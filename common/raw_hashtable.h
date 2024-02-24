// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_HASHTABLE_H_
#define CARBON_COMMON_RAW_HASHTABLE_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "common/hashing.h"
#include "common/raw_hashtable_metadata_group.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

// A namespace collecting a set of low-level utilities for building hashtable
// data structures. These should only be used as implementation details of
// higher-level data structure APIs.
//
// For example, see `set.h` for a hashtable-based set data structure, and
// `map.h` for a hashtable-based map data structure.
//
// The utilities in this namespace fall into a few categories:
//
// - Primitives to manage "groups" of hashtable entries that have densely packed
//   control bytes we can scan rapidly as a group, often using SIMD facilities
//   to process the entire group at once.
//
// - Tools to manipulate and work with the storage of offsets needed to
//   represent both key and key-value hashtables using these groups to organize
//   their entries.
//
// - Abstractions around efficiently probing across the hashtable consisting of
//   these "groups" of entries, and scanning within them to implement
//   traditional open-hashing hashtable operations.
//
// - Base classes to provide as much of the implementation of the user-facing
//   APIs as possible in a common way. This includes the most performance
//   sensitive code paths for the implementation of the data structures.
namespace Carbon::RawHashtable {

// A global variable whose address is used as a seed. This allows ASLR to
// introduce some variation in hashtable ordering.
extern volatile std::byte global_addr_seed;

// If allocating storage, allocate a minimum of one cacheline of group metadata
// and a minimum of one group.
constexpr ssize_t MinAllocatedSize = std::max<ssize_t>(64, MaxGroupSize);

[[clang::always_inline]] inline void Prefetch(const void* address) {
  // Currently we just hard code a single "low" temporal locality prefetch as
  // we're primarily expecting a brief use of the storage and then to return to
  // application code.
  __builtin_prefetch(address, /*read*/ 0, /*low-locality*/ 1);
}

template <typename KeyT, typename ValueT>
struct StorageEntry {
  static constexpr bool IsTriviallyDestructible =
      std::is_trivially_destructible_v<KeyT> &&
      std::is_trivially_destructible_v<ValueT>;

  static constexpr bool IsTriviallyRelocatable =
      IsTriviallyDestructible && std::is_trivially_move_constructible_v<KeyT> &&
      std::is_trivially_move_constructible_v<ValueT>;

  auto key() -> KeyT& {
    // Ensure we don't need more alignment than available.
    static_assert(
        alignof(StorageEntry) <= MinAllocatedSize,
        "The minimum allocated size turns into the alignment of our array of "
        "storage entries as they follow the metadata byte array.");

    return *std::launder(reinterpret_cast<KeyT*>(&key_storage));
  }

  auto value() -> ValueT& {
    return *std::launder(reinterpret_cast<ValueT*>(&value_storage));
  }

  // We handle destruction and move manually as we only want to expose distinct
  // `KeyT` and `ValueT` subobjects to user code that may need to do in-place
  // construction. As a consequence, this struct only provides the storage and
  // we have to manually manage the construction, move, and destruction of the
  // objects.
  auto Destroy() -> void {
    static_assert(!IsTriviallyDestructible,
                  "Should never instantiate when trivial!");
    key().~KeyT();
    value().~ValueT();
  }
  auto Move(StorageEntry& new_entry) -> void {
    if constexpr (IsTriviallyRelocatable) {
      memcpy(&new_entry, this, sizeof(StorageEntry));
    } else {
      new (&new_entry.key_storage) KeyT(std::move(key()));
      key().~KeyT();
      new (&new_entry.value_storage) KeyT(std::move(value()));
      value().~ValueT();
    }
  }

  alignas(KeyT) std::byte key_storage[sizeof(KeyT)];
  alignas(ValueT) std::byte value_storage[sizeof(ValueT)];
};

template <typename KeyT>
struct StorageEntry<KeyT, void> {
  static constexpr bool IsTriviallyDestructible =
      std::is_trivially_destructible_v<KeyT>;

  static constexpr bool IsTriviallyRelocatable =
      IsTriviallyDestructible && std::is_trivially_move_constructible_v<KeyT>;

  auto key() -> KeyT& {
    // Ensure we don't need more alignment than available.
    static_assert(
        alignof(StorageEntry) <= MinAllocatedSize,
        "The minimum allocated size turns into the alignment of our array of "
        "storage entries as they follow the metadata byte array.");

    return *std::launder(reinterpret_cast<KeyT*>(&key_storage));
  }

  auto Destroy() -> void {
    static_assert(!IsTriviallyDestructible,
                  "Should never instantiate when trivial!");
    key().~KeyT();
  }
  auto Move(StorageEntry& new_entry) -> void {
    if constexpr (IsTriviallyRelocatable) {
      memcpy(&new_entry, this, sizeof(StorageEntry));
    } else {
      new (&new_entry.key_storage) KeyT(std::move(key()));
      key().~KeyT();
    }
  }

  alignas(KeyT) std::byte key_storage[sizeof(KeyT)];
};

struct Storage {};

template <typename KeyT, typename ValueT>
struct StorageLayout {
  using EntryT = StorageEntry<KeyT, ValueT>;

  static constexpr ssize_t Alignment = std::max<ssize_t>(
      {alignof(MetadataGroup), alignof(StorageEntry<KeyT, ValueT>)});

  static auto Metadata(Storage* storage) -> uint8_t* {
    return reinterpret_cast<uint8_t*>(storage);
  }

  static constexpr auto EntriesOffset(ssize_t size) -> ssize_t {
    CARBON_DCHECK(llvm::isPowerOf2_64(size))
        << "Size must be a power of two for a hashed buffer!";
    // The size is always a power of two, which is typically perfectly aligned
    // and we prevent any too-small sizes to have adequate alignment
    // statically. As a result, the offset is exactly the size. But we
    // validate this here to catch alignment bugs early.
    CARBON_DCHECK(static_cast<uint64_t>(size) ==
                  llvm::alignTo<alignof(EntryT)>(size));
    return size;
  }

  static auto Entries(Storage* storage, ssize_t size) -> EntryT* {
    return reinterpret_cast<EntryT*>(reinterpret_cast<unsigned char*>(storage) +
                                     EntriesOffset(size));
  }

  static constexpr auto Size(ssize_t size) -> ssize_t {
    return EntriesOffset(size) + sizeof(EntryT) * size;
  }
};

template <typename KeyT, typename ValueT, ssize_t SmallSize>
struct alignas(StorageLayout<KeyT, ValueT>::Alignment) SmallStorageImpl
    : Storage {
  // Do early validation of the small size here.
  static_assert(llvm::isPowerOf2_64(SmallSize),
                "SmallSize must be a power of two for a hashed buffer!");
  static_assert(SmallSize >= MaxGroupSize,
                "We require all small sizes to multiples of the largest group "
                "size supported to ensure it can be used portably.  ");
  static_assert((SmallSize % MaxGroupSize) == 0,
                "Small size must be a multiple of the max group size supported "
                "so that we can allocate a whole number of groups.");
  // Implied by the max asserts above.
  static_assert(SmallSize >= GroupSize);
  static_assert((SmallSize % GroupSize) == 0);

  // Also ensure that for the specific types we will have enough alignment
  // without padding for the entries.
  static_assert(SmallSize >= alignof(StorageEntry<KeyT, ValueT>),
                "Requested a small size that would require padding between "
                "metadata bytes and correctly aligned key and value types. "
                "Either a larger small size or a zero small size and heap "
                "allocation are required for this key and value type.");

  static constexpr ssize_t SmallNumGroups = SmallSize / GroupSize;
  static_assert(llvm::isPowerOf2_64(SmallNumGroups),
                "The number of groups must be a power of two when hashing!");

  SmallStorageImpl() {
    static_assert(
        SmallSize == 0 || (offsetof(SmallStorageImpl, entries) == SmallSize),
        "Offset to keys in small size storage doesn't match computed offset!");
  }

  alignas(MetadataGroup) uint8_t metadata[SmallNumGroups * GroupSize];
  mutable StorageEntry<KeyT, ValueT> entries[SmallSize];
};

template <typename KeyT, typename ValueT>
struct SmallStorageImpl<KeyT, ValueT, 0> : Storage {
  SmallStorageImpl() = default;

  uint8_t metadata[0];
  mutable StorageEntry<KeyT, ValueT> entries[0];
};

// Base class that encodes either the absence of a value or a value type.
template <typename KeyT, typename ValueT = void>
class Base;

template <typename InputKeyT, typename InputValueT = void>
class ViewBase {
 protected:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using LayoutT = StorageLayout<KeyT, ValueT>;
  using EntryT = typename LayoutT::EntryT;

  using ConstViewBaseT = ViewBase<const KeyT, const ValueT>;

  friend class Base<KeyT, ValueT>;

  // Make more-`const` types friends to enable conversions that add `const`.
  friend class ViewBase<const KeyT, ValueT>;
  friend class ViewBase<KeyT, const ValueT>;
  friend class ViewBase<const KeyT, const ValueT>;

  ViewBase() = default;
  ViewBase(ssize_t size, Storage* storage) : size_(size), storage_(storage) {}

  // Support adding `const` to either key or value type of some other view.
  template <typename OtherKeyT, typename OtherValueT>
  // NOLINTNEXTLINE(google-explicit-constructor)
  ViewBase(ViewBase<OtherKeyT, OtherValueT> other_view)
    requires(std::same_as<KeyT, OtherKeyT> ||
             std::same_as<KeyT, const OtherKeyT>) &&
                (std::same_as<ValueT, OtherValueT> ||
                 std::same_as<ValueT, const OtherValueT>)
      : size_(other_view.size_), storage_(other_view.storage_) {}

  auto size() const -> ssize_t { return size_; }
  auto metadata() const -> uint8_t* { return LayoutT::Metadata(storage_); }
  auto entries_offset() const -> ssize_t {
    return LayoutT::EntriesOffset(size());
  }
  auto entries() const -> EntryT* { return LayoutT::Entries(storage_, size()); }

  template <typename LookupKeyT>
  auto LookupIndexHashed(LookupKeyT lookup_key) const -> EntryT*;

  template <typename IndexCallbackT, typename GroupCallbackT>
  void ForEachIndex(IndexCallbackT index_callback,
                    GroupCallbackT group_callback);

  auto CountProbedKeys() const -> ssize_t;

  ssize_t size_;
  Storage* storage_;
};

template <typename InputKeyT, typename InputValueT>
class Base {
 protected:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewBaseT = ViewBase<KeyT, ValueT>;
  using LayoutT = typename ViewBaseT::LayoutT;
  using EntryT = typename ViewBaseT::EntryT;

  template <ssize_t SmallSize>
  using SmallStorageT = SmallStorageImpl<KeyT, ValueT, SmallSize>;

  static auto Allocate(ssize_t size) -> Storage* {
    return reinterpret_cast<Storage*>(__builtin_operator_new(
        LayoutT::Size(size), static_cast<std::align_val_t>(LayoutT::Alignment),
        std::nothrow_t()));
  }

  static auto Deallocate(Storage* storage, ssize_t size) -> void {
    ssize_t allocated_size = LayoutT::Size(size);
    // We don't need the size, but make sure it always compiles.
    static_cast<void>(allocated_size);
    return __builtin_operator_delete(
        storage,
#if __cpp_sized_deallocation
        allocated_size,
#endif
        static_cast<std::align_val_t>(LayoutT::Alignment));
  }

  Base(int small_size, Storage* small_storage) : small_size_(small_size) {
    CARBON_CHECK(small_size >= 0);
    ConstructImpl(small_storage);
  }

  ~Base();

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewBaseT() const { return impl_view_; }

  auto size() const -> ssize_t { return impl_view_.size_; }
  auto size() -> ssize_t& { return impl_view_.size_; }
  auto storage() const -> Storage* { return impl_view_.storage_; }
  auto storage() -> Storage*& { return impl_view_.storage_; }
  auto metadata() const -> uint8_t* { return impl_view_.metadata(); }
  auto entries() const -> EntryT* { return impl_view_.entries(); }

  auto is_small() const -> bool { return size() <= small_size(); }
  auto small_size() const -> ssize_t {
    return static_cast<unsigned>(small_size_);
  }

  void Init(ssize_t init_size, Storage* init_storage);

  template <typename LookupKeyT>
  auto InsertIntoEmptyIndex(LookupKeyT lookup_key) -> EntryT*;

  template <typename LookupKeyT>
  auto EraseKey(LookupKeyT lookup_key) -> bool;

  template <typename LookupKeyT>
  auto GrowRehashAndInsertIndex(LookupKeyT lookup_key) -> EntryT*;
  template <typename LookupKeyT>
  auto InsertIndexHashed(LookupKeyT lookup_key) -> std::pair<EntryT*, bool>;

  auto ClearImpl() -> void;
  auto DestroyImpl() -> void;
  auto ConstructImpl(Storage* small_storage) -> void;

  ViewBaseT impl_view_;
  int growth_budget_;
  int small_size_;
};

inline auto ComputeProbeMaskFromSize(ssize_t size) -> size_t {
  CARBON_DCHECK(llvm::isPowerOf2_64(size))
      << "Size must be a power of two for a hashed buffer!";
  // The probe mask needs to mask down to keep the index within
  // `size`. Since `size` is a power of two, this is equivalent to
  // `size - 1`. We also mask off the low bits while here to match the size of
  // the groups of entries.
  return (size - 1) & ~GroupMask;
}

// This class handles building a sequence of probe indices from a given
// starting point, including both the quadratic growth and masking the index
// to stay within the bucket array size. The starting point doesn't need to be
// clamped to the size ahead of time (or even by positive), we will do it
// internally.
//
// We compute the quadratic probe index incrementally, but we can also compute
// it mathematically and will check that the incremental result matches our
// mathematical expectation. We use the quadratic probing formula of:
//
//   p(x,s) = (x + (s + s^2)/2) mod (Size / GroupSize)
//
// This particular quadratic sequence will visit every value modulo the
// provided size divided by the group size.
//
// However, we compute it scaled to the group size constant G and have it visit
// each G multiple modulo the size using the scaled formula:
//
//   p(x,s) = (x + (s + (s * s * G)/(G * G))/2) mod Size
class ProbeSequence {
  ssize_t Step = 0;
  size_t Mask;
  ssize_t i;
#ifndef NDEBUG
  ssize_t Start;
  ssize_t Size;
#endif

 public:
  ProbeSequence(ssize_t start, ssize_t size) {
    Mask = ComputeProbeMaskFromSize(size);
    i = start & Mask;
#ifndef NDEBUG
    this->Start = start & Mask;
    this->Size = size;
#endif
  }

  void step() {
    Step += GroupSize;
    i = (i + Step) & Mask;
#ifndef NDEBUG
    CARBON_DCHECK(
        i ==
        ((Start +
          ((Step + (Step * Step * GroupSize) / (GroupSize * GroupSize)) / 2)) %
         Size))
        << "Index in probe sequence does not match the expected formula.";
    CARBON_DCHECK(Step < Size) << "We necessarily visit all groups, so we "
                                  "can't have more probe steps than groups.";
#endif
  }

  auto getIndex() const -> ssize_t { return i; }
};

inline auto ComputeSeed() -> uint64_t {
  return reinterpret_cast<uint64_t>(&global_addr_seed);
}

inline auto ComputeMetadataByte(size_t tag) -> uint8_t {
  // Mask one over the high bit so that engaged control bytes are easily
  // identified.
  return tag;  // | 0b10000000;
}

// TODO: Evaluate keeping this outlined to see if macro benchmarks observe the
// same perf hit as micros.
template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
auto ViewBase<InputKeyT, InputValueT>::LookupIndexHashed(
    LookupKeyT lookup_key) const -> EntryT* {
  ssize_t local_size = size();
  CARBON_DCHECK(local_size > 0);

  uint8_t* local_metadata = metadata();
  HashCode hash = HashValue(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();
  uint8_t metadata_byte = ComputeMetadataByte(tag);

  EntryT* local_entries = entries();
  ProbeSequence s(hash_index, local_size);
  do {
    ssize_t group_index = s.getIndex();
    MetadataGroup g = MetadataGroup::Load(local_metadata, group_index);
    auto metadata_matched_range = g.Match(metadata_byte);
    if (LLVM_LIKELY(metadata_matched_range)) {
      auto byte_it = metadata_matched_range.begin();
      auto byte_end = metadata_matched_range.end();
      do {
        ssize_t index = group_index + *byte_it;
        EntryT* entry = &local_entries[index];
        if (LLVM_LIKELY(entry->key() == lookup_key)) {
          __builtin_assume(entry != nullptr);
          return entry;
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // empty slots and we're done probing.
    auto empty_byte_matched_range = g.MatchEmpty();
    if (LLVM_LIKELY(empty_byte_matched_range)) {
      return nullptr;
    }

    s.step();
  } while (LLVM_UNLIKELY(true));
}

template <typename InputKeyT, typename InputValueT>
template <typename IndexCallbackT, typename GroupCallbackT>
[[clang::always_inline]] void ViewBase<InputKeyT, InputValueT>::ForEachIndex(
    IndexCallbackT index_callback, GroupCallbackT group_callback) {
  uint8_t* local_metadata = metadata();
  EntryT* local_entries = entries();

  ssize_t local_size = size();
  for (ssize_t group_index = 0; group_index < local_size;
       group_index += GroupSize) {
    auto g = MetadataGroup::Load(local_metadata, group_index);
    auto present_matched_range = g.MatchPresent();
    if (!present_matched_range) {
      continue;
    }
    for (ssize_t byte_index : present_matched_range) {
      ssize_t index = group_index + byte_index;
      index_callback(local_entries, index);
    }

    group_callback(local_metadata, group_index);
  }
}

template <typename InputKeyT, typename InputValueT>
auto ViewBase<InputKeyT, InputValueT>::CountProbedKeys() const -> ssize_t {
  uint8_t* local_metadata = this->metadata();
  EntryT* local_entries = this->entries();
  ssize_t local_size = this->size();
  ssize_t count = 0;
  for (ssize_t group_index = 0; group_index < local_size;
       group_index += GroupSize) {
    auto g = MetadataGroup::Load(local_metadata, group_index);
    auto present_matched_range = g.MatchPresent();
    for (ssize_t byte_index : present_matched_range) {
      ssize_t index = group_index + byte_index;
      HashCode hash = HashValue(local_entries[index].key(), ComputeSeed());
      ssize_t hash_index = hash.ExtractIndexAndTag<7>().first &
                           ComputeProbeMaskFromSize(local_size);
      count += static_cast<ssize_t>(hash_index != group_index);
    }
  }
  return count;
}

template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
[[clang::noinline]] auto Base<InputKeyT, InputValueT>::InsertIntoEmptyIndex(
    LookupKeyT lookup_key) -> EntryT* {
  HashCode hash = HashValue(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();
  uint8_t metadata_byte = ComputeMetadataByte(tag);
  uint8_t* local_metadata = metadata();
  EntryT* local_entries = entries();

  for (ProbeSequence s(hash_index, size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = MetadataGroup::Load(local_metadata, group_index);

    if (auto empty_match = g.MatchEmpty()) {
      ssize_t index = group_index + empty_match.index();
      local_metadata[index] = metadata_byte;
      return &local_entries[index];
    }

    // Otherwise we continue probing.
  }
}

inline auto ComputeNewSize(ssize_t old_size) -> ssize_t {
  // We want the next power of two. This should always be a power of two coming
  // in, and so we just verify that.
  CARBON_DCHECK(old_size == static_cast<ssize_t>(llvm::PowerOf2Ceil(old_size)))
      << "Expected a power of two!";
  ssize_t new_size;
  bool overflow = __builtin_mul_overflow(old_size, 2, &new_size);
  CARBON_CHECK(!overflow) << "Computing the new size overflowed `ssize_t`!";
  return new_size;
}

inline auto GrowthThresholdForSize(ssize_t size) -> ssize_t {
  // We use a 7/8ths load factor to trigger growth.
  return size - size / 8;
}

template <typename InputKeyT, typename InputValueT>
void Base<InputKeyT, InputValueT>::Init(ssize_t init_size,
                                        Storage* init_storage) {
  size() = init_size;
  storage() = init_storage;
  std::memset(metadata(), MetadataGroup::Empty, init_size);
  growth_budget_ = GrowthThresholdForSize(init_size);
}

template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
auto Base<InputKeyT, InputValueT>::EraseKey(LookupKeyT lookup_key) -> bool {
  EntryT* entry = impl_view_.LookupIndexHashed(lookup_key);
  if (!entry) {
    return false;
  }

  // If there are empty slots in this group then nothing will probe past this
  // group looking for an entry so we can simply set this slot to empty as
  // well. However, if every slot in this group is full, it might be part of
  // a long probe chain that we can't disrupt. In that case we mark the group
  // as deleted to keep probes continuing past it.
  //
  // If we mark the slot as empty, we'll also need to increase the growth
  // budget.
  uint8_t* local_metadata = this->metadata();
  EntryT* local_entries = entries();
  ssize_t index = entry - local_entries;
  ssize_t group_index = index & ~GroupMask;
  auto g = MetadataGroup::Load(local_metadata, group_index);
  auto empty_matched_range = g.MatchEmpty();
  if (empty_matched_range) {
    local_metadata[index] = MetadataGroup::Empty;
    ++this->growth_budget_;
  } else {
    local_metadata[index] = MetadataGroup::Deleted;
  }

  if constexpr (!EntryT::IsTriviallyDestructible) {
    entry->Destroy();
  }

  return true;
}

template <typename InputKeyT, typename InputValueT>
Base<InputKeyT, InputValueT>::~Base() {
  DestroyImpl();
}

template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
[[clang::noinline]] auto Base<InputKeyT, InputValueT>::GrowRehashAndInsertIndex(
    LookupKeyT lookup_key) -> EntryT* {
  // We collect the probed elements in a small vector for re-insertion. It is
  // tempting to reuse the already allocated storage, but doing so appears to
  // be a (very slight) performance regression. These are relatively rare and
  // storing them into the existing storage creates stores to the same regions
  // of memory we're reading. Moreover, it requires moving both the key and the
  // value twice, and doing the `memcpy` widening for relocatable types before
  // the group walk rather than after the group walk. In practice, between the
  // statistical rareness and using a large small size buffer on the stack, we
  // can handle this most efficiently with temporary storage.
  llvm::SmallVector<ssize_t, 128> probed_indices;

  // We grow into a new `MapBase` so that both the new and old maps are
  // fully functional until all the entries are moved over. However, we directly
  // manipulate the internals to short circuit many aspects of the growth.
  ssize_t old_size = this->size();
  CARBON_DCHECK(old_size > 0);
  CARBON_DCHECK(this->growth_budget_ == 0);

  bool old_small = this->is_small();
  Storage* old_storage = this->storage();
  uint8_t* old_metadata = this->metadata();
  EntryT* old_entries = this->entries();

#ifndef NDEBUG
  ssize_t debug_empty_count =
      llvm::count(llvm::ArrayRef(old_metadata, old_size),
                  MetadataGroup::Empty) +
      llvm::count(llvm::ArrayRef(old_metadata, old_size),
                  MetadataGroup::Deleted);
  CARBON_DCHECK(debug_empty_count >=
                (old_size - GrowthThresholdForSize(old_size)))
      << "debug_empty_count: " << debug_empty_count << ", size: " << old_size;
#endif

  // Compute the new size and grow the storage in place (if possible).
  ssize_t new_size = ComputeNewSize(old_size);
  this->size() = new_size;
  this->storage() = Allocate(new_size);
  this->growth_budget_ = GrowthThresholdForSize(new_size);

  // Now extract the new components of the table.
  uint8_t* new_metadata = this->metadata();
  EntryT* new_entries = this->entries();

  // The common case, especially for large sizes, is that we double the size
  // when we grow. This allows an important optimization -- we're adding
  // exactly one more high bit to the hash-computed index for each entry. This
  // in turn means we can classify every entry in the table into three cases:
  //
  // 1) The new high bit is zero, the entry is at the same index in the new
  //    table as the old.
  //
  // 2) The new high bit is one, the entry is at the old index plus the old
  //    size.
  //
  // 3) The entry's current index doesn't match the initial hash index because
  //    it required some amount of probing to find an empty slot.
  //
  // The design of the hash table is specifically to minimize how many entries
  // fall into case (3), so we expect the vast majority of entries to be in
  // (1) or (2). This lets us model growth notionally as duplicating the hash
  // table up by `old_size` bytes, clearing out the empty slots, and inserting
  // any probed elements.

  ssize_t count = 0;
  for (ssize_t group_index = 0; group_index < old_size;
       group_index += GroupSize) {
    auto low_g = MetadataGroup::Load(old_metadata, group_index);
    // Make sure to match present elements first to enable pipelining with
    // clearing.
    auto present_matched_range = low_g.MatchPresent();
    low_g.ClearDeleted();
    MetadataGroup high_g;
    if constexpr (MetadataGroup::FastByteClear) {
      // When we have a fast byte clear, we can update the metadata for the
      // growth in-register and store at the end.
      high_g = low_g;
    } else {
      // If we don't have a fast byte clear, we can store the metadata group
      // eagerly here and overwrite bytes with a byte store below instead of
      // clearing the byte in-register.
      low_g.Store(new_metadata, group_index);
      low_g.Store(new_metadata, group_index | old_size);
    }
    for (ssize_t byte_index : present_matched_range) {
      ++count;
      ssize_t old_index = group_index + byte_index;
      if constexpr (!MetadataGroup::FastByteClear) {
        CARBON_DCHECK(new_metadata[old_index] == old_metadata[old_index]);
        CARBON_DCHECK(new_metadata[old_index | old_size] ==
                      old_metadata[old_index]);
      }
      HashCode hash = HashValue(old_entries[old_index].key(), ComputeSeed());
      ssize_t old_hash_index = hash.ExtractIndexAndTag<7>().first &
                               ComputeProbeMaskFromSize(old_size);
      if (LLVM_UNLIKELY(old_hash_index != group_index)) {
        probed_indices.push_back(old_index);
        if constexpr (MetadataGroup::FastByteClear) {
          low_g.ClearByte(byte_index);
          high_g.ClearByte(byte_index);
        } else {
          new_metadata[old_index] = MetadataGroup::Empty;
          new_metadata[old_index | old_size] = MetadataGroup::Empty;
        }
        continue;
      }
      ssize_t new_index = hash.ExtractIndexAndTag<7>().first &
                          ComputeProbeMaskFromSize(new_size);
      CARBON_DCHECK(new_index == old_hash_index ||
                    new_index == (old_hash_index | old_size));
      // Toggle the newly added bit of the index to get to the other possible
      // target index.
      if constexpr (MetadataGroup::FastByteClear) {
        (new_index == old_hash_index ? high_g : low_g).ClearByte(byte_index);
        new_index += byte_index;
      } else {
        new_index += byte_index;
        new_metadata[new_index ^ old_size] = MetadataGroup::Empty;
      }

      // If we need to explicitly move (and destroy) the key or value, do so
      // here where we already know its target.
      if constexpr (!EntryT::IsTriviallyRelocatable) {
        old_entries[old_index].Move(new_entries[new_index]);
      }
    }
    if constexpr (MetadataGroup::FastByteClear) {
      low_g.Store(new_metadata, group_index);
      high_g.Store(new_metadata, (group_index | old_size));
    }
  }
  CARBON_DCHECK((count - static_cast<ssize_t>(probed_indices.size())) ==
                (new_size - llvm::count(llvm::ArrayRef(new_metadata, new_size),
                                        MetadataGroup::Empty)));
#ifndef NDEBUG
  CARBON_DCHECK(debug_empty_count == (old_size - count));
  CARBON_DCHECK(llvm::count(llvm::ArrayRef(new_metadata, new_size),
                            MetadataGroup::Empty) ==
                debug_empty_count +
                    static_cast<ssize_t>(probed_indices.size()) + old_size);
#endif

  // If the keys or values are trivially relocatable, we do a bulk memcpy of
  // them into place. This will copy them into both possible locations, which is
  // fine. One will be empty and clobbered if reused or ignored. The other will
  // be the one used. This might seem like it needs it to be valid for us to
  // create two copies, but it doesn't. This produces the exact same storage as
  // copying the storage into the wrong location first, and then again into the
  // correct location. Only one is live and only one is destroyed.
  if constexpr (EntryT::IsTriviallyRelocatable) {
    memcpy(new_entries, old_entries, old_size * sizeof(EntryT));
    memcpy(new_entries + old_size, old_entries, old_size * sizeof(EntryT));
  }

  // We have to use the normal insert for anything that was probed before, but
  // we know we'll find an empty slot, so leverage that. We extract the probed
  // keys from the bottom of the old keys storage.
  for (ssize_t old_index : probed_indices) {
    // We may end up needing to do a sequence of re-inserts, swapping out keys
    // and values each time, so we enter a loop here and break out of it for the
    // simple cases of re-inserting into a genuinely empty slot.
    EntryT* new_entry =
        this->InsertIntoEmptyIndex(old_entries[old_index].key());
    old_entries[old_index].Move(*new_entry);
  }
  CARBON_DCHECK(count ==
                (new_size - llvm::count(llvm::ArrayRef(new_metadata, new_size),
                                        MetadataGroup::Empty)));
  this->growth_budget_ -= count;
  CARBON_DCHECK(this->growth_budget_ ==
                (GrowthThresholdForSize(new_size) -
                 (new_size - llvm::count(llvm::ArrayRef(new_metadata, new_size),
                                         MetadataGroup::Empty))));
  CARBON_DCHECK(this->growth_budget_ > 0 &&
                "Must still have a growth budget after rehash!");

  if (!old_small) {
    // Old isn't a small buffer, so we need to deallocate it.
    Deallocate(old_storage, old_size);
  }

  // And lastly insert the lookup_key into an index in the newly grown map and
  // return that index for use.
  --this->growth_budget_;
  return this->InsertIntoEmptyIndex(lookup_key);
}

// Tries to insert the given lookup key into the map. Returns three pieces of
// data compressed into two registers (in order to avoid an in-memory return).
// These are the group pointer, a bool representing whether insertion is in fact
// required, and the byte index of either the found entry in the group or the
// slot of the group to insert into. The index will be `-1` if insertion
// isn't possible without growing. Last but not least, because we leave this
// outlined for code size, we also need to encode the `bool` in a way that is
// effective with various encodings and ABIs. Currently this is `uint32_t` as
// that seems to result in good code.
template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
//[[clang::noinline]]
auto Base<InputKeyT, InputValueT>::InsertIndexHashed(LookupKeyT lookup_key)
    -> std::pair<EntryT*, bool> {
  CARBON_DCHECK(this->size() > 0);

  uint8_t* local_metadata = this->metadata();

  HashCode hash = HashValue(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();
  uint8_t metadata_byte = ComputeMetadataByte(tag);

  // We re-purpose the empty control byte to signal no insert is needed to the
  // caller. This is guaranteed to not be a control byte we're inserting.
  // constexpr uint8_t NoInsertNeeded = Group::Empty;

  ssize_t group_with_deleted_index;
  MetadataGroup::MatchIndex deleted_match = {};

  EntryT* local_entries = this->entries();

  auto return_insert_at_index = [&](ssize_t index) -> std::pair<EntryT*, bool> {
    // We'll need to insert at this index so set the control group byte to the
    // proper value.
    local_metadata[index] = metadata_byte;
    return {&local_entries[index], true};
  };

  for (ProbeSequence s(hash_index, this->size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = MetadataGroup::Load(local_metadata, group_index);

    auto control_byte_matched_range = g.Match(metadata_byte);
    auto empty_match = g.MatchEmpty();
    if (control_byte_matched_range) {
      EntryT* group_entries = &local_entries[group_index];
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        EntryT* entry = &group_entries[*byte_it];
        if (LLVM_LIKELY(entry->key() == lookup_key)) {
          return {entry, false};
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // Track the first group with a deleted entry that we could insert over.
    if (!deleted_match) {
      deleted_match = g.MatchDeleted();
      group_with_deleted_index = group_index;
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // no empty slots. In that case, we'll continue probing.
    if (!empty_match) {
      continue;
    }
    // Ok, we've finished probing without finding anything and need to insert
    // instead.

    // If we found a deleted slot, we don't need the probe sequence to insert
    // so just bail. We want to ensure building up a table is fast so we
    // de-prioritize this a bit. In practice this doesn't have too much of an
    // effect.
    if (LLVM_UNLIKELY(deleted_match)) {
      return return_insert_at_index(group_with_deleted_index +
                                    deleted_match.index());
    }

    // We're going to need to grow by inserting into an empty slot. Check that
    // we have the budget for that before we compute the exact index of the
    // empty slot. Without the growth budget we'll have to completely rehash and
    // so we can just bail here.
    if (LLVM_UNLIKELY(this->growth_budget_ == 0)) {
      return {this->GrowRehashAndInsertIndex(lookup_key), true};
    }

    --this->growth_budget_;
    return return_insert_at_index(group_index + empty_match.index());
  }

  CARBON_FATAL() << "We should never finish probing without finding the entry "
                    "or an empty slot.";
}

template <typename InputKeyT, typename InputValueT>
auto Base<InputKeyT, InputValueT>::ClearImpl() -> void {
  this->impl_view_.ForEachIndex(
      [this](EntryT* /*entries*/, ssize_t index) {
        // FIXME
        static_cast<void>(this);
        if constexpr (!EntryT::IsTriviallyDestructible) {
          this->entries()[index].Destroy();
        }
      },
      [](uint8_t* metadata, ssize_t group_index) {
        // Clear the group.
        std::memset(metadata + group_index, MetadataGroup::Empty, GroupSize);
      });
  this->growth_budget_ = GrowthThresholdForSize(this->size());
}

template <typename InputKeyT, typename InputValueT>
auto Base<InputKeyT, InputValueT>::DestroyImpl() -> void {
  // Nothing to do when in the un-allocated and unused state.
  if (this->size() == 0) {
    return;
  }

  // Destroy all the entries.
  if constexpr (!EntryT::IsTriviallyDestructible) {
    this->impl_view_.ForEachIndex(
        [this](EntryT* /*entries*/, ssize_t index) {
          this->entries()[index].Destroy();
        },
        [](auto...) {});
  }

  // If small, nothing to deallocate.
  if (this->is_small()) {
    return;
  }

  // Just deallocate the storage without updating anything when destroying the
  // object.
  Deallocate(this->storage(), this->size());
}

template <typename InputKeyT, typename InputValueT>
auto Base<InputKeyT, InputValueT>::ConstructImpl(Storage* small_storage)
    -> void {
  if (small_size_ > 0) {
    Init(small_size_, small_storage);
  } else {
    // Directly allocate the initial buffer so that the hashtable is never in
    // an empty state.
    Init(MinAllocatedSize, Allocate(MinAllocatedSize));
  }
}

}  // namespace Carbon::RawHashtable

#endif  // CARBON_COMMON_RAW_HASHTABLE_H_
