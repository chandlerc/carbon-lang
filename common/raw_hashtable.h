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
// higher-level data-structure APIs.
//
// These utilities support hashtables following a *specific* API design pattern,
// and using Small-Size Optimization, or "SSO", when desired. We expect there to
// be three layers to any hashtable design:
//
// - A *view* type: a read-only view of the hashtable contents. This type should
//   be a value type and is expected to be passed by-value in APIs. However, it
//   will have `const`-reference semantics, much like a `std::string_view`. Note
//   that the *entries* will continue to be mutable, it is only the *table* that
//   is read-only.
//
// - A *base* type: a base class type of the actual hashtable, which allows
//   almost all mutable operations but erases any specific SSO buffer size.
//   Because this is a base of the actual hash table, it is designed to be
//   passed as a non-`const` reference or pointer.
//
// - A *table* type: the actual hashtable which derives from the base type and
//   adds any desired SSO storage buffer. Beyond the physical storage, it also
//   allows resetting the table to its initial state & allocated size, as well
//   as copying and moving the table.
//
// For complete examples of the API design, see `set.h` for a hashtable-based
// set data structure, and `map.h` for a hashtable-based map data structure.
//
// The hashtable design implemented here has several key invariants and design
// elements that are essential to all three of the types above and the
// functionality they provide.
//
// - The underlying hashtable use open addressing, a power-of-two table size,
//   and quadratic probing rather than closed addressing and chaining.
//
// - The allocated table storage is organized into an array of one-byte metadata
//   for every entry, followed by an array of storage for key/value pairs. The
//   storage for an entry is an internal type that should not be exposed to
//   users, and instead only the underlying keys and values.
//
// - The hash addressing and probing occurs over *groups* of entries rather than
//   individual entries. These are not necessarily buckets of colliding entries,
//   however, as they may be filled by probing with entries that have different
//   initial hashed positions in the open addressed table.
//
// - Groups are scanned rapidly using the one-byte metadata for each entry in
//   the group and CPU instructions that allow comparing all of the metadata for
//   a group in parallel. Either SIMD or a large (64-bit) integer algorithm is
//   used. For more details on the metadata group encoding and scanning, see
//   `raw_hashtable_metadata_group.h` which contains all of those details.
//
// - `GroupSize` is a platform-specific relatively small power of two that fits
//   in some hardware register. However, `MaxGroupSize` is provided as a
//   portable max that is also a power of two. The table storage, whether
//   provided by an SSO buffer or allocated, is required to be a multiple of
//   `MaxGroupSize` to keep the requirement portable but sufficient for all
//   platforms.
//
// - There is *always* an allocated table of some multiple of `MaxGroupSize`.
//   This allows accesses to be branchless. When heap allocated, we pro-actively
//   allocate at least a minimum heap size table. When there is a small-size
//   optimization (SSO) buffer, that provides the initial allocation.
//
// - The table performs a minimal amount of book keeping that limits the APIs it
//   can support. For example, it doesn't track the exact number of entries in a
//   table and so doesn't support a container-style `size` API. The `alloc_size`
//   is the size of the table *allocated*, not *used*.
//
// - There is no direct iterator support because of the complexity of embedding
//   the group-based metadata scanning into an iterator model. Instead, there is
//   just a for-each method with a lambda to observe all entries. The order of
//   this observation is also not guaranteed.
namespace Carbon::RawHashtable {

// If allocating storage, allocate a minimum of one cacheline of group metadata
// or a minimum of one group, whichever is larger.
constexpr ssize_t MinAllocatedSize = std::max<ssize_t>(64, MaxGroupSize);

// An entry in the hashtable storage of a `KeyT` and `ValueT` object.
//
// Allows manual construction, destruction, and access to these values so we can
// create arrays af the entries prior to populating them with actual keys and
// values.
template <typename KeyT, typename ValueT>
struct StorageEntry {
  static constexpr bool IsTriviallyDestructible =
      std::is_trivially_destructible_v<KeyT> &&
      std::is_trivially_destructible_v<ValueT>;

  static constexpr bool IsTriviallyRelocatable =
      IsTriviallyDestructible && std::is_trivially_move_constructible_v<KeyT> &&
      std::is_trivially_move_constructible_v<ValueT>;

  auto key() const -> const KeyT& {
    // Ensure we don't need more alignment than available. Inside a method body
    // to apply to the complete type.
    static_assert(
        alignof(StorageEntry) <= MinAllocatedSize,
        "The minimum allocated size turns into the alignment of our array of "
        "storage entries as they follow the metadata byte array.");

    return *std::launder(reinterpret_cast<const KeyT*>(&key_storage));
  }
  auto key() -> KeyT& {
    return const_cast<KeyT&>(const_cast<const StorageEntry*>(this)->key());
  }

  auto value() const -> const ValueT& {
    return *std::launder(reinterpret_cast<const ValueT*>(&value_storage));
  }
  auto value() -> ValueT& {
    return const_cast<ValueT&>(const_cast<const StorageEntry*>(this)->value());
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

  auto CopyFrom(const StorageEntry& entry) -> void {
    if constexpr (IsTriviallyRelocatable) {
      memcpy(this, &entry, sizeof(StorageEntry));
    } else {
      new (&key_storage) KeyT(entry.key());
      new (&value_storage) ValueT(entry.value());
    }
  }

  // Move from an expiring entry and destroy that entry's key and value.
  // Optimizes to directly use `memcpy` when correct.
  auto MoveFrom(StorageEntry&& entry) -> void {
    if constexpr (IsTriviallyRelocatable) {
      memcpy(this, &entry, sizeof(StorageEntry));
    } else {
      new (&key_storage) KeyT(std::move(entry.key()));
      entry.key().~KeyT();
      new (&value_storage) ValueT(std::move(entry.value()));
      entry.value().~ValueT();
    }
  }

  alignas(KeyT) std::byte key_storage[sizeof(KeyT)];
  alignas(ValueT) std::byte value_storage[sizeof(ValueT)];
};

// A specialization of the storage entry for sets without a distinct value type.
// Somewhat duplicative with the key-value version, but C++ specialization makes
// doing better difficult.
template <typename KeyT>
struct StorageEntry<KeyT, void> {
  static constexpr bool IsTriviallyDestructible =
      std::is_trivially_destructible_v<KeyT>;

  static constexpr bool IsTriviallyRelocatable =
      IsTriviallyDestructible && std::is_trivially_move_constructible_v<KeyT>;

  auto key() const -> const KeyT& {
    // Ensure we don't need more alignment than available.
    static_assert(
        alignof(StorageEntry) <= MinAllocatedSize,
        "The minimum allocated size turns into the alignment of our array of "
        "storage entries as they follow the metadata byte array.");

    return *std::launder(reinterpret_cast<const KeyT*>(&key_storage));
  }
  auto key() -> KeyT& {
    return const_cast<KeyT&>(const_cast<const StorageEntry*>(this)->key());
  }

  auto Destroy() -> void {
    static_assert(!IsTriviallyDestructible,
                  "Should never instantiate when trivial!");
    key().~KeyT();
  }

  auto CopyFrom(const StorageEntry& entry) -> void {
    if constexpr (IsTriviallyRelocatable) {
      memcpy(this, &entry, sizeof(StorageEntry));
    } else {
      new (&key_storage) KeyT(entry.key());
    }
  }

  auto MoveFrom(StorageEntry&& entry) -> void {
    if constexpr (IsTriviallyRelocatable) {
      memcpy(this, &entry, sizeof(StorageEntry));
    } else {
      new (&key_storage) KeyT(std::move(entry.key()));
      entry.key().~KeyT();
    }
  }

  alignas(KeyT) std::byte key_storage[sizeof(KeyT)];
};

// A placeholder empty type used to model pointers to the allocated buffer of
// storage.
//
// The allocated storage doesn't have a meaningful static layout -- it consists
// of an array of metadata groups followed by an array of storage entries.
// However, we want to be able to mark pointers to this and so use pointers to
// this placeholder type as that signifier.
//
// This is a complete, empty type so that it can be used as a base class of a
// specific concrete storage type for compile-time sized storage.
struct Storage {};

// Forward declaration to support friending, see the definition below.
template <typename KeyT, typename ValueT = void>
class BaseImpl;

// Implementation helper for defining a read-only view type for a hashtable.
//
// A specific user-facing hashtable view type should derive privately from this
// type, and forward the implementation of its interface to functions in this
// type.
//
// The methods available to user-facing hashtable types are `protected`, and
// where they are expected to directly map to a public API, named with an
// `Impl`. The suffix naming ensures types don't `using` in these low-level APIs
// but declare their own and implement them by forwarding to these APIs. We
// don't want users to have to read these implementation details to understand
// their container's API, so none of these methods should be `using`-ed into the
// user facing types.
//
// Some of the types are just convenience aliases and aren't important to
// surface as part of the user-facing type API for readers and so those are
// reasonable to add via a `using`.
//
// Some methods are used by other parts of the raw hashtable implementation.
// Those are kept `private` and where necessary the other components of the raw
// hashtable implementation are friended to give access to them.
template <typename InputKeyT, typename InputValueT = void>
class ViewImpl {
 protected:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using EntryT = StorageEntry<KeyT, ValueT>;

  friend class BaseImpl<KeyT, ValueT>;

  // Make more-`const` types friends to enable conversions that add `const`.
  friend class ViewImpl<const KeyT, ValueT>;
  friend class ViewImpl<KeyT, const ValueT>;
  friend class ViewImpl<const KeyT, const ValueT>;

  ViewImpl() = default;

  // Support adding `const` to either key or value type of some other view.
  template <typename OtherKeyT, typename OtherValueT>
  // NOLINTNEXTLINE(google-explicit-constructor)
  ViewImpl(ViewImpl<OtherKeyT, OtherValueT> other_view)
    requires(std::same_as<KeyT, OtherKeyT> ||
             std::same_as<KeyT, const OtherKeyT>) &&
                (std::same_as<ValueT, OtherValueT> ||
                 std::same_as<ValueT, const OtherValueT>)
      : alloc_size_(other_view.alloc_size_), storage_(other_view.storage_) {}

  // Looks up an entry in the hashtable and returns its address or null if not
  // present.
  template <typename LookupKeyT>
  auto LookupEntry(LookupKeyT lookup_key) const -> EntryT*;

  // Calls `entry_callback` for each entry in the hashtable. All the entries
  // within a specific group are visited first, and then `group_callback` is
  // called on the group itself. The `group_callback` is typically only used by
  // the internals of the hashtable.
  template <typename EntryCallbackT, typename GroupCallbackT>
  auto ForEachEntry(EntryCallbackT entry_callback,
                    GroupCallbackT group_callback) const -> void;

  // Counts the number of keys in the hashtable that required probing beyond the
  // initial group.
  auto CountProbedKeys() const -> ssize_t;

 private:
  ViewImpl(ssize_t alloc_size, Storage* storage)
      : alloc_size_(alloc_size), storage_(storage) {}

  // Computes the offset from the metadata array to the entries array for a
  // given size. This is trivial, but we use this routine to enforce invariants
  // on the sizes.
  static constexpr auto EntriesOffset(ssize_t alloc_size) -> ssize_t {
    CARBON_DCHECK(llvm::isPowerOf2_64(alloc_size))
        << "Size must be a power of two for a hashed buffer!";
    // The size is always a power of two. We prevent any too-small sizes so it
    // being a power of two provides the needed alignment. As a result, the
    // offset is exactly the size. We validate this here to catch alignment bugs
    // early.
    CARBON_DCHECK(static_cast<uint64_t>(alloc_size) ==
                  llvm::alignTo<alignof(EntryT)>(alloc_size));
    return alloc_size;
  }

  auto metadata() const -> uint8_t* {
    return reinterpret_cast<uint8_t*>(storage_);
  }
  auto entries() const -> EntryT* {
    return reinterpret_cast<EntryT*>(reinterpret_cast<std::byte*>(storage_) +
                                     EntriesOffset(alloc_size_));
  }

  ssize_t alloc_size_;
  Storage* storage_;
};

// Implementation helper for defining a read-write base type for a hashtable
// that type-erases any SSO buffer.
//
// A specific user-facing hashtable base type should use *`protected`*
// inheritance for this type, and forward the implementation of its API to the
// protected helpers here. The derivation must be `protected` rather than
// `private` because the SSO-optimized table will in turn derive from it.
//
// The methods available to user-facing hashtable types are `protected`, and
// where relevant named with an `Impl` suffix. The suffix naming ensures types
// don't `using` in these low-level APIs but declare their own and implement
// them by forwarding to these APIs. We don't want users to have to read these
// implementation details to understand their container's API.
//
// Many method are `private` and used purely by other parts of the raw hashtable
// implementation. Where needed, friendship is used to access those.
template <typename InputKeyT, typename InputValueT>
class BaseImpl {
 protected:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewImplT = ViewImpl<KeyT, ValueT>;
  using EntryT = typename ViewImplT::EntryT;

  BaseImpl(int small_alloc_size, Storage* small_storage)
      : small_alloc_size_(small_alloc_size) {
    CARBON_CHECK(small_alloc_size >= 0);
    Construct(small_storage);
  }

  ~BaseImpl();

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewImplT() const { return view_impl(); }

  auto view_impl() const -> ViewImplT { return view_impl_; }

  // Looks up the provided key in the hashtable. If found, returns a pointer to
  // that entry and `false`.
  //
  // If not found, will locate an empty entry for inserting into, set the
  // metadata for that entry, and return a pointer to the entry and `true`. When
  // necessary, this will grow the hashtable to cause there to be sufficient
  // empty entries.
  template <typename LookupKeyT>
  auto InsertImpl(LookupKeyT lookup_key) -> std::pair<EntryT*, bool>;

  // Looks up the entry in the hashtable, and if found erases its entry and
  // returns `true`. If not found, returns `false`.
  //
  // Does not release any memory, just leaves a tombstone behind so this cannot
  // be found and the slot can in theory be re-used.
  template <typename LookupKeyT>
  auto EraseImpl(LookupKeyT lookup_key) -> bool;

  // Erases all entries in the hashtable but leaves the allocated storage.
  auto ClearImpl() -> void;

 private:
  template <typename InputBaseT, ssize_t SmallSize>
  friend class TableImpl;

  static constexpr ssize_t Alignment = std::max<ssize_t>(
      {alignof(MetadataGroup), alignof(StorageEntry<KeyT, ValueT>)});

  static constexpr auto AllocByteSize(ssize_t alloc_size) -> ssize_t {
    return ViewImplT::EntriesOffset(alloc_size) + sizeof(EntryT) * alloc_size;
  }
  static auto Allocate(ssize_t alloc_size) -> Storage*;
  static auto Deallocate(Storage* storage, ssize_t alloc_size) -> void;

  auto growth_budget() const -> ssize_t { return growth_budget_; }
  auto alloc_size() const -> ssize_t { return view_impl_.alloc_size_; }
  auto alloc_size() -> ssize_t& { return view_impl_.alloc_size_; }
  auto storage() const -> Storage* { return view_impl_.storage_; }
  auto storage() -> Storage*& { return view_impl_.storage_; }
  auto metadata() const -> uint8_t* { return view_impl_.metadata(); }
  auto entries() const -> EntryT* { return view_impl_.entries(); }
  auto small_alloc_size() const -> ssize_t {
    return static_cast<unsigned>(small_alloc_size_);
  }
  auto is_small() const -> bool { return alloc_size() <= small_alloc_size(); }

  auto Construct(Storage* small_storage) -> void;
  auto CopyFrom(const BaseImpl& arg) -> void;
  auto MoveFrom(BaseImpl&& arg) -> void;
  auto Destroy() -> void;

  template <typename LookupKeyT>
  auto InsertIntoEmpty(LookupKeyT lookup_key) -> EntryT*;

  static auto ComputeNextAllocSize(ssize_t old_alloc_size) -> ssize_t;
  static auto GrowthThresholdForAllocSize(ssize_t alloc_size) -> ssize_t;

  template <typename LookupKeyT>
  auto GrowAndInsert(LookupKeyT lookup_key) -> EntryT*;

  ViewImplT view_impl_;
  int growth_budget_;
  int small_alloc_size_;
};

// Implementation helper for defining a hashtable type with an SSO buffer.
//
// A specific user-facing hashtable should derive privately from this
// type, and forward the implementation of its interface to functions in this
// type. It should provide the corresponding user-facing hashtable base type as
// the `InputBaseT` type parameter (rather than a key/value pair), and this type
// will in turn derive from that provided base type. This allows derived-to-base
// conversion from the user-facing hashtable type to the user-facing hashtable
// base type. And it does so keeping the inheritance linear. This is similar but
// not quite the same as CRTP. The resulting linear inheritance hierarchy for a
// `Map<K, T>` type will look like:
//
//   Map<K, T>
//    ↓
//   TableImpl<MapBase<K, T>>
//    ↓
//   MapBase<K, T>
//    ↓
//   BaseImpl<K, T>
//
// The methods available to user-facing hashtable types are `protected`, and
// where relevant named with an `Impl` suffix. The suffix naming ensures types
// don't `using` in these low-level APIs but declare their own and implement
// them by forwarding to these APIs. We don't want users to have to read these
// implementation details to understand their container's API.
template <typename InputBaseT, ssize_t SmallSize>
class TableImpl : public InputBaseT {
 protected:
  using BaseT = InputBaseT;

  TableImpl() : BaseT(SmallSize, small_storage()) {
    // Specifically validate the offset of the small-size entries when we have
    // a non-zero SSO size. This needs to be in an inline body to see the
    // complete type.
    static_assert(offsetof(SmallStorage, entries) == SmallSize,
                  "Offset to entries in small size storage doesn't match "
                  "computed offset!");
  }
  TableImpl(const TableImpl& arg) : TableImpl() { this->CopyFrom(arg); }
  template <ssize_t OtherSmallSize>
  explicit TableImpl(const TableImpl<BaseT, OtherSmallSize>& arg)
      : TableImpl() {
    this->CopyFrom(arg);
  }
  TableImpl(TableImpl&& arg) noexcept : TableImpl() {
    this->MoveFrom(std::move(arg));
  }
  template <ssize_t OtherSmallSize>
  explicit TableImpl(TableImpl<BaseT, OtherSmallSize>&& arg) : TableImpl() {
    this->MoveFrom(std::move(arg));
  }

  // Resets the hashtable to its initial state, clearing all entries and
  // releasing all memory. If the hashtable had an SSO buffer, that is restored
  // as the storage. Otherwise, a minimum sized table storage is allocated.
  auto ResetImpl() -> void;

 private:
  using KeyT = BaseT::KeyT;
  using ValueT = BaseT::ValueT;

  // Do a bunch of validation of the small size to establish our invariants.
  // Using `static_assert` instead of a `requires` clause lets us associate a
  // message with each constraint.
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

  static_assert(SmallSize >= alignof(StorageEntry<KeyT, ValueT>),
                "Requested a small size that would require padding between "
                "metadata bytes and correctly aligned key and value types. "
                "Either a larger small size or a zero small size and heap "
                "allocation are required for this key and value type.");

  // A concrete implementation of storage for the provided key type, value type,
  // and small size.
  struct SmallStorage : Storage {
    alignas(BaseT::Alignment) uint8_t metadata[SmallSize];
    mutable StorageEntry<KeyT, ValueT> entries[SmallSize];
  };

  auto small_storage() const -> RawHashtable::Storage* {
    return &small_storage_;
  }

  mutable SmallStorage small_storage_;
};

////////////////////////////////////////////////////////////////////////////////
//
// Only implementation details below this point.
//
////////////////////////////////////////////////////////////////////////////////

// A template specialization for when no small size optimization buffer is
// desired. There is no API difference, only an implementation difference.
template <typename InputBaseT>
class TableImpl<InputBaseT, 0> : public InputBaseT {
 protected:
  using BaseT = InputBaseT;

  TableImpl() : BaseT(0, nullptr) {}
  TableImpl(const TableImpl& arg) : TableImpl() { this->CopyFrom(arg); }
  template <ssize_t OtherMinSmallSize>
  explicit TableImpl(const TableImpl<BaseT, OtherMinSmallSize>& arg)
      : TableImpl() {
    this->CopyFrom(arg);
  }
  TableImpl(TableImpl&& arg) noexcept : TableImpl() {
    this->MoveFrom(std::move(arg));
  }
  template <ssize_t OtherMinSmallSize>
  explicit TableImpl(TableImpl<BaseT, OtherMinSmallSize>&& arg) : TableImpl() {
    this->MoveFrom(std::move(arg));
  }

  auto ResetImpl() -> void;
};

// Computes a seed that provides a small amount of entropy from ASLR where
// available with minimal cost. The priority is speed, and this computes the
// entropy in a way that doesn't require loading from memory, merely accessing
// entropy already available without accessing memory.
inline auto ComputeSeed() -> uint64_t {
  // A global variable whose address is used as a seed. This allows ASLR to
  // introduce some variation in hashtable ordering when enabled via the code
  // model for globals.
  extern volatile std::byte global_addr_seed;

  return reinterpret_cast<uint64_t>(&global_addr_seed);
}

inline auto ComputeProbeMaskFromSize(ssize_t size) -> size_t {
  CARBON_DCHECK(llvm::isPowerOf2_64(size))
      << "Size must be a power of two for a hashed buffer!";
  // Since `size` is a power of two, we can make sure the probes are less
  // than `size` by making the mask `size - 1`. We also mask off the low
  // bits so the probes are a multiple of the size of the groups of entries.
  return (size - 1) & ~GroupMask;
}

// This class handles building a sequence of probe indices from a given
// starting point, including both the quadratic growth and masking the index
// to stay within the bucket array size. The starting point doesn't need to be
// clamped to the size ahead of time (or even be positive), we will do it
// internally.
//
// For reference on quadratic probing:
// https://en.wikipedia.org/wiki/Quadratic_probing
//
// We compute the quadratic probe index incrementally, but we can also compute
// it mathematically and will check that the incremental result matches our
// mathematical expectation. We use the quadratic probing formula of:
//
//   p(start, step) = (start + (step + step^2) / 2) (mod size / GroupSize)
//
// However, we compute it incrementally and scaled to the group size. We scale
// everything back down in order to validate that we are in fact correctly
// following the above formula.
class ProbeSequence {
 public:
  ProbeSequence(ssize_t start, ssize_t size) {
    mask_ = ComputeProbeMaskFromSize(size);
    p_ = start & mask_;
#ifndef NDEBUG
    start_ = start & mask_;
    size_ = size;
#endif
  }

  void step() {
    step_ += GroupSize;
    p_ = (p_ + step_) & mask_;
#ifndef NDEBUG
    // Verify against the quadratic formula we expect to be following by scaling
    // everything down by `GroupSize`.
    CARBON_DCHECK(
        (p_ / GroupSize) ==
        ((start_ / GroupSize +
          (step_ / GroupSize + (step_ / GroupSize) * (step_ / GroupSize)) / 2) %
         (size_ / GroupSize)))
        << "Index in probe sequence does not match the expected formula.";
    CARBON_DCHECK(step_ < size_) << "We necessarily visit all groups, so we "
                                    "can't have more probe steps than groups.";
#endif
  }

  auto index() const -> ssize_t { return p_; }

 private:
  ssize_t step_ = 0;
  size_t mask_;
  ssize_t p_;
#ifndef NDEBUG
  ssize_t start_;
  ssize_t size_;
#endif
};

// TODO: Evaluate keeping this outlined to see if macro benchmarks observe the
// same perf hit as micro benchmarks.
template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
auto ViewImpl<InputKeyT, InputValueT>::LookupEntry(LookupKeyT lookup_key) const
    -> EntryT* {
  // Prefetch with a "low" temporal locality as we're primarily expecting a
  // brief use of the storage and then to return to application code.
  __builtin_prefetch(storage_, /*read*/ 0, /*low-locality*/ 1);

  ssize_t local_size = alloc_size_;
  CARBON_DCHECK(local_size > 0);

  uint8_t* local_metadata = metadata();
  HashCode hash = HashValue(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();

  EntryT* local_entries = entries();

  // Walk through groups of entries using a quadratic probe starting from
  // `hash_index`.
  ProbeSequence s(hash_index, local_size);
  do {
    ssize_t group_index = s.index();

    // For each group, match the tag against the metadata to extract the
    // potentially matching entries within the group.
    MetadataGroup g = MetadataGroup::Load(local_metadata, group_index);
    auto metadata_matched_range = g.Match(tag);
    if (LLVM_LIKELY(metadata_matched_range)) {
      // If any entries in this group potentially match based on their metadata,
      // walk each candidate and compare its key to see if we have definitively
      // found a match.
      EntryT* group_entries = &local_entries[group_index];
      auto byte_it = metadata_matched_range.begin();
      auto byte_end = metadata_matched_range.end();
      do {
        EntryT* entry = byte_it.index_ptr(group_entries);
        if (LLVM_LIKELY(entry->key() == lookup_key)) {
          __builtin_assume(entry != nullptr);
          return entry;
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // empty slots as that indicates we're done probing -- no later probed index
    // could have a match.
    auto empty_byte_matched_range = g.MatchEmpty();
    if (LLVM_LIKELY(empty_byte_matched_range)) {
      return nullptr;
    }

    s.step();
  } while (LLVM_UNLIKELY(true));
}

template <typename InputKeyT, typename InputValueT>
template <typename EntryCallbackT, typename GroupCallbackT>
[[clang::always_inline]] auto ViewImpl<InputKeyT, InputValueT>::ForEachEntry(
    EntryCallbackT entry_callback, GroupCallbackT group_callback) const
    -> void {
  uint8_t* local_metadata = metadata();
  EntryT* local_entries = entries();

  ssize_t local_size = alloc_size_;
  for (ssize_t group_index = 0; group_index < local_size;
       group_index += GroupSize) {
    auto g = MetadataGroup::Load(local_metadata, group_index);
    auto present_matched_range = g.MatchPresent();
    if (!present_matched_range) {
      continue;
    }
    for (ssize_t byte_index : present_matched_range) {
      entry_callback(local_entries[group_index + byte_index]);
    }

    group_callback(&local_metadata[group_index]);
  }
}

template <typename InputKeyT, typename InputValueT>
auto ViewImpl<InputKeyT, InputValueT>::CountProbedKeys() const -> ssize_t {
  uint8_t* local_metadata = metadata();
  EntryT* local_entries = entries();
  ssize_t local_size = alloc_size_;
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
BaseImpl<InputKeyT, InputValueT>::~BaseImpl() {
  Destroy();
}

// TODO: Evaluate whether it is worth forcing this out-of-line given the
// reasonable ABI boundary it forms and large volume of code necessary to
// implement it.
template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
auto BaseImpl<InputKeyT, InputValueT>::InsertImpl(LookupKeyT lookup_key)
    -> std::pair<EntryT*, bool> {
  CARBON_DCHECK(alloc_size() > 0);

  uint8_t* local_metadata = metadata();

  HashCode hash = HashValue(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();

  // We re-purpose the empty control byte to signal no insert is needed to the
  // caller. This is guaranteed to not be a control byte we're inserting.
  // constexpr uint8_t NoInsertNeeded = Group::Empty;

  ssize_t group_with_deleted_index;
  MetadataGroup::MatchIndex deleted_match = {};

  EntryT* local_entries = entries();

  auto return_insert_at_index = [&](ssize_t index) -> std::pair<EntryT*, bool> {
    // We'll need to insert at this index so set the control group byte to the
    // proper value.
    local_metadata[index] = tag | MetadataGroup::PresentMask;
    return {&local_entries[index], true};
  };

  for (ProbeSequence s(hash_index, alloc_size());; s.step()) {
    ssize_t group_index = s.index();
    auto g = MetadataGroup::Load(local_metadata, group_index);

    auto control_byte_matched_range = g.Match(tag);
    if (control_byte_matched_range) {
      EntryT* group_entries = &local_entries[group_index];
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        EntryT* entry = byte_it.index_ptr(group_entries);
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
    auto empty_match = g.MatchEmpty();
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
    if (LLVM_UNLIKELY(growth_budget_ == 0)) {
      return {GrowAndInsert(lookup_key), true};
    }

    --growth_budget_;
    CARBON_DCHECK(growth_budget() >= 0)
        << "Growth budget shouldn't have gone negative!";
    return return_insert_at_index(group_index + empty_match.index());
  }

  CARBON_FATAL() << "We should never finish probing without finding the entry "
                    "or an empty slot.";
}

template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
auto BaseImpl<InputKeyT, InputValueT>::EraseImpl(LookupKeyT lookup_key)
    -> bool {
  EntryT* entry = view_impl_.LookupEntry(lookup_key);
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
  uint8_t* local_metadata = metadata();
  EntryT* local_entries = entries();
  ssize_t index = entry - local_entries;
  ssize_t group_index = index & ~GroupMask;
  auto g = MetadataGroup::Load(local_metadata, group_index);
  auto empty_matched_range = g.MatchEmpty();
  if (empty_matched_range) {
    local_metadata[index] = MetadataGroup::Empty;
    ++growth_budget_;
  } else {
    local_metadata[index] = MetadataGroup::Deleted;
  }

  if constexpr (!EntryT::IsTriviallyDestructible) {
    entry->Destroy();
  }

  return true;
}

template <typename InputKeyT, typename InputValueT>
auto BaseImpl<InputKeyT, InputValueT>::ClearImpl() -> void {
  view_impl_.ForEachEntry(
      [](EntryT& entry) {
        if constexpr (!EntryT::IsTriviallyDestructible) {
          entry.Destroy();
        }
      },
      [](uint8_t* metadata_group) {
        // Clear the group.
        std::memset(metadata_group, 0, GroupSize);
      });
  growth_budget_ = GrowthThresholdForAllocSize(alloc_size());
}

// Allocates the appropriate memory layout for a table of the given
// `alloc_size`, with space both for the metadata array and entries.
//
// The returned pointer *must* be deallocated by calling the below `Deallocate`
// function with the same `alloc_size` as used here.
template <typename InputKeyT, typename InputValueT>
auto BaseImpl<InputKeyT, InputValueT>::Allocate(ssize_t alloc_size)
    -> Storage* {
  return reinterpret_cast<Storage*>(__builtin_operator_new(
      AllocByteSize(alloc_size), static_cast<std::align_val_t>(Alignment),
      std::nothrow_t()));
}

// Deallocates a table's storage that was allocated with the `Allocate`
// function.
template <typename InputKeyT, typename InputValueT>
auto BaseImpl<InputKeyT, InputValueT>::Deallocate(Storage* storage,
                                                  ssize_t alloc_size) -> void {
  ssize_t allocated_size = AllocByteSize(alloc_size);
  // We don't need the size, but make sure it always compiles.
  static_cast<void>(allocated_size);
  __builtin_operator_delete(storage,
#if __cpp_sized_deallocation
                            allocated_size,
#endif
                            static_cast<std::align_val_t>(Alignment));
}

// Construct a table using the provided small storage if `small_size_` is
// non-zero. If `small_size_` is zero, than `small_storage` won't be used and
// can be zero. Regardless, after this the storage pointer is non-null and the
// size is non-zero so that we can directly begin inserting or querying the
// table.
template <typename InputKeyT, typename InputValueT>
auto BaseImpl<InputKeyT, InputValueT>::Construct(Storage* small_storage)
    -> void {
  if (small_alloc_size_ > 0) {
    alloc_size() = small_alloc_size_;
    storage() = small_storage;
  } else {
    // Directly allocate the initial buffer so that the hashtable is never in
    // an empty state.
    alloc_size() = MinAllocatedSize;
    storage() = Allocate(MinAllocatedSize);
  }
  std::memset(metadata(), 0, alloc_size());
  growth_budget_ = GrowthThresholdForAllocSize(alloc_size());
}

// Implementation detail for copying from an existing hashtable with the same
// key and value type.
template <typename InputKeyT, typename InputValueT>
auto BaseImpl<InputKeyT, InputValueT>::CopyFrom(const BaseImpl& arg) -> void {
  arg.view_impl_.ForEachEntry(
      [this](EntryT& arg_entry) {
        const KeyT& key = arg_entry.key();
        auto [new_entry, inserted] = InsertImpl(key);
        CARBON_CHECK(inserted) << "Duplicate insert when copying key: " << key;
        new_entry->CopyFrom(arg_entry);
      },
      [](auto...) {});
}

// Implementation details for moving from an existing hashtable with the same
// key and value type.
//
// While this is implemented generically for the base type of the hashtable, it
// correctly handles both an incoming SSO table and this table's SSO buffer,
// trying to use the SSO buffer when it can and falling back to a copy or moving
// the allocated table.
//
// Puts the incoming table into a moved-from state that can be destroyed or
// re-initialized but must not be used otherwise.
template <typename InputKeyT, typename InputValueT>
auto BaseImpl<InputKeyT, InputValueT>::MoveFrom(BaseImpl&& arg) -> void {
  // If either the incoming table is small or it would fit into our small size,
  // we move the elements but not the allocation.
  if (arg.is_small() || arg.alloc_size() <= small_alloc_size()) {
    arg.view_impl_.ForEachEntry(
        [this](EntryT& arg_entry) {
          const KeyT& key = arg_entry.key();
          auto [new_entry, inserted] = InsertImpl(key);
          CARBON_CHECK(inserted) << "Duplicate insert when moving key: " << key;
          new_entry->MoveFrom(std::move(arg_entry));
        },
        [](uint8_t* metadata_group) {
          // Clear the group so that destructors aren't run.
          std::memset(metadata_group, 0, GroupSize);
        });
    // If not small, deallocate the table storage.
    if (!arg.is_small()) {
      arg.Deallocate(arg.storage(), arg.alloc_size());
      // Replace the pointer with null to ease debugging.
      arg.storage() = nullptr;
    }
    // Put the table into a "moved from" state that will be trivially destroyed
    // but can also be re-initialized.
    arg.alloc_size() = 0;
    return;
  }

  // We need the allocated table anyways, so just setup our state to point to
  // it.
  alloc_size() = arg.alloc_size();
  storage() = arg.storage();
  growth_budget_ = arg.growth_budget_;

  // Finally, put the incoming table into a moved-from state.
  arg.alloc_size() = 0;
  // Replace the pointer with null to ease debugging.
  arg.storage() = nullptr;
}

// Destroy the current table, releasing any memory used.
template <typename InputKeyT, typename InputValueT>
auto BaseImpl<InputKeyT, InputValueT>::Destroy() -> void {
  // Check for a moved-from state and don't do anything. Only a moved-from table
  // has a zero size.
  if (alloc_size() == 0) {
    return;
  }

  // Destroy all the entries.
  if constexpr (!EntryT::IsTriviallyDestructible) {
    view_impl_.ForEachEntry([](EntryT& entry) { entry.Destroy(); },
                            [](auto...) {});
  }

  // If small, nothing to deallocate.
  if (is_small()) {
    return;
  }

  // Just deallocate the storage without updating anything when destroying the
  // object.
  Deallocate(storage(), alloc_size());
}

// Optimized routine to insert a key into a table when that key *definitely*
// isn't present in the table and the table *definitely* has a viable empty slot
// (and growth space) to insert into before any deleted slots. When both of
// these are true, typically just after growth, we can dramatically simplify the
// insert position search.
template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
[[clang::noinline]] auto BaseImpl<InputKeyT, InputValueT>::InsertIntoEmpty(
    LookupKeyT lookup_key) -> EntryT* {
  HashCode hash = HashValue(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();
  uint8_t* local_metadata = metadata();
  EntryT* local_entries = entries();

  for (ProbeSequence s(hash_index, alloc_size());; s.step()) {
    ssize_t group_index = s.index();
    auto g = MetadataGroup::Load(local_metadata, group_index);

    if (auto empty_match = g.MatchEmpty()) {
      ssize_t index = group_index + empty_match.index();
      local_metadata[index] = tag | MetadataGroup::PresentMask;
      return &local_entries[index];
    }

    // Otherwise we continue probing.
  }
}

// Apply our doubling growth strategy and (re-)check invariants around table
// size.
template <typename InputKeyT, typename InputValueT>
auto BaseImpl<InputKeyT, InputValueT>::ComputeNextAllocSize(
    ssize_t old_alloc_size) -> ssize_t {
  // We want the next power of two. This should always be a power of two coming
  // in, and so we just verify that.
  CARBON_DCHECK(old_alloc_size ==
                static_cast<ssize_t>(llvm::PowerOf2Ceil(old_alloc_size)))
      << "Expected a power of two!";
  ssize_t new_alloc_size;
  bool overflow = __builtin_mul_overflow(old_alloc_size, 2, &new_alloc_size);
  CARBON_CHECK(!overflow) << "Computing the new size overflowed `ssize_t`!";
  return new_alloc_size;
}

// Compute the growth threshold for a given size.
template <typename InputKeyT, typename InputValueT>
auto BaseImpl<InputKeyT, InputValueT>::GrowthThresholdForAllocSize(
    ssize_t alloc_size) -> ssize_t {
  // We use a 7/8ths load factor to trigger growth.
  return alloc_size - alloc_size / 8;
}

// Grow the hashtable to create space and then insert into it. Returns the
// selected insertion entry. Never returns null. In addition to growing and
// selecting the insertion entry, this routine updates the metadata array so
// that this function can be directly called and the result returned from
// `InsertImpl`.
template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
[[clang::noinline]] auto BaseImpl<InputKeyT, InputValueT>::GrowAndInsert(
    LookupKeyT lookup_key) -> EntryT* {
  // We collect the probed elements in a small vector for re-insertion. It is
  // tempting to reuse the already allocated storage, but doing so appears to
  // be a (very slight) performance regression. These are relatively rare and
  // storing them into the existing storage creates stores to the same regions
  // of memory we're reading. Moreover, it requires moving both the key and the
  // value twice, and doing the `memcpy` widening for relocatable types before
  // the group walk rather than after the group walk. In practice, between the
  // statistical rareness and using a large small size buffer here on the stack,
  // we can handle this most efficiently with temporary, additional storage.
  llvm::SmallVector<ssize_t, 128> probed_indices;

  // We grow into a new `MapBase` so that both the new and old maps are
  // fully functional until all the entries are moved over. However, we directly
  // manipulate the internals to short circuit many aspects of the growth.
  ssize_t old_size = alloc_size();
  CARBON_DCHECK(old_size > 0);
  CARBON_DCHECK(growth_budget_ == 0);

  bool old_small = is_small();
  Storage* old_storage = storage();
  uint8_t* old_metadata = metadata();
  EntryT* old_entries = entries();

#ifndef NDEBUG
  ssize_t debug_empty_count =
      llvm::count(llvm::ArrayRef(old_metadata, old_size),
                  MetadataGroup::Empty) +
      llvm::count(llvm::ArrayRef(old_metadata, old_size),
                  MetadataGroup::Deleted);
  CARBON_DCHECK(debug_empty_count >=
                (old_size - GrowthThresholdForAllocSize(old_size)))
      << "debug_empty_count: " << debug_empty_count << ", size: " << old_size;
#endif

  // Compute the new size and grow the storage in place (if possible).
  ssize_t new_size = ComputeNextAllocSize(old_size);
  alloc_size() = new_size;
  storage() = Allocate(new_size);
  growth_budget_ = GrowthThresholdForAllocSize(new_size);

  // Now extract the new components of the table.
  uint8_t* new_metadata = metadata();
  EntryT* new_entries = entries();

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
        new_entries[new_index].MoveFrom(std::move(old_entries[old_index]));
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
    EntryT* new_entry = InsertIntoEmpty(old_entries[old_index].key());
    new_entry->MoveFrom(std::move(old_entries[old_index]));
  }
  CARBON_DCHECK(count ==
                (new_size - llvm::count(llvm::ArrayRef(new_metadata, new_size),
                                        MetadataGroup::Empty)));
  growth_budget_ -= count;
  CARBON_DCHECK(growth_budget_ ==
                (GrowthThresholdForAllocSize(new_size) -
                 (new_size - llvm::count(llvm::ArrayRef(new_metadata, new_size),
                                         MetadataGroup::Empty))));
  CARBON_DCHECK(growth_budget_ > 0 &&
                "Must still have a growth budget after rehash!");

  if (!old_small) {
    // Old isn't a small buffer, so we need to deallocate it.
    Deallocate(old_storage, old_size);
  }

  // And lastly insert the lookup_key into an index in the newly grown map and
  // return that index for use.
  --growth_budget_;
  return InsertIntoEmpty(lookup_key);
}

// Reset a table to its original state, including releasing any allocated
// memory.
template <typename InputBaseT, ssize_t SmallSize>
auto TableImpl<InputBaseT, SmallSize>::ResetImpl() -> void {
  this->Destroy();

  // Re-initialize the whole thing.
  CARBON_DCHECK(this->small_alloc_size() == SmallSize);
  this->Construct(small_storage());
}

// We specialize the absence of any small size buffer to force to zero and null.
template <typename InputBaseT>
auto TableImpl<InputBaseT, 0>::ResetImpl() -> void {
  this->Destroy();

  // Re-initialize the whole thing.
  CARBON_DCHECK(this->small_alloc_size() == 0);
  this->Construct(nullptr);
}

}  // namespace Carbon::RawHashtable

#endif  // CARBON_COMMON_RAW_HASHTABLE_H_
