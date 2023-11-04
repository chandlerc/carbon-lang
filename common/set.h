// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_SET_H_
#define CARBON_COMMON_SET_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "common/hashing.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/ReverseIteration.h"
#include "llvm/Support/type_traits.h"

// Detect whether we can use SIMD accelerated implementations of the control
// groups.
#if defined(__SSSE3__)
#include <x86intrin.h>
#define CARBON_USE_X86_SIMD_CONTROL_GROUP 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define CARBON_USE_NEON_SIMD_CONTROL_GROUP 1
#endif

namespace Carbon {

template <typename KeyT>
class SetView;
template <typename KeyT>
class SetBase;
template <typename KeyT, ssize_t MinSmallSize>
class Set;

namespace SetInternal {

template <typename KeyT>
class LookupResult {
 public:
  LookupResult() = default;
  explicit LookupResult(KeyT* key) : key_(key) {}

  explicit operator bool() const { return key_ != nullptr; }

  auto key() const -> KeyT& { return *key_; }

 private:
  KeyT* key_ = nullptr;
};

template <typename KeyT>
class InsertResult {
 public:
  InsertResult() = default;
  explicit InsertResult(bool inserted, KeyT& key)
      : key_and_inserted_(&key, inserted) {}

  auto is_inserted() const -> bool { return key_and_inserted_.getInt(); }

  auto key() const -> KeyT& { return *key_and_inserted_.getPointer(); }

 private:
  llvm::PointerIntPair<KeyT*, 1, bool> key_and_inserted_;
};

template <typename MaskT, int Shift = 0, MaskT ZeroMask = 0>
class BitIndexRange {
 public:
  class Iterator
      : public llvm::iterator_facade_base<Iterator, std::forward_iterator_tag,
                                          ssize_t, ssize_t> {
   public:
    Iterator() = default;
    explicit Iterator(MaskT mask) : mask_(mask) {}

    auto operator==(const Iterator& rhs) const -> bool {
      return mask_ == rhs.mask_;
    }

    auto operator*() -> ssize_t& {
      CARBON_DCHECK(mask_ != 0) << "Cannot get an index from a zero mask!";
      __builtin_assume(mask_ != 0);
      index_ = static_cast<size_t>(llvm::countr_zero(mask_)) >> Shift;
      return index_;
    }

    auto operator++() -> Iterator& {
      CARBON_DCHECK(mask_ != 0) << "Must not increment past the end!";
      __builtin_assume(mask_ != 0);
      mask_ &= (mask_ - 1);
      return *this;
    }

   private:
    ssize_t index_;
    MaskT mask_ = 0;
  };

  BitIndexRange() = default;
  explicit BitIndexRange(MaskT mask) : mask_(mask) {}

  explicit operator bool() const { return !empty(); }
  auto empty() const -> bool {
    CARBON_DCHECK((mask_ & ZeroMask) == 0) << "Unexpected non-zero bits!";
    __builtin_assume((mask_ & ZeroMask) == 0);
    return mask_ == 0;
  }

  auto begin() const -> Iterator { return Iterator(mask_); }
  auto end() const -> Iterator { return Iterator(); }

 private:
  MaskT mask_ = 0;
};

#if CARBON_USE_X86_SIMD_CONTROL_GROUP
// An X86 SIMD optimized control group representation. This uses a 128-bit
// vector register to implement the control group. While this could also be
// expanded to 256-bit vector widths on sufficiently modern x86 processors, that
// doesn't provide an especially large performance benefit. Largely, it would
// allow increasing load factor. But a major goal is to keep the load factor and
// other benefits of the control group design while minimizing latency of
// various critical path operations, and larger control groups fundamentally
// increase the cache pressure for the critical path.
struct X86Group {
  // Each control byte can have special values. All special values have the
  // most significant bit set to distinguish them from the seven hash bits
  // stored when the control byte represents a full bucket.
  //
  // Otherwise, their values are chose primarily to provide efficient SIMD
  // implementations of the common operations on an entire control group.
  static constexpr uint8_t Empty = 0;
  static constexpr uint8_t Deleted = 1;

  using MatchedRange =
      BitIndexRange<uint32_t, /*Shift=*/0, /*ZeroMask=*/0xFFFF0000>;

  __m128i byte_vec = {};

  static auto Load(uint8_t* groups, ssize_t index) -> X86Group {
    X86Group g;
    g.byte_vec = _mm_load_si128(reinterpret_cast<__m128i*>(groups + index));
    return g;
  }

  auto Match(uint8_t match_byte) const -> MatchedRange {
    auto match_byte_vec = _mm_set1_epi8(match_byte);
    auto match_byte_cmp_vec = _mm_cmpeq_epi8(byte_vec, match_byte_vec);
    uint32_t mask = _mm_movemask_epi8(match_byte_cmp_vec);
    return MatchedRange(mask);
  }

  auto MatchEmpty() const -> MatchedRange { return Match(Empty); }

  auto MatchDeleted() const -> MatchedRange { return Match(Deleted); }

  auto MatchPresent() const -> MatchedRange {
    // We arrange the byte vector for present bytes so that we can directly
    // extract it as a mask.
    return MatchedRange(_mm_movemask_epi8(byte_vec));
  }
};
#endif

#if CARBON_USE_NEON_SIMD_CONTROL_GROUP
// An ARM NEON optimized control group. This is the same size and in fact layout
// as the portable group, but largely uses NEON operations to implement the
// logic on an 8-byte vector.
struct NeonGroup {
  // Each control byte can have special values. All special values have the
  // most significant bit set to distinguish them from the seven hash bits
  // stored when the control byte represents a full bucket.
  //
  // Otherwise, their values are chose primarily to provide efficient SIMD
  // implementations of the common operations on an entire control group.
  static constexpr uint8_t Empty = 0;
  static constexpr uint8_t Deleted = 1;

  using MatchedRange = BitIndexRange<uint64_t, /*Shift=*/3>;

  uint8x8_t byte_vec = {};

  static auto Load(uint8_t* groups, ssize_t index) -> NeonGroup {
    NeonGroup g;
    g.byte_vec = vld1_u8(groups + index);
    return g;
  }

  auto Match(uint8_t match_byte) const -> MatchedRange {
    auto match_byte_vec = vdup_n_u8(match_byte);
    auto match_byte_cmp_vec = vceq_u8(byte_vec, match_byte_vec);
    uint64_t mask = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
    return MatchedRange(mask);
  }

  auto MatchEmpty() const -> MatchedRange {
    auto match_byte_cmp_vec = vceqz_u8(byte_vec);
    uint64_t mask = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
    return MatchedRange(mask);
  }

  auto MatchDeleted() const -> MatchedRange { return Match(Deleted); }

  auto MatchPresent() const -> MatchedRange {
    static constexpr uint64_t MSBs = 0x8080'8080'8080'8080ULL;
    uint64_t mask;
    std::memcpy(&mask, &byte_vec, sizeof(byte_vec));
    mask &= MSBs;
    return MatchedRange(mask);
  }
};
#endif

struct PortableGroup {
  // Each control byte can have special values. All special values have the
  // most significant bit set to distinguish them from the seven hash bits
  // stored when the control byte represents a full bucket.
  //
  // Otherwise, their values are chose primarily to provide efficient SIMD
  // implementations of the common operations on an entire control group.
  static constexpr uint8_t Empty = 0;
  static constexpr uint8_t Deleted = 1;

  // Constants used to implement the various bit manipulation tricks.
  static constexpr uint64_t LSBs = 0x0101'0101'0101'0101ULL;
  static constexpr uint64_t MSBs = 0x8080'8080'8080'8080ULL;

  using MatchedRange = BitIndexRange<uint64_t, 3>;

  uint64_t group = {};

  static auto Load(uint8_t* groups, ssize_t index) -> PortableGroup {
    PortableGroup g;
    std::memcpy(&g.group, groups + index, sizeof(group));
    return g;
  }

  auto Match(uint8_t match_byte) const -> MatchedRange {
    // This algorithm only works for matching *present* bytes. We leverage the
    // set high bit in the present case as part of the algorithm. The whole
    // algorithm has a critical path height of 4 operations, and does 6
    // operations total:
    //
    //          group | MSBs    LSBs * match_byte
    //                 \            /
    //                 mask ^ pattern
    //                      |
    // group & MSBs    MSBs - mask
    //        \            /
    //    group_MSBs & mask
    //
    // While it is superficially similar to the "find zero bytes in a word" bit
    // math trick, it is different because this is designed to
    // have no false positives and perfectly produce 0x80 for matching bytes and
    // 0x00 for non-matching bytes. This is do-able because we constrain to only
    // handle present matches which only require testing 7 bits and have a
    // particular layout.
    CARBON_DCHECK(match_byte & 0b1000'0000)
        << llvm::formatv("{0:b}", match_byte);
    // Set the high bit of every byte to `1`. The match byte always has this bit
    // set as well, which ensures the xor below, in addition to zeroing the byte
    // that matches, also clears the high bit of every byte.
    uint64_t mask = group | MSBs;
    // Broadcast the match byte to all bytes.
    uint64_t pattern = LSBs * match_byte;
    // Xor the broadcast pattern, making matched bytes become zero bytes.
    mask = mask ^ pattern;
    // Subtract the mask bytes from `0x80` bytes so that any non-zero mask byte
    // clears the high byte but zero leaves it intact.
    mask = MSBs - mask;
    // Mask down to the high bits, but only those in the original group.
    mask &= (group & MSBs);
#ifndef NDEBUG
    const auto* group_bytes = reinterpret_cast<const uint8_t*>(&group);
    for (ssize_t byte_index : llvm::seq<ssize_t>(0, sizeof(group))) {
      uint8_t byte = (mask >> (byte_index * 8)) & 0xFF;
      if (group_bytes[byte_index] == match_byte) {
        CARBON_DCHECK(byte == 0x80)
            << "Should have a high bit set for a matched byte, found: "
            << llvm::formatv("{0:x}", byte);
      } else {
        CARBON_DCHECK(byte == 0)
            << "Should have no bits set for an unmatched byte, found: "
            << llvm::formatv("{0:x}", byte);
      }
    }
#endif
    return MatchedRange(mask);
  }

  auto MatchEmpty() const -> MatchedRange {
    // Materialize the group into a word.
    uint64_t mask = group | (group << 7);
    mask = ~mask & MSBs;
#ifndef NDEBUG
    const auto* group_bytes = reinterpret_cast<const uint8_t*>(&group);
    for (ssize_t byte_index : llvm::seq<ssize_t>(0, sizeof(group))) {
      uint8_t byte = (mask >> (byte_index * 8)) & 0xFF;
      if (group_bytes[byte_index] == Empty) {
        CARBON_DCHECK(byte == 0x80)
            << "Should have a high bit set for a matched byte, found: "
            << llvm::formatv("{0:x}", byte);
      } else {
        CARBON_DCHECK(byte == 0)
            << "Should have no bits set for an unmatched byte, found: "
            << llvm::formatv("{0:x}", byte);
      }
    }
#endif
    return MatchedRange(mask);
  }

  auto MatchDeleted() const -> MatchedRange {
    // Materialize the group into a word.
    uint64_t mask = group | (~group << 7);
    mask = ~mask & MSBs;
#ifndef NDEBUG
    const auto* group_bytes = reinterpret_cast<const uint8_t*>(&group);
    for (ssize_t byte_index : llvm::seq<ssize_t>(0, sizeof(group))) {
      uint8_t byte = (mask >> (byte_index * 8)) & 0xFF;
      if (group_bytes[byte_index] == Deleted) {
        CARBON_DCHECK(byte == 0x80)
            << "Should have a high bit set for a matched byte, found: "
            << llvm::formatv("{0:x}", byte);
      } else {
        CARBON_DCHECK(byte == 0)
            << "Should have no bits set for an unmatched byte, found: "
            << llvm::formatv("{0:x}", byte);
      }
    }
#endif
    return MatchedRange(mask);
  }

  auto MatchPresent() const -> MatchedRange {
    // Materialize the group into a word.
    uint64_t mask = group & MSBs;
#ifndef NDEBUG
    const auto* group_bytes = reinterpret_cast<const uint8_t*>(&group);
    for (ssize_t byte_index : llvm::seq<ssize_t>(0, sizeof(group))) {
      uint8_t byte = (mask >> (byte_index * 8)) & 0xFF;
      if (group_bytes[byte_index] & 0b1000'0000U) {
        CARBON_DCHECK(byte == 0x80)
            << "Should have a high bit set for a present byte, found: "
            << llvm::formatv("{0:x}", byte);
      } else {
        CARBON_DCHECK(byte == 0)
            << "Should have no bits set for a not-present byte, found: "
            << llvm::formatv("{0:x}", byte);
      }
    }
#endif
    return MatchedRange(mask);
  }
};

#if CARBON_USE_X86_SIMD_CONTROL_GROUP
using Group = X86Group;
#elif CARBON_USE_NEON_SIMD_CONTROL_GROUP
using Group = NeonGroup;
#else
using Group = PortableGroup;
#endif

constexpr ssize_t GroupSize = sizeof(Group);
static_assert(llvm::isPowerOf2_64(GroupSize),
              "The group size must be a constant power of two so dividing by "
              "it is a simple shift.");
constexpr ssize_t GroupMask = GroupSize - 1;

[[clang::always_inline]] inline void Prefetch(const void* address) {
  // Currently we just hard code a single "low" temporal locality prefetch as
  // we're primarily expecting a brief use of the storage and then to return to
  // application code.
  __builtin_prefetch(address, /*read*/ 0, /*low-locality*/ 1);
}

// We use pointers to this empty class to model the pointer to a dynamically
// allocated structure of arrays with the groups, keys, and values.
//
// This also lets us define statically allocated storage as subclasses.
struct Storage {};

template <typename KeyT>
constexpr ssize_t StorageAlignment =
    std::max<ssize_t>({GroupSize, alignof(Group), alignof(KeyT)});

template <typename KeyT>
constexpr auto ComputeKeyStorageOffset(ssize_t size) -> ssize_t {
  // There are `size` control bytes plus any alignment needed for the key type.
  return llvm::alignTo<alignof(KeyT)>(size);
}

template <typename KeyT>
constexpr auto ComputeStorageSize(ssize_t size) -> ssize_t {
  return ComputeKeyStorageOffset<KeyT>(size) + sizeof(KeyT) * size;
}

constexpr ssize_t CachelineSize = 64;

template <typename KeyT>
constexpr auto NumKeysInCacheline() -> int {
  return CachelineSize / sizeof(KeyT);
}

template <typename KeyT>
constexpr auto DefaultMinSmallSize() -> ssize_t {
  return (CachelineSize - 3 * sizeof(void*)) / sizeof(KeyT);
}

template <typename KeyT>
constexpr auto ShouldUseLinearLookup(int small_size) -> bool {
  // return false;
  return small_size >= 0 && small_size <= NumKeysInCacheline<KeyT>();
}

template <typename KeyT, ssize_t MinSmallSize>
constexpr auto ComputeSmallSize() -> ssize_t {
  constexpr ssize_t LinearSizeInPointer = sizeof(void*) / sizeof(KeyT);
  constexpr ssize_t SmallSizeFloor =
      MinSmallSize < LinearSizeInPointer ? LinearSizeInPointer : MinSmallSize;
  constexpr bool UseLinearLookup =
      ShouldUseLinearLookup<KeyT>(SmallSizeFloor);

  return UseLinearLookup ? SmallSizeFloor
                         : llvm::alignTo<GroupSize>(SmallSizeFloor);
}

template <typename KeyT, bool UseLinearLookup, ssize_t SmallSize>
struct SmallSizeStorage;

template <typename KeyT>
struct SmallSizeStorage<KeyT, true, 0> : Storage {
  SmallSizeStorage() {}
  union {
    KeyT keys[0];
  };
};

template <typename KeyT>
struct SmallSizeStorage<KeyT, false, 0> : Storage {
  SmallSizeStorage() {}
  union {
    KeyT keys[0];
  };
};

template <typename KeyT, ssize_t SmallSize>
struct SmallSizeStorage<KeyT, true, SmallSize> : Storage {
  SmallSizeStorage() {}
  union {
    KeyT keys[SmallSize];
  };
};

template <typename KeyT, ssize_t SmallSize>
struct alignas(StorageAlignment<KeyT>) SmallSizeStorage<KeyT, false, SmallSize>
    : Storage {
  SmallSizeStorage() {}

  // FIXME: One interesting question is whether the small size should be a
  // minimum here or an exact figure.
  static_assert(llvm::isPowerOf2_64(SmallSize),
                "SmallSize must be a power of two for a hashed buffer!");
  static_assert(SmallSize >= GroupSize,
                "SmallSize must be at least the size of one group!");
  static_assert((SmallSize % GroupSize) == 0,
                "SmallSize must be a multiple of the group size!");
  static constexpr ssize_t SmallNumGroups = SmallSize / GroupSize;
  static_assert(llvm::isPowerOf2_64(SmallNumGroups),
                "The number of groups must be a power of two when hashing!");

  Group groups[SmallNumGroups];

  union {
    KeyT keys[SmallSize];
  };
};

}  // namespace SetInternal

template <typename InputKeyT>
class SetView {
 public:
  using KeyT = InputKeyT;
  using LookupResultT = typename SetInternal::LookupResult<KeyT>;

  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool;

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupResultT;

  template <typename CallbackT>
  void ForEach(CallbackT callback);

 private:
  template <typename SetKeyT, ssize_t MinSmallSize>
  friend class Set;
  friend class SetBase<KeyT>;

  SetView() = default;
  SetView(ssize_t size, bool is_linear, SetInternal::Storage* storage)
      : storage_(storage) {
    SetSize(size);
    if (is_linear) {
      MakeLinear();
    } else {
      MakeNonLinear();
    }
  }

  int64_t packed_size_;
  SetInternal::Storage* storage_;

  auto size() const -> ssize_t { return static_cast<uint32_t>(packed_size_); }

  auto is_linear() const -> bool { return packed_size_ >= 0; }
  auto linear_keys() const -> KeyT* {
    assert(is_linear() && "No linear keys when not linear!");
    return reinterpret_cast<KeyT*>(storage_);
  }

  auto groups_ptr() const -> uint8_t* {
    assert(!is_linear() && "No groups when linear!");
    return reinterpret_cast<uint8_t*>(storage_);
  }
  auto keys_ptr() const -> KeyT* {
    assert(!is_linear() && "No grouped keys when linear!");
    assert(llvm::isPowerOf2_64(size()) &&
           "Size must be a power of two for a hashed buffer!");
    assert(size() == SetInternal::ComputeKeyStorageOffset<KeyT>(size()) &&
           "Cannot be more aligned than a power of two.");
    return reinterpret_cast<KeyT*>(reinterpret_cast<unsigned char*>(storage_) +
                                   size());
  }

  template <typename LookupKeyT>
  inline auto ContainsHashed(LookupKeyT lookup_key) const -> bool;
  template <typename LookupKeyT>
  inline auto LookupSmallLinear(LookupKeyT lookup_key) const -> LookupResultT;
  template <typename LookupKeyT>
  inline auto LookupHashed(LookupKeyT lookup_key) const -> LookupResultT;

  template <typename CallbackT>
  void ForEachLinear(CallbackT callback);
  template <typename KeyCallbackT, typename GroupCallbackT>
  void ForEachHashed(KeyCallbackT key_callback, GroupCallbackT group_callback);

  void SetSize(ssize_t size) {
    assert(size >= 0 && "Cannot have a negative size!");
    assert(size <= INT_MAX && "Only 32-bit sizes are supported!");
    packed_size_ &= -1ULL << 32;
    packed_size_ |= size & ((1LL << 32) - 1);
  }
  void MakeNonLinear() { packed_size_ |= -1ULL << 32; }
  void MakeLinear() { packed_size_ &= (1ULL << 32) - 1; }
};

template <typename InputKeyT>
class SetBase {
 public:
  using KeyT = InputKeyT;
  using ViewT = SetView<KeyT>;
  using LookupResultT = SetInternal::LookupResult<KeyT>;
  using InsertResultT = SetInternal::InsertResult<KeyT>;

  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool {
    return ViewT(*this).Contains(lookup_key);
  }

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupResultT {
    return ViewT(*this).Lookup(lookup_key);
  }

  template <typename CallbackT>
  void ForEach(CallbackT callback) {
    return ViewT(*this).ForEach(callback);
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const { return impl_view_; }

  template <typename LookupKeyT>
  auto Insert(LookupKeyT lookup_key,
              typename std::__type_identity<llvm::function_ref<
                  auto(LookupKeyT lookup_key, void* key_storage)->KeyT*>>::type
                  insert_cb) -> InsertResultT;

  template <typename LookupKeyT>
  auto Insert(LookupKeyT lookup_key) -> InsertResultT {
    return Insert(lookup_key,
                  [](LookupKeyT lookup_key, void* key_storage) -> KeyT* {
                    return new (key_storage) KeyT(lookup_key);
                  });
  }

  template <typename LookupKeyT>
  auto Erase(LookupKeyT lookup_key) -> bool;

  void Clear();

 protected:
  SetBase(int small_size, SetInternal::Storage* small_storage) {
    Init(small_size, small_storage);
  }
  // An internal constructor used to build temporary map base objects with a
  // specific allocated size. This is used internally to build ephemeral maps.
  explicit SetBase(ssize_t alloc_size) { InitAlloc(alloc_size); }

  ~SetBase();

  auto size() const -> ssize_t { return impl_view_.size(); }
  auto storage() const -> SetInternal::Storage* { return impl_view_.storage_; }
  auto small_size() const -> ssize_t {
    return static_cast<unsigned>(small_size_);
  }

  auto is_small() const -> bool { return size() <= small_size(); }

  auto storage() -> SetInternal::Storage*& { return impl_view_.storage_; }

  auto linear_keys() -> KeyT* { return impl_view_.linear_keys(); }

  auto groups_ptr() -> uint8_t* { return impl_view_.groups_ptr(); }
  auto keys_ptr() -> KeyT* { return impl_view_.keys_ptr(); }

  void Init(int small_size, SetInternal::Storage* small_storage);
  void InitAlloc(ssize_t alloc_size);

  template <typename LookupKeyT>
  auto InsertIndexHashed(LookupKeyT lookup_key) -> std::pair<uint32_t, ssize_t>;
  template <typename LookupKeyT>
  auto InsertIntoEmptyIndex(LookupKeyT lookup_key) -> ssize_t;
  template <typename LookupKeyT>
  auto InsertHashed(
      LookupKeyT lookup_key,
      llvm::function_ref<auto(LookupKeyT lookup_key, void* key_storage)->KeyT*>
          insert_cb) -> InsertResultT;
  template <typename LookupKeyT>
  auto InsertSmallLinear(
      LookupKeyT lookup_key,
      llvm::function_ref<auto(LookupKeyT lookup_key, void* key_storage)->KeyT*>
          insert_cb) -> InsertResultT;

  template <typename LookupKeyT>
  auto GrowRehashAndInsertIndex(LookupKeyT lookup_key) -> ssize_t;

  template <typename LookupKeyT>
  auto EraseSmallLinear(LookupKeyT lookup_key) -> bool;
  template <typename LookupKeyT>
  auto EraseHashed(LookupKeyT lookup_key) -> bool;

  ViewT impl_view_;
  int growth_budget_;
  int small_size_;
};

template <typename InputKeyT,
          ssize_t MinSmallSize =
              SetInternal::DefaultMinSmallSize<InputKeyT>()>
class Set : public SetBase<InputKeyT> {
 public:
  using KeyT = InputKeyT;
  using ViewT = SetView<KeyT>;
  using BaseT = SetBase<KeyT>;
  using LookupResultT = SetInternal::LookupResult<KeyT>;
  using InsertResultT = SetInternal::InsertResult<KeyT>;

  Set() : BaseT(SmallSize, small_storage()) {}
  Set(const Set& arg) : Set() {
    arg.ForEach([this](KeyT& k) { insert(k); });
  }
  template <ssize_t OtherMinSmallSize>
  explicit Set(const Set<KeyT, OtherMinSmallSize>& arg) : Set() {
    arg.ForEach([this](KeyT& k) { insert(k); });
  }
  Set(Set&& arg) = delete;

  void Reset();

 private:
  static constexpr ssize_t SmallSize =
      SetInternal::ComputeSmallSize<KeyT, MinSmallSize>();
  static constexpr bool UseLinearLookup =
      SetInternal::ShouldUseLinearLookup<KeyT>(SmallSize);

  static_assert(SmallSize >= 0, "Cannot have a negative small size!");

  using SmallSizeStorageT =
      SetInternal::SmallSizeStorage<KeyT, UseLinearLookup, SmallSize>;

  // Validate a collection of invariants between the small size storage layout
  // and the dynamically computed storage layout. We need to do this after both
  // are complete but in the context of a specific key type, value type, and
  // small size, so here is the best place.
  static_assert(SmallSize == 0 || UseLinearLookup ||
                    (alignof(SmallSizeStorageT) ==
                     SetInternal::StorageAlignment<KeyT>),
                "Small size buffer must have the same alignment as a heap "
                "allocated buffer.");
  static_assert(
      SmallSize == 0 ||
          (offsetof(SmallSizeStorageT, keys) ==
           (UseLinearLookup
                ? 0
                : SetInternal::ComputeKeyStorageOffset<KeyT>(SmallSize))),
      "Offset to keys in small size storage doesn't match computed offset!");
  static_assert(SmallSize == 0 || UseLinearLookup ||
                    (sizeof(SmallSizeStorageT) ==
                     SetInternal::ComputeStorageSize<KeyT>(SmallSize)),
                "The small size storage needs to match the dynamically "
                "computed storage size.");

  auto small_storage() const -> SetInternal::Storage* {
    return &small_storage_;
  }

  mutable SetInternal::SmallSizeStorage<KeyT, UseLinearLookup, SmallSize>
      small_storage_;
};

namespace SetInternal {

template <typename KeyT, typename LookupKeyT>
inline auto ContainsSmallLinear(const LookupKeyT& lookup_key, ssize_t size,
                                KeyT* keys) -> bool {
  for (ssize_t i : llvm::seq<ssize_t>(0, size)) {
    if (keys[i] == lookup_key) {
      return true;
    }
  }

  return false;
}

inline auto ComputeProbeMaskFromSize(ssize_t size) -> size_t {
  assert(llvm::isPowerOf2_64(size) &&
         "Size must be a power of two for a hashed buffer!");
  // The probe mask needs to mask down to keep the index within
  // `groups_size`. Since `groups_size` is a power of two, this is equivalent to
  // `groups_size - 1`.
  return (size - 1) & ~GroupMask;
}

/// This class handles building a sequence of probe indices from a given
/// starting point, including both the quadratic growth and masking the index
/// to stay within the bucket array size. The starting point doesn't need to be
/// clamped to the size ahead of time (or even by positive), we will do it
/// internally.
///
/// We compute the quadratic probe index incrementally, but we can also compute
/// it mathematically and will check that the incremental result matches our
/// mathematical expectation. We use the quadratic probing formula of:
///
///   p(x,s) = (x + (s + s^2)/2) mod (Size / GroupSize)
///
/// This particular quadratic sequence will visit every value modulo the
/// provided size divided by the group size.
///
/// However, we compute it scaled to the group size constant G and have it visit
/// each G multiple modulo the size using the scaled formula:
///
///   p(x,s) = (x + (s + (s * s * G)/(G * G))/2) mod Size
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
    assert(i == ((Start + ((Step + (Step * Step * GroupSize) /
                                       (GroupSize * GroupSize)) /
                           2)) %
                 Size) &&
           "Index in probe sequence does not match the expected formula.");
    assert(Step < Size &&
           "We necessarily visit all groups, so we can't have "
           "more probe steps than groups.");
  }

  auto getIndex() const -> ssize_t { return i; }
};

inline auto ComputeControlByte(size_t hash) -> uint8_t {
  // Mask one over the high bit so that engaged control bytes are easily
  // identified.
  return (hash >> (sizeof(hash) * 8 - 7)) | 0b10000000;
}

inline auto ComputeHashIndex(size_t hash, const void* ptr) -> ssize_t {
  return hash ^ reinterpret_cast<uintptr_t>(ptr);
}

template <typename KeyT, typename LookupKeyT>
[[clang::noinline]] auto LookupIndexHashed(LookupKeyT lookup_key, ssize_t size,
                                           Storage* storage) -> ssize_t {
  uint8_t* groups = reinterpret_cast<uint8_t*>(storage);
  size_t hash = static_cast<uint64_t>(HashValue(lookup_key));
  uint8_t control_byte = ComputeControlByte(hash);
  ssize_t hash_index = ComputeHashIndex(hash, groups);

  KeyT* keys =
      reinterpret_cast<KeyT*>(reinterpret_cast<unsigned char*>(storage) + size);
  ProbeSequence s(hash_index, size);
  do {
    ssize_t group_index = s.getIndex();
    Group g = Group::Load(groups, group_index);
    auto control_byte_matched_range = g.Match(control_byte);
    if (LLVM_LIKELY(control_byte_matched_range)) {
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        ssize_t index = group_index + *byte_it;
        if (LLVM_LIKELY(keys[index] == lookup_key)) {
          __builtin_assume(index >= 0);
          return index;
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // empty slots and we're done probing.
    auto empty_byte_matched_range = g.MatchEmpty();
    if (LLVM_LIKELY(empty_byte_matched_range)) {
      return -1;
    }

    s.step();
  } while (LLVM_UNLIKELY(true));
}

}  // namespace SetInternal

template <typename KT>
template <typename LookupKeyT>
inline auto SetView<KT>::ContainsHashed(LookupKeyT lookup_key) const
    -> bool {
  return SetInternal::LookupIndexHashed<KeyT>(lookup_key, size(), storage_) >=
         0;
}

template <typename KT>
template <typename LookupKeyT>
inline auto SetView<KT>::LookupSmallLinear(LookupKeyT lookup_key) const
    -> LookupResultT {
  KeyT* key = linear_keys();
  KeyT* key_end = &key[size()];
  do {
    if (*key == lookup_key) {
      return LookupResultT(key);
    }
    ++key;
  } while (key < key_end);

  return LookupResultT();
}

template <typename KT>
template <typename LookupKeyT>
inline auto SetView<KT>::LookupHashed(LookupKeyT lookup_key) const
    -> LookupResultT {
  ssize_t index =
      SetInternal::LookupIndexHashed<KeyT>(lookup_key, size(), storage_);
  if (index < 0) {
    return LookupResultT();
  }

  return LookupResultT(&keys_ptr()[index]);
}

template <typename KT>
template <typename LookupKeyT>
auto SetView<KT>::Contains(LookupKeyT lookup_key) const -> bool {
  SetInternal::Prefetch(storage_);
  if (is_linear()) {
    return SetInternal::ContainsSmallLinear<KeyT>(lookup_key, size(),
                                                  linear_keys());
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return ContainsHashed(lookup_key);
}

template <typename KT>
template <typename LookupKeyT>
auto SetView<KT>::Lookup(LookupKeyT lookup_key) const -> LookupResultT {
  SetInternal::Prefetch(storage_);
  if (is_linear()) {
    return LookupSmallLinear(lookup_key);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return LookupHashed(lookup_key);
}

template <typename KeyT>
template <typename CallbackT>
void SetView<KeyT>::ForEachLinear(CallbackT callback) {
  KeyT* keys = linear_keys();
  for (ssize_t i : llvm::seq<ssize_t>(0, size())) {
    callback(keys[i]);
  }
}

template <typename KeyT>
template <typename KeyCallbackT, typename GroupCallbackT>
[[clang::always_inline]] void SetView<KeyT>::ForEachHashed(
    KeyCallbackT key_callback, GroupCallbackT group_callback) {
  uint8_t* groups = groups_ptr();
  KeyT* keys = keys_ptr();

  ssize_t local_size = size();
  for (ssize_t group_index = 0; group_index < local_size;
       group_index += SetInternal::GroupSize) {
    auto g = SetInternal::Group::Load(groups, group_index);
    auto present_matched_range = g.MatchPresent();
    if (!present_matched_range) {
      continue;
    }
    for (ssize_t byte_index : present_matched_range) {
      ssize_t index = group_index + byte_index;
      key_callback(keys[index]);
    }

    group_callback(groups, group_index);
  }
}

template <typename KT>
template <typename CallbackT>
void SetView<KT>::ForEach(CallbackT callback) {
  if (is_linear()) {
    ForEachLinear(callback);
    return;
  }

  // Otherwise walk the non-empty slots in each control group.
  ForEachHashed(callback, [](auto...) {});
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
template <typename KT>
template <typename LookupKeyT>
[[clang::noinline]] auto SetBase<KT>::InsertIndexHashed(
    LookupKeyT lookup_key) -> std::pair<uint32_t, ssize_t> {
  uint8_t* groups = groups_ptr();

  size_t hash = static_cast<uint64_t>(HashValue(lookup_key));
  uint8_t control_byte = SetInternal::ComputeControlByte(hash);
  ssize_t hash_index = SetInternal::ComputeHashIndex(hash, groups);

  ssize_t group_with_deleted_index = -1;
  SetInternal::Group::MatchedRange deleted_matched_range;

  auto return_insert_at_index = [&](ssize_t index) -> std::pair<bool, ssize_t> {
    // We'll need to insert at this index so set the control group byte to the
    // proper value.
    groups[index] = control_byte;
    return {/*needs_insertion=*/true, index};
  };

  for (SetInternal::ProbeSequence s(hash_index, size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = SetInternal::Group::Load(groups, group_index);

    auto control_byte_matched_range = g.Match(control_byte);
    if (control_byte_matched_range) {
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        ssize_t index = group_index + *byte_it;
        if (LLVM_LIKELY(keys_ptr()[index] == lookup_key)) {
          return {/*needs_insertion=*/false, index};
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // Track the first group with a deleted entry that we could insert over.
    if (group_with_deleted_index < 0) {
      deleted_matched_range = g.MatchDeleted();
      if (deleted_matched_range) {
        group_with_deleted_index = group_index;
      }
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // no empty slots. In that case, we'll continue probing.
    auto empty_matched_range = g.MatchEmpty();
    if (LLVM_LIKELY(!empty_matched_range)) {
      continue;
    }

    // Ok, we've finished probing without finding anything and need to insert
    // instead.
    if (LLVM_UNLIKELY(group_with_deleted_index >= 0)) {
      // If we found a deleted slot, we don't need the probe sequence to insert
      // so just bail.
      break;
    }

    // Otherwise, we're going to need to grow by inserting over one of these
    // empty slots. Check that we have the budget for that before we compute the
    // exact index of the empty slot. Without the growth budget we'll have to
    // completely rehash and so we can just bail here.
    if (LLVM_UNLIKELY(growth_budget_ == 0)) {
      // Without room to grow, return that no group is viable but also set the
      // index to be negative. This ensures that a positive index is always
      // sufficient to determine that an existing was found.
      return {/*needs_insertion=*/true, -1};
    }

    return return_insert_at_index(group_index + *empty_matched_range.begin());
  }

  return return_insert_at_index(group_with_deleted_index +
                                *deleted_matched_range.begin());
}

template <typename KT>
template <typename LookupKeyT>
[[clang::noinline]] auto SetBase<KT>::InsertIntoEmptyIndex(
    LookupKeyT lookup_key) -> ssize_t {
  size_t hash = static_cast<uint64_t>(HashValue(lookup_key));
  uint8_t control_byte = SetInternal::ComputeControlByte(hash);
  uint8_t* groups = groups_ptr();
  ssize_t hash_index = SetInternal::ComputeHashIndex(hash, groups);

  for (SetInternal::ProbeSequence s(hash_index, size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = SetInternal::Group::Load(groups, group_index);

    if (auto empty_matched_range = g.MatchEmpty()) {
      ssize_t index = group_index + *empty_matched_range.begin();
      groups[index] = control_byte;
      return index;
    }

    // Otherwise we continue probing.
  }
}

namespace SetInternal {

template <typename KeyT>
inline auto AllocateStorage(ssize_t size) -> Storage* {
  ssize_t allocated_size = ComputeStorageSize<KeyT>(size);
  return reinterpret_cast<Storage*>(__builtin_operator_new(
      allocated_size, std::align_val_t(StorageAlignment<KeyT>),
      std::nothrow_t()));
}

template <typename KeyT>
inline void DeallocateStorage(Storage* storage, ssize_t size) {
#if __cpp_sized_deallocation
  ssize_t allocated_size = computeStorageSize<KeyT>(size);
  return __builtin_operator_delete(
      storage, allocated_size,
      std::align_val_t(StorageAlignment<KeyT>));
#else
  // Ensure `size` is used even in the fallback non-sized deallocation case.
  (void)size;
  return __builtin_operator_delete(
      storage, std::align_val_t(StorageAlignment<KeyT>));
#endif
}

inline auto ComputeNewSize(ssize_t old_size) -> ssize_t {
  if (old_size < (4 * GroupSize)) {
    // If we're going to heap allocate, get at least four groups.
    return 4 * GroupSize;
  }

  // Otherwise, we want the next power of two. This should always be a power of
  // two coming in, and so we just verify that. Also verify that this doesn't
  // overflow.
  assert(old_size == (ssize_t)llvm::PowerOf2Ceil(old_size) &&
         "Expected a power of two!");
  return old_size * 2;
}

inline auto GrowthThresholdForSize(ssize_t size) -> ssize_t {
  // We use a 7/8ths load factor to trigger growth.
  return size - size / 8;
}

}  // namespace SetInternal

template <typename KeyT>
template <typename LookupKeyT>
[[clang::noinline]] auto SetBase<KeyT>::GrowRehashAndInsertIndex(
    LookupKeyT lookup_key) -> ssize_t {
  // We grow into a new `MapBase` so that both the new and old maps are
  // fully functional until all the entries are moved over. However, we directly
  // manipulate the internals to short circuit many aspects of the growth.
  SetBase<KeyT> new_map(SetInternal::ComputeNewSize(size()));

  // We specially handle the linear and small case to make it easy to optimize
  // that.
  if (LLVM_LIKELY(impl_view_.is_linear())) {
    impl_view_.ForEachLinear([&](KeyT& old_key) {
      ssize_t index = new_map.InsertIntoEmptyIndex(old_key);
      KeyT* new_keys = new_map.keys_ptr();
      new (&new_keys[index]) KeyT(std::move(old_key));
      old_key.~KeyT();
    });
    assert(new_map.growth_budget_ > size() &&
           "Must still have a growth budget after rehash!");
    new_map.growth_budget_ -= size();
    assert(is_small() && "Should only have linear scans in the small mode!");
  } else {
    ssize_t insert_count = 0;
    impl_view_.ForEachHashed(
        [&](KeyT& old_key) {
          ++insert_count;
          ssize_t index = new_map.InsertIntoEmptyIndex(old_key);
          KeyT* new_keys = new_map.keys_ptr();
          new (&new_keys[index]) KeyT(std::move(old_key));
          old_key.~KeyT();
        },
        [](auto...) {});
    new_map.growth_budget_ -= insert_count;
    assert(new_map.growth_budget_ >= 0 &&
           "Must still have a growth budget after rehash!");

    if (LLVM_LIKELY(!is_small())) {
      // Old isn't a small buffer, so we need to deallocate it.
      SetInternal::DeallocateStorage<KeyT>(storage(), size());
    }
  }

  // Now that we've fully built the new, grown structures, replace the entries
  // in the data structure. At this point we can be certain to not clobber
  // anything aliasing a small buffer.
  impl_view_ = new_map.impl_view_;
  growth_budget_ = new_map.growth_budget_;

  // Mark the map as non-linear.
  impl_view_.MakeNonLinear();

  // Prevent the ephemeral new map object from doing anything when destroyed as
  // we've taken over it's internals.
  new_map.storage() = nullptr;
  new_map.impl_view_.SetSize(0);

  // And lastly insert the lookup_key into an index in the newly grown map and
  // return that index for use.
  --growth_budget_;
  return InsertIntoEmptyIndex(lookup_key);
}

template <typename KeyT>
template <typename LookupKeyT>
auto SetBase<KeyT>::InsertHashed(
    LookupKeyT lookup_key,
    llvm::function_ref<auto(LookupKeyT lookup_key, void* key_storage)->KeyT*>
        insert_cb) -> InsertResultT {
  ssize_t index = -1;
  // Try inserting if we have storage at all.
  if (size() > 0) {
    bool needs_insertion;
    std::tie(needs_insertion, index) = InsertIndexHashed(lookup_key);
    if (LLVM_LIKELY(!needs_insertion)) {
      assert(index >= 0 &&
             "Must have a valid group when we find an existing entry.");
      return InsertResultT(false, keys_ptr()[index]);
    }
  }

  if (index < 0) {
    assert(
        growth_budget_ == 0 &&
        "Shouldn't need to grow the table until we exhaust our growth budget!");

    index = GrowRehashAndInsertIndex(lookup_key);
  } else {
    assert(growth_budget_ >= 0 && "Cannot insert with zero budget!");
    --growth_budget_;
  }

  assert(index >= 0 && "Should have a group to insert into now.");

  KeyT* k = insert_cb(lookup_key, &keys_ptr()[index]);
  return InsertResultT(true, *k);
}

template <typename KeyT>
template <typename LookupKeyT>
auto SetBase<KeyT>::InsertSmallLinear(
    LookupKeyT lookup_key,
    llvm::function_ref<auto(
        LookupKeyT lookup_key, void* key_storage)->KeyT*>
        insert_cb) -> InsertResultT {
  KeyT* keys = linear_keys();
  for (ssize_t i : llvm::seq<ssize_t>(0, size())) {
    if (keys[i] == lookup_key) {
      return InsertResultT(false, keys[i]);
    }
  }

  ssize_t old_size = size();
  ssize_t index = old_size;

  // We need to insert. First see if we have space.
  if (old_size < small_size()) {
    // We can do the easy linear insert, just increment the size.
    impl_view_.SetSize(old_size + 1);
  } else {
    // No space for a linear insert so grow into a hash table and then do
    // a hashed insert.
    index = GrowRehashAndInsertIndex(lookup_key);
    keys = keys_ptr();
  }
  KeyT* k = insert_cb(lookup_key, &keys[index]);
  return InsertResultT(true, *k);
}

template <typename KT>
template <typename LookupKeyT>
auto SetBase<KT>::Insert(
    LookupKeyT lookup_key,
    typename std::__type_identity<llvm::function_ref<
        auto(LookupKeyT lookup_key, void* key_storage)->KeyT*>>::type insert_cb)
    -> InsertResultT {
  if (impl_view_.is_linear()) {
    return InsertSmallLinear(lookup_key, insert_cb);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return InsertHashed(lookup_key, insert_cb);
}

template <typename KeyT>
template <typename LookupKeyT>
auto SetBase<KeyT>::EraseSmallLinear(LookupKeyT lookup_key) -> bool {
  KeyT* keys = linear_keys();
  for (ssize_t i : llvm::seq<ssize_t>(0, size())) {
    if (keys[i] == lookup_key) {
      // Found the key, so clobber this entry with the last one and decrease
      // the size by one.
      impl_view_.SetSize(size() - 1);
      keys[i] = std::move(keys[size()]);
      keys[size()].~KeyT();
      return true;
    }
  }

  return false;
}

template <typename KeyT>
template <typename LookupKeyT>
auto SetBase<KeyT>::EraseHashed(LookupKeyT lookup_key) -> bool {
  ssize_t index =
      SetInternal::LookupIndexHashed<KeyT>(lookup_key, size(), storage());
  if (index < 0) {
    return false;
  }

  KeyT* keys = keys_ptr();
  keys[index].~KeyT();

  // If there are empty slots in this group then nothing will probe past this
  // group looking for an entry so we can simply set this slot to empty as
  // well. However, if every slot in this group is full, it might be part of
  // a long probe chain that we can't disrupt. In that case we mark the group
  // as deleted to keep probes continuing past it.
  //
  // If we mark the slot as empty, we'll also need to increase the growth
  // budget.
  uint8_t* groups = groups_ptr();
  ssize_t group_index = index & ~SetInternal::GroupMask;
  auto g = SetInternal::Group::Load(groups, group_index);
  auto empty_matched_range = g.MatchEmpty();
  if (empty_matched_range) {
    groups[index] = SetInternal::Group::Empty;
    ++growth_budget_;
  } else {
    groups[index] = SetInternal::Group::Deleted;
  }
  return true;
}

template <typename KeyT>
template <typename LookupKeyT>
auto SetBase<KeyT>::Erase(LookupKeyT lookup_key) -> bool {
  if (impl_view_.is_linear()) {
    return EraseSmallLinear(lookup_key);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return EraseHashed(lookup_key);
}

template <typename KeyT>
void SetBase<KeyT>::Clear() {
  auto destroy_cb = [](KeyT& k) {
    // Destroy this key.
    k.~KeyT();
  };

  if (impl_view_.is_linear()) {
    // Destroy all the keys and values.
    impl_view_.ForEachLinear(destroy_cb);

    // Now reset the size to zero and we'll start again inserting into the
    // beginning of the small linear buffer.
    impl_view_.SetSize(0);
    return;
  }

  // Otherwise walk the non-empty slots in the control group destroying each
  // one and clearing out the group.
  impl_view_.ForEachHashed(
      destroy_cb, [](uint8_t* groups, ssize_t group_index) {
        // Clear the group.
        std::memset(groups + group_index, 0, SetInternal::GroupSize);
      });

  // And reset the growth budget.
  growth_budget_ = SetInternal::GrowthThresholdForSize(size());
}

template <typename KeyT>
SetBase<KeyT>::~SetBase() {
  // Nothing to do when in the un-allocated and unused state.
  if (size() == 0) {
    return;
  }

  // Destroy all the keys and values.
  ForEach([](KeyT& k) {
    k.~KeyT();
  });

  // If small, nothing to deallocate.
  if (is_small()) {
    return;
  }

  // Just deallocate the storage without updating anything when destroying the
  // object.
  SetInternal::DeallocateStorage<KeyT>(storage(), size());
}

template <typename KeyT>
void SetBase<KeyT>::Init(int small_size_arg,
                                 SetInternal::Storage* small_storage) {
  storage() = small_storage;
  small_size_ = small_size_arg;

  if (SetInternal::ShouldUseLinearLookup<KeyT>(small_size_arg)) {
    // We use size to mean empty when doing linear lookups.
    impl_view_.SetSize(0);
    impl_view_.MakeLinear();
  } else {
    // We're not using linear lookups in the small size, so initialize it as
    // an initial hash table.
    impl_view_.SetSize(small_size_arg);
    impl_view_.MakeNonLinear();
    growth_budget_ = SetInternal::GrowthThresholdForSize(small_size_arg);
    std::memset(groups_ptr(), 0, small_size_arg);
  }
}

template <typename KeyT>
void SetBase<KeyT>::InitAlloc(ssize_t alloc_size) {
  assert(alloc_size > 0 && "Can only allocate positive size tables!");
  impl_view_.SetSize(alloc_size);
  impl_view_.MakeNonLinear();
  storage() = SetInternal::AllocateStorage<KeyT>(alloc_size);
  std::memset(groups_ptr(), 0, alloc_size);
  growth_budget_ = SetInternal::GrowthThresholdForSize(alloc_size);
  small_size_ = 0;
}

template <typename KeyT, ssize_t MinSmallSize>
void Set<KeyT, MinSmallSize>::Reset() {
  // Nothing to do when in the un-allocated and unused state.
  if (this->size() == 0) {
    return;
  }

  // If in the small rep, just clear the objects.
  if (this->is_small()) {
    this->Clear();
    return;
  }

  // Otherwise do the first part of the clear to destroy all the elements.
  this->ForEach([](KeyT& k) {
    k.~KeyT();
  });

  // Deallocate the buffer.
  SetInternal::DeallocateStorage<KeyT>(this->storage(), this->size());

  // Re-initialize the whole thing.
  this->Init(SmallSize, small_storage());
}

}  // namespace Carbon

#endif  // CARBON_COMMON_SET_H_
