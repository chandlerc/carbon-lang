// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_MAP_H_
#define CARBON_COMMON_MAP_H_

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
#include "common/set.h"
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

namespace Carbon {

template <typename KeyT, typename ValueT>
class MapView;
template <typename KeyT, typename ValueT>
class MapBase;
template <typename KeyT, typename ValueT, ssize_t MinSmallSize>
class Map;

namespace MapInternal {

template <typename KeyT, typename ValueT>
class LookupKVResult : public SetInternal::LookupResult<KeyT> {
 public:
  LookupKVResult() = default;
  LookupKVResult(KeyT* key, ValueT* value)
      : SetInternal::LookupResult<KeyT>(key), value_(value) {}

  auto value() const -> ValueT& { return *value_; }

 private:
  ValueT* value_ = nullptr;
};

template <typename KeyT, typename ValueT>
class InsertKVResult : public SetInternal::InsertResult<KeyT> {
 public:
  InsertKVResult() = default;
  InsertKVResult(bool inserted, KeyT& key, ValueT& value)
      : SetInternal::InsertResult<KeyT>(inserted, key), value_(&value) {}

  auto value() const -> ValueT& { return *value_; }

 private:
  ValueT* value_ = nullptr;
};

using Storage = SetInternal::Storage;

template <typename KeyT, typename ValueT>
constexpr ssize_t StorageAlignment =
    std::max<ssize_t>({SetInternal::StorageAlignment<KeyT>, alignof(ValueT)});

template <typename KeyT, typename ValueT>
constexpr auto ComputeValueStorageOffset(ssize_t size) -> ssize_t {
  // Skip the keys themselves.
  ssize_t offset = sizeof(KeyT) * size;

  // If the value type alignment is smaller than the key's, we're done.
  if constexpr (alignof(ValueT) <= alignof(KeyT)) {
    return offset;
  }

  // Otherwise, skip the alignment for the value type.
  return llvm::alignTo<alignof(ValueT)>(offset);
}

template <typename KeyT, typename ValueT>
constexpr auto ComputeStorageSize(ssize_t size) -> ssize_t {
  return SetInternal::ComputeKeyStorageOffset<KeyT>(size) +
         ComputeValueStorageOffset<KeyT, ValueT>(size) + sizeof(ValueT) * size;
}

constexpr ssize_t CachelineSize = 64;

template <typename KeyT>
constexpr auto NumKeysInCacheline() -> int {
  return CachelineSize / sizeof(KeyT);
}

template <typename KeyT, typename ValueT>
constexpr auto DefaultMinSmallSize() -> ssize_t {
  return (CachelineSize - 3 * sizeof(void*)) / (sizeof(KeyT) + sizeof(ValueT));
}

template <typename KeyT, typename ValueT, ssize_t SmallSize>
struct SmallSizeStorage;

template <typename KeyT, typename ValueT>
struct SmallSizeStorage<KeyT, ValueT, 0> : SetInternal::SmallSizeStorage<KeyT, 0> {
  SmallSizeStorage() {}
  union {
    ValueT values[0];
  };
};

template <typename KeyT, typename ValueT, ssize_t SmallSize>
struct alignas(StorageAlignment<KeyT, ValueT>) SmallSizeStorage
    : SetInternal::SmallSizeStorage<KeyT, SmallSize> {
  SmallSizeStorage() {}

  union {
    ValueT values[SmallSize];
  };
};

}  // namespace MapInternal

template <typename InputKeyT, typename InputValueT>
class MapView {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using LookupKVResultT = typename MapInternal::LookupKVResult<KeyT, ValueT>;

  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool;

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupKVResultT;

  template <typename LookupKeyT>
  auto operator[](LookupKeyT lookup_key) const -> ValueT*;

  template <typename CallbackT>
  void ForEach(CallbackT callback);

 private:
  template <typename MapKeyT, typename MapValueT, ssize_t MinSmallSize>
  friend class Map;
  friend class MapBase<KeyT, ValueT>;

  MapView() = default;
  MapView(ssize_t size, MapInternal::Storage* storage)
      : size_(size), storage_(storage) {
  }

  auto size() const -> ssize_t { return size_; }

  auto groups_ptr() const -> uint8_t* {
    return reinterpret_cast<uint8_t*>(storage_);
  }
  auto keys_ptr() const -> KeyT* {
    assert(llvm::isPowerOf2_64(size()) &&
           "Size must be a power of two for a hashed buffer!");
    assert(size() == SetInternal::ComputeKeyStorageOffset<KeyT>(size()) &&
           "Cannot be more aligned than a power of two.");
    return reinterpret_cast<KeyT*>(reinterpret_cast<unsigned char*>(storage_) +
                                   size());
  }
  auto values_ptr() const -> ValueT* {
    return reinterpret_cast<ValueT*>(
        reinterpret_cast<unsigned char*>(keys_ptr()) +
        MapInternal::ComputeValueStorageOffset<KeyT, ValueT>(size()));
  }

  template <typename LookupKeyT>
  inline auto ContainsHashed(LookupKeyT lookup_key) const -> bool;
  template <typename LookupKeyT>
  inline auto LookupHashed(LookupKeyT lookup_key) const -> LookupKVResultT;

  template <typename KVCallbackT, typename GroupCallbackT>
  void ForEachHashed(KVCallbackT kv_callback, GroupCallbackT group_callback);

  ssize_t size_;
  MapInternal::Storage* storage_;
};

template <typename InputKeyT, typename InputValueT>
class MapBase {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using LookupKVResultT = MapInternal::LookupKVResult<KeyT, ValueT>;
  using InsertKVResultT = MapInternal::InsertKVResult<KeyT, ValueT>;

  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool {
    return ViewT(*this).Contains(lookup_key);
  }

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupKVResultT {
    return ViewT(*this).Lookup(lookup_key);
  }

  template <typename LookupKeyT>
  auto operator[](LookupKeyT lookup_key) const -> ValueT* {
    return ViewT(*this)[lookup_key];
  }

  template <typename CallbackT>
  void ForEach(CallbackT callback) {
    return ViewT(*this).ForEach(callback);
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const { return impl_view_; }

  template <typename LookupKeyT>
  auto Insert(
      LookupKeyT lookup_key,
      typename std::__type_identity<llvm::function_ref<
          std::pair<KeyT*, ValueT*>(LookupKeyT lookup_key, void* key_storage,
                                    void* value_storage)>>::type insert_cb)
      -> InsertKVResultT;

  template <typename LookupKeyT>
  auto Insert(LookupKeyT lookup_key, ValueT new_v) -> InsertKVResultT {
    return Insert(lookup_key,
                  [&new_v](LookupKeyT lookup_key, void* key_storage,
                           void* value_storage) -> std::pair<KeyT*, ValueT*> {
                    KeyT* k = new (key_storage) KeyT(lookup_key);
                    auto* v = new (value_storage) ValueT(std::move(new_v));
                    return {k, v};
                  });
  }

  template <typename LookupKeyT, typename ValueCallbackT>
  auto Insert(LookupKeyT lookup_key, ValueCallbackT value_cb) ->
      typename std::enable_if<
          !std::is_same<ValueT, ValueCallbackT>::value &&
              std::is_same<ValueT,
                           decltype(std::declval<ValueCallbackT>()())>::value,

          InsertKVResultT>::type {
    return Insert(
        lookup_key,
        [&value_cb](LookupKeyT lookup_key, void* key_storage,
                    void* value_storage) -> std::pair<KeyT*, ValueT*> {
          KeyT* k = new (key_storage) KeyT(lookup_key);
          auto* v = new (value_storage) ValueT(value_cb());
          return {k, v};
        });
  }

  template <typename LookupKeyT>
  auto Update(
      LookupKeyT lookup_key,
      typename std::__type_identity<llvm::function_ref<
          std::pair<KeyT*, ValueT*>(LookupKeyT lookup_key, void* key_storage,
                                    void* value_storage)>>::type insert_cb,
      llvm::function_ref<ValueT&(KeyT& key, ValueT& value)> update_cb)
      -> InsertKVResultT;

  template <typename LookupKeyT, typename ValueCallbackT>
  auto Update(LookupKeyT lookup_key, ValueCallbackT value_cb) ->
      typename std::enable_if<
          !std::is_same<ValueT, ValueCallbackT>::value &&
              std::is_same<ValueT,
                           decltype(std::declval<ValueCallbackT>()())>::value,

          InsertKVResultT>::type {
    return Update(
        lookup_key,
        [&value_cb](LookupKeyT lookup_key, void* key_storage,
                    void* value_storage) -> std::pair<KeyT*, ValueT*> {
          KeyT* k = new (key_storage) KeyT(lookup_key);
          auto* v = new (value_storage) ValueT(value_cb());
          return {k, v};
        },
        [&value_cb](KeyT& /*Key*/, ValueT& value) -> ValueT& {
          value.~ValueT();
          return *new (&value) ValueT(value_cb());
        });
  }

  template <typename LookupKeyT>
  auto Update(LookupKeyT lookup_key, ValueT new_v) -> InsertKVResultT {
    return Update(
        lookup_key,
        [&new_v](LookupKeyT lookup_key, void* key_storage,
                 void* value_storage) -> std::pair<KeyT*, ValueT*> {
          KeyT* k = new (key_storage) KeyT(lookup_key);
          auto* v = new (value_storage) ValueT(std::move(new_v));
          return {k, v};
        },
        [&new_v](KeyT& /*Key*/, ValueT& value) -> ValueT& {
          value.~ValueT();
          return *new (&value) ValueT(std::move(new_v));
        });
  }

  template <typename LookupKeyT>
  auto Erase(LookupKeyT lookup_key) -> bool;

  void Clear();

 protected:
  MapBase(int small_size, MapInternal::Storage* small_storage) {
    Init(small_size, small_storage);
  }
  // An internal constructor used to build temporary map base objects with a
  // specific allocated size. This is used internally to build ephemeral maps.
  explicit MapBase(ssize_t alloc_size) { InitAlloc(alloc_size); }

  ~MapBase();

  auto size() const -> ssize_t { return impl_view_.size(); }
  auto storage() const -> MapInternal::Storage* { return impl_view_.storage_; }
  auto small_size() const -> ssize_t {
    return static_cast<unsigned>(small_size_);
  }

  auto is_small() const -> bool { return size() <= small_size(); }

  auto storage() -> MapInternal::Storage*& { return impl_view_.storage_; }

  auto groups_ptr() -> uint8_t* { return impl_view_.groups_ptr(); }
  auto keys_ptr() -> KeyT* { return impl_view_.keys_ptr(); }
  auto values_ptr() -> ValueT* { return impl_view_.values_ptr(); }

  void Init(int small_size, MapInternal::Storage* small_storage);
  void InitAlloc(ssize_t alloc_size);

  template <typename LookupKeyT>
  auto InsertIndexHashed(LookupKeyT lookup_key) -> std::pair<uint32_t, ssize_t>;
  template <typename LookupKeyT>
  auto InsertIntoEmptyIndex(LookupKeyT lookup_key) -> ssize_t;
  template <typename LookupKeyT>
  auto InsertHashed(
      LookupKeyT lookup_key,
      llvm::function_ref<std::pair<KeyT*, ValueT*>(
          LookupKeyT lookup_key, void* key_storage, void* value_storage)>
          insert_cb) -> InsertKVResultT;

  template <typename LookupKeyT>
  auto GrowRehashAndInsertIndex(LookupKeyT lookup_key) -> ssize_t;

  template <typename LookupKeyT>
  auto EraseHashed(LookupKeyT lookup_key) -> bool;

  ViewT impl_view_;
  int growth_budget_;
  int small_size_;
};

template <typename InputKeyT, typename InputValueT,
          ssize_t MinSmallSize =
              MapInternal::DefaultMinSmallSize<InputKeyT, InputValueT>()>
class Map : public MapBase<InputKeyT, InputValueT> {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using BaseT = MapBase<KeyT, ValueT>;
  using LookupKVResultT = MapInternal::LookupKVResult<KeyT, ValueT>;
  using InsertKVResultT = MapInternal::InsertKVResult<KeyT, ValueT>;

  Map() : BaseT(SmallSize, small_storage()) {}
  Map(const Map& arg) : Map() {
    arg.ForEach([this](KeyT& k, ValueT& v) { insert(k, v); });
  }
  template <ssize_t OtherMinSmallSize>
  explicit Map(const Map<KeyT, ValueT, OtherMinSmallSize>& arg) : Map() {
    arg.ForEach([this](KeyT& k, ValueT& v) { insert(k, v); });
  }
  Map(Map&& arg) = delete;

  void Reset();

 private:
  static constexpr ssize_t SmallSize =
      SetInternal::ComputeSmallSize<MinSmallSize>();

  static_assert(SmallSize >= 0, "Cannot have a negative small size!");

  using SmallSizeStorageT =
      MapInternal::SmallSizeStorage<KeyT, ValueT, SmallSize>;

  // Validate a collection of invariants between the small size storage layout
  // and the dynamically computed storage layout. We need to do this after both
  // are complete but in the context of a specific key type, value type, and
  // small size, so here is the best place.
  static_assert(SmallSize == 0 || alignof(SmallSizeStorageT) ==
                                   MapInternal::StorageAlignment<KeyT, ValueT>,
                "Small size buffer must have the same alignment as a heap "
                "allocated buffer.");
  static_assert(
      SmallSize == 0 ||
          offsetof(SmallSizeStorageT, keys) ==
              SetInternal::ComputeKeyStorageOffset<KeyT>(SmallSize),
      "Offset to keys in small size storage doesn't match computed offset!");
  static_assert(
      SmallSize == 0 ||
          offsetof(SmallSizeStorageT, values) ==
              (SetInternal::ComputeKeyStorageOffset<KeyT>(SmallSize) +
               MapInternal::ComputeValueStorageOffset<KeyT, ValueT>(SmallSize)),
      "Offset from keys to values in small size storage doesn't match computed "
      "offset!");
  static_assert(
      SmallSize == 0 ||
          sizeof(SmallSizeStorageT) ==
              MapInternal::ComputeStorageSize<KeyT, ValueT>(SmallSize),
      "The small size storage needs to match the dynamically computed storage "
      "size.");

  mutable MapInternal::SmallSizeStorage<KeyT, ValueT, SmallSize> small_storage_;

  auto small_storage() const -> MapInternal::Storage* {
    return &small_storage_;
  }
};

template <typename KT, typename VT>
template <typename LookupKeyT>
inline auto MapView<KT, VT>::ContainsHashed(LookupKeyT lookup_key) const
    -> bool {
  return SetInternal::LookupIndexHashed<KeyT>(lookup_key, size(), storage_) >=
         0;
}

template <typename KT, typename VT>
template <typename LookupKeyT>
inline auto MapView<KT, VT>::LookupHashed(LookupKeyT lookup_key) const
    -> LookupKVResultT {
  ssize_t index =
      SetInternal::LookupIndexHashed<KeyT>(lookup_key, size(), storage_);
  if (index < 0) {
    return {nullptr, nullptr};
  }

  return {&keys_ptr()[index], &values_ptr()[index]};
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::Contains(LookupKeyT lookup_key) const -> bool {
  SetInternal::Prefetch(storage_);
  return ContainsHashed(lookup_key);
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::Lookup(LookupKeyT lookup_key) const -> LookupKVResultT {
  SetInternal::Prefetch(storage_);
  return LookupHashed(lookup_key);
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::operator[](LookupKeyT lookup_key) const -> ValueT* {
  auto result = Lookup(lookup_key);
  return result ? &result.value() : nullptr;
}

template <typename KeyT, typename ValueT>
template <typename KVCallbackT, typename GroupCallbackT>
[[clang::always_inline]] void MapView<KeyT, ValueT>::ForEachHashed(
    KVCallbackT kv_callback, GroupCallbackT group_callback) {
  uint8_t* groups = groups_ptr();
  KeyT* keys = keys_ptr();
  ValueT* values = values_ptr();

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
      kv_callback(keys[index], values[index]);
    }

    group_callback(groups, group_index);
  }
}

template <typename KT, typename VT>
template <typename CallbackT>
void MapView<KT, VT>::ForEach(CallbackT callback) {
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
template <typename KT, typename VT>
template <typename LookupKeyT>
[[clang::noinline]] auto MapBase<KT, VT>::InsertIndexHashed(
    LookupKeyT lookup_key) -> std::pair<uint32_t, ssize_t> {
  uint8_t* groups = groups_ptr();

  size_t hash = static_cast<uint64_t>(HashValue(lookup_key));
  uint8_t control_byte = MapInternal::ComputeControlByte(hash);
  ssize_t hash_index = MapInternal::ComputeHashIndex(hash, groups);

  ssize_t group_with_deleted_index = -1;
  MapInternal::Group::MatchedRange deleted_matched_range;

  auto return_insert_at_index = [&](ssize_t index) -> std::pair<bool, ssize_t> {
    // We'll need to insert at this index so set the control group byte to the
    // proper value.
    groups[index] = control_byte;
    return {/*needs_insertion=*/true, index};
  };

  for (MapInternal::ProbeSequence s(hash_index, size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = MapInternal::Group::Load(groups, group_index);

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

template <typename KT, typename VT>
template <typename LookupKeyT>
[[clang::noinline]] auto MapBase<KT, VT>::InsertIntoEmptyIndex(
    LookupKeyT lookup_key) -> ssize_t {
  size_t hash = static_cast<uint64_t>(HashValue(lookup_key));
  uint8_t control_byte = MapInternal::ComputeControlByte(hash);
  uint8_t* groups = groups_ptr();
  ssize_t hash_index = MapInternal::ComputeHashIndex(hash, groups);

  for (MapInternal::ProbeSequence s(hash_index, size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = MapInternal::Group::Load(groups, group_index);

    if (auto empty_matched_range = g.MatchEmpty()) {
      ssize_t index = group_index + *empty_matched_range.begin();
      groups[index] = control_byte;
      return index;
    }

    // Otherwise we continue probing.
  }
}

namespace MapInternal {

template <typename KeyT, typename ValueT>
inline auto AllocateStorage(ssize_t size) -> Storage* {
  ssize_t allocated_size = ComputeStorageSize<KeyT, ValueT>(size);
  return reinterpret_cast<Storage*>(__builtin_operator_new(
      allocated_size, std::align_val_t(StorageAlignment<KeyT, ValueT>),
      std::nothrow_t()));
}

template <typename KeyT, typename ValueT>
inline void DeallocateStorage(Storage* storage, ssize_t size) {
#if __cpp_sized_deallocation
  ssize_t allocated_size = computeStorageSize<KeyT, ValueT>(size);
  return __builtin_operator_delete(
      storage, allocated_size,
      std::align_val_t(StorageAlignment<KeyT, ValueT>));
#else
  // Ensure `size` is used even in the fallback non-sized deallocation case.
  (void)size;
  return __builtin_operator_delete(
      storage, std::align_val_t(StorageAlignment<KeyT, ValueT>));
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

}  // namespace MapInternal

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
[[clang::noinline]] auto MapBase<KeyT, ValueT>::GrowRehashAndInsertIndex(
    LookupKeyT lookup_key) -> ssize_t {
  // We grow into a new `MapBase` so that both the new and old maps are
  // fully functional until all the entries are moved over. However, we directly
  // manipulate the internals to short circuit many aspects of the growth.
  MapBase<KeyT, ValueT> new_map(MapInternal::ComputeNewSize(size()));

  // We specially handle the linear and small case to make it easy to optimize
  // that.
  if (LLVM_LIKELY(impl_view_.is_linear())) {
    impl_view_.ForEachLinear([&](KeyT& old_key, ValueT& old_value) {
      ssize_t index = new_map.InsertIntoEmptyIndex(old_key);
      KeyT* new_keys = new_map.keys_ptr();
      ValueT* new_values = new_map.values_ptr();
      new (&new_keys[index]) KeyT(std::move(old_key));
      old_key.~KeyT();
      new (&new_values[index]) ValueT(std::move(old_value));
      old_value.~ValueT();
    });
    assert(new_map.growth_budget_ > size() &&
           "Must still have a growth budget after rehash!");
    new_map.growth_budget_ -= size();
    assert(is_small() && "Should only have linear scans in the small mode!");
  } else {
    ssize_t insert_count = 0;
    impl_view_.ForEachHashed(
        [&](KeyT& old_key, ValueT& old_value) {
          ++insert_count;
          ssize_t index = new_map.InsertIntoEmptyIndex(old_key);
          KeyT* new_keys = new_map.keys_ptr();
          ValueT* new_values = new_map.values_ptr();
          new (&new_keys[index]) KeyT(std::move(old_key));
          old_key.~KeyT();
          new (&new_values[index]) ValueT(std::move(old_value));
          old_value.~ValueT();
        },
        [](auto...) {});
    new_map.growth_budget_ -= insert_count;
    assert(new_map.growth_budget_ >= 0 &&
           "Must still have a growth budget after rehash!");

    if (LLVM_LIKELY(!is_small())) {
      // Old isn't a small buffer, so we need to deallocate it.
      MapInternal::DeallocateStorage<KeyT, ValueT>(storage(), size());
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

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::InsertHashed(
    LookupKeyT lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        LookupKeyT lookup_key, void* key_storage, void* value_storage)>
        insert_cb) -> InsertKVResultT {
  ssize_t index = -1;
  // Try inserting if we have storage at all.
  if (size() > 0) {
    bool needs_insertion;
    std::tie(needs_insertion, index) = InsertIndexHashed(lookup_key);
    if (LLVM_LIKELY(!needs_insertion)) {
      assert(index >= 0 &&
             "Must have a valid group when we find an existing entry.");
      return {false, keys_ptr()[index], values_ptr()[index]};
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

  KeyT* k;
  ValueT* v;
  std::tie(k, v) =
      insert_cb(lookup_key, &keys_ptr()[index], &values_ptr()[index]);
  return {true, *k, *v};
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::InsertSmallLinear(
    LookupKeyT lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        LookupKeyT lookup_key, void* key_storage, void* value_storage)>
        insert_cb) -> InsertKVResultT {
  KeyT* keys = linear_keys();
  ValueT* values = linear_values();
  for (ssize_t i : llvm::seq<ssize_t>(0, size())) {
    if (keys[i] == lookup_key) {
      return {false, keys[i], values[i]};
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
    values = values_ptr();
  }
  KeyT* k;
  ValueT* v;
  std::tie(k, v) = insert_cb(lookup_key, &keys[index], &values[index]);
  return {true, *k, *v};
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapBase<KT, VT>::Insert(
    LookupKeyT lookup_key,
    typename std::__type_identity<llvm::function_ref<std::pair<KeyT*, ValueT*>(
        LookupKeyT lookup_key, void* key_storage, void* value_storage)>>::type
        insert_cb) -> InsertKVResultT {
  if (impl_view_.is_linear()) {
    return InsertSmallLinear(lookup_key, insert_cb);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return InsertHashed(lookup_key, insert_cb);
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapBase<KT, VT>::Update(
    LookupKeyT lookup_key,
    typename std::__type_identity<llvm::function_ref<std::pair<KeyT*, ValueT*>(
        LookupKeyT lookup_key, void* key_storage, void* value_storage)>>::type
        insert_cb,
    llvm::function_ref<ValueT&(KeyT& key, ValueT& value)> update_cb)
    -> InsertKVResultT {
  ssize_t index = -1;

  KeyT* keys;
  ValueT* values;
  if (impl_view_.is_linear()) {
    keys = linear_keys();
    values = linear_values();
    for (ssize_t i : llvm::seq<ssize_t>(0, size())) {
      if (keys[i] == lookup_key) {
        KeyT& k = keys[i];
        ValueT& v = update_cb(k, values[i]);
        return {false, k, v};
      }
    }
    ssize_t old_size = size();
    if (old_size < small_size()) {
      index = old_size;
      impl_view_.SetSize(old_size + 1);
    }
  } else {
    if (size() > 0) {
      bool needs_insertion = true;
      std::tie(needs_insertion, index) = InsertIndexHashed(lookup_key);
      keys = keys_ptr();
      values = values_ptr();
      if (LLVM_LIKELY(!needs_insertion)) {
        assert(index >= 0 &&
               "Must have a valid group when we find an existing entry.");
        KeyT& k = keys[index];
        ValueT& v = update_cb(k, values[index]);
        return {false, k, v};
      }

      if (index >= 0) {
        // If inserting without growth, track that we've used that budget.
        --growth_budget_;
      }
    }
  }

  if (LLVM_UNLIKELY(index < 0)) {
    assert(impl_view_.is_linear() || growth_budget_ == 0 &&
                                         "Shouldn't need to grow the table "
                                         "until we exhaust our growth budget!");

    index = GrowRehashAndInsertIndex(lookup_key);
    // Refresh the keys and values.
    keys = keys_ptr();
    values = values_ptr();
  }

  assert(index >= 0 && "Should have a group to insert into now.");
  KeyT* k;
  ValueT* v;
  std::tie(k, v) = insert_cb(lookup_key, &keys[index], &values[index]);
  return {true, *k, *v};
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::EraseSmallLinear(LookupKeyT lookup_key) -> bool {
  KeyT* keys = linear_keys();
  ValueT* values = linear_values();
  for (ssize_t i : llvm::seq<ssize_t>(0, size())) {
    if (keys[i] == lookup_key) {
      // Found the key, so clobber this entry with the last one and decrease
      // the size by one.
      impl_view_.SetSize(size() - 1);
      keys[i] = std::move(keys[size()]);
      keys[size()].~KeyT();
      values[i] = std::move(values[size()]);
      values[size()].~ValueT();
      return true;
    }
  }

  return false;
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::EraseHashed(LookupKeyT lookup_key) -> bool {
  ssize_t index =
      MapInternal::LookupIndexHashed<KeyT>(lookup_key, size(), storage());
  if (index < 0) {
    return false;
  }

  KeyT* keys = keys_ptr();
  keys[index].~KeyT();
  values_ptr()[index].~ValueT();

  // If there are empty slots in this group then nothing will probe past this
  // group looking for an entry so we can simply set this slot to empty as
  // well. However, if every slot in this group is full, it might be part of
  // a long probe chain that we can't disrupt. In that case we mark the group
  // as deleted to keep probes continuing past it.
  //
  // If we mark the slot as empty, we'll also need to increase the growth
  // budget.
  uint8_t* groups = groups_ptr();
  ssize_t group_index = index & ~MapInternal::GroupMask;
  auto g = MapInternal::Group::Load(groups, group_index);
  auto empty_matched_range = g.MatchEmpty();
  if (empty_matched_range) {
    groups[index] = MapInternal::Group::Empty;
    ++growth_budget_;
  } else {
    groups[index] = MapInternal::Group::Deleted;
  }
  return true;
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::Erase(LookupKeyT lookup_key) -> bool {
  if (impl_view_.is_linear()) {
    return EraseSmallLinear(lookup_key);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return EraseHashed(lookup_key);
}

template <typename KeyT, typename ValueT>
void MapBase<KeyT, ValueT>::Clear() {
  auto destroy_cb = [](KeyT& k, ValueT& v) {
    // Destroy this key and value.
    k.~KeyT();
    v.~ValueT();
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
        std::memset(groups + group_index, 0, MapInternal::GroupSize);
      });

  // And reset the growth budget.
  growth_budget_ = MapInternal::GrowthThresholdForSize(size());
}

template <typename KeyT, typename ValueT>
MapBase<KeyT, ValueT>::~MapBase() {
  // Nothing to do when in the un-allocated and unused state.
  if (size() == 0) {
    return;
  }

  // Destroy all the keys and values.
  ForEach([](KeyT& k, ValueT& v) {
    k.~KeyT();
    v.~ValueT();
  });

  // If small, nothing to deallocate.
  if (is_small()) {
    return;
  }

  // Just deallocate the storage without updating anything when destroying the
  // object.
  MapInternal::DeallocateStorage<KeyT, ValueT>(storage(), size());
}

template <typename KeyT, typename ValueT>
void MapBase<KeyT, ValueT>::Init(int small_size_arg,
                                 MapInternal::Storage* small_storage) {
  storage() = small_storage;
  small_size_ = small_size_arg;

  if (MapInternal::ShouldUseLinearLookup<KeyT>(small_size_arg)) {
    // We use size to mean empty when doing linear lookups.
    impl_view_.SetSize(0);
    impl_view_.SetLinearValueOffset(small_size_arg);
  } else {
    // We're not using linear lookups in the small size, so initialize it as
    // an initial hash table.
    impl_view_.SetSize(small_size_arg);
    impl_view_.MakeNonLinear();
    growth_budget_ = MapInternal::GrowthThresholdForSize(small_size_arg);
    std::memset(groups_ptr(), 0, small_size_arg);
  }
}

template <typename KeyT, typename ValueT>
void MapBase<KeyT, ValueT>::InitAlloc(ssize_t alloc_size) {
  assert(alloc_size > 0 && "Can only allocate positive size tables!");
  impl_view_.SetSize(alloc_size);
  impl_view_.MakeNonLinear();
  storage() = MapInternal::AllocateStorage<KeyT, ValueT>(alloc_size);
  std::memset(groups_ptr(), 0, alloc_size);
  growth_budget_ = MapInternal::GrowthThresholdForSize(alloc_size);
  small_size_ = 0;
}

template <typename KeyT, typename ValueT, ssize_t MinSmallSize>
void Map<KeyT, ValueT, MinSmallSize>::Reset() {
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
  this->ForEach([](KeyT& k, ValueT& v) {
    k.~KeyT();
    v.~ValueT();
  });

  // Deallocate the buffer.
  MapInternal::DeallocateStorage<KeyT, ValueT>(this->storage(), this->size());

  // Re-initialize the whole thing.
  this->Init(SmallSize, small_storage());
}

}  // namespace Carbon

#endif  // CARBON_COMMON_MAP_H_
