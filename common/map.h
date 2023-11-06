// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_MAP_H_
#define CARBON_COMMON_MAP_H_

#include <algorithm>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "common/raw_hashtable.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

template <typename KeyT, typename ValueT>
class MapView;
template <typename KeyT, typename ValueT>
class MapBase;
template <typename KeyT, typename ValueT, ssize_t MinSmallSize>
class Map;

template <typename InputKeyT, typename InputValueT>
class MapView : public RawHashtable::RawHashtableViewBase<InputKeyT> {
 public:
  using BaseT = RawHashtable::RawHashtableViewBase<InputKeyT>;
  using KeyT = typename BaseT::KeyT;
  using ValueT = InputValueT;
  class LookupKVResult {
   public:
    LookupKVResult() = default;
    explicit LookupKVResult(KeyT* key, ValueT* value)
        : key_(key), value_(value) {}

    explicit operator bool() const { return key_ != nullptr; }

    auto key() const -> KeyT& { return *key_; }
    auto value() const -> ValueT& { return *value_; }

   private:
    KeyT* key_ = nullptr;
    ValueT* value_;
  };

  using BaseT::Contains;

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupKVResult;

  template <typename LookupKeyT>
  auto operator[](LookupKeyT lookup_key) const -> ValueT*;

  template <typename CallbackT>
  void ForEach(CallbackT callback);

 private:
  template <typename MapKeyT, typename MapValueT, ssize_t MinSmallSize>
  friend class Map;
  friend class MapBase<KeyT, ValueT>;
  friend class RawHashtable::RawHashtableBase<KeyT>;

  MapView() = default;
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit by design.
  MapView(BaseT base) : BaseT(base) {}
  MapView(ssize_t size, RawHashtable::Storage* storage)
      : BaseT(size, storage) {}

  auto values_ptr() const -> ValueT* {
    return reinterpret_cast<ValueT*>(
        reinterpret_cast<unsigned char*>(this->keys_ptr()) +
        RawHashtable::ComputeValueStorageOffset<KeyT, ValueT>(this->size()));
  }
};

template <typename InputKeyT, typename InputValueT>
class MapBase : public RawHashtable::RawHashtableBase<InputKeyT> {
 public:
  using BaseT = RawHashtable::RawHashtableBase<InputKeyT>;
  using KeyT = typename BaseT::KeyT;
  using ValueT = InputValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using LookupKVResult = typename ViewT::LookupKVResult;

  class InsertKVResult {
   public:
    InsertKVResult() = default;
    explicit InsertKVResult(bool inserted, KeyT& key, ValueT& value)
        : key_and_inserted_(&key, inserted), value_(&value) {}

    auto is_inserted() const -> bool { return key_and_inserted_.getInt(); }

    auto key() const -> KeyT& { return *key_and_inserted_.getPointer(); }
    auto value() const -> ValueT& { return *value_; }

   private:
    llvm::PointerIntPair<KeyT*, 1, bool> key_and_inserted_;
    ValueT* value_;
  };

  using BaseT::Contains;

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const { return this->impl_view_; }

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupKVResult {
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

  template <typename LookupKeyT>
  auto Insert(
      LookupKeyT lookup_key,
      typename std::__type_identity<llvm::function_ref<
          std::pair<KeyT*, ValueT*>(LookupKeyT lookup_key, void* key_storage,
                                    void* value_storage)>>::type insert_cb)
      -> InsertKVResult;

  template <typename LookupKeyT>
  auto Insert(LookupKeyT lookup_key, ValueT new_v) -> InsertKVResult {
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

          InsertKVResult>::type {
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
      -> InsertKVResult;

  template <typename LookupKeyT, typename ValueCallbackT>
  auto Update(LookupKeyT lookup_key, ValueCallbackT value_cb) ->
      typename std::enable_if<
          !std::is_same<ValueT, ValueCallbackT>::value &&
              std::is_same<ValueT,
                           decltype(std::declval<ValueCallbackT>()())>::value,

          InsertKVResult>::type {
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
  auto Update(LookupKeyT lookup_key, ValueT new_v) -> InsertKVResult {
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
  constexpr static auto ComputeStorageSize(ssize_t size) -> ssize_t {
    return RawHashtable::ComputeKeyValueStorageSize<KeyT, ValueT>(size);
  }

  static auto Allocate(ssize_t size) -> RawHashtable::Storage* {
    ssize_t allocated_size = ComputeStorageSize(size);
    return reinterpret_cast<RawHashtable::Storage*>(__builtin_operator_new(
        allocated_size,
        std::align_val_t(RawHashtable::StorageAlignment<KeyT, ValueT>),
        std::nothrow_t()));
  }

  MapBase(int small_size, RawHashtable::Storage* small_storage)
      : BaseT(small_size, small_storage) {}
  // An internal constructor used to build temporary map base objects with a
  // specific allocated size. This is used internally to build ephemeral maps.
  explicit MapBase(ssize_t arg_size);

  ~MapBase();

  auto values_ptr() -> ValueT* { return ViewT(*this).values_ptr(); }

  template <typename LookupKeyT>
  auto GrowRehashAndInsertIndex(LookupKeyT lookup_key) -> ssize_t;

  auto Deallocate() -> void {
    CARBON_DCHECK(!this->is_small());
    ssize_t allocated_size = ComputeStorageSize(this->size());
    // We don't need the size, but make sure it always compiles.
    (void)allocated_size;
    return __builtin_operator_delete(
        this->storage(),
#if __cpp_sized_deallocation
        allocated_size,
#endif
        std::align_val_t(RawHashtable::StorageAlignment<KeyT, ValueT>));
  }
};

template <typename InputKeyT, typename InputValueT,
          ssize_t SmallSize = 0>
class Map : public MapBase<InputKeyT, InputValueT> {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using BaseT = MapBase<KeyT, ValueT>;
  using LookupKVResult = typename BaseT::LookupKVResult;
  using InsertKVResult = typename BaseT::InsertKVResult;

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
  using SmallSizeStorageT =
      RawHashtable::SmallSizeKeyValueStorage<KeyT, ValueT, SmallSize>;

  auto small_storage() const -> RawHashtable::Storage* {
    return &small_storage_;
  }

  mutable SmallSizeStorageT small_storage_;
};

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::Lookup(LookupKeyT lookup_key) const -> LookupKVResult {
  RawHashtable::Prefetch(this->storage_);
  ssize_t index = RawHashtable::LookupIndexHashed<KeyT>(
      lookup_key, this->size(), this->storage_);
  if (index < 0) {
    return LookupKVResult(nullptr, nullptr);
  }

  return LookupKVResult(&this->keys_ptr()[index], &this->values_ptr()[index]);
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::operator[](LookupKeyT lookup_key) const -> ValueT* {
  auto result = Lookup(lookup_key);
  return result ? &result.value() : nullptr;
}

template <typename KT, typename VT>
template <typename CallbackT>
void MapView<KT, VT>::ForEach(CallbackT callback) {
  this->ForEachIndex(
      [this, callback](KeyT* keys, ssize_t i) {
        callback(keys[i], values_ptr()[i]);
      },
      [](auto...) {});
}

template <typename InputKeyT, typename InputValueT>
MapBase<InputKeyT, InputValueT>::MapBase(ssize_t arg_size)
    : BaseT(arg_size, Allocate(arg_size)) {}

template <typename KeyT, typename ValueT>
MapBase<KeyT, ValueT>::~MapBase() {
  // Nothing to do when in the un-allocated and unused state.
  if (this->size() == 0) {
    return;
  }

  // Destroy all the keys and values.
  ForEach([](KeyT& k, ValueT& v) {
    k.~KeyT();
    v.~ValueT();
  });

  // If small, nothing to deallocate.
  if (this->is_small()) {
    return;
  }

  // Just deallocate the storage without updating anything when destroying the
  // object.
  Deallocate();
}

template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
[[clang::noinline]] auto
MapBase<InputKeyT, InputValueT>::GrowRehashAndInsertIndex(LookupKeyT lookup_key)
    -> ssize_t {
  // We grow into a new `MapBase` so that both the new and old maps are
  // fully functional until all the entries are moved over. However, we directly
  // manipulate the internals to short circuit many aspects of the growth.
  MapBase<KeyT, ValueT> new_map(RawHashtable::ComputeNewSize(this->size()));

  ValueT* old_values = values_ptr();

  ssize_t insert_count = 0;
  KeyT* new_keys = new_map.keys_ptr();
  ValueT* new_values = new_map.values_ptr();
  ViewT(*this).ForEachIndex(
      [&](KeyT* old_keys, ssize_t old_index) {
        ++insert_count;

        // We optimize for small keys that are likely to fit into a register and
        // move twice to avoid loading the key twice from the old table -- first
        // to hash it and a second time prior to storing it into the new table.
        KeyT& old_key_ref = old_keys[old_index];
        KeyT old_key = std::move(old_key_ref);
        old_key_ref.~KeyT();

        ssize_t new_index = new_map.InsertIntoEmptyIndex(old_key);
        new (&new_keys[new_index]) KeyT(std::move(old_key));
        old_key.~KeyT();

        // Move directly from the old value to the new one.
        ValueT& old_value_ref = old_values[old_index];
        new (&new_values[new_index]) ValueT(std::move(old_value_ref));
        old_value_ref.~ValueT();
      },
      [](auto...) {});
  new_map.growth_budget_ -= insert_count;
  CARBON_DCHECK(new_map.growth_budget_ >= 0 &&
                "Must still have a growth budget after rehash!");

  if (LLVM_LIKELY(!this->is_small())) {
    // Old isn't a small buffer, so we need to deallocate it.
    Deallocate();
  }

  // Now that we've fully built the new, grown structures, replace the entries
  // in the data structure. At this point we can be certain to not clobber
  // anything aliasing a small buffer.
  this->impl_view_ = new_map.impl_view_;
  this->growth_budget_ = new_map.growth_budget_;

  // Prevent the ephemeral new map object from doing anything when destroyed as
  // we've taken over it's internals.
  new_map.storage() = nullptr;
  new_map.size() = 0;

  // And lastly insert the lookup_key into an index in the newly grown map and
  // return that index for use.
  --this->growth_budget_;
  return this->InsertIntoEmptyIndex(lookup_key);
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapBase<KT, VT>::Insert(
    LookupKeyT lookup_key,
    typename std::__type_identity<llvm::function_ref<std::pair<KeyT*, ValueT*>(
        LookupKeyT lookup_key, void* key_storage, void* value_storage)>>::type
        insert_cb) -> InsertKVResult {
  ssize_t index = -1;
  // Try inserting if we have storage at all.
  if (this->size() > 0) {
    bool needs_insertion;
    std::tie(needs_insertion, index) = this->InsertIndexHashed(lookup_key);
    if (LLVM_LIKELY(!needs_insertion)) {
      CARBON_DCHECK(index >= 0)
          << "Must have a valid group when we find an existing entry.";
      return InsertKVResult(false, this->keys_ptr()[index],
                            values_ptr()[index]);
    }
  }

  if (index < 0) {
    CARBON_DCHECK(this->growth_budget_ == 0)
        << "Shouldn't need to grow the table until we exhaust our growth "
           "budget!";

    index = GrowRehashAndInsertIndex(lookup_key);
  } else {
    CARBON_DCHECK(this->growth_budget_ >= 0)
        << "Cannot insert with zero budget!";
    --this->growth_budget_;
  }

  CARBON_DCHECK(index >= 0) << "Should have a group to insert into now.";

  KeyT* k;
  ValueT* v;
  std::tie(k, v) =
      insert_cb(lookup_key, &this->keys_ptr()[index], &values_ptr()[index]);
  return InsertKVResult(true, *k, *v);
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapBase<KT, VT>::Update(
    LookupKeyT lookup_key,
    typename std::__type_identity<llvm::function_ref<std::pair<KeyT*, ValueT*>(
        LookupKeyT lookup_key, void* key_storage, void* value_storage)>>::type
        insert_cb,
    llvm::function_ref<ValueT&(KeyT& key, ValueT& value)> update_cb)
    -> InsertKVResult {
  ssize_t index = -1;

  KeyT* keys;
  ValueT* values;
  if (this->size() > 0) {
    bool needs_insertion = true;
    std::tie(needs_insertion, index) = this->InsertIndexHashed(lookup_key);
    keys = this->keys_ptr();
    values = values_ptr();
    if (LLVM_LIKELY(!needs_insertion)) {
      CARBON_DCHECK(index >= 0)
          << "Must have a valid group when we find an existing entry.";
      KeyT& k = keys[index];
      ValueT& v = update_cb(k, values[index]);
      return InsertKVResult(false, k, v);
    }

    if (index >= 0) {
      // If inserting without growth, track that we've used that budget.
      --this->growth_budget_;
    }
  }

  if (LLVM_UNLIKELY(index < 0)) {
    CARBON_DCHECK(this->growth_budget_ == 0)
        << "Shouldn't need to grow the table until we exhaust our growth "
           "budget!";

    index = GrowRehashAndInsertIndex(lookup_key);
    // Refresh the keys and values.
    keys = this->keys_ptr();
    values = values_ptr();
  }

  CARBON_DCHECK(index >= 0) << "Should have a group to insert into now.";
  KeyT* k;
  ValueT* v;
  std::tie(k, v) = insert_cb(lookup_key, &keys[index], &values[index]);
  return InsertKVResult(true, *k, *v);
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::Erase(LookupKeyT lookup_key) -> bool {
  ssize_t erased_index = this->EraseKey(lookup_key);
  if (erased_index < 0) {
    return false;
  }

  values_ptr()[erased_index].~ValueT();
  return true;
}

template <typename KeyT, typename ValueT>
void MapBase<KeyT, ValueT>::Clear() {
  ValueT* values = values_ptr();
  auto index_cb = [values](KeyT* keys, ssize_t i) {
    // Destroy this key and value.
    keys[i].~KeyT();
    values[i].~ValueT();
  };
  this->ClearImpl(index_cb);
}

template <typename KeyT, typename ValueT, ssize_t SmallSize>
void Map<KeyT, ValueT, SmallSize>::Reset() {
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
  this->Deallocate();

  // Re-initialize the whole thing.
  CARBON_DCHECK(this->small_size() == SmallSize);
  this->Init(SmallSize, small_storage());
}

}  // namespace Carbon

#endif  // CARBON_COMMON_MAP_H_
