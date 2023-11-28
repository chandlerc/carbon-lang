// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_MAP_H_
#define CARBON_COMMON_MAP_H_

#include <algorithm>
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
class MapView : RawHashtable::RawHashtableViewBase<InputKeyT> {
  using BaseT = RawHashtable::RawHashtableViewBase<InputKeyT>;

 public:
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

  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool;

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupKVResult;

  template <typename LookupKeyT>
  auto operator[](LookupKeyT lookup_key) const -> ValueT*;

  template <typename CallbackT>
  void ForEach(CallbackT callback);

  auto CountProbedKeys() -> ssize_t { return BaseT::CountProbedKeys(); }

 private:
  template <typename MapKeyT, typename MapValueT, ssize_t MinSmallSize>
  friend class Map;
  friend class MapBase<KeyT, ValueT>;
  friend class RawHashtable::RawHashtableKeyBase<KeyT>;

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
class MapBase
    : protected RawHashtable::RawHashtableBase<InputKeyT, InputValueT> {
  using BaseT = RawHashtable::RawHashtableBase<InputKeyT, InputValueT>;

 public:
  using KeyT = typename BaseT::KeyT;
  using ValueT = typename BaseT::ValueT;
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

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const { return this->impl_view_; }

  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool {
    return ViewT(*this).Contains(lookup_key);
  }

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
  
  auto CountProbedKeys() const -> ssize_t {
    return ViewT(*this).CountProbedKeys();
  }

 protected:
  MapBase(int small_size, RawHashtable::Storage* small_storage)
      : BaseT(small_size, small_storage) {}

  auto values_ptr() -> ValueT* { return ViewT(*this).values_ptr(); }
};

template <typename InputKeyT, typename InputValueT, ssize_t SmallSize = 0>
class Map : public MapBase<InputKeyT, InputValueT> {
 public:
  using BaseT = MapBase<InputKeyT, InputValueT>;

  using KeyT = typename BaseT::KeyT;
  using ValueT = typename BaseT::ValueT;
  using ViewT = MapView<KeyT, ValueT>;
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

template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
auto MapView<InputKeyT, InputValueT>::Contains(LookupKeyT lookup_key) const
    -> bool {
  RawHashtable::Prefetch(this->storage_);
  return this->LookupIndexHashed(lookup_key) >= 0;
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::Lookup(LookupKeyT lookup_key) const -> LookupKVResult {
  RawHashtable::Prefetch(this->storage_);
  ssize_t index = this->LookupIndexHashed(lookup_key);
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

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapBase<KT, VT>::Insert(
    LookupKeyT lookup_key,
    typename std::__type_identity<llvm::function_ref<std::pair<KeyT*, ValueT*>(
        LookupKeyT lookup_key, void* key_storage, void* value_storage)>>::type
        insert_cb) -> InsertKVResult {
  ssize_t index;
  uint8_t control_byte;
  std::tie(index, control_byte) = this->InsertIndexHashed(lookup_key);
  CARBON_DCHECK(index >= 0) << "Should always result in a valid index.";
  if (LLVM_LIKELY(control_byte == 0)) {
    return InsertKVResult(false, this->keys_ptr()[index], values_ptr()[index]);
  }

  CARBON_DCHECK(this->growth_budget_ >= 0)
      << "Growth budget shouldn't have gone negative!";
  this->groups_ptr()[index] = control_byte;
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
  ssize_t index;
  uint8_t control_byte;
  std::tie(index, control_byte) = this->InsertIndexHashed(lookup_key);
  CARBON_DCHECK(index >= 0) << "Should always result in a valid index.";
  if (LLVM_LIKELY(control_byte == 0)) {
    KeyT& k = this->keys_ptr()[index];
    ValueT& v = update_cb(k, this->values_ptr()[index]);
    return InsertKVResult(false, k, v);
  }

  CARBON_DCHECK(this->growth_budget_ >= 0)
      << "Growth budget shouldn't have gone negative!";
  this->groups_ptr()[index] = control_byte;
  KeyT* k;
  ValueT* v;
  std::tie(k, v) =
      insert_cb(lookup_key, &this->keys_ptr()[index], &values_ptr()[index]);
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
  this->ClearImpl();
}

template <typename KeyT, typename ValueT, ssize_t SmallSize>
void Map<KeyT, ValueT, SmallSize>::Reset() {
  this->DestroyImpl();

  // Re-initialize the whole thing.
  CARBON_DCHECK(this->small_size() == SmallSize);
  this->Init(SmallSize, small_storage());
}

}  // namespace Carbon

#endif  // CARBON_COMMON_MAP_H_
