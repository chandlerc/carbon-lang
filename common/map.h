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
#include "llvm/Support/Compiler.h"

namespace Carbon {

template <typename KeyT, typename ValueT>
class MapView;
template <typename KeyT, typename ValueT>
class MapBase;
template <typename KeyT, typename ValueT, ssize_t MinSmallSize>
class Map;

template <typename InputKeyT, typename InputValueT>
class MapView : RawHashtable::RawHashtableViewBase<InputKeyT, InputValueT> {
  using BaseT = RawHashtable::RawHashtableViewBase<InputKeyT, InputValueT>;
  using EntryT = typename BaseT::EntryT;

 public:
  using KeyT = typename BaseT::KeyT;
  using ValueT = typename BaseT::ValueT;

  class LookupKVResult {
   public:
    LookupKVResult() = default;
    explicit LookupKVResult(EntryT* entry) : entry_(entry) {}

    explicit operator bool() const { return entry_ != nullptr; }

    auto key() const -> KeyT& { return entry_->key; }
    auto value() const -> ValueT& { return entry_->value; }

   private:
    EntryT* entry_ = nullptr;
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

  MapView() = default;
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit by design.
  MapView(BaseT base) : BaseT(base) {}
  MapView(ssize_t size, RawHashtable::Storage* storage)
      : BaseT(size, storage) {}
};

template <typename InputKeyT, typename InputValueT>
class MapBase
    : protected RawHashtable::RawHashtableBase<InputKeyT, InputValueT> {
  using BaseT = RawHashtable::RawHashtableBase<InputKeyT, InputValueT>;

 protected:
  using EntryT = typename BaseT::EntryT;

 public:
  using KeyT = typename BaseT::KeyT;
  using ValueT = typename BaseT::ValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using LookupKVResult = typename ViewT::LookupKVResult;

  class InsertKVResult {
   public:
    InsertKVResult() = default;
    explicit InsertKVResult(bool inserted, EntryT& entry)
        : entry_(&entry), inserted_(inserted) {}

    auto is_inserted() const -> bool { return inserted_; }

    auto key() const -> KeyT& { return entry_->key; }
    auto value() const -> ValueT& { return entry_->value; }

   private:
    EntryT* entry_;
    bool inserted_;
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
  auto Insert(LookupKeyT lookup_key, ValueT new_v) -> InsertKVResult;

  template <typename LookupKeyT, typename ValueCallbackT>
  auto
  Insert(LookupKeyT lookup_key, ValueCallbackT value_cb) -> std::enable_if_t<
      !std::is_same_v<ValueT, ValueCallbackT> &&
          std::is_same_v<ValueT, decltype(std::declval<ValueCallbackT>()())>,
      InsertKVResult>;

  template <typename LookupKeyT, typename InsertCallbackT>
  auto Insert(LookupKeyT lookup_key, InsertCallbackT insert_cb)
      -> std::enable_if_t<
          !std::is_same_v<ValueT, InsertCallbackT> &&
              std::is_invocable_v<InsertCallbackT, LookupKeyT, void*, void*>,
          InsertKVResult>;

  template <typename LookupKeyT>
  auto Update(LookupKeyT lookup_key, ValueT new_v) -> InsertKVResult;

  template <typename LookupKeyT, typename ValueCallbackT>
  auto
  Update(LookupKeyT lookup_key, ValueCallbackT value_cb) -> std::enable_if_t<
      !std::is_same_v<ValueT, ValueCallbackT> &&
          std::is_same_v<ValueT, decltype(std::declval<ValueCallbackT>()())>,
      InsertKVResult>;

  template <typename LookupKeyT, typename InsertCallbackT,
            typename UpdateCallbackT>
  auto Update(LookupKeyT lookup_key, InsertCallbackT insert_cb,
              UpdateCallbackT update_cb)
      -> std::enable_if_t<
          !std::is_same_v<ValueT, InsertCallbackT> &&
              std::is_invocable_v<InsertCallbackT, LookupKeyT, void*, void*> &&
              std::is_invocable_v<UpdateCallbackT, KeyT&, ValueT&>,
          InsertKVResult>;

  template <typename LookupKeyT>
  auto Erase(LookupKeyT lookup_key) -> bool;

  void Clear();

  auto CountProbedKeys() const -> ssize_t {
    return ViewT(*this).CountProbedKeys();
  }

 protected:
  template <ssize_t SmallSize>
  using SmallStorageT = typename BaseT::template SmallStorageT<SmallSize>;

  MapBase(int small_size, RawHashtable::Storage* small_storage)
      : BaseT(small_size, small_storage) {}
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
  using EntryT = typename BaseT::EntryT;
  using SmallSizeStorageT = typename BaseT::template SmallStorageT<SmallSize>;

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
  return this->LookupIndexHashed(lookup_key) != nullptr;
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::Lookup(LookupKeyT lookup_key) const -> LookupKVResult {
  RawHashtable::Prefetch(this->storage_);
  return LookupKVResult(this->LookupIndexHashed(lookup_key));
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
      [this, callback](EntryT* /*entries*/, ssize_t i) {
        EntryT& entry = this->entries()[i];
        callback(entry.key, entry.value);
      },
      [](auto...) {});
}

template <typename KT, typename VT>
template <typename LookupKeyT>
[[clang::always_inline]] auto MapBase<KT, VT>::Insert(LookupKeyT lookup_key,
                                                      ValueT new_v)
    -> InsertKVResult {
  return Insert(lookup_key,
                [&new_v](LookupKeyT lookup_key, void* key_storage,
                         void* value_storage) -> std::pair<KeyT*, ValueT*> {
                  KeyT* k = new (key_storage) KeyT(lookup_key);
                  auto* v = new (value_storage) ValueT(std::move(new_v));
                  return {k, v};
                });
}

template <typename KT, typename VT>
template <typename LookupKeyT, typename ValueCallbackT>
[[clang::always_inline]] auto MapBase<KT, VT>::Insert(LookupKeyT lookup_key,
                                                      ValueCallbackT value_cb)
    -> std::enable_if_t<
        !std::is_same_v<ValueT, ValueCallbackT> &&
            std::is_same_v<ValueT, decltype(std::declval<ValueCallbackT>()())>,

        InsertKVResult> {
  return Insert(lookup_key,
                [&value_cb](LookupKeyT lookup_key, void* key_storage,
                            void* value_storage) -> std::pair<KeyT*, ValueT*> {
                  KeyT* k = new (key_storage) KeyT(lookup_key);
                  auto* v = new (value_storage) ValueT(value_cb());
                  return {k, v};
                });
}

template <typename KT, typename VT>
template <typename LookupKeyT, typename InsertCallbackT>
[[clang::always_inline]] auto MapBase<KT, VT>::Insert(LookupKeyT lookup_key,
                                                      InsertCallbackT insert_cb)
    -> std::enable_if_t<
        !std::is_same_v<ValueT, InsertCallbackT> &&
            std::is_invocable_v<InsertCallbackT, LookupKeyT, void*, void*>,
        InsertKVResult> {
  auto [entry, inserted] = this->InsertIndexHashed(lookup_key);
  CARBON_DCHECK(entry) << "Should always result in a valid index.";

  if (LLVM_LIKELY(!inserted)) {
    return InsertKVResult(false, *entry);
  }

  CARBON_DCHECK(this->growth_budget_ >= 0)
      << "Growth budget shouldn't have gone negative!";
  insert_cb(lookup_key, static_cast<void*>(&entry->key),
            static_cast<void*>(&entry->value));
  return InsertKVResult(true, *entry);
}

template <typename KT, typename VT>
template <typename LookupKeyT>
[[clang::always_inline]] auto MapBase<KT, VT>::Update(LookupKeyT lookup_key,
                                                      ValueT new_v)
    -> InsertKVResult {
  return Update(
      lookup_key,
      [&new_v](LookupKeyT lookup_key, void* key_storage, void* value_storage) {
        new (key_storage) KeyT(lookup_key);
        new (value_storage) ValueT(std::move(new_v));
      },
      [&new_v](KeyT& /*key*/, ValueT& value) {
        value.~ValueT();
        new (&value) ValueT(std::move(new_v));
      });
}

template <typename KT, typename VT>
template <typename LookupKeyT, typename ValueCallbackT>
[[clang::always_inline]] auto MapBase<KT, VT>::Update(LookupKeyT lookup_key,
                                                      ValueCallbackT value_cb)
    -> std::enable_if_t<
        !std::is_same_v<ValueT, ValueCallbackT> &&
            std::is_same_v<ValueT, decltype(std::declval<ValueCallbackT>()())>,
        InsertKVResult> {
  return Update(
      lookup_key,
      [&value_cb](LookupKeyT lookup_key, void* key_storage,
                  void* value_storage) {
        new (key_storage) KeyT(lookup_key);
        new (value_storage) ValueT(value_cb());
      },
      [&value_cb](KeyT& /*key*/, ValueT& value) {
        value.~ValueT();
        new (&value) ValueT(value_cb());
      });
}

template <typename KT, typename VT>
template <typename LookupKeyT, typename InsertCallbackT,
          typename UpdateCallbackT>
[[clang::always_inline]] auto MapBase<KT, VT>::Update(LookupKeyT lookup_key,
                                                      InsertCallbackT insert_cb,
                                                      UpdateCallbackT update_cb)
    -> std::enable_if_t<
        !std::is_same_v<ValueT, InsertCallbackT> &&
            std::is_invocable_v<InsertCallbackT, LookupKeyT, void*, void*> &&
            std::is_invocable_v<UpdateCallbackT, KeyT&, ValueT&>,
        InsertKVResult> {
  auto [entry, inserted] = this->InsertIndexHashed(lookup_key);
  CARBON_DCHECK(entry) << "Should always result in a valid index.";
  // EntryT& entry = this->entries()[index];

  if (LLVM_LIKELY(!inserted)) {
    update_cb(entry->key, entry->value);
    return InsertKVResult(false, *entry);
  }

  CARBON_DCHECK(this->growth_budget_ >= 0)
      << "Growth budget shouldn't have gone negative!";
  insert_cb(lookup_key, static_cast<void*>(&entry->key),
            static_cast<void*>(&entry->value));
  return InsertKVResult(true, *entry);
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::Erase(LookupKeyT lookup_key) -> bool {
  return this->EraseKey(lookup_key);
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
