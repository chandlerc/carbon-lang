// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_SET_H_
#define CARBON_COMMON_SET_H_

#include "common/check.h"
#include "common/raw_hashtable.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

template <typename KeyT>
class SetView;
template <typename KeyT>
class SetBase;
template <typename KeyT, ssize_t MinSmallSize>
class Set;

template <typename InputKeyT>
class SetView : RawHashtable::RawHashtableViewBase<InputKeyT> {
 public:
  using BaseT = RawHashtable::RawHashtableViewBase<InputKeyT>;
  using KeyT = typename BaseT::KeyT;

  class LookupResult {
   public:
    LookupResult() = default;
    explicit LookupResult(KeyT* key) : key_(key) {}

    explicit operator bool() const { return key_ != nullptr; }

    auto key() const -> KeyT& { return *key_; }

   private:
    KeyT* key_ = nullptr;
  };

  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool;

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupResult;

  template <typename CallbackT>
  void ForEach(CallbackT callback);

  auto CountProbedKeys() -> ssize_t { return BaseT::CountProbedKeys(); }

 private:
  template <typename SetKeyT, ssize_t MinSmallSize>
  friend class Set;
  friend class SetBase<KeyT>;
  friend class RawHashtable::RawHashtableBase<KeyT>;

  using EntryT = typename BaseT::EntryT;

  SetView() = default;
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit by design.
  SetView(BaseT base) : BaseT(base) {}
  SetView(ssize_t size, RawHashtable::Storage* storage)
      : BaseT(size, storage) {}
};

template <typename InputKeyT>
class SetBase : protected RawHashtable::RawHashtableBase<InputKeyT> {
 public:
  using BaseT = RawHashtable::RawHashtableBase<InputKeyT>;
  using KeyT = typename BaseT::KeyT;
  using ViewT = SetView<KeyT>;
  using LookupResult = typename ViewT::LookupResult;

  class InsertResult {
   public:
    InsertResult() = default;
    explicit InsertResult(bool inserted, KeyT& key)
        : key_(&key), inserted_(inserted) {}

    auto is_inserted() const -> bool { return inserted_; }

    auto key() const -> KeyT& { return *key_; }

   private:
    KeyT* key_;
    bool inserted_;
  };

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const { return this->impl_view_; }

  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool {
    return ViewT(*this).Contains(lookup_key);
  }

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupResult {
    return ViewT(*this).Lookup(lookup_key);
  }

  template <typename CallbackT>
  void ForEach(CallbackT callback) {
    return ViewT(*this).ForEach(callback);
  }

  auto CountProbedKeys() const -> ssize_t {
    return ViewT(*this).CountProbedKeys();
  }

  template <typename LookupKeyT>
  auto Insert(LookupKeyT lookup_key,
              typename std::__type_identity<llvm::function_ref<
                  auto(LookupKeyT lookup_key, void* key_storage)->KeyT*>>::type
                  insert_cb) -> InsertResult;

  template <typename LookupKeyT>
  auto Insert(LookupKeyT lookup_key) -> InsertResult {
    return Insert(lookup_key,
                  [](LookupKeyT lookup_key, void* key_storage) -> KeyT* {
                    return new (key_storage) KeyT(std::move(lookup_key));
                  });
  }

  template <typename LookupKeyT>
  auto Erase(LookupKeyT lookup_key) -> bool;

  void Clear();

 protected:
  template <ssize_t SmallSize>
  using SmallStorageT = typename BaseT::template SmallStorageT<SmallSize>;

  using EntryT = typename BaseT::EntryT;

  SetBase(int small_size, RawHashtable::Storage* small_storage)
      : BaseT(small_size, small_storage) {}
};

template <typename InputKeyT, ssize_t SmallSize = 0>
class Set : public SetBase<InputKeyT> {
 public:
  using KeyT = InputKeyT;
  using ViewT = SetView<KeyT>;
  using BaseT = SetBase<KeyT>;
  using LookupResult = typename BaseT::LookupResult;
  using InsertResult = typename BaseT::InsertResult;

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
  using EntryT = typename BaseT::EntryT;
  using SmallSizeStorageT = typename BaseT::template SmallStorageT<SmallSize>;

  auto small_storage() const -> RawHashtable::Storage* {
    return &small_storage_;
  }

  mutable SmallSizeStorageT small_storage_;
};

template <typename InputKeyT>
template <typename LookupKeyT>
auto SetView<InputKeyT>::Contains(LookupKeyT lookup_key) const -> bool {
  RawHashtable::Prefetch(this->storage_);
  return this->LookupIndexHashed(lookup_key) != nullptr;
}

template <typename KT>
template <typename LookupKeyT>
auto SetView<KT>::Lookup(LookupKeyT lookup_key) const -> LookupResult {
  RawHashtable::Prefetch(this->storage_);
  EntryT* entry = this->LookupIndexHashed(lookup_key);
  if (!entry) {
    return LookupResult();
  }

  return LookupResult(&entry->key);
}

template <typename KT>
template <typename CallbackT>
void SetView<KT>::ForEach(CallbackT callback) {
  this->ForEachIndex(
      [callback](EntryT* entries, ssize_t i) { callback(entries[i].key); },
      [](auto...) {});
}

template <typename KT>
template <typename LookupKeyT>
auto SetBase<KT>::Insert(
    LookupKeyT lookup_key,
    typename std::__type_identity<llvm::function_ref<
        auto(LookupKeyT lookup_key, void* key_storage)->KeyT*>>::type insert_cb)
    -> InsertResult {
  auto [entry, inserted] = this->InsertIndexHashed(lookup_key);
  CARBON_DCHECK(entry) << "Should always result in a valid index.";
  if (LLVM_LIKELY(!inserted)) {
    return InsertResult(false, entry->key);
  }

  CARBON_DCHECK(this->growth_budget_ >= 0)
      << "Growth budget shouldn't have gone negative!";
  KeyT* k = &entry->key;
  k = insert_cb(lookup_key, k);
  return InsertResult(true, *k);
}

template <typename KeyT>
template <typename LookupKeyT>
auto SetBase<KeyT>::Erase(LookupKeyT lookup_key) -> bool {
  return this->EraseKey(lookup_key);
}

template <typename KeyT>
void SetBase<KeyT>::Clear() {
  this->ClearImpl();
}

template <typename KeyT, ssize_t SmallSize>
void Set<KeyT, SmallSize>::Reset() {
  this->DestroyImpl();

  // Re-initialize the whole thing.
  CARBON_DCHECK(this->small_size() == SmallSize);
  this->Init(SmallSize, small_storage());
}

}  // namespace Carbon

#endif  // CARBON_COMMON_SET_H_
