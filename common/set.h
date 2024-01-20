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
class SetView : RawHashtable::ViewImpl<InputKeyT> {
  using ImplT = RawHashtable::ViewImpl<InputKeyT>;

 public:
  using KeyT = typename ImplT::KeyT;

  class LookupResult {
   public:
    LookupResult() = default;
    explicit LookupResult(KeyT& key) : key_(&key) {}

    explicit operator bool() const { return key_ != nullptr; }

    auto key() const -> KeyT& { return *key_; }

   private:
    KeyT* key_ = nullptr;
  };

  // Enable implicit conversions that add `const`-ness to the key type.
  // NOLINTNEXTLINE(google-explicit-constructor)
  SetView(SetView<std::remove_const_t<KeyT>> other_view)
    requires(!std::same_as<KeyT, std::remove_const_t<KeyT>>)
      : ImplT(other_view) {}

  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool;

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupResult;

  template <typename CallbackT>
  void ForEach(CallbackT callback);

  auto CountProbedKeys() -> ssize_t { return ImplT::CountProbedKeys(); }

 private:
  template <typename SetKeyT, ssize_t MinSmallSize>
  friend class Set;
  friend class SetBase<KeyT>;
  friend class SetView<const KeyT>;

  using EntryT = typename ImplT::EntryT;

  SetView() = default;
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit by design.
  SetView(ImplT base) : ImplT(base) {}
  SetView(ssize_t size, RawHashtable::Storage* storage)
      : ImplT(size, storage) {}
};

template <typename InputKeyT>
class SetBase : protected RawHashtable::BaseImpl<InputKeyT> {
  using ImplT = RawHashtable::BaseImpl<InputKeyT>;

 public:
  using KeyT = typename ImplT::KeyT;
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
  operator ViewT() const { return this->view_impl(); }

  // We can't chain the above conversion with the conversions on `ViewT` to add
  // const, so explicitly support adding const to produce a view here.
  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator SetView<const KeyT>() const { return ViewT(*this); }

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
  auto Insert(LookupKeyT lookup_key) -> InsertResult;

  template <typename LookupKeyT, typename InsertCallbackT>
  auto Insert(LookupKeyT lookup_key, InsertCallbackT insert_cb) -> InsertResult
    requires std::invocable<InsertCallbackT, LookupKeyT, void*>;

  template <typename LookupKeyT>
  auto Erase(LookupKeyT lookup_key) -> bool;

  void Clear();

 protected:
  using ImplT::ImplT;
};

template <typename InputKeyT, ssize_t SmallSize = 0>
class Set : public RawHashtable::TableImpl<SetBase<InputKeyT>, SmallSize> {
  using BaseT = SetBase<InputKeyT>;
  using ImplT = RawHashtable::TableImpl<BaseT, SmallSize>;

 public:
  using KeyT = InputKeyT;

  Set() = default;
  Set(const Set& arg) = default;
  template <ssize_t OtherMinSmallSize>
  explicit Set(const Set<KeyT, OtherMinSmallSize>& arg) : ImplT(arg) {}
  Set(Set&& arg) = default;
  template <ssize_t OtherMinSmallSize>
  explicit Set(Set<KeyT, OtherMinSmallSize>&& arg) : ImplT(std::move(arg)) {}

  void Reset();
};

template <typename InputKeyT>
template <typename LookupKeyT>
auto SetView<InputKeyT>::Contains(LookupKeyT lookup_key) const -> bool {
  return this->LookupEntry(lookup_key) != nullptr;
}

template <typename KT>
template <typename LookupKeyT>
auto SetView<KT>::Lookup(LookupKeyT lookup_key) const -> LookupResult {
  EntryT* entry = this->LookupEntry(lookup_key);
  if (!entry) {
    return LookupResult();
  }

  return LookupResult(entry->key());
}

template <typename KT>
template <typename CallbackT>
void SetView<KT>::ForEach(CallbackT callback) {
  this->ForEachEntry([callback](EntryT& entry) { callback(entry.key()); },
                     [](auto...) {});
}

template <typename KT>
template <typename LookupKeyT>
auto SetBase<KT>::Insert(LookupKeyT lookup_key) -> InsertResult {
  return Insert(lookup_key, [](LookupKeyT lookup_key, void* key_storage) {
    new (key_storage) KeyT(std::move(lookup_key));
  });
}

template <typename KT>
template <typename LookupKeyT, typename InsertCallbackT>
auto SetBase<KT>::Insert(LookupKeyT lookup_key, InsertCallbackT insert_cb)
    -> InsertResult
  requires std::invocable<InsertCallbackT, LookupKeyT, void*>
{
  auto [entry, inserted] = this->InsertImpl(lookup_key);
  CARBON_DCHECK(entry) << "Should always result in a valid index.";

  if (LLVM_LIKELY(!inserted)) {
    return InsertResult(false, entry->key());
  }

  insert_cb(lookup_key, static_cast<void*>(&entry->key_storage));
  return InsertResult(true, entry->key());
}

template <typename KeyT>
template <typename LookupKeyT>
auto SetBase<KeyT>::Erase(LookupKeyT lookup_key) -> bool {
  return this->EraseImpl(lookup_key);
}

template <typename KeyT>
void SetBase<KeyT>::Clear() {
  this->ClearImpl();
}

template <typename KeyT, ssize_t SmallSize>
void Set<KeyT, SmallSize>::Reset() {
  this->ResetImpl();
}

}  // namespace Carbon

#endif  // CARBON_COMMON_SET_H_
