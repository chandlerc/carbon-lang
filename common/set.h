// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_SET_H_
#define CARBON_COMMON_SET_H_

#include <algorithm>
#include <new>
#include <tuple>
#include <utility>

#include "common/check.h"
#include "common/raw_hashtable.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

template <typename KeyT>
class SetView;
template <typename KeyT>
class SetBase;
template <typename KeyT, ssize_t MinSmallSize>
class Set;

template <typename InputKeyT>
class SetView : public RawHashtable::RawHashtableViewBase<InputKeyT> {
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

  using BaseT::Contains;

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupResult;

  template <typename CallbackT>
  void ForEach(CallbackT callback);

 private:
  template <typename SetKeyT, ssize_t MinSmallSize>
  friend class Set;
  friend class SetBase<KeyT>;
  friend class RawHashtable::RawHashtableBase<KeyT>;

  SetView() = default;
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit by design.
  SetView(BaseT base) : BaseT(base) {}
  SetView(ssize_t size, RawHashtable::Storage* storage)
      : BaseT(size, storage) {}
};

template <typename InputKeyT>
class SetBase : public RawHashtable::RawHashtableBase<InputKeyT> {
 public:
  using BaseT = RawHashtable::RawHashtableBase<InputKeyT>;
  using KeyT = typename BaseT::KeyT;
  using ViewT = SetView<KeyT>;
  using LookupResult = typename ViewT::LookupResult;

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

  using BaseT::Contains;

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const { return this->impl_view_; }

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupResult {
    return ViewT(*this).Lookup(lookup_key);
  }

  template <typename CallbackT>
  void ForEach(CallbackT callback) {
    return ViewT(*this).ForEach(callback);
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
                    return new (key_storage) KeyT(lookup_key);
                  });
  }

  template <typename LookupKeyT>
  auto Erase(LookupKeyT lookup_key) -> bool;

  void Clear();

 protected:
  constexpr static auto ComputeStorageSize(ssize_t size) -> ssize_t {
    return RawHashtable::ComputeKeyStorageSize<KeyT>(size);
  }

  static auto Allocate(ssize_t size) -> RawHashtable::Storage* {
    ssize_t allocated_size = ComputeStorageSize(size);
    return reinterpret_cast<RawHashtable::Storage*>(__builtin_operator_new(
        allocated_size, std::align_val_t(RawHashtable::StorageAlignment<KeyT>),
        std::nothrow_t()));
  }

  SetBase(int small_size, RawHashtable::Storage* small_storage)
      : BaseT(small_size, small_storage) {}
  // An internal constructor used to build temporary map base objects with a
  // specific allocated size. This is used internally to build ephemeral maps.
  explicit SetBase(ssize_t arg_size);

  ~SetBase();

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
        std::align_val_t(RawHashtable::StorageAlignment<KeyT>));
  }
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
  using SmallSizeStorageT = RawHashtable::SmallSizeKeyStorage<KeyT, SmallSize>;

  auto small_storage() const -> RawHashtable::Storage* {
    return &small_storage_;
  }

  mutable SmallSizeStorageT small_storage_;
};

template <typename KT>
template <typename LookupKeyT>
auto SetView<KT>::Lookup(LookupKeyT lookup_key) const -> LookupResult {
  RawHashtable::Prefetch(this->storage_);
  ssize_t index = RawHashtable::LookupIndexHashed<KeyT>(
      lookup_key, this->size(), this->storage_);
  if (index < 0) {
    return LookupResult();
  }

  return LookupResult(&this->keys_ptr()[index]);
}

template <typename KT>
template <typename CallbackT>
void SetView<KT>::ForEach(CallbackT callback) {
  this->ForEachIndex([callback](KeyT* keys, ssize_t i) { callback(keys[i]); },
                     [](auto...) {});
}

template <typename InputKeyT>
SetBase<InputKeyT>::SetBase(ssize_t arg_size)
    : BaseT(arg_size, Allocate(arg_size)) {}

template <typename InputKeyT>
SetBase<InputKeyT>::~SetBase() {
  // Nothing to do when in the un-allocated and unused state.
  if (this->size() == 0) {
    return;
  }

  // Destroy all the keys and values.
  ForEach([](KeyT& k) { k.~KeyT(); });

  // If small, nothing to deallocate.
  if (this->is_small()) {
    return;
  }

  // Just deallocate the storage without updating anything when destroying the
  // object.
  Deallocate();
}

template <typename InputKeyT>
template <typename LookupKeyT>
[[clang::noinline]] auto SetBase<InputKeyT>::GrowRehashAndInsertIndex(
    LookupKeyT lookup_key) -> ssize_t {
  // We grow into a new `MapBase` so that both the new and old maps are
  // fully functional until all the entries are moved over. However, we directly
  // manipulate the internals to short circuit many aspects of the growth.
  SetBase<KeyT> new_map(RawHashtable::ComputeNewSize(this->size()));

  ssize_t insert_count = 0;
  KeyT* new_keys = new_map.keys_ptr();
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

template <typename KT>
template <typename LookupKeyT>
auto SetBase<KT>::Insert(
    LookupKeyT lookup_key,
    typename std::__type_identity<llvm::function_ref<
        auto(LookupKeyT lookup_key, void* key_storage)->KeyT*>>::type insert_cb)
    -> InsertResult {
  ssize_t index = -1;
  // Try inserting if we have storage at all.
  if (this->size() > 0) {
    bool needs_insertion;
    std::tie(needs_insertion, index) = this->InsertIndexHashed(lookup_key);
    if (LLVM_LIKELY(!needs_insertion)) {
      CARBON_DCHECK(index >= 0)
          << "Must have a valid group when we find an existing entry.";
      return InsertResult(false, this->keys_ptr()[index]);
    }
  }

  if (index < 0) {
    CARBON_DCHECK(this->growth_budget_ == 0)
        << "Shouldn't need to grow the table until we exhaust our growth "
           "budget!";

    index = this->GrowRehashAndInsertIndex(lookup_key);
  } else {
    CARBON_DCHECK(this->growth_budget_ >= 0)
        << "Cannot insert with zero budget!";
    --this->growth_budget_;
  }

  CARBON_DCHECK(index >= 0) << "Should have a group to insert into now.";

  KeyT* k = insert_cb(lookup_key, &this->keys_ptr()[index]);
  return InsertResult(true, *k);
}

template <typename KeyT>
template <typename LookupKeyT>
auto SetBase<KeyT>::Erase(LookupKeyT lookup_key) -> bool {
  return this->EraseKey(lookup_key) >= 0;
}

template <typename KeyT>
void SetBase<KeyT>::Clear() {
  auto index_cb = [](KeyT* keys, ssize_t i) {
    // Destroy the key.
    keys[i].~KeyT();
  };
  this->ClearImpl(index_cb);
}

template <typename KeyT, ssize_t SmallSize>
void Set<KeyT, SmallSize>::Reset() {
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
  this->ForEach([](KeyT& k) { k.~KeyT(); });

  // Deallocate the buffer.
  this->Deallocate();

  // Re-initialize the whole thing.
  CARBON_DCHECK(this->small_size() == SmallSize);
  this->Init(SmallSize, small_storage());
}

}  // namespace Carbon

#endif  // CARBON_COMMON_SET_H_
