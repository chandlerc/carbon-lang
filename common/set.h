// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_SET_H_
#define CARBON_COMMON_SET_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <new>
#include <tuple>
#include <utility>

#include "common/check.h"
#include "common/raw_hashtable.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

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

template <ssize_t MinSmallSize>
constexpr auto ComputeSmallSize() -> ssize_t {
  return llvm::alignTo<GroupSize>(MinSmallSize);
}

template <typename KeyT, ssize_t SmallSize>
struct SmallSizeStorage;

template <typename KeyT>
struct SmallSizeStorage<KeyT, 0> : Storage {
  SmallSizeStorage() {}
  union {
    KeyT keys[0];
  };
};

template <typename KeyT, ssize_t SmallSize>
struct alignas(StorageAlignment<KeyT>) SmallSizeStorage : Storage {
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
class SetView : public SetInternal::RawHashtableViewBase<InputKeyT> {
 public:
  using BaseT = SetInternal::RawHashtableViewBase<InputKeyT>;
  using KeyT = typename BaseT::KeyT;
  using LookupResultT = typename SetInternal::LookupResult<KeyT>;

  using BaseT::Contains;

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupResultT;

  template <typename CallbackT>
  void ForEach(CallbackT callback);

 private:
  template <typename SetKeyT, ssize_t MinSmallSize>
  friend class Set;
  friend class SetBase<KeyT>;
  friend class SetInternal::RawHashtableBase<KeyT>;

  SetView() = default;
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit by design.
  SetView(BaseT base) : BaseT(base) {}
  SetView(ssize_t size, SetInternal::Storage* storage) : BaseT(size, storage) {}

  template <typename LookupKeyT>
  inline auto LookupHashed(LookupKeyT lookup_key) const -> LookupResultT;

  template <typename KeyCallbackT, typename GroupCallbackT>
  void ForEachHashed(KeyCallbackT key_callback, GroupCallbackT group_callback);
};

template <typename InputKeyT>
class SetBase : public SetInternal::RawHashtableBase<InputKeyT> {
 public:
  using BaseT = SetInternal::RawHashtableBase<InputKeyT>;
  using KeyT = typename BaseT::KeyT;
  using ViewT = SetView<KeyT>;
  using LookupResultT = SetInternal::LookupResult<KeyT>;
  using InsertResultT = SetInternal::InsertResult<KeyT>;

  using BaseT::Contains;

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const { return this->impl_view_; }

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupResultT {
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
  SetBase(int small_size, SetInternal::Storage* small_storage)
      : BaseT(small_size, small_storage) {}
  // An internal constructor used to build temporary map base objects with a
  // specific allocated size. This is used internally to build ephemeral maps.
  explicit SetBase(ssize_t arg_size);

  ~SetBase();

  template <typename LookupKeyT>
  auto GrowRehashAndInsertIndex(LookupKeyT lookup_key) -> ssize_t;
};

template <typename InputKeyT,
          ssize_t MinSmallSize = SetInternal::DefaultMinSmallSize<InputKeyT>()>
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
      SetInternal::ComputeSmallSize<MinSmallSize>();

  static_assert(SmallSize >= 0, "Cannot have a negative small size!");

  using SmallSizeStorageT =
      SetInternal::SmallSizeStorage<KeyT, SmallSize>;

  // Validate a collection of invariants between the small size storage layout
  // and the dynamically computed storage layout. We need to do this after both
  // are complete but in the context of a specific key type, value type, and
  // small size, so here is the best place.
  static_assert(SmallSize == 0 || alignof(SmallSizeStorageT) ==
                                      SetInternal::StorageAlignment<KeyT>,
                "Small size buffer must have the same alignment as a heap "
                "allocated buffer.");
  static_assert(
      SmallSize == 0 || (offsetof(SmallSizeStorageT, keys) ==
                         SetInternal::ComputeKeyStorageOffset<KeyT>(SmallSize)),
      "Offset to keys in small size storage doesn't match computed offset!");
  static_assert(SmallSize == 0 ||
                    sizeof(SmallSizeStorageT) ==
                        SetInternal::ComputeStorageSize<KeyT>(SmallSize),
                "The small size storage needs to match the dynamically "
                "computed storage size.");

  auto small_storage() const -> SetInternal::Storage* {
    return &small_storage_;
  }

  mutable SetInternal::SmallSizeStorage<KeyT, SmallSize> small_storage_;
};

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
  return __builtin_operator_delete(storage, allocated_size,
                                   std::align_val_t(StorageAlignment<KeyT>));
#else
  // Ensure `size` is used even in the fallback non-sized deallocation case.
  (void)size;
  return __builtin_operator_delete(storage,
                                   std::align_val_t(StorageAlignment<KeyT>));
#endif
}

}  // namespace SetInternal

template <typename KT>
template <typename LookupKeyT>
inline auto SetView<KT>::LookupHashed(LookupKeyT lookup_key) const
    -> LookupResultT {
  ssize_t index = SetInternal::LookupIndexHashed<KeyT>(lookup_key, this->size(),
                                                       this->storage_);
  if (index < 0) {
    return LookupResultT();
  }

  return LookupResultT(&this->keys_ptr()[index]);
}

template <typename KT>
template <typename LookupKeyT>
auto SetView<KT>::Lookup(LookupKeyT lookup_key) const -> LookupResultT {
  SetInternal::Prefetch(this->storage_);
  return this->LookupHashed(lookup_key);
}

template <typename KT>
template <typename CallbackT>
void SetView<KT>::ForEach(CallbackT callback) {
  this->ForEachIndex([callback](KeyT* keys, ssize_t i) { callback(keys[i]); },
                     [](auto...) {});
}

template <typename InputKeyT>
SetBase<InputKeyT>::SetBase(ssize_t arg_size)
    : BaseT(arg_size, SetInternal::AllocateStorage<KeyT>(arg_size)) {}

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
  SetInternal::DeallocateStorage<KeyT>(this->storage(), this->size());
}

template <typename InputKeyT>
template <typename LookupKeyT>
[[clang::noinline]] auto SetBase<InputKeyT>::GrowRehashAndInsertIndex(
    LookupKeyT lookup_key) -> ssize_t {
  // We grow into a new `MapBase` so that both the new and old maps are
  // fully functional until all the entries are moved over. However, we directly
  // manipulate the internals to short circuit many aspects of the growth.
  SetBase<KeyT> new_map(SetInternal::ComputeNewSize(this->size()));

  ssize_t insert_count = 0;
  ViewT(*this).ForEachIndex(
      [&](KeyT* old_keys, ssize_t old_index) {
        ++insert_count;
        KeyT& old_key = old_keys[old_index];
        ssize_t new_index = new_map.InsertIntoEmptyIndex(old_key);
        KeyT* new_keys = new_map.keys_ptr();
        new (&new_keys[new_index]) KeyT(std::move(old_key));
        old_key.~KeyT();
      },
      [](auto...) {});
  new_map.growth_budget_ -= insert_count;
  assert(new_map.growth_budget_ >= 0 &&
          "Must still have a growth budget after rehash!");

  if (LLVM_LIKELY(!this->is_small())) {
    // Old isn't a small buffer, so we need to deallocate it.
    SetInternal::DeallocateStorage<KeyT>(this->storage(), this->size());
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
    -> InsertResultT {
  ssize_t index = -1;
  // Try inserting if we have storage at all.
  if (this->size() > 0) {
    bool needs_insertion;
    std::tie(needs_insertion, index) = this->InsertIndexHashed(lookup_key);
    if (LLVM_LIKELY(!needs_insertion)) {
      assert(index >= 0 &&
             "Must have a valid group when we find an existing entry.");
      return InsertResultT(false, this->keys_ptr()[index]);
    }
  }

  if (index < 0) {
    assert(
        this->growth_budget_ == 0 &&
        "Shouldn't need to grow the table until we exhaust our growth budget!");

    index = this->GrowRehashAndInsertIndex(lookup_key);
  } else {
    assert(this->growth_budget_ >= 0 && "Cannot insert with zero budget!");
    --this->growth_budget_;
  }

  assert(index >= 0 && "Should have a group to insert into now.");

  KeyT* k = insert_cb(lookup_key, &this->keys_ptr()[index]);
  return InsertResultT(true, *k);
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
  this->ForEach([](KeyT& k) { k.~KeyT(); });

  // Deallocate the buffer.
  SetInternal::DeallocateStorage<KeyT>(this->storage(), this->size());

  // Re-initialize the whole thing.
  assert(this->small_size() == SmallSize);
  this->Init(SmallSize, small_storage());
}

}  // namespace Carbon

#endif  // CARBON_COMMON_SET_H_
