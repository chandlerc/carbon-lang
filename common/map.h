// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_MAP_H_
#define CARBON_COMMON_MAP_H_

#include <algorithm>
#include <concepts>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "common/raw_hashtable.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

// Forward declarations to resolve cyclic references.
template <typename KeyT, typename ValueT>
class MapView;
template <typename KeyT, typename ValueT>
class MapBase;
template <typename KeyT, typename ValueT, ssize_t MinSmallSize>
class Map;

// A read-only view type for a map from key to value.
//
// This view is a cheap-to-copy type that should be passed by value, but
// provides view or read-only reference semantics to the underlying map data
// structure.
//
// This should always be preferred to a `const`-ref parameter for the `MapBase`
// or `Map` type as it provides more flexibility and a cleaner API.
//
// Note that while this type is a read-only view, that applies to the underlying
// *map* data structure, not the individual entries stored within it. Those can
// be mutated freely. If we applied a deep-`const` design here, it would prevent
// using this type in many useful situations where the elements are mutated but
// the associative container is not. A view of immutable data can always be
// obtained by using `MapView<const T, const V>`, and we enable conversions to
// more-const views. This mirrors the semantics of views like `std::span`.
template <typename InputKeyT, typename InputValueT>
class MapView : RawHashtable::ViewImpl<InputKeyT, InputValueT> {
  using ImplT = RawHashtable::ViewImpl<InputKeyT, InputValueT>;
  using EntryT = typename ImplT::EntryT;

 public:
  using KeyT = typename ImplT::KeyT;
  using ValueT = typename ImplT::ValueT;

  // Result type used by lookup operations encodes whether the lookup was a
  // success as well as accessors for the key and value.
  class LookupKVResult {
   public:
    LookupKVResult() = default;
    explicit LookupKVResult(EntryT* entry) : entry_(entry) {}

    explicit operator bool() const { return entry_ != nullptr; }

    auto key() const -> KeyT& { return entry_->key(); }
    auto value() const -> ValueT& { return entry_->value(); }

   private:
    EntryT* entry_ = nullptr;
  };

  // Enable implicit conversions that add `const`-ness to either key or value
  // type. This is always safe to do with a view. We use a template to avoid
  // needing all 3 versions.
  template <typename OtherKeyT, typename OtherValueT>
  // NOLINTNEXTLINE(google-explicit-constructor)
  MapView(MapView<OtherKeyT, OtherValueT> other_view)
    requires(std::same_as<KeyT, OtherKeyT> ||
             std::same_as<KeyT, const OtherKeyT>) &&
            (std::same_as<ValueT, OtherValueT> ||
             std::same_as<ValueT, const OtherValueT>)
      : ImplT(other_view) {}

  // Tests whether a key is present in the map.
  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool;

  // Lookup a key in the map.
  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupKVResult;

  // Lookup a key in the map and try to return its value. Returns null on a
  // missing key.
  template <typename LookupKeyT>
  auto operator[](LookupKeyT lookup_key) const -> ValueT*;

  // Run the provided callback for every key and value in the map.
  template <typename CallbackT>
  void ForEach(CallbackT callback);

  // Count the probed keys. This routine is purely informational and for use in
  // benchmarking or logging of performance anomalies. It's returns have no
  // semantic guarantee at all.
  auto CountProbedKeys() -> ssize_t { return ImplT::CountProbedKeys(); }

 private:
  template <typename MapKeyT, typename MapValueT, ssize_t MinSmallSize>
  friend class Map;
  friend class MapBase<KeyT, ValueT>;
  friend class MapView<const KeyT, ValueT>;
  friend class MapView<KeyT, const ValueT>;
  friend class MapView<const KeyT, const ValueT>;

  MapView() = default;
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit by design.
  MapView(ImplT base) : ImplT(base) {}
  MapView(ssize_t size, RawHashtable::Storage* storage)
      : ImplT(size, storage) {}
};

// A base class for a `Map` type that remains mutable while type-erasing any SSO
// size.
//
// A pointer or reference to this type is the preferred way to pass a mutable
// handle to a `Map` type across API boundaries as it avoids encoding specific
// SSO sizing information while providing a near-complete mutable API.
template <typename InputKeyT, typename InputValueT>
class MapBase : protected RawHashtable::BaseImpl<InputKeyT, InputValueT> {
  using ImplT = RawHashtable::BaseImpl<InputKeyT, InputValueT>;
  using EntryT = typename ImplT::EntryT;

 public:
  using KeyT = typename ImplT::KeyT;
  using ValueT = typename ImplT::ValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using LookupKVResult = typename ViewT::LookupKVResult;

  // The result type for insertion operations both indicates whether an insert
  // was needed (as opposed to finding an existing element), and provides access
  // to the element's key and value.
  class InsertKVResult {
   public:
    InsertKVResult() = default;
    explicit InsertKVResult(bool inserted, EntryT& entry)
        : entry_(&entry), inserted_(inserted) {}

    auto is_inserted() const -> bool { return inserted_; }

    auto key() const -> KeyT& { return entry_->key(); }
    auto value() const -> ValueT& { return entry_->value(); }

   private:
    EntryT* entry_;
    bool inserted_;
  };

  // Implicitly convertible to the relevant view type.
  //
  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const { return this->view_impl(); }

  // We can't chain the above conversion with the conversions on `ViewT` to add
  // const, so explicitly support adding const to produce a view here.
  template <typename OtherKeyT, typename OtherValueT>
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator MapView<OtherKeyT, OtherValueT>() const
    requires(std::same_as<KeyT, OtherKeyT> ||
             std::same_as<const KeyT, OtherKeyT>) &&
            (std::same_as<ValueT, OtherValueT> ||
             std::same_as<const ValueT, OtherValueT>)
  {
    return ViewT(*this);
  }

  // Convenience forwarder to the view type.
  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool {
    return ViewT(*this).Contains(lookup_key);
  }

  // Convenience forwarder to the view type.
  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupKVResult {
    return ViewT(*this).Lookup(lookup_key);
  }

  // Convenience forwarder to the view type.
  template <typename LookupKeyT>
  auto operator[](LookupKeyT lookup_key) const -> ValueT* {
    return ViewT(*this)[lookup_key];
  }

  // Convenience forwarder to the view type.
  template <typename CallbackT>
  void ForEach(CallbackT callback) {
    return ViewT(*this).ForEach(callback);
  }

  // Insert a key and value into the map.
  template <typename LookupKeyT>
  auto Insert(LookupKeyT lookup_key, ValueT new_v) -> InsertKVResult;

  // Insert a key into the map and call the provided callback if necessary to
  // produce a new value when no existing value is found.
  //
  // TODO: The `;` formatting below appears to be bugs in clang-format with
  // concepts that should be filed upstream.
  template <typename LookupKeyT, typename ValueCallbackT>
  auto Insert(LookupKeyT lookup_key, ValueCallbackT value_cb) -> InsertKVResult
    requires(
        !std::same_as<ValueT, ValueCallbackT> &&
        std::convertible_to<decltype(std::declval<ValueCallbackT>()()), ValueT>)
  ;

  // Insert a key into the map and call the provided callback to allow in-place
  // construction of both the key and value when needed.
  template <typename LookupKeyT, typename InsertCallbackT>
  auto Insert(LookupKeyT lookup_key, InsertCallbackT insert_cb)
      -> InsertKVResult
    requires(!std::same_as<ValueT, InsertCallbackT> &&
             std::invocable<InsertCallbackT, LookupKeyT, void*, void*>);

  // Similar to insert, but an existing value is replaced with the provided one.
  template <typename LookupKeyT>
  auto Update(LookupKeyT lookup_key, ValueT new_v) -> InsertKVResult;

  // Similar to insert, but an existing value is replaced with the result of the
  // callback.
  template <typename LookupKeyT, typename ValueCallbackT>
  auto Update(LookupKeyT lookup_key, ValueCallbackT value_cb) -> InsertKVResult
    requires(
        !std::same_as<ValueT, ValueCallbackT> &&
        std::convertible_to<decltype(std::declval<ValueCallbackT>()()), ValueT>)
  ;

  // Similar to import, but with a distinct callback for updating an existing
  // key/value pair as opposed to inserting a new one.
  template <typename LookupKeyT, typename InsertCallbackT,
            typename UpdateCallbackT>
  auto Update(LookupKeyT lookup_key, InsertCallbackT insert_cb,
              UpdateCallbackT update_cb) -> InsertKVResult
    requires(!std::same_as<ValueT, InsertCallbackT> &&
             std::invocable<InsertCallbackT, LookupKeyT, void*, void*> &&
             std::invocable<UpdateCallbackT, KeyT&, ValueT&>);

  // Erase a key from the map.
  template <typename LookupKeyT>
  auto Erase(LookupKeyT lookup_key) -> bool;

  // Clear all key/value pairs from the map but leave the underlying hashtable
  // allocated and in place.
  void Clear();

  // Count the probed keys. This routine is purely informational and for use in
  // benchmarking or logging of performance anomalies. It's returns have no
  // semantic guarantee at all.
  auto CountProbedKeys() const -> ssize_t {
    return ViewT(*this).CountProbedKeys();
  }

 protected:
  using ImplT::ImplT;
};

// A data structure mapping from key to value.
//
// This map also supports SSO or small size optimization. The provided
// `SmallSize` type parameter indicates the small size buffer embedded. The
// default is zero, which always allocates a heap buffer on construction. When
// non-zero, must be a multiple of the `MaxGroupSize` of the underlying
// hashtable implementation.
//
// This data structure optimizes heavily for small, cheap to move and even copy
// key values. Ideally code can be shifted to ensure their keys fit this
// description.
//
// Note that this type should typically not appear on API boundaries and either
// `MapBase` or `MapView` should be used instead.
template <typename InputKeyT, typename InputValueT, ssize_t SmallSize = 0>
class Map : public RawHashtable::TableImpl<MapBase<InputKeyT, InputValueT>,
                                           SmallSize> {
  using BaseT = MapBase<InputKeyT, InputValueT>;
  using ImplT =
      RawHashtable::TableImpl<MapBase<InputKeyT, InputValueT>, SmallSize>;

 public:
  using KeyT = typename BaseT::KeyT;
  using ValueT = typename BaseT::ValueT;

  Map() = default;
  Map(const Map& arg) = default;
  template <ssize_t OtherMinSmallSize>
  explicit Map(const Map<KeyT, ValueT, OtherMinSmallSize>& arg) : ImplT(arg) {}
  Map(Map&& arg) = default;
  template <ssize_t OtherMinSmallSize>
  explicit Map(Map<KeyT, ValueT, OtherMinSmallSize>&& arg)
      : ImplT(std::move(arg)) {}

  // Reset the entire state of the hashtable to as it was when constructed,
  // throwing away any intervening allocations.
  void Reset();
};

template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
auto MapView<InputKeyT, InputValueT>::Contains(LookupKeyT lookup_key) const
    -> bool {
  return this->LookupEntry(lookup_key) != nullptr;
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::Lookup(LookupKeyT lookup_key) const -> LookupKVResult {
  return LookupKVResult(this->LookupEntry(lookup_key));
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
  this->ForEachEntry(
      [callback](EntryT& entry) { callback(entry.key(), entry.value()); },
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
    -> InsertKVResult
  requires(
      !std::same_as<ValueT, ValueCallbackT> &&
      std::convertible_to<decltype(std::declval<ValueCallbackT>()()), ValueT>)
{
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
    -> InsertKVResult
  requires(!std::same_as<ValueT, InsertCallbackT> &&
           std::invocable<InsertCallbackT, LookupKeyT, void*, void*>)
{
  auto [entry, inserted] = this->InsertImpl(lookup_key);
  CARBON_DCHECK(entry) << "Should always result in a valid index.";

  if (LLVM_LIKELY(!inserted)) {
    return InsertKVResult(false, *entry);
  }

  insert_cb(lookup_key, static_cast<void*>(&entry->key_storage),
            static_cast<void*>(&entry->value_storage));
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
    -> InsertKVResult
  requires(
      !std::same_as<ValueT, ValueCallbackT> &&
      std::convertible_to<decltype(std::declval<ValueCallbackT>()()), ValueT>)
{
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
    -> InsertKVResult
  requires(!std::same_as<ValueT, InsertCallbackT> &&
           std::invocable<InsertCallbackT, LookupKeyT, void*, void*> &&
           std::invocable<UpdateCallbackT, KeyT&, ValueT&>)
{
  auto [entry, inserted] = this->InsertImpl(lookup_key);
  CARBON_DCHECK(entry) << "Should always result in a valid index.";
  // EntryT& entry = this->entries()[index];

  if (LLVM_LIKELY(!inserted)) {
    update_cb(entry->key(), entry->value());
    return InsertKVResult(false, *entry);
  }

  insert_cb(lookup_key, static_cast<void*>(&entry->key_storage),
            static_cast<void*>(&entry->value_storage));
  return InsertKVResult(true, *entry);
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::Erase(LookupKeyT lookup_key) -> bool {
  return this->EraseImpl(lookup_key);
}

template <typename KeyT, typename ValueT>
void MapBase<KeyT, ValueT>::Clear() {
  this->ClearImpl();
}

template <typename KeyT, typename ValueT, ssize_t SmallSize>
void Map<KeyT, ValueT, SmallSize>::Reset() {
  this->ResetImpl();
}

}  // namespace Carbon

#endif  // CARBON_COMMON_MAP_H_
