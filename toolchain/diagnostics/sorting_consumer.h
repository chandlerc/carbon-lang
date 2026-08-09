// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_SORTING_CONSUMER_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_SORTING_CONSUMER_H_

#include <utility>

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "toolchain/diagnostics/consumer.h"
#include "toolchain/diagnostics/diagnostic.h"

namespace Carbon::Diagnostics {

// Buffers incoming diagnostics for printing and sorting.
//
// Sorting is based on `last_byte_offset` without taking the filename into
// account. When processing multiple files, it's expected that separate
// consumers will be used in order to keep diagnostics distinct. Typically the
// location leading a diagnostic will be in the consumer's primary file, but if
// it needs to correspond to a different file, the `last_byte_offset` must
// still indicate an offset within the primary file.
class SortingConsumer : public Consumer {
 public:
  explicit SortingConsumer(Consumer& next_consumer)
      : next_consumer_(&next_consumer) {}

  ~SortingConsumer() override {
    // We choose not to automatically flush diagnostics here, because they are
    // likely to refer to data that gets destroyed before the diagnostics
    // consumer is destroyed, because the diagnostics consumer is typically
    // created before the objects that diagnostics refer into are created.
    CARBON_CHECK(diagnostics_.empty(),
                 "Must flush diagnostics consumer before destroying it");
  }

  // Buffers the diagnostic.
  auto HandleDiagnostic(Diagnostic diagnostic) -> void override {
    diagnostics_.push_back(std::move(diagnostic));
  }

  // Sorts and flushes buffered diagnostics.
  auto Flush() -> void override {
    llvm::stable_sort(
        diagnostics_, [](const Diagnostic& lhs, const Diagnostic& rhs) {
          if (lhs.last_byte_offset != rhs.last_byte_offset) {
            return lhs.last_byte_offset < rhs.last_byte_offset;
          }

          // A diagnostic generated on a scope is about everything reported
          // inside it, so it comes after those.
          if (lhs.is_on_scope != rhs.is_on_scope) {
            return !lhs.is_on_scope;
          }

          // Two diagnostics found at the same point print in the order their
          // messages appear, so that reading the output top to bottom reads the
          // file top to bottom. Which was emitted first says nothing the reader
          // can see: a diagnostic names the token it is about, which is not
          // always the one the phase had reached when it noticed.
          //
          // The position compared is that of whatever leads the diagnostic,
          // which is a context when it has one -- the same thing
          // `last_byte_offset` is rooted at, and the line the reader sees
          // first. One that names the file rather than anything in it has no
          // position to be ordered by: those compare equal among themselves,
          // keeping the order they were emitted in, and before anything
          // positioned at the same offset.
          auto position = [](const Diagnostic& diag) {
            const Context* leading = LeadingContext(diag);
            const Loc& loc = leading ? leading->loc : diag.message.loc;
            return loc.line_number > 0
                       ? std::pair(loc.line_number, loc.column_number)
                       : std::pair(0, 0);
          };
          return position(lhs) < position(rhs);
        });
    for (auto& diag : diagnostics_) {
      next_consumer_->HandleDiagnostic(std::move(diag));
    }
    diagnostics_.clear();
  }

 private:
  // A Diagnostic is undesirably large for inline storage by SmallVector, so we
  // specify 0.
  llvm::SmallVector<Diagnostic, 0> diagnostics_;

  Consumer* next_consumer_;
};

}  // namespace Carbon::Diagnostics

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_SORTING_CONSUMER_H_
