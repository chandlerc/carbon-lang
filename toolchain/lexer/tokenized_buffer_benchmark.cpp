// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include "llvm/ADT/StringRef.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "common/check.h"

namespace Carbon {
namespace {

static auto MakeSourceBuffer(llvm::StringRef source_text) -> SourceBuffer {
  return SourceBuffer::CreateFromText(source_text, "benchmark.carbon");
}

static void BM_BasicLexing(benchmark::State& state) {
  auto source = MakeSourceBuffer("fn F() {}");
  for (auto _ : state) {
    auto tokens = TokenizedBuffer::Lex(source, NullDiagnosticConsumer());
    CHECK(!tokens.HasErrors()) << "Errors while benchmarking the lexer!";
    benchmark::DoNotOptimize(&tokens);
  }
}

BENCHMARK(BM_BasicLexing);

}  // namespace
}  // namespace Carbon

BENCHMARK_MAIN();
