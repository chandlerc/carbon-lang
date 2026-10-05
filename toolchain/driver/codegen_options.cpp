// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/codegen_options.h"

#include <string>

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/TargetParser/RISCVISAInfo.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/TargetParser/Triple.h"

namespace Carbon {

auto CodegenOptions::Build(CommandLine::CommandBuilder& b) -> void {
  b.AddStringOption(
      {
          .name = "target",
          .help = R"""(
Select a target platform. Uses the LLVM target syntax. Also known as a "triple"
for historical reasons.

This corresponds to the `target` flag to Clang and accepts the same strings
documented there:
https://clang.llvm.org/docs/CrossCompilation.html#target-triple
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(host);
        arg_b.Set(&target);
      });

  b.AddStringOption(
      {
          .name = "target-cpu",
          .value_name = "CPU",
          .help = R"""(
Select a target CPU or architecture level.

Accepts CPU names (such as `znver4`, `apple-m1`, `spacemit-x60`, or `native`) as
well as architecture strings or profiles (such as `x86-64-v3`, `armv9-a`,
`rv64gc`, or `rva22u64`). This corresponds to Clang's `-march` and `-mcpu`
flags, selecting the appropriate underlying flag for the target architecture.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&target_cpu); });

  b.AddStringOption(
      {
          .name = "target-cpu-tune",
          .value_name = "CPU",
          .help = R"""(
Select a target CPU to tune code generation for without enabling new
instruction set extensions.

Accepts CPU names (such as `znver4`, `apple-m1`, `spacemit-x60`, `generic`, or
`native`). This corresponds to Clang's `-mtune` flag.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&target_cpu_tune); });

  b.AddStringOption(
      {
          .name = "target-cpu-features",
          .value_name = "FEATURES",
          .help = R"""(
Select target CPU features to enable or disable.

Accepts a comma-separated list of LLVM target feature names, optionally prefixed
with `+` to enable or `-` to disable (such as `+avx2,-fma` or `sve,crc`). Bare
feature names without a prefix are enabled.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&target_cpu_features); });
}

auto CodegenOptions::GetTargetCpuClangArg() const -> std::string {
  if (target_cpu.empty()) {
    return "";
  }

  llvm::Triple triple(target);
  if (triple.isX86()) {
    // On x86, Clang uses `-march=` for both microarchitecture levels (such as
    // `x86-64-v3`) and specific CPUs (such as `znver4` or `native`), and does
    // not support `-mcpu=`.
    return llvm::formatv("-march={0}", target_cpu).str();
  }

  if (triple.isAArch64()) {
    // On AArch64, architecture strings (such as `armv8-a` or `armv9.2-a+sve`)
    // use `-march=`, while CPU names (such as `neoverse-v2`, `apple-m1`, or
    // `native`) use `-mcpu=`.
    auto [base, _] = target_cpu.split('+');
    if (llvm::AArch64::parseArch(base) != nullptr || base.starts_with("armv")) {
      return llvm::formatv("-march={0}", target_cpu).str();
    }
    return llvm::formatv("-mcpu={0}", target_cpu).str();
  }

  if (triple.isRISCV()) {
    // On RISC-V, ISA strings (such as `rv64gc`) and profiles (such as
    // `rva22u64`) use `-march=`, while CPU names (such as `spacemit-x60`,
    // `sifive-u74`, or `native`) use `-mcpu=`.
    auto isa_info = llvm::RISCVISAInfo::parseArchString(
        target_cpu, /*EnableExperimentalExtension=*/true);
    if (isa_info) {
      return llvm::formatv("-march={0}", target_cpu).str();
    }
    llvm::consumeError(isa_info.takeError());
    if (!llvm::RISCV::parseCPU(target_cpu, triple.isRISCV64()) &&
        (target_cpu.starts_with("rv32") || target_cpu.starts_with("rv64") ||
         target_cpu.starts_with("rva") || target_cpu.starts_with("rvb"))) {
      return llvm::formatv("-march={0}", target_cpu).str();
    }
    return llvm::formatv("-mcpu={0}", target_cpu).str();
  }

  return llvm::formatv("-mcpu={0}", target_cpu).str();
}

auto CodegenOptions::AppendClangArgs(
    llvm::SmallVectorImpl<std::string>& args) const -> void {
  std::string target_cpu_arg = GetTargetCpuClangArg();
  if (!target_cpu_arg.empty()) {
    args.push_back(std::move(target_cpu_arg));
  }
  if (!target_cpu_tune.empty()) {
    args.push_back(llvm::formatv("-mtune={0}", target_cpu_tune).str());
  }
  if (!target_cpu_features.empty()) {
    llvm::SmallVector<llvm::StringRef> features;
    target_cpu_features.split(features, ',', /*MaxSplit=*/-1,
                              /*KeepEmpty=*/false);
    for (llvm::StringRef feature : features) {
      args.push_back("-Xclang");
      args.push_back("-target-feature");
      args.push_back("-Xclang");
      if (feature.size() > 1 &&
          (feature.starts_with('+') || feature.starts_with('-'))) {
        args.push_back(feature.str());
      } else {
        args.push_back(llvm::formatv("+{0}", feature).str());
      }
    }
  }
}

auto CodegenOptions::GetClangArgs() const -> llvm::SmallVector<std::string> {
  llvm::SmallVector<std::string> args;
  AppendClangArgs(args);
  return args;
}

}  // namespace Carbon
