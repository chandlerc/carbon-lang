// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_runtimes.h"

#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <mutex>
#include <numeric>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>

#include "common/check.h"
#include "common/error.h"
#include "common/filesystem.h"
#include "common/latch.h"
#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/runtimes_build_info.h"
#include "toolchain/driver/clang_runner.h"
#include "toolchain/driver/runtimes_cache.h"

namespace Carbon {

auto ClangRuntimesBuilderBase::ArchiveBuilder::Setup(Latch::Handle latch_handle)
    -> void {
  // `NewArchiveMember` isn't default constructable unfortunately, so we have to
  // manually populate the vector with errors that we'll replace with the actual
  // result in each thread.
  objs_.reserve(src_files_.size());
  for (const auto& _ : src_files_) {
    objs_.push_back(Error("Never constructed archive member!"));
  }

  // Finish building the archive when the last compile finishes.
  Latch::Handle comp_latch_handle =
      compilation_latch_.Init([this, latch_handle] { result_ = Finish(); });

  // Add all the compiles to the thread pool to run concurrently. The latch
  // handle ensures the last one triggers the finishing closure above.
  for (auto [src_file, obj] : llvm::zip_equal(src_files_, objs_)) {
    builder_->tasks_.async([this, comp_latch_handle, src_file, &obj]() mutable {
      obj = CompileMember(src_file);
    });
  }
}

auto ClangRuntimesBuilderBase::ArchiveBuilder::Finish() -> ErrorOr<Success> {
  // We build this directly into the desired location as this is expected to be
  // a staging directory and cleaned up on errors. We do need to create any
  // intermediate directories.
  Filesystem::DirRef runtimes_dir = builder_->runtimes_builder_->dir();
  if (archive_path_.has_parent_path()) {
    CARBON_RETURN_IF_ERROR(
        runtimes_dir.CreateDirectories(archive_path_.parent_path()));
  }

  // Check if any compilations ended up producing an error. If so, return the
  // first error for the entire function. Otherwise, move the archive member
  // into a direct vector to match the required archive building API.
  llvm::SmallVector<llvm::NewArchiveMember> unwrapped_objs;
  unwrapped_objs.reserve(objs_.size());
  for (auto& obj : objs_) {
    if (!obj.ok()) {
      return std::move(obj).error();
    }
    unwrapped_objs.push_back(*std::move(obj));
  }
  objs_.clear();

  // Remove any directories created for the object files, the files should
  // already be removed. We walk the sorted list of these in reverse so we
  // remove child directories before parent directories.
  for (const auto& obj_dir : llvm::reverse(obj_dirs_)) {
    auto rmdir_result = runtimes_dir.Rmdir(obj_dir);
    // Don't return an error on failure here as this has no problematic
    // effect, just log that we couldn't clean up a directory.
    if (!rmdir_result.ok()) {
      CARBON_VLOG("Unable to remove object directory `{0}` in the runtime: {1}",
                  obj_dir.native(), rmdir_result.error());
    }
  }

  // Write the actual archive.
  CARBON_ASSIGN_OR_RETURN(
      Filesystem::WriteFile archive_file,
      runtimes_dir.OpenWriteOnly(archive_path_, Filesystem::CreateAlways));
  {
    llvm::raw_fd_ostream archive_os = archive_file.WriteStream();
    llvm::Error archive_err = llvm::writeArchiveToStream(
        archive_os, unwrapped_objs, llvm::SymtabWritingMode::NormalSymtab,
        builder_->target_triple_.isOSDarwin() ? llvm::object::Archive::K_DARWIN
                                              : llvm::object::Archive::K_GNU,
        /*Deterministic=*/true, /*Thin=*/false);
    // The presence of an error is `true`.
    if (archive_err) {
      (void)std::move(archive_file).Close();
      return Error(llvm::toString(std::move(archive_err)));
    }
  }
  // Close and return any errors, potentially from the writes above.
  CARBON_RETURN_IF_ERROR(std::move(archive_file).Close());

  if (generate_syms_) {
    CARBON_RETURN_IF_ERROR(WriteSymsFile(unwrapped_objs));
  }
  return Success();
}

auto ClangRuntimesBuilderBase::ArchiveBuilder::WriteSymsFile(
    llvm::ArrayRef<llvm::NewArchiveMember> members) -> ErrorOr<Success> {
  static constexpr llvm::StringLiteral NewDeleteSymbols[] = {
      "_Znam",
      "_ZnamRKSt9nothrow_t",
      "_Znwm",
      "_ZnwmRKSt9nothrow_t",
      "_Znaj",
      "_ZnajRKSt9nothrow_t",
      "_Znwj",
      "_ZnwjRKSt9nothrow_t",
      "_ZnwmSt11align_val_t",
      "_ZnwmSt11align_val_tRKSt9nothrow_t",
      "_ZnwjSt11align_val_t",
      "_ZnwjSt11align_val_tRKSt9nothrow_t",
      "_ZnamSt11align_val_t",
      "_ZnamSt11align_val_tRKSt9nothrow_t",
      "_ZnajSt11align_val_t",
      "_ZnajSt11align_val_tRKSt9nothrow_t",
      "_ZdaPv",
      "_ZdaPvRKSt9nothrow_t",
      "_ZdlPv",
      "_ZdlPvRKSt9nothrow_t",
      "_ZdaPvm",
      "_ZdlPvm",
      "_ZdaPvj",
      "_ZdlPvj",
      "_ZdlPvSt11align_val_t",
      "_ZdlPvSt11align_val_tRKSt9nothrow_t",
      "_ZdaPvSt11align_val_t",
      "_ZdaPvSt11align_val_tRKSt9nothrow_t",
      "_ZdlPvmSt11align_val_t",
      "_ZdaPvmSt11align_val_t",
      "_ZdlPvjSt11align_val_t",
      "_ZdaPvjSt11align_val_t",
  };
  static constexpr llvm::StringLiteral VersionedFunctions[] = {
      "memcpy",
      "pthread_attr_getaffinity_np",
      "pthread_cond_broadcast",
      "pthread_cond_destroy",
      "pthread_cond_init",
      "pthread_cond_signal",
      "pthread_cond_timedwait",
      "pthread_cond_wait",
      "realpath",
      "sched_getaffinity",
  };

  llvm::StringSet<> function_set;
  for (const llvm::NewArchiveMember& member : members) {
    auto obj_or_err = llvm::object::ObjectFile::createObjectFile(
        member.Buf->getMemBufferRef());
    if (!obj_or_err) {
      return Error(llvm::toString(obj_or_err.takeError()));
    }
    const auto* elf_obj =
        llvm::dyn_cast<llvm::object::ELFObjectFileBase>(obj_or_err->get());
    if (!elf_obj) {
      continue;
    }

    for (llvm::object::ELFSymbolRef sym : elf_obj->symbols()) {
      auto flags_or_err = sym.getFlags();
      if (!flags_or_err) {
        return Error(llvm::toString(flags_or_err.takeError()));
      }
      uint32_t flags = *flags_or_err;
      if (!(flags & llvm::object::BasicSymbolRef::SF_Global) ||
          (flags & (llvm::object::BasicSymbolRef::SF_Undefined |
                    llvm::object::BasicSymbolRef::SF_Common |
                    llvm::object::BasicSymbolRef::SF_Absolute))) {
        continue;
      }
      if (sym.getELFType() == llvm::ELF::STT_GNU_IFUNC) {
        continue;
      }

      bool is_func_sym = false;
      if (flags & llvm::object::BasicSymbolRef::SF_Weak) {
        // Matches `llvm-nm` type 'W' (weak symbol other than STT_OBJECT).
        is_func_sym = sym.getELFType() != llvm::ELF::STT_OBJECT;
      } else if (sym.getBinding() == llvm::ELF::STB_GLOBAL) {
        auto sec_or_err = sym.getSection();
        if (!sec_or_err) {
          return Error(llvm::toString(sec_or_err.takeError()));
        }
        if (*sec_or_err != elf_obj->section_end()) {
          llvm::object::ELFSectionRef elf_sec(**sec_or_err);
          uint64_t sec_flags = elf_sec.getFlags();
          // Matches `llvm-nm` type 'T' (or 'D' on PowerPC).
          if (sec_flags & llvm::ELF::SHF_EXECINSTR) {
            is_func_sym = true;
          } else if (builder_->target_triple_.isPPC() &&
                     (sec_flags & llvm::ELF::SHF_ALLOC) &&
                     (sec_flags & llvm::ELF::SHF_WRITE)) {
            is_func_sym = true;
          }
        }
      }
      if (!is_func_sym) {
        continue;
      }

      auto name_or_err = sym.getName();
      if (!name_or_err) {
        return Error(llvm::toString(name_or_err.takeError()));
      }
      if (!name_or_err->empty()) {
        function_set.insert(*name_or_err);
      }
    }
  }

  llvm::StringSet<> exported;
  for (const auto& entry : function_set) {
    llvm::StringRef func = entry.getKey();
    if (llvm::is_contained(NewDeleteSymbols, func)) {
      exported.insert(func);
      continue;
    }
    llvm::StringRef interceptor_rest = func;
    if (interceptor_rest.consume_front("___interceptor_") ||
        interceptor_rest.consume_front("__interceptor_")) {
      exported.insert(func);
      if (function_set.contains(interceptor_rest) &&
          !llvm::is_contained(VersionedFunctions, interceptor_rest)) {
        exported.insert(interceptor_rest);
      }
      continue;
    }
    if (func.starts_with("__sanitizer_")) {
      exported.insert(func);
    }
  }

  if (syms_extra_path_) {
    CARBON_ASSIGN_OR_RETURN(
        std::string extra_content,
        Filesystem::Cwd().ReadFileToString(*syms_extra_path_));
    llvm::SmallVector<llvm::StringRef> lines;
    llvm::StringRef(extra_content).split(lines, '\n');
    for (llvm::StringRef line : lines) {
      line = line.rtrim();
      if (!line.empty()) {
        exported.insert(line);
      }
    }
  }

  llvm::SmallVector<llvm::StringRef> sorted_syms(exported.keys());
  llvm::sort(sorted_syms);

  std::filesystem::path syms_path = archive_path_;
  syms_path += ".syms";
  Filesystem::DirRef runtimes_dir = builder_->runtimes_builder_->dir();
  CARBON_ASSIGN_OR_RETURN(
      Filesystem::WriteFile syms_file,
      runtimes_dir.OpenWriteOnly(syms_path, Filesystem::CreateAlways));
  {
    llvm::raw_fd_ostream syms_os = syms_file.WriteStream();
    syms_os << "{\n";
    for (llvm::StringRef sym : sorted_syms) {
      syms_os << "  " << sym << ";\n";
    }
    syms_os << "};\n";
  }
  CARBON_RETURN_IF_ERROR(std::move(syms_file).Close());
  return Success();
}

auto ClangRuntimesBuilderBase::ArchiveBuilder::CreateObjDir(
    const std::filesystem::path& src_path) -> ErrorOr<Success> {
  auto obj_dir_path = src_path.parent_path();
  if (obj_dir_path.empty()) {
    return Success();
  }

  std::scoped_lock lock(obj_dirs_mu_);
  auto* it = std::lower_bound(obj_dirs_.begin(), obj_dirs_.end(), obj_dir_path);
  if (it != obj_dirs_.end() && *it == obj_dir_path) {
    return Success();
  }

  auto create_result =
      builder_->runtimes_builder_->dir().CreateDirectories(obj_dir_path);
  if (!create_result.ok()) {
    return Error(llvm::formatv(
        "Unable to create object directory mirroring source file `{0}`: {1}",
        src_path, create_result.error()));
  }

  it = obj_dirs_.insert(it, obj_dir_path);

  // Also insert any parent paths. These should always sort earlier.
  CARBON_DCHECK(!obj_dir_path.has_parent_path() ||
                obj_dir_path.parent_path() < obj_dir_path);
  obj_dir_path = obj_dir_path.parent_path();
  while (!obj_dir_path.empty()) {
    it = std::lower_bound(obj_dirs_.begin(), it, obj_dir_path);
    if (*it != obj_dir_path) {
      it = obj_dirs_.insert(it, obj_dir_path);
    }
    obj_dir_path = obj_dir_path.parent_path();
  }
  return Success();
}

auto ClangRuntimesBuilderBase::ArchiveBuilder::CompileMember(
    llvm::StringRef src_file) -> ErrorOr<llvm::NewArchiveMember> {
  // Create any obj subdirectories needed for this file.
  std::filesystem::path rel_obj_path = objs_dir_ / std::string_view(src_file);
  rel_obj_path += ".o";
  CARBON_RETURN_IF_ERROR(CreateObjDir(rel_obj_path));
  std::filesystem::path src_path = srcs_root_ / std::string_view(src_file);
  std::filesystem::path obj_path =
      builder_->runtimes_builder_->path() / rel_obj_path;
  CARBON_VLOG("Building `{0}' from `{1}`...\n", obj_path, src_file);

  llvm::SmallVector<llvm::StringRef> args(cflags_);

  // Add language-specific flags based on file extension.
  //
  // Currently, we hard code a sufficiently "recent" C++ standard, but this is
  // arbitrary and brittle. We'll have to update these any time one of the
  // libraries uses a too-new feature.
  //
  // TODO: We should eventually switch to something more like `/std:c++latest`
  // in MSVC-style command lines, but would need that implemented in Clang.
  if (src_file.ends_with(".c")) {
    args.push_back("-std=c11");
  } else if (src_file.ends_with(".cpp")) {
    args.push_back("-std=c++26");
  }

  // Collect the additional required flags and dynamic flags for this builder.
  args.push_back("-c");
  args.push_back(builder_->target_flag_);
  for (const std::string& target_arg : builder_->target_args_) {
    args.push_back(target_arg);
  }
  args.append({
      "-o",
      obj_path.native(),
      src_path.native(),
  });
  CARBON_ASSIGN_OR_RETURN(bool success,
                          builder_->clang_->RunWithNoRuntimes(args));
  if (!success) {
    return Error(
        llvm::formatv("Failed to compile runtime source file '{0}'", src_file));
  }

  auto obj_result = llvm::NewArchiveMember::getFile(obj_path.native(),
                                                    /*Deterministic=*/true);
  if (!obj_result) {
    return Error(llvm::formatv("Unable to read `{0}` object file: {1}",
                               src_file,
                               llvm::fmt_consume(obj_result.takeError())));
  }

  // Only use the basename as the member name to match the behavior of `ar`. We
  // also specifically use the LLVM path function rather than the standard
  // library as it allows us to get the filename within the member-owned
  // filename storage.
  obj_result->MemberName = llvm::sys::path::filename(obj_result->MemberName);

  // Unlink the object file once we've read it -- we only want to retain the
  // copy inside the archive member and there's no advantage to using
  // thin-archives or something else that leaves the object file in place.
  // However, we log and ignore any errors here as they aren't fatal.
  auto unlink_result = builder_->runtimes_builder_->dir().Unlink(obj_path);
  if (!unlink_result.ok()) {
    CARBON_VLOG("Unable to unlink object file `{0}`: {1}\n", obj_path,
                unlink_result.error());
  }

  return std::move(*obj_result);
}

template <Runtimes::Component Component>
  requires IsClangArchiveRuntimes<Component>
ClangArchiveRuntimesBuilder<Component>::ClangArchiveRuntimesBuilder(
    ClangRunner* clang, llvm::ThreadPoolInterface* threads,
    llvm::Triple target_triple, Runtimes* runtimes,
    const Runtimes::Cache::Features& features)
    : ClangRuntimesBuilderBase(clang, threads, std::move(target_triple),
                               features) {
  // Ensure we're on a platform where we _can_ build a working runtime.
  if (target_triple_.isOSWindows()) {
    result_ =
        Error("TODO: Windows runtimes are untested and not yet supported.");
    return;
  }

  auto build_dir_or_error = runtimes->Build(Component);
  if (!build_dir_or_error.ok()) {
    result_ = std::move(build_dir_or_error).error();
    return;
  }
  auto build_dir = *(std::move(build_dir_or_error));
  CARBON_KIND_SWITCH(std::move(build_dir)) {
    case CARBON_KIND(std::filesystem::path build_dir_path): {
      // Found cached build.
      result_ = std::move(build_dir_path);
      return;
    }
    case CARBON_KIND(Runtimes::Builder builder): {
      runtimes_builder_ = std::move(builder);
      // Building the runtimes is handled below.
      break;
    }
  }

  if constexpr (Component == Runtimes::LibUnwind) {
    archive_path_ = std::filesystem::path("lib") / "libunwind.a";
    include_paths_ = {installation().libunwind_path() / "include"};
  } else if constexpr (Component == Runtimes::Libcxx) {
    archive_path_ = std::filesystem::path("lib") / "libc++.a";
    include_paths_ = {
        installation().libcxx_path() / "include",
        // Some private headers of libc++ are nested in the source directory.
        installation().libcxx_path() / "src",
        installation().libcxxabi_path() / "include",
        // Libc++ also uses llvm-libc header-only libraries for parts of its
        // implementation. All the `#include`s are relative to the root of the
        // internal libc source tree rather than an `include` directory.
        installation().libc_path() / "internal",
    };
  } else {
    static_assert(false,
                  "Invalid runtimes component for an archive runtime builder.");
  }

  archive_.emplace(this, archive_path_, installation().root(),
                   CollectSrcFiles(), CollectCflags());
  tasks_.async([this]() mutable { Setup(); });
}

template <Runtimes::Component Component>
  requires IsClangArchiveRuntimes<Component>
auto ClangArchiveRuntimesBuilder<Component>::CollectSrcFiles()
    -> llvm::SmallVector<llvm::StringRef> {
  if constexpr (Component == Runtimes::LibUnwind) {
    return llvm::to_vector_of<llvm::StringRef>(llvm::make_filter_range(
        RuntimesBuildInfo::LibunwindSrcs, [](llvm::StringRef src) {
          return src.ends_with(".c") || src.ends_with(".cpp") ||
                 src.ends_with(".S");
        }));
  } else if constexpr (Component == Runtimes::Libcxx) {
    auto libcxx_target_srcs =
        target_triple_.isOSWindows()
            ? llvm::ArrayRef(RuntimesBuildInfo::LibcxxWin32Srcs)
        : target_triple_.isMacOSX()
            ? llvm::ArrayRef(RuntimesBuildInfo::LibcxxMacosSrcs)
            : llvm::ArrayRef(RuntimesBuildInfo::LibcxxLinuxSrcs);
    auto libcxx_srcs = llvm::make_filter_range(
        libcxx_target_srcs,
        [](llvm::StringRef src) { return src.ends_with(".cpp"); });

    auto libcxxabi_srcs = llvm::make_filter_range(
        RuntimesBuildInfo::LibcxxabiSrcs,
        [](llvm::StringRef src) { return src.ends_with(".cpp"); });
    return llvm::to_vector(
        llvm::concat<llvm::StringRef>(libcxx_srcs, libcxxabi_srcs));
  } else {
    static_assert(false,
                  "Invalid runtimes component for an archive runtime builder.");
  }
}

template <Runtimes::Component Component>
  requires IsClangArchiveRuntimes<Component>
auto ClangArchiveRuntimesBuilder<Component>::CollectCflags()
    -> llvm::SmallVector<llvm::StringRef> {
  // Start with some hard-coded flags used across any runtime.
  //
  // TODO: It would be nice to plumb through an option to enable (some) warnings
  // when building runtimes, especially for folks working directly on the Carbon
  // toolchain to validate our builds of runtimes.
  llvm::SmallVector<llvm::StringRef> cflags = {
      "-no-canonical-prefixes",
      "-w",
  };

  if constexpr (Component == Runtimes::LibUnwind) {
    llvm::append_range(cflags, RuntimesBuildInfo::LibunwindCopts);
  } else if constexpr (Component == Runtimes::Libcxx) {
    llvm::append_range(cflags, RuntimesBuildInfo::LibcxxCopts);
  } else {
    static_assert(false,
                  "Invalid runtimes component for an archive runtime builder.");
  }

  for (const auto& include_path : include_paths_) {
    cflags.append({"-I", include_path.native()});
  }
  if (asan_) {
    cflags.push_back("-fsanitize=address");
  }
  return cflags;
}

template <Runtimes::Component Component>
  requires IsClangArchiveRuntimes<Component>
auto ClangArchiveRuntimesBuilder<Component>::Setup() -> void {
  // Finish building the runtime once the archive is built.
  Latch::Handle latch_handle = step_counter_.Init(
      [this]() mutable { tasks_.async([this]() mutable { Finish(); }); });

  // Start building the archive itself with a handle to detect when complete.
  archive_->Setup(std::move(latch_handle));
}

template <Runtimes::Component Component>
  requires IsClangArchiveRuntimes<Component>
auto ClangArchiveRuntimesBuilder<Component>::Finish() -> void {
  CARBON_VLOG("Finished building {0}...\n", archive_path_);
  if (!archive_->result().ok()) {
    result_ = std::move(archive_->result()).error();
    return;
  }

  result_ = (*std::move(runtimes_builder_)).Commit();
}

template class ClangArchiveRuntimesBuilder<Runtimes::LibUnwind>;
template class ClangArchiveRuntimesBuilder<Runtimes::Libcxx>;

auto ClangResourceDirBuilder::GetDarwinOsSuffix(llvm::Triple target_triple)
    -> llvm::StringRef {
  switch (target_triple.getOS()) {
    case llvm::Triple::IOS:
      return target_triple.isSimulatorEnvironment() ? "iossim" : "ios";
    case llvm::Triple::WatchOS:
      return target_triple.isSimulatorEnvironment() ? "watchossim" : "watchos";
    case llvm::Triple::TvOS:
      return target_triple.isSimulatorEnvironment() ? "tvossim" : "tvos";
    case llvm::Triple::XROS:
      return target_triple.isSimulatorEnvironment() ? "xrossim" : "xros";
    default:
      return "osx";
  }
}

ClangResourceDirBuilder::ClangResourceDirBuilder(
    ClangRunner* clang, llvm::ThreadPoolInterface* threads,
    llvm::Triple target_triple, Runtimes* runtimes,
    const Runtimes::Cache::Features& features)
    : ClangRuntimesBuilderBase(clang, threads, std::move(target_triple),
                               features),
      crt_begin_result_(Error("Never built CRT begin file!")),
      crt_end_result_(Error("Never built CRT end file!")) {
  // Ensure we're on a platform where we _can_ build a working runtime.
  if (target_triple_.isOSWindows()) {
    result_ =
        Error("TODO: Windows runtimes are untested and not yet supported.");
    return;
  }

  auto build_dir_or_error = runtimes->Build(Runtimes::ClangResourceDir);
  if (!build_dir_or_error.ok()) {
    result_ = std::move(build_dir_or_error).error();
    return;
  }
  auto build_dir = *std::move(build_dir_or_error);
  if (std::holds_alternative<std::filesystem::path>(build_dir)) {
    // Found cached build.
    result_ = std::get<std::filesystem::path>(std::move(build_dir));
    return;
  }

  runtimes_builder_ = std::get<Runtimes::Builder>(std::move(build_dir));
  lib_path_ = std::filesystem::path("lib");
  std::filesystem::path builtins_name = "libclang_rt.builtins.a";
  if (target_triple.isOSDarwin()) {
    // Darwin targets don't use the full triple, and don't include the
    // architecture in the resource directory naming structure.
    //
    // TODO: We should add support for embedded Darwin as well which uses a
    // different layout.
    lib_path_ /= "darwin";

    // Darwin targets also use a custom naming convention for the builtins
    // archive.
    builtins_name =
        llvm::formatv("libclang_rt.{0}.a", GetDarwinOsSuffix(target_triple_))
            .str();
  } else {
    lib_path_ /= target_triple_.str();
  }

  // TODO: Currently, we only need a single include path to see headers inside
  // the `builtins` directory. However, we're anticipating needing more, for
  // example to support SipHash. If that need doesn't materialize, we should
  // simplify this to a single path instead of a vector.
  include_paths_.push_back(installation().runtimes_root() / "builtins");

  llvm::SmallVector<llvm::StringRef> copts = {
      "-no-canonical-prefixes",
      "-w",
      "-fno-sanitize=all",
  };
  llvm::append_range(copts, RuntimesBuildInfo::BuiltinsCopts);
  for (const auto& include_path : include_paths_) {
    copts.append({"-I", include_path.native()});
  }
  archive_.emplace(this, lib_path_ / builtins_name, installation().root(),
                   CollectBuiltinsSrcFiles(), copts);

  if (asan_) {
    asan_include_path_ = installation().runtimes_root() / "compiler-rt/lib";

    auto make_asan_copts = [&](llvm::ArrayRef<llvm::StringLiteral> base_copts) {
      llvm::SmallVector<llvm::StringRef> asan_copts = {
          "-no-canonical-prefixes",
          "-w",
          "-fno-sanitize=all",
      };
      llvm::append_range(asan_copts, base_copts);
      if (target_triple_.isOSDarwin()) {
        llvm::append_range(asan_copts, RuntimesBuildInfo::AsanDarwinCopts);
      }
      asan_copts.append({"-I", asan_include_path_.native()});
      return asan_copts;
    };

    if (target_triple_.isOSDarwin()) {
      llvm::SmallVector<llvm::StringRef> asan_cxx_srcs;
      llvm::append_range(asan_cxx_srcs, RuntimesBuildInfo::AsanCxxSrcs);
      llvm::append_range(asan_cxx_srcs, RuntimesBuildInfo::UbsanCxxSrcs);

      asan_archive_.emplace(
          this, lib_path_ / "libclang_rt.asan.a", installation().root(),
          llvm::to_vector_of<llvm::StringRef>(RuntimesBuildInfo::AsanSrcs),
          make_asan_copts(RuntimesBuildInfo::AsanCopts));
      asan_cxx_archive_.emplace(
          this, lib_path_ / "libclang_rt.asan_cxx.a", installation().root(),
          std::move(asan_cxx_srcs),
          make_asan_copts(RuntimesBuildInfo::AsanCxxCopts));
    } else {
      llvm::SmallVector<llvm::StringRef> asan_srcs;
      llvm::append_range(asan_srcs, RuntimesBuildInfo::AsanPreinitSrcs);
      llvm::append_range(asan_srcs, RuntimesBuildInfo::AsanSrcs);

      asan_archive_.emplace(
          this, lib_path_ / "libclang_rt.asan.a", installation().root(),
          std::move(asan_srcs), make_asan_copts(RuntimesBuildInfo::AsanCopts),
          /*generate_syms=*/true,
          installation().root() /
              std::string_view(RuntimesBuildInfo::AsanSymsExtra));
      asan_cxx_archive_.emplace(
          this, lib_path_ / "libclang_rt.asan_cxx.a", installation().root(),
          llvm::to_vector_of<llvm::StringRef>(RuntimesBuildInfo::AsanCxxSrcs),
          make_asan_copts(RuntimesBuildInfo::AsanCxxCopts),
          /*generate_syms=*/true);
      asan_static_archive_.emplace(
          this, lib_path_ / "libclang_rt.asan_static.a", installation().root(),
          llvm::to_vector_of<llvm::StringRef>(
              RuntimesBuildInfo::AsanStaticSrcs),
          make_asan_copts(RuntimesBuildInfo::AsanCopts));
    }
  }

  tasks_.async([this]() { Setup(); });
}

auto ClangResourceDirBuilder::CollectBuiltinsSrcFiles()
    -> llvm::SmallVector<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef> src_files;
  if (target_triple_.isAArch64()) {
    llvm::append_range(src_files, RuntimesBuildInfo::BuiltinsAarch64Srcs);
  } else if (target_triple_.isX86()) {
    if (target_triple_.isArch64Bit()) {
      llvm::append_range(src_files, RuntimesBuildInfo::BuiltinsX86_64Srcs);
    } else {
      // TODO: This should be turned into a nice user-facing diagnostic about an
      // unsupported target.
      CARBON_CHECK(
          target_triple_.isArch32Bit(),
          "The Carbon toolchain doesn't currently support 16-bit x86.");
      llvm::append_range(src_files, RuntimesBuildInfo::BuiltinsI386Srcs);
    }
  } else {
    // TODO: This should be turned into a nice user-facing diagnostic about an
    // unsupported target.
    CARBON_FATAL("Target architecture is not supported: {0}",
                 target_triple_.str());
  }

  // Only compile source files, not headers.
  llvm::erase_if(src_files,
                 [](llvm::StringRef file) { return file.ends_with(".h"); });
  return src_files;
}

auto ClangResourceDirBuilder::Setup() -> void {
  // Symlink the installation's `include` and `share` directories.
  std::filesystem::path install_resource_path =
      installation().clang_resource_path();
  for (const char* dir_name : {"include", "share"}) {
    if (auto result = runtimes_builder_->dir().Symlink(
            dir_name, install_resource_path / dir_name);
        !result.ok()) {
      result_ = std::move(result).error();
      return;
    }
  }

  // Create the target's `lib` directory.
  auto lib_dir_result = runtimes_builder_->dir().CreateDirectories(lib_path_);
  if (!lib_dir_result.ok()) {
    result_ = std::move(lib_dir_result).error();
    return;
  }
  lib_dir_ = *std::move(lib_dir_result);

  Latch::Handle latch_handle =
      step_counter_.Init([this] { tasks_.async([this] { Finish(); }); });

  // For Linux targets, the system libc (typically glibc) doesn't necessarily
  // provide the CRT begin/end files, and so we need to build them.
  if (target_triple_.isOSLinux()) {
    tasks_.async([this, latch_handle] {
      crt_begin_result_ = BuildCrtFile(RuntimesBuildInfo::CrtBegin);
    });
    tasks_.async([this, latch_handle] {
      crt_end_result_ = BuildCrtFile(RuntimesBuildInfo::CrtEnd);
    });
  }

  if (asan_archive_) {
    asan_archive_->Setup(latch_handle);
    asan_cxx_archive_->Setup(latch_handle);
    if (asan_static_archive_) {
      asan_static_archive_->Setup(latch_handle);
    }
  }
  archive_->Setup(std::move(latch_handle));
}

auto ClangResourceDirBuilder::Finish() -> void {
  CARBON_VLOG("Finished building resource dir...\n");
  for (std::optional<ArchiveBuilder>* archive :
       {&archive_, &asan_archive_, &asan_cxx_archive_, &asan_static_archive_}) {
    if (*archive && !(*archive)->result().ok()) {
      result_ = std::move((*archive)->result()).error();
      return;
    }
  }
  if (target_triple_.isOSLinux()) {
    for (ErrorOr<Success>* result : {&crt_begin_result_, &crt_end_result_}) {
      if (!result->ok()) {
        result_ = std::move(*result).error();
        return;
      }
    }
  }
  if (asan_ && target_triple_.isOSDarwin()) {
    if (auto result = BuildDarwinAsanDylib(); !result.ok()) {
      result_ = std::move(result).error();
      return;
    }
  }

  result_ = (*std::move(runtimes_builder_)).Commit();
}

auto ClangResourceDirBuilder::BuildCrtFile(llvm::StringRef src_file)
    -> ErrorOr<Success> {
  CARBON_CHECK(src_file == RuntimesBuildInfo::CrtBegin ||
               src_file == RuntimesBuildInfo::CrtEnd);
  std::filesystem::path out_path =
      runtimes_builder_->path() / lib_path_ /
      (src_file == RuntimesBuildInfo::CrtBegin ? "clang_rt.crtbegin.o"
                                               : "clang_rt.crtend.o");
  std::filesystem::path src_path =
      installation().root() / std::string_view(src_file);
  CARBON_VLOG("Building `{0}' from `{1}`...\n", out_path, src_path);

  llvm::SmallVector<llvm::StringRef> copts = {
      "-no-canonical-prefixes",
      "-w",
      "-fno-sanitize=all",
      target_flag_,
  };
  for (const std::string& target_arg : target_args_) {
    copts.push_back(target_arg);
  }
  llvm::append_range(copts, RuntimesBuildInfo::CrtCopts);
  copts.append({
      "-c",
      "-o",
      out_path.native(),
      src_path.native(),
  });

  CARBON_ASSIGN_OR_RETURN(bool success, clang_->RunWithNoRuntimes(copts));

  if (success) {
    return Success();
  }
  return Error(llvm::formatv("Failed to compile CRT file: {0}", src_file));
}

auto ClangResourceDirBuilder::BuildDarwinAsanDylib() -> ErrorOr<Success> {
  llvm::StringRef os_suffix = GetDarwinOsSuffix(target_triple_);
  std::string dylib_name =
      llvm::formatv("libclang_rt.asan_{0}_dynamic.dylib", os_suffix).str();
  std::filesystem::path lib_dir_path = runtimes_builder_->path() / lib_path_;
  std::filesystem::path dylib_path = lib_dir_path / dylib_name;
  std::filesystem::path asan_archive_path = lib_dir_path / "libclang_rt.asan.a";
  std::filesystem::path asan_cxx_archive_path =
      lib_dir_path / "libclang_rt.asan_cxx.a";
  std::filesystem::path builtins_archive_path =
      lib_dir_path / llvm::formatv("libclang_rt.{0}.a", os_suffix).str();
  CARBON_VLOG("Linking `{0}'...\n", dylib_path);

  std::string install_name_arg =
      llvm::formatv("-Wl,-install_name,@rpath/{0}", dylib_name).str();
  std::string force_load_asan_arg =
      llvm::formatv("-Wl,-force_load,{0}", asan_archive_path.native()).str();
  std::string force_load_asan_cxx_arg =
      llvm::formatv("-Wl,-force_load,{0}", asan_cxx_archive_path.native())
          .str();

  llvm::SmallVector<llvm::StringRef> link_args = {
      "-no-canonical-prefixes",
      "-w",
      "-shared",
      "-fno-sanitize=all",
      target_flag_,
      install_name_arg,
      force_load_asan_arg,
      force_load_asan_cxx_arg,
      builtins_archive_path.native(),
  };
  for (const std::string& target_arg : target_args_) {
    link_args.push_back(target_arg);
  }
  llvm::append_range(link_args, RuntimesBuildInfo::AsanDarwinLinkopts);
  link_args.append({
      "-o",
      dylib_path.native(),
  });

  CARBON_ASSIGN_OR_RETURN(bool success, clang_->RunWithNoRuntimes(link_args));
  if (!success) {
    return Error(
        llvm::formatv("Failed to link Darwin ASan dylib: {0}", dylib_name));
  }

  for (const char* archive_name :
       {"libclang_rt.asan.a", "libclang_rt.asan_cxx.a"}) {
    CARBON_RETURN_IF_ERROR(lib_dir_.Unlink(archive_name));
  }
  return Success();
}

}  // namespace Carbon
