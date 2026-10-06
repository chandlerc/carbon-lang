// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_runtimes.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "testing/base/capture_std_streams.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/base/install_paths.h"
#include "toolchain/base/llvm_tools.h"
#include "toolchain/driver/clang_runner.h"
#include "toolchain/driver/llvm_runner.h"
#include "toolchain/driver/runtimes_cache.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace Carbon {

class ClangResourceDirBuilderTestPeer {
 public:
  static auto GetDarwinOsSuffix(llvm::Triple target_triple) -> llvm::StringRef {
    return ClangResourceDirBuilder::GetDarwinOsSuffix(target_triple);
  }
};

namespace {

using ::bazel::tools::cpp::runfiles::Runfiles;
using ::testing::Each;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsSupersetOf;

// NOLINTNEXTLINE(modernize-use-trailing-return-type): Macro based function.
MATCHER_P(TextSymbolNamed, name_matcher, "") {
  llvm::Expected<llvm::StringRef> name = arg.getName();
  if (auto error = name.takeError()) {
    *result_listener << "with an error instead of a name: " << error;
    return false;
  }
  if (!testing::ExplainMatchResult(name_matcher, *name, result_listener)) {
    return false;
  }
  // We have to dig out the section to determine if this was a text symbol.
  auto expected_section_it = arg.getSection();
  if (auto error = expected_section_it.takeError()) {
    *result_listener << "without a section: " << error;
    return false;
  }
  llvm::object::SectionRef section = **expected_section_it;
  if (!section.isText()) {
    *result_listener << "in the non-text section: " << *section.getName();
    return false;
  }
  return true;
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type): Macro based function.
MATCHER(IsBasename, "") {
  std::filesystem::path path = arg;
  return path == path.filename();
}

class ClangRuntimesTest : public ::testing::Test {
 public:
  ClangRuntimesTest() {
    std::string error;
    test_runfiles_.reset(Runfiles::Create(exe_path_, &error));
    CARBON_CHECK(test_runfiles_ != nullptr, "{0}", error);
  }

  // Helper to get the `llvm-nm` listing of defined symbols for an archive.
  //
  // TODO: It would be nice to use a library API and matchers instead of
  // `llvm-nm` and matching text on the output.
  auto NmListDefinedSymbols(const std::filesystem::path& archive)
      -> std::string {
    LLVMRunner llvm_runner(&install_paths_, &llvm::errs());
    std::string out;
    std::string err;
    bool result = Testing::CallWithCapturedOutput(out, err, [&] {
      return llvm_runner.Run(
          LLVMTool::Nm, {"--format=just-symbols", "--defined-only", "--quiet",
                         archive.native()});
    });
    CARBON_CHECK(result, "Unable to run `llvm-nm`:\n{0}", err);

    return out;
  }

  // Helper to get the `llvm-nm` listing of undefined symbols for an archive.
  auto NmListUndefinedSymbols(const std::filesystem::path& archive)
      -> std::string {
    LLVMRunner llvm_runner(&install_paths_, &llvm::errs());
    std::string out;
    std::string err;
    bool result = Testing::CallWithCapturedOutput(out, err, [&] {
      return llvm_runner.Run(
          LLVMTool::Nm, {"--format=just-symbols", "--undefined-only", "--quiet",
                         archive.native()});
    });
    CARBON_CHECK(result, "Unable to run `llvm-nm`:\n{0}", err);

    return out;
  }

  // Helper to expect a specific symbol in the `llvm-nm` list.
  //
  // This handles platform-specific formatting of symbols.
  auto ExpectSymbol(llvm::StringRef nm_list, llvm::StringRef symbol) -> void {
    std::string symbol_substr = llvm::formatv(
        target_triple_.isMacOSX() ? "\n_{0}\n" : "\n{0}\n", symbol);

    // Do the actual match with `HasSubstr` so it can explain failures.
    EXPECT_THAT(nm_list, HasSubstr(symbol_substr));
  }

  // Helper to get the names of archive members.
  auto ListArchiveMemberNames(const std::filesystem::path& archive_path)
      -> llvm::SmallVector<std::string> {
    llvm::SmallVector<std::string> result;

    auto archive_buffer_result =
        llvm::MemoryBuffer::getFile(archive_path.native());
    CARBON_CHECK(!archive_buffer_result.getError(), "Unable to open {0}: {1}",
                 archive_path, archive_buffer_result.getError().message());
    auto archive = llvm::cantFail(llvm::object::Archive::create(
        archive_buffer_result.get()->getMemBufferRef()));

    llvm::Error error = llvm::Error::success();
    for (const auto& child : archive->children(error)) {
      result.push_back(child.getName()->str());
    }
    CARBON_CHECK(!error, "Error reading members of archive {0}: {1}",
                 archive_path, error);

    return result;
  }

  auto TestResourceDir(std::filesystem::path resource_dir_path,
                       bool asan = false) -> void {
    // For Linux we can directly check the CRT begin/end object files.
    if (target_triple_.isOSLinux()) {
      std::filesystem::path crt_begin_path =
          resource_dir_path / "lib" / target_ / "clang_rt.crtbegin.o";
      ASSERT_TRUE(std::filesystem::is_regular_file(crt_begin_path));
      auto begin_result =
          llvm::object::ObjectFile::createObjectFile(crt_begin_path.native());
      llvm::object::ObjectFile& crtbegin = *begin_result->getBinary();
      EXPECT_TRUE(crtbegin.isELF());
      EXPECT_TRUE(crtbegin.isObject());
      EXPECT_THAT(crtbegin.getArch(), Eq(target_triple_.getArch()));

      llvm::SmallVector<llvm::object::SymbolRef> symbols(crtbegin.symbols());
      // The first symbol should come from the source file.
      EXPECT_THAT(*symbols.front().getName(), Eq("crtbegin.c"));

      // Check for representative symbols of `crtbegin.o` -- we always use
      // `.init_array` in our runtimes build so we have predictable functions.
      EXPECT_THAT(symbols, IsSupersetOf({TextSymbolNamed("__do_init"),
                                         TextSymbolNamed("__do_fini")}));

      std::filesystem::path crt_end_path =
          resource_dir_path / "lib" / target_ / "clang_rt.crtend.o";
      ASSERT_TRUE(std::filesystem::is_regular_file(crt_end_path));
      auto end_result =
          llvm::object::ObjectFile::createObjectFile(crt_end_path.native());
      llvm::object::ObjectFile& crtend = *end_result->getBinary();
      EXPECT_TRUE(crtend.isELF());
      EXPECT_TRUE(crtend.isObject());
      EXPECT_THAT(crtend.getArch(), Eq(target_triple_.getArch()));

      // Just check the source file symbol, not much of interest in the end.
      llvm::object::SymbolRef crtend_front_symbol = *crtend.symbol_begin();
      EXPECT_THAT(*crtend_front_symbol.getName(), Eq("crtend.c"));
    }

    // Across all targets, check that the builtins archive exists, and contains
    // a relevant symbol by running the `llvm-nm` tool over it. Using `nm`
    // rather than directly inspecting the objects is a bit awkward, but lets us
    // easily ignore the wrapping in an archive file.
    std::filesystem::path lib_path = "lib";
    std::string builtins_name = "libclang_rt.builtins.a";
    if (target_triple_.isOSDarwin()) {
      lib_path /= "darwin";
      builtins_name =
          llvm::formatv("libclang_rt.{0}.a",
                        ClangResourceDirBuilderTestPeer::GetDarwinOsSuffix(
                            target_triple_))
              .str();
    } else {
      lib_path /= target_;
    }
    std::filesystem::path builtins_path =
        resource_dir_path / lib_path / builtins_name;
    std::string builtins_symbols = NmListDefinedSymbols(builtins_path);

    // Check that we found a definition of `__mulodi4`, a builtin function
    // provided by Compiler-RT.
    ExpectSymbol(builtins_symbols, "__mulodi4");

    // Compiler-RT builtins should never be instrumented with ASan.
    EXPECT_THAT(NmListUndefinedSymbols(builtins_path),
                Not(HasSubstr("__asan_version_mismatch_check")));

    // Check that we don't include the `chkstk` builtins outside of Windows.
    if (!target_triple_.isOSWindows()) {
      EXPECT_THAT(builtins_symbols, Not(HasSubstr("chkstk")));
    }

    // Check that member names don't contain full paths, as that is the
    // canonical format produced by `ar`.
    auto member_names = ListArchiveMemberNames(builtins_path);
    EXPECT_THAT(member_names, Each(IsBasename()));

    EXPECT_TRUE(std::filesystem::is_regular_file(resource_dir_path /
                                                 "share/asan_ignorelist.txt"));

    if (target_triple_.isOSDarwin()) {
      std::filesystem::path asan_dylib_path =
          resource_dir_path / lib_path /
          llvm::formatv("libclang_rt.asan_{0}_dynamic.dylib",
                        ClangResourceDirBuilderTestPeer::GetDarwinOsSuffix(
                            target_triple_))
              .str();
      if (asan) {
        ASSERT_TRUE(std::filesystem::is_regular_file(asan_dylib_path));

        std::string asan_symbols = NmListDefinedSymbols(asan_dylib_path);
        ExpectSymbol(asan_symbols, "__asan_init");
        ExpectSymbol(asan_symbols, "__asan_report_load1");
        ExpectSymbol(asan_symbols, "__ubsan_handle_type_mismatch_v1");
        ExpectSymbol(asan_symbols, "__ubsan_handle_dynamic_type_cache_miss");
        ExpectSymbol(asan_symbols, "wrap__Znwm");
        ExpectSymbol(asan_symbols, "wrap__ZdlPv");

        // Sanitizer runtimes themselves must never be instrumented with ASan.
        EXPECT_THAT(NmListUndefinedSymbols(asan_dylib_path),
                    Not(HasSubstr("__asan_version_mismatch_check")));

        // Temporary static archives must not be left in the Darwin resource
        // directory.
        EXPECT_FALSE(std::filesystem::exists(resource_dir_path / lib_path /
                                             "libclang_rt.asan.a"));
        EXPECT_FALSE(std::filesystem::exists(resource_dir_path / lib_path /
                                             "libclang_rt.asan_cxx.a"));
      } else {
        EXPECT_FALSE(std::filesystem::exists(asan_dylib_path));
      }
    } else {
      std::filesystem::path asan_path =
          resource_dir_path / lib_path / "libclang_rt.asan.a";
      std::filesystem::path asan_syms_path =
          resource_dir_path / lib_path / "libclang_rt.asan.a.syms";
      std::filesystem::path asan_cxx_path =
          resource_dir_path / lib_path / "libclang_rt.asan_cxx.a";
      std::filesystem::path asan_cxx_syms_path =
          resource_dir_path / lib_path / "libclang_rt.asan_cxx.a.syms";
      std::filesystem::path asan_static_path =
          resource_dir_path / lib_path / "libclang_rt.asan_static.a";

      if (asan) {
        ASSERT_TRUE(std::filesystem::is_regular_file(asan_path));
        ASSERT_TRUE(std::filesystem::is_regular_file(asan_syms_path));
        ASSERT_TRUE(std::filesystem::is_regular_file(asan_cxx_path));
        ASSERT_TRUE(std::filesystem::is_regular_file(asan_cxx_syms_path));
        ASSERT_TRUE(std::filesystem::is_regular_file(asan_static_path));

        std::string asan_symbols = NmListDefinedSymbols(asan_path);
        ExpectSymbol(asan_symbols, "__asan_init");
        ExpectSymbol(asan_symbols, "__asan_report_load1");
        ExpectSymbol(asan_symbols, "__ubsan_handle_type_mismatch_v1");

        auto asan_syms_content =
            Filesystem::Cwd().ReadFileToString(asan_syms_path);
        ASSERT_TRUE(asan_syms_content.ok()) << asan_syms_content.error();
        EXPECT_THAT(*asan_syms_content, HasSubstr("\n  __asan_*;\n"));
        EXPECT_THAT(*asan_syms_content,
                    HasSubstr("\n  __interceptor_malloc;\n"));
        EXPECT_THAT(*asan_syms_content, HasSubstr("\n  malloc;\n"));
        EXPECT_THAT(*asan_syms_content,
                    HasSubstr("\n  __interceptor_memcpy;\n"));
        // Versioned functions like `memcpy` must not be directly exported in
        // `.syms`.
        EXPECT_THAT(*asan_syms_content, Not(HasSubstr("\n  memcpy;\n")));

        std::string asan_cxx_symbols = NmListDefinedSymbols(asan_cxx_path);
        ExpectSymbol(asan_cxx_symbols, "_Znwm");
        ExpectSymbol(asan_cxx_symbols, "_ZdlPv");

        auto asan_cxx_syms_content =
            Filesystem::Cwd().ReadFileToString(asan_cxx_syms_path);
        ASSERT_TRUE(asan_cxx_syms_content.ok())
            << asan_cxx_syms_content.error();
        EXPECT_THAT(*asan_cxx_syms_content, HasSubstr("\n  _Znwm;\n"));
        EXPECT_THAT(*asan_cxx_syms_content, HasSubstr("\n  _ZdlPv;\n"));

        std::string asan_static_symbols =
            NmListDefinedSymbols(asan_static_path);
        ExpectSymbol(asan_static_symbols, "__asan_report_load1_asm");

        // Sanitizer runtimes themselves must never be instrumented with ASan.
        for (const auto& archive_path :
             {asan_path, asan_cxx_path, asan_static_path}) {
          EXPECT_THAT(NmListUndefinedSymbols(archive_path),
                      Not(HasSubstr("__asan_version_mismatch_check")));
          EXPECT_THAT(ListArchiveMemberNames(archive_path), Each(IsBasename()));
        }
      } else {
        EXPECT_FALSE(std::filesystem::exists(asan_path));
        EXPECT_FALSE(std::filesystem::exists(asan_syms_path));
        EXPECT_FALSE(std::filesystem::exists(asan_cxx_path));
        EXPECT_FALSE(std::filesystem::exists(asan_cxx_syms_path));
        EXPECT_FALSE(std::filesystem::exists(asan_static_path));
      }
    }
  }

  auto TestLibunwind(std::filesystem::path libunwind_path, bool asan = false)
      -> void {
    std::string libunwind_symbols = NmListDefinedSymbols(libunwind_path);

    // Check a few of the main exported symbols here. The set here is somewhat
    // arbitrary, but chosen to be among the more stable names and have at least
    // one from most of the object files that should be linked into the archive.
    ExpectSymbol(libunwind_symbols, "_Unwind_Resume");
    ExpectSymbol(libunwind_symbols, "_Unwind_Backtrace");
    ExpectSymbol(libunwind_symbols, "__unw_getcontext");
    ExpectSymbol(libunwind_symbols, "__unw_get_proc_info");

    std::string undef_symbols = NmListUndefinedSymbols(libunwind_path);
    if (asan) {
      EXPECT_THAT(undef_symbols, HasSubstr("__asan_version_mismatch_check"));
    } else {
      EXPECT_THAT(undef_symbols,
                  Not(HasSubstr("__asan_version_mismatch_check")));
    }

    // Check that member names don't contain full paths, as that is the
    // canonical format produced by `ar`.
    auto member_names = ListArchiveMemberNames(libunwind_path);
    EXPECT_THAT(member_names, Each(IsBasename()));
  }

  auto TestLibcxx(std::filesystem::path libcxx_path, bool asan = false)
      -> void {
    std::string libcxx_symbols = NmListDefinedSymbols(libcxx_path);

    // First check a few fundamental symbols from libc++.a, including symbols
    // both within the ABI namespace and outside of it.
    ExpectSymbol(libcxx_symbols, "_ZNKSt12bad_any_cast4whatEv");
    ExpectSymbol(libcxx_symbols, "_ZSt17current_exceptionv");
    if (asan) {
      ExpectSymbol(libcxx_symbols, "_ZNSt6__asan8to_charsEPcS0_d");
      ExpectSymbol(libcxx_symbols,
                   "_ZNKSt6__asan10filesystem4path10__filenameEv");
      EXPECT_THAT(NmListUndefinedSymbols(libcxx_path),
                  HasSubstr("__asan_version_mismatch_check"));
    } else {
      ExpectSymbol(libcxx_symbols, "_ZNSt2_C8to_charsEPcS0_d");
      ExpectSymbol(libcxx_symbols, "_ZNKSt2_C10filesystem4path10__filenameEv");
      EXPECT_THAT(NmListUndefinedSymbols(libcxx_path),
                  Not(HasSubstr("__asan_version_mismatch_check")));
    }

    // Check that several of the libc++abi object files are also included in the
    // archive.
    ExpectSymbol(libcxx_symbols, "__cxa_bad_cast");
    ExpectSymbol(libcxx_symbols, "__cxa_new_handler");
    ExpectSymbol(libcxx_symbols, "__cxa_demangle");
    ExpectSymbol(libcxx_symbols, "__cxa_get_globals");
    ExpectSymbol(libcxx_symbols, "_ZSt9terminatev");

    // Check that member names don't contain full paths, as that is the
    // canonical format produced by `ar`.
    auto member_names = ListArchiveMemberNames(libcxx_path);
    EXPECT_THAT(member_names, Each(IsBasename()));
  }

#if __has_feature(address_sanitizer)
  static constexpr bool PrebuiltAsan = true;
#else
  static constexpr bool PrebuiltAsan = false;
#endif

  std::string exe_path_ = Testing::GetExePath().str();
  std::unique_ptr<Runfiles> test_runfiles_;
  InstallPaths install_paths_ = InstallPaths::MakeForBazelRunfiles(exe_path_);
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs_ =
      llvm::vfs::getRealFileSystem();
  // Note that for debugging, you can pass `llvm::errs()` as the vlog stream,
  // but this makes the output both very verbose and hard to use with multiple
  // threads.
  ClangRunner runner_{&install_paths_, vfs_};

  // Note that we can't test arbitrary targets here as we need to be able to
  // compile the builtin functions for the target. We use the default target as
  // the most likely to pass.
  std::string target_ = llvm::sys::getDefaultTargetTriple();
  llvm::Triple target_triple_{target_};

  Runtimes::Cache runtimes_cache_ =
      *Runtimes::Cache::MakeSystem(install_paths_);
  Runtimes::Cache::Features features = {.target = target_};
  Runtimes runtimes_ = *runtimes_cache_.Lookup(features);
  Runtimes::Cache::Features asan_features_ = {.target = target_, .asan = true};
  Runtimes asan_runtimes_ = *runtimes_cache_.Lookup(asan_features_);

  // Note that for debugging it may be useful to replace this with a
  // single-threaded thread pool. However the test will be _much_ slower.
  llvm::DefaultThreadPool threads_{llvm::optimal_concurrency()};
};

TEST_F(ClangRuntimesTest, ResourceDir) {
  ClangResourceDirBuilder resource_dir_builder(
      &runner_, &threads_, target_triple_, &runtimes_, features);
  auto build_result = std::move(resource_dir_builder).Wait();
  ASSERT_TRUE(build_result.ok()) << build_result.error();
  TestResourceDir(std::move(*build_result), /*asan=*/false);
}

TEST_F(ClangRuntimesTest, ResourceDirAsan) {
  ClangResourceDirBuilder resource_dir_builder(
      &runner_, &threads_, target_triple_, &asan_runtimes_, asan_features_);
  auto build_result = std::move(resource_dir_builder).Wait();
  ASSERT_TRUE(build_result.ok()) << build_result.error();
  TestResourceDir(*build_result, /*asan=*/true);

  if (!target_triple_.isOSDarwin()) {
    std::filesystem::path prebuilt_lib_path =
        std::filesystem::path(test_runfiles_->Rlocation(
            "carbon/toolchain/install/carbon_stage1_runtimes_build")) /
        "clang_resource_dir/lib" / target_;
    std::filesystem::path built_lib_path = *build_result / "lib" / target_;
    for (const char* syms_name :
         {"libclang_rt.asan.a.syms", "libclang_rt.asan_cxx.a.syms"}) {
      auto built_syms =
          Filesystem::Cwd().ReadFileToString(built_lib_path / syms_name);
      auto prebuilt_syms =
          Filesystem::Cwd().ReadFileToString(prebuilt_lib_path / syms_name);
      ASSERT_TRUE(built_syms.ok()) << built_syms.error();
      ASSERT_TRUE(prebuilt_syms.ok()) << prebuilt_syms.error();
      EXPECT_THAT(*built_syms, Eq(*prebuilt_syms));
    }
  }
}

TEST_F(ClangRuntimesTest, Libunwind) {
  LibunwindBuilder libunwind_builder(&runner_, &threads_, target_triple_,
                                     &runtimes_, features);
  auto build_result = std::move(libunwind_builder).Wait();
  ASSERT_TRUE(build_result.ok()) << build_result.error();
  std::filesystem::path runtimes_path = std::move(*build_result);
  TestLibunwind(runtimes_path / "lib/libunwind.a", /*asan=*/false);
}

TEST_F(ClangRuntimesTest, LibunwindAsan) {
  LibunwindBuilder libunwind_builder(&runner_, &threads_, target_triple_,
                                     &asan_runtimes_, asan_features_);
  auto build_result = std::move(libunwind_builder).Wait();
  ASSERT_TRUE(build_result.ok()) << build_result.error();
  std::filesystem::path runtimes_path = std::move(*build_result);
  TestLibunwind(runtimes_path / "lib/libunwind.a", /*asan=*/true);
}

// ASan causes Clang and LLVM to be _egregiously_ inefficient at compiling
// libc++, taking 5x - 10x longer than without ASan. Rough estimate is that it
// would take 5-10 minutes on GitHub's Linux runner.
//
// We test libc++ in the prebuilt runtimes below in a more cache friendly and
// sustainable way. Given that, we disable this test by default but include it
// for debugging purposes.
TEST_F(ClangRuntimesTest, DISABLED_Libcxx) {
  LibcxxBuilder libcxx_builder(&runner_, &threads_, target_triple_, &runtimes_,
                               features);
  auto build_result = std::move(libcxx_builder).Wait();
  ASSERT_TRUE(build_result.ok()) << build_result.error();
  std::filesystem::path runtimes_path = std::move(*build_result);
  TestLibcxx(runtimes_path / "lib/libc++.a", /*asan=*/false);
}

TEST_F(ClangRuntimesTest, DISABLED_LibcxxAsan) {
  LibcxxBuilder libcxx_builder(&runner_, &threads_, target_triple_,
                               &asan_runtimes_, asan_features_);
  auto build_result = std::move(libcxx_builder).Wait();
  ASSERT_TRUE(build_result.ok()) << build_result.error();
  std::filesystem::path runtimes_path = std::move(*build_result);
  TestLibcxx(runtimes_path / "lib/libc++.a", /*asan=*/true);
}

TEST_F(ClangRuntimesTest, PrebuiltResourceDir) {
  std::filesystem::path prebuilt_runtimes_path = test_runfiles_->Rlocation(
      "carbon/toolchain/install/carbon_stage1_runtimes_build");
  TestResourceDir(prebuilt_runtimes_path / "clang_resource_dir", /*asan=*/true);
}

TEST_F(ClangRuntimesTest, PrebuiltLibunwind) {
  std::filesystem::path prebuilt_runtimes_path = test_runfiles_->Rlocation(
      "carbon/toolchain/install/carbon_stage1_runtimes_build");
  TestLibunwind(prebuilt_runtimes_path / "libunwind/lib/libunwind.a",
                PrebuiltAsan);
}

TEST_F(ClangRuntimesTest, PrebuiltLibcxx) {
  std::filesystem::path prebuilt_runtimes_path = test_runfiles_->Rlocation(
      "carbon/toolchain/install/carbon_stage1_runtimes_build");
  TestLibcxx(prebuilt_runtimes_path / "libcxx/lib/libc++.a", PrebuiltAsan);
}

}  // namespace
}  // namespace Carbon
