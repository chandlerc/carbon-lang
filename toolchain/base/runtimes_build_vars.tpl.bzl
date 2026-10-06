# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A Starlark file exporting Carbon toolchain runtimes build info variables.

This file is a template that is expanded into a starlark file for the installed
Carbon toolchain that provides a trivial textual definition of the relevant
build info in variables.
"""

llvm_version_major = LLVM_VERSION_MAJOR

crtbegin_src = CRTBEGIN_SRC
crtend_src = CRTEND_SRC

crt_copts = [CRT_COPTS]

builtins_aarch64_srcs = [BUILTINS_AARCH64_SRCS]
builtins_x86_64_srcs = [BUILTINS_X86_64_SRCS]
builtins_i386_srcs = [BUILTINS_I386_SRCS]

builtins_aarch64_textual_srcs = [BUILTINS_AARCH64_TEXTUAL_SRCS]
builtins_x86_64_textual_srcs = [BUILTINS_X86_64_TEXTUAL_SRCS]
builtins_i386_textual_srcs = [BUILTINS_I386_TEXTUAL_SRCS]

builtins_copts = [BUILTINS_COPTS]

asan_hdrs = [ASAN_HDRS]
asan_textual_srcs = [ASAN_TEXTUAL_SRCS]
asan_preinit_srcs = [ASAN_PREINIT_SRCS]
asan_srcs = [ASAN_SRCS]
asan_cxx_srcs = [ASAN_CXX_SRCS]
ubsan_cxx_srcs = [UBSAN_CXX_SRCS]
asan_static_srcs = [ASAN_STATIC_SRCS]
asan_syms_extra = ASAN_SYMS_EXTRA
gen_dynamic_list = GEN_DYNAMIC_LIST

asan_copts = [ASAN_COPTS]
asan_cxx_copts = [ASAN_CXX_COPTS]
asan_darwin_copts = [ASAN_DARWIN_COPTS]
asan_darwin_linkopts = [ASAN_DARWIN_LINKOPTS]

libcxx_hdrs = [LIBCXX_HDRS]
libcxx_linux_srcs = [LIBCXX_LINUX_SRCS]
libcxx_macos_srcs = [LIBCXX_MACOS_SRCS]
libcxx_win32_srcs = [LIBCXX_WIN32_SRCS]

libc_internal_libcxx_hdrs = [LIBCXX_SHARED_HEADERS_HDRS]

libcxxabi_hdrs = [LIBCXXABI_HDRS]
libcxxabi_srcs = [LIBCXXABI_SRCS]
libcxxabi_textual_srcs = [LIBCXXABI_TEXTUAL_SRCS]

libcxx_copts = [LIBCXX_AND_ABI_COPTS]

libunwind_hdrs = [LIBUNWIND_HDRS]
libunwind_srcs = [LIBUNWIND_SRCS]

libunwind_copts = [LIBUNWIND_COPTS]

carbon_core_srcs = [PRELUDE_FILES]
