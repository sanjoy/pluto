"""Macros for the uniformly configured cuTile layer libraries and tests."""

load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

def layer_library(name, extra_deps = []):
    cuda_library(
        name = name,
        srcs = ["layers/{}.cc".format(name)],
        hdrs = ["layers/{}.h".format(name)],
        copts = [
            "--enable-tile",
            "-std=c++20",
        ],
        host_copts = ["-std=c++20"],
        visibility = ["//visibility:public"],
        deps = [
            ":layer",
            ":layer_util",
            "//src/util:status_macros",
            "//src/cuda:buffer",
            "//src/cuda:executor",
            "//src/cuda:page_locked_host_array",
            "@abseil-cpp//absl/memory",
            "@abseil-cpp//absl/status",
            "@abseil-cpp//absl/status:statusor",
            "@abseil-cpp//absl/strings",
            "@abseil-cpp//absl/types:span",
            "@cuda//:cuda_runtime",
        ] + extra_deps,
    )

def layer_reference_library(name, extra_deps = []):
    """Defines a host-only scalar reference implementation for one layer."""
    cc_library(
        name = name + "_reference",
        srcs = ["layers/{}_reference.cc".format(name)],
        hdrs = ["layers/{}.h".format(name)],
        copts = ["-std=c++20"],
        visibility = ["//visibility:public"],
        deps = [
            ":layer",
            ":layer_reference_internal",
            "//src/util:status_macros",
            "//src/host:buffer",
            "@abseil-cpp//absl/memory",
            "@abseil-cpp//absl/status",
            "@abseil-cpp//absl/status:statusor",
            "@abseil-cpp//absl/types:span",
        ] + extra_deps,
    )

def layer_test(name, extra_deps = []):
    cuda_test(
        name = name + "_test",
        size = "small",
        srcs = ["layers/{}_test.cc".format(name)],
        copts = ["-std=c++20"],
        host_copts = ["-std=c++20"],
        tags = ["requires-gpu"],
        deps = [
            ":" + name,
            ":layer",
            ":layer_test_util",
            "//src/cuda:buffer",
            "//src/cuda:executor",
            "//src/cuda:page_locked_host_array",
            "@cuda//:cuda_runtime",
            "@googletest//:gtest_main",
        ] + extra_deps,
    )

def layer_reference_test(name, extra_deps = []):
    """Compares a CUDA layer with its scalar CPU executable specification."""
    cuda_test(
        name = name + "_reference_test",
        size = "small",
        srcs = ["layers/{}_reference_test.cc".format(name)],
        copts = ["-std=c++20"],
        host_copts = ["-std=c++20"],
        tags = ["requires-gpu"],
        deps = [
            ":" + name,
            ":" + name + "_reference",
            ":layer",
            ":layer_reference_test_util",
            "//src/util:status_macros",
            "//src/cuda:buffer",
            "//src/cuda:executor",
            "//src/cuda:page_locked_host_array",
            "//src/host:buffer",
            "@abseil-cpp//absl/status",
            "@abseil-cpp//absl/status:statusor",
            "@abseil-cpp//absl/types:span",
            "@cuda//:cuda_runtime",
            "@googletest//:gtest_main",
        ] + extra_deps,
    )
