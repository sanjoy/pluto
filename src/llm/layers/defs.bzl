"""Macros for the uniformly configured cuTile layer libraries and tests."""

load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_test")

def layer_library(name):
    cuda_library(
        name = name,
        srcs = ["layers/{}.cu".format(name)],
        hdrs = ["layers/{}.h".format(name)],
        copts = [
            "--enable-tile",
            "-std=c++20",
        ],
        host_copts = ["-std=c++20"],
        visibility = ["//visibility:public"],
        deps = [
            ":layer",
            ":layer_internal",
            "//src/gpu:buffer",
            "@abseil-cpp//absl/status",
            "@abseil-cpp//absl/status:statusor",
            "@abseil-cpp//absl/strings",
            "@abseil-cpp//absl/types:span",
            "@cuda//:cuda_runtime",
        ],
    )

def layer_test(name, extra_deps = []):
    cuda_test(
        name = name + "_test",
        size = "small",
        srcs = ["layers/{}_test.cu".format(name)],
        copts = ["-std=c++20"],
        host_copts = ["-std=c++20"],
        tags = ["requires-gpu"],
        deps = [
            ":" + name,
            ":layer",
            ":layer_test_util",
            "//src/gpu:buffer",
            "@cuda//:cuda_runtime",
            "@googletest//:gtest_main",
        ] + extra_deps,
    )
