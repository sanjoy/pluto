#include "src/cuda/executor.h"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <memory>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace pluto::cuda {
namespace {

absl::Status CudaStatus(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return absl::OkStatus();
  return absl::InternalError(absl::StrCat(operation,
                                          " failed: ", cudaGetErrorName(error),
                                          ": ", cudaGetErrorString(error)));
}

void ReportCleanupError(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return;
  std::fprintf(stderr, "%s failed: %s: %s\n", operation,
               cudaGetErrorName(error), cudaGetErrorString(error));
}

}  // namespace

absl::StatusOr<std::unique_ptr<Executor>> Executor::Create() {
  cudaStream_t stream = nullptr;
  const cudaError_t error =
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaStreamCreateWithFlags");
  }
  return std::unique_ptr<Executor>(new Executor(stream));
}

Executor::~Executor() {
  // Executor must outlive all Buffers allocated through it. Synchronizing here
  // completes any queued frees before the native stream is destroyed.
  ReportCleanupError(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  ReportCleanupError(cudaStreamDestroy(stream_), "cudaStreamDestroy");
}

absl::Status Executor::Synchronize() const {
  return CudaStatus(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

}  // namespace pluto::cuda
