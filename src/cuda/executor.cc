#include "src/cuda/executor.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <memory>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace pluto::cuda {

absl::Status CudaStatus(cudaError_t error, const char* operation) {
  if (error == cudaSuccess)
    return absl::OkStatus();
  return absl::InternalError(absl::StrCat(operation,
                                          " failed: ", cudaGetErrorName(error),
                                          ": ", cudaGetErrorString(error)));
}

namespace {

void ReportCleanupError(cudaError_t error, const char* operation) {
  if (error == cudaSuccess)
    return;
  std::fprintf(stderr, "%s failed: %s: %s\n", operation,
               cudaGetErrorName(error), cudaGetErrorString(error));
}

}  // namespace

absl::StatusOr<std::unique_ptr<Executor>> Executor::Create() {
  // Build directly into an owned object so every failure path releases all
  // streams/pools already created, without an exception-dependent scope guard.
  auto executor = std::unique_ptr<Executor>(new Executor());
  int device;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess)
    return CudaStatus(error, "cudaGetDevice");
  error = cudaStreamCreateWithFlags(&executor->stream_, cudaStreamNonBlocking);
  if (error != cudaSuccess)
    return CudaStatus(error, "cudaStreamCreateWithFlags");
  error = cudaStreamCreateWithFlags(&executor->trivial_stream_,
                                    cudaStreamNonBlocking);
  if (error != cudaSuccess)
    return CudaStatus(error, "cudaStreamCreateWithFlags(trivial stream)");

  cudaMemPoolProps properties{};
  properties.allocType = cudaMemAllocationTypePinned;
  properties.handleTypes = cudaMemHandleTypeNone;
  properties.location.type = cudaMemLocationTypeHost;
  error = cudaMemPoolCreate(&executor->host_memory_pool_, &properties);
  if (error != cudaSuccess)
    return CudaStatus(error, "cudaMemPoolCreate(host)");

  // Host pools are CPU-accessible by default; grant this executor's GPU access
  // explicitly so H2D and D2H transfers can use the same pinned allocation.
  cudaMemAccessDesc access{};
  access.location.type = cudaMemLocationTypeDevice;
  access.location.id = device;
  access.flags = cudaMemAccessFlagsProtReadWrite;
  error = cudaMemPoolSetAccess(executor->host_memory_pool_, &access, 1);
  if (error != cudaSuccess)
    return CudaStatus(error, "cudaMemPoolSetAccess(host)");

  // Never recycle storage by inserting a wait on compute: the CPU needs to
  // fill its next host buffer even while the previous transfer is still
  // queued. Completed-free opportunistic reuse remains enabled, otherwise
  // cross-stream frees may not be recycled until compute is synchronized.
  int disabled = 0;
  error = cudaMemPoolSetAttribute(executor->host_memory_pool_,
                                  cudaMemPoolReuseFollowEventDependencies,
                                  &disabled);
  if (error != cudaSuccess)
    return CudaStatus(error, "cudaMemPoolSetAttribute(follow events)");
  error = cudaMemPoolSetAttribute(executor->host_memory_pool_,
                                  cudaMemPoolReuseAllowInternalDependencies,
                                  &disabled);
  if (error != cudaSuccess)
    return CudaStatus(error, "cudaMemPoolSetAttribute(internal dependencies)");
  int enabled = 1;
  error = cudaMemPoolSetAttribute(executor->host_memory_pool_,
                                  cudaMemPoolReuseAllowOpportunistic, &enabled);
  if (error != cudaSuccess)
    return CudaStatus(error, "cudaMemPoolSetAttribute(opportunistic reuse)");

  // Cache a modest amount of pinned memory across synchronization points.
  // This is a cache-release threshold, not a cap on outstanding allocations.
  uint64_t release_threshold = 64 * 1024 * 1024;
  error = cudaMemPoolSetAttribute(executor->host_memory_pool_,
                                  cudaMemPoolAttrReleaseThreshold,
                                  &release_threshold);
  if (error != cudaSuccess)
    return CudaStatus(error, "cudaMemPoolSetAttribute(release threshold)");
  return executor;
}

Executor::~Executor() {
  // Complete queued frees before releasing pool and stream ownership. Null
  // handles occur only while unwinding a partially successful Create().
  if (stream_ != nullptr)
    ReportCleanupError(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  if (trivial_stream_ != nullptr) {
    ReportCleanupError(cudaStreamSynchronize(trivial_stream_),
                       "cudaStreamSynchronize(trivial stream)");
  }
  if (host_memory_pool_ != nullptr)
    ReportCleanupError(cudaMemPoolDestroy(host_memory_pool_),
                       "cudaMemPoolDestroy");
  if (trivial_stream_ != nullptr) {
    ReportCleanupError(cudaStreamDestroy(trivial_stream_),
                       "cudaStreamDestroy(trivial stream)");
  }
  if (stream_ != nullptr)
    ReportCleanupError(cudaStreamDestroy(stream_), "cudaStreamDestroy");
}

absl::Status Executor::Synchronize() const {
  return CudaStatus(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

}  // namespace pluto::cuda
