#pragma once

#include <cuda_runtime_api.h>

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace pluto::cuda {

// Owns the CUDA execution context used by one stream-ordered computation.
//
// Executor is deliberately small today: its only state is one explicitly
// created, non-default CUDA stream. Passing the Executor through APIs keeps
// stream selection explicit and leaves room for future execution-wide state
// without changing every layer interface again.
class Executor final {
 public:
  static absl::StatusOr<std::unique_ptr<Executor>> Create();

  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;
  ~Executor();

  // Waits for all work previously submitted to this executor.
  absl::Status Synchronize() const;

  // CUDA launch syntax and runtime calls require the native stream handle.
  // Code must obtain it from the Executor passed to the current operation.
  cudaStream_t stream() const { return stream_; }

 private:
  explicit Executor(cudaStream_t stream) : stream_(stream) {}

  cudaStream_t stream_;
};

}  // namespace pluto::cuda
