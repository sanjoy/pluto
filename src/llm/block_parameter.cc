#include "src/llm/block_parameter.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <limits>

#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {
template <class From, class To>
__tile_global__ void Convert(const From* input, To* output, int elements) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto x = ct::partition_view{ct::tensor_span{input, ct::extents{elements}},
                              ct::shape{256_ic}};
  auto y = ct::partition_view{ct::tensor_span{output, ct::extents{elements}},
                              ct::shape{256_ic}};
  y.store_masked(ct::element_cast<To>(x.load_masked(ct::bid().x)), ct::bid().x);
}
}  // namespace

absl::StatusOr<std::shared_ptr<BlockParameter>> BlockParameter::Create(
    cuda::Executor& executor, Buffer value, DataType storage) {
  if (storage != DataType::BF16 && storage != DataType::FP32)
    return absl::InvalidArgumentError("block parameter must use BF16 or FP32");
  const size_t size = storage == DataType::BF16 ? 2 : 4;
  if (&value.executor() != &executor || value.size_bytes() == 0 ||
      value.size_bytes() % size ||
      value.size_bytes() / size > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError("invalid block parameter buffer");
  const size_t elements = value.size_bytes() / size;
  return std::shared_ptr<BlockParameter>(
      new BlockParameter(std::move(value), storage, elements));
}

absl::Status BlockParameter::Activate() {
  if (active())
    return absl::OkStatus();
  auto& executor = value_.executor();
  ASSIGN_OR_RETURN(auto master, Buffer::Allocate(executor, elements_ * 4));
  ASSIGN_OR_RETURN(auto gradient, Buffer::Allocate(executor, elements_ * 4));
  if (storage_ == DataType::BF16) {
    Convert<<<(elements_ + 255) / 256, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(value_.data()),
        static_cast<float*>(master.data()), static_cast<int>(elements_));
    RETURN_IF_ERROR(
        cuda::CudaStatus(cudaGetLastError(), "activate block weight"));
  } else {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(master.data(), value_.data(), value_.size_bytes(),
                        cudaMemcpyDeviceToDevice, executor.stream()),
        "activate FP32 block weight"));
  }
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemsetAsync(gradient.data(), 0, gradient.size_bytes(),
                      executor.stream()),
      "zero block gradient"));
  master_ = std::move(master);
  gradient_ = std::move(gradient);
  return absl::OkStatus();
}

absl::Status BlockParameter::Publish() {
  if (!active())
    return absl::OkStatus();
  auto& executor = value_.executor();
  if (storage_ == DataType::BF16) {
    Convert<<<(elements_ + 255) / 256, 1, 0, executor.stream()>>>(
        static_cast<const float*>(master_->data()),
        static_cast<__nv_bfloat16*>(value_.data()),
        static_cast<int>(elements_));
    return cuda::CudaStatus(cudaGetLastError(), "publish block weight");
  }
  return cuda::CudaStatus(
      cudaMemcpyAsync(value_.data(), master_->data(), value_.size_bytes(),
                      cudaMemcpyDeviceToDevice, executor.stream()),
      "publish FP32 block weight");
}

absl::Status BlockParameter::Deactivate() {
  master_.reset();
  gradient_.reset();
  return absl::OkStatus();
}
}  // namespace pluto::llm
