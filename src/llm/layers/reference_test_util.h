#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"
#include "src/llm/layers/reference_internal.h"
#include "src/llm/layers/test_util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {

struct BufferPair {
  Buffer device;
  HostBuffer host;
};

template <class Element>
absl::StatusOr<BufferPair> MakeRawBufferPair(cuda::Executor& executor,
                                             absl::Span<const Element> values) {
  static_assert(std::is_trivially_copyable_v<Element>);
  const size_t bytes = values.size() * sizeof(Element);
  ASSIGN_OR_RETURN(auto host, HostBuffer::Allocate(bytes));
  if (bytes != 0) std::memcpy(host.data(), values.data(), bytes);
  ASSIGN_OR_RETURN(auto device, Buffer::Allocate(executor, bytes));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device.data(), host.data(), bytes, cudaMemcpyHostToDevice,
                      executor.stream()),
      "copy test buffer to device"));
  return BufferPair{std::move(device), std::move(host)};
}

inline absl::StatusOr<BufferPair> MakeActivationBufferPair(
    cuda::Executor& executor, absl::Span<const float> values,
    DataType data_type) {
  RETURN_IF_ERROR(reference_internal::ValidateComputeType(data_type));
  ASSIGN_OR_RETURN(auto host, reference_internal::AllocateActivation(
                                  values.size(), data_type));
  for (size_t index = 0; index < values.size(); ++index) {
    reference_internal::StoreActivation(&host, index, data_type, values[index]);
  }
  ASSIGN_OR_RETURN(auto device, Buffer::Allocate(executor, host.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "copy activation to device"));
  return BufferPair{std::move(device), std::move(host)};
}

inline absl::Status SetFloatBufferPair(cuda::Executor& executor,
                                       const Buffer& device, HostBuffer* host,
                                       absl::Span<const float> values) {
  const size_t bytes = values.size() * sizeof(float);
  if (device.size_bytes() != bytes || host->size_bytes() != bytes) {
    return absl::InvalidArgumentError("parameter pair has the wrong size");
  }
  std::memcpy(host->data(), values.data(), bytes);
  return cuda::CudaStatus(
      cudaMemcpyAsync(device.data(), values.data(), bytes,
                      cudaMemcpyHostToDevice, executor.stream()),
      "copy paired parameter to device");
}

inline absl::Status ZeroBufferPair(cuda::Executor& executor,
                                   const Buffer& device, HostBuffer* host) {
  std::memset(host->data(), 0, host->size_bytes());
  return cuda::CudaStatus(
      cudaMemsetAsync(device.data(), 0, device.size_bytes(), executor.stream()),
      "clear paired buffer");
}

inline absl::StatusOr<std::vector<float>> ReadDeviceFloats(
    cuda::Executor& executor, const Buffer& buffer) {
  if (buffer.size_bytes() % sizeof(float) != 0) {
    return absl::InvalidArgumentError("device buffer is not FP32");
  }
  std::vector<float> values(buffer.size_bytes() / sizeof(float));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(values.data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "copy FP32 device buffer to host"));
  RETURN_IF_ERROR(executor.Synchronize());
  return values;
}

inline std::vector<float> ReadHostFloats(const HostBuffer& buffer) {
  const auto* values = static_cast<const float*>(buffer.data());
  return std::vector<float>(values,
                            values + buffer.size_bytes() / sizeof(float));
}

inline absl::StatusOr<std::vector<float>> ReadDeviceActivations(
    cuda::Executor& executor, const Buffer& buffer, DataType data_type) {
  ASSIGN_OR_RETURN(auto host, HostBuffer::Allocate(buffer.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "copy activation device buffer to host"));
  RETURN_IF_ERROR(executor.Synchronize());
  const size_t elements = buffer.size_bytes() /
                          reference_internal::ActivationElementBytes(data_type);
  std::vector<float> values(elements);
  for (size_t index = 0; index < elements; ++index) {
    values[index] = reference_internal::LoadActivation(host, index, data_type);
  }
  return values;
}

inline std::vector<float> ReadHostActivations(const HostBuffer& buffer,
                                              DataType data_type) {
  const size_t elements = buffer.size_bytes() /
                          reference_internal::ActivationElementBytes(data_type);
  std::vector<float> values(elements);
  for (size_t index = 0; index < elements; ++index) {
    values[index] =
        reference_internal::LoadActivation(buffer, index, data_type);
  }
  return values;
}

inline testing::AssertionResult VectorsNear(absl::Span<const float> actual,
                                            absl::Span<const float> expected,
                                            float absolute_tolerance,
                                            float relative_tolerance = 0.0f) {
  if (actual.size() != expected.size()) {
    return testing::AssertionFailure()
           << "size mismatch: " << actual.size() << " vs " << expected.size();
  }
  for (size_t index = 0; index < actual.size(); ++index) {
    if (actual[index] == expected[index]) continue;
    const float tolerance =
        absolute_tolerance +
        relative_tolerance *
            std::max(std::abs(actual[index]), std::abs(expected[index]));
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]) ||
        std::abs(actual[index] - expected[index]) > tolerance) {
      return testing::AssertionFailure()
             << "element " << index << ": " << actual[index] << " vs "
             << expected[index] << " (tolerance " << tolerance << ")";
    }
  }
  return testing::AssertionSuccess();
}

class LayerReferenceTest : public LayersTest {
 protected:
  testing::AssertionResult FloatBuffersNear(const Buffer& device,
                                            const HostBuffer& host,
                                            float absolute_tolerance,
                                            float relative_tolerance = 0.0f) {
    auto actual = ReadDeviceFloats(*executor_, device);
    if (!actual.ok()) {
      return testing::AssertionFailure() << actual.status();
    }
    return VectorsNear(*actual, ReadHostFloats(host), absolute_tolerance,
                       relative_tolerance);
  }

  testing::AssertionResult ActivationBuffersNear(
      const Buffer& device, const HostBuffer& host, DataType data_type,
      float absolute_tolerance, float relative_tolerance = 0.0f) {
    auto actual = ReadDeviceActivations(*executor_, device, data_type);
    if (!actual.ok()) {
      return testing::AssertionFailure() << actual.status();
    }
    return VectorsNear(*actual, ReadHostActivations(host, data_type),
                       absolute_tolerance, relative_tolerance);
  }
};

}  // namespace pluto::llm
