#include <cuda_runtime.h>

#include <cstdio>

#include "src/cuda/page_locked_host_array.h"

namespace {

constexpr int kValueCount = 8;

__global__ void DoubleValues(int* values) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < kValueCount) {
    values[index] *= 2;
  }
}

bool CheckCuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }

  std::fprintf(stderr, "%s failed: %s\n", operation,
               cudaGetErrorString(status));
  return false;
}

}  // namespace

int main() {
  auto values = pluto::cuda::PageLockedHostArray<int>::Allocate(kValueCount);
  if (!values.ok()) {
    std::fprintf(stderr, "host allocation failed: %s\n",
                 values.status().ToString().c_str());
    return 1;
  }
  for (int index = 0; index < kValueCount; ++index) {
    (*values)[index] = index + 1;
  }
  int* device_values = nullptr;

  if (!CheckCuda(cudaMalloc(&device_values, values->size_bytes()),
                 "cudaMalloc")) {
    return 1;
  }

  if (!CheckCuda(cudaMemcpy(device_values, values->data(), values->size_bytes(),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy to device")) {
    cudaFree(device_values);
    return 1;
  }

  DoubleValues<<<1, kValueCount>>>(device_values);
  if (!CheckCuda(cudaGetLastError(), "DoubleValues launch") ||
      !CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    cudaFree(device_values);
    return 1;
  }

  if (!CheckCuda(cudaMemcpy(values->data(), device_values, values->size_bytes(),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy to host")) {
    cudaFree(device_values);
    return 1;
  }

  if (!CheckCuda(cudaFree(device_values), "cudaFree")) {
    return 1;
  }

  for (int index = 0; index < kValueCount; ++index) {
    const int expected = 2 * (index + 1);
    if ((*values)[index] != expected) {
      std::fprintf(stderr, "output[%d] = %d, expected %d\n", index,
                   (*values)[index], expected);
      return 1;
    }
  }

  std::puts("CUDA kernel output verified");
  return 0;
}
