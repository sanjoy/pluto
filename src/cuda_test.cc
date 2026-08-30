#include <cuda_runtime.h>

#include <array>
#include <cstdio>

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
  std::array<int, kValueCount> values = {1, 2, 3, 4, 5, 6, 7, 8};
  int* device_values = nullptr;

  if (!CheckCuda(cudaMalloc(&device_values, sizeof(values)), "cudaMalloc")) {
    return 1;
  }

  if (!CheckCuda(cudaMemcpy(device_values, values.data(), sizeof(values),
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

  if (!CheckCuda(cudaMemcpy(values.data(), device_values, sizeof(values),
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
    if (values[index] != expected) {
      std::fprintf(stderr, "output[%d] = %d, expected %d\n", index,
                   values[index], expected);
      return 1;
    }
  }

  std::puts("CUDA kernel output verified");
  return 0;
}
