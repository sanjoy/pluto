#include "src/llm/experiments/completion_trace/trace.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm::completion_trace {
namespace {

using HostBytes = cuda::PageLockedHostArray<uint8_t>;

absl::StatusOr<size_t> Product(size_t a, size_t b) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    return absl::OutOfRangeError("completion trace tensor size overflows");
  return a * b;
}

struct ModelShape {
  size_t context;
  size_t vocabulary_stride;
  size_t output_bytes;
};

absl::StatusOr<ModelShape> ValidateModel(cuda::Executor& executor,
                                         const Layer& model,
                                         absl::Span<const int> prefix,
                                         int eos_token, int vocabulary_size) {
  if (prefix.empty() || vocabulary_size <= 0 || eos_token < 0 ||
      eos_token >= vocabulary_size)
    return absl::InvalidArgumentError(
        "trace requires a nonempty prefix and valid vocabulary/EOS");
  for (int token : prefix)
    if (token < 0 || token >= vocabulary_size)
      return absl::InvalidArgumentError("trace prefix token is out of range");
  const auto inputs = model.input_types();
  const auto outputs = model.output_types();
  if (inputs.size() != 1 || outputs.size() != 1)
    return absl::InvalidArgumentError("trace requires one model input/output");
  RETURN_IF_ERROR(inputs[0].Validate());
  RETURN_IF_ERROR(outputs[0].Validate());
  const auto in = inputs[0].dimensions();
  const auto out = outputs[0].dimensions();
  if (inputs[0].data_type() != DataType::INT32 || in.size() != 2 ||
      in[0] != ActivationType::kBatchDimension ||
      in[1] > std::numeric_limits<int>::max() ||
      prefix.size() > static_cast<size_t>(in[1]))
    return absl::InvalidArgumentError(
        "trace requires INT32 [batch,context] and a fitting prefix");
  if (outputs[0].data_type() != DataType::FP32 || out.size() != 3 ||
      out[0] != ActivationType::kBatchDimension || out[1] != in[1] ||
      out[2] < vocabulary_size || out[2] > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "trace requires FP32 [batch,context,vocabulary] output");
  for (const auto& weight : model.weights())
    if (&weight.executor() != &executor)
      return absl::InvalidArgumentError("trace model uses another executor");
  ASSIGN_OR_RETURN(auto count, Product(in[1], out[2]));
  ASSIGN_OR_RETURN(auto bytes, Product(count, sizeof(float)));
  return ModelShape{static_cast<size_t>(in[1]), static_cast<size_t>(out[2]),
                    bytes};
}

absl::StatusOr<size_t> ElementBytes(DataType type) {
  switch (type) {
    case DataType::BF16:
    case DataType::FP16:
      return 2;
    case DataType::FP32:
    case DataType::INT32:
      return 4;
    case DataType::FP8:
      // The backend itself rejects FP8 without an explicit scaling policy.
      // Silently choosing E4M3/E5M2 here would misrepresent captured values.
      return absl::UnimplementedError(
          "FP8 trace decoding requires a format and scaling policy");
  }
  return absl::InvalidArgumentError("unknown trace activation dtype");
}

// Binary16 decoding is independent of the CPU compiler's half support. FP16
// compute-policy layers currently expose FP32 storage; this also handles an
// explicitly FP16 physical tensor should a layer provide one in the future.
float DecodeHalf(uint16_t value) {
  const bool negative = (value & 0x8000u) != 0;
  const int exponent = (value >> 10) & 0x1fu;
  const int fraction = value & 0x3ffu;
  float result;
  if (exponent == 0)
    result = std::ldexp(static_cast<float>(fraction), -24);
  else if (exponent == 31)
    result = fraction == 0 ? std::numeric_limits<float>::infinity()
                           : std::numeric_limits<float>::quiet_NaN();
  else
    result = std::ldexp(static_cast<float>(1024 + fraction), exponent - 25);
  return negative ? -result : result;
}

void Decode(TensorSnapshot& tensor, size_t element_bytes) {
  tensor.values.resize(tensor.raw_bytes.size() / element_bytes);
  for (size_t i = 0; i < tensor.values.size(); ++i) {
    const uint8_t* source = tensor.raw_bytes.data() + i * element_bytes;
    if (tensor.data_type == DataType::BF16 ||
        tensor.data_type == DataType::FP16) {
      uint16_t bits;
      std::memcpy(&bits, source, sizeof(bits));
      tensor.values[i] = tensor.data_type == DataType::BF16
                             ? std::bit_cast<float>(uint32_t{bits} << 16)
                             : DecodeHalf(bits);
    } else if (tensor.data_type == DataType::INT32) {
      int32_t value;
      std::memcpy(&value, source, sizeof(value));
      // Original integer bits remain available even beyond FP32's exact range.
      tensor.values[i] = static_cast<float>(value);
    } else {
      std::memcpy(&tensor.values[i], source, sizeof(float));
    }
  }
}

struct PendingTensor {
  TensorSnapshot tensor;
  size_t element_bytes;
  HostBytes staging;
};

struct Occurrence {
  std::vector<std::string> scope;
  std::string name;
  size_t count;
};

class Capture {
 public:
  Capture(cuda::Executor& executor, const ModelShape& shape, size_t prefix)
      : executor_(executor), shape_(shape), prefix_(prefix) {
    hooks_.enter_combinator = [this](cuda::Executor& executor,
                                     absl::string_view name) {
      RETURN_IF_ERROR(CheckExecutor(executor));
      scope_.emplace_back(name);
      return absl::OkStatus();
    };
    hooks_.exit_combinator = [this](cuda::Executor& executor) {
      RETURN_IF_ERROR(CheckExecutor(executor));
      if (scope_.empty())
        return absl::FailedPreconditionError("trace scope stack underflow");
      scope_.pop_back();
      return absl::OkStatus();
    };
    hooks_.activation_hook = [this](cuda::Executor& executor,
                                    absl::string_view name,
                                    absl::Span<const ActivationType> types,
                                    absl::Span<Buffer> buffers) {
      return Activation(executor, name, types, buffers);
    };
    hooks_.attention_probabilities_hook =
        [this](cuda::Executor& executor, absl::string_view name,
               const ActivationType& type, const Buffer& buffer) {
          return Attention(executor, name, type, buffer);
        };
  }

  // Callback closures refer to this object's scope stack and pending copies.
  Capture(const Capture&) = delete;
  Capture& operator=(const Capture&) = delete;
  Capture(Capture&&) = delete;
  Capture& operator=(Capture&&) = delete;

  LayerHooks* hooks() { return &hooks_; }

  absl::Status Finish(ForwardTrace& result) {
    if (!scope_.empty())
      return absl::FailedPreconditionError("trace scopes were not closed");
    // Every copy was enqueued immediately after its producing operation. One
    // wait suffices: no device buffer is modified, and staging allocations
    // remain alive until their transfers finish, including on error paths.
    RETURN_IF_ERROR(executor_.Synchronize());
    for (auto& pending : activations_) {
      auto& tensor = pending.tensor;
      tensor.raw_bytes.assign(pending.staging.begin(), pending.staging.end());
      Decode(tensor, pending.element_bytes);
      result.activations.push_back(std::move(tensor));
    }
    for (auto& pending : attention_) {
      auto& tensor = pending.tensor;
      tensor.raw_bytes.assign(pending.staging.begin(), pending.staging.end());
      Decode(tensor, sizeof(float));
      const size_t heads = tensor.dimensions[1];
      for (size_t head = 0; head < heads; ++head)
        for (size_t query = 0; query < prefix_; ++query) {
          double sum = 0;
          for (size_t key = 0; key < prefix_; ++key) {
            const float p =
                tensor.values[(head * prefix_ + query) * prefix_ + key];
            // The observer reconstructs FP32 probabilities from FlashAttention
            // statistics. Keep its original values rather than renormalizing.
            if (!std::isfinite(p) || p < 0 || p > 1.0001f ||
                (key > query && p != 0))
              return absl::DataLossError(
                  "invalid causal attention probability");
            sum += p;
          }
          if (std::abs(sum - 1.0) > 1e-4)
            return absl::DataLossError(
                "attention probability row is not normalized");
        }
      result.attention.push_back(std::move(tensor));
    }
    return absl::OkStatus();
  }

 private:
  absl::Status CheckExecutor(const cuda::Executor& executor) const {
    return &executor == &executor_
               ? absl::OkStatus()
               : absl::InvalidArgumentError("trace hook executor mismatch");
  }

  size_t NextOccurrence(absl::string_view name,
                        std::vector<Occurrence>& occurrences) {
    for (auto& entry : occurrences)
      if (entry.scope == scope_ && entry.name == name)
        return entry.count++;
    occurrences.push_back({scope_, std::string(name), 1});
    return 0;
  }

  absl::Status Activation(cuda::Executor& executor, absl::string_view name,
                          absl::Span<const ActivationType> types,
                          absl::Span<Buffer> buffers) {
    RETURN_IF_ERROR(CheckExecutor(executor));
    if (types.size() != buffers.size())
      return absl::InvalidArgumentError("trace activation type/count mismatch");
    const size_t occurrence = NextOccurrence(name, activation_occurrences_);
    for (size_t output = 0; output < buffers.size(); ++output) {
      const auto& type = types[output];
      RETURN_IF_ERROR(type.Validate());
      const auto dimensions = type.dimensions();
      if (dimensions.size() < 2 ||
          dimensions[0] != ActivationType::kBatchDimension ||
          dimensions[1] != static_cast<int64_t>(shape_.context))
        return absl::InvalidArgumentError(
            "trace activation requires [batch,context,...] shape");
      ASSIGN_OR_RETURN(size_t element_bytes, ElementBytes(type.data_type()));
      size_t row_bytes = element_bytes;
      for (size_t axis = 2; axis < dimensions.size(); ++axis) {
        ASSIGN_OR_RETURN(row_bytes, Product(row_bytes, dimensions[axis]));
      }
      ASSIGN_OR_RETURN(size_t total_bytes, Product(shape_.context, row_bytes));
      if (buffers[output].size_bytes() != total_bytes ||
          &buffers[output].executor() != &executor_)
        return absl::InvalidArgumentError(
            "trace activation size/executor mismatch");
      ASSIGN_OR_RETURN(size_t captured_bytes, Product(prefix_, row_bytes));
      ASSIGN_OR_RETURN(auto staging,
                       HostBytes::Allocate(executor_, captured_bytes));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(staging.data(), buffers[output].data(),
                          captured_bytes, cudaMemcpyDeviceToHost,
                          executor_.stream()),
          "capture active prefix layer output"));
      TensorSnapshot snapshot{
          .scope = absl::StrJoin(scope_, "/"),
          .name = std::string(name),
          .occurrence = occurrence,
          .output_index = output,
          .data_type = type.data_type(),
          .dimensions = {dimensions.begin(), dimensions.end()}};
      snapshot.dimensions[0] = 1;
      snapshot.dimensions[1] = prefix_;
      activations_.push_back(
          {std::move(snapshot), element_bytes, std::move(staging)});
    }
    return absl::OkStatus();
  }

  absl::Status Attention(cuda::Executor& executor, absl::string_view name,
                         const ActivationType& type, const Buffer& buffer) {
    RETURN_IF_ERROR(CheckExecutor(executor));
    RETURN_IF_ERROR(type.Validate());
    const auto dimensions = type.dimensions();
    if (type.data_type() != DataType::FP32 || dimensions.size() != 4 ||
        dimensions[0] != 1 ||
        dimensions[2] != static_cast<int64_t>(shape_.context) ||
        dimensions[3] != static_cast<int64_t>(shape_.context) ||
        &buffer.executor() != &executor_)
      return absl::InvalidArgumentError(
          "trace attention requires FP32 [1,heads,context,context]");
    const size_t heads = dimensions[1];
    ASSIGN_OR_RETURN(size_t head_elements,
                     Product(shape_.context, shape_.context));
    ASSIGN_OR_RETURN(size_t elements, Product(heads, head_elements));
    ASSIGN_OR_RETURN(size_t bytes, Product(elements, sizeof(float)));
    if (buffer.size_bytes() != bytes)
      return absl::InvalidArgumentError("trace attention buffer size mismatch");
    ASSIGN_OR_RETURN(size_t prefix_elements, Product(prefix_, prefix_));
    ASSIGN_OR_RETURN(size_t captured_elements, Product(heads, prefix_elements));
    ASSIGN_OR_RETURN(size_t captured_bytes,
                     Product(captured_elements, sizeof(float)));
    ASSIGN_OR_RETURN(auto staging,
                     HostBytes::Allocate(executor_, captured_bytes));
    for (size_t head = 0; head < heads; ++head)
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpy2DAsync(
              staging.data() + head * prefix_elements * sizeof(float),
              prefix_ * sizeof(float),
              static_cast<const float*>(buffer.data()) + head * head_elements,
              shape_.context * sizeof(float), prefix_ * sizeof(float), prefix_,
              cudaMemcpyDeviceToHost, executor_.stream()),
          "capture active causal attention square"));
    TensorSnapshot snapshot{
        .scope = absl::StrJoin(scope_, "/"),
        .name = std::string(name),
        .occurrence = NextOccurrence(name, attention_occurrences_),
        .data_type = DataType::FP32,
        .dimensions = {1, static_cast<int64_t>(heads),
                       static_cast<int64_t>(prefix_),
                       static_cast<int64_t>(prefix_)}};
    attention_.push_back(
        {std::move(snapshot), sizeof(float), std::move(staging)});
    return absl::OkStatus();
  }

  cuda::Executor& executor_;
  const ModelShape shape_;
  const size_t prefix_;
  std::vector<std::string> scope_;
  std::vector<Occurrence> activation_occurrences_;
  std::vector<Occurrence> attention_occurrences_;
  std::vector<PendingTensor> activations_;
  std::vector<PendingTensor> attention_;
  LayerHooks hooks_;
};

absl::StatusOr<ForwardTrace> Forward(cuda::Executor& executor,
                                     const Layer& model,
                                     absl::Span<const int> prefix,
                                     int eos_token, int vocabulary_size,
                                     bool capture) {
  static_assert(sizeof(int) == sizeof(int32_t));  // Model input is INT32.
  ASSIGN_OR_RETURN(auto shape, ValidateModel(executor, model, prefix, eos_token,
                                             vocabulary_size));
  ASSIGN_OR_RETURN(auto staging, cuda::PageLockedHostArray<int>::Allocate(
                                     executor, shape.context));
  std::fill(staging.begin(), staging.end(), eos_token);
  std::copy(prefix.begin(), prefix.end(), staging.begin());
  ASSIGN_OR_RETURN(auto input, Buffer::Allocate(executor, staging.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(input.data(), staging.data(), staging.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload prefix with EOS in unused context positions"));
  Capture collector(executor, shape, prefix.size());
  ASSIGN_OR_RETURN(
      auto result,
      model.fwd(executor, {&input, 1}, capture ? collector.hooks() : nullptr));
  if (result.outputs.size() != 1 ||
      result.outputs[0].size_bytes() != shape.output_bytes ||
      &result.outputs[0].executor() != &executor)
    return absl::InvalidArgumentError(
        "trace model output differs from signature");
  ASSIGN_OR_RETURN(auto logits, cuda::PageLockedHostArray<float>::Allocate(
                                    executor, vocabulary_size));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(logits.data(),
                      static_cast<const float*>(result.outputs[0].data()) +
                          (prefix.size() - 1) * shape.vocabulary_stride,
                      logits.size_bytes(), cudaMemcpyDeviceToHost,
                      executor.stream()),
      "capture logical next-token logits"));
  ForwardTrace trace{.prefix = {prefix.begin(), prefix.end()}};
  RETURN_IF_ERROR(collector.Finish(trace));
  trace.next_logits.assign(logits.begin(), logits.end());
  for (float value : trace.next_logits)
    if (!std::isfinite(value))
      return absl::DataLossError("trace logical next-token logit is nonfinite");
  return trace;
}

}  // namespace

absl::StatusOr<ForwardTrace> TraceForward(cuda::Executor& executor,
                                          const Layer& model,
                                          absl::Span<const int> prefix,
                                          int eos_token, int vocabulary_size) {
  return Forward(executor, model, prefix, eos_token, vocabulary_size, true);
}

absl::StatusOr<std::vector<float>> PredictNextLogits(
    cuda::Executor& executor, const Layer& model, absl::Span<const int> prefix,
    int eos_token, int vocabulary_size) {
  ASSIGN_OR_RETURN(auto trace, Forward(executor, model, prefix, eos_token,
                                       vocabulary_size, false));
  return std::move(trace.next_logits);
}

}  // namespace pluto::llm::completion_trace
