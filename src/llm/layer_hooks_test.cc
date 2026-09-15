#include "src/llm/layer_hooks.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/test_util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

ActivationType FloatRows() {
  return ActivationType(DataType::FP32, {-2, 1, 16});
}

absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              absl::Span<const float> values) {
  ASSIGN_OR_RETURN(auto staging,
                   cuda::PageLockedHostArray<float>::CopyFrom(executor, values));
  ASSIGN_OR_RETURN(auto result,
                   Buffer::Allocate(executor, values.size() * sizeof(float)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(result.data(), staging.data(), result.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload hooks test values"));
  return result;
}

absl::StatusOr<std::vector<float>> Download(cuda::Executor& executor,
                                            const Buffer& buffer) {
  ASSIGN_OR_RETURN(auto staging,
                   cuda::PageLockedHostArray<float>::Allocate(
                       executor, buffer.size_bytes() / sizeof(float)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(staging.data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download hooks test values"));
  RETURN_IF_ERROR(executor.Synchronize());
  return std::vector<float>(staging.begin(), staging.end());
}

// All spans are inspected synchronously. Only copied metadata and raw addresses
// are retained for assertions; the test owns buffers until those assertions.
struct Observation {
  cuda::Executor* executor;
  std::string layer;
  std::vector<ActivationType> types;
  std::vector<const void*> addresses;
};

class RecordingLayerHooks final : public LayerHooks {
 public:
  using Hook = std::function<absl::Status(cuda::Executor&, absl::string_view,
                                          absl::Span<const ActivationType>,
                                          absl::Span<Buffer>)>;

  absl::Status ActivationHook(cuda::Executor& executor, absl::string_view layer,
                              absl::Span<const ActivationType> types,
                              absl::Span<Buffer> buffers) override {
    Record(executor, layer, types, buffers, "activation:", activations);
    return activation_hook ? activation_hook(executor, layer, types, buffers)
                           : absl::OkStatus();
  }

  absl::Status GradientHook(cuda::Executor& executor, absl::string_view layer,
                            absl::Span<const ActivationType> types,
                            absl::Span<Buffer> buffers) override {
    Record(executor, layer, types, buffers, "gradient:", gradients);
    return gradient_hook ? gradient_hook(executor, layer, types, buffers)
                         : absl::OkStatus();
  }

  absl::Status EnterCombinator(cuda::Executor& executor,
                               absl::string_view layer) override {
    events.push_back("enter:" + std::string(layer));
    scope_executors.push_back(&executor);
    if (enter_hook) {
      const auto status = enter_hook(layer);
      if (!status.ok())
        return status;
    }
    scopes.emplace_back(layer);
    return absl::OkStatus();
  }

  absl::Status ExitCombinator(cuda::Executor& executor) override {
    EXPECT_FALSE(scopes.empty());
    if (scopes.empty())
      return absl::InternalError("unexpected unmatched hooks exit");
    const std::string layer = scopes.back();
    scopes.pop_back();
    events.push_back("exit:" + layer);
    scope_executors.push_back(&executor);
    return exit_hook ? exit_hook(layer) : absl::OkStatus();
  }

  std::vector<std::string> events;
  std::vector<std::string> scopes;
  std::vector<cuda::Executor*> scope_executors;
  std::vector<Observation> activations;
  std::vector<Observation> gradients;
  Hook activation_hook;
  Hook gradient_hook;
  std::function<absl::Status(absl::string_view)> enter_hook;
  std::function<absl::Status(absl::string_view)> exit_hook;

 private:
  void Record(cuda::Executor& executor, absl::string_view layer,
              absl::Span<const ActivationType> types,
              absl::Span<Buffer> buffers, const char* prefix,
              std::vector<Observation>& destination) {
    events.push_back(prefix + std::string(layer));
    Observation observation{
        &executor, std::string(layer), {types.begin(), types.end()}, {}};
    for (const auto& buffer : buffers)
      observation.addresses.push_back(buffer.data());
    destination.push_back(std::move(observation));
  }
};

// Identity data path with configurable signatures and failures. Saving output
// handles deliberately models attention/SAE aliases, without hiding extra GPU
// operations in the dispatch/ordering tests.
class SpyLayer final : public Layer {
 public:
  explicit SpyLayer(absl::string_view label,
                    std::vector<std::string>* events = nullptr)
      : SpyLayer(label, {FloatRows()}, {FloatRows()}, events) {}
  SpyLayer(absl::string_view label, std::vector<ActivationType> inputs,
           std::vector<ActivationType> outputs,
           std::vector<std::string>* events = nullptr)
      : label_(label),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        events_(events) {}

  absl::string_view name() const override { return label_; }
  absl::Span<const ActivationType> input_types() const override {
    return inputs_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return outputs_;
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

  absl::Status forward_status;
  absl::Status backward_status;
  std::optional<BufferVec> forward_outputs;
  std::optional<BufferVec> backward_outputs;
  bool save_outputs = true;
  bool copy_forward_output = false;
  mutable int forward_calls = 0;
  int backward_calls = 0;
  BufferVec received_gradients;

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    ++forward_calls;
    if (events_ != nullptr)
      events_->push_back("fwd:" + std::string(label_));
    RETURN_IF_ERROR(forward_status);
    BufferVec outputs = forward_outputs
                            ? *forward_outputs
                            : BufferVec(inputs.begin(), inputs.end());
    if (copy_forward_output) {
      if (outputs.size() != 1)
        return absl::InvalidArgumentError("copy test requires one output");
      ASSIGN_OR_RETURN(auto fresh,
                       Buffer::Allocate(executor, outputs[0].size_bytes()));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(fresh.data(), outputs[0].data(), fresh.size_bytes(),
                          cudaMemcpyDeviceToDevice, executor.stream()),
          "copy fake forward output"));
      outputs[0] = std::move(fresh);
    }
    BackwardState state;
    if (save_outputs)
      state.intermediates = outputs;
    return FwdResult{std::move(outputs), std::move(state)};
  }

  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState, LayerHooks*) override {
    ++backward_calls;
    if (events_ != nullptr)
      events_->push_back("bwd:" + std::string(label_));
    RETURN_IF_ERROR(backward_status);
    received_gradients.assign(gradients.begin(), gradients.end());
    if (backward_outputs)
      return *backward_outputs;
    return BufferVec(gradients.begin(), gradients.end());
  }

  absl::string_view label_;
  const std::vector<ActivationType> inputs_;
  const std::vector<ActivationType> outputs_;
  std::vector<std::string>* events_;
};

struct NestedModel {
  std::unique_ptr<ComposedLayer> model;
  SpyLayer* first;
  SpyLayer* second;
  SpyLayer* last;
};

absl::StatusOr<NestedModel> MakeNested(RecordingLayerHooks& hooks) {
  auto first = absl::make_unique<SpyLayer>("A", &hooks.events);
  auto second = absl::make_unique<SpyLayer>("B", &hooks.events);
  auto last = absl::make_unique<SpyLayer>("C", &hooks.events);
  SpyLayer* first_pointer = first.get();
  SpyLayer* second_pointer = second.get();
  SpyLayer* last_pointer = last.get();
  ComposedLayerBuilder inner;
  RETURN_IF_ERROR(inner.add(std::move(first)));
  RETURN_IF_ERROR(inner.add(std::move(second)));
  ASSIGN_OR_RETURN(auto inside, inner.create());
  ASSIGN_OR_RETURN(auto residual, ResidualLayer::Create(std::move(inside)));
  ComposedLayerBuilder outer;
  RETURN_IF_ERROR(outer.add(std::move(residual)));
  RETURN_IF_ERROR(outer.add(std::move(last)));
  ASSIGN_OR_RETURN(auto model, outer.create());
  return NestedModel{std::move(model), first_pointer, second_pointer,
                     last_pointer};
}

class LayerHooksTest : public LayersTest {};

TEST_F(LayerHooksTest, DefaultScopeCallbacksPreserveNumericalResults) {
  LayerHooks hooks;
  auto residual = ResidualLayer::Create(absl::make_unique<SpyLayer>("branch"));
  ASSERT_TRUE(residual.ok()) << residual.status();
  auto input = Upload(*executor_, std::vector<float>(16, 1.0f));
  ASSERT_TRUE(input.ok()) << input.status();
  auto forward = (*residual)->fwd(*executor_, {*input}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  auto values = Download(*executor_, forward->outputs[0]);
  ASSERT_TRUE(values.ok()) << values.status();
  EXPECT_EQ(*values, std::vector<float>(16, 2.0f));
  auto backward =
      (*residual)->bwd(*executor_, {*input}, std::move(forward->state), &hooks);
  ASSERT_TRUE(backward.ok()) << backward.status();
  auto gradients = Download(*executor_, (*backward)[0]);
  ASSERT_TRUE(gradients.ok()) << gradients.status();
  EXPECT_EQ(*gradients, std::vector<float>(16, 2.0f));
}

TEST_F(LayerHooksTest, DefaultAndNullHooksDoNotObserveCallsOnTheSameExecutor) {
  RecordingLayerHooks hooks;
  SpyLayer layer("identity");
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  auto observed = layer.fwd(*executor_, {*input}, &hooks);
  ASSERT_TRUE(observed.ok()) << observed.status();
  ASSERT_TRUE(
      layer.bwd(*executor_, {*input}, std::move(observed->state), &hooks).ok());
  ASSERT_EQ(hooks.activations.size(), 1);
  ASSERT_EQ(hooks.gradients.size(), 1);
  const auto events = hooks.events;

  auto default_forward = layer.fwd(*executor_, {*input});
  ASSERT_TRUE(default_forward.ok()) << default_forward.status();
  EXPECT_TRUE(
      layer.bwd(*executor_, {*input}, std::move(default_forward->state)).ok());
  auto null_forward = layer.fwd(*executor_, {*input}, nullptr);
  ASSERT_TRUE(null_forward.ok()) << null_forward.status();
  EXPECT_TRUE(
      layer.bwd(*executor_, {*input}, std::move(null_forward->state), nullptr)
          .ok());
  EXPECT_EQ(hooks.events, events);
  EXPECT_EQ(layer.forward_calls, 3);
  EXPECT_EQ(layer.backward_calls, 3);
}

TEST_F(LayerHooksTest, DispatchesActivationAfterAndGradientBeforeImpl) {
  RecordingLayerHooks hooks;
  const ActivationType bf16(DataType::BF16, {-2, 2, 16});
  SpyLayer layer("bf16", {bf16}, {bf16}, &hooks.events);
  auto input = Buffer::Allocate(*executor_, 32 * sizeof(uint16_t));
  auto gradient = Buffer::Allocate(*executor_, 32 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  auto forward = layer.fwd(*executor_, {*input}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  auto backward =
      layer.bwd(*executor_, {*gradient}, std::move(forward->state), &hooks);
  ASSERT_TRUE(backward.ok()) << backward.status();
  EXPECT_EQ(hooks.events,
            (std::vector<std::string>{"fwd:bf16", "activation:bf16",
                                      "gradient:bf16", "bwd:bf16"}));
  ASSERT_EQ(hooks.activations.size(), 1);
  EXPECT_EQ(hooks.activations[0].executor, executor_.get());
  EXPECT_EQ(hooks.activations[0].types, std::vector<ActivationType>{bf16});
  ASSERT_EQ(hooks.gradients.size(), 1);
  EXPECT_EQ(hooks.gradients[0].executor, executor_.get());
  EXPECT_EQ(
      hooks.gradients[0].types,
      std::vector<ActivationType>{ActivationType(DataType::FP32, {-2, 2, 16})});
  EXPECT_EQ(hooks.gradients[0].addresses[0], gradient->data());
}

TEST_F(LayerHooksTest, ReportsEmptyTerminalAndPrefixSaeGradients) {
  RecordingLayerHooks hooks;
  const ActivationType reconstruction(DataType::BF16, {-2, 1, 16});
  const ActivationType latents(DataType::BF16, {-2, 1, 32});
  const ActivationType decoder(DataType::FP32, {16, 32});
  SpyLayer sae("sae", {reconstruction}, {reconstruction, latents, decoder});
  auto x = Buffer::Allocate(*executor_, 16 * sizeof(uint16_t));
  auto z = Buffer::Allocate(*executor_, 32 * sizeof(uint16_t));
  auto d = Buffer::Allocate(*executor_, 16 * 32 * sizeof(float));
  auto dx = Buffer::Allocate(*executor_, 16 * sizeof(float));
  auto dz = Buffer::Allocate(*executor_, 32 * sizeof(float));
  ASSERT_TRUE(x.ok()) << x.status();
  ASSERT_TRUE(z.ok()) << z.status();
  ASSERT_TRUE(d.ok()) << d.status();
  ASSERT_TRUE(dx.ok()) << dx.status();
  ASSERT_TRUE(dz.ok()) << dz.status();
  sae.forward_outputs = BufferVec{*x, *z, *d};
  for (bool auxiliary : {false, true}) {
    auto forward = sae.fwd(*executor_, {*x}, &hooks);
    ASSERT_TRUE(forward.ok()) << forward.status();
    BufferVec gradients{*dx};
    if (auxiliary) {
      gradients.push_back(*dz);
      gradients.push_back(*d);
    }
    auto backward =
        sae.bwd(*executor_, gradients, std::move(forward->state), &hooks);
    ASSERT_TRUE(backward.ok()) << backward.status();
    std::vector<ActivationType> expected{
        ActivationType(DataType::FP32, {-2, 1, 16})};
    if (auxiliary) {
      expected.emplace_back(DataType::FP32,
                            absl::InlinedVector<int64_t, 4>{-2, 1, 32});
      expected.push_back(decoder);
    }
    EXPECT_EQ(hooks.gradients.back().types, expected);
  }

  SpyLayer loss("loss", {reconstruction, latents, decoder, reconstruction},
                {ActivationType(DataType::FP32, {-2, 1})});
  auto scalar = Buffer::Allocate(*executor_, sizeof(float));
  ASSERT_TRUE(scalar.ok()) << scalar.status();
  loss.forward_outputs = BufferVec{*scalar};
  loss.backward_outputs = BufferVec{*dx, *dz, *d};
  auto forward = loss.fwd(*executor_, {*x, *z, *d, *x}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  auto backward = loss.bwd(*executor_, {}, std::move(forward->state), &hooks);
  ASSERT_TRUE(backward.ok()) << backward.status();
  EXPECT_EQ(backward->size(), 3);
  EXPECT_TRUE(hooks.gradients.back().types.empty());
  EXPECT_TRUE(hooks.gradients.back().addresses.empty());
}

TEST_F(LayerHooksTest, EmbeddingStillReportsIncomingGradient) {
  RecordingLayerHooks hooks;
  SpyLayer embedding("embedding", {ActivationType(DataType::INT32, {-2, 1})},
                     {FloatRows()});
  auto token = Buffer::Allocate(*executor_, sizeof(int32_t));
  auto activation = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(token.ok()) << token.status();
  ASSERT_TRUE(activation.ok()) << activation.status();
  embedding.forward_outputs = BufferVec{*activation};
  embedding.backward_outputs = BufferVec{};
  auto forward = embedding.fwd(*executor_, {*token}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  auto backward = embedding.bwd(*executor_, {*activation},
                                std::move(forward->state), &hooks);
  ASSERT_TRUE(backward.ok()) << backward.status();
  EXPECT_TRUE(backward->empty());
  ASSERT_EQ(hooks.gradients.size(), 1);
  EXPECT_EQ(hooks.gradients[0].types, std::vector<ActivationType>{FloatRows()});
}

TEST_F(LayerHooksTest, InvalidBackwardStateDoesNotInvokeHooks) {
  RecordingLayerHooks hooks;
  SpyLayer first("first");
  SpyLayer other("other");
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  auto forward = first.fwd(*executor_, {*input}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  hooks.events.clear();
  auto backward =
      other.bwd(*executor_, {*input}, std::move(forward->state), &hooks);
  EXPECT_EQ(backward.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(other.backward_calls, 0);
  EXPECT_TRUE(hooks.events.empty());
}

TEST_F(LayerHooksTest, FailedImplDoesNotPublishActivation) {
  RecordingLayerHooks hooks;
  SpyLayer layer("broken", &hooks.events);
  layer.forward_status = absl::NotFoundError("forward body failed");
  const auto result = layer.fwd(*executor_, {}, &hooks);
  EXPECT_EQ(result.status(), layer.forward_status);
  EXPECT_EQ(hooks.events, std::vector<std::string>{"fwd:broken"});
  EXPECT_TRUE(hooks.activations.empty());
}

TEST_F(LayerHooksTest, CallbackErrorsPropagateAndStopBackwardImpl) {
  RecordingLayerHooks hooks;
  SpyLayer layer("identity");
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  hooks.activation_hook = [](auto&, auto, auto, auto) {
    return absl::PermissionDeniedError("activation callback failed");
  };
  auto failed = layer.fwd(*executor_, {*input}, &hooks);
  EXPECT_EQ(failed.status().code(), absl::StatusCode::kPermissionDenied);
  EXPECT_NE(failed.status().message().find("activation callback failed"),
            absl::string_view::npos);
  hooks.activation_hook = {};
  auto forward = layer.fwd(*executor_, {*input}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  hooks.gradient_hook = [](auto&, auto, auto, auto) {
    return absl::CancelledError("gradient callback failed");
  };
  auto backward =
      layer.bwd(*executor_, {*input}, std::move(forward->state), &hooks);
  EXPECT_EQ(backward.status().code(), absl::StatusCode::kCancelled);
  EXPECT_EQ(layer.backward_calls, 0);
}

TEST_F(LayerHooksTest, ReplacementsPreserveOriginalAndSavedHandles) {
  RecordingLayerHooks hooks;
  SpyLayer layer("identity");
  auto original = Upload(*executor_, std::vector<float>(16, 2.0f));
  auto replacement = Upload(*executor_, std::vector<float>(16, 5.0f));
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_TRUE(replacement.ok()) << replacement.status();
  hooks.activation_hook = [&](auto&, auto, auto, absl::Span<Buffer> buffers) {
    buffers[0] = *replacement;
    return absl::OkStatus();
  };
  auto forward = layer.fwd(*executor_, {*original}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  EXPECT_EQ(forward->outputs[0].data(), replacement->data());
  ASSERT_EQ(forward->state.intermediates.size(), 1);
  EXPECT_EQ(forward->state.intermediates[0].data(), original->data());
  auto original_values = Download(*executor_, *original);
  ASSERT_TRUE(original_values.ok()) << original_values.status();
  EXPECT_EQ(*original_values, std::vector<float>(16, 2.0f));

  hooks.gradient_hook = [&](auto&, auto, auto, absl::Span<Buffer> buffers) {
    buffers[0] = *replacement;
    return absl::OkStatus();
  };
  BufferVec incoming{*original};
  auto backward =
      layer.bwd(*executor_, incoming, std::move(forward->state), &hooks);
  ASSERT_TRUE(backward.ok()) << backward.status();
  EXPECT_EQ(incoming[0].data(), original->data());
  ASSERT_EQ(layer.received_gradients.size(), 1);
  EXPECT_EQ(layer.received_gradients[0].data(), replacement->data());
}

TEST_F(LayerHooksTest, RetainsOldOutputWhileCallbackQueuesCopy) {
  RecordingLayerHooks hooks;
  SpyLayer layer("fresh");
  layer.copy_forward_output = true;
  layer.save_outputs = false;
  auto input = Upload(*executor_, std::vector<float>(16, 7.0f));
  ASSERT_TRUE(input.ok()) << input.status();
  hooks.activation_hook = [](cuda::Executor& executor, auto, auto,
                             absl::Span<Buffer> buffers) -> absl::Status {
    const void* old_address = buffers[0].data();
    const size_t bytes = buffers[0].size_bytes();
    ASSIGN_OR_RETURN(auto replacement, Buffer::Allocate(executor, bytes));
    buffers[0] = std::move(replacement);
    // The framework must still retain the old handle here: queuing its async
    // free before this copy would make old_address a use-after-free.
    return cuda::CudaStatus(
        cudaMemcpyAsync(buffers[0].data(), old_address, bytes,
                        cudaMemcpyDeviceToDevice, executor.stream()),
        "copy replaced hooks output");
  };
  auto forward = layer.fwd(*executor_, {*input}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  EXPECT_NE(forward->outputs[0].data(), input->data());
  auto values = Download(*executor_, forward->outputs[0]);
  ASSERT_TRUE(values.ok()) << values.status();
  EXPECT_EQ(*values, std::vector<float>(16, 7.0f));
}

TEST_F(LayerHooksTest, RejectsWrongSizeAndForeignExecutorReplacements) {
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  auto wrong_size = Buffer::Allocate(*executor_, 17 * sizeof(float));
  auto foreign = Buffer::Allocate(**other, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(wrong_size.ok()) << wrong_size.status();
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  RecordingLayerHooks hooks;
  SpyLayer layer("identity");
  for (const Buffer& replacement : {*wrong_size, *foreign}) {
    hooks.activation_hook = [&](auto&, auto, auto, absl::Span<Buffer> buffers) {
      buffers[0] = replacement;
      return absl::OkStatus();
    };
    auto invalid_forward = layer.fwd(*executor_, {*input}, &hooks);
    EXPECT_EQ(invalid_forward.status().code(),
              absl::StatusCode::kInvalidArgument);
    hooks.activation_hook = {};
    auto forward = layer.fwd(*executor_, {*input}, &hooks);
    ASSERT_TRUE(forward.ok()) << forward.status();
    hooks.gradient_hook = [&](auto&, auto, auto, absl::Span<Buffer> buffers) {
      buffers[0] = replacement;
      return absl::OkStatus();
    };
    auto invalid_backward =
        layer.bwd(*executor_, {*input}, std::move(forward->state), &hooks);
    EXPECT_EQ(invalid_backward.status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(layer.backward_calls, 0);
    hooks.gradient_hook = {};
  }
}

TEST_F(LayerHooksTest, InvalidArityNeverPairsBuffersWithWrongMetadata) {
  RecordingLayerHooks hooks;
  SpyLayer layer("identity");
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  layer.forward_outputs = BufferVec{*input, *input};
  EXPECT_EQ(layer.fwd(*executor_, {*input}, &hooks).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(hooks.activations.empty());
  layer.forward_outputs.reset();
  auto forward = layer.fwd(*executor_, {*input}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  auto backward = layer.bwd(*executor_, {*input, *input},
                            std::move(forward->state), &hooks);
  EXPECT_EQ(backward.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(layer.backward_calls, 0);
  EXPECT_TRUE(hooks.gradients.empty());
}

TEST_F(LayerHooksTest, NestedScopesBracketChildrenInBothDirections) {
  RecordingLayerHooks hooks;
  auto nested = MakeNested(hooks);
  ASSERT_TRUE(nested.ok()) << nested.status();
  auto input = Upload(*executor_, std::vector<float>(16, 1.0f));
  ASSERT_TRUE(input.ok()) << input.status();
  auto forward = nested->model->fwd(*executor_, {*input}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  EXPECT_EQ(
      hooks.events,
      (std::vector<std::string>{
          "enter:ComposedLayer", "enter:ResidualLayer", "enter:ComposedLayer",
          "fwd:A", "activation:A", "fwd:B", "activation:B",
          "exit:ComposedLayer", "activation:ComposedLayer",
          "exit:ResidualLayer", "activation:ResidualLayer", "fwd:C",
          "activation:C", "exit:ComposedLayer", "activation:ComposedLayer"}));
  EXPECT_TRUE(hooks.scopes.empty());
  hooks.events.clear();
  auto backward = nested->model->bwd(*executor_, {*input},
                                     std::move(forward->state), &hooks);
  ASSERT_TRUE(backward.ok()) << backward.status();
  EXPECT_EQ(hooks.events,
            (std::vector<std::string>{
                "gradient:ComposedLayer", "enter:ComposedLayer", "gradient:C",
                "bwd:C", "gradient:ResidualLayer", "enter:ResidualLayer",
                "gradient:ComposedLayer", "enter:ComposedLayer", "gradient:B",
                "bwd:B", "gradient:A", "bwd:A", "exit:ComposedLayer",
                "exit:ResidualLayer", "exit:ComposedLayer"}));
  EXPECT_TRUE(hooks.scopes.empty());
  for (const auto* executor : hooks.scope_executors)
    EXPECT_EQ(executor, executor_.get());
}

TEST_F(LayerHooksTest, NestedForwardFailureUnwindsEveryEnteredScope) {
  RecordingLayerHooks hooks;
  auto nested = MakeNested(hooks);
  ASSERT_TRUE(nested.ok()) << nested.status();
  nested->second->forward_status = absl::NotFoundError("second forward failed");
  auto input = Upload(*executor_, std::vector<float>(16, 1.0f));
  ASSERT_TRUE(input.ok()) << input.status();
  auto forward = nested->model->fwd(*executor_, {*input}, &hooks);
  EXPECT_EQ(forward.status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(hooks.events, (std::vector<std::string>{
                              "enter:ComposedLayer", "enter:ResidualLayer",
                              "enter:ComposedLayer", "fwd:A", "activation:A",
                              "fwd:B", "exit:ComposedLayer",
                              "exit:ResidualLayer", "exit:ComposedLayer"}));
  EXPECT_TRUE(hooks.scopes.empty());
  EXPECT_EQ(nested->last->forward_calls, 0);
}

TEST_F(LayerHooksTest, NestedBackwardFailureUnwindsEveryEnteredScope) {
  RecordingLayerHooks hooks;
  auto nested = MakeNested(hooks);
  ASSERT_TRUE(nested.ok()) << nested.status();
  auto input = Upload(*executor_, std::vector<float>(16, 1.0f));
  ASSERT_TRUE(input.ok()) << input.status();
  auto forward = nested->model->fwd(*executor_, {*input}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  nested->second->backward_status =
      absl::NotFoundError("second backward failed");
  hooks.events.clear();
  auto backward = nested->model->bwd(*executor_, {*input},
                                     std::move(forward->state), &hooks);
  EXPECT_EQ(backward.status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(hooks.events,
            (std::vector<std::string>{
                "gradient:ComposedLayer", "enter:ComposedLayer", "gradient:C",
                "bwd:C", "gradient:ResidualLayer", "enter:ResidualLayer",
                "gradient:ComposedLayer", "enter:ComposedLayer", "gradient:B",
                "bwd:B", "exit:ComposedLayer", "exit:ResidualLayer",
                "exit:ComposedLayer"}));
  EXPECT_TRUE(hooks.scopes.empty());
  EXPECT_EQ(nested->first->backward_calls, 0);
}

TEST_F(LayerHooksTest, FailedEnterIsNotExitedButParentStillIs) {
  RecordingLayerHooks hooks;
  auto nested = MakeNested(hooks);
  ASSERT_TRUE(nested.ok()) << nested.status();
  hooks.enter_hook = [](absl::string_view name) {
    return name == "ResidualLayer" ? absl::UnavailableError("enter failed")
                                   : absl::OkStatus();
  };
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  auto forward = nested->model->fwd(*executor_, {*input}, &hooks);
  EXPECT_EQ(forward.status().code(), absl::StatusCode::kUnavailable);
  EXPECT_EQ(hooks.events, (std::vector<std::string>{"enter:ComposedLayer",
                                                    "enter:ResidualLayer",
                                                    "exit:ComposedLayer"}));
  EXPECT_TRUE(hooks.scopes.empty());
  EXPECT_EQ(nested->first->forward_calls, 0);
}

TEST_F(LayerHooksTest, ExitFailurePropagatesAndCombinesWithBodyError) {
  for (bool body_fails : {false, true}) {
    RecordingLayerHooks hooks;
    auto nested = MakeNested(hooks);
    ASSERT_TRUE(nested.ok()) << nested.status();
    if (body_fails)
      nested->second->forward_status = absl::NotFoundError("body marker");
    hooks.exit_hook = [](absl::string_view name) {
      return name == "ResidualLayer" ? absl::AbortedError("exit marker")
                                     : absl::OkStatus();
    };
    auto input = Upload(*executor_, std::vector<float>(16, 1.0f));
    ASSERT_TRUE(input.ok()) << input.status();
    auto forward = nested->model->fwd(*executor_, {*input}, &hooks);
    EXPECT_EQ(forward.status().code(), body_fails ? absl::StatusCode::kNotFound
                                                  : absl::StatusCode::kAborted);
    EXPECT_NE(forward.status().message().find("exit marker"),
              absl::string_view::npos);
    if (body_fails)
      EXPECT_NE(forward.status().message().find("body marker"),
                absl::string_view::npos);
    EXPECT_TRUE(hooks.scopes.empty());
    EXPECT_EQ(hooks.events.back(), "exit:ComposedLayer");
    EXPECT_EQ(nested->last->forward_calls, 0);
  }
}

TEST_F(LayerHooksTest, CallbackFailureStillExitsNestedScopes) {
  RecordingLayerHooks hooks;
  auto nested = MakeNested(hooks);
  ASSERT_TRUE(nested.ok()) << nested.status();
  auto input = Upload(*executor_, std::vector<float>(16, 1.0f));
  ASSERT_TRUE(input.ok()) << input.status();
  hooks.activation_hook = [](auto&, absl::string_view name, auto, auto) {
    return name == "B" ? absl::CancelledError("activation marker")
                       : absl::OkStatus();
  };
  auto failed = nested->model->fwd(*executor_, {*input}, &hooks);
  EXPECT_EQ(failed.status().code(), absl::StatusCode::kCancelled);
  EXPECT_TRUE(hooks.scopes.empty());
  EXPECT_EQ(hooks.events.back(), "exit:ComposedLayer");
  hooks.activation_hook = {};
  auto forward = nested->model->fwd(*executor_, {*input}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  hooks.gradient_hook = [](auto&, absl::string_view name, auto, auto) {
    return name == "B" ? absl::CancelledError("gradient marker")
                       : absl::OkStatus();
  };
  auto backward = nested->model->bwd(*executor_, {*input},
                                     std::move(forward->state), &hooks);
  EXPECT_EQ(backward.status().code(), absl::StatusCode::kCancelled);
  EXPECT_TRUE(hooks.scopes.empty());
  EXPECT_EQ(hooks.events.back(), "exit:ComposedLayer");
  EXPECT_EQ(nested->second->backward_calls, 0);
}

TEST_F(LayerHooksTest, ReplacingBranchGradientPreservesResidualSkipPath) {
  RecordingLayerHooks hooks;
  auto branch = absl::make_unique<SpyLayer>("branch");
  SpyLayer* branch_pointer = branch.get();
  auto residual = ResidualLayer::Create(std::move(branch));
  ASSERT_TRUE(residual.ok()) << residual.status();
  auto original = Upload(*executor_, std::vector<float>(16, 3.0f));
  auto replacement = Upload(*executor_, std::vector<float>(16, 7.0f));
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_TRUE(replacement.ok()) << replacement.status();
  auto forward = (*residual)->fwd(*executor_, {*original}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  hooks.gradient_hook = [&](auto&, absl::string_view name, auto,
                            absl::Span<Buffer> buffers) {
    if (name == "branch")
      buffers[0] = *replacement;
    return absl::OkStatus();
  };
  auto backward = (*residual)->bwd(*executor_, {*original},
                                   std::move(forward->state), &hooks);
  ASSERT_TRUE(backward.ok()) << backward.status();
  ASSERT_EQ(backward->size(), 1);
  auto values = Download(*executor_, (*backward)[0]);
  ASSERT_TRUE(values.ok()) << values.status();
  // Branch sees 7 while the independently retained skip path still sees 3.
  EXPECT_EQ(*values, std::vector<float>(16, 10.0f));
  auto unchanged = Download(*executor_, *original);
  ASSERT_TRUE(unchanged.ok()) << unchanged.status();
  EXPECT_EQ(*unchanged, std::vector<float>(16, 3.0f));
  EXPECT_EQ(branch_pointer->received_gradients[0].data(), replacement->data());
}

TEST_F(LayerHooksTest, GradientHookChangesDenseParameterGradients) {
  RecordingLayerHooks hooks;
  auto dense = FullyConnectedLayer::Create(*executor_, 16, DataType::FP16);
  ASSERT_TRUE(dense.ok()) << dense.status();
  ASSERT_TRUE((*dense)->InitializeIdentity().ok());
  auto input = Upload(*executor_, std::vector<float>(16 * 16, 1.0f));
  auto zero = Upload(*executor_, std::vector<float>(16 * 16, 0.0f));
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(zero.ok()) << zero.status();
  // Establish a nonzero baseline, then clear its parameter accumulators. The
  // second pass must become zero because of interception, not untouched
  // buffers.
  auto baseline_forward = (*dense)->fwd(*executor_, {*input});
  ASSERT_TRUE(baseline_forward.ok()) << baseline_forward.status();
  auto baseline_backward =
      (*dense)->bwd(*executor_, {*input}, std::move(baseline_forward->state));
  ASSERT_TRUE(baseline_backward.ok()) << baseline_backward.status();
  for (const Buffer& parameter_gradient : (*dense)->gradients()) {
    auto values = Download(*executor_, parameter_gradient);
    ASSERT_TRUE(values.ok()) << values.status();
    EXPECT_EQ(*values, std::vector<float>(values->size(), 16.0f));
    ASSERT_EQ(
        cudaMemsetAsync(parameter_gradient.data(), 0,
                        parameter_gradient.size_bytes(), executor_->stream()),
        cudaSuccess);
  }
  auto forward = (*dense)->fwd(*executor_, {*input}, &hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  hooks.gradient_hook = [&](auto&, absl::string_view name, auto,
                            absl::Span<Buffer> buffers) {
    EXPECT_EQ(name, "FullyConnectedLayer");
    buffers[0] = *zero;
    return absl::OkStatus();
  };
  auto backward =
      (*dense)->bwd(*executor_, {*input}, std::move(forward->state), &hooks);
  ASSERT_TRUE(backward.ok()) << backward.status();
  auto input_gradient = Download(*executor_, (*backward)[0]);
  ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
  EXPECT_EQ(*input_gradient, std::vector<float>(16 * 16, 0.0f));
  for (const Buffer& parameter_gradient : (*dense)->gradients()) {
    auto values = Download(*executor_, parameter_gradient);
    ASSERT_TRUE(values.ok()) << values.status();
    EXPECT_EQ(*values, std::vector<float>(values->size(), 0.0f));
  }
}

class ActivationOnlyHooks final : public LayerHooks {
 public:
  absl::Status ActivationHook(cuda::Executor&, absl::string_view,
                              absl::Span<const ActivationType>,
                              absl::Span<Buffer>) override {
    ++calls;
    return absl::OkStatus();
  }
  int calls = 0;
};

class GradientOnlyHooks final : public LayerHooks {
 public:
  absl::Status GradientHook(cuda::Executor&, absl::string_view,
                            absl::Span<const ActivationType>,
                            absl::Span<Buffer>) override {
    ++calls;
    return absl::OkStatus();
  }
  int calls = 0;
};

TEST_F(LayerHooksTest, CanOverrideOnlyOneHook) {
  SpyLayer layer("identity");
  ActivationOnlyHooks activation;
  GradientOnlyHooks gradient;
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();

  auto first = layer.fwd(*executor_, {*input}, &activation);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_TRUE(
      layer.bwd(*executor_, {*input}, std::move(first->state), &activation)
          .ok());
  EXPECT_EQ(activation.calls, 1);

  auto second = layer.fwd(*executor_, {*input}, &gradient);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_TRUE(
      layer.bwd(*executor_, {*input}, std::move(second->state), &gradient)
          .ok());
  EXPECT_EQ(gradient.calls, 1);
}

TEST_F(LayerHooksTest, DifferentHooksAreSelectedIndependentlyOnEachCall) {
  SpyLayer layer("identity");
  RecordingLayerHooks first;
  RecordingLayerHooks second;
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  auto forward = layer.fwd(*executor_, {*input}, &first);
  ASSERT_TRUE(forward.ok()) << forward.status();
  EXPECT_TRUE(
      layer.bwd(*executor_, {*input}, std::move(forward->state), &second).ok());
  EXPECT_EQ(first.activations.size(), 1);
  EXPECT_TRUE(first.gradients.empty());
  EXPECT_TRUE(second.activations.empty());
  EXPECT_EQ(second.gradients.size(), 1);

  auto another = layer.fwd(*executor_, {*input}, &second);
  ASSERT_TRUE(another.ok()) << another.status();
  EXPECT_TRUE(
      layer.bwd(*executor_, {*input}, std::move(another->state), &first).ok());
  EXPECT_EQ(first.activations.size(), 1);
  EXPECT_EQ(first.gradients.size(), 1);
  EXPECT_EQ(second.activations.size(), 1);
  EXPECT_EQ(second.gradients.size(), 1);
}

TEST_F(LayerHooksTest, ForwardStateDoesNotRetainHooksForBackward) {
  SpyLayer layer("identity");
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  BackwardState state;
  {
    RecordingLayerHooks temporary;
    auto forward = layer.fwd(*executor_, {*input}, &temporary);
    ASSERT_TRUE(forward.ok()) << forward.status();
    EXPECT_EQ(temporary.activations.size(), 1);
    state = std::move(forward->state);
  }
  // No observer is retained in the layer, executor, or saved backward state.
  // This remains valid after the forward hook object has been destroyed.
  auto backward = layer.bwd(*executor_, {*input}, std::move(state));
  ASSERT_TRUE(backward.ok()) << backward.status();
  EXPECT_EQ(layer.backward_calls, 1);
}

}  // namespace
}  // namespace pluto::llm
