#include "src/llm/layer_hooks.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
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

static_assert(std::is_aggregate_v<LayerHooks>);
static_assert(!std::is_polymorphic_v<LayerHooks>);

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

class RecordingLayerHooks final {
 public:
  using Hook = std::function<absl::Status(cuda::Executor&, absl::string_view,
                                          absl::Span<const ActivationType>,
                                          absl::Span<Buffer>)>;

  RecordingLayerHooks()
      : hooks{[this](auto& executor, auto layer, auto types, auto buffers) {
                return ActivationHook(executor, layer, types, buffers);
              },
              [this](auto& executor, auto layer, auto types, auto buffers) {
                return GradientHook(executor, layer, types, buffers);
              },
              [this](auto& executor, auto layer) {
                return EnterCombinator(executor, layer);
              },
              [this](auto& executor) { return ExitCombinator(executor); }} {}

  absl::Status ActivationHook(cuda::Executor& executor, absl::string_view layer,
                              absl::Span<const ActivationType> types,
                              absl::Span<Buffer> buffers) {
    Record(executor, layer, types, buffers, "activation:", activations);
    return activation_hook ? activation_hook(executor, layer, types, buffers)
                           : absl::OkStatus();
  }

  absl::Status GradientHook(cuda::Executor& executor, absl::string_view layer,
                            absl::Span<const ActivationType> types,
                            absl::Span<Buffer> buffers) {
    Record(executor, layer, types, buffers, "gradient:", gradients);
    return gradient_hook ? gradient_hook(executor, layer, types, buffers)
                         : absl::OkStatus();
  }

  absl::Status EnterCombinator(cuda::Executor& executor,
                               absl::string_view layer) {
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

  absl::Status ExitCombinator(cuda::Executor& executor) {
    EXPECT_FALSE(scopes.empty());
    if (scopes.empty())
      return absl::InternalError("unexpected unmatched hooks exit");
    const std::string layer = scopes.back();
    scopes.pop_back();
    events.push_back("exit:" + layer);
    scope_executors.push_back(&executor);
    return exit_hook ? exit_hook(layer) : absl::OkStatus();
  }

  LayerHooks hooks;
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
    ++output_type_calls;
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
  mutable int output_type_calls = 0;
  int backward_calls = 0;
  mutable LayerHooks* forward_hooks = nullptr;
  LayerHooks* backward_hooks = nullptr;
  const Buffer* received_gradient_handles = nullptr;
  BufferVec received_gradients;

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const override {
    ++forward_calls;
    forward_hooks = hooks;
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
                                     BackwardState,
                                     LayerHooks* hooks) override {
    ++backward_calls;
    backward_hooks = hooks;
    received_gradient_handles = gradients.data();
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
  ASSIGN_OR_RETURN(auto inside, inner.create("inner_pipeline"));
  ASSIGN_OR_RETURN(auto residual, ResidualLayer::Create(std::move(inside)));
  ComposedLayerBuilder outer;
  RETURN_IF_ERROR(outer.add(std::move(residual)));
  RETURN_IF_ERROR(outer.add(std::move(last)));
  ASSIGN_OR_RETURN(auto model, outer.create("outer_pipeline"));
  return NestedModel{std::move(model), first_pointer, second_pointer,
                     last_pointer};
}

class LayerHooksTest : public LayersTest {};

TEST(LayerHooks, CallbacksAreEmptyByDefault) {
  LayerHooks hooks;
  EXPECT_FALSE(hooks.activation_hook);
  EXPECT_FALSE(hooks.gradient_hook);
  EXPECT_FALSE(hooks.attention_probabilities_hook);
  EXPECT_FALSE(hooks.enter_combinator);
  EXPECT_FALSE(hooks.exit_combinator);
}

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
  auto observed = layer.fwd(*executor_, {*input}, &hooks.hooks);
  ASSERT_TRUE(observed.ok()) << observed.status();
  ASSERT_TRUE(
      layer.bwd(*executor_, {*input}, std::move(observed->state), &hooks.hooks)
          .ok());
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
  auto forward = layer.fwd(*executor_, {*input}, &hooks.hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  auto backward = layer.bwd(*executor_, {*gradient}, std::move(forward->state),
                            &hooks.hooks);
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
    auto forward = sae.fwd(*executor_, {*x}, &hooks.hooks);
    ASSERT_TRUE(forward.ok()) << forward.status();
    BufferVec gradients{*dx};
    if (auxiliary) {
      gradients.push_back(*dz);
      gradients.push_back(*d);
    }
    auto backward =
        sae.bwd(*executor_, gradients, std::move(forward->state), &hooks.hooks);
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
  auto forward = loss.fwd(*executor_, {*x, *z, *d, *x}, &hooks.hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  auto backward =
      loss.bwd(*executor_, {}, std::move(forward->state), &hooks.hooks);
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
  auto forward = embedding.fwd(*executor_, {*token}, &hooks.hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  auto backward = embedding.bwd(*executor_, {*activation},
                                std::move(forward->state), &hooks.hooks);
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
  auto forward = first.fwd(*executor_, {*input}, &hooks.hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  hooks.events.clear();
  auto backward =
      other.bwd(*executor_, {*input}, std::move(forward->state), &hooks.hooks);
  EXPECT_EQ(backward.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(other.backward_calls, 0);
  EXPECT_TRUE(hooks.events.empty());
}

TEST_F(LayerHooksTest, FailedImplDoesNotPublishActivation) {
  RecordingLayerHooks hooks;
  SpyLayer layer("broken", &hooks.events);
  layer.forward_status = absl::NotFoundError("forward body failed");
  const auto result = layer.fwd(*executor_, {}, &hooks.hooks);
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
  auto failed = layer.fwd(*executor_, {*input}, &hooks.hooks);
  EXPECT_EQ(failed.status().code(), absl::StatusCode::kPermissionDenied);
  EXPECT_NE(failed.status().message().find("activation callback failed"),
            absl::string_view::npos);
  hooks.activation_hook = {};
  auto forward = layer.fwd(*executor_, {*input}, &hooks.hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  hooks.gradient_hook = [](auto&, auto, auto, auto) {
    return absl::CancelledError("gradient callback failed");
  };
  auto backward =
      layer.bwd(*executor_, {*input}, std::move(forward->state), &hooks.hooks);
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
  auto forward = layer.fwd(*executor_, {*original}, &hooks.hooks);
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
      layer.bwd(*executor_, incoming, std::move(forward->state), &hooks.hooks);
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
  auto forward = layer.fwd(*executor_, {*input}, &hooks.hooks);
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
    auto invalid_forward = layer.fwd(*executor_, {*input}, &hooks.hooks);
    EXPECT_EQ(invalid_forward.status().code(),
              absl::StatusCode::kInvalidArgument);
    hooks.activation_hook = {};
    auto forward = layer.fwd(*executor_, {*input}, &hooks.hooks);
    ASSERT_TRUE(forward.ok()) << forward.status();
    hooks.gradient_hook = [&](auto&, auto, auto, absl::Span<Buffer> buffers) {
      buffers[0] = replacement;
      return absl::OkStatus();
    };
    auto invalid_backward = layer.bwd(*executor_, {*input},
                                      std::move(forward->state), &hooks.hooks);
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
  EXPECT_EQ(layer.fwd(*executor_, {*input}, &hooks.hooks).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(hooks.activations.empty());
  layer.forward_outputs.reset();
  auto forward = layer.fwd(*executor_, {*input}, &hooks.hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  auto backward = layer.bwd(*executor_, {*input, *input},
                            std::move(forward->state), &hooks.hooks);
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
  auto forward = nested->model->fwd(*executor_, {*input}, &hooks.hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  EXPECT_EQ(
      hooks.events,
      (std::vector<std::string>{
          "enter:outer_pipeline", "enter:ResidualLayer", "enter:inner_pipeline",
          "fwd:A", "activation:A", "fwd:B", "activation:B",
          "exit:inner_pipeline", "activation:inner_pipeline",
          "exit:ResidualLayer", "activation:ResidualLayer", "fwd:C",
          "activation:C", "exit:outer_pipeline", "activation:outer_pipeline"}));
  EXPECT_TRUE(hooks.scopes.empty());
  hooks.events.clear();
  auto backward = nested->model->bwd(*executor_, {*input},
                                     std::move(forward->state), &hooks.hooks);
  ASSERT_TRUE(backward.ok()) << backward.status();
  EXPECT_EQ(hooks.events,
            (std::vector<std::string>{
                "gradient:outer_pipeline", "enter:outer_pipeline", "gradient:C",
                "bwd:C", "gradient:ResidualLayer", "enter:ResidualLayer",
                "gradient:inner_pipeline", "enter:inner_pipeline", "gradient:B",
                "bwd:B", "gradient:A", "bwd:A", "exit:inner_pipeline",
                "exit:ResidualLayer", "exit:outer_pipeline"}));
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
  auto forward = nested->model->fwd(*executor_, {*input}, &hooks.hooks);
  EXPECT_EQ(forward.status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(hooks.events, (std::vector<std::string>{
                              "enter:outer_pipeline", "enter:ResidualLayer",
                              "enter:inner_pipeline", "fwd:A", "activation:A",
                              "fwd:B", "exit:inner_pipeline",
                              "exit:ResidualLayer", "exit:outer_pipeline"}));
  EXPECT_TRUE(hooks.scopes.empty());
  EXPECT_EQ(nested->last->forward_calls, 0);
}

TEST_F(LayerHooksTest, NestedBackwardFailureUnwindsEveryEnteredScope) {
  RecordingLayerHooks hooks;
  auto nested = MakeNested(hooks);
  ASSERT_TRUE(nested.ok()) << nested.status();
  auto input = Upload(*executor_, std::vector<float>(16, 1.0f));
  ASSERT_TRUE(input.ok()) << input.status();
  auto forward = nested->model->fwd(*executor_, {*input}, &hooks.hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  nested->second->backward_status =
      absl::NotFoundError("second backward failed");
  hooks.events.clear();
  auto backward = nested->model->bwd(*executor_, {*input},
                                     std::move(forward->state), &hooks.hooks);
  EXPECT_EQ(backward.status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(hooks.events,
            (std::vector<std::string>{
                "gradient:outer_pipeline", "enter:outer_pipeline", "gradient:C",
                "bwd:C", "gradient:ResidualLayer", "enter:ResidualLayer",
                "gradient:inner_pipeline", "enter:inner_pipeline", "gradient:B",
                "bwd:B", "exit:inner_pipeline", "exit:ResidualLayer",
                "exit:outer_pipeline"}));
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
  auto forward = nested->model->fwd(*executor_, {*input}, &hooks.hooks);
  EXPECT_EQ(forward.status().code(), absl::StatusCode::kUnavailable);
  EXPECT_EQ(hooks.events, (std::vector<std::string>{"enter:outer_pipeline",
                                                    "enter:ResidualLayer",
                                                    "exit:outer_pipeline"}));
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
    auto forward = nested->model->fwd(*executor_, {*input}, &hooks.hooks);
    EXPECT_EQ(forward.status().code(), body_fails ? absl::StatusCode::kNotFound
                                                  : absl::StatusCode::kAborted);
    EXPECT_NE(forward.status().message().find("exit marker"),
              absl::string_view::npos);
    if (body_fails)
      EXPECT_NE(forward.status().message().find("body marker"),
                absl::string_view::npos);
    EXPECT_TRUE(hooks.scopes.empty());
    EXPECT_EQ(hooks.events.back(), "exit:outer_pipeline");
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
  auto failed = nested->model->fwd(*executor_, {*input}, &hooks.hooks);
  EXPECT_EQ(failed.status().code(), absl::StatusCode::kCancelled);
  EXPECT_TRUE(hooks.scopes.empty());
  EXPECT_EQ(hooks.events.back(), "exit:outer_pipeline");
  hooks.activation_hook = {};
  auto forward = nested->model->fwd(*executor_, {*input}, &hooks.hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  hooks.gradient_hook = [](auto&, absl::string_view name, auto, auto) {
    return name == "B" ? absl::CancelledError("gradient marker")
                       : absl::OkStatus();
  };
  auto backward = nested->model->bwd(*executor_, {*input},
                                     std::move(forward->state), &hooks.hooks);
  EXPECT_EQ(backward.status().code(), absl::StatusCode::kCancelled);
  EXPECT_TRUE(hooks.scopes.empty());
  EXPECT_EQ(hooks.events.back(), "exit:outer_pipeline");
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
  auto forward = (*residual)->fwd(*executor_, {*original}, &hooks.hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  hooks.gradient_hook = [&](auto&, absl::string_view name, auto,
                            absl::Span<Buffer> buffers) {
    if (name == "branch")
      buffers[0] = *replacement;
    return absl::OkStatus();
  };
  auto backward = (*residual)->bwd(*executor_, {*original},
                                   std::move(forward->state), &hooks.hooks);
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
  auto forward = (*dense)->fwd(*executor_, {*input}, &hooks.hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  hooks.gradient_hook = [&](auto&, absl::string_view name, auto,
                            absl::Span<Buffer> buffers) {
    EXPECT_EQ(name, "FullyConnectedLayer");
    buffers[0] = *zero;
    return absl::OkStatus();
  };
  auto backward = (*dense)->bwd(*executor_, {*input}, std::move(forward->state),
                                &hooks.hooks);
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

TEST_F(LayerHooksTest, CanInstallOnlyOneCapturingCallback) {
  SpyLayer layer("identity");
  int activation_calls = 0;
  int gradient_calls = 0;
  LayerHooks activation{.activation_hook = [&](auto&, auto, auto, auto) {
    ++activation_calls;
    return absl::OkStatus();
  }};
  LayerHooks gradient{.gradient_hook = [&](auto&, auto, auto, auto) {
    ++gradient_calls;
    return absl::OkStatus();
  }};
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();

  auto first = layer.fwd(*executor_, {*input}, &activation);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_TRUE(
      layer.bwd(*executor_, {*input}, std::move(first->state), &activation)
          .ok());
  EXPECT_EQ(activation_calls, 1);

  auto second = layer.fwd(*executor_, {*input}, &gradient);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_TRUE(
      layer.bwd(*executor_, {*input}, std::move(second->state), &gradient)
          .ok());
  EXPECT_EQ(gradient_calls, 1);
}

TEST_F(LayerHooksTest, EveryCallbackPresenceCombinationPreservesOrdering) {
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  for (int mask = 0; mask < 16; ++mask) {
    SCOPED_TRACE(mask);
    std::vector<std::string> events;
    LayerHooks hooks;
    if (mask & 1) {
      hooks.activation_hook = [&](auto& executor, auto name, auto, auto) {
        EXPECT_EQ(&executor, executor_.get());
        events.push_back("activation:" + std::string(name));
        return absl::OkStatus();
      };
    }
    if (mask & 2) {
      hooks.gradient_hook = [&](auto& executor, auto name, auto, auto) {
        EXPECT_EQ(&executor, executor_.get());
        events.push_back("gradient:" + std::string(name));
        return absl::OkStatus();
      };
    }
    if (mask & 4) {
      hooks.enter_combinator = [&](auto& executor, auto name) {
        EXPECT_EQ(&executor, executor_.get());
        events.push_back("enter:" + std::string(name));
        return absl::OkStatus();
      };
    }
    if (mask & 8) {
      hooks.exit_combinator = [&](auto& executor) {
        EXPECT_EQ(&executor, executor_.get());
        events.push_back("exit");
        return absl::OkStatus();
      };
    }
    ComposedLayerBuilder builder;
    auto child = absl::make_unique<SpyLayer>("child", &events);
    auto* child_pointer = child.get();
    ASSERT_TRUE(builder.add(std::move(child)).ok());
    auto model = builder.create("parent");
    ASSERT_TRUE(model.ok()) << model.status();
    auto forward = (*model)->fwd(*executor_, {*input}, &hooks);
    ASSERT_TRUE(forward.ok()) << forward.status();
    auto backward =
        (*model)->bwd(*executor_, {*input}, std::move(forward->state), &hooks);
    ASSERT_TRUE(backward.ok()) << backward.status();
    EXPECT_EQ(child_pointer->forward_hooks, &hooks);
    EXPECT_EQ(child_pointer->backward_hooks, &hooks);

    std::vector<std::string> expected;
    if (mask & 4)
      expected.push_back("enter:parent");
    expected.push_back("fwd:child");
    if (mask & 1)
      expected.push_back("activation:child");
    if (mask & 8)
      expected.push_back("exit");
    if (mask & 1)
      expected.push_back("activation:parent");
    if (mask & 2)
      expected.push_back("gradient:parent");
    if (mask & 4)
      expected.push_back("enter:parent");
    if (mask & 2)
      expected.push_back("gradient:child");
    expected.push_back("bwd:child");
    if (mask & 8)
      expected.push_back("exit");
    EXPECT_EQ(events, expected);
  }
}

TEST_F(LayerHooksTest, MissingCallbacksSkipMetadataAndArityValidation) {
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  for (bool other_callback_present : {false, true}) {
    SCOPED_TRACE(other_callback_present);
    SpyLayer layer("identity");
    LayerHooks hooks;
    if (other_callback_present) {
      hooks.gradient_hook = [](auto&, auto, auto, auto) {
        ADD_FAILURE() << "gradient callback must not run during forward";
        return absl::OkStatus();
      };
    }
    layer.forward_outputs = BufferVec{*input, *input};
    auto forward = layer.fwd(*executor_, {*input}, &hooks);
    ASSERT_TRUE(forward.ok()) << forward.status();
    EXPECT_EQ(forward->outputs.size(), 2);
    EXPECT_EQ(layer.output_type_calls, 0);

    hooks.gradient_hook = {};
    if (other_callback_present) {
      hooks.activation_hook = [](auto&, auto, auto, auto) {
        ADD_FAILURE() << "activation callback must not run during backward";
        return absl::OkStatus();
      };
    }
    const BufferVec gradients{*input, *input};
    auto backward =
        layer.bwd(*executor_, gradients, std::move(forward->state), &hooks);
    ASSERT_TRUE(backward.ok()) << backward.status();
    EXPECT_EQ(backward->size(), 2);
    EXPECT_EQ(layer.output_type_calls, 0);
    EXPECT_EQ(layer.backward_calls, 1);
    EXPECT_EQ(layer.received_gradient_handles, gradients.data());
  }
}

TEST_F(LayerHooksTest, MissingGradientCallbackStillValidatesStateOwner) {
  SpyLayer first("first");
  SpyLayer other("other");
  LayerHooks empty;
  LayerHooks activation{.activation_hook = [](auto&, auto, auto, auto) {
    return absl::OkStatus();
  }};
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  for (LayerHooks* hooks :
       {static_cast<LayerHooks*>(nullptr), &empty, &activation}) {
    auto forward = first.fwd(*executor_, {*input});
    ASSERT_TRUE(forward.ok()) << forward.status();
    auto backward =
        other.bwd(*executor_, {*input}, std::move(forward->state), hooks);
    EXPECT_EQ(backward.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(other.backward_calls, 0);
  }
}

TEST_F(LayerHooksTest, EnterOnlyCallbackPropagatesEnterAndBodyFailures) {
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  for (bool backward : {false, true}) {
    for (bool enter_fails : {false, true}) {
      for (bool body_fails : {false, true}) {
        SCOPED_TRACE(backward);
        SCOPED_TRACE(enter_fails);
        SCOPED_TRACE(body_fails);
        std::vector<std::string> events;
        auto child = absl::make_unique<SpyLayer>("child", &events);
        auto* child_pointer = child.get();
        ComposedLayerBuilder builder;
        ASSERT_TRUE(builder.add(std::move(child)).ok());
        auto model = builder.create("parent");
        ASSERT_TRUE(model.ok()) << model.status();
        auto forward = (*model)->fwd(*executor_, {*input});
        ASSERT_TRUE(forward.ok()) << forward.status();
        if (body_fails) {
          child_pointer->forward_status = absl::NotFoundError("body marker");
          child_pointer->backward_status = child_pointer->forward_status;
        }
        events.clear();
        LayerHooks hooks{.enter_combinator = [&](auto&, auto name) {
          events.push_back("enter:" + std::string(name));
          return enter_fails ? absl::UnavailableError("enter marker")
                             : absl::OkStatus();
        }};
        const auto status =
            backward ? (*model)
                           ->bwd(*executor_, {*input},
                                 std::move(forward->state), &hooks)
                           .status()
                     : (*model)->fwd(*executor_, {*input}, &hooks).status();
        EXPECT_EQ(status.code(), enter_fails  ? absl::StatusCode::kUnavailable
                                 : body_fails ? absl::StatusCode::kNotFound
                                              : absl::StatusCode::kOk);
        std::vector<std::string> expected{"enter:parent"};
        if (!enter_fails)
          expected.push_back(backward ? "bwd:child" : "fwd:child");
        EXPECT_EQ(events, expected);
      }
    }
  }
}

TEST_F(LayerHooksTest, ExitOnlyCallbackRunsAfterSuccessfulOrFailedBody) {
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  for (bool backward : {false, true}) {
    for (bool exit_fails : {false, true}) {
      for (bool body_fails : {false, true}) {
        SCOPED_TRACE(backward);
        SCOPED_TRACE(exit_fails);
        SCOPED_TRACE(body_fails);
        std::vector<std::string> events;
        auto child = absl::make_unique<SpyLayer>("child", &events);
        auto* child_pointer = child.get();
        ComposedLayerBuilder builder;
        ASSERT_TRUE(builder.add(std::move(child)).ok());
        auto model = builder.create("parent");
        ASSERT_TRUE(model.ok()) << model.status();
        auto forward = (*model)->fwd(*executor_, {*input});
        ASSERT_TRUE(forward.ok()) << forward.status();
        if (body_fails) {
          child_pointer->forward_status = absl::NotFoundError("body marker");
          child_pointer->backward_status = child_pointer->forward_status;
        }
        events.clear();
        LayerHooks hooks{.exit_combinator = [&](auto&) {
          events.push_back("exit");
          return exit_fails ? absl::AbortedError("exit marker")
                            : absl::OkStatus();
        }};
        const auto status =
            backward ? (*model)
                           ->bwd(*executor_, {*input},
                                 std::move(forward->state), &hooks)
                           .status()
                     : (*model)->fwd(*executor_, {*input}, &hooks).status();
        EXPECT_EQ(status.code(), body_fails   ? absl::StatusCode::kNotFound
                                 : exit_fails ? absl::StatusCode::kAborted
                                              : absl::StatusCode::kOk);
        EXPECT_EQ(events, (std::vector<std::string>{
                              backward ? "bwd:child" : "fwd:child", "exit"}));
        if (body_fails)
          EXPECT_NE(status.message().find("body marker"),
                    absl::string_view::npos);
        if (exit_fails)
          EXPECT_NE(status.message().find("exit marker"),
                    absl::string_view::npos);
        if (body_fails && exit_fails)
          EXPECT_NE(status.message().find("exit_combinator failed"),
                    absl::string_view::npos);
      }
    }
  }
}

TEST_F(LayerHooksTest, DifferentHooksAreSelectedIndependentlyOnEachCall) {
  SpyLayer layer("identity");
  RecordingLayerHooks first;
  RecordingLayerHooks second;
  auto input = Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  auto forward = layer.fwd(*executor_, {*input}, &first.hooks);
  ASSERT_TRUE(forward.ok()) << forward.status();
  EXPECT_TRUE(
      layer.bwd(*executor_, {*input}, std::move(forward->state), &second.hooks)
          .ok());
  EXPECT_EQ(first.activations.size(), 1);
  EXPECT_TRUE(first.gradients.empty());
  EXPECT_TRUE(second.activations.empty());
  EXPECT_EQ(second.gradients.size(), 1);

  auto another = layer.fwd(*executor_, {*input}, &second.hooks);
  ASSERT_TRUE(another.ok()) << another.status();
  EXPECT_TRUE(
      layer.bwd(*executor_, {*input}, std::move(another->state), &first.hooks)
          .ok());
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
    auto forward = layer.fwd(*executor_, {*input}, &temporary.hooks);
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
