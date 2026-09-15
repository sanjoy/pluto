#include "src/llm/layer.h"

#include <utility>

#include "absl/strings/str_cat.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

absl::Status ValidateHookBuffers(cuda::Executor& executor,
                                 absl::Span<const Buffer> buffers) {
  for (size_t i = 0; i < buffers.size(); ++i)
    if (&buffers[i].executor() != &executor)
      return absl::InvalidArgumentError(
          absl::StrCat("hook buffer ", i, " belongs to another executor"));
  return absl::OkStatus();
}

absl::Status ValidateReplacements(cuda::Executor& executor,
                                  absl::Span<const Buffer> original,
                                  absl::Span<const Buffer> replacement) {
  RETURN_IF_ERROR(ValidateHookBuffers(executor, replacement));
  // A mutable Span permits replacing handles, not changing the output arity.
  for (size_t i = 0; i < original.size(); ++i)
    if (replacement[i].size_bytes() != original[i].size_bytes())
      return absl::InvalidArgumentError(
          absl::StrCat("hook changed buffer ", i, " byte size"));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<FwdResult> Layer::fwd(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const {
  ASSIGN_OR_RETURN(auto result, fwd_impl(executor, inputs, hooks));
  if (hooks != nullptr) {
    const auto types = output_types();
    if (result.outputs.size() != types.size())
      return absl::InvalidArgumentError(
          "activation hook requires one activation type per forward output");
    RETURN_IF_ERROR(ValidateHookBuffers(executor, result.outputs));
    // Retain the original handles through the callback: replacing a handle
    // must not enqueue its free before a callback queues a read of its bytes.
    const BufferVec original = result.outputs;
    RETURN_IF_ERROR(hooks->ActivationHook(executor, name(), types,
                                          absl::MakeSpan(result.outputs)));
    RETURN_IF_ERROR(ValidateReplacements(executor, original, result.outputs));
  }
  result.state.layer = this;
  return result;
}

absl::StatusOr<BufferVec> Layer::bwd(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state, LayerHooks* hooks) {
  if (state.layer != this)
    return absl::InvalidArgumentError(
        "bwd requires a state from this layer's successful fwd");
  if (hooks == nullptr)
    return bwd_impl(executor, output_gradients, std::move(state), hooks);

  const auto types = output_types();
  if (output_gradients.size() > types.size())
    return absl::InvalidArgumentError(
        "gradient hook received more gradients than forward output types");
  RETURN_IF_ERROR(ValidateHookBuffers(executor, output_gradients));
  absl::InlinedVector<ActivationType, 2> gradient_types;
  for (size_t i = 0; i < output_gradients.size(); ++i) {
    const auto dimensions = types[i].dimensions();
    gradient_types.emplace_back(
        DataType::FP32,
        absl::InlinedVector<int64_t, 4>(dimensions.begin(), dimensions.end()));
  }
  // Only copy refcounted handles. The caller's original gradients stay alive
  // and unchanged, including a residual's gradient shared by its skip branch.
  BufferVec gradients(output_gradients.begin(), output_gradients.end());
  RETURN_IF_ERROR(hooks->GradientHook(executor, name(), gradient_types,
                                      absl::MakeSpan(gradients)));
  RETURN_IF_ERROR(ValidateReplacements(executor, output_gradients, gradients));
  return bwd_impl(executor, gradients, std::move(state), hooks);
}

}  // namespace pluto::llm
