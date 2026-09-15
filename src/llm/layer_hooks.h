#pragma once

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"

namespace pluto::llm {

class ActivationType;

// Optional synchronous host callbacks around asynchronous GPU layer work.
// Pass a non-owning instance explicitly to Layer::fwd/bwd; combinators forward
// it to their children. The instance only needs to outlive that call and is
// never stored on the Executor, layer, or saved backward state. Passing nullptr
// disables instrumentation. Every callback defaults to a no-op, so subclasses
// need only override the hooks they use. Sharing one instance across threads
// requires synchronization in the hooks.
//
// Callbacks may read buffers or replace their handles with initialized buffers
// of the SAME byte size, physical dtype, logical shape, and Executor. Do not
// modify the original device bytes: they may alias saved backward state,
// another branch's inputs/gradients, or parameters (for example SAE's decoder
// output). Allocate a new buffer and queue the transformation on executor.
// Replacement size and Executor are checked; untyped Buffer cannot verify
// dtype or shape. Every entry must remain a valid, non-moved-from Buffer.
//
// Work already queued on executor is not necessarily complete. Queue GPU work
// on that executor, and synchronize explicitly if reading values on the CPU.
// Names, types, and spans are borrowed only for the callback; retain Buffer
// copies, not spans, if needed afterwards. Signatures use the symbolic batch
// dimension from ActivationType (defined in layer.h), not a resolved batch.
//
// Forward interventions do not rewrite saved state or automatically acquire a
// derivative. Supply a corresponding gradient intervention when needed;
// otherwise backward uses the original saved state (straight-through at the
// replacement). Replacing a loss output changes reporting, not its terminal
// backward rule. Changing an SAE auxiliary output does not rerun its decoder.
class LayerHooks {
 public:
  virtual ~LayerHooks() = default;

  // Called after a successful fwd_impl, before its output/state is published.
  // Types and buffers are the ordered forward outputs, not layer inputs.
  virtual absl::Status ActivationHook(
      cuda::Executor& executor, absl::string_view layer_name,
      absl::Span<const ActivationType> activation_types,
      absl::Span<cuda::Buffer> activations) {
    return absl::OkStatus();
  }

  // Called after saved-state identity validation and BEFORE bwd_impl, so an
  // intervention affects parameter gradients as well as input gradients.
  // These are gradients of this layer's forward outputs, all stored as FP32.
  // The types have the corresponding output dimensions (a prefix for SAE's
  // optional auxiliary gradients). Terminal losses have no upstream gradient:
  // they receive empty type/buffer spans, not an invented gradient of one.
  virtual absl::Status GradientHook(
      cuda::Executor& executor, absl::string_view layer_name,
      absl::Span<const ActivationType> activation_types,
      absl::Span<cuda::Buffer> gradients) {
    return absl::OkStatus();
  }

  // Bracket a combinator's implementation, in both forward and backward.
  // A successful Enter gets exactly one Exit, even if a child fails. A failed
  // Enter must leave hook scope state unchanged and does not get an Exit.
  // Exit must unwind its scope even when returning an error. Errors propagate
  // to the layer caller; if both a child and Exit fail, both are reported.
  //
  // Thus forward order is Enter, children, Exit, own activation callback;
  // backward order is own gradient callback, Enter, children, Exit. Names are
  // diagnostic class names, not unique IDs: use nesting/order to distinguish
  // repeated blocks. These callbacks do not wait for GPU work to complete.
  virtual absl::Status EnterCombinator(
      cuda::Executor& executor, absl::string_view combinator_layer_name) {
    return absl::OkStatus();
  }
  virtual absl::Status ExitCombinator(cuda::Executor& executor) {
    return absl::OkStatus();
  }
};

}  // namespace pluto::llm
