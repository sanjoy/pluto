#pragma once

#include <functional>

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
// disables instrumentation. Each std::function is empty by default and layers
// invoke only the callbacks that are set. Captured references must also outlive
// the call; copying this struct copies callbacks, not the objects they refer
// to. Sharing captured state across threads requires synchronization in the
// hooks. Do not change callback fields while a layer call is using this
// instance.
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
struct LayerHooks {
  // Called after a successful fwd_impl, before its output/state is published.
  // Types and buffers are the ordered forward outputs, not layer inputs.
  std::function<absl::Status(cuda::Executor& executor,
                             absl::string_view layer_name,
                             absl::Span<const ActivationType> activation_types,
                             absl::Span<cuda::Buffer> activations)>
      activation_hook;

  // Called after saved-state identity validation and BEFORE bwd_impl, so an
  // intervention affects parameter gradients as well as input gradients.
  // These are gradients of this layer's forward outputs, all stored as FP32.
  // The types have the corresponding output dimensions (a prefix for SAE's
  // optional auxiliary gradients). Terminal losses have no upstream gradient:
  // they receive empty type/buffer spans, not an invented gradient of one.
  std::function<absl::Status(cuda::Executor& executor,
                             absl::string_view layer_name,
                             absl::Span<const ActivationType> activation_types,
                             absl::Span<cuda::Buffer> gradients)>
      gradient_hook;

  // Bracket a combinator's implementation, in both forward and backward.
  // The two callbacks are independently optional. A failed enter_combinator
  // must leave scope state unchanged and skips both the body and the exit.
  // Otherwise, an installed exit_combinator runs exactly once after the body,
  // even if no enter callback is set or a child fails. Install both callbacks
  // when maintaining a scope stack; exit must unwind it even on an error.
  // Errors propagate to the layer caller; if both body and exit fail, both
  // are reported.
  //
  // Forward order is enter_combinator, children, exit_combinator, own
  // activation_hook; backward order is own gradient_hook, enter_combinator,
  // children, exit_combinator. Names are diagnostic labels, not unique IDs:
  // use nesting/order to distinguish repeated blocks. These callbacks do not
  // wait for GPU work to complete.
  std::function<absl::Status(cuda::Executor& executor,
                             absl::string_view combinator_layer_name)>
      enter_combinator;
  std::function<absl::Status(cuda::Executor& executor)> exit_combinator;
};

}  // namespace pluto::llm
