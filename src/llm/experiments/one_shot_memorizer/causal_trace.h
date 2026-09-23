#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm::one_shot_memorizer {

struct CausalTraceSite {
  // Exact activation-hook name, for example "transformer_block_0".
  std::string layer_name;
  // If nonempty, this exact combinator name must be an active ancestor.
  std::string enclosing_scope = "";
  // -1 requires exactly one matching activation per forward. Nonnegative
  // values select a zero-based occurrence among scoped matches. In GPT-2,
  // ResidualLayer inside transformer_block_N has occurrence 0 after attention
  // and occurrence 1 after the MLP; these are the post-addition residuals.
  int occurrence = -1;
};

struct CausalTraceResult {
  CausalTraceSite site;
  int64_t targets = 0;
  // Cases whose corrupted input prefix differs in at least one token.
  int64_t changed_prefixes = 0;
  int64_t clean_correct = 0;
  int64_t corrupt_correct = 0;
  int64_t rescue_correct = 0;
  int64_t damage_correct = 0;
  // Denominator for rescued: targets - corrupt_correct, including nonfinite.
  int64_t baseline_wrong = 0;
  // Corrupt-wrong cases made correct by clean-query-row -> corrupt patching.
  int64_t rescued = 0;
  // Corrupt-correct cases made wrong by that patch; denominator
  // corrupt_correct.
  int64_t newly_broken_by_rescue = 0;
  // Clean-correct cases made wrong by corrupt-query-row -> clean patching;
  // denominator clean_correct. All four accuracy denominators are targets.
  int64_t damaged = 0;
  int64_t clean_nonfinite = 0;
  int64_t corrupt_nonfinite = 0;
  int64_t rescue_nonfinite = 0;
  int64_t damage_nonfinite = 0;
};

// Tests whether a selected residual activation carries information affecting
// independent teacher-forced next-token predictions. Every corpus target from
// prompt_tokens through terminal EOS is evaluated at its original position.
// Corruption replaces the prefix before its final retained_tokens tokens with
// tokens from the next corpus sentence, cycling that donor when necessary.
// Future positions are EOS; raw input sentences exclude EOS and padding.
//
// Clean and corrupt baselines capture each requested activation. Each site is
// then tested independently in both directions: clone the recipient's complete
// activation and overwrite ONLY the final query-position row of each valid
// sample with the donor baseline's row. Earlier positions, future positions,
// and unused batch samples retain the recipient's values. Patches are never
// accumulated across sites, and the original activation is never mutated.
//
// Sites must expose one FP32 or BF16 [batch, context, width] activation with
// matching physical storage. Missing/ambiguous sites, malformed signatures,
// foreign-executor buffers, and invalid corpus/options are errors. Inputs use
// the checkpoint's token IDs. Results preserve supplied site order.
//
// A rescue/damage measures mediation under this particular intervention, not
// a unique storage location or proof of a learned circuit. Donor corruption
// and patched states may be out of distribution. A late query-row patch can
// replace an already-computed answer; interpretation must respect site depth.
absl::StatusOr<std::vector<CausalTraceResult>> TracePrefixMediation(
    cuda::Executor& executor, const Layer& model,
    absl::Span<const std::vector<int>> sentences, int vocabulary_size,
    int eos_token, int prompt_tokens, int batch_size,
    absl::Span<const CausalTraceSite> sites, int retained_tokens = 9);

}  // namespace pluto::llm::one_shot_memorizer
