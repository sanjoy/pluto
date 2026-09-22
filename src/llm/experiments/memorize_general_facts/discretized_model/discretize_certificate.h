#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_model.h"

namespace pluto::llm::discretized::generator {

enum class CertificateStatus { kInconclusive, kProven };

// One boundary whose every state pair has been proved distinguishable.
struct CertificateStage {
  int stage = 0;
  int64_t states = 0;
  int64_t pairs = 0;
  std::string argument;  // Explanation of the sufficient collision argument.
  auto operator<=>(const CertificateStage&) const = default;
};

// A sufficient proof of complete pairwise compaction, not a claim that no
// different global partition could use fewer states. Inconclusive results
// identify the first boundary and optional state pair the proof could not
// distinguish.
struct CertificateResult {
  CertificateStatus status = CertificateStatus::kInconclusive;
  int64_t states = 0;
  int64_t boundaries = 0;
  int64_t same_boundary_pairs = 0;
  int64_t proven_pairs = 0;
  int64_t attention_pairs_checked = 0;
  bool pairwise_compaction_complete = false;
  bool global_minimum_proven = false;
  std::vector<CertificateStage> stages;
  std::string reason;
  std::optional<int> unresolved_stage;
  std::optional<std::array<int, 2>> unresolved_pair;
  auto operator<=>(const CertificateResult&) const = default;
};

// Independent backward collision proof over immutable transition tables. It
// reads no compactor state, search history, or rejection cache. "Inconclusive"
// is not failure: this sufficient argument need not prove compaction complete
// for every model. A successful proof is pairwise, not global-partition
// minimality.
absl::StatusOr<CertificateResult> CertifyCompaction(const SymbolicModel& model);

}  // namespace pluto::llm::discretized::generator
