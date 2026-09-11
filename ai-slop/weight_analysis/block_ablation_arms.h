#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ai-slop/weight_analysis/causal_probe.h"

namespace pluto::weight_analysis {

// Frozen single-model study. Each ablation starts from restored clean weights;
// cumulative names describe the set of disabled blocks, not sequential edits.
// Whole branches remove both their output projection and output bias. Learned
// input projections and LayerNorms remain untouched. Only isolated_b0 also
// zeros position embeddings, yielding the embedding-only B0 MLP readout.
inline std::vector<ProbeArm> BlockAblationArms() {
  const auto indices = [](int first, int end, bool attention, bool mlp) {
    std::vector<int> result;
    for (int block = first; block < end; ++block) {
      if (attention) {
        result.push_back(6 + 12 * block);
        result.push_back(7 + 12 * block);
      }
      if (mlp) {
        result.push_back(12 + 12 * block);
        result.push_back(13 + 12 * block);
      }
    }
    return result;
  };
  std::vector<ProbeArm> arms;
  arms.reserve(39);
  arms.push_back({"clean_before", {}, 1.0f});
  arms.push_back({"clean_repeat", {}, 1.0f});
  arms.push_back({"drop_b0_attention", indices(0, 1, true, false), 0.0f});
  arms.push_back({"drop_b0_mlp", indices(0, 1, false, true), 0.0f});
  for (int block = 1; block < 8; ++block) {
    const std::string name = "drop_b" + std::to_string(block);
    arms.push_back({name, indices(block, block + 1, true, true), 0.0f});
    arms.push_back(
        {name + "_attention", indices(block, block + 1, true, false), 0.0f});
    arms.push_back(
        {name + "_mlp", indices(block, block + 1, false, true), 0.0f});
  }
  for (int last = 0; last < 7; ++last) {
    arms.push_back({"keep_through_b" + std::to_string(last),
                    indices(last + 1, 8, true, true), 0.0f});
  }
  arms.push_back({"drop_later_attention", indices(1, 8, true, false), 0.0f});
  arms.push_back({"drop_later_mlp", indices(1, 8, false, true), 0.0f});

  auto without_positions = indices(1, 8, true, true);
  without_positions.insert(without_positions.begin(), {1, 6, 7});
  arms.push_back({"isolated_b0", std::move(without_positions), 0.0f});
  auto with_positions = indices(1, 8, true, true);
  with_positions.insert(with_positions.begin(), {6, 7});
  arms.push_back({"b0_mlp_with_positions", std::move(with_positions), 0.0f});

  arms.push_back({"drop_all_attention", indices(0, 8, true, false), 0.0f});
  arms.push_back({"drop_all_mlp", indices(0, 8, false, true), 0.0f});
  arms.push_back({"clean_after", {}, 1.0f});
  return arms;
}

}  // namespace pluto::weight_analysis
