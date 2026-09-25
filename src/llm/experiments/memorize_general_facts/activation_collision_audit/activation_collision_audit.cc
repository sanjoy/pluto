// A read-only diagnostic: exact activation equality, not a distance threshold.
// No weights are updated and no compacted symbolic states participate.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/memorize_general_facts/activation_collision_audit/audit.h"
#include "src/llm/extract_top1_ids.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Checkpoint directory (never modified).");
ABSL_FLAG(std::string, tokenizer, "", "Original GPT-2 tokenizer directory.");
ABSL_FLAG(std::string, corpus, "", "One fact per line.");
ABSL_FLAG(int, layers, 4, "Checkpoint transformer block count.");
ABSL_FLAG(int, block, 0, "Zero-based transformer block to audit.");
ABSL_FLAG(int, model_width, 10, "Checkpoint hidden width.");
ABSL_FLAG(int, context_length, 27, "Checkpoint context length.");
ABSL_FLAG(int, attention_heads, 1, "Checkpoint head count.");
ABSL_FLAG(int, feed_forward_width, 20, "Checkpoint MLP expansion width.");
ABSL_FLAG(int, prompt_tokens, 5, "Number of supplied tokens, not scored.");
ABSL_FLAG(int, expected_samples, 1024, "Require this many corpus lines.");
ABSL_FLAG(bool, verify_prefixes, true,
          "Also re-run every scored causal prefix and compare exact vectors.");
ABSL_FLAG(int, max_examples, 10,
          "Maximum conflicting vector examples per tap.");

namespace pluto::llm::activation_collision_audit {
namespace {

// Tap order matches Capture's scope selection for the chosen block.
std::array<std::string, 4> TapNames(int block) {
  return {"Token + position embedding",
          absl::StrCat("Block ", block, " attention + residual"),
          absl::StrCat("Block ", block, " pre-MLP LayerNorm"),
          absl::StrCat("Block ", block, " MLP + residual")};
}

// These buffers are allocated once and remain alive until the last copy has
// completed. Every transfer uses the model's executor and pinned host memory.
struct CaptureStorage {
  cuda::PageLockedHostArray<int> context;
  cuda::PageLockedHostArray<int> predictions;
  std::array<cuda::PageLockedHostArray<uint16_t>, 4> taps;
  Buffer device_context;
  Buffer device_mask;
};

absl::StatusOr<CaptureStorage> AllocateCapture(cuda::Executor& executor,
                                               int context_length, int width) {
  ASSIGN_OR_RETURN(auto context, cuda::PageLockedHostArray<int>::Allocate(
                                     executor, context_length));
  ASSIGN_OR_RETURN(auto predictions, cuda::PageLockedHostArray<int>::Allocate(
                                         executor, context_length));
  ASSIGN_OR_RETURN(auto device_context,
                   Buffer::Allocate(executor, context.size_bytes()));
  ASSIGN_OR_RETURN(auto device_mask,
                   Buffer::Allocate(executor, context.size_bytes()));
  // A zero target enables every row for top-1 extraction. The audit itself
  // excludes prompt and padding rows instead of treating them as targets.
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemsetAsync(device_mask.data(), 0, device_mask.size_bytes(),
                      executor.stream()),
      "initialize prediction mask"));
  CaptureStorage storage{std::move(context),
                         std::move(predictions),
                         {},
                         std::move(device_context),
                         std::move(device_mask)};
  for (auto& tap : storage.taps) {
    ASSIGN_OR_RETURN(tap,
                     cuda::PageLockedHostArray<uint16_t>::Allocate(
                         executor, static_cast<size_t>(context_length) * width));
  }
  return storage;
}

absl::Status Capture(cuda::Executor& executor, const Layer& model,
                     const Gpt2Config& config, int block, int eos,
                     absl::Span<const int> tokens, CaptureStorage& storage) {
  if (tokens.empty() || tokens.size() > storage.context.size())
    return absl::InvalidArgumentError("invalid capture prefix length");
  std::fill(storage.context.begin(), storage.context.end(), eos);
  std::copy(tokens.begin(), tokens.end(), storage.context.begin());
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(storage.device_context.data(), storage.context.data(),
                      storage.context.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "upload capture prefix"));
  std::vector<std::string> scopes;
  const std::string block_name = absl::StrCat("transformer_block_", block);
  std::array<bool, 4> seen{};
  int residual_count = 0;
  LayerHooks hooks;
  hooks.enter_combinator = [&](auto&, auto name) {
    scopes.emplace_back(name);
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](auto&) {
    if (scopes.empty())
      return absl::DataLossError("unbalanced capture scope");
    scopes.pop_back();
    return absl::OkStatus();
  };
  hooks.activation_hook = [&](auto& hook_executor, auto name, auto types,
                              auto buffers) -> absl::Status {
    std::optional<size_t> tap;
    if (name == "PositionEmbeddingLayer" && scopes.size() == 1 &&
        scopes[0] == "gpt2")
      tap = 0;
    if (scopes.size() >= 2 && scopes[0] == "gpt2" && scopes[1] == block_name) {
      if (name == "ResidualLayer" && scopes.size() == 2) {
        if (residual_count >= 2)
          return absl::DataLossError("unexpected extra audited-block residual");
        tap = residual_count++ == 0 ? 1 : 3;
      }
      if (name == "LayerNormLayer" && scopes.size() == 4 &&
          scopes[2] == "ResidualLayer" && scopes[3] == "mlp")
        tap = 2;
    }
    if (!tap)
      return absl::OkStatus();
    const ActivationType expected(
        DataType::BF16, {ActivationType::kBatchDimension, config.context_length,
                         config.model_width});
    if (&hook_executor != &executor || seen[*tap] || types.size() != 1 ||
        types[0] != expected || buffers.size() != 1 ||
        &buffers[0].executor() != &executor ||
        buffers[0].size_bytes() != storage.taps[*tap].size_bytes())
      return absl::DataLossError("unexpected BF16 activation tap");
    seen[*tap] = true;
    return cuda::CudaStatus(
        cudaMemcpyAsync(storage.taps[*tap].data(), buffers[0].data(),
                        storage.taps[*tap].size_bytes(), cudaMemcpyDeviceToHost,
                        executor.stream()),
        "download exact BF16 activation");
  };
  ASSIGN_OR_RETURN(auto forward,
                   model.fwd(executor, {&storage.device_context, 1}, &hooks));
  if (!scopes.empty() ||
      !std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }))
    return absl::DataLossError("did not capture all four expected taps");
  forward.state = BackwardState{};
  if (forward.outputs.size() != 1)
    return absl::DataLossError("expected one logits output");
  ASSIGN_OR_RETURN(auto predicted,
                   ExtractTop1Ids(executor, forward.outputs[0],
                                  storage.device_mask, config.vocabulary_size));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(storage.predictions.data(), predicted.data(),
                      storage.predictions.size_bytes(), cudaMemcpyDeviceToHost,
                      executor.stream()),
      "download exact top-1 predictions"));
  return executor.Synchronize();
}

absl::StatusOr<std::string> Decode(
    const tokenizer::Gpt2Detokenizer& detokenizer,
    const tokenizer::CompactVocabularyTokenizer& vocabulary,
    absl::Span<const int> compact) {
  std::vector<int> original;
  for (int id : compact)
    original.push_back(vocabulary.original_token_ids()[id]);
  return detokenizer.Decode(original);
}

absl::Status Run() {
  const auto checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const auto tokenizer_path = absl::GetFlag(FLAGS_tokenizer);
  const auto corpus_path = absl::GetFlag(FLAGS_corpus);
  const int prompt = absl::GetFlag(FLAGS_prompt_tokens);
  const int max_examples = absl::GetFlag(FLAGS_max_examples);
  const int block = absl::GetFlag(FLAGS_block);
  if (block < 0 || block >= absl::GetFlag(FLAGS_layers))
    return absl::InvalidArgumentError("block must be in [0, layers)");
  const auto names = TapNames(block);
  if (checkpoint.empty() || tokenizer_path.empty() || corpus_path.empty() ||
      prompt <= 0 || absl::GetFlag(FLAGS_expected_samples) <= 0 ||
      absl::GetFlag(FLAGS_layers) <= 0 || max_examples < 0)
    return absl::InvalidArgumentError(
        "required paths or audit options invalid");
  ASSIGN_OR_RETURN(auto original,
                   tokenizer::Gpt2Tokenizer::Load(tokenizer_path));
  ASSIGN_OR_RETURN(auto detokenizer,
                   tokenizer::Gpt2Detokenizer::Load(tokenizer_path));
  ASSIGN_OR_RETURN(auto vocabulary,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *original, std::filesystem::path(checkpoint) /
                                      "compact_vocabulary.tsv"));
  const int eos = vocabulary->eos_token_id();
  if (original->vocab_size() != detokenizer->vocab_size() ||
      original->eos_token_id() != detokenizer->eos_token_id() ||
      vocabulary->original_eos_token_id() != original->eos_token_id())
    return absl::InvalidArgumentError("tokenizer/checkpoint identity mismatch");
  const Gpt2Config config{
      .transformer_block_count = absl::GetFlag(FLAGS_layers),
      .model_width = absl::GetFlag(FLAGS_model_width),
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = vocabulary->vocab_size(),
      .pad_vocabulary = false,
      .context_length = absl::GetFlag(FLAGS_context_length)};
  RETURN_IF_ERROR(config.Validate());
  if (prompt > config.context_length)
    return absl::InvalidArgumentError("prompt exceeds context length");
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(corpus_path));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto data, PaddedLineDataSetIterator::Create(
                                  *executor, corpus.text(), *vocabulary,
                                  {.batch_size = 1,
                                   .context_length = config.context_length,
                                   .prompt_tokens = prompt,
                                   .eos_token = eos,
                                   .shuffle = false}));
  if (data->sample_count() !=
      static_cast<size_t>(absl::GetFlag(FLAGS_expected_samples)))
    return absl::InvalidArgumentError("unexpected corpus sample count");
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(ReadFromDirectory(*executor, *model, checkpoint,
                                    /*allow_prefix=*/false));
  ASSIGN_OR_RETURN(
      auto storage,
      AllocateCapture(*executor, config.context_length, config.model_width));
  std::vector<Audit> audits;
  for (size_t tap = 0; tap < names.size(); ++tap)
    audits.emplace_back(config.model_width);
  std::array<absl::flat_hash_set<std::vector<uint16_t>>, 4> all_vectors;
  int64_t checked_prefixes = 0;
  int64_t total_rows = 0;
  for (size_t sample = 0; sample < data->sample_count(); ++sample) {
    const auto tokens = data->sample_tokens(sample);
    RETURN_IF_ERROR(
        Capture(*executor, *model, config, block, eos, tokens, storage));
    std::array<std::vector<uint16_t>, 4> full_sample;
    for (size_t tap = 0; tap < names.size(); ++tap) {
      full_sample[tap].assign(storage.taps[tap].begin(),
                              storage.taps[tap].end());
      for (size_t row = 0; row < tokens.size(); ++row) {
        const auto bits = storage.taps[tap].span().subspan(
            row * config.model_width, config.model_width);
        all_vectors[tap].emplace(bits.begin(), bits.end());
        if (row < static_cast<size_t>(prompt - 1))
          continue;  // Earlier prompt rows do not predict required suffixes.
        const int target = row + 1 < tokens.size() ? tokens[row + 1] : eos;
        if (storage.predictions[row] != target)
          return absl::DataLossError(
              absl::StrCat("checkpoint prediction wrong at sample ", sample + 1,
                           ", position ", row));
        RETURN_IF_ERROR(audits[tap].Add(
            bits, {static_cast<int>(sample), static_cast<int>(row), target}));
      }
    }
    total_rows += tokens.size();
    if (absl::GetFlag(FLAGS_verify_prefixes)) {
      for (size_t row = prompt - 1; row < tokens.size(); ++row) {
        RETURN_IF_ERROR(Capture(*executor, *model, config, block, eos,
                                tokens.first(row + 1), storage));
        const int target = row + 1 < tokens.size() ? tokens[row + 1] : eos;
        if (storage.predictions[row] != target)
          return absl::DataLossError("prefix prediction differs from target");
        for (size_t tap = 0; tap < names.size(); ++tap)
          if (!std::equal(
                  storage.taps[tap].begin(),
                  storage.taps[tap].begin() + (row + 1) * config.model_width,
                  full_sample[tap].begin()))
            return absl::DataLossError(absl::StrCat(
                "prefix activations differ at sample ", sample + 1,
                ", prefix length ", row + 1, ", tap ", names[tap]));
        ++checked_prefixes;
      }
    }
    if ((sample + 1) % 128 == 0 || sample + 1 == data->sample_count())
      std::cerr << "Audited " << sample + 1 << '/' << data->sample_count()
                << " facts; verified " << checked_prefixes << " prefixes\n";
  }
  std::cout << "# Exact activation collision audit\n\nCheckpoint: `"
            << checkpoint << "`\n\nCorpus: `" << corpus_path
            << "`\n\nConfiguration: " << config.transformer_block_count
            << " blocks, width " << config.model_width << ", MLP width "
            << config.feed_forward_width << ", context "
            << config.context_length << "; audited block " << block
            << " (zero-based).\n\nSamples: " << data->sample_count()
            << "; real input rows: " << total_rows
            << "; scored suffix/EOS rows: " << data->supervised_row_count()
            << "; native prediction errors: 0; verified prefixes: "
            << checked_prefixes << ".\n\n"
            << "Equality is bitwise equality of all BF16 channels. Prompt and "
               "padding rows are excluded from the target audit. All-vector "
               "counts include real prompt rows, not padding.\n\n"
            << "| Tap | All unique vectors | Scored unique vectors | Repeated "
               "scored groups | Conflicting groups | Rows in conflicts | "
               "Minimum unavoidable errors |\n"
            << "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n";
  std::vector<Summary> summaries;
  for (size_t tap = 0; tap < names.size(); ++tap) {
    summaries.push_back(audits[tap].Summarize());
    const auto& s = summaries.back();
    if (s.total_rows != data->supervised_row_count())
      return absl::DataLossError("audit target count mismatch");
    std::cout << "| " << names[tap] << " | " << all_vectors[tap].size() << " | "
              << s.unique_vectors << " | " << s.repeated_groups << " | "
              << s.conflicting_groups << " | " << s.conflicting_rows << " | "
              << s.minimum_errors << " |\n";
  }
  std::cout << "\nMinimum errors assumes an arbitrary deterministic function "
               "of this vector alone: each equality group must choose one "
               "target. For pre-MLP LayerNorm this bound applies to the "
               "normalized input alone, not the full residual block, which "
               "also receives the unnormalized residual. Zero conflicts is "
               "not proof that the existing small MLP can learn the mapping."
               "\n";
  for (size_t tap = 0; tap < names.size(); ++tap) {
    std::cout << "\n## " << names[tap] << "\n\n";
    if (summaries[tap].conflicts.empty()) {
      std::cout << "No exact vector is associated with different targets.\n";
      continue;
    }
    int shown = 0;
    for (const auto& group : summaries[tap].conflicts) {
      if (shown++ >= max_examples)
        break;
      std::cout << "### Conflicting vector " << shown << "\n\nBF16 bits: `";
      for (uint16_t word : group.bits)
        std::cout << absl::StrFormat("%04x ", word);
      std::cout << "`\n\nValues: `";
      for (uint16_t word : group.bits)
        std::cout << std::bit_cast<float>(static_cast<uint32_t>(word) << 16)
                  << ' ';
      std::cout << "`\n\n```text\n";
      // One witness per target is sufficient; frequencies retain the complete
      // group census without drowning a report in identical-label duplicates.
      std::map<int, std::vector<Occurrence>> by_target;
      for (const auto& occurrence : group.occurrences)
        by_target[occurrence.target].push_back(occurrence);
      for (const auto& [target, occurrences] : by_target) {
        const auto& occurrence = occurrences.front();
        const auto tokens = data->sample_tokens(occurrence.sample);
        ASSIGN_OR_RETURN(auto prefix,
                         Decode(*detokenizer, *vocabulary,
                                tokens.first(occurrence.position + 1)));
        ASSIGN_OR_RETURN(auto text,
                         Decode(*detokenizer, *vocabulary, {&target, 1}));
        std::cout << "sample " << occurrence.sample + 1 << ", position "
                  << occurrence.position << ": \"" << absl::CEscape(prefix)
                  << "\" -> \"" << absl::CEscape(text) << "\" (target "
                  << target << ", " << occurrences.size() << " rows)\n";
      }
      std::cout << "```\n\n";
    }
  }
  return std::cout ? absl::OkStatus()
                   : absl::UnknownError("writing audit report failed");
}

}  // namespace
}  // namespace pluto::llm::activation_collision_audit

int main(int argc, char** argv) {
  const auto positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "Unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::llm::activation_collision_audit::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
