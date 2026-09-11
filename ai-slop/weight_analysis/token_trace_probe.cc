#include "ai-slop/weight_analysis/token_trace_probe.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "ai-slop/weight_analysis/causal_probe.h"
#include "ai-slop/weight_analysis/phrase_probe.h"
#include "src/llm/checkpoint.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Read-only full GPT-2 checkpoint.");
ABSL_FLAG(
    std::string, tokens_file, "",
    "Exact generated prefix as little-endian int32 IDs; no retokenizing.");
ABSL_FLAG(int, target_id, -1, "Emitted token being investigated (logical ID).");
ABSL_FLAG(std::string, output_dir, "", "New exclusive output directory.");
ABSL_FLAG(bool, interventions, false,
          "Run independent zero interventions on 16 branches and 64 heads.");
ABSL_FLAG(std::string, ablate_neuron, "",
          "Optional comma-separated zero-based block:neuron interventions.");

namespace pluto::weight_analysis {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int kWidth = llm::kGpt2ModelWidth;
constexpr int kVocabulary = llm::kGpt2VocabularySize;
constexpr int kPaddedVocabulary = llm::kGpt2PaddedVocabularySize;
constexpr int kContext = llm::kGpt2ContextLength;
static_assert(kWidth == 512 && kVocabulary == 50257 &&
              kPaddedVocabulary == 50272 && kContext == 1024);
static_assert(llm::kGpt2TransformerBlockCount == 8 &&
              llm::kGpt2AttentionHeads == 8 &&
              llm::kGpt2AttentionHeadDimension == 64 &&
              llm::kGpt2FeedForwardWidth == 2048);
static_assert(sizeof(int) == sizeof(int32_t) && sizeof(float) == 4);

absl::Status CheckFinite(const cuda::PageLockedHostArray<uint8_t>& bytes,
                         bool fp32) {
  const size_t stride = fp32 ? 4 : 2;
  if (bytes.size_bytes() % stride)
    return absl::InvalidArgumentError("misaligned native tensor");
  for (size_t i = 0; i < bytes.size_bytes(); i += stride) {
    float value;
    if (fp32) {
      std::memcpy(&value, bytes.data() + i, 4);
    } else {
      uint16_t bits;
      std::memcpy(&bits, bytes.data() + i, 2);
      value = std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
    }
    if (!std::isfinite(value))
      return absl::DataLossError("nonfinite native tensor");
  }
  return absl::OkStatus();
}

absl::Status CheckEqual(const cuda::PageLockedHostArray<uint8_t>& a,
                        const cuda::PageLockedHostArray<uint8_t>& b,
                        absl::string_view description) {
  if (a.size_bytes() != b.size_bytes() ||
      std::memcmp(a.data(), b.data(), a.size_bytes()) != 0)
    return absl::DataLossError(absl::StrCat(description, " is not byte-equal"));
  return absl::OkStatus();
}

absl::Status Upload(cuda::Executor& executor,
                    const cuda::PageLockedHostArray<int32_t>& tokens,
                    const cuda::Buffer& input) {
  if (tokens.size_bytes() != input.size_bytes() ||
      &input.executor() != &executor)
    return absl::InvalidArgumentError("token upload shape/executor mismatch");
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(input.data(), tokens.data(), tokens.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "exact prefix token upload"));
  return executor.Synchronize();
}

absl::StatusOr<cuda::PageLockedHostArray<uint8_t>> ForwardRow(
    cuda::Executor& executor, const llm::Layer& model,
    const cuda::Buffer& input, int selected_row) {
  llm::Tape tape;
  ASSIGN_OR_RETURN(auto logits,
                   model.fwd(executor, absl::MakeConstSpan(&input, 1), &tape));
  ASSIGN_OR_RETURN(auto host, ReadSelectedRow(executor, logits, selected_row,
                                              kPaddedVocabulary, 4));
  RETURN_IF_ERROR(CheckFinite(host, true));
  return host;
}

std::string TensorJson(const std::string& filename, const std::string& dtype,
                       int rows, int columns, const std::string& source,
                       const std::string& role) {
  return absl::StrCat("{\"file\":", JsonQuote(filename),
                      ",\"dtype\":", JsonQuote(dtype), ",\"shape\":[", rows,
                      ",", columns, "],\"source\":", JsonQuote(source),
                      ",\"role\":", JsonQuote(role), "}");
}

absl::Status Run() {
  if constexpr (std::endian::native != std::endian::little)
    return absl::UnimplementedError("requires little-endian native bytes");
  const auto started = Clock::now();
  const int target = absl::GetFlag(FLAGS_target_id);
  if (absl::GetFlag(FLAGS_checkpoint).empty() ||
      absl::GetFlag(FLAGS_tokens_file).empty() ||
      absl::GetFlag(FLAGS_output_dir).empty() || target < 0 ||
      target >= kVocabulary)
    return absl::InvalidArgumentError(
        "required: --checkpoint --tokens_file "
        "--output_dir --target_id=0..50256");
  ASSIGN_OR_RETURN(auto neurons, ParseNeuronInterventions(
                                     absl::GetFlag(FLAGS_ablate_neuron)));
  const auto checkpoint = fs::canonical(absl::GetFlag(FLAGS_checkpoint));
  // Validate the original path before canonicalization, so a file symlink is
  // rejected rather than silently resolved. IDs are never passed to a
  // tokenizer.
  const auto token_path = fs::absolute(absl::GetFlag(FLAGS_tokens_file));
  ASSIGN_OR_RETURN(auto encoded,
                   ReadTokenIds(token_path, kVocabulary, kContext));
  const auto token_bytes = fs::file_size(token_path);
  const auto token_modified = fs::last_write_time(token_path);
  const auto output = fs::absolute(absl::GetFlag(FLAGS_output_dir));
  const auto resolved_output =
      (fs::canonical(output.parent_path()) / output.filename())
          .lexically_normal();
  for (auto parent = resolved_output;; parent = parent.parent_path()) {
    if (parent == checkpoint)
      return absl::InvalidArgumentError("output must not be inside checkpoint");
    if (parent == parent.parent_path()) break;
  }
  // Validate the supplied path, not its already-canonicalized symlink target.
  ASSIGN_OR_RETURN(auto checkpoint_files,
                   InspectGpt2CheckpointFiles(
                       fs::absolute(absl::GetFlag(FLAGS_checkpoint))));
  RETURN_IF_ERROR(CreateNewOutputDirectory(output));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  const int rows = static_cast<int>(encoded.size());
  const int selected_row = rows - 1;
  const int padding = encoded[selected_row];
  const int alternate_padding = padding == 0 ? 1 : 0;
  ASSIGN_OR_RETURN(auto tokens,
                   cuda::PageLockedHostArray<int32_t>::Allocate(kContext));
  std::fill(tokens.begin(), tokens.end(), padding);
  std::copy(encoded.begin(), encoded.end(), tokens.begin());
  ASSIGN_OR_RETURN(auto input,
                   cuda::Buffer::Allocate(*executor, tokens.size_bytes()));
  RETURN_IF_ERROR(Upload(*executor, tokens, input));
  ASSIGN_OR_RETURN(auto model,
                   llm::CreateGpt2(*executor, llm::DataType::BF16, 0));
  ASSIGN_OR_RETURN(auto weights, UniqueWeights(*executor, model->weights()));
  RETURN_IF_ERROR(ValidateGpt2Weights(weights));
  if (model->weights().size() != 101 ||
      model->weights().front().data() != model->weights().back().data())
    return absl::FailedPreconditionError("changed tied-embedding traversal");
  RETURN_IF_ERROR(llm::ReadFromDirectory(*executor, *model, checkpoint));
  llm::Tape original_tape;
  ASSIGN_OR_RETURN(
      auto original_logits,
      model->fwd(*executor, absl::MakeConstSpan(&input, 1), &original_tape));
  ASSIGN_OR_RETURN(auto baseline,
                   ReadSelectedRow(*executor, original_logits, selected_row,
                                   kPaddedVocabulary, 4));
  RETURN_IF_ERROR(CheckFinite(baseline, true));
  // A sampled target need not be greedy. Preserve its exact native logit and
  // rank rather than silently replacing the investigated token with top-1.
  const auto* baseline_values = reinterpret_cast<const float*>(baseline.data());
  const float target_logit = baseline_values[target];
  int target_rank = 1, greedy_id = 0;
  for (int id = 0; id < kVocabulary; ++id) {
    if (baseline_values[id] > target_logit ||
        (baseline_values[id] == target_logit && id < target))
      ++target_rank;
    if (baseline_values[id] > baseline_values[greedy_id]) greedy_id = id;
  }
  std::cout << "Native token trace: prefix=" << rows << ", target=" << target
            << ", target rank=" << target_rank << ", greedy=" << greedy_id
            << std::endl;
  ASSIGN_OR_RETURN(auto frames, CollectGpt2Trace(*executor, original_tape,
                                                 original_logits, weights));
  ASSIGN_OR_RETURN(auto lens, NativeLogitLens::Create(*executor, weights));
  ASSIGN_OR_RETURN(auto embedding, lens->Embed(*executor, input));
  frames.insert(frames.begin(),
                {"embedding", std::move(embedding), kWidth, false, true});
  std::map<std::string, const TraceFrame*> by_name;
  for (const auto& frame : frames) {
    if (!by_name.emplace(frame.name, &frame).second)
      return absl::InternalError("duplicate trace name");
  }
  for (int block = 0; block < 8; ++block) {
    const auto p = absl::StrCat("blocks.", block, ".");
    const std::string before =
        block == 0 ? "positioned"
                   : absl::StrCat("blocks.", block - 1, ".after_mlp");
    RETURN_IF_ERROR(VerifyResidualReplay(
        *executor, by_name.at(before)->buffer,
        by_name.at(p + "attention_projected")->buffer,
        by_name.at(p + "after_attention")->buffer, rows, kWidth));
    RETURN_IF_ERROR(VerifyResidualReplay(
        *executor, by_name.at(p + "after_attention")->buffer,
        by_name.at(p + "mlp_projected")->buffer,
        by_name.at(p + "after_mlp")->buffer, rows, kWidth));
  }
  std::ostringstream files_json;
  bool first_file = true;
  for (const auto& frame : frames) {
    // Full BF16 prefix is needed to reconstruct which earlier keys/values the
    // selected query attends to. Vocabulary logits need only the selected row.
    auto host_result =
        frame.fp32 ? ReadSelectedRow(*executor, frame.buffer, selected_row,
                                     frame.width, 4)
                   : ReadPrefix(*executor, frame.buffer, rows, frame.width, 2);
    if (!host_result.ok()) return host_result.status();
    auto host = std::move(*host_result);
    RETURN_IF_ERROR(CheckFinite(host, frame.fp32));
    const auto filename = frame.name + (frame.fp32 ? ".f32" : ".bf16");
    RETURN_IF_ERROR(
        WriteExclusive(output / filename, host.data(), host.size_bytes()));
    if (!first_file) files_json << ',';
    first_file = false;
    files_json << JsonQuote(frame.name) << ':'
               << TensorJson(filename, frame.fp32 ? "float32" : "bf16",
                             frame.fp32 ? 1 : rows, frame.width,
                             frame.native_replay ? "native_replay_saved_input"
                                                 : "native_original_forward",
                             frame.fp32 ? "selected_row" : "full_prefix");
  }
  RETURN_IF_ERROR(WriteExclusive(output / "tokens.i32", encoded.data(),
                                 encoded.size_bytes()));
  files_json << ",\"tokens\":"
             << TensorJson("tokens.i32", "int32", rows, 1,
                           "exact_input_token_ids", "full_prefix");
  std::vector<std::string> residual_names{"positioned"};
  for (int block = 0; block < 8; ++block) {
    residual_names.push_back(
        absl::StrCat("blocks.", block, ".after_attention"));
    residual_names.push_back(absl::StrCat("blocks.", block, ".after_mlp"));
  }
  std::ostringstream lens_json;
  for (size_t i = 0; i < residual_names.size(); ++i) {
    const auto& name = residual_names[i];
    ASSIGN_OR_RETURN(auto logits,
                     lens->Apply(*executor, by_name.at(name)->buffer));
    ASSIGN_OR_RETURN(auto host, ReadSelectedRow(*executor, logits, selected_row,
                                                kPaddedVocabulary, 4));
    RETURN_IF_ERROR(CheckFinite(host, true));
    if (i + 1 == residual_names.size())
      RETURN_IF_ERROR(CheckEqual(baseline, host, "final lens selected logits"));
    const auto filename = "lens." + name + ".f32";
    RETURN_IF_ERROR(
        WriteExclusive(output / filename, host.data(), host.size_bytes()));
    if (i) lens_json << ',';
    lens_json << JsonQuote(name) << ':'
              << TensorJson(filename, "float32", 1, kPaddedVocabulary,
                            "native_diagnostic_final_norm_and_tied_head",
                            "selected_row");
  }
  std::cout << "Captured full-prefix stages and 17 selected-row readouts"
            << std::endl;
  std::fill(tokens.begin() + rows, tokens.end(), alternate_padding);
  RETURN_IF_ERROR(Upload(*executor, tokens, input));
  ASSIGN_OR_RETURN(auto alternate,
                   ForwardRow(*executor, *model, input, selected_row));
  RETURN_IF_ERROR(
      CheckEqual(baseline, alternate, "alternate-padding selected logits"));
  RETURN_IF_ERROR(WriteExclusive(output / "alternate_padding.logits.f32",
                                 alternate.data(), alternate.size_bytes()));
  std::fill(tokens.begin() + rows, tokens.end(), padding);
  RETURN_IF_ERROR(Upload(*executor, tokens, input));

  std::ostringstream interventions_json;
  int arm_count = 0;
  const auto save_arm =
      [&](const std::string& name, const std::string& kind, int block,
          int element,
          const cuda::PageLockedHostArray<uint8_t>& result) -> absl::Status {
    const auto filename = name + ".logits.f32";
    RETURN_IF_ERROR(
        WriteExclusive(output / filename, result.data(), result.size_bytes()));
    if (arm_count++) interventions_json << ',';
    interventions_json << "{\"name\":" << JsonQuote(name)
                       << ",\"kind\":" << JsonQuote(kind)
                       << ",\"block\":" << block;
    if (element >= 0) interventions_json << ",\"head_or_neuron\":" << element;
    interventions_json
        << ",\"scale\":0,\"logits_file\":" << JsonQuote(filename)
        << ",\"shape\":[1," << kPaddedVocabulary
        << "],\"role\":\"selected_row\",\"restoration_verified_bytes\":true}";
    std::cout << "Intervention " << arm_count << ": " << name << std::endl;
    return absl::OkStatus();
  };
  if (absl::GetFlag(FLAGS_interventions)) {
    for (int block = 0; block < 8; ++block) {
      for (bool mlp : {false, true}) {
        const int index = 12 * block + (mlp ? 12 : 6);
        const std::vector<int> indices{index, index + 1};
        ASSIGN_OR_RETURN(auto intervention, WeightIntervention::Capture(
                                                *executor, weights, indices));
        RETURN_IF_ERROR(intervention->Apply(0));
        auto result = ForwardRow(*executor, *model, input, selected_row);
        RETURN_IF_ERROR(intervention->RestoreAndVerify());
        if (!result.ok()) return result.status();
        const std::string kind = mlp ? "mlp_branch" : "attention_branch";
        RETURN_IF_ERROR(
            save_arm(absl::StrCat("ablation.block", block, ".", kind), kind,
                     block, -1, *result));
      }
      for (int head = 0; head < 8; ++head) {
        std::vector<int> head_rows(64);
        for (int lane = 0; lane < 64; ++lane)
          head_rows[lane] = head * 64 + lane;
        ASSIGN_OR_RETURN(
            auto intervention,
            MlpRowIntervention::Capture(*executor, weights[6 + 12 * block],
                                        kWidth, kWidth, head_rows));
        RETURN_IF_ERROR(intervention->Apply(0));
        auto result = ForwardRow(*executor, *model, input, selected_row);
        RETURN_IF_ERROR(intervention->RestoreAndVerify());
        if (!result.ok()) return result.status();
        RETURN_IF_ERROR(
            save_arm(absl::StrCat("ablation.block", block, ".head", head),
                     "attention_head", block, head, *result));
      }
    }
  }
  for (const auto& [block, neuron] : neurons) {
    const std::vector<int> selected{neuron};
    ASSIGN_OR_RETURN(auto intervention,
                     MlpRowIntervention::Capture(
                         *executor, weights[12 + 12 * block],
                         llm::kGpt2FeedForwardWidth, kWidth, selected));
    RETURN_IF_ERROR(intervention->Apply(0));
    auto result = ForwardRow(*executor, *model, input, selected_row);
    RETURN_IF_ERROR(intervention->RestoreAndVerify());
    if (!result.ok()) return result.status();
    RETURN_IF_ERROR(
        save_arm(absl::StrCat("ablation.block", block, ".neuron", neuron),
                 "mlp_neuron", block, neuron, *result));
  }
  ASSIGN_OR_RETURN(auto clean,
                   ForwardRow(*executor, *model, input, selected_row));
  RETURN_IF_ERROR(CheckEqual(baseline, clean, "clean replay selected logits"));
  RETURN_IF_ERROR(WriteExclusive(output / "clean_replay.logits.f32",
                                 clean.data(), clean.size_bytes()));
  RETURN_IF_ERROR(VerifyCheckpointFilesUnchanged(checkpoint_files));
  if (!fs::is_regular_file(token_path) || fs::is_symlink(token_path) ||
      fs::file_size(token_path) != token_bytes ||
      fs::last_write_time(token_path) != token_modified)
    return absl::DataLossError("token input file stat changed during probe");
  RETURN_IF_ERROR(executor->Synchronize());

  std::ostringstream metadata;
  metadata << std::setprecision(17)
           << "{\"schema_version\":1,\"probe_kind\":\"token_trace\","
              "\"complete\":true,\"token_ids\":[";
  for (int i = 0; i < rows; ++i) {
    if (i) metadata << ',';
    metadata << encoded[i];
  }
  metadata
      << "],\"prompt_rows\":" << rows << ",\"selected_row\":" << selected_row
      << ",\"target_id\":" << target << ",\"target_logit\":" << target_logit
      << ",\"target_rank\":" << target_rank << ",\"greedy_id\":" << greedy_id
      << ",\"context_length\":" << kContext << ",\"pad_token_id\":" << padding
      << ",\"alternate_pad_token_id\":" << alternate_padding
      << ",\"no_bos\":true,\"autoregressive\":false,\"retokenized\":false"
      << ",\"vocab_size\":" << kVocabulary
      << ",\"padded_vocab_size\":" << kPaddedVocabulary
      << ",\"checkpoint_directory\":" << JsonQuote(checkpoint.string())
      << ",\"tokens_file\":" << JsonQuote(fs::canonical(token_path).string())
      << ",\"binary_file\":"
      << JsonQuote(fs::canonical("/proc/self/exe").string())
      << ",\"checkpoint_unique_weight_count\":100,\"checkpoint_raw_weight_"
         "count\":101"
      << ",\"compute_type\":\"BF16 activations and MMA operands; FP32 master "
         "weights, reductions and logits\""
      << ",\"byte_order\":\"little\",\"original_forward\":\"unmodified "
         "CreateGpt2\""
      << ",\"optimizer_steps\":0,\"backward_calls\":0,\"checkpoint_writes\":0"
      << ",\"integrity_hashes\":\"external runner hashes complete inputs and "
         "outputs\""
      << ",\"files\":{" << files_json.str() << "},\"lens\":{" << lens_json.str()
      << "},\"interventions\":[" << interventions_json.str() << ']'
      << ",\"checks\":{\"final_lens_logits_byte_equal\":true,"
         "\"alternate_padding_logits_byte_equal\":true,"
         "\"clean_replay_logits_byte_equal\":true,"
         "\"attention_residual_replay_byte_equal\":true,"
         "\"mlp_residual_replay_byte_equal\":true,"
         "\"checkpoint_file_stats_unchanged\":true,"
         "\"token_file_stats_unchanged\":true,"
         "\"all_exported_activations_finite\":true}"
      << ",\"logit_parity_scope\":\"selected_row\",\"residual_parity_scope\":"
         "\"full_prefix\""
      << ",\"parity_files\":{\"alternate_padding\":\"alternate_padding.logits."
         "f32\","
         "\"clean_replay\":\"clean_replay.logits.f32\"}"
      << ",\"elapsed_seconds\":"
      << std::chrono::duration<double>(Clock::now() - started).count() << "}\n";
  const auto json = metadata.str();
  RETURN_IF_ERROR(
      WriteExclusive(output / "metadata.json", json.data(), json.size()));
  std::cout << "Complete: " << output << std::endl;
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::weight_analysis

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "Unexpected positional arguments\n";
    return 1;
  }
  try {
    const auto status = pluto::weight_analysis::Run();
    if (status.ok()) return 0;
    std::cerr << status << '\n';
  } catch (const std::exception& error) {
    std::cerr << "Token trace failed: " << error.what() << '\n';
  }
  return 1;
}
