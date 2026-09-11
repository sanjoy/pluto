#include "ai-slop/weight_analysis/phrase_probe.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "ai-slop/weight_analysis/causal_probe.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Read-only full GPT-2 checkpoint directory.");
ABSL_FLAG(std::string, tokenizer, "", "Runtime GPT-2 tokenizer directory.");
ABSL_FLAG(std::string, output_dir, "",
          "New exclusive diagnostic output directory.");
ABSL_FLAG(std::string, prompt, "to be or not to be,",
          "Exact literal prompt, without automatic BOS/EOS/newline insertion.");
ABSL_FLAG(bool, interventions, false,
          "Also run all 16 independent branch and 64 head zero interventions.");
ABSL_FLAG(std::string, ablate_neuron, "",
          "Optional comma-separated zero-based block:neuron interventions.");

namespace pluto::weight_analysis {
namespace {
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;
constexpr int kWidth = llm::kGpt2ModelWidth;
constexpr int kVocabulary = llm::kGpt2VocabularySize;
constexpr int kPaddedVocabulary = llm::kGpt2PaddedVocabularySize;
constexpr int kContext = llm::kGpt2ContextLength;
static_assert(kWidth == 512 && kVocabulary == 50257 &&
              kPaddedVocabulary == 50272);
static_assert(kContext == 1024 && llm::kGpt2TransformerBlockCount == 8);
static_assert(llm::kGpt2AttentionHeads == 8 &&
              llm::kGpt2AttentionHeadDimension == 64 &&
              llm::kGpt2FeedForwardWidth == 2048);
static_assert(sizeof(int) == sizeof(int32_t) && sizeof(float) == 4);

absl::Status CheckFinite(const cuda::PageLockedHostArray<uint8_t>& bytes,
                         bool fp32) {
  const size_t stride = fp32 ? 4 : 2;
  if (bytes.size_bytes() % stride)
    return absl::InvalidArgumentError("misaligned native tensor byte count");
  for (size_t i = 0; i < bytes.size_bytes(); i += stride) {
    float value;
    if (fp32) {
      std::memcpy(&value, bytes.data() + i, 4);
    } else {
      uint16_t bits;
      std::memcpy(&bits, bytes.data() + i, 2);
      value = std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
    }
    if (!std::isfinite(value)) {
      return absl::DataLossError(
          absl::StrCat("nonfinite native tensor at ", i / stride));
    }
  }
  return absl::OkStatus();
}

absl::Status CheckEqual(const cuda::PageLockedHostArray<uint8_t>& first,
                        const cuda::PageLockedHostArray<uint8_t>& second,
                        absl::string_view description) {
  if (first.size_bytes() != second.size_bytes() ||
      std::memcmp(first.data(), second.data(), first.size_bytes()) != 0) {
    return absl::DataLossError(
        absl::StrCat(description, " is not byte-identical"));
  }
  return absl::OkStatus();
}

absl::Status Upload(cuda::Executor& executor,
                    const cuda::PageLockedHostArray<int>& tokens,
                    const cuda::Buffer& input) {
  if (tokens.size_bytes() != input.size_bytes() ||
      &input.executor() != &executor) {
    return absl::InvalidArgumentError("token upload shape/executor mismatch");
  }
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(input.data(), tokens.data(), tokens.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "phrase tokens upload"));
  // The caller may immediately change the pinned suffix for another pass.
  return executor.Synchronize();
}

absl::StatusOr<cuda::PageLockedHostArray<uint8_t>> ForwardPrefix(
    cuda::Executor& executor, const llm::Layer& model,
    const cuda::Buffer& input, int rows) {
  llm::Tape tape;
  ASSIGN_OR_RETURN(auto logits,
                   model.fwd(executor, absl::MakeConstSpan(&input, 1), &tape));
  ASSIGN_OR_RETURN(auto host,
                   ReadPrefix(executor, logits, rows, kPaddedVocabulary, 4));
  RETURN_IF_ERROR(CheckFinite(host, true));
  return host;
}

std::string TensorJson(const std::string& filename, const std::string& dtype,
                       int rows, int columns, const std::string& source) {
  return absl::StrCat("{\"file\":", JsonQuote(filename),
                      ",\"dtype\":", JsonQuote(dtype), ",\"shape\":[", rows,
                      ",", columns, "],\"source\":", JsonQuote(source), "}");
}

absl::Status Run() {
  if constexpr (std::endian::native != std::endian::little) {
    return absl::UnimplementedError(
        "probe requires little-endian native bytes");
  }
  const auto started = Clock::now();
  if (absl::GetFlag(FLAGS_checkpoint).empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() ||
      absl::GetFlag(FLAGS_output_dir).empty()) {
    return absl::InvalidArgumentError(
        "required: --checkpoint --tokenizer --output_dir");
  }
  ASSIGN_OR_RETURN(auto neurons,
                   ParseNeuronInterventions(absl::GetFlag(FLAGS_ablate_neuron)));
  const auto checkpoint = fs::canonical(absl::GetFlag(FLAGS_checkpoint));
  const auto tokenizer_path = fs::canonical(absl::GetFlag(FLAGS_tokenizer));
  const auto output = fs::absolute(absl::GetFlag(FLAGS_output_dir));
  const auto resolved_output =
      (fs::canonical(output.parent_path()) / output.filename())
          .lexically_normal();
  for (auto parent = resolved_output;; parent = parent.parent_path()) {
    if (parent == checkpoint || parent == tokenizer_path) {
      return absl::InvalidArgumentError(
          "output must not be inside an input directory");
    }
    if (parent == parent.parent_path())
      break;
  }
  // Validate the supplied path, not its already-canonicalized symlink target.
  ASSIGN_OR_RETURN(
      auto checkpoint_files,
      InspectGpt2CheckpointFiles(fs::absolute(absl::GetFlag(FLAGS_checkpoint))));
  RETURN_IF_ERROR(CreateNewOutputDirectory(output));

  // Executor outlives every device buffer and all queued stream-ordered frees.
  // There is deliberately no training, backward pass, or checkpoint writer.
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto tokenizer,
                   tokenizer::Gpt2Tokenizer::Load(tokenizer_path));
  ASSIGN_OR_RETURN(auto detokenizer,
                   tokenizer::Gpt2Detokenizer::Load(tokenizer_path));
  if (tokenizer->vocab_size() != kVocabulary ||
      detokenizer->vocab_size() != kVocabulary) {
    return absl::InvalidArgumentError(
        "tokenizer vocabulary does not match GPT-2");
  }
  const std::string prompt = absl::GetFlag(FLAGS_prompt);
  ASSIGN_OR_RETURN(auto encoded, tokenizer->Encode(*executor, prompt));
  if (encoded.empty() || encoded.size() > kContext)
    return absl::InvalidArgumentError("prompt must encode to 1..1024 tokens");
  ASSIGN_OR_RETURN(auto decoded, detokenizer->Decode(encoded.span()));
  if (decoded != prompt)
    return absl::DataLossError("native tokenizer round trip failed");
  const int rows = static_cast<int>(encoded.size());
  // This exactly matches production inference's right-padding convention.
  // Causal attention makes the unused suffix invisible to all exported rows.
  const int padding = encoded[rows - 1];
  const int alternate_padding = padding == 0 ? tokenizer->eos_token_id() : 0;
  ASSIGN_OR_RETURN(auto tokens, cuda::PageLockedHostArray<int>::Allocate(
                                    *executor, kContext));
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
      model->weights().front().data() != model->weights().back().data()) {
    return absl::FailedPreconditionError("changed tied-embedding traversal");
  }
  RETURN_IF_ERROR(llm::ReadFromDirectory(*executor, *model, checkpoint));
  llm::Tape original_tape;
  ASSIGN_OR_RETURN(
      auto original_logits,
      model->fwd(*executor, absl::MakeConstSpan(&input, 1), &original_tape));
  ASSIGN_OR_RETURN(auto baseline, ReadPrefix(*executor, original_logits, rows,
                                             kPaddedVocabulary, 4));
  RETURN_IF_ERROR(CheckFinite(baseline, true));
  std::cout << "Original native forward complete: " << rows << " prompt tokens"
            << std::endl;

  ASSIGN_OR_RETURN(auto frames, CollectGpt2Trace(*executor, original_tape,
                                                 original_logits, weights));
  ASSIGN_OR_RETURN(auto lens, NativeLogitLens::Create(*executor, weights));
  ASSIGN_OR_RETURN(auto embedding, lens->Embed(*executor, input));
  frames.insert(frames.begin(),
                {"embedding", std::move(embedding), kWidth, false, true});
  std::map<std::string, const TraceFrame*> by_name;
  for (const auto& frame : frames)
    if (!by_name.emplace(frame.name, &frame).second)
      return absl::InternalError("duplicate native frame name");
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
    ASSIGN_OR_RETURN(auto host, ReadPrefix(*executor, frame.buffer, rows,
                                           frame.width, frame.fp32 ? 4 : 2));
    RETURN_IF_ERROR(CheckFinite(host, frame.fp32));
    const auto filename = frame.name + (frame.fp32 ? ".f32" : ".bf16");
    RETURN_IF_ERROR(
        WriteExclusive(output / filename, host.data(), host.size_bytes()));
    if (!first_file)
      files_json << ',';
    first_file = false;
    files_json << JsonQuote(frame.name) << ':'
               << TensorJson(filename, frame.fp32 ? "float32" : "bf16", rows,
                             frame.width,
                             frame.native_replay ? "native_replay_saved_input"
                                                 : "native_original_forward");
  }
  RETURN_IF_ERROR(WriteExclusive(output / "tokens.i32", encoded.data(),
                                 encoded.size_bytes()));
  files_json << ",\"tokens\":"
             << TensorJson("tokens.i32", "int32", rows, 1, "native_tokenizer");

  // A native logit lens intentionally adds separate readout passes. These
  // never feed back into the original forward and are not generation steps.
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
    ASSIGN_OR_RETURN(auto host,
                     ReadPrefix(*executor, logits, rows, kPaddedVocabulary, 4));
    RETURN_IF_ERROR(CheckFinite(host, true));
    if (i + 1 == residual_names.size())
      RETURN_IF_ERROR(CheckEqual(baseline, host, "final native lens logits"));
    const auto filename = "lens." + name + ".f32";
    RETURN_IF_ERROR(
        WriteExclusive(output / filename, host.data(), host.size_bytes()));
    if (i)
      lens_json << ',';
    lens_json << JsonQuote(name) << ':'
              << TensorJson(filename, "float32", rows, kPaddedVocabulary,
                            "native_diagnostic_final_norm_and_tied_head");
  }
  std::cout << "Captured " << frames.size()
            << " stages and 17 native logit-lens readouts" << std::endl;

  std::fill(tokens.begin() + rows, tokens.end(), alternate_padding);
  RETURN_IF_ERROR(Upload(*executor, tokens, input));
  ASSIGN_OR_RETURN(auto alternate,
                   ForwardPrefix(*executor, *model, input, rows));
  RETURN_IF_ERROR(
      CheckEqual(baseline, alternate, "alternate-padding prefix logits"));
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
    if (arm_count++)
      interventions_json << ',';
    interventions_json << "{\"name\":" << JsonQuote(name)
                       << ",\"kind\":" << JsonQuote(kind)
                       << ",\"block\":" << block;
    if (element >= 0)
      interventions_json << ",\"head_or_neuron\":" << element;
    interventions_json << ",\"scale\":0,\"logits_file\":" << JsonQuote(filename)
                       << ",\"shape\":[" << rows << ',' << kPaddedVocabulary
                       << ']' << ",\"restoration_verified_bytes\":true}";
    std::cout << "Intervention " << arm_count << ": " << name << " complete"
              << std::endl;
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
        auto result = ForwardPrefix(*executor, *model, input, rows);
        // Even an invalid forward result must restore before propagating its
        // error. RestoreAndVerify checks device bytes against pristine weights.
        RETURN_IF_ERROR(intervention->RestoreAndVerify());
        if (!result.ok())
          return result.status();
        const std::string kind = mlp ? "mlp_branch" : "attention_branch";
        RETURN_IF_ERROR(
            save_arm(absl::StrCat("ablation.block", block, ".", kind), kind,
                     block, -1, *result));
      }
      for (int head = 0; head < 8; ++head) {
        std::vector<int> head_rows(64);
        for (int lane = 0; lane < 64; ++lane)
          head_rows[lane] = head * 64 + lane;
        // W_o is [input_channel, output_channel]. Zeroing one head's 64
        // input rows removes its projected contribution and keeps W_o's bias.
        ASSIGN_OR_RETURN(
            auto intervention,
            MlpRowIntervention::Capture(*executor, weights[6 + 12 * block],
                                        kWidth, kWidth, head_rows));
        RETURN_IF_ERROR(intervention->Apply(0));
        auto result = ForwardPrefix(*executor, *model, input, rows);
        RETURN_IF_ERROR(intervention->RestoreAndVerify());
        if (!result.ok())
          return result.status();
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
    auto result = ForwardPrefix(*executor, *model, input, rows);
    RETURN_IF_ERROR(intervention->RestoreAndVerify());
    if (!result.ok())
      return result.status();
    RETURN_IF_ERROR(
        save_arm(absl::StrCat("ablation.block", block, ".neuron", neuron),
                 "mlp_neuron", block, neuron, *result));
  }
  ASSIGN_OR_RETURN(auto clean, ForwardPrefix(*executor, *model, input, rows));
  RETURN_IF_ERROR(
      CheckEqual(baseline, clean, "final clean replay prefix logits"));
  RETURN_IF_ERROR(WriteExclusive(output / "clean_replay.logits.f32",
                                 clean.data(), clean.size_bytes()));
  RETURN_IF_ERROR(VerifyCheckpointFilesUnchanged(checkpoint_files));
  RETURN_IF_ERROR(executor->Synchronize());

  std::ostringstream metadata;
  metadata << std::setprecision(17)
           << "{\"schema_version\":1,\"complete\":true,\"prompt\":"
           << JsonQuote(prompt) << ",\"token_ids\":[";
  for (int i = 0; i < rows; ++i) {
    if (i)
      metadata << ',';
    metadata << encoded[i];
  }
  metadata << "],\"token_pieces\":[";
  for (int i = 0; i < rows; ++i) {
    ASSIGN_OR_RETURN(auto piece,
                     detokenizer->Decode(encoded.span().subspan(i, 1)));
    if (i)
      metadata << ',';
    metadata << JsonQuote(piece);
  }
  metadata
      << "],\"prompt_rows\":" << rows << ",\"context_length\":" << kContext
      << ",\"pad_token_id\":" << padding
      << ",\"alternate_pad_token_id\":" << alternate_padding
      << ",\"no_bos\":true,\"autoregressive\":false"
      << ",\"vocab_size\":" << kVocabulary
      << ",\"padded_vocab_size\":" << kPaddedVocabulary
      << ",\"checkpoint_directory\":" << JsonQuote(checkpoint.string())
      << ",\"tokenizer_directory\":" << JsonQuote(tokenizer_path.string())
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
         "\"all_exported_activations_finite\":true}"
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
  const auto remaining = absl::ParseCommandLine(argc, argv);
  if (remaining.size() != 1) {
    std::cerr << "Unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::weight_analysis::Run();
  if (status.ok())
    return 0;
  std::cerr << status << '\n';
  return 1;
}
