// Dataset-only numeric construction. Neural construction and inference run on
// CPU; the existing tokenizer still needs a CUDA Executor for pinned host
// output allocation. No checkpoint, teacher forward or optimizer is used.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_split.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/experiments/one_shot_memorizer/projected_relu_memory.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, mode, "", "compile or infer");
ABSL_FLAG(std::string, tokenizer, "", "Original GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "One sentence per line, compile mode only");
ABSL_FLAG(std::string, output_dir, "", "Fresh compile output directory");
ABSL_FLAG(std::string, model_file, "", "Saved numeric weights, infer only");
ABSL_FLAG(std::string, prompt, "",
          "Text prefix with at least five tokens, infer only");
ABSL_FLAG(int, max_new_tokens, 1024, "Positive inference limit including EOS");

namespace pluto::llm::one_shot_memorizer {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int kVerificationLimit = 1024;

absl::Status Write(const fs::path& path, absl::string_view bytes) {
  std::ofstream output(path, std::ios::binary);
  output.write(bytes.data(), bytes.size());
  output.close();
  if (!output)
    return absl::UnknownError("cannot write projected ReLU artifact");
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<int>>> TokenizeLines(
    cuda::Executor& executor, const tokenizer::Gpt2Tokenizer& tokenizer,
    absl::string_view text) {
  if (!text.empty() && text.back() == '\n')
    text.remove_suffix(1);
  std::vector<std::vector<int>> sentences;
  for (absl::string_view line : absl::StrSplit(text, '\n')) {
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    ASSIGN_OR_RETURN(auto ids, tokenizer.Encode(executor, line));
    if (ids.size() < kProjectedReluPrompt || ids.size() > kVerificationLimit)
      return absl::InvalidArgumentError(
          "each corpus sentence must contain 5..1024 tokens");
    sentences.emplace_back(ids.begin(), ids.end());
  }
  return sentences;
}

absl::Status Compile(cuda::Executor& executor,
                     const tokenizer::Gpt2Tokenizer& tokenizer) {
  const fs::path directory = absl::GetFlag(FLAGS_output_dir);
  if (directory.empty() || !absl::GetFlag(FLAGS_model_file).empty() ||
      !absl::GetFlag(FLAGS_prompt).empty())
    return absl::InvalidArgumentError(
        "compile requires output_dir and forbids model_file/prompt");
  std::error_code error;
  if (fs::exists(directory, error))
    return absl::AlreadyExistsError("output_dir must be fresh");
  if (error)
    return absl::UnknownError(error.message());
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto sentences,
                   TokenizeLines(executor, tokenizer, corpus.text()));
  const auto started = Clock::now();
  ASSIGN_OR_RETURN(auto model,
                   BuildProjectedReluMemory(sentences, tokenizer.vocab_size(),
                                            tokenizer.eos_token_id()));
  const double construction_seconds =
      std::chrono::duration<double>(Clock::now() - started).count();
  ASSIGN_OR_RETURN(auto bytes, SerializeProjectedReluMemory(model));
  if (!fs::create_directory(directory, error) || error)
    return absl::UnknownError("cannot create fresh output directory");
  const fs::path weight_path = directory / "projected_relu.weights";
  RETURN_IF_ERROR(Write(weight_path, bytes));
  // Read the actual saved artifact back from disk before any verification.
  ASSIGN_OR_RETURN(auto file, LoadTextCorpus(weight_path.string()));
  ASSIGN_OR_RETURN(auto loaded, DeserializeProjectedReluMemory(file.text()));
  ASSIGN_OR_RETURN(auto reserialized, SerializeProjectedReluMemory(loaded));
  if (reserialized != bytes)
    return absl::DataLossError(
        "saved projected ReLU weights did not round trip");
  const auto actual_bytes = fs::file_size(weight_path, error);
  if (error || actual_bytes != bytes.size())
    return absl::DataLossError("saved projected ReLU byte count disagrees");

  std::ofstream verification(directory / "verification.tsv");
  std::ofstream coefficients(directory / "projection.tsv");
  if (!verification || !coefficients)
    return absl::UnknownError("cannot create projected ReLU reports");
  verification << "line_1based\tinput_tokens\texpected_suffix_including_eos\t"
                  "generated_tokens\tterminated_eos\texact\n";
  const auto verifying = Clock::now();
  size_t exact = 0, targets = 0, input_tokens = 0;
  absl::flat_hash_set<int> active{tokenizer.eos_token_id()};
  for (size_t index = 0; index < sentences.size(); ++index) {
    const auto& sentence = sentences[index];
    input_tokens += sentence.size();
    active.insert(sentence.begin(), sentence.end());
    const auto prefix =
        absl::MakeConstSpan(sentence).first(kProjectedReluPrompt);
    std::vector<int> expected(sentence.begin() + kProjectedReluPrompt,
                              sentence.end());
    expected.push_back(tokenizer.eos_token_id());
    targets += expected.size();
    // The fixed cap is not supplied by the answer length. Only generated IDs
    // feed subsequent queries; expected IDs are consulted afterward to score.
    ASSIGN_OR_RETURN(auto generated, ProjectedReluGreedyContinuation(
                                         loaded, prefix, kVerificationLimit));
    const bool matches = generated == expected;
    exact += matches;
    verification << index + 1 << '\t' << sentence.size() << '\t'
                 << expected.size() << '\t' << generated.size() << '\t'
                 << (!generated.empty() &&
                     generated.back() == tokenizer.eos_token_id())
                 << '\t' << matches << '\n';
  }
  const double verification_seconds =
      std::chrono::duration<double>(Clock::now() - verifying).count();
  coefficients << "suffix_coordinate\tFP64_projection_weight\n"
               << std::setprecision(17);
  for (size_t j = 0; j < model.projection.size(); ++j)
    coefficients << j << '\t' << model.projection[j] << '\n';
  double maximum_hash = 0;
  double minimum_gap = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < loaded.units.size(); ++i) {
    maximum_hash =
        std::max(maximum_hash, std::abs(loaded.units[i].positive_bias));
    if (i > 0)
      minimum_gap =
          std::min(minimum_gap, loaded.units[i].positive_bias -
                                    loaded.units[i - 1].positive_bias);
  }
  const uint64_t units = loaded.units.size();
  const uint64_t stored_scalars = units * 3 + kProjectedReluWindow;
  std::ofstream summary(directory / "summary.tsv");
  if (!summary)
    return absl::UnknownError("cannot create projected ReLU summary");
  summary << std::setprecision(17) << "metric\tvalue\ncorpus\t"
          << absl::GetFlag(FLAGS_corpus) << "\ntokenizer\t"
          << absl::GetFlag(FLAGS_tokenizer)
          << "\nconstruction\tdataset/tokenizer only; no trained checkpoint"
             "\nexecution\tCPU neural arithmetic; tokenizer needs CUDA pinned "
             "allocation"
             "\noutput\tscalar original-token ID, not softmax or 16D activation"
             "\nunseen_contexts\tMAY ALIAS stored hashes; no full-key "
             "rejection guarantee"
             "\nprecision\tFP64 exact integers within validated bounds"
             "\nprompt_tokens\t5\ncontext_window\t9\nseed\t0"
             "\ncoefficient_min\t1\ncoefficient_max\t1048576"
             "\nmax_projection_attempts\t256\nprojection_attempts\t"
          << loaded.projection_attempts << "\nsentences\t" << sentences.size()
          << "\ninput_tokens\t" << input_tokens
          << "\nactive_tokens_including_eos\t" << active.size()
          << "\ntargets_including_eos\t" << targets
          << "\nexact_autoregressive_completions\t" << exact
          << "\ndistinct_compiled_contexts\t" << units << "\nfirst_relu_units\t"
          << units * 2 << "\nsecond_relu_units\t" << units
          << "\nstored_FP64_scalars\t" << stored_scalars
          << "\nstored_FP64_weight_bytes\t" << stored_scalars * sizeof(double)
          << "\nfixed_sparse_weights\t" << units * 4
          << "\nfixed_second_layer_biases\t" << units << "\nserialized_bytes\t"
          << actual_bytes << "\nmetadata_bytes\t"
          << actual_bytes - stored_scalars * sizeof(double)
          << "\nmaximum_absolute_stored_hash\t" << maximum_hash
          << "\nminimum_hash_gap\t" << (units > 1 ? minimum_gap : 0)
          << "\nconstruction_seconds\t" << construction_seconds
          << "\nverification_seconds\t" << verification_seconds
          << "\nartifact_round_trip\tPASS\ncompleted\t"
          << (exact == sentences.size() ? "true" : "false") << '\n';
  for (auto* stream : {&verification, &coefficients, &summary}) {
    stream->flush();
    if (!*stream)
      return absl::UnknownError("cannot finish projected ReLU reports");
  }
  std::cout << exact << '/' << sentences.size()
            << " exact suffix-plus-EOS completions; " << units << " units; "
            << actual_bytes << " artifact bytes.\n"
            << "Construction " << construction_seconds << "s; verification "
            << verification_seconds << "s. Artifacts: " << directory << '\n';
  if (exact != sentences.size())
    return absl::FailedPreconditionError(
        "projected ReLU corpus verification failed");
  return absl::OkStatus();
}

absl::Status Infer(cuda::Executor& executor,
                   const tokenizer::Gpt2Tokenizer& tokenizer) {
  if (absl::GetFlag(FLAGS_model_file).empty() ||
      absl::GetFlag(FLAGS_prompt).empty() ||
      !absl::GetFlag(FLAGS_output_dir).empty() ||
      absl::GetFlag(FLAGS_max_new_tokens) <= 0)
    return absl::InvalidArgumentError(
        "infer requires model_file/prompt/positive max_new_tokens; no "
        "output_dir");
  ASSIGN_OR_RETURN(auto file, LoadTextCorpus(absl::GetFlag(FLAGS_model_file)));
  ASSIGN_OR_RETURN(auto model, DeserializeProjectedReluMemory(file.text()));
  if (model.vocabulary_size != tokenizer.vocab_size() ||
      model.eos_token_id != tokenizer.eos_token_id())
    return absl::InvalidArgumentError("tokenizer vocabulary/EOS mismatch");
  const auto prompt = absl::GetFlag(FLAGS_prompt);
  ASSIGN_OR_RETURN(auto prefix, tokenizer.Encode(executor, prompt));
  ASSIGN_OR_RETURN(auto suffix, ProjectedReluGreedyContinuation(
                                    model, prefix.span(),
                                    absl::GetFlag(FLAGS_max_new_tokens)));
  if (!suffix.empty() && suffix.back() == model.eos_token_id)
    suffix.pop_back();
  ASSIGN_OR_RETURN(auto decoder, tokenizer::Gpt2Detokenizer::Load(
                                     absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto text, decoder->Decode(suffix));
  std::cout << prompt << text << '\n';
  return absl::OkStatus();
}

absl::Status Run() {
  const auto mode = absl::GetFlag(FLAGS_mode);
  if ((mode != "compile" && mode != "infer") ||
      absl::GetFlag(FLAGS_tokenizer).empty())
    return absl::InvalidArgumentError(
        "mode=compile|infer and tokenizer required");
  std::cerr << "Warning: unseen contexts can alias stored projected hashes; "
               "this is a CPU FP64 memorizer, not the learned GPT-2 model.\n";
  ASSIGN_OR_RETURN(auto tokenizer, tokenizer::Gpt2Tokenizer::Load(
                                       absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  return mode == "compile" ? Compile(*executor, *tokenizer)
                           : Infer(*executor, *tokenizer);
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::llm::one_shot_memorizer::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
