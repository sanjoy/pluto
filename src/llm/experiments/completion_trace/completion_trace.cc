// Read-only, genuine autoregressive traces of a small selection of facts.
// Corpus continuations are used for scoring only, never as model inputs.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/completion_trace/report.h"
#include "src/llm/experiments/completion_trace/trace.h"
#include "src/llm/generate_greedy_continuation.h"
#include "src/llm/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Required compact-vocabulary checkpoint");
ABSL_FLAG(std::string, tokenizer, "",
          "Required original GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Fact corpus");
ABSL_FLAG(std::string, output_dir, "", "Required NEW report directory");
ABSL_FLAG(
    std::vector<std::string>, lines,
    (std::vector<std::string>{"80", "406", "411", "631", "1"}),
    "One-based corpus line numbers: France, Greece, Peru, durian, mammals");
ABSL_FLAG(int, prompt_tokens, 5, "Tokens supplied from each fact");
ABSL_FLAG(int, generation_tokens, 5, "New greedy tokens, stopping at EOS");
ABSL_FLAG(int, layers, 8, "Transformer block count");
ABSL_FLAG(int, model_width, 16, "Residual/embedding width");
ABSL_FLAG(int, attention_heads, 1, "Attention heads per block");
ABSL_FLAG(int, feed_forward_width, 64, "MLP inner width");
ABSL_FLAG(
    std::string, source_revision, "unspecified",
    "Source revision/provenance recorded verbatim (e.g. Git hash + dirty)");

namespace pluto::llm::completion_trace {
namespace {

// Snapshot unique FP32 master tensors, preserving checkpoint order and ties.
// Pinned staging is required even for these small read-only transfers.
absl::StatusOr<std::vector<std::vector<float>>> CopyWeights(
    cuda::Executor& executor, const Layer& model) {
  std::vector<std::vector<float>> result;
  absl::flat_hash_set<const void*> seen;
  for (const auto& weight : model.weights()) {
    if (!seen.insert(weight.data()).second)
      continue;
    if (weight.size_bytes() % sizeof(float) != 0)
      return absl::DataLossError("non-FP32 master weight byte count");
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<float>::Allocate(
                         executor, weight.size_bytes() / sizeof(float)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), weight.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "copy trace parameters"));
    RETURN_IF_ERROR(executor.Synchronize());
    result.emplace_back(host.begin(), host.end());
  }
  return result;
}

// A readable parameter index makes the projected vectors independently
// inspectable. The model recipe defines this order; verify each tensor's size
// before assigning a label so a future recipe edit cannot silently mislabel it.
absl::Status WriteParameters(const std::filesystem::path& path,
                             const Gpt2Config& config,
                             const std::vector<std::vector<float>>& weights) {
  struct WeightShape {
    std::string name;
    int rows;
    int columns;
  };
  const int d = config.model_width;
  const int f = config.feed_forward_width;
  std::vector<WeightShape> shapes = {
      {"token_embedding (also tied language-model head)",
       config.vocabulary_size, d},
      {"position_embedding", kGpt2ContextLength, d}};
  for (int block = 0; block < config.transformer_block_count; ++block) {
    const std::string b = absl::StrCat("block_", block, "/");
    shapes.insert(shapes.end(),
                  {{b + "attention_norm/gamma", 1, d},
                   {b + "attention_norm/beta", 1, d},
                   {b + "QKV/W (Q then K then V columns)", d, 3 * d},
                   {b + "QKV/b", 1, 3 * d},
                   {b + "attention_output/W", d, d},
                   {b + "attention_output/b", 1, d},
                   {b + "mlp_norm/gamma", 1, d},
                   {b + "mlp_norm/beta", 1, d},
                   {b + "mlp_expand/W", d, f},
                   {b + "mlp_expand/b", 1, f},
                   {b + "mlp_contract/W", f, d},
                   {b + "mlp_contract/b", 1, d}});
  }
  shapes.push_back({"final_norm/gamma", 1, d});
  shapes.push_back({"final_norm/beta", 1, d});
  if (weights.size() != shapes.size())
    return absl::DataLossError(
        "recipe weight count changed; update trace labels");
  for (size_t i = 0; i < shapes.size(); ++i)
    if (weights[i].size() !=
        static_cast<size_t>(shapes[i].rows) * shapes[i].columns)
      return absl::DataLossError(
          absl::StrCat("unexpected weight shape at ", i));
  std::ofstream out(path);
  out << std::setprecision(std::numeric_limits<float>::max_digits10);
  out << "FP32 master weights, checkpoint order. Matrices are row-major xW. "
         "Matrix operands and token-embedding outputs are cast to BF16; "
         "position additions, biases and LayerNorm parameters participate in "
         "FP32 calculations before their BF16 output rounding. The tied LM "
         "head uses the same embedding master tensor (BF16 matrix operands, "
         "FP32 accumulated logits).\n";
  for (size_t i = 0; i < weights.size(); ++i) {
    out << "\nweight_" << i << ".bin " << shapes[i].name << " ["
        << shapes[i].rows << "," << shapes[i].columns << "]\n";
    for (int row = 0; row < shapes[i].rows; ++row) {
      out << "row " << row << ":";
      for (int col = 0; col < shapes[i].columns; ++col)
        out << ' '
            << weights[i][static_cast<size_t>(row) * shapes[i].columns + col];
      out << '\n';
    }
  }
  out.close();
  return out ? absl::OkStatus()
             : absl::UnknownError("writing parameters failed");
}

bool EqualBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Autoregression must not revise a previous position's causal activations.
// Compare original bytes, including both row axes of each attention matrix.
absl::Status CheckEarlierPositions(const ForwardTrace& before,
                                   const ForwardTrace& after) {
  if (after.prefix.size() != before.prefix.size() + 1 ||
      !std::equal(before.prefix.begin(), before.prefix.end(),
                  after.prefix.begin()))
    return absl::DataLossError("trace prefixes are not autoregressive");
  auto check = [&](const std::vector<TensorSnapshot>& old,
                   const std::vector<TensorSnapshot>& next, bool attention) {
    if (old.size() != next.size())
      return absl::DataLossError("layer count changed between prefixes");
    for (size_t i = 0; i < old.size(); ++i) {
      const auto& a = old[i];
      const auto& b = next[i];
      if (a.scope != b.scope || a.name != b.name ||
          a.occurrence != b.occurrence || a.output_index != b.output_index ||
          a.data_type != b.data_type)
        return absl::DataLossError("layer identities changed between prefixes");
      if (!attention) {
        if (a.raw_bytes.size() > b.raw_bytes.size() ||
            !std::equal(a.raw_bytes.begin(), a.raw_bytes.end(),
                        b.raw_bytes.begin()))
          return absl::DataLossError(absl::StrCat(
              "earlier activation changed: ", a.scope, "/", a.name));
      } else {
        const size_t p = before.prefix.size();
        const size_t q = after.prefix.size();
        for (int64_t h = 0; h < a.dimensions[1]; ++h)
          for (size_t row = 0; row < p; ++row)
            if (std::memcmp(a.raw_bytes.data() + (h * p * p + row * p) * 4,
                            b.raw_bytes.data() + (h * q * q + row * q) * 4,
                            p * 4) != 0)
              return absl::DataLossError(
                  "earlier attention probabilities changed");
      }
    }
    return absl::OkStatus();
  };
  RETURN_IF_ERROR(check(before.activations, after.activations, false));
  return check(before.attention, after.attention, true);
}

absl::Status Run() {
  const std::string checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const std::string tokenizer_path = absl::GetFlag(FLAGS_tokenizer);
  const std::string corpus_path = absl::GetFlag(FLAGS_corpus);
  const std::filesystem::path directory = absl::GetFlag(FLAGS_output_dir);
  const int prompt_count = absl::GetFlag(FLAGS_prompt_tokens);
  const int new_count = absl::GetFlag(FLAGS_generation_tokens);
  if (checkpoint.empty() || tokenizer_path.empty() || directory.empty())
    return absl::InvalidArgumentError(
        "--checkpoint, --tokenizer, --output_dir required");
  if (prompt_count <= 0 || new_count <= 0 ||
      int64_t{prompt_count} + new_count > kGpt2ContextLength)
    return absl::InvalidArgumentError(
        "positive prompt/generation counts must fit context");
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(corpus_path));
  std::vector<absl::string_view> lines = absl::StrSplit(corpus.text(), '\n');
  if (!lines.empty() && lines.back().empty())
    lines.pop_back();
  std::vector<int> selected;
  absl::flat_hash_set<int> seen;
  for (const auto& text : absl::GetFlag(FLAGS_lines)) {
    int n = 0;
    if (!absl::SimpleAtoi(text, &n) || n < 1 ||
        static_cast<size_t>(n) > lines.size() || !seen.insert(n).second)
      return absl::InvalidArgumentError(
          "--lines must be distinct existing positive line numbers");
    selected.push_back(n);
  }
  if (selected.empty())
    return absl::InvalidArgumentError("at least one corpus line is required");
  ASSIGN_OR_RETURN(auto base, tokenizer::Gpt2Tokenizer::Load(tokenizer_path));
  ASSIGN_OR_RETURN(
      auto tokenizer,
      tokenizer::CompactVocabularyTokenizer::LoadFromFile(
          *base, std::filesystem::path(checkpoint) / "compact_vocabulary.tsv"));
  ASSIGN_OR_RETURN(auto detokenizer,
                   tokenizer::Gpt2Detokenizer::Load(tokenizer_path));
  const int vocab = tokenizer->vocab_size();
  const int eos = tokenizer->eos_token_id();
  Gpt2Config config{
      .transformer_block_count = absl::GetFlag(FLAGS_layers),
      .model_width = absl::GetFlag(FLAGS_model_width),
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = vocab,
      .pad_vocabulary = false};
  RETURN_IF_ERROR(config.Validate());
  ReportMetadata metadata{
      .checkpoint = checkpoint,
      .tokenizer = tokenizer_path,
      .corpus = corpus_path,
      .source_revision = absl::GetFlag(FLAGS_source_revision),
      .model_width = config.model_width,
      .layers = config.transformer_block_count,
      .heads = config.attention_heads,
      .feed_forward_width = config.feed_forward_width,
      .eos_token = eos};
  const auto ids = tokenizer->original_token_ids();
  metadata.original_ids.assign(ids.begin(), ids.end());
  for (int id : ids) {
    ASSIGN_OR_RETURN(auto text, detokenizer->Decode({&id, 1}));
    metadata.token_text.push_back(std::move(text));
  }
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(ReadFromDirectory(*executor, *model, checkpoint, false));
  ASSIGN_OR_RETURN(auto initial_weights, CopyWeights(*executor, *model));
  std::error_code error;
  if (!std::filesystem::create_directory(directory, error))
    return absl::InvalidArgumentError(
        absl::StrCat("output directory must be new: ", directory.string(), " ",
                     error.message()));
  RETURN_IF_ERROR(
      WriteParameters(directory / "parameters.txt", config, initial_weights));
  // Absence of COMPLETE means interrupted/failed capture; never present a
  // partial directory as a verified report.
  std::ofstream controls(directory / "controls.tsv");
  controls << "check\tline\tstep\tresult\n";
  std::vector<CompletionExample> examples;
  int correct = 0;
  int decisions = 0;
  for (int line : selected) {
    absl::string_view sentence = lines[line - 1];
    if (!sentence.empty() && sentence.back() == '\r')
      sentence.remove_suffix(1);
    ASSIGN_OR_RETURN(auto gold, tokenizer->Encode(*executor, sentence));
    if (gold.size() < static_cast<size_t>(prompt_count))
      return absl::InvalidArgumentError("selected fact is shorter than prompt");
    CompletionExample example{.corpus_line = line,
                              .sentence = std::string(sentence)};
    example.prompt.assign(gold.begin(), gold.begin() + prompt_count);
    std::vector<int> prefix = example.prompt;
    std::vector<int> generated;
    for (int step = 0; step < new_count; ++step) {
      ASSIGN_OR_RETURN(auto plain_before,
                       PredictNextLogits(*executor, *model, prefix, eos, vocab));
      ASSIGN_OR_RETURN(auto forward,
                       TraceForward(*executor, *model, prefix, eos, vocab));
      ASSIGN_OR_RETURN(auto plain_after,
                       PredictNextLogits(*executor, *model, prefix, eos, vocab));
      if (!EqualBits(plain_before, forward.next_logits) ||
          !EqualBits(plain_after, forward.next_logits))
        return absl::DataLossError("hooks changed logical-vocabulary logits");
      controls << "all_logit_bits_plain_before_and_after\t" << line << '\t'
               << step + 1 << "\tPASS\n";
      if (!example.steps.empty()) {
        RETURN_IF_ERROR(
            CheckEarlierPositions(example.steps.back().forward, forward));
        controls << "all_earlier_activation_and_attention_bits\t" << line
                 << '\t' << step + 1 << "\tPASS\n";
      }
      const int predicted = std::max_element(forward.next_logits.begin(),
                                             forward.next_logits.end()) -
                            forward.next_logits.begin();
      const size_t target_position = prefix.size();
      const int expected = target_position < gold.size() ? gold[target_position]
                           : target_position == gold.size() ? eos
                                                            : -1;
      correct += predicted == expected;
      ++decisions;
      example.steps.push_back({std::move(forward), predicted, expected});
      if (predicted == eos)
        break;
      prefix.push_back(predicted);
      generated.push_back(predicted);
    }
    ASSIGN_OR_RETURN(auto production, GenerateGreedyContinuation(
                                          *executor, *model, example.prompt,
                                          vocab, eos, new_count));
    if (production.size() != generated.size() ||
        !std::equal(generated.begin(), generated.end(), production.begin()))
      return absl::DataLossError(
          "trace generation differs from production generator");
    controls << "production_generation_ids\t" << line << "\tall\tPASS\n";
    std::string prompt_text, generated_text;
    for (int id : example.prompt)
      prompt_text += metadata.token_text[id];
    for (int id : generated)
      generated_text += metadata.token_text[id];
    std::cout << "line " << line << ": \"" << absl::CEscape(prompt_text)
              << "\" -> \"" << absl::CEscape(generated_text) << "\"\n"
              << std::flush;
    examples.push_back(std::move(example));
  }
  ASSIGN_OR_RETURN(auto final_weights, CopyWeights(*executor, *model));
  for (size_t i = 0; i < initial_weights.size(); ++i)
    if (!EqualBits(initial_weights[i], final_weights[i]))
      return absl::DataLossError("trace changed model parameters");
  controls << "all_unique_weight_bytes_unchanged\tall\tall\tPASS\n";
  controls.close();
  if (!controls)
    return absl::UnknownError("writing controls failed");
  RETURN_IF_ERROR(WriteReports(directory, metadata, examples));
  std::ofstream complete(directory / "COMPLETE");
  complete << "All capture controls passed. Corpus next-token matches: "
           << correct << '/' << decisions << "\n";
  complete.close();
  if (!complete)
    return absl::UnknownError("writing completion marker failed");
  std::cout << "Report: " << directory / "report.html"
            << "\nCorpus next-token matches: " << correct << '/' << decisions
            << '\n';
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::completion_trace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const auto status = pluto::llm::completion_trace::Run();
  if (status.ok())
    return 0;
  std::cerr << status << '\n';
  return 1;
}
