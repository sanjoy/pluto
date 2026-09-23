// Conditional robustness of a memorized checkpoint to coarse branch weights.
// This is not a dataset-only constructor or a measurement of information
// entropy: all unmodified parameters retain their learned values.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/mlp_probe.h"
#include "src/llm/experiments/one_shot_memorizer/quantization.h"
#include "src/llm/experiments/one_shot_memorizer/sentence_ablation.h"
#include "src/llm/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Exact memorized checkpoint directory");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Full corpus");
ABSL_FLAG(std::string, output_dir, "", "Fresh directory for capacity reports");
ABSL_FLAG(int, batch_size, 32, "Sentences per evaluation batch");
ABSL_FLAG(int, prompt_tokens, 5, "Supplied prefix length");
ABSL_FLAG(int, expected_sentences, 1024,
          "Reject a corpus with a different sentence count");
ABSL_FLAG(int, layers, 8, "Transformer block count");
ABSL_FLAG(int, model_width, 16, "Residual width");
ABSL_FLAG(int, attention_heads, 1, "Attention head count");
ABSL_FLAG(int, feed_forward_width, 64, "MLP expansion width");
ABSL_FLAG(std::vector<std::string>, bits,
          (std::vector<std::string>{"8", "4", "2"}),
          "Per-tensor symmetric signed quantizer widths, comma-separated");
ABSL_FLAG(bool, joint_branches, false,
          "Quantize all transformer attention/MLP branches together instead "
          "of individually; leaves embeddings and final norm unchanged");

namespace pluto::llm::one_shot_memorizer {
namespace {

namespace fs = std::filesystem;
using HostTensor = cuda::PageLockedHostArray<float>;

struct Group {
  std::string name;
  int block = -1;
  std::string branch;
  std::vector<size_t> tensors;
  size_t parameters = 0;
};

struct Result {
  std::string name;
  int block = -1;
  std::string branch;
  int bits = 0;
  size_t tensors = 0;
  size_t parameters = 0;
  size_t changed_master_values = 0;
  uint64_t scale_bits = 0;
  uint64_t nominal_group_bits = 0;
  uint64_t nominal_model_bits = 0;
  MlpEvaluation teacher;
  MlpGreedyEvaluation greedy;
  double seconds = 0;
};

std::string EscapeHtml(const std::string& text) {
  std::string out;
  for (char ch : text)
    switch (ch) {
      case '&':
        out += "&amp;";
        break;
      case '<':
        out += "&lt;";
        break;
      case '>':
        out += "&gt;";
        break;
      case '"':
        out += "&quot;";
        break;
      case '\'':
        out += "&#39;";
        break;
      default:
        out += ch;
    }
  return out;
}

std::string Shape(const TensorSpec& tensor) {
  std::string out;
  for (size_t dimension : tensor.shape) {
    if (!out.empty())
      out += 'x';
    out += std::to_string(dimension);
  }
  return out;
}

bool HasBf16OperandWeights(size_t tensor, int blocks) {
  if (tensor == 0)
    return true;  // Tied input/output token embedding.
  if (tensor < 2 || tensor >= 2 + static_cast<size_t>(blocks) * 12)
    return false;
  const size_t within_block = (tensor - 2) % 12;
  return within_block == 2 || within_block == 4 || within_block == 8 ||
         within_block == 10;
}

absl::StatusOr<std::vector<Buffer>> UniqueWeights(
    cuda::Executor& executor, Layer& model,
    absl::Span<const TensorSpec> layout) {
  // The tied language-model head returns the same allocation as the token
  // embedding. Preserve first-occurrence order, exactly like checkpoint I/O.
  absl::flat_hash_set<const void*> seen;
  std::vector<Buffer> result;
  for (const Buffer& weight : model.weights()) {
    if (&weight.executor() != &executor)
      return absl::InvalidArgumentError(
          "capacity probe weight executor mismatch");
    if (seen.insert(weight.data()).second)
      result.push_back(weight);
  }
  if (result.size() != layout.size())
    return absl::InvalidArgumentError(
        "unique model weights differ from GPT-2 inventory");
  for (size_t tensor = 0; tensor < result.size(); ++tensor)
    if (result[tensor].size_bytes() !=
        layout[tensor].element_count * sizeof(float))
      return absl::InvalidArgumentError(
          absl::StrCat("checkpoint tensor shape mismatch at ", tensor));
  return result;
}

absl::StatusOr<std::vector<HostTensor>> CopyD2H(
    cuda::Executor& executor, absl::Span<const Buffer> weights) {
  std::vector<HostTensor> result;
  result.reserve(weights.size());
  for (const Buffer& weight : weights) {
    ASSIGN_OR_RETURN(
        auto host,
        HostTensor::Allocate(executor, weight.size_bytes() / sizeof(float)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "read original capacity-probe weight"));
    result.push_back(std::move(host));
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return result;
}

absl::Status Restore(cuda::Executor& executor,
                     absl::Span<const HostTensor> originals,
                     absl::Span<Buffer> weights) {
  if (originals.size() != weights.size())
    return absl::InternalError("capacity restore tensor count changed");
  for (size_t tensor = 0; tensor < weights.size(); ++tensor) {
    if (originals[tensor].size_bytes() != weights[tensor].size_bytes())
      return absl::InternalError("capacity restore tensor shape changed");
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(weights[tensor].data(), originals[tensor].data(),
                        originals[tensor].size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "restore every original capacity-probe weight"));
  }
  return absl::OkStatus();
}

absl::Status Upload(cuda::Executor& executor, absl::Span<const float> values,
                    Buffer& destination) {
  if (destination.size_bytes() != values.size() * sizeof(float))
    return absl::InternalError("capacity intervention tensor shape changed");
  ASSIGN_OR_RETURN(auto host, HostTensor::CopyFrom(executor, values));
  return cuda::CudaStatus(
      cudaMemcpyAsync(destination.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload quantized capacity-probe weight");
}

absl::StatusOr<std::vector<Group>> MakeGroups(
    absl::Span<const TensorSpec> layout, int blocks) {
  std::vector<Group> groups;
  for (int block = 0; block < blocks; ++block)
    for (const char* branch : {"attention", "mlp"}) {
      Group group{.name = absl::StrCat("block_", block, "_", branch),
                  .block = block,
                  .branch = branch};
      const std::string prefix =
          absl::StrCat("transformer_block_", block, ".", branch, ".");
      for (size_t tensor = 0; tensor < layout.size(); ++tensor)
        if (layout[tensor].name.starts_with(prefix)) {
          group.tensors.push_back(tensor);
          group.parameters += layout[tensor].element_count;
        }
      if (group.tensors.size() != 6)
        return absl::InternalError(
            "capacity branch inventory must contain six tensors");
      groups.push_back(std::move(group));
    }
  return groups;
}

absl::Status WriteHtml(const fs::path& directory, const fs::path& checkpoint,
                       const std::vector<Result>& results, size_t parameters,
                       bool complete) {
  std::ofstream out(directory / "report.html", std::ios::trunc);
  if (!out)
    return absl::UnknownError("cannot open capacity HTML report");
  out << std::setprecision(8)
      << "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
         "<meta name=\"viewport\" "
         "content=\"width=device-width,initial-scale=1\">"
         "<title>Checkpoint branch quantization</title><style>"
         "body{font:15px system-ui,sans-serif;margin:2rem;max-width:1500px}"
         "table{border-collapse:collapse;font-variant-numeric:tabular-nums}"
         "td,th{border:1px solid #ccc;padding:.4rem;text-align:right}"
         "td:first-child,th:first-child{text-align:left}thead{background:#eee}"
         "</style><body><h1>Checkpoint branch quantization</h1><p>Checkpoint: "
         "<code>"
      << EscapeHtml(checkpoint.string()) << "</code></p><p>"
      << (complete ? "Complete."
                   : "Partial: additional conditions are still pending.")
      << " Unique FP32 master parameters: " << parameters
      << ".</p>"
         "<p>Each condition starts by restoring <strong>all original "
         "weights</strong>. "
         "Only the named branch or joint group is quantized, including its "
         "LayerNorm and "
         "biases. "
         "Individual and joint groups are explicitly named; cases are never "
         "cumulative across evaluations. "
         "A final restored control checks that the original model remains "
         "intact.</p>"
         "<p>Uniform symmetric quantization uses one double-precision scale "
         "per tensor "
         "and integer levels from -(2^(bits-1)-1) through +(2^(bits-1)-1). "
         "Ties round "
         "away from zero. GPU inference still uses dequantized FP32 master "
         "weights "
         "and the model's usual BF16 arithmetic. Nominal coding budgets "
         "include 64 "
         "bits per scale, even for all-zero tensors. They are "
         "<strong>not</strong> "
         "serialized sizes, entropy estimates, fact ownership, or a joint "
         "compression "
         "guarantee. Unchanged masters are counted as 32 bits each.</p>"
         "<p>The BF16-master control only rounds token embeddings and dense "
         "matrices "
         "already used as BF16 operands; position embeddings, norms and biases "
         "stay "
         "FP32. Teacher-forced accuracy scores gold prefixes. Greedy exactness "
         "independently feeds predictions back through the full suffix and "
         "EOS.</p>"
         "<p>The zero-key-bias control removes only the K slice of each Q/K/V "
         "projection bias. In real arithmetic its score contribution is "
         "constant across keys for each query and cancels in softmax; BF16 "
         "rounding of K can break that invariance. This is a budget-neutral "
         "control: zero coded parameters are reported rather than assuming "
         "a compression scheme; actual changed masters are still counted.</p>"
         "<table><thead><tr><th>Condition</th><th>Bits</th><th>Changed "
         "masters</th>"
         "<th>Group parameters</th><th>Scale bits</th><th>Nominal group "
         "bits</th>"
         "<th>Nominal model bytes</th><th>Correct targets</th><th>Teacher "
         "exact</th>"
         "<th>Greedy exact</th><th>Seconds</th></tr></thead><tbody>";
  for (const auto& result : results)
    out << "<tr><td>" << EscapeHtml(result.name) << "</td><td>" << result.bits
        << "</td><td>" << result.changed_master_values << "</td><td>"
        << result.parameters << "</td><td>" << result.scale_bits << "</td><td>"
        << result.nominal_group_bits << "</td><td>"
        << static_cast<double>(result.nominal_model_bits) / 8 << "</td><td>"
        << result.teacher.correct_targets << '/' << result.teacher.targets
        << "</td><td>" << result.teacher.exact_sentences << '/'
        << result.teacher.sentences << "</td><td>"
        << result.greedy.exact_sentences << '/' << result.greedy.sentences
        << "</td><td>" << result.seconds << "</td></tr>\n";
  out << "</tbody></table><p>See conditions.tsv, per_sentence.tsv, "
         "inventory.tsv "
         "and quantization_scales.tsv for machine-readable "
         "details.</p></body></html>\n";
  out.close();
  if (!out)
    return absl::UnknownError("cannot write capacity HTML report");
  return absl::OkStatus();
}

absl::Status Run() {
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path directory = absl::GetFlag(FLAGS_output_dir);
  const int blocks = absl::GetFlag(FLAGS_layers);
  const int width = absl::GetFlag(FLAGS_model_width);
  const int features = absl::GetFlag(FLAGS_feed_forward_width);
  if (checkpoint.empty() || directory.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() || blocks <= 0 ||
      absl::GetFlag(FLAGS_expected_sentences) <= 0)
    return absl::InvalidArgumentError(
        "checkpoint, tokenizer, fresh output_dir and positive dimensions "
        "required");
  std::vector<int> widths;
  for (const std::string& text : absl::GetFlag(FLAGS_bits)) {
    int bits;
    if (!absl::SimpleAtoi(text, &bits) || bits < 2 || bits > 16 ||
        std::find(widths.begin(), widths.end(), bits) != widths.end())
      return absl::InvalidArgumentError(
          "bits must be distinct integers in [2,16]");
    widths.push_back(bits);
  }
  if (widths.empty())
    return absl::InvalidArgumentError("at least one precision is required");
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto tokenizer,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint / "compact_vocabulary.tsv"));
  if (tokenizer->original_eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError("tokenizer EOS differs from checkpoint");
  const Gpt2Config config{
      .transformer_block_count = blocks,
      .model_width = width,
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = features,
      .vocabulary_size = tokenizer->vocab_size(),
      .pad_vocabulary = false};
  RETURN_IF_ERROR(config.Validate());
  ASSIGN_OR_RETURN(auto layout, BuildGpt2ParameterLayout(
                                    {.vocabulary_size = config.vocabulary_size,
                                     .model_width = width,
                                     .feed_forward_width = features,
                                     .context_length = kGpt2ContextLength,
                                     .transformer_block_count = blocks}));
  ASSIGN_OR_RETURN(auto groups, MakeGroups(layout, blocks));
  if (absl::GetFlag(FLAGS_joint_branches)) {
    Group joint{.name = "all_transformer_branches", .branch = "joint"};
    for (const auto& group : groups) {
      joint.parameters += group.parameters;
      joint.tensors.insert(joint.tensors.end(), group.tensors.begin(),
                           group.tensors.end());
    }
    groups = {std::move(joint)};
  }
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto dataset,
                   PaddedLineDataSetIterator::Create(
                       *executor, corpus.text(), *tokenizer,
                       {.batch_size = absl::GetFlag(FLAGS_batch_size),
                        .context_length = kGpt2ContextLength,
                        .prompt_tokens = absl::GetFlag(FLAGS_prompt_tokens),
                        .eos_token = tokenizer->eos_token_id(),
                        .shuffle = false}));
  if (dataset->sample_count() !=
      static_cast<size_t>(absl::GetFlag(FLAGS_expected_sentences)))
    return absl::InvalidArgumentError(
        "corpus sentence count differs from expected_sentences");
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  ASSIGN_OR_RETURN(auto weights, UniqueWeights(*executor, *model, layout));
  ASSIGN_OR_RETURN(auto originals, CopyD2H(*executor, weights));
  std::vector<absl::Span<const float>> original_views;
  size_t parameters = 0;
  for (const auto& tensor : originals) {
    original_views.push_back(tensor.span());
    parameters += tensor.size();
  }
  RETURN_IF_ERROR(ValidateParameterValues(layout, original_views));
  if (parameters > std::numeric_limits<uint64_t>::max() / 32)
    return absl::OutOfRangeError("nominal parameter bit count overflows");
  const uint64_t full_master_bits = static_cast<uint64_t>(parameters) * 32;
  std::error_code error;
  if (!fs::create_directory(directory, error))
    return absl::InvalidArgumentError(
        absl::StrCat("output_dir must be fresh: ", error.message()));
  std::ofstream conditions(directory / "conditions.tsv");
  std::ofstream sentences(directory / "per_sentence.tsv");
  std::ofstream inventory(directory / "inventory.tsv");
  std::ofstream scales(directory / "quantization_scales.tsv");
  if (!conditions || !sentences || !inventory || !scales)
    return absl::UnknownError("cannot create capacity report files");
  for (std::ofstream* file : {&conditions, &sentences, &inventory, &scales})
    *file << std::setprecision(std::numeric_limits<double>::max_digits10);
  conditions << "# checkpoint=" << checkpoint.string()
             << "\n# corpus=" << absl::GetFlag(FLAGS_corpus)
             << "\n# tokenizer=" << absl::GetFlag(FLAGS_tokenizer)
             << "\n# unique_tensors=" << weights.size()
             << "\n# parameters=" << parameters << "\n# model_width=" << width
             << "\n# layers=" << blocks << "\n# feed_forward_width=" << features
             << "\n# attention_heads=" << config.attention_heads
             << "\n# vocabulary=" << config.vocabulary_size
             << "\n# context_length=" << kGpt2ContextLength
             << "\n# batch_size=" << dataset->options().batch_size
             << "\n# prompt_tokens=" << dataset->options().prompt_tokens
             << "\n# joint_branches=" << absl::GetFlag(FLAGS_joint_branches)
             << "\n# Each branch intervention starts from all original "
                "weights. No training."
                "\n# Nominal budgets count integer codes plus one 64-bit scale "
                "per tensor;"
                " not actual packed size or entropy. Other weights stay FP32 "
                "masters.\n"
                "condition\tblock\tbranch\tbits\ttensors\tparameters\tchanged_"
                "master_values"
                "\tquantized_level_bits\tscale_bits\tnominal_group_"
                "bits\tnominal_model_bits"
                "\ttargets\tcorrect_targets\tsentences\tteacher_exact\tgreedy_"
                "exact\tseconds\n";
  sentences << "condition\tsentence_index\ttargets\tcorrect_targets\tteacher_"
               "exact\tgreedy_exact\n";
  inventory << "checkpoint_index\tname\tshape\tparameters\tfp32_master_"
               "bits\tbf16_operand_weights\n";
  for (size_t tensor = 0; tensor < layout.size(); ++tensor)
    inventory << tensor << '\t' << layout[tensor].name << '\t'
              << Shape(layout[tensor]) << '\t' << layout[tensor].element_count
              << '\t'
              << static_cast<uint64_t>(layout[tensor].element_count) * 32
              << '\t' << HasBf16OperandWeights(tensor, blocks) << '\n';
  scales << "condition\tcheckpoint_index\ttensor\tbits\tparameters\tmaximum_"
            "code\tscale_double\tscale_bits\n";
  inventory.flush();
  std::vector<Result> results;
  auto evaluate = [&](Result result) -> absl::Status {
    const auto started = std::chrono::steady_clock::now();
    std::cout << "Evaluating " << result.name << std::endl;
    ASSIGN_OR_RETURN(result.teacher,
                     EvaluateMlpReplacements(*executor, *model, *dataset, {},
                                             config.vocabulary_size));
    ASSIGN_OR_RETURN(result.greedy,
                     VerifyMlpGreedyCompletions(*executor, *model, *dataset, {},
                                                config.vocabulary_size));
    result.seconds = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - started)
                         .count();
    if (result.teacher.sentences !=
            static_cast<int64_t>(dataset->sample_count()) ||
        result.greedy.sentences != result.teacher.sentences ||
        result.teacher.targets != dataset->supervised_row_count() ||
        result.teacher.targets_per_sentence.size() != dataset->sample_count() ||
        result.teacher.correct_per_sentence.size() != dataset->sample_count() ||
        result.greedy.exact_per_sentence.size() != dataset->sample_count())
      return absl::InternalError("capacity evaluation cardinalities changed");
    for (size_t sentence = 0; sentence < dataset->sample_count(); ++sentence) {
      const bool teacher_exact =
          result.teacher.targets_per_sentence[sentence] ==
          result.teacher.correct_per_sentence[sentence];
      if (teacher_exact != result.greedy.exact_per_sentence[sentence])
        return absl::InternalError(
            "teacher-forced/greedy per-sentence exactness disagrees");
      sentences << result.name << '\t' << sentence << '\t'
                << result.teacher.targets_per_sentence[sentence] << '\t'
                << result.teacher.correct_per_sentence[sentence] << '\t'
                << teacher_exact << '\t'
                << result.greedy.exact_per_sentence[sentence] << '\n';
    }
    const uint64_t level_bits =
        static_cast<uint64_t>(result.parameters) * result.bits;
    result.nominal_group_bits = level_bits + result.scale_bits;
    result.nominal_model_bits = full_master_bits -
                                static_cast<uint64_t>(result.parameters) * 32 +
                                result.nominal_group_bits;
    conditions << result.name << '\t' << result.block << '\t' << result.branch
               << '\t' << result.bits << '\t' << result.tensors << '\t'
               << result.parameters << '\t' << result.changed_master_values
               << '\t' << level_bits << '\t' << result.scale_bits << '\t'
               << result.nominal_group_bits << '\t' << result.nominal_model_bits
               << '\t' << result.teacher.targets << '\t'
               << result.teacher.correct_targets << '\t'
               << result.teacher.sentences << '\t'
               << result.teacher.exact_sentences << '\t'
               << result.greedy.exact_sentences << '\t' << result.seconds
               << '\n';
    conditions.flush();
    sentences.flush();
    scales.flush();
    if (!conditions || !sentences || !scales)
      return absl::UnknownError("cannot write capacity evaluation results");
    std::cout << result.name << " correct=" << result.teacher.correct_targets
              << '/' << result.teacher.targets
              << " greedy_exact=" << result.greedy.exact_sentences << '/'
              << result.greedy.sentences << " seconds=" << result.seconds
              << std::endl;
    results.push_back(std::move(result));
    return WriteHtml(directory, checkpoint, results, parameters, false);
  };
  RETURN_IF_ERROR(evaluate({.name = "original", .branch = "none"}));
  if (results.back().teacher.correct_targets !=
          results.back().teacher.targets ||
      results.back().greedy.exact_sentences != results.back().greedy.sentences)
    return absl::FailedPreconditionError(
        "checkpoint is not fully memorized on the full supplied corpus");

  RETURN_IF_ERROR(Restore(*executor, originals, absl::MakeSpan(weights)));
  Result bf16{.name = "baseline_bf16_matrix_masters",
              .branch = "bf16_operands",
              .bits = 16};
  for (size_t tensor = 0; tensor < layout.size(); ++tensor) {
    if (!HasBf16OperandWeights(tensor, blocks))
      continue;
    std::vector<float> rounded(originals[tensor].begin(),
                               originals[tensor].end());
    for (size_t index = 0; index < rounded.size(); ++index) {
      const uint32_t original_bits = std::bit_cast<uint32_t>(rounded[index]);
      uint32_t rounded_bits =
          original_bits + 0x7fffu + ((original_bits >> 16) & 1u);
      rounded[index] = std::bit_cast<float>(rounded_bits & 0xffff0000u);
      if (!std::isfinite(rounded[index]))
        return absl::OutOfRangeError("BF16 matrix-master control overflowed");
      bf16.changed_master_values +=
          original_bits != std::bit_cast<uint32_t>(rounded[index]);
    }
    ++bf16.tensors;
    bf16.parameters += rounded.size();
    RETURN_IF_ERROR(Upload(*executor, rounded, weights[tensor]));
  }
  RETURN_IF_ERROR(evaluate(std::move(bf16)));
  if (results.back().teacher.correct_per_sentence !=
          results.front().teacher.correct_per_sentence ||
      results.back().greedy.exact_per_sentence !=
          results.front().greedy.exact_per_sentence)
    return absl::FailedPreconditionError(
        "BF16 matrix-master control differs from the fully memorized baseline");

  RETURN_IF_ERROR(Restore(*executor, originals, absl::MakeSpan(weights)));
  Result key_biases{.name = "control_zero_key_biases",
                    .branch = "key_bias_control",
                    .bits = 32,
                    .tensors = static_cast<size_t>(blocks)};
  // For q_i dot (W_k x_j + b_k), q_i dot b_k is the same additive
  // score for every allowed key j and therefore cancels in exact softmax.
  // The actual kernel rounds K to BF16 before attention, however, so removing
  // b_k can change those rounded values. Measure rather than assert invariance.
  // parameters remains zero for budget accounting: this control does not
  // claim a particular packed representation or a reduction in storage size.
  for (int block = 0; block < blocks; ++block) {
    const size_t tensor = 2 + static_cast<size_t>(block) * 12 + 3;
    if (layout[tensor].shape.size() != 1 ||
        layout[tensor].qkv_width != static_cast<size_t>(width) ||
        originals[tensor].size() != static_cast<size_t>(width) * 3)
      return absl::InternalError(
          "invalid QKV bias inventory for key-bias control");
    std::vector<float> bias(originals[tensor].begin(), originals[tensor].end());
    for (int channel = 0; channel < width; ++channel) {
      const size_t index = static_cast<size_t>(width) + channel;
      key_biases.changed_master_values +=
          std::bit_cast<uint32_t>(bias[index]) != 0;
      bias[index] = 0.0f;
    }
    RETURN_IF_ERROR(Upload(*executor, bias, weights[tensor]));
  }
  conditions
      << "# control_zero_key_biases: budget-neutral control; parameters=0 "
         "is a coding-accounting sentinel, not a claim that no weights "
         "changed.\n";
  RETURN_IF_ERROR(evaluate(std::move(key_biases)));
  for (int bits : widths)
    for (const auto& group : groups) {
      // Restoring ALL tensors is essential: branch errors otherwise accumulate
      // and the result no longer measures an individual branch intervention.
      RETURN_IF_ERROR(Restore(*executor, originals, absl::MakeSpan(weights)));
      Result result{
          .name = absl::StrCat(group.name, "_", bits, "bit"),
          .block = group.block,
          .branch = group.branch,
          .bits = bits,
          .tensors = group.tensors.size(),
          .parameters = group.parameters,
          .scale_bits = static_cast<uint64_t>(group.tensors.size()) * 64};
      for (size_t tensor : group.tensors) {
        ASSIGN_OR_RETURN(auto quantized, QuantizeTensorSymmetric(
                                             originals[tensor].span(), bits));
        for (size_t index = 0; index < quantized.values.size(); ++index)
          result.changed_master_values +=
              std::bit_cast<uint32_t>(quantized.values[index]) !=
              std::bit_cast<uint32_t>(originals[tensor][index]);
        RETURN_IF_ERROR(Upload(*executor, quantized.values, weights[tensor]));
        scales << result.name << '\t' << tensor << '\t' << layout[tensor].name
               << '\t' << bits << '\t' << layout[tensor].element_count << '\t'
               << quantized.maximum_code << '\t' << quantized.scale << "\t64\n";
      }
      RETURN_IF_ERROR(evaluate(std::move(result)));
    }
  RETURN_IF_ERROR(Restore(*executor, originals, absl::MakeSpan(weights)));
  ASSIGN_OR_RETURN(auto restored, CopyD2H(*executor, weights));
  for (size_t tensor = 0; tensor < originals.size(); ++tensor)
    if (std::memcmp(originals[tensor].data(), restored[tensor].data(),
                    originals[tensor].size_bytes()) != 0)
      return absl::InternalError(
          "final original-weight restoration was not bit-identical");
  RETURN_IF_ERROR(evaluate({.name = "restored_original", .branch = "none"}));
  if (results.back().teacher.correct_per_sentence !=
          results.front().teacher.correct_per_sentence ||
      results.back().greedy.exact_per_sentence !=
          results.front().greedy.exact_per_sentence)
    return absl::InternalError(
        "restored capacity control differs from original baseline");
  RETURN_IF_ERROR(WriteHtml(directory, checkpoint, results, parameters, true));
  conditions << "# Complete; restored all original parameter bytes and "
                "reverified baseline.\n";
  for (std::ofstream* file : {&conditions, &sentences, &inventory, &scales}) {
    file->close();
    if (!*file)
      return absl::UnknownError("cannot close capacity report files");
  }
  return absl::OkStatus();
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
