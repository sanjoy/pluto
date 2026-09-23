// Tests fixed nonlinear MLP bases after fitting only their output projections:
// initialization-time GELU directions, or a separately selected quadratic
// basis. Inputs and regression targets come from the learned model: this is a
// conditional construction, not a model constructed from text alone.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/closed_form_map.h"
#include "src/llm/experiments/one_shot_memorizer/frozen_mlp_basis.h"
#include "src/llm/experiments/one_shot_memorizer/mlp_probe.h"
#include "src/llm/experiments/one_shot_memorizer/quadratic_features.h"
#include "src/llm/gpt2.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Memorized step_16128 checkpoint");
ABSL_FLAG(std::string, experiment, "initialization",
          "initialization or quadratic; fixed separately chosen protocols");
ABSL_FLAG(std::string, initial_checkpoint, "",
          "Matching step_0 checkpoint; required only for initialization, "
          "disallowed for quadratic");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "The original 1,024-fact corpus");
ABSL_FLAG(std::string, output_dir, "", "Fresh directory for TSV reports");
ABSL_FLAG(int, batch_size, 32, "Sentences per evaluation batch (1 through 32)");

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr int kWidth = 16;
constexpr int kFeatures = 64;
constexpr int kQuadraticFeatures = 16 + 16 * 17 / 2;
constexpr int kBlocks = 8;
constexpr int kVocabulary = 4475;
constexpr int kPromptTokens = 5;
constexpr int kHeldStride = 5;
constexpr double kRidge = 1e-6;

uint16_t Bf16Bits(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

float ExpandBf16(uint16_t value) {
  return std::bit_cast<float>(uint32_t{value} << 16);
}

using HostBytes = cuda::PageLockedHostArray<uint8_t>;

absl::StatusOr<std::vector<HostBytes>> CopyMasterBytes(cuda::Executor& executor,
                                                       const Layer& model) {
  std::vector<HostBytes> result;
  for (const Buffer& weight : model.weights()) {
    ASSIGN_OR_RETURN(auto host,
                     HostBytes::Allocate(executor, weight.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "snapshot original model master weights"));
    result.push_back(std::move(host));
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return result;
}

absl::Status CheckUnchanged(absl::Span<const HostBytes> before,
                            absl::Span<const HostBytes> after) {
  if (before.size() != after.size())
    return absl::FailedPreconditionError("source weight inventory changed");
  for (size_t i = 0; i < before.size(); ++i)
    if (before[i].size() != after[i].size() ||
        std::memcmp(before[i].data(), after[i].data(), before[i].size()) != 0)
      return absl::FailedPreconditionError(
          absl::StrCat("source master tensor changed: ", i));
  return absl::OkStatus();
}

// These are the exact CreateGpt2 shapes, including the tied embedding alias.
// Complete checkpoint loading independently rejects absent/extra files.
absl::Status CheckInventory(const Layer& model) {
  std::vector<size_t> sizes{kVocabulary * kWidth, kGpt2ContextLength * kWidth};
  for (int block = 0; block < kBlocks; ++block) {
    const size_t block_sizes[] = {kWidth,
                                  kWidth,
                                  kWidth * 3 * kWidth,
                                  3 * kWidth,
                                  kWidth * kWidth,
                                  kWidth,
                                  kWidth,
                                  kWidth,
                                  kWidth * kFeatures,
                                  kFeatures,
                                  kFeatures * kWidth,
                                  kWidth};
    sizes.insert(sizes.end(), std::begin(block_sizes), std::end(block_sizes));
  }
  sizes.insert(sizes.end(), {kWidth, kWidth, kVocabulary * kWidth});
  const auto weights = model.weights();
  if (weights.size() != sizes.size() ||
      weights.front().data() != weights.back().data())
    return absl::FailedPreconditionError("unexpected GPT-2 weight inventory");
  for (size_t i = 0; i < sizes.size(); ++i)
    if (weights[i].size_bytes() != sizes[i] * sizeof(float))
      return absl::FailedPreconditionError(
          absl::StrCat("unexpected GPT-2 tensor size at ", i));
  return absl::OkStatus();
}

struct Expansion {
  std::vector<float> weights;  // Effective BF16 [width, features].
  std::vector<float> bias;     // FP32 [features], never rounded to BF16.
};

// The initialization checkpoint supplies only its eight W1/b1 tensors. No
// initial embedding, attention, LayerNorm, W2, or output target enters a fit.
absl::StatusOr<std::vector<Expansion>> CopyExpansions(cuda::Executor& executor,
                                                      const Layer& model) {
  RETURN_IF_ERROR(CheckInventory(model));
  std::vector<Expansion> result;
  for (int block = 0; block < kBlocks; ++block) {
    Expansion expansion;
    for (int part = 0; part < 2; ++part) {
      const Buffer& buffer = model.weights()[10 + 12 * block + part];
      ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::Allocate(
                                      executor, buffer.size_bytes() / 4));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(host.data(), buffer.data(), host.size_bytes(),
                          cudaMemcpyDeviceToHost, executor.stream()),
          "copy expansion parameters"));
      RETURN_IF_ERROR(executor.Synchronize());
      auto& values = part == 0 ? expansion.weights : expansion.bias;
      for (float value : host) {
        if (!std::isfinite(value))
          return absl::DataLossError("nonfinite expansion master parameter");
        if (part == 0)
          value = ExpandBf16(Bf16Bits(value));
        if (!std::isfinite(value))
          return absl::DataLossError("nonfinite expansion parameter");
        values.push_back(value);
      }
    }
    result.push_back(std::move(expansion));
  }
  return result;
}

absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> MakeProjection(
    cuda::Executor& executor, int input_width, int output_width,
    absl::Span<const float> weights, absl::Span<const float> bias) {
  if (weights.size() != static_cast<size_t>(input_width) * output_width ||
      bias.size() != static_cast<size_t>(output_width))
    return absl::InvalidArgumentError("replacement projection shape mismatch");
  ASSIGN_OR_RETURN(auto layer, FullyConnectedLayer::Create(
                                   executor, input_width, output_width,
                                   DataType::BF16, kGpt2ContextLength));
  const absl::Span<const float> parts[] = {weights, bias};
  for (int part = 0; part < 2; ++part) {
    for (float value : parts[part])
      if (!std::isfinite(value) ||
          (part == 0 && !std::isfinite(ExpandBf16(Bf16Bits(value)))))
        return absl::OutOfRangeError("replacement parameter exceeds GPU range");
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::CopyFrom(
                                    executor, parts[part]));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(layer->weights()[part].data(), host.data(),
                        host.size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "upload replacement projection"));
  }
  return layer;
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> MakeFeatures(
    cuda::Executor& executor, const Expansion& expansion) {
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(MakeProjection(
      executor, kWidth, kFeatures, expansion.weights, expansion.bias)));
  RETURN_IF_ERROR(builder.add(GeluLayer::Create(
      executor, kFeatures, DataType::BF16, kGpt2ContextLength)));
  return builder.create("frozen_mlp_features");
}

// Run the very same BF16 production FC/GELU layers used in the replacement.
// Padding is zero and discarded; these pointwise maps never mix token rows.
// This is used to fit W2 only, NEVER as a replay buffer during evaluation.
absl::StatusOr<std::vector<float>> ApplyToRecordedInputs(
    cuda::Executor& executor, const Layer& layer,
    absl::Span<const float> inputs, int output_width) {
  if (inputs.empty() || inputs.size() % kWidth != 0)
    return absl::InvalidArgumentError("malformed recorded input matrix");
  const size_t rows = inputs.size() / kWidth;
  const size_t padded_rows =
      (rows + kGpt2ContextLength - 1) / kGpt2ContextLength * kGpt2ContextLength;
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint16_t>::Allocate(
                                  executor, padded_rows * kWidth));
  std::fill(host.begin(), host.end(), 0);
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!std::isfinite(inputs[i]) ||
        std::bit_cast<uint32_t>(ExpandBf16(Bf16Bits(inputs[i]))) !=
            std::bit_cast<uint32_t>(inputs[i]))
      return absl::DataLossError("recorded input is not finite exact BF16");
    host[i] = Bf16Bits(inputs[i]);
  }
  ASSIGN_OR_RETURN(auto device, Buffer::Allocate(executor, host.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload recorded normalized inputs"));
  ASSIGN_OR_RETURN(auto forward, layer.fwd(executor, {&device, 1}));
  if (forward.outputs.size() != 1 ||
      forward.outputs[0].size_bytes() !=
          padded_rows * output_width * sizeof(uint16_t))
    return absl::InternalError("replacement produced an unexpected shape");
  ASSIGN_OR_RETURN(auto output, cuda::PageLockedHostArray<uint16_t>::Allocate(
                                    executor, padded_rows * output_width));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(output.data(), forward.outputs[0].data(),
                      output.size_bytes(), cudaMemcpyDeviceToHost,
                      executor.stream()),
      "download production BF16 feature/update matrix"));
  RETURN_IF_ERROR(executor.Synchronize());
  std::vector<float> result(rows * output_width);
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = ExpandBf16(output[i]);
    if (!std::isfinite(result[i]))
      return absl::DataLossError("replacement produced nonfinite values");
  }
  return result;
}

std::vector<float> FittingRows(absl::Span<const float> values, int width,
                               const std::vector<bool>& fitting_rows) {
  std::vector<float> result;
  for (size_t row = 0; row < fitting_rows.size(); ++row)
    if (fitting_rows[row])
      result.insert(result.end(), values.begin() + row * width,
                    values.begin() + (row + 1) * width);
  return result;
}

bool SameFloats(absl::Span<const float> left, absl::Span<const float> right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) ==
             0;
}

absl::StatusOr<std::vector<float>> ToFp32(absl::Span<const double> values) {
  std::vector<float> result;
  result.reserve(values.size());
  for (double value : values) {
    if (!std::isfinite(value) ||
        std::abs(value) > std::numeric_limits<float>::max())
      return absl::OutOfRangeError("fitted coefficient exceeds FP32 range");
    result.push_back(static_cast<float>(value));
  }
  return result;
}

absl::Status ReportUpdateErrors(std::ostream& out, absl::string_view condition,
                                int block, absl::Span<const float> predicted,
                                absl::Span<const float> expected,
                                const std::vector<bool>& fitting_rows) {
  if (predicted.size() != expected.size() ||
      expected.size() != fitting_rows.size() * kWidth)
    return absl::InternalError("update error shape mismatch");
  for (bool held : {false, true}) {
    size_t rows = 0;
    double error = 0, energy = 0;
    double sums[kWidth]{};
    for (size_t row = 0; row < fitting_rows.size(); ++row) {
      if (fitting_rows[row] == held)
        continue;
      ++rows;
      for (int dim = 0; dim < kWidth; ++dim) {
        const size_t i = row * kWidth + dim;
        const double target = expected[i];
        const double delta = predicted[i] - target;
        error += delta * delta;
        energy += target * target;
        sums[dim] += target;
      }
    }
    if (rows == 0)
      return absl::FailedPreconditionError("empty update error group");
    double centered_energy = energy;
    for (double sum : sums)
      centered_energy -= sum * sum / rows;
    if (energy <= 0 || centered_energy <= 0)
      return absl::FailedPreconditionError("degenerate update error group");
    out << condition << '\t' << block << '\t' << (held ? "held" : "fit") << '\t'
        << rows << '\t' << std::sqrt(error / (rows * kWidth)) << '\t'
        << std::sqrt(error / energy) << '\t'
        << std::sqrt(error / centered_energy) << '\n';
  }
  return out ? absl::OkStatus()
             : absl::UnknownError("cannot write update error report");
}

absl::Status Run() {
  namespace fs = std::filesystem;
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path initial_checkpoint = absl::GetFlag(FLAGS_initial_checkpoint);
  const fs::path output_dir = absl::GetFlag(FLAGS_output_dir);
  const int batch_size = absl::GetFlag(FLAGS_batch_size);
  const std::string experiment = absl::GetFlag(FLAGS_experiment);
  if (experiment != "initialization" && experiment != "quadratic")
    return absl::InvalidArgumentError(
        "experiment must be initialization or quadratic");
  const bool quadratic = experiment == "quadratic";
  if ((!quadratic && initial_checkpoint.empty()) ||
      (quadratic && !initial_checkpoint.empty()))
    return absl::InvalidArgumentError(
        "initial_checkpoint is required for initialization and disallowed for "
        "quadratic");
  if (checkpoint.empty() || output_dir.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() || batch_size < 1 ||
      batch_size > 32)
    return absl::InvalidArgumentError(
        "checkpoint, tokenizer, fresh output_dir required; "
        "batch_size must be in [1,32]");
  std::error_code error;
  if (fs::exists(output_dir, error))
    return absl::AlreadyExistsError("output_dir must be fresh");
  if (error)
    return absl::UnknownError(error.message());
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto tokenizer,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint / "compact_vocabulary.tsv"));
  if (tokenizer->vocab_size() != kVocabulary ||
      tokenizer->original_eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError("invalid checkpoint compact vocabulary");
  if (!quadratic) {
    ASSIGN_OR_RETURN(auto initial_tokenizer,
                     tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                         *base, initial_checkpoint / "compact_vocabulary.tsv"));
    if (tokenizer->original_eos_token_id() !=
            initial_tokenizer->original_eos_token_id() ||
        tokenizer->original_token_ids() !=
            initial_tokenizer->original_token_ids())
      return absl::InvalidArgumentError(
          "checkpoint compact vocabularies differ");
  }
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto dataset, PaddedLineDataSetIterator::Create(
                                     *executor, corpus.text(), *tokenizer,
                                     {.batch_size = batch_size,
                                      .context_length = kGpt2ContextLength,
                                      .prompt_tokens = kPromptTokens,
                                      .eos_token = tokenizer->eos_token_id()}));
  const Gpt2Config config{.transformer_block_count = kBlocks,
                          .model_width = kWidth,
                          .attention_heads = 1,
                          .feed_forward_width = kFeatures,
                          .vocabulary_size = kVocabulary,
                          .pad_vocabulary = false};
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  RETURN_IF_ERROR(CheckInventory(*model));
  ASSIGN_OR_RETURN(auto before, CopyMasterBytes(*executor, *model));
  ASSIGN_OR_RETURN(auto learned_expansions, CopyExpansions(*executor, *model));
  std::vector<Expansion> initial_expansions;
  if (!quadratic) {
    ASSIGN_OR_RETURN(auto initial,
                     CreateGpt2(*executor, DataType::BF16, 0, config));
    RETURN_IF_ERROR(ReadFromDirectory(*executor, *initial,
                                      initial_checkpoint.string(), false));
    ASSIGN_OR_RETURN(initial_expansions, CopyExpansions(*executor, *initial));
    // The local initial model is destroyed here. Only its W1/b1 copies remain.
  }
  if (!fs::create_directory(output_dir, error) || error)
    return absl::UnknownError(
        absl::StrCat("cannot create output_dir: ", error.message()));
  std::ofstream manifest(output_dir / "manifest.tsv");
  std::ofstream scores(output_dir / "evaluations.tsv");
  std::ofstream cases(output_dir / "sentences.tsv");
  std::ofstream fits(output_dir / "fits.tsv");
  std::ofstream errors(output_dir / "update_errors.tsv");
  std::ofstream coefficients(output_dir / "coefficients.tsv");
  std::ofstream statistics(output_dir / "features.tsv");
  std::ofstream basis_statistics(output_dir / "standardization.tsv");
  if (!manifest || !scores || !cases || !fits || !errors || !coefficients ||
      !statistics || !basis_statistics)
    return absl::UnknownError("cannot create frozen MLP reports");
  for (auto* stream : {&manifest, &scores, &cases, &fits, &errors,
                       &coefficients, &statistics, &basis_statistics})
    *stream << std::setprecision(17);
  manifest
      << "key\tvalue\ncompleted\tfalse\ncheckpoint\t" << checkpoint.string()
      << "\nexperiment\t" << experiment << "\ninitial_checkpoint\t"
      << initial_checkpoint.string() << "\ncorpus\t"
      << absl::GetFlag(FLAGS_corpus) << "\ntokenizer\t"
      << absl::GetFlag(FLAGS_tokenizer)
      << "\nblocks\t8\nwidth\t16\nfeatures\t64\ncontext\t1024"
         "\nprompt_tokens\t5\nridge\t0.000001\nheld_stride\t5"
      << "\nconstruction_feature_width\t"
      << (quadratic ? kQuadraticFeatures : kFeatures) << "\nbatch_size\t"
      << batch_size
      << "\nheld_definition\tzero-based sentence index divisible by 5"
         "\npositive_control_features\tproduction GPU BF16 learned FC then GELU"
         "\nfit_target\tlearned BF16 branch updates; not token labels"
         "\nevaluation\tfresh recipient LayerNorm inputs at all positions; "
         "no recorded activation replay"
         "\nheld_scope\texcluded from regression only; backbone trained on "
         "them\n";
  if (quadratic)
    manifest
        << "initial_inputs\tnone; no initial checkpoint opened\n"
           "feature_order\tx_0 through x_15, then x_i*x_j for i=0..15, "
           "j=i..15 in lexicographic order\n"
           "feature_arithmetic\tBF16 linear coordinates copied; products "
           "computed in FP32 from decoded BF16 then rounded once to BF16\n"
           "standardization\tnone; no feature scaling or outcome tuning\n"
           "quadratic_parameters\t152*16+16 per block; no W1/b1\n"
           "intercept\tunpenalized output bias separate from 152 features\n";
  else
    manifest
        << "initial_inputs\teffective BF16 W1 directions and FP32 b1 only\n"
           "standardization\tfitting-only FP64 projected moments; unit "
           "population variance; initial bias cancels; FP32 master output\n"
           "feature_arithmetic\tproduction GPU BF16 FC then GELU\n";
  manifest.flush();
  std::cout
      << "Capturing original learned inputs, features and branch updates..."
      << std::endl;
  ASSIGN_OR_RETURN(auto capture,
                   CaptureGpt2Mlps(*executor, *model, *dataset, kWidth,
                                   kFeatures, kBlocks, kVocabulary));
  std::vector<bool> fitting_rows;
  for (size_t sentence = 0; sentence < dataset->sample_count(); ++sentence)
    fitting_rows.insert(fitting_rows.end(),
                        dataset->sample_tokens(sentence).size(),
                        sentence % kHeldStride != 0);
  const size_t fitted_rows =
      std::count(fitting_rows.begin(), fitting_rows.end(), true);
  if (dataset->sample_count() != 1024 || fitting_rows.size() != 14098 ||
      fitted_rows != 11285 || capture.target_count != 10002 ||
      capture.correct_targets != capture.target_count ||
      capture.exact_sentences != 1024 || capture.blocks.size() != kBlocks)
    return absl::FailedPreconditionError(
        "corpus/capture differs from frozen 1024-fact memorized protocol");
  for (const auto& sample : capture.blocks)
    if (sample.normalized_inputs.size() != fitting_rows.size() * kWidth ||
        sample.features.size() != fitting_rows.size() * kFeatures ||
        sample.outputs.size() != sample.normalized_inputs.size())
      return absl::InternalError(
          "capture matrices have inconsistent row counts");
  manifest << "real_rows\t" << fitting_rows.size() << "\nfit_rows\t"
           << fitted_rows << "\nheld_rows\t"
           << fitting_rows.size() - fitted_rows
           << "\nfit_sentences\t819\nheld_sentences\t205\n";
  manifest.flush();
  scores << "condition\tselection\tgroup\tcorrect_targets\ttargets\t"
            "teacher_exact\tgreedy_exact\tsentences\tseconds\n";
  cases << "condition\tselection\tline_1based\tgroup\tcorrect_targets\t"
           "targets\tgreedy_exact\n";
  fits
      << "condition\tblock\trows\trank\tfit_real_rmse\tfit_real_relative_rmse\t"
         "qr_diagonal_spread\tseconds\n";
  errors << "condition\tblock\tgroup\trows\tgpu_rmse\tgpu_relative_rmse\t"
            "gpu_centered_relative_rmse\n";
  coefficients << "condition\tblock\ttensor\tflat_index\tfp32_master\n";
  statistics << "condition\tblock\tfeature\tfit_feature_mean\t"
                "fit_feature_stddev\n";
  basis_statistics << "block\tfeature\tfit_initial_projection_mean\t"
                      "fit_initial_projection_stddev\n";
  auto evaluate = [&](absl::string_view condition, absl::string_view selection,
                      absl::Span<const MlpReplacement> replacements,
                      bool require_perfect) -> absl::Status {
    const auto started = std::chrono::steady_clock::now();
    ASSIGN_OR_RETURN(auto teacher,
                     EvaluateMlpReplacements(*executor, *model, *dataset,
                                             replacements, kVocabulary));
    ASSIGN_OR_RETURN(auto greedy,
                     VerifyMlpGreedyCompletions(*executor, *model, *dataset,
                                                replacements, kVocabulary));
    if (teacher.targets_per_sentence.size() != 1024 ||
        teacher.correct_per_sentence.size() != 1024 ||
        greedy.exact_per_sentence.size() != 1024 || teacher.targets != 10002 ||
        teacher.sentences != 1024 || greedy.sentences != 1024)
      return absl::InternalError("incomplete corpus evaluation");
    int64_t correct[2]{}, targets[2]{}, exact[2]{}, count[2]{};
    for (size_t sentence = 0; sentence < 1024; ++sentence) {
      const bool held = sentence % kHeldStride == 0;
      const bool teacher_exact = teacher.correct_per_sentence[sentence] ==
                                 teacher.targets_per_sentence[sentence];
      if (teacher_exact != greedy.exact_per_sentence[sentence])
        return absl::FailedPreconditionError(
            "teacher-forced and greedy exactness disagree");
      correct[held] += teacher.correct_per_sentence[sentence];
      targets[held] += teacher.targets_per_sentence[sentence];
      exact[held] += teacher_exact;
      ++count[held];
      cases << condition << '\t' << selection << '\t' << sentence + 1 << '\t'
            << (held ? "held" : "fit") << '\t'
            << teacher.correct_per_sentence[sentence] << '\t'
            << teacher.targets_per_sentence[sentence] << '\t' << teacher_exact
            << '\n';
    }
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    for (int group = 0; group < 3; ++group) {
      const bool all = group == 2;
      scores << condition << '\t' << selection << '\t'
             << (all          ? "all"
                 : group == 0 ? "fit"
                              : "held")
             << '\t' << (all ? teacher.correct_targets : correct[group]) << '\t'
             << (all ? teacher.targets : targets[group]) << '\t'
             << (all ? teacher.exact_sentences : exact[group]) << '\t'
             << (all ? greedy.exact_sentences : exact[group]) << '\t'
             << (all ? teacher.sentences : count[group]) << '\t' << seconds
             << '\n';
    }
    scores.flush();
    cases.flush();
    std::cout << condition << " selection=" << selection << ": "
              << teacher.correct_targets << "/10002 teacher targets, "
              << greedy.exact_sentences << "/1024 exact greedy; " << seconds
              << " s" << std::endl;
    if (!scores || !cases)
      return absl::UnknownError("cannot write evaluation report");
    if (require_perfect &&
        (teacher.correct_targets != 10002 || greedy.exact_sentences != 1024))
      return absl::FailedPreconditionError(
          "original/cloned positive control failed");
    return absl::OkStatus();
  };
  RETURN_IF_ERROR(evaluate("original", "all", {}, true));
  std::vector<absl::string_view> conditions{"original_mlp_clone",
                                            "learned_w1_refit"};
  if (quadratic)
    conditions.push_back("quadratic_refit");
  else {
    conditions.push_back("initial_w1_raw");
    conditions.push_back("initial_w1_standardized");
  }
  for (absl::string_view condition : conditions) {
    const bool use_quadratic = condition == "quadratic_refit";
    const int feature_width = use_quadratic ? kQuadraticFeatures : kFeatures;
    std::vector<std::unique_ptr<ComposedLayer>> layers;
    std::vector<MlpReplacement> replacements;
    for (int block = 0; block < kBlocks; ++block) {
      const auto& sample = capture.blocks[block];
      Expansion expansion;
      if (!use_quadratic)
        expansion =
            condition == "original_mlp_clone" || condition == "learned_w1_refit"
                ? learned_expansions[block]
                : initial_expansions[block];
      if (condition == "initial_w1_standardized") {
        const auto fitting_inputs =
            FittingRows(sample.normalized_inputs, kWidth, fitting_rows);
        ASSIGN_OR_RETURN(auto basis, StandardizeFrozenMlpBasis(
                                         fitting_inputs, expansion.weights,
                                         kWidth, kFeatures));
        for (int feature = 0; feature < kFeatures; ++feature)
          basis_statistics << block << '\t' << feature << '\t'
                           << basis.projection_means[feature] << '\t'
                           << basis.projection_standard_deviations[feature]
                           << '\n';
        expansion.weights = std::move(basis.weights);
        expansion.bias = std::move(basis.bias);
      }
      std::unique_ptr<Layer> features;
      if (use_quadratic) {
        ASSIGN_OR_RETURN(features, QuadraticFeaturesLayer::Create(
                                       *executor, kGpt2ContextLength));
      } else {
        ASSIGN_OR_RETURN(features, MakeFeatures(*executor, expansion));
      }
      ASSIGN_OR_RETURN(
          auto all_features,
          ApplyToRecordedInputs(*executor, *features, sample.normalized_inputs,
                                feature_width));
      if ((condition == "original_mlp_clone" ||
           condition == "learned_w1_refit") &&
          !SameFloats(all_features, sample.features))
        return absl::FailedPreconditionError(
            "regenerated learned GELU features differ from original capture");
      const auto fitting_features =
          FittingRows(all_features, feature_width, fitting_rows);
      for (int feature = 0; feature < feature_width; ++feature) {
        double mean = 0, variance = 0;
        for (size_t row = 0; row < fitted_rows; ++row)
          mean += fitting_features[row * feature_width + feature] / fitted_rows;
        for (size_t row = 0; row < fitted_rows; ++row) {
          const double delta =
              fitting_features[row * feature_width + feature] - mean;
          variance += delta * delta / fitted_rows;
        }
        statistics << condition << '\t' << block << '\t' << feature << '\t'
                   << mean << '\t' << std::sqrt(variance) << '\n';
      }
      std::vector<float> output_weights, output_bias;
      if (condition == "original_mlp_clone") {
        output_weights = sample.effective_output_weights;
        output_bias = sample.output_bias;
      } else {
        const auto fitting_outputs =
            FittingRows(sample.outputs, kWidth, fitting_rows);
        const auto started = std::chrono::steady_clock::now();
        ASSIGN_OR_RETURN(auto fit,
                         FitAffineMap(fitting_features, fitting_outputs,
                                      feature_width, kWidth, {.ridge = kRidge}));
        const auto [smallest, largest] =
            std::minmax_element(fit.qr_diagonal_magnitudes.begin(),
                                fit.qr_diagonal_magnitudes.end());
        fits << condition << '\t' << block << '\t' << fit.sample_count << '\t'
             << fit.numerical_rank << '\t' << fit.rmse << '\t'
             << fit.relative_rmse << '\t' << *largest / *smallest << '\t'
             << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              started)
                    .count()
             << '\n';
        ASSIGN_OR_RETURN(output_weights, ToFp32(fit.weights));
        ASSIGN_OR_RETURN(output_bias, ToFp32(fit.biases));
      }
      const absl::Span<const float> tensors[] = {
          expansion.weights, expansion.bias, output_weights, output_bias};
      const char* names[] = {"W1", "b1", "W2", "b2"};
      for (int tensor = 0; tensor < 4; ++tensor)
        for (size_t i = 0; i < tensors[tensor].size(); ++i)
          coefficients << condition << '\t' << block << '\t' << names[tensor]
                       << '\t' << i << '\t' << tensors[tensor][i] << '\n';
      ComposedLayerBuilder builder;
      RETURN_IF_ERROR(builder.add(std::move(features)));
      RETURN_IF_ERROR(builder.add(MakeProjection(
          *executor, feature_width, kWidth, output_weights, output_bias)));
      ASSIGN_OR_RETURN(auto replacement,
                       builder.create("frozen_mlp_replacement"));
      ASSIGN_OR_RETURN(auto predicted_updates,
                       ApplyToRecordedInputs(*executor, *replacement,
                                             sample.normalized_inputs, kWidth));
      if (condition == "original_mlp_clone" &&
          !SameFloats(predicted_updates, sample.outputs))
        return absl::FailedPreconditionError(
            "cloned full MLP updates differ from original capture");
      RETURN_IF_ERROR(ReportUpdateErrors(errors, condition, block,
                                         predicted_updates, sample.outputs,
                                         fitting_rows));
      replacements.push_back({block, MlpSource::kLayerNorm, replacement.get()});
      layers.push_back(std::move(replacement));
      if (condition != "original_mlp_clone")
        RETURN_IF_ERROR(evaluate(condition, absl::StrCat(block),
                                 {&replacements.back(), 1}, false));
      fits.flush();
      errors.flush();
      coefficients.flush();
      statistics.flush();
      basis_statistics.flush();
    }
    RETURN_IF_ERROR(evaluate(
        condition, "all", replacements,
        condition == "original_mlp_clone" || condition == "learned_w1_refit"));
  }
  RETURN_IF_ERROR(evaluate("original_post", "all", {}, true));
  ASSIGN_OR_RETURN(auto after, CopyMasterBytes(*executor, *model));
  RETURN_IF_ERROR(CheckUnchanged(before, after));
  manifest << "learned_features_bitwise_control\tPASS\n"
              "cloned_mlp_updates_bitwise_control\tPASS\n"
              "source_all_master_bytes_unchanged\tPASS\n"
              "original_post_control\tPASS\ncompleted\ttrue\n";
  for (auto* stream : {&manifest, &scores, &cases, &fits, &errors,
                       &coefficients, &statistics, &basis_statistics}) {
    stream->flush();
    if (!*stream)
      return absl::UnknownError("cannot finish frozen MLP report");
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  const auto positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "Unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::llm::one_shot_memorizer::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
