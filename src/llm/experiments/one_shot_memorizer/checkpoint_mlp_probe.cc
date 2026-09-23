// System identification and nonlinear-feature ablation for the learned MLPs.
// Teacher hidden updates are NOT dataset target tokens: success at
// reconstructing their linear output projections does not construct the
// backbone from text.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/closed_form_map.h"
#include "src/llm/experiments/one_shot_memorizer/mlp_probe.h"
#include "src/llm/gpt2.h"
#include "src/llm/layers/fully_connected.h"
#include "src/util/status_macros.h"
#include "src/util/tee_stream.h"

ABSL_FLAG(std::string, checkpoint, "", "Exact memorized checkpoint directory");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Evaluation corpus");
ABSL_FLAG(std::string, output_dir, "", "Fresh report directory");
ABSL_FLAG(int, batch_size, 32, "Corpus sentences per forward batch");
ABSL_FLAG(int, prompt_tokens, 5, "Supplied prefix length");
ABSL_FLAG(int, layers, 8, "Transformer block count");
ABSL_FLAG(int, model_width, 16, "Residual width");
ABSL_FLAG(int, attention_heads, 1, "Attention heads");
ABSL_FLAG(int, feed_forward_width, 64, "Inner MLP width");
ABSL_FLAG(
    int, fit_sentence_stride, 5,
    "Hold out every Nth whole sentence from the affine fit (0 fits all; N>=2)");
ABSL_FLAG(bool, verify_greedy, true,
          "Also run generated-prefix suffix-plus-EOS verification for each "
          "joint intervention");
ABSL_FLAG(std::vector<std::string>, ridges,
          (std::vector<std::string>{"0", "0.000001", "0.001"}),
          "Nonnegative mean-square ridge penalties, comma separated");

namespace pluto::llm::one_shot_memorizer {
namespace {

float EffectiveBf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return std::bit_cast<float>(bits & 0xffff0000u);
}

absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> MakeProjection(
    cuda::Executor& executor, int input_dim, int output_dim,
    absl::Span<const double> weights, absl::Span<const double> biases) {
  if (input_dim <= 0 || output_dim <= 0 ||
      weights.size() != static_cast<size_t>(input_dim) * output_dim ||
      biases.size() != static_cast<size_t>(output_dim))
    return absl::InvalidArgumentError(
        "invalid replacement projection dimensions");
  ASSIGN_OR_RETURN(auto layer, FullyConnectedLayer::Create(
                                   executor, input_dim, output_dim,
                                   DataType::BF16, kGpt2ContextLength));
  const absl::Span<const double> values[] = {weights, biases};
  for (int index = 0; index < 2; ++index) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::Allocate(
                                    executor, values[index].size()));
    for (size_t i = 0; i < host.size(); ++i) {
      host[i] = static_cast<float>(values[index][i]);
      if (!std::isfinite(host[i]) ||
          (index == 0 && !std::isfinite(EffectiveBf16(host[i]))))
        return absl::OutOfRangeError(
            "fitted coefficients exceed GPU precision range");
    }
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(layer->weights()[index].data(), host.data(),
                        host.size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "upload closed-form projection"));
  }
  return layer;
}

// Report effective matrix error: the GPU will round fitted FP32 master
// weights to BF16. Biases instead remain FP32 throughout the accumulation.
double RelativeWeightError(const ClosedFormMap& map,
                           const MlpSamples& samples) {
  double error = 0, scale = 0;
  for (size_t i = 0; i < map.weights.size(); ++i) {
    const double original = samples.effective_output_weights[i];
    const double difference =
        EffectiveBf16(static_cast<float>(map.weights[i])) - original;
    error += difference * difference;
    scale += original * original;
  }
  return scale > 0 ? std::sqrt(error / scale) : std::sqrt(error);
}

std::vector<float> SelectFittingRows(absl::Span<const float> values, int width,
                                     const std::vector<bool>& fitting_rows) {
  std::vector<float> selected;
  selected.reserve(values.size());
  for (size_t row = 0; row < fitting_rows.size(); ++row)
    if (fitting_rows[row])
      selected.insert(selected.end(), values.begin() + row * width,
                      values.begin() + (row + 1) * width);
  return selected;
}

// Keep update error separate from next-token accuracy. Report both raw-target
// and centered-target norms: a large constant component must not make a poor
// variable update look nearly exact. These CPU calculations intentionally do
// not round the output, so the known original map exposes the discrepancy
// introduced by GPU accumulation/output rounding.
absl::Status ReportUpdateErrors(std::ostream& report, const std::string& name,
                                const std::string& ridge, int block,
                                const ClosedFormMap& map,
                                absl::Span<const float> inputs,
                                absl::Span<const float> targets,
                                const std::vector<bool>& fitting_rows) {
  ASSIGN_OR_RETURN(auto predictions, ApplyClosedFormMap(map, inputs));
  if (predictions.size() != targets.size() ||
      predictions.size() != fitting_rows.size() * map.output_dim)
    return absl::InternalError("update error matrices disagree");
  size_t counts[2]{};
  double error[2]{}, energy[2]{};
  std::vector<double> sums(static_cast<size_t>(2) * map.output_dim);
  for (size_t row = 0; row < fitting_rows.size(); ++row) {
    const int held = !fitting_rows[row];
    ++counts[held];
    for (int dim = 0; dim < map.output_dim; ++dim) {
      const size_t index = row * map.output_dim + dim;
      const double difference = predictions[index] - targets[index];
      error[held] += difference * difference;
      energy[held] += static_cast<double>(targets[index]) * targets[index];
      sums[held * map.output_dim + dim] += targets[index];
    }
  }
  for (int held = 0; held < 2; ++held) {
    if (counts[held] == 0)
      continue;
    double centered_energy = energy[held];
    for (int dim = 0; dim < map.output_dim; ++dim) {
      const double sum = sums[held * map.output_dim + dim];
      centered_energy -= sum * sum / counts[held];
    }
    centered_energy = std::max(0.0, centered_energy);
    const double values = static_cast<double>(counts[held]) * map.output_dim;
    report << name << '\t' << ridge << '\t' << block << '\t'
           << (held ? "held_sentence" : "fitting_sentence") << '\t'
           << counts[held] << '\t' << std::sqrt(error[held] / values) << '\t'
           << std::sqrt(energy[held] / values) << '\t'
           << std::sqrt(centered_energy / values) << '\t';
    if (energy[held] > 0)
      report << std::sqrt(error[held] / energy[held]);
    else
      report << "NA";
    report << '\t';
    if (centered_energy > 0)
      report << std::sqrt(error[held] / centered_energy);
    else
      report << "NA";
    report << '\n';
  }
  if (!report)
    return absl::UnknownError("cannot write update error report");
  return absl::OkStatus();
}

absl::Status RunProbe() {
  namespace fs = std::filesystem;
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path output_dir = absl::GetFlag(FLAGS_output_dir);
  const int width = absl::GetFlag(FLAGS_model_width);
  const int features = absl::GetFlag(FLAGS_feed_forward_width);
  const int blocks = absl::GetFlag(FLAGS_layers);
  const int held_stride = absl::GetFlag(FLAGS_fit_sentence_stride);
  if (checkpoint.empty() || output_dir.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() || blocks <= 0 ||
      held_stride < 0 || held_stride == 1)
    return absl::InvalidArgumentError(
        "checkpoint, output_dir, tokenizer and positive layers required");
  std::vector<double> ridges;
  for (const auto& text : absl::GetFlag(FLAGS_ridges)) {
    double ridge;
    if (!absl::SimpleAtod(text, &ridge) || !std::isfinite(ridge) || ridge < 0)
      return absl::InvalidArgumentError(
          "ridges must be finite nonnegative numbers");
    ridges.push_back(ridge);
  }
  if (ridges.empty())
    return absl::InvalidArgumentError("at least one ridge is required");
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto tokenizer,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint / "compact_vocabulary.tsv"));
  if (tokenizer->original_eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError("tokenizer EOS differs from checkpoint");
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto dataset,
                   PaddedLineDataSetIterator::Create(
                       *executor, corpus.text(), *tokenizer,
                       {.batch_size = absl::GetFlag(FLAGS_batch_size),
                        .context_length = kGpt2ContextLength,
                        .prompt_tokens = absl::GetFlag(FLAGS_prompt_tokens),
                        .eos_token = tokenizer->eos_token_id()}));
  const int vocab = tokenizer->vocab_size();
  const Gpt2Config config{
      .transformer_block_count = blocks,
      .model_width = width,
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = features,
      .vocabulary_size = vocab,
      .pad_vocabulary = false};
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  std::error_code error;
  const bool created = fs::create_directory(output_dir, error);
  if (error)
    return absl::UnknownError(
        absl::StrCat("cannot create output_dir: ", error.message()));
  if (!created)
    return absl::AlreadyExistsError("output_dir must be fresh");
  std::ofstream fit_file(output_dir / "mlp_fits.tsv");
  std::ofstream evaluation_file(output_dir / "mlp_replacements.tsv");
  std::ofstream update_file(output_dir / "mlp_update_errors.tsv");
  std::ofstream greedy_file(output_dir / "mlp_greedy.tsv");
  if (!fit_file || !evaluation_file || !update_file || !greedy_file)
    return absl::UnknownError("cannot create MLP probe reports");
  util::TeeStream report(std::cout, evaluation_file);
  greedy_file << "# Predictions are fed back; each sentence stops at correct "
                 "EOS or first mismatch.\n"
              << "source\tridge\texact_sentences\tsentences\tgenerated_"
                 "targets\tfit_exact\tfit_sentences\theld_exact\theld_"
                 "sentences\tseconds\n";
  report << "# checkpoint=" << checkpoint.string()
         << "\n# corpus=" << absl::GetFlag(FLAGS_corpus)
         << "\n# tokenizer=" << absl::GetFlag(FLAGS_tokenizer)
         << "\n# width=" << width << "\n# feature_width=" << features
         << "\n# blocks=" << blocks << "\n# heads=" << config.attention_heads
         << "\n# batch_size=" << dataset->options().batch_size
         << "\n# prompt_tokens=" << dataset->options().prompt_tokens
         << "\n# fit_sentence_stride=" << held_stride
         << "\n# Capture includes ALL real tokens, including supplied prompt "
            "positions.\n"
         << "# All-position branch replacement; earlier attention, LayerNorm, "
            "residuals and head remain learned.\n"
         << "# Fitted targets are teacher hidden updates, not next-token "
            "labels; this is conditional system identification.\n"
         << "# Held-out sentences are excluded from fitting only; the original "
            "backbone trained on them.\n"
         << "# Scores are teacher-forced target and all-rows-correct sentence "
            "counts, not an independent greedy run.\n"
         << "# Capturing original MLP features and outputs...\n";
  ASSIGN_OR_RETURN(auto capture,
                   CaptureGpt2Mlps(*executor, *model, *dataset, width, features,
                                   blocks, vocab));
  if (capture.correct_targets != capture.target_count ||
      capture.exact_sentences != capture.sentences)
    return absl::FailedPreconditionError(
        "unaltered checkpoint is not fully memorized");
  std::vector<bool> fitting_rows;
  for (size_t sentence = 0; sentence < dataset->sample_count(); ++sentence) {
    const bool fitting = held_stride == 0 || sentence % held_stride != 0;
    fitting_rows.insert(fitting_rows.end(),
                        dataset->sample_tokens(sentence).size(), fitting);
  }
  const size_t fitted_rows =
      std::count(fitting_rows.begin(), fitting_rows.end(), true);
  if (fitted_rows == 0)
    return absl::InvalidArgumentError("sentence split leaves no fitting rows");
  for (const auto& samples : capture.blocks)
    if (samples.outputs.size() != fitting_rows.size() * width ||
        samples.normalized_inputs.size() != samples.outputs.size() ||
        samples.features.size() != fitting_rows.size() * features)
      return absl::InternalError(
          "MLP capture row count disagrees with real token lengths");
  report
      << "source\tridge\tselection\tcorrect_targets\ttargets\texact_"
         "sentences\tsentences"
      << "\tfit_correct\tfit_targets\tfit_exact\tfit_sentences"
      << "\theld_correct\theld_targets\theld_exact\theld_sentences\tseconds\n";
  auto evaluate = [&](const std::string& name, const std::string& ridge,
                      const std::string& selection,
                      absl::Span<const MlpReplacement> replacements)
      -> absl::StatusOr<MlpEvaluation> {
    const auto start = std::chrono::steady_clock::now();
    ASSIGN_OR_RETURN(auto result,
                     EvaluateMlpReplacements(*executor, *model, *dataset,
                                             replacements, vocab));
    if (result.targets_per_sentence.size() != dataset->sample_count() ||
        result.correct_per_sentence.size() != dataset->sample_count())
      return absl::InternalError("missing per-sentence replacement scores");
    int64_t correct[2]{}, targets[2]{}, exact[2]{}, sentences[2]{};
    for (size_t sentence = 0; sentence < dataset->sample_count(); ++sentence) {
      const int held = held_stride != 0 && sentence % held_stride == 0;
      correct[held] += result.correct_per_sentence[sentence];
      targets[held] += result.targets_per_sentence[sentence];
      exact[held] += result.correct_per_sentence[sentence] ==
                     result.targets_per_sentence[sentence];
      ++sentences[held];
    }
    report << name << '\t' << ridge << '\t' << selection << '\t'
           << result.correct_targets << '\t' << result.targets << '\t'
           << result.exact_sentences << '\t' << result.sentences << '\t'
           << correct[0] << '\t' << targets[0] << '\t' << exact[0] << '\t'
           << sentences[0] << '\t' << correct[1] << '\t' << targets[1] << '\t'
           << exact[1] << '\t' << sentences[1] << '\t'
           << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            start)
                  .count()
           << '\n';
    if (!evaluation_file)
      return absl::UnknownError("cannot write MLP evaluation");
    if (selection == "all" && absl::GetFlag(FLAGS_verify_greedy)) {
      const auto greedy_start = std::chrono::steady_clock::now();
    ASSIGN_OR_RETURN(auto greedy,
                     VerifyMlpGreedyCompletions(*executor, *model, *dataset,
                                                replacements, vocab));
      if (greedy.exact_per_sentence.size() != dataset->sample_count())
        return absl::InternalError("missing per-sentence greedy scores");
      int64_t greedy_exact[2]{};
      for (size_t sentence = 0; sentence < dataset->sample_count();
           ++sentence) {
        const bool teacher_exact = result.correct_per_sentence[sentence] ==
                                   result.targets_per_sentence[sentence];
        if (greedy.exact_per_sentence[sentence] != teacher_exact)
          return absl::FailedPreconditionError(absl::StrCat(
              "teacher-forced and autoregressive exactness disagree for "
              "sentence ",
              sentence));
        const int held = held_stride != 0 && sentence % held_stride == 0;
        greedy_exact[held] += greedy.exact_per_sentence[sentence];
      }
      greedy_file << name << '\t' << ridge << '\t' << greedy.exact_sentences
                  << '\t' << greedy.sentences << '\t'
                  << greedy.generated_targets << '\t' << greedy_exact[0] << '\t'
                  << sentences[0] << '\t' << greedy_exact[1] << '\t'
                  << sentences[1] << '\t'
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - greedy_start)
                         .count()
                  << '\n';
      greedy_file.flush();
      if (!greedy_file)
        return absl::UnknownError("cannot write MLP greedy report");
      report << "# Greedy " << name << " ridge=" << ridge << ": "
             << greedy.exact_sentences << '/' << greedy.sentences
             << " exact suffix-plus-EOS completions\n";
      if (greedy.exact_sentences != result.exact_sentences ||
          greedy.sentences != result.sentences)
        return absl::FailedPreconditionError(
            "teacher-forced and autoregressive exact counts disagree");
    }
    return result;
  };
  ASSIGN_OR_RETURN(auto baseline, evaluate("original", "NA", "all", {}));
  if (baseline.correct_targets != capture.correct_targets)
    return absl::FailedPreconditionError(
        "baseline predictions changed after capture");
  // Clone all effective W2 matrices and FP32 biases before interpreting a fit.
  // This must reproduce every original prediction through the same hook path.
  std::vector<std::unique_ptr<FullyConnectedLayer>> clones;
  std::vector<MlpReplacement> clone_replacements;
  update_file
      << "# Double-precision affine predictions versus recorded BF16 branch "
         "outputs.\n"
      << "# Original weights are effective BF16, bias FP32; known-original "
         "residual measures GPU accumulation/output-rounding error.\n"
      << "source\tridge\tblock\tsubset\trows\trmse\ttarget_rms\tcentered_"
         "target_rms\trelative_rmse\tcentered_relative_rmse\n";
  for (int block = 0; block < blocks; ++block) {
    const auto& sample = capture.blocks[block];
    const std::vector<double> weights(sample.effective_output_weights.begin(),
                                      sample.effective_output_weights.end());
    const std::vector<double> biases(sample.output_bias.begin(),
                                     sample.output_bias.end());
    ClosedFormMap original_map;
    original_map.input_dim = features;
    original_map.output_dim = width;
    original_map.weights = weights;
    original_map.biases = biases;
    RETURN_IF_ERROR(ReportUpdateErrors(
        update_file, "original_w2_real_arithmetic", "NA", block, original_map,
        sample.features, sample.outputs, fitting_rows));
    ASSIGN_OR_RETURN(
        auto clone, MakeProjection(*executor, features, width, weights, biases));
    clone_replacements.push_back({block, MlpSource::kGelu, clone.get()});
    clones.push_back(std::move(clone));
  }
  ASSIGN_OR_RETURN(auto control, evaluate("original_w2_clone", "NA", "all",
                                          clone_replacements));
  if (control.correct_targets != capture.correct_targets ||
      control.exact_sentences != capture.exact_sentences)
    return absl::FailedPreconditionError(
        "cloned output projections do not reproduce original decisions");
  for (bool use_mean : {false, true}) {
    const std::string name = use_mean ? "mean_update" : "zero_update";
    std::vector<std::unique_ptr<FullyConnectedLayer>> projections;
    std::vector<MlpReplacement> replacements;
    for (int block = 0; block < blocks; ++block) {
      std::vector<double> weights(static_cast<size_t>(width) * width),
          biases(width);
      if (use_mean) {
        const auto& outputs = capture.blocks[block].outputs;
        for (size_t row = 0; row < fitting_rows.size(); ++row)
          if (fitting_rows[row])
            for (int dim = 0; dim < width; ++dim)
              biases[dim] += outputs[row * width + dim];
        for (double& bias : biases)
          bias /= fitted_rows;
      }
      ASSIGN_OR_RETURN(auto projection,
                       MakeProjection(*executor, width, width, weights, biases));
      replacements.push_back({block, MlpSource::kLayerNorm, projection.get()});
      projections.push_back(std::move(projection));
      ASSIGN_OR_RETURN(auto result, evaluate(name, "NA", absl::StrCat(block),
                                             {&replacements.back(), 1}));
      (void)result;
    }
    ASSIGN_OR_RETURN(auto result, evaluate(name, "NA", "all", replacements));
    (void)result;
  }
  fit_file << "source\tridge\tblock\trows\tinput_dim\toutput_dim\tfit_"
              "rank\trmse\ttarget_rms\trelative_rmse\teffective_weight_"
              "relative_error\tqr_diagonal_spread\tseconds\tstatus\n";
  for (MlpSource source : {MlpSource::kGelu, MlpSource::kLayerNorm}) {
    const std::string name = source == MlpSource::kGelu
                                 ? "learned_gelu_refit"
                                 : "affine_after_layernorm";
    const int input_dim = source == MlpSource::kGelu ? features : width;
    for (double ridge : ridges) {
      std::vector<std::unique_ptr<FullyConnectedLayer>> projections;
      std::vector<MlpReplacement> replacements;
      for (int block = 0; block < blocks; ++block) {
        const auto& samples = capture.blocks[block];
        const auto& all_inputs = source == MlpSource::kGelu
                                     ? samples.features
                                     : samples.normalized_inputs;
        const auto inputs =
            SelectFittingRows(all_inputs, input_dim, fitting_rows);
        const auto outputs =
            SelectFittingRows(samples.outputs, width, fitting_rows);
        const auto start = std::chrono::steady_clock::now();
        auto fitted =
            FitAffineMap(inputs, outputs, input_dim, width, {.ridge = ridge});
        if (!fitted.ok()) {
          // A rank failure is evidence, not permission to change the requested
          // ridge silently. Leave this block and joint replacement unmeasured.
          if (fitted.status().code() != absl::StatusCode::kFailedPrecondition)
            return fitted.status();
          fit_file << name << '\t' << ridge << '\t' << block << '\t'
                   << inputs.size() / input_dim << '\t' << input_dim << '\t'
                   << width << "\tNA\tNA\tNA\tNA\tNA\tNA\t0\t"
                   << fitted.status().message() << '\n';
          report << "# Fit skipped: " << name << " block=" << block
                 << " ridge=" << ridge << " " << fitted.status() << '\n';
          continue;
        }
        fit_file << name << '\t' << ridge << '\t' << block << '\t'
                 << fitted->sample_count << '\t' << input_dim << '\t' << width
                 << '\t' << fitted->numerical_rank << '\t' << fitted->rmse
                 << '\t' << fitted->target_rms << '\t' << fitted->relative_rmse
                 << '\t';
        if (source == MlpSource::kGelu)
          fit_file << RelativeWeightError(*fitted, samples);
        else
          fit_file << "NA";
        // Diagonal spread is a conditioning diagnostic, not the spectral
        // condition number. Positive ridge changes the augmented design.
        const auto [minimum, maximum] =
            std::minmax_element(fitted->qr_diagonal_magnitudes.begin(),
                                fitted->qr_diagonal_magnitudes.end());
        fit_file << '\t' << *maximum / *minimum;
        fit_file << '\t'
                 << std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start)
                        .count()
                 << "\tOK\n";
        RETURN_IF_ERROR(ReportUpdateErrors(
            update_file, name, absl::StrCat(ridge), block, *fitted, all_inputs,
            samples.outputs, fitting_rows));
        ASSIGN_OR_RETURN(auto projection,
                         MakeProjection(*executor, input_dim, width,
                                        fitted->weights, fitted->biases));
        replacements.push_back({block, source, projection.get()});
        projections.push_back(std::move(projection));
        ASSIGN_OR_RETURN(auto result, evaluate(name, absl::StrCat(ridge),
                                               absl::StrCat(block),
                                               {&replacements.back(), 1}));
        (void)result;
      }
      if (replacements.size() == static_cast<size_t>(blocks)) {
        ASSIGN_OR_RETURN(auto result, evaluate(name, absl::StrCat(ridge), "all",
                                               replacements));
        (void)result;
      }
    }
  }
  fit_file.flush();
  update_file.flush();
  if (!fit_file || !update_file)
    return absl::UnknownError("cannot write MLP fit report");
  report << "# Complete\n";
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::llm::one_shot_memorizer::RunProbe();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
