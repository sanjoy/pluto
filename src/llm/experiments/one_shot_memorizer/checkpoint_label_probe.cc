// Test whether corpus labels, instead of teacher MLP outputs, can specify a
// final projection in the checkpoint's already-learned feature space.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/decision_readout.h"
#include "src/llm/experiments/one_shot_memorizer/feature_capture.h"
#include "src/llm/experiments/one_shot_memorizer/final_mlp_probe.h"
#include "src/llm/experiments/one_shot_memorizer/label_projection.h"
#include "src/llm/experiments/one_shot_memorizer/margin_projection.h"
#include "src/llm/experiments/one_shot_memorizer/mlp_probe.h"
#include "src/llm/experiments/one_shot_memorizer/token_codes.h"
#include "src/llm/extract_top1_ids.h"
#include "src/llm/gpt2.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/norm.h"
#include "src/util/status_macros.h"
#include "src/util/tee_stream.h"

ABSL_FLAG(std::string, checkpoint, "", "Exact memorized checkpoint");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt", "Corpus");
ABSL_FLAG(std::string, output_dir, "",
          "Fresh directory for label-only reports");
ABSL_FLAG(int, batch_size, 32, "Sentences per GPU batch");
ABSL_FLAG(int, prompt_tokens, 5, "Supplied prompt length");
ABSL_FLAG(int, layers, 8, "Transformer blocks");
ABSL_FLAG(int, model_width, 16, "Residual width");
ABSL_FLAG(int, attention_heads, 1, "Attention head count");
ABSL_FLAG(int, feed_forward_width, 64, "MLP feature width");
ABSL_FLAG(int, fit_sentence_stride, 5,
          "Hold out every Nth sentence; 0 fits all");
ABSL_FLAG(double, ridge, 1e-6, "Mean-square ridge penalty in the direct solve");
ABSL_FLAG(double, code_scale, 0.25,
          "Desired residual amplitude per code coordinate");
ABSL_FLAG(uint64_t, code_seed, 0,
          "Fixed token-to-code assignment/permutation seed");
ABSL_FLAG(
    std::string, projection_objective, "code_vector",
    "code_vector: fit target vectors; token_margin: solve target-vs-rival "
    "linear inequalities with the original head and gamma, beta=0");
ABSL_FLAG(double, decision_coefficient_bound, 1,
          "Absolute bound on every fitted decision projection coefficient");
ABSL_FLAG(int, decision_rounds, 30, "Maximum decision cutting-plane rounds");
ABSL_FLAG(int, decision_new_cuts, 1024,
          "Maximum new target-versus-rival constraints per round");
ABSL_FLAG(int, decision_total_cuts, 30000,
          "Maximum accumulated target-versus-rival constraints");
ABSL_FLAG(int, decision_solve_seconds, 60,
          "Time limit for each simplex solve; exhaustion is inconclusive");
ABSL_FLAG(bool, decision_dual_simplex, false,
          "Use dual simplex in both solver phases for incremental cuts");

namespace pluto::llm::one_shot_memorizer {
namespace {

namespace fs = std::filesystem;

absl::Status Upload(cuda::Executor& executor, absl::Span<const float> values,
                    Buffer& destination) {
  if (destination.size_bytes() != values.size() * sizeof(float))
    return absl::InvalidArgumentError("projection upload size mismatch");
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::Allocate(
                                  executor, values.size()));
  std::copy(values.begin(), values.end(), host.begin());
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(destination.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload label-only parameter"));
  return absl::OkStatus();
}

struct Readout {
  // Destruction order matters: the head borrows this independent output table.
  std::unique_ptr<EmbeddingLookupLayer> table;
  std::unique_ptr<LanguageModelingHeadLayer> head;
};

absl::StatusOr<Readout> MakeReadout(cuda::Executor& executor,
                                    const TokenCodes& codes) {
  ASSIGN_OR_RETURN(auto table, EmbeddingLookupLayer::Create(
                                   executor, codes.vocab_size, codes.width,
                                   DataType::BF16, kGpt2ContextLength, false));
  RETURN_IF_ERROR(Upload(executor, codes.values, table->weights()[0]));
  ASSIGN_OR_RETURN(auto head, LanguageModelingHeadLayer::Create(table.get()));
  return Readout{std::move(table), std::move(head)};
}

absl::StatusOr<std::unique_ptr<LayerNormLayer>> MakeNorm(
    cuda::Executor& executor, absl::Span<const float> gamma,
    absl::Span<const float> beta) {
  if (gamma.size() != beta.size())
    return absl::InvalidArgumentError("normalization parameter sizes differ");
  ASSIGN_OR_RETURN(auto norm,
                   LayerNormLayer::Create(executor, gamma.size(), 1e-5f,
                                          DataType::BF16, kGpt2ContextLength));
  RETURN_IF_ERROR(Upload(executor, gamma, norm->weights()[0]));
  RETURN_IF_ERROR(Upload(executor, beta, norm->weights()[1]));
  return norm;
}

absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> MakeProjection(
    cuda::Executor& executor, const ClosedFormMap& map) {
  ASSIGN_OR_RETURN(auto layer, FullyConnectedLayer::Create(
                                   executor, map.input_dim, map.output_dim,
                                   DataType::BF16, kGpt2ContextLength));
  const absl::Span<const double> parameters[] = {map.weights, map.biases};
  for (int index = 0; index < 2; ++index) {
    std::vector<float> values(parameters[index].begin(),
                              parameters[index].end());
    for (float value : values) {
      uint32_t bits = std::bit_cast<uint32_t>(value);
      bits += 0x7fff + ((bits >> 16) & 1);
      if (!std::isfinite(value) ||
          (index == 0 &&
           !std::isfinite(std::bit_cast<float>(bits & 0xffff0000u))))
        return absl::OutOfRangeError(
            "fitted coefficients exceed GPU precision");
    }
    RETURN_IF_ERROR(Upload(executor, values, layer->weights()[index]));
  }
  return layer;
}

// The desired codes must actually decode to their own IDs under the GPU's
// BF16 rounding and normalization. This control does NOT access the corpus.
absl::StatusOr<int> CodeOracle(cuda::Executor& executor,
                               const TokenCodes& codes, const Layer& norm,
                               const Layer& head, double scale) {
  const int rows = (codes.vocab_size + kGpt2ContextLength - 1) /
                   kGpt2ContextLength * kGpt2ContextLength;
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<uint16_t>::Allocate(
                       executor, static_cast<size_t>(rows) * codes.width));
  ASSIGN_OR_RETURN(auto mask,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  std::fill(host.begin(), host.end(), uint16_t{0});
  std::fill(mask.begin(), mask.end(), -1);
  for (int token = 0; token < codes.vocab_size; ++token) {
    mask[token] = token;
    for (int dim = 0; dim < codes.width; ++dim) {
      uint32_t bits = std::bit_cast<uint32_t>(
          static_cast<float>(scale * codes.values[token * codes.width + dim]));
      bits += 0x7fff + ((bits >> 16) & 1);
      host[token * codes.width + dim] = static_cast<uint16_t>(bits >> 16);
    }
  }
  ASSIGN_OR_RETURN(auto input, Buffer::Allocate(executor, host.size_bytes()));
  ASSIGN_OR_RETURN(auto targets, Buffer::Allocate(executor, mask.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(input.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload ideal code vectors"));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(targets.data(), mask.data(), mask.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload oracle mask"));
  ASSIGN_OR_RETURN(auto normalized, norm.fwd(executor, {&input, 1}, nullptr));
  ASSIGN_OR_RETURN(auto logits, head.fwd(executor, normalized.outputs, nullptr));
  ASSIGN_OR_RETURN(auto ids, ExtractTop1Ids(executor, logits.outputs[0],
                                            targets, codes.vocab_size));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(mask.data(), ids.data(), mask.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download code oracle"));
  RETURN_IF_ERROR(executor.Synchronize());
  int correct = 0;
  for (int token = 0; token < codes.vocab_size; ++token)
    correct += mask[token] == token;
  return correct;
}

template <class T>
void Append(std::vector<T>& output, const std::vector<T>& input) {
  output.insert(output.end(), input.begin(), input.end());
}

// Inspect the requested final residual code, not just branch-update error:
// cancelling the incoming residual can dominate the latter's target norm.
// These diagnostics use real-valued fitted coefficients, not BF16 inference.
absl::Status ReportCodeError(util::TeeStream& report, const std::string& name,
                             const ClosedFormMap& map,
                             const FinalMlpBatch& captured,
                             const TokenCodes& codes,
                             const LabelProjectionOptions& options) {
  ASSIGN_OR_RETURN(auto update, ApplyClosedFormMap(map, captured.features));
  double error[2]{}, desired_energy[2]{}, cosine[2]{};
  size_t rows[2]{}, zero_rows[2]{};
  for (size_t row = 0; row < captured.labels.size(); ++row) {
    const int held =
        options.held_sentence_stride != 0 &&
        captured.sample_indices[row] % options.held_sentence_stride == 0;
    ++rows[held];
    double mean = 0;
    for (int dim = 0; dim < codes.width; ++dim)
      mean += update[row * codes.width + dim] +
              captured.residuals[row * codes.width + dim];
    mean /= codes.width;
    double dot = 0, predicted_norm = 0, desired_norm = 0;
    for (int dim = 0; dim < codes.width; ++dim) {
      const double predicted = update[row * codes.width + dim] +
                               captured.residuals[row * codes.width + dim] -
                               mean;
      const double desired =
          options.code_scale *
          codes.values[static_cast<size_t>(captured.labels[row]) * codes.width +
                       dim];
      const double difference = predicted - desired;
      error[held] += difference * difference;
      dot += predicted * desired;
      predicted_norm += predicted * predicted;
      desired_norm += desired * desired;
    }
    desired_energy[held] += desired_norm;
    if (predicted_norm == 0)
      ++zero_rows[held];
    else
      cosine[held] += dot / std::sqrt(predicted_norm * desired_norm);
  }
  for (int held = 0; held < 2; ++held) {
    if (rows[held] == 0)
      continue;
    report << "# " << name << (held ? " held" : " fitting")
           << " code_relative_rmse="
           << std::sqrt(error[held] / desired_energy[held])
           << " mean_code_cosine=" << cosine[held] / rows[held]
           << " zero_predicted_rows=" << zero_rows[held]
           << " (double coefficients, before GPU rounding)\n";
  }
  return absl::OkStatus();
}

absl::Status Run() {
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path directory = absl::GetFlag(FLAGS_output_dir);
  const int width = absl::GetFlag(FLAGS_model_width);
  const int features = absl::GetFlag(FLAGS_feed_forward_width);
  const int blocks = absl::GetFlag(FLAGS_layers);
  const std::string objective = absl::GetFlag(FLAGS_projection_objective);
  const LabelProjectionOptions options{
      .held_sentence_stride = absl::GetFlag(FLAGS_fit_sentence_stride),
      .code_scale = absl::GetFlag(FLAGS_code_scale),
      .affine = {.ridge = absl::GetFlag(FLAGS_ridge)}};
  if (checkpoint.empty() || directory.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() || blocks <= 0 ||
      options.held_sentence_stride < 0 || options.held_sentence_stride == 1 ||
      !std::isfinite(options.code_scale) || options.code_scale <= 0 ||
      !std::isfinite(options.affine.ridge) || options.affine.ridge < 0)
    return absl::InvalidArgumentError("invalid label-probe arguments");
  if (objective != "code_vector" && objective != "token_margin")
    return absl::InvalidArgumentError("unknown projection_objective");
  if (objective == "token_margin" && options.held_sentence_stride != 0)
    return absl::InvalidArgumentError(
        "token_margin currently requires --fit_sentence_stride=0");
  if (objective == "token_margin" &&
      (!std::isfinite(absl::GetFlag(FLAGS_decision_coefficient_bound)) ||
       absl::GetFlag(FLAGS_decision_coefficient_bound) <= 0 ||
       absl::GetFlag(FLAGS_decision_rounds) <= 0 ||
       absl::GetFlag(FLAGS_decision_new_cuts) <= 0 ||
       absl::GetFlag(FLAGS_decision_total_cuts) <= 0 ||
       absl::GetFlag(FLAGS_decision_solve_seconds) <= 0))
    return absl::InvalidArgumentError("invalid decision solver limits");
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
  const int vocabulary = tokenizer->vocab_size();
  Gpt2Config config{.transformer_block_count = blocks,
                    .model_width = width,
                    .attention_heads = absl::GetFlag(FLAGS_attention_heads),
                    .feed_forward_width = features,
                    .vocabulary_size = vocabulary,
                    .pad_vocabulary = false};
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  std::error_code error;
  if (!fs::create_directory(directory, error))
    return absl::InvalidArgumentError(
        absl::StrCat("output_dir must be fresh: ", error.message()));
  std::ofstream file(directory / "label_projection.tsv");
  if (!file)
    return absl::UnknownError("cannot create label projection report");
  util::TeeStream report(std::cout, file);
  report
      << "# checkpoint=" << checkpoint.string()
      << "\n# corpus=" << absl::GetFlag(FLAGS_corpus)
      << "\n# tokenizer=" << absl::GetFlag(FLAGS_tokenizer)
      << "\n# model_width=" << width << "\n# feature_width=" << features
      << "\n# layers=" << blocks
      << "\n# attention_heads=" << config.attention_heads
      << "\n# batch_size=" << dataset->options().batch_size
      << "\n# prompt_tokens=" << dataset->options().prompt_tokens
      << "\n# fit_sentence_stride=" << options.held_sentence_stride
      << "\n# code_scale=" << options.code_scale
      << "\n# ridge=" << options.affine.ridge
      << "\n# code_seed=" << absl::GetFlag(FLAGS_code_seed)
      << "\n# projection_objective=" << objective
      << "\n# Code heads are independent output tables; tied input "
         "embeddings remain unchanged.\n"
      << "# Fits use corpus labels, not teacher branch outputs. Backbone "
         "remains trained on all sentences.\n"
      << "# Target accuracy is teacher-forced; greedy exactness independently "
         "feeds predictions back through EOS or the first mismatch.\n";
  FinalMlpBatch captured;
  size_t first_sample = 0;
  RETURN_IF_ERROR(dataset->Reset());
  for (size_t i = 0; i < dataset->batches_per_epoch(); ++i) {
    ASSIGN_OR_RETURN(auto batch, dataset->Next());
    ASSIGN_OR_RETURN(auto values,
                     CaptureFinalMlpBatch(*executor, *model, batch, width,
                                          features, blocks, vocabulary));
    if (i == 0) {
      captured.width = width;
      captured.feature_width = features;
      captured.finalnorm_gamma = values.finalnorm_gamma;
      captured.finalnorm_beta = values.finalnorm_beta;
    } else if (captured.finalnorm_gamma != values.finalnorm_gamma ||
               captured.finalnorm_beta != values.finalnorm_beta) {
      return absl::InternalError("final normalization changed during capture");
    }
    for (size_t row = 0; row < values.labels.size(); ++row) {
      if (values.original_predictions[row] != values.labels[row])
        return absl::FailedPreconditionError(
            "checkpoint is not fully memorized");
      values.sample_indices[row] += first_sample;
    }
    Append(captured.features, values.features);
    Append(captured.normalized_inputs, values.normalized_inputs);
    Append(captured.residuals, values.residuals);
    Append(captured.labels, values.labels);
    Append(captured.sample_indices, values.sample_indices);
    first_sample += batch.batch_size;
  }
  if (first_sample != dataset->sample_count() ||
      captured.labels.size() !=
          static_cast<size_t>(dataset->supervised_row_count()))
    return absl::InternalError("captured corpus counts disagree");
  std::vector<bool> seen(vocabulary);
  size_t fitting = 0, unsupported_held_targets = 0;
  for (size_t row = 0; row < captured.labels.size(); ++row)
    if (options.held_sentence_stride == 0 ||
        captured.sample_indices[row] % options.held_sentence_stride != 0) {
      seen[captured.labels[row]] = true;
      ++fitting;
    }
  for (size_t row = 0; row < captured.labels.size(); ++row)
    if (options.held_sentence_stride != 0 &&
        captured.sample_indices[row] % options.held_sentence_stride == 0 &&
        !seen[captured.labels[row]])
      ++unsupported_held_targets;
  report << "# supervised_rows=" << captured.labels.size()
         << "\n# fitting_rows=" << fitting
         << "\n# held_rows_whose_target_never_occurs_in_fitting="
         << unsupported_held_targets << "\n# Decoder always includes all "
         << vocabulary << " compact tokens.\n"
         << "condition\tcorrect_targets\ttargets\tgreedy_exact\tsentences\tfit_"
            "correct\tfit_targets\tfit_greedy_exact\tfit_sentences\theld_"
            "correct\theld_targets\theld_greedy_exact\theld_sentences\n";
  auto evaluate = [&](const std::string& name,
                      const Layer& candidate) -> absl::Status {
    ASSIGN_OR_RETURN(
        auto scored,
        EvaluateMlpReplacements(*executor, candidate, *dataset, {}, vocabulary));
    ASSIGN_OR_RETURN(auto greedy,
                     VerifyMlpGreedyCompletions(*executor, candidate, *dataset,
                                                {}, vocabulary));
    if (scored.targets_per_sentence.size() != dataset->sample_count() ||
        greedy.exact_per_sentence.size() != dataset->sample_count())
      return absl::InternalError("missing per-sentence label-probe scores");
    int64_t correct[2]{}, targets[2]{}, exact[2]{}, sentences[2]{};
    for (size_t i = 0; i < dataset->sample_count(); ++i) {
      const int held = options.held_sentence_stride != 0 &&
                       i % options.held_sentence_stride == 0;
      if (greedy.exact_per_sentence[i] !=
          (scored.correct_per_sentence[i] == scored.targets_per_sentence[i]))
        return absl::FailedPreconditionError(
            "greedy and teacher-forced sentence identity differs");
      correct[held] += scored.correct_per_sentence[i];
      targets[held] += scored.targets_per_sentence[i];
      exact[held] += greedy.exact_per_sentence[i];
      ++sentences[held];
    }
    report << name << '\t' << scored.correct_targets << '\t' << scored.targets
           << '\t' << greedy.exact_sentences << '\t' << greedy.sentences;
    for (int subset = 0; subset < 2; ++subset)
      report << '\t' << correct[subset] << '\t' << targets[subset] << '\t'
             << exact[subset] << '\t' << sentences[subset];
    report << '\n';
    if (name == "original" || name == "original_norm_head_clone")
      if (greedy.exact_sentences != greedy.sentences)
        return absl::FailedPreconditionError(
            "original/clone control is not exact");
    if (!file)
      return absl::UnknownError("cannot write label projection report");
    return absl::OkStatus();
  };
  RETURN_IF_ERROR(evaluate("original", *model));
  ASSIGN_OR_RETURN(auto effective,
                   CopyEffectiveBf16Readout(*executor, model->weights()[0],
                                            width, vocabulary));
  TokenCodes original_codes{width, vocabulary, effective};
  ASSIGN_OR_RETURN(auto original_head, MakeReadout(*executor, original_codes));
  ASSIGN_OR_RETURN(
      auto original_norm,
      MakeNorm(*executor, captured.finalnorm_gamma, captured.finalnorm_beta));
  ASSIGN_OR_RETURN(auto cloned,
                   CreateFinalMlpReplacement(*model, nullptr, *original_norm,
                                             *original_head.head, blocks - 1));
  RETURN_IF_ERROR(evaluate("original_norm_head_clone", *cloned));
  std::vector<float> ones(width, 1), zeros(width, 0);
  ASSIGN_OR_RETURN(auto zero_beta,
                   MakeNorm(*executor, captured.finalnorm_gamma, zeros));
  ASSIGN_OR_RETURN(auto beta_control,
                   CreateFinalMlpReplacement(*model, nullptr, *zero_beta,
                                             *original_head.head, blocks - 1));
  RETURN_IF_ERROR(evaluate("original_head_zero_final_beta", *beta_control));
  if (objective == "token_margin") {
    ASSIGN_OR_RETURN(auto decoder, MakeDecisionReadout(effective, width,
                                                       captured.finalnorm_gamma,
                                                       captured.finalnorm_beta));
    const MarginProjectionOptions decision_options{
        .coefficient_bound = absl::GetFlag(FLAGS_decision_coefficient_bound),
        .center_coefficients = true,
        .dual_simplex = absl::GetFlag(FLAGS_decision_dual_simplex),
        .max_rounds = absl::GetFlag(FLAGS_decision_rounds),
        .max_new_cuts =
            static_cast<size_t>(absl::GetFlag(FLAGS_decision_new_cuts)),
        .max_total_cuts =
            static_cast<size_t>(absl::GetFlag(FLAGS_decision_total_cuts)),
        .per_solve_timeout_seconds =
            absl::GetFlag(FLAGS_decision_solve_seconds),
        .progress = [&](const MarginProjectionProgress& progress) {
          report << "# decision round=" << progress.round
                 << " checked_round=" << progress.checked_round
                 << " cuts=" << progress.cut_count
                 << " correct=" << progress.correct_count << '/'
                 << captured.labels.size()
                 << " minimum_margin=" << progress.minimum_margin
                 << " restricted_lp_objective=" << progress.lp_objective
                 << " solver_status=" << progress.solver_status
                 << " status=" << progress.status
                 << " elapsed_seconds=" << progress.elapsed_seconds << '\n';
        }};
    report << "# Decision fit keeps original effective embedding rows and "
              "gamma; beta is zero ONLY in the linear constraint model.\n"
           << "# decision_coefficient_bound="
           << decision_options.coefficient_bound
           << "\n# decision_margin_cap=" << decision_options.margin_cap
           << "\n# decision_acceptance_tolerance="
           << decision_options.acceptance_tolerance
           << "\n# decision_center_coefficients="
           << decision_options.center_coefficients
           << "\n# decision_dual_simplex=" << decision_options.dual_simplex
           << "\n# decision_round_limit=" << decision_options.max_rounds
           << "\n# decision_new_cut_limit=" << decision_options.max_new_cuts
           << "\n# decision_total_cut_limit=" << decision_options.max_total_cuts
           << "\n# decision_solve_seconds="
           << decision_options.per_solve_timeout_seconds << '\n';
    ASSIGN_OR_RETURN(
        auto result,
        FitMarginProjection(captured.features, features, captured.residuals,
                            captured.labels, decoder.directions, width,
                            vocabulary, decision_options));
    report << "# decision_outcome="
           << MarginProjectionOutcomeName(result.outcome) << '\n';
    // The simplex result contains only a projection, not a complete model.
    // Preserve the last independently checked candidate even on a time limit,
    // with its outcome recorded above; a saved matrix is not a success claim.
    std::ofstream parameters(directory / "decision_projection.tsv");
    if (!parameters)
      return absl::UnknownError("cannot create decision projection file");
    parameters << std::setprecision(17)
               << "kind\tinput_coordinate\toutput_coordinate\tvalue\n";
    for (int input = 0; input < features; ++input)
      for (int output = 0; output < width; ++output)
        parameters
            << "weight\t" << input << '\t' << output << '\t'
            << result.weights[static_cast<size_t>(input) * width + output]
            << '\n';
    for (int output = 0; output < width; ++output)
      parameters << "bias\t-1\t" << output << '\t' << result.biases[output]
                 << '\n';
    parameters.close();
    if (!parameters)
      return absl::UnknownError("cannot write decision projection file");
    ClosedFormMap map;
    map.input_dim = result.input_dim;
    map.output_dim = result.output_dim;
    map.weights = std::move(result.weights);
    map.biases = std::move(result.biases);
    ASSIGN_OR_RETURN(auto projection, MakeProjection(*executor, map));
    ASSIGN_OR_RETURN(
        auto zero_beta_candidate,
        CreateFinalMlpReplacement(*model, projection.get(), *zero_beta,
                                  *original_head.head, blocks - 1));
    RETURN_IF_ERROR(
        evaluate("decision_fit_zero_final_beta", *zero_beta_candidate));
    // This separate check is on the unmodified learned final normalization;
    // success in the zero-beta real-valued LP does not imply success here.
    ASSIGN_OR_RETURN(
        auto original_beta_candidate,
        CreateFinalMlpReplacement(*model, projection.get(), *original_norm,
                                  *original_head.head, blocks - 1));
    RETURN_IF_ERROR(
        evaluate("decision_fit_original_final_beta", *original_beta_candidate));
    report << "# Complete\n";
    return absl::OkStatus();
  }
  ASSIGN_OR_RETURN(auto unit_norm, MakeNorm(*executor, ones, zeros));
  ASSIGN_OR_RETURN(
      auto balanced,
      MakeBalancedTokenCodes(vocabulary, width, absl::GetFlag(FLAGS_code_seed)));
  ASSIGN_OR_RETURN(auto learned, NormalizeTokenCodes(effective, width));
  ASSIGN_OR_RETURN(auto permuted,
                   PermuteTokenCodes(learned, absl::GetFlag(FLAGS_code_seed)));
  const TokenCodes* codebooks[] = {&balanced, &learned, &permuted};
  const char* names[] = {"fixed_balanced", "normalized_learned",
                         "permuted_learned"};
  for (int index = 0; index < 3; ++index) {
    const auto& codes = *codebooks[index];
    const std::string name = names[index];
    ASSIGN_OR_RETURN(auto readout, MakeReadout(*executor, codes));
    ASSIGN_OR_RETURN(auto oracle, CodeOracle(*executor, codes, *unit_norm,
                                             *readout.head, options.code_scale));
    report << "# " << name << " ideal_code_oracle=" << oracle << '/'
           << vocabulary << '\n';
    if (oracle != vocabulary)
      return absl::FailedPreconditionError(
          "codebook cannot decode every ideal code under GPU rounding");
    ASSIGN_OR_RETURN(auto head_only,
                     CreateFinalMlpReplacement(*model, nullptr, *unit_norm,
                                               *readout.head, blocks - 1));
    RETURN_IF_ERROR(evaluate(name + "_head_only", *head_only));
    ClosedFormMap zero;
    zero.input_dim = features;
    zero.output_dim = width;
    zero.weights.resize(static_cast<size_t>(features) * width);
    zero.biases.resize(width);
    ASSIGN_OR_RETURN(auto zero_projection, MakeProjection(*executor, zero));
    ASSIGN_OR_RETURN(
        auto residual_only,
        CreateFinalMlpReplacement(*model, zero_projection.get(), *unit_norm,
                                  *readout.head, blocks - 1));
    RETURN_IF_ERROR(evaluate(name + "_incoming_residual_only", *residual_only));
    ASSIGN_OR_RETURN(
        auto fit, FitTokenCodeProjection(
                      captured.features, features, captured.residuals,
                      captured.labels, captured.sample_indices, codes, options));
    report << "# " << name << " fit_rows=" << fit.sample_count
           << " rank=" << fit.numerical_rank
           << " update_relative_rmse=" << fit.relative_rmse << '\n';
    RETURN_IF_ERROR(
        ReportCodeError(report, name, fit, captured, codes, options));
    ASSIGN_OR_RETURN(auto projection, MakeProjection(*executor, fit));
    ASSIGN_OR_RETURN(auto candidate, CreateFinalMlpReplacement(
                                         *model, projection.get(), *unit_norm,
                                         *readout.head, blocks - 1));
    RETURN_IF_ERROR(evaluate(name + "_label_fit", *candidate));
  }
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
  const auto status = pluto::llm::one_shot_memorizer::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
