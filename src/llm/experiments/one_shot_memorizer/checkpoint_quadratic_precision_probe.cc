// Captures one unmodified learned forward over the corpus, then audits whether
// quadratic branch-fit error persists without BF16 product rounding, and in a
// separate penalty-free fit. All fits and diagnostic predictions run on the
// CPU. This is not an alternate model
// forward: downstream completion accuracy is not measured by this executable.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/mlp_probe.h"
#include "src/llm/experiments/one_shot_memorizer/quadratic_precision_audit.h"
#include "src/llm/experiments/one_shot_memorizer/quadratic_probe_artifacts.h"
#include "src/llm/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Original memorized width16 checkpoint");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "The original 1,024-fact corpus");
ABSL_FLAG(
    std::string, reference_report, "",
    "Completed quadratic experiment directory containing coefficients.tsv");
ABSL_FLAG(std::string, output_dir, "",
          "Fresh directory for precision-audit reports");
ABSL_FLAG(int, batch_size, 32,
          "Sentences per capture batch, from 1 through 32");

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr int kWidth = 16;
constexpr int kFeatures = 152;
constexpr int kBlocks = 8;
constexpr int kVocabulary = 4475;

absl::StatusOr<std::vector<std::vector<uint8_t>>> SnapshotWeights(
    cuda::Executor& executor, const Layer& model) {
  std::vector<std::vector<uint8_t>> result;
  for (const Buffer& weight : model.weights()) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                    executor, weight.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "snapshot precision-audit source weights"));
    RETURN_IF_ERROR(executor.Synchronize());
    result.emplace_back(host.begin(), host.end());
  }
  return result;
}

void WriteErrors(std::ostream& stream, int block,
                 absl::string_view feature_mode,
                 absl::string_view parameter_mode,
                 const QuadraticPrecisionEvaluation& evaluation) {
  for (bool held : {false, true}) {
    const auto& error = held ? evaluation.held : evaluation.fitting;
    stream << block << '\t' << feature_mode << '\t' << parameter_mode << '\t'
           << (held ? "held" : "fit") << '\t' << error.rows << '\t'
           << error.rmse << '\t' << error.relative_rmse << '\t'
           << error.centered_relative_rmse << '\n';
  }
}

absl::Status CheckParity(std::ostream& report, int block,
                         const ClosedFormMap& fitted,
                         const QuadraticCoefficients& reference) {
  const absl::Span<const double> actual_parts[] = {fitted.weights,
                                                   fitted.biases};
  const absl::Span<const float> reference_parts[] = {reference.weights,
                                                     reference.bias};
  const char* names[] = {"W2", "b2"};
  bool all_equal = true;
  for (int part = 0; part < 2; ++part) {
    if (actual_parts[part].size() != reference_parts[part].size())
      return absl::DataLossError(
          "reference quadratic coefficient shape differs");
    size_t equal = 0;
    double maximum_difference = 0;
    for (size_t coordinate = 0; coordinate < actual_parts[part].size();
         ++coordinate) {
      // AuditQuadraticPrecision already range-checks FP32 coefficient casts.
      const float actual = static_cast<float>(actual_parts[part][coordinate]);
      const float expected = reference_parts[part][coordinate];
      equal +=
          std::bit_cast<uint32_t>(actual) == std::bit_cast<uint32_t>(expected);
      maximum_difference = std::max(
          maximum_difference, std::abs(static_cast<double>(actual) - expected));
    }
    all_equal &= equal == actual_parts[part].size();
    report << block << '\t' << names[part] << '\t' << actual_parts[part].size()
           << '\t' << equal << '\t' << maximum_difference << '\n';
  }
  if (!all_equal)
    return absl::FailedPreconditionError(absl::StrCat(
        "rounded-feature coefficients differ from original report at block ",
        block));
  return absl::OkStatus();
}

void WriteUnregularized(
    std::ostream& fits, std::ostream& errors, std::ostream& coefficients,
    int block, size_t fitting_rows,
    const absl::StatusOr<QuadraticUnregularizedFit>& diagnostic) {
  // Preserve the primary reports verbatim: penalty-free diagnostics have
  // separate files. Rank failure is an observed outcome, not a retry trigger.
  fits << block << '\t' << (diagnostic.ok() ? "OK" : "FAILED_PRECONDITION")
       << '\t' << diagnostic.status().message() << '\t'
       << AffineMapOptions{}.relative_rank_tolerance << '\t' << fitting_rows;
  if (!diagnostic.ok()) {
    fits << "\tNA\tNA\tNA\tNA\tNA\tNA\tNA\n";
    return;
  }
  const auto& map = diagnostic->fitted_map;
  const auto [min_qr, max_qr] = std::minmax_element(
      map.qr_diagonal_magnitudes.begin(), map.qr_diagonal_magnitudes.end());
  const auto [min_weight, max_weight] =
      std::minmax_element(map.weights.begin(), map.weights.end());
  const auto [min_bias, max_bias] =
      std::minmax_element(map.biases.begin(), map.biases.end());
  fits << '\t' << map.numerical_rank << '\t' << *min_qr << '\t' << *max_qr
       << '\t' << *min_weight << '\t' << *max_weight << '\t' << *min_bias
       << '\t' << *max_bias << '\n';
  WriteErrors(errors, block, "fp32_products", "fp64", diagnostic->evaluation);
  const absl::Span<const double> parts[] = {map.weights, map.biases};
  const char* tensors[] = {"W2", "b2"};
  for (int part = 0; part < 2; ++part)
    for (size_t coordinate = 0; coordinate < parts[part].size(); ++coordinate)
      coefficients << block << '\t' << tensors[part] << '\t' << coordinate
                   << '\t' << parts[part][coordinate] << '\n';
}

absl::Status Run() {
  namespace fs = std::filesystem;
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path reference_report = absl::GetFlag(FLAGS_reference_report);
  const fs::path output_dir = absl::GetFlag(FLAGS_output_dir);
  const int batch_size = absl::GetFlag(FLAGS_batch_size);
  if (checkpoint.empty() || reference_report.empty() || output_dir.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() || batch_size < 1 ||
      batch_size > 32)
    return absl::InvalidArgumentError(
        "checkpoint, tokenizer, reference_report, fresh output_dir required; "
        "batch_size must be in [1,32]");
  std::error_code error;
  if (fs::exists(output_dir, error))
    return absl::AlreadyExistsError("output_dir must be fresh");
  if (error)
    return absl::UnknownError(error.message());
  std::ifstream reference_file(reference_report / "coefficients.tsv");
  if (!reference_file)
    return absl::NotFoundError("cannot open reference coefficients.tsv");
  ASSIGN_OR_RETURN(auto reference, ReadQuadraticCoefficients(reference_file));
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto tokenizer,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint / "compact_vocabulary.tsv"));
  if (tokenizer->vocab_size() != kVocabulary ||
      tokenizer->eos_token_id() != 4474 ||
      tokenizer->original_eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError("unexpected compact vocabulary");
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto dataset, PaddedLineDataSetIterator::Create(
                                     *executor, corpus.text(), *tokenizer,
                                     {.batch_size = batch_size,
                                      .context_length = kGpt2ContextLength,
                                      .prompt_tokens = 5,
                                      .eos_token = tokenizer->eos_token_id()}));
  if (dataset->sample_count() != 1024)
    return absl::InvalidArgumentError("expected the original 1024 sentences");
  std::vector<size_t> lengths;
  size_t rows = 0, fitting_rows = 0;
  for (size_t sentence = 0; sentence < dataset->sample_count(); ++sentence) {
    const size_t length = dataset->sample_tokens(sentence).size();
    lengths.push_back(length);
    rows += length;
    if (sentence % 5 != 0)
      fitting_rows += length;
  }
  if (rows != 14098 || fitting_rows != 11285)
    return absl::FailedPreconditionError("unexpected corpus row accounting");
  const Gpt2Config config{.transformer_block_count = kBlocks,
                          .model_width = kWidth,
                          .attention_heads = 1,
                          .feed_forward_width = 64,
                          .vocabulary_size = kVocabulary,
                          .pad_vocabulary = false};
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  ASSIGN_OR_RETURN(auto before, SnapshotWeights(*executor, *model));
  if (!fs::create_directory(output_dir, error) || error)
    return absl::UnknownError(
        absl::StrCat("cannot create output directory: ", error.message()));
  std::ofstream manifest(output_dir / "manifest.tsv"),
      errors(output_dir / "errors.tsv"), fits(output_dir / "fits.tsv"),
      coefficients(output_dir / "coefficients.tsv"),
      parity(output_dir / "parity.tsv"),
      unregularized_fits(output_dir / "ridge_zero_unrounded_fits.tsv"),
      unregularized_errors(output_dir / "ridge_zero_unrounded_errors.tsv"),
      unregularized_coefficients(output_dir /
                                 "ridge_zero_unrounded_coefficients.tsv");
  for (auto* stream :
       {&manifest, &errors, &fits, &coefficients, &parity, &unregularized_fits,
        &unregularized_errors, &unregularized_coefficients}) {
    if (!*stream)
      return absl::UnknownError("cannot create precision audit reports");
    *stream << std::setprecision(17);
  }
  manifest
      << "key\tvalue\ncompleted\tfalse\ncheckpoint\t" << checkpoint.string()
      << "\nreference_report\t" << reference_report.string() << "\ntokenizer\t"
      << absl::GetFlag(FLAGS_tokenizer) << "\ncorpus\t"
      << absl::GetFlag(FLAGS_corpus) << "\nbatch_size\t" << batch_size
      << "\nblocks\t8\ninput_width\t16\nfeatures\t152\noutput_width\t16"
         "\nreal_rows\t14098\nfit_rows\t11285\nheld_rows\t2813"
         "\nfit_sentences\t819\nheld_sentences\t205\nridge\t0.000001"
         "\nheld_definition\tzero-based sentence index divisible by five"
         "\nfeatures\tlinear16 then raw upper-triangle136; no scaling"
         "\nfit_target\tlearned physical BF16 branch updates"
         "\nfit_policy\tindependent FP64 ridge fit for each feature precision"
         "\nridge_zero_unrounded\tseparate FP64 QR fit on the same unrounded "
         "fitting rows; no penalty, coefficient rounding, or tolerance retry"
         "\nridge_zero_relative_rank_tolerance\t1e-12"
         "\nprecision_policies\tFP64 coefficients; FP32 coefficients; "
         "BF16 weights with FP32 bias"
         "\nevaluation\tCPU FP64 dot products; NO output rounding or GPU MMA "
         "emulation"
         "\nparity\trounded-feature fitted FP32 coefficients must match "
         "reference W2/b2 bits"
         "\nscope\tconditional branch approximation; retained learned inputs "
         "and targets; "
         "no downstream completion claim\n";
  manifest.flush();
  errors << "block\tfeature_precision\tparameter_"
            "precision\tgroup\trows\trmse\trelative_rmse\tcentered_relative_"
            "rmse\n";
  fits << "block\tfeature_precision\tfit_rows\trank\trmse\trelative_rmse\tqr_"
          "diagonal_spread\n";
  coefficients
      << "block\tfeature_precision\ttensor\tflat_index\tfp64_coefficient\n";
  parity << "block\ttensor\tcoordinates\tbitwise_equal_fp32\tmaximum_absolute_"
            "difference\n";
  unregularized_fits
      << "block\tstatus\tstatus_message\trelative_rank_tolerance\tfit_rows\t"
         "numerical_rank\tqr_min\tqr_max\tweight_min\tweight_max\tbias_min\t"
         "bias_max\n";
  unregularized_errors
      << "block\tfeature_precision\tparameter_precision\tgroup\trows\trmse\t"
         "relative_rmse\tcentered_relative_rmse\n";
  unregularized_coefficients << "block\ttensor\tflat_index\tfp64_coefficient\n";
  std::cout << "Capturing one unmodified corpus pass..." << std::endl;
  ASSIGN_OR_RETURN(auto capture,
                   CaptureGpt2Mlps(*executor, *model, *dataset, kWidth, 64,
                                   kBlocks, kVocabulary));
  if (capture.blocks.size() != kBlocks || capture.sentences != 1024 ||
      capture.target_count != 10002 || capture.correct_targets != 10002 ||
      capture.exact_sentences != 1024)
    return absl::FailedPreconditionError(
        "original capture accuracy/control failed");
  ASSIGN_OR_RETURN(auto after, SnapshotWeights(*executor, *model));
  if (before != after)
    return absl::FailedPreconditionError(
        "capture changed original model weight bytes");
  manifest << "capture_targets\t10002/10002\ncapture_sentences\t1024/1024\n"
              "source_master_bytes_unchanged\tPASS\n";
  manifest.flush();
  std::cout << "GPU capture and source-byte controls complete; CPU fits only."
            << std::endl;
  // Everything after the capture and byte check is CPU-only. Reference W2/b2
  // are used exclusively for parity, never supplied to the regression solver.
  for (int block = 0; block < kBlocks; ++block) {
    const auto started = std::chrono::steady_clock::now();
    const auto& samples = capture.blocks[block];
    ASSIGN_OR_RETURN(auto audit,
                     AuditQuadraticPrecision(samples.normalized_inputs,
                                             samples.outputs, lengths));
    if (audit.fitting_sentences != 819 || audit.held_sentences != 205)
      return absl::InternalError("precision audit sentence split changed");
    const QuadraticPrecisionVariant* variants[] = {&audit.unrounded_products,
                                                   &audit.bf16_products};
    const char* names[] = {"fp32_products", "bf16_products"};
    for (int variant = 0; variant < 2; ++variant) {
      const auto& value = *variants[variant];
      const auto& map = value.fitted_map;
      if (map.sample_count != fitting_rows || map.input_dim != kFeatures ||
          map.output_dim != kWidth)
        return absl::InternalError("precision fit dimensions changed");
      const auto [minimum, maximum] = std::minmax_element(
          map.qr_diagonal_magnitudes.begin(), map.qr_diagonal_magnitudes.end());
      fits << block << '\t' << names[variant] << '\t' << map.sample_count
           << '\t' << map.numerical_rank << '\t' << map.rmse << '\t'
           << map.relative_rmse << '\t' << *maximum / *minimum << '\n';
      WriteErrors(errors, block, names[variant], "fp64", value.fp64_parameters);
      WriteErrors(errors, block, names[variant], "fp32", value.fp32_parameters);
      WriteErrors(errors, block, names[variant], "bf16_weights_fp32_bias",
                  value.bf16_weights_fp32_bias);
      const absl::Span<const double> parts[] = {map.weights, map.biases};
      const char* tensors[] = {"W2", "b2"};
      for (int part = 0; part < 2; ++part)
        for (size_t coordinate = 0; coordinate < parts[part].size();
             ++coordinate)
          coefficients << block << '\t' << names[variant] << '\t'
                       << tensors[part] << '\t' << coordinate << '\t'
                       << parts[part][coordinate] << '\n';
    }
    RETURN_IF_ERROR(CheckParity(parity, block, audit.bf16_products.fitted_map,
                                reference[block]));
    WriteUnregularized(unregularized_fits, unregularized_errors,
                       unregularized_coefficients, block, fitting_rows,
                       audit.ridge_zero_unrounded);
    for (auto* stream :
         {&errors, &fits, &coefficients, &parity, &unregularized_fits,
          &unregularized_errors, &unregularized_coefficients}) {
      stream->flush();
      if (!*stream)
        return absl::UnknownError("precision audit report write failed");
    }
    std::cout << "block " << block
              << ": held relative error without product rounding="
              << audit.unrounded_products.fp64_parameters.held.relative_rmse
              << ", with BF16 products="
              << audit.bf16_products.fp64_parameters.held.relative_rmse
              << "; rounded-fit parity PASS; "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - started)
                     .count()
              << " s" << std::endl;
    if (audit.ridge_zero_unrounded.ok())
      std::cout << "  ridge-zero unrounded: fit relative error="
                << audit.ridge_zero_unrounded->evaluation.fitting.relative_rmse
                << ", held relative error="
                << audit.ridge_zero_unrounded->evaluation.held.relative_rmse
                << std::endl;
    else
      std::cout << "  ridge-zero unrounded: "
                << audit.ridge_zero_unrounded.status().message() << std::endl;
  }
  manifest << "rounded_feature_reference_parity\tPASS\ncompleted\ttrue\n";
  manifest.flush();
  return manifest ? absl::OkStatus()
                  : absl::UnknownError("manifest write failed");
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
