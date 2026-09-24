// Replay actual shuffled training batches to find the first vocabulary-
// permutation equivariance failure. This instruments production kernels;
// it does not substitute a reference forward/backward implementation.
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/batch_validation.h"
#include "src/llm/experiments/memorize_general_facts/permutation_trace/trace_util.h"
#include "src/llm/experiments/memorize_general_facts/token_corpus.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, run_dir, "",
          "Pinned identical-embedding experiment directory");
ABSL_FLAG(std::string, output_dir, "", "Fresh external trace output directory");
ABSL_FLAG(int, max_steps, 32,
          "Maximum actual shuffled optimizer steps to replay");
ABSL_FLAG(int, support, 2,
          "Select rename_000_NNNN trial by permutation support");
ABSL_FLAG(bool, canonical_token_order, false,
          "Use the saved permutation as the renamed model/loss reduction "
          "order, and require exact aligned equality throughout the replay");

namespace pluto::llm::permutation_trace {
namespace {

constexpr int kBatch = 32;
constexpr int kContext = 27;
constexpr int kWidth = 10;
constexpr int kVocabulary = 4475;
constexpr int kPaddedVocabulary = 4480;
constexpr int kSeed = 1337;

absl::StatusOr<std::string> ReadFile(const std::filesystem::path& file) {
  std::ifstream input(file, std::ios::binary);
  if (!input)
    return absl::NotFoundError(absl::StrCat("cannot read ", file.string()));
  std::string text((std::istreambuf_iterator<char>(input)), {});
  if (input.bad())
    return absl::DataLossError(absl::StrCat("failed reading ", file.string()));
  return text;
}

// Match the production experiment's double intermediate and final float cast.
float LearningRate(int step) {
  const double peak = 0.0012;
  if (step <= 100)
    return peak * step / 100;
  const double progress =
      std::clamp(static_cast<double>(step - 100) / 119900, 0.0, 1.0);
  return peak *
         (0.1 + 0.9 * (1 + std::cos(3.14159265358979323846 * progress)) / 2);
}

struct CapturedTensor {
  TensorDescription description;
  cuda::PageLockedHostArray<uint8_t> bytes;
};

// Every snapshot queues its own pinned D2H copy before a later kernel could
// mutate/reuse the source. Synchronize once after the complete training step;
// no callback changes activations or inserts a replacement GPU computation.
class Recorder {
 public:
  Recorder() {
    hooks_.enter_combinator = [this](cuda::Executor&, absl::string_view name) {
      scopes_.emplace_back(name);
      return absl::OkStatus();
    };
    hooks_.exit_combinator = [this](cuda::Executor&) {
      if (scopes_.empty())
        return absl::InternalError("unbalanced combinator hook scopes");
      scopes_.pop_back();
      return absl::OkStatus();
    };
    hooks_.activation_hook = [this](cuda::Executor& executor,
                                    absl::string_view name,
                                    absl::Span<const ActivationType> types,
                                    absl::Span<Buffer> buffers) {
      return Hook(executor, "fwd", name, types, buffers, false);
    };
    hooks_.gradient_hook = [this](cuda::Executor& executor,
                                  absl::string_view name,
                                  absl::Span<const ActivationType> types,
                                  absl::Span<Buffer> buffers) {
      return Hook(executor, "bwd", name, types, buffers, true);
    };
  }

  LayerHooks& hooks() { return hooks_; }
  const std::vector<CapturedTensor>& tensors() const { return tensors_; }

  absl::Status Capture(cuda::Executor& executor, TensorDescription description,
                       const Buffer& buffer) {
    ASSIGN_OR_RETURN(auto bytes, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                     executor, buffer.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(bytes.data(), buffer.data(), buffer.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "cudaMemcpyAsync(trace snapshot)"));
    tensors_.push_back({std::move(description), std::move(bytes)});
    return absl::OkStatus();
  }

 private:
  absl::Status Hook(cuda::Executor& executor, absl::string_view phase,
                    absl::string_view name,
                    absl::Span<const ActivationType> types,
                    absl::Span<const Buffer> buffers, bool gradient) {
    if (types.size() != buffers.size())
      return absl::InternalError("hook type/buffer count mismatch");
    std::string path(phase);
    for (const auto& scope : scopes_)
      absl::StrAppend(&path, "/", scope);
    absl::StrAppend(&path, "/", name);
    for (size_t i = 0; i < buffers.size(); ++i) {
      const DataType type = gradient ? DataType::FP32 : types[i].data_type();
      std::string dtype;
      if (type == DataType::BF16)
        dtype = "bf16";
      else if (type == DataType::FP32)
        dtype = "fp32";
      else if (type == DataType::INT32)
        dtype = "int32";
      else
        return absl::UnimplementedError("trace encountered unexpected dtype");
      std::vector<int64_t> shape(types[i].dimensions().begin(),
                                 types[i].dimensions().end());
      for (auto& dimension : shape)
        if (dimension == ActivationType::kBatchDimension)
          dimension = kBatch;
      const std::string alignment =
          !shape.empty() && shape.back() == kPaddedVocabulary ? "vocab_columns"
                                                              : "none";
      RETURN_IF_ERROR(Capture(
          executor,
          {absl::StrCat(path, "/", i), dtype, std::move(shape), alignment},
          buffers[i]));
    }
    return absl::OkStatus();
  }

  std::vector<std::string> scopes_;
  std::vector<CapturedTensor> tensors_;
  LayerHooks hooks_;
};

// Dense matrices use [input_dimension, output_dimension], matching the actual
// row-major storage, not the transposed mathematical notation sometimes used.
std::vector<TensorDescription> ParameterDescriptions() {
  std::vector<TensorDescription> descriptions{
      {"token_embedding", "fp32", {kVocabulary, kWidth}, "vocab_rows"},
      {"position_embedding", "fp32", {kContext, kWidth}}};
  for (int block = 0; block < 4; ++block) {
    const std::string p = absl::StrCat("block_", block, "/");
    for (const auto& description : std::vector<TensorDescription>{
             {p + "attention_norm/scale", "fp32", {kWidth}},
             {p + "attention_norm/bias", "fp32", {kWidth}},
             {p + "qkv/matrix", "fp32", {kWidth, 3 * kWidth}},
             {p + "qkv/bias", "fp32", {3 * kWidth}},
             {p + "attention_projection/matrix", "fp32", {kWidth, kWidth}},
             {p + "attention_projection/bias", "fp32", {kWidth}},
             {p + "mlp_norm/scale", "fp32", {kWidth}},
             {p + "mlp_norm/bias", "fp32", {kWidth}},
             {p + "mlp_up/matrix", "fp32", {kWidth, 2 * kWidth}},
             {p + "mlp_up/bias", "fp32", {2 * kWidth}},
             {p + "mlp_down/matrix", "fp32", {2 * kWidth, kWidth}},
             {p + "mlp_down/bias", "fp32", {kWidth}}})
      descriptions.push_back(description);
  }
  descriptions.push_back({"final_norm/scale", "fp32", {kWidth}});
  descriptions.push_back({"final_norm/bias", "fp32", {kWidth}});
  return descriptions;
}

absl::Status CaptureParameters(cuda::Executor& executor, Recorder& recorder,
                               absl::string_view phase,
                               absl::Span<const Buffer> parameters) {
  const auto descriptions = ParameterDescriptions();
  if (parameters.size() != descriptions.size() + 1 ||
      parameters.front().data() != parameters.back().data())
    return absl::FailedPreconditionError("unexpected GPT-2 parameter layout");
  for (size_t i = 0; i < descriptions.size(); ++i) {
    auto description = descriptions[i];
    description.stage = absl::StrCat(phase, "/", description.stage);
    RETURN_IF_ERROR(
        recorder.Capture(executor, std::move(description), parameters[i]));
  }
  return absl::OkStatus();
}

struct ModelRun {
  std::unique_ptr<Layer> model;
  std::unique_ptr<Layer> loss;
  std::unique_ptr<AdamWOptimizer> optimizer;
  std::unique_ptr<PaddedLineDataSetIterator> data;
};

absl::StatusOr<ModelRun> CreateRun(cuda::Executor& executor,
                                   const tokenizer::Tokenizer& tokenizer,
                                   absl::string_view corpus,
                                   absl::Span<const int32_t> token_order = {}) {
  const Gpt2Config config{.transformer_block_count = 4,
                          .model_width = kWidth,
                          .attention_heads = 1,
                          .feed_forward_width = 2 * kWidth,
                          .vocabulary_size = kVocabulary,
                          .pad_vocabulary = false,
                          .context_length = kContext,
                          .identical_token_embeddings = true};
  const PaddedLineDataSetOptions options{.batch_size = kBatch,
                                         .context_length = kContext,
                                         .prompt_tokens = 5,
                                         .eos_token = kVocabulary - 1,
                                         .shuffle = true,
                                         .seed = kSeed};
  ASSIGN_OR_RETURN(auto data, PaddedLineDataSetIterator::Create(
                                  executor, corpus, tokenizer, options));
  if (data->sample_count() != 1024 || data->supervised_row_count() != 10002)
    return absl::FailedPreconditionError(
        "unexpected experiment corpus dimensions");
  ASSIGN_OR_RETURN(auto model, CreateGpt2(executor, DataType::BF16, kSeed,
                                          config, token_order));
  ASSIGN_OR_RETURN(auto loss, CrossEntropyLossLayer::Create(
                                  executor, kVocabulary, DataType::BF16,
                                  kContext, token_order));
  ASSIGN_OR_RETURN(auto optimizer,
                   AdamWOptimizer::Create(executor, *model,
                                          {.learning_rate = LearningRate(1),
                                           .beta1 = 0.9f,
                                           .beta2 = 0.99f,
                                           .epsilon = 1e-8f,
                                           .weight_decay = 0.0f}));
  return ModelRun{std::move(model), std::move(loss), std::move(optimizer),
                  std::move(data)};
}

absl::Status RunStep(cuda::Executor& executor, ModelRun& run, int step,
                     Recorder& recorder) {
  ASSIGN_OR_RETURN(auto batch, run.data->Next());
  RETURN_IF_ERROR(
      ValidateTrainingBatch(executor, *run.model, *run.loss, batch));
  RETURN_IF_ERROR(recorder.Capture(
      executor, {"input_ids", "int32", {kBatch, kContext}, "token_ids"},
      batch.inputs));
  RETURN_IF_ERROR(recorder.Capture(
      executor, {"targets", "int32", {kBatch, kContext}, "token_ids"},
      batch.targets));
  RETURN_IF_ERROR(CaptureParameters(executor, recorder, "weights_before",
                                    run.model->weights()));
  ASSIGN_OR_RETURN(auto forward,
                   run.model->fwd(executor, {batch.inputs}, &recorder.hooks()));
  ASSIGN_OR_RETURN(auto loss_forward,
                   run.loss->fwd(executor, {forward.outputs[0], batch.targets}));
  RETURN_IF_ERROR(recorder.Capture(executor,
                                   {"loss/fwd", "fp32", {kBatch, kContext}},
                                   loss_forward.outputs[0]));
  ASSIGN_OR_RETURN(auto gradients,
                   run.loss->bwd(executor, {}, std::move(loss_forward.state)));
  RETURN_IF_ERROR(recorder.Capture(executor,
                                   {"loss/dlogits",
                                    "fp32",
                                    {kBatch, kContext, kPaddedVocabulary},
                                    "vocab_columns"},
                                   gradients[0]));
  ASSIGN_OR_RETURN(auto unused,
                   run.model->bwd(executor, gradients, std::move(forward.state),
                                  &recorder.hooks()));
  (void)unused;
  RETURN_IF_ERROR(CaptureParameters(executor, recorder, "parameter_gradients",
                                    run.model->gradients()));
  RETURN_IF_ERROR(run.optimizer->SetLearningRate(LearningRate(step)));
  RETURN_IF_ERROR(run.optimizer->ApplyStep());
  RETURN_IF_ERROR(CaptureParameters(executor, recorder, "weights_after",
                                    run.model->weights()));
  return executor.Synchronize();
}

absl::Status WriteTrace(const Recorder& recorder,
                        const std::filesystem::path& directory) {
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error)
    return absl::InternalError(error.message());
  std::ofstream manifest(directory / "tensors.tsv");
  manifest << "sequence\tstage\tdtype\tshape\talignment\tfile\tbytes\n";
  for (size_t i = 0; i < recorder.tensors().size(); ++i) {
    const auto& tensor = recorder.tensors()[i];
    const auto file = absl::StrFormat("%05d.bin", i);
    std::ofstream output(directory / file, std::ios::binary);
    output.write(reinterpret_cast<const char*>(tensor.bytes.data()),
                 tensor.bytes.size());
    output.close();
    if (!output)
      return absl::InternalError("writing tensor snapshot failed");
    const auto& d = tensor.description;
    manifest << i << '\t' << d.stage << '\t' << d.dtype << '\t'
             << ShapeText(d.shape) << '\t' << d.alignment << '\t' << file
             << '\t' << tensor.bytes.size() << '\n';
  }
  manifest.close();
  return manifest ? absl::OkStatus()
                  : absl::InternalError("writing trace manifest failed");
}

absl::Status CheckInitialization(cuda::Executor& executor, ModelRun& run,
                                 const std::filesystem::path& checkpoint) {
  Recorder recorder;
  RETURN_IF_ERROR(
      CaptureParameters(executor, recorder, "initial", run.model->weights()));
  RETURN_IF_ERROR(executor.Synchronize());
  for (size_t i = 0; i < recorder.tensors().size(); ++i) {
    ASSIGN_OR_RETURN(auto saved,
                     ReadFile(checkpoint / absl::StrCat("weight_", i, ".bin")));
    const auto& tensor = recorder.tensors()[i];
    if (saved.size() != tensor.bytes.size() ||
        std::memcmp(saved.data(), tensor.bytes.data(), saved.size()) != 0)
      return absl::FailedPreconditionError(
          absl::StrCat("initial checkpoint mismatch at weight ", i));
  }
  const auto& embedding = recorder.tensors()[0].bytes;
  for (int row = 1; row < kVocabulary; ++row)
    if (std::memcmp(embedding.data(),
                    embedding.data() + row * kWidth * sizeof(float),
                    kWidth * sizeof(float)) != 0)
      return absl::FailedPreconditionError(
          "initial embedding rows are not identical");
  return absl::OkStatus();
}

absl::Status Run() {
  const std::filesystem::path root(absl::GetFlag(FLAGS_run_dir));
  const std::filesystem::path output(absl::GetFlag(FLAGS_output_dir));
  const int max_steps = absl::GetFlag(FLAGS_max_steps);
  const int support = absl::GetFlag(FLAGS_support);
  const bool canonical_token_order = absl::GetFlag(FLAGS_canonical_token_order);
  if (root.empty() || output.empty() || max_steps <= 0 || max_steps > 256 ||
      support < 2)
    return absl::InvalidArgumentError(
        "require run_dir, fresh output_dir, 1..256 max_steps, support>=2");
  if (std::endian::native != std::endian::little)
    return absl::UnimplementedError("trace format requires little-endian host");
  std::error_code error;
  if (std::filesystem::exists(output, error))
    return absl::AlreadyExistsError("output directory must be fresh");
  if (error)
    return absl::InternalError(error.message());
  const auto renamed_dir = root / absl::StrFormat("rename_000_%04d", support);
  ASSIGN_OR_RETURN(auto permutation_text,
                   ReadFile(renamed_dir / "permutation.tsv"));
  ASSIGN_OR_RETURN(
      auto permutation,
      ParsePermutation(permutation_text, kVocabulary, kVocabulary - 1));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto corpus,
                   LoadTextCorpus((root / "inputs/corpus.txt").string()));
  ASSIGN_OR_RETURN(auto base,
                   tokenizer::Gpt2Tokenizer::Load(root / "inputs/tokenizer"));
  ASSIGN_OR_RETURN(auto mapping,
                   tokenizer::BuildCompactVocabularyMapping(
                       *executor, *base, corpus.text(), base->eos_token_id()));
  ASSIGN_OR_RETURN(auto compact, tokenizer::CompactVocabularyTokenizer::Create(
                                     *base, std::move(mapping)));
  if (compact->vocab_size() != kVocabulary ||
      compact->eos_token_id() != kVocabulary - 1)
    return absl::FailedPreconditionError("unexpected compact vocabulary");
  ASSIGN_OR_RETURN(auto base_tokens, ReadFile(root / "inputs/base_tokens.tsv"));
  ASSIGN_OR_RETURN(auto renamed_tokens, ReadFile(renamed_dir / "tokens.tsv"));
  ASSIGN_OR_RETURN(auto baseline_tokenizer,
                   memorize_general_facts::TokenCorpusTokenizer::Create(
                       *executor, *compact, corpus.text(), base_tokens,
                       compact->eos_token_id()));
  ASSIGN_OR_RETURN(auto renamed_tokenizer,
                   memorize_general_facts::TokenCorpusTokenizer::Create(
                       *executor, *compact, corpus.text(), renamed_tokens,
                       compact->eos_token_id()));
  ASSIGN_OR_RETURN(auto baseline,
                   CreateRun(*executor, *baseline_tokenizer, corpus.text()));
  ASSIGN_OR_RETURN(auto repeat,
                   CreateRun(*executor, *baseline_tokenizer, corpus.text()));
  ASSIGN_OR_RETURN(
      auto renamed,
      CreateRun(*executor, *renamed_tokenizer, corpus.text(),
                canonical_token_order ? absl::Span<const int32_t>(permutation)
                                      : absl::Span<const int32_t>()));
  const auto initial_checkpoint = root / "baseline/checkpoints/layers_4/step_0";
  RETURN_IF_ERROR(CheckInitialization(*executor, baseline, initial_checkpoint));
  RETURN_IF_ERROR(CheckInitialization(*executor, repeat, initial_checkpoint));
  RETURN_IF_ERROR(CheckInitialization(*executor, renamed, initial_checkpoint));
  std::filesystem::create_directories(output, error);
  if (error)
    return absl::InternalError(error.message());
  std::filesystem::copy_file(renamed_dir / "permutation.tsv",
                             output / "permutation.tsv", error);
  if (error)
    return absl::InternalError(error.message());
  std::ofstream summary(output / "summary.tsv");
  // Raw (unaligned) differences locate when a renamed token first occurs and
  // when its embedding rows first become distinct, before numeric divergence.
  std::ofstream batches(output / "batches.tsv");
  batches << "step\tchanged_input_ids\tchanged_target_ids\t"
             "unaligned_embedding_values_before\n";
  summary << "step\tsequence\tstage\trepeat_mismatches\trenamed_"
             "mismatches\tmax_abs\tl2\talignment\n";
  summary << std::setprecision(17);
  for (int step = 1; step <= max_steps; ++step) {
    Recorder a, b, c;
    RETURN_IF_ERROR(RunStep(*executor, baseline, step, a));
    RETURN_IF_ERROR(RunStep(*executor, repeat, step, b));
    RETURN_IF_ERROR(RunStep(*executor, renamed, step, c));
    if (a.tensors().size() != b.tensors().size() ||
        a.tensors().size() != c.tensors().size())
      return absl::InternalError("trace lengths disagree");
    batches << step;
    for (size_t i : {size_t{0}, size_t{1}, size_t{2}}) {
      auto description = a.tensors()[i].description;
      description.alignment = "none";
      ASSIGN_OR_RETURN(auto raw,
                       Compare(description, a.tensors()[i].bytes.span(),
                               c.tensors()[i].bytes.span(), {}));
      batches << '\t' << raw.mismatches;
    }
    batches << '\n';
    batches.flush();
    std::string first_difference, first_bit_difference, repeat_difference;
    for (size_t i = 0; i < a.tensors().size(); ++i) {
      const auto& x = a.tensors()[i];
      const auto& y = b.tensors()[i];
      const auto& z = c.tensors()[i];
      if (x.description.stage != y.description.stage ||
          x.description.stage != z.description.stage)
        return absl::InternalError("trace stage ordering disagrees");
      ASSIGN_OR_RETURN(auto repeated, Compare(x.description, x.bytes.span(),
                                              y.bytes.span(), {}));
      ASSIGN_OR_RETURN(auto compared, Compare(x.description, x.bytes.span(),
                                              z.bytes.span(), permutation));
      summary << step << '\t' << i << '\t' << x.description.stage << '\t'
              << repeated.mismatches << '\t' << compared.mismatches << '\t'
              << compared.max_abs << '\t' << compared.l2 << '\t'
              << x.description.alignment << '\n';
      if (repeated.mismatches != 0 && repeat_difference.empty())
        repeat_difference = x.description.stage;
      if (compared.mismatches != 0 && first_bit_difference.empty())
        first_bit_difference = x.description.stage;
      if (compared.max_abs != 0 && first_difference.empty())
        first_difference = x.description.stage;
    }
    summary.flush();
    std::cout << "step=" << step
              << " first_numerical_difference=" << first_difference
              << " first_bit_difference=" << first_bit_difference
              << " repeat_difference=" << repeat_difference << std::endl;
    if (!first_difference.empty() || !repeat_difference.empty() ||
        (canonical_token_order && !first_bit_difference.empty()) ||
        step == max_steps) {
      RETURN_IF_ERROR(WriteTrace(a, output / "baseline"));
      RETURN_IF_ERROR(WriteTrace(b, output / "repeat"));
      RETURN_IF_ERROR(WriteTrace(c, output / "renamed"));
      std::ofstream metadata(output / "metadata.txt");
      metadata << "run_dir=" << root.string()
               << "\nvocabulary_size=" << kVocabulary
               << "\nmodel_width=" << kWidth
               << "\npadded_vocabulary_size=" << kPaddedVocabulary
               << "\nrows=" << kBatch * kContext << "\ncompute_type=BF16"
               << "\nbatch_size=" << kBatch << "\ncontext_length=" << kContext
               << "\nsupport=" << support << "\nmax_steps=" << max_steps
               << "\ncanonical_token_order=" << canonical_token_order
               << "\ndumped_step=" << step
               << "\ndivergent_step=" << (first_difference.empty() ? -1 : step)
               << "\nfirst_difference=" << first_difference
               << "\nfirst_bit_difference=" << first_bit_difference
               << "\nrepeat_difference=" << repeat_difference
               << "\ninitial_checkpoints_bitwise_equal=1\ninitial_embedding_"
                  "rows_equal=1\n";
      metadata.close();
      summary.close();
      batches.close();
      if (!metadata || !summary || !batches)
        return absl::InternalError("failed writing trace reports");
      if (!repeat_difference.empty())
        return absl::DataLossError(
            "baseline replicas diverged; investigate determinism before "
            "interpreting renaming");
      if (canonical_token_order && !first_bit_difference.empty())
        return absl::DataLossError(
            "canonical token order did not preserve bitwise equality");
      return absl::OkStatus();
    }
  }
  return absl::InternalError("trace loop ended unexpectedly");
}

}  // namespace
}  // namespace pluto::llm::permutation_trace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto status = pluto::llm::permutation_trace::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
