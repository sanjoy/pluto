// Separate two sources of non-additivity using actual production AdamW:
// changed second-step gradients, and optimizer history with frozen gradients.
// This is two fixed training updates, not a fitted parameter construction.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/batch_validation.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/fact_superposition.h"
#include "src/llm/experiments/one_shot_memorizer/sentence_ablation.h"
#include "src/llm/experiments/one_shot_memorizer/sentence_ablation_training.h"
#include "src/llm/gpt2.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, initial_checkpoint, "", "Original common step_0");
ABSL_FLAG(std::string, baseline_step1, "",
          "Matched France-first baseline/step_1 control");
ABSL_FLAG(std::string, a_only_step2, "",
          "Matched France-then-masked-Greece omit_line_2/step_2");
ABSL_FLAG(std::string, tokenizer, "",
          "Original base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "", "Fixed two-line France then Greece corpus");
ABSL_FLAG(std::string, output_dir, "",
          "Fresh directory for two-step diagnostic artifacts");

namespace pluto::llm::one_shot_memorizer {
namespace {
namespace fs = std::filesystem;
using Snapshot = std::vector<float>;
constexpr int kVocabulary = 4475;
constexpr int kPhysicalVocabulary = 4480;
constexpr Gpt2Config kConfig{.transformer_block_count = 8,
                             .model_width = 16,
                             .attention_heads = 1,
                             .feed_forward_width = 64,
                             .vocabulary_size = kVocabulary,
                             .pad_vocabulary = false};

bool Identical(const Snapshot& a, const Snapshot& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<absl::Span<const float>> Views(absl::Span<const TensorSpec> layout,
                                           const Snapshot& snapshot) {
  std::vector<absl::Span<const float>> result;
  for (const auto& tensor : layout)
    result.push_back(absl::MakeConstSpan(snapshot).subspan(
        tensor.flat_offset, tensor.element_count));
  return result;
}

// Both lists use checkpoint order. Tied input/head weights MUST share one
// gradient accumulator; two different weights MUST NOT alias a gradient.
struct Parameters {
  std::vector<Buffer> weights;
  std::vector<Buffer> gradients;
};

absl::StatusOr<Parameters> GetParameters(cuda::Executor& executor, Layer& model,
                                         absl::Span<const TensorSpec> layout) {
  const auto weights = model.weights();
  const auto gradients = model.gradients();
  if (weights.size() != gradients.size())
    return absl::FailedPreconditionError(
        "weights and gradients differ in cardinality");
  Parameters result;
  absl::flat_hash_map<const void*, const void*> paired;
  absl::flat_hash_set<const void*> gradient_addresses;
  for (size_t i = 0; i < weights.size(); ++i) {
    if (&weights[i].executor() != &executor ||
        &gradients[i].executor() != &executor ||
        weights[i].size_bytes() != gradients[i].size_bytes())
      return absl::FailedPreconditionError(
          "weight/gradient size or executor mismatch");
    const auto [entry, inserted] =
        paired.emplace(weights[i].data(), gradients[i].data());
    if (!inserted) {
      if (entry->second != gradients[i].data())
        return absl::FailedPreconditionError(
            "tied weight has distinct gradient accumulators");
      continue;
    }
    if (!gradient_addresses.insert(gradients[i].data()).second)
      return absl::FailedPreconditionError(
          "distinct weights alias a gradient accumulator");
    result.weights.push_back(weights[i]);
    result.gradients.push_back(gradients[i]);
  }
  if (result.weights.size() != layout.size() || layout.size() != 100 ||
      weights.size() != 101)
    return absl::FailedPreconditionError(
        "expected 100 unique tensors and one tied alias");
  for (size_t i = 0; i < layout.size(); ++i)
    if (result.weights[i].size_bytes() !=
        layout[i].element_count * sizeof(float))
      return absl::FailedPreconditionError("parameter layout shape mismatch");
  return result;
}

absl::StatusOr<Snapshot> CopyD2H(cuda::Executor& executor,
                                 absl::Span<const Buffer> buffers,
                                 absl::Span<const TensorSpec> layout) {
  const size_t count = layout.back().flat_offset + layout.back().element_count;
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<float>::Allocate(executor, count));
  for (size_t i = 0; i < layout.size(); ++i)
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data() + layout[i].flat_offset, buffers[i].data(),
                        buffers[i].size_bytes(), cudaMemcpyDeviceToHost,
                        executor.stream()),
        "copy two-step FP32 state"));
  RETURN_IF_ERROR(executor.Synchronize());
  Snapshot result(host.begin(), host.end());
  RETURN_IF_ERROR(ValidateParameterValues(layout, Views(layout, result)));
  return result;
}

absl::Status CopyH2D(cuda::Executor& executor, const Snapshot& values,
                     absl::Span<const Buffer> buffers,
                     absl::Span<const TensorSpec> layout) {
  RETURN_IF_ERROR(ValidateParameterValues(layout, Views(layout, values)));
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<float>::CopyFrom(executor, values));
  for (size_t i = 0; i < layout.size(); ++i)
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(buffers[i].data(), host.data() + layout[i].flat_offset,
                        buffers[i].size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "inject captured FP32 gradients"));
  return executor.Synchronize();
}

absl::Status Check(std::ostream& controls, absl::string_view name, bool ok) {
  controls << name << '\t' << (ok ? "PASS" : "FAIL") << std::endl;
  return ok ? absl::OkStatus()
            : absl::FailedPreconditionError(
                  absl::StrCat("failed control: ", name));
}

struct ProductionRun {
  std::unique_ptr<Layer> model;  // Fresh model/master allocations.
  std::unique_ptr<AdamWOptimizer>
      optimizer;          // Fresh, zero first/second moments.
  Parameters parameters;  // Borrowed handles owned by model.
};

absl::StatusOr<ProductionRun> NewRun(cuda::Executor& executor,
                                     const fs::path& initial_path,
                                     absl::Span<const TensorSpec> layout,
                                     const Snapshot& initial,
                                     std::ostream& controls,
                                     absl::string_view name) {
  ProductionRun run;
  ASSIGN_OR_RETURN(run.model,
                   CreateGpt2(executor, DataType::BF16, 1337, kConfig));
  RETURN_IF_ERROR(ReadFromDirectory(executor, *run.model, initial_path, false));
  ASSIGN_OR_RETURN(run.parameters, GetParameters(executor, *run.model, layout));
  ASSIGN_OR_RETURN(auto before,
                   CopyD2H(executor, run.parameters.weights, layout));
  RETURN_IF_ERROR(Check(controls, absl::StrCat(name, "_starts_at_W0"),
                        Identical(before, initial)));
  ASSIGN_OR_RETURN(float rate, AblationLearningRate(1, 0.0006, 100, 40000));
  ASSIGN_OR_RETURN(run.optimizer, AdamWOptimizer::Create(executor, *run.model,
                                                         {.learning_rate = rate,
                                                          .beta1 = 0.9f,
                                                          .beta2 = 0.99f,
                                                          .epsilon = 1e-8f,
                                                          .weight_decay = 0}));
  RETURN_IF_ERROR(
      Check(controls, absl::StrCat(name, "_fresh_optimizer"),
            run.optimizer->step() == 0 &&
                run.optimizer->parameter_tensor_count() == layout.size()));
  return run;
}

// No borrowed batch escapes this function: Reset/Next's reusable buffers are
// consumed by both forward and backward, then the returned gradients are an
// owning CPU snapshot. Every call starts with explicitly cleared accumulators.
absl::StatusOr<Snapshot> CaptureGradient(cuda::Executor& executor,
                                         ProductionRun& run, Layer& loss,
                                         PaddedLineDataSetIterator& sample,
                                         absl::Span<const TensorSpec> layout,
                                         bool masked, std::ostream& controls,
                                         absl::string_view name) {
  ASSIGN_OR_RETURN(auto before,
                   CopyD2H(executor, run.parameters.weights, layout));
  RETURN_IF_ERROR(run.optimizer->ZeroGrad());
  RETURN_IF_ERROR(sample.Reset());
  ASSIGN_OR_RETURN(auto batch, sample.Next());
  RETURN_IF_ERROR(ValidateTrainingBatch(executor, *run.model, loss, batch));
  ASSIGN_OR_RETURN(auto forward, run.model->fwd(executor, {batch.inputs}));
  ASSIGN_OR_RETURN(auto loss_forward,
                   loss.fwd(executor, {forward.outputs[0], batch.targets}));
  ASSIGN_OR_RETURN(auto gradients,
                   loss.bwd(executor, {}, std::move(loss_forward.state)));
  if (gradients.size() != 1)
    return absl::InternalError("loss must return one logits gradient");
  if (masked) {
    ASSIGN_OR_RETURN(int slots, ZeroSentenceContribution(
                                    executor, gradients[0],
                                    sample.last_batch_sample_indices(), 0,
                                    kGpt2ContextLength, kPhysicalVocabulary));
    RETURN_IF_ERROR(
        Check(controls, absl::StrCat(name, "_one_masked_slot"), slots == 1));
  }
  ASSIGN_OR_RETURN(auto unused, run.model->bwd(executor, gradients,
                                               std::move(forward.state)));
  (void)unused;
  ASSIGN_OR_RETURN(auto captured,
                   CopyD2H(executor, run.parameters.gradients, layout));
  ASSIGN_OR_RETURN(auto after,
                   CopyD2H(executor, run.parameters.weights, layout));
  RETURN_IF_ERROR(Check(controls, absl::StrCat(name, "_weights_unchanged"),
                        Identical(before, after)));
  if (masked)
    RETURN_IF_ERROR(Check(controls, absl::StrCat(name, "_all_gradients_zero"),
                          std::all_of(captured.begin(), captured.end(),
                                      [](float value) { return value == 0; })));
  return captured;
}

absl::StatusOr<Snapshot> Apply(cuda::Executor& executor, ProductionRun& run,
                               int step, absl::Span<const TensorSpec> layout,
                               std::ostream& controls, absl::string_view name) {
  if (run.optimizer->step() != step - 1)
    return absl::InternalError("optimizer clock differs from requested step");
  ASSIGN_OR_RETURN(float rate, AblationLearningRate(step, 0.0006, 100, 40000));
  RETURN_IF_ERROR(run.optimizer->SetLearningRate(rate));
  RETURN_IF_ERROR(run.optimizer->ApplyStep());
  ASSIGN_OR_RETURN(auto cleared,
                   CopyD2H(executor, run.parameters.gradients, layout));
  RETURN_IF_ERROR(
      Check(controls, absl::StrCat(name, "_step", step, "_clears_gradients"),
            std::all_of(cleared.begin(), cleared.end(), [](float value) {
              return std::bit_cast<uint32_t>(value) == 0;
            })));
  return CopyD2H(executor, run.parameters.weights, layout);
}

absl::StatusOr<Snapshot> Replay(cuda::Executor& executor,
                                const fs::path& initial_path,
                                absl::Span<const TensorSpec> layout,
                                const Snapshot& initial, const Snapshot& first,
                                const Snapshot& second, std::ostream& controls,
                                absl::string_view name) {
  ASSIGN_OR_RETURN(
      auto run, NewRun(executor, initial_path, layout, initial, controls, name));
  Snapshot result;
  for (int step = 1; step <= 2; ++step) {
    const auto& gradient = step == 1 ? first : second;
    RETURN_IF_ERROR(run.optimizer->ZeroGrad());
    RETURN_IF_ERROR(
        CopyH2D(executor, gradient, run.parameters.gradients, layout));
    ASSIGN_OR_RETURN(auto uploaded,
                     CopyD2H(executor, run.parameters.gradients, layout));
    RETURN_IF_ERROR(Check(controls,
                          absl::StrCat(name, "_step", step, "_injected_bytes"),
                          Identical(uploaded, gradient)));
    ASSIGN_OR_RETURN(result, Apply(executor, run, step, layout, controls, name));
  }
  return result;
}

absl::StatusOr<Snapshot> ReadSnapshot(
    cuda::Executor& executor, const fs::path& path, int expected_step,
    const tokenizer::CompactVocabularyTokenizer& vocabulary,
    absl::Span<const TensorSpec> layout) {
  ASSIGN_OR_RETURN(auto info, InspectCheckpointDirectory(path));
  if (info.step != expected_step)
    return absl::InvalidArgumentError(
        "source checkpoint step differs from protocol");
  RETURN_IF_ERROR(vocabulary.ValidateFile(path / "compact_vocabulary.tsv"));
  ASSIGN_OR_RETURN(auto model,
                   CreateGpt2(executor, DataType::BF16, 1337, kConfig));
  RETURN_IF_ERROR(ReadFromDirectory(executor, *model, path, false));
  ASSIGN_OR_RETURN(auto parameters, GetParameters(executor, *model, layout));
  return CopyD2H(executor, parameters.weights, layout);
}

struct Norms {
  double total_squared = 0;
  double trajectory_squared = 0;
  double history_squared = 0;
  double interaction_inner_product = 0;
  double gradient_change_squared = 0;
  double original_gradient_squared = 0;
  double max_closure = 0;
  size_t changed = 0;
  void Add(double total, double trajectory, double history,
           double gradient_change, double original_gradient) {
    total_squared += total * total;
    trajectory_squared += trajectory * trajectory;
    history_squared += history * history;
    interaction_inner_product += trajectory * history;
    gradient_change_squared += gradient_change * gradient_change;
    original_gradient_squared += original_gradient * original_gradient;
    max_closure =
        std::max(max_closure, std::abs(total - (trajectory + history)));
    changed += total != 0;
  }
};

void WriteNorms(std::ostream& stream, absl::string_view name, size_t size,
                const Norms& value) {
  stream << name << '\t' << size << '\t' << value.changed << '\t'
         << std::sqrt(value.total_squared) << '\t'
         << std::sqrt(value.trajectory_squared) << '\t'
         << std::sqrt(value.history_squared) << '\t'
         << value.interaction_inner_product << '\t'
         << std::sqrt(value.gradient_change_squared) << '\t'
         << std::sqrt(value.original_gradient_squared) << '\t'
         << value.max_closure << '\n';
}

absl::Status Run() {
  const fs::path initial_path = absl::GetFlag(FLAGS_initial_checkpoint),
                 first_path = absl::GetFlag(FLAGS_baseline_step1),
                 component_path = absl::GetFlag(FLAGS_a_only_step2),
                 output = absl::GetFlag(FLAGS_output_dir);
  if (initial_path.empty() || first_path.empty() || component_path.empty() ||
      output.empty() || absl::GetFlag(FLAGS_tokenizer).empty() ||
      absl::GetFlag(FLAGS_corpus).empty())
    return absl::InvalidArgumentError(
        "initial, step1, A-only step2, tokenizer, corpus and fresh output "
        "paths required");
  std::error_code error;
  if (fs::exists(output, error) || error)
    return absl::AlreadyExistsError("output_dir must be fresh and accessible");
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto vocabulary,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, initial_path / "compact_vocabulary.tsv"));
  if (vocabulary->vocab_size() != kVocabulary ||
      vocabulary->eos_token_id() != 4474 ||
      vocabulary->original_eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError("unexpected compact vocabulary");
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  constexpr absl::string_view france =
      "The capital of France is Paris, a city on the Seine.";
  constexpr absl::string_view greece =
      "The capital of Greece is Athens, an ancient Mediterranean city.";
  absl::string_view text = corpus.text();
  if (!text.empty() && text.back() == '\n')
    text.remove_suffix(1);
  if (text != absl::StrCat(france, "\n", greece))
    return absl::InvalidArgumentError(
        "corpus differs from fixed France/Greece protocol");
  ASSIGN_OR_RETURN(
      auto layout,
      BuildGpt2ParameterLayout({kVocabulary, 16, 64, kGpt2ContextLength, 8}));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  const PaddedLineDataSetOptions options{
      .batch_size = 1,
      .context_length = kGpt2ContextLength,
      .prompt_tokens = 5,
      .eos_token = vocabulary->eos_token_id(),
      .shuffle = false};
  ASSIGN_OR_RETURN(auto sample_a, PaddedLineDataSetIterator::Create(
                                      *executor, france, *vocabulary, options));
  ASSIGN_OR_RETURN(auto sample_b, PaddedLineDataSetIterator::Create(
                                      *executor, greece, *vocabulary, options));
  if (sample_a->supervised_row_count() != 10 ||
      sample_b->supervised_row_count() != 8)
    return absl::FailedPreconditionError("fixed fact target counts differ");
  ASSIGN_OR_RETURN(auto loss, CrossEntropyLossLayer::Create(
                                  *executor, kVocabulary, DataType::BF16,
                                  kGpt2ContextLength));
  ASSIGN_OR_RETURN(const auto initial, ReadSnapshot(*executor, initial_path, 0,
                                                    *vocabulary, layout));
  ASSIGN_OR_RETURN(const auto external_first,
                   ReadSnapshot(*executor, first_path, 1, *vocabulary, layout));
  ASSIGN_OR_RETURN(
      const auto external_component,
      ReadSnapshot(*executor, component_path, 2, *vocabulary, layout));
  if (initial.size() != 114256)
    return absl::InternalError("expected 114256 unique FP32 parameters");
  if (!fs::create_directory(output, error) || error)
    return absl::UnknownError("cannot create fresh output directory");
  std::ofstream manifest(output / "manifest.tsv"),
      controls(output / "controls.tsv"),
      coordinates(output / "coordinates.tsv"),
      summary(output / "tensor_summary.tsv"),
      gradients_file(output / "gradients.f32", std::ios::binary),
      weights_file(output / "weights.f32", std::ios::binary);
  for (auto* stream : {&manifest, &controls, &coordinates, &summary,
                       &gradients_file, &weights_file}) {
    if (!*stream)
      return absl::UnknownError("cannot create two-step reports");
    *stream << std::setprecision(17);
  }
  ASSIGN_OR_RETURN(float rate1, AblationLearningRate(1, 0.0006, 100, 40000));
  ASSIGN_OR_RETURN(float rate2, AblationLearningRate(2, 0.0006, 100, 40000));
  manifest
      << "key\tvalue\ncompleted\tfalse\ninitial_checkpoint\t"
      << initial_path.string() << "\nbaseline_step1\t" << first_path.string()
      << "\na_only_step2\t" << component_path.string() << "\ncorpus\t"
      << absl::GetFlag(FLAGS_corpus) << "\ntokenizer\t"
      << absl::GetFlag(FLAGS_tokenizer) << "\nstep1_rate\t" << rate1
      << "\nstep2_rate\t" << rate2
      << "\noptimizer\tproduction AdamW; "
         "beta1=0.9f,beta2=0.99f,epsilon=1e-8f,decay=0"
         "\nschedule\tFrance then Greece; batch1; supervised targets10 then8; "
         "normal loss mean"
         "\nmask\tzero logits gradient AFTER CE backward; production Adam "
         "still steps"
         "\nmodel\tBF168blocks,width16,head1,FF64,context1024,storedvocab4475,"
         "logits4480"
         "\nunique_tensors\t100\ncoordinates\t114256\nprecision\tFP32 master "
         "weights and gradients"
         "\nreplays\tjoint=[gA(W0),gB(W1)];frozen=[gA(W0),gB(W0)];Aonly=[gA(W0)"
         ",0];Bonly=[0,gB(W0)]"
         "\ndecomposition\tjoint-SUM=(joint-frozen)+(frozen-SUM); FP64 "
         "differences"
         "\nlimitations\tordered two-step local interaction; not memorized "
         "accuracy, causal ownership or additive norm fractions\n";
  manifest.flush();
  controls << "control\tresult\n";
  ASSIGN_OR_RETURN(auto direct, NewRun(*executor, initial_path, layout, initial,
                                       controls, "direct_joint"));
  ASSIGN_OR_RETURN(const auto ga,
                   CaptureGradient(*executor, direct, *loss, *sample_a, layout,
                                   false, controls, "gA_W0"));
  ASSIGN_OR_RETURN(auto ga_again,
                   CaptureGradient(*executor, direct, *loss, *sample_a, layout,
                                   false, controls, "gA_W0_repeat"));
  RETURN_IF_ERROR(
      Check(controls, "gA_W0_repeat_bytes", Identical(ga, ga_again)));
  ASSIGN_OR_RETURN(const auto w1, Apply(*executor, direct, 1, layout, controls,
                                        "direct_joint"));
  RETURN_IF_ERROR(Check(controls, "external_baseline_step1",
                        Identical(w1, external_first)));
  ASSIGN_OR_RETURN(const auto gb1,
                   CaptureGradient(*executor, direct, *loss, *sample_b, layout,
                                   false, controls, "gB_W1"));
  ASSIGN_OR_RETURN(auto gb1_again,
                   CaptureGradient(*executor, direct, *loss, *sample_b, layout,
                                   false, controls, "gB_W1_repeat"));
  RETURN_IF_ERROR(
      Check(controls, "gB_W1_repeat_bytes", Identical(gb1, gb1_again)));
  ASSIGN_OR_RETURN(const auto direct_joint, Apply(*executor, direct, 2, layout,
                                                  controls, "direct_joint"));
  ASSIGN_OR_RETURN(auto origin, NewRun(*executor, initial_path, layout, initial,
                                       controls, "gradient_at_origin"));
  ASSIGN_OR_RETURN(const auto gb0,
                   CaptureGradient(*executor, origin, *loss, *sample_b, layout,
                                   false, controls, "gB_W0"));
  ASSIGN_OR_RETURN(auto gb0_again,
                   CaptureGradient(*executor, origin, *loss, *sample_b, layout,
                                   false, controls, "gB_W0_repeat"));
  RETURN_IF_ERROR(
      Check(controls, "gB_W0_repeat_bytes", Identical(gb0, gb0_again)));
  const Snapshot zero(initial.size(), 0);
  ASSIGN_OR_RETURN(const auto joint,
                   Replay(*executor, initial_path, layout, initial, ga, gb1,
                          controls, "replay_joint"));
  ASSIGN_OR_RETURN(const auto frozen,
                   Replay(*executor, initial_path, layout, initial, ga, gb0,
                          controls, "replay_frozen"));
  ASSIGN_OR_RETURN(const auto only_a,
                   Replay(*executor, initial_path, layout, initial, ga, zero,
                          controls, "replay_Aonly"));
  ASSIGN_OR_RETURN(const auto only_b,
                   Replay(*executor, initial_path, layout, initial, zero, gb0,
                          controls, "replay_Bonly"));
  RETURN_IF_ERROR(Check(controls, "direct_joint_equals_replayed_joint",
                        Identical(direct_joint, joint)));
  for (bool a_only : {true, false}) {
    const char* name = a_only ? "direct_Aonly" : "direct_Bonly";
    ASSIGN_OR_RETURN(auto masked, NewRun(*executor, initial_path, layout,
                                         initial, controls, name));
    ASSIGN_OR_RETURN(
        auto first_gradient,
        CaptureGradient(*executor, masked, *loss, *sample_a, layout, !a_only,
                        controls, absl::StrCat(name, "_first")));
    if (a_only)
      RETURN_IF_ERROR(Check(controls, "direct_Aonly_first_gradient",
                            Identical(first_gradient, ga)));
    ASSIGN_OR_RETURN(auto first,
                     Apply(*executor, masked, 1, layout, controls, name));
    RETURN_IF_ERROR(Check(controls, absl::StrCat(name, "_first_weights"),
                          Identical(first, a_only ? w1 : initial)));
    ASSIGN_OR_RETURN(
        auto second_gradient,
        CaptureGradient(*executor, masked, *loss, *sample_b, layout, a_only,
                        controls, absl::StrCat(name, "_second")));
    if (!a_only)
      RETURN_IF_ERROR(Check(controls, "direct_Bonly_second_gradient",
                            Identical(second_gradient, gb0)));
    ASSIGN_OR_RETURN(auto last,
                     Apply(*executor, masked, 2, layout, controls, name));
    RETURN_IF_ERROR(Check(controls, absl::StrCat(name, "_equals_replay"),
                          Identical(last, a_only ? only_a : only_b)));
  }
  RETURN_IF_ERROR(Check(controls, "external_Aonly_step2",
                        Identical(only_a, external_component)));
  ASSIGN_OR_RETURN(
      const auto sum,
      SuperposeFactParameters(initial, only_a, only_b, FactSuperposition::kSum));
  ASSIGN_OR_RETURN(const auto decomposition,
                   DecomposeFactSuperposition(joint, frozen, sum));
  RETURN_IF_ERROR(Check(controls, "coordinatewise_exact_closure",
                        decomposition.maximum_absolute_closure_error == 0));
  coordinates << "checkpoint_index\ttensor\telement_"
                 "index\trow\tcolumn\tinitial\tjoint\tfrozen\tAonly\tBonly\tSUM"
                 "\tgA_W0\tgB_W0\tgB_W1\tgradient_change\ttotal\tgradient_"
                 "trajectory\toptimizer_history\tclosure\n";
  summary << "tensor\tcoordinates\tnonzero_total\ttotal_l2\tgradient_"
             "trajectory_l2\toptimizer_history_l2\ttrajectory_history_inner_"
             "product\tgradient_change_l2\tgB_W0_l2\tmaximum_closure_error\n";
  Norms all;
  for (const auto& tensor : layout) {
    Norms norms;
    for (size_t element = 0; element < tensor.element_count; ++element) {
      const size_t index = tensor.flat_offset + element;
      const double total = decomposition.total[index],
                   trajectory = decomposition.gradient_trajectory[index],
                   history = decomposition.optimizer_history[index],
                   gradient_change =
                       static_cast<double>(gb1[index]) - gb0[index];
      ASSIGN_OR_RETURN(auto coordinate, LocateParameter(tensor, element));
      coordinates << tensor.checkpoint_index << '\t' << tensor.name << '\t'
                  << element << '\t' << coordinate.row << '\t';
      if (coordinate.column)
        coordinates << *coordinate.column;
      else
        coordinates << "NA";
      coordinates << '\t' << initial[index] << '\t' << joint[index] << '\t'
                  << frozen[index] << '\t' << only_a[index] << '\t'
                  << only_b[index] << '\t' << sum[index] << '\t' << ga[index]
                  << '\t' << gb0[index] << '\t' << gb1[index] << '\t'
                  << gradient_change << '\t' << total << '\t' << trajectory
                  << '\t' << history << '\t' << total - (trajectory + history)
                  << '\n';
      norms.Add(total, trajectory, history, gradient_change, gb0[index]);
      all.Add(total, trajectory, history, gradient_change, gb0[index]);
    }
    WriteNorms(summary, tensor.name, tensor.element_count, norms);
  }
  WriteNorms(summary, "ALL", initial.size(), all);
  static_assert(std::endian::native == std::endian::little &&
                sizeof(float) == 4);
  for (const auto* values : {&ga, &gb0, &gb1})
    gradients_file.write(reinterpret_cast<const char*>(values->data()),
                         values->size() * sizeof(float));
  for (const auto* values :
       {&initial, &w1, &joint, &frozen, &only_a, &only_b, &sum})
    weights_file.write(reinterpret_cast<const char*>(values->data()),
                       values->size() * sizeof(float));
  const std::array<std::pair<fs::path, const Snapshot*>, 3> sources{
      {{initial_path, &initial},
       {first_path, &external_first},
       {component_path, &external_component}}};
  for (size_t i = 0; i < sources.size(); ++i) {
    ASSIGN_OR_RETURN(auto after, ReadSnapshot(*executor, sources[i].first, i,
                                              *vocabulary, layout));
    RETURN_IF_ERROR(Check(controls,
                          absl::StrCat("source_", i, "_weight_bytes_unchanged"),
                          Identical(after, *sources[i].second)));
  }
  manifest
      << "gradient_binary_order\tgA_W0,gB_W0,gB_W1;3x114256 little-endianFP32\n"
         "weight_binary_order\tW0,W1,joint,frozen,Aonly,Bonly,SUM;7x114256 "
         "little-endianFP32\n"
         "sum_precision\tFP64 construction then one FP32 cast\n"
         "norm_convention\tcomponent norms are not fractions; cancellation is "
         "retained\n";
  for (auto* stream : {&manifest, &controls, &coordinates, &summary,
                       &gradients_file, &weights_file}) {
    stream->flush();
    if (!*stream)
      return absl::DataLossError("two-step report write failed");
  }
  manifest << "completed\ttrue\n";
  manifest.flush();
  if (!manifest)
    return absl::DataLossError("two-step manifest write failed");
  std::cout << "total L2=" << std::sqrt(all.total_squared)
            << ", gradient trajectory L2=" << std::sqrt(all.trajectory_squared)
            << ", optimizer history L2=" << std::sqrt(all.history_squared)
            << ", closure=" << all.max_closure << '\n';
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
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
