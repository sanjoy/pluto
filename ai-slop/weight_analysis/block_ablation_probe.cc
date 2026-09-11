// One-off, read-only checkpoint study. See ai-slop/README.md. Reuses the
// tested native forward, loss, token selection and weight-restoration tools.
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "ai-slop/weight_analysis/block_ablation_arms.h"
#include "ai-slop/weight_analysis/causal_probe.h"
#include "ai-slop/weight_analysis/token_trace_probe.h"
#include "src/llm/checkpoint.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Read-only full checkpoint");
ABSL_FLAG(std::string, batch_tokens, "",
          "Frozen packed corpus input/target batch");
ABSL_FLAG(std::string, trace_directory, "", "Verified exact Exeunt trace root");
ABSL_FLAG(std::string, output_dir, "", "New exclusive results directory");

namespace pluto::weight_analysis {
namespace {
using Clock = std::chrono::steady_clock;
constexpr int kContext = llm::kGpt2ContextLength;
constexpr int kVocab = llm::kGpt2VocabularySize;
constexpr int kPadded = llm::kGpt2PaddedVocabularySize;
constexpr int kCases = 4;
static_assert(kContext == 1024 && kVocab == 50257 && kPadded == 50272);

absl::StatusOr<cuda::PageLockedHostArray<float>> TraceRows(
    cuda::Executor& executor, const llm::Layer& model,
    const cuda::Buffer& tokens) {
  llm::Tape tape;
  ASSIGN_OR_RETURN(auto logits, model.fwd(executor, {tokens}, &tape));
  ASSIGN_OR_RETURN(auto result, cuda::PageLockedHostArray<float>::Allocate(
                                    kCases * kPadded));
  for (int item = 0; item < kCases; ++item) {
    ASSIGN_OR_RETURN(
        auto row, ReadSelectedRow(executor, logits, (item + 1) * kContext - 1,
                                  kPadded, sizeof(float)));
    std::memcpy(result.data() + item * kPadded, row.data(), row.size_bytes());
  }
  for (float value : result) {
    if (!std::isfinite(value))
      return absl::DataLossError("nonfinite trace logit");
  }
  return result;
}

template <class T>
absl::Status Equal(const cuda::PageLockedHostArray<T>& actual,
                   const cuda::PageLockedHostArray<T>& expected) {
  if (actual.size_bytes() != expected.size_bytes() ||
      std::memcmp(actual.data(), expected.data(), actual.size_bytes()) != 0) {
    return absl::DataLossError("clean repeat differs byte-for-byte");
  }
  return absl::OkStatus();
}

absl::Status Run() {
  const auto checkpoint =
      std::filesystem::absolute(absl::GetFlag(FLAGS_checkpoint));
  const auto batch_path =
      std::filesystem::absolute(absl::GetFlag(FLAGS_batch_tokens));
  const auto trace_root =
      std::filesystem::absolute(absl::GetFlag(FLAGS_trace_directory));
  const auto output =
      std::filesystem::absolute(absl::GetFlag(FLAGS_output_dir));
  for (const auto& value :
       {absl::GetFlag(FLAGS_checkpoint), absl::GetFlag(FLAGS_batch_tokens),
        absl::GetFlag(FLAGS_trace_directory),
        absl::GetFlag(FLAGS_output_dir)}) {
    if (value.empty())
      return absl::InvalidArgumentError("all four paths are required");
  }
  const auto resolved_checkpoint = std::filesystem::canonical(checkpoint);
  const auto resolved_output =
      (std::filesystem::canonical(output.parent_path()) / output.filename())
          .lexically_normal();
  for (auto ancestor = resolved_output;; ancestor = ancestor.parent_path()) {
    if (ancestor == resolved_checkpoint) {
      return absl::InvalidArgumentError("output cannot be inside checkpoint");
    }
    if (ancestor == ancestor.parent_path()) break;
  }
  ASSIGN_OR_RETURN(auto snapshot, InspectGpt2CheckpointFiles(checkpoint));
  RETURN_IF_ERROR(CreateNewOutputDirectory(output));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto batch, LoadPackedBatch(batch_path, kContext, kVocab));
  if (batch.passage_count != 8) {
    return absl::InvalidArgumentError("frozen study requires eight passages");
  }
  ASSIGN_OR_RETURN(auto trace_ids, cuda::PageLockedHostArray<int32_t>::Allocate(
                                       kCases * kContext));
  for (int item = 0; item < kCases; ++item) {
    const auto path =
        trace_root / absl::StrCat("prefix_step_", 1340 + item, ".i32");
    ASSIGN_OR_RETURN(auto ids, ReadTokenIds(path, kVocab, kContext));
    if (ids.size() != kContext)
      return absl::InvalidArgumentError("need full exact contexts");
    std::copy(ids.begin(), ids.end(), trace_ids.begin() + item * kContext);
  }
  ASSIGN_OR_RETURN(auto trace_tokens,
                   cuda::Buffer::Allocate(*executor, trace_ids.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(trace_tokens.data(), trace_ids.data(),
                      trace_ids.size_bytes(), cudaMemcpyHostToDevice,
                      executor->stream()),
      "upload fixed trace contexts"));
  RETURN_IF_ERROR(executor->Synchronize());
  ASSIGN_OR_RETURN(auto model,
                   llm::CreateGpt2(*executor, llm::DataType::BF16, 0));
  RETURN_IF_ERROR(llm::ReadFromDirectory(*executor, *model, checkpoint));
  ASSIGN_OR_RETURN(auto weights, UniqueWeights(*executor, model->weights()));
  RETURN_IF_ERROR(ValidateGpt2Weights(weights));
  ASSIGN_OR_RETURN(auto loss, llm::CrossEntropyLossLayer::Create(
                                  *executor, kVocab, llm::DataType::BF16));
  cuda::PageLockedHostArray<float> clean_losses;
  cuda::PageLockedHostArray<int32_t> clean_argmax;
  cuda::PageLockedHostArray<float> clean_trace;
  const auto arms = BlockAblationArms();
  std::ostringstream metadata;
  metadata
      << std::setprecision(17)
      << "{\"complete\":true,\"checkpoint\":" << JsonQuote(checkpoint.string())
      << ",\"batch_tokens\":" << JsonQuote(batch_path.string())
      << ",\"trace_directory\":" << JsonQuote(trace_root.string())
      << ",\"context_length\":1024,\"passages\":8,\"vocabulary\":50257,"
         "\"padded_vocabulary\":50272,\"trace_steps\":[1340,1341,1342,1343],"
         "\"target_ids\":[1475,68,2797,1750],\"optimizer_steps\":0,\"arms\":[";
  for (size_t index = 0; index < arms.size(); ++index) {
    const auto& arm = arms[index];
    const auto started = Clock::now();
    std::unique_ptr<WeightIntervention> intervention;
    if (!arm.weight_indices.empty()) {
      ASSIGN_OR_RETURN(
          intervention,
          WeightIntervention::Capture(*executor, weights, arm.weight_indices));
      RETURN_IF_ERROR(intervention->Apply(arm.scale));
    }
    auto measured =
        EvaluatePassages(*executor, *model, *loss, batch, 4, kVocab, kPadded);
    auto trace = TraceRows(*executor, *model, trace_tokens);
    // Drain and explicitly certify restoration even when an evaluation failed.
    if (intervention) RETURN_IF_ERROR(intervention->RestoreAndVerify());
    if (!measured.ok()) return measured.status();
    if (!trace.ok()) return trace.status();
    if (index == 0) {
      clean_losses = measured->losses;
      clean_argmax = measured->argmax;
      clean_trace = *trace;
    } else if (arm.weight_indices.empty()) {
      RETURN_IF_ERROR(Equal(measured->losses, clean_losses));
      RETURN_IF_ERROR(Equal(measured->argmax, clean_argmax));
      RETURN_IF_ERROR(Equal(*trace, clean_trace));
    }
    const std::string losses = arm.name + ".losses.f32";
    const std::string argmax = arm.name + ".argmax.i32";
    const std::string trace_file = arm.name + ".trace_logits.f32";
    RETURN_IF_ERROR(WriteExclusive(output / losses, measured->losses.data(),
                                   measured->losses.size_bytes()));
    RETURN_IF_ERROR(WriteExclusive(output / argmax, measured->argmax.data(),
                                   measured->argmax.size_bytes()));
    RETURN_IF_ERROR(WriteExclusive(output / trace_file, trace->data(),
                                   trace->size_bytes()));
    if (index) metadata << ',';
    metadata << "{\"name\":" << JsonQuote(arm.name) << ",\"indices\":[";
    for (size_t item = 0; item < arm.weight_indices.size(); ++item) {
      if (item) metadata << ',';
      metadata << arm.weight_indices[item];
    }
    metadata << "],\"scale\":" << arm.scale
             << ",\"restored_byte_equal\":true,\"losses\":" << JsonQuote(losses)
             << ",\"argmax\":" << JsonQuote(argmax)
             << ",\"trace_logits\":" << JsonQuote(trace_file) << '}';
    double sum = 0;
    int correct = 0;
    const auto targets = batch.targets(0, batch.passage_count);
    for (size_t item = 0; item < measured->losses.size(); ++item) {
      sum += measured->losses[item];
      correct += measured->argmax[item] == targets[item];
    }
    std::cout << index + 1 << '/' << arms.size() << ' ' << arm.name
              << " nll=" << sum / measured->losses.size() << " accuracy="
              << static_cast<double>(correct) / measured->losses.size()
              << " seconds="
              << std::chrono::duration<double>(Clock::now() - started).count()
              << std::endl;
  }
  RETURN_IF_ERROR(VerifyCheckpointFilesUnchanged(snapshot));
  metadata << "],\"clean_replays_byte_equal\":true}\n";
  const std::string document = metadata.str();
  RETURN_IF_ERROR(WriteExclusive(output / "metadata.json", document.data(),
                                 document.size()));
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::weight_analysis

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) return 1;
  try {
    const auto status = pluto::weight_analysis::Run();
    if (status.ok()) return 0;
    std::cerr << status << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
  }
  return 1;
}
