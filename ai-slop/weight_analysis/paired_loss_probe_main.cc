// Analysis-only teacher-forced scoring. Each passage starts at position zero;
// supplied right-padding is unchanged and the caller selects meaningful rows.
// Native cross entropy is -log P(target | prefix) at temperature 1. There is
// no sampling, generation, backward pass, or checkpoint-writing path here.
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ai-slop/weight_analysis/causal_probe.h"
#include "src/llm/checkpoint.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Read-only full GPT-2 step_N checkpoint.");
ABSL_FLAG(std::string, batch, "", "Frozen LE int32 [all inputs][all targets].");
ABSL_FLAG(std::string, output_dir, "", "New exclusive output directory.");
ABSL_FLAG(int, batch_sequences, 1, "Passages per execution microbatch.");

namespace pluto::weight_analysis {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static_assert(llm::kGpt2VocabularySize == 50257 &&
              llm::kGpt2PaddedVocabularySize == 50272 &&
              llm::kGpt2ContextLength == 1024 &&
              llm::kGpt2TransformerBlockCount == 8 &&
              llm::kGpt2ModelWidth == 512 && llm::kGpt2AttentionHeads == 8 &&
              llm::kGpt2AttentionHeadDimension == 64 &&
              llm::kGpt2FeedForwardWidth == 2048);

absl::Status ValidateCheckpointFiles(const fs::path &directory) {
  const auto sizes = Gpt2WeightByteSizes();
  size_t count = 0;
  for (const auto &entry : fs::directory_iterator(directory)) {
    const auto name = entry.path().filename().string();
    if (name.starts_with("weight_") && name.ends_with(".bin")) ++count;
  }
  if (count != sizes.size())
    return absl::InvalidArgumentError(
        "expected exactly 100 GPT-2 weight files");
  for (size_t i = 0; i < sizes.size(); ++i) {
    const auto path = directory / absl::StrCat("weight_", i, ".bin");
    if (fs::is_symlink(path) || !fs::is_regular_file(path) ||
        fs::file_size(path) != sizes[i])
      return absl::InvalidArgumentError(
          absl::StrCat("checkpoint weight layout mismatch: ", path.string()));
  }
  return absl::OkStatus();
}

absl::Status Run() {
  const auto started = Clock::now();
  const int microbatch = absl::GetFlag(FLAGS_batch_sequences);
  if (absl::GetFlag(FLAGS_checkpoint).empty() ||
      absl::GetFlag(FLAGS_batch).empty() ||
      absl::GetFlag(FLAGS_output_dir).empty() || microbatch <= 0)
    return absl::InvalidArgumentError(
        "required: --checkpoint --batch --output_dir; --batch_sequences > 0");
  ASSIGN_OR_RETURN(auto checkpoint,
                   llm::InspectCheckpointDirectory(
                       fs::canonical(absl::GetFlag(FLAGS_checkpoint))));
  const auto batch_path = fs::canonical(absl::GetFlag(FLAGS_batch));
  const auto output = fs::absolute(absl::GetFlag(FLAGS_output_dir));
  const auto resolved_output =
      (fs::canonical(output.parent_path()) / output.filename())
          .lexically_normal();
  for (auto parent = resolved_output;; parent = parent.parent_path()) {
    if (parent == checkpoint.directory)
      return absl::InvalidArgumentError("output must not be inside checkpoint");
    if (parent == parent.parent_path()) break;
  }
  RETURN_IF_ERROR(ValidateCheckpointFiles(checkpoint.directory));
  RETURN_IF_ERROR(CreateNewOutputDirectory(output));
  // Executor outlives pinned arrays, model buffers, and queued stream work.
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto batch,
                   LoadPackedBatch(batch_path, llm::kGpt2ContextLength,
                                   llm::kGpt2VocabularySize));
  ASSIGN_OR_RETURN(auto model,
                   llm::CreateGpt2(*executor, llm::DataType::BF16, 0));
  ASSIGN_OR_RETURN(auto weights, UniqueWeights(*executor, model->weights()));
  RETURN_IF_ERROR(ValidateGpt2Weights(weights));
  const auto raw_weights = model->weights();
  if (raw_weights.size() != 101 ||
      raw_weights.front().data() != raw_weights.back().data())
    return absl::FailedPreconditionError(
        "changed GPT-2 tied-embedding traversal");
  RETURN_IF_ERROR(
      llm::ReadFromDirectory(*executor, *model, checkpoint.directory));
  ASSIGN_OR_RETURN(auto loss_layer, llm::CrossEntropyLossLayer::Create(
                                        *executor, llm::kGpt2VocabularySize,
                                        llm::DataType::BF16));
  ASSIGN_OR_RETURN(auto measured,
                   EvaluatePassages(*executor, *model, *loss_layer, batch,
                                    microbatch, llm::kGpt2VocabularySize,
                                    llm::kGpt2PaddedVocabularySize));
  RETURN_IF_ERROR(WriteExclusive(output / "losses.f32.bin",
                                 measured.losses.data(),
                                 measured.losses.size_bytes()));
  RETURN_IF_ERROR(WriteExclusive(output / "argmax.i32.bin",
                                 measured.argmax.data(),
                                 measured.argmax.size_bytes()));
  RETURN_IF_ERROR(executor->Synchronize());

  std::ostringstream metadata;
  metadata
      << std::setprecision(17)
      << "{\"schema_version\":1,\"complete\":true,\"kind\":\"paired_loss_"
         "probe\","
      << "\"checkpoint_directory\":" << JsonQuote(checkpoint.directory.string())
      << ",\"checkpoint_step\":" << checkpoint.step
      << ",\"batch_file\":" << JsonQuote(batch_path.string())
      << ",\"batch_bytes\":" << batch.tokens.size_bytes() << ",\"binary_file\":"
      << JsonQuote(fs::canonical("/proc/self/exe").string())
      << ",\"output_directory\":" << JsonQuote(resolved_output.string())
      << ",\"batch_sequences\":" << microbatch
      << ",\"passage_count\":" << batch.passage_count
      << ",\"case_count\":" << batch.passage_count
      << ",\"context_length\":" << batch.context_length
      << ",\"vocab_size\":50257,\"padded_vocab_size\":50272,"
         "\"layers\":8,\"width\":512,\"heads\":8,\"head_dimension\":64,"
         "\"feed_forward_width\":2048,\"checkpoint_unique_weight_count\":100,"
         "\"checkpoint_raw_weight_count\":101,\"temperature\":1,"
         "\"loss_definition\":\"negative natural log probability of supplied "
         "target\","
         "\"compute_type\":\"native BF16 activations; FP32 weights, logits and "
         "loss\","
         "\"byte_order\":\"little\",\"batch_layout\":\"[all inputs][all "
         "targets], passage-major\","
         "\"padding\":\"supplied unchanged; caller selects meaningful output "
         "rows\","
         "\"output_shape\":["
      << batch.passage_count << ',' << batch.context_length
      << "],\"loss_file\":\"losses.f32.bin\",\"loss_dtype\":\"<f4\","
         "\"argmax_file\":\"argmax.i32.bin\",\"argmax_dtype\":\"<i4\","
         "\"finite_losses\":true,\"argmax_valid\":true,\"non_default_"
         "executor\":true,"
         "\"optimizer_steps\":0,\"backward_calls\":0,\"checkpoint_writes\":0,"
         "\"generation_steps\":0,\"weight_interventions\":0,"
         "\"integrity_hashes\":\"not computed here; external runner must hash "
         "inputs and checkpoint\","
         "\"elapsed_seconds\":"
      << std::chrono::duration<double>(Clock::now() - started).count() << "}\n";
  const auto json = metadata.str();
  // Publish completion only after both validated output arrays exist.
  RETURN_IF_ERROR(
      WriteExclusive(output / "metadata.json", json.data(), json.size()));
  std::cout << "Scored " << batch.passage_count
            << " passages at temperature 1\n";
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::weight_analysis

int main(int argc, char **argv) {
  const auto remaining = absl::ParseCommandLine(argc, argv);
  if (remaining.size() != 1) {
    std::cerr << "Unexpected positional arguments\n";
    return 1;
  }
  try {
    const auto status = pluto::weight_analysis::Run();
    if (status.ok()) return 0;
    std::cerr << status << '\n';
  } catch (const std::exception &error) {
    std::cerr << "Scoring failed: " << error.what() << '\n';
  }
  return 1;
}
