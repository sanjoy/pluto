// Factor the effect of a tied embedding patch into input-side, output-side,
// and interaction effects. All four cells use the exact same teacher-forced
// input IDs. No generation, training, or on-disk checkpoint mutation occurs.
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
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "ai-slop/weight_analysis/causal_probe.h"
#include "ai-slop/weight_analysis/embedding_factorial_probe.h"
#include "ai-slop/weight_analysis/phrase_probe.h"
#include "src/llm/checkpoint.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, recipient, "", "Original recipient A checkpoint.");
ABSL_FLAG(std::string, patched, "",
          "Checkpoint J; only logical token-embedding rows may differ from A.");
ABSL_FLAG(std::string, batch, "", "Frozen LE int32 [all inputs][all targets].");
ABSL_FLAG(std::string, rows, "",
          "LE int32 [case_count, rows_per_case] selected loss rows.");
ABSL_FLAG(int, rows_per_case, 3, "Number of selected rows in each case.");
ABSL_FLAG(int, batch_sequences, 1,
          "Execution microbatch; never changes cases.");
ABSL_FLAG(std::string, output_dir, "", "New exclusive evidence directory.");

namespace pluto::weight_analysis {
namespace {
namespace fs = std::filesystem;
using WeightBytes = std::vector<cuda::PageLockedHostArray<uint8_t>>;
constexpr int kContext = llm::kGpt2ContextLength;
constexpr int kVocabulary = llm::kGpt2VocabularySize;
static_assert(kContext == 1024 && kVocabulary == 50257 &&
              llm::kGpt2PaddedVocabularySize == 50272 &&
              llm::kGpt2ModelWidth == 512 &&
              llm::kGpt2TransformerBlockCount == 8 &&
              llm::kGpt2AttentionHeads == 8 &&
              llm::kGpt2AttentionHeadDimension == 64 &&
              llm::kGpt2FeedForwardWidth == 2048);

// Byte comparisons are stronger than file timestamps. All disk reads use
// bounded ordinary CPU scratch; every CUDA transfer uses page-locked storage.
absl::Status CompareFileBytes(const fs::path& path, const void* expected,
                              size_t size) {
  if (fs::is_symlink(path) || !fs::is_regular_file(path) ||
      fs::file_size(path) != size) {
    return absl::DataLossError(
        absl::StrCat("input file changed: ", path.string()));
  }
  std::ifstream file(path, std::ios::binary);
  std::array<char, 65536> scratch;
  const auto* source = static_cast<const uint8_t*>(expected);
  for (size_t offset = 0; offset < size;) {
    const size_t count = std::min(scratch.size(), size - offset);
    if (!file.read(scratch.data(), count) ||
        std::memcmp(scratch.data(), source + offset, count)) {
      return absl::DataLossError(
          absl::StrCat("input bytes differ: ", path.string()));
    }
    offset += count;
  }
  if (file.peek() != EOF)
    return absl::DataLossError("unexpected trailing bytes");
  return absl::OkStatus();
}

absl::StatusOr<WeightBytes> SnapshotWeights(
    cuda::Executor& executor, absl::Span<const cuda::Buffer> weights) {
  RETURN_IF_ERROR(ValidateGpt2Weights(weights));
  WeightBytes result;
  for (const auto& weight : weights) {
    ASSIGN_OR_RETURN(auto bytes, ReadPrefix(executor, weight,
                                            weight.size_bytes() / 4, 1, 4));
    const auto* values = reinterpret_cast<const float*>(bytes.data());
    for (size_t i = 0; i < bytes.size_bytes() / 4; ++i) {
      if (!std::isfinite(values[i])) {
        return absl::DataLossError("nonfinite checkpoint parameter");
      }
    }
    result.push_back(std::move(bytes));
  }
  return result;
}

absl::Status VerifyWeightBytes(cuda::Executor& executor,
                               absl::Span<const cuda::Buffer> weights,
                               const WeightBytes& expected,
                               const fs::path& directory, bool check_device) {
  if (weights.size() != expected.size()) {
    return absl::InternalError("weight snapshot shape changed");
  }
  for (size_t i = 0; i < weights.size(); ++i) {
    if (check_device) {
      ASSIGN_OR_RETURN(
          auto actual,
          ReadPrefix(executor, weights[i], weights[i].size_bytes() / 4, 1, 4));
      if (actual.size_bytes() != expected[i].size_bytes() ||
          std::memcmp(actual.data(), expected[i].data(), actual.size_bytes())) {
        return absl::DataLossError("in-memory model weight bytes changed");
      }
    }
    RETURN_IF_ERROR(
        CompareFileBytes(directory / absl::StrCat("weight_", i, ".bin"),
                         expected[i].data(), expected[i].size_bytes()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> SnapshotMetadata(
    const std::vector<CheckpointFileInfo>& files) {
  if (files.size() == 100) return std::string();
  const auto& file = files.back();
  if (file.bytes > 16 * 1024 * 1024) {
    return absl::InvalidArgumentError("patch metadata exceeds 16 MiB");
  }
  std::string bytes(file.bytes, '\0');
  std::ifstream stream(file.path, std::ios::binary);
  if (!stream.read(bytes.data(), bytes.size()) || stream.peek() != EOF) {
    return absl::DataLossError("could not read exact patch metadata");
  }
  return bytes;
}

absl::Status Run() {
  if constexpr (std::endian::native != std::endian::little) {
    return absl::UnimplementedError("native evidence requires little endian");
  }
  const int rows_per_case = absl::GetFlag(FLAGS_rows_per_case);
  const int microbatch = absl::GetFlag(FLAGS_batch_sequences);
  if (absl::GetFlag(FLAGS_recipient).empty() ||
      absl::GetFlag(FLAGS_patched).empty() ||
      absl::GetFlag(FLAGS_batch).empty() || absl::GetFlag(FLAGS_rows).empty() ||
      absl::GetFlag(FLAGS_output_dir).empty() || rows_per_case <= 0 ||
      rows_per_case > kContext || microbatch <= 0 ||
      microbatch > std::numeric_limits<int>::max() / kContext) {
    return absl::InvalidArgumentError(
        "required: --recipient --patched --batch --rows --output_dir; "
        "positive --batch_sequences and --rows_per_case=1..1024");
  }
  // Validate the supplied paths before canonicalizing: directory symlinks must
  // not silently select another experiment. A and J may coincide for a control.
  const std::array<fs::path, 2> checkpoints{
      fs::absolute(absl::GetFlag(FLAGS_recipient)),
      fs::absolute(absl::GetFlag(FLAGS_patched))};
  std::array<std::vector<CheckpointFileInfo>, 2> checkpoint_files;
  std::array<std::string, 2> checkpoint_metadata;
  for (int side = 0; side < 2; ++side) {
    ASSIGN_OR_RETURN(checkpoint_files[side],
                     InspectGpt2CheckpointFiles(checkpoints[side]));
    ASSIGN_OR_RETURN(checkpoint_metadata[side],
                     SnapshotMetadata(checkpoint_files[side]));
  }
  const auto output = fs::absolute(absl::GetFlag(FLAGS_output_dir));
  const auto resolved_output = fs::weakly_canonical(output);
  for (const auto& checkpoint : checkpoints) {
    for (auto parent = resolved_output;; parent = parent.parent_path()) {
      if (parent == fs::canonical(checkpoint)) {
        return absl::InvalidArgumentError(
            "output must not be inside checkpoint");
      }
      if (parent == parent.parent_path()) break;
    }
  }
  const auto batch_path = fs::absolute(absl::GetFlag(FLAGS_batch));
  const auto rows_path = fs::absolute(absl::GetFlag(FLAGS_rows));
  if (fs::is_symlink(batch_path) || !fs::is_regular_file(batch_path)) {
    return absl::InvalidArgumentError(
        "batch must be a regular non-symlink file");
  }
  // Executor precedes all CUDA-owned objects and therefore outlives them.
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto batch,
                   LoadPackedBatch(batch_path, kContext, kVocabulary));
  ASSIGN_OR_RETURN(auto selected,
                   LoadFactorialRows(rows_path, batch.passage_count,
                                     rows_per_case, kContext));
  RETURN_IF_ERROR(CompareFileBytes(batch_path, batch.tokens.data(),
                                   batch.tokens.size_bytes()));
  RETURN_IF_ERROR(CompareFileBytes(rows_path, selected.data(),
                                   selected.size() * sizeof(int32_t)));

  std::array<std::unique_ptr<llm::ComposedLayer>, 2> models;
  std::array<std::vector<cuda::Buffer>, 2> weights;
  std::array<WeightBytes, 2> snapshots;
  std::array<std::unique_ptr<NativeLogitLens>, 2> lenses;
  for (int side = 0; side < 2; ++side) {
    ASSIGN_OR_RETURN(models[side],
                     llm::CreateGpt2(*executor, llm::DataType::BF16, 0));
    RETURN_IF_ERROR(
        llm::ReadFromDirectory(*executor, *models[side], checkpoints[side]));
    ASSIGN_OR_RETURN(weights[side],
                     UniqueWeights(*executor, models[side]->weights()));
    ASSIGN_OR_RETURN(snapshots[side],
                     SnapshotWeights(*executor, weights[side]));
    RETURN_IF_ERROR(VerifyWeightBytes(*executor, weights[side], snapshots[side],
                                      checkpoints[side], false));
  }
  // A joint patch is ONLY an embedding intervention. Otherwise interpreting
  // the input-side cell as an embedding effect would confound changed blocks
  // or final normalization. Compare the actual loaded bytes, not patch labels.
  for (size_t i = 1; i < snapshots[0].size(); ++i) {
    if (snapshots[0][i].size_bytes() != snapshots[1][i].size_bytes() ||
        std::memcmp(snapshots[0][i].data(), snapshots[1][i].data(),
                    snapshots[0][i].size_bytes())) {
      return absl::InvalidArgumentError(
          absl::StrCat("nonembedding weights differ: weight_", i, ".bin"));
    }
  }
  ASSIGN_OR_RETURN(
      auto changed_rows,
      ChangedEmbeddingRows(
          absl::MakeConstSpan(
              reinterpret_cast<const float*>(snapshots[0][0].data()),
              snapshots[0][0].size_bytes() / sizeof(float)),
          absl::MakeConstSpan(
              reinterpret_cast<const float*>(snapshots[1][0].data()),
              snapshots[1][0].size_bytes() / sizeof(float)),
          kVocabulary, llm::kGpt2ModelWidth));
  for (int side = 0; side < 2; ++side) {
    // Both final LNs were verified byte-identical; replacing these handles with
    // A's explicitly makes the fixed-normalization contract visible here too.
    auto readout_weights = weights[side];
    readout_weights[98] = weights[0][98];
    readout_weights[99] = weights[0][99];
    ASSIGN_OR_RETURN(lenses[side],
                     NativeLogitLens::Create(*executor, readout_weights));
  }

  // Persist only selected rows, not the 1024-by-vocabulary padded forwards.
  // Output order is [case, selected row, token ID] for each of AA/AJ/JA/JJ.
  std::array<cuda::PageLockedHostArray<float>, 4> all_logits;
  std::array<std::vector<FactorialTokenScore>, 4> all_scores;
  for (auto& values : all_logits) {
    ASSIGN_OR_RETURN(values, cuda::PageLockedHostArray<float>::Allocate(
                                 selected.size() * kVocabulary));
  }
  RETURN_IF_ERROR(CreateNewOutputDirectory(output));
  for (int first = 0; first < batch.passage_count;) {
    const int count = std::min(microbatch, batch.passage_count - first);
    const auto inputs = batch.inputs(first, count);
    ASSIGN_OR_RETURN(
        auto input,
        cuda::Buffer::Allocate(*executor, inputs.size() * sizeof(int32_t)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(input.data(), inputs.data(), input.size_bytes(),
                        cudaMemcpyHostToDevice, executor->stream()),
        "factorial frozen input upload"));
    std::array<llm::Tape, 2> tapes;
    std::vector<cuda::Buffer> native_logits;
    std::vector<cuda::Buffer> residuals;
    for (int side = 0; side < 2; ++side) {
      ASSIGN_OR_RETURN(
          auto logits,
          models[side]->fwd(*executor, absl::MakeConstSpan(&input, 1),
                            &tapes[side]));
      native_logits.push_back(std::move(logits));
      RETURN_IF_ERROR(ValidateGpt2Tape(tapes[side]));
      // ValidateGpt2Tape guards this exact production path: final LayerNorm
      // saves its input, i.e. the post-block-7 residual before final readout.
      residuals.push_back(tapes[side].children[10].intermediates[0]);
    }
    std::vector<int32_t> rows, targets;
    const auto batch_targets = batch.targets(first, count);
    for (int c = 0; c < count; ++c) {
      for (int i = 0; i < rows_per_case; ++i) {
        const int row =
            c * kContext +
            selected[static_cast<size_t>(first + c) * rows_per_case + i];
        rows.push_back(row);
        targets.push_back(batch_targets[row]);
      }
    }
    ASSIGN_OR_RETURN(auto measured,
                     EvaluateEmbeddingFactorial(
                         *executor, {lenses[0].get(), lenses[1].get()},
                         {residuals[0], residuals[1]},
                         {native_logits[0], native_logits[1]}, rows, targets));
    for (int cell = 0; cell < 4; ++cell) {
      std::memcpy(all_logits[cell].data() +
                      static_cast<size_t>(first) * rows_per_case * kVocabulary,
                  measured.logits[cell].data(),
                  measured.logits[cell].size_bytes());
      all_scores[cell].insert(all_scores[cell].end(),
                              measured.scores[cell].begin(),
                              measured.scores[cell].end());
    }
    first += count;
    std::cout << "Factorial cases " << first << '/' << batch.passage_count
              << ": all four cells, both exact native diagonals" << std::endl;
  }
  RETURN_IF_ERROR(executor->Synchronize());
  for (int side = 0; side < 2; ++side) {
    RETURN_IF_ERROR(VerifyWeightBytes(*executor, weights[side], snapshots[side],
                                      checkpoints[side], true));
    RETURN_IF_ERROR(VerifyCheckpointFilesUnchanged(checkpoint_files[side]));
    if (checkpoint_files[side].size() == 101) {
      RETURN_IF_ERROR(CompareFileBytes(checkpoint_files[side].back().path,
                                       checkpoint_metadata[side].data(),
                                       checkpoint_metadata[side].size()));
    }
  }
  RETURN_IF_ERROR(CompareFileBytes(batch_path, batch.tokens.data(),
                                   batch.tokens.size_bytes()));
  RETURN_IF_ERROR(CompareFileBytes(rows_path, selected.data(),
                                   selected.size() * sizeof(int32_t)));

  std::ostringstream metadata;
  metadata
      << std::setprecision(17)
      << "{\"format\":\"pluto-embedding-factorial-v1\",\"complete\":true"
      << ",\"recipient\":" << JsonQuote(fs::canonical(checkpoints[0]).string())
      << ",\"patched\":" << JsonQuote(fs::canonical(checkpoints[1]).string())
      << ",\"binary\":" << JsonQuote(fs::canonical("/proc/self/exe").string())
      << ",\"batch\":" << JsonQuote(batch_path.string())
      << ",\"rows\":" << JsonQuote(rows_path.string())
      << ",\"case_count\":" << batch.passage_count
      << ",\"rows_per_case\":" << rows_per_case
      << ",\"batch_sequences\":" << microbatch
      << ",\"context_length\":1024,\"vocabulary\":50257,\"temperature\":1"
      << ",\"logits_dtype\":\"<f4\",\"logits_shape\":[" << batch.passage_count
      << ',' << rows_per_case << ',' << kVocabulary << ']'
      << ",\"normalization\":\"recipient final LayerNorm fixed in all four "
         "cells\""
      << ",\"score_definition\":\"CPU FP64 log-sum-exp over all logical native "
         "FP32 logits\""
      << ",\"interpretation\":\"A/J first letter chooses input embedding; "
         "second chooses output dictionary; mixed cells are intentionally "
         "untied\""
      << ",\"checks\":{\"native_diagonals_byte_equal\":true,"
         "\"native_diagonal_verification_scope\":"
         "\"selected_rows_full_padded_vocabulary\",\"nonembedding_"
         "weights_identical\":true,\"padding_weights_identical\":true,\"device_"
         "weight_bytes_unchanged\":true,\"disk_weight_bytes_unchanged\":true,"
         "\"input_bytes_unchanged\":true}"
      << ",\"changed_embedding_rows\":[";
  for (size_t i = 0; i < changed_rows.size(); ++i) {
    if (i) metadata << ',';
    metadata << changed_rows[i];
  }
  metadata << "],\"cells\":{";
  for (int cell = 0; cell < 4; ++cell) {
    if (cell) metadata << ',';
    const auto filename =
        absl::StrCat(kEmbeddingFactorialCells[cell], ".logits.f32.bin");
    RETURN_IF_ERROR(WriteExclusive(output / filename, all_logits[cell].data(),
                                   all_logits[cell].size_bytes()));
    metadata << JsonQuote(kEmbeddingFactorialCells[cell])
             << ":{\"logits_file\":" << JsonQuote(filename) << ",\"cases\":[";
    for (int c = 0; c < batch.passage_count; ++c) {
      if (c) metadata << ',';
      metadata << "{\"case_index\":" << c << ",\"tokens\":[";
      double sum_nll = 0;
      for (int i = 0; i < rows_per_case; ++i) {
        if (i) metadata << ',';
        const size_t index = static_cast<size_t>(c) * rows_per_case + i;
        const auto& score = all_scores[cell][index];
        sum_nll += score.nll;
        metadata << "{\"row\":" << selected[index]
                 << ",\"target\":" << batch.targets(c, 1)[selected[index]]
                 << ",\"nll\":" << score.nll
                 << ",\"rank\":" << score.target_rank
                 << ",\"argmax\":" << score.argmax << '}';
      }
      metadata << "],\"selected_rows_nll_sum\":" << sum_nll << '}';
    }
    metadata << "]}";
  }
  metadata
      << "},\"limitations\":[\"Teacher-forced exact token-sequence scores; no "
         "alternative tokenizations summed.\",\"A following delimiter is "
         "included only when explicitly selected in --rows.\",\"Only "
         "consecutive selected rows constitute a contiguous continuation "
         "probability.\",\"Native diagonal byte equality is checked only at "
         "selected rows, including their padded vocabulary lanes; nonselected "
         "rows are not compared.\",\"Checkpoint patch labels are not trusted; "
         "changed "
         "rows and unchanged remaining bytes were measured.\",\"No "
         "cryptographic hashes are computed here; the external evidence runner "
         "must hash checkpoints, batch, rows, binary, and outputs.\"]}\n";
  const auto json = metadata.str();
  // A partial or failed run cannot advertise completed measurements.
  RETURN_IF_ERROR(
      WriteExclusive(output / "metadata.json", json.data(), json.size()));
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::weight_analysis

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "Unexpected positional arguments\n";
    return 1;
  }
  try {
    const auto status = pluto::weight_analysis::Run();
    if (status.ok()) return 0;
    std::cerr << status << '\n';
  } catch (const std::exception& error) {
    std::cerr << "Factorial probe failed: " << error.what() << '\n';
  }
  return 1;
}
