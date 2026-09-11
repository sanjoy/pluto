// A source-specific attention intervention, not a head ablation: change one
// source V, run native attention, retain only one query/head of that result,
// and replay the native downstream model. The original model is read-only.
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
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "ai-slop/weight_analysis/causal_probe.h"
#include "ai-slop/weight_analysis/embedding_factorial_probe.h"
#include "ai-slop/weight_analysis/phrase_probe.h"
#include "ai-slop/weight_analysis/source_value_probe.h"
#include "ai-slop/weight_analysis/token_trace_probe.h"
#include "src/llm/checkpoint.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Read-only full GPT-2 checkpoint.");
ABSL_FLAG(std::string, tokens_file, "",
          "Exact LE int32 prefix; no retokenizing.");
ABSL_FLAG(std::string, output_dir, "", "New exclusive evidence directory.");
ABSL_FLAG(int, block, -1, "Zero-based transformer block.");
ABSL_FLAG(int, head, -1, "Zero-based attention head.");
ABSL_FLAG(int, query, -1,
          "Query position; -1 selects the final prefix position.");
ABSL_FLAG(std::vector<std::string>, sources, {},
          "Comma-separated unique source positions, chosen before probing.");
ABSL_FLAG(int, target_id, -1, "Logical target token predicted at the query.");
ABSL_FLAG(
    std::string, expected_logits, "",
    "Optional prior native FP32 padded-vocabulary row; exact parity gate.");

namespace pluto::weight_analysis {
namespace {
namespace fs = std::filesystem;
using Bytes = cuda::PageLockedHostArray<uint8_t>;
constexpr int kContext = llm::kGpt2ContextLength;
constexpr int kWidth = llm::kGpt2ModelWidth;
constexpr int kVocabulary = llm::kGpt2VocabularySize;
constexpr int kPadded = llm::kGpt2PaddedVocabularySize;

// Disk scratch is ordinary CPU memory. All host memory supplied to CUDA is
// page locked, and every read completes before a snapshot is inspected.
absl::Status CompareFile(const fs::path& path, const void* expected,
                         size_t size) {
  if (fs::is_symlink(path) || !fs::is_regular_file(path) ||
      fs::file_size(path) != size) {
    return absl::DataLossError(
        absl::StrCat("file type/size changed: ", path.string()));
  }
  std::ifstream stream(path, std::ios::binary);
  std::array<char, 65536> scratch;
  for (size_t offset = 0; offset < size;) {
    const size_t count = std::min(scratch.size(), size - offset);
    if (!stream.read(scratch.data(), count) ||
        std::memcmp(scratch.data(),
                    static_cast<const uint8_t*>(expected) + offset, count)) {
      return absl::DataLossError(
          absl::StrCat("file bytes differ: ", path.string()));
    }
    offset += count;
  }
  if (stream.peek() != EOF)
    return absl::DataLossError("trailing file bytes");
  return absl::OkStatus();
}

absl::Status CheckFinite(const Bytes& bytes, size_t element_bytes) {
  if ((element_bytes != 2 && element_bytes != 4) ||
      bytes.size_bytes() % element_bytes) {
    return absl::InvalidArgumentError("invalid finite-check shape");
  }
  for (size_t offset = 0; offset < bytes.size_bytes();
       offset += element_bytes) {
    float value;
    if (element_bytes == 4) {
      std::memcpy(&value, bytes.data() + offset, 4);
    } else {
      uint16_t bits;
      std::memcpy(&bits, bytes.data() + offset, 2);
      value = std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
    }
    if (!std::isfinite(value))
      return absl::DataLossError("nonfinite native tensor");
  }
  return absl::OkStatus();
}

absl::Status CheckBytes(const Bytes& a, const Bytes& b) {
  if (a.size_bytes() != b.size_bytes() ||
      std::memcmp(a.data(), b.data(), a.size_bytes())) {
    return absl::DataLossError("read-only native buffer changed");
  }
  return absl::OkStatus();
}

absl::Status SaveTensor(cuda::Executor& executor, const fs::path& path,
                        const cuda::Buffer& buffer, int width,
                        size_t element_bytes) {
  ASSIGN_OR_RETURN(auto bytes,
                   ReadPrefix(executor, buffer, kContext, width, element_bytes));
  RETURN_IF_ERROR(CheckFinite(bytes, element_bytes));
  return WriteExclusive(path, bytes.data(), bytes.size_bytes());
}

absl::StatusOr<FactorialTokenScore> SaveLogits(cuda::Executor& executor,
                                               const fs::path& path,
                                               const cuda::Buffer& logits,
                                               int query, int target) {
  ASSIGN_OR_RETURN(auto row,
                   ReadSelectedRow(executor, logits, query, kPadded, 4));
  RETURN_IF_ERROR(CheckFinite(row, 4));
  ASSIGN_OR_RETURN(
      auto score,
      ScoreFactorialToken(
          absl::MakeConstSpan(reinterpret_cast<const float*>(row.data()),
                              kVocabulary),
          target));
  RETURN_IF_ERROR(WriteExclusive(path, row.data(), row.size_bytes()));
  return score;
}

absl::Status Run() {
  if constexpr (std::endian::native != std::endian::little)
    return absl::UnimplementedError("native evidence requires little endian");
  const int block = absl::GetFlag(FLAGS_block);
  const int head = absl::GetFlag(FLAGS_head);
  const int target = absl::GetFlag(FLAGS_target_id);
  if (absl::GetFlag(FLAGS_checkpoint).empty() ||
      absl::GetFlag(FLAGS_tokens_file).empty() ||
      absl::GetFlag(FLAGS_output_dir).empty() || target < 0 ||
      target >= kVocabulary || absl::GetFlag(FLAGS_query) < -1) {
    return absl::InvalidArgumentError(
        "required: --checkpoint --tokens_file --output_dir --block --head "
        "--sources --target_id; --query must be -1 or a prefix position");
  }
  std::vector<int> sources;
  std::set<int> unique;
  for (const auto& text : absl::GetFlag(FLAGS_sources)) {
    int source;
    if (!absl::SimpleAtoi(text, &source) || !unique.insert(source).second) {
      return absl::InvalidArgumentError(
          "sources must be unique integer positions");
    }
    RETURN_IF_ERROR(ValidateSourceValueSelection(
        {block, head, 0, kContext - 1, source, 1.0f}, kContext));
    sources.push_back(source);
  }
  if (sources.empty())
    return absl::InvalidArgumentError("at least one source required");

  const auto checkpoint = fs::absolute(absl::GetFlag(FLAGS_checkpoint));
  ASSIGN_OR_RETURN(auto files, InspectGpt2CheckpointFiles(checkpoint));
  const auto input_path = fs::absolute(absl::GetFlag(FLAGS_tokens_file));
  const auto output = fs::absolute(absl::GetFlag(FLAGS_output_dir));
  for (auto parent = fs::weakly_canonical(output);;
       parent = parent.parent_path()) {
    if (parent == fs::canonical(checkpoint))
      return absl::InvalidArgumentError("output must not be inside checkpoint");
    if (parent == parent.parent_path())
      break;
  }
  // Fail before touching CUDA for an already-used evidence directory.
  RETURN_IF_ERROR(CreateNewOutputDirectory(output));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto encoded,
                   ReadTokenIds(input_path, kVocabulary, kContext));
  const int query = absl::GetFlag(FLAGS_query) == -1
                        ? static_cast<int>(encoded.size()) - 1
                        : absl::GetFlag(FLAGS_query);
  if (query >= static_cast<int>(encoded.size()))
    return absl::InvalidArgumentError("query must be inside the actual prefix");
  for (int source : sources) {
    if (source >= static_cast<int>(encoded.size())) {
      return absl::InvalidArgumentError(
          "source must be inside the actual prefix");
    }
    RETURN_IF_ERROR(ValidateSourceValueSelection(
        {block, head, 0, query, source, 1.0f}, kContext));
  }
  RETURN_IF_ERROR(
      CompareFile(input_path, encoded.data(), encoded.size_bytes()));
  ASSIGN_OR_RETURN(auto padded,
                   cuda::PageLockedHostArray<int32_t>::Allocate(kContext));
  std::fill(padded.begin(), padded.end(), encoded[encoded.size() - 1]);
  std::copy(encoded.begin(), encoded.end(), padded.begin());
  ASSIGN_OR_RETURN(auto input,
                   cuda::Buffer::Allocate(*executor, padded.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(input.data(), padded.data(), padded.size_bytes(),
                      cudaMemcpyHostToDevice, executor->stream()),
      "source-value prefix upload"));
  ASSIGN_OR_RETURN(auto model,
                   llm::CreateGpt2(*executor, llm::DataType::BF16, 0));
  RETURN_IF_ERROR(llm::ReadFromDirectory(*executor, *model, checkpoint));
  ASSIGN_OR_RETURN(auto weights, UniqueWeights(*executor, model->weights()));
  RETURN_IF_ERROR(ValidateGpt2Weights(weights));
  if (model->weights().size() != 101 ||
      model->weights().front().data() != model->weights().back().data()) {
    return absl::FailedPreconditionError("unexpected tied weight traversal");
  }
  std::vector<Bytes> snapshots;
  for (size_t i = 0; i < weights.size(); ++i) {
    ASSIGN_OR_RETURN(auto bytes, ReadPrefix(*executor, weights[i],
                                            weights[i].size_bytes() / 4, 1, 4));
    RETURN_IF_ERROR(CheckFinite(bytes, 4));
    RETURN_IF_ERROR(
        CompareFile(files[i].path, bytes.data(), bytes.size_bytes()));
    snapshots.push_back(std::move(bytes));
  }
  // Optional patch metadata is not part of the layer. Snapshot and compare it
  // too; a stat-only check would not authenticate these input bytes.
  std::string patch;
  if (files.size() == 101) {
    if (files.back().bytes > 16 * 1024 * 1024)
      return absl::InvalidArgumentError("patch metadata exceeds 16 MiB");
    patch.resize(files.back().bytes);
    std::ifstream stream(files.back().path, std::ios::binary);
    if (!stream.read(patch.data(), patch.size()) || stream.peek() != EOF)
      return absl::DataLossError("could not snapshot patch metadata");
  }
  llm::Tape tape;
  ASSIGN_OR_RETURN(auto original,
                   model->fwd(*executor, absl::MakeConstSpan(&input, 1), &tape));
  ASSIGN_OR_RETURN(auto baseline,
                   ReadSelectedRow(*executor, original, query, kPadded, 4));
  const auto expected = absl::GetFlag(FLAGS_expected_logits);
  if (!expected.empty()) {
    RETURN_IF_ERROR(
        CompareFile(expected, baseline.data(), baseline.size_bytes()));
  }
  // Create refuses to return unless its unmodified native attention AND
  // complete native downstream replay agree at every row, including padding.
  ASSIGN_OR_RETURN(auto probe, SourceValueProbe::Create(
                                   *executor, tape, original, weights, block));
  ASSIGN_OR_RETURN(
      auto baseline_score,
      SaveLogits(*executor, output / "baseline.f32", original, query, target));
  // Pick the rival from the clean forward once and hold it fixed across doses.
  const auto* baseline_values = reinterpret_cast<const float*>(baseline.data());
  int rival = target == 0 ? 1 : 0;
  for (int id = 0; id < kVocabulary; ++id)
    if (id != target && baseline_values[id] > baseline_values[rival])
      rival = id;
  const double baseline_margin =
      static_cast<double>(baseline_values[target]) - baseline_values[rival];
  RETURN_IF_ERROR(WriteExclusive(output / "tokens.i32", encoded.data(),
                                 encoded.size_bytes()));
  RETURN_IF_ERROR(WriteExclusive(output / "padded_tokens.i32", padded.data(),
                                 padded.size_bytes()));
  RETURN_IF_ERROR(SaveTensor(*executor, output / "original_qkv.bf16",
                             probe->original_qkv(), 3 * kWidth, 2));
  RETURN_IF_ERROR(SaveTensor(*executor, output / "original_context.bf16",
                             probe->original_context(), kWidth, 2));
  std::ostringstream arms;
  arms << std::setprecision(17);
  bool first = true;
  for (int source : sources) {
    // The identity dose is genuinely replayed, never a shortcut returning the
    // saved logits. Repeat it after interventions to detect state
    // contamination.
    for (const auto& [label, dose] :
         std::array<std::pair<const char*, float>, 4>{{{"one", 1.0f},
                                                       {"half", 0.5f},
                                                       {"zero", 0.0f},
                                                       {"one_after", 1.0f}}}) {
      const auto name = absl::StrCat("source_", source, "_", label);
      const auto arm = output / name;
      RETURN_IF_ERROR(CreateNewOutputDirectory(arm));
      ASSIGN_OR_RETURN(
          auto result,
          probe->Apply(*executor, {block, head, 0, query, source, dose}));
      ASSIGN_OR_RETURN(auto score, SaveLogits(*executor, arm / "logits.f32",
                                              result.logits, query, target));
      ASSIGN_OR_RETURN(auto row, ReadSelectedRow(*executor, result.logits,
                                                 query, kPadded, 4));
      const auto* values = reinterpret_cast<const float*>(row.data());
      RETURN_IF_ERROR(SaveTensor(*executor, arm / "qkv.bf16",
                                 result.modified_qkv, 3 * kWidth, 2));
      RETURN_IF_ERROR(SaveTensor(*executor, arm / "replayed_attention.bf16",
                                 result.replayed_attention, kWidth, 2));
      RETURN_IF_ERROR(SaveTensor(*executor, arm / "spliced_context.bf16",
                                 result.spliced_context, kWidth, 2));
      if (!first)
        arms << ',';
      first = false;
      arms << "{\"directory\":" << JsonQuote(name) << ",\"source\":" << source
           << ",\"source_token_id\":" << encoded[source]
           << ",\"scale\":" << dose << ",\"nll\":" << score.nll
           << ",\"probability\":" << std::exp(-score.nll)
           << ",\"argmax\":" << score.argmax
           << ",\"target_rank\":" << score.target_rank
           << ",\"target_logit\":" << values[target]
           << ",\"rival_logit\":" << values[rival] << ",\"fixed_rival_margin\":"
           << static_cast<double>(values[target]) - values[rival] << '}';
      std::cout << name << ": P(target)=" << std::setprecision(10)
                << std::exp(-score.nll) << ", rank=" << score.target_rank
                << std::endl;
    }
  }
  llm::Tape after_tape;
  ASSIGN_OR_RETURN(
      auto after,
      model->fwd(*executor, absl::MakeConstSpan(&input, 1), &after_tape));
  ASSIGN_OR_RETURN(auto original_all,
                   ReadPrefix(*executor, original, kContext, kPadded, 4));
  ASSIGN_OR_RETURN(auto after_all,
                   ReadPrefix(*executor, after, kContext, kPadded, 4));
  RETURN_IF_ERROR(CheckBytes(original_all, after_all));
  // Check all original model and input bytes, including after the final clean
  // forward. No model computation occurs after this certification.
  // Evidence orchestrators additionally hash all files/binaries before/after.
  for (size_t i = 0; i < weights.size(); ++i) {
    ASSIGN_OR_RETURN(auto actual, ReadPrefix(*executor, weights[i],
                                             weights[i].size_bytes() / 4, 1, 4));
    RETURN_IF_ERROR(CheckBytes(actual, snapshots[i]));
    RETURN_IF_ERROR(CompareFile(files[i].path, snapshots[i].data(),
                                snapshots[i].size_bytes()));
  }
  RETURN_IF_ERROR(VerifyCheckpointFilesUnchanged(files));
  if (files.size() == 101)
    RETURN_IF_ERROR(CompareFile(files.back().path, patch.data(), patch.size()));
  RETURN_IF_ERROR(
      CompareFile(input_path, encoded.data(), encoded.size_bytes()));
  ASSIGN_OR_RETURN(auto input_after,
                   ReadPrefix(*executor, input, kContext, 1, 4));
  if (std::memcmp(input_after.data(), padded.data(), padded.size_bytes()))
    return absl::DataLossError("original device input changed");
  if (!expected.empty())
    RETURN_IF_ERROR(
        CompareFile(expected, baseline.data(), baseline.size_bytes()));
  std::ostringstream metadata;
  metadata << std::setprecision(17)
           << "{\"format\":\"pluto-source-value-probe-v1\",\"complete\":true"
           << ",\"checkpoint\":"
           << JsonQuote(fs::canonical(checkpoint).string())
           << ",\"tokens_file\":"
           << JsonQuote(fs::canonical(input_path).string()) << ",\"binary\":"
           << JsonQuote(fs::canonical("/proc/self/exe").string())
           << ",\"expected_logits\":" << JsonQuote(expected)
           << ",\"prefix_rows\":" << encoded.size()
           << ",\"execution_rows\":" << kContext << ",\"block\":" << block
           << ",\"head\":" << head << ",\"query\":" << query
           << ",\"query_token_id\":" << encoded[query]
           << ",\"target_id\":" << target << ",\"vocabulary\":" << kVocabulary
           << ",\"padded_vocabulary\":" << kPadded
           << ",\"baseline_nll\":" << baseline_score.nll
           << ",\"baseline_probability\":" << std::exp(-baseline_score.nll)
           << ",\"baseline_rank\":" << baseline_score.target_rank
           << ",\"baseline_argmax\":" << baseline_score.argmax
           << ",\"fixed_rival_id\":" << rival
           << ",\"baseline_fixed_rival_margin\":" << baseline_margin
           << ",\"native_full_attention_identity\":true"
           << ",\"native_all_row_padded_logit_identity\":true"
           << ",\"weight_disk_and_device_bytes_unchanged\":true"
           << ",\"original_full_forward_replay_equal\":true"
           << ",\"scope\":\"one source V; only one query/head context "
              "retained; Q/K unchanged; no masking or renormalization\""
           << ",\"arms\":[" << arms.str() << "]}\n";
  const auto text = metadata.str();
  return WriteExclusive(output / "metadata.json", text.data(), text.size());
}
}  // namespace
}  // namespace pluto::weight_analysis

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const auto status = pluto::weight_analysis::Run();
  if (status.ok())
    return 0;
  std::cerr << status << '\n';
  return 1;
}
