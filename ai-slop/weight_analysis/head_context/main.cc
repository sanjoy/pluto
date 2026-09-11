// Calibrated historical head-context interventions. The CLI is buildable
// independently of frozen parent diagnostics; native execution is reserved
// for after the training/analysis queue and requires two archived parity gates.
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ai-slop/weight_analysis/head_context/probe.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/checkpoint.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Exact full read-only GPT-2 checkpoint.");
ABSL_FLAG(std::string, tokens_file, "",
          "Exact historical LE int32 prefix, 1..1024 tokens.");
ABSL_FLAG(std::string, output_dir, "",
          "New exclusive evidence directory; parent must exist.");
ABSL_FLAG(int, block, -1, "Zero-based transformer block.");
ABSL_FLAG(int, head, -1, "Zero-based attention head.");
ABSL_FLAG(int, query, -1,
          "Selected query inside the actual historical prefix.");
ABSL_FLAG(int, target_id, -1,
          "Logical token whose prediction is investigated.");
ABSL_FLAG(std::string, expected_clean_logits, "",
          "Required archived 50,272-FP32 selected clean row.");
ABSL_FLAG(std::string, expected_all_query_zero_logits, "",
          "Required archived 50,272-FP32 selected all-query head-zero row.");

namespace pluto::weight_analysis::head_context {
namespace {
namespace fs = std::filesystem;
constexpr int kContext = llm::kGpt2ContextLength;
constexpr int kWidth = llm::kGpt2ModelWidth;
constexpr int kVocabulary = llm::kGpt2VocabularySize;
constexpr int kPadded = llm::kGpt2PaddedVocabularySize;
using Bytes = cuda::PageLockedHostArray<uint8_t>;
static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4 &&
              sizeof(uint16_t) == 2);
static_assert(std::numeric_limits<float>::is_iec559);
static_assert(kContext == 1024 && kWidth == 512 && kVocabulary == 50257 &&
              kPadded == 50272 && llm::kGpt2TransformerBlockCount == 8 &&
              llm::kGpt2AttentionHeads == 8 &&
              llm::kGpt2AttentionHeadDimension == 64);

struct FileInfo {
  fs::path path;
  uintmax_t bytes;
  fs::file_time_type modified;
};

absl::Status Bad(absl::string_view text) {
  return absl::InvalidArgumentError(text);
}

absl::StatusOr<fs::path> CanonicalExisting(const std::string& text,
                                           bool directory) {
  if (text.empty())
    return Bad("required input path is empty");
  const fs::path supplied = fs::absolute(text).lexically_normal();
  if (fs::is_symlink(supplied) ||
      (directory ? !fs::is_directory(supplied)
                 : !fs::is_regular_file(supplied)) ||
      fs::canonical(supplied) != supplied) {
    return Bad("input must be a canonical nonsymlink file/directory");
  }
  return supplied;
}

absl::StatusOr<FileInfo> Inspect(const fs::path& path) {
  if (fs::is_symlink(path) || !fs::is_regular_file(path) ||
      fs::canonical(path) != path) {
    return Bad(absl::StrCat("not a canonical regular input: ", path.string()));
  }
  return FileInfo{path, fs::file_size(path), fs::last_write_time(path)};
}

absl::Status VerifyStat(const FileInfo& expected) {
  ASSIGN_OR_RETURN(auto current, Inspect(expected.path));
  if (current.bytes != expected.bytes ||
      current.modified != expected.modified) {
    return absl::DataLossError(
        absl::StrCat("input file stat changed: ", expected.path.string()));
  }
  return absl::OkStatus();
}

// Disk I/O uses ordinary CPU scratch storage; only CUDA-facing memory must
// be page-locked. This comparison authenticates contents, not merely mtime.
absl::Status CompareDisk(const FileInfo& file, const void* bytes, size_t size) {
  RETURN_IF_ERROR(VerifyStat(file));
  if (file.bytes != size)
    return absl::DataLossError("disk/snapshot size differs");
  std::ifstream input(file.path, std::ios::binary);
  std::array<char, 65536> scratch;
  for (size_t offset = 0; offset < size;) {
    const size_t count = std::min(scratch.size(), size - offset);
    if (!input.read(scratch.data(), count) ||
        std::memcmp(scratch.data(), static_cast<const uint8_t*>(bytes) + offset,
                    count)) {
      return absl::DataLossError(
          absl::StrCat("disk/snapshot bytes differ: ", file.path.string()));
    }
    offset += count;
  }
  if (input.peek() != EOF)
    return absl::DataLossError("trailing input bytes");
  return VerifyStat(file);
}

template <class T>
absl::StatusOr<std::vector<T>> ReadSmall(const FileInfo& file, size_t minimum,
                                         size_t maximum) {
  if (file.bytes % sizeof(T) || file.bytes / sizeof(T) < minimum ||
      file.bytes / sizeof(T) > maximum)
    return Bad("small input file size differs");
  std::vector<T> values(file.bytes / sizeof(T));
  std::ifstream input(file.path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(values.data()), file.bytes) ||
      input.peek() != EOF) {
    return absl::DataLossError("short or changed input file");
  }
  RETURN_IF_ERROR(CompareDisk(file, values.data(), values.size() * sizeof(T)));
  return values;
}

std::vector<size_t> WeightSizes() {
  std::vector<size_t> sizes{size_t{kPadded} * kWidth,
                            size_t{kContext} * kWidth};
  for (int block = 0; block < 8; ++block)
    for (size_t n :
         {512ULL, 512ULL, 512ULL * 1536, 1536ULL, 512ULL * 512, 512ULL, 512ULL,
          512ULL, 512ULL * 2048, 2048ULL, 2048ULL * 512, 512ULL})
      sizes.push_back(n);
  sizes.push_back(kWidth);
  sizes.push_back(kWidth);
  for (auto& size : sizes)
    size *= sizeof(float);
  return sizes;
}

absl::StatusOr<std::vector<FileInfo>> InspectCheckpoint(
    const fs::path& directory) {
  const auto sizes = WeightSizes();
  std::set<std::string> expected;
  for (size_t i = 0; i < sizes.size(); ++i)
    expected.insert(absl::StrCat("weight_", i, ".bin"));
  std::set<std::string> actual;
  for (const auto& entry : fs::directory_iterator(directory)) {
    if (entry.is_symlink() || !entry.is_regular_file())
      return Bad("checkpoint contains link or nonregular file");
    actual.insert(entry.path().filename().string());
  }
  // Historical calibration is deliberately not the broader prefix/patch
  // checkpoint interface: exactly 100 canonical full-model files are required.
  if (actual != expected)
    return Bad(
        "checkpoint must contain exactly weight_0.bin through weight_99.bin");
  std::vector<FileInfo> files;
  for (size_t i = 0; i < sizes.size(); ++i) {
    ASSIGN_OR_RETURN(auto file,
                     Inspect(directory / absl::StrCat("weight_", i, ".bin")));
    if (file.bytes != sizes[i])
      return Bad("checkpoint weight byte size differs");
    files.push_back(std::move(file));
  }
  return files;
}

absl::Status CreateDirectory(const fs::path& path) {
  if (fs::exists(path) || fs::is_symlink(path) || !fs::create_directory(path)) {
    return absl::AlreadyExistsError(absl::StrCat(
        "evidence directory exists/cannot be exclusively created: ",
        path.string()));
  }
  return absl::OkStatus();
}

absl::Status WriteExclusive(const fs::path& path, const void* bytes,
                            size_t size) {
  const int fd = ::open(
      path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0444);
  if (fd < 0)
    return absl::InternalError(
        absl::StrCat("exclusive output open failed: ", path.string(), ": ",
                     std::strerror(errno)));
  size_t offset = 0;
  while (offset < size) {
    const size_t count = std::min<size_t>(size - offset, 64 * 1024 * 1024);
    const ssize_t written =
        ::write(fd, static_cast<const uint8_t*>(bytes) + offset, count);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0) {
      const std::string reason = std::strerror(errno);
      ::close(fd);
      return absl::InternalError(absl::StrCat("output write failed: ", reason));
    }
    offset += static_cast<size_t>(written);
  }
  if (::close(fd) != 0)
    return absl::InternalError("output close failed");
  return absl::OkStatus();
}

std::string Quote(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (unsigned char byte : value)
    if (byte == '"' || byte == '\\')
      output << '\\' << static_cast<char>(byte);
    else if (byte < 32)
      output << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
             << static_cast<int>(byte) << std::dec;
    else
      output << static_cast<char>(byte);
  output << '"';
  return output.str();
}

template <class T>
absl::StatusOr<cuda::PageLockedHostArray<T>> Download(
    cuda::Executor& executor, const cuda::Buffer& buffer) {
  if (&buffer.executor() != &executor || buffer.size_bytes() % sizeof(T))
    return Bad("download shape/executor differs");
  ASSIGN_OR_RETURN(auto values, cuda::PageLockedHostArray<T>::Allocate(
                                    executor, buffer.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(values.data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download calibrated head-context evidence"));
  RETURN_IF_ERROR(executor.Synchronize());
  return values;
}

absl::Status SaveContext(cuda::Executor& executor, const fs::path& path,
                         const cuda::Buffer& buffer, int width) {
  if (buffer.size_bytes() != size_t{kContext} * width * 2)
    return Bad("BF16 export shape differs");
  ASSIGN_OR_RETURN(auto values, Download<uint16_t>(executor, buffer));
  if (!std::all_of(values.begin(), values.end(),
                   [](uint16_t bits) { return (bits & 0x7f80) != 0x7f80; })) {
    return absl::DataLossError("nonfinite exported BF16 tensor");
  }
  return WriteExclusive(path, values.data(), values.size_bytes());
}

absl::StatusOr<cuda::PageLockedHostArray<float>> SaveLogits(
    cuda::Executor& executor, const fs::path& path,
    const cuda::Buffer& buffer) {
  if (buffer.size_bytes() != size_t{kContext} * kPadded * 4)
    return Bad("full-logit export shape differs");
  ASSIGN_OR_RETURN(auto values, Download<float>(executor, buffer));
  if (!std::all_of(values.begin(), values.end(),
                   [](float x) { return std::isfinite(x); })) {
    return absl::DataLossError("nonfinite exported full logits");
  }
  RETURN_IF_ERROR(WriteExclusive(path, values.data(), values.size_bytes()));
  return values;
}

absl::Status MatchSelected(const cuda::PageLockedHostArray<float>& all,
                           int query, const std::vector<float>& expected,
                           absl::string_view name) {
  if (all.size() != size_t{kContext} * kPadded || expected.size() != kPadded ||
      std::memcmp(all.data() + static_cast<size_t>(query) * kPadded,
                  expected.data(), kPadded * sizeof(float))) {
    return absl::DataLossError(
        absl::StrCat("archived ", name, " selected logits differ"));
  }
  return absl::OkStatus();
}

absl::Status Run(const std::vector<std::string>& argv) {
  if constexpr (std::endian::native != std::endian::little)
    return absl::UnimplementedError("requires little-endian native evidence");
  const int block = absl::GetFlag(FLAGS_block),
            head = absl::GetFlag(FLAGS_head);
  const int query = absl::GetFlag(FLAGS_query),
            target = absl::GetFlag(FLAGS_target_id);
  RETURN_IF_ERROR(ValidateSelection(
      {8, kContext, 8, 64}, {block, head, 0, query, QueryScope::kAllQueries, 0},
      kContext));
  if (target < 0 || target >= kVocabulary)
    return Bad("target_id must be a logical vocabulary ID");
  ASSIGN_OR_RETURN(auto checkpoint,
                   CanonicalExisting(absl::GetFlag(FLAGS_checkpoint), true));
  ASSIGN_OR_RETURN(auto tokens_path,
                   CanonicalExisting(absl::GetFlag(FLAGS_tokens_file), false));
  ASSIGN_OR_RETURN(
      auto clean_path,
      CanonicalExisting(absl::GetFlag(FLAGS_expected_clean_logits), false));
  ASSIGN_OR_RETURN(
      auto zero_path,
      CanonicalExisting(absl::GetFlag(FLAGS_expected_all_query_zero_logits),
                        false));
  ASSIGN_OR_RETURN(auto files, InspectCheckpoint(checkpoint));
  ASSIGN_OR_RETURN(auto input_record, Inspect(tokens_path));
  ASSIGN_OR_RETURN(auto clean_record, Inspect(clean_path));
  ASSIGN_OR_RETURN(auto zero_record, Inspect(zero_path));
  ASSIGN_OR_RETURN(auto tokens, ReadSmall<int32_t>(input_record, 1, kContext));
  ASSIGN_OR_RETURN(auto expected_clean,
                   ReadSmall<float>(clean_record, kPadded, kPadded));
  ASSIGN_OR_RETURN(auto expected_zero,
                   ReadSmall<float>(zero_record, kPadded, kPadded));
  if (query >= static_cast<int>(tokens.size()) ||
      !std::all_of(tokens.begin(), tokens.end(),
                   [](int32_t id) { return id >= 0 && id < kVocabulary; })) {
    return Bad("invalid token IDs or query outside actual prefix");
  }
  for (const auto* expected : {&expected_clean, &expected_zero}) {
    if (!std::all_of(expected->begin(), expected->end(),
                     [](float x) { return std::isfinite(x); }) ||
        !std::all_of(
            expected->begin() + kVocabulary, expected->end(),
            [](float x) { return x == -std::numeric_limits<float>::max(); })) {
      return Bad(
          "archived rows must be finite with exact -FLT_MAX physical padding");
    }
  }
  const auto output_text = absl::GetFlag(FLAGS_output_dir);
  if (output_text.empty())
    return Bad("output_dir is required");
  const fs::path output = fs::absolute(output_text).lexically_normal();
  ASSIGN_OR_RETURN(auto output_parent,
                   CanonicalExisting(output.parent_path().string(), true));
  (void)output_parent;
  for (auto parent = output;; parent = parent.parent_path()) {
    if (parent == checkpoint)
      return Bad("output cannot be inside checkpoint");
    if (parent == parent.parent_path())
      break;
  }
  // All flags, disk layouts, historical rows and prefix checks above are CPU
  // only. Existing output fails before creating any CUDA context.
  RETURN_IF_ERROR(CreateDirectory(output));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto padded, cuda::PageLockedHostArray<int32_t>::Allocate(
                                    *executor, kContext));
  std::fill(padded.begin(), padded.end(), tokens.back());
  std::copy(tokens.begin(), tokens.end(), padded.begin());
  ASSIGN_OR_RETURN(auto input,
                   cuda::Buffer::Allocate(*executor, padded.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(input.data(), padded.data(), padded.size_bytes(),
                      cudaMemcpyHostToDevice, executor->stream()),
      "upload exact historical padded prefix"));
  ASSIGN_OR_RETURN(auto model,
                   llm::CreateGpt2(*executor, llm::DataType::BF16, 17));
  RETURN_IF_ERROR(llm::ReadFromDirectory(*executor, *model, checkpoint));
  const auto traversal = model->weights();
  if (traversal.size() != 101 ||
      traversal.front().data() != traversal.back().data())
    return Bad("unexpected GPT-2 tied weight traversal");
  const auto weights = traversal.subspan(0, 100);
  std::vector<Bytes> snapshots;
  for (size_t i = 0; i < weights.size(); ++i) {
    ASSIGN_OR_RETURN(auto bytes, Download<uint8_t>(*executor, weights[i]));
    RETURN_IF_ERROR(CompareDisk(files[i], bytes.data(), bytes.size_bytes()));
    snapshots.push_back(std::move(bytes));
  }

  ASSIGN_OR_RETURN(auto clean_fwd,
                   model->fwd(*executor, absl::MakeConstSpan(&input, 1)));
  auto clean = std::move(clean_fwd.output);

  ASSIGN_OR_RETURN(auto probe, Probe::Create(*executor, *model, input,
                                             clean_fwd.state, clean, block));
  ASSIGN_OR_RETURN(auto clean_values,
                   SaveLogits(*executor, output / "clean_logits.f32", clean));
  RETURN_IF_ERROR(MatchSelected(clean_values, query, expected_clean, "clean"));
  RETURN_IF_ERROR(WriteExclusive(output / "tokens.i32", tokens.data(),
                                 tokens.size() * sizeof(int32_t)));
  RETURN_IF_ERROR(WriteExclusive(output / "padded_tokens.i32", padded.data(),
                                 padded.size_bytes()));
  const auto& branch = clean_fwd.state.children[block + 2].children[0];
  RETURN_IF_ERROR(SaveContext(*executor, output / "original_before.bf16",
                              branch.intermediates[0], kWidth));
  RETURN_IF_ERROR(SaveContext(*executor, output / "original_qkv.bf16",
                              branch.children[0].children[2].intermediates[0],
                              3 * kWidth));
  RETURN_IF_ERROR(SaveContext(*executor, output / "original_context.bf16",
                              probe->original_context(), kWidth));

  // Calibrate both clean and all-query zero at the archived selected row
  // BEFORE running the factorial. A plausible clean score alone is not enough
  // to establish equivalence with the historical projection-row ablation.
  const fs::path calibration = output / "calibration_all_queries_zero";
  RETURN_IF_ERROR(CreateDirectory(calibration));
  ASSIGN_OR_RETURN(auto zero,
                   probe->Apply(*executor, {block, head, 0, query,
                                            QueryScope::kAllQueries, 0}));
  RETURN_IF_ERROR(SaveContext(*executor, calibration / "context.bf16",
                              zero.context, kWidth));
  ASSIGN_OR_RETURN(
      auto zero_values,
      SaveLogits(*executor, calibration / "logits.f32", zero.logits));
  RETURN_IF_ERROR(
      MatchSelected(zero_values, query, expected_zero, "all-query zero"));

  std::ostringstream arms;
  bool first_arm = true;
  for (const auto& [scope_name, scope] :
       std::array<std::pair<const char*, QueryScope>, 3>{
           {{"query_only", QueryScope::kSelectedQuery},
            {"other_queries", QueryScope::kOtherQueries},
            {"all_queries", QueryScope::kAllQueries}}}) {
    for (const auto& [label, dose] :
         std::array<std::pair<const char*, float>, 4>{{{"one", 1.0f},
                                                       {"half", 0.5f},
                                                       {"zero", 0.0f},
                                                       {"one_after", 1.0f}}}) {
      const auto name = absl::StrCat(scope_name, "_", label);
      const auto directory = output / name;
      RETURN_IF_ERROR(CreateDirectory(directory));
      ASSIGN_OR_RETURN(
          auto result,
          probe->Apply(*executor, {block, head, 0, query, scope, dose}));
      RETURN_IF_ERROR(SaveContext(*executor, directory / "context.bf16",
                                  result.context, kWidth));
      ASSIGN_OR_RETURN(
          auto values,
          SaveLogits(*executor, directory / "logits.f32", result.logits));
      if (dose == 1.0f && std::memcmp(values.data(), clean_values.data(),
                                      values.size_bytes())) {
        return absl::DataLossError(
            "exported identity full logits differ from clean");
      }
      if (scope == QueryScope::kAllQueries && dose == 0.0f &&
          std::memcmp(values.data(), zero_values.data(), values.size_bytes())) {
        return absl::DataLossError(
            "repeated all-query zero differs from calibration full logits");
      }
      if (!first_arm)
        arms << ',';
      first_arm = false;
      arms << "{\"directory\":" << Quote(name)
           << ",\"scope\":" << Quote(scope_name)
           << ",\"dose_label\":" << Quote(label) << ",\"scale\":" << dose
           << ",\"context_file\":\"context.bf16\",\"logits_file\":\"logits."
              "f32\"}";
      std::cout << "Completed " << name << std::endl;
    }
  }
  // Final clean full-model replay followed by byte checks of all original
  // device inputs/weights and their actual checkpoint files. No optimizer or
  // backward call exists anywhere in this executable.

  ASSIGN_OR_RETURN(auto after_fwd,
                   model->fwd(*executor, absl::MakeConstSpan(&input, 1)));
  auto after = std::move(after_fwd.output);

  ASSIGN_OR_RETURN(auto after_values, Download<float>(*executor, after));
  if (after_values.size_bytes() != clean_values.size_bytes() ||
      std::memcmp(after_values.data(), clean_values.data(),
                  clean_values.size_bytes())) {
    return absl::DataLossError("final clean full-model forward differs");
  }
  RETURN_IF_ERROR(probe->VerifyOriginals(*executor));
  ASSIGN_OR_RETURN(auto current_files, InspectCheckpoint(checkpoint));
  if (current_files.size() != files.size())
    return absl::DataLossError("checkpoint inventory changed");
  for (size_t i = 0; i < weights.size(); ++i) {
    ASSIGN_OR_RETURN(auto bytes, Download<uint8_t>(*executor, weights[i]));
    if (bytes.size_bytes() != snapshots[i].size_bytes() ||
        std::memcmp(bytes.data(), snapshots[i].data(), bytes.size_bytes()))
      return absl::DataLossError("original device weight changed");
    RETURN_IF_ERROR(
        CompareDisk(files[i], snapshots[i].data(), snapshots[i].size_bytes()));
  }
  ASSIGN_OR_RETURN(auto input_after, Download<int32_t>(*executor, input));
  if (input_after.size_bytes() != padded.size_bytes() ||
      std::memcmp(input_after.data(), padded.data(), padded.size_bytes()))
    return absl::DataLossError("original device input changed");
  RETURN_IF_ERROR(CompareDisk(input_record, tokens.data(),
                              tokens.size() * sizeof(int32_t)));
  RETURN_IF_ERROR(CompareDisk(clean_record, expected_clean.data(),
                              expected_clean.size() * sizeof(float)));
  RETURN_IF_ERROR(CompareDisk(zero_record, expected_zero.data(),
                              expected_zero.size() * sizeof(float)));

  std::ostringstream metadata;
  metadata << "{\"format\":\"pluto-head-context-probe-v1\",\"complete\":true"
           << ",\"binary\":" << Quote(fs::canonical("/proc/self/exe").string())
           << ",\"checkpoint\":" << Quote(checkpoint.string())
           << ",\"tokens_file\":" << Quote(tokens_path.string())
           << ",\"output_dir\":" << Quote(output.string())
           << ",\"expected_clean_logits\":" << Quote(clean_path.string())
           << ",\"expected_all_query_zero_logits\":"
           << Quote(zero_path.string()) << ",\"block\":" << block
           << ",\"head\":" << head << ",\"sequence\":0,\"query\":" << query
           << ",\"query_token_id\":" << tokens[query]
           << ",\"target_id\":" << target
           << ",\"prefix_rows\":" << tokens.size()
           << ",\"execution_rows\":1024,\"context_length\":1024"
           << ",\"width\":512,\"heads\":8,\"head_dimension\":64,\"blocks\":8"
           << ",\"vocabulary\":50257,\"padded_vocabulary\":50272,\"byte_"
              "order\":\"little\""
           << ",\"padding\":\"repeat_last_token\",\"context_dtype\":\"BF16/"
              "<u2\",\"logits_dtype\":\"<f4\""
           << ",\"context_shape\":[1024,512],\"qkv_shape\":[1024,1536],"
              "\"logits_shape\":[1024,50272]"
           << ",\"scopes\":[\"query_only\",\"other_queries\",\"all_queries\"],"
              "\"doses\":[1,0.5,0,1]"
           << ",\"calibration_directory\":\"calibration_all_queries_zero\","
              "\"arm_count\":12,\"arms\":["
           << arms.str() << ']'
           << ",\"native_full_model_all_row_padded_identity\":true,\"native_"
              "clean_attention_identity\":true"
           << ",\"native_clean_tail_all_row_padded_identity\":true,\"identity_"
              "all_row_padded_logits\":true"
           << ",\"causally_unaffected_rows_equal\":true,\"physical_padding_"
              "logits_equal\":true"
           << ",\"archived_clean_selected_logits_equal\":true,\"archived_all_"
              "query_zero_selected_logits_equal\":true"
           << ",\"calibration_precedes_factorial\":true,\"repeated_all_query_"
              "zero_full_logits_equal\":true"
           << ",\"checkpoint_disk_and_device_bytes_unchanged\":true,\"original_"
              "input_bytes_unchanged\":true"
           << ",\"original_full_forward_replay_equal\":true,\"new_paired_model_"
              "result\":false,\"goal_completion_claimed\":false"
           << ",\"command\":[";
  for (size_t i = 0; i < argv.size(); ++i) {
    if (i)
      metadata << ',';
    metadata << Quote(argv[i]);
  }
  metadata << "]}\n";
  const auto text = metadata.str();
  // Completion is the final exclusive file, after every native/disk gate.
  return WriteExclusive(output / "metadata.json", text.data(), text.size());
}
}  // namespace
}  // namespace pluto::weight_analysis::head_context

int main(int argc, char** argv) {
  const std::vector<std::string> command(argv, argv + argc);
  absl::ParseCommandLine(argc, argv);
  const auto status = pluto::weight_analysis::head_context::Run(command);
  if (status.ok())
    return 0;
  std::cerr << status << '\n';
  return 1;
}
