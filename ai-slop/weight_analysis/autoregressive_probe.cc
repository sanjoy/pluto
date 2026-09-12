#include "ai-slop/weight_analysis/autoregressive_probe.h"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "ai-slop/weight_analysis/causal_probe.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Read-only full GPT-2 checkpoint directory.");
ABSL_FLAG(std::string, tokenizer, "", "Runtime GPT-2 tokenizer directory.");
ABSL_FLAG(std::string, output_dir, "",
          "New exclusive diagnostic output directory.");
ABSL_FLAG(std::string, prompt, "to be or not to be", "Exact initial prompt.");
ABSL_FLAG(int, steps, 512,
          "Total generated token count, including any EOS tokens.");
ABSL_FLAG(double, temperature, .8, "Production sampling temperature.");
ABSL_FLAG(int, seed, 17, "Production seed; sampling mt19937 starts at seed+1.");

namespace pluto::weight_analysis {
namespace {
namespace fs = std::filesystem;
constexpr int kContext = llm::kGpt2ContextLength;
constexpr int kVocabulary = llm::kGpt2VocabularySize;
constexpr int kPaddedVocabulary = llm::kGpt2PaddedVocabularySize;
static_assert(kContext == 1024 && kVocabulary == 50257 &&
              kPaddedVocabulary == 50272);
static_assert(llm::kGpt2ModelWidth == 512 &&
              llm::kGpt2TransformerBlockCount == 8 &&
              llm::kGpt2AttentionHeads == 8 &&
              llm::kGpt2FeedForwardWidth == 2048);
static_assert(sizeof(float) == 4 && sizeof(int) == 4);

// The two append-only streams can be inspected during a partial run. Only the
// final metadata marker certifies completion. O_EXCL prevents replacement even
// if a caller races an output pathname inside the new run directory.
class ExclusiveStream {
 public:
  static absl::StatusOr<std::unique_ptr<ExclusiveStream>> Create(
      const fs::path& path) {
    const int fd =
        open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
      return absl::InternalError(
          absl::StrCat("exclusive open failed: ", path.string(), ": ",
                       std::strerror(errno)));
    }
    return absl::WrapUnique(new ExclusiveStream(fd));
  }
  ~ExclusiveStream() {
    if (fd_ >= 0)
      close(fd_);
  }
  absl::Status Append(const void* bytes, size_t count) {
    const auto* cursor = static_cast<const char*>(bytes);
    while (count) {
      const auto written = write(fd_, cursor, count);
      if (written < 0 && errno == EINTR)
        continue;
      if (written <= 0)
        return absl::InternalError("diagnostic stream write failed");
      cursor += written;
      count -= static_cast<size_t>(written);
    }
    return absl::OkStatus();
  }
  absl::Status Finish() {
    if (fsync(fd_) != 0)
      return absl::InternalError("diagnostic stream fsync failed");
    const int fd = std::exchange(fd_, -1);
    if (close(fd) != 0)
      return absl::InternalError("diagnostic stream close failed");
    return absl::OkStatus();
  }

 private:
  explicit ExclusiveStream(int fd) : fd_(fd) {}
  int fd_;
};

struct CheckpointFile {
  fs::path path;
  uintmax_t bytes;
  fs::file_time_type modified;
};

absl::StatusOr<std::vector<CheckpointFile>> InspectCheckpoint(
    const fs::path& path) {
  const auto expected = Gpt2WeightByteSizes();
  size_t count = 0;
  for (const auto& entry : fs::directory_iterator(path)) {
    if (entry.is_symlink() || !entry.is_regular_file()) {
      return absl::InvalidArgumentError(
          "checkpoint contains a non-regular entry");
    }
    ++count;
  }
  if (count != expected.size()) {
    return absl::InvalidArgumentError(
        "full GPT-2 checkpoint must contain 100 files");
  }
  std::vector<CheckpointFile> result;
  for (size_t index = 0; index < expected.size(); ++index) {
    const auto file = path / absl::StrCat("weight_", index, ".bin");
    if (fs::is_symlink(file) || !fs::is_regular_file(file) ||
        fs::file_size(file) != expected[index]) {
      return absl::InvalidArgumentError(
          "checkpoint weight has wrong name/type/size");
    }
    result.push_back({file, expected[index], fs::last_write_time(file)});
  }
  return result;
}

std::string Hex(absl::string_view bytes) {
  constexpr char kDigits[] = "0123456789abcdef";
  std::string result;
  result.reserve(2 * bytes.size());
  for (unsigned char byte : bytes) {
    result.push_back(kDigits[byte >> 4]);
    result.push_back(kDigits[byte & 15]);
  }
  return result;
}

std::string Ints(absl::Span<const int> values) {
  std::ostringstream output;
  output << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i)
      output << ',';
    output << values[i];
  }
  output << ']';
  return output.str();
}

// Same BF16 model, row selection and physical vocabulary stride as production
// Predict. Only the logical selected-row logits cross D2H, into pinned memory.
absl::StatusOr<cuda::PageLockedHostArray<float>> PredictSelectedRow(
    cuda::Executor& executor, const llm::Layer& model,
    const cuda::PageLockedHostArray<int>& padded_input,
    const cuda::Buffer& token_buffer, size_t selected_row) {
  if (&token_buffer.executor() != &executor ||
      padded_input.size() != kContext ||
      token_buffer.size_bytes() != padded_input.size_bytes() ||
      selected_row >= kContext) {
    return absl::InvalidArgumentError(
        "invalid selected-row prediction shape/executor");
  }
  ASSIGN_OR_RETURN(auto host_logits, cuda::PageLockedHostArray<float>::Allocate(
                                         executor, kVocabulary));
  // On an error path, preserve pinned input/output until all enqueued work has
  // stopped. The fence is declared after host_logits and dies before it.
  struct Fence {
    cuda::Executor& executor;
    ~Fence() { (void)executor.Synchronize(); }
  } fence{executor};
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(token_buffer.data(), padded_input.data(),
                      padded_input.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "autoregressive prompt upload"));

  ASSIGN_OR_RETURN(auto logits_fwd,
                   model.fwd(executor, absl::MakeConstSpan(&token_buffer, 1)));
  if (logits_fwd.outputs.size() != 1)
    return absl::FailedPreconditionError("model must return one logits tensor");
  auto logits = std::move(logits_fwd.outputs[0]);

  if (&logits.executor() != &executor ||
      logits.size_bytes() !=
          static_cast<size_t>(kContext) * kPaddedVocabulary * sizeof(float)) {
    return absl::FailedPreconditionError(
        "native GPT-2 logits shape/executor changed");
  }
  const auto* selected = static_cast<const float*>(logits.data()) +
                         selected_row * kPaddedVocabulary;
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host_logits.data(), selected, host_logits.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "autoregressive selected logits download"));
  RETURN_IF_ERROR(executor.Synchronize());
  return host_logits;
}

absl::Status Run() {
  if constexpr (std::endian::native != std::endian::little)
    return absl::UnimplementedError("probe requires a little-endian host");
  const auto started = std::chrono::steady_clock::now();
  const int steps = absl::GetFlag(FLAGS_steps);
  const int seed = absl::GetFlag(FLAGS_seed);
  const double temperature = absl::GetFlag(FLAGS_temperature);
  RETURN_IF_ERROR(ValidateGenerationOptions(steps, temperature, seed));
  if (absl::GetFlag(FLAGS_checkpoint).empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() ||
      absl::GetFlag(FLAGS_output_dir).empty()) {
    return absl::InvalidArgumentError(
        "required: --checkpoint --tokenizer --output_dir");
  }
  const auto checkpoint = fs::canonical(absl::GetFlag(FLAGS_checkpoint));
  const auto tokenizer_directory =
      fs::canonical(absl::GetFlag(FLAGS_tokenizer));
  const auto output = fs::absolute(absl::GetFlag(FLAGS_output_dir));
  const auto resolved_output =
      (fs::canonical(output.parent_path()) / output.filename())
          .lexically_normal();
  for (auto ancestor = resolved_output;; ancestor = ancestor.parent_path()) {
    if (ancestor == checkpoint || ancestor == tokenizer_directory) {
      return absl::InvalidArgumentError(
          "output directory must not be inside input data");
    }
    if (ancestor == ancestor.parent_path())
      break;
  }
  ASSIGN_OR_RETURN(auto files, InspectCheckpoint(checkpoint));
  RETURN_IF_ERROR(CreateNewOutputDirectory(output));
  ASSIGN_OR_RETURN(auto logits_file,
                   ExclusiveStream::Create(output / "logits.f32"));
  ASSIGN_OR_RETURN(auto events_file,
                   ExclusiveStream::Create(output / "events.jsonl"));

  // One explicit executor owns every stream-bound allocation, including model
  // weights, and outlives them all. This binary has no training or writer API.
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto encoder,
                   tokenizer::Gpt2Tokenizer::Load(tokenizer_directory));
  ASSIGN_OR_RETURN(auto decoder,
                   tokenizer::Gpt2Detokenizer::Load(tokenizer_directory));
  if (encoder->vocab_size() != kVocabulary ||
      decoder->vocab_size() != kVocabulary) {
    return absl::InvalidArgumentError(
        "tokenizer vocabulary differs from native GPT-2");
  }
  const auto requested_prompt = absl::GetFlag(FLAGS_prompt);
  const std::string prompt = requested_prompt.empty() ? "\n" : requested_prompt;
  ASSIGN_OR_RETURN(auto encoded, encoder->Encode(*executor, prompt));
  if (encoded.empty())
    return absl::InvalidArgumentError("prompt encoded to no tokens");
  ASSIGN_OR_RETURN(auto round_trip, decoder->Decode(encoded.span()));
  if (round_trip != prompt)
    return absl::DataLossError("initial prompt token bytes differ");
  std::vector<int> history(encoded.begin(), encoded.end());
  std::vector<int> generated;
  generated.reserve(steps);
  std::string generated_bytes;
  ASSIGN_OR_RETURN(auto padded, cuda::PageLockedHostArray<int>::Allocate(
                                    *executor, kContext));
  ASSIGN_OR_RETURN(auto token_buffer,
                   cuda::Buffer::Allocate(*executor, padded.size_bytes()));
  ASSIGN_OR_RETURN(auto model,
                   llm::CreateGpt2(*executor, llm::DataType::BF16, seed));
  ASSIGN_OR_RETURN(auto weights, UniqueWeights(*executor, model->weights()));
  RETURN_IF_ERROR(ValidateGpt2Weights(weights));
  if (model->weights().size() != 101 ||
      model->weights().front().data() != model->weights().back().data()) {
    return absl::FailedPreconditionError(
        "production tied-embedding traversal changed");
  }
  RETURN_IF_ERROR(llm::ReadFromDirectory(*executor, *model, checkpoint));
  std::mt19937 random(seed + 1);
  std::ostringstream initial_rng_state;
  initial_rng_state << random;
  for (int index = 0; index < steps; ++index) {
    ASSIGN_OR_RETURN(
        auto window,
        FillPredictionInput(history, kContext, kVocabulary, padded.span()));
    ASSIGN_OR_RETURN(auto logits,
                     PredictSelectedRow(*executor, *model, padded, token_buffer,
                                        window.output_row));
    ASSIGN_OR_RETURN(auto decision,
                     SampleProductionLogits(logits.span(), temperature, random));
    const int next = decision.token_id;
    ASSIGN_OR_RETURN(auto piece, decoder->Decode(absl::MakeConstSpan(&next, 1)));
    const size_t byte_start = generated_bytes.size();
    generated_bytes.append(piece);
    RETURN_IF_ERROR(logits_file->Append(logits.data(), logits.size_bytes()));
    std::ostringstream event;
    event << std::setprecision(17) << "{\"index\":" << index
          << ",\"step\":" << index
          << ",\"absolute_token_index\":" << history.size()
          << ",\"context_start\":" << window.start
          << ",\"context_length\":" << window.length
          << ",\"output_row\":" << window.output_row << ",\"token_id\":" << next
          << ",\"piece_hex\":" << JsonQuote(Hex(piece))
          << ",\"generated_byte_start\":" << byte_start
          << ",\"generated_byte_end\":" << generated_bytes.size()
          << ",\"logits_byte_offset\":"
          << static_cast<size_t>(index) * kVocabulary * sizeof(float)
          << ",\"raw_logit\":" << decision.raw_logit
          << ",\"raw_rank\":" << decision.raw_rank
          << ",\"raw_argmax_token_id\":" << decision.raw_argmax_token_id
          << ",\"temperature\":" << temperature
          << ",\"sampled_probability\":" << decision.sampled_probability
          << ",\"uniform\":" << decision.uniform
          << ",\"cdf_lower\":" << decision.cdf_lower
          << ",\"cdf_upper\":" << decision.cdf_upper
          << ",\"rng_word_0\":" << decision.rng_words[0]
          << ",\"rng_word_1\":" << decision.rng_words[1]
          << ",\"rng_state_verified\":true,\"cdf_selected_verified\":true}\n";
    const auto record = event.str();
    RETURN_IF_ERROR(events_file->Append(record.data(), record.size()));
    history.push_back(next);
    generated.push_back(next);
    if ((index + 1) % 32 == 0 || index + 1 == steps) {
      std::cout << "Generated " << index + 1 << '/' << steps << " tokens; "
                << generated_bytes.size()
                << " bytes; context_start=" << window.start << std::endl;
    }
  }
  ASSIGN_OR_RETURN(auto completion, decoder->Decode(generated));
  if (completion != generated_bytes) {
    return absl::DataLossError(
        "concatenated per-token bytes differ from production Decode");
  }
  RETURN_IF_ERROR(logits_file->Finish());
  RETURN_IF_ERROR(events_file->Finish());
  RETURN_IF_ERROR(WriteExclusive(output / "tokens.i32", history.data(),
                                 history.size() * sizeof(int)));
  RETURN_IF_ERROR(WriteExclusive(output / "generated.bin",
                                 generated_bytes.data(),
                                 generated_bytes.size()));
  for (const auto& file : files) {
    if (fs::is_symlink(file.path) || !fs::is_regular_file(file.path) ||
        fs::file_size(file.path) != file.bytes ||
        fs::last_write_time(file.path) != file.modified) {
      return absl::DataLossError(
          "checkpoint file stat changed during generation");
    }
  }
  RETURN_IF_ERROR(executor->Synchronize());
  std::ostringstream final_rng_state;
  final_rng_state << random;
  const auto initial_ids = Ints(encoded.span());
  const auto all_ids = Ints(history);
  std::ostringstream metadata;
  metadata
      << std::setprecision(17)
      << "{\"schema_version\":1,\"complete\":true,\"autoregressive\":true"
      << ",\"requested_prompt\":" << JsonQuote(requested_prompt)
      << ",\"prompt\":" << JsonQuote(prompt)
      << ",\"empty_prompt_newline_fallback\":"
      << (requested_prompt.empty() ? "true" : "false")
      << ",\"initial_token_ids\":" << initial_ids
      << ",\"prompt_token_ids\":" << initial_ids
      << ",\"generated_token_ids\":" << Ints(generated)
      << ",\"all_token_ids\":" << all_ids << ",\"token_ids\":" << all_ids
      << ",\"steps\":" << steps << ",\"temperature\":" << temperature
      << ",\"seed\":" << seed << ",\"rng_seed\":" << seed + 1
      << ",\"context_length\":" << kContext << ",\"vocab_size\":" << kVocabulary
      << ",\"padded_vocab_size\":" << kPaddedVocabulary
      << ",\"special_token_ids\":[" << encoder->eos_token_id() << ']'
      << ",\"stop_on_eos\":false,\"no_bos\":true,\"batch_size\":1"
      << ",\"checkpoint_directory\":" << JsonQuote(checkpoint.string())
      << ",\"tokenizer_directory\":" << JsonQuote(tokenizer_directory.string())
      << ",\"binary_file\":"
      << JsonQuote(fs::canonical("/proc/self/exe").string())
      << ",\"compute_type\":\"native BF16 activations/MMA operands; FP32 "
         "master weights/reductions/logits\""
      << ",\"window_policy\":\"last1024 IDs; absolute positions reset to0; pad "
         "last input ID\""
      << ",\"sampling\":\"FP32(logit-max)/double_temperature; double exp; "
         "std::discrete_distribution<int>\""
      << ",\"rng\":\"std::mt19937(seed+1); verified "
         "generate_canonical<double,53>\""
      << ",\"cdf_interval_convention\":\"lower_bound: lower exclusive, upper "
         "inclusive; initial lower endpoint zero included\""
      << ",\"initial_rng_state\":" << JsonQuote(initial_rng_state.str())
      << ",\"final_rng_state\":" << JsonQuote(final_rng_state.str())
      << ",\"cpp_version\":" << __cplusplus
      << ",\"compiler\":" << JsonQuote(__VERSION__)
#ifdef __GLIBCXX__
      << ",\"libstdcxx_date\":" << __GLIBCXX__
#endif
      << ",\"logits_file\":\"logits.f32\",\"logits_shape\":[" << steps << ','
      << kVocabulary << ']'
      << ",\"events_file\":\"events.jsonl\",\"byte_order\":\"little\""
      << ",\"files\":{\"logits\":{\"file\":\"logits.f32\",\"dtype\":"
         "\"float32\",\"shape\":["
      << steps << ',' << kVocabulary
      << "]},"
         "\"tokens\":{\"file\":\"tokens.i32\",\"dtype\":\"int32\",\"shape\":["
      << history.size()
      << "]},"
         "\"events\":{\"file\":\"events.jsonl\"},\"generated_bytes\":{\"file\":"
         "\"generated.bin\",\"bytes\":"
      << generated_bytes.size() << "}}"
      << ",\"checks\":{\"initial_token_roundtrip\":true,\"generated_piece_"
         "roundtrip\":true,"
         "\"all_sampling_rng_states_verified\":true,\"all_sampling_cdf_"
         "selections_verified\":true,"
         "\"all_logical_logits_finite\":true,\"checkpoint_stats_unchanged\":"
         "true}"
      << ",\"optimizer_steps\":0,\"backward_calls\":0,\"checkpoint_writes\":0"
      << ",\"integrity_hashes\":\"external runner hashes full "
         "inputs/source/binary and outputs\""
      << ",\"elapsed_seconds\":"
      << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       started)
             .count()
      << "}\n";
  const auto json = metadata.str();
  RETURN_IF_ERROR(
      WriteExclusive(output / "metadata.json", json.data(), json.size()));
  std::cout << "Complete: " << output << std::endl;
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::weight_analysis

int main(int argc, char** argv) {
  const auto remaining = absl::ParseCommandLine(argc, argv);
  if (remaining.size() != 1) {
    std::cerr << "Unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::weight_analysis::Run();
  if (status.ok())
    return 0;
  std::cerr << status << '\n';
  return 1;
}
