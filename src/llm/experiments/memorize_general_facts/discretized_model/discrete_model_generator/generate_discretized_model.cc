#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/generate_discretized_model.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model_util.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/code_generator.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/discretize_certificate.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/generator_report.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/model_recorder.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/state_compactor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/utils.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {
namespace fs = std::filesystem;

absl::Status FilesystemError(absl::string_view operation,
                             const std::error_code& error) {
  return absl::UnknownError(absl::StrCat(operation, ": ", error.message()));
}

absl::Status CheckOptions(const GeneratorOptions& options) {
  if (options.output.empty() || options.recorder.expected_samples <= 0)
    return absl::InvalidArgumentError(
        "output is required and expected_samples must be positive");
  if (options.compaction &&
      (options.compaction_options.neighbors < 1 ||
       options.compaction_options.max_passes < 1 ||
       options.compaction_options.exhaustive_pair_limit < 0 ||
       (options.compaction_options.max_attempts &&
        *options.compaction_options.max_attempts < 0)))
    return absl::InvalidArgumentError("invalid compaction settings");
  std::error_code error;
  auto status = fs::symlink_status(options.output, error);
  if (error && error != std::errc::no_such_file_or_directory)
    return FilesystemError("cannot inspect output", error);
  if (fs::exists(status))
    return absl::AlreadyExistsError(
        "output must be a fresh directory; refusing to overwrite");
  if (!fs::is_regular_file(options.clang_format_config, error))
    return absl::InvalidArgumentError(
        "missing declared clang-format configuration");
  // Validate host tooling before running thousands of GPU forwards.
  ASSIGN_OR_RETURN(auto formatter, FindExecutable("clang-format"));
  for (auto parent = options.output.parent_path(); !parent.empty();
       parent = parent.parent_path()) {
    auto parent_status = fs::status(parent, error);
    if (error && error != std::errc::no_such_file_or_directory)
      return FilesystemError("cannot inspect output parent", error);
    if (fs::exists(parent_status)) {
      if (!fs::is_directory(parent_status))
        return absl::InvalidArgumentError("output parent is not a directory");
      break;
    }
  }
  return absl::OkStatus();
}

// Temporary formatting work is not visible at the final destination. The
// emitter publishes the complete formatted and hashed directory with no-clobber
// rename only after every subprocess and validation step has succeeded.
absl::Status FormatSources(FileMap& files, const fs::path& style) {
  ASSIGN_OR_RETURN(auto formatter, FindExecutable("clang-format"));
  std::error_code error;
  auto temp_root = fs::temp_directory_path(error);
  if (error)
    return FilesystemError("cannot locate temporary directory", error);
  std::string temporary = (temp_root / "pluto-format.XXXXXX").string();
  if (mkdtemp(temporary.data()) == nullptr)
    return absl::UnknownError(
        absl::StrCat("cannot create format staging: ", std::strerror(errno)));
  struct Cleanup {
    fs::path directory;
    ~Cleanup() {
      std::error_code ignored;
      fs::remove_all(directory, ignored);
    }
  } cleanup{temporary};
  std::vector<std::string> names;
  for (auto& [name, content] : files) {
    fs::path path(name);
    if (path.extension() != ".h" && path.extension() != ".cc")
      continue;
    RETURN_IF_ERROR(WriteFile(fs::path(temporary) / name, content));
    names.push_back(name);
  }
  std::atomic<size_t> next{0};
  std::mutex mutex;
  absl::Status status;
  std::vector<std::thread> workers;
  // Limit concurrent formatter processes while allowing independent C++ files
  // to format in parallel. Neither filenames nor style paths enter a shell.
  for (size_t worker = 0; worker < std::min<size_t>(4, names.size()); ++worker)
    workers.emplace_back([&] {
      for (;;) {
        size_t index = next.fetch_add(1);
        if (index >= names.size())
          return;
        auto result =
            RunProcess({formatter.string(), "--style=file:" + style.string(),
                        "-i", (fs::path(temporary) / names[index]).string()});
        if (!result.ok()) {
          std::lock_guard lock(mutex);
          status.Update(result);
        }
      }
    });
  for (auto& worker : workers)
    worker.join();
  RETURN_IF_ERROR(status);
  for (const auto& name : names) {
    ASSIGN_OR_RETURN(files[name], ReadFile(fs::path(temporary) / name));
  }
  return absl::OkStatus();
}

void AppendSection(std::string& report, absl::string_view name,
                   const std::string& section) {
  absl::StrAppend(&report, name, ":\n");
  std::istringstream lines(section);
  for (std::string line; std::getline(lines, line);)
    absl::StrAppend(&report, "  ", line, "\n");
}

absl::StatusOr<std::string> Provenance(const GeneratorOptions& options,
                                       const CapturedModel& model) {
  ASSIGN_OR_RETURN(auto verification, EvaluateModel(model));
  std::string result = absl::StrCat(
      "protocol: first ", model.metadata.prompt_tokens,
      " tokens; autonomous suffix and explicit EOS\n",
      "source: in-memory GPU execution trace\n",
      "model_width: ", options.recorder.model_width,
      "\ncontext_length: ", options.recorder.context_length,
      "\nattention_heads: ", options.recorder.attention_heads,
      "\nfeed_forward_width: ", options.recorder.feed_forward_width,
      "\nnative_greedy_verified: ",
      options.recorder.verify_greedy ? "true" : "false", "\n");
  AppendSection(result, "verification", FormatVerification(verification));
  AppendSection(result, "stats", FormatStatistics(model.stats));
  if (model.stats.compaction_search &&
      model.stats.compaction_search->pairwise_compaction_complete) {
    ASSIGN_OR_RETURN(auto certificate, CertifyCompaction(model));
    AppendSection(result, "compaction_certificate",
                  FormatCertificate(certificate));
  }
  if (!options.recorder.checkpoint.empty()) {
    std::error_code error;
    std::vector<fs::path> weights;
    fs::directory_iterator iterator(options.recorder.checkpoint, error), end;
    if (error)
      return FilesystemError("cannot read checkpoint", error);
    for (; iterator != end; iterator.increment(error)) {
      if (error)
        return FilesystemError("cannot list checkpoint", error);
      const auto name = iterator->path().filename().string();
      if (name.starts_with("weight_") && name.ends_with(".bin"))
        weights.push_back(iterator->path());
    }
    if (error)
      return FilesystemError("cannot list checkpoint", error);
    if (weights.empty())
      return absl::InvalidArgumentError(
          "checkpoint must contain weights and compact_vocabulary.tsv");
    std::sort(weights.begin(), weights.end());
    weights.push_back(options.recorder.checkpoint / "compact_vocabulary.tsv");
    absl::StrAppend(&result, "checkpoint: ",
                    absl::CEscape(options.recorder.checkpoint.string()),
                    "\ncheckpoint_files_sha256:\n");
    for (const auto& path : weights) {
      ASSIGN_OR_RETURN(auto hash, Sha256File(path));
      absl::StrAppend(&result, "  ", absl::CEscape(path.filename().string()),
                      ": ", hash, "\n");
    }
  }
  for (auto [name, path] : std::vector<std::pair<std::string, fs::path>>{
           {"corpus", options.recorder.corpus},
           {"tokenizer", options.recorder.tokenizer.empty()
                             ? fs::path{}
                             : options.recorder.tokenizer / "tokenizer.json"}})
    if (!path.empty()) {
      ASSIGN_OR_RETURN(auto hash, Sha256File(path));
      absl::StrAppend(&result, name, ": ", absl::CEscape(path.string()), "\n",
                      name, "_sha256: ", hash, "\n");
    }
  return result;
}

void Report(const GeneratorOptions& options, VerificationResult verification,
            const CapturedModel& model, GenerationPhase phase) {
  if (!options.progress)
    return;
  options.progress(GenerationProgress{
      .phase = phase,
      .states = static_cast<int64_t>(model.states.size()),
      .verification = verification,
      .output = phase == GenerationPhase::kGenerated ? options.output.string()
                                                     : std::string{}});
}

// Internal pipeline continuation: only Generate can enter after checking the
// destination/tooling and capturing a valid checkpoint on the GPU.
absl::StatusOr<CapturedModel> GenerateFromModel(
    CapturedModel model, const GeneratorOptions& options,
    const CapturedStateVectors& vectors) {
  RETURN_IF_ERROR(ValidateModel(model));
  if (model.samples.size() !=
      static_cast<size_t>(options.recorder.expected_samples))
    return absl::InvalidArgumentError(
        "captured model has the wrong number of samples");
  ASSIGN_OR_RETURN(auto verification, EvaluateModel(model));
  Report(options, verification, model, GenerationPhase::kBaseline);
  if (options.compaction) {
    auto compaction = options.compaction_options;
    if (options.progress)
      compaction.progress = [&](const CompactionProgress& progress) {
        if (options.compaction_options.progress)
          options.compaction_options.progress(progress);
        options.progress(progress);
      };
    ASSIGN_OR_RETURN(model,
                     CompactModel(model, compaction, &vectors.original_states));
  }
  if (options.compact_transitions) {
    ASSIGN_OR_RETURN(model, RelabelMlpOutputs(model));
  }
  ASSIGN_OR_RETURN(verification, EvaluateModel(model));
  ASSIGN_OR_RETURN(auto provenance, Provenance(options, model));
  ASSIGN_OR_RETURN(auto files,
                   RenderModel(model, options.state_index,
                               options.compact_transitions, &vectors));
  RETURN_IF_ERROR(FormatSources(files, options.clang_format_config));
  absl::StrAppend(
      &provenance, "generator: generate_discretized_model (C++)\n",
      "transition_representation: ",
      options.compact_transitions ? "control_flow" : "tables",
      "\nstates: ", model.states.size(), "\nlayers: ", model.metadata.layers,
      "\nvocabulary_size: ", model.metadata.vocab_size,
      "\narchived_original_vectors: ", vectors.original_states.size(),
      "\narchived_vector_width: ", model.metadata.width,
      "\narchived_vector_encoding: exact_bf16_inspection_only",
      "\ninteger_only_inference: true\ngenerated_sources_sha256:\n");
  for (const auto& [name, content] : files)
    absl::StrAppend(&provenance, "  ", name, ": ", Sha256(content), "\n");
  provenance += "formatted_source_bytes:\n";
  for (const auto& [name, content] : files)
    absl::StrAppend(&provenance, "  ", name, ": ", content.size(), "\n");
  files["generation_report.txt"] = std::move(provenance);
  RETURN_IF_ERROR(PublishFiles(files, options.output));
  Report(options, verification, model, GenerationPhase::kGenerated);
  return model;
}

}  // namespace

absl::StatusOr<CapturedModel> Generate(const GeneratorOptions& options) {
  RETURN_IF_ERROR(CheckOptions(options));
  auto recorder = options.recorder;
  if (options.progress)
    recorder.progress = [&](const CaptureProgress& progress) {
      if (options.recorder.progress)
        options.recorder.progress(progress);
      options.progress(progress);
    };
  // Preserve original vectors even without compaction. Current state IDs may
  // be relabeled later, but each state's original membership remains intact.
  // The archive is immutable; inference never consults its emitted copy.
  CapturedStateVectors vectors;
  ASSIGN_OR_RETURN(auto model,
                   ModelRecorder::Record(recorder, &vectors.original_states));
  return GenerateFromModel(std::move(model), options, vectors);
}

}  // namespace pluto::llm::discretized::generator
