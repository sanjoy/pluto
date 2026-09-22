#include "src/llm/experiments/memorize_general_facts/discretized_model/generate_discretized_model.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_certificate.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_emit.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_pointwise.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_capture.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {
namespace fs = std::filesystem;

absl::Status FilesystemError(absl::string_view operation,
                             const std::error_code& error) {
  return absl::UnknownError(absl::StrCat(operation, ": ", error.message()));
}

absl::Status CheckOptions(const GeneratorOptions& options) {
  if (options.output.empty() || options.expected_samples <= 0)
    return absl::InvalidArgumentError(
        "output is required and expected_samples must be positive");
  if (options.reduce &&
      (options.reduction.neighbors < 1 || options.reduction.max_passes < 1 ||
       options.reduction.exhaustive_pair_limit < 0 ||
       (options.reduction.max_attempts && *options.reduction.max_attempts < 0)))
    return absl::InvalidArgumentError("invalid reduction settings");
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

absl::StatusOr<Json> Provenance(const GeneratorOptions& options,
                                const Json& model) {
  Json stats = model.value("stats", Json::object());
  Json merges = stats.value("accepted_merges", Json::array());
  stats.erase("accepted_merges");
  if (!merges.empty()) {
    stats["accepted_merges_sha256"] = Sha256(merges.dump());
    stats["accepted_merge_records"] = merges.size();
    double minimum = merges[0]["euclidean_distance"].get<double>();
    double maximum = minimum;
    int64_t induced = 0;
    for (const auto& merge : merges) {
      minimum = std::min(minimum, merge["euclidean_distance"].get<double>());
      maximum = std::max(maximum, merge["euclidean_distance"].get<double>());
      induced += merge["induced_unions"].get<int64_t>();
    }
    stats["merge_distance_summary"] = {{"minimum", minimum},
                                       {"maximum", maximum},
                                       {"induced_unions", induced}};
  }
  ASSIGN_OR_RETURN(auto verification, EvaluateModel(model));
  Json result = {
      {"schema", 1},
      {"protocol", absl::StrCat("first ", model["prompt_tokens"].get<int>(),
                                " tokens; autonomous suffix and explicit EOS")},
      {"verification", verification},
      {"stats", stats}};
  result["source"] = "in-memory GPU execution trace";
  if (stats.value("search", Json::object())
          .value("pairwise_irreducible", false)) {
    ASSIGN_OR_RETURN(result["irreducibility_certificate"], CertifyModel(model));
  }
  if (!options.checkpoint.empty()) {
    std::error_code error;
    std::vector<fs::path> weights;
    fs::directory_iterator iterator(options.checkpoint, error), end;
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
    weights.push_back(options.checkpoint / "compact_vocabulary.tsv");
    result["checkpoint"] = options.checkpoint.string();
    for (const auto& path : weights) {
      ASSIGN_OR_RETURN(
          result["checkpoint_files_sha256"][path.filename().string()],
          Sha256File(path));
    }
  }
  for (auto [name, path] : std::vector<std::pair<std::string, fs::path>>{
           {"corpus", options.corpus},
           {"tokenizer", options.tokenizer.empty()
                             ? fs::path{}
                             : options.tokenizer / "tokenizer.json"}})
    if (!path.empty()) {
      result[name] = path.string();
      ASSIGN_OR_RETURN(result[name + "_sha256"], Sha256File(path));
    }
  return result;
}

void Report(const GeneratorOptions& options, Json verification,
            const Json& model, const char* phase) {
  if (!options.progress)
    return;
  verification["phase"] = phase;
  verification["states"] = model["states"].size();
  if (std::string_view(phase) == "generated")
    verification["output"] = options.output.string();
  options.progress(verification);
}
}  // namespace

absl::StatusOr<Json> Generate(const GeneratorOptions& options) {
  RETURN_IF_ERROR(CheckOptions(options));
  ASSIGN_OR_RETURN(auto model, CaptureCheckpoint(options));
  return GenerateFromModel(std::move(model), options);
}

absl::StatusOr<Json> GenerateFromModel(Json model,
                                       const GeneratorOptions& options) {
  RETURN_IF_ERROR(CheckOptions(options));
  RETURN_IF_ERROR(ValidateModel(model));
  if (model["samples"].size() != static_cast<size_t>(options.expected_samples))
    return absl::InvalidArgumentError(
        "saved model has the wrong number of samples");
  ASSIGN_OR_RETURN(auto verification, EvaluateModel(model));
  Report(options, verification, model, "baseline");
  if (options.reduce) {
    auto reduction = options.reduction;
    reduction.progress = options.progress;
    ASSIGN_OR_RETURN(model, ReduceModel(model, reduction));
  }
  if (options.compact_transitions) {
    ASSIGN_OR_RETURN(model, RelabelMlpOutputs(model));
  }
  ASSIGN_OR_RETURN(verification, EvaluateModel(model));
  ASSIGN_OR_RETURN(auto provenance, Provenance(options, model));
  ASSIGN_OR_RETURN(auto files, RenderModel(model, options.state_index,
                                           options.compact_transitions));
  RETURN_IF_ERROR(FormatSources(files, options.clang_format_config));
  provenance["generator"] = "generate_discretized_model (C++)";
  provenance["transition_representation"] =
      options.compact_transitions ? "control_flow" : "tables";
  provenance["states"] = model["states"].size();
  provenance["layers"] = model["layers"];
  provenance["vocabulary_size"] = model["vocab_size"];
  provenance["integer_only_inference"] = true;
  provenance["generated_sources_sha256"] = Json::object();
  for (const auto& [name, content] : files) {
    provenance["generated_sources_sha256"][name] = Sha256(content);
    provenance["formatted_source_bytes"][name] = content.size();
  }
  files["generation_report.txt"] = TextReport(provenance);
  RETURN_IF_ERROR(PublishFiles(files, options.output));
  Report(options, verification, model, "generated");
  return model;
}

}  // namespace pluto::llm::discretized::generator
