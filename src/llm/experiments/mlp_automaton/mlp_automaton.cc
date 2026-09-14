#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <syncstream>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "src/cuda/thread_pool.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/llm/experiments/mlp_automaton/graph.h"
#include "src/llm/experiments/mlp_automaton/history.h"
#include "src/llm/experiments/mlp_automaton/model.h"
#include "src/util/status_macros.h"

ABSL_FLAG(
    std::string, checkpoint, "",
    "GPT-2 checkpoint directory, or parent containing step_N directories");
ABSL_FLAG(int, mlp_block, 0,
          "Zero-based transformer block to read (0 through 7); omit to analyze "
          "all blocks");
ABSL_FLAG(std::string, tokenizer, "datasets/tokenizer/gpt2",
          "Directory containing the matching GPT-2 tokenizer.json");
ABSL_FLAG(std::string, corpus, "testdata/shakespeare.txt",
          "Text corpus used for training; only paths present in its training "
          "portion are printed or saved in samples.json");
ABSL_FLAG(double, test_fraction, 0.1,
          "Held-out fraction, matching the training run; zero means corpus "
          "already contains only training text");
ABSL_FLAG(
    std::string, output_dir, "",
    "New output directory; checkpoint histories have step_N/ subdirectories "
    "and combined_history.txt/json");
ABSL_FLAG(int, batch_size, 256, "GPU batch rows; positive multiple of 16");
ABSL_FLAG(int, threads, -1,
          "Checkpoint workers; positive count, or -1 for one per logical CPU. "
          "Each active worker keeps a private GPU readout");
ABSL_FLAG(double, threshold, 0.75,
          "Keep edges with probability strictly greater than this [0.5,1)");
ABSL_FLAG(int, samples, 50,
          "Number of distinct random starting tokens per block before corpus "
          "filtering");
ABSL_FLAG(int, max_tokens, 16, "Maximum tokens per path, including the start");
ABSL_FLAG(uint64_t, seed, 17, "Seed for sampling starting tokens");
ABSL_FLAG(std::vector<std::string>, start_tokens, {},
          "Additional explicit starting token IDs, comma-separated");

namespace pluto::llm::mlp_automaton {
namespace {

absl::StatusOr<std::filesystem::path> AbsolutePath(
    const std::filesystem::path& path) {
  std::error_code error;
  auto absolute = std::filesystem::absolute(path, error);
  if (error)
    return absl::InternalError(absl::StrCat(
        "cannot resolve path ", path.string(), ": ", error.message()));
  return absolute;
}

// Shared configuration is fixed for every checkpoint so changing a range
// reflects the weights, not a different tokenizer, corpus or sampling rule.
struct ScanOptions {
  bool all_blocks;
  int requested_block;
  std::string tokenizer;
  std::string corpus_path;
  double test_fraction;
  int batch_size;
  int sample_count;
  int max_tokens;
  uint64_t seed;
  std::vector<int> starts;
  bool history;
};

struct CheckpointResult {
  std::vector<CombinedPath> sampled;
  std::vector<CombinedPath> complete;
};

// Retain the existing graph/sample layout within each checkpoint's output.
// Complete memberships are needed only for history: they let a text sampled
// later be attributed to earlier checkpoints without rerunning GPU inference.
absl::StatusOr<CheckpointResult> ScanCheckpoint(
    cuda::Executor& executor, Layer& readout, const Graph& vocabulary,
    const TextCorpus& training, const ScanOptions& options,
    const std::filesystem::path& checkpoint,
    const std::filesystem::path& output_directory) {
  const auto& [all_blocks, requested_block, tokenizer, corpus_path,
               test_fraction, batch_size, sample_count, max_tokens, seed,
               starts, history] = options;
  const int vocab_size = static_cast<int>(vocabulary.token_bytes.size());
  const double threshold = vocabulary.threshold;
  std::error_code error;
  std::vector<BlockPaths> analyses;
  std::vector<int> block_ids;
  const int first_block = all_blocks ? 0 : requested_block;
  const int end_block =
      all_blocks ? kGpt2TransformerBlockCount : requested_block + 1;
  for (int mlp_block = first_block; mlp_block < end_block; ++mlp_block) {
    block_ids.push_back(mlp_block);
    const auto block_output =
        all_blocks ? output_directory / ("block_" + std::to_string(mlp_block))
                   : output_directory;
    if (all_blocks &&
        (!std::filesystem::create_directory(block_output, error) || error))
      return absl::FailedPreconditionError("could not create block output");
    Graph graph = vocabulary;
    RETURN_IF_ERROR(LoadMlpWeights(executor, readout, checkpoint, mlp_block));
    const auto began = std::chrono::steady_clock::now();
    auto last_update = began;
    std::osyncstream(std::cout) << "Loaded " << checkpoint << " isolated B"
                                << mlp_block << " MLP. Scanning " << vocab_size
                                << " tokens, batch_size=" << batch_size
                                << ", p > " << threshold << std::endl;
    ASSIGN_OR_RETURN(
        auto transitions,
        ScanVocabulary(
            executor, readout, vocab_size, batch_size,
            [&](int completed) {
              const auto now = std::chrono::steady_clock::now();
              if (completed == vocab_size ||
                  now - last_update >= std::chrono::seconds(5)) {
              std::osyncstream(std::cout)
                  << checkpoint << " B" << mlp_block << ": scanned "
                  << completed << '/' << vocab_size << " tokens in "
                  << std::chrono::duration<double>(now - began).count()
                  << " seconds" << std::endl;
              last_update = now;
              }
            }));
    for (int token = 0; token < vocab_size; ++token) {
      if (transitions[token].probability > threshold) {
        graph.edges.push_back(
            {token, transitions[token].token, transitions[token].probability});
      }
    }
    ASSIGN_OR_RETURN(auto paths,
                     SamplePaths(graph, sample_count, max_tokens, seed));
    for (int token : starts) {
      ASSIGN_OR_RETURN(auto path, Walk(graph, token, max_tokens));
      paths.push_back(std::move(path));
    }
    const size_t candidate_count = paths.size();
    ASSIGN_OR_RETURN(paths, FilterPathsInCorpus(graph, paths, training.text()));
    std::ofstream graph_file(block_output / "graph.json");
    RETURN_IF_ERROR(WriteGraphJson(graph_file, graph));
    graph_file.close();
    std::ofstream samples_file(block_output / "samples.json");
    RETURN_IF_ERROR(WritePathsJson(samples_file, graph, paths));
    samples_file.close();
    std::ofstream metadata(block_output / "metadata.txt");
    metadata
        << std::setprecision(17) << "checkpoint=" << checkpoint.string()
        << "\ntokenizer=" << tokenizer << "\ncorpus=" << corpus_path
        << "\ntest_fraction=" << test_fraction
        << "\ntraining_bytes=" << training.size()
        << "\npath_filter=exact_training_substring"
        << "\nblock_membership=complete_walk_with_at_least_one_edge"
        << "\nmlp_block=" << mlp_block
        << "\nmlp_checkpoint_indices=" << 8 + 12 * mlp_block << ".."
        << 13 + 12 * mlp_block << "\nformula=x=E[token]; h=x+FC2_B" << mlp_block
        << "(GELU(FC1_B" << mlp_block << "(LN2_B" << mlp_block
        << "(x)))); logits=LN_final(h)*E^T\n"
        << "position_embeddings=false\nattention=false\nother_blocks=false\n"
        << "activations=BF16\nmaster_weights=FP32\nlogits=FP32\n"
        << "softmax_reductions=FP32\ntemperature=1\n"
        << "vocab_size=" << kGpt2VocabularySize
        << "\npadded_vocab_size=" << kGpt2PaddedVocabularySize
        << "\nmodel_width=" << kGpt2ModelWidth
        << "\nfeed_forward_width=" << kGpt2FeedForwardWidth
        << "\nthreshold_strictly_greater_than=" << threshold
        << "\nbatch_size=" << batch_size << "\nseed=" << seed
        << "\nmax_tokens=" << max_tokens << "\nrandom_samples=" << sample_count
        << "\nstart_tokens=" << absl::StrJoin(starts, ",")
        << "\ncandidate_paths=" << candidate_count
        << "\nedges=" << graph.edges.size() << "\npaths=" << paths.size()
        << "\n";
    metadata.close();
    if (!graph_file || !samples_file || !metadata)
      return absl::InternalError("could not finish writing automaton output");
    std::osyncstream(std::cout)
        << "Wrote " << graph.token_bytes.size() << " nodes, "
        << graph.edges.size() << " edges, " << paths.size()
        << " training-text paths (from " << candidate_count
        << " candidates) to " << block_output << '\n'
        << "B" << mlp_block << " scan complete.\n";
    analyses.push_back({mlp_block, std::move(graph), std::move(paths)});
  }

  ASSIGN_OR_RETURN(auto combined, CombinePaths(analyses, max_tokens));
  std::ofstream combined_file(output_directory / "combined_paths.json");
  RETURN_IF_ERROR(WriteCombinedPathsJson(combined_file, combined));
  combined_file.close();
  if (!combined_file)
    return absl::InternalError("could not finish writing combined paths");

  if (all_blocks) {
    // Per-block metadata retains each graph's exact formula, weights, and
    // sample counts. The root records the shared configuration and merge rule.
    std::ofstream metadata(output_directory / "metadata.txt");
    metadata << std::setprecision(17) << "checkpoint=" << checkpoint.string()
             << "\ntokenizer=" << tokenizer << "\ncorpus=" << corpus_path
             << "\ntest_fraction=" << test_fraction
             << "\ntraining_bytes=" << training.size()
             << "\nmlp_blocks=" << absl::StrJoin(block_ids, ",")
             << "\npath_filter=exact_training_substring"
             << "\nblock_membership=complete_walk_with_at_least_one_edge"
             << "\npath_identity=exact_decoded_bytes"
             << "\nthreshold_strictly_greater_than=" << threshold
             << "\nbatch_size=" << batch_size
             << "\nrandom_samples_per_block=" << sample_count
             << "\nseed=" << seed << "\nmax_tokens=" << max_tokens
             << "\nstart_tokens=" << absl::StrJoin(starts, ",")
             << "\ncombined_paths=" << combined.size() << '\n';
    metadata.close();
    if (!metadata)
      return absl::InternalError("could not finish writing combined metadata");
  }
  if (!history) {
    std::osyncstream(std::cout)
        << "Combined " << combined.size()
        << " distinct training-text paths across MLP blocks "
        << absl::StrJoin(block_ids, ", ") << ".\n";
    for (const auto& path : combined)
      std::osyncstream(std::cout)
          << '"' << absl::CEscape(path.bytes)
          << "\" (MLP blocks: " << absl::StrJoin(path.mlp_blocks, ", ")
          << ")\n";
  }
  std::vector<CombinedPath> complete;
  if (history && (sample_count > 0 || !starts.empty())) {
    ASSIGN_OR_RETURN(complete, CollectAllPaths(analyses, max_tokens));
  }
  return CheckpointResult{std::move(combined), std::move(complete)};
}

absl::Status Run() {
  std::filesystem::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const bool all_blocks = !FLAGS_mlp_block.IsSpecifiedOnCommandLine();
  const int requested_block = absl::GetFlag(FLAGS_mlp_block);
  if (requested_block < 0 || requested_block >= kGpt2TransformerBlockCount)
    return absl::InvalidArgumentError("mlp_block must be in [0, 8)");
  std::string tokenizer = absl::GetFlag(FLAGS_tokenizer);
  std::string corpus_path = absl::GetFlag(FLAGS_corpus);
  const double test_fraction = absl::GetFlag(FLAGS_test_fraction);
  std::filesystem::path output_directory = absl::GetFlag(FLAGS_output_dir);
  const int batch_size = absl::GetFlag(FLAGS_batch_size);
  const int num_threads = absl::GetFlag(FLAGS_threads);
  const int sample_count = absl::GetFlag(FLAGS_samples);
  const int max_tokens = absl::GetFlag(FLAGS_max_tokens);
  const double threshold = absl::GetFlag(FLAGS_threshold);
  if (checkpoint.empty() || tokenizer.empty() || output_directory.empty() ||
      batch_size <= 0 || batch_size % 16 != 0 || sample_count < 0 ||
      max_tokens <= 0 || !std::isfinite(threshold) || threshold < 0.5 ||
      threshold >= 1.0) {
    return absl::InvalidArgumentError(
        "supply --checkpoint, --tokenizer and a new --output_dir; "
        "batch_size must be a positive multiple of 16, samples >= 0, "
        "max_tokens > 0 and 0.5 <= threshold < 1");
  }
  if (num_threads != -1 && num_threads <= 0)
    return absl::InvalidArgumentError("threads must be positive or -1");
  if (corpus_path.empty() || !std::isfinite(test_fraction) ||
      test_fraction < 0.0 || test_fraction >= 1.0)
    return absl::InvalidArgumentError(
        "supply --corpus and a finite test_fraction in [0, 1)");
  // Resolve once with nonthrowing filesystem APIs before producing output.
  // Checkpoint discovery then returns absolute child paths as well.
  ASSIGN_OR_RETURN(checkpoint, AbsolutePath(checkpoint));
  ASSIGN_OR_RETURN(output_directory, AbsolutePath(output_directory));
  ASSIGN_OR_RETURN(auto tokenizer_path, AbsolutePath(tokenizer));
  ASSIGN_OR_RETURN(auto corpus_file, AbsolutePath(corpus_path));
  tokenizer = tokenizer_path.string();
  corpus_path = corpus_file.string();
  ASSIGN_OR_RETURN(auto selection, DiscoverCheckpoints(checkpoint));
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(corpus_path));
  TextCorpus training = corpus;
  if (test_fraction > 0.0) {
    ASSIGN_OR_RETURN(auto split, SplitCorpus(corpus, test_fraction));
    training = std::move(split.training);
  }
  std::vector<int> starts;
  for (const std::string& text : absl::GetFlag(FLAGS_start_tokens)) {
    int token;
    if (!absl::SimpleAtoi(text, &token) || token < 0 ||
        token >= kGpt2VocabularySize) {
      return absl::InvalidArgumentError("start_tokens contains an invalid ID");
    }
    starts.push_back(token);
  }
  ASSIGN_OR_RETURN(auto decoder, tokenizer::Gpt2Detokenizer::Load(tokenizer));
  if (decoder->vocab_size() != kGpt2VocabularySize) {
    return absl::InvalidArgumentError(
        "expected a 50,257-token GPT-2 vocabulary");
  }
  // Never overwrite a prior graph (or a checkpoint) by accident. Creation is
  // exclusive at the directory level; a failed run may leave a partial output.
  std::error_code error;
  if (!std::filesystem::create_directories(output_directory, error) || error) {
    return absl::FailedPreconditionError(
        "output_dir must be a new, writable directory");
  }
  Graph vocabulary;
  vocabulary.threshold = threshold;
  vocabulary.eos_token_id = decoder->eos_token_id();
  for (int token = 0; token < decoder->vocab_size(); ++token) {
    ASSIGN_OR_RETURN(auto bytes, decoder->Decode({&token, 1}));
    vocabulary.token_bytes.push_back(std::move(bytes));
  }
  ASSIGN_OR_RETURN(auto pool, cuda::ThreadPool::Create(num_threads));
  const ScanOptions options{
      all_blocks,        requested_block,     tokenizer,
      corpus_path,       test_fraction,       batch_size,
      sample_count,      max_tokens,          absl::GetFlag(FLAGS_seed),
      std::move(starts), selection.is_history};
  const size_t checkpoint_count = selection.checkpoints.size();
  const size_t worker_count = static_cast<size_t>(pool->size());
  std::vector<CheckpointResult> results(checkpoint_count);
  std::atomic<size_t> next_worker{0};
  RETURN_IF_ERROR(
      pool->ParallelFor([&](cuda::Executor& executor) -> absl::Status {
        // ParallelFor calls us once per worker. Balanced contiguous slices
        // cover every checkpoint exactly once, including when the count is not
        // divisible by the pool size. This arithmetic never multiplies two
        // large counts.
        const size_t worker =
            next_worker.fetch_add(1, std::memory_order_relaxed);
        const size_t quotient = checkpoint_count / worker_count;
        const size_t remainder = checkpoint_count % worker_count;
        const size_t begin = worker * quotient + std::min(worker, remainder);
        const size_t end = begin + quotient + (worker < remainder ? 1 : 0);
        if (begin == end)
          return absl::OkStatus();

        // Loading changes weights, and every GPU allocation belongs to its
        // worker's Executor. Reuse this private model across the slice and its
        // serial block scans; never share a readout between concurrent
        // checkpoints.
        ASSIGN_OR_RETURN(auto readout, CreateReadout(executor));
        for (size_t index = begin; index < end; ++index) {
          const auto& selected = selection.checkpoints[index];
          const auto checkpoint_output =
              selection.is_history
                  ? output_directory / ("step_" + std::to_string(selected.step))
                  : output_directory;
          if (selection.is_history) {
            std::error_code directory_error;
            if (!std::filesystem::create_directory(checkpoint_output,
                                                   directory_error) ||
                directory_error)
              return absl::FailedPreconditionError(
                  absl::StrCat("could not create checkpoint output: ",
                               checkpoint_output.string()));
            std::osyncstream(std::cout)
                << "Checkpoint " << selected.step << " (" << index + 1 << '/'
                << checkpoint_count << "): " << selected.directory << std::endl;
          }
          auto result =
              ScanCheckpoint(executor, *readout, vocabulary, training, options,
                             selected.directory, checkpoint_output);
          if (!result.ok())
            return absl::Status(
                result.status().code(),
                absl::StrCat("checkpoint ", selected.directory.string(), ": ",
                             result.status().message()));
          // The vector never resizes, and only this worker touches this slot.
          results[index] = std::move(*result);
        }
        return absl::OkStatus();
      }));
  HistoryAccumulator history;
  std::vector<int64_t> steps;
  for (size_t index = 0; index < checkpoint_count; ++index) {
    const auto& selected = selection.checkpoints[index];
    const auto& result = results[index];
    if (selection.is_history) {
      // Completion order must not affect ranges or attribution. Merge only on
      // the caller, in discovered numeric step order, after every scan
      // succeeds. Only sampled strings can reach the final report, and samples
      // have already passed the corpus filter. Keep complete graph memberships
      // without searching the corpus for every unsampled vocabulary path.
      RETURN_IF_ERROR(history.AddCheckpoint(selected.step, result.complete,
                                            result.sampled));
      steps.push_back(selected.step);
    }
  }
  if (!selection.is_history)
    return absl::OkStatus();

  const auto combined = history.Finish();
  std::ofstream json(output_directory / "combined_history.json");
  RETURN_IF_ERROR(WriteHistoryJson(json, combined, steps));
  json.close();
  std::ofstream text(output_directory / "combined_history.txt");
  RETURN_IF_ERROR(WriteHistoryText(text, combined));
  text.close();
  std::ofstream metadata(output_directory / "metadata.txt");
  metadata << std::setprecision(17)
           << "checkpoint_parent=" << checkpoint.string()
           << "\ncheckpoint_steps=" << absl::StrJoin(steps, ",")
           << "\ncheckpoint_count=" << steps.size()
           << "\nthreads=" << pool->size() << "\ntokenizer=" << tokenizer
           << "\ncorpus=" << corpus_path << "\ntest_fraction=" << test_fraction
           << "\ntraining_bytes=" << training.size() << "\nmlp_blocks="
           << (all_blocks ? "all" : std::to_string(requested_block))
           << "\npath_filter=exact_training_substring"
           << "\nblock_membership=complete_walk_with_at_least_one_edge"
           << "\npath_identity=exact_decoded_bytes"
           << "\ncandidates=union_of_per_checkpoint_samples"
           << "\nrange_semantics=consecutive_analyzed_checkpoints"
           << "\nthreshold_strictly_greater_than=" << threshold
           << "\nbatch_size=" << batch_size
           << "\nrandom_samples_per_block_per_checkpoint=" << sample_count
           << "\nseed=" << options.seed << "\nmax_tokens=" << max_tokens
           << "\nstart_tokens=" << absl::StrJoin(options.starts, ",")
           << "\ncombined_paths=" << combined.size() << '\n';
  metadata.close();
  if (!json || !text || !metadata)
    return absl::InternalError("could not finish writing checkpoint history");
  std::cout << "Combined " << combined.size()
            << " distinct training-text paths across " << steps.size()
            << " checkpoints. Ranges cover consecutive analyzed checkpoints.\n";
  RETURN_IF_ERROR(WriteHistoryText(std::cout, combined));
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::mlp_automaton

int main(int argc, char** argv) {
  const auto positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "Use named flags; positional arguments are not accepted.\n";
    return 1;
  }
  const absl::Status status = pluto::llm::mlp_automaton::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
