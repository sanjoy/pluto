#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "src/dataset/detokenizer.h"
#include "src/llm/recipes/mlp_automaton/graph.h"
#include "src/llm/recipes/mlp_automaton/model.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "GPT-2 checkpoint directory to read");
ABSL_FLAG(std::string, tokenizer, "datasets/tokenizer/gpt2",
          "Directory containing the matching GPT-2 tokenizer.json");
ABSL_FLAG(std::string, output_dir, "",
          "New directory for graph.json, samples.json and metadata.txt");
ABSL_FLAG(int, batch_size, 256, "GPU batch rows; positive multiple of 16");
ABSL_FLAG(double, threshold, 0.75,
          "Keep edges with probability strictly greater than this [0.5,1)");
ABSL_FLAG(int, samples, 50,
          "Number of distinct random starting tokens to sample");
ABSL_FLAG(int, max_tokens, 16, "Maximum tokens per path, including the start");
ABSL_FLAG(uint64_t, seed, 17, "Seed for sampling starting tokens");
ABSL_FLAG(std::vector<std::string>, start_tokens, {},
          "Additional explicit starting token IDs, comma-separated");

namespace pluto::llm::mlp_automaton {
namespace {

absl::Status Run() {
  const std::string checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const std::string tokenizer = absl::GetFlag(FLAGS_tokenizer);
  const std::filesystem::path output_directory =
      absl::GetFlag(FLAGS_output_dir);
  const int batch_size = absl::GetFlag(FLAGS_batch_size);
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
  Graph graph;
  graph.threshold = threshold;
  graph.eos_token_id = decoder->eos_token_id();
  for (int token = 0; token < decoder->vocab_size(); ++token) {
    ASSIGN_OR_RETURN(auto bytes, decoder->Decode({&token, 1}));
    graph.token_bytes.push_back(std::move(bytes));
  }
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto readout, CreateReadout(*executor));
  RETURN_IF_ERROR(LoadB0Weights(*executor, *readout, checkpoint));
  const auto began = std::chrono::steady_clock::now();
  auto last_update = began;
  std::cout << "Loaded isolated B0 MLP. Scanning " << decoder->vocab_size()
            << " tokens, batch_size=" << batch_size << ", p > " << threshold
            << std::endl;
  ASSIGN_OR_RETURN(
      auto transitions,
      ScanVocabulary(
          *executor, *readout, decoder->vocab_size(), batch_size,
          [&](int completed) {
            const auto now = std::chrono::steady_clock::now();
            if (completed == decoder->vocab_size() ||
                now - last_update >= std::chrono::seconds(5)) {
              std::cout << "Scanned " << completed << '/'
                        << decoder->vocab_size() << " tokens in "
                        << std::chrono::duration<double>(now - began).count()
                        << " seconds" << std::endl;
              last_update = now;
            }
          }));
  for (int token = 0; token < decoder->vocab_size(); ++token) {
    if (transitions[token].probability > threshold) {
      graph.edges.push_back(
          {token, transitions[token].token, transitions[token].probability});
    }
  }
  ASSIGN_OR_RETURN(auto paths, SamplePaths(graph, sample_count, max_tokens,
                                           absl::GetFlag(FLAGS_seed)));
  for (int token : starts) {
    ASSIGN_OR_RETURN(auto path, Walk(graph, token, max_tokens));
    paths.push_back(std::move(path));
  }
  std::ofstream graph_file(output_directory / "graph.json");
  RETURN_IF_ERROR(WriteGraphJson(graph_file, graph));
  graph_file.close();
  std::ofstream samples_file(output_directory / "samples.json");
  RETURN_IF_ERROR(WritePathsJson(samples_file, graph, paths));
  samples_file.close();
  std::ofstream metadata(output_directory / "metadata.txt");
  metadata << std::setprecision(17)
           << "checkpoint=" << std::filesystem::absolute(checkpoint).string()
           << "\ntokenizer=" << std::filesystem::absolute(tokenizer).string()
           << "\nformula=x=E[token]; h=x+FC2(GELU(FC1(LN2_B0(x)))); "
              "logits=LN_final(h)*E^T\n"
           << "position_embeddings=false\nattention=false\nlater_blocks=false\n"
           << "activations=BF16\nmaster_weights=FP32\nlogits=FP32\n"
           << "softmax_reductions=FP32\ntemperature=1\n"
           << "vocab_size=" << kGpt2VocabularySize
           << "\npadded_vocab_size=" << kGpt2PaddedVocabularySize
           << "\nmodel_width=" << kGpt2ModelWidth
           << "\nfeed_forward_width=" << kGpt2FeedForwardWidth
           << "\nthreshold_strictly_greater_than=" << threshold
           << "\nbatch_size=" << batch_size
           << "\nseed=" << absl::GetFlag(FLAGS_seed)
           << "\nmax_tokens=" << max_tokens
           << "\nrandom_samples=" << sample_count
           << "\nstart_tokens=" << absl::StrJoin(starts, ",")
           << "\nedges=" << graph.edges.size() << "\npaths=" << paths.size()
           << "\n";
  metadata.close();
  if (!graph_file || !samples_file || !metadata) {
    return absl::InternalError("could not finish writing automaton output");
  }
  std::cout << "Wrote " << graph.token_bytes.size() << " nodes, "
            << graph.edges.size() << " edges, " << paths.size()
            << " sampled paths to " << output_directory << '\n'
            << "Paths are local token transitions, not necessarily words.\n";
  for (const auto& path : paths) {
    std::string bytes;
    std::cout << '[';
    for (size_t index = 0; index < path.tokens.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << path.tokens[index];
      bytes += graph.token_bytes[path.tokens[index]];
    }
    std::cout << "] \"" << absl::CEscape(bytes) << "\" ("
              << TerminationName(path.termination) << ")\n";
  }
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
