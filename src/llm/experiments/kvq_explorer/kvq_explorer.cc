#include <iomanip>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/experiments/kvq_explorer/readout.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "GPT-2 recipe checkpoint directory");
ABSL_FLAG(std::string, tokenizer, "datasets/tokenizer/gpt2",
          "Directory containing the matching GPT-2 tokenizer.json");
ABSL_FLAG(std::string, prompt, "", "Input string to tokenize and inspect");

namespace pluto::llm::kvq_explorer {
namespace {

absl::Status Run() {
  const std::string checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const std::string prompt = absl::GetFlag(FLAGS_prompt);
  if (checkpoint.empty() || prompt.empty())
    return absl::InvalidArgumentError(
        "--checkpoint and nonempty --prompt are required");
  ASSIGN_OR_RETURN(auto tokenizer, tokenizer::Gpt2Tokenizer::Load(
                                       absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto decoder, tokenizer::Gpt2Detokenizer::Load(
                                     absl::GetFlag(FLAGS_tokenizer)));
  if (tokenizer->vocab_size() != kGpt2VocabularySize ||
      decoder->vocab_size() != kGpt2VocabularySize)
    return absl::InvalidArgumentError("tokenizer must have 50,257 tokens");
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto tokens, tokenizer->Encode(*executor, prompt));
  if (tokens.empty())
    return absl::InvalidArgumentError("prompt produced no tokens");
  ASSIGN_OR_RETURN(auto readout, Readout::Load(*executor, checkpoint));
  ASSIGN_OR_RETURN(auto results, readout->Explore(*executor, tokens));
  std::cout << "Checkpoint: " << checkpoint << "\n"
            << "Isolated FP32 readout: softmax((E[token] W_s + b_s) E^T).\n"
            << "No positions, LayerNorm or context; all heads concatenated.\n"
            << "Probabilities normalize over all 50,257 tokens, not just the "
               "top 3.\n";
  std::cout << std::setprecision(8);
  for (size_t position = 0; position < tokens.size(); ++position) {
    const int token = tokens[position];
    ASSIGN_OR_RETURN(auto bytes, decoder->Decode({&token, 1}));
    std::cout << "\nToken " << position << ": id=" << token << " text=\""
              << absl::CEscape(bytes) << "\"\n";
    for (int block = 0; block < kGpt2TransformerBlockCount; ++block) {
      std::cout << "  Block " << block << "\n";
      for (const Projection projection : {kKey, kValue, kQuery}) {
        const auto& top =
            results[(position * kGpt2TransformerBlockCount + block) * 3 +
                    projection];
        std::cout << "    "
                  << (projection == kKey     ? 'K'
                      : projection == kValue ? 'V'
                                             : 'Q')
                  << ':';
        for (int rank = 0; rank < 3; ++rank) {
          const int winner = top.tokens[rank];
          ASSIGN_OR_RETURN(auto text, decoder->Decode({&winner, 1}));
          std::cout << "  id=" << winner << " text=\"" << absl::CEscape(text)
                    << "\" p=" << top.probabilities[rank];
        }
        std::cout << '\n';
      }
    }
  }
  if (!std::cout)
    return absl::InternalError("failed writing KVQ results");
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::kvq_explorer

int main(int argc, char** argv) {
  const auto positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "Use --prompt to supply the input string.\n";
    return 1;
  }
  const auto status = pluto::llm::kvq_explorer::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
