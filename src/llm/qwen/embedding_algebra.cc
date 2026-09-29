#include "src/llm/qwen/embedding_algebra.h"

#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "linenoise.h"
#include "src/cuda/executor.h"
#include "src/dataset/qwen_tokenizer.h"
#include "src/llm/qwen/embedding_algebra_expression.h"
#include "src/llm/qwen/embedding_algebra_table.h"
#include "src/util/status_macros.h"

namespace pluto::llm::qwen {
namespace {

// A parsed expression with one verified vocabulary ID per literal symbol.
// Signs remain separate from IDs so repeated terms are summed in input order.
struct PreparedExpression {
  bool raw;
  std::vector<EmbeddingTerm> terms;
};

// Tokenize symbols independently, never the complete arithmetic expression.
// In particular, neither syntactic spaces nor a synthetic leading space become
// part of an unquoted symbol; quoted leading spaces remain significant.
absl::StatusOr<PreparedExpression> PrepareExpression(
    absl::string_view text, const tokenizer::QwenTokenizer& tokenizer) {
  ASSIGN_OR_RETURN(auto parsed, ParseAlgebraExpression(text));
  ASSIGN_OR_RETURN(auto ids, EncodeAlgebraSymbols(
                                 parsed,
                                 [&](absl::string_view symbol) {
                                   return tokenizer.Encode(symbol);
                                 }));
  PreparedExpression result{parsed.raw, {}};
  for (size_t i = 0; i < ids.size(); ++i)
    result.terms.push_back({ids[i], parsed.terms[i].coefficient});
  return result;
}

// Arithmetic uses the unnormalized input embeddings. Cosine similarity only
// affects ranking: it is not a softmax, probability or language-model output.
absl::Status PrintResult(const PreparedExpression& expression,
                         EmbeddingAlgebraTable& table,
                         const tokenizer::QwenTokenizer& tokenizer) {
  ASSIGN_OR_RETURN(auto vector, table.Evaluate(expression.terms));
  if (expression.raw) {
    // Nine significant decimal digits round-trip every finite FP32 element;
    // no components are omitted, normalized or replaced with an ellipsis.
    std::cout << '[';
    for (size_t i = 0; i < vector.size(); ++i)
      std::cout << (i == 0 ? "" : ", ") << absl::StrFormat("%.9g", vector[i]);
    std::cout << "]\n";
  } else {
    ASSIGN_OR_RETURN(auto matches, table.Nearest(vector.span(), 3));
    std::cout << "Top " << matches.size()
              << " by cosine similarity (1 = identical direction; not a "
                 "probability):\n";
    for (size_t i = 0; i < matches.size(); ++i) {
      const auto& match = matches[i];
      ASSIGN_OR_RETURN(auto text, tokenizer.Decode({match.token_id}));
      // Escape arbitrary byte-BPE spellings, including terminal controls and
      // incomplete UTF-8. Token IDs disambiguate visually similar spellings.
      std::cout << absl::StrFormat(
          "  %d. \"%s\" [token %d]  cosine=%.6f  L2=%.6f\n", i + 1,
          absl::CEscape(text), match.token_id, match.cosine_similarity,
          match.l2_distance);
    }
  }
  std::cout.flush();
  return std::cout ? absl::OkStatus()
                   : absl::InternalError("could not write embedding output");
}

void PrintHelp() {
  std::cout
      << "Embedding algebra: king - queen + boy\n"
         "  raw king - queen + boy  (or append raw): print every dimension\n"
         "  Quote exact token text, e.g. \" king\"; each symbol must encode "
         "as one token.\n"
         "  + and - are supported; quotes preserve spaces/operators.\n"
         "  Neighbors use cosine similarity, not LM-head probabilities.\n"
         "  Up/Down: session history; Ctrl-C: cancel line; Ctrl-D: exit.\n"
         "  :help shows this message; :quit exits. No history is saved "
         "to disk.\n"
      << std::flush;
}

// Linenoise provides terminal editing/history. Plain stdin avoids terminal
// escapes and prompts, which also makes scripted sessions easy to compose.
absl::Status RunSession(EmbeddingAlgebraTable& table,
                        const tokenizer::QwenTokenizer& tokenizer) {
  const bool interactive = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
  if (interactive) {
    linenoiseSetMultiLine(1);
    if (!linenoiseHistorySetMaxLen(1000))
      return absl::ResourceExhaustedError("cannot allocate command history");
    PrintHelp();
  }
  bool failed = false;
  for (;;) {
    std::string input;
    if (interactive) {
      errno = 0;
      char* line = linenoise("embedding> ");
      if (line == nullptr) {
        if (errno == EAGAIN || errno == EINTR) {
          std::cout << '\n';
          continue;
        }
        if (errno != 0 && errno != ENOENT)
          return absl::InternalError(
              absl::StrCat("terminal input failed: ", std::strerror(errno)));
        break;
      }
      input = line;
      linenoiseFree(line);
    } else if (!std::getline(std::cin, input)) {
      if (std::cin.bad())
        return absl::InternalError("could not read embedding expression");
      break;
    }
    const auto trimmed = absl::StripAsciiWhitespace(input);
    if (trimmed.empty())
      continue;
    if (interactive)
      linenoiseHistoryAdd(input.c_str());
    if (trimmed == ":quit" || trimmed == ":q" || trimmed == ":exit")
      break;
    if (trimmed == ":help") {
      PrintHelp();
      continue;
    }
    auto prepared = PrepareExpression(input, tokenizer);
    const auto status = prepared.ok() ? PrintResult(*prepared, table, tokenizer)
                                      : prepared.status();
    if (!status.ok()) {
      std::cerr << "Error: " << status << '\n';
      failed = true;
    }
  }
  return !interactive && failed
             ? absl::InvalidArgumentError("one or more expressions failed")
             : absl::OkStatus();
}

}  // namespace

absl::Status RunEmbeddingAlgebra(const std::filesystem::path& checkpoint,
                                 absl::string_view expression) {
  ASSIGN_OR_RETURN(auto tokenizer, tokenizer::QwenTokenizer::Load(checkpoint));
  // Fail malformed/multi-token one-shot inputs before loading GPU weights.
  std::optional<PreparedExpression> prepared;
  if (!expression.empty()) {
    ASSIGN_OR_RETURN(prepared, PrepareExpression(expression, *tokenizer));
  }
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto table,
                   EmbeddingAlgebraTable::Load(*executor, checkpoint,
                                               tokenizer->vocab_size()));
  std::cerr << "Loaded only input embeddings: " << table->vocab_size() << " x "
            << table->dimensions() << "; searching " << tokenizer->vocab_size()
            << " token IDs\n";
  return prepared ? PrintResult(*prepared, *table, *tokenizer)
                  : RunSession(*table, *tokenizer);
}

}  // namespace pluto::llm::qwen
