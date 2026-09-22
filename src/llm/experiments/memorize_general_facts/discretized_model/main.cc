#include <charconv>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"

namespace pluto::llm::discretized {
// Separate generated targets. Neither is visible to the integer model.
absl::Status VerifyGeneratedModel(const Model& model, std::ostream& output);
absl::StatusOr<std::vector<TokenId>> EncodeGeneratedPrompt(
    absl::string_view text);
}  // namespace pluto::llm::discretized

namespace {
namespace dm = pluto::llm::discretized;

absl::StatusOr<std::vector<dm::TokenId>> ParseIds(absl::string_view text) {
  std::vector<dm::TokenId> ids;
  while (!text.empty()) {
    const size_t separator = text.find(',');
    const auto item = text.substr(0, separator);
    dm::TokenId id;
    auto result = std::from_chars(item.data(), item.data() + item.size(), id);
    if (item.empty() || result.ec != std::errc() ||
        result.ptr != item.data() + item.size())
      return absl::InvalidArgumentError(
          "--token_ids requires comma-separated integer compact IDs");
    ids.push_back(id);
    if (separator == absl::string_view::npos)
      break;
    text.remove_prefix(separator + 1);
    if (text.empty())
      return absl::InvalidArgumentError("--token_ids has a trailing comma");
  }
  if (ids.empty())
    return absl::InvalidArgumentError("--token_ids must not be empty");
  return ids;
}

int Run(int argc, char** argv) {
  bool verify = false;
  bool have_prompt = false;
  bool have_ids = false;
  std::string prompt;
  std::string raw_ids;
  size_t limit = 64;
  bool have_limit = false;
  for (int i = 1; i < argc; ++i) {
    absl::string_view arg(argv[i]);
    if (arg == "--help") {
      std::cout
          << "Usage: discretized_model --verify\n"
             "       discretized_model --token_ids=COMPACT,IDS "
             "[--generation_tokens=N]\n"
             "       discretized_model --prompt=TEXT [--generation_tokens=N]\n"
             "Text encoding accepts captured corpus prefixes at token "
             "boundaries only.\n"
             "Inference executes integer tables and fails for unsupported "
             "histories.\n";
      return 0;
    }
    if (arg == "--verify" && !verify) {
      verify = true;
    } else if (arg.substr(0, 9) == "--prompt=" && !have_prompt) {
      have_prompt = true;
      prompt = std::string(arg.substr(9));
    } else if (arg.substr(0, 12) == "--token_ids=" && !have_ids) {
      have_ids = true;
      raw_ids = std::string(arg.substr(12));
    } else if (arg.substr(0, 20) == "--generation_tokens=" && !have_limit) {
      have_limit = true;
      const auto number = arg.substr(20);
      const auto result =
          std::from_chars(number.data(), number.data() + number.size(), limit);
      if (number.empty() || result.ec != std::errc() ||
          result.ptr != number.data() + number.size()) {
        std::cerr << "invalid --generation_tokens\n";
        return 2;
      }
    } else {
      std::cerr << "unknown or repeated option: " << arg << "\n";
      return 2;
    }
  }
  if ((verify && (have_prompt || have_ids || have_limit)) ||
      (!verify && have_prompt == have_ids)) {
    std::cerr << "choose exactly --verify, --prompt=TEXT, or --token_ids=IDS\n";
    return 2;
  }
  const auto& model = dm::GeneratedModel();
  auto valid = dm::ValidateModel(model);
  if (!valid.ok()) {
    std::cerr << valid << "\n";
    return 1;
  }
  if (verify) {
    auto result = dm::VerifyGeneratedModel(model, std::cout);
    if (!result.ok())
      std::cerr << result << "\n";
    return result.ok() ? 0 : 1;
  }
  auto tokens =
      have_ids ? ParseIds(raw_ids) : dm::EncodeGeneratedPrompt(prompt);
  if (!tokens.ok()) {
    std::cerr << tokens.status() << "\n";
    return 2;
  }
  auto continuation = dm::Generate(model, *tokens, limit);
  if (!continuation.ok()) {
    std::cerr << continuation.status() << "\n";
    return 1;
  }
  for (dm::TokenId token : *continuation) {
    if (token == model.eos_token)
      break;
    tokens->push_back(token);
  }
  auto decoded = dm::Decode(model, *tokens);
  if (!decoded.ok()) {
    std::cerr << decoded.status() << "\n";
    return 1;
  }
  std::cout << *decoded << "\n";
  return 0;
}
}  // namespace

int main(int argc, char** argv) { return Run(argc, argv); }
