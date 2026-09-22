#include "src/llm/experiments/memorize_general_facts/discretized_model/naming_utils.h"

#include <algorithm>
#include <set>
#include <string>

#include "absl/strings/ascii.h"

namespace pluto::llm::discretized::generator {
namespace {

bool ValidName(absl::string_view name) {
  static const std::set<std::string> keywords = {
      "alignas",       "alignof",     "and",
      "and_eq",        "asm",         "auto",
      "bitand",        "bitor",       "bool",
      "break",         "case",        "catch",
      "char",          "char8_t",     "char16_t",
      "char32_t",      "class",       "compl",
      "concept",       "const",       "consteval",
      "constexpr",     "constinit",   "const_cast",
      "continue",      "co_await",    "co_return",
      "co_yield",      "decltype",    "default",
      "delete",        "do",          "double",
      "dynamic_cast",  "else",        "enum",
      "explicit",      "export",      "extern",
      "false",         "float",       "for",
      "friend",        "goto",        "if",
      "inline",        "int",         "long",
      "mutable",       "namespace",   "new",
      "noexcept",      "not",         "not_eq",
      "nullptr",       "operator",    "or",
      "or_eq",         "private",     "protected",
      "public",        "register",    "reinterpret_cast",
      "requires",      "return",      "short",
      "signed",        "sizeof",      "static",
      "static_assert", "static_cast", "struct",
      "switch",        "template",    "this",
      "thread_local",  "throw",       "true",
      "try",           "typedef",     "typeid",
      "typename",      "union",       "unsigned",
      "using",         "virtual",     "void",
      "volatile",      "wchar_t",     "while",
      "xor",           "xor_eq"};
  return !name.empty() && absl::ascii_isalpha(name.front()) &&
         name.find("__") == absl::string_view::npos &&
         !keywords.contains(std::string(name)) &&
         std::all_of(name.begin(), name.end(), [](unsigned char c) {
           return absl::ascii_isalnum(c) || c == '_';
         });
}

}  // namespace

absl::Status ValidateTransitionFunctionName(absl::string_view name) {
  if (!ValidName(name))
    return absl::InvalidArgumentError(
        "transition function name must be a nonreserved C++ identifier");
  return absl::OkStatus();
}

}  // namespace pluto::llm::discretized::generator
