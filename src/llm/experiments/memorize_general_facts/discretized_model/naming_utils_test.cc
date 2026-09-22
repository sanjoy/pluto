#include "src/llm/experiments/memorize_general_facts/discretized_model/naming_utils.h"

#include <string>

#include "gtest/gtest.h"

namespace pluto::llm::discretized::generator {
namespace {

TEST(NamingUtilsTest, AcceptsUnqualifiedAsciiFunctionNames) {
  for (const char* name : {"Attention", "GeneratedMlp0", "entry_lookup", "x_",
                           "Class", "final", "override"})
    EXPECT_TRUE(ValidateTransitionFunctionName(name).ok()) << name;
}

TEST(NamingUtilsTest, RejectsKeywordsReservedNamesAndDeclarationInjection) {
  for (const char* name : {"",         "class",      "and",
                           "xor_eq",   "char8_t",    "concept",
                           "requires", "consteval",  "constinit",
                           "co_await", "co_return",  "co_yield",
                           "_private", "__reserved", "Bad__Name",
                           "1Bad",     "Bad::Name",  "x()",
                           "x;}",      "two words",  "name\n#error injected"}) {
    const auto status = ValidateTransitionFunctionName(name);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << name;
    EXPECT_EQ(status.message(),
              "transition function name must be a nonreserved C++ identifier");
  }
}

TEST(NamingUtilsTest, ChecksEveryByteIncludingEmbeddedNull) {
  // Exercise signed-char platforms and string_view length handling. Only
  // ASCII letters can begin a name; digits and '_' can also continue one.
  for (int byte = 0; byte < 256; ++byte) {
    SCOPED_TRACE(byte);
    const char c = static_cast<char>(byte);
    const bool letter = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    const bool suffix = letter || (c >= '0' && c <= '9') || c == '_';
    EXPECT_EQ(ValidateTransitionFunctionName(std::string(1, c)).ok(), letter);
    EXPECT_EQ(ValidateTransitionFunctionName(std::string("x") + c).ok(),
              suffix);
  }
}

}  // namespace
}  // namespace pluto::llm::discretized::generator
