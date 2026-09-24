#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "pluto/discretized/gen/model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"

namespace pluto::llm::discretized {
namespace {

struct CommandResult {
  int exit_code = -1;
  std::string output;
};

class DiscretizedModelCliTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* root = std::getenv("TEST_SRCDIR");
    const char* workspace = std::getenv("TEST_WORKSPACE");
    ASSERT_NE(root, nullptr);
    ASSERT_NE(workspace, nullptr);
    executable_ =
        (std::filesystem::path(root) / workspace /
         "src/llm/experiments/memorize_general_facts/discretized_model/"
         "generated/discretized_model")
            .string();
    ASSERT_TRUE(std::filesystem::exists(executable_));
  }

  // Invoke the actual generated CPU binary without shell interpretation. Read
  // concurrently with the child so even large state listings cannot fill the
  // pipe and deadlock the test. Merge stdout/stderr to inspect error messages.
  CommandResult Run(std::vector<std::string> arguments) const {
    arguments.insert(arguments.begin(), executable_);
    std::vector<char*> argv;
    for (auto& argument : arguments)
      argv.push_back(argument.data());
    argv.push_back(nullptr);
    int descriptors[2];
    if (::pipe(descriptors) != 0) {
      ADD_FAILURE() << "pipe failed";
      return {};
    }
    const pid_t child = ::fork();
    if (child == 0) {
      ::close(descriptors[0]);
      if (::dup2(descriptors[1], STDOUT_FILENO) < 0 ||
          ::dup2(descriptors[1], STDERR_FILENO) < 0)
        ::_exit(126);
      ::close(descriptors[1]);
      ::execv(executable_.c_str(), argv.data());
      ::_exit(127);
    }
    ::close(descriptors[1]);
    if (child < 0) {
      ::close(descriptors[0]);
      ADD_FAILURE() << "fork failed";
      return {};
    }
    CommandResult result;
    char buffer[4096];
    while (true) {
      const ssize_t count = ::read(descriptors[0], buffer, sizeof(buffer));
      if (count > 0) {
        result.output.append(buffer, count);
      } else if (count == 0) {
        break;
      } else if (errno != EINTR) {
        ADD_FAILURE() << "read failed";
        break;
      }
    }
    ::close(descriptors[0]);
    int status;
    pid_t waited;
    do {
      waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    EXPECT_EQ(waited, child);
    if (waited == child && WIFEXITED(status))
      result.exit_code = WEXITSTATUS(status);
    return result;
  }

  std::string executable_;
};

TEST_F(DiscretizedModelCliTest, PrintsExactlyTheGeneratedCallbackOutput) {
  const auto& model = gen::GeneratedModel();
  ASSERT_TRUE(model.print_state);
  std::optional<DiscreteHiddenState> state;
  for (size_t token = 0; token < model.vocabulary.size(); ++token) {
    state = model.position_embedding(DiscreteToken{static_cast<int>(token)}, 0);
    if (state)
      break;
  }
  ASSERT_TRUE(state);
  std::ostringstream expected;
  const auto status = model.print_state(*state, expected);
  ASSERT_TRUE(status.ok()) << status;
  ASSERT_FALSE(expected.str().empty());
  const auto result = Run({"--print_state=" + std::to_string(state->value)});
  EXPECT_EQ(result.exit_code, 0) << result.output;
  EXPECT_EQ(result.output, expected.str());
}

TEST_F(DiscretizedModelCliTest, RejectsUnsupportedHiddenStates) {
  for (const char* option :
       {"--print_state=-1", "--print_state=0", "--print_state=2147483647"}) {
    const auto result = Run({option});
    EXPECT_EQ(result.exit_code, 1) << result.output;
    EXPECT_NE(result.output.find("NOT_FOUND"), std::string::npos)
        << result.output;
  }
}

TEST_F(DiscretizedModelCliTest, RejectsMalformedStateIds) {
  for (const char* option : {"--print_state=", "--print_state=1x",
                             "--print_state=1,2", "--print_state=2147483648"}) {
    const auto result = Run({option});
    EXPECT_EQ(result.exit_code, 2) << result.output;
    EXPECT_NE(result.output.find("integer hidden-state ID"), std::string::npos)
        << result.output;
  }
}

TEST_F(DiscretizedModelCliTest,
       ModesAreExclusiveAndGenerationLimitNeedsPrompt) {
  const std::vector<std::string> modes = {
      "--verify", "--prompt=The capital of France is", "--token_ids=1",
      "--print_state=4475"};
  for (size_t first = 0; first < modes.size(); ++first)
    for (size_t second = first + 1; second < modes.size(); ++second)
      EXPECT_EQ(Run({modes[first], modes[second]}).exit_code, 2);
  EXPECT_EQ(Run({"--print_state=4475", "--generation_tokens=1"}).exit_code, 2);
  EXPECT_EQ(Run({"--verify", "--generation_tokens=1"}).exit_code, 2);
  EXPECT_EQ(Run({"--generation_tokens=1"}).exit_code, 2);
  EXPECT_EQ(Run({"--print_state=4475", "--print_state=4475"}).exit_code, 2);
  EXPECT_EQ(Run({}).exit_code, 2);
}

TEST_F(DiscretizedModelCliTest, HelpMentionsStateInspection) {
  const auto result = Run({"--help"});
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.output.find("--print_state=ID"), std::string::npos);
}

}  // namespace
}  // namespace pluto::llm::discretized
