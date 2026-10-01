#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::fsm {
namespace {

// Help and invalid flags exit before opening CUDA or loading a dataset, so
// these checks exercise the real executable without requiring a GPU.
class FiniteStateMachineCliTest : public testing::Test {
 protected:
  void Run(std::vector<std::string> arguments) {
    const char* runfiles = std::getenv("TEST_SRCDIR");
    const char* workspace = std::getenv("TEST_WORKSPACE");
    ASSERT_NE(runfiles, nullptr);
    const auto binary =
        std::filesystem::path(runfiles) /
        (workspace == nullptr ? "_main" : workspace) /
        "src/llm/experiments/finite_state_machine/finite_state_machine";
    arguments.insert(arguments.begin(), binary.string());
    std::vector<char*> argv;
    for (std::string& argument : arguments)
      argv.push_back(argument.data());
    argv.push_back(nullptr);

    int output_pipe[2];
    ASSERT_EQ(pipe(output_pipe), 0);
    const pid_t child = fork();
    if (child == -1) {
      close(output_pipe[0]);
      close(output_pipe[1]);
      FAIL() << "fork failed: " << errno;
    }
    if (child == 0) {
      close(output_pipe[0]);
      if (dup2(output_pipe[1], STDOUT_FILENO) == -1 ||
          dup2(output_pipe[1], STDERR_FILENO) == -1)
        _exit(127);
      close(output_pipe[1]);
      execv(argv[0], argv.data());
      _exit(127);
    }
    close(output_pipe[1]);
    output_.clear();
    std::array<char, 4096> buffer;
    int read_error = 0;
    for (;;) {
      const ssize_t count = read(output_pipe[0], buffer.data(), buffer.size());
      if (count > 0) {
        output_.append(buffer.data(), count);
      } else if (count == 0) {
        break;
      } else if (errno != EINTR) {
        read_error = errno;
        break;
      }
    }
    close(output_pipe[0]);
    int status = 0;
    pid_t waited;
    do {
      waited = waitpid(child, &status, 0);
    } while (waited == -1 && errno == EINTR);
    ASSERT_EQ(waited, child);
    ASSERT_EQ(read_error, 0);
    ASSERT_TRUE(WIFEXITED(status)) << output_;
    exit_code_ = WEXITSTATUS(status);
  }

  // Restrict the default search to this flag's help section, including wrapped
  // descriptions, so another flag cannot supply a missing default.
  std::optional<std::string> DefaultValue(std::string_view name) const {
    const auto begin = output_.find("    --" + std::string(name) + " (");
    if (begin == std::string::npos)
      return std::nullopt;
    const auto end = output_.find("\n    --", begin);
    const auto section = std::string_view(output_).substr(
        begin, end == std::string::npos ? end : end - begin);
    constexpr std::string_view marker = "default: ";
    const auto marker_begin = section.find(marker);
    if (marker_begin == std::string_view::npos)
      return std::nullopt;
    const auto value_begin = marker_begin + marker.size();
    const auto value_end = section.find(';', value_begin);
    if (value_end == std::string_view::npos)
      return std::nullopt;
    return std::string(section.substr(value_begin, value_end - value_begin));
  }

  void ExpectInvalidFlags(std::vector<std::string> arguments,
                          std::string_view diagnostic) {
    const char* temporary_directory = std::getenv("TEST_TMPDIR");
    ASSERT_NE(temporary_directory, nullptr);
    const auto checkpoint_directory =
        std::filesystem::path(temporary_directory) /
        (std::string("fsm_cli_") +
         testing::UnitTest::GetInstance()->current_test_info()->name() + "_" +
         std::to_string(invocations_++));
    std::error_code error;
    ASSERT_FALSE(std::filesystem::exists(checkpoint_directory, error));
    ASSERT_FALSE(error) << error.message();
    arguments.push_back("--checkpoint_dir=" + checkpoint_directory.string());

    ASSERT_NO_FATAL_FAILURE(Run(std::move(arguments)));
    EXPECT_EQ(exit_code_, 2) << output_;
    EXPECT_NE(output_.find(diagnostic), std::string::npos) << output_;
    EXPECT_FALSE(std::filesystem::exists(checkpoint_directory, error))
        << "Invalid flags must not create a checkpoint directory";
    EXPECT_FALSE(error) << error.message();
  }

  std::string output_;
  int exit_code_ = -1;
  int invocations_ = 0;
};

TEST_F(FiniteStateMachineCliTest, HelpExposesDepthWithOriginalDefault) {
  ASSERT_NO_FATAL_FAILURE(Run({"--helpfull"}));
  // Abseil's help handler deliberately exits with status 1.
  ASSERT_EQ(exit_code_, 1) << output_;
  const auto layers = DefaultValue("layers");
  ASSERT_TRUE(layers.has_value()) << output_;
  EXPECT_EQ(*layers, "8");
  EXPECT_EQ(output_.find("--tokenizer_dir"), std::string::npos) << output_;
}

TEST_F(FiniteStateMachineCliTest,
       HelpExposesAttentionHeadsWithOriginalDefault) {
  ASSERT_NO_FATAL_FAILURE(Run({"--helpfull"}));
  ASSERT_EQ(exit_code_, 1) << output_;
  const auto attention_heads = DefaultValue("attention_heads");
  ASSERT_TRUE(attention_heads.has_value()) << output_;
  EXPECT_EQ(*attention_heads, "8");
}

TEST_F(FiniteStateMachineCliTest,
       RejectsInvalidAttentionHeadsBeforeCreatingRun) {
  for (const char* argument :
       {"--attention_heads=0", "--attention_heads=-1", "--attention_heads=3",
        "--attention_heads=1024"}) {
    SCOPED_TRACE(argument);
    ASSERT_NO_FATAL_FAILURE(ExpectInvalidFlags(
        {argument},
        "--attention_heads must be positive and divide model width 512"));
  }
}

TEST_F(FiniteStateMachineCliTest, TwoHeadsPassesAttentionHeadValidation) {
  // A later invalid option lets the requested two-head configuration pass
  // validation without opening CUDA, creating a run, or starting training.
  ASSERT_NO_FATAL_FAILURE(
      ExpectInvalidFlags({"--attention_heads=2", "--batch_size=0"},
                         "--batch_size must be positive and fit the context"));
  EXPECT_EQ(output_.find("--attention_heads must be positive and divide model "
                         "width 512"),
            std::string::npos)
      << output_;
}

TEST_F(FiniteStateMachineCliTest, RejectsNonpositiveDepthBeforeCreatingRun) {
  for (const char* argument : {"--layers=0", "--layers=-1"}) {
    SCOPED_TRACE(argument);
    ASSERT_NO_FATAL_FAILURE(
        ExpectInvalidFlags({argument}, "--layers must be positive"));
  }
}

TEST_F(FiniteStateMachineCliTest, SixteenLayersPassesDepthValidation) {
  // A later invalid option guarantees the larger depth can be validated
  // without starting training, even on a machine with a CUDA device.
  ASSERT_NO_FATAL_FAILURE(
      ExpectInvalidFlags({"--layers=16", "--batch_size=0"},
                         "--batch_size must be positive and fit the context"));
  EXPECT_EQ(output_.find("--layers must be positive"), std::string::npos)
      << output_;
}

}  // namespace
}  // namespace pluto::llm::fsm
