#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::memorize_general_facts {
namespace {

// Read the executable's registered flags, not a second set of default values in
// a test fixture. Help and invalid flags both exit before opening a CUDA
// device, so these checks need neither a GPU nor a tokenizer/checkpoint
// installation.
class MemorizeGeneralFactsDefaultsTest : public testing::Test {
 protected:
  void Run(std::vector<std::string> arguments) {
    const char* runfiles = std::getenv("TEST_SRCDIR");
    const char* workspace = std::getenv("TEST_WORKSPACE");
    ASSERT_NE(runfiles, nullptr);
    const auto binary =
        std::filesystem::path(runfiles) /
        (workspace == nullptr ? "_main" : workspace) /
        "src/llm/experiments/memorize_general_facts/memorize_general_facts";
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

  // Flag descriptions wrap across lines. Restrict each search to its own
  // section so a missing default cannot accidentally match the following flag.
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

  std::string output_;
  int exit_code_ = -1;
};

TEST_F(MemorizeGeneralFactsDefaultsTest, DefaultsMatchSmallestMemorizedModel) {
  ASSERT_NO_FATAL_FAILURE(Run({"--mode=train_model", "--helpfull"}));
  // Abseil's help handler deliberately exits with status 1.
  ASSERT_EQ(exit_code_, 1) << output_;
  const std::pair<std::string_view, std::string_view> expected[] = {
      {"layers", "4"},
      {"model_width", "10"},
      {"attention_heads", "1"},
      {"feed_forward_width", "20"},
      {"context_length", "27"},
      {"compact_vocabulary", "true"},
      {"batch_size", "32"},
      {"steps", "120000"},
      {"learning_rate", "0.0012"},
      {"eval_every", "256"},
      {"seed", "1337"},
      {"warmup_steps", "100"},
      {"search", "false"},
      {"train_mlp", "false"},
      {"mlp_width", "150"},
      {"mlp_depth", "1"},
      {"match_mlp_parameter_budget", "true"},
      {"mode", "\"\""},
  };
  for (const auto& [name, value] : expected) {
    SCOPED_TRACE(name);
    const auto actual = DefaultValue(name);
    ASSERT_TRUE(actual.has_value()) << output_;
    EXPECT_EQ(*actual, value);
  }
}

TEST_F(MemorizeGeneralFactsDefaultsTest, ModeRemainsExplicitlyRequired) {
  ASSERT_NO_FATAL_FAILURE(Run({}));
  EXPECT_EQ(exit_code_, 1) << output_;
  EXPECT_NE(output_.find("--mode must be one of: train_model, infer_model"),
            std::string::npos)
      << output_;
}

TEST_F(MemorizeGeneralFactsDefaultsTest,
       InferenceRejectsExplicitTrainingFlags) {
  // Even explicitly specifying the default value must obey the mode policy.
  ASSERT_NO_FATAL_FAILURE(
      Run({"--mode=infer_model", "--infer_checkpoint=unused_checkpoint",
           "--steps=120000"}));
  EXPECT_EQ(exit_code_, 1) << output_;
  EXPECT_NE(output_.find("--steps is not valid in --mode=infer_model"),
            std::string::npos)
      << output_;
}

TEST_F(MemorizeGeneralFactsDefaultsTest, StackedMlpFlagsRequirePuzzleTraining) {
  const std::vector<std::string> paths[] = {
      {"--mode=train_model"},
      {"--mode=infer_model", "--infer_checkpoint=unused_checkpoint"},
      {"--mode=infer_model", "--verify_checkpoint=unused_checkpoint"},
      {"--mode=puzzle", "--puzzle_checkpoint=unused_checkpoint"},
  };
  // Explicit defaults and a disabled budget flag all count as supplied flags.
  for (const auto& path : paths) {
    SCOPED_TRACE(path.front());
    for (const std::string flag :
         {"--mlp_depth=1", "--match_mlp_parameter_budget=true",
          "--match_mlp_parameter_budget=false"}) {
      SCOPED_TRACE(flag);
      auto arguments = path;
      arguments.push_back(flag);
      ASSERT_NO_FATAL_FAILURE(Run(std::move(arguments)));
      EXPECT_EQ(exit_code_, 1) << output_;
      EXPECT_NE(output_.find(flag.substr(0, flag.find('=')) +
                             " is not valid in --mode="),
                std::string::npos)
          << output_;
    }
  }
}

TEST_F(MemorizeGeneralFactsDefaultsTest,
       PuzzleRejectsNonpositiveMlpDepthBeforeOpeningCuda) {
  for (const std::string depth : {"0", "-1"}) {
    SCOPED_TRACE(depth);
    ASSERT_NO_FATAL_FAILURE(
        Run({"--mode=puzzle", "--train_mlp", "--tokenizer=unused_tokenizer",
             "--puzzle_checkpoint=unused_checkpoint", "--mlp_depth=" + depth}));
    EXPECT_EQ(exit_code_, 1) << output_;
    EXPECT_NE(output_.find("--mlp_depth must be positive"), std::string::npos)
        << output_;
  }
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
