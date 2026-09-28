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
      {"train_stacked_mlp", "false"},
      {"train_mlp_transformer", "false"},
      {"mlp_width", "150"},
      {"mode", "\"\""},
  };
  for (const auto& [name, value] : expected) {
    SCOPED_TRACE(name);
    const auto actual = DefaultValue(name);
    ASSERT_TRUE(actual.has_value()) << output_;
    EXPECT_EQ(*actual, value);
  }
  EXPECT_NE(output_.find("10/150/10"), std::string::npos) << output_;
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

TEST_F(MemorizeGeneralFactsDefaultsTest,
       OtherModesRejectExplicitPuzzleTrainingFlags) {
  for (const char* flag : {"train_stacked_mlp", "train_mlp_transformer"}) {
    for (const char* mode : {"train_model", "infer_model"}) {
      for (const char* value : {"false", "true"}) {
        SCOPED_TRACE(flag);
        SCOPED_TRACE(mode);
        SCOPED_TRACE(value);
        std::vector<std::string> arguments = {
            std::string("--mode=") + mode,
            std::string("--") + flag + "=" + value};
        if (std::string_view(mode) == "infer_model")
          arguments.push_back("--infer_checkpoint=unused_checkpoint");
        ASSERT_NO_FATAL_FAILURE(Run(std::move(arguments)));
        EXPECT_EQ(exit_code_, 1) << output_;
        EXPECT_NE(
            output_.find(std::string("--") + flag + " is not valid in --mode="),
            std::string::npos)
            << output_;
      }
    }
  }
}

TEST_F(MemorizeGeneralFactsDefaultsTest,
       StackedTrainingRejectsInvalidOptionsBeforeOpeningCuda) {
  const std::pair<std::string_view, std::string_view> cases[] = {
      {"--train_mlp", "mutually exclusive"},
      {"--train_mlp_transformer", "mutually exclusive"},
      {"--model_width=20", "--model_width must be 10"},
      {"--mlp_width=150", "--mlp_width is not valid"},
      {"--steps=-1", "--steps must be nonnegative"},
      {"--eval_every=0", "--eval_every must be positive"},
      {"--learning_rate=0", "--learning_rate must be finite and positive"},
      {"--seed=-1", "--seed must be nonnegative"},
  };
  for (const auto& [argument, diagnostic] : cases) {
    SCOPED_TRACE(argument);
    ASSERT_NO_FATAL_FAILURE(
        Run({"--mode=puzzle", "--train_stacked_mlp",
             "--puzzle_checkpoint=unused_checkpoint",
             "--tokenizer=unused_tokenizer", std::string(argument)}));
    EXPECT_EQ(exit_code_, 1) << output_;
    EXPECT_NE(output_.find(diagnostic), std::string::npos) << output_;
  }
}

TEST_F(MemorizeGeneralFactsDefaultsTest,
       MlpTransformerTrainingRejectsInvalidOptionsBeforeOpeningCuda) {
  const std::pair<std::string_view, std::string_view> cases[] = {
      {"--train_mlp", "mutually exclusive"},
      {"--train_stacked_mlp", "mutually exclusive"},
      {"--layers=-1", "--layers must be 4 with --train_mlp_transformer"},
      {"--layers=3", "--layers must be 4 with --train_mlp_transformer"},
      {"--layers=5", "--layers must be 4 with --train_mlp_transformer"},
      {"--model_width=0", "--model_width must be positive"},
      {"--mlp_width=150", "--mlp_width is not valid"},
      {"--steps=-1", "--steps must be nonnegative"},
      {"--eval_every=0", "--eval_every must be positive"},
      {"--learning_rate=0", "--learning_rate must be finite and positive"},
      {"--seed=-1", "--seed must be nonnegative"},
  };
  for (const auto& [argument, diagnostic] : cases) {
    SCOPED_TRACE(argument);
    ASSERT_NO_FATAL_FAILURE(
        Run({"--mode=puzzle", "--train_mlp_transformer",
             "--puzzle_checkpoint=unused_checkpoint",
             "--tokenizer=unused_tokenizer", std::string(argument)}));
    EXPECT_EQ(exit_code_, 1) << output_;
    EXPECT_NE(output_.find(diagnostic), std::string::npos) << output_;
  }
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
