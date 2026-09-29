#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::qwen {
namespace {

// Exercise the executable's actual flags without constructing a CUDA executor
// or requiring model files. Invalid arguments must fail before either happens.
class QwenLlmCliTest : public testing::Test {
 protected:
  void Run(std::vector<std::string> arguments) {
    const char* runfiles = std::getenv("TEST_SRCDIR");
    const char* workspace = std::getenv("TEST_WORKSPACE");
    ASSERT_NE(runfiles, nullptr);
    const auto binary = std::filesystem::path(runfiles) /
                        (workspace == nullptr ? "_main" : workspace) /
                        "src/llm/qwen/qwen_llm";
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

  void ExpectInvalid(std::string_view diagnostic) const {
    EXPECT_EQ(exit_code_, 1) << output_;
    EXPECT_NE(output_.find("INVALID_ARGUMENT"), std::string::npos) << output_;
    EXPECT_NE(output_.find(diagnostic), std::string::npos) << output_;
  }

  std::string output_;
  int exit_code_ = -1;
};

TEST_F(QwenLlmCliTest, RequiresExplicitSupportedMode) {
  for (const std::vector<std::string>& arguments :
       {std::vector<std::string>{},
        std::vector<std::string>{"--checkpoint=unused_checkpoint"},
        std::vector<std::string>{"--mode="},
        std::vector<std::string>{"--mode=train"}}) {
    ASSERT_NO_FATAL_FAILURE(Run(arguments));
    ExpectInvalid("--mode");
    EXPECT_NE(output_.find("infer_model"), std::string::npos) << output_;
    EXPECT_NE(output_.find("train_model"), std::string::npos) << output_;
  }
}

TEST_F(QwenLlmCliTest, BothModesRequireCheckpoint) {
  for (const char* mode : {"infer_model", "train_model"}) {
    SCOPED_TRACE(mode);
    ASSERT_NO_FATAL_FAILURE(Run({std::string("--mode=") + mode}));
    ExpectInvalid("--checkpoint");
  }
}

TEST_F(QwenLlmCliTest, InferenceRejectsExplicitTrainingFlags) {
  // Supplying the default value still counts as supplying a forbidden flag.
  for (const char* argument :
       {"--text=", "--sequence_length=8", "--batch_size=1", "--steps=2",
        "--switch_every=50", "--start_block=-1", "--learning_rate=1e-5",
        "--max_active_gib=0", "--resume_weights=", "--save_weights="}) {
    SCOPED_TRACE(argument);
    ASSERT_NO_FATAL_FAILURE(Run(
        {"--mode=infer_model", "--checkpoint=unused_checkpoint", argument}));
    ExpectInvalid(std::string_view(argument).substr(
        0, std::string_view(argument).find('=')));
    EXPECT_NE(output_.find("infer_model"), std::string::npos) << output_;
  }
}

TEST_F(QwenLlmCliTest, TrainingRejectsExplicitInferenceFlags) {
  for (const char* argument :
       {"--prompt=", "--max_new_tokens=32", "--context_length=512",
        "--raw_prompt=false", "--thinking=false", "--nothinking"}) {
    SCOPED_TRACE(argument);
    ASSERT_NO_FATAL_FAILURE(Run(
        {"--mode=train_model", "--checkpoint=unused_checkpoint", argument}));
    // Abseil normalizes --nothinking to the registered --thinking flag.
    const std::string_view flag = argument;
    ExpectInvalid(flag == "--nothinking" ? "--thinking"
                                         : flag.substr(0, flag.find('=')));
    EXPECT_NE(output_.find("train_model"), std::string::npos) << output_;
  }
}

TEST_F(QwenLlmCliTest, RejectsPositionalArguments) {
  for (const char* mode : {"infer_model", "train_model"}) {
    SCOPED_TRACE(mode);
    ASSERT_NO_FATAL_FAILURE(
        Run({std::string("--mode=") + mode, "--checkpoint=unused_checkpoint",
             "unexpected positional input"}));
    ExpectInvalid("positional");
  }
}

TEST_F(QwenLlmCliTest, InferenceRejectsInvalidSettingsBeforeLoadingModel) {
  for (const char* argument : {"--max_new_tokens=0", "--max_new_tokens=-1",
                               "--context_length=0", "--context_length=-1"}) {
    SCOPED_TRACE(argument);
    ASSERT_NO_FATAL_FAILURE(Run(
        {"--mode=infer_model", "--checkpoint=unused_checkpoint", argument}));
    ExpectInvalid(std::string_view(argument).substr(
        0, std::string_view(argument).find('=')));
  }
  ASSERT_NO_FATAL_FAILURE(
      Run({"--mode=infer_model", "--checkpoint=unused_checkpoint",
           "--raw_prompt", "--thinking"}));
  ExpectInvalid("--thinking");
}

TEST_F(QwenLlmCliTest, TrainingRejectsInvalidSettingsBeforeLoadingModel) {
  for (const char* argument :
       {"--sequence_length=0", "--sequence_length=129", "--batch_size=0",
        "--batch_size=2", "--steps=0", "--steps=-1", "--switch_every=0",
        "--start_block=-2", "--learning_rate=0", "--learning_rate=-1",
        "--learning_rate=2", "--learning_rate=nan", "--learning_rate=inf",
        "--learning_rate=1e-100", "--max_active_gib=-1",
        "--max_active_gib=1025", "--max_active_gib=nan",
        "--max_active_gib=inf"}) {
    SCOPED_TRACE(argument);
    ASSERT_NO_FATAL_FAILURE(Run(
        {"--mode=train_model", "--checkpoint=unused_checkpoint", argument}));
    ExpectInvalid(std::string_view(argument).substr(
        0, std::string_view(argument).find('=')));
  }
}

TEST_F(QwenLlmCliTest, FlagsFromFileObeyModePolicyEvenAtDefaultValue) {
  const char* directory = std::getenv("TEST_TMPDIR");
  ASSERT_NE(directory, nullptr);
  const auto path = std::filesystem::path(directory) / "qwen_cli_flags.txt";
  {
    std::ofstream flags(path);
    ASSERT_TRUE(flags.is_open());
    flags << "--steps=2\n";
    flags.close();
    ASSERT_TRUE(flags.good());
  }
  ASSERT_NO_FATAL_FAILURE(
      Run({"--mode=infer_model", "--checkpoint=unused_checkpoint",
           "--flagfile=" + path.string()}));
  std::error_code error;
  std::filesystem::remove(path, error);
  EXPECT_FALSE(error) << error.message();
  ExpectInvalid("--steps");
  EXPECT_NE(output_.find("infer_model"), std::string::npos) << output_;
}

TEST_F(QwenLlmCliTest, HelpDescribesBothModesWithoutLoadingModel) {
  ASSERT_NO_FATAL_FAILURE(Run({"--helpfull"}));
  // Abseil deliberately exits with status 1 after printing help.
  EXPECT_EQ(exit_code_, 1) << output_;
  for (const char* text :
       {"infer_model", "train_model", "--mode", "--prompt", "--max_new_tokens",
        "--text", "--switch_every", "--resume_weights", "--save_weights"})
    EXPECT_NE(output_.find(text), std::string::npos) << text << '\n' << output_;
  EXPECT_EQ(output_.find("INVALID_ARGUMENT"), std::string::npos) << output_;
}

}  // namespace
}  // namespace pluto::llm::qwen
