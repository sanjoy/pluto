#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_io.h"

#include <unistd.h>

#include <filesystem>
#include <string>

#include "gtest/gtest.h"

namespace pluto::llm::discretized::generator {
namespace {

TEST(GeneratorIoTest, StandardSha256Vectors) {
  EXPECT_EQ(Sha256(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(Sha256("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(Sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  EXPECT_EQ(Sha256(std::string(1000000, 'a')),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(GeneratorIoTest, RoundTripAndErrors) {
  std::string directory = testing::TempDir() + "/generator-io.XXXXXX";
  ASSERT_NE(mkdtemp(directory.data()), nullptr);
  auto path = std::filesystem::path(directory) / "artifact.txt";
  EXPECT_FALSE(ReadFile(directory).ok());
  EXPECT_FALSE(Sha256File(directory).ok());
  ASSERT_TRUE(WriteFile(path, "original").ok());
  auto text = ReadFile(path);
  ASSERT_TRUE(text.ok());
  auto hash = Sha256File(path);
  ASSERT_TRUE(hash.ok());
  EXPECT_EQ(*hash, Sha256(*text));
  ASSERT_TRUE(WriteFile(path, "replacement").ok());
  EXPECT_EQ(*ReadFile(path), "replacement");
  EXPECT_FALSE(ReadFile(path / "absent").ok());
  EXPECT_FALSE(WriteFile(path / "absent", "not written").ok());
  EXPECT_EQ(*ReadFile(path), "replacement");
  std::filesystem::remove_all(directory);
}

TEST(GeneratorIoTest, RunsWithoutShellInterpretation) {
  EXPECT_TRUE(FindExecutable("clang-format").ok());
  EXPECT_FALSE(FindExecutable("pluto-nonexistent-executable-99887").ok());
  EXPECT_TRUE(
      RunProcess({"/usr/bin/test", "$(exit 1)", "=", "$(exit 1)"}).ok());
  EXPECT_FALSE(RunProcess({"/usr/bin/test", "1", "=", "2"}).ok());
  EXPECT_FALSE(RunProcess({"/nonexistent/pluto/program"}).ok());
  EXPECT_FALSE(RunProcess({}).ok());
}

}  // namespace
}  // namespace pluto::llm::discretized::generator
