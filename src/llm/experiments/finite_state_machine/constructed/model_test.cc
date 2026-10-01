#include "src/llm/experiments/finite_state_machine/constructed/model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::fsm::constructed {
namespace {

constexpr int kError = 1028;
constexpr int kEnd = 1029;

std::string StateText(int state) {
  std::string text(3, '0');
  text[0] += state / 100;
  text[1] += state / 10 % 10;
  text[2] += state % 10;
  return text;
}

std::string WithoutSpaces(std::string text) {
  text.erase(std::remove(text.begin(), text.end(), ' '), text.end());
  return text;
}

// This ordinary interpreter is deliberately independent of the neural model:
// it uses integer table lookup, never the model's tokenizer or affine maps.
struct Transition {
  int source;
  char letter;
  int destination;
};

std::vector<int> Interpret(const std::vector<Transition>& transitions,
                           const std::string& input) {
  std::vector<int> table(1000 * 26, -1);
  for (const auto& transition : transitions)
    table[transition.source * 26 + transition.letter - 'A'] =
        transition.destination;
  std::vector<int> output{0};
  for (const char letter : input) {
    const int next = table[output.back() * 26 + letter - 'A'];
    if (next < 0) {
      output.push_back(kError);
      break;
    }
    output.push_back(next);
  }
  return output;
}

std::string MakePrompt(const std::vector<Transition>& transitions,
                       const std::string& input) {
  std::string prompt;
  for (const auto& transition : transitions)
    prompt += StateText(transition.source) + transition.letter +
              StateText(transition.destination) + ';';
  return prompt + input + '>';
}

class ConstructedModelTest : public testing::Test {
 protected:
  void SetUp() override {
    auto model = Model::Create();
    ASSERT_TRUE(model.ok()) << model.status();
    model_ = std::move(*model);
  }

  void ExpectOutput(const std::string& prompt,
                    const std::vector<int>& expected) {
    auto generation = model_->Generate(prompt);
    ASSERT_TRUE(generation.ok()) << prompt << ": " << generation.status();
    EXPECT_EQ(generation->tokens, expected) << prompt;
    // END must be a real final prediction, not a visible output token or a
    // caller stopping after the known expected answer length.
    ASSERT_EQ(generation->steps.size(), generation->tokens.size() + 1);
    for (size_t index = 0; index < generation->tokens.size(); ++index)
      EXPECT_EQ(generation->steps[index].output_token,
                generation->tokens[index]);
    EXPECT_EQ(generation->steps.back().output_token, kEnd);
    for (const auto& step : generation->steps) {
      EXPECT_TRUE(std::isfinite(step.lookup_probability));
      EXPECT_GE(step.lookup_probability, 0);
      EXPECT_LE(step.lookup_probability, 1);
    }
  }

  std::unique_ptr<Model> model_;
};

TEST_F(ConstructedModelTest, FollowsTransitionsAndReportsMissingTransitions) {
  ExpectOutput("000X093;093A044;XA>", {0, 93, 44});
  ExpectOutput("000X093;093A044;XB>", {0, 93, kError});
  ExpectOutput("000X093;093A044;B>", {0, kError});
}

TEST_F(ConstructedModelTest, EmptyInputProducesOnlyTheInitialState) {
  ExpectOutput("000A001;>", {0});
}

TEST_F(ConstructedModelTest, StopsAtFirstErrorWithoutConsumingRemainingInput) {
  ExpectOutput("000X093;093A044;BXAXA>", {0, kError});
  ExpectOutput("000X093;093A044;XBAAAA>", {0, 93, kError});
}

TEST_F(ConstructedModelTest, LoopsPreserveEveryVisitIncludingInitialZero) {
  ExpectOutput("000A000;AAAA>", {0, 0, 0, 0, 0});
  ExpectOutput("000A999;999A000;AAAAA>", {0, 999, 0, 999, 0, 999});
  ExpectOutput("000A000;" + std::string(100, 'A') + ">",
               std::vector<int>(101, 0));
}

TEST_F(ConstructedModelTest, TableOrderAndIrrelevantRowsDoNotChangeExecution) {
  ExpectOutput("999Z888;093A044;777X111;000X093;XA>", {0, 93, 44});
  ExpectOutput("000X093;777X111;093A044;999Z888;XA>", {0, 93, 44});
  // There are multiple X transitions, but only the current state's row may win.
  ExpectOutput("093X044;044X000;000X093;XXX>", {0, 93, 44, 0});
}

TEST_F(ConstructedModelTest, NearMaximumContextToleratesManyDistractorRows) {
  std::vector<Transition> transitions{{0, 'A', 999}, {999, 'A', 999}};
  for (int state = 1; state <= 247; ++state)
    transitions.push_back({state, 'A', state * 97 % 999});
  // 998 differs from 999 in only its low bit, making this a close wrong key.
  transitions.push_back({998, 'A', 0});
  std::mt19937 random(1729);
  std::shuffle(transitions.begin(), transitions.end(), random);
  ASSERT_EQ(transitions.size(), size_t{250});

  // 1,000 table tokens + 10 input tokens + '>' + 11 output tokens leaves only
  // two positions below the limit. None of the 248 distractors may be selected.
  const std::string input(10, 'A');
  ExpectOutput(MakePrompt(transitions, input), Interpret(transitions, input));
  const std::string failing = std::string(9, 'A') + 'B';
  ExpectOutput(MakePrompt(transitions, failing),
               Interpret(transitions, failing));
}

TEST_F(ConstructedModelTest, CanPredictEndAtTheLastSupportedPosition) {
  // The prompt occupies 514 tokens and the visited-state trace adds 510. END
  // must still be predicted from position 1023 without embedding another token.
  const std::string prompt = "000A000;" + std::string(509, 'A') + '>';
  auto generation = model_->Generate(prompt);
  ASSERT_TRUE(generation.ok()) << generation.status();
  EXPECT_EQ(generation->tokens, std::vector<int>(510, 0));
  ASSERT_EQ(generation->steps.size(), size_t{511});
  EXPECT_EQ(generation->steps.back().position, 1023);
  EXPECT_EQ(generation->steps.back().output_token, kEnd);
}

TEST_F(ConstructedModelTest, SpacesAreIgnoredEvenInsideStateLabels) {
  ExpectOutput(" 0 0 0 X 0 9 3 ; 0 93 A 04 4 ; X A > ", {0, 93, 44});
}

TEST_F(ConstructedModelTest, ErrInInputIsThreeOrdinaryLetters) {
  ExpectOutput("000E001;001R999;999R000;ERR>", {0, 1, 999, 0});
}

TEST_F(ConstructedModelTest, EveryInputLetterSelectsItsOwnTransition) {
  std::vector<Transition> transitions;
  for (int letter = 0; letter < 26; ++letter)
    transitions.push_back({0, static_cast<char>('A' + letter), 100 + letter});
  std::reverse(transitions.begin(), transitions.end());
  for (int letter = 0; letter < 26; ++letter) {
    SCOPED_TRACE(letter);
    ExpectOutput(MakePrompt(transitions, std::string(1, 'A' + letter)),
                 {0, 100 + letter});
  }
}

TEST_F(ConstructedModelTest, EveryStateLabelCanBeWrittenAndReadBack) {
  for (int state = 0; state < 1000; ++state) {
    SCOPED_TRACE(state);
    const std::vector<Transition> transitions = {{state, 'B', 0},
                                                 {0, 'A', state}};
    ExpectOutput(MakePrompt(transitions, "AB"), {0, state, 0});
  }
}

TEST_F(ConstructedModelTest, RandomNovelMachinesMatchIndependentInterpreter) {
  std::mt19937 random(728531);
  std::array<int, 999> other_states;
  std::iota(other_states.begin(), other_states.end(), 1);
  std::array<char, 26> letters;
  std::iota(letters.begin(), letters.end(), 'A');

  // None of these 250 machines come from a corpus. State IDs, input symbols,
  // transition destinations, and serialization order are randomized separately.
  // Each machine has a successful walk and a walk ending in a missing edge.
  for (int trial = 0; trial < 250; ++trial) {
    SCOPED_TRACE(trial);
    std::shuffle(other_states.begin(), other_states.end(), random);
    std::shuffle(letters.begin(), letters.end(), random);
    const int state_count = 2 + random() % 11;
    const int letter_count = 2 + random() % 7;
    std::vector<int> states{0};
    states.insert(states.end(), other_states.begin(),
                  other_states.begin() + state_count - 1);
    std::vector<Transition> transitions;
    for (const int source : states) {
      std::vector<char> outgoing(letters.begin(),
                                 letters.begin() + letter_count);
      std::shuffle(outgoing.begin(), outgoing.end(), random);
      const int degree = 1 + random() % letter_count;
      for (int index = 0; index < degree; ++index)
        transitions.push_back(
            {source, outgoing[index], states[random() % states.size()]});
    }

    std::string input;
    int current = 0;
    const int input_length = 1 + random() % 32;
    for (int index = 0; index < input_length; ++index) {
      std::vector<Transition> outgoing;
      for (const auto& transition : transitions)
        if (transition.source == current)
          outgoing.push_back(transition);
      const auto& selected = outgoing[random() % outgoing.size()];
      input += selected.letter;
      current = selected.destination;
    }
    const auto expected = Interpret(transitions, input);
    std::shuffle(transitions.begin(), transitions.end(), random);
    ExpectOutput(MakePrompt(transitions, input), expected);

    // This unused letter guarantees failure only after the successful prefix.
    // Trailing letters must not cause extra states to be generated after ERR.
    const std::string failing = input + letters[letter_count] + input;
    ExpectOutput(MakePrompt(transitions, failing),
                 Interpret(transitions, failing));
    std::shuffle(transitions.begin(), transitions.end(), random);
    ExpectOutput(MakePrompt(transitions, input), expected);
  }
}

TEST_F(ConstructedModelTest, RejectsBothConflictingAndIdenticalDuplicateRows) {
  for (const char* prompt : {"000A001;000A002;A>", "000A001;000A001;A>"}) {
    SCOPED_TRACE(prompt);
    EXPECT_FALSE(model_->Generate(prompt).ok());
  }
}

TEST_F(ConstructedModelTest, RejectsMalformedPromptsAndAlreadySuppliedAnswers) {
  for (const char* prompt :
       {"", ">", "A>", "000A001", "000A001;A", "000a001;A>", "000A01;A>",
        "000A1000;A>", "000A001;;A>", "000A001;A>>", "000A001;1>",
        "000A001;A;>", "000A001;A>000001", "000A001;A>ERR"}) {
    SCOPED_TRACE(prompt);
    EXPECT_FALSE(model_->Generate(prompt).ok());
  }
}

TEST(ConstructedModelLimitsTest, RejectsUnsupportedContextSizes) {
  EXPECT_FALSE(Model::Create(0).ok());
  EXPECT_FALSE(Model::Create(-1).ok());
  EXPECT_FALSE(Model::Create(1025).ok());
}

TEST(ConstructedModelLimitsTest, RejectsInputAndAutoregressiveContextOverflow) {
  auto model = Model::Create(10);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ((*model)->max_context(), 10);
  // The first prompt is already too long; the second fits but its trace does
  // not.
  EXPECT_FALSE((*model)->Generate("000A000;AAAAAA>").ok());
  EXPECT_FALSE((*model)->Generate("000A000;AAA>").ok());
}

TEST_F(ConstructedModelTest, WeightExportIsNonemptyDeterministicAndImmutable) {
  EXPECT_GT(model_->nonzero_weight_count(), size_t{0});
  std::ostringstream first;
  ASSERT_TRUE(model_->WriteWeights(first).ok());
  EXPECT_FALSE(first.str().empty());

  auto other = Model::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  EXPECT_EQ(model_->nonzero_weight_count(), (*other)->nonzero_weight_count());
  std::ostringstream second;
  ASSERT_TRUE((*other)->WriteWeights(second).ok());
  EXPECT_EQ(first.str(), second.str());

  ExpectOutput("000X093;093A044;XA>", {0, 93, 44});
  std::ostringstream after_inference;
  ASSERT_TRUE(model_->WriteWeights(after_inference).ok());
  EXPECT_EQ(first.str(), after_inference.str());
}

TEST_F(ConstructedModelTest,
       RenderIncludesPaddedStatesAndErrButNotInternalEnd) {
  auto generation = model_->Generate("000X093;093A044;XB>");
  ASSERT_TRUE(generation.ok()) << generation.status();
  EXPECT_EQ(WithoutSpaces(Render(*generation)), "000093ERR");
}

class ConstructedModelCorpusTest
    : public ConstructedModelTest,
      public testing::WithParamInterface<std::pair<const char*, int>> {};

TEST_P(ConstructedModelCorpusTest,
       CompletesEveryFullTraceWithoutReadingLabels) {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  ASSERT_NE(runfiles, nullptr);
  const auto path = std::filesystem::path(runfiles) /
                    (workspace == nullptr ? "_main" : workspace) / "testdata" /
                    GetParam().first;
  std::ifstream file(path);
  ASSERT_TRUE(file.is_open()) << path;
  std::string line;
  int line_count = 0;
  while (std::getline(file, line)) {
    SCOPED_TRACE(testing::Message() << path << ":" << line_count + 1);
    const size_t separator = line.find('>');
    ASSERT_NE(separator, std::string::npos);
    // Labels are used ONLY for the assertion. Generate never receives them or
    // their length, so END and error stopping must be predicted by the network.
    auto generation = model_->Generate(line.substr(0, separator + 1));
    ASSERT_TRUE(generation.ok()) << generation.status();
    EXPECT_EQ(WithoutSpaces(Render(*generation)),
              WithoutSpaces(line.substr(separator + 1)));
    ASSERT_FALSE(generation->steps.empty());
    EXPECT_EQ(generation->steps.back().output_token, kEnd);
    ++line_count;
  }
  EXPECT_TRUE(file.eof());
  EXPECT_EQ(line_count, GetParam().second);
}

INSTANTIATE_TEST_SUITE_P(
    FullData, ConstructedModelCorpusTest,
    testing::Values(
        std::make_pair("finite_state_machine_full_training_data.txt", 4096),
        std::make_pair("finite_state_machine_full_test_data.txt", 128)));

}  // namespace
}  // namespace pluto::llm::fsm::constructed
