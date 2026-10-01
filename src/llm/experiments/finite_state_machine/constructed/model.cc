#include "src/llm/experiments/finite_state_machine/constructed/model.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <iomanip>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "src/llm/experiments/finite_state_machine/constructed/numeric.h"
#include "src/llm/experiments/finite_state_machine/token_ids.h"
#include "src/util/status_macros.h"

namespace pluto::llm::fsm::constructed {
namespace {

// The residual registers are binary codes, not numeric state magnitudes. Dot
// products of bipolar codes therefore count bit agreements independently of
// the arbitrary names given to states or letters.
constexpr int kStateBits = 10;
constexpr int kLetterBits = 5;
constexpr int kState = 15;
constexpr int kLetter = 16;
constexpr int kSemicolon = 17;
constexpr int kSeparator = 18;
constexpr int kError = 19;
constexpr int kTokenWidth = 20;
constexpr int kPositionBits = 10;
constexpr int kPosition = 30;
constexpr int kPositionSquared = 31;
constexpr int kConstant = 32;
constexpr int kWidth = 33;
constexpr int kEndToken = kVocabularySize;
constexpr double kScale = 16;

// Validate syntax and produce the same tokens as FsmTokenizer, without CUDA.
// Parsing discards the transition relation: only token IDs enter the network.
// The set below rejects ambiguous machines; it never supplies inference values.
absl::StatusOr<std::vector<int>> TokenizePrompt(absl::string_view text) {
  std::string prompt;
  for (char c : text)
    if (c != ' ')
      prompt += c;
  if (prompt.empty() || prompt.back() != '>' ||
      prompt.find('>') != prompt.size() - 1)
    return absl::InvalidArgumentError("prompt must end in exactly one '>'");
  const size_t separator = prompt.rfind(';');
  if (separator == std::string::npos || separator == 0)
    return absl::InvalidArgumentError("expected transitions followed by ';'");
  auto state = [](absl::string_view digits) -> int {
    int value = 0;
    for (char c : digits) {
      if (c < '0' || c > '9')
        return -1;
      value = 10 * value + c - '0';
    }
    return value;
  };
  std::vector<int> tokens;
  absl::flat_hash_set<int> pairs;
  size_t begin = 0;
  while (begin <= separator) {
    size_t end = prompt.find(';', begin);
    if (end == std::string::npos || end - begin != 7)
      return absl::InvalidArgumentError(
          "transition must be three digits, A-Z, three digits");
    const absl::string_view row(prompt.data() + begin, 7);
    const int source = state(row.substr(0, 3));
    const int destination = state(row.substr(4, 3));
    if (source < 0 || destination < 0 || row[3] < 'A' || row[3] > 'Z')
      return absl::InvalidArgumentError("invalid transition token");
    if (!pairs.insert(source * 26 + row[3] - 'A').second)
      return absl::InvalidArgumentError("duplicate (state, letter) transition");
    tokens.insert(tokens.end(), {source, kLetterOffset + row[3] - 'A',
                                 destination, kSemicolonToken});
    begin = end + 1;
  }
  for (size_t i = separator + 1; i + 1 < prompt.size(); ++i) {
    if (prompt[i] < 'A' || prompt[i] > 'Z')
      return absl::InvalidArgumentError("input must contain only A-Z");
    tokens.push_back(kLetterOffset + prompt[i] - 'A');
  }
  tokens.push_back(kOutputSeparatorToken);
  return tokens;
}

Vector Concat(absl::Span<const double> a, absl::Span<const double> b) {
  Vector result(a.begin(), a.end());
  result.insert(result.end(), b.begin(), b.end());
  return result;
}

// These names identify actual matrices in the exported weight file. Every
// computation below uses these weights rather than a host-language FSM lookup.
enum Weight {
  kTokenEmbedding,
  kPositionEmbedding,
  kPositionKey,
  kTokenValue,
  kPreviousLetterQuery,
  kPreviousStateQuery,
  kSemicolonKey,
  kMarkerQuery,
  kPositionValue,
  kBoundaryKey,
  kBoundaryValue,
  kValidTransition,
  kTransitionBits,
  kTransitionKey,
  kTransitionValue,
  kCounter,
  kInputQuery,
  kTransitionQuery,
  kControlHidden,
  kControlFlags,
  kOutputGate,
  kOutputCode,
  kOutputProjection,
  kCleanTokenHidden,
  kCleanTokenOutput,
  kCleanPositionHidden,
  kCleanPositionOutput,
  kCleanBoundaryHidden,
  kCleanBoundaryOutput,
  kWeightCount
};

// A cache entry is a projected numeric key/value pair, never a transition map.
// Appending before each query implements a causal mask by construction.
struct Cache {
  std::vector<MemoryEntry> position;
  std::vector<MemoryEntry> semicolon;
  std::vector<MemoryEntry> boundary;
  std::vector<MemoryEntry> transition;
  Vector current;
};

}  // namespace

struct Model::Impl {
  int max_context;
  std::vector<Affine> weights;
  std::vector<Vector> token_embeddings;
  std::vector<Vector> position_embeddings;

  explicit Impl(int context) : max_context(context) {
    // Add the layers in Weight order, making export deterministic.
    auto add = [&](Weight id, std::string name, int input,
                   int output) -> Affine& {
      assert(weights.size() == static_cast<size_t>(id));
      return weights.emplace_back(std::move(name), input, output);
    };
    weights.reserve(kWeightCount);
    auto& embedding =
        add(kTokenEmbedding, "token_embedding", kVocabularySize, kWidth);
    for (int token = 0; token < kVocabularySize; ++token) {
      if (token < kStateCount) {
        for (int bit = 0; bit < kStateBits; ++bit)
          embedding.Add(bit, token, (token >> bit) & 1);
        embedding.Add(kState, token, 1);
      } else if (token < kSemicolonToken) {
        for (int bit = 0; bit < kLetterBits; ++bit)
          embedding.Add(kStateBits + bit, token,
                        ((token - kLetterOffset) >> bit) & 1);
        embedding.Add(kLetter, token, 1);
      } else {
        embedding.Add(kSemicolon + token - kSemicolonToken, token, 1);
      }
    }
    embedding.Bias(kConstant, 1);
    auto& positions =
        add(kPositionEmbedding, "position_embedding", context, kWidth);
    for (int p = 0; p < context; ++p) {
      for (int bit = 0; bit < kPositionBits; ++bit)
        positions.Add(kTokenWidth + bit, p, (p >> bit) & 1);
      positions.Add(kPosition, p, p);
      positions.Add(kPositionSquared, p, p * p);
    }
    auto& position_key = add(kPositionKey, "position_key", kWidth, 2);
    position_key.Add(0, kPosition, 1);
    position_key.Add(1, kPositionSquared, 1);
    auto& token_value = add(kTokenValue, "token_value", kWidth, kTokenWidth);
    for (int i = 0; i < kTokenWidth; ++i)
      token_value.Add(i, i, 1);
    for (int lag = 1; lag <= 2; ++lag) {
      auto& q = add(lag == 1 ? kPreviousLetterQuery : kPreviousStateQuery,
                    absl::StrCat("previous_", lag, "_query"), kWidth, 2);
      q.Add(0, kPosition, 2 * kScale);
      q.Bias(0, -2 * lag * kScale);
      q.Bias(1, -kScale);
    }
    auto& semicolon_key = add(kSemicolonKey, "semicolon_key", kWidth, 2);
    semicolon_key.Add(0, kPosition, 1);
    semicolon_key.Add(1, kSemicolon, 1);
    auto& marker_query = add(kMarkerQuery, "latest_marker_query", kWidth, 2);
    marker_query.Bias(0, kScale);
    marker_query.Bias(1, 2 * context * kScale);
    auto& position_value =
        add(kPositionValue, "position_bits_value", kWidth, kPositionBits);
    for (int bit = 0; bit < kPositionBits; ++bit)
      position_value.Add(bit, kTokenWidth + bit, 1);
    auto& boundary_key = add(kBoundaryKey, "boundary_key", kWidth, 2);
    boundary_key.Add(0, kPosition, 1);
    boundary_key.Add(1, kSeparator, 1);
    auto& boundary_value = add(kBoundaryValue, "boundary_value",
                               kWidth + kPositionBits, 2 * kPositionBits);
    for (int bit = 0; bit < kPositionBits; ++bit) {
      boundary_value.Add(bit, kTokenWidth + bit, 1);
      boundary_value.Add(kPositionBits + bit, kWidth + bit, 1);
    }
    // A destination row has exactly the pattern [state, letter, state]. This
    // is a learned-style AND, ReLU(a+b+c-2), with explicitly set coefficients.
    auto& valid =
        add(kValidTransition, "transition_valid", kWidth + 2 * kTokenWidth, 1);
    valid.Add(0, kState, 1);
    valid.Add(0, kWidth + kLetter, 1);
    valid.Add(0, kWidth + kTokenWidth + kState, 1);
    valid.Bias(0, -2);
    const int valid_index = kWidth + 2 * kTokenWidth;
    auto& bits =
        add(kTransitionBits, "transition_bit_gates", valid_index + 1, 16);
    for (int bit = 0; bit < 15; ++bit) {
      bits.Add(bit, bit < 10 ? kWidth + kTokenWidth + bit : kWidth + bit, 1);
      bits.Add(bit, valid_index, 1);
      bits.Bias(bit, -1);
    }
    bits.Add(15, valid_index, 1);
    auto& key = add(kTransitionKey, "transition_key", 16 + kWidth, 16);
    for (int bit = 0; bit < 15; ++bit) {
      key.Add(bit, bit, 2);
      key.Add(bit, 15, -1);
    }
    // Exact matches score 15, nearest wrong pairs <=13, '>' fallback 14.
    // Invalid rows score -32. Multiplication by 16 is in the query matrix.
    key.Add(15, 15, 32);
    key.Add(15, 16 + kSeparator, 46);
    key.Bias(15, -32);
    auto& value =
        add(kTransitionValue, "transition_value", kWidth, kTokenWidth);
    for (int i = 0; i < kTokenWidth; ++i)
      if (i != kSeparator)
        value.Add(i, i, 1);
    value.Add(kError, kSeparator, 1);
    // With current position p, separator g, and last semicolon b, the next
    // input position is b+p-g. All three are neural registers, not host
    // indices.
    auto& counter = add(kCounter, "input_counter", kWidth + 20, 2);
    counter.Add(0, kPosition, 1);
    for (int bit = 0; bit < 10; ++bit) {
      counter.Add(0, kWidth + bit, -(1 << bit));
      counter.Add(0, kWidth + 10 + bit, 1 << bit);
      counter.Add(1, kWidth + bit, 1 << bit);
    }
    auto& input_q = add(kInputQuery, "input_query", 2, 2);
    input_q.Add(0, 0, 2 * kScale);
    input_q.Bias(1, -kScale);
    auto& lookup_q =
        add(kTransitionQuery, "transition_query", kWidth + kTokenWidth, 16);
    for (int bit = 0; bit < 15; ++bit) {
      lookup_q.Add(bit, bit < 10 ? bit : kWidth + bit, 2 * kScale);
      lookup_q.Bias(bit, -kScale);
    }
    lookup_q.Bias(15, kScale);
    auto& control_hidden = add(kControlHidden, "control_hidden", kWidth + 2, 3);
    for (int i = 0; i < 2; ++i) {
      control_hidden.Add(i, kWidth, 1);
      control_hidden.Add(i, kWidth + 1, -1);
      control_hidden.Add(i, kError, 2 * context);
    }
    control_hidden.Bias(0, 1);
    control_hidden.Add(2, kSeparator, 1);
    auto& control = add(kControlFlags, "control_flags", 3, 2);
    control.Add(0, 2, 1);  // start
    control.Add(1, 0, 1);  // done = clipped integer step(t-g+2*context*ERR)
    control.Add(1, 1, -1);
    auto& gate = add(kOutputGate, "output_gates", kTokenWidth + 2, 14);
    for (int i = 0; i < 12; ++i) {
      gate.Add(i, i < 10 ? i : (i == 10 ? kState : kError), 1);
      gate.Add(i, kTokenWidth, -1);
      gate.Add(i, kTokenWidth + 1, -1);
    }
    gate.Add(12, kTokenWidth + 1, 1);
    gate.Add(13, kTokenWidth, 1);
    auto& code = add(kOutputCode, "output_code", 14, 13);
    for (int i = 0; i < 13; ++i)
      code.Add(i, i, 1);
    code.Add(10, 13,
             1);  // initial output is state 000, whose bits are all zero.
    auto& output =
        add(kOutputProjection, "language_model_head", 13, kVocabularySize + 1);
    for (int token = 0; token < kStateCount; ++token) {
      int constant = -20;
      for (int bit = 0; bit < 10; ++bit) {
        const int sign = 2 * ((token >> bit) & 1) - 1;
        output.Add(token, bit, 2 * sign);
        constant -= sign;
      }
      output.Add(token, 10, 20);
      output.Bias(token, constant);
    }
    for (int token = kLetterOffset; token < kErrorToken; ++token)
      output.Bias(token, -100);
    output.Add(kErrorToken, 11, 20);
    output.Add(kEndToken, 12, 20);
    // If one attention row carries >3/4 of the mass and values are bits, this
    // fixed two-ReLU map restores exact bits: relu(2x-.5)-relu(2x-1.5).
    for (auto [hidden, result, name, width] :
         {std::tuple{kCleanTokenHidden, kCleanTokenOutput, "token", 20},
          std::tuple{kCleanPositionHidden, kCleanPositionOutput, "position",
                     10},
          std::tuple{kCleanBoundaryHidden, kCleanBoundaryOutput, "boundary",
                     20}}) {
      auto& h = add(hidden, absl::StrCat("clean_", name, "_hidden"), width,
                    2 * width);
      auto& o = add(result, absl::StrCat("clean_", name, "_output"), 2 * width,
                    width);
      for (int i = 0; i < width; ++i) {
        h.Add(2 * i, i, 2);
        h.Bias(2 * i, -0.5);
        h.Add(2 * i + 1, i, 2);
        h.Bias(2 * i + 1, -1.5);
        o.Add(i, 2 * i, 1);
        o.Add(i, 2 * i + 1, -1);
      }
    }
    // Cache the affine maps of one-hot inputs, exactly as an ordinary embedding
    // lookup does. These are derived from exported weights, not from a machine.
    for (int token = 0; token < kVocabularySize; ++token) {
      Vector one_hot(kVocabularySize);
      one_hot[token] = 1;
      token_embeddings.push_back(Apply(kTokenEmbedding, one_hot));
    }
    for (int p = 0; p < context; ++p) {
      Vector one_hot(context);
      one_hot[p] = 1;
      position_embeddings.push_back(Apply(kPositionEmbedding, one_hot));
    }
  }

  Vector Apply(Weight id, absl::Span<const double> input) const {
    return weights[id].Apply(input);
  }

  Vector Clean(Weight hidden, Weight output,
               absl::Span<const double> input) const {
    return Apply(output, Relu(Apply(hidden, input)));
  }

  // Standard embedding lookup is performed by an affine map of a one-hot ID.
  // Position-dependent behavior likewise lives in explicit embedding weights.
  Vector Embed(int token, int position) const {
    return AddVectors(token_embeddings[token], position_embeddings[position]);
  }

  void Append(int token, Cache& cache) const {
    Vector x = Embed(token, static_cast<int>(cache.position.size()));
    cache.position.push_back(
        {Sparsify(Apply(kPositionKey, x)), Sparsify(Apply(kTokenValue, x))});
    Vector previous_letter = Clean(
        kCleanTokenHidden, kCleanTokenOutput,
        Attend(Apply(kPreviousLetterQuery, x), cache.position, kTokenWidth)
            .output);
    Vector previous_state =
        Clean(kCleanTokenHidden, kCleanTokenOutput,
              Attend(Apply(kPreviousStateQuery, x), cache.position, kTokenWidth)
                  .output);
    Vector triple = Concat(Concat(x, previous_letter), previous_state);
    Vector valid = Relu(Apply(kValidTransition, triple));
    Vector bits = Relu(Apply(kTransitionBits, Concat(triple, valid)));
    cache.transition.push_back(
        {Sparsify(Apply(kTransitionKey, Concat(bits, x))),
         Sparsify(Apply(kTransitionValue, x))});
    cache.semicolon.push_back({Sparsify(Apply(kSemicolonKey, x)),
                               Sparsify(Apply(kPositionValue, x))});
    Vector semicolon = Clean(
        kCleanPositionHidden, kCleanPositionOutput,
        Attend(Apply(kMarkerQuery, x), cache.semicolon, kPositionBits).output);
    cache.boundary.push_back(
        {Sparsify(Apply(kBoundaryKey, x)),
         Sparsify(Apply(kBoundaryValue, Concat(x, semicolon)))});
    cache.current = std::move(x);
  }

  // The only data-dependent host choice is the final vocabulary argmax. In
  // particular, winning attention indices below are used only for diagnostics.
  StepTrace Predict(const Cache& cache) const {
    const Vector& x = cache.current;
    Vector boundary =
        Clean(kCleanBoundaryHidden, kCleanBoundaryOutput,
              Attend(Apply(kMarkerQuery, x), cache.boundary, 20).output);
    Vector counters = Apply(kCounter, Concat(x, boundary));
    AttentionResult input =
        Attend(Apply(kInputQuery, counters), cache.position, kTokenWidth);
    Vector letter = Clean(kCleanTokenHidden, kCleanTokenOutput, input.output);
    AttentionResult transition =
        Attend(Apply(kTransitionQuery, Concat(x, letter)), cache.transition,
               kTokenWidth);
    Vector next =
        Clean(kCleanTokenHidden, kCleanTokenOutput, transition.output);
    Vector flags =
        Apply(kControlFlags, Relu(Apply(kControlHidden, Concat(x, counters))));
    Vector code =
        Apply(kOutputCode, Relu(Apply(kOutputGate, Concat(next, flags))));
    Vector logits = Apply(kOutputProjection, code);
    int output = static_cast<int>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
    return {static_cast<int>(cache.position.size()) - 1,
            static_cast<int>(input.winning_index),
            static_cast<int>(transition.winning_index),
            transition.winning_probability, output};
  }
};

Model::Model(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Model::~Model() = default;

absl::StatusOr<std::unique_ptr<Model>> Model::Create(int max_context) {
  if (max_context < 1 || max_context > 1024)
    return absl::InvalidArgumentError("max_context must be between 1 and 1024");
  return absl::WrapUnique(new Model(absl::make_unique<Impl>(max_context)));
}

absl::StatusOr<Generation> Model::Generate(absl::string_view prompt) const {
  ASSIGN_OR_RETURN(auto tokens, TokenizePrompt(prompt));
  if (tokens.size() >= static_cast<size_t>(impl_->max_context))
    return absl::ResourceExhaustedError(
        "prompt leaves no room for the state trace");
  Cache cache;
  for (int token : tokens)
    impl_->Append(token, cache);
  Generation result;
  for (;;) {
    StepTrace next = impl_->Predict(cache);
    result.steps.push_back(next);
    if (next.output_token == kEndToken)
      return result;
    result.tokens.push_back(next.output_token);
    if (cache.position.size() == static_cast<size_t>(impl_->max_context))
      return absl::ResourceExhaustedError("state trace exceeds max_context");
    impl_->Append(next.output_token, cache);
  }
}

absl::Status Model::WriteWeights(std::ostream& output) const {
  output << "# Constructed FSM affine weights; implicit entries are zero.\n"
         << "# matrix NAME OUTPUT_SIZE INPUT_SIZE\n"
         << "# w OUTPUT_INDEX INPUT_INDEX COEFFICIENT; b OUTPUT_INDEX BIAS\n"
         << std::setprecision(17);
  for (const auto& weight : impl_->weights) {
    output << "matrix " << weight.name() << ' ' << weight.output_size() << ' '
           << weight.input_size() << '\n';
    for (auto coefficient : weight.coefficients())
      output << "w " << coefficient.row << ' ' << coefficient.column << ' '
             << coefficient.value << '\n';
    for (int i = 0; i < weight.output_size(); ++i)
      if (weight.bias()[i] != 0)
        output << "b " << i << ' ' << weight.bias()[i] << '\n';
  }
  if (!output)
    return absl::InternalError("could not write constructed weights");
  return absl::OkStatus();
}

size_t Model::nonzero_weight_count() const {
  size_t count = 0;
  for (const auto& weight : impl_->weights) {
    count += weight.coefficients().size();
    count += std::count_if(weight.bias().begin(), weight.bias().end(),
                           [](double x) { return x != 0; });
  }
  return count;
}

int Model::max_context() const { return impl_->max_context; }

std::string Render(const Generation& generation) {
  std::string result;
  for (int token : generation.tokens) {
    if (!result.empty())
      result += ' ';
    if (token == kErrorToken) {
      result += "ERR";
    } else {
      result += static_cast<char>('0' + token / 100);
      result += static_cast<char>('0' + token / 10 % 10);
      result += static_cast<char>('0' + token % 10);
    }
  }
  return result;
}

}  // namespace pluto::llm::fsm::constructed
