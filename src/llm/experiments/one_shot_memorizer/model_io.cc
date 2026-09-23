#include "src/llm/experiments/one_shot_memorizer/model_io.h"

#include <bit>
#include <cstdint>
#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {
constexpr absl::string_view kMagic = "PLUTO_WFA_V1\n";
constexpr absl::string_view kReluMagic = "PLUTO_RELU_V1\n";
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

void Append(uint64_t value, int width, std::string& bytes) {
  for (int i = 0; i < width; ++i)
    bytes.push_back(static_cast<char>((value >> (8 * i)) & 255));
}

absl::StatusOr<uint64_t> Read(int width, absl::string_view& bytes) {
  if (bytes.size() < static_cast<size_t>(width))
    return absl::DataLossError("truncated automaton weights");
  uint64_t value = 0;
  for (int i = 0; i < width; ++i)
    value |= uint64_t{static_cast<unsigned char>(bytes[i])} << (8 * i);
  bytes.remove_prefix(width);
  return value;
}

absl::StatusOr<int> ReadInt(absl::string_view& bytes) {
  ASSIGN_OR_RETURN(auto value, Read(4, bytes));
  if (value > std::numeric_limits<int>::max())
    return absl::DataLossError("invalid signed integer in automaton weights");
  return static_cast<int>(value);
}

absl::StatusOr<size_t> ReadSize(absl::string_view& bytes) {
  ASSIGN_OR_RETURN(auto value, Read(8, bytes));
  if (value > std::numeric_limits<size_t>::max())
    return absl::DataLossError("automaton dimension exceeds host size_t");
  return static_cast<size_t>(value);
}
}  // namespace

absl::StatusOr<std::string> SerializeModel(const Model& model) {
  RETURN_IF_ERROR(ValidateModel(model));
  std::string bytes(kMagic);
  Append(model.vocabulary_size, 4, bytes);
  Append(model.eos_token_id, 4, bytes);
  Append(model.initial_state, 8, bytes);
  Append(model.trie_state_count, 8, bytes);
  Append(model.sentence_count, 8, bytes);
  Append(model.states.size(), 8, bytes);
  for (const auto& state : model.states) {
    Append(state.terminal_count, 8, bytes);
    Append(state.suffix_count, 8, bytes);
    Append(state.transitions.size(), 8, bytes);
    for (const auto& edge : state.transitions) {
      Append(edge.token, 4, bytes);
      Append(edge.target, 8, bytes);
    }
  }
  return bytes;
}

absl::StatusOr<Model> DeserializeModel(absl::string_view bytes) {
  if (!bytes.starts_with(kMagic))
    return absl::DataLossError("invalid automaton weights format");
  bytes.remove_prefix(kMagic.size());
  Model model;
  ASSIGN_OR_RETURN(model.vocabulary_size, ReadInt(bytes));
  ASSIGN_OR_RETURN(model.eos_token_id, ReadInt(bytes));
  ASSIGN_OR_RETURN(model.initial_state, ReadSize(bytes));
  ASSIGN_OR_RETURN(model.trie_state_count, ReadSize(bytes));
  ASSIGN_OR_RETURN(model.sentence_count, Read(8, bytes));
  ASSIGN_OR_RETURN(const uint64_t states, Read(8, bytes));
  if (states > bytes.size() / 24)
    return absl::DataLossError("state count exceeds available weights");
  model.states.reserve(states);
  for (uint64_t i = 0; i < states; ++i) {
    State state;
    ASSIGN_OR_RETURN(state.terminal_count, Read(8, bytes));
    ASSIGN_OR_RETURN(state.suffix_count, Read(8, bytes));
    ASSIGN_OR_RETURN(const uint64_t edges, Read(8, bytes));
    if (edges > bytes.size() / 12)
      return absl::DataLossError("transition count exceeds available weights");
    state.transitions.reserve(edges);
    for (uint64_t j = 0; j < edges; ++j) {
      ASSIGN_OR_RETURN(const int token, ReadInt(bytes));
      ASSIGN_OR_RETURN(const size_t target, ReadSize(bytes));
      state.transitions.push_back({token, target});
    }
    model.states.push_back(std::move(state));
  }
  if (!bytes.empty())
    return absl::DataLossError("trailing bytes after automaton weights");
  RETURN_IF_ERROR(ValidateModel(model));
  return model;
}

absl::StatusOr<std::string> SerializeReluMemory(const ReluMemory& model) {
  RETURN_IF_ERROR(ValidateReluMemory(model));
  std::string bytes(kReluMagic);
  Append(model.vocabulary_size, 4, bytes);
  Append(model.eos_token_id, 4, bytes);
  Append(model.prompt_tokens, 8, bytes);
  Append(model.context_window, 8, bytes);
  Append(model.units.size(), 8, bytes);
  for (const auto& unit : model.units) {
    for (float bias : unit.input_biases)
      Append(std::bit_cast<uint32_t>(bias), 4, bytes);
    for (float value : unit.output_code)
      Append(std::bit_cast<uint32_t>(value), 4, bytes);
  }
  return bytes;
}

absl::StatusOr<ReluMemory> DeserializeReluMemory(absl::string_view bytes) {
  if (!bytes.starts_with(kReluMagic))
    return absl::DataLossError("invalid ReLU memory format");
  bytes.remove_prefix(kReluMagic.size());
  ReluMemory model;
  ASSIGN_OR_RETURN(model.vocabulary_size, ReadInt(bytes));
  ASSIGN_OR_RETURN(model.eos_token_id, ReadInt(bytes));
  ASSIGN_OR_RETURN(model.prompt_tokens, ReadSize(bytes));
  ASSIGN_OR_RETURN(model.context_window, ReadSize(bytes));
  ASSIGN_OR_RETURN(const size_t units, ReadSize(bytes));
  if (model.context_window > (std::numeric_limits<size_t>::max() / 4 - 16) / 2)
    return absl::DataLossError("ReLU dimensions overflow size_t");
  const size_t unit_bytes = (2 * model.context_window + 16) * 4;
  if (units > bytes.size() / unit_bytes || units * unit_bytes != bytes.size())
    return absl::DataLossError("ReLU dimensions disagree with weight bytes");
  model.units.reserve(units);
  for (size_t i = 0; i < units; ++i) {
    ReluMemoryUnit unit;
    unit.input_biases.resize(2 * model.context_window);
    for (float& bias : unit.input_biases) {
      ASSIGN_OR_RETURN(auto bits, Read(4, bytes));
      bias = std::bit_cast<float>(static_cast<uint32_t>(bits));
    }
    for (float& value : unit.output_code) {
      ASSIGN_OR_RETURN(auto bits, Read(4, bytes));
      value = std::bit_cast<float>(static_cast<uint32_t>(bits));
    }
    model.units.push_back(std::move(unit));
  }
  RETURN_IF_ERROR(ValidateReluMemory(model));
  return model;
}

}  // namespace pluto::llm::one_shot_memorizer
