#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/llm/experiments/finite_state_machine/constructed/model.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, prompt, "",
          "Run one FSM prompt ending in '>' instead of evaluating corpora");
ABSL_FLAG(bool, trace, false,
          "Print attention and output diagnostics for each --prompt step");
ABSL_FLAG(std::string, training_data,
          "testdata/finite_state_machine_full_training_data.txt",
          "Training corpus to evaluate without fitting weights");
ABSL_FLAG(std::string, test_data,
          "testdata/finite_state_machine_full_test_data.txt",
          "Test corpus to evaluate without teacher forcing");
ABSL_FLAG(
    std::string, weights, "",
    "Optional output file for all nonzero constructed weights and biases");
ABSL_FLAG(int, max_context, 1024,
          "Maximum context tokens, including generated tokens (1..1024)");

namespace pluto::llm::fsm::constructed {
namespace {

struct Evaluation {
  size_t total = 0;
  size_t exact = 0;
};

std::string WithoutSpaces(std::string text) {
  text.erase(std::remove(text.begin(), text.end(), ' '), text.end());
  return text;
}

void PrintTrace(const Generation& generation) {
  std::cout << "position\tinput_position\ttransition_position\t"
               "lookup_probability\toutput_token\n"
            << std::setprecision(17);
  for (const StepTrace& step : generation.steps) {
    std::cout << step.position << '\t' << step.input_position << '\t'
              << step.transition_position << '\t' << step.lookup_probability
              << '\t' << step.output_token << '\n';
  }
}

absl::StatusOr<Evaluation> Evaluate(const Model& model, const std::string& path,
                                    const std::string& split) {
  std::ifstream input(path);
  if (!input.is_open())
    return absl::NotFoundError("could not open " + split + " corpus: " + path);

  Evaluation result;
  size_t line_number = 0;
  std::string line;
  while (std::getline(input, line)) {
    ++line_number;
    // Accommodate CRLF files without changing the comparison's space policy.
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (WithoutSpaces(line).empty())
      continue;
    const std::string location = path + ":" + std::to_string(line_number);
    const size_t separator = line.find('>');
    if (separator == std::string::npos ||
        line.find('>', separator + 1) != std::string::npos) {
      return absl::InvalidArgumentError(
          location + ": expected exactly one '>' before the trace label");
    }
    const std::string expected = WithoutSpaces(line.substr(separator + 1));
    if (expected.empty())
      return absl::InvalidArgumentError(location + ": missing trace label");

    // Labels are used only after free-running generation has finished.
    auto generation = model.Generate(line.substr(0, separator + 1));
    if (!generation.ok()) {
      return absl::Status(generation.status().code(),
                          location + ": " + generation.status().ToString());
    }
    const std::string actual = Render(*generation);
    ++result.total;
    if (WithoutSpaces(actual) == expected) {
      ++result.exact;
    } else if (result.total - result.exact <= 5) {
      std::cerr << location << ": trace mismatch\n"
                << "  expected: " << line.substr(separator + 1) << '\n'
                << "  actual:   " << actual << '\n';
    }
    if (result.total % 512 == 0) {
      std::cout << split << " progress: exact=" << result.exact
                << " processed=" << result.total << '\n'
                << std::flush;
      if (!std::cout)
        return absl::DataLossError("failed writing evaluation progress");
    }
  }
  if (input.bad() || !input.eof())
    return absl::DataLossError("failed reading " + split + " corpus: " + path);
  if (result.total == 0)
    return absl::InvalidArgumentError(split + " corpus is empty: " + path);
  std::cout << split << ": exact=" << result.exact << " total=" << result.total
            << " accuracy=" << std::fixed << std::setprecision(2)
            << (100.0 * result.exact / result.total) << "%\n";
  return result;
}

absl::Status WriteWeights(const Model& model, const std::string& path) {
  std::ofstream output(path);
  if (!output.is_open()) {
    return absl::PermissionDeniedError("could not open weights output: " +
                                       path);
  }
  RETURN_IF_ERROR(model.WriteWeights(output));
  output.close();
  if (!output)
    return absl::DataLossError("failed writing weights output: " + path);
  return absl::OkStatus();
}

absl::Status Run() {
  const int max_context = absl::GetFlag(FLAGS_max_context);
  if (max_context < 1 || max_context > 1024) {
    return absl::InvalidArgumentError(
        "--max_context must be between 1 and 1024");
  }
  ASSIGN_OR_RETURN(auto model, Model::Create(max_context));
  std::cout << "constructed model: nonzero_weights="
            << model->nonzero_weight_count()
            << " max_context=" << model->max_context() << '\n'
            << std::flush;
  if (!std::cout)
    return absl::DataLossError("failed writing model summary");
  const std::string weights = absl::GetFlag(FLAGS_weights);
  if (!weights.empty())
    RETURN_IF_ERROR(WriteWeights(*model, weights));

  const std::string prompt = absl::GetFlag(FLAGS_prompt);
  if (!prompt.empty()) {
    ASSIGN_OR_RETURN(auto generation, model->Generate(prompt));
    std::cout << Render(generation) << '\n';
    if (absl::GetFlag(FLAGS_trace))
      PrintTrace(generation);
  } else {
    ASSIGN_OR_RETURN(
        auto training,
        Evaluate(*model, absl::GetFlag(FLAGS_training_data), "training"));
    ASSIGN_OR_RETURN(auto test,
                     Evaluate(*model, absl::GetFlag(FLAGS_test_data), "test"));
    std::cout.flush();
    if (!std::cout)
      return absl::DataLossError("failed writing evaluation results");
    if (training.exact != training.total || test.exact != test.total) {
      return absl::FailedPreconditionError(
          "free-running evaluation did not exactly match every trace");
    }
  }
  std::cout.flush();
  if (!std::cout)
    return absl::DataLossError("failed writing model output");
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::fsm::constructed

int main(int argc, char** argv) {
  const auto positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "Use --prompt for a single input, or --training_data and "
                 "--test_data for corpus evaluation.\n";
    return 1;
  }
  const auto started = std::chrono::steady_clock::now();
  const absl::Status status = pluto::llm::fsm::constructed::Run();
  const std::chrono::duration<double> elapsed =
      std::chrono::steady_clock::now() - started;
  std::cerr << "elapsed_seconds=" << std::fixed << std::setprecision(2)
            << elapsed.count() << '\n';
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
