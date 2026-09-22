#pragma once

#include <string>

namespace pluto::llm::discretized::generator {

// Complete C++ text for one pure transition function and its private helpers.
// The caller supplies namespace, headers, and the runtime interface wrapper.
struct SerializedCppProgram {
  std::string source;
};

}  // namespace pluto::llm::discretized::generator
