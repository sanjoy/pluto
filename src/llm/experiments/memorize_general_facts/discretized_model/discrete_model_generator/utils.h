#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace pluto::llm::discretized::generator {

// Qualified C++ vocabulary names indexed by compact token ID.
using TokenNames = std::map<int, std::string>;

absl::StatusOr<std::string> ReadFile(const std::filesystem::path& path);
absl::Status WriteFile(const std::filesystem::path& path,
                       absl::string_view data);
std::string Sha256(absl::string_view data);
absl::StatusOr<std::string> Sha256File(const std::filesystem::path& path);
absl::StatusOr<std::filesystem::path> FindExecutable(absl::string_view name);

// Launch directly, without interpreting paths or arguments as shell syntax.
absl::Status RunProcess(const std::vector<std::string>& arguments);

// Safe C++ identifier validation shared by all layer lowerers.
absl::Status ValidateTransitionFunctionName(absl::string_view name);

// Exact byte literals, stable vocabulary names, and generated source wrappers.
std::string Literal(absl::string_view bytes);
std::string TokenName(absl::string_view bytes, int compact_id, int eos_token);
std::string Source(
    absl::string_view body, absl::string_view header = "\"tables.h\"",
    absl::string_view description = "", bool vocabulary = false,
    absl::string_view name_space = "pluto::llm::discretized::gen::internal",
    const std::vector<std::string>& extra_headers = {});
std::string TransitionObject(absl::string_view body,
                             absl::string_view interface,
                             absl::string_view factory);

// Emit a nonempty C++ array declaration, with a harmless sentinel for no rows.
std::string CppArray(absl::string_view type, absl::string_view name,
                     const std::vector<std::string>& rows);

// Append a local constexpr array in readable, fixed-width rows.
// Callers provide nonempty values and a strictly positive column count.
void AppendCppArray(std::vector<std::string>& lines, absl::string_view name,
                    absl::string_view type,
                    const std::vector<std::string>& values,
                    size_t columns = 12);

// Join source lines with newlines, including a final newline.
std::string JoinCppLines(const std::vector<std::string>& lines);

}  // namespace pluto::llm::discretized::generator
