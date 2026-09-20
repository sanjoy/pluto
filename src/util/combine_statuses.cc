#include "src/util/combine_statuses.h"

#include <cstddef>
#include <string>

#include "absl/strings/str_cat.h"

namespace pluto::util {

absl::Status CombineStatuses(absl::Span<const absl::Status> statuses) {
  absl::StatusCode code = absl::StatusCode::kOk;
  std::string message;
  for (size_t i = 0; i < statuses.size(); ++i) {
    if (statuses[i].ok())
      continue;
    if (code == absl::StatusCode::kOk) {
      code = statuses[i].code();
      message = "Failures:";
    }
    absl::StrAppend(&message, "\n  [", i, "] ", statuses[i].ToString());
  }
  return absl::Status(code, message);
}

}  // namespace pluto::util
