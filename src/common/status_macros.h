#pragma once

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

// Returns from the current function when `expression` produces a non-OK
// absl::Status. The expression is evaluated exactly once.
#define RETURN_IF_ERROR(expression)                         \
  do {                                                      \
    const ::absl::Status pluto_status_macro_status =        \
        (expression);                                       \
    if (!pluto_status_macro_status.ok()) {                  \
      return pluto_status_macro_status;                     \
    }                                                       \
  } while (false)

// Assigns the value produced by a successful absl::StatusOr expression to
// `lhs`, or returns its error. `lhs` may be either an existing variable or a
// declaration such as `auto value`; a declared variable remains in the
// surrounding scope. Use this macro as a complete statement, including the
// trailing semicolon.
#define ASSIGN_OR_RETURN(lhs, expression)                                  \
  PLUTO_STATUS_MACROS_ASSIGN_OR_RETURN_(                                   \
      PLUTO_STATUS_MACROS_CONCAT_(pluto_status_macro_value, __COUNTER__),  \
      lhs, expression)

#define PLUTO_STATUS_MACROS_CONCAT_INNER_(left, right) left##right
#define PLUTO_STATUS_MACROS_CONCAT_(left, right) \
  PLUTO_STATUS_MACROS_CONCAT_INNER_(left, right)

#define PLUTO_STATUS_MACROS_ASSIGN_OR_RETURN_(status_or, lhs, expression) \
  auto status_or = (expression);                                           \
  if (!status_or.ok()) {                                                   \
    return status_or.status();                                             \
  }                                                                        \
  lhs = std::move(status_or).value()
