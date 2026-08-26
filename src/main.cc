#include <iostream>

#include "absl/strings/str_format.h"

int main() {
  std::cout << absl::StrFormat("%s %s\n", "hello", "world");
  return 0;
}
