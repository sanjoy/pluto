#include "src/llm/optimizer.h"

#include <memory>
#include <utility>

#include "src/llm/adamw_optimizer.h"
#include "src/util/status_macros.h"

namespace pluto::llm {

absl::StatusOr<std::unique_ptr<Optimizer>> Optimizer::Create(
    cuda::Executor& executor, Layer& model, AdamWConfig config) {
  ASSIGN_OR_RETURN(auto optimizer,
                   AdamWOptimizer::Create(executor, model, config));
  return std::unique_ptr<Optimizer>(std::move(optimizer));
}

}  // namespace pluto::llm
