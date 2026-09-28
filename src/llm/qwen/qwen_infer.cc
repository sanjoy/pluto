#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/qwen_tokenizer.h"
#include "src/llm/qwen/model.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Downloaded Qwen3.8-27B-FP8 HF directory");
ABSL_FLAG(std::string, prompt, "What is the capital of France? Answer briefly.",
          "Text prompt");
ABSL_FLAG(int, max_new_tokens, 32, "Maximum greedy continuation tokens");
ABSL_FLAG(int, context_length, 512,
          "Maximum total prompt plus generated tokens");
ABSL_FLAG(bool, raw_prompt, false, "Do not apply the model's chat template");
ABSL_FLAG(bool, thinking, false, "Enable the chat template's thinking mode");

namespace {

// Keep the command intentionally small: native tokenization, cached decoder
// steps, and greedy selection. No Python runtime or external serving engine.
absl::Status Run() {
  if (absl::GetFlag(FLAGS_checkpoint).empty() ||
      absl::GetFlag(FLAGS_max_new_tokens) <= 0 ||
      absl::GetFlag(FLAGS_context_length) <= 0)
    return absl::InvalidArgumentError(
        "--checkpoint and positive token/context limits are required");
  if (absl::GetFlag(FLAGS_raw_prompt) && absl::GetFlag(FLAGS_thinking))
    return absl::InvalidArgumentError(
        "--thinking is only valid with the chat template");
  ASSIGN_OR_RETURN(auto tokenizer, pluto::tokenizer::QwenTokenizer::Load(
                                       absl::GetFlag(FLAGS_checkpoint)));
  std::string prompt = absl::GetFlag(FLAGS_prompt);
  if (!absl::GetFlag(FLAGS_raw_prompt))
    prompt = pluto::tokenizer::QwenTokenizer::ChatPrompt(
        prompt, absl::GetFlag(FLAGS_thinking));
  ASSIGN_OR_RETURN(auto tokens, tokenizer->Encode(prompt));
  if (tokens.empty() ||
      tokens.size() + size_t(absl::GetFlag(FLAGS_max_new_tokens)) >
          size_t(absl::GetFlag(FLAGS_context_length)))
    return absl::InvalidArgumentError(
        "prompt plus requested continuation exceeds --context_length");
  ASSIGN_OR_RETURN(auto executor, pluto::cuda::Executor::Create());
  pluto::llm::qwen::InferenceOptions options;
  options.context_length = absl::GetFlag(FLAGS_context_length);
  options.load_progress = [](int loaded, int total) {
    if (loaded % 8 == 0 || loaded == total)
      std::cerr << "Loaded decoder blocks " << loaded << '/' << total << '\n';
  };
  auto start = std::chrono::steady_clock::now();
  ASSIGN_OR_RETURN(auto model,
                   pluto::llm::qwen::Model::Load(
                       *executor, absl::GetFlag(FLAGS_checkpoint), options));
  if (tokenizer->vocab_size() > model->config().vocab_size)
    return absl::InvalidArgumentError("tokenizer exceeds model vocabulary");
  std::cerr << "GPU weight storage: " << model->weight_bytes()
            << " bytes; prompt: " << tokens.size() << " tokens\n";
  for (int token : tokens)
    RETURN_IF_ERROR(model->Step(token));
  ASSIGN_OR_RETURN(auto scores,
                   pluto::cuda::PageLockedHostArray<float>::Allocate(
                       *executor, model->config().vocab_size));
  std::vector<int> continuation;
  bool eos = false;
  for (int step = 0; step < absl::GetFlag(FLAGS_max_new_tokens); ++step) {
    ASSIGN_OR_RETURN(auto logits, model->Logits());
    RETURN_IF_ERROR(pluto::cuda::CudaStatus(
        cudaMemcpyAsync(scores.data(), logits.data(), scores.size_bytes(),
                        cudaMemcpyDeviceToHost, executor->stream()),
        "copy Qwen logits"));
    RETURN_IF_ERROR(executor->Synchronize());
    for (float score : scores)
      if (!std::isfinite(score))
        return absl::DataLossError("non-finite Qwen logits");
    // Padded head rows have no token spelling and cannot be generated.
    int next = static_cast<int>(
        std::max_element(scores.begin(),
                         scores.begin() + tokenizer->vocab_size()) -
        scores.begin());
    if (next == tokenizer->eos_token_id() ||
        next == model->config().eos_token_id) {
      eos = true;
      break;
    }
    // Concatenate token bytes before printing, since byte-BPE tokens can end
    // inside a Unicode scalar. This also keeps diagnostic output on stderr.
    continuation.push_back(next);
    if (step + 1 < absl::GetFlag(FLAGS_max_new_tokens))
      RETURN_IF_ERROR(model->Step(next));
  }
  ASSIGN_OR_RETURN(auto text, tokenizer->Decode(continuation));
  std::cout << text << '\n';
  double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::cerr << "Generated " << continuation.size() << " tokens; "
            << (eos ? "EOS" : "token limit") << "; load + inference " << seconds
            << " seconds\n";
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  auto positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "Use --prompt for input text\n";
    return 1;
  }
  auto status = Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
