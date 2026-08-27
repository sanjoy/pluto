#ifndef PLUTO_SRC_LLM_LAYERS_COMBINATORS_H_
#define PLUTO_SRC_LLM_LAYERS_COMBINATORS_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Wraps a unary layer as x + layer(x), retaining the child's tape and weights.
class ResidualLayer final : public Layer {
 public:
  explicit ResidualLayer(std::unique_ptr<Layer> layer);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override {
    return absl::MakeSpan(weights_);
  }
  DataType output_type() const override { return layer_->output_type(); }

 private:
  std::unique_ptr<Layer> layer_;
  std::vector<Buffer> weights_;
};

// Sequentially composes unary layers. Multi-input terminal operations, such as
// cross entropy with labels, intentionally remain outside the predictor.
class ComposedLayer final : public Layer {
 public:
  ComposedLayer(DataType data_type,
                std::vector<std::unique_ptr<Layer>> layers);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override {
    return absl::MakeSpan(weights_);
  }
  DataType output_type() const override { return output_type_; }

 private:
  DataType output_type_;
  std::vector<std::unique_ptr<Layer>> layers_;
  std::vector<Buffer> weights_;
};

// Repeats a concrete Layer type while retaining ordinary Layer polymorphism.
// Each repetition has independent parameters; callers that want tied weights
// can instead pass one layer explicitly from a custom composite.
template <class LayerToRepeat>
class RepeatedLayer final : public Layer {
 public:
  RepeatedLayer(DataType data_type,
                std::vector<std::unique_ptr<LayerToRepeat>> layers)
      : output_type_(data_type), layers_(std::move(layers)) {
    for (const auto& layer : layers_) {
      for (Buffer& weight : layer->weights()) weights_.push_back(weight);
    }
  }

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override {
    if (inputs.size() != 1) {
      return absl::InvalidArgumentError(
          "RepeatedLayer fwd expects one activation buffer");
    }
    tape->intermediates.clear();
    tape->children.clear();
    Buffer activation = inputs.front();
    for (auto& layer : layers_) {
      Tape child_tape;
      BufferVec child_inputs = {activation};
      auto output = layer->fwd(child_inputs, &child_tape);
      if (!output.ok()) return output.status();
      activation = *std::move(output);
      tape->children.push_back(std::move(child_tape));
    }
    return activation;
  }

  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override {
    if (output_gradients.size() != 1 ||
        tape.children.size() != layers_.size()) {
      return absl::InvalidArgumentError(
          "RepeatedLayer bwd received an incompatible gradient or tape");
    }
    Buffer gradient = output_gradients.front();
    for (size_t index = layers_.size(); index-- > 0;) {
      BufferVec child_gradients = {gradient};
      auto inputs = layers_[index]->bwd(
          child_gradients, std::move(tape.children[index]));
      if (!inputs.ok()) return inputs.status();
      if (inputs->size() != 1) {
        return absl::InternalError(
            "a repeated unary layer returned multiple input gradients");
      }
      gradient = inputs->front();
    }
    return BufferVec{gradient};
  }

  absl::Span<Buffer> weights() override {
    return absl::MakeSpan(weights_);
  }
  DataType output_type() const override { return output_type_; }

 private:
  DataType output_type_;
  std::vector<std::unique_ptr<LayerToRepeat>> layers_;
  std::vector<Buffer> weights_;
};

}  // namespace pluto::llm

#endif  // PLUTO_SRC_LLM_LAYERS_COMBINATORS_H_
