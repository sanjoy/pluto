#include <memory>
#include <type_traits>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/llm/layers/sparse_autoencoder.h"
#include "src/llm/layers/test_util.h"

namespace pluto::llm {
namespace {

static_assert(std::is_same_v<decltype(&Layer::name),
                             absl::string_view (Layer::*)() const>);
static_assert(std::is_same_v<decltype(&LayerReference::name),
                             absl::string_view (LayerReference::*)() const>);

// Exercise virtual dispatch through a const base, not just the concrete
// overload. No forward/backward execution or initialized weights are needed
// to inspect a layer, and the returned view must remain stable across calls.
template <class Base, class Concrete>
void ExpectName(const absl::StatusOr<std::unique_ptr<Concrete>>& result,
                absl::string_view expected) {
  ASSERT_TRUE(result.ok()) << result.status();
  const Base& layer = **result;
  const absl::string_view name = layer.name();
  EXPECT_EQ(name, expected);
  EXPECT_EQ(layer.name(), name);
  EXPECT_EQ(layer.name().data(), name.data());
}

TEST_F(LayersTest, EveryGpuLayerReportsItsDiagnosticName) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    SCOPED_TRACE(static_cast<int>(type));
    auto embedding = EmbeddingLookupLayer::Create(*executor_, 33, 16, type);
    ExpectName<Layer>(embedding, "EmbeddingLookupLayer");
    ASSERT_TRUE(embedding.ok()) << embedding.status();
    ExpectName<Layer>(LanguageModelingHeadLayer::Create(embedding->get()),
                      "LanguageModelingHeadLayer");
    ExpectName<Layer>(PositionEmbeddingLayer::Create(*executor_, 4, 16, type),
                      "PositionEmbeddingLayer");
    ExpectName<Layer>(FullyConnectedLayer::Create(*executor_, 16, 32, type),
                      "FullyConnectedLayer");
    ExpectName<Layer>(AttentionLayer::Create(*executor_, 4, 1, 16, type),
                      "AttentionLayer");
    ExpectName<Layer>(LayerNormLayer::Create(*executor_, 16, 1e-5f, type),
                      "LayerNormLayer");
    ExpectName<Layer>(GeluLayer::Create(*executor_, 16, type), "GeluLayer");
    ExpectName<Layer>(CrossEntropyLossLayer::Create(*executor_, 33, type),
                      "CrossEntropyLossLayer");
    for (auto mode : {SparseAutoEncoderLayer::Mode::kDefault,
                      SparseAutoEncoderLayer::Mode::kCollectStatistics})
      ExpectName<Layer>(
          SparseAutoEncoderLayer::Create(*executor_, 16, 32, type, mode),
          "SparseAutoEncoderLayer");
    ExpectName<Layer>(
        SparseAutoEncoderLossLayer::Create(*executor_, 16, 32, 0.5f, type),
        "SparseAutoEncoderLossLayer");

    auto branch = FullyConnectedLayer::Create(*executor_, 16, type);
    ASSERT_TRUE(branch.ok()) << branch.status();
    ExpectName<Layer>(ResidualLayer::Create(std::move(*branch)),
                      "ResidualLayer");
    ComposedLayerBuilder builder;
    ASSERT_TRUE(builder.add(GeluLayer::Create(*executor_, 16, type)).ok());
    ExpectName<Layer>(builder.create("activation_pipeline"),
                      "activation_pipeline");
  }
}

TEST(LayerNameTest, EveryReferenceLayerReportsItsDiagnosticName) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    SCOPED_TRACE(static_cast<int>(type));
    auto embedding = EmbeddingLookupLayerReference::Create(33, 16, type);
    ExpectName<LayerReference>(embedding, "EmbeddingLookupLayerReference");
    ASSERT_TRUE(embedding.ok()) << embedding.status();
    ExpectName<LayerReference>(
        LanguageModelingHeadLayerReference::Create(embedding->get()),
        "LanguageModelingHeadLayerReference");
    ExpectName<LayerReference>(
        PositionEmbeddingLayerReference::Create(4, 16, type),
        "PositionEmbeddingLayerReference");
    ExpectName<LayerReference>(
        FullyConnectedLayerReference::Create(16, 32, type),
        "FullyConnectedLayerReference");
    ExpectName<LayerReference>(AttentionLayerReference::Create(4, 1, 16, type),
                               "AttentionLayerReference");
    ExpectName<LayerReference>(LayerNormLayerReference::Create(16, 1e-5f, type),
                               "LayerNormLayerReference");
    ExpectName<LayerReference>(GeluLayerReference::Create(16, type),
                               "GeluLayerReference");
    ExpectName<LayerReference>(CrossEntropyLossLayerReference::Create(33, type),
                               "CrossEntropyLossLayerReference");
    ExpectName<LayerReference>(
        SparseAutoEncoderLayerReference::Create(16, 32, type),
        "SparseAutoEncoderLayerReference");
    ExpectName<LayerReference>(
        SparseAutoEncoderLossLayerReference::Create(16, 32, 0.5f, type),
        "SparseAutoEncoderLossLayerReference");

    auto branch = FullyConnectedLayerReference::Create(16, type);
    ASSERT_TRUE(branch.ok()) << branch.status();
    ExpectName<LayerReference>(
        ResidualLayerReference::Create(std::move(*branch)),
        "ResidualLayerReference");
    ComposedLayerReferenceBuilder builder;
    ASSERT_TRUE(builder.add(GeluLayerReference::Create(16, type)).ok());
    ExpectName<LayerReference>(builder.create("activation_pipeline_reference"),
                               "activation_pipeline_reference");
  }
}

}  // namespace
}  // namespace pluto::llm
