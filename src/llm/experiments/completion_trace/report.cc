#include "src/llm/experiments/completion_trace/report.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::completion_trace {
namespace {

std::string Html(absl::string_view text) {
  std::string result;
  for (char c : text)
    switch (c) {
      case '&':
        result += "&amp;";
        break;
      case '<':
        result += "&lt;";
        break;
      case '>':
        result += "&gt;";
        break;
      case '"':
        result += "&quot;";
        break;
      case '\'':
        result += "&#39;";
        break;
      default:
        result += c;
    }
  return result;
}

std::string Quote(absl::string_view text) {
  return absl::StrCat("\"", absl::CEscape(text), "\"");
}

const char* Dtype(DataType type) {
  switch (type) {
    case DataType::FP32:
      return "FP32";
    case DataType::BF16:
      return "BF16";
    case DataType::FP16:
      return "FP16";
    case DataType::INT32:
      return "INT32";
    case DataType::FP8:
      return "FP8";
  }
  return "unknown";
}

size_t ElementBytes(DataType type) {
  switch (type) {
    case DataType::FP32:
    case DataType::INT32:
      return 4;
    case DataType::BF16:
    case DataType::FP16:
      return 2;
    case DataType::FP8:
      return 1;
  }
  return 0;
}

std::string Key(const TensorSnapshot& tensor) {
  return absl::StrCat(tensor.scope, "/", tensor.name, "[", tensor.occurrence,
                      "]/output", tensor.output_index);
}

std::string Tokens(const ReportMetadata& metadata, absl::Span<const int> ids) {
  std::string result;
  for (int id : ids)
    result += metadata.token_text[id];
  return result;
}

absl::Status ValidateTensor(const TensorSnapshot& tensor, size_t prefix,
                            bool attention) {
  const auto& dims = tensor.dimensions;
  if (dims.size() < 2 || dims[0] != 1 || ElementBytes(tensor.data_type) == 0)
    return absl::InvalidArgumentError("invalid snapshot shape/dtype");
  size_t elements = 1;
  for (int64_t dimension : dims) {
    if (dimension <= 0 || static_cast<uint64_t>(dimension) >
                              std::numeric_limits<size_t>::max() / elements)
      return absl::InvalidArgumentError("snapshot shape overflows");
    elements *= dimension;
  }
  if (elements != tensor.values.size() ||
      elements >
          std::numeric_limits<size_t>::max() / ElementBytes(tensor.data_type) ||
      tensor.raw_bytes.size() != elements * ElementBytes(tensor.data_type))
    return absl::InvalidArgumentError(
        "snapshot values/raw bytes differ from shape");
  if (!attention) {
    if (dims[1] != static_cast<int64_t>(prefix))
      return absl::InvalidArgumentError(
          "snapshot does not cover the active prefix");
    return absl::OkStatus();
  }
  if (tensor.data_type != DataType::FP32 || dims.size() != 4 ||
      dims[2] != static_cast<int64_t>(prefix) || dims[3] != dims[2])
    return absl::InvalidArgumentError("invalid attention snapshot shape");
  for (int64_t head = 0; head < dims[1]; ++head)
    for (size_t row = 0; row < prefix; ++row) {
      double sum = 0;
      for (size_t col = 0; col < prefix; ++col) {
        const float value = tensor.values[(head * prefix + row) * prefix + col];
        if (!std::isfinite(value) || value < 0 || value > 1.0001f ||
            (col > row && value != 0))
          return absl::InvalidArgumentError(
              "invalid causal attention probability");
        sum += value;
      }
      if (std::abs(sum - 1) > 0.0001)
        return absl::InvalidArgumentError("attention row does not sum to one");
    }
  return absl::OkStatus();
}

absl::Status Validate(const ReportMetadata& metadata,
                      absl::Span<const CompletionExample> examples) {
  if (examples.empty() || metadata.token_text.empty() ||
      metadata.original_ids.size() != metadata.token_text.size() ||
      metadata.model_width <= 0 || metadata.layers < 0 || metadata.heads <= 0 ||
      metadata.feed_forward_width <= 0 || metadata.eos_token < 0 ||
      static_cast<size_t>(metadata.eos_token) >= metadata.token_text.size())
    return absl::InvalidArgumentError("invalid completion report metadata");
  auto valid_id = [&](int id) {
    return id >= 0 && static_cast<size_t>(id) < metadata.token_text.size();
  };
  // Prefix lengths legitimately grow between forwards, but a particular
  // layer/output site must retain its dtype, head count, and channel shape.
  // Statistics compare these vectors coordinate-wise; accepting a different
  // width would both mislabel the comparison and permit out-of-bounds access.
  using Signature = std::pair<DataType, std::vector<int64_t>>;
  std::map<std::pair<bool, std::string>, Signature> signatures;
  auto validate_site = [&](const TensorSnapshot& tensor, size_t prefix,
                           bool attention) -> absl::Status {
    RETURN_IF_ERROR(ValidateTensor(tensor, prefix, attention));
    auto dimensions = tensor.dimensions;
    if (attention) {
      dimensions[2] = -1;
      dimensions[3] = -1;
    } else {
      dimensions[1] = -1;
    }
    Signature signature{tensor.data_type, std::move(dimensions)};
    const auto [entry, inserted] =
        signatures.emplace(std::make_pair(attention, Key(tensor)), signature);
    if (!inserted && entry->second != signature)
      return absl::InvalidArgumentError(
          "snapshot site changes dtype or non-prefix dimensions");
    return absl::OkStatus();
  };
  for (const auto& example : examples) {
    if (example.corpus_line <= 0 || example.prompt.empty() ||
        example.steps.empty())
      return absl::InvalidArgumentError("empty completion example");
    std::vector<int> prefix = example.prompt;
    for (int id : prefix)
      if (!valid_id(id) || id == metadata.eos_token)
        return absl::InvalidArgumentError("invalid prompt token");
    for (size_t index = 0; index < example.steps.size(); ++index) {
      const auto& step = example.steps[index];
      if (step.forward.prefix != prefix || !valid_id(step.predicted) ||
          (step.expected != -1 && !valid_id(step.expected)) ||
          step.forward.next_logits.size() != metadata.token_text.size() ||
          step.forward.activations.empty())
        return absl::InvalidArgumentError("invalid autoregressive trace step");
      for (float value : step.forward.next_logits)
        if (!std::isfinite(value))
          return absl::InvalidArgumentError(
              "nonfinite logical next-token logit");
      const int winner = std::max_element(step.forward.next_logits.begin(),
                                          step.forward.next_logits.end()) -
                         step.forward.next_logits.begin();
      if (winner != step.predicted)
        return absl::InvalidArgumentError(
            "recorded token is not the actual argmax");
      for (const auto& tensor : step.forward.activations)
        RETURN_IF_ERROR(validate_site(tensor, prefix.size(), false));
      for (const auto& tensor : step.forward.attention)
        RETURN_IF_ERROR(validate_site(tensor, prefix.size(), true));
      if (step.predicted == metadata.eos_token &&
          index + 1 != example.steps.size())
        return absl::InvalidArgumentError("trace continued after EOS");
      prefix.push_back(step.predicted);
    }
  }
  return absl::OkStatus();
}

std::string Operation(const TensorSnapshot& tensor) {
  if (tensor.name == "EmbeddingLookupLayer")
    return "Select the input token's learned embedding row; store in "
           "activation dtype.";
  if (tensor.name == "PositionEmbeddingLayer")
    return "Add the learned absolute-position embedding to the token "
           "embedding.";
  if (tensor.name == "LayerNormLayer")
    return "Per row: gamma * (x - mean(x)) / sqrt(mean((x-mean)^2) + 1e-5) + "
           "beta. Statistics are FP32.";
  if (tensor.name == "GeluLayer")
    return "Elementwise: 0.5*x*(1+tanh(0.7978845608*(x+0.044715*x*x*x))); then "
           "activation-dtype storage.";
  if (tensor.name == "AttentionLayer")
    return "Causal softmax(Q*K^T/sqrt(head_width))*V. Q/K/V come from the "
           "preceding packed projection; FlashAttention computes the output.";
  if (tensor.name == "FullyConnectedLayer") {
    if (tensor.scope.ends_with("/attention") && tensor.occurrence == 0)
      return "x*W_qkv+b_qkv; columns are Q, then K, then V (each model_width "
             "entries).";
    if (tensor.scope.ends_with("/attention"))
      return "Attention output projection: x*W_o+b_o.";
    if (tensor.scope.ends_with("/mlp") && tensor.occurrence == 0)
      return "MLP expansion: normalized residual * W_1 + b_1.";
    return "MLP contraction: GELU output * W_2 + b_2.";
  }
  if (tensor.name == "ResidualLayer")
    return "Add this branch's output to its saved input residual; round to "
           "activation dtype.";
  if (tensor.name == "LanguageModelingHeadLayer")
    return "Dot final normalized hidden state with every tied token-embedding "
           "row; output FP32 logits. Padding lanes are not vocabulary tokens.";
  return "Combinator output: the last child output (recorded again to preserve "
         "layer boundaries).";
}

// All elements are printed, eight per line. Logit columns are compact IDs and
// can be resolved through vocabulary.tsv. Original bits disambiguate signed
// zero/exceptional values and make physical storage explicit.
void DumpVector(std::ostream& out, const TensorSnapshot& tensor, size_t offset,
                size_t count, bool include_raw_bytes = true) {
  const size_t bytes = ElementBytes(tensor.data_type);
  for (size_t begin = 0; begin < count; begin += 8) {
    out << "  [" << begin << "] ";
    for (size_t i = begin; i < std::min(count, begin + 8); ++i)
      out << tensor.values[offset + i]
          << (i + 1 < std::min(count, begin + 8) ? ", " : "");
    out << '\n';
  }
  if (!include_raw_bytes)
    return;
  out << "  raw storage bytes (hex, element order):\n";
  const auto flags = out.flags();
  for (size_t begin = 0; begin < count; begin += 16) {
    out << "  [" << begin << "] ";
    for (size_t i = begin; i < std::min(count, begin + 16); ++i) {
      for (size_t byte = 0; byte < bytes; ++byte)
        out << std::hex << std::setw(2) << std::setfill('0')
            << unsigned(tensor.raw_bytes[(offset + i) * bytes + byte]);
      out << ' ';
    }
    out.flags(flags);
    out << std::setfill(' ') << '\n';
  }
}

std::string DumpTensor(const TensorSnapshot& tensor,
                       const ForwardTrace& forward,
                       const ReportMetadata& metadata, bool attention) {
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<float>::max_digits10);
  out << Key(tensor) << " dtype=" << Dtype(tensor.data_type) << " shape=[";
  for (size_t i = 0; i < tensor.dimensions.size(); ++i)
    out << (i == 0 ? "" : ",") << tensor.dimensions[i];
  out << "]\n";
  const size_t n = forward.prefix.size();
  if (attention) {
    out << "Reconstructed FP32 causal probabilities; upper triangle is zero "
           "and omitted.\n";
    for (int64_t head = 0; head < tensor.dimensions[1]; ++head) {
      out << "Head " << head
          << ": columns are input positions 1..query_position.\n";
      for (size_t row = 0; row < n; ++row) {
        out << "Query position " << row + 1 << ' '
            << Quote(metadata.token_text[forward.prefix[row]]) << '\n';
        DumpVector(out, tensor, (head * n + row) * n, row + 1);
      }
    }
  } else {
    out << Operation(tensor) << '\n';
    const size_t columns = tensor.values.size() / n;
    for (size_t row = 0; row < n; ++row) {
      out << "Position " << row + 1 << " (input "
          << Quote(metadata.token_text[forward.prefix[row]]) << ", compact ID "
          << forward.prefix[row] << "), flattened channels 0.." << columns - 1
          << ":\n";
      DumpVector(out, tensor, row * columns, columns);
    }
  }
  return out.str();
}

// A compact companion keeps the actual query's complete feature vectors, not
// a selection of dimensions. Large vocabulary logits and earlier positions
// remain available in the full report; no values there are changed or omitted.
void WriteQueryWalkthrough(std::ostream& out, const ForwardTrace& forward,
                           const ReportMetadata& metadata) {
  const size_t prefix = forward.prefix.size();
  out << "\nLast active input position " << prefix << " ("
      << Quote(metadata.token_text[forward.prefix.back()])
      << ") -- complete feature vectors in execution order:\n";
  for (const auto& tensor : forward.activations) {
    if (tensor.dimensions.size() != 3 ||
        tensor.name == "LanguageModelingHeadLayer" ||
        (tensor.scope.empty() && tensor.name == "gpt2"))
      continue;
    const int64_t width = tensor.dimensions.back();
    if (width != metadata.model_width &&
        width != int64_t{3} * metadata.model_width &&
        width != metadata.feed_forward_width)
      continue;
    out << '\n'
        << Key(tensor) << " dtype=" << Dtype(tensor.data_type)
        << " channels=" << width << '\n'
        << Operation(tensor) << '\n';
    DumpVector(out, tensor, (prefix - 1) * width, width, false);
  }
  out << "\nAttention weights used at that last input position:\n";
  for (const auto& tensor : forward.attention) {
    out << '\n'
        << Key(tensor)
        << " -- FP32 probabilities reconstructed from FlashAttention "
           "statistics\n";
    for (int64_t head = 0; head < tensor.dimensions[1]; ++head) {
      out << "Head " << head << ", query position " << prefix << ":\n";
      for (size_t key = 0; key < prefix; ++key)
        out << "  key position " << key + 1 << ' '
            << Quote(metadata.token_text[forward.prefix[key]]) << " (compact "
            << forward.prefix[key] << "): "
            << tensor.values[(head * prefix + prefix - 1) * prefix + key]
            << '\n';
    }
  }
}

double Norm(absl::Span<const float> values) {
  double sum = 0;
  for (double value : values)
    sum += value * value;
  return std::sqrt(sum);
}

struct QueryVector {
  int line;
  size_t step;
  int token;
  std::vector<float> values;
};

// Ordered maps make reports stable across runs. These are descriptive pairs,
// not independent statistical samples, and do not imply causal ownership.
using Queries = std::map<std::string, std::vector<QueryVector>>;

void WriteStatistics(std::ostream& pairs, std::ostream& statistics,
                     std::ostream& attention_rows, std::ostream& residuals,
                     const ReportMetadata& metadata,
                     absl::Span<const CompletionExample> examples) {
  Queries queries;
  pairs << "layer\tstep\tline_a\tline_b\tpredicted_token_a\tpredicted_token_"
           "b\tsame_predicted_token\tcosine\tcentered_cosine\n";
  statistics
      << "layer\tquery_vectors\tsame_token_pairs\tsame_token_mean_"
         "cosine\tdifferent_token_pairs\tdifferent_token_mean_cosine\tsame_"
         "token_mean_centered_cosine\tdifferent_token_mean_centered_cosine\n";
  attention_rows
      << "line\tstep\tlayer\thead\tkey_position\tinput_token\tprobability\n";
  residuals << "line\tstep\tlayer\tinput_norm\toutput_norm\tupdate_"
               "norm\tcosine_to_input\n";
  for (const auto& example : examples)
    for (size_t step_index = 0; step_index < example.steps.size();
         ++step_index) {
      const auto& step = example.steps[step_index];
      const size_t n = step.forward.prefix.size();
      std::vector<float> previous_residual;
      for (const auto& tensor : step.forward.activations) {
        if (tensor.dimensions.size() != 3 ||
            tensor.dimensions.back() >
                std::max(int64_t{3} * metadata.model_width,
                         int64_t{metadata.feed_forward_width}))
          continue;
        const size_t width = tensor.dimensions.back();
        const auto query = absl::MakeConstSpan(tensor.values).last(width);
        const bool residual_boundary =
            tensor.name == "ResidualLayer" &&
            tensor.scope.starts_with("gpt2/transformer_block_");
        if (residual_boundary && previous_residual.size() == width) {
          std::vector<float> difference(width);
          for (size_t i = 0; i < width; ++i)
            difference[i] = query[i] - previous_residual[i];
          residuals << example.corpus_line << '\t' << step_index + 1 << '\t'
                    << Key(tensor) << '\t' << Norm(previous_residual) << '\t'
                    << Norm(query) << '\t' << Norm(difference) << '\t';
          const auto cosine = CosineSimilarity(previous_residual, query);
          if (cosine)
            residuals << *cosine;
          else
            residuals << "NA";
          residuals << '\n';
        }
        if (tensor.name == "PositionEmbeddingLayer" || residual_boundary)
          previous_residual.assign(query.begin(), query.end());
        if (residual_boundary || tensor.name == "PositionEmbeddingLayer" ||
            tensor.name == "GeluLayer" ||
            (tensor.scope == "gpt2" && tensor.name == "LayerNormLayer"))
          queries[Key(tensor)].push_back({example.corpus_line,
                                          step_index + 1,
                                          step.predicted,
                                          {query.begin(), query.end()}});
      }
      for (const auto& tensor : step.forward.attention)
        for (int64_t head = 0; head < tensor.dimensions[1]; ++head)
          for (size_t key = 0; key < n; ++key)
            attention_rows << example.corpus_line << '\t' << step_index + 1
                           << '\t' << Key(tensor) << '\t' << head << '\t'
                           << key + 1 << '\t'
                           << Quote(
                                  metadata.token_text[step.forward.prefix[key]])
                           << '\t'
                           << tensor.values[(head * n + n - 1) * n + key]
                           << '\n';
    }
  for (const auto& [key, vectors] : queries) {
    const size_t width = vectors.front().values.size();
    std::vector<double> mean(width, 0);
    for (const auto& vector : vectors)
      for (size_t i = 0; i < width; ++i)
        mean[i] += vector.values[i];
    for (double& value : mean)
      value /= vectors.size();
    std::array<double, 2> sums{}, centered_sums{};
    std::array<size_t, 2> counts{}, centered_counts{};
    for (size_t a = 0; a < vectors.size(); ++a)
      for (size_t b = a + 1; b < vectors.size(); ++b) {
        if (vectors[a].step != vectors[b].step)
          continue;
        const bool same = vectors[a].token == vectors[b].token;
        const auto cosine =
            CosineSimilarity(vectors[a].values, vectors[b].values);
        std::vector<float> centered_a(width), centered_b(width);
        for (size_t i = 0; i < width; ++i) {
          centered_a[i] = static_cast<float>(vectors[a].values[i] - mean[i]);
          centered_b[i] = static_cast<float>(vectors[b].values[i] - mean[i]);
        }
        const auto centered = CosineSimilarity(centered_a, centered_b);
        pairs << key << '\t' << vectors[a].step << '\t' << vectors[a].line
              << '\t' << vectors[b].line << '\t' << vectors[a].token << '\t'
              << vectors[b].token << '\t' << same << '\t';
        if (cosine) {
          pairs << *cosine;
          sums[same] += *cosine;
          ++counts[same];
        } else
          pairs << "NA";
        pairs << '\t';
        if (centered) {
          pairs << *centered;
          centered_sums[same] += *centered;
          ++centered_counts[same];
        } else
          pairs << "NA";
        pairs << '\n';
      }
    statistics << key << '\t' << vectors.size();
    for (int same : {1, 0}) {
      statistics << '\t' << counts[same] << '\t';
      if (counts[same])
        statistics << sums[same] / counts[same];
      else
        statistics << "NA";
    }
    for (int same : {1, 0}) {
      statistics << '\t';
      if (centered_counts[same])
        statistics << centered_sums[same] / centered_counts[same];
      else
        statistics << "NA";
    }
    statistics << '\n';
  }
}

}  // namespace

std::optional<double> CosineSimilarity(absl::Span<const float> a,
                                       absl::Span<const float> b) {
  if (a.empty() || a.size() != b.size())
    return std::nullopt;
  double aa = 0, bb = 0, ab = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      return std::nullopt;
    aa += static_cast<double>(a[i]) * a[i];
    bb += static_cast<double>(b[i]) * b[i];
    ab += static_cast<double>(a[i]) * b[i];
  }
  if (aa == 0 || bb == 0)
    return std::nullopt;
  return std::clamp(ab / std::sqrt(aa * bb), -1.0, 1.0);
}

absl::Status WriteReports(const std::filesystem::path& directory,
                          const ReportMetadata& metadata,
                          absl::Span<const CompletionExample> examples) {
  RETURN_IF_ERROR(Validate(metadata, examples));
  const std::array<const char*, 9> names{"report.txt",
                                         "report.html",
                                         "vocabulary.tsv",
                                         "correlations.tsv",
                                         "correlation_summary.tsv",
                                         "query_attention.tsv",
                                         "residual_updates.tsv",
                                         "completions.tsv",
                                         "query_walkthrough.txt"};
  for (const char* name : names) {
    std::error_code error;
    if (std::filesystem::exists(directory / name, error))
      return absl::AlreadyExistsError("completion report file already exists");
    if (error)
      return absl::UnknownError(error.message());
  }
  std::array<std::ofstream, 9> files;
  for (size_t i = 0; i < files.size(); ++i) {
    files[i].open(directory / names[i]);
    if (!files[i])
      return absl::UnknownError("cannot open completion report file");
    files[i] << std::setprecision(std::numeric_limits<float>::max_digits10);
  }
  auto& text = files[0];
  auto& html = files[1];
  auto& vocabulary = files[2];
  auto& completions = files[7];
  auto& walkthrough = files[8];
  walkthrough
      << "Query-position walkthrough\nCheckpoint: " << metadata.checkpoint
      << "\nSource revision: " << metadata.source_revision
      << "\nEach step shows the last active input position that predicts the "
         "next token.\n"
      << "All feature-vector dimensions are printed. Combinator boundaries may "
         "repeat child outputs.\n"
      << "See report.txt for ALL positions, ALL vocabulary logits, full causal "
         "attention matrices, and exact storage bytes.\n"
      << "Top next-token candidates below are the same full-vocabulary softmax "
         "diagnostics as the complete report; generation uses argmax.\n";
  const std::string introduction = absl::StrCat(
      "Autoregressive completion trace\nCheckpoint: ", metadata.checkpoint,
      "\nTokenizer: ", metadata.tokenizer, "\nCorpus: ", metadata.corpus,
      "\nSource revision: ", metadata.source_revision,
      "\nArchitecture: ", metadata.layers, " blocks, width ",
      metadata.model_width, ", heads ", metadata.heads, ", MLP width ",
      metadata.feed_forward_width,
      ". BF16 activations; FP32 statistics/logits; no dropout.\n",
      "Positions are one-based; channels and token IDs are zero-based. A token "
      "at output position N is predicted from input position N-1.\n",
      "Only generated IDs feed subsequent steps. Expected IDs are scoring "
      "metadata, never future model input.\n",
      "Every layer output at every active input position is shown, including "
      "complete physical logits. Future EOS padding and private kernel scratch "
      "are not model-text activations.\n",
      "Attention probabilities are reconstructed from FlashAttention "
      "statistics; capturing them does not replace the fused output path.\n",
      "The attention matrices are causal triangles. Their omitted upper "
      "entries are validated zero. FP32 decimal values have round-trip "
      "precision; raw storage bytes are also shown.\n",
      "Each model block uses row vectors: qkv = LN(x)*W_qkv+b_qkv; x <- x + "
      "Attention(qkv)*W_o+b_o; x <- x + GELU(LN(x)*W_1+b_1)*W_2+b_2.\n",
      "Combinator snapshots can repeat child outputs. Top-token percentages "
      "are FP64 softmax diagnostics computed from actual FP32 logits; "
      "generation is greedy argmax.\n",
      "Correlations compare different examples at the SAME generation step and "
      "group by the predicted NEXT token, not by the current input token. "
      "Centering subtracts the per-layer mean across all recorded query "
      "vectors, then rounds to FP32. This small dependent sample is "
      "descriptive, not causal evidence or an independent statistical test.\n");
  text << introduction << '\n';
  html << "<!doctype html><html><head><meta "
          "charset=\"utf-8\"><title>Completion traces</title><style>"
          "body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:0 "
          "1em;color:#18202a}pre{font:12px "
          "monospace;white-space:pre;overflow:auto;background:#f5f7fa;padding:"
          "1em}summary{cursor:pointer;padding:.35em}details{border-left:2px "
          "solid #cdd6e0;margin:.5em "
          "0;padding-left:.6em}a{color:#155caf}h2{border-top:2px solid "
          "#ddd;padding-top:1em}</style></head><body><h1>Completion "
          "traces</h1><pre>"
       << Html(introduction) << "</pre><nav>";
  for (size_t i = 0; i < examples.size(); ++i)
    html << "<p><a href=\"#example" << i << "\">Line "
         << examples[i].corpus_line << ": "
         << Html(Quote(Tokens(metadata, examples[i].prompt))) << "</a></p>";
  html << "</nav>";
  vocabulary << "compact_id\toriginal_id\ttext\n";
  for (size_t i = 0; i < metadata.token_text.size(); ++i)
    vocabulary << i << '\t' << metadata.original_ids[i] << '\t'
               << Quote(metadata.token_text[i]) << '\n';
  completions << "line\tstep\toutput_position\tcompact_id\toriginal_"
                 "id\ttext\texpected_id\tcorrect\tprobability\tmargin\n";
  for (size_t example_index = 0; example_index < examples.size();
       ++example_index) {
    const auto& example = examples[example_index];
    std::vector<int> generated;
    for (const auto& step : example.steps)
      generated.push_back(step.predicted);
    const std::string title =
        absl::StrCat("Line ", example.corpus_line,
                     "\nPrompt: ", Quote(Tokens(metadata, example.prompt)),
                     "\nGenerated: ", Quote(Tokens(metadata, generated)), "\n");
    text << "\n====================\n" << title;
    walkthrough << "\n====================\n" << title;
    html << "<h2 id=\"example" << example_index << "\">Line "
         << example.corpus_line << "</h2><pre>" << Html(title) << "</pre>";
    for (size_t step_index = 0; step_index < example.steps.size();
         ++step_index) {
      const auto& step = example.steps[step_index];
      const auto& logits = step.forward.next_logits;
      std::vector<int> order(logits.size());
      std::iota(order.begin(), order.end(), 0);
      const size_t count = std::min<size_t>(10, order.size());
      std::partial_sort(
          order.begin(), order.begin() + count, order.end(), [&](int a, int b) {
            return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
          });
      const double maximum = logits[order[0]];
      double denominator = 0;
      for (double logit : logits)
        denominator += std::exp(logit - maximum);
      const double probability = 1.0 / denominator;
      const double margin = order.size() > 1 ? maximum - logits[order[1]] : 0;
      std::ostringstream overview;
      overview << std::setprecision(9) << "Step " << step_index + 1
               << ": output position " << step.forward.prefix.size() + 1
               << " emits " << Quote(metadata.token_text[step.predicted])
               << " (compact " << step.predicted << ", original "
               << metadata.original_ids[step.predicted] << ")\nInput tokens:\n";
      for (size_t p = 0; p < step.forward.prefix.size(); ++p) {
        const int id = step.forward.prefix[p];
        overview << "  Position " << p + 1
                 << (p < example.prompt.size() ? " [prompt] " : " [generated] ")
                 << Quote(metadata.token_text[id]) << " compact=" << id
                 << " original=" << metadata.original_ids[id] << '\n';
      }
      overview << "Top next-token candidates (all " << logits.size()
               << " logical tokens in denominator):\n";
      for (size_t i = 0; i < count; ++i) {
        const int id = order[i];
        overview << "  " << id << ' ' << Quote(metadata.token_text[id])
                 << " logit=" << logits[id] << " probability="
                 << 100 * std::exp(logits[id] - maximum) / denominator << "%\n";
      }
      text << '\n' << overview.str();
      walkthrough << '\n' << overview.str();
      WriteQueryWalkthrough(walkthrough, step.forward, metadata);
      html << "<h3>Step " << step_index + 1 << ": "
           << Html(Quote(metadata.token_text[step.predicted])) << "</h3><pre>"
           << Html(overview.str()) << "</pre>";
      completions << example.corpus_line << '\t' << step_index + 1 << '\t'
                  << step.forward.prefix.size() + 1 << '\t' << step.predicted
                  << '\t' << metadata.original_ids[step.predicted] << '\t'
                  << Quote(metadata.token_text[step.predicted]) << '\t'
                  << step.expected << '\t' << (step.predicted == step.expected)
                  << '\t' << probability << '\t' << margin << '\n';
      auto dump = [&](const TensorSnapshot& tensor, bool attention) {
        const auto contents =
            DumpTensor(tensor, step.forward, metadata, attention);
        text << '\n' << contents;
        html << "<details><summary>"
             << (attention ? "Attention probabilities: " : "")
             << Html(Key(tensor)) << "</summary><pre>" << Html(contents)
             << "</pre></details>";
      };
      html << "<details><summary>All layer activations (execution "
              "order)</summary>";
      for (const auto& tensor : step.forward.activations)
        dump(tensor, false);
      html << "</details><details><summary>All causal attention "
              "matrices</summary>";
      for (const auto& tensor : step.forward.attention)
        dump(tensor, true);
      html << "</details>";
    }
  }
  html << "</body></html>\n";
  WriteStatistics(files[3], files[4], files[5], files[6], metadata, examples);
  for (auto& file : files) {
    file.close();
    if (!file)
      return absl::DataLossError("completion report write failed");
  }
  return absl::OkStatus();
}

}  // namespace pluto::llm::completion_trace
