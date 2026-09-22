#pragma once

#include <string>

#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_certificate.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_model.h"

namespace pluto::llm::discretized::generator {

// Human-readable reports are presentation only. Generation, reduction, and
// verification exchange typed data and never parse these strings back.
std::string FormatProgress(const ProgressEvent& event);
std::string FormatVerification(const VerificationResult& result);
std::string FormatStatistics(const ModelStatistics& stats);
std::string FormatCertificate(const CertificateResult& certificate);

}  // namespace pluto::llm::discretized::generator
