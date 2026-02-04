#include "model.h"
#include <iostream>
#include <iomanip>

namespace qec {

// ============================================================================
// OutputHead Implementation
// ============================================================================

OutputHeadImpl::OutputHeadImpl(int d_model) {
    // Linear projection from d_model to 1 (scalar logit per logical qubit)
    output_projection_ = register_module(
        "output_projection",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, 1))
    );

    // Initialize weights
    torch::nn::init::xavier_uniform_(output_projection_->weight);
    torch::nn::init::zeros_(output_projection_->bias);
}

torch::Tensor OutputHeadImpl::forward(torch::Tensor x) {
    // Input: [batch_size, num_logical, d_model]

    // Project each logical qubit representation to a scalar logit
    // [batch_size, num_logical, d_model] → [batch_size, num_logical, 1]
    x = output_projection_->forward(x);

    // Squeeze the last dimension
    // [batch_size, num_logical, 1] → [batch_size, num_logical]
    x = x.squeeze(-1);

    return x;
}

// ============================================================================
// QECTransformer Implementation
// ============================================================================

QECTransformerImpl::QECTransformerImpl(
    int num_checks,
    int num_logical,
    int d_model,
    int nhead,
    int num_encoder_layers,
    int num_decoder_layers,
    int dim_feedforward,
    float dropout)
    : num_checks_(num_checks),
      num_logical_(num_logical),
      d_model_(d_model) {

    syndrome_embedding_ = register_module(
        "syndrome_embedding",
        SyndromeEmbedding(num_checks, d_model, dropout)
    );

    encoder_ = register_module(
        "encoder",
        TransformerEncoder(num_encoder_layers, d_model, nhead, dim_feedforward, dropout)
    );

    logical_queries_ = register_parameter(
        "logical_queries",
        torch::randn({num_logical, d_model}) * 0.02
    );

    decoder_ = register_module(
        "decoder",
        TransformerDecoder(num_decoder_layers, d_model, nhead, dim_feedforward, dropout)
    );

    output_head_ = register_module(
        "output_head",
        OutputHead(d_model)
    );
}

torch::Tensor QECTransformerImpl::forward(torch::Tensor syndromes) {
    TORCH_CHECK(syndromes.dim() == 2,
        "Expected 2D input tensor, got ", syndromes.dim(), "D");
    TORCH_CHECK(syndromes.size(1) == num_checks_,
        "Expected ", num_checks_, " syndrome values, got ", syndromes.size(1));

    auto batch_size = syndromes.size(0);

    // 1. Embedding
    auto embedded = syndrome_embedding_->forward(syndromes);

    // 2. Encoder
    auto encoded = encoder_->forward(embedded);

    // 3. Prepare queries
    auto queries = logical_queries_.unsqueeze(0).expand({batch_size, -1, -1}).clone();

    // 4. Decoder
    auto decoded = decoder_->forward(queries, encoded);

    // 5. Output
    return output_head_->forward(decoded);
}

int64_t QECTransformerImpl::num_parameters() const {
    int64_t total = 0;
    for (const auto& param : parameters()) {
        total += param.numel();
    }
    return total;
}

void QECTransformerImpl::print_summary() const {
    std::cout << "QEC Transformer: " << num_checks_ << " -> " << num_logical_
              << " | d=" << d_model_ << " heads=" << NHEAD
              << " enc=" << NUM_ENCODER_LAYERS << " dec=" << NUM_DECODER_LAYERS
              << " ffn=" << DIM_FEEDFORWARD
              << " | " << std::fixed << std::setprecision(2) << (num_parameters() / 1e6) << "M params"
              << std::endl;
}

} // namespace qec
