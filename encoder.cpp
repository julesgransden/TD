#include "encoder.h"
#include <iostream>

namespace qec {

// ============================================================================
// FeedForward Implementation
// ============================================================================

FeedForwardImpl::FeedForwardImpl(int d_model, int dim_feedforward, float dropout) {
    // First linear layer: d_model → dim_feedforward
    linear1_ = register_module(
        "linear1",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, dim_feedforward))
    );

    // Second linear layer: dim_feedforward → d_model
    linear2_ = register_module(
        "linear2",
        torch::nn::Linear(torch::nn::LinearOptions(dim_feedforward, d_model))
    );

    // Dropout
    dropout_ = register_module(
        "dropout",
        torch::nn::Dropout(torch::nn::DropoutOptions(dropout))
    );

    // Initialize weights
    torch::nn::init::xavier_uniform_(linear1_->weight);
    torch::nn::init::zeros_(linear1_->bias);
    torch::nn::init::xavier_uniform_(linear2_->weight);
    torch::nn::init::zeros_(linear2_->bias);
}

torch::Tensor FeedForwardImpl::forward(torch::Tensor x) {
    // FFN: Linear → ReLU → Dropout → Linear
    x = linear1_->forward(x);
    x = torch::relu(x);
    x = dropout_->forward(x);
    x = linear2_->forward(x);
    return x;
}

// ============================================================================
// TransformerEncoderLayer Implementation
// ============================================================================

TransformerEncoderLayerImpl::TransformerEncoderLayerImpl(
    int d_model, int nhead, int dim_feedforward, float dropout) {

    // Multi-head self-attention
    // LibTorch expects (seq_len, batch, embed) format by default
    // We'll transpose in forward() to handle (batch, seq_len, embed) input
    self_attn_ = register_module(
        "self_attn",
        torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(d_model, nhead)
                .dropout(dropout)
        )
    );

    // Feed-forward network
    ffn_ = register_module(
        "ffn", 
        FeedForward(d_model, dim_feedforward, dropout));

    // Layer normalization
    norm1_ = register_module(
        "norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}))
    );
    norm2_ = register_module(
        "norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}))
    );

    // Dropout for residual connections
    dropout_ = register_module(
        "dropout",
        torch::nn::Dropout(torch::nn::DropoutOptions(dropout))
    );

    init_weights();
}

void TransformerEncoderLayerImpl::init_weights() {
    // LayerNorm is already initialized with ones and zeros by default
    // MultiheadAttention uses Xavier initialization by default
}

torch::Tensor TransformerEncoderLayerImpl::forward(
    torch::Tensor x, torch::Tensor src_mask) {

    // Input: x [batch_size, seq_len, d_model]

    // =========================================
    // 1. Self-Attention with Residual Connection
    // =========================================
    // Transpose to (seq_len, batch, d_model) for MultiheadAttention
    auto x_t = x.transpose(0, 1);  // [seq_len, batch, d_model]

    // Query, Key, Value are all the same (self-attention)
    auto attn_output = std::get<0>(self_attn_->forward(x_t, x_t, x_t, {}, false, src_mask));
    // attn_output: [seq_len, batch, d_model]

    // Transpose back to (batch, seq_len, d_model)
    attn_output = attn_output.transpose(0, 1);

    // Add & Norm (Pre-LN variant: x + dropout(attn))
    x = norm1_->forward(x + dropout_->forward(attn_output));

    // =========================================
    // 2. Feed-Forward with Residual Connection
    // =========================================
    auto ffn_output = ffn_->forward(x);
    // ffn_output: [batch_size, seq_len, d_model]

    // Add & Norm
    x = norm2_->forward(x + dropout_->forward(ffn_output));

    // Output: [batch_size, seq_len, d_model]
    return x;
}

// ============================================================================
// TransformerEncoder Implementation
// ============================================================================

TransformerEncoderImpl::TransformerEncoderImpl(
    int num_layers, int d_model, int nhead, int dim_feedforward, float dropout)
    : num_layers_(num_layers) {

    // Create stack of encoder layers
    layers_ = register_module("layers", torch::nn::ModuleList());

    for (int i = 0; i < num_layers; ++i) {
        layers_->push_back(
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        );
    }
}

torch::Tensor TransformerEncoderImpl::forward(
    torch::Tensor x, torch::Tensor src_mask) {

    for (int i = 0; i < num_layers_; ++i) {
        auto layer = layers_->ptr<TransformerEncoderLayerImpl>(i);
        x = layer->forward(x, src_mask);
    }

    return x;
}

} // namespace qec
