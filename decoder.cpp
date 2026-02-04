#include "decoder.h"
#include <iostream>

namespace qec {

// ============================================================================
// TransformerDecoderLayer Implementation
// ============================================================================

TransformerDecoderLayerImpl::TransformerDecoderLayerImpl(
    int d_model, int nhead, int dim_feedforward, float dropout) {

    // =========================================
    // Self-Attention: Logical qubits attend to each other
    // LibTorch expects (seq_len, batch, embed) format by default
    // We'll transpose in forward() to handle (batch, seq_len, embed) input
    // =========================================
    self_attn_ = register_module(
        "self_attn",
        torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(d_model, nhead)
                .dropout(dropout)
        )
    );

    // =========================================
    // Cross-Attention: Logical qubits attend to syndromes
    // THIS IS THE KEY COMPONENT that allows the decoder to
    // learn syndrome → logical error mappings!
    // =========================================
    cross_attn_ = register_module(
        "cross_attn",
        torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(d_model, nhead)
                .dropout(dropout)
        )
    );

    // Feed-forward network
    ffn_ = register_module("ffn", FeedForward(d_model, dim_feedforward, dropout));

    // Layer normalization (three for decoder: self-attn, cross-attn, ffn)
    norm1_ = register_module(
        "norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}))
    );
    norm2_ = register_module(
        "norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}))
    );
    norm3_ = register_module(
        "norm3",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}))
    );

    // Dropout for residual connections
    dropout_ = register_module(
        "dropout",
        torch::nn::Dropout(torch::nn::DropoutOptions(dropout))
    );

    init_weights();
}

void TransformerDecoderLayerImpl::init_weights() {
    // LayerNorm and MultiheadAttention use sensible defaults
}

torch::Tensor TransformerDecoderLayerImpl::forward(
    torch::Tensor tgt,
    torch::Tensor memory,
    torch::Tensor tgt_mask,
    torch::Tensor memory_mask) {

    // Input:
    //   tgt: [batch_size, tgt_len, d_model] - logical qubit queries
    //   memory: [batch_size, src_len, d_model] - encoded syndromes

    // =========================================
    // 1. Self-Attention: Logical qubits attend to each other
    // =========================================
    // Transpose to (seq_len, batch, d_model) for MultiheadAttention
    auto tgt_t = tgt.transpose(0, 1);  // [tgt_len, batch, d_model]

    // This allows the model to learn dependencies between logical qubits
    auto self_out = std::get<0>(self_attn_->forward(tgt_t, tgt_t, tgt_t, {}, false, tgt_mask));
    // self_out: [tgt_len, batch, d_model]

    // Transpose back to (batch, tgt_len, d_model)
    self_out = self_out.transpose(0, 1);

    // Add & Norm
    tgt = norm1_->forward(tgt + dropout_->forward(self_out));

    // =========================================
    // 2. Cross-Attention: Logical qubits attend to syndromes
    // KEY COMPONENT: This is where the decoder learns to map
    // syndrome patterns to logical errors!
    // - Query: logical qubit representations (what we want to predict)
    // - Key/Value: encoded syndromes (what we condition on)
    // =========================================
    // Transpose for attention
    tgt_t = tgt.transpose(0, 1);  // [tgt_len, batch, d_model]
    auto memory_t = memory.transpose(0, 1);  // [src_len, batch, d_model]

    auto cross_out = std::get<0>(cross_attn_->forward(
        tgt_t,      // Query: logical qubits ask "which syndromes are relevant to me?"
        memory_t,   // Key: syndromes provide the context
        memory_t,   // Value: syndromes provide the information
        {},         // key_padding_mask
        false,      // need_weights
        memory_mask
    ));
    // cross_out: [tgt_len, batch, d_model]

    // Transpose back
    cross_out = cross_out.transpose(0, 1);

    // Add & Norm
    tgt = norm2_->forward(tgt + dropout_->forward(cross_out));

    // =========================================
    // 3. Feed-Forward Network
    // =========================================
    auto ffn_out = ffn_->forward(tgt);
    // ffn_out: [batch_size, tgt_len, d_model]

    // Add & Norm
    tgt = norm3_->forward(tgt + dropout_->forward(ffn_out));

    // Output: [batch_size, tgt_len, d_model]
    return tgt;
}

// ============================================================================
// TransformerDecoder Implementation
// ============================================================================

TransformerDecoderImpl::TransformerDecoderImpl(
    int num_layers, int d_model, int nhead, int dim_feedforward, float dropout)
    : num_layers_(num_layers) {

    // Create stack of decoder layers
    layers_ = register_module("layers", torch::nn::ModuleList());

    for (int i = 0; i < num_layers; ++i) {
        layers_->push_back(
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        );
    }
}

torch::Tensor TransformerDecoderImpl::forward(
    torch::Tensor tgt,
    torch::Tensor memory,
    torch::Tensor tgt_mask,
    torch::Tensor memory_mask) {

    // Input:
    //   tgt: [batch_size, tgt_len, d_model] - logical qubit queries
    //   memory: [batch_size, src_len, d_model] - encoded syndromes

    std::cout << "  [Decoder] Input tgt shape: [" << tgt.size(0) << ", "
              << tgt.size(1) << ", " << tgt.size(2) << "]" << std::endl;
    std::cout << "  [Decoder] Input memory shape: [" << memory.size(0) << ", "
              << memory.size(1) << ", " << memory.size(2) << "]" << std::endl;

    // Pass through each decoder layer
    for (int i = 0; i < num_layers_; ++i) {
        auto layer = layers_->ptr<TransformerDecoderLayerImpl>(i);
        tgt = layer->forward(tgt, memory, tgt_mask, memory_mask);
    }

    std::cout << "  [Decoder] Output shape: [" << tgt.size(0) << ", "
              << tgt.size(1) << ", " << tgt.size(2) << "]" << std::endl;

    // Output: [batch_size, tgt_len, d_model]
    return tgt;
}

} // namespace qec
