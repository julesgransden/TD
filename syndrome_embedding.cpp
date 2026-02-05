#include "syndrome_embedding.h"
#include <iostream>

namespace qec {

SyndromeEmbeddingImpl::SyndromeEmbeddingImpl(int num_positions, int d_model, float dropout)
    : num_positions_(num_positions), d_model_(d_model) {

    // Linear layer: maps each binary syndrome value (0 or 1) to d_model dimensions
    // Input size: 1 (single scalar syndrome value)
    // Output size: d_model
    syndrome_embed_ = register_module(
        "syndrome_embed",
        torch::nn::Linear(torch::nn::LinearOptions(1, d_model))
    );

    // Learned positional embeddings: one embedding per syndrome position
    // num_embeddings: num_positions (one for each syndrome check)
    // embedding_dim: d_model
    position_embed_ = register_module(
        "position_embed",
        torch::nn::Embedding(torch::nn::EmbeddingOptions(num_positions, d_model))
    );

    // Dropout for regularization
    dropout_ = register_module(
        "dropout",
        torch::nn::Dropout(torch::nn::DropoutOptions(dropout))
    );

    // Initialize weights
    init_weights();
}

void SyndromeEmbeddingImpl::init_weights() {
    // Xavier uniform initialization for linear layer
    torch::nn::init::xavier_uniform_(syndrome_embed_->weight);
    torch::nn::init::zeros_(syndrome_embed_->bias);

    // Normal initialization for positional embeddings (std = 0.02)
    torch::nn::init::normal_(position_embed_->weight, 0.0, 0.02);
}

torch::Tensor SyndromeEmbeddingImpl::forward(torch::Tensor syndromes) {
    auto seq_len = syndromes.size(1);
    auto input_dtype = syndromes.dtype();

    // Reshape and embed syndromes
    auto x = syndromes.unsqueeze(-1);
    x = syndrome_embed_->forward(x);

    // Add positional embeddings (cast to match input dtype for float16 support)
    auto positions = torch::arange(seq_len, torch::TensorOptions().dtype(torch::kLong).device(syndromes.device()));
    auto pos_emb = position_embed_->forward(positions).to(input_dtype);
    x = x + pos_emb;

    // Apply dropout
    return dropout_->forward(x);
}

} // namespace qec
