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
    // Input: syndromes [batch_size, num_positions] - binary values {0, 1}

    auto batch_size = syndromes.size(0);
    auto seq_len = syndromes.size(1);

    // Debug: Print input shape
    std::cout << "  [SyndromeEmbedding] Input shape: [" << batch_size << ", " << seq_len << "]" << std::endl;

    // Reshape syndromes for linear layer: [batch_size, num_positions] → [batch_size, num_positions, 1]
    // This allows the linear layer to process each syndrome value independently
    auto x = syndromes.unsqueeze(-1).to(torch::kFloat);
    // Shape: [batch_size, num_positions, 1]

    // Apply syndrome embedding linear layer
    // [batch_size, num_positions, 1] → [batch_size, num_positions, d_model]
    x = syndrome_embed_->forward(x);
    std::cout << "  [SyndromeEmbedding] After syndrome_embed: [" << x.size(0) << ", "
              << x.size(1) << ", " << x.size(2) << "]" << std::endl;

    // Create position indices: [0, 1, 2, ..., num_positions-1]
    auto positions = torch::arange(seq_len, syndromes.options().dtype(torch::kLong));
    // Shape: [num_positions]

    // Get positional embeddings and add to syndrome embeddings
    // position_embed: [num_positions] → [num_positions, d_model]
    auto pos_emb = position_embed_->forward(positions);
    // Shape: [num_positions, d_model]

    // Add positional embeddings (broadcasts over batch dimension)
    // [batch_size, num_positions, d_model] + [num_positions, d_model]
    x = x + pos_emb;
    std::cout << "  [SyndromeEmbedding] After position_embed: [" << x.size(0) << ", "
              << x.size(1) << ", " << x.size(2) << "]" << std::endl;

    // Apply dropout
    x = dropout_->forward(x);

    // Output: [batch_size, num_positions, d_model]
    return x;
}

} // namespace qec
