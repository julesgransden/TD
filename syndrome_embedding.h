#ifndef SYNDROME_EMBEDDING_H
#define SYNDROME_EMBEDDING_H

#include <torch/torch.h>
#include "config.h"

/**
 * @file syndrome_embedding.h
 * @brief Syndrome Embedding Layer for QEC Transformer
 *
 * This module converts binary syndrome vectors into dense embeddings
 * with positional encoding for the Transformer encoder.
 *
 * Input: Binary syndrome vector [batch_size, NUM_CHECKS] with values {0, 1}
 * Output: Dense embeddings [batch_size, NUM_CHECKS, D_MODEL]
 */

namespace qec {

/**
 * @class SyndromeEmbeddingImpl
 * @brief Embeds binary syndrome values with positional encoding
 *
 * For each syndrome value (binary 0 or 1):
 * - Linear embedding: syndrome_value → D_MODEL dimensions
 * - Positional encoding: learned embedding for each of NUM_CHECKS positions
 *
 * The final embedding is the sum of syndrome embedding and positional embedding.
 */
class SyndromeEmbeddingImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new Syndrome Embedding module
     * @param num_positions Number of syndrome positions (NUM_CHECKS)
     * @param d_model Embedding dimension
     * @param dropout Dropout rate
     */
    SyndromeEmbeddingImpl(int num_positions = NUM_CHECKS,
                          int d_model = D_MODEL,
                          float dropout = DROPOUT);

    /**
     * @brief Forward pass: embed syndromes with positional encoding
     * @param syndromes Input tensor [batch_size, num_positions] of binary values
     * @return Embedded tensor [batch_size, num_positions, d_model]
     */
    torch::Tensor forward(torch::Tensor syndromes);

    /**
     * @brief Initialize weights using Xavier uniform initialization
     */
    void init_weights();

private:
    int num_positions_;
    int d_model_;

    // Linear layer to embed each binary syndrome value → d_model dimensions
    // Input: 1 (single syndrome value), Output: D_MODEL
    torch::nn::Linear syndrome_embed_{nullptr};

    // Learned positional embeddings for each syndrome position
    // NUM_CHECKS embeddings of size D_MODEL
    torch::nn::Embedding position_embed_{nullptr};

    // Dropout for regularization
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(SyndromeEmbedding);

} // namespace qec

#endif // SYNDROME_EMBEDDING_H
