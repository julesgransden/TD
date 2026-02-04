#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include "config.h"
#include "syndrome_embedding.h"
#include "encoder.h"
#include "decoder.h"

/**
 * @file model.h
 * @brief QEC Transformer Model - Main Architecture
 *
 * This is the main Transformer-based quantum error correction decoder
 * for a [[72,12,6]] QLDPC code. It learns to predict logical errors
 * from syndrome measurements.
 *
 * Architecture:
 * 1. SyndromeEmbedding: Binary syndromes → Dense embeddings with position
 * 2. TransformerEncoder: Bidirectional self-attention on syndromes
 * 3. Logical Queries: Learned embeddings for each logical qubit
 * 4. TransformerDecoder: Cross-attention from queries to syndromes
 * 5. OutputHead: Project to binary logits
 *
 * Input: [batch_size, NUM_CHECKS] binary syndrome vector
 * Output: [batch_size, NUM_LOGICAL] logits for each logical qubit
 */

namespace qec {

/**
 * @class OutputHeadImpl
 * @brief Final projection layer for binary classification
 *
 * Projects the decoder output to scalar logits for each logical qubit.
 * Input: [batch_size, NUM_LOGICAL, D_MODEL]
 * Output: [batch_size, NUM_LOGICAL]
 */
class OutputHeadImpl : public torch::nn::Module {
public:
    OutputHeadImpl(int d_model = D_MODEL);

    torch::Tensor forward(torch::Tensor x);

private:
    // Linear projection: D_MODEL → 1 (one logit per logical qubit)
    torch::nn::Linear output_projection_{nullptr};
};

TORCH_MODULE(OutputHead);

/**
 * @class QECTransformerImpl
 * @brief Main QEC Transformer Decoder Model
 *
 * Complete encoder-decoder transformer architecture for learning
 * syndrome → logical error mappings.
 */
class QECTransformerImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct the QEC Transformer model
     * @param num_checks Number of syndrome checks (input dimension)
     * @param num_logical Number of logical qubits (output dimension)
     * @param d_model Embedding dimension
     * @param nhead Number of attention heads
     * @param num_encoder_layers Number of encoder layers
     * @param num_decoder_layers Number of decoder layers
     * @param dim_feedforward FFN hidden dimension
     * @param dropout Dropout rate
     */
    QECTransformerImpl(int num_checks = NUM_CHECKS,
                       int num_logical = NUM_LOGICAL,
                       int d_model = D_MODEL,
                       int nhead = NHEAD,
                       int num_encoder_layers = NUM_ENCODER_LAYERS,
                       int num_decoder_layers = NUM_DECODER_LAYERS,
                       int dim_feedforward = DIM_FEEDFORWARD,
                       float dropout = DROPOUT);

    /**
     * @brief Forward pass through the complete model
     * @param syndromes Input tensor [batch_size, num_checks] of binary values {0, 1}
     * @return Logits tensor [batch_size, num_logical] for binary classification
     */
    torch::Tensor forward(torch::Tensor syndromes);

    /**
     * @brief Get the number of trainable parameters
     */
    int64_t num_parameters() const;

    /**
     * @brief Print model architecture summary
     */
    void print_summary() const;

private:
    int num_checks_;
    int num_logical_;
    int d_model_;

    // =========================================
    // Model Components
    // =========================================

    // 1. Syndrome Embedding: Binary syndromes → Dense embeddings
    SyndromeEmbedding syndrome_embedding_{nullptr};

    // 2. Transformer Encoder: Bidirectional self-attention on syndromes
    TransformerEncoder encoder_{nullptr};

    // 3. Logical Qubit Queries: Learned embeddings for each logical qubit
    // Shape: [NUM_LOGICAL, D_MODEL]
    // These are learnable parameters that represent each logical qubit
    torch::Tensor logical_queries_;

    // 4. Transformer Decoder: Cross-attention from queries to syndromes
    TransformerDecoder decoder_{nullptr};

    // 5. Output Head: Project to binary logits
    OutputHead output_head_{nullptr};
};

TORCH_MODULE(QECTransformer);

} // namespace qec

#endif // MODEL_H
