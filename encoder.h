#ifndef ENCODER_H
#define ENCODER_H

#include <torch/torch.h>
#include "config.h"

/**
 * @file encoder.h
 * @brief Transformer Encoder for QEC Decoder
 *
 * The encoder processes the embedded syndrome sequence using
 * bidirectional self-attention, allowing each syndrome to attend
 * to all other syndromes.
 *
 * Input: [batch_size, NUM_CHECKS, D_MODEL]
 * Output: [batch_size, NUM_CHECKS, D_MODEL]
 */

namespace qec {

/**
 * @class FeedForwardImpl
 * @brief Position-wise Feed-Forward Network
 *
 * Two-layer MLP with ReLU activation:
 * d_model → dim_feedforward → d_model
 */
 //inherit the Module class from Torch to allow for automatic parameter registration and optimization
class FeedForwardImpl : public torch::nn::Module {
public:
    FeedForwardImpl(int d_model = D_MODEL,
                    int dim_feedforward = DIM_FEEDFORWARD,
                    float dropout = DROPOUT);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear linear1_{nullptr};
    torch::nn::Linear linear2_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(FeedForward);

/**
 * @class TransformerEncoderLayerImpl
 * @brief Single Transformer Encoder Layer
 *
 * Each layer contains:
 * 1. Multi-head self-attention (syndromes attend to all syndromes)
 * 2. Add & LayerNorm
 * 3. Feed-forward network (d_model → dim_feedforward → d_model)
 * 4. Add & LayerNorm
 */
class TransformerEncoderLayerImpl : public torch::nn::Module {
public:
    TransformerEncoderLayerImpl(int d_model = D_MODEL,
                                 int nhead = NHEAD,
                                 int dim_feedforward = DIM_FEEDFORWARD,
                                 float dropout = DROPOUT);

    /**
     * @brief Forward pass through encoder layer
     * @param x Input tensor [batch_size, seq_len, d_model]
     * @param src_mask Optional attention mask
     * @return Output tensor [batch_size, seq_len, d_model]
     */
    torch::Tensor forward(torch::Tensor x,
                          torch::Tensor src_mask = {});

    void init_weights();

private:
    // Multi-head self-attention
    torch::nn::MultiheadAttention self_attn_{nullptr};

    // Feed-forward network
    FeedForward ffn_{nullptr};

    // Layer normalization
    torch::nn::LayerNorm norm1_{nullptr};
    torch::nn::LayerNorm norm2_{nullptr};

    // Dropout
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(TransformerEncoderLayer);

/**
 * @class TransformerEncoderImpl
 * @brief Stack of Transformer Encoder Layers
 *
 * Stacks NUM_ENCODER_LAYERS encoder layers to process
 * the syndrome sequence with deep bidirectional attention.
 */
class TransformerEncoderImpl : public torch::nn::Module {
public:
    TransformerEncoderImpl(int num_layers = NUM_ENCODER_LAYERS,
                           int d_model = D_MODEL,
                           int nhead = NHEAD,
                           int dim_feedforward = DIM_FEEDFORWARD,
                           float dropout = DROPOUT);

    /**
     * @brief Forward pass through all encoder layers
     * @param x Input tensor [batch_size, seq_len, d_model]
     * @param src_mask Optional attention mask
     * @return Output tensor [batch_size, seq_len, d_model]
     */
    torch::Tensor forward(torch::Tensor x,
                          torch::Tensor src_mask = {});

private:
    int num_layers_;
    torch::nn::ModuleList layers_{nullptr};
};

TORCH_MODULE(TransformerEncoder);

} // namespace qec

#endif // ENCODER_H
