#ifndef DECODER_H
#define DECODER_H

#include <torch/torch.h>
#include "config.h"
#include "encoder.h"  // For FeedForward

/**
 * @file decoder.h
 * @brief Transformer Decoder for QEC Decoder
 *
 * The decoder uses learned logical qubit queries to attend to the
 * encoded syndrome sequence via cross-attention, learning to predict
 * which logical qubits have errors.
 *
 * Input tgt: [batch_size, NUM_LOGICAL, D_MODEL] (logical qubit queries)
 * Input memory: [batch_size, NUM_CHECKS, D_MODEL] (encoded syndromes)
 * Output: [batch_size, NUM_LOGICAL, D_MODEL]
 */

namespace qec {

/**
 * @class TransformerDecoderLayerImpl
 * @brief Single Transformer Decoder Layer
 *
 * Each layer contains:
 * 1. Multi-head self-attention (logical qubits attend to each other)
 * 2. Add & LayerNorm
 * 3. Multi-head cross-attention (logical qubits attend to syndromes) - KEY!
 * 4. Add & LayerNorm
 * 5. Feed-forward network
 * 6. Add & LayerNorm
 */
class TransformerDecoderLayerImpl : public torch::nn::Module {
public:
    TransformerDecoderLayerImpl(int d_model = D_MODEL,
                                 int nhead = NHEAD,
                                 int dim_feedforward = DIM_FEEDFORWARD,
                                 float dropout = DROPOUT);

    /**
     * @brief Forward pass through decoder layer
     * @param tgt Target tensor (logical queries) [batch_size, tgt_len, d_model]
     * @param memory Encoder output (syndromes) [batch_size, src_len, d_model]
     * @param tgt_mask Optional self-attention mask for target
     * @param memory_mask Optional cross-attention mask for memory
     * @return Output tensor [batch_size, tgt_len, d_model]
     */
    torch::Tensor forward(torch::Tensor tgt,
                          torch::Tensor memory,
                          torch::Tensor tgt_mask = {},
                          torch::Tensor memory_mask = {});

    void init_weights();

private:
    // Self-attention: logical qubits attend to each other
    torch::nn::MultiheadAttention self_attn_{nullptr};

    // Cross-attention: logical qubits attend to encoded syndromes (KEY COMPONENT!)
    torch::nn::MultiheadAttention cross_attn_{nullptr};

    // Feed-forward network
    FeedForward ffn_{nullptr};

    // Layer normalization (three norms: after self-attn, cross-attn, ffn)
    torch::nn::LayerNorm norm1_{nullptr};
    torch::nn::LayerNorm norm2_{nullptr};
    torch::nn::LayerNorm norm3_{nullptr};

    // Dropout
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(TransformerDecoderLayer);

/**
 * @class TransformerDecoderImpl
 * @brief Stack of Transformer Decoder Layers
 *
 * Stacks NUM_DECODER_LAYERS decoder layers to process
 * the logical qubit queries with self-attention and
 * cross-attention to the encoded syndromes.
 */
class TransformerDecoderImpl : public torch::nn::Module {
public:
    TransformerDecoderImpl(int num_layers = NUM_DECODER_LAYERS,
                           int d_model = D_MODEL,
                           int nhead = NHEAD,
                           int dim_feedforward = DIM_FEEDFORWARD,
                           float dropout = DROPOUT);

    /**
     * @brief Forward pass through all decoder layers
     * @param tgt Target tensor (logical queries) [batch_size, tgt_len, d_model]
     * @param memory Encoder output (syndromes) [batch_size, src_len, d_model]
     * @param tgt_mask Optional self-attention mask
     * @param memory_mask Optional cross-attention mask
     * @return Output tensor [batch_size, tgt_len, d_model]
     */
    torch::Tensor forward(torch::Tensor tgt,
                          torch::Tensor memory,
                          torch::Tensor tgt_mask = {},
                          torch::Tensor memory_mask = {});

private:
    int num_layers_;
    torch::nn::ModuleList layers_{nullptr};
};

TORCH_MODULE(TransformerDecoder);

} // namespace qec

#endif // DECODER_H
