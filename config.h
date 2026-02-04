#ifndef CONFIG_H
#define CONFIG_H

/**
 * @file config.h
 * @brief Hyperparameters for the QEC Transformer Decoder
 *
 * This configuration file defines all hyperparameters for the
 * Transformer-based quantum error correction decoder for a [[72,12,6]] QLDPC code.
 */

namespace qec {

// ============================================================================
// Model Architecture Hyperparameters
// ============================================================================

// Input/Output dimensions
constexpr int NUM_CHECKS = 432;        // Number of syndrome values (input dimension)
constexpr int NUM_LOGICAL = 12;        // Number of logical qubits (output dimension)

// Transformer dimensions
constexpr int D_MODEL = 256;           // Embedding dimension
constexpr int NHEAD = 8;               // Number of attention heads
constexpr int NUM_ENCODER_LAYERS = 6;  // Number of encoder layers
constexpr int NUM_DECODER_LAYERS = 6;  // Number of decoder layers
constexpr int DIM_FEEDFORWARD = 1024;  // FFN hidden dimension (typically 4 × d_model)

// Regularization
constexpr float DROPOUT = 0.1f;        // Dropout rate

// ============================================================================
// Training Hyperparameters
// ============================================================================

constexpr float LEARNING_RATE = 1e-4f; // Adam optimizer learning rate
constexpr int BATCH_SIZE = 256;         // Training batch size
constexpr int NUM_EPOCHS = 1;        // Number of training epochs
constexpr int LOG_INTERVAL = 100;       // Print loss every N batches

// ============================================================================
// Derived Constants (do not modify)
// ============================================================================

constexpr int HEAD_DIM = D_MODEL / NHEAD;  // Dimension per attention head

// Static assertion to ensure d_model is divisible by nhead
static_assert(D_MODEL % NHEAD == 0, "D_MODEL must be divisible by NHEAD");

} // namespace qec

#endif // CONFIG_H
