#ifndef TRAIN_H
#define TRAIN_H

#include <torch/torch.h>
#include "model.h"
#include "config.h"
#include "data_loader.h"

/**
 * @file train.h
 * @brief Training utilities for QEC Transformer
 *
 * Provides training loop, data loading, and evaluation utilities
 * for the quantum error correction transformer decoder.
 */

namespace qec {

/**
 * @class Trainer
 * @brief Training manager for QEC Transformer
 *
 * Handles the training loop, loss computation, and optimization.
 */
class Trainer {
public:
    /**
     * @brief Construct a trainer for the model
     * @param model The QEC Transformer model to train
     * @param learning_rate Learning rate for Adam optimizer
     */
    Trainer(QECTransformer model, float learning_rate = LEARNING_RATE);

    /**
     * @brief Perform one training step
     * @param syndromes Input syndromes [batch_size, num_checks]
     * @param targets Target logical errors [batch_size, num_logical]
     * @return Training loss for this batch
     */
    float train_step(torch::Tensor syndromes, torch::Tensor targets);

    /**
     * @brief Evaluate model on a batch
     * @param syndromes Input syndromes [batch_size, num_checks]
     * @param targets Target logical errors [batch_size, num_logical]
     * @return Evaluation loss and accuracy
     */
    std::pair<float, float> eval_step(torch::Tensor syndromes, torch::Tensor targets);

    /**
     * @brief Generate dummy training data for testing (fallback when no HDF5)
     * @param batch_size Number of samples
     * @param num_checks Number of syndrome checks
     * @param num_logical Number of logical qubits
     * @return Pair of (syndromes, logical_errors) tensors
     */
    static std::pair<torch::Tensor, torch::Tensor> generate_dummy_data(
        int batch_size,
        int num_checks = NUM_CHECKS,
        int num_logical = NUM_LOGICAL);

    /**
     * @brief Run training loop with HDF5 data
     * @param data_loader Data loader with real training data
     * @param num_epochs Number of training epochs
     * @param log_interval Print loss every N batches
     * @param val_loader Optional validation data loader
     */
    void train_loop(DataLoader& data_loader,
                    int num_epochs = NUM_EPOCHS,
                    int log_interval = LOG_INTERVAL,
                    DataLoader* val_loader = nullptr);

    /**
     * @brief Run training loop with dummy data (for testing)
     * @param num_epochs Number of training epochs
     * @param batch_size Batch size
     * @param batches_per_epoch Number of batches per epoch
     * @param log_interval Print loss every N batches
     */
    void train_loop_dummy(int num_epochs = NUM_EPOCHS,
                          int batch_size = BATCH_SIZE,
                          int batches_per_epoch = 100,
                          int log_interval = LOG_INTERVAL);

    /**
     * @brief Evaluate model on entire validation set
     * @param data_loader Validation data loader
     * @return Pair of (average_loss, average_accuracy)
     */
    std::pair<float, float> evaluate(DataLoader& data_loader);

    /**
     * @brief Save model checkpoint
     * @param path Path to save the model
     */
    void save_checkpoint(const std::string& path);

    /**
     * @brief Load model checkpoint
     * @param path Path to load the model from
     */
    void load_checkpoint(const std::string& path);

private:
    QECTransformer model_;
    torch::optim::Adam optimizer_;
    torch::nn::BCEWithLogitsLoss criterion_;
};

/**
 * @brief Compute accuracy for binary predictions
 * @param logits Model output logits [batch_size, num_logical]
 * @param targets Ground truth [batch_size, num_logical]
 * @return Accuracy (fraction of correct predictions)
 */
float compute_accuracy(torch::Tensor logits, torch::Tensor targets);

/**
 * @brief Compute per-qubit accuracy
 * @param logits Model output logits [batch_size, num_logical]
 * @param targets Ground truth [batch_size, num_logical]
 * @return Tensor of accuracies for each logical qubit [num_logical]
 */
torch::Tensor compute_per_qubit_accuracy(torch::Tensor logits, torch::Tensor targets);

} // namespace qec

#endif // TRAIN_H
