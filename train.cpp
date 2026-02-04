#include "train.h"
#include <iostream>
#include <iomanip>

namespace qec {

// ============================================================================
// Trainer Implementation
// ============================================================================

Trainer::Trainer(QECTransformer model, float learning_rate)
    : model_(model),
      optimizer_(model->parameters(), torch::optim::AdamOptions(learning_rate)),
      criterion_(torch::nn::BCEWithLogitsLossOptions()) {

    std::cout << "Trainer initialized with learning rate: " << learning_rate << std::endl;
}

float Trainer::train_step(torch::Tensor syndromes, torch::Tensor targets) {
    // Set model to training mode
    model_->train();

    // Zero gradients
    optimizer_.zero_grad();

    // Forward pass
    auto logits = model_->forward(syndromes);

    // Compute loss (BCEWithLogitsLoss expects float targets)
    auto loss = criterion_(logits, targets.to(torch::kFloat));

    // Backward pass
    loss.backward();

    // Update weights
    optimizer_.step();

    return loss.item<float>();
}

std::pair<float, float> Trainer::eval_step(torch::Tensor syndromes, torch::Tensor targets) {
    // Set model to evaluation mode
    model_->eval();

    // Disable gradient computation
    torch::NoGradGuard no_grad;

    // Forward pass
    auto logits = model_->forward(syndromes);

    // Compute loss
    auto loss = criterion_(logits, targets.to(torch::kFloat));

    // Compute accuracy
    float accuracy = compute_accuracy(logits, targets);

    return {loss.item<float>(), accuracy};
}

std::pair<torch::Tensor, torch::Tensor> Trainer::generate_dummy_data(
    int batch_size, int num_checks, int num_logical) {

    // Generate random binary syndromes
    auto syndromes = torch::randint(0, 2, {batch_size, num_checks},
        torch::TensorOptions().dtype(torch::kFloat));

    // Generate random binary logical errors
    auto logical_errors = torch::randint(0, 2, {batch_size, num_logical},
        torch::TensorOptions().dtype(torch::kFloat));

    return {syndromes, logical_errors};
}

void Trainer::train_loop(DataLoader& data_loader,
                         int num_epochs,
                         int log_interval,
                         DataLoader* val_loader) {

    auto& info = data_loader.dataset()->info();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting Training with HDF5 Data" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Dataset:          " << info.code_name << std::endl;
    std::cout << "Num samples:      " << info.num_shots << std::endl;
    std::cout << "Num checks:       " << info.num_checks << std::endl;
    std::cout << "Num logicals:     " << info.num_logicals << std::endl;
    std::cout << "Epochs:           " << num_epochs << std::endl;
    std::cout << "Batches/epoch:    " << data_loader.num_batches() << std::endl;
    std::cout << "Log interval:     " << log_interval << std::endl;
    std::cout << "Validation:       " << (val_loader ? "Yes" : "No") << std::endl;
    std::cout << "========================================\n" << std::endl;

    float best_val_acc = 0.0f;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        float epoch_accuracy = 0.0f;
        int batch_count = 0;

        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        // Reset data loader for new epoch (shuffles data)
        data_loader.reset();

        while (data_loader.has_next()) {
            // Get next batch from HDF5 data
            auto [syndromes, targets] = data_loader.next();

            // Training step
            float loss = train_step(syndromes, targets);
            epoch_loss += loss;

            // Compute accuracy for logging
            {
                torch::NoGradGuard no_grad;
                model_->eval();
                auto logits = model_->forward(syndromes);
                epoch_accuracy += compute_accuracy(logits, targets);
                model_->train();
            }

            batch_count++;

            // Log progress
            if (batch_count % log_interval == 0) {
                float avg_loss = epoch_loss / batch_count;
                float avg_acc = epoch_accuracy / batch_count;
                std::cout << "  Batch " << std::setw(4) << batch_count
                          << "/" << data_loader.num_batches()
                          << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
                          << " | Acc: " << std::fixed << std::setprecision(2) << (avg_acc * 100) << "%"
                          << std::endl;
            }
        }

        // Epoch summary
        float avg_epoch_loss = epoch_loss / batch_count;
        float avg_epoch_acc = epoch_accuracy / batch_count;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Epoch " << (epoch + 1) << " Complete"
                  << " | Train Loss: " << std::fixed << std::setprecision(4) << avg_epoch_loss
                  << " | Train Acc: " << std::fixed << std::setprecision(2) << (avg_epoch_acc * 100) << "%";

        // Validation
        if (val_loader) {
            auto [val_loss, val_acc] = evaluate(*val_loader);
            std::cout << " | Val Loss: " << std::fixed << std::setprecision(4) << val_loss
                      << " | Val Acc: " << std::fixed << std::setprecision(2) << (val_acc * 100) << "%";

            if (val_acc > best_val_acc) {
                best_val_acc = val_acc;
                std::cout << " *";  // Mark best epoch
            }
        }
        std::cout << std::endl << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Training Complete!" << std::endl;
    if (val_loader) {
        std::cout << "Best Validation Accuracy: " << std::fixed << std::setprecision(2)
                  << (best_val_acc * 100) << "%" << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
}

void Trainer::train_loop_dummy(int num_epochs, int batch_size, int batches_per_epoch, int log_interval) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting Training with Dummy Data" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Epochs:           " << num_epochs << std::endl;
    std::cout << "Batch size:       " << batch_size << std::endl;
    std::cout << "Batches/epoch:    " << batches_per_epoch << std::endl;
    std::cout << "Log interval:     " << log_interval << std::endl;
    std::cout << "========================================\n" << std::endl;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        float epoch_accuracy = 0.0f;

        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        for (int batch = 0; batch < batches_per_epoch; ++batch) {
            // Generate dummy data for this batch
            auto [syndromes, targets] = generate_dummy_data(batch_size);

            // Training step
            float loss = train_step(syndromes, targets);
            epoch_loss += loss;

            // Compute accuracy for logging
            {
                torch::NoGradGuard no_grad;
                auto logits = model_->forward(syndromes);
                epoch_accuracy += compute_accuracy(logits, targets);
            }

            // Log progress
            if ((batch + 1) % log_interval == 0) {
                float avg_loss = epoch_loss / (batch + 1);
                float avg_acc = epoch_accuracy / (batch + 1);
                std::cout << "  Batch " << std::setw(4) << (batch + 1)
                          << "/" << batches_per_epoch
                          << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
                          << " | Acc: " << std::fixed << std::setprecision(2) << (avg_acc * 100) << "%"
                          << std::endl;
            }
        }

        // Epoch summary
        float avg_epoch_loss = epoch_loss / batches_per_epoch;
        float avg_epoch_acc = epoch_accuracy / batches_per_epoch;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Epoch " << (epoch + 1) << " Complete"
                  << " | Avg Loss: " << std::fixed << std::setprecision(4) << avg_epoch_loss
                  << " | Avg Acc: " << std::fixed << std::setprecision(2) << (avg_epoch_acc * 100) << "%"
                  << std::endl;
        std::cout << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Training Complete!" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

std::pair<float, float> Trainer::evaluate(DataLoader& data_loader) {
    model_->eval();
    torch::NoGradGuard no_grad;

    float total_loss = 0.0f;
    float total_accuracy = 0.0f;
    int batch_count = 0;

    data_loader.reset();

    while (data_loader.has_next()) {
        auto [syndromes, targets] = data_loader.next();

        auto logits = model_->forward(syndromes);
        auto loss = criterion_(logits, targets.to(torch::kFloat));

        total_loss += loss.item<float>();
        total_accuracy += compute_accuracy(logits, targets);
        batch_count++;
    }

    model_->train();

    return {total_loss / batch_count, total_accuracy / batch_count};
}

void Trainer::save_checkpoint(const std::string& path) {
    torch::save(model_, path);
    std::cout << "Model saved to: " << path << std::endl;
}

void Trainer::load_checkpoint(const std::string& path) {
    torch::load(model_, path);
    std::cout << "Model loaded from: " << path << std::endl;
}

// ============================================================================
// Utility Functions
// ============================================================================

float compute_accuracy(torch::Tensor logits, torch::Tensor targets) {
    // Convert logits to predictions (threshold at 0, since sigmoid(0) = 0.5)
    auto predictions = (logits > 0).to(torch::kFloat);

    // Compare with targets
    auto correct = (predictions == targets.to(torch::kFloat)).to(torch::kFloat);

    // Return mean accuracy
    return correct.mean().item<float>();
}

torch::Tensor compute_per_qubit_accuracy(torch::Tensor logits, torch::Tensor targets) {
    // Convert logits to predictions
    auto predictions = (logits > 0).to(torch::kFloat);

    // Compare with targets
    auto correct = (predictions == targets.to(torch::kFloat)).to(torch::kFloat);

    // Return mean accuracy per qubit (average over batch dimension)
    return correct.mean(0);
}

} // namespace qec
