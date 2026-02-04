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

void Trainer::train_loop(DataLoader& data_loader,
                         int num_epochs,
                         int log_interval,
                         DataLoader* val_loader) {

    constexpr int LOG_EVERY = 100;  // Print every 100 batches
    float best_val_acc = 0.0f;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float recent_loss = 0.0f;
        int batch_count = 0;

        data_loader.reset();

        while (data_loader.has_next()) {
            auto [syndromes, targets] = data_loader.next();

            float loss = train_step(syndromes, targets);
            recent_loss += loss;
            batch_count++;

            if (batch_count % LOG_EVERY == 0) {
                float avg_loss = recent_loss / LOG_EVERY;

                // Compute accuracy on current batch
                float acc;
                {
                    torch::NoGradGuard no_grad;
                    model_->eval();
                    auto logits = model_->forward(syndromes);
                    acc = compute_accuracy(logits, targets);
                    model_->train();
                }

                std::cout << "Epoch " << (epoch + 1) << " Batch " << batch_count
                          << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
                          << " | Acc: " << std::fixed << std::setprecision(2) << (acc * 100) << "%"
                          << std::endl;

                recent_loss = 0.0f;
            }
        }

        // Validation at end of epoch
        if (val_loader) {
            auto [val_loss, val_acc] = evaluate(*val_loader);
            std::cout << "Epoch " << (epoch + 1) << " Val | Loss: " << std::fixed << std::setprecision(4) << val_loss
                      << " | Acc: " << std::fixed << std::setprecision(2) << (val_acc * 100) << "%";
            if (val_acc > best_val_acc) {
                best_val_acc = val_acc;
                std::cout << " *";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Training complete. Best val acc: " << std::fixed << std::setprecision(2)
              << (best_val_acc * 100) << "%" << std::endl;
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
