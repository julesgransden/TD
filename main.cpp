/**
 * @file main.cpp
 * @brief Entry point for QEC Transformer Decoder
 *
 * Transformer-based quantum error correction decoder for a [[72,12,6]] QLDPC code.
 * This encoder-decoder transformer architecture learns syndrome → logical error mappings.
 *
 * Usage:
 *   ./qec_transformer                    # Train with default training_data.h5
 *   ./qec_transformer data.h5            # Train with specified HDF5 data file
 *   ./qec_transformer data.h5 --epochs 50 --batch 64
 *
 * Expected HDF5 structure:
 *   - "syndromes": [num_shots, num_detectors], uint8
 *   - "logical_errors": [num_shots, num_logicals], uint8
 *   - Attributes: num_shots, num_checks, num_logicals, code
 */

#include <iostream>
#include <string>
#include <torch/torch.h>

#include "config.h"
#include "model.h"
#include "train.h"
#include "data_loader.h"

using namespace qec;

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [data.h5] [options]\n"
              << "\nOptions:\n"
              << "  --epochs N     Number of training epochs (default: " << NUM_EPOCHS << ")\n"
              << "  --batch N      Batch size (default: " << BATCH_SIZE << ")\n"
              << "  --lr RATE      Learning rate (default: " << LEARNING_RATE << ")\n"
              << "  --samples N    Limit training to first N samples (default: all)\n"
              << "  --val-split F  Validation split ratio (default: 0.1)\n"
              << "  --save PATH    Save model checkpoint after training\n"
              << "  --help         Show this help message\n"
              << "\nIf no HDF5 file is provided, uses training_data.h5 in current directory.\n";
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string h5_path = "training_data.h5";
    int num_epochs = NUM_EPOCHS;
    int batch_size = BATCH_SIZE;
    float learning_rate = LEARNING_RATE;
    int64_t max_samples = -1;
    float val_split = 0.1f;
    std::string save_path = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--epochs" && i + 1 < argc) {
            num_epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            learning_rate = std::stof(argv[++i]);
        } else if (arg == "--samples" && i + 1 < argc) {
            max_samples = std::stoll(argv[++i]);
        } else if (arg == "--val-split" && i + 1 < argc) {
            val_split = std::stof(argv[++i]);
        } else if (arg == "--save" && i + 1 < argc) {
            save_path = argv[++i];
        } else if (arg[0] != '-') {
            h5_path = arg;
        }
    }

    // Check CUDA availability
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }

    // Load HDF5 training data
    int num_checks = NUM_CHECKS;
    int num_logical = NUM_LOGICAL;
    std::shared_ptr<QECDataset> train_dataset;

    try {
        train_dataset = std::make_shared<QECDataset>(h5_path, device);
        if (max_samples > 0) {
            train_dataset->limit_samples(max_samples);
        }
        num_checks = train_dataset->info().num_checks;
        num_logical = train_dataset->info().num_logicals;
    } catch (const std::exception& e) {
        std::cerr << "Error loading HDF5 file: " << e.what() << std::endl;
        return 1;
    }

    // Create model
    QECTransformer model(num_checks, num_logical, D_MODEL, NHEAD,
                         NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                         DIM_FEEDFORWARD, DROPOUT);
    model->to(device);
    model->print_summary();

    // Create trainer
    Trainer trainer(model, learning_rate);

    // Split and train
    auto [train_split, val_split_data] = train_dataset->split(val_split, true);
    DataLoader train_loader(train_split, batch_size, true);
    DataLoader val_loader(val_split_data, batch_size, false);

    std::cout << "Training " << num_epochs << " epochs: "
              << train_split->size().value() << " train, "
              << val_split_data->size().value() << " val samples (batch=" << batch_size << ")"
              << std::endl;

    trainer.train_loop(train_loader, num_epochs, LOG_INTERVAL, &val_loader);

    if (!save_path.empty()) {
        trainer.save_checkpoint(save_path);
    }

    return 0;
}
