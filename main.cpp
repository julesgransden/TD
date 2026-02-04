/**
 * @file main.cpp
 * @brief Entry point for QEC Transformer Decoder
 *
 * Transformer-based quantum error correction decoder for a [[72,12,6]] QLDPC code.
 * This encoder-decoder transformer architecture learns syndrome → logical error mappings.
 *
 * Usage:
 *   ./qec_transformer                    # Run with dummy data (demo mode)
 *   ./qec_transformer data.h5            # Train with HDF5 data file
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
              << "\nIf no HDF5 file is provided, runs demo with dummy data.\n";
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "QEC Transformer Decoder" << std::endl;
    std::cout << "Quantum Error Correction for [[72,12,6]] QLDPC Code" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // =========================================
    // Parse command line arguments
    // =========================================
    std::string h5_path = "";
    int num_epochs = NUM_EPOCHS;
    int batch_size = BATCH_SIZE;
    float learning_rate = LEARNING_RATE;
    int64_t max_samples = -1;  // -1 means use all samples
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

    // =========================================
    // Check CUDA availability
    // =========================================
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "CUDA not available. Training on CPU." << std::endl;
    }
    std::cout << std::endl;

    // =========================================
    // Determine model dimensions from data or defaults
    // =========================================
    int num_checks = NUM_CHECKS;
    int num_logical = NUM_LOGICAL;

    // If HDF5 file provided, we'll get dimensions from it
    std::shared_ptr<QECDataset> train_dataset;
    std::shared_ptr<QECDataset> val_dataset;

    if (!h5_path.empty()) {
        std::cout << "Loading data from: " << h5_path << std::endl;
        try {
            train_dataset = std::make_shared<QECDataset>(h5_path, device);

            // Limit samples if requested
            if (max_samples > 0) {
                train_dataset->limit_samples(max_samples);
            }

            train_dataset->print_summary();

            // Get dimensions from dataset
            num_checks = train_dataset->info().num_checks;
            num_logical = train_dataset->info().num_logicals;

            std::cout << "Using dimensions from HDF5 file:" << std::endl;
            std::cout << "  num_checks:  " << num_checks << std::endl;
            std::cout << "  num_logical: " << num_logical << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error loading HDF5 file: " << e.what() << std::endl;
            std::cerr << "Falling back to dummy data mode." << std::endl;
            h5_path = "";
        }
    }

    // =========================================
    // Create Model
    // =========================================
    std::cout << "\nCreating QEC Transformer model..." << std::endl;
    QECTransformer model(
        num_checks,           // num_checks from data or default
        num_logical,          // num_logical from data or default
        D_MODEL,              // d_model: 256
        NHEAD,                // nhead: 8 attention heads
        NUM_ENCODER_LAYERS,   // encoder_layers: 6
        NUM_DECODER_LAYERS,   // decoder_layers: 6
        DIM_FEEDFORWARD,      // dim_feedforward: 1024
        DROPOUT               // dropout: 0.1
    );

    // Move model to device
    model->to(device);

    // Print model summary
    model->print_summary();

    // =========================================
    // Create Trainer
    // =========================================
    Trainer trainer(model, learning_rate);

    // =========================================
    // Training Mode: HDF5 Data vs Dummy Data
    // =========================================
    if (!h5_path.empty() && train_dataset) {
        // =========================================
        // Train with HDF5 Data
        // =========================================
        std::cout << "========================================" << std::endl;
        std::cout << "Training with HDF5 Data" << std::endl;
        std::cout << "========================================\n" << std::endl;

        // Split into train and validation sets
        auto [train_split, val_split_data] = train_dataset->split(val_split, /*shuffle=*/true);

        std::cout << "Data split:" << std::endl;
        std::cout << "  Total:      " << train_dataset->size().value() << std::endl;
        std::cout << "  Training:   " << train_split->size().value()
                  << " (" << (1.0f - val_split) * 100 << "%)" << std::endl;
        std::cout << "  Validation: " << val_split_data->size().value()
                  << " (" << val_split * 100 << "%)" << std::endl;

        // Create data loaders for train and validation
        DataLoader train_loader(train_split, batch_size, /*shuffle=*/true);
        DataLoader val_loader(val_split_data, batch_size, /*shuffle=*/false);

        // Run training with validation
        trainer.train_loop(train_loader, num_epochs, LOG_INTERVAL, &val_loader);

        // Save checkpoint if requested
        if (!save_path.empty()) {
            trainer.save_checkpoint(save_path);
        }

    } else {
        // =========================================
        // Demo Mode: Dummy Data
        // =========================================
        std::cout << "========================================" << std::endl;
        std::cout << "Demo Mode (No HDF5 file provided)" << std::endl;
        std::cout << "========================================\n" << std::endl;

        // Test forward pass first
        std::cout << "Testing forward pass..." << std::endl;
        int test_batch_size = 4;
        auto test_syndromes = torch::randint(0, 2, {test_batch_size, num_checks},
            torch::TensorOptions().dtype(torch::kFloat).device(device));

        std::cout << "Input syndromes shape: [" << test_syndromes.size(0) << ", "
                  << test_syndromes.size(1) << "]" << std::endl;

        auto logits = model->forward(test_syndromes);

        std::cout << "Output logits shape: [" << logits.size(0) << ", "
                  << logits.size(1) << "]" << std::endl;

        auto probabilities = torch::sigmoid(logits);
        std::cout << "\nSample predictions (first batch item):" << std::endl;
        std::cout << "Probabilities: " << probabilities[0] << std::endl;

        // Run short training demo
        std::cout << "\n========================================" << std::endl;
        std::cout << "Running Short Training Demo" << std::endl;
        std::cout << "========================================\n" << std::endl;

        trainer.train_loop_dummy(
            3,    // num_epochs
            16,   // batch_size
            10,   // batches_per_epoch
            5     // log_interval
        );
    }

    // =========================================
    // Final Summary
    // =========================================
    std::cout << "========================================" << std::endl;
    std::cout << "QEC Transformer Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model parameters: " << model->num_parameters() << std::endl;
    std::cout << "Input dimension:  " << num_checks << " syndromes" << std::endl;
    std::cout << "Output dimension: " << num_logical << " logical qubits" << std::endl;

    if (h5_path.empty()) {
        std::cout << "\nTo train on real data, run:" << std::endl;
        std::cout << "  ./qec_transformer data.h5 --epochs 100 --batch 32" << std::endl;
    }
    std::cout << "========================================\n" << std::endl;

    return 0;
}
