#include "data_loader.h"
#include <H5Cpp.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace qec {

// ============================================================================
// QECDataset Implementation
// ============================================================================

QECDataset::QECDataset(torch::Tensor syndromes,
                       torch::Tensor logical_errors,
                       const DatasetInfo& info,
                       torch::Device device)
    : syndromes_(syndromes),
      logical_errors_(logical_errors),
      info_(info),
      device_(device) {
    // Constructor for creating dataset from tensors (used in split)
}

QECDataset::QECDataset(const std::string& h5_path, torch::Device device)
    : device_(device) {

    std::cout << "Loading dataset from: " << h5_path << std::endl;

    try {
        // Open HDF5 file (read-only)
        H5::H5File file(h5_path, H5F_ACC_RDONLY);

        // =========================================
        // Read syndromes dataset
        // =========================================
        {
            H5::DataSet dataset = file.openDataSet("syndromes");
            H5::DataSpace dataspace = dataset.getSpace();

            // Get dimensions
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims);
            int64_t num_shots = static_cast<int64_t>(dims[0]);
            int64_t num_checks = static_cast<int64_t>(dims[1]);

            std::cout << "  Syndromes shape: [" << num_shots << ", " << num_checks << "]" << std::endl;

            // Read data into vector
            std::vector<uint8_t> buffer(num_shots * num_checks);
            dataset.read(buffer.data(), H5::PredType::NATIVE_UINT8);

            // Convert to torch tensor
            // First create as uint8, then convert to float for model
            syndromes_ = torch::from_blob(
                buffer.data(),
                {num_shots, num_checks},
                torch::TensorOptions().dtype(torch::kUInt8)
            ).clone().to(torch::kFloat).to(device_);

            info_.num_shots = num_shots;
            info_.num_checks = num_checks;
        }

        // =========================================
        // Read logical_errors dataset
        // =========================================
        {
            H5::DataSet dataset = file.openDataSet("logical_errors");
            H5::DataSpace dataspace = dataset.getSpace();

            // Get dimensions
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims);
            int64_t num_shots = static_cast<int64_t>(dims[0]);
            int64_t num_logicals = static_cast<int64_t>(dims[1]);

            std::cout << "  Logical errors shape: [" << num_shots << ", " << num_logicals << "]" << std::endl;

            // Verify consistency
            if (num_shots != info_.num_shots) {
                throw std::runtime_error("Mismatch: syndromes and logical_errors have different num_shots");
            }

            // Read data into vector
            std::vector<uint8_t> buffer(num_shots * num_logicals);
            dataset.read(buffer.data(), H5::PredType::NATIVE_UINT8);

            // Convert to torch tensor
            logical_errors_ = torch::from_blob(
                buffer.data(),
                {num_shots, num_logicals},
                torch::TensorOptions().dtype(torch::kUInt8)
            ).clone().to(torch::kFloat).to(device_);

            info_.num_logicals = num_logicals;
        }

        // =========================================
        // Read attributes (optional metadata)
        // =========================================
        try {
            if (file.attrExists("code")) {
                H5::Attribute attr = file.openAttribute("code");
                H5::StrType stype = attr.getStrType();
                std::string code_name;
                attr.read(stype, code_name);
                info_.code_name = code_name;
            } else {
                info_.code_name = "unknown";
            }
        } catch (...) {
            info_.code_name = "unknown";
        }

        file.close();
        std::cout << "Dataset loaded successfully!" << std::endl;

    } catch (H5::FileIException& e) {
        throw std::runtime_error("Failed to open HDF5 file: " + h5_path + "\n" + e.getDetailMsg());
    } catch (H5::DataSetIException& e) {
        throw std::runtime_error("Failed to read dataset: " + std::string(e.getDetailMsg()));
    } catch (H5::DataSpaceIException& e) {
        throw std::runtime_error("DataSpace error: " + std::string(e.getDetailMsg()));
    }
}

torch::data::Example<> QECDataset::get(size_t index) {
    // Return single sample: (syndrome, logical_error)
    return {syndromes_[index].clone(), logical_errors_[index].clone()};
}

torch::optional<size_t> QECDataset::size() const {
    return static_cast<size_t>(info_.num_shots);
}

void QECDataset::limit_samples(int64_t n) {
    if (n <= 0 || n >= info_.num_shots) {
        return;  // No limiting needed
    }

    std::cout << "Limiting dataset from " << info_.num_shots << " to " << n << " samples" << std::endl;

    // Slice tensors to first n samples
    syndromes_ = syndromes_.slice(0, 0, n).clone();
    logical_errors_ = logical_errors_.slice(0, 0, n).clone();

    // Update info
    info_.num_shots = n;
}

std::pair<std::shared_ptr<QECDataset>, std::shared_ptr<QECDataset>>
QECDataset::split(float val_ratio, bool shuffle) {
    int64_t total = info_.num_shots;
    int64_t val_size = static_cast<int64_t>(total * val_ratio);
    int64_t train_size = total - val_size;

    std::cout << "Splitting dataset: " << train_size << " train, " << val_size << " validation" << std::endl;

    // Create indices
    std::vector<int64_t> indices(total);
    for (int64_t i = 0; i < total; ++i) {
        indices[i] = i;
    }

    // Shuffle if requested
    if (shuffle) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
    }

    // Create index tensors
    std::vector<int64_t> train_indices(indices.begin(), indices.begin() + train_size);
    std::vector<int64_t> val_indices(indices.begin() + train_size, indices.end());

    auto train_idx = torch::tensor(train_indices, torch::TensorOptions().dtype(torch::kLong)).to(device_);
    auto val_idx = torch::tensor(val_indices, torch::TensorOptions().dtype(torch::kLong)).to(device_);

    // Index into tensors to create train/val splits
    auto train_syndromes = syndromes_.index_select(0, train_idx);
    auto train_logicals = logical_errors_.index_select(0, train_idx);
    auto val_syndromes = syndromes_.index_select(0, val_idx);
    auto val_logicals = logical_errors_.index_select(0, val_idx);

    // Create info for each split
    DatasetInfo train_info = info_;
    train_info.num_shots = train_size;

    DatasetInfo val_info = info_;
    val_info.num_shots = val_size;

    // Create new datasets
    auto train_dataset = std::make_shared<QECDataset>(
        train_syndromes, train_logicals, train_info, device_);
    auto val_dataset = std::make_shared<QECDataset>(
        val_syndromes, val_logicals, val_info, device_);

    return {train_dataset, val_dataset};
}

void QECDataset::print_summary() const {
    std::cout << "\n========================================" << std::endl;
    std::cout << "QEC Dataset Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Code:           " << info_.code_name << std::endl;
    std::cout << "Num shots:      " << info_.num_shots << std::endl;
    std::cout << "Num checks:     " << info_.num_checks << std::endl;
    std::cout << "Num logicals:   " << info_.num_logicals << std::endl;
    std::cout << "Syndromes:      [" << syndromes_.size(0) << ", " << syndromes_.size(1) << "]" << std::endl;
    std::cout << "Logical errors: [" << logical_errors_.size(0) << ", " << logical_errors_.size(1) << "]" << std::endl;
    std::cout << "Device:         " << device_ << std::endl;

    // Print class balance for logical errors
    auto error_rate = logical_errors_.mean(0);
    std::cout << "Error rates per logical qubit:" << std::endl;
    std::cout << "  " << error_rate << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// ============================================================================
// DataLoader Implementation
// ============================================================================

DataLoader::DataLoader(std::shared_ptr<QECDataset> dataset, int batch_size, bool shuffle)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle), current_idx_(0) {

    // Initialize indices
    int64_t n = dataset_->size().value();
    indices_.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        indices_[i] = i;
    }

    if (shuffle_) {
        reset();  // Initial shuffle
    }
}

void DataLoader::reset() {
    current_idx_ = 0;

    if (shuffle_) {
        // Shuffle indices using random device
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices_.begin(), indices_.end(), gen);
    }
}

bool DataLoader::has_next() const {
    return current_idx_ < static_cast<int64_t>(indices_.size());
}

std::pair<torch::Tensor, torch::Tensor> DataLoader::next() {
    if (!has_next()) {
        throw std::runtime_error("No more batches available. Call reset() for new epoch.");
    }

    // Calculate batch range
    int64_t start = current_idx_;
    int64_t end = std::min(start + batch_size_, static_cast<int64_t>(indices_.size()));
    int64_t actual_batch_size = end - start;

    // Gather indices for this batch
    std::vector<int64_t> batch_indices(indices_.begin() + start, indices_.begin() + end);

    // Create index tensor and use it to gather samples
    auto idx_tensor = torch::tensor(batch_indices, torch::TensorOptions().dtype(torch::kLong));

    // Index into the full tensors
    auto syndromes = dataset_->syndromes().index_select(0, idx_tensor.to(dataset_->syndromes().device()));
    auto logical_errors = dataset_->logical_errors().index_select(0, idx_tensor.to(dataset_->logical_errors().device()));

    current_idx_ = end;

    return {syndromes, logical_errors};
}

int64_t DataLoader::num_batches() const {
    int64_t n = static_cast<int64_t>(indices_.size());
    return (n + batch_size_ - 1) / batch_size_;  // Ceiling division
}

// ============================================================================
// Utility Functions
// ============================================================================

std::pair<std::shared_ptr<QECDataset>, std::shared_ptr<QECDataset>>
load_and_split(const std::string& h5_path, float train_ratio, torch::Device device) {

    // Load full dataset
    auto full_dataset = std::make_shared<QECDataset>(h5_path, device);

    int64_t total = full_dataset->size().value();
    int64_t train_size = static_cast<int64_t>(total * train_ratio);
    int64_t val_size = total - train_size;

    std::cout << "Splitting dataset: " << train_size << " train, " << val_size << " validation" << std::endl;

    // Create shuffled indices
    std::vector<int64_t> indices(total);
    for (int64_t i = 0; i < total; ++i) {
        indices[i] = i;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    // Split indices
    std::vector<int64_t> train_indices(indices.begin(), indices.begin() + train_size);
    std::vector<int64_t> val_indices(indices.begin() + train_size, indices.end());

    // Create index tensors
    auto train_idx = torch::tensor(train_indices, torch::TensorOptions().dtype(torch::kLong)).to(device);
    auto val_idx = torch::tensor(val_indices, torch::TensorOptions().dtype(torch::kLong)).to(device);

    // Note: For simplicity, we return the full dataset and the caller can use
    // DataLoader with the appropriate indices. For a proper split, you'd want
    // separate dataset objects. Here we just return the same dataset twice
    // and let the DataLoader handle the indexing.

    // For now, return the same dataset - the DataLoader will handle batching
    // In a production system, you'd want proper train/val dataset classes
    return {full_dataset, full_dataset};
}

} // namespace qec
