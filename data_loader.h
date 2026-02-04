#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>

/**
 * @file data_loader.h
 * @brief HDF5 Data Loader for QEC Transformer
 *
 * Loads syndrome and logical error data from HDF5 files.
 *
 * Expected HDF5 structure:
 * - "syndromes": [num_shots, num_detectors], uint8, binary {0, 1}
 * - "logical_errors": [num_shots, num_logicals], uint8, binary {0, 1}
 * - Attributes: num_shots, num_checks, num_logicals, code
 */

namespace qec {

/**
 * @struct DatasetInfo
 * @brief Metadata about the loaded dataset
 */
struct DatasetInfo {
    int64_t num_shots;      // Number of samples
    int64_t num_checks;     // Number of syndrome checks (detectors)
    int64_t num_logicals;   // Number of logical qubits
    std::string code_name;  // Code identifier (e.g., "[[72,12,6]]")
};

/**
 * @class QECDataset
 * @brief PyTorch-style dataset for QEC data
 *
 * Loads data from HDF5 and provides batched access via indexing.
 */
class QECDataset : public torch::data::datasets::Dataset<QECDataset> {
public:
    /**
     * @brief Load dataset from HDF5 file
     * @param h5_path Path to the HDF5 file
     * @param device Device to load tensors to (CPU or CUDA)
     */
    explicit QECDataset(const std::string& h5_path,
                        torch::Device device = torch::kCPU);

    /**
     * @brief Create dataset from tensors (used for splitting)
     * @param syndromes Syndrome tensor [num_shots, num_checks]
     * @param logical_errors Logical error tensor [num_shots, num_logicals]
     * @param info Dataset metadata
     * @param device Device tensors are on
     */
    QECDataset(torch::Tensor syndromes,
               torch::Tensor logical_errors,
               const DatasetInfo& info,
               torch::Device device);

    /**
     * @brief Get a single sample by index
     * @param index Sample index
     * @return Pair of (syndrome, logical_error) tensors
     */
    torch::data::Example<> get(size_t index) override;

    /**
     * @brief Get the total number of samples
     */
    torch::optional<size_t> size() const override;

    /**
     * @brief Get dataset metadata
     */
    const DatasetInfo& info() const { return info_; }

    /**
     * @brief Get all syndromes tensor
     */
    const torch::Tensor& syndromes() const { return syndromes_; }

    /**
     * @brief Get all logical errors tensor
     */
    const torch::Tensor& logical_errors() const { return logical_errors_; }

    /**
     * @brief Print dataset summary
     */
    void print_summary() const;

    /**
     * @brief Limit dataset to first N samples
     * @param n Maximum number of samples to keep
     */
    void limit_samples(int64_t n);

    /**
     * @brief Split dataset into train and validation sets
     * @param val_ratio Fraction of data for validation (e.g., 0.1 for 10%)
     * @param shuffle Whether to shuffle before splitting
     * @return Pair of (train_dataset, val_dataset)
     */
    std::pair<std::shared_ptr<QECDataset>, std::shared_ptr<QECDataset>>
    split(float val_ratio, bool shuffle = true);

private:
    torch::Tensor syndromes_;       // [num_shots, num_checks]
    torch::Tensor logical_errors_;  // [num_shots, num_logicals]
    DatasetInfo info_;
    torch::Device device_;
};

/**
 * @class DataLoader
 * @brief Batched data loader with shuffling support
 *
 * Wraps QECDataset to provide batched iteration with optional shuffling.
 */
class DataLoader {
public:
    /**
     * @brief Create a data loader
     * @param dataset The QEC dataset
     * @param batch_size Batch size
     * @param shuffle Whether to shuffle data each epoch
     */
    DataLoader(std::shared_ptr<QECDataset> dataset,
               int batch_size,
               bool shuffle = true);

    /**
     * @brief Reset loader for new epoch (reshuffles if enabled)
     */
    void reset();

    /**
     * @brief Check if more batches available
     */
    bool has_next() const;

    /**
     * @brief Get next batch
     * @return Pair of (syndromes, logical_errors) tensors for batch
     */
    std::pair<torch::Tensor, torch::Tensor> next();

    /**
     * @brief Get number of batches per epoch
     */
    int64_t num_batches() const;

    /**
     * @brief Get the underlying dataset
     */
    std::shared_ptr<QECDataset> dataset() const { return dataset_; }

private:
    std::shared_ptr<QECDataset> dataset_;
    int batch_size_;
    bool shuffle_;
    std::vector<int64_t> indices_;
    int64_t current_idx_;
};

/**
 * @brief Load HDF5 file and create train/validation split
 * @param h5_path Path to HDF5 file
 * @param train_ratio Fraction of data for training (default 0.8)
 * @param device Device to load data to
 * @return Pair of (train_dataset, val_dataset)
 */
std::pair<std::shared_ptr<QECDataset>, std::shared_ptr<QECDataset>>
load_and_split(const std::string& h5_path,
               float train_ratio = 0.8f,
               torch::Device device = torch::kCPU);

} // namespace qec

#endif // DATA_LOADER_H
