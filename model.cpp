#include "model.h"
#include <iostream>
#include <iomanip>

namespace qec {

// ============================================================================
// OutputHead Implementation
// ============================================================================

OutputHeadImpl::OutputHeadImpl(int d_model) {
    // Linear projection from d_model to 1 (scalar logit per logical qubit)
    output_projection_ = register_module(
        "output_projection",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, 1))
    );

    // Initialize weights
    torch::nn::init::xavier_uniform_(output_projection_->weight);
    torch::nn::init::zeros_(output_projection_->bias);
}

torch::Tensor OutputHeadImpl::forward(torch::Tensor x) {
    // Input: [batch_size, num_logical, d_model]

    // Project each logical qubit representation to a scalar logit
    // [batch_size, num_logical, d_model] → [batch_size, num_logical, 1]
    x = output_projection_->forward(x);

    // Squeeze the last dimension
    // [batch_size, num_logical, 1] → [batch_size, num_logical]
    x = x.squeeze(-1);

    return x;
}

// ============================================================================
// QECTransformer Implementation
// ============================================================================

QECTransformerImpl::QECTransformerImpl(
    int num_checks,
    int num_logical,
    int d_model,
    int nhead,
    int num_encoder_layers,
    int num_decoder_layers,
    int dim_feedforward,
    float dropout)
    : num_checks_(num_checks),
      num_logical_(num_logical),
      d_model_(d_model) {

    std::cout << "========================================" << std::endl;
    std::cout << "Initializing QEC Transformer Model" << std::endl;
    std::cout << "========================================" << std::endl;

    // =========================================
    // 1. SYNDROME EMBEDDING
    // =========================================
    // Converts binary syndrome vector to dense embeddings with positional encoding
    std::cout << "Creating SyndromeEmbedding..." << std::endl;
    syndrome_embedding_ = register_module(
        "syndrome_embedding",
        SyndromeEmbedding(num_checks, d_model, dropout)
    );

    // =========================================
    // 2. TRANSFORMER ENCODER
    // =========================================
    // Processes syndrome embeddings with bidirectional self-attention
    std::cout << "Creating TransformerEncoder (" << num_encoder_layers << " layers)..." << std::endl;
    encoder_ = register_module(
        "encoder",
        TransformerEncoder(num_encoder_layers, d_model, nhead, dim_feedforward, dropout)
    );

    // =========================================
    // 3. LOGICAL QUBIT QUERIES
    // =========================================
    // Learned embeddings representing each of the logical qubits
    // These queries will attend to the encoded syndromes via cross-attention
    std::cout << "Creating Logical Qubit Queries (" << num_logical << " queries)..." << std::endl;
    logical_queries_ = register_parameter(
        "logical_queries",
        torch::randn({num_logical, d_model}) * 0.02  // Small initialization
    );

    // =========================================
    // 4. TRANSFORMER DECODER
    // =========================================
    // Uses cross-attention to let logical queries attend to encoded syndromes
    std::cout << "Creating TransformerDecoder (" << num_decoder_layers << " layers)..." << std::endl;
    decoder_ = register_module(
        "decoder",
        TransformerDecoder(num_decoder_layers, d_model, nhead, dim_feedforward, dropout)
    );

    // =========================================
    // 5. OUTPUT HEAD
    // =========================================
    // Projects decoder output to binary logits
    std::cout << "Creating OutputHead..." << std::endl;
    output_head_ = register_module(
        "output_head",
        OutputHead(d_model)
    );

    std::cout << "Model initialization complete!" << std::endl;
    std::cout << "========================================" << std::endl;
}

torch::Tensor QECTransformerImpl::forward(torch::Tensor syndromes) {
    // =========================================
    // Input Validation
    // =========================================
    // syndromes: [batch_size, NUM_CHECKS] binary values {0, 1}
    TORCH_CHECK(syndromes.dim() == 2,
        "Expected 2D input tensor, got ", syndromes.dim(), "D");
    TORCH_CHECK(syndromes.size(1) == num_checks_,
        "Expected ", num_checks_, " syndrome values, got ", syndromes.size(1));

    auto batch_size = syndromes.size(0);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Forward Pass" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Input syndromes: [" << batch_size << ", " << num_checks_ << "]" << std::endl;

    // =========================================
    // 1. EMBEDDING: Syndrome embedding + positional encoding
    // =========================================
    // [batch_size, NUM_CHECKS] → [batch_size, NUM_CHECKS, D_MODEL]
    std::cout << "\n[Step 1] Syndrome Embedding" << std::endl;
    auto embedded = syndrome_embedding_->forward(syndromes);
    std::cout << "  Output: [" << embedded.size(0) << ", " << embedded.size(1)
              << ", " << embedded.size(2) << "]" << std::endl;

    // =========================================
    // 2. ENCODER: Bidirectional self-attention on syndromes
    // =========================================
    // [batch_size, NUM_CHECKS, D_MODEL] → [batch_size, NUM_CHECKS, D_MODEL]
    std::cout << "\n[Step 2] Transformer Encoder" << std::endl;
    auto encoded = encoder_->forward(embedded);
    std::cout << "  Output: [" << encoded.size(0) << ", " << encoded.size(1)
              << ", " << encoded.size(2) << "]" << std::endl;

    // =========================================
    // 3. PREPARE DECODER QUERIES: Expand learned logical queries for batch
    // =========================================
    // [NUM_LOGICAL, D_MODEL] → [batch_size, NUM_LOGICAL, D_MODEL]
    std::cout << "\n[Step 3] Prepare Logical Qubit Queries" << std::endl;
    auto queries = logical_queries_.unsqueeze(0).expand({batch_size, -1, -1});
    // Need to clone to avoid in-place modification issues
    queries = queries.clone();
    std::cout << "  Queries: [" << queries.size(0) << ", " << queries.size(1)
              << ", " << queries.size(2) << "]" << std::endl;

    // =========================================
    // 4. DECODER: Cross-attention from logical qubits to syndromes
    // =========================================
    // tgt: [batch_size, NUM_LOGICAL, D_MODEL] (queries)
    // memory: [batch_size, NUM_CHECKS, D_MODEL] (encoded syndromes)
    // Output: [batch_size, NUM_LOGICAL, D_MODEL]
    std::cout << "\n[Step 4] Transformer Decoder" << std::endl;
    auto decoded = decoder_->forward(queries, encoded);
    std::cout << "  Output: [" << decoded.size(0) << ", " << decoded.size(1)
              << ", " << decoded.size(2) << "]" << std::endl;

    // =========================================
    // 5. OUTPUT: Project to binary logits
    // =========================================
    // [batch_size, NUM_LOGICAL, D_MODEL] → [batch_size, NUM_LOGICAL]
    std::cout << "\n[Step 5] Output Head" << std::endl;
    auto logits = output_head_->forward(decoded);
    std::cout << "  Output logits: [" << logits.size(0) << ", " << logits.size(1) << "]" << std::endl;

    std::cout << "========================================\n" << std::endl;

    return logits;
}

int64_t QECTransformerImpl::num_parameters() const {
    int64_t total = 0;
    for (const auto& param : parameters()) {
        total += param.numel();
    }
    return total;
}

void QECTransformerImpl::print_summary() const {
    std::cout << "\n========================================" << std::endl;
    std::cout << "QEC Transformer Model Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Architecture:" << std::endl;
    std::cout << "  - Input (syndromes):    " << num_checks_ << std::endl;
    std::cout << "  - Output (logical):     " << num_logical_ << std::endl;
    std::cout << "  - Embedding dim:        " << d_model_ << std::endl;
    std::cout << "  - Attention heads:      " << NHEAD << std::endl;
    std::cout << "  - Encoder layers:       " << NUM_ENCODER_LAYERS << std::endl;
    std::cout << "  - Decoder layers:       " << NUM_DECODER_LAYERS << std::endl;
    std::cout << "  - FFN dimension:        " << DIM_FEEDFORWARD << std::endl;
    std::cout << "  - Dropout:              " << DROPOUT << std::endl;
    std::cout << std::endl;
    std::cout << "Total parameters: " << std::fixed << std::setprecision(2)
              << (num_parameters() / 1e6) << "M" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

} // namespace qec
