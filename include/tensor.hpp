#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// ============================================================================
// Operations & Types
// ============================================================================

enum class ElementWiseOp {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE
};

// ============================================================================
// Storage (Raw 1D Memory Management)
// ============================================================================

class Storage {
private:
    std::vector<float> data_;

public:
    explicit Storage(size_t size);
    explicit Storage(std::vector<float> values);

    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    float* data();
    const float* data() const;

    size_t size() const;
};

// ============================================================================
// Tensor (Multidimensional View & Operations)
// ============================================================================

class Tensor {
private:
    std::shared_ptr<Storage> storage_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t dim_{0};
    size_t numel_{0};

    static size_t compute_numel(const std::vector<size_t>& shape);
    static std::vector<size_t> compute_contiguous_strides(const std::vector<size_t>& shape);

public:
    // Constructors
    explicit Tensor(std::vector<size_t> shape);
    Tensor(std::vector<float> values, std::vector<size_t> shape);

    // Metadata Getters
    size_t dim() const;
    size_t numel() const;
    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;

    // Element Access
    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;

    // Iterators
    float* begin();
    float* end();
    const float* begin() const;
    const float* end() const;

    // Arithmetic & Elementwise Operations
    Tensor element_wise_op(const Tensor& other, ElementWiseOp op) const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
};
