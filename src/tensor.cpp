#include <iostream>
#include <vector>
#include <string>
#include <cstddef>
#include <memory>
#include <stdexcept>

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
    // Constructors
    explicit Storage(size_t size) : data_(size) {}
    explicit Storage(std::vector<float> values) : data_(std::move(values)) {}

    // Element Access
    float& operator[](size_t index) { return data_[index]; }
    const float& operator[](size_t index) const { return data_[index]; }

    // Direct Buffer Access
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    // Size
    size_t size() const { return data_.size(); }
};

// ============================================================================
// Tensor (Multidimensional View & Operations)
// ============================================================================

class Tensor {
private:
    // --- Member Variables ---
    std::shared_ptr<Storage> storage_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t dim_{0};
    size_t numel_{0};

    // --- Private Helpers ---
    static size_t compute_numel(const std::vector<size_t>& shape) {
        size_t n_elem = 1;
        for (size_t d : shape) {
            n_elem *= d;
        }
        return n_elem;
    }

    static std::vector<size_t> compute_contiguous_strides(const std::vector<size_t>& shape) {
        size_t dim = shape.size();
        std::vector<size_t> strides(dim, 1);
        for (size_t i = 0; i < dim; i++) {
            size_t stride_i = dim - i - 1;
            if (stride_i > 0) {
                strides[stride_i - 1] = strides[stride_i] * shape[stride_i];
            }
        }
        return strides;
    }

public:
    // --- Constructors ---

    // 1. Allocate uninitialized/zeroed tensor from shape
    explicit Tensor(std::vector<size_t> shape)
        : shape_(std::move(shape)),
          strides_(compute_contiguous_strides(shape_)),
          dim_(shape_.size()),
          numel_(compute_numel(shape_)) {
        storage_ = std::make_shared<Storage>(numel_);
    }

    // 2. Initialize tensor from values and shape
    Tensor(std::vector<float> values, std::vector<size_t> shape)
        : shape_(std::move(shape)),
          strides_(compute_contiguous_strides(shape_)),
          dim_(shape_.size()),
          numel_(compute_numel(shape_)) {
        if (values.size() != numel_) {
            throw std::invalid_argument("Size of values does not match shape total elements.");
        }
        storage_ = std::make_shared<Storage>(std::move(values));
    }

    // --- Metadata Getters ---
    size_t dim() const { return dim_; }
    size_t numel() const { return numel_; }
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }

    // --- Element Access ---

    // Mutable access
    float& at(const std::vector<size_t>& indices) {
        if (indices.size() != dim_) {
            throw std::invalid_argument(
                "Expected " + std::to_string(dim_) + " indices, but got " + std::to_string(indices.size())
            );
        }
        size_t flat_index = 0;
        for (size_t i = 0; i < dim_; i++) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range(
                    "Index " + std::to_string(indices[i]) +
                    " is out of bounds for dimension " + std::to_string(i) +
                    " with size " + std::to_string(shape_[i])
                );
            }
            flat_index += strides_[i] * indices[i];
        }
        return (*storage_)[flat_index];
    }

    // Const (read-only) access
    const float& at(const std::vector<size_t>& indices) const {
        if (indices.size() != dim_) {
            throw std::invalid_argument(
                "Expected " + std::to_string(dim_) + " indices, but got " + std::to_string(indices.size())
            );
        }
        size_t flat_index = 0;
        for (size_t i = 0; i < dim_; i++) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range(
                    "Index " + std::to_string(indices[i]) +
                    " is out of bounds for dimension " + std::to_string(i) +
                    " with size " + std::to_string(shape_[i])
                );
            }
            flat_index += strides_[i] * indices[i];
        }
        return (*storage_)[flat_index];
    }

    // --- Iterators ---
    float* begin() { return storage_->data(); }
    float* end() { return storage_->data() + numel_; }

    const float* begin() const { return storage_->data(); }
    const float* end() const { return storage_->data() + numel_; }

    // --- Arithmetic & Elementwise Operations ---
    Tensor element_wise_op(const Tensor& other, ElementWiseOp op) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensors must have matching shapes for elementwise operations.");
        }

        Tensor result(shape_);
        switch (op) {
            case ElementWiseOp::ADD:
                for (size_t i = 0; i < numel_; ++i) {
                    (*result.storage_)[i] = (*storage_)[i] + (*other.storage_)[i];
                }
                break;
            case ElementWiseOp::SUBTRACT:
                for (size_t i = 0; i < numel_; ++i) {
                    (*result.storage_)[i] = (*storage_)[i] - (*other.storage_)[i];
                }
                break;
            case ElementWiseOp::MULTIPLY:
                for (size_t i = 0; i < numel_; ++i) {
                    (*result.storage_)[i] = (*storage_)[i] * (*other.storage_)[i];
                }
                break;
            case ElementWiseOp::DIVIDE:
                for (size_t i = 0; i < numel_; ++i) {
                    (*result.storage_)[i] = (*storage_)[i] / (*other.storage_)[i];
                }
                break;
        }
        return result;
    }

    Tensor operator+(const Tensor& other) const { return element_wise_op(other, ElementWiseOp::ADD); }
    Tensor operator-(const Tensor& other) const { return element_wise_op(other, ElementWiseOp::SUBTRACT); }
    Tensor operator*(const Tensor& other) const { return element_wise_op(other, ElementWiseOp::MULTIPLY); }
    Tensor operator/(const Tensor& other) const { return element_wise_op(other, ElementWiseOp::DIVIDE); }
};

int main() {
    Tensor t1({1,2,3,4,5,6,7,8,9,10,11,12}, {2,2,3});
    Tensor t2({10,20,30,40,50,60,70,80,90,100,110,120}, {2,2,3});
    Tensor t3 = t1 * t2;

    std::cout << "t1[1, 0, 2] = " << t1.at({1,1,2}) << "\n";
    std::cout << "t3[1, 0, 2] = " << t3.at({1,1,2}) << "\n";
    std::cout << "t3 numel    = " << t3.numel() << "\n";

    for (float val: t3) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    return 0;
}