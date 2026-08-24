#include "tensor.hpp"
#include <iostream>
#include <stdexcept>

// ============================================================================
// Storage Implementation
// ============================================================================

Storage::Storage(size_t size) : data_(size) {}

Storage::Storage(std::vector<float> values) : data_(std::move(values)) {}

float& Storage::operator[](size_t index) {
    return data_[index];
}

const float& Storage::operator[](size_t index) const {
    return data_[index];
}

float* Storage::data() {
    return data_.data();
}

const float* Storage::data() const {
    return data_.data();
}

size_t Storage::size() const {
    return data_.size();
}

// ============================================================================
// Tensor Implementation
// ============================================================================

size_t Tensor::compute_numel(const std::vector<size_t>& shape) {
    size_t n_elem = 1;
    for (size_t d : shape) {
        n_elem *= d;
    }
    return n_elem;
}

std::vector<size_t> Tensor::compute_contiguous_strides(const std::vector<size_t>& shape) {
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

Tensor::Tensor(std::vector<size_t> shape)
    : shape_(std::move(shape)),
      strides_(compute_contiguous_strides(shape_)),
      dim_(shape_.size()),
      numel_(compute_numel(shape_)) {
    storage_ = std::make_shared<Storage>(numel_);
}

Tensor::Tensor(std::vector<float> values, std::vector<size_t> shape)
    : shape_(std::move(shape)),
      strides_(compute_contiguous_strides(shape_)),
      dim_(shape_.size()),
      numel_(compute_numel(shape_)) {
    if (values.size() != numel_) {
        throw std::invalid_argument("Size of values does not match shape total elements.");
    }
    storage_ = std::make_shared<Storage>(std::move(values));
}

size_t Tensor::dim() const {
    return dim_;
}

size_t Tensor::numel() const {
    return numel_;
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

const std::vector<size_t>& Tensor::strides() const {
    return strides_;
}

float& Tensor::at(const std::vector<size_t>& indices) {
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

const float& Tensor::at(const std::vector<size_t>& indices) const {
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

float* Tensor::begin() {
    return storage_->data();
}

float* Tensor::end() {
    return storage_->data() + numel_;
}

const float* Tensor::begin() const {
    return storage_->data();
}

const float* Tensor::end() const {
    return storage_->data() + numel_;
}

Tensor Tensor::element_wise_op(const Tensor& other, ElementWiseOp op) const {
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

Tensor Tensor::operator+(const Tensor& other) const {
    return element_wise_op(other, ElementWiseOp::ADD);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return element_wise_op(other, ElementWiseOp::SUBTRACT);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return element_wise_op(other, ElementWiseOp::MULTIPLY);
}

Tensor Tensor::operator/(const Tensor& other) const {
    return element_wise_op(other, ElementWiseOp::DIVIDE);
}