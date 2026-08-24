#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<float>, std::vector<size_t>>(), py::arg("values"), py::arg("shape"))
        .def(py::init<std::vector<size_t>>(), py::arg("shape"))
        .def("numel", &Tensor::numel)
        .def("dim", &Tensor::dim)
        .def("shape", &Tensor::shape)
        .def("strides", &Tensor::strides)
        .def("at", [](Tensor& t, const std::vector<size_t>& idx) {
            return t.at(idx);
        })
        .def("__getitem__", [](Tensor& t, py::object idx_obj) -> float {
            std::vector<size_t> indices;
            if (py::isinstance<py::tuple>(idx_obj)) {
                py::tuple tup = idx_obj.cast<py::tuple>();
                for (auto item : tup) {
                    indices.push_back(item.cast<size_t>());
                }
            } else if (py::isinstance<py::int_>(idx_obj)) {
                indices.push_back(idx_obj.cast<size_t>());
            } else if (py::isinstance<py::list>(idx_obj)) {
                indices = idx_obj.cast<std::vector<size_t>>();
            } else {
                throw std::invalid_argument("Indices must be an int, tuple, or list.");
            }
            return t.at(indices);
        })
        .def("__setitem__", [](Tensor& t, py::object idx_obj, float val) {
            std::vector<size_t> indices;
            if (py::isinstance<py::tuple>(idx_obj)) {
                py::tuple tup = idx_obj.cast<py::tuple>();
                for (auto item : tup) {
                    indices.push_back(item.cast<size_t>());
                }
            } else if (py::isinstance<py::int_>(idx_obj)) {
                indices.push_back(idx_obj.cast<size_t>());
            } else if (py::isinstance<py::list>(idx_obj)) {
                indices = idx_obj.cast<std::vector<size_t>>();
            } else {
                throw std::invalid_argument("Indices must be an int, tuple, or list.");
            }
            t.at(indices) = val;
        })
        .def("__add__", &Tensor::operator+)
        .def("__sub__", &Tensor::operator-)
        .def("__mul__", &Tensor::operator*)
        .def("__truediv__", &Tensor::operator/)
        .def("__repr__", [](const Tensor& t) {
            std::string repr = "Tensor(shape=[";
            for (size_t i = 0; i < t.shape().size(); ++i) {
                repr += std::to_string(t.shape()[i]) + (i + 1 < t.shape().size() ? ", " : "");
            }
            repr += "], numel=" + std::to_string(t.numel()) + ")";
            return repr;
        });
}