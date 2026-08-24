# Step 01 — CPU Tensor Core

## Goal
Build a Tensor type that lives entirely on CPU, with no autograd yet. This forces you to understand what a tensor *is* before you worry about where it runs or how it learns.

## Ask yourself before starting
- Is a tensor its data, or is it a *view* over data? What's the difference between a tensor and its underlying storage?
- What are strides, and why do they let you reshape/transpose without copying memory?
- What should happen when you add two tensors of different shapes?

## Checklist
- [ ] Design and implement a `Storage` concept — the raw buffer that owns memory
- [ ] Design and implement a `Tensor` concept — shape, strides, dtype, and a reference to storage
- [ ] Implement tensor creation: from raw data, filled with zeros/ones, random
- [ ] Implement indexing and slicing without copying underlying data
- [ ] Implement reshape, transpose, and view operations correctly using strides
- [ ] Implement elementwise operations (add, subtract, multiply, divide)
- [ ] Implement broadcasting rules for elementwise operations
- [ ] Implement reduction operations (sum, mean, max) along arbitrary dimensions
- [ ] Implement matrix multiplication (naive is fine for now)
- [ ] Write a test suite that checks your results against known correct values by hand (not against real PyTorch yet — do the math yourself first)
- [ ] Only after your own tests pass: cross-check a handful of results against real PyTorch outputs

> ## 🔴 CAUTION
> Do not reach for CUDA in this step, even if you're impatient. A wrong CPU tensor will produce a wrong GPU tensor twice as fast — get correctness first.

## Definition of done
You have a CPU-only tensor library with elementwise ops, broadcasting, reductions, and matmul, all independently verified.
