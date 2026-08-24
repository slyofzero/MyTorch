# Step 10 — Python Bindings

## Goal
Make your C++/CUDA framework usable from Python, the way real PyTorch is. This is where you learn *why* PyTorch is Python on top of C++, not Python all the way down.

## Ask yourself before starting
- What has to happen for a Python object to "point to" memory that was allocated in C++/CUDA?
- Why would a framework choose to write performance-critical code in C++/CUDA but expose it through Python instead of just writing everything in Python?
- What should happen to GPU memory when a Python tensor object is garbage collected?

## Checklist
- [ ] Choose and set up a Python binding tool for your C++ library
- [ ] Expose your Tensor class to Python, including creation, basic ops, and printing/representation
- [ ] Expose device movement (`.cpu()`, `.cuda()`) to Python
- [ ] Expose autograd functionality to Python (`requires_grad`, `.backward()`, `.grad`)
- [ ] Expose your layers/modules and optimizers to Python
- [ ] Confirm memory is correctly managed — no leaks and no crashes when Python tensors are created and destroyed repeatedly
- [ ] Write a pure-Python script that builds a small model, trains it on a toy problem, using only your library (no real PyTorch)
- [ ] Package the project so it can be installed and imported like a normal Python package
- [ ] Write a couple of small Python-side tests confirming the bindings behave correctly (not re-testing the C++ logic, just the binding layer itself)

> ## 🔴 CAUTION
> Do not let Python silently hide a C++/CUDA error. Make sure exceptions and errors from the C++ layer propagate up as real, readable Python errors — a segfault with no message is not acceptable here.

## Definition of done
You can `import` your framework in a plain Python script, build a model, and train it end-to-end using only Python-facing code, with no direct C++/CUDA calls needed.
