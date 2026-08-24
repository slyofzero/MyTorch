# Step 00 — Foundations & Environment Setup

## Goal
Before any tensor or kernel exists, your toolchain, repo, and mental model of "what am I even building" need to exist. This step has no ML and no GPU code — it's the workbench.

## Ask yourself before starting
- What does it mean for code to be "compiled for a GPU" vs "compiled for a CPU"?
- Why does PyTorch need C++/CUDA at all if Python already works?

> ## 🔴 CAUTION
> Do not write a single tensor or kernel line in this step. If you catch yourself opening a `.cu` file, stop — this step is scaffolding only.

## Checklist
- [ ] Confirm GPU + driver + CUDA toolkit are installed and a sample NVIDIA CUDA program compiles and runs on your machine
- [ ] Set up a repo structure that separates: core library, CUDA kernels, Python bindings, tests
- [ ] Set up a build system (decide and justify your choice — don't just default without understanding trade-offs)
- [ ] Write a "hello world" C++ program and confirm it builds through your chosen build system
- [ ] Write a "hello world" CUDA program (a kernel that does nothing but print from a thread) and confirm it runs
- [ ] Set up a testing framework for C++/CUDA code
- [ ] Set up version control hygiene: `.gitignore` for build artifacts, a README describing the project's purpose
- [ ] Write a one-page personal doc (in your own words) answering: "What is the minimum set of things a deep learning framework must do?"

## Definition of done
You can build and run a trivial C++ and a trivial CUDA program from a clean checkout, and you've written your own framing of the problem before looking at how PyTorch actually solves it.
