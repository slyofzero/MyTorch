# Step 06 — GPU Tensor Backend & Device Abstraction

## Goal
Now merge Steps 01–05: take your CPU tensor library and give it a GPU-backed twin, plus a clean way to move data between them. This is where "CPU vs GPU" stops being an exercise and becomes an actual design decision in your codebase.

## Ask yourself before starting
- What should a tensor "know" about which device it lives on?
- When you call an operation on a tensor, how should the code decide whether to run the CPU or GPU implementation?
- What does it mean to "move" a tensor to the GPU — is data copied, or something else?

## Checklist
- [ ] Design a `Device` concept that a tensor carries (e.g. CPU vs GPU, and which GPU if multiple)
- [ ] Implement GPU-backed storage — allocating and freeing device memory correctly, with no leaks
- [ ] Implement a `.to(device)` / `.cuda()` / `.cpu()` style operation that copies a tensor's data across devices
- [ ] Reimplement your Step 01 operations (elementwise ops, broadcasting, reductions, matmul) as CUDA kernels operating on GPU storage
- [ ] Design the dispatch mechanism that routes a tensor operation to the CPU or GPU implementation based on the tensor's device
- [ ] Write tests that create identical tensors on CPU and GPU, run the same operation, and assert the results match
- [ ] Add a guardrail: operations between tensors on different devices should fail with a clear error, not silently do something wrong
- [ ] Benchmark the same operation on CPU vs GPU across a range of tensor sizes, and note at what size the GPU actually wins

> ## 🔴 CAUTION
> Do not let a single operation "sometimes" support GPU and "sometimes" not without an explicit, deliberate error message. Silent CPU fallback is exactly the kind of bug that's invisible until it's expensive.

## Definition of done
The same tensor API works identically whether the tensor lives on CPU or GPU, moving between devices works correctly, and you have benchmark data showing where the crossover point is.
