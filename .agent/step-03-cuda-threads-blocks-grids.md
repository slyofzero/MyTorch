# Step 03 — CUDA Fundamentals: Threads, Blocks, Grids

## Goal
This is your first real GPU programming step, deliberately kept separate from the tensor library. Learn the execution model in isolation, on toy problems, before you ask it to power a tensor op.

## Ask yourself before starting
- What is the actual difference between a thread, a block, and a grid — not the definition, but *why* GPUs are organized this way?
- If you launch a kernel with more threads than data elements, what happens to the extra threads?
- Why can't a kernel simply "return a value" the way a CPU function does?

## Checklist
- [ ] Write a kernel that fills an array with the thread's global index
- [ ] Write a kernel that adds two arrays elementwise, with correct bounds-checking for arbitrary array sizes
- [ ] Experiment with different block sizes and observe how performance changes — form a hypothesis about why
- [ ] Write a kernel operating on 2D data using 2D blocks/grids (e.g. filling or transposing a matrix)
- [ ] Deliberately write a kernel with an out-of-bounds bug, observe what happens, then fix it — understand what tool you used to catch it
- [ ] Learn and use a CUDA error-checking pattern for every kernel launch and memory operation
- [ ] Write your own short explanation (in your own words) of the relationship between threads, warps, blocks, and the SM (streaming multiprocessor)

> ## 🔴 CAUTION
> Do not connect this to your Tensor class yet. Keep these experiments in scratch files — the goal here is intuition, not integration.

## Definition of done
You can write, launch, and debug a simple CUDA kernel from scratch without looking anything up mid-way, and you can explain the thread/block/grid hierarchy to someone else without notes.
