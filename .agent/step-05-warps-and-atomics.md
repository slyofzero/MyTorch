# Step 05 — Warp-Level Programming & Atomics

## Goal
The last "pure GPU concepts" step before you go build the actual tensor backend. Warps and atomics matter most in reductions and anywhere multiple threads must safely touch the same memory.

## Ask yourself before starting
- What is a warp, and why does it mean threads within it execute in lockstep?
- What happens if two threads write to the same memory location at the same time without any coordination?
- Why is a naive parallel sum (reduction) harder to get right than a parallel elementwise add?

## Checklist
- [ ] Write a kernel that demonstrates warp divergence (threads in a warp taking different branches) and observe its performance cost
- [ ] Implement a parallel reduction (e.g. sum of an array) using shared memory, without atomics, handling arbitrary array sizes
- [ ] Implement the same reduction using atomic operations instead, and compare
- [ ] Benchmark both reduction approaches and explain the performance difference in your own words
- [ ] Explore at least one warp-level primitive (e.g. shuffle-based communication between threads in a warp) and use it in a small kernel
- [ ] Identify one place in your future tensor library where atomics will likely be necessary (don't implement it yet — just identify it)

> ## 🔴 CAUTION
> Do not reach for atomics as your default tool for every parallel problem. Try the non-atomic version first each time — atomics are often a last resort, not a starting point.

## Definition of done
You have working, benchmarked reduction kernels (both non-atomic and atomic versions) and can explain warp divergence and race conditions from first principles.
