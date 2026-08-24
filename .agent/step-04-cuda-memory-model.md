# Step 04 — CUDA Memory Model: Global, Shared, and Coalescing

## Goal
Same trick as Step 03 — isolate the concept before applying it. Memory access patterns are where naive GPU code goes to die in terms of performance, and where most of the "why is my kernel slow" intuition comes from.

## Ask yourself before starting
- Why does the *pattern* in which threads access memory matter, not just *how much* memory they access?
- What is shared memory actually shared *among* — all threads on the GPU, or something narrower?
- Why would you ever copy data into shared memory instead of reading it directly from global memory?

## Checklist
- [ ] Write a kernel that deliberately accesses global memory in a coalesced pattern, and one that deliberately accesses it in a strided/non-coalesced pattern
- [ ] Benchmark both and confirm (with numbers, not assumption) that coalescing matters
- [ ] Write a kernel that uses shared memory to cache a tile of data before computing on it
- [ ] Implement a classic shared-memory example: a tiled matrix multiplication kernel
- [ ] Benchmark your tiled matmul against your naive global-memory matmul kernel
- [ ] Learn about bank conflicts in shared memory and try to construct an example that triggers one
- [ ] Learn the difference between global, shared, local, constant, and register memory — write a short comparison in your own words

> ## 🔴 CAUTION
> Do not skip writing the "bad" (non-coalesced) version on purpose. Seeing the slow version with your own benchmark numbers is what makes the fast version make sense — don't just take it on faith.

## Definition of done
You have a working, benchmarked tiled matrix multiplication kernel, and concrete numbers proving coalesced access and shared memory usage improve performance over the naive version.
