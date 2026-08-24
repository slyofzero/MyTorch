# Step 02 — Build System & Testing Infrastructure

## Goal
Before mixing CUDA into the codebase, your build system needs to gracefully handle "some files are C++, some are CUDA, and one day some will be Python." Retrofitting this later is painful — do it now while the codebase is still small.

## Ask yourself before starting
- How does a build system know to hand a `.cu` file to `nvcc` and a `.cpp` file to a regular compiler?
- What happens if someone builds this project on a machine with no GPU at all — should it still compile?

## Checklist
- [ ] Extend your build system to detect and compile `.cu` files alongside `.cpp` files
- [ ] Add a build flag/switch for "CPU-only" builds that exclude CUDA code entirely
- [ ] Set up continuous test running (a single command that builds and runs the full test suite)
- [ ] Add a benchmarking harness — something that times operations and reports throughput, separate from correctness tests
- [ ] Establish a convention for where CPU implementations live vs GPU implementations of the "same" operation
- [ ] Document (for yourself) the build commands in the README so future-you doesn't have to rediscover them

> ## 🔴 CAUTION
> Do not skip the "CPU-only build" switch even if you always have a GPU handy. You'll need it later to isolate whether a bug is in your logic or in your CUDA code.

## Definition of done
One command builds the whole project (CPU + GPU code together), one command runs all tests, and one command runs benchmarks.
