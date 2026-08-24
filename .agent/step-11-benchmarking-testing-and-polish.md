# Step 11 — Benchmarking, Testing & Polish

## Goal
The framework works — now find out how well, where it's weak, and make it something you'd trust to hand to someone else.

## Ask yourself before starting
- Where do you expect your framework to be dramatically slower than real PyTorch, and why?
- What's the difference between a correctness bug and a performance gap — and how would you tell them apart from a bad benchmark result alone?
- What would someone else need to know to use this without reading your source code?

## Checklist
- [ ] Build a benchmark suite comparing your framework's core operations against real PyTorch on the same hardware
- [ ] Identify the largest performance gaps and form a hypothesis for each (don't have to fix all of them)
- [ ] Pick at least one identified gap and attempt an optimization pass, then re-benchmark to confirm improvement
- [ ] Expand test coverage to edge cases: empty tensors, mismatched shapes, single-element tensors, very large tensors
- [ ] Stress-test memory management (repeated allocation/deallocation on GPU) for leaks
- [ ] Write end-to-end documentation: what the project is, what it supports, what it doesn't, how to build and run it
- [ ] Write a short retrospective (in your own words) on which GPU programming concept was hardest to internalize and why
- [ ] Decide and document what you would build next if you continued this project

> ## 🔴 CAUTION
> Do not treat "close to PyTorch's speed" as the success bar. Real PyTorch has years of optimization behind it — the goal here was understanding, and the benchmark numbers are a diagnostic tool, not a scoreboard.

## Definition of done
You have benchmark data comparing your framework to PyTorch, a documented understanding of where and why the gaps exist, and documentation someone else could follow to build and use your project.
