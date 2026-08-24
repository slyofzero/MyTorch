# Step 07 — Autograd Engine

## Goal
This is the conceptual heart of a deep learning framework. Everything before this was "just" a tensor library — this step is what makes it a *learning* system.

## Ask yourself before starting
- What is a computational graph, and when is it actually built — before or during the forward pass?
- If a tensor is used in two different operations, how should gradients from both be combined?
- Why does every differentiable operation need to know its own derivative, not just its own forward computation?

## Checklist
- [ ] Design how a tensor will track its history (what operation created it, from what inputs)
- [ ] Implement a `requires_grad` flag and a `.grad` field on tensors
- [ ] Implement the forward+backward pair for elementwise addition, multiplication, and division
- [ ] Implement the forward+backward pair for matrix multiplication
- [ ] Implement the forward+backward pair for reductions (sum, mean)
- [ ] Implement the forward+backward pair for broadcasting (this is the trickiest one — think about what "undoing" a broadcast means for gradients)
- [ ] Implement topological sort / traversal of the computational graph for the backward pass
- [ ] Implement gradient accumulation for tensors used multiple times in a graph
- [ ] Implement a `.backward()` entry point that triggers the full backward traversal from a scalar output
- [ ] Implement gradient zeroing/resetting between training steps
- [ ] Write numerical gradient checking (finite differences) and validate your analytical gradients against it for every operation above
- [ ] Confirm autograd works correctly for tensors on both CPU and GPU

> ## 🔴 CAUTION
> Do not trust an analytical gradient implementation just because the forward pass looks right. Numerically verify every single backward function before moving on — this is the step where silent bugs are most expensive later.

## Definition of done
You can build an arbitrary small expression out of your tensors, call `.backward()`, and get gradients that match numerical gradient checks, on both CPU and GPU.
