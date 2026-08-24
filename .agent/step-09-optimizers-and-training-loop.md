# Step 09 — Optimizers & the Training Loop

## Goal
Turn gradients into learning. This step is small in scope but it's the moment your framework can actually train something end-to-end.

## Ask yourself before starting
- What is the actual difference between plain SGD and something like Adam, conceptually — not the formula, but *why* the extra terms exist?
- Why must gradients be zeroed between training steps — what breaks if you forget?
- What's the minimum loop structure needed to train any model, regardless of architecture?

## Checklist
- [ ] Implement a basic SGD optimizer that updates parameters using their gradients
- [ ] Implement at least one more advanced optimizer (e.g. SGD with momentum, or Adam)
- [ ] Implement a learning rate as a configurable value
- [ ] Implement a standard training loop structure: forward pass, loss computation, backward pass, optimizer step, gradient zeroing
- [ ] Train a small model on a toy problem you can verify by hand (e.g. fitting a known linear function) and confirm the loss decreases
- [ ] Train the same toy problem on CPU and on GPU and confirm both converge to the same result
- [ ] Add basic logging/printing of loss over training steps so you can visually confirm learning is happening

> ## 🔴 CAUTION
> Do not move on to a "real" dataset or architecture until your toy problem's loss actually converges to the known correct answer. If it doesn't converge on something you can verify by hand, something upstream (autograd, layers) is still broken.

## Definition of done
You can train a small model end-to-end on a toy problem, on both CPU and GPU, and watch the loss decrease to the expected result.
