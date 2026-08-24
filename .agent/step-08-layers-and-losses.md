# Step 08 — Neural Network Layers & Loss Functions

## Goal
With tensors + autograd working, layers are "just" compositions of differentiable operations with learnable parameters. This step is about API design as much as math.

## Ask yourself before starting
- What does a "layer" actually need to store, versus what should be computed fresh on every forward pass?
- How should learnable parameters be tracked so an optimizer can later find and update all of them?
- Why do activation functions need backward passes too, even though they have no learnable parameters?

## Checklist
- [ ] Design a base "Module" concept that layers will share (holds parameters, has a forward method)
- [ ] Implement a Linear (fully connected) layer with weight and bias parameters
- [ ] Implement common activation functions (e.g. ReLU, Sigmoid, Tanh) with correct backward passes
- [ ] Implement a way to compose layers into a larger model (sequential or otherwise)
- [ ] Implement at least one loss function (e.g. mean squared error or cross-entropy) with a correct backward pass
- [ ] Implement parameter initialization strategies (not just zeros — think about why zero-init is a bad idea)
- [ ] Implement a way to collect all parameters of a model (for the optimizer step later)
- [ ] Verify a full forward + backward pass through a small multi-layer model produces correct gradients (via numerical checking again)
- [ ] Confirm the same model runs correctly on both CPU and GPU tensors

> ## 🔴 CAUTION
> Do not hardcode a layer's behavior to only work on CPU or only on GPU. If Step 06 was done properly, layers shouldn't need to know or care which device their tensors live on.

## Definition of done
You can construct a small multi-layer network, run a forward pass, compute a loss, and call `.backward()` to get gradients for every parameter — on CPU and GPU.
