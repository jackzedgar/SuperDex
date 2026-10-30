"""Focused mathematical checks before SuperDex implementation; no policy training.

Run in the already verified Isaac Gym PyTorch environment, on physical GPU 0.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import uuid

EXPECTED_GPU = "GPU-810d1bb5-dcdc-3d20-c3c5-22bffb1fcf9d"
if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
    raise RuntimeError("Legacy PyTorch requires numeric CUDA_VISIBLE_DEVICES=0 and PCI_BUS_ID order")
# PyTorch 1.13's Python device-count path does not handle UUID masks reliably.
# Verify the numeric mapping against the actual CUDA driver UUID before allocation.
driver = ctypes.CDLL("libcuda.so.1")
assert driver.cuInit(0) == 0
cuda_device = ctypes.c_int()
assert driver.cuDeviceGet(ctypes.byref(cuda_device), 0) == 0
cuda_uuid = ctypes.create_string_buffer(16)
assert driver.cuDeviceGetUuid(cuda_uuid, cuda_device) == 0
assert "GPU-" + str(uuid.UUID(bytes=cuda_uuid.raw)) == EXPECTED_GPU

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1
    device, dtype = "cuda:0", torch.float64
    checks = {}

    # Freezing parameters must preserve action Jacobians through the base model.
    base = torch.nn.Linear(2, 2, bias=False).to(device=device, dtype=dtype)
    residual = torch.nn.Linear(2, 2, bias=False).to(device=device, dtype=dtype)
    with torch.no_grad():
        base.weight.copy_(2 * torch.eye(2, device=device, dtype=dtype))
        residual.weight.zero_()
    for p in base.parameters():
        p.requires_grad_(False)
    a = torch.tensor([[0.3, 0.7]], dtype=dtype, device=device, requires_grad=True)
    y = base(a) + residual(a)
    grad = torch.autograd.grad(y.sum(), a)[0]
    assert torch.allclose(grad, torch.full_like(a, 2))
    bad = base(a).detach() + residual(a)
    bad_grad = torch.autograd.grad(bad.sum(), a)[0]
    assert torch.equal(bad_grad, torch.zeros_like(a))
    assert all(p.grad is None for p in base.parameters())
    checks["frozen_parameters_preserve_input_gradient"] = "PASS"
    checks["detached_base_counterexample"] = dict(correct=grad.tolist(), detached=bad_grad.tolist())

    # One scalar interpolation defines both forward values and backward derivatives.
    coords = torch.linspace(-1, 1, 9, dtype=dtype, device=device)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    volume = (x + 2*y - 0.5*z)[None, None]

    def field(points):
        projected = points.clamp(-1, 1)
        grid = projected.reshape(1, 1, 1, -1, 3)
        values = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border",
                               align_corners=True).reshape(-1)
        return values + torch.linalg.vector_norm(points - projected, dim=-1)

    pts = torch.tensor([[0.12, 0.31, -0.17], [1.4, 0.31, -0.17]],
                       dtype=dtype, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(field, (pts,), eps=1e-6, atol=1e-5, rtol=1e-4)
    grads = torch.autograd.grad(field(pts).sum(), pts)[0]
    expected = torch.tensor([[1., 2., -0.5], [1., 2., -0.5]], dtype=dtype, device=device)
    assert torch.allclose(grads, expected, atol=1e-10)
    checks["scalar_grid_forward_backward_finite_difference"] = "PASS"
    checks["outside_box_keeps_tangential_derivatives"] = grads[1].tolist()

    # End-to-end 32-step gradient against numerical differentiation.
    initial = torch.tensor(0.3, device=device, dtype=dtype)
    def objective(theta):
        state = initial
        total = torch.zeros((), device=device, dtype=dtype)
        for h in range(32):
            action = theta * state
            state = 0.98 * state + 0.1 * action
            total = total - (0.99 ** h) * (state - 0.7).square()
        return total - 0.99 ** 32 * (state - 0.7).square()
    theta = torch.tensor(0.2, device=device, dtype=dtype, requires_grad=True)
    assert torch.autograd.gradcheck(objective, (theta,), eps=1e-6, atol=1e-5, rtol=1e-4)
    checks["32_step_bptt_finite_difference"] = "PASS"

    # A terminal transition earns its immediate reward, but no later reward/value.
    rewards = torch.tensor([1., 100., 100.], device=device, dtype=dtype)
    done = torch.tensor([1., 0., 0.], device=device, dtype=dtype)
    survival = torch.cat([torch.ones(1, device=device, dtype=dtype), (1-done).cumprod(0)])
    value = (rewards * survival[:-1] * 0.99 ** torch.arange(3, device=device)).sum()
    value += 0.99 ** 3 * survival[-1] * 1000
    assert value.item() == 1.0
    checks["termination_reward_and_bootstrap_indexing"] = "PASS"
    result = dict(status="PASS", scope="synthetic mathematical regression checks; not G1/G2",
                  gpu_uuid=EXPECTED_GPU, torch=torch.__version__, checks=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
