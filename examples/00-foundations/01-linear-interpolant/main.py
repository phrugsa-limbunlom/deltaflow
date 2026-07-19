"""
00-foundations/01-linear-interpolant: build the linear probability path and
inspect its boundary conditions.

Run: python examples/00-foundations/01-linear-interpolant/main.py
"""

import torch

from deltaflow.interpolants import LinearInterpolant


def main():
    torch.manual_seed(0)
    interpolant = LinearInterpolant()

    x1 = torch.randn(4, 2)  # "data"
    x0 = torch.randn(4, 2)  # "noise"

    for t_val in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = torch.full((4,), t_val)
        x_t, u_t = interpolant.interpolate(x1, t, x0=x0)
        print(f"t={t_val:.2f}  x_t[0]={x_t[0].tolist()}  u_t[0]={u_t[0].tolist()}")

    # Boundary conditions: x_t == x0 at t=0, x_t == x1 at t=1.
    x_t0, _ = interpolant.interpolate(x1, torch.zeros(4), x0=x0)
    x_t1, _ = interpolant.interpolate(x1, torch.ones(4), x0=x0)
    assert torch.allclose(x_t0, x0)
    assert torch.allclose(x_t1, x1)
    print("Boundary conditions OK.")


if __name__ == "__main__":
    main()
