# Inverse Problems

DeltaFlow can turn a pretrained *unconditional* velocity field into a
**posterior sampler** for inverse problems (reconstructing a signal \(x\) from
a noisy measurement \(y = A(x) + \varepsilon\)) without retraining the
backbone.

## Design: wrap, don't rewrite

`PosteriorSolver` **wraps** an existing ODE solver (Euler, Heun, ...) and
injects a measurement-likelihood gradient at every step, so the base
integrator is reused. This follows FlowDPS and Flower: both modify the
sampling-time ODE and never touch the pretrained field.

Per step, given state \(x_t\) and velocity \(v_t = v_\theta(x_t, t)\):

\[
\hat{x}_0 = \text{Tweedie}(x_t, v_t, t), \qquad
g = \nabla_{x_t}\big[-\log p(y \mid \hat{x}_0)\big],
\]
\[
x_{t+\mathrm{d}t} = \text{BaseSolver.step}(x_t, t, \mathrm{d}t) - \eta\, g,
\]

with \(\eta\) the `guidance_scale`. The gradient is computed by autograd.

## Usage

```python
from deltaflow.solvers import EulerSolver, PosteriorSolver
from deltaflow.inverse import GaussianLikelihood, BlurOperator

operator = BlurOperator(kernel_size=9, sigma=2.0)
likelihood = GaussianLikelihood(operator=operator, y=measurement, sigma=0.05)

solver = PosteriorSolver(
    base_solver=EulerSolver(model),
    likelihood=likelihood,
    guidance_scale=1.0,
)
recon = solver.sample(torch.randn_like(x_init), n_steps=100)
```

## Measurement operators

[`deltaflow.inverse`](../api/inverse.md) ships common linear operators:

| Operator | Task |
|---|---|
| `IdentityOperator` | Denoising |
| `MaskOperator` | Inpainting |
| `BlurOperator` | Deblurring |
| `DownsampleOperator` | Super-resolution |

## Latent-space problems

If the velocity field operates on VAE latents while the measurement operator is
defined on pixels, pass a decoder to the likelihood object: the gradient is
pulled back through the decoder automatically, no extra bookkeeping.

## Tweedie decomposition

The map \((x_t, v_t, t) \mapsto \hat{x}_0\) is provided by
[`BaseTweedie`](../api/inverse.md). Use `LinearTweedie` for a linear-interpolant
(rectified-flow) backbone and `VPTweedie` for a variance-preserving one. It
must match the training path.
