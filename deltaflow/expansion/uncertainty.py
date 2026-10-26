"""Verifier-uncertainty estimators over a flow representation.

ActFlow treats the black-box verifier as an unknown validity function on the
representation space ``Z_s`` and queries where that function is least
determined. The closed-form estimator of the paper (its Eq. 10) is the
posterior standard deviation of kernel (Bayesian linear) regression,

    sigma_t^2(z) = k(z, z) - k(z, Z_t) (K_t + lambda I)^{-1} k(Z_t, z),

which depends only on *where* the verifier has been queried, not on the
labels it returned. Queried regions therefore become certain, and the
exploration tilt is pushed toward the unexplored frontier of the generable
set.

Any object that subclasses `BaseUncertainty` is a drop-in replacement (for
example a bootstrapped MLP ensemble, which the paper uses for the molecular
experiments).
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch

from ..utils.numerical import get_eps, safe_sqrt


class BaseUncertainty(ABC):
    """Base class for an uncertainty score ``sigma(z)`` over features ``z``.

    Subclasses implement `fit` (refit from the full query history) and
    `__call__` (a per-sample score, differentiable in ``z`` so it can drive
    sampling-time guidance).
    """

    @abstractmethod
    def fit(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> None:
        """Refit on queried features ``[t, D]`` and their verifier labels ``[t]``."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Return the uncertainty ``sigma(z)`` with shape ``[B]``."""
        raise NotImplementedError


class KernelUncertainty(BaseUncertainty):
    r"""Posterior standard deviation of kernel regression (ActFlow Eq. 10).

    Given queried features \(Z_t = (z_1, \dots, z_t)\) with kernel matrix
    \((K_t)_{ij} = k(z_i, z_j)\), the uncertainty of a new point is

    \[
    \sigma_t^2(z) = k(z, z) - k(z, Z_t)\,\bigl(K_t + \lambda I\bigr)^{-1}
        k(Z_t, z), \qquad \sigma_t(z) = \sqrt{\sigma_t^2(z)}.
    \]

    This is the posterior variance of Bayesian linear regression under
    Gaussian observations in the feature space of \(k\). With no queries it
    reduces to the prior \(k(z, z)\).

    For the linear kernel \(k(z, z') = \langle z, z' \rangle\) the push-through
    identity gives the equivalent primal form

    \[
    \sigma_t^2(z) = \lambda\, z^\top \bigl(Z_t^\top Z_t + \lambda I_D\bigr)^{-1} z,
    \]

    which costs \(\mathcal{O}(D^3)\) instead of \(\mathcal{O}(t^3)\). It is
    selected automatically once \(t > D\).

    The score is differentiable in \(z\) (the stored Cholesky factor is a
    constant), so it can be back-propagated through `FlowRepresentation`
    to the sampling state. Linear algebra runs in float64 for stability and
    the result is cast back to the input dtype.

    References:
        De Santi et al., "Active Flow Expansion for Out-of-Distribution
        Discovery: from Theory to Molecules" (2026),
        https://arxiv.org/abs/2606.08802. Rasmussen and Williams, "Gaussian
        Processes for Machine Learning" (2006), Ch. 2.

    Args:
        kernel: ``"linear"`` (the paper's default) or ``"rbf"``
            (\(k(z, z') = \exp(-\lVert z - z' \rVert^2 / 2\ell^2)\), used in the
            paper's 2D experiment).
        reg: regulariser \(\lambda > 0\) (observation-noise variance).
        lengthscale: RBF lengthscale \(\ell\). Ignored for the linear kernel.
        max_points: if set, only the most recent ``max_points`` queries are
            kept, which bounds the \(\mathcal{O}(t^3)\) refit cost.
    """

    def __init__(
        self,
        kernel: str = "linear",
        reg: float = 1.0,
        lengthscale: float = 1.0,
        max_points: Optional[int] = None,
    ):
        if kernel not in ("linear", "rbf"):
            raise ValueError(f"kernel must be 'linear' or 'rbf', got {kernel!r}")
        if reg <= 0:
            raise ValueError(f"reg must be positive, got {reg}")
        self.kernel = kernel
        self.reg = float(reg)
        self.lengthscale = float(lengthscale)
        self.max_points = max_points
        self._z: Optional[torch.Tensor] = None
        self._y: Optional[torch.Tensor] = None
        self._chol: Optional[torch.Tensor] = None
        self._primal = False

    @property
    def n_points(self) -> int:
        return 0 if self._z is None else self._z.shape[0]

    def _k(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.kernel == "linear":
            return a @ b.T
        sq = torch.cdist(a, b).pow(2)
        return torch.exp(-0.5 * sq / (self.lengthscale**2))

    def _k_diag(self, a: torch.Tensor) -> torch.Tensor:
        if self.kernel == "linear":
            return a.pow(2).sum(-1)
        # k(z, z) = 1 for the RBF kernel, written so it stays in z's autograd graph
        return torch.exp(-0.5 * (a - a).pow(2).sum(-1) / (self.lengthscale**2))

    def fit(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> None:
        z = features.detach().to(torch.float64).flatten(1)
        y = None if labels is None else labels.detach().to(torch.float64).flatten()
        if self.max_points is not None and z.shape[0] > self.max_points:
            z = z[-self.max_points :]
            y = None if y is None else y[-self.max_points :]
        if z.shape[0] == 0:
            self._z, self._y, self._chol = None, None, None
            return
        self._z, self._y = z, y
        self._primal = self.kernel == "linear" and z.shape[0] > z.shape[1]
        if self._primal:
            gram = z.T @ z
        else:
            gram = self._k(z, z)
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        self._chol = torch.linalg.cholesky(gram + self.reg * eye)

    def variance(self, features: torch.Tensor) -> torch.Tensor:
        """Return \\(\\sigma_t^2(z)\\) with shape ``[B]``."""
        z = features.flatten(1).to(torch.float64)
        if self._z is None:
            return self._k_diag(z).to(features.dtype)
        z_ref = self._z.to(z.device)
        chol = self._chol.to(z.device)
        if self._primal:
            # sigma^2 = lambda * z^T (Z^T Z + lambda I)^{-1} z
            w = torch.cholesky_solve(z.T, chol)
            var = self.reg * (z.T * w).sum(0)
        else:
            k_star = self._k(z_ref, z)
            w = torch.cholesky_solve(k_star, chol)
            var = self._k_diag(z) - (k_star * w).sum(0)
        return var.to(features.dtype)

    def mean(self, features: torch.Tensor) -> torch.Tensor:
        """Posterior mean of the label regression, a cheap validity surrogate.

        Returns zeros if no labels were provided to `fit`.
        """
        z = features.flatten(1).to(torch.float64)
        if self._z is None or self._y is None:
            return torch.zeros(z.shape[0], dtype=features.dtype, device=features.device)
        z_ref = self._z.to(z.device)
        y = self._y.to(z.device)
        chol = self._chol.to(z.device)
        if self._primal:
            w = torch.cholesky_solve((z_ref.T @ y).unsqueeze(-1), chol).squeeze(-1)
            mu = z @ w
        else:
            alpha = torch.cholesky_solve(y.unsqueeze(-1), chol).squeeze(-1)
            mu = self._k(z, z_ref) @ alpha
        return mu.to(features.dtype)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return safe_sqrt(self.variance(features), eps=get_eps(features.dtype))


__all__ = ["BaseUncertainty", "KernelUncertainty"]
