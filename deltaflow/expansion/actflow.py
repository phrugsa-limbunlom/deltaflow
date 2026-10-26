"""Active Flow Expansion (ActFlow), the continued pre-training loop.

ActFlow (De Santi et al., 2026) grows the *generable set* of a pre-trained
flow model, the region it covers with non-negligible density,

    Omega_theta^tau = { x : p_1^theta(x) >= tau },

toward the full valid design space ``Omega*`` defined by a black-box
verifier, rather than only matching the training distribution. Each round of
Algorithm 1 is

1. fit the verifier uncertainty ``sigma_t`` on the query history ``D_t``,
   using features ``phi_s`` read from the *current* model,
2. self-generate a batch from the uncertainty tilt
   ``p_theta * exp(sigma_t / beta)`` (active exploration in ``Z_s``),
3. query the verifier and append the labelled batch to ``D_t``,
4. update the flow by signed replay on accepted and rejected samples.

In the displacement view of DeltaFlow, plain self-training only re-learns
the displacement field toward modes the model already reaches. ActFlow
chooses *which* endpoints the field should be retrained to reach, steering
self-generation to the uncertain frontier of the generable set so the learned
transport extends into new valid regions.

Every stage is a swappable DeltaFlow component, the sampler is
`PosteriorSolver` wrapping any base solver, the tilt is
`UncertaintyGuidance`, the uncertainty is any `BaseUncertainty`, and the
update is `SignedReplayFlowMatchingLoss`.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..core.base_solver import BaseSolver
from ..inverse.tweedie import BaseTweedie
from ..losses.signed_replay import SignedReplayFlowMatchingLoss
from ..solvers.euler import EulerSolver
from ..solvers.posterior_solver import PosteriorSolver
from .guidance import UncertaintyGuidance
from .representation import FlowRepresentation
from .uncertainty import BaseUncertainty, KernelUncertainty


@dataclass
class ActFlowRound:
    """Summary of one ActFlow round.

    Attributes:
        round: zero-based round index.
        samples: the self-generated queries of this round.
        labels: their verifier labels (bool, ``True`` means accepted).
        acceptance_rate: fraction of this round's queries accepted.
        mean_uncertainty: mean \\(\\sigma_t(\\phi_s(x))\\) of the queries under the
            uncertainty fitted at the start of the round.
        n_accepted: size of the accepted buffer after the round.
        n_rejected: size of the rejected buffer after the round.
        finetuned: whether the flow was updated (``False`` during warm-up).
        loss: last signed-replay loss of the update, if any.
        alpha: last calibrated repulsion weight \\(\\alpha_t\\).
    """

    round: int
    samples: torch.Tensor
    labels: torch.Tensor
    acceptance_rate: float
    mean_uncertainty: float
    n_accepted: int
    n_rejected: int
    finetuned: bool
    loss: Optional[float]
    alpha: float


class ActFlow:
    r"""Active Flow Expansion, verifier-guided growth of a flow's generable set.

    Runs Algorithm 1 of De Santi et al. (2026). At round \(t\),

    \[
    \begin{aligned}
    &\sigma_t \leftarrow \textsc{Fit}\bigl(\{\phi^{\theta_t}_s(x_i)\}_{i \le t}\bigr), \\
    &x_{t+1} \sim \tilde{p}_t \propto p_1^{\theta_t}\,
        \exp\!\bigl(\sigma_t(\phi^{\theta_t}_s(x)) / \beta\bigr), \\
    &y_{t+1} \leftarrow \tilde{v}(x_{t+1}), \qquad
        \mathcal{D}_{t+1} \leftarrow \mathcal{D}_t \cup \{(x_{t+1}, y_{t+1})\}, \\
    &\theta_{t+1} \leftarrow \theta_t - \eta\,\bigl(\nabla \hat{\mathcal{L}}^{+}_t
        - \alpha_t \nabla \hat{\mathcal{L}}^{-}_t\bigr)
        \quad (\text{repeated } \texttt{finetune\_steps} \text{ times}).
    \end{aligned}
    \]

    The representation \(\phi^{\theta_t}_s\) is re-read from the current model
    every round, so the uncertainty refit uses the model's evolving features,
    as in the paper. ``beta=math.inf`` turns off the tilt and gives the
    self-training baseline (REC-F in the paper when ``alpha=0``).

    References:
        De Santi et al., "Active Flow Expansion for Out-of-Distribution
        Discovery: from Theory to Molecules" (2026),
        https://arxiv.org/abs/2606.08802.

    Args:
        model: pre-trained velocity field (an ``nn.Module``), fine-tuned in
            place.
        verifier: black-box validity oracle ``verifier(x) -> [B]`` returning
            bools or ``{0, 1}`` (values above ``0.5`` count as accepted). It is
            called without gradients and may be non-differentiable.
        optimizer: optimizer over ``model``'s parameters.
        sample_shape: shape of one design without the batch axis, e.g.
            ``(2,)`` or ``(1, 32, 32)``. The noise prior is
            \(\mathcal{N}(0, I)\) of this shape.
        representation: feature map \(\phi_s\). Defaults to
            ``FlowRepresentation(model, s=0.9)`` on the model output. Pass a
            ``layer`` to read hidden features.
        uncertainty: verifier-uncertainty estimator. Defaults to the paper's
            linear-kernel `KernelUncertainty`.
        loss: continued pre-training objective. Defaults to
            ``SignedReplayFlowMatchingLoss(alpha=0.0)``, i.e. flow matching on
            accepted samples.
        base_solver: integrator wrapped by `PosteriorSolver`. Defaults to
            `EulerSolver`.
        tweedie: Tweedie decomposition matching the training path. Defaults
            to `LinearTweedie` (inside `PosteriorSolver`).
        beta: KL weight \(\beta\). Smaller explores harder.
        guidance_scale: step size on the tilt gradient (``PosteriorSolver``'s
            ``guidance_scale``).
        grad_normalize: forwarded to `PosteriorSolver`.
        n_steps: ODE steps per self-generated batch.
        finetune_steps: gradient steps of the flow update per round.
        batch_size: minibatch size drawn from each replay buffer.
        min_accepted: warm-up, the flow is not updated until the accepted
            buffer holds at least this many samples.
        feature_batch_size: chunk size for reading features of the history.
        cond: conditioning forwarded unchanged to the model in sampling,
            features, and the loss. It must not depend on the batch size.
    """

    def __init__(
        self,
        model: nn.Module,
        verifier: Callable[[torch.Tensor], Any],
        optimizer: torch.optim.Optimizer,
        sample_shape: Sequence[int],
        representation: Optional[FlowRepresentation] = None,
        uncertainty: Optional[BaseUncertainty] = None,
        loss: Optional[SignedReplayFlowMatchingLoss] = None,
        base_solver: Optional[BaseSolver] = None,
        tweedie: Optional[BaseTweedie] = None,
        beta: float = 0.1,
        guidance_scale: float = 1.0,
        grad_normalize: bool = False,
        n_steps: int = 50,
        finetune_steps: int = 100,
        batch_size: int = 64,
        min_accepted: int = 0,
        feature_batch_size: int = 1024,
        cond: Optional[Dict[str, Any]] = None,
    ):
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.model = model
        self.verifier = verifier
        self.optimizer = optimizer
        self.sample_shape = tuple(sample_shape)
        self.representation = representation or FlowRepresentation(model)
        self.uncertainty = uncertainty or KernelUncertainty()
        self.loss = loss or SignedReplayFlowMatchingLoss()
        self.base_solver = base_solver or EulerSolver(model)
        self.tweedie = tweedie
        self.beta = float(beta)
        self.guidance_scale = guidance_scale
        self.grad_normalize = grad_normalize
        self.n_steps = n_steps
        self.finetune_steps = finetune_steps
        self.batch_size = batch_size
        self.min_accepted = min_accepted
        self.feature_batch_size = feature_batch_size
        self.cond = dict(cond or {})

        try:
            ref = next(model.parameters())
            self.device, self.dtype = ref.device, ref.dtype
        except StopIteration:
            self.device, self.dtype = torch.device("cpu"), torch.get_default_dtype()

        empty = torch.empty(0, *self.sample_shape, device=self.device, dtype=self.dtype)
        self.queries = empty
        self.labels = torch.empty(0, dtype=torch.bool, device=self.device)
        self.history: List[ActFlowRound] = []

    @property
    def accepted(self) -> torch.Tensor:
        """Accepted replay buffer \\(\\mathcal{D}^{+}_t\\)."""
        return self.queries[self.labels]

    @property
    def rejected(self) -> torch.Tensor:
        """Rejected replay buffer \\(\\mathcal{D}^{-}_t\\)."""
        return self.queries[~self.labels]

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Read \\(\\phi_s(x)\\) from the current model in chunks."""
        chunks = [
            self.representation(x[i : i + self.feature_batch_size], **self.cond)
            for i in range(0, x.shape[0], self.feature_batch_size)
        ]
        return torch.cat(chunks) if chunks else x.new_empty(0, 0)

    def fit_uncertainty(self) -> None:
        """Refit \\(\\sigma_t\\) on the full history with the current features."""
        was_training = self.model.training
        self.model.eval()
        try:
            feats = self.features(self.queries)
        finally:
            self.model.train(was_training)
        self.uncertainty.fit(feats, self.labels.to(feats.dtype))

    def guidance(self) -> UncertaintyGuidance:
        """The uncertainty tilt for the current \\(\\sigma_t\\)."""
        return UncertaintyGuidance(self.representation, self.uncertainty, self.beta, self.cond)

    def self_generate(self, n: int) -> torch.Tensor:
        """Draw ``n`` designs from the uncertainty-tilted model (Eq. 9)."""
        x0 = torch.randn(n, *self.sample_shape, device=self.device, dtype=self.dtype)
        was_training = self.model.training
        self.model.eval()
        try:
            if math.isinf(self.beta):
                x = self.base_solver.sample(
                    x0, n_steps=self.n_steps, show_progress=False, **self.cond
                )
            else:
                solver = PosteriorSolver(
                    base_solver=self.base_solver,
                    likelihood=self.guidance(),
                    tweedie=self.tweedie,
                    guidance_scale=self.guidance_scale,
                    grad_normalize=self.grad_normalize,
                )
                x = solver.sample(x0, n_steps=self.n_steps, show_progress=False, **self.cond)
        finally:
            self.model.train(was_training)
        return x.detach()

    @torch.no_grad()
    def query(self, x: torch.Tensor) -> torch.Tensor:
        """Label ``x`` with the verifier and return a bool tensor ``[B]``."""
        y = torch.as_tensor(self.verifier(x), device=self.device).flatten()
        if y.dtype != torch.bool:
            y = y.float() > 0.5
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"verifier returned {y.shape[0]} labels for {x.shape[0]} samples")
        return y

    def update_flow(self) -> Optional[float]:
        """Signed replay continued pre-training. Returns the last loss or ``None``."""
        pos, neg = self.accepted, self.rejected
        if pos.shape[0] == 0 or pos.shape[0] < self.min_accepted or self.finetune_steps <= 0:
            return None
        was_training = self.model.training
        self.model.train()
        last = None
        try:
            for _ in range(self.finetune_steps):
                x_pos = pos[torch.randint(pos.shape[0], (self.batch_size,), device=pos.device)]
                x_neg = None
                if neg.shape[0] > 0 and self.loss.alpha > 0:
                    x_neg = neg[torch.randint(neg.shape[0], (self.batch_size,), device=neg.device)]
                loss = self.loss(self.model, x_pos, x_neg, **self.cond)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                last = float(loss.detach())
        finally:
            self.model.train(was_training)
        return last

    def step(self, n_samples: int) -> ActFlowRound:
        """Run one round of Algorithm 1 with a batch of ``n_samples`` queries."""
        self.fit_uncertainty()
        x = self.self_generate(n_samples)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                sigma = self.uncertainty(self.representation(x, **self.cond))
        finally:
            self.model.train(was_training)

        y = self.query(x)
        self.queries = torch.cat([self.queries, x])
        self.labels = torch.cat([self.labels, y])

        loss = self.update_flow()
        result = ActFlowRound(
            round=len(self.history),
            samples=x,
            labels=y,
            acceptance_rate=float(y.float().mean()),
            mean_uncertainty=float(sigma.mean()),
            n_accepted=int(self.labels.sum()),
            n_rejected=int((~self.labels).sum()),
            finetuned=loss is not None,
            loss=loss,
            alpha=self.loss.last_alpha if loss is not None else 0.0,
        )
        self.history.append(result)
        return result

    def run(
        self,
        n_rounds: int,
        n_samples: int,
        callback: Optional[Callable[[ActFlowRound], None]] = None,
    ) -> List[ActFlowRound]:
        """Run ``n_rounds`` rounds, calling ``callback(round)`` after each."""
        rounds = []
        for _ in range(n_rounds):
            r = self.step(n_samples)
            rounds.append(r)
            if callback is not None:
                callback(r)
        return rounds


__all__ = ["ActFlow", "ActFlowRound"]
