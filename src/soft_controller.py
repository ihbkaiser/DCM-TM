"""Lightweight gates for soft continual topic memory updates."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class SoftGateController(nn.Module):
    """Predict survival and novelty gates from compact topic features."""

    def __init__(self, feature_dim: int = 3, hidden: int = 16):
        super().__init__()
        self.global_gate = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.local_gate = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_features: torch.Tensor,
                local_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        retain = torch.sigmoid(self.global_gate(global_features)).squeeze(-1)
        novelty = torch.sigmoid(self.local_gate(local_features)).squeeze(-1)
        return retain, novelty


def train_soft_controller(
    retain_features: np.ndarray,
    novelty_features: np.ndarray,
    retain_priors: np.ndarray,
    novelty_priors: np.ndarray,
    epochs: int = 20,
    lr: float = 1e-3,
    lambda_llm: float = 1.0,
    hidden: int = 16,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train a small controller against Bernoulli LLM-prior KL targets.

    The controller is intentionally local to one timestamp. This keeps the
    first implementation simple while preserving the method's soft gates.
    """
    retain_priors = np.clip(retain_priors.astype(np.float32), 1e-4, 1.0 - 1e-4)
    novelty_priors = np.clip(novelty_priors.astype(np.float32), 1e-4, 1.0 - 1e-4)

    if epochs <= 0:
        return retain_priors, novelty_priors, {"loss": 0.0, "epochs": 0}

    g_feat = torch.tensor(retain_features, dtype=torch.float32, device=device)
    l_feat = torch.tensor(novelty_features, dtype=torch.float32, device=device)
    p = torch.tensor(retain_priors, dtype=torch.float32, device=device)
    q = torch.tensor(novelty_priors, dtype=torch.float32, device=device)

    feature_dim = int(retain_features.shape[1])
    model = SoftGateController(feature_dim=feature_dim, hidden=hidden).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    final_loss = 0.0
    for _ in range(epochs):
        r, n = model(g_feat, l_feat)
        loss = lambda_llm * (
            _bernoulli_kl(r, p).mean() + _bernoulli_kl(n, q).mean()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    with torch.no_grad():
        r, n = model(g_feat, l_feat)

    return (
        r.detach().cpu().numpy(),
        n.detach().cpu().numpy(),
        {"loss": final_loss, "epochs": epochs},
    )


def build_gate_features(sim_matrix: np.ndarray,
                        retain_priors: np.ndarray,
                        novelty_priors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build compact controller features from similarities and LLM priors.

    sim_matrix has shape (J local topics, K global topics).
    """
    if sim_matrix.size == 0:
        raise ValueError("sim_matrix must be non-empty")

    local_max = sim_matrix.max(axis=1)
    local_mean = sim_matrix.mean(axis=1)
    global_max = sim_matrix.max(axis=0)
    global_mean = sim_matrix.mean(axis=0)

    retain_features = np.stack(
        [global_max, global_mean, retain_priors],
        axis=1,
    ).astype(np.float32)
    novelty_features = np.stack(
        [local_max, local_mean, novelty_priors],
        axis=1,
    ).astype(np.float32)
    return retain_features, novelty_features


def _bernoulli_kl(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    pred = pred.clamp(eps, 1.0 - eps)
    target = target.clamp(eps, 1.0 - eps)
    return (
        pred * (pred.log() - target.log())
        + (1.0 - pred) * ((1.0 - pred).log() - (1.0 - target).log())
    )
