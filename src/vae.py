"""
ProdLDA-style VAE for topic modeling.

Architecture:
  Encoder: BoW → FC → μ, log(σ²) → Logistic-Normal (Laplace approx to Dirichlet)
  Decoder: θ (topic proportions) → β (topic-word matrix) → Reconstructed BoW

Supports residual delta learning:
  At timestamp t, the decoder topic-word matrix is:
    β_t = softmax(β_global + Δβ_local)
  where β_global comes from the previous global state.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
from dataclasses import dataclass


@dataclass
class VAEOutput:
    """Output from a single VAE forward pass."""
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor
    loss: torch.Tensor
    theta: torch.Tensor       # (batch, n_topics) topic proportions


class ProdLDAEncoder(nn.Module):
    """Encoder: BoW → (μ, log σ²) in logistic-normal space."""

    def __init__(self, vocab_size: int, n_topics: int, hidden: int, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.mu = nn.Linear(hidden, n_topics)
        self.log_var = nn.Linear(hidden, n_topics)

    def forward(self, x: torch.Tensor):
        # x: (batch, vocab_size) normalized BoW
        h = self.drop(x)
        h = F.softplus(self.bn1(self.fc1(h)))
        h = F.softplus(self.bn2(self.fc2(h)))
        return self.mu(h), self.log_var(h)


class ProdLDADecoder(nn.Module):
    """Decoder: θ → reconstructed BoW via topic-word matrix β.

    Standard ProdLDA: the decoder computes log p(w|d) = log softmax(θ · β_logits)
    where β_logits are unnormalized topic-word scores.

    Uses BatchNorm on the pre-softmax logits for stability.
    """

    def __init__(self, vocab_size: int, n_topics: int):
        super().__init__()
        # Topic-word logits (local delta when doing residual learning)
        self.topic_word_logits = nn.Linear(n_topics, vocab_size, bias=False)
        # BN on pre-softmax logits (critical for ProdLDA stability)
        self.bn = nn.BatchNorm1d(vocab_size, affine=True)
        # Small dropout on decoder
        self.drop = nn.Dropout(0.1)

    def forward(self, theta: torch.Tensor, global_beta: Optional[torch.Tensor] = None):
        """
        Args:
            theta: (batch, n_topics) topic proportions (softmax output)
            global_beta: (n_topics, vocab_size) global topic-word logits (optional)
        Returns:
            log_recon: (batch, vocab_size) log-probability of words
        """
        # Get local topic-word logits
        local_logits = self.topic_word_logits.weight.T  # (n_topics, vocab_size)

        if global_beta is not None:
            combined = global_beta + local_logits
        else:
            combined = local_logits

        # θ · β_logits → (batch, vocab_size) unnormalized scores
        logits = torch.mm(theta, combined)  # (batch, vocab_size)
        logits = self.bn(logits)
        logits = self.drop(logits)

        # Log-softmax for numerical stability
        log_recon = F.log_softmax(logits, dim=-1)
        return log_recon


class ProdLDA(nn.Module):
    """ProdLDA: Product of Experts LDA with logistic-normal prior.

    Implements the Neural Variational Document Model with Dirichlet-like prior
    via logistic-normal approximation.
    """

    def __init__(self, vocab_size: int, n_topics: int, enc_hidden: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_topics = n_topics

        self.encoder = ProdLDAEncoder(vocab_size, n_topics, enc_hidden, dropout)
        self.decoder = ProdLDADecoder(vocab_size, n_topics)

        # Prior parameters (standard normal)
        self.prior_mu = nn.Parameter(torch.zeros(n_topics), requires_grad=False)
        self.prior_var = nn.Parameter(torch.ones(n_topics), requires_grad=False)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample θ using reparameterization trick + softmax."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std  # (batch, n_topics)
        theta = F.softmax(z, dim=-1)
        return theta

    def forward(self, x: torch.Tensor, global_beta: Optional[torch.Tensor] = None,
                kl_weight: float = 1.0) -> VAEOutput:
        """
        Args:
            x: (batch, vocab_size) raw BoW counts
            global_beta: optional (n_topics, vocab_size) global topic-word logits
            kl_weight: weight for KL divergence (for annealing)
        """
        # Normalize BoW
        x_norm = x / (x.sum(dim=1, keepdim=True) + 1e-12)

        # Encode
        mu, log_var = self.encoder(x_norm)

        # Sample
        theta = self.reparameterize(mu, log_var)

        # Decode
        log_recon = self.decoder(theta, global_beta)

        # Reconstruction loss (negative log-likelihood)
        recon_loss = -(x * log_recon).sum(dim=1).mean()

        # KL divergence: KL(q(z|x) || p(z)) for logistic-normal
        prior_var = self.prior_var
        prior_mu = self.prior_mu
        var_ratio = torch.exp(log_var) / prior_var
        diff = mu - prior_mu
        kl = 0.5 * (var_ratio + diff.pow(2) / prior_var - 1.0 - log_var + prior_var.log())
        kl_loss = kl.sum(dim=1).mean()

        loss = recon_loss + kl_weight * kl_loss

        return VAEOutput(
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            loss=loss,
            theta=theta,
        )

    def get_topic_word_dist(self, global_beta: Optional[torch.Tensor] = None) -> np.ndarray:
        """Get topic-word distributions β: (n_topics, vocab_size).

        Returns softmax-normalized distributions over words for each topic.
        """
        with torch.no_grad():
            local_logits = self.decoder.topic_word_logits.weight.T
            if global_beta is not None:
                combined = global_beta + local_logits
            else:
                combined = local_logits
            beta = F.softmax(combined, dim=-1)
        return beta.cpu().numpy()

    def get_topic_word_logits(self, global_beta: Optional[torch.Tensor] = None) -> np.ndarray:
        """Get raw topic-word logits: (n_topics, vocab_size).

        These are the unnormalized logits (useful for residual learning).
        """
        with torch.no_grad():
            local_logits = self.decoder.topic_word_logits.weight.T
            if global_beta is not None:
                combined = global_beta + local_logits
            else:
                combined = local_logits
        return combined.cpu().numpy()


# ─── Training ────────────────────────────────────────────────────────────────

def train_vae(
    model: ProdLDA,
    train_loader,
    test_loader=None,
    global_beta: Optional[torch.Tensor] = None,
    epochs: int = 100,
    lr: float = 0.002,
    weight_decay: float = 1e-6,
    kl_warmup_epochs: int = 20,
    patience: int = 10,
    device: str = "cpu",
) -> dict:
    """Train ProdLDA model on a single timestamp's data.

    Returns:
        dict with training history and best topic-word distributions.
    """
    model = model.to(device)
    if global_beta is not None:
        global_beta = global_beta.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "train_recon": [], "train_kl": [],
               "test_loss": [], "kl_weight": []}

    for epoch in range(1, epochs + 1):
        # KL annealing
        if kl_warmup_epochs > 0:
            kl_weight = min(1.0, epoch / kl_warmup_epochs)
        else:
            kl_weight = 1.0
        history["kl_weight"].append(kl_weight)

        # ── Train ──
        model.train()
        total_loss = total_recon = total_kl = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            output = model(batch, global_beta, kl_weight)

            optimizer.zero_grad()
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += output.loss.item()
            total_recon += output.recon_loss.item()
            total_kl += output.kl_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches
        history["train_loss"].append(avg_loss)
        history["train_recon"].append(avg_recon)
        history["train_kl"].append(avg_kl)

        # ── Test ──
        test_loss = None
        if test_loader is not None:
            model.eval()
            total_test = 0.0
            n_test = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    output = model(batch, global_beta, kl_weight=1.0)
                    total_test += output.loss.item()
                    n_test += 1
            test_loss = total_test / n_test
            history["test_loss"].append(test_loss)
        else:
            history["test_loss"].append(None)

        monitor_loss = test_loss if test_loss is not None else avg_loss
        scheduler.step(monitor_loss)

        # Early stopping
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1 or wait == 0:
            msg = f"  Epoch {epoch:3d} | Loss {avg_loss:.2f} (recon {avg_recon:.2f} + kl {avg_kl:.2f} * {kl_weight:.2f})"
            if test_loss is not None:
                msg += f" | Test {test_loss:.2f}"
            if wait == 0:
                msg += " *"
            print(msg)

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return {
        "history": history,
        "best_loss": best_loss,
        "final_epoch": epoch,
    }
