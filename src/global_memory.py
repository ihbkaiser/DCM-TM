"""
Global memory for continual topic model.

Dynamic-K design: G_t ∈ R^{K_t × V} where K_t changes over time.

  K_t = K_{t-1} - |removed| + |novel|

Because K_t is always the exact count of active global topics, the local VAE
at timestamp t is created with n_topics = K_{t-1}, so residual learning

    L_t = G_{t-1} + ΔL_t

is always applicable — both sides are (K_{t-1} × V).

After curation:
  - Removed topics are dropped from the list.
  - Novel topics are appended.
  - beta_logits is exactly (K_t, V) — no zero rows, no padding.
"""

import json
import numpy as np
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from src.topic_utils import Topic, embed_topics


@dataclass
class GlobalUpdate:
    """Record of what changed in one timestamp update."""
    timestamp: int
    n_retained: int = 0
    n_removed: int = 0
    n_novel: int = 0
    retained_ids: list[int] = field(default_factory=list)
    removed_ids: list[int] = field(default_factory=list)
    novel_ids: list[int] = field(default_factory=list)


class GlobalMemory:
    """Manages the evolving global topic set with dynamic K.

    Attributes:
        topics      : list of Topic objects (length K_t — every entry is active)
        beta_logits : (K_t, vocab_size) topic-word logit matrix (no zero rows)
        history     : list of GlobalUpdate records
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_topics: int = 500,
    ):
        self.topics: list[Topic] = []
        self.beta_logits: Optional[np.ndarray] = None   # (K_t, vocab_size)
        self.history: list[GlobalUpdate] = []
        self.embedding_model = embedding_model
        self.max_topics = max_topics
        self._next_id = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def K(self) -> int:
        """Number of active global topics."""
        return len(self.topics)

    @property
    def n_topics(self) -> int:
        """Alias for K."""
        return len(self.topics)

    @property
    def active_topics(self) -> list[Topic]:
        """All topics (every entry is active in dynamic-K design)."""
        return self.topics

    # ── Initialisation ─────────────────────────────────────────────────────

    def initialize_from_local(
        self,
        local_topics: list[Topic],
        local_beta: np.ndarray,
        timestamp: int = 0,
    ):
        """Initialize global memory from the first timestamp's local topics.

        All K_0 local topics become global topics; beta_logits = (K_0, V).
        """
        K = len(local_topics)
        self.topics = []
        for t in local_topics:
            new_topic = deepcopy(t)
            new_topic.id = self._next_id
            new_topic.source = "global"
            new_topic.metadata["origin_timestamp"] = timestamp
            new_topic.metadata["origin_source"] = t.source
            self.topics.append(new_topic)
            self._next_id += 1

        self.beta_logits = local_beta.copy()   # (K_0, V)

        # Ensure embeddings
        self.topics = embed_topics(self.topics, self.embedding_model)

        update = GlobalUpdate(
            timestamp=timestamp,
            n_novel=K,
            novel_ids=[t.id for t in self.topics],
        )
        self.history.append(update)
        print(f"  Global memory initialized: K_0 = {K} topics from T{timestamp}")

    # ── Update ─────────────────────────────────────────────────────────────

    def update(
        self,
        retained_indices: list[int],
        retained_refined_words: dict[int, list[str]],
        novel_topics: list[Topic],
        novel_beta_rows: Optional[np.ndarray],
        local_beta: np.ndarray,
        timestamp: int,
    ):
        """Update global memory after curation.

        retained_indices      : indices into self.topics to keep
        retained_refined_words: {idx: [refined_words]} optional word refinement
        novel_topics          : new Topic objects to append
        novel_beta_rows       : (n_novel, V) logit rows for novel topics
        local_beta            : (n_local, V) full local beta (fallback source)
        timestamp             : current timestamp index

        Result: self.topics becomes length K_t = |retained| + |novel|,
                self.beta_logits becomes (K_t, V) — exactly, no zero rows.
        """
        retained_set = set(retained_indices)
        removed_ids = []
        retained_ids = []

        new_topics: list[Topic] = []
        new_beta_rows: list[np.ndarray] = []

        # Keep retained topics
        for idx, topic in enumerate(self.topics):
            if idx in retained_set:
                t = deepcopy(topic)
                if idx in retained_refined_words and retained_refined_words[idx]:
                    t.words = retained_refined_words[idx][:len(t.words)]
                new_topics.append(t)
                new_beta_rows.append(self.beta_logits[idx])
                retained_ids.append(t.id)
            else:
                removed_ids.append(topic.id)

        # Append novel topics
        novel_ids = []
        for i, t in enumerate(novel_topics):
            if len(new_topics) >= self.max_topics:
                print(f"  Warning: max_topics={self.max_topics} reached, "
                      f"skipping novel topic {i}")
                break

            if novel_beta_rows is not None and i < len(novel_beta_rows):
                beta_row = novel_beta_rows[i]
            elif t.id < local_beta.shape[0]:
                beta_row = local_beta[t.id]
            else:
                beta_row = np.zeros(local_beta.shape[1])

            new_t = deepcopy(t)
            new_t.id = self._next_id
            new_t.source = "global"
            new_t.metadata["origin_timestamp"] = timestamp
            new_t.metadata["origin_source"] = t.source
            new_topics.append(new_t)
            new_beta_rows.append(beta_row)
            novel_ids.append(self._next_id)
            self._next_id += 1

        # Commit — K_t is now dynamic
        self.topics = new_topics
        self.beta_logits = (
            np.stack(new_beta_rows) if new_beta_rows
            else np.zeros((0, local_beta.shape[1]))
        )

        # Re-embed all topics
        self.topics = embed_topics(self.topics, self.embedding_model)

        update = GlobalUpdate(
            timestamp=timestamp,
            n_retained=len(retained_ids),
            n_removed=len(removed_ids),
            n_novel=len(novel_ids),
            retained_ids=retained_ids,
            removed_ids=removed_ids,
            novel_ids=novel_ids,
        )
        self.history.append(update)

        print(f"  Global memory updated at T{timestamp}: "
              f"{len(retained_ids)} retained, {len(removed_ids)} removed, "
              f"{len(novel_ids)} novel → K_{timestamp} = {self.K}")

    # ── Tensor access ─────────────────────────────────────────────────────

    def get_beta_tensor(self, device: str = "cpu"):
        """Return the (K_t, vocab_size) logit matrix as a torch tensor.

        Shape always matches self.n_topics so the VAE can apply:
            L_t = G_{t-1} + ΔL_t   (both K_{t-1} × V)
        """
        import torch
        if self.beta_logits is None or self.K == 0:
            return None
        return torch.tensor(self.beta_logits, dtype=torch.float32, device=device)

    # ── Summaries ──────────────────────────────────────────────────────────

    def get_summary(self) -> str:
        """Human-readable summary of active topics."""
        lines = [f"Global Memory: K = {self.K} active topics"]
        for i, t in enumerate(self.topics):
            lines.append(f"  [{i:3d} | id {t.id:3d}] {t.to_string()}")
        return "\n".join(lines)

    def get_evolution_summary(self) -> str:
        """Summary of topic evolution across timestamps."""
        lines = ["Topic Evolution:"]
        n_active = 0
        for update in self.history:
            n_active = n_active - update.n_removed + update.n_novel
            lines.append(
                f"  T{update.timestamp}: +{update.n_novel} novel, "
                f"-{update.n_removed} removed, ={update.n_retained} retained "
                f"→ K_{update.timestamp} = {n_active}"
            )
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        """Save global memory to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)


        # Save (K_t, V) logit matrix — no zero rows, no padding
        if self.beta_logits is not None:
            np.save(str(path / "beta_logits.npy"), self.beta_logits)

        # Save topic list (every entry is active)
        topics_data = []
        for t in self.topics:
            topics_data.append({
                "id": t.id,
                "words": t.words,
                "word_weights": t.word_weights,
                "source": t.source,
                "metadata": t.metadata,
            })
        with open(path / "topics.json", "w") as f:
            json.dump(topics_data, f, indent=2)

        # Save history
        history_data = []
        for h in self.history:
            history_data.append({
                "timestamp": h.timestamp,
                "n_retained": h.n_retained,
                "n_removed": h.n_removed,
                "n_novel": h.n_novel,
                "retained_ids": h.retained_ids,
                "removed_ids": h.removed_ids,
                "novel_ids": h.novel_ids,
            })
        with open(path / "history.json", "w") as f:
            json.dump(history_data, f, indent=2)

    def load(self, path: str):
        """Load global memory from disk."""
        path = Path(path)

        # Load (K_t, V) logit matrix
        beta_path = path / "beta_logits.npy"
        if beta_path.exists():
            self.beta_logits = np.load(str(beta_path))

        # Load topic list
        with open(path / "topics.json") as f:
            topics_data = json.load(f)
        self.topics = []
        for td in topics_data:
            self.topics.append(Topic(
                id=td["id"],
                words=td["words"],
                word_weights=td["word_weights"],
                source=td["source"],
                metadata=td["metadata"],
            ))

        self._next_id = max(t.id for t in self.topics) + 1 if self.topics else 0

        # Re-embed
        self.topics = embed_topics(self.topics, self.embedding_model)

        # Load history
        with open(path / "history.json") as f:
            history_data = json.load(f)
        self.history = [GlobalUpdate(**h) for h in history_data]
