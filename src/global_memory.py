"""
Global memory for continual topic model.

Maintains a dynamic set of global topics G that evolves across timestamps.
At each timestamp:
  G_t = Retained(G_{t-1}) ∪ Novel(L_t)

Tracks:
  - Current global topics with embeddings
  - Topic evolution history (additions, removals per timestamp)
  - Global topic-word logits (β) for residual learning
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
    """Manages the evolving global topic set."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.topics: list[Topic] = []
        self.beta_logits: Optional[np.ndarray] = None  # (n_global, vocab_size)
        self.history: list[GlobalUpdate] = []
        self.embedding_model = embedding_model
        self._next_id = 0

    @property
    def n_topics(self) -> int:
        return len(self.topics)

    def initialize_from_local(
        self,
        local_topics: list[Topic],
        local_beta: np.ndarray,
        timestamp: int = 0,
    ):
        """Initialize global memory from the first timestamp's local topics.

        All local topics become global topics.
        """
        self.topics = []
        for t in local_topics:
            new_topic = deepcopy(t)
            new_topic.id = self._next_id
            new_topic.source = "global"
            new_topic.metadata["origin_timestamp"] = timestamp
            new_topic.metadata["origin_source"] = t.source
            self.topics.append(new_topic)
            self._next_id += 1

        self.beta_logits = local_beta.copy()

        # Ensure embeddings
        self.topics = embed_topics(self.topics, self.embedding_model)

        update = GlobalUpdate(
            timestamp=timestamp,
            n_novel=len(self.topics),
            novel_ids=[t.id for t in self.topics],
        )
        self.history.append(update)
        print(f"  Global memory initialized: {self.n_topics} topics from T{timestamp}")

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

        Args:
            retained_indices: indices into self.topics that are retained
            retained_refined_words: {topic_idx: [refined_words]} for retained topics
            novel_topics: new topics to add
            novel_beta_rows: (n_novel, vocab_size) beta rows for novel topics
            local_beta: (n_local, vocab_size) full local beta matrix
            timestamp: current timestamp
        """
        removed_ids = []
        retained_ids = []

        # Identify removed topics
        retained_set = set(retained_indices)
        old_topics = self.topics
        old_beta = self.beta_logits

        new_topics = []
        new_beta_rows = []

        for idx, topic in enumerate(old_topics):
            if idx in retained_set:
                # Optionally refine words
                if idx in retained_refined_words and retained_refined_words[idx]:
                    topic = deepcopy(topic)
                    topic.words = retained_refined_words[idx][:len(topic.words)]
                retained_ids.append(topic.id)
                new_topics.append(topic)
                new_beta_rows.append(old_beta[idx])
            else:
                removed_ids.append(topic.id)

        # Add novel topics
        novel_ids = []
        for i, t in enumerate(novel_topics):
            new_topic = deepcopy(t)
            new_topic.id = self._next_id
            new_topic.source = "global"
            new_topic.metadata["origin_timestamp"] = timestamp
            new_topic.metadata["origin_source"] = t.source
            novel_ids.append(self._next_id)
            new_topics.append(new_topic)
            self._next_id += 1

            if novel_beta_rows is not None and i < len(novel_beta_rows):
                new_beta_rows.append(novel_beta_rows[i])
            else:
                # Use the local topic's beta row
                if t.id < local_beta.shape[0]:
                    new_beta_rows.append(local_beta[t.id])
                else:
                    new_beta_rows.append(np.zeros(local_beta.shape[1]))

        self.topics = new_topics
        if new_beta_rows:
            self.beta_logits = np.stack(new_beta_rows, axis=0)
        else:
            self.beta_logits = np.zeros((0, local_beta.shape[1]))

        # Re-embed all topics (some may have refined words)
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
              f"{len(novel_ids)} novel → {self.n_topics} total")

    def get_beta_tensor(self, device: str = "cpu"):
        """Get global beta logits as a torch tensor for residual learning."""
        import torch
        if self.beta_logits is None or len(self.beta_logits) == 0:
            return None
        return torch.tensor(self.beta_logits, dtype=torch.float32, device=device)

    def get_summary(self) -> str:
        """Get a human-readable summary of the global memory."""
        lines = [f"Global Memory: {self.n_topics} topics"]
        for t in self.topics:
            lines.append(f"  [{t.id}] {t.to_string()}")
        return "\n".join(lines)

    def get_evolution_summary(self) -> str:
        """Get a summary of how topics evolved across timestamps."""
        lines = ["Topic Evolution:"]
        for update in self.history:
            lines.append(
                f"  T{update.timestamp}: +{update.n_novel} novel, "
                f"-{update.n_removed} removed, ={update.n_retained} retained "
                f"→ total topics after update"
            )
        return "\n".join(lines)

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save global memory to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save beta logits
        if self.beta_logits is not None:
            np.save(str(path / "beta_logits.npy"), self.beta_logits)

        # Save topics (without embeddings — they'll be recomputed)
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

        # Load beta logits
        beta_path = path / "beta_logits.npy"
        if beta_path.exists():
            self.beta_logits = np.load(str(beta_path))

        # Load topics
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
