"""
Pipeline: Orchestrates the 4-step continual topic model.

For each timestamp t:
  Step 1 — Local VAE Training:
      Train ProdLDA on D_t. If use_residual and t > 0,
      the decoder uses β = G_{t-1} + Δβ (residual learning).
      Output: local topics L_t with embeddings.

  Step 2 — LLM Stage 1 (Prune & Refine):
      For each global topic in G_{t-1}, find top-K nearest local topics,
      ask LLM: RETAIN or REMOVE? If RETAIN, optionally refine words.

  Step 3 — LLM Stage 2 (Detect Novel):
      For each local topic in L_t, find top-K nearest retained globals,
      ask LLM: NOVEL or COVERED? If NOVEL, add to G_t.

  Step 4 — Update Global State:
      G_t = Retained(G_{t-1}) ∪ Novel(L_t)
      Discard batch data and local topics.
"""

import os
import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from src.data_loader import Corpus, TimestampData, make_dataloader
from src.vae import ProdLDA, train_vae
from src.topic_utils import (
    extract_topics, embed_topics, topic_diversity,
    topic_coherence_pmi, cosine_similarity_matrix,
)
from src.llm_curator import LLMCurator, CurationDecision
from src.global_memory import GlobalMemory


class ContinualTopicPipeline:
    """Continual Topic Model pipeline."""

    def __init__(self, config: dict, corpus: Corpus):
        self.config = config
        self.corpus = corpus

        # VAE config
        vae_cfg = config["vae"]
        self.n_topics = vae_cfg["n_topics"]
        self.enc_hidden = vae_cfg["enc_hidden"]
        self.dropout = vae_cfg["dropout"]
        self.lr = vae_cfg["lr"]
        self.weight_decay = vae_cfg["weight_decay"]
        self.batch_size = vae_cfg["batch_size"]
        self.epochs = vae_cfg["epochs"]
        self.patience = vae_cfg["patience"]
        self.kl_warmup = vae_cfg["kl_warmup_epochs"]
        self.use_residual = vae_cfg.get("use_residual", True)

        # Topic config
        topic_cfg = config["topics"]
        self.top_m = topic_cfg["top_m_words"]
        self.embedding_model = topic_cfg["embedding_model"]
        self.top_k_nearest = topic_cfg["top_k_nearest"]

        # Pipeline config
        pipe_cfg = config["pipeline"]
        self.output_dir = Path(pipe_cfg["output_dir"])
        self.save_intermediate = pipe_cfg.get("save_intermediate", True)
        self.seed = pipe_cfg.get("seed", 42)

        # Device
        device_str = pipe_cfg.get("device", "auto")
        if device_str == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_str
        print(f"Using device: {self.device}")

        # LLM Curator
        self.curator = LLMCurator(config.get("llm", {}))

        # Global memory
        self.global_memory = GlobalMemory(embedding_model=self.embedding_model)

        # Results tracking
        self.results = []

        # Set seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def run(self):
        """Run the full pipeline across all timestamps."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamps = self.corpus.timestamps

        print(f"\n{'='*70}")
        print(f"CONTINUAL TOPIC MODEL PIPELINE")
        print(f"{'='*70}")
        print(f"Timestamps: {len(timestamps)}")
        print(f"Topics per timestamp: {self.n_topics}")
        print(f"Vocab size: {self.corpus.vocab_size}")
        print(f"Residual learning: {self.use_residual}")
        print(f"LLM provider: {self.curator.provider}")
        print(f"{'='*70}\n")

        for i, ts in enumerate(timestamps):
            t_start = time.time()
            print(f"\n{'─'*70}")
            label = self.corpus.time_labels.get(ts, str(ts))
            n_docs = self.corpus.train_data[ts].n_docs
            print(f"TIMESTAMP T{ts} ({label}) — {n_docs} documents")
            print(f"{'─'*70}")

            result = self._process_timestamp(ts, is_first=(i == 0))
            result["wall_time"] = time.time() - t_start
            self.results.append(result)

            # Print summary
            self._print_timestamp_summary(result)

            # Save intermediate
            if self.save_intermediate:
                self._save_timestamp_results(ts, result)

        # Final summary
        self._print_final_summary()
        self._save_final_results()

    def _process_timestamp(self, ts: int, is_first: bool) -> dict:
        """Process a single timestamp through the 4-step pipeline."""
        result = {"timestamp": ts, "label": self.corpus.time_labels.get(ts, str(ts))}

        # ── Step 1: Local VAE Training ───────────────────────────────────────
        print(f"\n  Step 1: Training local VAE...")
        local_topics, local_beta, train_info = self._step1_train_vae(ts)
        result["train_info"] = {
            "final_epoch": train_info["final_epoch"],
            "best_loss": float(train_info["best_loss"]),
        }
        result["n_local_topics"] = len(local_topics)

        # Compute local topic metrics
        diversity = topic_diversity(local_topics)
        result["local_diversity"] = diversity
        print(f"  Local topics: {len(local_topics)}, diversity: {diversity:.3f}")

        if is_first:
            # First timestamp: all local topics become global
            print(f"\n  First timestamp — initializing global memory from local topics")
            self.global_memory.initialize_from_local(local_topics, local_beta, ts)
            result["n_retained"] = 0
            result["n_removed"] = 0
            result["n_novel"] = len(local_topics)
            result["n_global"] = self.global_memory.n_topics
            return result

        # ── Step 2: LLM Stage 1 — Prune & Refine ────────────────────────────
        print(f"\n  Step 2: Prune & Refine ({self.global_memory.n_topics} global topics)...")
        stage1_decisions = self.curator.stage1_prune_and_refine(
            global_topics=self.global_memory.topics,
            local_topics=local_topics,
            top_k=self.top_k_nearest,
        )

        retained_indices = []
        retained_refined = {}
        removed_count = 0
        for i, dec in enumerate(stage1_decisions):
            if dec.action == "RETAIN":
                retained_indices.append(i)
                if dec.refined_words:
                    retained_refined[i] = dec.refined_words
            else:
                removed_count += 1

        print(f"  Stage 1 result: {len(retained_indices)} retained, {removed_count} removed")

        # Build retained global topics list for Stage 2
        retained_global = [self.global_memory.topics[i] for i in retained_indices]

        # ── Step 3: LLM Stage 2 — Detect Novel ──────────────────────────────
        print(f"\n  Step 3: Detect Novel ({len(local_topics)} local topics)...")
        stage2_decisions = self.curator.stage2_detect_novel(
            local_topics=local_topics,
            retained_global_topics=retained_global,
            top_k=self.top_k_nearest,
        )

        novel_topics = []
        novel_beta_rows = []
        covered_count = 0
        for dec in stage2_decisions:
            if dec.action == "NOVEL":
                # Find the local topic
                lt = local_topics[dec.topic_id]
                if dec.refined_words:
                    lt.words = dec.refined_words[:self.top_m]
                novel_topics.append(lt)
                novel_beta_rows.append(local_beta[dec.topic_id])
            else:
                covered_count += 1

        novel_beta_arr = np.stack(novel_beta_rows) if novel_beta_rows else None
        print(f"  Stage 2 result: {len(novel_topics)} novel, {covered_count} covered")

        # ── Step 4: Update Global State ──────────────────────────────────────
        print(f"\n  Step 4: Updating global state...")
        self.global_memory.update(
            retained_indices=retained_indices,
            retained_refined_words=retained_refined,
            novel_topics=novel_topics,
            novel_beta_rows=novel_beta_arr,
            local_beta=local_beta,
            timestamp=ts,
        )

        result["n_retained"] = len(retained_indices)
        result["n_removed"] = removed_count
        result["n_novel"] = len(novel_topics)
        result["n_global"] = self.global_memory.n_topics
        result["global_diversity"] = topic_diversity(self.global_memory.topics)

        return result

    def _step1_train_vae(self, ts: int):
        """Train VAE on a single timestamp's data and extract local topics."""
        ts_data = self.corpus.train_data[ts]
        train_loader = make_dataloader(ts_data, batch_size=self.batch_size, shuffle=True)

        # Optional test loader
        test_loader = None
        if self.corpus.test_data and ts in self.corpus.test_data:
            test_loader = make_dataloader(
                self.corpus.test_data[ts],
                batch_size=self.batch_size,
                shuffle=False,
            )

        # Create model
        model = ProdLDA(
            vocab_size=self.corpus.vocab_size,
            n_topics=self.n_topics,
            enc_hidden=self.enc_hidden,
            dropout=self.dropout,
        )

        # Global beta for residual learning
        global_beta = None
        if self.use_residual and self.global_memory.n_topics > 0:
            gb = self.global_memory.get_beta_tensor(self.device)
            if gb is not None and gb.shape[0] == self.n_topics:
                global_beta = gb
                print(f"  Using residual learning with {gb.shape[0]} global topic logits")
            else:
                if gb is not None:
                    print(f"  Skipping residual: global has {gb.shape[0]} topics, "
                          f"local expects {self.n_topics}")

        # Train
        train_info = train_vae(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            global_beta=global_beta,
            epochs=self.epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            kl_warmup_epochs=self.kl_warmup,
            patience=self.patience,
            device=self.device,
        )

        # Extract topic-word distributions
        model.eval()
        model = model.to(self.device)
        beta = model.get_topic_word_dist(global_beta)  # (n_topics, vocab_size)

        # Extract topics
        local_topics = extract_topics(
            beta, self.corpus.vocab, top_m=self.top_m,
            source=f"local_T{ts}",
        )

        # Compute embeddings
        local_topics = embed_topics(local_topics, self.embedding_model)

        return local_topics, beta, train_info

    # ─── Reporting ───────────────────────────────────────────────────────────

    def _print_timestamp_summary(self, result: dict):
        print(f"\n  Summary T{result['timestamp']} ({result['label']}):")
        print(f"    Local topics: {result['n_local_topics']}")
        if "local_diversity" in result:
            print(f"    Local diversity: {result['local_diversity']:.3f}")
        print(f"    Retained: {result['n_retained']}, "
              f"Removed: {result['n_removed']}, "
              f"Novel: {result['n_novel']}")
        print(f"    Global topics: {result['n_global']}")
        if "global_diversity" in result:
            print(f"    Global diversity: {result['global_diversity']:.3f}")
        if "wall_time" in result:
            print(f"    Wall time: {result['wall_time']:.1f}s")

    def _print_final_summary(self):
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Final global topics: {self.global_memory.n_topics}")
        print(f"\nTopic evolution:")
        for r in self.results:
            print(f"  T{r['timestamp']:2d} ({r['label']:>9s}): "
                  f"+{r['n_novel']:3d} novel, -{r['n_removed']:3d} removed, "
                  f"={r['n_retained']:3d} retained → {r['n_global']:3d} global")

        print(f"\n{self.global_memory.get_summary()}")

    def _save_timestamp_results(self, ts: int, result: dict):
        """Save per-timestamp results."""
        ts_dir = self.output_dir / f"T{ts}"
        ts_dir.mkdir(parents=True, exist_ok=True)

        # Save result summary
        serializable = {k: v for k, v in result.items()
                        if not isinstance(v, np.ndarray)}
        with open(ts_dir / "result.json", "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        # Save global memory snapshot
        self.global_memory.save(str(ts_dir / "global_memory"))

    def _save_final_results(self):
        """Save final pipeline results."""
        # Save evolution summary
        with open(self.output_dir / "evolution.json", "w") as f:
            results_serializable = []
            for r in self.results:
                results_serializable.append(
                    {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
                )
            json.dump(results_serializable, f, indent=2, default=str)

        # Save final global memory
        self.global_memory.save(str(self.output_dir / "final_global_memory"))

        # Save final topics as readable text
        with open(self.output_dir / "final_topics.txt", "w") as f:
            f.write(self.global_memory.get_summary())
            f.write("\n\n")
            f.write(self.global_memory.get_evolution_summary())

        print(f"\nResults saved to {self.output_dir}/")
