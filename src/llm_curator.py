"""
LLM-based topic curator for continual topic model.

Two-stage curation:
  Stage 1 — Prune & Refine: For each global topic, decide RETAIN or REMOVE
            based on evidence from local topics.
  Stage 2 — Detect Novel: For each local topic, decide NOVEL or COVERED
            based on similarity to retained global topics.

Supports:
  - "gemini" provider (Google Generative AI)
  - "none" provider (similarity-based fallback, no LLM needed)
"""

import json
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.topic_utils import Topic, cosine_similarity_matrix, find_nearest_topics


@dataclass
class CurationDecision:
    """Decision for a single topic."""
    topic_id: int
    action: str          # "RETAIN", "REMOVE", "NOVEL", "COVERED"
    refined_words: Optional[list[str]] = None
    reason: str = ""
    confidence: float = 0.0


# ─── LLM Interface ───────────────────────────────────────────────────────────

class LLMCurator:
    """LLM-based topic curator."""

    def __init__(self, config: dict):
        self.provider = config.get("provider", "none")
        self.model_name = config.get("model", "gemini-2.0-flash")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 1024)
        self.rate_limit_delay = config.get("rate_limit_delay", 0.5)
        self.max_retries = config.get("max_retries", 3)
        self.fallback_config = config.get("fallback", {})

        self._client = None
        if self.provider == "gemini":
            self._init_gemini()

    def _init_gemini(self):
        """Initialize Google Generative AI client."""
        try:
            from google import genai
            self._client = genai.Client()
            print(f"  LLM Curator: Using Gemini ({self.model_name})")
        except Exception as e:
            print(f"  Warning: Failed to init Gemini: {e}. Falling back to similarity-based.")
            self.provider = "none"

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return text response."""
        if self.provider == "gemini":
            from google import genai
            for attempt in range(self.max_retries):
                try:
                    response = self._client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_tokens,
                        },
                    )
                    time.sleep(self.rate_limit_delay)
                    return response.text
                except Exception as e:
                    print(f"    LLM call failed (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)
            return ""
        return ""

    # ─── Stage 1: Prune & Refine ─────────────────────────────────────────────

    def stage1_prune_and_refine(
        self,
        global_topics: list[Topic],
        local_topics: list[Topic],
        top_k: int = 5,
    ) -> list[CurationDecision]:
        """For each global topic, decide RETAIN or REMOVE.

        A global topic is RETAINED if it still has support in the current
        local topics, otherwise REMOVED (outdated/redundant).
        """
        if not global_topics:
            return []

        if self.provider == "none":
            return self._stage1_fallback(global_topics, local_topics)

        # Find nearest local topics for each global topic
        nearest = find_nearest_topics(global_topics, local_topics, top_k=top_k)

        decisions = []
        for g_idx, g_topic in enumerate(global_topics):
            neighbors = nearest[g_idx]
            neighbor_strs = []
            for l_idx, sim in neighbors:
                neighbor_strs.append(
                    f"  - Local topic (sim={sim:.3f}): {local_topics[l_idx].to_string()}"
                )

            prompt = self._build_stage1_prompt(g_topic, neighbor_strs)
            response = self._call_llm(prompt)
            decision = self._parse_stage1_response(response, g_topic.id)
            decisions.append(decision)

        return decisions

    def _build_stage1_prompt(self, global_topic: Topic, neighbor_strs: list[str]) -> str:
        return f"""You are a topic modeling expert. Evaluate whether a GLOBAL topic should be RETAINED or REMOVED based on evidence from local (current) topics.

GLOBAL TOPIC: {global_topic.to_string()}

NEAREST LOCAL TOPICS (by embedding similarity):
{chr(10).join(neighbor_strs)}

INSTRUCTIONS:
- RETAIN: The global topic still represents a meaningful, active research theme evidenced by local topics.
- REMOVE: The global topic is outdated, redundant, or no longer supported by current data.

If RETAIN, optionally refine the topic's top words to better reflect current usage.

Respond in JSON format:
{{"action": "RETAIN" or "REMOVE", "refined_words": ["word1", "word2", ...] or null, "reason": "brief explanation"}}
"""

    def _parse_stage1_response(self, response: str, topic_id: int) -> CurationDecision:
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return CurationDecision(
                    topic_id=topic_id,
                    action=data.get("action", "RETAIN").upper(),
                    refined_words=data.get("refined_words"),
                    reason=data.get("reason", ""),
                )
        except (json.JSONDecodeError, KeyError):
            pass
        # Default to RETAIN on parse failure
        return CurationDecision(topic_id=topic_id, action="RETAIN", reason="LLM parse failure")

    def _stage1_fallback(
        self,
        global_topics: list[Topic],
        local_topics: list[Topic],
    ) -> list[CurationDecision]:
        """Similarity-based fallback for Stage 1."""
        threshold = self.fallback_config.get("prune_threshold", 0.3)

        if not local_topics:
            # No local topics → remove all globals
            return [
                CurationDecision(topic_id=g.id, action="REMOVE",
                                 reason=f"No local topics to support")
                for g in global_topics
            ]

        sim_matrix = cosine_similarity_matrix(global_topics, local_topics)
        decisions = []

        for g_idx, g_topic in enumerate(global_topics):
            max_sim = float(sim_matrix[g_idx].max())
            if max_sim >= threshold:
                # Find the closest local topic to potentially refine
                best_local_idx = int(sim_matrix[g_idx].argmax())
                best_local = local_topics[best_local_idx]
                # Merge words: keep global words that appear in local, add new local words
                refined = _merge_topic_words(g_topic, best_local)
                decisions.append(CurationDecision(
                    topic_id=g_topic.id,
                    action="RETAIN",
                    refined_words=refined,
                    reason=f"max_sim={max_sim:.3f} >= {threshold}",
                    confidence=max_sim,
                ))
            else:
                decisions.append(CurationDecision(
                    topic_id=g_topic.id,
                    action="REMOVE",
                    reason=f"max_sim={max_sim:.3f} < {threshold}",
                    confidence=1.0 - max_sim,
                ))

        return decisions

    # ─── Stage 2: Detect Novel ───────────────────────────────────────────────

    def stage2_detect_novel(
        self,
        local_topics: list[Topic],
        retained_global_topics: list[Topic],
        top_k: int = 5,
    ) -> list[CurationDecision]:
        """For each local topic, decide NOVEL or COVERED.

        A local topic is NOVEL if it represents a genuinely new theme
        not covered by any retained global topic.
        """
        if not local_topics:
            return []

        if self.provider == "none":
            return self._stage2_fallback(local_topics, retained_global_topics)

        if not retained_global_topics:
            # All local topics are novel if no global topics exist
            return [
                CurationDecision(
                    topic_id=l.id, action="NOVEL",
                    refined_words=l.words,
                    reason="No global topics to compare against",
                )
                for l in local_topics
            ]

        nearest = find_nearest_topics(local_topics, retained_global_topics, top_k=top_k)

        decisions = []
        for l_idx, l_topic in enumerate(local_topics):
            neighbors = nearest[l_idx]
            neighbor_strs = []
            for g_idx, sim in neighbors:
                neighbor_strs.append(
                    f"  - Global topic (sim={sim:.3f}): {retained_global_topics[g_idx].to_string()}"
                )

            prompt = self._build_stage2_prompt(l_topic, neighbor_strs)
            response = self._call_llm(prompt)
            decision = self._parse_stage2_response(response, l_topic.id, l_topic.words)
            decisions.append(decision)

        return decisions

    def _build_stage2_prompt(self, local_topic: Topic, neighbor_strs: list[str]) -> str:
        return f"""You are a topic modeling expert. Determine whether a LOCAL topic represents a NOVEL research theme or is already COVERED by existing global topics.

LOCAL TOPIC: {local_topic.to_string()}

NEAREST GLOBAL TOPICS (by embedding similarity):
{chr(10).join(neighbor_strs)}

INSTRUCTIONS:
- NOVEL: The local topic represents a genuinely new or emerging theme not adequately covered by any global topic.
- COVERED: The local topic is essentially the same as or a subset of an existing global topic.

If NOVEL, provide refined representative words for the new topic.

Respond in JSON format:
{{"action": "NOVEL" or "COVERED", "refined_words": ["word1", "word2", ...] or null, "reason": "brief explanation"}}
"""

    def _parse_stage2_response(self, response: str, topic_id: int,
                                default_words: list[str]) -> CurationDecision:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return CurationDecision(
                    topic_id=topic_id,
                    action=data.get("action", "COVERED").upper(),
                    refined_words=data.get("refined_words", default_words),
                    reason=data.get("reason", ""),
                )
        except (json.JSONDecodeError, KeyError):
            pass
        return CurationDecision(
            topic_id=topic_id, action="COVERED",
            reason="LLM parse failure",
        )

    def _stage2_fallback(
        self,
        local_topics: list[Topic],
        retained_global_topics: list[Topic],
    ) -> list[CurationDecision]:
        """Similarity-based fallback for Stage 2."""
        threshold = self.fallback_config.get("novel_threshold", 0.5)

        if not retained_global_topics:
            return [
                CurationDecision(
                    topic_id=l.id, action="NOVEL",
                    refined_words=l.words,
                    reason="No global topics to compare",
                    confidence=1.0,
                )
                for l in local_topics
            ]

        sim_matrix = cosine_similarity_matrix(local_topics, retained_global_topics)
        decisions = []

        for l_idx, l_topic in enumerate(local_topics):
            max_sim = float(sim_matrix[l_idx].max())
            if max_sim < threshold:
                decisions.append(CurationDecision(
                    topic_id=l_topic.id,
                    action="NOVEL",
                    refined_words=l_topic.words,
                    reason=f"max_sim={max_sim:.3f} < {threshold}",
                    confidence=1.0 - max_sim,
                ))
            else:
                decisions.append(CurationDecision(
                    topic_id=l_topic.id,
                    action="COVERED",
                    reason=f"max_sim={max_sim:.3f} >= {threshold}",
                    confidence=max_sim,
                ))

        return decisions


# ─── Helper ──────────────────────────────────────────────────────────────────

def _merge_topic_words(topic_a: Topic, topic_b: Topic, max_words: int = 15) -> list[str]:
    """Merge words from two topics, preferring topic_a's order."""
    seen = set()
    merged = []
    for w in topic_a.words:
        if w not in seen:
            merged.append(w)
            seen.add(w)
    for w in topic_b.words:
        if w not in seen and len(merged) < max_words:
            merged.append(w)
            seen.add(w)
    return merged[:max_words]
