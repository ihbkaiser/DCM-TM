"""
Topic utilities: extraction, embedding, similarity.

Provides functions to:
  - Extract top-m words from topic-word distributions
  - Compute dense embeddings for topics (using sentence-transformers)
  - Compute pairwise cosine similarity between topic sets
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sentence_transformers import SentenceTransformer


@dataclass
class Topic:
    """A single topic representation."""
    id: int
    words: list[str]           # top-m words
    word_weights: list[float]  # corresponding weights
    embedding: Optional[np.ndarray] = None   # dense embedding
    source: str = ""           # e.g. "local_T3", "global"
    metadata: dict = field(default_factory=dict)

    def to_string(self) -> str:
        """Human-readable topic string."""
        return ", ".join(self.words)


# ─── Topic Extraction ────────────────────────────────────────────────────────

def extract_topics(
    beta: np.ndarray,
    vocab: list[str],
    top_m: int = 15,
    source: str = "",
) -> list[Topic]:
    """Extract topics from a topic-word distribution matrix.

    Args:
        beta: (n_topics, vocab_size) topic-word distribution
        vocab: vocabulary list
        top_m: number of top words per topic
        source: label for where these topics came from

    Returns:
        List of Topic objects.
    """
    n_topics, vocab_size = beta.shape
    topics = []

    for k in range(n_topics):
        # Get top-m word indices by weight
        top_indices = np.argsort(beta[k])[::-1][:top_m]
        words = [vocab[i] for i in top_indices]
        weights = [float(beta[k, i]) for i in top_indices]

        topics.append(Topic(
            id=k,
            words=words,
            word_weights=weights,
            source=source,
        ))

    return topics


# ─── Topic Embedding ─────────────────────────────────────────────────────────

_model_cache: dict[str, SentenceTransformer] = {}
_word_embedding_cache: dict[tuple[str, int, int], np.ndarray] = {}


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Get or create a sentence-transformer model (cached)."""
    if model_name not in _model_cache:
        print(f"  Loading embedding model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_topics(
    topics: list[Topic],
    model_name: str = "all-MiniLM-L6-v2",
) -> list[Topic]:
    """Compute dense embeddings for a list of topics.

    Embeds the topic's top words as a single sentence.
    """
    model = get_embedding_model(model_name)

    # Create text representations
    texts = [t.to_string() for t in topics]

    # Batch encode
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=128)

    for topic, emb in zip(topics, embeddings):
        topic.embedding = emb / (np.linalg.norm(emb) + 1e-12)  # L2 normalize

    return topics


def embed_topics_from_beta(
    topics: list[Topic],
    beta: np.ndarray,
    vocab: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 512,
) -> list[Topic]:
    """Compute topic embeddings as alpha_k = sum_v beta_kv * e_v.

    Top words are still kept for readable prompts and summaries, but nearest
    topic retrieval uses these full-distribution weighted embeddings.
    """
    if len(topics) == 0:
        return topics

    beta = np.asarray(beta, dtype=np.float32)
    if beta.shape[0] != len(topics):
        raise ValueError("beta rows must match number of topics")
    if beta.shape[1] != len(vocab):
        raise ValueError("beta columns must match vocabulary size")

    word_embeddings = get_word_embedding_matrix(
        vocab=vocab,
        model_name=model_name,
        batch_size=batch_size,
    )

    weights = beta / (beta.sum(axis=1, keepdims=True) + 1e-12)
    topic_embeddings = weights @ word_embeddings
    topic_embeddings = topic_embeddings / (
        np.linalg.norm(topic_embeddings, axis=1, keepdims=True) + 1e-12
    )

    for topic, emb in zip(topics, topic_embeddings):
        topic.embedding = emb
        topic.metadata["embedding_source"] = "beta_weighted_vocab"

    return topics


def get_word_embedding_matrix(
    vocab: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 512,
) -> np.ndarray:
    """Return L2-normalized word embeddings for the full vocabulary."""
    cache_key = (model_name, len(vocab), hash("\n".join(vocab)))
    if cache_key not in _word_embedding_cache:
        model = get_embedding_model(model_name)
        print(f"  Embedding vocabulary with {model_name}: {len(vocab)} words")
        embeddings = model.encode(
            vocab,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        embeddings = embeddings.astype(np.float32)
        embeddings = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        )
        _word_embedding_cache[cache_key] = embeddings
    return _word_embedding_cache[cache_key]


# ─── Similarity ──────────────────────────────────────────────────────────────

def cosine_similarity_matrix(
    topics_a: list[Topic],
    topics_b: list[Topic],
) -> np.ndarray:
    """Compute pairwise cosine similarity between two sets of topics.

    Args:
        topics_a: first set of topics (must have embeddings)
        topics_b: second set of topics (must have embeddings)

    Returns:
        (len(topics_a), len(topics_b)) cosine similarity matrix
    """
    emb_a = np.stack([t.embedding for t in topics_a])  # (A, D)
    emb_b = np.stack([t.embedding for t in topics_b])  # (B, D)
    return emb_a @ emb_b.T


def find_nearest_topics(
    query_topics: list[Topic],
    reference_topics: list[Topic],
    top_k: int = 5,
) -> list[list[tuple[int, float]]]:
    """For each query topic, find top-k nearest reference topics.

    Returns:
        List of lists. Each inner list has (ref_idx, similarity) tuples,
        sorted by descending similarity.
    """
    sim_matrix = cosine_similarity_matrix(query_topics, reference_topics)
    results = []

    for i in range(len(query_topics)):
        sims = sim_matrix[i]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results.append([(int(j), float(sims[j])) for j in top_indices])

    return results


def topic_diversity(topics: list[Topic]) -> float:
    """Compute topic diversity: proportion of unique words in top words.

    Higher is better (means less redundancy across topics).
    """
    all_words = []
    for t in topics:
        all_words.extend(t.words)
    if len(all_words) == 0:
        return 0.0
    return len(set(all_words)) / len(all_words)


def topic_coherence_pmi(
    topics: list[Topic],
    bow_matrix,
    vocab: list[str],
    top_n: int = 10,
) -> float:
    """Compute approximate topic coherence using PMI on the BoW matrix.

    Uses the normalized pointwise mutual information (NPMI) metric.
    """
    import scipy.sparse as sp

    n_docs = bow_matrix.shape[0]
    # Binary document-word matrix
    binary = (bow_matrix > 0).astype(np.float32)
    if sp.issparse(binary):
        binary = binary.toarray()

    word2idx = {w: i for i, w in enumerate(vocab)}
    doc_freq = binary.sum(axis=0).flatten()  # (vocab_size,)

    coherences = []
    for topic in topics:
        words = topic.words[:top_n]
        indices = [word2idx[w] for w in words if w in word2idx]
        if len(indices) < 2:
            continue

        pairs_npmi = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                wi, wj = indices[i], indices[j]
                df_i = doc_freq[wi]
                df_j = doc_freq[wj]
                # Co-occurrence
                df_ij = (binary[:, wi] * binary[:, wj]).sum()
                if df_ij == 0:
                    pairs_npmi.append(-1.0)
                else:
                    pmi = np.log((df_ij * n_docs) / (df_i * df_j + 1e-12) + 1e-12)
                    npmi = pmi / (-np.log(df_ij / n_docs + 1e-12) + 1e-12)
                    pairs_npmi.append(float(npmi))

        if pairs_npmi:
            coherences.append(np.mean(pairs_npmi))

    return float(np.mean(coherences)) if coherences else 0.0
