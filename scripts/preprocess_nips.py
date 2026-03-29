"""
Preprocess NIPS corpus from Kaggle archive.zip.

Steps:
1. Read papers.csv (source_id, year, title, abstract, full_text)
2. Split each paper's full_text into paragraphs (each paragraph = 1 document)
3. Group years into 3-year bins → 11 timestamps
4. Lowercase, tokenize with spaCy, remove stopwords & punctuation
5. Build BoW with sklearn CountVectorizer (min_df=0.05%, max_df=95%)
6. Save: train_bow.npz, vocab.txt, train_times.txt, train_texts.txt, etc.

Target statistics (from paper):
  - ~276,657 documents
  - ~6,278 vocab
  - 11 timestamps
"""

import csv
import sys
import os
import re
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

# ── Year → timestamp bin mapping ─────────────────────────────────────────────
YEAR_BINS = [
    (1987, 1989),  # T0
    (1990, 1992),  # T1
    (1993, 1995),  # T2
    (1996, 1998),  # T3
    (1999, 2001),  # T4
    (2002, 2004),  # T5
    (2005, 2007),  # T6
    (2008, 2010),  # T7
    (2011, 2013),  # T8
    (2014, 2016),  # T9
    (2017, 2019),  # T10
]


def year_to_timestamp(year: int) -> int:
    for i, (lo, hi) in enumerate(YEAR_BINS):
        if lo <= year <= hi:
            return i
    return -1


def split_into_paragraphs(text: str, min_words: int = 15) -> list[str]:
    """Split full_text into paragraphs, keeping only those with >= min_words."""
    # Split on double newlines (common paragraph boundary)
    raw_paras = re.split(r"\n\s*\n", text)
    paras = []
    for p in raw_paras:
        p = p.strip()
        p = re.sub(r"\s+", " ", p)  # collapse whitespace
        if len(p.split()) >= min_words:
            paras.append(p)
    return paras


def tokenize_spacy(texts: list[str], batch_size: int = 10000) -> list[list[str]]:
    """Tokenize with spaCy: lowercase, remove stopwords, punctuation, numbers.

    Uses spacy.blank("en") instead of en_core_web_sm — we only need the
    tokenizer, not POS/NER/parser which are 10-50x slower.
    Runs with multiprocessing for speed on large corpora.
    """
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS

    # blank("en") gives us the English tokenizer without any statistical models
    nlp = spacy.blank("en")
    nlp.max_length = 2_000_000

    stopwords = STOP_WORDS
    n_cpus = min(os.cpu_count() or 1, 16)
    print(f"  Using spaCy blank tokenizer with {n_cpus} processes, batch_size={batch_size}")

    all_tokens = []
    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size, n_process=n_cpus),
        total=len(texts),
        desc="Tokenizing (spaCy)",
    ):
        tokens = []
        for tok in doc:
            t = tok.text.lower().strip()
            if t in stopwords or tok.is_punct or tok.is_space:
                continue
            # Keep only alphabetic tokens of length >= 3
            if len(t) >= 3 and t.isalpha():
                tokens.append(t)
        all_tokens.append(tokens)
    return all_tokens


def main():
    parser = argparse.ArgumentParser(description="Preprocess NIPS corpus")
    parser.add_argument(
        "--input",
        default="data/NIPS_raw/papers.csv",
        help="Path to papers.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/NIPS",
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--min-para-words",
        type=int,
        default=15,
        help="Minimum words per paragraph to keep",
    )
    parser.add_argument(
        "--min-df",
        type=float,
        default=0.0005,
        help="min_df for CountVectorizer (as fraction)",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="max_df for CountVectorizer (as fraction)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Read papers and split into paragraphs ────────────────────────
    print("Step 1: Reading papers and splitting into paragraphs...")
    documents = []  # list of (text, timestamp)
    skipped_year = 0

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading papers"):
            year = int(row["year"])
            ts = year_to_timestamp(year)
            if ts < 0:
                skipped_year += 1
                continue

            full_text = row.get("full_text", "")
            if not full_text:
                continue

            paras = split_into_paragraphs(full_text, min_words=args.min_para_words)
            for p in paras:
                documents.append((p, ts))

    print(f"  Total paragraphs (docs): {len(documents)}")
    print(f"  Skipped papers (out-of-range year): {skipped_year}")

    ts_counts = Counter(ts for _, ts in documents)
    for t in sorted(ts_counts):
        lo, hi = YEAR_BINS[t]
        print(f"  T{t} ({lo}-{hi}): {ts_counts[t]} docs")

    # ── Step 2: Tokenize with spaCy ──────────────────────────────────────────
    print("\nStep 2: Tokenizing with spaCy...")
    raw_texts = [d[0] for d in documents]
    tokenized = tokenize_spacy(raw_texts)

    # Rejoin tokens for CountVectorizer (it expects strings)
    processed_texts = [" ".join(toks) for toks in tokenized]

    # Filter out empty documents after tokenization
    valid_indices = [i for i, t in enumerate(processed_texts) if len(t.strip()) > 0]
    processed_texts = [processed_texts[i] for i in valid_indices]
    timestamps = [documents[i][1] for i in valid_indices]
    original_texts = [documents[i][0] for i in valid_indices]
    print(f"  Documents after tokenization filter: {len(processed_texts)}")

    # ── Step 3: Build BoW with sklearn ───────────────────────────────────────
    print(f"\nStep 3: Building BoW (min_df={args.min_df}, max_df={args.max_df})...")
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        min_df=args.min_df,
        max_df=args.max_df,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    bow_matrix = vectorizer.fit_transform(processed_texts)
    vocab = vectorizer.get_feature_names_out()

    print(f"  BoW shape: {bow_matrix.shape}")
    print(f"  Vocab size: {len(vocab)}")

    # Filter out documents with zero words in the final vocab
    row_sums = np.array(bow_matrix.sum(axis=1)).flatten()
    nonzero_idx = np.where(row_sums > 0)[0]
    bow_matrix = bow_matrix[nonzero_idx]
    timestamps = [timestamps[i] for i in nonzero_idx]
    original_texts = [original_texts[i] for i in nonzero_idx]

    print(f"  Documents after vocab filter: {bow_matrix.shape[0]}")

    ts_counts_final = Counter(timestamps)
    print(f"  Timestamps: {len(ts_counts_final)}")
    for t in sorted(ts_counts_final):
        lo, hi = YEAR_BINS[t]
        print(f"    T{t} ({lo}-{hi}): {ts_counts_final[t]} docs")

    # ── Step 4: Train/Test split (90/10 stratified by timestamp) ─────────────
    print("\nStep 4: Train/test split (90/10)...")
    from sklearn.model_selection import train_test_split

    indices = np.arange(bow_matrix.shape[0])
    train_idx, test_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=timestamps
    )

    train_bow = bow_matrix[train_idx]
    test_bow = bow_matrix[test_idx]
    train_times = [timestamps[i] for i in train_idx]
    test_times = [timestamps[i] for i in test_idx]
    train_texts = [original_texts[i] for i in train_idx]
    test_texts = [original_texts[i] for i in test_idx]

    print(f"  Train: {train_bow.shape[0]} docs")
    print(f"  Test:  {test_bow.shape[0]} docs")

    # ── Step 5: Save ─────────────────────────────────────────────────────────
    print(f"\nStep 5: Saving to {args.output_dir}/...")
    from scipy import sparse

    sparse.save_npz(os.path.join(args.output_dir, "train_bow.npz"), train_bow)
    sparse.save_npz(os.path.join(args.output_dir, "test_bow.npz"), test_bow)

    with open(os.path.join(args.output_dir, "vocab.txt"), "w") as f:
        for w in vocab:
            f.write(w + "\n")

    with open(os.path.join(args.output_dir, "train_times.txt"), "w") as f:
        for t in train_times:
            f.write(str(t) + "\n")

    with open(os.path.join(args.output_dir, "test_times.txt"), "w") as f:
        for t in test_times:
            f.write(str(t) + "\n")

    with open(os.path.join(args.output_dir, "train_texts.txt"), "w") as f:
        for t in train_texts:
            f.write(t.replace("\n", " ") + "\n")

    with open(os.path.join(args.output_dir, "test_texts.txt"), "w") as f:
        for t in test_texts:
            f.write(t.replace("\n", " ") + "\n")

    # Save time2id mapping
    with open(os.path.join(args.output_dir, "time2id.txt"), "w") as f:
        for i, (lo, hi) in enumerate(YEAR_BINS):
            f.write(f"{i}\t{lo}-{hi}\n")

    # Save train/test indices
    np.savetxt(
        os.path.join(args.output_dir, "train_idx.txt"), train_idx, fmt="%d"
    )
    np.savetxt(
        os.path.join(args.output_dir, "test_idx.txt"), test_idx, fmt="%d"
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Total documents:  {bow_matrix.shape[0]}")
    print(f"  Vocab size:       {len(vocab)}")
    print(f"  Timestamps:       {len(ts_counts_final)}")
    print(f"  Train docs:       {train_bow.shape[0]}")
    print(f"  Test docs:        {test_bow.shape[0]}")
    print(f"  BoW density:      {bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1]):.6f}")


if __name__ == "__main__":
    main()
