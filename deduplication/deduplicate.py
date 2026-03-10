from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from deduplication.embedding_index import EmbeddingIndex
from utils.environment_check import has_package
from utils.io_utils import safe_write_csv


def _dedup_with_faiss(embeddings: np.ndarray, threshold: float) -> list[int]:
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    sims, idxs = index.search(embeddings, 2)

    keep = []
    removed = set()
    for i, (sim_row, idx_row) in enumerate(zip(sims, idxs)):
        if i in removed:
            continue
        keep.append(i)
        neighbor = int(idx_row[1])
        if float(sim_row[1]) >= threshold:
            removed.add(neighbor)
    return keep


def _dedup_with_sklearn(embeddings: np.ndarray, threshold: float) -> list[int]:
    sim = cosine_similarity(embeddings)
    keep = []
    removed = set()
    for i in range(len(sim)):
        if i in removed:
            continue
        keep.append(i)
        dupes = np.where(sim[i] >= threshold)[0]
        for d in dupes:
            if d != i:
                removed.add(int(d))
    return keep


def deduplicate_dataset(input_csv: str, output_csv: str, similarity_threshold: float = 0.92) -> pd.DataFrame:
    df = pd.read_csv(input_csv, engine="python")
    text_col = "instruction" if "instruction" in df.columns else "instruction_1"
    texts = df[text_col].astype(str).tolist()

    embeddings = EmbeddingIndex().encode(texts)

    if has_package("faiss"):
        keep_idx = _dedup_with_faiss(embeddings, similarity_threshold)
    else:
        keep_idx = _dedup_with_sklearn(embeddings, similarity_threshold)

    out_df = df.iloc[sorted(keep_idx)].reset_index(drop=True)
    safe_write_csv(out_df, output_csv)
    return out_df
