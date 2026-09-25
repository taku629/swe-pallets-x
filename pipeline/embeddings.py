#!/usr/bin/env python3
"""Generate 256-dim float32 node embeddings (.npz) for a graph file.

CPU-only, reproducible: TF-IDF (char+word n-grams over node text) ->
TruncatedSVD(256) -> L2-normalize. Keys are node ids, matching the
competition's `embeddings/<instance_id>.npz` layout.

Usage: python embeddings.py <graph.json> <out.npz>
"""
import json
import sys

import numpy as np


def main():
    g = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    ids = [n["id"] for n in g["nodes"]]
    docs = [(n.get("text") or n["id"]) for n in g["nodes"]]
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=200_000
    )
    X = vec.fit_transform(docs)
    n_comp = min(256, max(2, min(X.shape) - 1))
    E = TruncatedSVD(n_components=n_comp, random_state=0).fit_transform(X)
    if E.shape[1] < 256:
        E = np.pad(E, ((0, 0), (0, 256 - E.shape[1])))
    E = normalize(E.astype(np.float32))
    np.savez_compressed(out, **{i: E[k] for k, i in enumerate(ids)})
    print(f"{len(ids)} embeddings ({E.shape[1]}d) -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
