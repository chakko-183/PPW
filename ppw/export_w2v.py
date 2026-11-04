"""Jalankan SEKALI di komputer Anda (yang punya gensim) untuk mengekspor vektor kata.

Hasil: w2v_vectors.npy dan w2v_meta.json -> taruh di folder yang sama dengan app.py
"""
import json

import numpy as np
from gensim.models import Word2Vec

m = Word2Vec.load("model_skipgram.model")

np.save("w2v_vectors.npy", m.wv.vectors.astype("float32"))
json.dump(
    {
        "words": list(m.wv.index_to_key),
        "window": int(m.window),
        "vector_size": int(m.vector_size),
    },
    open("w2v_meta.json", "w", encoding="utf-8"),
)

print("Selesai:", len(m.wv.index_to_key), "kata,", m.vector_size, "dimensi")
