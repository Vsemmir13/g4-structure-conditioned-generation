import numpy as np


def encode_sequences(embedder, seq_ids, batch_size):
    chunks = []
    for start in range(0, seq_ids.size(0), batch_size):
        end = min(start + batch_size, seq_ids.size(0))
        chunks.append(embedder.encode(seq_ids[start:end]))
    return np.concatenate(chunks, axis=0)
