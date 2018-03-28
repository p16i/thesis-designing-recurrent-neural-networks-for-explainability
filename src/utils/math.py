import numpy as np


def normalize_vector(x):
    dist = np.sqrt(np.sum(x * x, axis=1)).reshape(x.shape[0], 1)
    return x / dist


def cosine_similarity(u, v):
    normed_u = normalize_vector(u)
    normed_v = normalize_vector(v)

    return np.sum(normed_u * normed_v, axis=1)
