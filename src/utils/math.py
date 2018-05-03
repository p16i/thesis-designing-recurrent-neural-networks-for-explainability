import numpy as np


def compute_length(x):
    dist = np.sqrt(np.sum(x * x, axis=1))
    return dist


def cosine_similarity(u, v):

    dot_prod = np.sum(u * v, axis=1)

    length = compute_length(u) * compute_length(v)

    cosine_sim = dot_prod / length
    cosine_sim[length == 0] = 0

    return cosine_sim
