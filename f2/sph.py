"""icosphere hierarchy with parent tracking + hierarchical (wavelet-like) coding of per-vertex fields."""
import numpy as np

def icosphere_levels(L):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
                  [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
                  [8, 6, 7], [9, 8, 1]])
    levels = [(V.copy(), F.copy(), None)]
    for _ in range(L):
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        Eu, inv = np.unique(E, axis=0, return_inverse=True)
        inv = inv.reshape(3, -1).T  # per face: edge ids of (01,12,20)
        n0 = len(V)
        M = V[Eu[:, 0]] + V[Eu[:, 1]]; M /= np.linalg.norm(M, axis=1, keepdims=True)
        V = np.concatenate([V, M])
        a, b, c = F[:, 0], F[:, 1], F[:, 2]
        ab, bc, ca = inv[:, 0] + n0, inv[:, 1] + n0, inv[:, 2] + n0
        F = np.concatenate([np.stack([a, ab, ca], 1), np.stack([b, bc, ab], 1), np.stack([c, ca, bc], 1), np.stack([ab, bc, ca], 1)])
        levels.append((V.copy(), F.copy(), Eu))  # Eu = parents of the new vertices n0..
    return levels

def entropy_bits(q):
    if len(q) == 0: return 0.0
    _, cnt = np.unique(q, return_counts=True); p = cnt / cnt.sum()
    return float(-(cnt * np.log2(p)).sum()) + 16 * len(cnt)  # + crude model cost

def hier_code(levels, field, step):
    """field: (Nfinest, C). base level raw 16 bit; each finer level predicted by parent-edge mean; residual quantized.
    returns reconstructed field and total bits."""
    field = np.atleast_2d(field.T).T
    n0 = len(levels[0][0])
    rec = np.zeros_like(field)
    rec[:n0] = field[:n0]; bits = 16.0 * field[:n0].size
    nprev = n0
    for V, F, Eu in levels[1:]:
        n = len(V)
        pred = 0.5 * (rec[Eu[:, 0]] + rec[Eu[:, 1]])
        q = np.round((field[nprev:n] - pred) / step)
        rec[nprev:n] = pred + q * step
        for ch in range(field.shape[1]): bits += entropy_bits(q[:, ch].astype(np.int64))
        nprev = n
    return rec, bits
