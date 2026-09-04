import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity (rotation + scale + translation) mapping src onto dst:
    dst ~= scale * (r @ src.T).T + t. To go the other way (map a dst-frame point back into
    src's frame), invert explicitly: src_pt = (r.T @ (dst_pt - t)) / scale."""
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = (dst_c.T @ src_c) / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2, 2] = -1
    r = u @ d @ vt
    var_src = (src_c ** 2).sum() / len(src)
    scale = np.trace(np.diag(s) @ d) / var_src
    t = mu_dst - scale * r @ mu_src
    return r, scale, t


def robust_umeyama(src: np.ndarray, dst: np.ndarray, trim_ratio: float = 0.2, n_iters: int = 4):
    """Iteratively refits Umeyama after keeping only the best-fitting (1-trim_ratio) fraction
    of points under the CURRENT fit, each round. A single least-squares fit over the whole
    trajectory gets dragged toward compromising between segments that need different
    alignments - e.g. an early segment that drifted before a later loop closure corrected it
    back, which stays "off" in a way plain Umeyama can't distinguish from genuine estimation
    error. Trimming the worst-fitting fraction each round converges to an alignment fit
    mostly on the well-converged part of the trajectory, at the cost of no longer being a
    single objective, ground-truth-free number - it's an aid for understanding a specific
    known-lopsided case, not a universal replacement for the plain metric everywhere.

    Returns (r, s, t, inlier_mask) - inlier_mask is which points survived the last round."""
    r, s, t = umeyama(src, dst)
    mask = np.ones(len(src), dtype=bool)
    for _ in range(n_iters):
        aligned = s * (r @ src.T).T + t
        err = np.linalg.norm(aligned - dst, axis=1)
        n_keep = max(3, int(len(src) * (1 - trim_ratio)))
        keep_idx = np.argsort(err)[:n_keep]
        mask = np.zeros(len(src), dtype=bool)
        mask[keep_idx] = True
        r, s, t = umeyama(src[mask], dst[mask])
    return r, s, t, mask
