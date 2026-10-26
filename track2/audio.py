"""Signal helpers shared by training and inference."""
import numpy as np

SR = 16000


def si_sdr(ref, est):
    """Scale-invariant SDR in dB (numpy, float64)."""
    n = min(len(ref), len(est))
    r, x = ref[:n].astype(np.float64), est[:n].astype(np.float64)
    a = np.dot(x, r) / max(np.dot(r, r), 1e-12)
    t = a * r
    e = x - t
    return float(10 * np.log10(max(np.dot(t, t), 1e-12) / max(np.dot(e, e), 1e-12)))


def si_sdr_loss(est, ref, eps=1e-8):
    """Negative mean SI-SDR of a batch [B, T] (torch), zero-mean signals."""
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / ((ref * ref).sum(-1, keepdim=True) + eps)
    t = a * ref
    e = est - t
    import torch
    return -(10 * torch.log10((t * t).sum(-1) / ((e * e).sum(-1) + eps) + eps)).mean()


def ls_scale(e, y):
    """Least-squares gain of e onto y."""
    return e * (np.dot(e, y) / max(np.dot(e, e), 1e-12))


def fit(x, n):
    """Cut or zero-pad x to n samples."""
    x = np.asarray(x)[:n]
    return np.pad(x, (0, n - len(x)))
