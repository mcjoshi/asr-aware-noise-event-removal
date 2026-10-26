"""Post-processing applied identically to every clip after the fusion.

1. Span-limit: keep the model output where it changed the audio (smoothed |input - output| above 1.5% of its clip
   maximum, dilated by 150 ms, cross-faded over 10 ms); elsewhere the model left the audio essentially unchanged.
2. Silence gate at -50 dB relative to the clip's 95th-percentile envelope.
3. Peak guard: scale down to 0.99 if the clip would clip.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

THR, PAD_MS, GATE_DB = 0.015, 150, -50.0
XF, GS = 161, 321           # 10 ms cross-fade and 20 ms envelope windows at 16 kHz


def post(y, e, thr=THR, pad_ms=PAD_MS, gate_db=GATE_DB):
    if len(e) != len(y):
        e = e[:len(y)] if len(e) > len(y) else np.concatenate([e, np.zeros(len(y) - len(e), e.dtype)])
    s = uniform_filter1d(np.abs(y - e), XF)
    msk = (s > thr * s.max()).astype(np.float32) if s.max() > 0 else np.zeros(len(y), np.float32)
    msk = maximum_filter1d(msk, 2 * int(pad_ms * 16) + 1)
    w = uniform_filter1d(msk, XF)
    o = w * e + (1 - w) * y
    env = uniform_filter1d(np.abs(o), GS)
    pk = np.percentile(env, 95)
    if pk > 0:
        keep = maximum_filter1d((env > pk * 10 ** (gate_db / 20)).astype(np.float32), GS)
        o = o * uniform_filter1d(keep, XF)
    p = np.abs(o).max()
    return o * (0.99 / p) if p > 0.99 else o


def post_weights(y, e, thr=THR, pad_ms=PAD_MS, gate_db=GATE_DB):
    """The span-limit weight w and gate g of post() (post = (w * e + (1 - w) * y) * g before the peak guard), for training
    the fusion through the shipped post-processing."""
    s = uniform_filter1d(np.abs(y - e), XF)
    m = (s > thr * s.max()).astype(np.float32) if s.max() > 0 else np.zeros(len(y), np.float32)
    w = uniform_filter1d(maximum_filter1d(m, 2 * int(pad_ms * 16) + 1), XF)
    o = w * e + (1 - w) * y
    env = uniform_filter1d(np.abs(o), GS)
    pk = np.percentile(env, 95)
    g = np.ones(len(y), np.float32)
    if pk > 0:
        g = uniform_filter1d(maximum_filter1d((env > pk * 10 ** (gate_db / 20)).astype(np.float32), GS), XF)
    return w.astype(np.float32), g.astype(np.float32)
