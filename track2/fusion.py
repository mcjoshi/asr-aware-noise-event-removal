"""Learned per-cell fusion of the three member estimates (0.35M parameters).

Inputs are the members' STFTs (each least-squares scaled onto the noisy input) and the noisy STFT. A dilated
convolutional network predicts, for every time-frequency cell, softmax weights over the members (initialised at the
fixed mix w0) and a bounded readmission of the noisy cell; corrections are anchored to how far the fixed mix moved
that cell, so cells the members left untouched stay untouched.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIL = ((1, 1), (2, 2), (4, 4), (8, 8), (16, 1), (32, 2), (64, 4), (1, 8))


class Block(nn.Module):
    def __init__(self, c, dt, df):
        super().__init__()
        self.n = nn.GroupNorm(1, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=(dt, df), dilation=(dt, df))
        self.c2 = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x + self.c2(F.gelu(self.c1(F.gelu(self.n(x)))))


class FusionNet(nn.Module):
    """Inputs [B, M, F, T] complex member STFTs and [B, F, T] noisy STFT; returns the fused complex STFT."""

    def __init__(self, m=3, ch=64, w0=(0.2, 0.35, 0.45), nfft=512, hop=128, gmax=0.0, rmax=0.2, anchor=True,
                 composite=False, dil=DIL):
        super().__init__()
        m = len(w0)
        self.cfg = dict(m=m, ch=ch, w0=tuple(w0), nfft=nfft, hop=hop, gmax=gmax, rmax=rmax, anchor=anchor,
                        composite=composite, dil=tuple(map(tuple, dil)))
        self.m, self.nfft, self.hop, self.gmax, self.rmax, self.anchor = m, nfft, hop, gmax, rmax, anchor
        self.register_buffer("lw0", torch.log(torch.tensor(w0, dtype=torch.float32).clamp(min=1e-3)))
        self.register_buffer("w0", torch.tensor(w0, dtype=torch.float32), persistent=False)
        self.register_buffer("win", torch.hann_window(nfft))
        self.stem = nn.Conv2d(3 * m + 3, ch, 3, padding=1)
        self.blocks = nn.ModuleList(Block(ch, dt, df) for dt, df in dil)
        self.head = nn.Conv2d(ch, m + 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.ckpt = True

    def stft(self, x):
        return torch.stft(x, self.nfft, self.hop, window=self.win, return_complex=True)

    def istft(self, X, n):
        return torch.istft(X, self.nfft, self.hop, window=self.win, length=n)

    def forward(self, X, Y):
        B = (self.w0[None, :, None, None] * X).sum(1)                       # fixed mix
        lp = lambda z: torch.log(z.abs().pow(2) + 1e-10)
        ref = lp(Y).mean((-2, -1), keepdim=True)
        rel = X * B.conj()[:, None] / (X.abs() * B.abs()[:, None] + 1e-8)   # member phase and level against the mix
        fpos = torch.linspace(-1, 1, Y.shape[-2], device=Y.device)[None, None, :, None].expand(Y.shape[0], 1, -1, Y.shape[-1])
        h = torch.cat([lp(X) - ref[:, None], (lp(Y) - ref)[:, None], (lp(B) - ref)[:, None], rel.real, rel.imag, fpos], 1)
        h = self.stem(h.transpose(-1, -2))                                  # [B, C, T, F]
        for b in self.blocks:
            h = torch.utils.checkpoint.checkpoint(b, h, use_reentrant=False) if (self.ckpt and self.training) else b(h)
        o = self.head(F.gelu(h)).transpose(-1, -2)                          # [B, M+4, F, T]
        w = torch.softmax(self.lw0[None, :, None, None] + o[:, :self.m], 1)
        g = torch.complex(*(self.gmax * torch.tanh(o[:, self.m:self.m + 2])).unbind(1))
        r = torch.complex(*(self.rmax * torch.tanh(o[:, self.m + 2:self.m + 4])).unbind(1))
        if self.anchor:
            a = ((Y - B).abs() / (Y.abs() + 1e-8)).clamp(max=1.0)
            g, r = g * a, r * a
        self.last_w = w
        return (w * X).sum(1) * (1 + g) + r * Y

    def wave(self, x, y):
        """x [B, M, n] LS-scaled member waveforms, y [B, n] noisy -> fused waveform [B, n]."""
        n = y.shape[-1]
        X = self.stft(x.reshape(-1, n)).reshape(x.shape[0], x.shape[1], -1, (n // self.hop) + 1)
        return self.istft(self.forward(X, self.stft(y)), n)


def load_net(path, dev):
    ck = torch.load(path, map_location="cpu")
    net = FusionNet(**ck["cfg"])
    net.load_state_dict(ck["state"])
    return net.to(dev).eval()


def run_fusion(net, dev, y, X):
    """y [n] float32 noisy, X [M, n] float32 LS-scaled members -> fused waveform [n] float32."""
    with torch.no_grad():
        return net.wave(torch.from_numpy(X)[None].to(dev), torch.from_numpy(y)[None].to(dev))[0].float().cpu().numpy()
