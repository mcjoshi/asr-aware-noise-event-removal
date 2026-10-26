"""The three enhancement members: MossFormerGAN, 104M BSRNN and TF-Locoformer.

Each member wraps a public pretrained network, loads our fine-tuned weights on top, and exposes
`enhance(y)` (numpy waveform in, numpy waveform out, any length) for inference and `loss` / `asr_loss`
for training. Training losses live in track2.losses.
"""
import os
import sys

import numpy as np
import torch

from . import mossgan as M
from .audio import SR

THIRD_PARTY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")


class MossganMember:
    """MossFormerGAN_SE_16K (ClearVoice). Clips up to 10 s decode in one pass, longer ones with ClearVoice's decoder."""

    def __init__(self, dev, init=None):
        self.dev = dev
        _, self.net = M.load_model(dev, init or None)

    def loss(self, mx, cl, lab=None):
        from .losses import enc_term, si_sdr_loss
        est, prs, _, nf = M.forward(self.net, mx)
        with torch.no_grad():
            from clearvoice.utils.misc import power_compress, stft
            sc = power_compress(stft((cl.T * nf).T, M.A, center=True, periodic=True, onesided=True).to(torch.float32))
        n = min(est.shape[-1], cl.shape[-1])
        return M.loss_fn(est[:, :n], cl[:, :n], prs, sc, si_sdr_loss) + enc_term(est[:, :n], cl[:, :n], lab)

    def asr_loss(self, mx, cl, lab=None):
        from .losses import enc_term
        est = M.forward(self.net, mx)[0]
        n = min(est.shape[-1], cl.shape[-1])
        return enc_term(est[:, :n], cl[:, :n], lab)

    def enhance(self, y):
        f = M.enhance if len(y) <= int(M.A.decode_window * SR) else M.enhance_full
        return np.asarray(f(self.net, y, self.dev), np.float32)


class BsrnnMember:
    """Band-split RNN, large-double (104M; ESPnet BSRNNSeparator, VCTK + DNS-2020 + WHAMR! universal-SE checkpoint).
    The input is variance-normalised as the checkpoint was trained; the output is least-squares scaled onto the input."""

    SRC = ("wyz/vctk_dns2020_whamr_bsrnn_large_double_noncausal",
           "exp_vctk_dns20_whamr/enh_train_enh_bsrnn_large_double_noncausal_raw/91epoch.pth", "separator.", (256, 12))
    CHUNK, HOP = 20 * SR, 15 * SR     # clips over 20 s decode in 20 s windows, 15 s hop, linear cross-fade

    def __init__(self, dev, init=None):
        from huggingface_hub import hf_hub_download
        from espnet2.enh.decoder.stft_decoder import STFTDecoder
        from espnet2.enh.encoder.stft_encoder import STFTEncoder
        from espnet2.enh.separator.bsrnn_separator import BSRNNSeparator
        repo, fn, pre, (ch, nl) = self.SRC
        sd = torch.load(hf_hub_download(repo, fn), map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        net = torch.nn.Module()
        net.enc = STFTEncoder(n_fft=960, hop_length=480, use_builtin_complex=True, default_fs=48000)
        net.dec = STFTDecoder(n_fft=960, hop_length=480, default_fs=48000)
        net.sep = BSRNNSeparator(input_dim=481, num_spk=1, num_channels=ch, num_layers=nl, target_fs=48000, causal=False)
        net.sep.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=True)
        self.net, self.dev = net.to(dev), dev
        if init:
            self.net.load_state_dict(torch.load(init, map_location=dev))

    @staticmethod
    def scale(wav):
        return wav.std(dim=-1, keepdim=True).clamp_min(1e-8)      # ESPnet normalize_variance_per_ch

    def forward(self, wav, s=None, ls=True):
        T_ = wav.shape[-1]
        ln = torch.full((wav.shape[0],), T_, device=wav.device, dtype=torch.long)
        s = self.scale(wav) if s is None else s
        f, fl = self.net.enc(wav / s, ln, fs=SR)
        pre, _, _ = self.net.sep(f, fl, {"fs": SR})
        est = (self.net.dec(pre[0], ln, fs=SR)[0] * s)[:, :T_]
        return self._ls(est, wav) if ls else est

    @staticmethod
    def _ls(est, wav):
        return est * (est * wav).sum(-1, keepdim=True) / est.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)

    def loss(self, mx, cl, lab=None):
        from .losses import enc_term, si_sdr_loss
        est = self.forward(mx)
        return si_sdr_loss(est, cl) + enc_term(est, cl, lab)

    def asr_loss(self, mx, cl, lab=None):
        from .losses import enc_term
        return enc_term(self.forward(mx), cl, lab)

    @torch.no_grad()
    def enhance(self, y):
        y = torch.from_numpy(np.asarray(y, np.float32)).to(self.dev)
        if len(y) <= self.CHUNK:
            return self.forward(y[None])[0].float().cpu().numpy()
        s, out, wsum = self.scale(y), torch.zeros_like(y), torch.zeros_like(y)
        fade = torch.linspace(0, 1, self.CHUNK - self.HOP, device=self.dev)
        for st in range(0, len(y) - (self.CHUNK - self.HOP), self.HOP):
            seg = y[st:st + self.CHUNK]
            w = torch.ones_like(seg)
            if st > 0:
                w[:len(fade)] = fade[:len(seg)]
            if st + self.CHUNK < len(y):
                w[-len(fade):] = fade.flip(0)
            out[st:st + len(seg)] += self.forward(seg[None], s, ls=False)[0] * w
            wsum[st:st + len(seg)] += w
        return self._ls((out / wsum.clamp_min(1e-8))[None], y[None])[0].cpu().numpy()


class LocoMember(BsrnnMember):
    """TF-Locoformer M (MERL, Apache-2.0; DNS-2020 checkpoint, 15M): time and frequency self-attention on a 16 ms STFT.
    Clips over 8 s decode in 8 s windows with a 6 s hop, each window least-squares scaled onto its own input, linear
    cross-fade over the overlap. Training runs under bf16 autocast; inference in fp32."""

    DIR = os.path.join(THIRD_PARTY, "tflocoformer")
    ONE_PASS, CHUNK, HOP = 8 * SR, 8 * SR, 6 * SR

    def __init__(self, dev, init=None, pretrained=None):
        import yaml
        from espnet2.enh.decoder.stft_decoder import STFTDecoder
        from espnet2.enh.encoder.stft_encoder import STFTEncoder
        sys.path.insert(1, self.DIR)
        from tflocoformer_separator import TFLocoformerSeparator
        cfg = yaml.safe_load(open(os.path.join(self.DIR, "config.yaml")))
        net = torch.nn.Module()
        net.enc, net.dec = STFTEncoder(**cfg["encoder_conf"]), STFTDecoder(**cfg["decoder_conf"])
        net.sep = TFLocoformerSeparator(input_dim=net.enc.output_dim, **cfg["separator_conf"])
        if pretrained:     # MERL's DNS checkpoint (valid.loss.ave_5best.pth), the starting point of training
            sd = torch.load(pretrained, map_location="cpu", weights_only=True)
            for mod, pre in ((net.enc, "encoder."), (net.dec, "decoder."), (net.sep, "separator.")):
                mod.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=True)
        self.net, self.dev = net.to(dev), dev
        if init:
            self.net.load_state_dict(torch.load(init, map_location=dev))

    def forward(self, wav, s=None, ls=True):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.is_grad_enabled()):
            return super().forward(wav, s, ls)

    @torch.no_grad()
    def enhance(self, y):
        y = torch.from_numpy(np.asarray(y, np.float32)).to(self.dev)
        if len(y) <= max(self.ONE_PASS, self.CHUNK):
            return self.forward(y[None])[0].float().cpu().numpy()
        s, out, wsum = self.scale(y), torch.zeros_like(y), torch.zeros_like(y)
        fade = torch.linspace(0, 1, self.CHUNK - self.HOP, device=self.dev)
        starts = list(range(0, len(y) - self.CHUNK, self.HOP)) + [len(y) - self.CHUNK]
        for st in starts:
            w = torch.ones(self.CHUNK, device=self.dev)
            if st > 0:
                w[:len(fade)] = fade
            if st + self.CHUNK < len(y):
                w[-len(fade):] = fade.flip(0)
            out[st:st + self.CHUNK] += self.forward(y[None, st:st + self.CHUNK], s)[0] * w
            wsum[st:st + self.CHUNK] += w
        return self._ls((out / wsum.clamp_min(1e-8))[None], y[None])[0].cpu().numpy()


MEMBERS = {"mossgan": MossganMember, "bsrnn_xl": BsrnnMember, "loco": LocoMember}


def enhance_tta(member, y, shifts=(0,)):
    """Shift test-time augmentation: delay the input by each shift, enhance, undo the delay, average."""
    y = np.asarray(y, np.float32)
    e = np.zeros(len(y), np.float32)
    for k in shifts:
        o = np.asarray(member.enhance(np.pad(y, (k, 0))), np.float32)[k:k + len(y)]
        e[:len(o)] += o / len(shifts)
    return e
