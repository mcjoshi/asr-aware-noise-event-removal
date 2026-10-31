"""MossFormerGAN_SE_16K (ClearVoice) with its decode path made differentiable for fine-tuning.

The forward replicates ClearVoice's own decoder exactly (RMS normalisation, padding to a multiple of the hop, STFT, power
compression, model, uncompression, iSTFT, denormalisation), so training and inference use the same signal path.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F


class A:
    """STFT and decode settings the ClearVoice helpers read."""
    win_len, win_inc, fft_len, win_type = 400, 100, 400, "hamming"
    sampling_rate = 16000
    decode_window = 10.0            # seconds; longer clips go through ClearVoice's segmenting decoder
    one_time_decode_length = 10.0


def forward(model, wav):
    """wav [B, T] on the model's device -> (estimate [B, T], (power-compressed real, imag), input spec, norm factor)."""
    from clearvoice.utils.misc import power_compress, power_uncompress, stft, istft
    T = wav.shape[-1]
    nf = torch.sqrt(wav.shape[-1] / torch.sum(wav ** 2, dim=-1).clamp_min(1e-12))
    nfr = int(np.ceil(T / A.win_inc))
    pad = nfr * A.win_inc - T
    x = torch.cat([wav, wav[:, :pad]], -1) if pad else wav
    x = (x.T * nf).T
    spec = stft(x, A, center=True, periodic=True, onesided=True).to(torch.float32)
    spec = power_compress(spec).permute(0, 1, 3, 2)
    out = model(spec)
    pr, pi = out[0].permute(0, 1, 3, 2), out[1].permute(0, 1, 3, 2)
    y = istft(power_uncompress(pr, pi).squeeze(1), A, center=True, periodic=True, onesided=True)
    y = y.squeeze(1) if y.dim() == 3 else y
    return (y.T / nf).T[:, :T], (pr, pi), spec, nf


def load_model(dev, ckpt=None):
    """ClearVoice's pretrained MossFormerGAN_SE_16K (downloaded on first use), optionally overwritten by a fine-tuned state dict."""
    from clearvoice import ClearVoice
    from clearvoice.networks import SpeechModel
    if dev.startswith("cuda"):
        torch.cuda.set_device(0)
        torch.zeros(1, device="cuda")
    SpeechModel.get_free_gpu = lambda self: 0 if dev.startswith("cuda") else None
    cv = ClearVoice(task="speech_enhancement", model_names=["MossFormerGAN_SE_16K"])
    sm = cv.models[0]
    ckdir = getattr(getattr(sm, "args", None), "checkpoint_dir", "checkpoints/MossFormerGAN_SE_16K")
    if not os.path.exists(os.path.join(ckdir, "last_best_checkpoint")):   # ClearVoice ignores download failures
        raise RuntimeError(f"MossFormerGAN_SE_16K weights missing in {ckdir}: ClearVoice could not download them")
    sm.device = torch.device(dev)
    if hasattr(sm, "args"):
        sm.args.use_cuda = 1 if dev.startswith("cuda") else 0
    m = sm.model.to(dev)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location=dev))
    return cv, m


@torch.no_grad()
def enhance(model, y, dev):
    """Whole-clip forward (clips up to A.decode_window seconds)."""
    return forward(model, torch.from_numpy(np.asarray(y, np.float32)).to(dev).unsqueeze(0))[0].squeeze().cpu().numpy().astype(np.float32)


@torch.no_grad()
def enhance_full(model, y, dev):
    """ClearVoice's own decoder for long clips (10 s segments, 75% stride)."""
    from clearvoice.utils.decode import decode_one_audio_mossformergan_se_16k as dec
    out = np.asarray(dec(model, torch.device(dev), np.asarray(y, np.float32)[None, :], A), np.float32).squeeze()
    return out[:len(y)] if len(out) >= len(y) else np.concatenate([out, np.asarray(y, np.float32)[len(out):]])


def loss_fn(est, cl, prs, spec_cl, si_sdr_loss):
    """SI-SDR plus the L1 loss on power-compressed real and imaginary spectra (ClearVoice's MossFormerGAN recipe)."""
    pr, pi = prs
    return si_sdr_loss(est, cl) + 15.0 * (F.l1_loss(pr.squeeze(1), spec_cl[:, 0]) + F.l1_loss(pi.squeeze(1), spec_cl[:, 1]))
