"""Training losses through the mandated scoring ASR (SraVaani-1.0, TDT, frozen): gradients reach the enhancer through the
ASR's front end and encoder, never into the ASR.

- encoder distance: mean per-frame cosine distance between the encoder states of the estimate and of clean speech
- transcript NLL: TDT negative log-likelihood of the clean speech's greedy SraVaani tokens (teacher labels) on the
  estimate's lattice, with crop-edge handling for windows cut out of longer recordings
- MWER: deletion-weighted sampled minimum word error; n ancestral samples from the TDT decoder per window, REINFORCE with a
  mean baseline on their word cost against the teacher transcript (a deletion costs MWER_DEL words)
"""
import os
import sys

import torch
import torch.nn.functional as F

from .audio import si_sdr_loss  # noqa: F401  (re-exported for the members)

ASR_REPO, ASR_REV = "ARTPARK-IISc/SraVaani-1.0", "f5dd5358325a5208775b91dad98918e079ea2b27"
ENC_W = 0.0     # encoder-distance weight (--enc-loss)
NLL_W = 0.0     # transcript-NLL weight (--nll-loss)
MWER_W, MWER_N, MWER_DEL, MWER_MIN_S = 0.0, 4, 2.0, 8.0   # MWER weight, samples per window, deletion cost, shortest window (s)
SR = 16000
FRAME, MARGIN = 1280, 3     # samples per SraVaani encoder frame; frames dropped at crop edges cut out of a recording
_ASR = {}


def asr_path():
    """The pinned SraVaani-1.0 snapshot (SRAVAANI=<dir> overrides)."""
    if os.environ.get("SRAVAANI"):
        return os.environ["SRAVAANI"]
    from huggingface_hub import snapshot_download
    return snapshot_download(ASR_REPO, revision=ASR_REV)


def _asr(dev):
    """The mandated ASR, frozen: gradients flow through its front end and encoder, never into them."""
    if dev not in _ASR:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(asr_path(), trust_remote_code=True).to(dev).eval()
        m._ensure_loaded()
        for p in m.parameters():
            p.requires_grad_(False)
        _ASR[dev] = m
    return _ASR[dev]


def asr_feats(wav):
    """SraVaani's feature extractor (differentiable copy of extract_features): log-mel [B, 128, T] and lengths."""
    m = _asr(wav.device)
    P = m._p
    x = torch.cat((wav[:, :1], wav[:, 1:] - P["preemph"] * wav[:, :-1]), dim=1).float()
    spec = torch.stft(x, n_fft=P["n_fft"], hop_length=P["hop_length"], win_length=P["win_length"],
                      window=m._window.float(), center=True, pad_mode="constant", return_complex=True)
    x = (spec.real.pow(2) + spec.imag.pow(2) + 1e-12).sqrt().pow(P["mag_power"])
    x = torch.log(torch.matmul(m._fb.float(), x) + P["log_zero_guard_value"])
    sl = torch.full((wav.shape[0],), wav.shape[1] // P["hop_length"], dtype=torch.long, device=wav.device)
    x = sys.modules[type(m).__module__]._normalize_per_feature(x, sl, P["CONSTANT"])
    pad = torch.arange(x.size(-1), device=wav.device)[None] >= sl[:, None]
    return x.masked_fill(pad[:, None], P["pad_value"]).to(m._io_dtype), sl


def asr_encode(wav):
    """SraVaani's feature extractor and encoder: [B, D, T]."""
    out = _asr(wav.device).encoder(*asr_feats(wav))
    return (out[0] if isinstance(out, (tuple, list)) else out).float()


def enc_dist(est, cl, ee=None):
    """Mean per-frame cosine distance between the encoder states of the estimate (ee, when already
    computed) and of clean speech."""
    n = min(est.shape[-1], cl.shape[-1])
    with torch.no_grad():
        ec = asr_encode(cl[..., :n])
    ee = asr_encode(est[..., :n]) if ee is None else ee
    t = min(ee.shape[-1], ec.shape[-1])
    return (1 - F.cosine_similarity(ee[..., :t], ec[..., :t], dim=1)).mean()


def load_teacher(teacher_dir):
    """Greedy SraVaani tokens and frames of every decoded clean bed, by bed key (t2p2_tdt_teacher.py)."""
    lab = {}
    for f in sorted(os.listdir(teacher_dir)):
        if f.endswith(".pt"):
            lab.update(torch.load(os.path.join(teacher_dir, f)))
    return lab


def crop_labels(src, beds, seg):
    """Teacher labels per crop: (seq, C, U, lo, hi). seq holds the C bed tokens before the crop's scored
    window (prediction-network context only) followed by the U tokens scored inside it; lo and hi are the
    frame ranges where a lattice path may start and end. src rows are Stream's (pool, index, offset);
    beds[pool][index] is (tokens, frames, bed length). An edge cut out of a longer bed drops the tokens of its
    outer MARGIN frames, where crop context differs from the full bed's and words straddle the cut, and the
    path may start (end) anywhere in those frames, so straddling speech is neither an insertion nor pushed
    towards blank."""
    seqs, C, U, lo, hi = [], [], [], [], []
    for p, k, off in src.tolist():
        tk, fr, n = beds[p][k]
        a = MARGIN if off > 0 else 0
        b = seg // FRAME - MARGIN if off + seg < n else 1 << 30
        rel = fr.double() - off / FRAME
        c, u = int((rel < a).sum()), int(((rel >= a) & (rel < b)).sum())
        seqs.append(tk[:c + u].long()); C.append(c); U.append(u)
        lo.append((0, a)); hi.append((b, seg // FRAME) if b < 1 << 30 else (b, b))
    pad = torch.zeros(len(seqs), max(1, max(map(len, seqs))), dtype=torch.long)
    for i, s in enumerate(seqs):
        pad[i, :len(s)] = s
    return pad, torch.tensor(C), torch.tensor(U), torch.tensor(lo), torch.tensor(hi)


def tdt_nll(ee, seq, C, U, lo, hi):
    """Per-item NLL of the teacher tokens seq[C:C+U] on the TDT lattice joint(ee, pred(seq)), paths starting
    in the frame range lo [B, 2] and ending in hi [B, 2] (see tdt_forward); seq[:C] only conditions the
    prediction network. The ASR is frozen: gradients reach ee, never its weights."""
    m = _asr(ee.device)
    cfg, dev = m.config, ee.device
    nd, blank = cfg.num_durations, cfg.blank_id
    B, _, T = ee.shape
    seq, C = seq.to(dev), C.to(dev)
    Ux = max(1, int(U.max()))
    with torch.no_grad():
        h0 = torch.zeros(cfg.pred_rnn_layers, B, cfg.pred_hidden, device=dev, dtype=m._io_dtype)
        g = m.decoder_joint.decoder(torch.cat([seq.new_full((B, 1), blank), seq], 1).int(), h0, h0)[0]
        j = (C[:, None] + torch.arange(Ux + 1, device=dev)).clamp(max=seq.shape[1])
        g = g.gather(2, j[:, None].expand(-1, g.shape[1], -1))
        toks = seq.gather(1, j[:, :-1].clamp(max=seq.shape[1] - 1))
    z = m.decoder_joint.joint(ee.to(m._io_dtype), g)                           # [B, T, Ux+1, V+1+nd]
    z = z.to(torch.promote_types(z.dtype, torch.float32))
    lse = torch.logsumexp(z[..., :-nd], -1)
    emit = z[:, :, :Ux].gather(-1, toks[:, None, :, None].expand(B, T, Ux, 1))[..., 0] - lse[:, :, :Ux]
    ld = F.log_softmax(z[..., -nd:], -1)
    return tdt_forward(emit.double(), (z[..., blank] - lse).double(), ld.double(), U.to(dev), lo.to(dev),
                       hi.to(dev).clamp(max=T), list(cfg.durations))


def tdt_forward(emit, blk, ld, U, lo, hi, durs, NEG=-1e30):
    """NeMo's TDT forward recursion (only blanks end a path) on emit [B, T, Ux], blk [B, T, Ux+1] and
    ld [B, T, Ux+1, len(durs)] log-probs: -log-likelihood of paths that start at token 0 on a frame in
    [lo[:, 0], lo[:, 1]] and end after token U on a frame in [hi[:, 0], hi[:, 1]], both uniformly likely;
    0 for an empty window. Looped over frames, vectorised over batch and tokens; duration-0 emissions chain
    tokens within a frame, solved as a log-cumsum along tokens. Finite NEG instead of -inf keeps the backward
    pass NaN-free."""
    B, T, Ux = emit.shape
    live = torch.arange(Ux + 1, device=emit.device)[None] <= U[:, None]
    tokm = live[:, 1:]
    start = torch.full((1, Ux + 1), NEG, dtype=emit.dtype, device=emit.device)
    start[0, 0] = 0.0
    inc, ll = [None] * T, start.new_full((B,), NEG)
    for t in range(T):
        a = start.new_full((B, Ux + 1), NEG) if inc[t] is None else inc[t]
        a = torch.where(((lo[:, 0] <= t) & (lo[:, 1] >= t))[:, None], torch.logaddexp(a, start), a)
        if 0 in durs:
            w = torch.where(tokm, emit[:, t] + ld[:, t, :Ux, durs.index(0)], 0.0)
            cs = torch.cat([w.new_zeros(B, 1), w.cumsum(1)], 1)
            a = cs + torch.logcumsumexp(a - cs, 1)
        a = torch.where(live, a, NEG)
        for k, d in enumerate(durs):
            if d == 0:
                continue
            vb = a + blk[:, t] + ld[:, t, :, k]
            end = (hi[:, 0] <= t + d) & (hi[:, 1] >= t + d)
            ll = torch.logaddexp(ll, torch.where(end, vb.gather(1, U[:, None])[:, 0], NEG))
            if t + d < T:
                ok = (hi[:, 1] > t + d)[:, None]
                vt = torch.where(ok & tokm, a[:, :-1] + emit[:, t] + ld[:, t, :Ux, k], NEG)
                v = torch.where(ok, torch.cat([vb[:, :1], torch.logaddexp(vb[:, 1:], vt)], 1), NEG)
                inc[t + d] = v if inc[t + d] is None else torch.logaddexp(inc[t + d], v)
    n = ((lo[:, 1] - lo[:, 0] + 1) * (hi[:, 1] - hi[:, 0] + 1)).to(ll.dtype)
    return torch.where(hi[:, 1] > lo[:, 0], n.log() - ll, 0.0)


@torch.no_grad()
def tdt_sample(ee, ctx, a, b, n):
    """n ancestral samples per item from SraVaani's TDT decoder on encoder states ee [B, D, T], each starting on frame a[i]
    with the prediction network primed on ctx[i] (tokens before the window) and stopping at frame b[i]. Returns lists of
    token lists, item-major (item i, sample j at i * n + j). A blank never takes duration 0."""
    m = _asr(ee.device)
    cfg, dj, dev = m.config, m.decoder_joint, ee.device
    nd, blank, durs = cfg.num_durations, cfg.blank_id, torch.tensor(cfg.durations, device=dev)
    rows = [i for i in range(ee.shape[0]) for _ in range(n)]
    g, h, c = [], [], []
    for i in rows:
        o = dj.decoder(torch.tensor([[blank] + ctx[i]], dtype=torch.int32, device=dev), *[torch.zeros(
            cfg.pred_rnn_layers, 1, cfg.pred_hidden, device=dev, dtype=m._io_dtype)] * 2)
        g.append(o[0][..., -1:]); h.append(o[1]); c.append(o[2])
    g, h, c = torch.cat(g), torch.cat(h, 1), torch.cat(c, 1)
    R, idx = len(rows), torch.tensor(rows, device=dev)
    t = torch.tensor([a[i] for i in rows], device=dev)
    end = torch.tensor([b[i] for i in rows], device=dev)
    sym, out = torch.zeros(R, dtype=torch.long, device=dev), [[] for _ in rows]
    while True:
        act = (t < end).nonzero()[:, 0]
        if not len(act):
            return out
        f = ee[idx[act], :, t[act]][..., None].to(m._io_dtype)
        z = dj.joint(f, g[act]).float()[:, 0, 0]
        k = torch.multinomial(F.softmax(z[:, :-nd], -1), 1)[:, 0]
        dl = z[:, -nd:].clone()
        dl[:, durs == 0] = torch.where((k == blank)[:, None], -1e9, dl[:, durs == 0])
        d = durs[torch.multinomial(F.softmax(dl, -1), 1)[:, 0]]
        emit = k != blank
        sym[act] = torch.where(d == 0, sym[act] + 1, 0)
        d = torch.where(sym[act] >= cfg.max_symbols, torch.ones_like(d), d)
        sym[act] = torch.where(d > 0, 0, sym[act])
        if emit.any():
            e = act[emit]
            o = dj.decoder(k[emit][:, None].int(), h[:, e], c[:, e])
            g[e], h[:, e], c[:, e] = o[0], o[1], o[2]
            for r, tk in zip(e.tolist(), k[emit].tolist()):
                out[r].append(tk)
        t[act] = t[act] + d


def word_cost(hyp, ref, w_del):
    """Levenshtein cost of hyp against ref (word lists), a deletion costing w_del, an insertion or substitution 1."""
    prev = [j * 1.0 for j in range(len(hyp) + 1)]
    for i, r in enumerate(ref, 1):
        cur = [i * w_del]
        for j, w in enumerate(hyp, 1):
            cur.append(min(prev[j] + w_del, cur[j - 1] + 1, prev[j - 1] + (r != w)))
        prev = cur
    return prev[-1]


def mwer_term(ee, lab):
    """Deletion-weighted minimum-word-error risk on the TDT lattice: MWER_N sampled hypotheses per window, each weighted by
    its word cost against the teacher transcript minus the window mean (REINFORCE with a mean baseline), so probability
    moves from the hypotheses that lose words to the ones that keep them. Divided by samples x reference words."""
    seq, C, U, lo, hi = lab
    T = ee.shape[-1]
    a, b = lo[:, 1].tolist(), hi[:, 0].clamp(max=T).tolist()
    ctx = [seq[i, :C[i]].tolist() for i in range(len(C))]
    hyps = tdt_sample(ee, ctx, a, b, MWER_N)
    sp = _asr(ee.device)._get_tokenizer()
    refs = [sp.decode(seq[i, C[i]:C[i] + U[i]].tolist()).split() for i in range(len(C))]
    cost = torch.tensor([word_cost(sp.decode(hp).split(), refs[r // MWER_N], MWER_DEL) for r, hp in enumerate(hyps)])
    adv = (cost.view(-1, MWER_N) - cost.view(-1, MWER_N).mean(1, keepdim=True)).flatten()
    keep = (adv != 0).nonzero()[:, 0]
    if not len(keep):
        return ee.new_zeros(())
    # one checkpointed lattice per window over its kept hypotheses: the backward recomputes them a window at a time, so
    # only one window's [n, T, U, vocab] joint is ever held
    z = ee.new_zeros(())
    for i in range(len(C)):
        ks = [r for r in keep.tolist() if r // MWER_N == i]
        if not ks:
            continue
        c = int(C[i])
        hs = torch.zeros(len(ks), c + max(1, max(len(hyps[r]) for r in ks)), dtype=torch.long)
        for q, r in enumerate(ks):
            hs[q, :c + len(hyps[r])] = torch.tensor(ctx[i] + hyps[r], dtype=torch.long)
        n = len(ks)
        nll = torch.utils.checkpoint.checkpoint(
            tdt_nll, ee[i:i + 1].expand(n, -1, -1), hs, C[i:i + 1].expand(n), torch.tensor([len(hyps[r]) for r in ks]),
            torch.tensor([[a[i], a[i]]]).expand(n, 2), torch.tensor([[b[i], b[i]]]).expand(n, 2), use_reentrant=False)
        z = z - (adv[ks].to(nll) * nll).sum().float()
    return z / (MWER_N * max(1, sum(len(r) for r in refs)))


def enc_term(est, cl, lab=None):
    """Encoder-distance and teacher-NLL terms on one shared encoder pass over the estimate; the NLL is
    summed over the batch and divided by its teacher-token count. Windows of MWER_MIN_S or longer add the MWER term."""
    nll = NLL_W > 0 and lab is not None
    z = est.new_zeros(())
    if ENC_W <= 0 and not nll:
        return z
    n = min(est.shape[-1], cl.shape[-1])
    ee = asr_encode(est[..., :n])
    if ENC_W > 0:
        z = z + ENC_W * enc_dist(est[..., :n], cl[..., :n], ee)
    if nll:
        seq, C, U, lo, hi = lab
        ntok = U[hi[:, 1].clamp(max=ee.shape[-1]) > lo[:, 0]].sum().clamp_min(1)
        z = z + NLL_W * tdt_nll(ee, seq, C, U, lo, hi).sum().float() / ntok.to(z.device)
        if MWER_W > 0 and n >= MWER_MIN_S * SR:
            z = z + MWER_W * mwer_term(ee, lab)
    return z
