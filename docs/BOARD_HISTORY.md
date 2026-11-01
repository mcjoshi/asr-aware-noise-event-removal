# Phase 2 board history

How the system was built on the Phase 2 board (Codabench upload names, Combined score).
Δ is against the parent upload, from unrounded scores.
Uploads before the organisers' unified-pipeline rule routed natural clips through a lighter path; every upload after it processes all clips identically.
‡ marks an upload with two changes.
Our final entry is the best of 45 Phase 2 uploads; one upload resolves about ±0.1 Combined, so steps below 0.1 rest on the local evidence in Table II of the report.

| upload | change (parent) | Combined | Δ |
|---|---|---|---|
| (first entry) | routed MP-SENet / FRCRN | 16.15 | |
| p2_02 | blend of MP-SENet, FRCRN, MossFormerGAN | 16.33 | +0.18 |
| p2_08 | ASR-encoder loss on MP-SENet (p2_02) | 16.51 | +0.18 |
| p2_09 | + BSRNN family, members on all pairs ‡ (p2_02) | 16.97 | +0.63 |
| p2_17a | re-weighted members, shift TTA ‡ | 17.29 | +0.32 |
| p2_18 | 104M BSRNN, encoder + transcript loss ‡ | 17.48 | +0.19 |
| p2_19 | learned per-cell fusion | 17.60 | +0.13 |
| p2_23 | fusion on members with ASR losses (p2_22, 17.55) | 17.79 | +0.24 |
| p2_28 | + TF-Locoformer with ASR losses (p2_23) | 19.16 | +1.37 |
| *unified-pipeline rule* | | | |
| p2_32 | unified pipeline, long stream on BSRNN ‡ | 19.30 | +0.14 |
| p2_33 | 8-member fusion, input add-back, test-length data ‡ | 19.68 | +0.38 |
| p2_35 | three members only | 19.70 | +0.02 |
| p2_36 | two members, no MossFormerGAN (p2_35) | 19.62 | −0.08 |
| p2_37 | long stream on all three members (p2_35) | 19.81 | +0.11 |
| p2_39 | MWER on one TF-Locoformer | 19.83 | +0.02 |
| p2_43 | TF-Locoformer weight soup | 19.86 | +0.03 |
| **p2_44** | **MWER in the fusion (final)** | **19.90** | +0.04 |
| p2_46 | words-first fusion (p2_44) | 19.88 | −0.02 |
| p2_47 | TF-Locoformer soup retrained on all pairs (p2_44) | 19.87 | −0.03 |

Final entry `p2_44_soupmwer`: SI-SDR 15.7934 dB, ΔWER +11.66 pp synthetic and +0.54 pp natural, pooled ΔWER +4.1039 pp, Combined 19.8973.
