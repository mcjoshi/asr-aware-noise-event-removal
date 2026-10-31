"""Draws docs/figures/system.pdf: the unified inference path and the ASR-driven training losses (needs matplotlib)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, ASR = "#222222", "#555555", "#b04a2f"
fig, ax = plt.subplots(figsize=(3.45, 2.7), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 8.0)
ax.axis("off")


def box(x, y, w, h, title, sub="", fc="#ffffff", ec="#333333", fs=5.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=0.7))
    n = sub.count("\n") + 1 if sub else 0
    ty = y + h - 0.32 if sub else y + h / 2
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=fs, weight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, ty - 0.3 - 0.19 * (n - 1), sub, ha="center", va="center", fontsize=4.3, color=MUTED, linespacing=1.15)


def arrow(p, q, color=INK, ls="-", lw=0.7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=5, lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0))


# inference path
box(0.05, 4.55, 1.3, 1.0, "noisy", "16 kHz", fc="#eeeeee")
members = [("MossFormerGAN", "3.1M, 4-shift TTA", "#e3f0e3", 6.25),
           ("BSRNN", "104M, 4-shift TTA", "#e2ebf7", 4.8),
           ("TF-Locoformer", "15M, weight soup", "#efe6f5", 3.35)]
for name, sub, fc, y in members:
    box(1.75, y, 2.65, 1.0, name, sub, fc=fc, fs=5.1)
    arrow((1.35, 5.05), (1.75, y + 0.5))
    arrow((4.4, y + 0.5), (4.9, 5.05))
box(4.9, 4.0, 2.35, 2.1, "fusion", "0.34M params\nweights $w_m$ per cell\n+ bounded\nreadmission $r\\,a\\,Y$", fc="#fbeedd")
# noisy input to the fusion, routed above the members
ax.plot([0.7, 0.7, 6.08], [5.55, 7.62, 7.62], color=MUTED, lw=0.6)
arrow((6.08, 7.62), (6.08, 6.1), color=MUTED, lw=0.6)
ax.text(3.1, 7.75, "noisy input $Y$", fontsize=4.4, color=MUTED, ha="center")
box(7.6, 4.45, 1.4, 1.2, "post", "same for\nall clips", fc="#eeeeee")
arrow((7.25, 5.05), (7.6, 5.05))
arrow((9.0, 5.05), (9.35, 5.05))
ax.text(9.7, 5.05, "out", fontsize=5.4, va="center", ha="center", weight="bold")

# training-only path through the frozen scoring ASR
ax.add_patch(FancyBboxPatch((0.1, 0.1), 9.8, 2.75, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#fafafa",
                            ec="#999999", lw=0.6, ls=(0, (3, 2))))
ax.text(0.3, 2.55, "training only", fontsize=4.6, color=MUTED, style="italic")
box(2.75, 1.3, 4.5, 1.05, "SraVaani-1.0, frozen", "scoring ASR: encoder + TDT decoder", fc="#ffffff", ec=ASR)
for (sym, txt), x in zip([("$d_{\\mathrm{enc}}$", "encoder distance"), ("$\\mathcal{L}_{\\mathrm{NLL}}$", "transcript likelihood"),
                          ("$\\mathcal{L}_{\\mathrm{MWER}}$", "deletion-weighted risk")], [1.6, 5.0, 8.4]):
    ax.text(x, 0.8, sym, fontsize=5.8, ha="center", color=ASR)
    ax.text(x, 0.38, txt, fontsize=4.3, ha="center", color=MUTED)
arrow((3.1, 3.35), (4.2, 2.35), color=ASR, ls=(0, (2, 1.5)), lw=0.6)
arrow((6.07, 4.0), (5.8, 2.35), color=ASR, ls=(0, (2, 1.5)), lw=0.6)
ax.text(2.1, 3.0, "members: $d_{\\mathrm{enc}}$, $\\mathcal{L}_{\\mathrm{NLL}}$", fontsize=4.3, color=ASR, ha="center")
ax.text(7.25, 3.12, "fused output: all three", fontsize=4.3, color=ASR, ha="center")
ax.text(8.6, 1.85, "gradients reach\nonly the enhancers", fontsize=4.3, color=MUTED, ha="center", linespacing=1.15)
plt.savefig("docs/figures/system.pdf", bbox_inches="tight", pad_inches=0.02)
plt.savefig("docs/figures/system.png", bbox_inches="tight", pad_inches=0.02)
