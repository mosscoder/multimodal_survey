import pandas as pd
import matplotlib.pyplot as plt

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
P = 3072  # patch count used as the "k=all" x position in topk_curves.csv

curves = pd.read_csv("results/topk_curves.csv", header=[0, 1], index_col=0)

ks_x = sorted(curves.index)
fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
for ax, c in zip(axes.ravel(), CLASSES):
    name = c.split()[0]
    ax.plot(ks_x, curves[(name, "mean")], marker="o")             # mean over 3 seeds
    ax.fill_between(ks_x, curves[(name, "min")], curves[(name, "max")], alpha=0.25)
    ax.set_xscale("log"); ax.set_title(c, fontsize=9, style="italic")
    ax.set_xticks([1, 4, 16, 64, 256, P]); ax.set_xticklabels(["1", "4", "16", "64", "256", "all"])
axes[0, 0].set_ylabel("pooled out-of-fold AP"); axes[1, 0].set_ylabel("pooled out-of-fold AP")
fig.suptitle("AP vs pooling aggressiveness, per species (band = seed min-max)")
fig.tight_layout(); fig.savefig("figs/ap_vs_k.png", dpi=150)
print("wrote figs/ap_vs_k.png")
