"""Fit linear probes on the cached DINOv3 features.

For each (backbone x image column) feature set:
  * species  -- 8 independent one-vs-rest logistic regressions (multilabel).
    Reported: macro / micro average precision, macro ROC-AUC, micro-F1 @ 0.5.
  * artifact_level -- one multinomial logistic regression (4 ordinal bands).
    Reported: accuracy, balanced accuracy, macro-F1, quadratic-weighted kappa.

Fit on ``train``, evaluate on ``test`` (and ``train_tier_2`` as a secondary set).
Features are standardised (train statistics) before fitting.

Writes modeling/results/probe_metrics.csv and probe_metrics.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from extract_features import IMAGE_COLS, MODELS, SPECIES

FEAT_DIR = Path(__file__).parent / "features"
RES_DIR = Path(__file__).parent / "results"
SP_COLS = [f"sp_{j}" for j in range(len(SPECIES))]


def load_split(model_key: str, col: str, split: str):
    X = np.load(FEAT_DIR / f"{model_key}__{col}__{split}.npy")
    lab = pd.read_parquet(FEAT_DIR / f"labels__{split}.parquet")
    return X, lab


def species_probe(Xtr, Ytr, evals: dict[str, tuple]) -> list[dict]:
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    out = []
    prob = {k: np.zeros_like(Y, dtype=float) for k, (_, Y) in evals.items()}
    for j in range(len(SPECIES)):
        y = Ytr[:, j]
        if y.sum() < 2 or y.sum() > len(y) - 2:
            for k in evals:
                prob[k][:, j] = y.mean()
            continue
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(Xtr_s, y)
        for k, (Xe, _) in evals.items():
            prob[k][:, j] = clf.predict_proba(scaler.transform(Xe))[:, 1]
    for k, (_, Ye) in evals.items():
        valid = [j for j in range(len(SPECIES)) if 0 < Ye[:, j].sum() < len(Ye)]
        P, Y = prob[k][:, valid], Ye[:, valid]
        pred = (P >= 0.5).astype(int)
        out.append({
            "target": "species", "eval": k, "n": len(Ye), "classes_scored": len(valid),
            "macro_AP": average_precision_score(Y, P, average="macro"),
            "micro_AP": average_precision_score(Y, P, average="micro"),
            "macro_AUC": roc_auc_score(Y, P, average="macro"),
            "micro_F1@0.5": f1_score(Y, pred, average="micro", zero_division=0),
        })
    return out


def artifact_probe(Xtr, ytr, evals: dict[str, tuple]) -> list[dict]:
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)
    out = []
    for k, (Xe, ye) in evals.items():
        pred = clf.predict(scaler.transform(Xe))
        out.append({
            "target": "artifact_level", "eval": k, "n": len(ye), "classes_scored": len(np.unique(ye)),
            "accuracy": (pred == ye).mean(),
            "balanced_acc": balanced_accuracy_score(ye, pred),
            "macro_F1": f1_score(ye, pred, average="macro", zero_division=0),
            "qwk": cohen_kappa_score(ye, pred, weights="quadratic"),
        })
    return out


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for model_key in MODELS:
        for col in IMAGE_COLS:
            tag = f"{model_key} x {col}"
            try:
                Xtr, ltr = load_split(model_key, col, "train")
                Xte, lte = load_split(model_key, col, "test")
                Xt2, lt2 = load_split(model_key, col, "train_tier_2")
            except FileNotFoundError as e:
                print(f"skip {tag}: {e}")
                continue
            print(f"\n== {tag} ==  train {Xtr.shape}  test {Xte.shape}")

            Ytr = ltr[SP_COLS].to_numpy()
            sp_evals = {"test": (Xte, lte[SP_COLS].to_numpy()),
                        "tier2": (Xt2, lt2[SP_COLS].to_numpy())}
            art_evals = {"test": (Xte, lte["artifact_level"].to_numpy()),
                         "tier2": (Xt2, lt2["artifact_level"].to_numpy())}

            for r in species_probe(Xtr, Ytr, sp_evals):
                r |= {"model": model_key, "image": col}
                rows.append(r)
                print(f"  species  {r['eval']:5s}  macroAP {r['macro_AP']:.3f}  "
                      f"microAP {r['micro_AP']:.3f}  macroAUC {r['macro_AUC']:.3f}  "
                      f"microF1 {r['micro_F1@0.5']:.3f}")
            for r in artifact_probe(Xtr, ltr["artifact_level"].to_numpy(), art_evals):
                r |= {"model": model_key, "image": col}
                rows.append(r)
                print(f"  artifact {r['eval']:5s}  acc {r['accuracy']:.3f}  "
                      f"balAcc {r['balanced_acc']:.3f}  macroF1 {r['macro_F1']:.3f}  "
                      f"qwk {r['qwk']:.3f}")

    df = pd.DataFrame(rows)
    cols = ["target", "model", "image", "eval", "n", "classes_scored",
            "macro_AP", "micro_AP", "macro_AUC", "micro_F1@0.5",
            "accuracy", "balanced_acc", "macro_F1", "qwk"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(RES_DIR / "probe_metrics.csv", index=False)
    (RES_DIR / "probe_metrics.md").write_text(
        "# DINOv3 linear-probe results\n\n"
        "Frozen backbone, CLS (`pooler_output`) features, standardised, "
        "logistic regression (`class_weight=balanced`). Fit on `train`.\n\n"
        "## species (multilabel, 8 classes)\n\n"
        + df[df.target == "species"].drop(columns=["target"]).to_markdown(index=False)
        + "\n\n## artifact_level (4 ordinal bands)\n\n"
        + df[df.target == "artifact_level"].drop(columns=["target"]).to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {RES_DIR / 'probe_metrics.csv'} and .md")


if __name__ == "__main__":
    main()
