"""
ML vs Physics-based Ice Detection Comparison
=============================================
Trains Random Forest on six polarimetric features and compares it
against the physics-based Bayesian posterior via ROC curves.

Output: results/figures/17_ml_comparison.png
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.data_generator   import load_all
from src.dfsar_analysis   import run_analysis
from src.ml_classifier    import run_ml_classifier, feature_importances
from sklearn.metrics       import roc_curve, auc as sklearn_auc


def main():
    print("[ML] Loading data...")
    data = load_all(cache=True)
    dem   = data["dem"]
    slope = data["slope"]
    illum = data["illum"]
    psr   = data["psr_mask"]
    dfsar = data["dfsar"]
    meta  = data["meta"]
    ps    = meta["pixel_scale"]

    print("[ML] Running physics pipeline...")
    res = run_analysis(dfsar, psr, slope, cpr_thresh=0.8, dop_thresh=0.13)
    CPR      = res["CPR"]
    DOP      = res["DOP"]
    ice_conf = res["ice_conf"]       # physics Bayesian posterior
    mchi     = res.get("mchi")

    if "ice_zone" not in dfsar:
        print("[ML] No ground-truth ice_zone — cannot run supervised ML.")
        return

    ice_zone = dfsar["ice_zone"].astype(bool)

    print("[ML] Training Random Forest (5-fold stratified CV)...")
    ml = run_ml_classifier(
        CPR, DOP, slope, illum, psr, ice_zone,
        dem=dem, mchi=mchi
    )
    if ml is None:
        print("[ML] Not enough class samples — aborting.")
        return

    mr  = ml["ml_result"]
    print(f"[ML] CV AUC: {mr['mean_auc']:.3f} +/- {mr['std_auc']:.3f}  "
          f"(AP={mr['mean_ap']:.3f})")

    # ── ROC comparison ──────────────────────────────────────────────────────
    psr_flat  = psr.ravel()
    yz        = ice_zone.ravel()[psr_flat]
    scores_ph = ice_conf.ravel()[psr_flat]
    scores_ml = ml["proba_map"].ravel()[psr_flat]

    fpr_ph, tpr_ph, _ = roc_curve(yz, scores_ph)
    fpr_ml, tpr_ml, _ = roc_curve(yz, scores_ml)
    auc_ph = sklearn_auc(fpr_ph, tpr_ph)
    auc_ml = sklearn_auc(fpr_ml, tpr_ml)

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0a0a1a")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    _KW = dict(facecolor="#0d0d25", edgecolor="#2a2a4a")

    # --- Panel A: ROC curves ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor("#0d0d25")
    ax0.plot(fpr_ph, tpr_ph, color="#00ccff", lw=2,
             label=f"Physics Bayesian  AUC={auc_ph:.3f}")
    ax0.plot(fpr_ml, tpr_ml, color="#ff6600", lw=2, ls="--",
             label=f"Random Forest     AUC={auc_ml:.3f}")
    ax0.plot([0,1],[0,1], color="#444", lw=1, ls=":")
    ax0.set_xlabel("False Positive Rate", color="white", fontsize=9)
    ax0.set_ylabel("True Positive Rate",  color="white", fontsize=9)
    ax0.set_title("ROC Comparison", color="#00ccff", fontsize=10, fontweight="bold")
    ax0.legend(fontsize=8, loc="lower right",
               facecolor="#0d0d25", edgecolor="#2a2a4a",
               labelcolor="white")
    ax0.tick_params(colors="white"); ax0.spines[:].set_color("#2a2a4a")

    # --- Panel B: Feature importances ---
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor("#0d0d25")
    imps  = ml["importances"]
    names = [i[0] for i in imps]
    vals  = [i[1] for i in imps]
    colors_bar = ["#00ccff","#ff6600","#66ff66","#ffcc00","#cc66ff","#ff6666"]
    bars = ax1.barh(names[::-1], vals[::-1],
                    color=colors_bar[:len(names)][::-1], alpha=0.85)
    ax1.set_xlabel("Gini Importance", color="white", fontsize=9)
    ax1.set_title("Feature Importances", color="#00ccff", fontsize=10, fontweight="bold")
    ax1.tick_params(colors="white"); ax1.spines[:].set_color("#2a2a4a")
    for bar, v in zip(bars, vals[::-1]):
        ax1.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f}", va="center", color="white", fontsize=8)

    # --- Panel C: CV fold AUCs ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#0d0d25")
    fold_aucs = mr["auc_folds"]
    fold_aps  = mr["ap_folds"]
    x = np.arange(len(fold_aucs))
    w = 0.35
    ax2.bar(x - w/2, fold_aucs, width=w, color="#00ccff", alpha=0.8, label="AUC")
    ax2.bar(x + w/2, fold_aps,  width=w, color="#ff6600", alpha=0.8, label="Avg Precision")
    ax2.axhline(mr["mean_auc"], color="#00ccff", ls="--", lw=1)
    ax2.axhline(mr["mean_ap"],  color="#ff6600", ls="--", lw=1)
    ax2.set_xticks(x); ax2.set_xticklabels([f"Fold {i+1}" for i in x], color="white")
    ax2.set_ylabel("Score", color="white", fontsize=9)
    ax2.set_title(f"5-Fold CV  (AUC={mr['mean_auc']:.3f}±{mr['std_auc']:.3f})",
                  color="#00ccff", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, facecolor="#0d0d25", edgecolor="#2a2a4a", labelcolor="white")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#2a2a4a")

    # --- Panel D: ML probability map ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("black")
    proba_disp = np.where(psr, ml["proba_map"], np.nan)
    im3 = ax3.imshow(proba_disp, cmap="plasma", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im3, ax=ax3, label="P(ice | features)", fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")
    ax3.set_title("RF Ice Probability (PSR)", color="#00ccff", fontsize=10, fontweight="bold")
    ax3.axis("off")

    # --- Panel E: Physics Bayesian posterior ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("black")
    conf_disp = np.where(psr, ice_conf, np.nan)
    im4 = ax4.imshow(conf_disp, cmap="plasma", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im4, ax=ax4, label="Bayesian P(ice)", fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")
    ax4.set_title("Physics Bayesian Posterior (PSR)", color="#00ccff", fontsize=10, fontweight="bold")
    ax4.axis("off")

    # --- Panel F: Agreement map ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("black")
    agree_map = np.where(psr, np.abs(ml["proba_map"] - ice_conf), np.nan)
    im5 = ax5.imshow(agree_map, cmap="RdYlGn_r", vmin=0, vmax=0.5, interpolation="nearest")
    plt.colorbar(im5, ax=ax5, label="|RF - Bayesian|", fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")
    ax5.set_title("Disagreement Map (PSR)", color="#00ccff", fontsize=10, fontweight="bold")
    ax5.axis("off")

    # ── Labels ──────────────────────────────────────────────────────────────
    for ax, lbl in zip([ax0,ax1,ax2,ax3,ax4,ax5], "ABCDEF"):
        ax.text(0.02, 0.97, lbl, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="white",
                va="top", ha="left")

    for ax in [ax0,ax1,ax2]:
        ax.tick_params(colors="white", which="both")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a4a")

    fig.suptitle(
        "Machine Learning vs Physics-Based Ice Detection\n"
        "Chandrayaan-2 DFSAR — Faustini PSR, Lunar South Pole",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    out = Path("results/figures/17_ml_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[ML] Saved: {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
