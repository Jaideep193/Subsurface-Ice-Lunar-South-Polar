"""
ML Classifier for Subsurface Ice Detection
==========================================

Random Forest binary classifier that combines six physically-motivated
per-pixel features to distinguish true ice signatures from:
  - Surface roughness / blocky ejecta (high CPR without DOP anomaly)
  - Illuminated rough terrain leaking into the PSR margin
  - Radiometric hot-pixels (addressed upstream by spatial coherence filter)

Features
--------
  CPR       : Circular Polarization Ratio  (primary ice proxy)
  DOP       : Degree of Polarization       (volume scatter signature)
  Pv_frac   : m-chi Pv / total power      (volume scatter from m-chi decomp)
  slope_deg : local slope in degrees       (geometric false-positive gate)
  illum     : mean annual illumination fraction (PSR proxy)
  roughness : surface roughness (local std of DEM)

Training strategy
-----------------
  Positive labels : pixels in planted ice_zone (synthetic ground truth)
  Negative labels : non-ice PSR pixels (balanced subsampling)
  CV              : 5-fold stratified cross-validation
  Metric          : ROC-AUC (averaged over folds)

References
----------
  Breiman 2001 – Random Forests; CART with bootstrap aggregation
  Fawcett 2006 – ROC analysis for binary classification
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _roughness(dem, window=5):
    """Local std of DEM in a square window — scalar proxy for surface roughness."""
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(dem.astype(np.float64), size=window)
    sq   = uniform_filter(dem.astype(np.float64) ** 2, size=window)
    return np.sqrt(np.clip(sq - mean**2, 0, None))


def build_feature_matrix(CPR, DOP, slope, illum, psr_mask, dem=None, mchi=None):
    """
    Build per-pixel feature matrix for all PSR pixels.

    Returns
    -------
    X   : (n_pixels, n_features) float32
    idx : boolean mask selecting the PSR pixels that were included
    feature_names : list[str]
    """
    idx = psr_mask.astype(bool)

    rough = _roughness(dem) if dem is not None else np.zeros_like(CPR)

    Pv = mchi["Pv_frac"] if (mchi is not None and "Pv_frac" in mchi) else np.zeros_like(CPR)

    X = np.column_stack([
        CPR[idx],
        DOP[idx],
        Pv[idx],
        slope[idx],
        illum[idx],
        rough[idx],
    ]).astype(np.float32)

    feature_names = ["CPR", "DOP", "Pv_frac", "slope_deg", "illum", "roughness"]
    return X, idx, feature_names


# ---------------------------------------------------------------------------
# Training and cross-validation
# ---------------------------------------------------------------------------

def train_and_evaluate(X, y, n_estimators=300, n_folds=5, random_state=42):
    """
    Train a Random Forest with stratified k-fold cross-validation.

    Parameters
    ----------
    X  : (n, p) feature matrix
    y  : (n,) binary labels {0, 1}
    n_estimators : number of trees
    n_folds      : CV folds

    Returns
    -------
    results dict with per-fold and mean AUC, AP, fitted pipeline
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestClassifier(
            n_estimators  = n_estimators,
            max_depth     = 8,
            min_samples_leaf = 4,
            class_weight  = "balanced",
            random_state  = random_state,
            n_jobs        = -1,
        )),
    ])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_res = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring=["roc_auc", "average_precision"],
        return_train_score=False,
    )

    auc_folds = cv_res["test_roc_auc"]
    ap_folds  = cv_res["test_average_precision"]

    # Fit final model on all data for full-image prediction
    pipe.fit(X, y)

    return dict(
        pipeline         = pipe,
        auc_folds        = auc_folds.tolist(),
        ap_folds         = ap_folds.tolist(),
        mean_auc         = float(auc_folds.mean()),
        std_auc          = float(auc_folds.std()),
        mean_ap          = float(ap_folds.mean()),
        std_ap           = float(ap_folds.std()),
        n_pos            = int(y.sum()),
        n_neg            = int((1 - y).sum()),
        n_folds          = n_folds,
        n_estimators     = n_estimators,
    )


def feature_importances(ml_result, feature_names):
    """Return sorted (name, importance) from the fitted RF."""
    rf   = ml_result["pipeline"].named_steps["rf"]
    imp  = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    return [(feature_names[i], float(imp[i])) for i in order]


# ---------------------------------------------------------------------------
# Prediction on full image
# ---------------------------------------------------------------------------

def predict_ice_probability(ml_result, X_full):
    """
    Apply fitted pipeline to feature matrix, return probability of ice class.

    Parameters
    ----------
    ml_result : dict from train_and_evaluate
    X_full    : (n_pixels, n_features) — same PSR pixel order as training

    Returns
    -------
    proba : (n_pixels,) float32, probability of ice
    """
    return ml_result["pipeline"].predict_proba(X_full)[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline: extract → train → predict
# ---------------------------------------------------------------------------

def run_ml_classifier(CPR, DOP, slope, illum, psr_mask, ice_zone,
                      dem=None, mchi=None):
    """
    End-to-end ML ice detection on PSR pixels.

    Parameters
    ----------
    ice_zone : boolean ndarray — ground-truth ice labels (synthetic)

    Returns
    -------
    dict with ml_result, proba_map, feature_names, importances
    """
    X, psr_idx, feature_names = build_feature_matrix(
        CPR, DOP, slope, illum, psr_mask, dem=dem, mchi=mchi
    )

    y = ice_zone[psr_idx].astype(np.int32)

    # Guard: need both classes
    if y.sum() == 0 or (1 - y).sum() == 0:
        return None

    ml_result = train_and_evaluate(X, y)
    proba_psr  = predict_ice_probability(ml_result, X)

    # Map probabilities back to full image grid
    proba_map = np.zeros(CPR.shape, dtype=np.float32)
    proba_map[psr_idx] = proba_psr

    imps = feature_importances(ml_result, feature_names)

    return dict(
        ml_result     = ml_result,
        proba_map     = proba_map,
        feature_names = feature_names,
        importances   = imps,
        psr_idx       = psr_idx,
    )
