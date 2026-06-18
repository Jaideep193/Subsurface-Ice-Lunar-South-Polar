"""
Project Results Summary
========================
Collects all key numerical results from the pipeline and writes
results/summary.json — a machine-readable record of every headline metric.

Useful for judges/reviewers who want to parse results programmatically,
and for cross-checking figures against the underlying numbers.

Output: results/summary.json
"""
import sys, os, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))


def _f(x, decimals=4):
    """Round a scalar to <decimals> significant figures for JSON output."""
    if x is None or np.isnan(float(x)):
        return None
    return round(float(x), decimals)


def collect():
    from src.data_generator import load_all
    from src.dfsar_analysis import run_analysis
    from src.thermal_model  import compute_surface_temperature
    from src.ice_volume     import (estimate_ice_volume, monte_carlo_uncertainty,
                                    compute_penetration_depth_map)
    from src.morphology     import run_morphology
    from src.landing_site   import composite_landing_score, select_landing_site

    t0 = time.time()
    print("[Summary] Loading data...")
    data  = load_all(cache=True)
    psr   = data["psr_mask"]; slope  = data["slope"]; illum = data["illum"]
    dem   = data["dem"];      dfsar  = data["dfsar"]; ohrc  = data["ohrc"]
    meta  = data["meta"]
    gs    = meta["grid_size"]; ps    = meta["pixel_scale"]
    dsc_c = meta["dsc_center"]; dsc_r = meta["dsc_radius"]

    print("[Summary] Running DFSAR analysis...")
    res  = run_analysis(dfsar, psr, slope, cpr_thresh=0.8, dop_thresh=0.13)
    CPR  = res["CPR"]; DOP = res["DOP"]; ice = res["ice_mask"]

    print("[Summary] Estimating ice volume...")
    ice_result = estimate_ice_volume(CPR, ice, pixel_scale=ps)
    depth_map  = compute_penetration_depth_map(CPR)

    print("[Summary] Running Monte Carlo (300 samples)...")
    unc = monte_carlo_uncertainty(CPR, ice, pixel_scale=ps, n_samples=300)

    print("[Summary] Computing temperature...")
    T = compute_surface_temperature(illum)

    print("[Summary] Morphology...")
    morph = run_morphology(dem, ohrc, dsc_c, dsc_r, pixel_scale=ps)

    print("[Summary] Landing site scoring...")
    roughness = morph["roughness"]; bd = morph["boulder_density"]
    composite, fmaps = composite_landing_score(
        slope, roughness, bd, illum, dem, psr, dsc_c, dsc_r, ps)
    candidates = select_landing_site(composite, psr, slope, illum,
                                     n_candidates=5, exclusion_radius=100,
                                     factor_maps=fmaps)

    # ── Thermal stability breakdown ───────────────────────────────────────
    stable   = ice & (T < 70.0)
    seasonal = ice & (T >= 70.0) & (T < 110.0)
    unstable = ice & (T >= 110.0)
    n_ice    = int(ice.sum())

    # ── Validation stats ─────────────────────────────────────────────────
    val  = res.get("validation", {})
    fp   = res.get("fp_analysis", {})
    roc  = res.get("roc")

    # ── Landing site ─────────────────────────────────────────────────────
    best = candidates[0] if candidates else {}
    CENTER_LAT = -87.0; CENTER_LON = 0.0; M_PER_DEG = 111_320.0
    import math
    def px_to_latlon(r, c):
        lat = CENTER_LAT + (gs/2.0 - r) * ps / M_PER_DEG
        lon = CENTER_LON + (c - gs/2.0) * ps / (M_PER_DEG * math.cos(math.radians(CENTER_LAT)))
        return round(lat, 6), round(lon, 6)

    best_lat, best_lon = px_to_latlon(best.get("row", gs//2), best.get("col", gs//2)) if best else (None, None)

    # ── ISRU propellant (rigorous) ────────────────────────────────────────
    ICE_DENSITY = 917.0
    water_p50_kg = unc["p50_m3"] * ICE_DENSITY
    water_p50_t  = water_p50_kg / 1000.0

    # PEM electrolysis: H2O → H2 + ½O2
    # Water: 11.19% H by mass, 88.81% O by mass
    # System efficiency η=0.70 (includes balance-of-plant losses)
    ETA_ELEC = 0.70
    H2_MASS_FRAC = 0.1119
    O2_MASS_FRAC = 0.8881
    h2_p50_t  = water_p50_t * H2_MASS_FRAC * ETA_ELEC
    o2_p50_t  = water_p50_t * O2_MASS_FRAC * ETA_ELEC

    # Sabatier reaction: CO2 + 4H2 → CH4 + 2H2O (ISRU methane path)
    # H2 feed mass → CH4: CH4/4H2 = 16/8 = 2.0 mass ratio
    ch4_p50_t = h2_p50_t * 2.0
    # O2 required to combust CH4: CH4 + 2O2 (mass: 16 + 64 = 80 → 64 O2 per 16 CH4)
    o2_for_ch4_t = ch4_p50_t * 4.0

    # Delta-v budget context: Lunar orbit → surface ~1.7 km/s
    # LOX/LH2 Isp~450 s, LOX/CH4 Isp~363 s; both achievable from ISRU
    dv_lunar_landing_kms = 1.70

    elapsed = time.time() - t0

    summary = {
        "meta": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pipeline_elapsed_s": round(elapsed, 1),
            "data_source": meta.get("source", "synthetic"),
            "target_region": "Faustini PSR, Lunar South Pole",
            "instrument": "Chandrayaan-2 DFSAR L-band 1.25 GHz",
            "grid_size_px": gs,
            "pixel_scale_m": ps,
        },

        "scene": {
            "psr_coverage_pct":   _f(psr.mean() * 100, 2),
            "dsc_center_row":     int(dsc_c[0]),
            "dsc_center_col":     int(dsc_c[1]),
            "dsc_radius_px":      int(dsc_r),
            "dsc_diameter_m":     _f(dsc_r * 2 * ps),
            "center_lat_deg":     CENTER_LAT,
            "center_lon_deg":     CENTER_LON,
        },

        "dfsar_detection": {
            "cpr_threshold":      0.8,
            "dop_threshold":      0.13,
            "dop_method":         res.get("dop_method", "fallback"),
            "ice_pixels":         n_ice,
            "ice_area_m2":        _f(n_ice * ps**2),
            "ice_area_km2":       _f(n_ice * ps**2 / 1e6, 6),
            "ice_pct_of_psr":     _f(n_ice / max(psr.sum(), 1) * 100, 3),
            "mean_cpr_ice":       _f(float(CPR[ice].mean()) if ice.any() else 0),
            "mean_dop_ice":       _f(float(DOP[ice].mean()) if ice.any() else 0),
        },

        "statistical_validation": {
            "cohens_d_cpr":       _f(val.get("cohens_d_cpr")),
            "effect_size":        val.get("effect_size", "unknown"),
            "ks_statistic":       _f(val.get("ks_statistic")),
            "ks_pvalue":          _f(val.get("ks_pvalue"), 6),
            "ks_significant":     bool(val.get("ks_significant", False)),
            "mw_pvalue_dop":      _f(val.get("mw_pvalue_dop"), 6),
            "n_ice_samples":      int(val.get("n_ice", 0)),
            "n_nonpsr_samples":   int(val.get("n_nonpsr", 0)),
        },

        "false_positive_analysis": {
            "n_candidates":       int(fp.get("n_candidates", 0)),
            "n_rough_fp":         int(fp.get("n_rough_fp", 0)),
            "pct_rough_fp":       _f(fp.get("pct_rough_fp"), 2),
            "n_ejecta_fp":        int(fp.get("n_ejecta_fp", 0)),
            "pct_ejecta_fp":      _f(fp.get("pct_ejecta_fp"), 2),
            "precision_pct":      _f(fp.get("precision", 0) * 100, 2),
        },

        "roc_performance": {
            "auc":                _f(roc["auc"], 4) if roc else None,
            "optimal_cpr_threshold": _f(roc["optimal_threshold"], 3) if roc else None,
            "optimal_tpr":        _f(roc["optimal_tpr"], 4) if roc else None,
            "optimal_fpr":        _f(roc["optimal_fpr"], 4) if roc else None,
        },

        "ice_volume": {
            "total_m3":           _f(ice_result["total_ice_volume_m3"]),
            "total_km3":          _f(ice_result["total_ice_volume_km3"], 8),
            "mean_ice_fraction":  _f(ice_result["mean_ice_concentration"], 4),
            "mean_depth_m":       _f(ice_result["penetration_depth_m"], 3),
            "bearing_area_km2":   _f(ice_result["ice_bearing_area_km2"], 6),
            "mass_kg":            _f(ice_result["estimated_mass_kg"], 1),
        },

        "monte_carlo_uncertainty": {
            "n_samples":          300,
            "sources": ["CPR noise ±0.1", "depth ±30% lognormal", "dielectric ±20% Gaussian"],
            "mean_m3":            _f(unc["mean_m3"]),
            "std_m3":             _f(unc["std_m3"]),
            "rel_unc_pct":        _f(unc["rel_unc_pct"], 1),
            "p5_m3":              _f(unc["p5_m3"]),
            "p25_m3":             _f(unc["p25_m3"]),
            "p50_m3":             _f(unc["p50_m3"]),
            "p75_m3":             _f(unc["p75_m3"]),
            "p95_m3":             _f(unc["p95_m3"]),
        },

        "thermal_stability": {
            "stable_ice_px":      int(stable.sum()),
            "stable_ice_pct":     _f(stable.sum() / max(n_ice, 1) * 100, 2),
            "stable_T_mean_K":    _f(float(T[stable].mean()) if stable.any() else None),
            "seasonal_ice_px":    int(seasonal.sum()),
            "seasonal_ice_pct":   _f(seasonal.sum() / max(n_ice, 1) * 100, 2),
            "unstable_ice_px":    int(unstable.sum()),
            "unstable_ice_pct":   _f(unstable.sum() / max(n_ice, 1) * 100, 2),
            "T_mean_psr_K":       _f(float(T[psr].mean()) if psr.any() else None),
            "T_min_psr_K":        _f(float(T[psr].min()) if psr.any() else None),
        },

        "isru_assessment": {
            "note": "P50 scenario; PEM electrolysis η=0.70; Sabatier CH4 path",
            "electrolysis_efficiency": 0.70,
            "water_p10_t":        _f(unc["p5_m3"]  * ICE_DENSITY / 1000.0),
            "water_p50_t":        _f(water_p50_t),
            "water_p90_t":        _f(unc["p95_m3"] * ICE_DENSITY / 1000.0),
            "h2_propellant_p50_t":  _f(h2_p50_t),
            "o2_propellant_p50_t":  _f(o2_p50_t),
            "ch4_sabatier_p50_t":   _f(ch4_p50_t),
            "o2_for_ch4_combust_t": _f(o2_for_ch4_t),
            "dv_context_kms":     dv_lunar_landing_kms,
            "crew_water_l_per_person_day": 1.83,
            "crew_days_p50":      _f(water_p50_t * 1000.0 / 1.83 / 1000.0, 0),
        },

        "landing_sites": {
            "n_candidates":       len(candidates),
            "best_score":         _f(best.get("score")) if best else None,
            "best_slope_deg":     _f(best.get("slope_deg")) if best else None,
            "best_illumination":  _f(best.get("illum")) if best else None,
            "best_lat_deg":       best_lat,
            "best_lon_deg":       best_lon,
        },

        "morphology": {
            "mean_slope_deg":         _f(float(slope.mean()), 3),
            "max_slope_deg":          _f(float(slope.max()), 2),
            "mean_roughness":         _f(float(morph["roughness"].mean()), 4),
            "mean_boulder_density":   _f(float(morph["boulder_density"].mean()), 4),
        },
    }
    return summary


if __name__ == "__main__":
    summary = collect()

    out = Path("results/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Summary] Saved: {out}  ({out.stat().st_size // 1024} KB)")

    # Print headline numbers
    d = summary
    print("\n  === KEY RESULTS ===")
    print(f"  Ice pixels          : {d['dfsar_detection']['ice_pixels']}")
    print(f"  Ice area            : {d['dfsar_detection']['ice_area_km2']} km2")
    print(f"  Ice volume (P50)    : {d['monte_carlo_uncertainty']['p50_m3']} m3")
    print(f"  Water mass (P50)    : {d['isru_assessment']['water_p50_t']} t")
    print(f"  H2 propellant (P50) : {d['isru_assessment']['h2_propellant_p50_t']} t")
    print(f"  CH4 via Sabatier    : {d['isru_assessment']['ch4_sabatier_p50_t']} t")
    print(f"  AUC (ROC)           : {d['roc_performance']['auc']}")
    print(f"  Cohen's d (CPR)     : {d['statistical_validation']['cohens_d_cpr']}")
    print(f"  Best landing score  : {d['landing_sites']['best_score']}")
    print(f"  Best site coords    : {d['landing_sites']['best_lat_deg']}, "
          f"{d['landing_sites']['best_lon_deg']}")
