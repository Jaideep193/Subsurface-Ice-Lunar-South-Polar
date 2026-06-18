"""
Main pipeline – Subsurface Ice Detection in Lunar South Polar DSC
=================================================================
Usage:
    python main.py                   # use synthetic data
    python main.py --real            # use real data from data/raw/
    python main.py --real --inspect  # just list contents of ZIPs

Place Chandrayaan-2 ZIP files in:
    data/raw/dfsar/   ← DFRS ZIPs (CH2O_*_DFRS_*.zip)
    data/raw/ohrc/    ← OHRC ZIPs (CH2O_*_OHRC_*.zip)

Or drop them flat into data/raw/ – the loader will find them.
"""

import argparse
import os
import sys
import time
import numpy as np

# ── ensure src is importable ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader     import DataLoader, inspect_zip
from src.data_generator  import load_all as load_synthetic
from src.psr_mapping     import (identify_psrs, identify_doubly_shadowed_craters,
                                  lobate_rim_score, psr_summary)
from src.dfsar_analysis  import run_analysis, analysis_summary
from src.morphology      import run_morphology, morphology_summary
from src.landing_site    import (composite_landing_score, select_landing_site,
                                  landing_site_summary)
from src.rover_traverse  import build_cost_map, plan_traverse, traverse_summary
from src.ice_volume      import (estimate_ice_volume, monte_carlo_uncertainty,
                                  ice_volume_summary, compute_penetration_depth_map)
from src.visualization   import (plot_overview, plot_dfsar, plot_morphology,
                                  plot_landing_site, plot_traverse,
                                  plot_ice_volume, plot_dashboard,
                                  plot_advanced_analysis, plot_scenarios)
from src.thermal_model   import (compute_surface_temperature, thermal_summary,
                                  isru_assessment)


def parse_args():
    p = argparse.ArgumentParser(description="Lunar Ice Detection Pipeline")
    p.add_argument("--real",    action="store_true",
                   help="Load real Chandrayaan-2 data from data/raw/")
    p.add_argument("--inspect", action="store_true",
                   help="List ZIP contents only (use with --real)")
    p.add_argument("--raw-dir", default="data/raw",
                   help="Directory containing raw data ZIPs (default: data/raw)")
    p.add_argument("--no-mc",  action="store_true",
                   help="Skip Monte Carlo uncertainty (faster)")
    p.add_argument("--cpr-thresh", type=float, default=1.0,
                   help="CPR threshold for ice detection (default: 1.0)")
    p.add_argument("--dop-thresh", type=float, default=0.13,
                   help="DOP threshold for ice detection (default: 0.13)")
    return p.parse_args()


def banner(text):
    width = 62
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main():
    args = parse_args()

    # ── Optional ZIP inspection ──────────────────────────────────────────
    if args.inspect and args.real:
        from pathlib import Path
        raw_dir = Path(args.raw_dir)
        for z in sorted(raw_dir.rglob("*.zip")):
            inspect_zip(z)
        return

    # ── Step 0 – Load data ───────────────────────────────────────────────
    banner("STEP 0 – Data Loading")
    t0 = time.time()

    if args.real:
        from src.real_data_loader import load_real_data
        data = load_real_data(args.raw_dir)
    else:
        data = load_synthetic()

    dem        = data["dem"]
    slope      = data["slope"]
    illum      = data["illum"]
    illum_mean = data.get("illum_mean", illum)   # seasonal mean (synthetic only)
    illum_std  = data.get("illum_std",  None)    # seasonal variability
    psr_mask   = data["psr_mask"]
    dsc_mask   = data["dsc_mask"]
    dfsar      = data["dfsar"]
    dfsar2     = data.get("dfsar2", None)   # second pass for temporal coherence
    ohrc       = data["ohrc"]
    meta       = data["meta"]

    gs           = meta["grid_size"]
    pixel_scale  = meta["pixel_scale"]
    dsc_center   = meta.get("dsc_center", (gs // 2, gs // 2))
    dsc_radius   = meta.get("dsc_radius", 80)

    print(f"  Grid: {gs}×{gs} pixels  |  Pixel scale: {pixel_scale} m/px")
    print(f"  PSR coverage: {psr_mask.sum() / gs**2 * 100:.1f}%")
    print(f"  DSC pixels: {dsc_mask.sum()}")
    print(f"  Data source: {meta.get('source', 'synthetic')}")

    # ── Step 1 – PSR mapping ─────────────────────────────────────────────
    banner("STEP 1 – PSR Mapping & DSC Identification")

    psr_patches, psr_labels, psr_stats = identify_psrs(illum, threshold=0.01)

    dsc_labels, dsc_stats = identify_doubly_shadowed_craters(
        dem, psr_mask, slope, min_depth_m=30, min_diameter_pixels=15
    )

    # Use the pre-known DSC if detection missed it (synthetic fallback)
    if not dsc_stats:
        print("  [PSR] Using pre-defined DSC location from metadata.")
        gs = dem.shape[0]
        xx, yy = np.meshgrid(np.arange(gs), np.arange(gs))
        r_px = np.sqrt((xx - dsc_center[1])**2 + (yy - dsc_center[0])**2)
        fallback_dsc_mask = r_px < dsc_radius
        floor_elev = float(dem[fallback_dsc_mask].min()) if fallback_dsc_mask.any() else -62.0
        rim_elev   = float(dem[fallback_dsc_mask].max()) if fallback_dsc_mask.any() else -44.0
        dsc_stats = [dict(
            id=1,
            centroid=dsc_center,
            area_pixels=int(dsc_mask.sum()) if dsc_mask.any() else int(fallback_dsc_mask.sum()),
            area_m2=float((dsc_mask.sum() if dsc_mask.any() else fallback_dsc_mask.sum()) * pixel_scale**2),
            diameter_m=float(dsc_radius * 2 * pixel_scale),
            depth_m=62.0,
            depth_to_diameter=float(62.0 / (dsc_radius * 2 * pixel_scale)),
            rim_slope_deg=28.0,
            floor_elevation_m=floor_elev,
            rim_elevation_m=rim_elev,
        )]

    lrs = lobate_rim_score(dem, dsc_center, dsc_radius, pixel_scale=pixel_scale)
    print(f"  Lobate Rim Score: {lrs:.3f}")
    print(psr_summary(psr_stats, dsc_stats))

    # Thermal cold-trap model (Diviner-calibrated, Paige et al. 2010)
    T_surface = compute_surface_temperature(illum)
    print(thermal_summary(T_surface, psr_mask))

    # ── Step 2 – DFSAR analysis ──────────────────────────────────────────
    banner("STEP 2 – DFSAR Polarimetric Ice Detection")

    # Use band-appropriate CPR threshold (L-band: 0.8, S-band: 1.0)
    cpr_thresh = args.cpr_thresh
    if dfsar.get("frequency_band", "S") == "L" and args.cpr_thresh == 1.0:
        cpr_thresh = 0.8
        print(f"  [Auto] L-band detected -> using CPR threshold = {cpr_thresh}")

    dfsar_results = run_analysis(
        dfsar, psr_mask, slope,
        cpr_thresh=cpr_thresh,
        dop_thresh=args.dop_thresh,
    )

    CPR      = dfsar_results["CPR"]
    DOP      = dfsar_results["DOP"]
    ice_mask = dfsar_results["ice_mask"]
    ice_conf = dfsar_results["ice_conf"]
    anomaly  = dfsar_results["anomaly"]

    print(analysis_summary(dfsar_results))

    # m-chi decomposition summary
    mchi = dfsar_results.get("mchi")
    if mchi is not None:
        print(f"  m-chi decomposition complete:")
        print(f"    Volume scatter fraction (ice):   mean={mchi['frac_volume'].mean():.3f}")
        print(f"    Single-bounce fraction (regolith): mean={mchi['frac_single'].mean():.3f}")
        print(f"    Double-bounce fraction (boulders): mean={mchi['frac_double'].mean():.3f}")

    # Temporal coherence (synthetic: perturb a second pass; real: use dfsar2)
    from src.dfsar_analysis import temporal_coherence
    if dfsar2 is not None:
        print("  Running temporal coherence with real second pass...")
        temporal_result = temporal_coherence(
            dfsar["CPR"], dfsar["DOP"], dfsar_results["ice_mask"],
            dfsar2["CPR"], dfsar2["DOP"],
            dfsar2.get("ice_mask", dfsar_results["ice_mask"]),
        )
    else:
        # Synthetic second pass: add small noise perturbation to CPR
        rng_t = np.random.default_rng(99)
        CPR2  = np.clip(dfsar["CPR"] + rng_t.normal(0, 0.05, dfsar["CPR"].shape), 0, 3).astype(np.float32)
        DOP2  = np.clip(dfsar["DOP"] + rng_t.normal(0, 0.01, dfsar["DOP"].shape), 0, 1).astype(np.float32)
        temporal_result = temporal_coherence(
            dfsar["CPR"], dfsar["DOP"], dfsar_results["ice_mask"],
            CPR2, DOP2, dfsar_results["ice_mask"],
        )

    print(f"  Temporal coherence: stable_ice={temporal_result['n_stable']}px  "
          f"frost_candidates={temporal_result['n_frost']}px")

    # Thermal correlation: how many ice pixels lie in the T<110K cold trap?
    n_cold = int((T_surface[ice_mask] < 110.0).sum()) if ice_mask.any() else 0
    print(f"  Thermal cold-trap correlation: {n_cold}/{ice_mask.sum()} "
          f"ice pixels lie at T < 110 K  (Paige et al. 2010 stability threshold)")

    # ── Step 3 – Morphology ──────────────────────────────────────────────
    banner("STEP 3 – Crater Morphology Characterization")

    morph = run_morphology(dem, ohrc, dsc_center, dsc_radius, pixel_scale)
    print(morphology_summary(morph))

    # ── Step 4 – Landing site ────────────────────────────────────────────
    banner("STEP 4 – Landing Site Evaluation")

    roughness      = morph["roughness"]
    boulder_density = morph["boulder_density"]
    hazard         = morph["hazard"]

    composite, factor_maps = composite_landing_score(
        slope, roughness, boulder_density, illum,
        dem, psr_mask, dsc_center, dsc_radius, pixel_scale
    )

    candidates = select_landing_site(composite, psr_mask, slope, illum,
                                      n_candidates=5, exclusion_radius=100,
                                      factor_maps=factor_maps)

    if not candidates:
        print("  [Landing] No valid candidates found – using fallback.")
        fallback_row = max(0, int(dsc_center[0]) - int(500 / pixel_scale))
        candidates = [dict(rank=1, row=fallback_row, col=int(dsc_center[1]),
                           score=0.5, slope_deg=5.0, illum=0.7)]

    best_site = candidates[0]
    print(landing_site_summary(candidates))

    # ── Step 5 – Rover traverse ──────────────────────────────────────────
    banner("STEP 5 – Rover Traverse Path Planning")

    cost_map = build_cost_map(slope, morph["roughness"], psr_mask, illum,
                               pixel_scale=pixel_scale)

    traverse = plan_traverse(cost_map, dem, slope, best_site, dsc_center,
                              dsc_radius, roughness=morph["roughness"],
                              illum_fraction=illum, pixel_scale=pixel_scale)
    print(traverse_summary(traverse))

    # ── Step 6 – Ice volume estimation ───────────────────────────────────
    banner("STEP 6 – Subsurface Ice Volume Estimation")

    ice_result = estimate_ice_volume(CPR, ice_mask, pixel_scale=pixel_scale)

    uncertainty = None
    if not args.no_mc:
        print("  Running Monte Carlo uncertainty (500 samples)...")
        uncertainty = monte_carlo_uncertainty(CPR, ice_mask,
                                               pixel_scale=pixel_scale)

    print(ice_volume_summary(ice_result, uncertainty))
    print(isru_assessment(ice_result["total_ice_volume_m3"], uncertainty))

    # Ice depth scenario comparison
    banner("STEP 6b – Ice Depth Scenario Analysis")
    from src.data_generator import generate_ice_scenarios
    print("  Generating three ice depth scenarios...")
    scenarios = generate_ice_scenarios(dem, psr_mask, dsc_mask, pixel_scale)

    # ── Step 7 – Visualisation ───────────────────────────────────────────
    banner("STEP 7 – Generating Figures")

    print("  Generating Fig 1: Overview...")
    plot_overview(dem, illum, psr_mask, dsc_mask, dsc_center, dsc_radius,
                  T_surface=T_surface)

    print("  Generating Fig 2: DFSAR analysis...")
    plot_dfsar(CPR, DOP, ice_mask, ice_conf, anomaly, psr_mask,
               sensitivity=dfsar_results.get("sensitivity"),
               sensitivity_dop=dfsar_results.get("sensitivity_dop"),
               cpr_thresh=cpr_thresh,
               dop_thresh=args.dop_thresh)

    print("  Generating Fig 3: Morphology...")
    plot_morphology(dem, morph["slope"], roughness, boulder_density,
                    ohrc, hazard, morph["rim_analysis"], dsc_center, dsc_radius)

    print("  Generating Fig 4: Landing site...")
    plot_landing_site(composite, factor_maps, candidates,
                       dem, psr_mask, dsc_center, dsc_radius)

    print("  Generating Fig 5: Traverse...")
    plot_traverse(dem, cost_map, morph["slope"], traverse, psr_mask,
                  dsc_center, dsc_radius, ice_mask, illum=illum)

    print("  Generating Fig 6: Ice volume...")
    full_depth_map = compute_penetration_depth_map(CPR)
    plot_ice_volume(CPR, ice_mask, ice_result, uncertainty, dsc_center, dsc_radius,
                    full_depth_map=full_depth_map)

    print("  Generating Fig 7: Master dashboard...")
    plot_dashboard(dem, CPR, ice_mask, ice_conf, psr_mask, dsc_center,
                   dsc_radius, composite, traverse, ice_result,
                   uncertainty=uncertainty)

    print("  Generating Fig 8: Advanced analysis (m-chi / ROC / temporal)...")
    roc_result = dfsar_results.get("roc")
    plot_advanced_analysis(CPR, DOP, ice_mask, psr_mask,
                           mchi=mchi, roc=roc_result,
                           temporal=temporal_result)

    print("  Generating Fig 9: Ice depth scenario comparison...")
    plot_scenarios(scenarios, psr_mask)

    # ── GeoTIFF export ───────────────────────────────────────────────────
    banner("STEP 7b – GeoTIFF Export")
    try:
        from src.export import export_all, export_summary
        center_lat = meta.get("center_lat", -87.2)
        center_lon = meta.get("center_lon",  84.1)
        written = export_all(
            results_dir   = "results",
            center_lat    = center_lat,
            center_lon    = center_lon,
            pixel_scale_m = pixel_scale,
            dem           = dem,
            ice_mask      = ice_mask,
            CPR           = CPR,
            DOP           = DOP,
            ice_conf      = ice_conf,
            composite_score = composite,
            hazard        = morph["hazard"],
            T_surface     = T_surface,
            mchi          = mchi,
        )
        print(export_summary(written))
    except ImportError:
        print("  [Export] rasterio not available — skipping GeoTIFF export.")
    except Exception as exc:
        print(f"  [Export] GeoTIFF export failed: {exc}")

    # ── Final summary ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print("""
  Key Results
  -----------""")
    print(f"  PSR patches identified   : {len(psr_stats)}")
    print(f"  Doubly shadowed craters  : {len(dsc_stats)}")
    print(f"  Ice-flagged pixels       : {ice_mask.sum():,}")
    print(f"  Ice-bearing area         : {ice_result['ice_bearing_area_km2']:.4f} km2")
    print(f"  Mean ice concentration   : {ice_result['mean_ice_concentration']*100:.1f} %")
    print(f"  Total ice volume         : {ice_result['total_ice_volume_m3']:.2f} m3")
    print(f"  Best landing site        : row={best_site['row']}, col={best_site['col']}")
    print(f"                             score={best_site['score']:.3f}, slope={best_site['slope_deg']:.1f} deg")
    print(f"  Traverse distance        : {traverse['metrics'].get('total_distance_m', 0):.0f} m")
    print(f"  Temporal coherence       : stable_ice={temporal_result['n_stable']}px, "
          f"frost={temporal_result['n_frost']}px")
    print("""
  Figures saved to : results/figures/
    00_dashboard.png        (master summary)
    01_overview.png
    02_dfsar_analysis.png
    03_morphology.png
    04_landing_site.png
    05_traverse.png
    06_ice_volume.png
    07_advanced_analysis.png (m-chi / ROC / temporal coherence)
    08_ice_scenarios.png     (depth scenario comparison)
  GeoTIFFs saved to : results/geotiff/
""")


if __name__ == "__main__":
    main()
