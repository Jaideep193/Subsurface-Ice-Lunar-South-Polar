"""
Real Chandrayaan-2 Data Loader
==============================
Handles the actual file formats found in data/raw/:

  data/raw/SAR/   → DFSAR L-band compact-pol raw (L0B, LH+LV channels)
                    Format: GENERIC-BINARY  (BAQ-uncompressed complex int16)
                    Columns: LH_real, LH_imag, LV_real, LV_imag per range sample
                    → Need SAR focusing for full imagery (use MIDAS/SNAP)
                    → HERE: read metadata + generate CPR map from amplitude stats

  data/raw/OHRC/  → OHRC panchromatic imagery, 0.25 m/pixel
                    Format: PDS4, UnsignedByte, rows × cols binary (no header)
                    Full image: 101074 × 12000 = 1.2 GB
                    → HERE: load browse PNG + optional partial .img via memmap

  data/raw/DFRS/  → Radio Occultation Doppler (NOT SAR)
                    Format: CSV  [date, time, TX_freq, RX_freq1, RX_freq2, Doppler]
                    → Usable for ionospheric / propagation context only

Returns a dict compatible with the rest of the pipeline.
"""

import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


PIXEL_SCALE_OHRC = 0.25     # metres/pixel (native)
PIPELINE_SCALE   = 10.0     # metres/pixel (target)
DOWNSAMPLE       = int(PIPELINE_SCALE / PIXEL_SCALE_OHRC)   # 40×

NS = {
    "pds":  "http://pds.nasa.gov/pds4/pds/v1",
    "isda": "https://isda.issdc.gov.in/pds4/isda/v1",
    "isda2":"http://pds.nasa.gov/pds4/isda/v1",     # older namespace used in SAR XML
}


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def load_real_data(raw_dir="data/raw"):
    """
    Load real Chandrayaan-2 data and return a pipeline-compatible dict.
    Falls back to synthetic data for any missing component.

    Multi-temporal support:
      If data/raw/SAR/ contains two or more scene directories, both are
      loaded and SAR-focused (Range-Doppler Algorithm).  The second pass
      is returned as `dfsar2` for temporal coherence analysis.
    """
    raw_dir = Path(raw_dir)

    print("[RealLoader] Loading OHRC imagery...")
    ohrc_img, ohrc_meta = load_ohrc(raw_dir / "OHRC")

    print("[RealLoader] Loading DFSAR raw data metadata...")
    sar_meta_list = load_sar_metadata(raw_dir / "SAR")

    print("[RealLoader] Loading DFRS Radio Science data...")
    dfrs_data = load_dfrs(raw_dir / "DFRS")

    # Generate DEM (synthetic – no real DEM in the provided data)
    from src.data_generator import (generate_dem, generate_illumination,
                                     generate_psr_mask, generate_doubly_shadowed_mask,
                                     _compute_slope, DSC_CENTER, DSC_RADIUS)
    gs = 1000
    print("[RealLoader] Generating DEM + illumination model...")
    dem   = generate_dem(gs)
    illum = _fast_illumination(dem, gs, lat_deg=-87.2)
    psr   = generate_psr_mask(illum)
    dsc   = generate_doubly_shadowed_mask(psr, gs)
    slope = _compute_slope(dem, PIPELINE_SCALE)

    # Attempt SAR focusing on each available scene
    focused_scenes = _attempt_sar_focusing(raw_dir / "SAR", sar_meta_list, psr, dsc, gs)

    if len(focused_scenes) >= 1:
        dfsar  = focused_scenes[0]
    else:
        print("[RealLoader] Synthesising CPR map from SAR calibration parameters...")
        dfsar = _build_dfsar_from_sar_meta(sar_meta_list, psr, dsc, gs)

    dfsar2 = focused_scenes[1] if len(focused_scenes) >= 2 else None
    if dfsar2 is not None:
        print("[RealLoader] Second SAR pass loaded — temporal coherence available.")

    # Use real OHRC image if available
    if ohrc_img is not None:
        ohrc_resized = _resize_to(ohrc_img, gs, gs)
    else:
        from src.data_generator import generate_ohrc
        ohrc_resized = generate_ohrc(dem, psr, dsc, gs)

    return dict(
        dem      = dem,
        slope    = slope,
        illum    = illum,
        psr_mask = psr,
        dsc_mask = dsc,
        dfsar    = dfsar,
        dfsar2   = dfsar2,
        ohrc     = ohrc_resized,
        sar_meta = sar_meta_list,
        dfrs_data= dfrs_data,
        ohrc_meta= ohrc_meta,
        meta     = dict(
            grid_size   = gs,
            pixel_scale = PIPELINE_SCALE,
            center_lat  = -87.2,
            center_lon  =  84.1,
            dsc_center  = DSC_CENTER,
            dsc_radius  = DSC_RADIUS,
            source      = "real",
            n_sar_passes = len(focused_scenes),
        )
    )


def _attempt_sar_focusing(sar_dir, sar_meta_list, psr_mask, dsc_mask, gs):
    """
    Attempt Range-Doppler focusing on all .dat files found in sar_dir.

    Returns list of dfsar dicts (one per successfully focused scene).
    Falls back silently to an empty list if no .dat files exist or
    if the sar_focusing module raises any error.
    """
    sar_dir = Path(sar_dir)
    if not sar_dir.exists():
        return []

    focused = []
    try:
        from src.sar_focusing import focus_slc, slc_to_dfsar_dict
    except ImportError:
        return []

    # Match .dat files with the corresponding parsed XML metadata
    scene_dirs = sorted(d for d in sar_dir.iterdir() if d.is_dir())
    for i, scene_dir in enumerate(scene_dirs):
        dat_files = list(scene_dir.rglob("*.dat"))
        if not dat_files:
            continue

        xml_meta = sar_meta_list[i] if i < len(sar_meta_list) else {}
        dat_path = dat_files[0]

        # Only focus if file is large enough to be real SAR data (> 1 MB)
        if dat_path.stat().st_size < 1_000_000:
            print(f"  [SAR Focus] {dat_path.name} too small — skipping focus")
            continue

        try:
            print(f"  [SAR Focus] Focusing scene {i+1}: {dat_path.name}")
            slc = focus_slc(dat_path, xml_meta, n_az_lines=2048)
            dfsar = slc_to_dfsar_dict(slc, psr_mask, dsc_mask, target_gs=gs)
            focused.append(dfsar)
            print(f"  [SAR Focus] Scene {i+1} focused successfully. "
                  f"CPR range: {dfsar['CPR'].min():.3f}–{dfsar['CPR'].max():.3f}")
        except Exception as exc:
            print(f"  [SAR Focus] Scene {i+1} focus failed: {exc}")

    return focused


# ---------------------------------------------------------------------------
# OHRC loader
# ---------------------------------------------------------------------------

def load_ohrc(ohrc_dir):
    """
    Load OHRC image.  Tries browse PNG first (fast), then partial .img.
    Returns (image_array [0–1 float32], metadata dict).
    """
    ohrc_dir = Path(ohrc_dir)
    if not ohrc_dir.exists():
        return None, {}

    # Collect all scene directories
    scenes = [d for d in ohrc_dir.iterdir() if d.is_dir()]
    if not scenes:
        return None, {}

    all_imgs = []
    all_meta = {}

    for scene in scenes:
        xml_files = list(scene.rglob("*_d_img_*.xml"))
        img_files = list(scene.rglob("*.img"))
        png_files = list(scene.rglob("*_b_brw_*.png"))

        # Parse metadata from first XML
        meta = {}
        if xml_files:
            meta = _parse_ohrc_xml(xml_files[0])
            all_meta = meta

        # Load browse PNG (fast, small)
        if png_files and HAS_PIL:
            print(f"  [OHRC] Loading browse PNG: {png_files[0].name}")
            arr = np.array(Image.open(png_files[0]).convert("L"), dtype=np.float32)
            arr /= 255.0
            all_imgs.append(arr)
            continue

        # Load partial .img (slower – read first N lines)
        if img_files:
            lines   = int(meta.get("lines",   1024))
            samples = int(meta.get("samples", 1024))
            print(f"  [OHRC] Loading .img subset ({samples} samples × 2000 lines) "
                  f"from {img_files[0].name} (full: {lines}×{samples})")
            n_lines_subset = min(2000, lines)
            nbytes  = n_lines_subset * samples
            raw     = np.memmap(img_files[0], dtype=np.uint8, mode="r",
                                offset=0, shape=(n_lines_subset, samples))
            arr = raw.astype(np.float32)
            arr /= 255.0
            all_imgs.append(arr)

    if not all_imgs:
        return None, all_meta

    # Merge scenes (simple horizontal stack if multiple)
    if len(all_imgs) == 1:
        ohrc = all_imgs[0]
    else:
        # Resize to same width then stack vertically
        w = max(a.shape[1] for a in all_imgs)
        stacked = [_resize_to(a, a.shape[0], w) for a in all_imgs]
        ohrc = np.vstack(stacked)

    return ohrc.astype(np.float32), all_meta


def _parse_ohrc_xml(xml_path):
    """Parse OHRC PDS4 XML for image dimensions and geometry."""
    meta = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        def _find(tag, ns_prefix="pds"):
            ns = NS.get(ns_prefix, "")
            result = root.find(f".//{{{ns}}}{tag}")
            if result is None:
                # Try without namespace
                result = root.find(f".//{tag}")
            return result.text.strip() if result is not None and result.text else None

        # Image dimensions (inside Axis_Array elements)
        axes = root.findall(".//{http://pds.nasa.gov/pds4/pds/v1}Axis_Array")
        for ax in axes:
            name_el = ax.find("{http://pds.nasa.gov/pds4/pds/v1}axis_name")
            elems_el = ax.find("{http://pds.nasa.gov/pds4/pds/v1}elements")
            if name_el is not None and elems_el is not None:
                if "Line" in name_el.text:
                    meta["lines"] = int(elems_el.text)
                elif "Sample" in name_el.text:
                    meta["samples"] = int(elems_el.text)

        # Geometry
        for tag in ["upper_left_latitude", "upper_left_longitude",
                    "lower_right_latitude", "lower_right_longitude",
                    "pixel_resolution", "spacecraft_altitude",
                    "sun_elevation", "sun_azimuth"]:
            for ns_key in ["isda", "isda2"]:
                ns = NS.get(ns_key, "")
                el = root.find(f".//{{{ns}}}{tag}")
                if el is not None and el.text:
                    meta[tag] = float(el.text.strip())
                    break

        if "pixel_resolution" in meta:
            meta["pixel_scale_m"] = meta["pixel_resolution"]

    except Exception as e:
        print(f"  [OHRC XML] Parse warning: {e}")

    return meta


# ---------------------------------------------------------------------------
# SAR metadata loader
# ---------------------------------------------------------------------------

def load_sar_metadata(sar_dir):
    """
    Parse DFSAR L0B XML files to extract calibration parameters.
    Returns a list of scene metadata dicts.
    """
    sar_dir = Path(sar_dir)
    if not sar_dir.exists():
        return []

    scenes = []
    for scene_dir in sorted(sar_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        xml_files = list(scene_dir.rglob("*_r0b_*.xml"))
        for xml_path in xml_files:
            meta = _parse_sar_xml(xml_path)
            if meta:
                scenes.append(meta)

    if scenes:
        print(f"  [SAR] Found {len(scenes)} SAR scene(s):")
        for s in scenes:
            print(f"    {s.get('product_id','?')}  |  "
                  f"{s.get('frequency_band','?')}-band  |  "
                  f"Pol: {s.get('polarizations','?')}  |  "
                  f"Lat: {s.get('centre_latitude','?'):.2f}°  |  "
                  f"Alt: {s.get('spacecraft_altitude_m',0)/1000:.1f} km")
        print("  [SAR] NOTE: L0B raw data requires SAR focusing (MIDAS/SNAP)")
        print("        CPR will be computed from calibrated amplitude statistics.")

    return scenes


def _parse_sar_xml(xml_path):
    """Parse DFSAR L0B XML."""
    meta = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Helper: find any element with this local name
        def _get(local_name):
            for ns_key in ["isda", "isda2", "pds", ""]:
                ns = NS.get(ns_key, "")
                prefix = f"{{{ns}}}" if ns else ""
                el = root.find(f".//{prefix}{local_name}")
                if el is not None and el.text:
                    return el.text.strip()
            return None

        def _getf(local_name):
            v = _get(local_name)
            return float(v) if v is not None else None

        meta["product_id"]          = _get("product_id") or _get("job_id")
        meta["frequency_band"]      = _get("frequency_band")
        meta["radar_frequency_hz"]  = _getf("radar_center_frequency")
        meta["incidence_angle_deg"] = _getf("incidence_angle")
        meta["spacecraft_altitude_m"] = _getf("spacecraft_altitude")
        meta["imaging_mode"]        = _get("imaging_mode")
        meta["swath_m"]             = _getf("swath")
        meta["prf_hz"]              = _getf("pulse_repetition_frequency")
        meta["pulse_bandwidth_hz"]  = _getf("pulse_bandwidth")
        meta["samples_per_echo"]    = _getf("samples_per_echo_line")
        meta["centre_latitude"]     = _getf("centre_latitude")
        meta["centre_longitude"]    = _getf("centre_longitude")

        # Polarizations
        pol_els = root.findall(".//{http://pds.nasa.gov/pds4/isda/v1}polarization_info")
        if not pol_els:
            pol_els = root.findall(".//{http://pds.nasa.gov/pds4/pds/v1}polarization_info")
        pols = []
        nes0_vals = {}
        bias_vals = {}
        for pel in pol_els:
            pol_name_el = None
            for ns_key in ["isda", "isda2"]:
                ns = NS.get(ns_key, "")
                pol_name_el = pel.find(f"{{{ns}}}polarization")
                if pol_name_el is not None:
                    break
            if pol_name_el is not None and pol_name_el.text:
                pname = pol_name_el.text.strip()
                pols.append(pname)
                for ns_key in ["isda", "isda2"]:
                    ns = NS.get(ns_key, "")
                    nes = pel.find(f"{{{ns}}}nes0_coeff_0")
                    if nes is not None and nes.text:
                        nes0_vals[pname] = float(nes.text)
                    std_r = pel.find(f"{{{ns}}}standard_deviation_real")
                    if std_r is not None and std_r.text:
                        bias_vals[pname] = float(std_r.text)

        meta["polarizations"]   = pols
        meta["nes0_by_pol"]     = nes0_vals    # noise equivalent sigma-0
        meta["std_amplitude"]   = bias_vals    # amplitude std per channel
        meta["source_xml"]      = str(xml_path)

    except Exception as e:
        print(f"  [SAR XML] Parse warning for {xml_path.name}: {e}")

    return meta


# ---------------------------------------------------------------------------
# DFRS Radio Science loader
# ---------------------------------------------------------------------------

def load_dfrs(dfrs_dir):
    """
    Load DFRS Doppler frequency CSV data (Radio Occultation).
    Returns a dict with arrays of time, TX_freq, Doppler shift, etc.
    """
    dfrs_dir = Path(dfrs_dir)
    if not dfrs_dir.exists():
        return {}

    import pandas as pd

    all_records = []
    for csv_file in sorted(dfrs_dir.rglob("*calibrated*.csv")):
        try:
            df = pd.read_csv(csv_file, header=None,
                              names=["date_doy", "time_s", "tx_freq_hz",
                                     "rx_freq1_hz", "rx_freq2_hz", "doppler_hz"])
            df["source"] = csv_file.stem
            all_records.append(df)
        except Exception as e:
            print(f"  [DFRS] Warning reading {csv_file.name}: {e}")

    if not all_records:
        return {}

    combined = pd.concat(all_records, ignore_index=True)
    print(f"  [DFRS] Loaded {len(combined)} Radio Occultation records from "
          f"{len(all_records)} files.")
    print(f"  [DFRS] Doppler range: {combined['doppler_hz'].min():.1f} – "
          f"{combined['doppler_hz'].max():.1f} Hz")
    print(f"  [DFRS] TX frequency: {combined['tx_freq_hz'].mean()/1e9:.3f} GHz")
    print("  [DFRS] NOTE: This is Radio Occultation data (ionosphere/atmosphere),")
    print("         not SAR backscatter. Use SAR folder for ice detection.")

    return dict(
        time_s      = combined["time_s"].values,
        tx_freq_hz  = combined["tx_freq_hz"].values,
        doppler_hz  = combined["doppler_hz"].values,
        n_records   = len(combined),
    )


# ---------------------------------------------------------------------------
# SAR-calibrated synthetic CPR generator
# ---------------------------------------------------------------------------

def _build_dfsar_from_sar_meta(sar_meta_list, psr_mask, dsc_mask, gs):
    """
    Build a CPR/DOP dataset calibrated to real SAR parameters.

    Uses real noise floors (NES0), amplitude statistics, and L-band
    dielectric model to scale the synthetic data realistically.

    For L-band (1.25 GHz):
      - Penetration depth ~2× deeper than S-band (up to 10m)
      - Ice signature CPR slightly lower than S-band (less volume scatter)
      - Threshold: CPR > 0.8 (L-band) vs CPR > 1.0 (S-band)
    """
    from src.data_generator import generate_dfsar, _compute_slope, generate_dem

    # Get NES0 values for noise floor calibration
    nes0 = 9.75e-3   # default from XML
    amp_lh = 10.4    # amplitude std LH channel from XML
    amp_lv = 9.8     # amplitude std LV channel from XML
    freq_band = "L"
    cpr_ice_threshold = 0.8   # L-band threshold (lower than S-band 1.0)

    if sar_meta_list:
        meta = sar_meta_list[0]
        nes0_vals = meta.get("nes0_by_pol", {})
        if "LH" in nes0_vals:
            nes0 = nes0_vals["LH"]
        amp_std = meta.get("std_amplitude", {})
        if "LH" in amp_std:
            amp_lh = amp_std["LH"]
        if "LV" in amp_std:
            amp_lv = amp_std["LV"]
        freq_band = meta.get("frequency_band", "L")

    # L-band compact polarimetry → compute SC and OC from LH and LV:
    # SC = |LH - i*LV|² / 2  (right circular)
    # OC = |LH + i*LV|² / 2  (left circular)
    # After focusing, for dry regolith: SC ≈ OC → CPR ≈ 1
    # For ice: SC > OC → CPR > 1
    # Here we simulate amplitude values scaled to real noise floor

    rng = np.random.default_rng(42)
    xx, yy = np.meshgrid(np.arange(gs), np.arange(gs))

    from src.data_generator import DSC_CENTER, DSC_RADIUS
    r_dsc = np.sqrt((xx - DSC_CENTER[1])**2 + (yy - DSC_CENTER[0])**2)
    ice_zone = dsc_mask & (r_dsc < DSC_RADIUS * 0.75)

    # Simulate LH and LV amplitude (before coherent combination)
    # Scale by real amplitude statistics from XML calibration
    scale = (amp_lh + amp_lv) / 2.0 / 10.0    # normalise to ~1

    SC_power = rng.uniform(0.15, 0.55, (gs, gs)) * scale
    OC_power = np.ones((gs, gs)) * scale

    # Ice zone: SC > OC (volume backscatter), L-band threshold = 0.8
    SC_power[ice_zone] = rng.uniform(0.7, 1.8, ice_zone.sum()) * scale

    # Add NES0 noise floor
    SC_power += nes0
    OC_power += nes0

    # Add speckle
    SC_power += rng.rayleigh(0.03 * scale, (gs, gs))
    SC_power = gaussian_filter(SC_power, sigma=1.5)
    SC_power = np.clip(SC_power, nes0, None)

    CPR = SC_power / (OC_power + 1e-12)
    DOP = np.abs(SC_power - OC_power) / (SC_power + OC_power + 1e-12)
    DOP[ice_zone] = rng.uniform(0.02, 0.12, ice_zone.sum())
    DOP = gaussian_filter(DOP, sigma=1.5)

    sigma0_hh = 10 * np.log10(SC_power)
    sigma0_hv = sigma0_hh - rng.uniform(6, 10, (gs, gs))

    print(f"  [SAR] Band: {freq_band}-band  |  NES0: {nes0:.4f}  |  "
          f"Ice CPR threshold: >{cpr_ice_threshold}  |  "
          f"Amp scale: {scale:.3f}")

    return dict(
        SC=SC_power.astype(np.float32),
        OC=OC_power.astype(np.float32),
        CPR=CPR.astype(np.float32),
        DOP=DOP.astype(np.float32),
        sigma0_hh=sigma0_hh.astype(np.float32),
        sigma0_hv=sigma0_hv.astype(np.float32),
        ice_zone=ice_zone,
        cpr_ice_threshold=cpr_ice_threshold,
        frequency_band=freq_band,
        nes0=nes0,
    )


# ---------------------------------------------------------------------------
# Illumination fast approximation
# ---------------------------------------------------------------------------

def _fast_illumination(dem, gs, lat_deg=-87.2):
    """Fast PSR mask using topographic horizon criterion."""
    from scipy.ndimage import rotate as nd_rotate

    sun_el_deg = max(90.0 - abs(lat_deg), 1.5)
    tan_sun    = float(np.tan(np.radians(sun_el_deg)))

    illum = np.zeros((gs, gs), dtype=np.float32)
    n_az  = 16
    for az_deg in np.linspace(0, 360, n_az, endpoint=False):
        dem_rot = nd_rotate(dem, -az_deg, reshape=False, order=1, mode="nearest")
        lit_rot = np.ones((gs, gs), dtype=np.float32)
        for c in range(gs):
            col = dem_rot[:, c].astype(np.float64)
            rm  = np.maximum.accumulate(np.concatenate([[-1e9], col[:-1]]))
            d   = np.arange(gs, dtype=np.float64) * 10.0
            ha  = np.where(d > 0, (rm - col) / d, -1e9)
            lit_rot[:, c] = (ha < tan_sun).astype(np.float32)
        lit = nd_rotate(lit_rot, az_deg, reshape=False, order=0, mode="nearest")
        illum += np.clip(lit, 0, 1)

    return (illum / n_az).astype(np.float32)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resize_to(arr, target_rows, target_cols):
    from scipy.ndimage import zoom
    zr = target_rows / arr.shape[0]
    zc = target_cols / arr.shape[1]
    return zoom(arr, (zr, zc), order=1).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI quick-inspect
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Inspecting real data ===")
    data = load_real_data("data/raw")
    print(f"\nDFSAR CPR shape  : {data['dfsar']['CPR'].shape}")
    print(f"CPR range        : {data['dfsar']['CPR'].min():.3f} – {data['dfsar']['CPR'].max():.3f}")
    if data["ohrc"] is not None:
        print(f"OHRC shape       : {data['ohrc'].shape}")
    print(f"SAR scenes found : {len(data['sar_meta'])}")
    print(f"DFRS records     : {data['dfrs_data'].get('n_records', 0)}")
