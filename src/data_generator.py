"""
Synthetic data generator – calibrated to real Faustini crater parameters.

DEM calibration (LOLA / SLDEM2015 statistics for Faustini floor):
  - Study area: 10×10 km sub-region of Faustini interior (~87.2°S, 84°E)
  - Floor elevation:     ~-3200 m (relative to mean lunar radius)
  - RMS roughness:        1.5–3.0 m at 10 m baseline (LOLA RDR)
  - Hurst exponent H:     0.55 (LOLA-derived, rough terrain)
  - Floor slope:          < 5° typical
  - DSC:                  ~400 m diameter, ~60 m deep, offset from centre
  - DSC rim height:       ~15–25 m above floor
  - Wall asymmetry:       oblique impact origin → one wall 20–30% steeper

PSR is computed by proper horizon-angle ray-tracing over 24 solar azimuths
at the south-polar sun elevation (~2.8° max at -87.2°S).

References:
  Zuber et al. 2012 (LOLA), Araki et al. 2009 (SELENE)
  Kreslavsky & Head 2016 (lunar roughness power spectra)
"""

import pickle
import pathlib
import numpy as np
from scipy.ndimage import gaussian_filter, rotate as nd_rotate


def _load_real_ohrc(grid_size):
    """
    Try to load a real OHRC browse PNG and resize to grid_size × grid_size.
    Returns float64 array in [0,1] or None if no PNG is found.
    Requires Pillow (PIL) — optional dependency.
    """
    import glob as _glob
    pattern = str(pathlib.Path("data/raw/OHRC") / "**" / "browse" / "calibrated" / "*.png")
    matches = _glob.glob(pattern, recursive=True)
    if not matches:
        return None
    try:
        from PIL import Image
        img = Image.open(matches[0]).convert("L")
        img = img.resize((grid_size, grid_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float64) / 255.0
        print(f"[DataGen] Real OHRC loaded: {pathlib.Path(matches[0]).name} "
              f"(resized to {grid_size}x{grid_size})")
        return arr
    except Exception as exc:
        print(f"[DataGen] OHRC load skipped ({exc}); using synthetic")
        return None

# ---------------------------------------------------------------------------
# Grid constants – 10 km × 10 km, 10 m/pixel
# ---------------------------------------------------------------------------
GRID_SIZE   = 1000
PIXEL_SCALE = 10.0          # metres per pixel
CENTER_LAT  = -87.2
CENTER_LON  =  84.1
MOON_RADIUS = 1_737_400.0

# Real Faustini DSC parameters (approximate, from Chandrayaan-2 OHRC study)
# DSC at r=300px from bowl centre: inside the PSR (PSR boundary ~330px with
# FAU_DEPTH=800m).  Landing sites sit at r>440px (lit exterior).
# Rover traverse: ~1400m across the crater wall – physically realistic.
DSC_CENTER  = (250, 500)    # r=250 px from bowl centre (500,500) -> inside PSR (PSR boundary ~281px)
DSC_RADIUS  = 20            # pixels = 200 m radius -> 400 m diameter
DSC_DEPTH   = 62.0          # metres (depth below bowl floor)
DSC_RIM_H   = 18.0          # metres (rim height above bowl floor)


# ---------------------------------------------------------------------------
# Realistic DEM
# ---------------------------------------------------------------------------

def generate_dem(grid_size=GRID_SIZE, seed=42):
    """
    Realistic Faustini-region DEM for 10×10 km study area.

    The study area spans from OUTSIDE the Faustini crater (lit terrain)
    across the crater rim into the PSR interior, centred on the DSC.
    This is the only geometry that allows the illumination model to
    correctly place PSR (exterior terrain blocks sun at 2.8° elevation).

    Calibrated parameters (LOLA SELENE / Zuber 2012):
      Faustini outer radius :  ~4.5 km (half-diameter in our 10 km grid)
      Faustini depth        :  ~2800 m (rim to floor elevation difference)
      Floor roughness (RMS) :  ~2 m at 10 m baseline
      Rim asymmetry         :  10–20% steeper on one wall (oblique impact)
      DSC diameter          :  ~400 m (20 px radius at 10 m/px)
      DSC depth             :  ~60 m below floor
      DSC rim height        :  ~18 m above floor

    Structure:
      1. Exterior high terrain (lit)         – outer 15% of grid
      2. Faustini crater rim (steep wall)    – 15–35% radius
      3. Faustini crater floor (PSR)         – inner 35–100% radius
      4. Asymmetric DSC sub-crater           – centred near DSC_CENTER
      5. Lobate DSC rim + ejecta             – NE of DSC
      6. Secondary craters on floor          – 4 small craters
      7. Fractal roughness (H=0.55, σ=2 m)  – all areas
    """
    rng = np.random.default_rng(seed)
    gs  = grid_size
    xx, yy = np.meshgrid(np.arange(gs), np.arange(gs))

    # ── Faustini crater bowl ──
    cx, cy   = gs // 2, gs // 2
    FAU_RAD  = int(gs * 0.44)     # ~4.4 km radius
    FAU_DEPTH= 800.0               # metres

    # Circular radius for exterior and secondary crater calculations
    r_fau    = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    # Slightly elliptical bowl (NE-SW elongation, 15% – oblique impact origin).
    # Keeps the crater shape from being a perfect circle of isolines.
    angle_ell = np.radians(55)  # long axis ~NE-SW
    dx_ell = (xx - cx) * np.cos(angle_ell) + (yy - cy) * np.sin(angle_ell)
    dy_ell = -(xx - cx) * np.sin(angle_ell) + (yy - cy) * np.cos(angle_ell)
    r_eff_fau = np.sqrt((dx_ell / 1.08)**2 + (dy_ell / 0.94)**2)

    # Radial profile with subtle undulations (creates terrace-like morphology)
    r_norm_fau = np.clip(r_eff_fau / FAU_RAD, 0, 1)
    profile = r_norm_fau**1.8 + 0.06 * np.sin(4 * np.pi * r_norm_fau) * r_norm_fau * (1 - r_norm_fau)

    mask_fau    = r_eff_fau < FAU_RAD
    floor_z     = -(FAU_DEPTH * np.clip(1 - profile, 0, 1))
    dem         = np.where(mask_fau, floor_z, 0.0)

    # ── Azimuthal wall variation (breaks circular symmetry) ──
    # Simulates degraded rim sectors, wall collapses and irregular ejecta erosion.
    angles_fau = np.arctan2(yy - cy, xx - cx)
    az_var = (0.18 * np.sin(3 * angles_fau)
            + 0.11 * np.cos(7 * angles_fau)
            + 0.07 * np.sin(5 * angles_fau + 0.9))
    # Apply strongest at mid-wall (~r=0.65 FAU_RAD), fades to zero at centre/rim
    az_weight = np.clip(1.0 - np.abs(r_eff_fau / FAU_RAD - 0.65) / 0.35, 0, 1)**2
    dem += mask_fau * az_weight * az_var * FAU_DEPTH * 0.22

    exterior = ~mask_fau
    dem[exterior] += 5.0 * (r_fau[exterior] - FAU_RAD) / FAU_RAD   # gentle outward tilt

    # ── DSC sub-crater on floor (asymmetric, oblique impact) ──
    cr, cc   = DSC_CENTER
    r_dsc    = np.sqrt((xx - cc)**2 + (yy - cr)**2)

    # N wall steeper (higher row = S is shallower)
    asym_dsc = 1.0 + 0.35 * (yy - cr) / (DSC_RADIUS + 1)
    r_eff_dsc = r_dsc / np.clip(asym_dsc, 0.6, 1.7)

    mask_dsc  = r_eff_dsc < DSC_RADIUS
    bowl_norm = r_eff_dsc / DSC_RADIUS
    dem[mask_dsc] -= DSC_DEPTH * (1 - bowl_norm[mask_dsc]**1.5)

    # DSC rim (lobate)
    angles_dsc = np.arctan2(yy - cr, xx - cc)
    rim_lobe   = 1.0 + 0.28 * np.sin(3 * angles_dsc) + 0.16 * np.cos(5 * angles_dsc)
    rim_dsc    = (r_dsc >= DSC_RADIUS * 0.9) & (r_dsc < DSC_RADIUS * 1.45)
    dem[rim_dsc] += DSC_RIM_H * rim_lobe[rim_dsc] * np.exp(
        -((r_dsc[rim_dsc] - DSC_RADIUS) / (DSC_RADIUS * 0.28))**2
    )

    # Ejecta deposit NE of DSC
    ej_r = np.sqrt((xx - (cc + int(DSC_RADIUS * 1.9)))**2 +
                   (yy - (cr - int(DSC_RADIUS * 1.6)))**2)
    dem += 8.0 * np.exp(-(ej_r / (DSC_RADIUS * 1.3))**2)

    # ── Secondary craters on Faustini floor (12 craters, broadly distributed) ──
    rng2 = np.random.default_rng(seed + 1)
    floor_zone = mask_fau & ~mask_dsc & (r_dsc > DSC_RADIUS * 2)
    fr, fc = np.where(floor_zone)
    if len(fr) > 0:
        for _ in range(12):
            idx    = rng2.integers(0, len(fr))
            sc_r, sc_c = int(fr[idx]), int(fc[idx])
            sc_rad = rng2.integers(3, 18)
            sc_dep = rng2.uniform(5, 45)
            r_sec  = np.sqrt((xx - sc_c)**2 + (yy - sc_r)**2)
            ins    = r_sec < sc_rad
            dem[ins] -= sc_dep * (1 - (r_sec[ins] / sc_rad)**1.5)
            rim_s  = (r_sec >= sc_rad) & (r_sec < sc_rad * 1.5)
            dem[rim_s] += sc_dep * 0.28 * np.exp(
                -((r_sec[rim_s] - sc_rad) / (sc_rad * 0.3))**2
            )

    # ── Fractal roughness (H=0.55, σ=2 m — LOLA floor statistics) ──
    rough = _fractal_surface(gs, H=0.55, rng=rng)
    rough = rough / rough.std() * 2.0
    dem  += rough

    dem = gaussian_filter(dem, sigma=0.4)
    return dem.astype(np.float32)


def _fractal_surface(n, H=0.55, rng=None):
    """Fractional Brownian surface via spectral synthesis (Hurst exponent H)."""
    if rng is None:
        rng = np.random.default_rng()
    f  = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    freq   = np.sqrt(fx**2 + fy**2)
    freq[0, 0] = 1.0
    power  = freq ** (-(H + 1))
    power[0, 0] = 0.0
    phase  = rng.uniform(0, 2 * np.pi, (n, n))
    spec   = np.sqrt(power) * np.exp(1j * phase)
    surf   = np.real(np.fft.ifft2(spec))
    surf  -= surf.mean()
    return surf


# ---------------------------------------------------------------------------
# Illumination model – proper horizon-angle ray tracing
# ---------------------------------------------------------------------------

def generate_illumination(dem, grid_size=GRID_SIZE, pixel_scale=PIXEL_SCALE,
                           n_sun_positions=24, lat_deg=CENTER_LAT):
    """
    Multi-azimuth illumination fraction using horizon-angle ray tracing.

    For each solar azimuth, checks whether terrain horizon angle exceeds
    the sun elevation angle.  Uses rotate-and-scan (vectorised NumPy):
    no per-pixel Python loop.

    Sun elevation at -87.2°S ≈ 2.8° maximum (grazing).

    Returns
    -------
    illum_fraction : ndarray [0,1]  — fraction of azimuths where pixel is lit
    """
    gs         = dem.shape[0]
    sun_el_deg = max(90.0 - abs(lat_deg), 1.0)
    tan_sun    = float(np.tan(np.radians(sun_el_deg)))

    illum = np.zeros((gs, gs), dtype=np.float32)
    for az_deg in np.linspace(0, 360, n_sun_positions, endpoint=False):
        illum += _horizon_pass(dem, az_deg, tan_sun, pixel_scale)

    return (illum / n_sun_positions).astype(np.float32)


def _horizon_pass(dem, az_deg, tan_sun, pixel_scale):
    """
    Single-azimuth illumination via 1-D horizon scan.

    Rotates the DEM so the sun direction aligns with the column axis,
    then sweeps each column with a running-max to find the terrain horizon.
    """
    dem_rot = nd_rotate(dem, -az_deg, reshape=False, order=1, mode="nearest")
    gs      = dem_rot.shape[0]
    lit_rot = np.ones((gs, gs), dtype=np.float32)

    for c in range(gs):
        col = dem_rot[:, c].astype(np.float64)
        # Running max of ALL previous rows (terrain visible looking back toward sun)
        running_max = np.maximum.accumulate(
            np.concatenate([[-1e9], col[:-1]])
        )
        dist_m = np.arange(gs, dtype=np.float64) * pixel_scale
        with np.errstate(divide="ignore", invalid="ignore"):
            horizon_tan = np.where(dist_m > 0,
                                   (running_max - col) / dist_m,
                                   -1e9)
        lit_rot[:, c] = (horizon_tan < tan_sun).astype(np.float32)

    # Rotate back
    lit = nd_rotate(lit_rot, az_deg, reshape=False, order=0, mode="nearest")
    return np.clip(lit, 0, 1).astype(np.float32)


def generate_psr_mask(illum_fraction, threshold=0.01):
    """PSR = pixels with < 1% illumination (terrain-driven, non-circular)."""
    return illum_fraction < threshold


def generate_doubly_shadowed_mask(psr_mask, grid_size=GRID_SIZE):
    """DSC floor = pixels inside DSC that are also PSR."""
    gs = grid_size
    xx, yy = np.meshgrid(np.arange(gs), np.arange(gs))
    r_dsc = np.sqrt((xx - DSC_CENTER[1])**2 + (yy - DSC_CENTER[0])**2)
    return (r_dsc < DSC_RADIUS) & psr_mask


# ---------------------------------------------------------------------------
# Seasonal illumination (lunar month variation)
# ---------------------------------------------------------------------------

def generate_illumination_seasonal(dem, n_seasons=4, n_az_per_season=16,
                                    lat_deg=CENTER_LAT):
    """
    Compute time-averaged and seasonally-variable illumination fractions.

    The lunar south-polar sun elevation oscillates between ~0° and ~2.8°
    over a synodic month (~29.5 days).  Some PSR boundaries shift with
    the sun's elevation: marginal pixels become lit or shadowed seasonally.

    Parameters
    ----------
    n_seasons        : int   number of seasonal epochs (evenly spaced 0→2.8°)
    n_az_per_season  : int   solar azimuths per epoch

    Returns
    -------
    illum_mean : float32 ndarray  time-averaged illumination fraction [0,1]
    illum_std  : float32 ndarray  seasonal variability [0,1]
                  High std = marginally shadowed (seasonally variable PSR)
    """
    # Sun elevation cycles from 0.3° (near new Moon) to 2.8° (full Sun)
    sun_elevations = np.linspace(0.3, 2.8, n_seasons)
    stack = []
    for el_deg in sun_elevations:
        # Effective latitude that gives this sun elevation angle at south pole:
        # sun_el = 90 - |lat| → |lat| = 90 - sun_el
        lat_eff = -(90.0 - el_deg)
        illum_s = generate_illumination(dem, grid_size=dem.shape[0],
                                         n_sun_positions=n_az_per_season,
                                         lat_deg=lat_eff)
        stack.append(illum_s)

    stack = np.stack(stack, axis=0)   # (n_seasons, gs, gs)
    illum_mean = stack.mean(axis=0).astype(np.float32)
    illum_std  = stack.std(axis=0).astype(np.float32)
    return illum_mean, illum_std


# ---------------------------------------------------------------------------
# Ice depth scenario generator
# ---------------------------------------------------------------------------

def generate_ice_scenarios(dem, psr_mask, dsc_mask, pixel_scale=PIXEL_SCALE):
    """
    Generate three physically distinct ice deposit scenarios for comparison.

    Scenarios differ in burial depth and ice fraction, which changes the
    observed CPR via two-way signal attenuation through overlying regolith:
        CPR_obs = CPR_surface * exp(-2 * alpha_regolith * depth_m)
    where alpha_regolith ~ 0.01 Np/m at L-band (dry regolith loss tangent).

    Scenarios
    ---------
    shallow_frost    : 0.3 m depth, f_ice=0.12, T-dependent seasonal deposition
    mid_depth_ice    : 2.0 m depth, f_ice=0.35, stable ancient ice
    deep_ancient_ice : 4.5 m depth, f_ice=0.55, deeply buried primordial ice

    Returns dict mapping scenario name → dict(dfsar, depth_m, f_ice_target,
                                               cpr_expected, detectable)
    """
    ALPHA_REGOLITH_L = 0.01   # Np/m, L-band dry regolith absorption

    scenario_defs = [
        ("shallow_frost",    0.3, 0.12, 1.17, 8),    # (name, depth, f_ice, CPR_expected, seed)
        ("mid_depth_ice",    2.0, 0.35, 1.13, 9),
        ("deep_ancient_ice", 4.5, 0.55, 1.08, 10),
    ]

    scenarios = {}
    for name, depth_m, f_ice_target, cpr_expected, seed in scenario_defs:
        attenuation = float(np.exp(-2 * ALPHA_REGOLITH_L * depth_m))
        dfsar = generate_dfsar(dem, psr_mask, dsc_mask, seed=seed)

        # Post-process: scale ice-zone CPR to simulate depth attenuation.
        # Attenuated CPR = original CPR * (depth_attenuation / baseline_factor).
        # The baseline dfsar ice CPR ~ 1.18; target = cpr_expected * attenuation.
        CPR      = dfsar["CPR"].copy()
        ice_zone = dfsar["ice_zone"]
        if ice_zone.any():
            baseline_mean = float(CPR[ice_zone].mean().clip(1e-3, None))
            cpr_obs       = cpr_expected * attenuation
            scale         = cpr_obs / baseline_mean
            CPR[ice_zone] = np.clip(CPR[ice_zone] * scale, 0.0, 3.0)

            # DOP scales inversely with ice purity / depth
            DOP = dfsar["DOP"].copy()
            dop_target = float(np.clip(0.11 - f_ice_target * 0.05 + depth_m * 0.005, 0.04, 0.13))
            DOP[ice_zone] = np.clip(DOP[ice_zone] * (dop_target / max(float(DOP[ice_zone].mean()), 1e-6)), 0.0, 1.0)
            dfsar = {**dfsar, "CPR": CPR.astype(np.float32), "DOP": DOP.astype(np.float32)}
            cpr_obs_actual = float(CPR[ice_zone].mean())
        else:
            cpr_obs_actual = cpr_expected * attenuation

        detectable = cpr_obs_actual > 0.8
        scenarios[name] = dict(
            dfsar         = dfsar,
            depth_m       = depth_m,
            f_ice_target  = f_ice_target,
            cpr_expected  = cpr_expected,
            attenuation   = attenuation,
            cpr_obs       = cpr_obs_actual,
            detectable    = detectable,
        )
        print(f"  [Scenario] {name}: depth={depth_m}m  f_ice={f_ice_target:.2f}  "
              f"CPR_obs={cpr_obs_actual:.3f}  "
              f"detectable={'YES' if detectable else 'NO'}")

    return scenarios


# ---------------------------------------------------------------------------
# DFSAR compact-polarimetry simulation (calibrated to Chandrayaan-2 DFSAR)
# ---------------------------------------------------------------------------

def generate_dfsar(dem, psr_mask, dsc_mask, grid_size=GRID_SIZE, seed=7,
                    frequency_ghz=1.25, nes0=9.75e-3):
    """
    Simulate DFSAR compact-polarimetry (LH + LV channels).

    Physics
    -------
    Chandrayaan-2 DFSAR compact-pol mode: Left circular transmit,
    receive in H and V linear (LH, LV).

    From LH and LV complex amplitudes:
        SC = (LH - j*LV) / sqrt(2)   [same-sense circular = left]
        OC = (LH + j*LV) / sqrt(2)   [opposite-sense = right]
        CPR = |SC|² / |OC|²

    Stokes parameters (for DOP):
        I = |LH|² + |LV|²
        Q = |LH|² - |LV|²
        U = 2 Re(LH · LV*)
        V = 2 Im(LH · LV*)
        DOP = sqrt(Q² + U² + V²) / I

    Scattering regimes (L-band, 26° incidence, calibrated to DFSAR XML):
      Dry regolith (PSR non-ice)  : CPR ~ 0.2–0.5,  DOP ~ 0.5–0.8
      Rough/rocky surface (rim)   : CPR ~ 0.5–0.9,  DOP ~ 0.4–0.7
      Blocky ejecta               : CPR ~ 0.7–1.1,  DOP ~ 0.3–0.6
      Ice-bearing (DSC floor)     : CPR ~ 0.9–1.8,  DOP ~ 0.05–0.12
        (L-band ice threshold CPR > 0.8; S-band > 1.0)

    Parameters
    ----------
    nes0 : float   Noise Equivalent Sigma-0 from real DFSAR XML (9.75e-3)
    """
    rng = np.random.default_rng(seed)
    gs  = grid_size

    slope = _compute_slope(dem, PIXEL_SCALE)
    FAU_RAD_local = int(gs * 0.44)

    # ── Terrain classification ──
    # Steep walls (radial ring from DEM) PLUS scattered boulder fields on the
    # crater floor (non-radial clusters) – breaks the "CPR ring" artefact.
    rocky  = (slope > 18.0) | _boulder_fields(gs, DSC_CENTER, FAU_RAD_local, rng)
    ejecta = _ejecta_mask(gs, DSC_CENTER, DSC_RADIUS, rng)

    # Ice zone: multi-octave fractal (Perlin-like) field → one connected,
    # organically-shaped deposit.  Avoids the "flower" artefact caused by
    # ellipse boundaries + circular internal voids.
    rng_ice = np.random.default_rng(seed + 77)
    xx, yy  = np.meshgrid(np.arange(gs), np.arange(gs))
    cr_f, cc_f = float(DSC_CENTER[0]), float(DSC_CENTER[1])

    # Build a 4-octave fractal field. Coarse octave (sigma=20 ≈ DSC_RADIUS)
    # dominates → one large contiguous region.  Fine octaves add irregular
    # texture at the margins so the boundary looks fractured, not circular.
    noise2d = np.zeros((gs, gs), dtype=np.float64)
    for sigma_k, amp_k in [(20.0, 1.0), (7.0, 0.45), (2.5, 0.18), (1.0, 0.07)]:
        layer  = gaussian_filter(rng_ice.normal(0, 1, (gs, gs)), sigma=sigma_k)
        layer /= (layer.std() + 1e-9)
        noise2d += amp_k * layer

    # Soft Gaussian bias centred on DSC floor: guarantees the deposit falls
    # inside the DSC regardless of which random phase the noise lands on.
    r_bias   = np.sqrt((xx - cc_f)**2 + (yy - cr_f)**2)
    noise2d += 1.5 * np.exp(-0.5 * (r_bias / (DSC_RADIUS * 0.60))**2)
    noise2d /= (noise2d.std() + 1e-9)

    # Threshold ≈ 30 % fill → one large connected deposit + small satellite patches
    ice_zone = dsc_mask & (noise2d > 0.52)

    # ── Simulate complex LH and LV amplitudes ──
    # Per terrain class: assign mean amplitude ratio LH/LV
    # For ice: LH ≈ LV in power but phase-correlated (volume scatter)
    # For surface scatter: LH > LV (linearly polarised return)

    A_H = rng.rayleigh(0.25, (gs, gs)).astype(np.float64)
    A_V = rng.rayleigh(0.20, (gs, gs)).astype(np.float64)

    # Phase (physically motivated per scattering regime):
    # Single-bounce (smooth regolith): phi_V = phi_H - pi/2 → V > 0 → OC > SC → CPR ~ 0.3
    # Double-bounce (boulders, walls): phi_V ≈ phi_H ± pi/2 + noise → CPR ~ 0.5-1.0
    # Volume scatter (ice):            phi_V = uniform random → V ≈ 0 → CPR ~ 1.0
    phi_H = rng.uniform(0, 2*np.pi, (gs, gs))
    phi_V = phi_H - np.pi/2 + rng.normal(0, 0.5, (gs, gs))  # smooth regolith -> CPR~0.3

    # Rocky terrain (steep slopes): partial double-bounce → CPR ~ 0.5-0.9
    phi_V[rocky] = phi_H[rocky] + rng.normal(0, np.pi/3, rocky.sum())
    A_H[rocky] += rng.rayleigh(0.20, rocky.sum())
    A_V[rocky] += rng.rayleigh(0.18, rocky.sum())

    # Ejecta: blocky double-bounce → CPR ~ 0.7-1.2 (true false-positive source)
    phi_V[ejecta] = phi_H[ejecta] + rng.normal(np.pi/4, 0.6, ejecta.sum())
    A_H[ejecta] += rng.rayleigh(0.25, ejecta.sum())
    A_V[ejecta] += rng.rayleigh(0.10, ejecta.sum())

    # Ice: high-amplitude volume scatter – equal amplitudes, random phase
    n_ice = int(ice_zone.sum())
    if n_ice > 0:
        A_ice = rng.rayleigh(0.50, n_ice)
        A_H[ice_zone] = A_ice
        A_V[ice_zone] = A_ice          # equal amps → Q→0 per pixel
        phi_V[ice_zone] = rng.uniform(0, 2*np.pi, n_ice)  # fully random phase

    # Add NES0 noise floor
    A_H += np.sqrt(nes0)
    A_V += np.sqrt(nes0)

    # Complex fields
    LH = A_H * np.exp(1j * phi_H)
    LV = A_V * np.exp(1j * phi_V)

    # ── 7×7 multi-look spatial averaging (49 looks) to reduce speckle ──
    # With 49 looks: DOP noise floor for ice ≈ 0.07 (below 0.13 threshold)
    from scipy.ndimage import uniform_filter
    I_LH = uniform_filter(np.abs(LH)**2, size=7)
    I_LV = uniform_filter(np.abs(LV)**2, size=7)
    cross = uniform_filter(np.real(LH * np.conj(LV)), size=7)
    cross_im = uniform_filter(np.imag(LH * np.conj(LV)), size=7)

    # ── Stokes parameters ──
    I = I_LH + I_LV
    Q = I_LH - I_LV
    U = 2 * cross
    V = 2 * cross_im

    DOP = np.where(I > 1e-9, np.sqrt(Q**2 + U**2 + V**2) / I, 0.0)

    # ── Circular polarisation ratio ──
    # SC = (LH - j*LV)/sqrt(2) → |SC|² = (I - V)/2
    # OC = (LH + j*LV)/sqrt(2) → |OC|² = (I + V)/2
    SC_power = np.clip((I - V) / 2, nes0, None)
    OC_power = np.clip((I + V) / 2, nes0, None)
    CPR      = SC_power / (OC_power + 1e-12)

    # ── Override ice zone with empirically calibrated CPR / DOP ──
    # Physics: volume scatter from ice enhances same-sense circular return.
    # CPR ~ 1.15–1.25 (Nozette 1996, Harmon & Slade 1992, DFSAR char. studies)
    # DOP ~ 0.06–0.11 (chaotic depolarisation, Raney 2012)
    # Constraint: DOP = (CPR-1)/(CPR+1) when Q=U=0 → CPR < 1.30 gives DOP < 0.13
    if n_ice > 0:
        cpr_ice = rng.normal(1.18, 0.09, n_ice).clip(0.90, 1.28)
        I_ice   = I[ice_zone]
        # Stokes V gives CPR: V = I*(1-CPR)/(1+CPR)  →  V < 0 when CPR > 1
        V_ice   = I_ice * (1.0 - cpr_ice) / (1.0 + cpr_ice)
        V[ice_zone]        = V_ice
        Q[ice_zone]        = 0.0       # equal amplitudes  → Q = 0
        U[ice_zone]        = rng.normal(0, 0.008 * I_ice, n_ice)  # tiny residual U
        SC_power[ice_zone] = np.clip((I_ice - V_ice) / 2, nes0, None)
        OC_power[ice_zone] = np.clip((I_ice + V_ice) / 2, nes0, None)
        CPR[ice_zone]      = SC_power[ice_zone] / (OC_power[ice_zone] + 1e-12)
        # DOP for ice: sqrt(Q^2+U^2+V^2)/I ≈ |V|/I = (CPR-1)/(CPR+1)
        DOP[ice_zone]      = np.sqrt(Q[ice_zone]**2 + U[ice_zone]**2 + V_ice**2) / (I_ice + 1e-12)

    # ── Sigma-0 (dB) ──
    sigma0_lh = 10 * np.log10(I_LH + 1e-9)
    sigma0_lv = 10 * np.log10(I_LV + 1e-9)

    return dict(
        SC=SC_power.astype(np.float32),
        OC=OC_power.astype(np.float32),
        CPR=CPR.astype(np.float32),
        DOP=DOP.astype(np.float32),
        stokes_I=I.astype(np.float32),
        stokes_Q=Q.astype(np.float32),
        stokes_U=U.astype(np.float32),
        stokes_V=V.astype(np.float32),
        sigma0_hh=sigma0_lh.astype(np.float32),
        sigma0_hv=sigma0_lv.astype(np.float32),
        ice_zone=ice_zone,
        terrain_rocky=rocky,
        terrain_ejecta=ejecta,
        nes0=nes0,
        frequency_band="L",
        cpr_ice_threshold=0.8,
    )


def _ejecta_mask(gs, dsc_center, dsc_radius, rng):
    """
    Realistic ejecta: multiple discrete clusters at random azimuths from DSC.
    Avoids the single-blob pattern that looks like a radially placed artifact.
    """
    xx, yy = np.meshgrid(np.arange(gs), np.arange(gs))
    cr, cc = dsc_center
    mask = np.zeros((gs, gs), dtype=bool)
    # 6 disconnected clusters at varying distances and azimuths
    azimuths  = [0.4, 1.3, 2.2, 3.5, 4.7, 5.5]   # fixed azimuths (0-2pi)
    distances = [1.8, 2.4, 1.5, 2.8, 2.1, 3.2]    # multiples of dsc_radius
    fill_fracs = [0.55, 0.45, 0.60, 0.40, 0.50, 0.65]
    radii     = [1.1, 0.7, 0.9, 0.6, 1.2, 0.8]    # cluster radius (× dsc_radius)
    for az, dist, fill, rad in zip(azimuths, distances, fill_fracs, radii):
        er = int(cr + dist * dsc_radius * np.sin(az))
        ec = int(cc + dist * dsc_radius * np.cos(az))
        if 0 <= er < gs and 0 <= ec < gs:
            r_ej = np.sqrt((xx - ec)**2 + (yy - er)**2)
            base  = r_ej < rad * dsc_radius
            noise = rng.uniform(0, 1, (gs, gs)) < fill
            mask |= (base & noise)
    return mask


def _boulder_fields(gs, dsc_center, fau_rad, rng, n_fields=8):
    """
    Scattered boulder field patches on the crater floor.
    These create spatially clustered high-CPR regions NOT following the
    radial slope pattern, breaking the 'CPR ring' visual artefact.
    """
    xx, yy = np.meshgrid(np.arange(gs), np.arange(gs))
    cx, cy = gs // 2, gs // 2
    r_fau  = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    floor  = r_fau < fau_rad * 0.88   # restrict to interior

    # DSC exclusion zone
    cr, cc = dsc_center
    r_dsc  = np.sqrt((xx - cc)**2 + (yy - cr)**2)

    mask = np.zeros((gs, gs), dtype=bool)
    # Random azimuths (not linspace) to avoid symmetric "flower-ring" CPR pattern
    angles = rng.uniform(0, 2 * np.pi, n_fields)
    distances = [0.30, 0.48, 0.55, 0.38, 0.62, 0.44, 0.50, 0.35]
    patch_radii = [12, 18, 14, 22, 16, 10, 20, 15]
    fill_fracs  = [0.55, 0.45, 0.60, 0.50, 0.40, 0.65, 0.50, 0.55]

    for i, (az, dist) in enumerate(zip(angles, distances)):
        pr = int(cy + dist * fau_rad * np.sin(az))
        pc = int(cx + dist * fau_rad * np.cos(az))
        if 0 <= pr < gs and 0 <= pc < gs:
            rad = patch_radii[i % len(patch_radii)]
            fill = fill_fracs[i % len(fill_fracs)]
            r_patch = np.sqrt((xx - pc)**2 + (yy - pr)**2)
            noise = rng.uniform(0, 1, (gs, gs)) < fill
            outside_dsc = r_dsc > 30
            mask |= ((r_patch < rad) & noise & floor & outside_dsc)
    return mask


# ---------------------------------------------------------------------------
# OHRC simulation (realistic texture from Faustini imagery statistics)
# ---------------------------------------------------------------------------

def generate_ohrc(dem, psr_mask, dsc_mask, grid_size=GRID_SIZE, seed=13):
    """
    Simulate OHRC panchromatic image with realistic texture.

    Real OHRC imagery (0.25 m resolution, simulated here at 10 m) shows:
      - Albedo heterogeneity from space weathering and rock abundance
      - Hapke-like slope-dependent shading
      - Cast shadows from DSC rim and boulders
      - Bright fresh ejecta and dark PSR interior
      - Boulder highlights with dark cast shadows

    Parameters calibrated to Faustini floor statistics (Li et al. 2018,
    Hazra et al. 2023 OHRC analysis).
    """
    rng = np.random.default_rng(seed)
    gs  = grid_size

    # Base albedo: highland anorthosite regolith (0.10–0.16)
    # Multi-scale albedo variation: large-scale maturity + small-scale rocks
    base_lf = gaussian_filter(rng.uniform(0.09, 0.15, (gs, gs)), sigma=30)
    base_mf = gaussian_filter(rng.uniform(-0.02, 0.02, (gs, gs)), sigma=8)
    albedo  = np.clip(base_lf + base_mf, 0.08, 0.18)

    # Fractal micro-texture (space weathering heterogeneity, H=0.6)
    texture = _fractal_surface(gs, H=0.60, rng=rng) * 0.018
    albedo += texture

    # Slope-dependent shading (simplified Hapke: L = cos(i) where i = incidence angle)
    # For grazing illumination at south pole (sun elev ~2.8°): shadows are long
    slope   = _compute_slope(dem, PIXEL_SCALE)
    dy, dx  = np.gradient(dem.astype(np.float64), PIXEL_SCALE)
    # Sun from the north (azimuth=0) at 2.8° elevation
    cos_inc = np.clip(0.049 - dx * np.cos(0) - dy * np.sin(0), 0, 1)
    albedo *= (0.3 + 0.7 * cos_inc)   # darken shadowed slopes

    # PSR: nearly black (thermal emission only, ~1–3% relative to lit terrain)
    albedo[psr_mask] = rng.uniform(0.004, 0.012, psr_mask.sum())

    # Fresh ejecta blanket NE of DSC: high albedo, bright halo
    xx, yy = np.meshgrid(np.arange(gs), np.arange(gs))
    r_dsc  = np.sqrt((xx - DSC_CENTER[1])**2 + (yy - DSC_CENTER[0])**2)
    cr, cc = DSC_CENTER
    ej_r   = np.sqrt((xx - (cc + int(DSC_RADIUS*1.9)))**2 +
                     (yy - (cr - int(DSC_RADIUS*1.6)))**2)
    albedo += 0.05 * np.exp(-(ej_r / (DSC_RADIUS * 1.4))**2)

    # DSC rim: bright exposed bedrock faces
    rim_bright = (r_dsc >= DSC_RADIUS * 0.85) & (r_dsc < DSC_RADIUS * 1.3)
    albedo[rim_bright] = np.clip(albedo[rim_bright] + 0.04, 0, 0.25)

    # Boulders: bright sunlit face + dark cast shadow (NE–SW elongated pair)
    rng2 = np.random.default_rng(seed + 7)
    rim_rows, rim_cols = np.where(rim_bright)
    n_boulders = 150
    if len(rim_rows) >= n_boulders:
        idx = rng2.choice(len(rim_rows), n_boulders, replace=False)
        for br, bc in zip(rim_rows[idx], rim_cols[idx]):
            size = rng2.integers(1, 3)
            bright_val = rng2.uniform(0.18, 0.28)
            r0 = max(0, br - size); r1 = min(gs, br + size + 1)
            c0 = max(0, bc - size); c1 = min(gs, bc + size + 1)
            albedo[r0:r1, c0:c1] = np.clip(bright_val, 0, 1)
            # Shadow trail in downslope direction (sun from N, shadow goes S)
            for k in range(1, rng2.integers(2, 5)):
                sr = min(gs-1, br + k)
                albedo[sr, bc] = max(albedo[sr, bc] - 0.03, 0.005)

    # Floor boulders (smaller, scattered)
    floor_zone = ~psr_mask & ~rim_bright & (r_dsc > DSC_RADIUS * 1.5)
    fr, fc = np.where(floor_zone)
    if len(fr) > 50:
        idx2 = rng2.choice(len(fr), 50, replace=False)
        for br, bc in zip(fr[idx2], fc[idx2]):
            albedo[br, bc] = np.clip(rng2.uniform(0.14, 0.22), 0, 1)

    albedo = gaussian_filter(albedo, sigma=0.4)
    return np.clip(albedo, 0, 0.35).astype(np.float32)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _compute_slope(dem, pixel_scale=PIXEL_SCALE):
    dy, dx = np.gradient(dem.astype(np.float64), pixel_scale)
    return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_all(grid_size=GRID_SIZE, cache=True):
    """Generate and return all synthetic datasets, with optional disk cache."""
    cache_dir  = pathlib.Path(".cache")
    cache_path = cache_dir / f"synthetic_{grid_size}.pkl"

    if cache and cache_path.exists():
        print(f"[DataGen] Loading from cache ({cache_path})...")
        with open(cache_path, "rb") as fh:
            data = pickle.load(fh)
        # After cache load, try to upgrade synthetic OHRC with real browse PNG
        real_ohrc = _load_real_ohrc(data["meta"]["grid_size"])
        if real_ohrc is not None:
            data["ohrc"] = real_ohrc
            data["meta"]["ohrc_source"] = "real_browse_png"
        return data

    print("[DataGen] Generating realistic Faustini-calibrated DEM...")
    dem = generate_dem(grid_size)

    print(f"[DataGen] DEM stats: mean={dem.mean():.1f} m, "
          f"std={dem.std():.1f} m, "
          f"range=[{dem.min():.0f}, {dem.max():.0f}] m")

    print("[DataGen] Computing illumination model (24 solar azimuths)...")
    illum = generate_illumination(dem, grid_size)

    print("[DataGen] Computing seasonal illumination variability...")
    illum_mean, illum_std = generate_illumination_seasonal(dem, n_seasons=4,
                                                            n_az_per_season=12)

    psr_mask = generate_psr_mask(illum)
    dsc_mask  = generate_doubly_shadowed_mask(psr_mask, grid_size)

    psr_pct = psr_mask.sum() / grid_size**2 * 100
    print(f"[DataGen] PSR coverage: {psr_pct:.1f}%  "
          f"(non-circular, terrain-driven)")

    print("[DataGen] Simulating DFSAR compact-pol (Stokes + CPR + DOP)...")
    dfsar = generate_dfsar(dem, psr_mask, dsc_mask, grid_size)

    print("[DataGen] Simulating OHRC optical image...")
    ohrc = generate_ohrc(dem, psr_mask, dsc_mask, grid_size)
    real_ohrc = _load_real_ohrc(grid_size)
    if real_ohrc is not None:
        ohrc = real_ohrc

    slope = _compute_slope(dem, PIXEL_SCALE)

    data = dict(
        dem=dem, slope=slope,
        illum=illum, illum_mean=illum_mean, illum_std=illum_std,
        psr_mask=psr_mask, dsc_mask=dsc_mask,
        dfsar=dfsar, ohrc=ohrc,
        meta=dict(
            grid_size=grid_size, pixel_scale=PIXEL_SCALE,
            center_lat=CENTER_LAT, center_lon=CENTER_LON,
            dsc_center=DSC_CENTER, dsc_radius=DSC_RADIUS,
            source="synthetic_calibrated",
            ohrc_source="real_browse_png" if real_ohrc is not None else "synthetic",
        )
    )

    if cache:
        cache_dir.mkdir(exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(data, fh)
        print(f"[DataGen] Cached to {cache_path}")

    return data
