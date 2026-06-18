"""
Surface temperature model for lunar south polar region.
Based on Stefan-Boltzmann equilibrium calibrated to Diviner/LRO observations
(Paige et al. 2010, Science 330, 479-482).

Physics
-------
  T_surface^4 = T_sunlit^4 * f_illum + T_shadow^4 * (1 - f_illum)

where f_illum = illumination fraction from ray-casting (0=always shadowed, 1=always lit).

Diviner calibration values for -87° S:
  Fully lit terrain:   90 – 110 K  (solar incidence ~2.8°, low insolation)
  Deep PSR:            40 –  70 K  (geothermal + scattered thermal IR only)
  Ice stability limit: T < 110 K   (Paige et al. 2010; valid for geological timescales)

The spatial coincidence of CPR anomaly with T < 110 K cold trap provides a second
independent line of evidence for subsurface ice.
"""

import numpy as np

ICE_STABILITY_T_K = 110.0   # Water ice thermally stable below this threshold


def compute_surface_temperature(illum_fraction,
                                 T_sunlit_K=100.0,
                                 T_shadow_K=42.0):
    """
    Stefan-Boltzmann equilibrium temperature for south polar surface.

    Parameters
    ----------
    illum_fraction : ndarray  Fraction of time pixel is sunlit [0, 1]
    T_sunlit_K     : float    Equilibrium temp for fully lit terrain (K)
    T_shadow_K     : float    Residual background temperature (K) — geothermal + IR
    """
    f   = np.clip(illum_fraction, 0.0, 1.0).astype(np.float64)
    T4  = T_sunlit_K**4 * f + T_shadow_K**4 * (1.0 - f)
    return np.power(T4, 0.25).astype(np.float32)


def ice_stability_mask(T_surface, threshold_K=ICE_STABILITY_T_K):
    """Boolean mask: True where T < threshold (ice thermally stable)."""
    return T_surface < threshold_K


def isru_assessment(ice_volume_m3, uncertainty=None):
    """
    Convert ice volume to mission-relevant ISRU resource metrics.

    Parameters
    ----------
    ice_volume_m3 : float  Total ice volume (m3) — already accounts for f_ice
    uncertainty   : dict   Monte Carlo result with p5_m3, p95_m3 keys (optional)
    """
    water_kg    = ice_volume_m3 * 917.0           # ice density 917 kg/m3
    water_L     = water_kg                         # 1 kg water ≈ 1 L
    prop_kg     = water_kg * 0.83                  # 83% electrolysis → H2/O2 prop
    crew_days   = water_L / 1.83                   # NASA 1.83 L/person/day potable

    lines = [
        "=" * 60,
        "ISRU RESOURCE ASSESSMENT",
        "=" * 60,
        f"  Ice volume (P50)      : {ice_volume_m3:>10.0f} m3",
        f"  Water mass            : {water_kg:>10.0f} kg  ({water_kg/1e3:.1f} tonnes)",
        f"  Water volume          : {water_L:>10.0f} L",
        f"  H2/O2 propellant      : {prop_kg:>10.0f} kg  (83% electrolysis eff.)",
        f"  Crew water supply     : {crew_days:>10.0f} person-days  ({crew_days/365:.1f} person-yrs)",
    ]
    if uncertainty:
        w_p5  = uncertainty["p5_m3"]  * 917.0
        w_p95 = uncertainty["p95_m3"] * 917.0
        lines += [
            f"  90% CI water mass     : [{w_p5:.0f}, {w_p95:.0f}] kg",
        ]
    lines += [
        "  Reference: NASA HEOMD water budget 1.83 L/person/day;",
        "             SpaceX Starship bipropellant ~3600 t (LOX+LH2)",
        "=" * 60,
    ]
    return "\n".join(lines)


def thermal_summary(T_surface, psr_mask, ice_mask=None):
    lines = [
        "=" * 60,
        "THERMAL MODEL  (Diviner-calibrated, Paige et al. 2010)",
        "=" * 60,
        f"  T fully-lit south polar terrain : ~100 K",
        f"  T deep PSR (background)         : ~42 K",
        f"  Ice stability threshold         : <{ICE_STABILITY_T_K:.0f} K",
        f"  Mean T (PSR pixels)             : {T_surface[psr_mask].mean():.1f} K",
        f"  Mean T (lit pixels)             : {T_surface[~psr_mask].mean():.1f} K",
        f"  Cold trap area (<{ICE_STABILITY_T_K:.0f} K)       : "
        f"{(T_surface < ICE_STABILITY_T_K).sum() * 1e-4:.2f} km2",
    ]
    if ice_mask is not None and ice_mask.any():
        lines += [
            f"  Mean T at ice pixels            : {T_surface[ice_mask].mean():.1f} K",
            f"  Ice pixels below stability T    : "
            f"{(T_surface[ice_mask] < ICE_STABILITY_T_K).sum()} / {ice_mask.sum()}",
        ]
    lines.append("=" * 60)
    return "\n".join(lines)
