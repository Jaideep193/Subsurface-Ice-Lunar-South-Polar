"""
Figure Gallery
==============
Generates results/figures/index.html — a dark-themed, responsive
thumbnail gallery linking every PNG produced by the pipeline.

Figures are grouped by category. Clicking a thumbnail opens the full PNG.
Also embeds the summary.json metrics if it exists.

Output: results/figures/index.html
"""
import json, base64, os
from pathlib import Path


FIGURES = [
    # (filename,  title,  description)
    ("00_dashboard.png",         "Master Dashboard",
     "8-panel master summary: CPR, ice mask, DEM, traverse, volume, confidence"),
    ("01_overview.png",          "Scene Overview",
     "DEM, illumination fraction, PSR map, DSC identification, temperature"),
    ("02_dfsar_analysis.png",    "DFSAR Polarimetric Analysis",
     "CPR/DOP maps, ice detection, ROC curve, sensitivity analysis, statistics"),
    ("03_morphology.png",        "Crater Morphology",
     "Slope, roughness, boulder density, OHRC imagery, hazard map, rim profile"),
    ("04_landing_site.png",      "Landing Site Evaluation",
     "Composite MCDA score, per-criterion maps, top-5 candidate sites"),
    ("05_traverse.png",          "Rover Traverse Path",
     "A* cost map, planned path, energy budget, illumination profile"),
    ("06_ice_volume.png",        "Ice Volume Estimation",
     "Polder-van Santen inversion, penetration depth, volume map, uncertainty"),
    ("07_advanced_analysis.png", "Advanced Analysis",
     "m-chi decomposition (Raney 2012), ROC curve, temporal coherence"),
    ("08_ice_scenarios.png",     "Ice Depth Scenarios",
     "Three modelled ice depth scenarios (shallow/mid/deep) for sensitivity"),
    ("09_isru_resource_map.png", "ISRU Resource Map",
     "Water-equivalent mass density, mining priority zones, propellant budget"),
    ("10_lband_sband_comparison.png", "L-band vs S-band",
     "Detection comparison: 1.25 GHz vs simulated 2.40 GHz; agreement analysis"),
    ("11_ice_volume_cdf.png",    "Ice Volume CDF",
     "Monte Carlo CDF, PDF, bootstrap convergence, propellant barchart, quantiles"),
    ("12_ice_stability_zones.png", "Ice Stability Zones",
     "Thermal classification: stable ancient ice, seasonal frost, unstable"),
    ("13_landing_sites.png",     "Top Landing Sites",
     "MCDA 4-criterion scoring, top-5 ranked candidate sites, GeoJSON export"),
    ("14_temporal_coherence.png","Temporal Coherence",
     "Two-pass comparison: stable subsurface ice vs transient surface frost"),
    ("15_sublimation_lifetime.png","Sublimation Lifetime",
     "Hertz-Knudsen equation, zone lifetime map, sensitivity to α and thickness"),
    ("16_depth_uncertainty.png", "Depth-Uncertainty Cross-Plot",
     "f_ice vs penetration depth per pixel, colored by temperature"),
    ("architecture.png",         "Pipeline Architecture",
     "System diagram: data flow from DFSAR ingestion to ISRU assessment"),
]

GEOTIFF_PNGS = [
    ("cpr.png",            "CPR (GeoTIFF export)"),
    ("dop.png",            "DOP (GeoTIFF export)"),
    ("ice_mask.png",       "Ice Mask (GeoTIFF export)"),
    ("ice_confidence.png", "Ice Confidence (GeoTIFF export)"),
    ("dem.png",            "DEM (GeoTIFF export)"),
    ("temperature.png",    "Temperature (GeoTIFF export)"),
    ("landing_score.png",  "Landing Score (GeoTIFF export)"),
    ("hazard.png",         "Hazard Map (GeoTIFF export)"),
    ("mchi_volume.png",    "m-chi Volume Scatter (GeoTIFF)"),
    ("mchi_single.png",    "m-chi Single Bounce (GeoTIFF)"),
    ("mchi_double.png",    "m-chi Double Bounce (GeoTIFF)"),
]


def thumb_b64(path: Path, max_px=400) -> str:
    """Return base64-encoded JPEG thumbnail, or empty string if file missing."""
    if not path.exists():
        return ""
    try:
        from PIL import Image
        img = Image.open(path)
        img.thumbnail((max_px, max_px))
        from io import BytesIO
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        # Pillow not available — just use the file path directly
        return ""


def read_summary(root: Path) -> dict:
    p = root.parent / "summary.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def figure_card(fname, title, desc, fig_dir: Path, use_b64: bool) -> str:
    fpath = fig_dir / fname
    exists = fpath.exists()
    if not exists:
        return f"""
        <div class="card missing">
          <div class="thumb-placeholder">missing</div>
          <div class="card-body">
            <div class="card-title">{title}</div>
            <div class="card-desc">{desc}</div>
            <span class="badge missing-badge">NOT GENERATED</span>
          </div>
        </div>"""

    if use_b64:
        b64 = thumb_b64(fpath)
        if b64:
            img_tag = f'<img src="data:image/jpeg;base64,{b64}" alt="{title}" loading="lazy">'
        else:
            img_tag = f'<img src="{fname}" alt="{title}" loading="lazy">'
    else:
        img_tag = f'<img src="{fname}" alt="{title}" loading="lazy">'

    size_kb = fpath.stat().st_size // 1024

    return f"""
        <a class="card" href="{fname}" target="_blank">
          <div class="thumb">{img_tag}</div>
          <div class="card-body">
            <div class="card-title">{title}</div>
            <div class="card-desc">{desc}</div>
            <span class="badge">{size_kb} KB</span>
          </div>
        </a>"""


def geotiff_card(fname, title, gt_dir: Path, use_b64: bool) -> str:
    fpath = gt_dir / fname
    if not fpath.exists():
        return ""
    if use_b64:
        b64 = thumb_b64(fpath, max_px=250)
        img_tag = (f'<img src="data:image/jpeg;base64,{b64}" alt="{title}" loading="lazy">'
                   if b64 else f'<img src="../geotiff_png/{fname}" alt="{title}" loading="lazy">')
    else:
        img_tag = f'<img src="../geotiff_png/{fname}" alt="{title}" loading="lazy">'
    size_kb = fpath.stat().st_size // 1024
    return f"""
        <a class="card small-card" href="../geotiff_png/{fname}" target="_blank">
          <div class="thumb">{img_tag}</div>
          <div class="card-body">
            <div class="card-title" style="font-size:11px">{title}</div>
            <span class="badge">{size_kb} KB</span>
          </div>
        </a>"""


def metric_pill(label, value) -> str:
    if value is None:
        return ""
    return f'<div class="metric"><span class="metric-label">{label}</span><span class="metric-value">{value}</span></div>'


def summary_section(s: dict) -> str:
    if not s:
        return ""
    d = s.get("dfsar_detection", {})
    iv = s.get("ice_volume", {})
    mc = s.get("monte_carlo_uncertainty", {})
    ls = s.get("landing_sites", {})
    isru = s.get("isru_assessment", {})
    sv = s.get("statistical_validation", {})
    src = s.get("meta", {}).get("data_source", "?")
    gen = s.get("meta", {}).get("generated_utc", "?")

    pills = "".join([
        metric_pill("Ice pixels",        d.get("ice_pixels")),
        metric_pill("Ice area (km²)",    d.get("ice_area_km2")),
        metric_pill("Mean CPR (ice)",    d.get("mean_cpr_ice")),
        metric_pill("Mean DOP (ice)",    d.get("mean_dop_ice")),
        metric_pill("Ice volume P50 (m³)", mc.get("p50_m3")),
        metric_pill("Uncertainty",       f"{mc.get('rel_unc_pct', '?')}%"),
        metric_pill("Water mass P50 (t)", isru.get("water_p50_t")),
        metric_pill("H₂ propellant (t)",  isru.get("h2_propellant_p50_t")),
        metric_pill("CH₄ via Sabatier (t)", isru.get("ch4_sabatier_p50_t")),
        metric_pill("Cohen's d (CPR)",   sv.get("cohens_d_cpr")),
        metric_pill("Effect size",       sv.get("effect_size")),
        metric_pill("Best landing score",ls.get("best_score")),
        metric_pill("Best site lat",     ls.get("best_lat_deg")),
        metric_pill("Best site lon",     ls.get("best_lon_deg")),
    ])

    return f"""
    <section class="summary-section">
      <h2>Key Results <span style="font-size:12px;color:#556">— {src} data, generated {gen}</span></h2>
      <div class="metrics-grid">{pills}</div>
    </section>"""


def build_html(fig_dir: Path) -> str:
    use_b64 = True
    try:
        from PIL import Image
    except ImportError:
        use_b64 = False

    gt_dir = fig_dir.parent / "geotiff_png"
    s      = read_summary(fig_dir)

    figure_cards = "\n".join(
        figure_card(f, t, d, fig_dir, use_b64) for f, t, d in FIGURES)
    gt_cards = "\n".join(
        geotiff_card(f, t, gt_dir, use_b64) for f, t in GEOTIFF_PNGS)
    summary_html = summary_section(s)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lunar Ice Detection — Figure Gallery</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #07070f;
    color: #c8d4e8;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 13px;
    min-height: 100vh;
  }}
  header {{
    background: linear-gradient(135deg, #0a0a2a 0%, #1a1a3a 100%);
    border-bottom: 1px solid #223;
    padding: 24px 32px 20px;
  }}
  header h1 {{
    font-size: 22px;
    font-weight: 700;
    color: #a8d8ff;
    letter-spacing: 0.03em;
  }}
  header p {{ color: #8899aa; margin-top: 6px; font-size: 12px; }}
  .container {{ max-width: 1600px; margin: 0 auto; padding: 24px 20px 60px; }}

  /* Summary */
  .summary-section {{
    background: #0c0c1e;
    border: 1px solid #2a3a55;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 32px;
  }}
  .summary-section h2 {{
    font-size: 15px;
    color: #7abbff;
    margin-bottom: 14px;
    font-weight: 600;
  }}
  .metrics-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }}
  .metric {{
    background: #111827;
    border: 1px solid #1e3050;
    border-radius: 6px;
    padding: 8px 14px;
    display: flex;
    flex-direction: column;
    min-width: 120px;
  }}
  .metric-label {{ font-size: 10px; color: #6677aa; text-transform: uppercase; letter-spacing: 0.06em; }}
  .metric-value {{ font-size: 15px; font-weight: 700; color: #88ddff; margin-top: 3px; }}

  /* Gallery */
  h2.section-title {{
    font-size: 16px;
    font-weight: 600;
    color: #7abbff;
    margin: 28px 0 14px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2a3a;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
  }}
  .small-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
  }}
  .card {{
    background: #0d1122;
    border: 1px solid #1e2a3a;
    border-radius: 10px;
    overflow: hidden;
    text-decoration: none;
    color: inherit;
    transition: border-color 0.18s, transform 0.18s, box-shadow 0.18s;
    display: flex;
    flex-direction: column;
  }}
  .card:hover {{
    border-color: #4488cc;
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.6);
  }}
  .card.missing {{ opacity: 0.45; cursor: default; }}
  .thumb {{
    width: 100%;
    aspect-ratio: 4/3;
    overflow: hidden;
    background: #0a0a1a;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .thumb img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.3s;
  }}
  .card:hover .thumb img {{ transform: scale(1.04); }}
  .thumb-placeholder {{
    color: #334;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}
  .card-body {{ padding: 12px 14px 10px; flex: 1; }}
  .card-title {{ font-size: 13px; font-weight: 600; color: #aac8e8; margin-bottom: 4px; }}
  .card-desc  {{ font-size: 11px; color: #6677aa; line-height: 1.45; }}
  .badge {{
    display: inline-block;
    margin-top: 8px;
    background: #1a2a3a;
    color: #5577aa;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 4px;
  }}
  .missing-badge {{ background: #3a1a1a; color: #aa5555; }}
  .small-card .card-body {{ padding: 8px 10px 8px; }}

  footer {{
    text-align: center;
    padding: 24px;
    color: #334;
    font-size: 11px;
    border-top: 1px solid #111;
  }}
</style>
</head>
<body>
<header>
  <h1>Detection and Characterization of Subsurface Ice in Lunar South Polar Regions</h1>
  <p>Chandrayaan-2 DFSAR L-band compact polarimetry &nbsp;|&nbsp; Faustini Doubly Shadowed Crater &nbsp;|&nbsp; Figure Gallery</p>
</header>
<div class="container">
  {summary_html}

  <h2 class="section-title">Pipeline Figures</h2>
  <div class="grid">
{figure_cards}
  </div>

  <h2 class="section-title">GeoTIFF Exports (PNG previews)</h2>
  <div class="small-grid">
{gt_cards}
  </div>
</div>
<footer>
  Subsurface Ice Detection Pipeline &nbsp;·&nbsp; Chandrayaan-2 DFSAR &nbsp;·&nbsp;
  CPR > 0.8 &amp; DOP &lt; 0.13 dual criterion &nbsp;·&nbsp; Faustini PSR, Lunar South Pole
</footer>
</body>
</html>"""


if __name__ == "__main__":
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    html = build_html(fig_dir)
    out  = fig_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"[Gallery] Saved: {out}  ({out.stat().st_size // 1024} KB)")
    n_present = sum(1 for f, _, _ in FIGURES if (fig_dir / f).exists())
    print(f"[Gallery] {n_present}/{len(FIGURES)} main figures present")
