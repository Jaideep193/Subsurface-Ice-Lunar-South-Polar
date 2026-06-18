"""
KML + GeoJSON Ice Detection Export
====================================
Exports ice-positive pixels as:
  - KML  with point placemarks (opens in Google Earth)
  - GeoJSON with square polygon features (opens in QGIS / any GIS tool)

Coordinates: EPSG:4326 (geographic lat/lon)

Outputs:
  results/export/ice_detections.kml
  results/export/ice_detections.geojson
"""
import sys, os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

# Faustini scene centre (same as data_generator.py)
CENTER_LAT = -87.0
CENTER_LON =   0.0
M_PER_DEG  = 111_320.0


def pixel_to_latlon(r, c, gs, ps_m):
    """Convert (row, col) pixel index to (lat, lon) degrees."""
    lat = CENTER_LAT + (gs / 2.0 - r) * ps_m / M_PER_DEG
    cos_lat = np.cos(np.radians(CENTER_LAT))
    lon = CENTER_LON + (c - gs / 2.0) * ps_m / (M_PER_DEG * cos_lat)
    return float(lat), float(lon)


def load_data():
    from src.data_generator import load_all
    from src.dfsar_analysis import run_analysis

    data  = load_all(cache=True)
    psr   = data["psr_mask"]; slope = data["slope"]
    dfsar = data["dfsar"]; meta = data["meta"]
    gs    = meta["grid_size"]; ps_m = meta["pixel_scale"]
    dsc_c = meta["dsc_center"]; dsc_r = meta["dsc_radius"]

    res = run_analysis(dfsar, psr, slope)
    return dict(ice_mask=res["ice_mask"], CPR=res["CPR"],
                DOP=res["DOP"], ice_conf=res["ice_conf"],
                gs=gs, ps=ps_m, dsc_c=dsc_c, dsc_r=dsc_r)


# ── KML writer ────────────────────────────────────────────────────────────────
KML_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"
     xmlns:gx="http://www.google.com/kml/ext/2.2">
<Document>
  <name>Chandrayaan-2 DFSAR Ice Detections</name>
  <description>
    Subsurface ice-positive pixels detected in the Faustini Doubly Shadowed Crater
    using L-band (1.25 GHz) CPR &gt; 0.8 AND DOP &lt; 0.13 dual-criterion.
    Chandrayaan-2 DFSAR Compact Polarimetry.
  </description>

  <Style id="ice_pixel">
    <IconStyle>
      <color>ff00ffcc</color>
      <scale>0.6</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-circle-lv.png</href></Icon>
    </IconStyle>
    <LabelStyle><scale>0</scale></LabelStyle>
    <BalloonStyle>
      <text><![CDATA[
        <b>Ice Detection Pixel</b><br/>
        CPR: $[CPR]<br/>
        DOP: $[DOP]<br/>
        P(ice): $[Confidence]<br/>
        Location: $[description]
      ]]></text>
    </BalloonStyle>
  </Style>

  <Style id="dsc_boundary">
    <LineStyle><color>ff00ffff</color><width>2</width></LineStyle>
    <PolyStyle><color>220000ff</color></PolyStyle>
  </Style>

  <Folder><name>Ice Detection Pixels</name>
"""

KML_FOOTER = """  </Folder>
</Document>
</kml>
"""

KML_PLACEMARK = """    <Placemark>
      <name>Ice-{idx}</name>
      <description>{lat:.5f}°N  {lon:.5f}°E</description>
      <styleUrl>#ice_pixel</styleUrl>
      <ExtendedData>
        <Data name="CPR"><value>{cpr:.4f}</value></Data>
        <Data name="DOP"><value>{dop:.4f}</value></Data>
        <Data name="Confidence"><value>{conf:.4f}</value></Data>
      </ExtendedData>
      <Point><coordinates>{lon:.7f},{lat:.7f},0</coordinates></Point>
    </Placemark>
"""


def write_kml(d, out_path):
    ice_rows, ice_cols = np.where(d["ice_mask"])
    gs = d["gs"]; ps = d["ps"]

    lines = [KML_HEADER]
    for idx, (r, c) in enumerate(zip(ice_rows, ice_cols)):
        lat, lon = pixel_to_latlon(r, c, gs, ps)
        lines.append(KML_PLACEMARK.format(
            idx=idx,
            lat=lat, lon=lon,
            cpr=float(d["CPR"][r, c]),
            dop=float(d["DOP"][r, c]),
            conf=float(d["ice_conf"][r, c]),
        ))
    lines.append(KML_FOOTER)

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[KML]     Saved: {out_path}  ({len(ice_rows)} placemarks)")


# ── GeoJSON writer ─────────────────────────────────────────────────────────────
def write_geojson(d, out_path):
    ice_rows, ice_cols = np.where(d["ice_mask"])
    gs = d["gs"]; ps = d["ps"]
    half_deg_lat = (ps / 2.0) / M_PER_DEG
    half_deg_lon = (ps / 2.0) / (M_PER_DEG * np.cos(np.radians(CENTER_LAT)))

    features = []
    for r, c in zip(ice_rows, ice_cols):
        lat, lon = pixel_to_latlon(r, c, gs, ps)
        # Square polygon for this 10m × 10m pixel
        coords = [[
            [lon - half_deg_lon, lat - half_deg_lat],
            [lon + half_deg_lon, lat - half_deg_lat],
            [lon + half_deg_lon, lat + half_deg_lat],
            [lon - half_deg_lon, lat + half_deg_lat],
            [lon - half_deg_lon, lat - half_deg_lat],   # close ring
        ]]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": coords},
            "properties": {
                "CPR":        round(float(d["CPR"][r, c]),   4),
                "DOP":        round(float(d["DOP"][r, c]),   4),
                "P_ice":      round(float(d["ice_conf"][r, c]), 4),
                "pixel_row":  int(r),
                "pixel_col":  int(c),
                "lat":        round(lat, 6),
                "lon":        round(lon, 6),
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "name": "Chandrayaan-2 DFSAR Ice Detections",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }
    out_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"[GeoJSON] Saved: {out_path}  ({len(features)} polygons, {size_kb:.0f} KB)")


# ── DSC boundary KML (separate file for overlay) ──────────────────────────────
def write_dsc_kml(d, out_path):
    gs = d["gs"]; ps = d["ps"]
    dsc_c = d["dsc_c"]; dsc_r = d["dsc_r"]
    theta = np.linspace(0, 2*np.pi, 120)
    rows  = dsc_c[0] + dsc_r * np.sin(theta)
    cols  = dsc_c[1] + dsc_r * np.cos(theta)
    coords = []
    for r, c in zip(rows, cols):
        lat, lon = pixel_to_latlon(r, c, gs, ps)
        coords.append(f"{lon:.7f},{lat:.7f},0")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Faustini DSC Boundary</name>
  <Style id="dsc"><LineStyle><color>ff00ffff</color><width>2.5</width></LineStyle>
    <PolyStyle><color>1100ffff</color></PolyStyle></Style>
  <Placemark>
    <name>Doubly Shadowed Crater (DSC)</name>
    <styleUrl>#dsc</styleUrl>
    <Polygon><outerBoundaryIs><LinearRing>
      <coordinates>{" ".join(coords)}</coordinates>
    </LinearRing></outerBoundaryIs></Polygon>
  </Placemark>
</Document>
</kml>"""
    out_path.write_text(kml, encoding="utf-8")
    print(f"[KML]     Saved: {out_path}  (DSC boundary polygon)")


def main():
    print("[Export] Loading data...")
    d = load_data()

    out_dir = Path("results/export")
    out_dir.mkdir(parents=True, exist_ok=True)

    write_kml(d,      out_dir / "ice_detections.kml")
    write_geojson(d,  out_dir / "ice_detections.geojson")
    write_dsc_kml(d,  out_dir / "dsc_boundary.kml")

    print(f"\n[Export] Files written to {out_dir}/")
    print("  ice_detections.kml    -> open in Google Earth")
    print("  ice_detections.geojson -> open in QGIS / ArcGIS")
    print("  dsc_boundary.kml       -> DSC polygon overlay")


if __name__ == "__main__":
    main()
