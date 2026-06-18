"""
Real data loader for Chandrayaan-2 DFSAR (DFRS) and OHRC data.

Expected file naming from ISRO PRADAN portal:
  DFSAR : CH2O_XXXXX_DFRS_DS95_YYYY_DDD_HH_MM.zip
  OHRC  : CH2O_XXXXX_OHRC_DS95_YYYY_DDD_HH_MM.zip

Place downloaded ZIPs in:  data/raw/dfsar/  and  data/raw/ohrc/

The ZIPs typically contain:
  *.img / *.dat   – raw binary raster (BSQ or BIL)
  *.lbl / *.xml   – PDS-style label with geometry and scale
  *.h5  / *.nc    – HDF5 or NetCDF (newer products)
  *_BROWSE.*      – quicklook image

Usage
-----
    from src.data_loader import DataLoader
    loader = DataLoader("data/raw")
    data   = loader.load()          # returns same dict as data_generator.load_all()
"""

import os
import zipfile
import glob
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import h5py
    HAS_H5 = True
except ImportError:
    HAS_H5 = False

try:
    import rasterio
    HAS_RIO = True
except ImportError:
    HAS_RIO = False


PIXEL_SCALE = 10.0   # metres – update if your product header says different


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Loads real Chandrayaan-2 DFSAR / OHRC data and returns a dictionary
    compatible with the rest of the pipeline.

    Parameters
    ----------
    raw_dir : str  path to data/raw/ which should contain dfsar/ and ohrc/ sub-dirs
                   OR flat ZIPs placed directly in raw_dir.
    """

    def __init__(self, raw_dir="data/raw"):
        self.raw_dir   = Path(raw_dir)
        self.dfsar_dir = self.raw_dir / "dfsar"
        self.ohrc_dir  = self.raw_dir / "ohrc"
        self.work_dir  = self.raw_dir / "extracted"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def load(self):
        """
        Extract ZIPs, read data, return pipeline-compatible dict.
        Falls back to synthetic data if no real files are found.
        """
        dfsar_zips = self._find_zips("DFRS")
        ohrc_zips  = self._find_zips("OHRC")

        if not dfsar_zips and not ohrc_zips:
            print("[DataLoader] No real data found – falling back to synthetic data.")
            from src.data_generator import load_all
            return load_all()

        print(f"[DataLoader] Found {len(dfsar_zips)} DFSAR ZIPs, "
              f"{len(ohrc_zips)} OHRC ZIPs.")

        dfsar_arrays = []
        for z in dfsar_zips:
            arr = self._load_dfsar_zip(z)
            if arr is not None:
                dfsar_arrays.append(arr)

        ohrc_arrays = []
        for z in ohrc_zips:
            arr = self._load_ohrc_zip(z)
            if arr is not None:
                ohrc_arrays.append(arr)

        # Mosaic multiple passes (simple average / median)
        SC, OC = self._mosaic_dfsar(dfsar_arrays)
        ohrc   = self._mosaic_ohrc(ohrc_arrays)

        # Derived products
        CPR  = np.where(OC > 0, SC / OC, 0).astype(np.float32)
        DOP  = self._compute_dop(SC, OC)

        # Placeholder DEM from DFSAR intensity (real workflow: use LOLA DEM)
        dem   = self._dem_from_intensity(SC)
        slope = self._compute_slope(dem)

        # PSR and DSC masks from illumination model on DEM
        from src.data_generator import (generate_illumination,
                                         generate_psr_mask,
                                         generate_doubly_shadowed_mask,
                                         _compute_slope)
        illum    = generate_illumination(dem, dem.shape[0])
        psr_mask = generate_psr_mask(illum)
        dsc_mask = generate_doubly_shadowed_mask(psr_mask, dem.shape[0])

        # Ice zone: CPR > 1 AND DOP < 0.13 AND inside PSR
        ice_zone = (CPR > 1.0) & (DOP < 0.13) & psr_mask

        sigma0_hh = 10 * np.log10(SC + 1e-9)
        sigma0_hv = 10 * np.log10(OC + 1e-9)

        return dict(
            dem      = dem,
            slope    = slope,
            illum    = illum,
            psr_mask = psr_mask,
            dsc_mask = dsc_mask,
            dfsar    = dict(SC=SC, OC=OC, CPR=CPR, DOP=DOP,
                            sigma0_hh=sigma0_hh, sigma0_hv=sigma0_hv,
                            ice_zone=ice_zone),
            ohrc     = ohrc,
            meta     = dict(
                grid_size   = SC.shape[0],
                pixel_scale = PIXEL_SCALE,
                source      = "real",
                dfsar_files = [str(z) for z in dfsar_zips],
                ohrc_files  = [str(z) for z in ohrc_zips],
            )
        )

    # ------------------------------------------------------------------
    # ZIP discovery
    # ------------------------------------------------------------------

    def _find_zips(self, sensor_tag):
        """Find all ZIPs matching the sensor tag anywhere under raw_dir."""
        pattern_upper = self.raw_dir.rglob(f"*{sensor_tag}*.zip")
        pattern_lower = self.raw_dir.rglob(f"*{sensor_tag.lower()}*.zip")
        found = sorted(set(list(pattern_upper) + list(pattern_lower)))
        return found

    # ------------------------------------------------------------------
    # DFSAR extraction
    # ------------------------------------------------------------------

    def _load_dfsar_zip(self, zip_path):
        """
        Extract one DFSAR ZIP and return (SC, OC) tuple of 2-D float32 arrays.
        Handles HDF5, GeoTIFF (via rasterio), and raw binary (via PDS label).
        """
        extract_to = self.work_dir / zip_path.stem
        extract_to.mkdir(exist_ok=True)

        print(f"  [DFSAR] Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
            names = zf.namelist()

        print(f"    Contents: {names[:10]}")

        # --- Try HDF5 first ---
        h5_files = list(extract_to.rglob("*.h5")) + list(extract_to.rglob("*.hdf5"))
        if h5_files and HAS_H5:
            return self._read_dfsar_h5(h5_files[0])

        # --- Try GeoTIFF ---
        tif_files = list(extract_to.rglob("*.tif")) + list(extract_to.rglob("*.tiff"))
        if tif_files and HAS_RIO:
            return self._read_dfsar_tif(tif_files)

        # --- Try PDS binary (.img) + label (.lbl / .xml) ---
        img_files = list(extract_to.rglob("*.img")) + list(extract_to.rglob("*.dat"))
        if img_files:
            return self._read_dfsar_pds(img_files, extract_to)

        print(f"    [WARN] Could not parse DFSAR from {zip_path.name}")
        return None

    def _read_dfsar_h5(self, h5_path):
        """Read DFSAR from HDF5 – tries common dataset name conventions."""
        import h5py
        with h5py.File(h5_path, "r") as f:
            print(f"    HDF5 keys: {list(f.keys())}")

            def _find_dataset(f, keywords):
                for key in f.keys():
                    for kw in keywords:
                        if kw.lower() in key.lower():
                            ds = f[key]
                            if isinstance(ds, h5py.Dataset):
                                return np.array(ds, dtype=np.float32)
                            elif isinstance(ds, h5py.Group):
                                sub = _find_dataset(ds, keywords)
                                if sub is not None:
                                    return sub
                return None

            SC = _find_dataset(f, ["SC", "SL", "SameCirc", "sigma_sc", "backscatter_sc"])
            OC = _find_dataset(f, ["OC", "OL", "OppCirc", "sigma_oc", "backscatter_oc"])

            # Fallback: first two 2-D datasets
            if SC is None or OC is None:
                datasets_2d = []
                def _collect(name, obj):
                    if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                        datasets_2d.append(np.array(obj, dtype=np.float32))
                f.visititems(_collect)
                if len(datasets_2d) >= 2:
                    SC, OC = datasets_2d[0], datasets_2d[1]
                elif len(datasets_2d) == 1:
                    SC = datasets_2d[0]
                    OC = np.ones_like(SC)

        if SC is None:
            return None
        if OC is None:
            OC = np.ones_like(SC)
        return SC, OC

    def _read_dfsar_tif(self, tif_files):
        """Read DFSAR from GeoTIFF files – expects separate SC and OC files."""
        import rasterio

        def _load(path):
            with rasterio.open(path) as src:
                return src.read(1).astype(np.float32)

        if len(tif_files) >= 2:
            SC = _load(tif_files[0])
            OC = _load(tif_files[1])
        else:
            with rasterio.open(tif_files[0]) as src:
                bands = [src.read(b).astype(np.float32) for b in src.indexes]
            SC = bands[0]
            OC = bands[1] if len(bands) > 1 else np.ones_like(SC)

        return SC, OC

    def _read_dfsar_pds(self, img_files, extract_dir):
        """Read DFSAR from PDS binary image + label."""
        # Find label
        lbl_files = list(extract_dir.rglob("*.lbl")) + list(extract_dir.rglob("*.xml"))
        meta = self._parse_pds_label(lbl_files[0]) if lbl_files else {}

        lines  = int(meta.get("LINES", 1024))
        samples = int(meta.get("LINE_SAMPLES", 1024))
        dtype_str = meta.get("SAMPLE_TYPE", "IEEE_REAL")
        dtype = np.float32 if "REAL" in dtype_str else np.int16

        bands = []
        for img in img_files[:2]:
            data = np.fromfile(img, dtype=dtype).reshape(lines, samples)
            bands.append(data.astype(np.float32))

        SC = bands[0] if len(bands) > 0 else None
        OC = bands[1] if len(bands) > 1 else (np.ones_like(SC) if SC is not None else None)
        return (SC, OC) if SC is not None else None

    def _parse_pds_label(self, lbl_path):
        """Parse PDS3 .lbl or PDS4 .xml label for image geometry."""
        meta = {}
        text = Path(lbl_path).read_text(errors="replace")
        for line in text.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                meta[k.strip()] = v.strip().strip('"').split()[0]
        return meta

    # ------------------------------------------------------------------
    # OHRC extraction
    # ------------------------------------------------------------------

    def _load_ohrc_zip(self, zip_path):
        """Extract OHRC ZIP and return 2-D float32 albedo/DN image."""
        extract_to = self.work_dir / zip_path.stem
        extract_to.mkdir(exist_ok=True)

        print(f"  [OHRC] Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)

        # GeoTIFF
        tif_files = list(extract_to.rglob("*.tif")) + list(extract_to.rglob("*.tiff"))
        if tif_files and HAS_RIO:
            import rasterio
            with rasterio.open(tif_files[0]) as src:
                img = src.read(1).astype(np.float32)
            img /= img.max() if img.max() > 0 else 1.0
            return img

        # JPEG / PNG browse
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            imgs = list(extract_to.rglob(ext))
            if imgs:
                from PIL import Image
                arr = np.array(Image.open(imgs[0]).convert("L"), dtype=np.float32)
                arr /= 255.0
                return arr

        # PDS binary
        img_files = list(extract_to.rglob("*.img"))
        if img_files:
            lbl_files = list(extract_to.rglob("*.lbl"))
            meta = self._parse_pds_label(lbl_files[0]) if lbl_files else {}
            lines   = int(meta.get("LINES", 1024))
            samples = int(meta.get("LINE_SAMPLES", 1024))
            data    = np.fromfile(img_files[0], dtype=np.uint16).reshape(lines, samples)
            return (data / data.max()).astype(np.float32)

        print(f"    [WARN] Could not parse OHRC from {zip_path.name}")
        return None

    # ------------------------------------------------------------------
    # Mosaicking
    # ------------------------------------------------------------------

    def _mosaic_dfsar(self, arrays):
        """Stack multiple DFSAR passes into a single SC, OC mosaic."""
        if not arrays:
            from src.data_generator import generate_dem, generate_dfsar, \
                generate_psr_mask, generate_doubly_shadowed_mask, generate_illumination
            dem  = generate_dem()
            illum = generate_illumination(dem, 1000)
            psr  = generate_psr_mask(illum)
            dsc  = generate_doubly_shadowed_mask(psr)
            dfsar = generate_dfsar(dem, psr, dsc)
            return dfsar["SC"], dfsar["OC"]

        # Resize all to same shape (largest)
        max_r = max(a[0].shape[0] for a in arrays)
        max_c = max(a[0].shape[1] for a in arrays)

        SC_stack, OC_stack = [], []
        for sc, oc in arrays:
            sc_r = self._resize(sc, max_r, max_c)
            oc_r = self._resize(oc, max_r, max_c)
            SC_stack.append(sc_r)
            OC_stack.append(oc_r)

        return np.median(SC_stack, axis=0), np.median(OC_stack, axis=0)

    def _mosaic_ohrc(self, arrays):
        """Stack OHRC images into a mosaic."""
        if not arrays:
            return None
        max_r = max(a.shape[0] for a in arrays)
        max_c = max(a.shape[1] for a in arrays)
        resized = [self._resize(a, max_r, max_c) for a in arrays]
        return np.median(resized, axis=0).astype(np.float32)

    @staticmethod
    def _resize(arr, rows, cols):
        from scipy.ndimage import zoom
        zr = rows / arr.shape[0]
        zc = cols / arr.shape[1]
        return zoom(arr, (zr, zc), order=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Derived products
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_dop(SC, OC):
        """
        Degree of Polarization from dual-circular backscatter.
        DOP = |SC - OC| / (SC + OC)
        """
        total = SC + OC
        return np.where(total > 0, np.abs(SC - OC) / total, 0).astype(np.float32)

    @staticmethod
    def _compute_slope(dem, pixel_scale=PIXEL_SCALE):
        dy, dx = np.gradient(dem, pixel_scale)
        return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)

    @staticmethod
    def _dem_from_intensity(SC, pixel_scale=PIXEL_SCALE):
        """
        Rough DEM proxy from SAR intensity via shape-from-shading.
        For production: replace with co-registered LOLA DEM.
        """
        log_sc = np.log(SC + 1e-6)
        smooth = gaussian_filter(log_sc, sigma=5)
        dem    = -smooth * 500          # empirical scale
        dem   -= dem.mean()
        return dem.astype(np.float32)


# ---------------------------------------------------------------------------
# CLI helper – inspect a single ZIP without running the full pipeline
# ---------------------------------------------------------------------------

def inspect_zip(zip_path):
    """Print the contents of a data ZIP to identify format."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        print(f"\nContents of {Path(zip_path).name}:")
        for info in zf.infolist():
            size_kb = info.file_size / 1024
            print(f"  {info.filename:<60}  {size_kb:>10.1f} KB")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        inspect_zip(sys.argv[1])
    else:
        print("Usage: python -m src.data_loader <path_to_zip>")
        print("\nTo load all data:")
        print("  from src.data_loader import DataLoader")
        print("  data = DataLoader('data/raw').load()")
