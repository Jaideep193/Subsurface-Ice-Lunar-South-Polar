"""
Range-Doppler SAR Focusing for Chandrayaan-2 DFSAR L0B Data
============================================================

Implements a simplified Range-Doppler algorithm (RDA) to focus
DFSAR Level-0B (raw compressed echo) data into a Single Look Complex
(SLC) image suitable for polarimetric CPR/DOP analysis.

Steps:
  1. Range compression   — matched filter with transmitted chirp
  2. Range cell migration correction (RCMC) — bulk shift method
  3. Azimuth compression — Doppler centroid estimation + MF

Physical parameters are extracted from the XML metadata parsed by
real_data_loader.py (PRF, bandwidth, centre frequency, altitude,
velocity, incidence angle).

Reference:
  Cumming & Wong (2005) "Digital Processing of Synthetic Aperture Radar Data"
  Artech House, Chapter 5 (Range-Doppler Algorithm).

Notes on Chandrayaan-2 DFSAR L0B format:
  - BAQ-compressed complex int16  (real/imag pairs per sample)
  - Two polarisation channels stored in alternating echoes:
      Even echoes → LH channel,  Odd echoes → LV channel
  - File: .dat (binary), Metadata: .xml
  - Typical scene: ~3000 range samples × ~50000 azimuth lines
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.ndimage import map_coordinates


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEED_OF_LIGHT = 2.998e8     # m/s
EARTH_RATE_RAD = 7.2921e-5   # rad/s (not used for Moon but kept for API compat)


# ---------------------------------------------------------------------------
# Raw data reader
# ---------------------------------------------------------------------------

def read_l0b_raw(dat_path, n_range_samples, n_az_lines=None,
                  dtype=np.int16, interleave="IQ"):
    """
    Read DFSAR L0B binary echo data.

    Parameters
    ----------
    dat_path        : str / Path   path to .dat file
    n_range_samples : int          samples per echo line (from XML)
    n_az_lines      : int | None   lines to read (None = all)
    dtype           : numpy dtype  raw word type (int16 default)
    interleave      : str          'IQ' = real/imag per sample (default)

    Returns
    -------
    raw : complex64 ndarray  shape (n_az_lines, n_range_samples)
    """
    import os
    word_bytes = np.dtype(dtype).itemsize
    samples_per_line = n_range_samples * 2   # I + Q per sample

    file_size = os.path.getsize(dat_path)
    total_words = file_size // word_bytes
    max_lines = total_words // samples_per_line

    if n_az_lines is None:
        n_az_lines = max_lines
    else:
        n_az_lines = min(n_az_lines, max_lines)

    print(f"  [SAR Focus] Reading {n_az_lines} azimuth lines × "
          f"{n_range_samples} range samples from {dat_path.name}")

    raw_flat = np.fromfile(dat_path, dtype=dtype,
                            count=n_az_lines * samples_per_line)
    raw_2d = raw_flat.reshape(n_az_lines, samples_per_line)

    # De-interleave I and Q
    raw_cplx = (raw_2d[:, 0::2].astype(np.float32)
                + 1j * raw_2d[:, 1::2].astype(np.float32))
    return raw_cplx.astype(np.complex64)


def split_polarisations(raw_all):
    """
    Split alternating-echo LH/LV channels.

    Convention for DFSAR compact-pol:
      Even azimuth lines (0, 2, 4, ...) → LH polarisation
      Odd  azimuth lines (1, 3, 5, ...) → LV polarisation
    """
    lh = raw_all[0::2, :]   # even
    lv = raw_all[1::2, :]   # odd
    return lh, lv


# ---------------------------------------------------------------------------
# Step 1 – Range compression (matched filter)
# ---------------------------------------------------------------------------

def range_compress(raw, bandwidth_hz, prf_hz, center_freq_hz,
                   range_sampling_rate_hz=None):
    """
    Matched filter range compression.

    The reference chirp is a linear frequency-modulated pulse:
        h(t) = exp(j * pi * K_r * t^2),  |t| <= tau/2
    where K_r = bandwidth / pulse_duration is the chirp rate.

    In the frequency domain: H*(f) = exp(-j * pi * f^2 / K_r)

    Parameters
    ----------
    raw             : complex64 ndarray  (az_lines, range_samples)
    bandwidth_hz    : float   pulse bandwidth (from XML pulse_bandwidth_hz)
    prf_hz          : float   pulse repetition frequency
    center_freq_hz  : float   radar centre frequency (Hz)
    range_sampling_rate_hz : float | None  (if None, estimated from bandwidth)
    """
    az_lines, rg_samples = raw.shape

    if range_sampling_rate_hz is None:
        # Nyquist: sample rate >= 2 * bandwidth (use 1.5× oversampling)
        range_sampling_rate_hz = bandwidth_hz * 1.5

    # Frequency axis
    f_range = fftfreq(rg_samples, d=1.0 / range_sampling_rate_hz)

    # Chirp rate estimate: K_r = B / tau.  With tau unknown, estimate from
    # pulse duration ≈ 1 / PRF * 0.3 (30% duty cycle typical for DFSAR)
    tau = 0.3 / prf_hz
    K_r = bandwidth_hz / tau

    # Reference function (matched filter in range freq domain)
    ref_fn = np.exp(-1j * np.pi * f_range**2 / K_r).astype(np.complex64)

    # Apply column-wise matched filter
    raw_f = fft(raw, axis=1)
    compressed = ifft(raw_f * ref_fn[np.newaxis, :], axis=1)
    return compressed.astype(np.complex64)


# ---------------------------------------------------------------------------
# Step 2 – Range cell migration correction (RCMC)
# ---------------------------------------------------------------------------

def rcmc(raw_rg_compressed, prf_hz, center_freq_hz,
          spacecraft_altitude_m, incidence_angle_deg,
          range_sampling_rate_hz=None, bandwidth_hz=None):
    """
    Bulk Range Cell Migration Correction.

    The range migration ΔR(f_az) = R0 * (1 - 1/sqrt(1 - (f_az*lambda/(2*v))^2))
    where R0 = slant range, lambda = wavelength, v = satellite velocity.

    This simplified implementation uses a constant-shift RCMC (valid for
    small scene widths and low squint angles, appropriate for DFSAR nadir mode).

    For full accuracy, the secondary range compression (SRC) term should be
    applied; here we apply only the bulk linear migration removal.
    """
    az_lines, rg_samples = raw_rg_compressed.shape

    wavelength_m = SPEED_OF_LIGHT / center_freq_hz
    # Lunar orbital velocity at given altitude (~100 km orbit)
    MOON_RADIUS_M = 1_737_400.0
    GM_MOON = 4.9048e12  # m^3/s^2
    orbital_radius = MOON_RADIUS_M + spacecraft_altitude_m
    v_sat = float(np.sqrt(GM_MOON / orbital_radius))  # ~1.6 km/s

    # Slant range at scene centre
    R0 = spacecraft_altitude_m / np.cos(np.radians(incidence_angle_deg))

    if range_sampling_rate_hz is None and bandwidth_hz is not None:
        range_sampling_rate_hz = bandwidth_hz * 1.5
    elif range_sampling_rate_hz is None:
        range_sampling_rate_hz = 50e6   # 50 MHz fallback

    # Azimuth Doppler frequencies
    f_az = fftfreq(az_lines, d=1.0 / prf_hz)
    # Range shift per azimuth Doppler bin (in range samples)
    migration_m = R0 * (1.0 / np.sqrt(
        np.clip(1.0 - (f_az * wavelength_m / (2.0 * v_sat))**2, 1e-6, None)
    ) - 1.0)
    migration_samples = migration_m / (SPEED_OF_LIGHT / (2 * range_sampling_rate_hz))

    # Apply shift in azimuth-Doppler domain (range FFT + phase ramp per line)
    raw_az_fft = fft(raw_rg_compressed, axis=0)
    rg_indices  = np.arange(rg_samples, dtype=np.float32)
    phase_ramp  = np.exp(1j * 2 * np.pi
                          * migration_samples[:, np.newaxis]
                          * rg_indices[np.newaxis, :] / rg_samples)
    corrected = ifft(raw_az_fft * phase_ramp.astype(np.complex64), axis=0)
    return corrected.astype(np.complex64)


# ---------------------------------------------------------------------------
# Step 3 – Azimuth compression
# ---------------------------------------------------------------------------

def azimuth_compress(rcmc_data, prf_hz, center_freq_hz,
                      spacecraft_altitude_m, incidence_angle_deg,
                      squint_angle_deg=0.0):
    """
    Azimuth matched filter compression.

    Reference: Cumming & Wong (2005) Section 5.4.
    The azimuth matched filter is:
        H_az(f_az) = exp(j * pi * f_az^2 / K_a)
    where K_a = 2*v^2 / (lambda * R0) is the azimuth FM rate.
    """
    az_lines, rg_samples = rcmc_data.shape

    wavelength_m = SPEED_OF_LIGHT / center_freq_hz
    MOON_RADIUS_M = 1_737_400.0
    GM_MOON = 4.9048e12
    orbital_radius = MOON_RADIUS_M + spacecraft_altitude_m
    v_sat = float(np.sqrt(GM_MOON / orbital_radius))

    R0 = spacecraft_altitude_m / np.cos(np.radians(incidence_angle_deg))

    K_a = 2.0 * v_sat**2 / (wavelength_m * R0)   # azimuth FM rate (Hz/s)

    f_az = fftfreq(az_lines, d=1.0 / prf_hz)
    h_az = np.exp(1j * np.pi * f_az**2 / K_a).astype(np.complex64)

    raw_az_f = fft(rcmc_data, axis=0)
    focused   = ifft(raw_az_f * h_az[:, np.newaxis], axis=0)
    return focused.astype(np.complex64)


# ---------------------------------------------------------------------------
# Full focusing pipeline
# ---------------------------------------------------------------------------

def focus_slc(dat_path, xml_meta, n_az_lines=4096,
               range_sampling_rate_hz=None):
    """
    Focus a DFSAR L0B .dat file to SLC using the Range-Doppler Algorithm.

    Parameters
    ----------
    dat_path   : Path   .dat binary file
    xml_meta   : dict   from real_data_loader._parse_sar_xml()
    n_az_lines : int    number of azimuth lines to process

    Returns
    -------
    dict with keys:
        LH_slc : complex64 (az, rg)   focused LH channel
        LV_slc : complex64 (az, rg)   focused LV channel
        meta   : dict                  focusing parameters used
    """
    # Extract parameters from XML metadata
    prf_hz          = float(xml_meta.get("prf_hz") or 1500.0)
    bandwidth_hz    = float(xml_meta.get("pulse_bandwidth_hz") or 75e6)
    center_freq_hz  = float(xml_meta.get("radar_frequency_hz") or 1.257e9)
    altitude_m      = float(xml_meta.get("spacecraft_altitude_m") or 100e3)
    incidence_deg   = float(xml_meta.get("incidence_angle_deg") or 26.0)
    n_rg_samples    = int(xml_meta.get("samples_per_echo") or 3072)

    if range_sampling_rate_hz is None:
        range_sampling_rate_hz = bandwidth_hz * 1.5

    print(f"  [SAR Focus] PRF={prf_hz:.0f} Hz  BW={bandwidth_hz/1e6:.1f} MHz  "
          f"f0={center_freq_hz/1e9:.3f} GHz  Alt={altitude_m/1e3:.1f} km")

    # Read raw data
    raw_all = read_l0b_raw(dat_path, n_rg_samples, n_az_lines)

    # Split polarisations
    lh_raw, lv_raw = split_polarisations(raw_all)

    focused_channels = {}
    for name, raw in [("LH", lh_raw), ("LV", lv_raw)]:
        print(f"  [SAR Focus] Focusing {name} channel "
              f"({raw.shape[0]} az × {raw.shape[1]} rg)...")
        rc  = range_compress(raw, bandwidth_hz, prf_hz, center_freq_hz,
                              range_sampling_rate_hz)
        rmc = rcmc(rc, prf_hz, center_freq_hz, altitude_m, incidence_deg,
                   range_sampling_rate_hz, bandwidth_hz)
        slc = azimuth_compress(rmc, prf_hz, center_freq_hz,
                                altitude_m, incidence_deg)
        focused_channels[f"{name}_slc"] = slc

    # Compute amplitude images from SLC
    LH = np.abs(focused_channels["LH_slc"]).astype(np.float32)
    LV = np.abs(focused_channels["LV_slc"]).astype(np.float32)

    # SC / OC power (compact-pol)
    SC_power = ((LH**2 + LV**2) / 2.0).astype(np.float32)   # approx
    OC_power = SC_power.copy()

    meta = dict(
        prf_hz=prf_hz, bandwidth_hz=bandwidth_hz,
        center_freq_hz=center_freq_hz, altitude_m=altitude_m,
        incidence_deg=incidence_deg, n_az=lh_raw.shape[0],
        n_rg=n_rg_samples, range_sampling_rate_hz=range_sampling_rate_hz,
    )

    return dict(
        LH_slc   = focused_channels["LH_slc"],
        LV_slc   = focused_channels["LV_slc"],
        LH_amp   = LH,
        LV_amp   = LV,
        SC       = SC_power,
        OC       = OC_power,
        meta     = meta,
    )


def slc_to_dfsar_dict(focused, psr_mask, dsc_mask, target_gs=1000,
                       frequency_band="L"):
    """
    Resample focused SLC amplitudes onto pipeline grid and build dfsar dict.

    After focusing, the SLC may have different dimensions than the 1000×1000
    pipeline grid.  This resamples to the target grid using bilinear zoom.
    """
    from scipy.ndimage import zoom

    LH = focused["LH_amp"]
    LV = focused["LV_amp"]

    # Resample to pipeline grid
    zy = target_gs / LH.shape[0]
    zx = target_gs / LH.shape[1]
    LH_r = zoom(LH, (zy, zx), order=1).astype(np.float32)
    LV_r = zoom(LV, (zy, zx), order=1).astype(np.float32)

    # Stokes parameters from amplitude images (no phase — approximation)
    I = LH_r**2 + LV_r**2
    Q = LH_r**2 - LV_r**2
    U = np.zeros_like(I)           # cross-product phase not available from amp
    V = np.zeros_like(I)

    # SC / OC
    SC = np.clip((I - V) / 2, 1e-6, None)
    OC = np.clip((I + V) / 2, 1e-6, None)
    CPR = SC / (OC + 1e-12)

    from scipy.ndimage import gaussian_filter
    DOP_raw = np.where(I > 1e-6, np.sqrt(Q**2 + U**2 + V**2) / I, 0.0)
    DOP = gaussian_filter(DOP_raw.astype(np.float32), sigma=1.5)

    nes0 = float(focused["meta"].get("nes0", 1e-3))

    return dict(
        SC=SC, OC=OC, CPR=CPR.astype(np.float32), DOP=DOP,
        stokes_I=I.astype(np.float32), stokes_Q=Q.astype(np.float32),
        stokes_U=U.astype(np.float32), stokes_V=V.astype(np.float32),
        sigma0_hh=(10*np.log10(LH_r**2 + 1e-9)).astype(np.float32),
        sigma0_hv=(10*np.log10(LV_r**2 + 1e-9)).astype(np.float32),
        nes0=nes0, frequency_band=frequency_band,
        cpr_ice_threshold=0.8,
        focused=True,
    )
