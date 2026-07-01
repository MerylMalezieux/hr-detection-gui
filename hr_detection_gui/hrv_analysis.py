# -*- coding: utf-8 -*-
"""
Heart Rate Variability (HRV) Analysis Module
Contains functions for calculating various HRV metrics.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def interpolate_nan_values(values):
    """Linearly interpolate NaN gaps; hold edge values at boundaries."""
    values = np.asarray(values, dtype=float).copy()
    if values.size == 0:
        return values

    nan_mask = np.isnan(values)
    if not np.any(nan_mask):
        return values
    if np.all(nan_mask):
        return values

    x = np.arange(values.size)
    valid = ~nan_mask
    interpolator = interp1d(
        x[valid],
        values[valid],
        kind='linear',
        bounds_error=False,
        fill_value=(values[valid][0], values[valid][-1]),
    )
    return interpolator(x)


def remove_outliers_and_interpolate(values, n_std=3):
    """
    Mark values outside mean ± n_std*std as outliers and linearly interpolate.

    Parameters
    ----------
    values : array-like
        Input signal (e.g. RR intervals, successive differences, or BPM).
    n_std : float
        Number of standard deviations for outlier bounds.

    Returns
    -------
    array
        Cleaned values with outliers replaced by interpolation.
    """
    values = np.asarray(values, dtype=float).copy()
    if values.size == 0:
        return values

    valid_values = values[~np.isnan(values)]
    if valid_values.size == 0:
        return values

    mean = np.nanmean(valid_values)
    std = np.nanstd(valid_values)
    if np.isnan(std) or std == 0:
        return values

    outlier_max = mean + std * n_std
    outlier_min = mean - std * n_std
    outlier_mask = (values > outlier_max) | (values < outlier_min)
    if not np.any(outlier_mask):
        return values

    values[outlier_mask] = np.nan
    return interpolate_nan_values(values)


def rr_intervals_from_peaks(hr_sp_times):
    """Compute RR intervals in milliseconds from sorted peak times."""
    hr_sp_times = np.asarray(hr_sp_times, dtype=float)
    return np.diff(hr_sp_times) * 1000.0


def clean_rr_intervals(hr_sp_times, n_std=3):
    """
    Clean RR intervals directly (each interval cleaned, not reconstructed from sd).

    Non-physical intervals (<= 0 ms) are removed and interpolated.
    """
    rr_ms = rr_intervals_from_peaks(hr_sp_times)
    if rr_ms.size == 0:
        return rr_ms

    rr_ms = rr_ms.copy()
    rr_ms[rr_ms <= 0] = np.nan
    rr_ms = interpolate_nan_values(rr_ms)
    return remove_outliers_and_interpolate(rr_ms, n_std=n_std)


def clean_successive_rr_differences(rr_ms, n_std=3):
    """Clean successive RR differences (sd) for RMSSD, without altering RR itself."""
    if rr_ms.size < 2:
        return np.array([])
    sd = np.diff(rr_ms)
    return remove_outliers_and_interpolate(sd, n_std=n_std)


def calculate_rmssd(hr_sp_times, window_size=30, n_std=3):
    """
    Calculate RMSSD (Root Mean Square of Successive Differences).

    Outlier cleaning is applied to successive RR differences (sd) only,
    matching the original analysis approach.
    """
    rr_ms = rr_intervals_from_peaks(hr_sp_times)
    if rr_ms.size < 2:
        return np.array([]), np.array([])

    sd_clean = clean_successive_rr_differences(rr_ms, n_std=n_std)
    if sd_clean.size == 0:
        return np.array([]), np.array([])

    ssd = np.square(sd_clean)
    mssd = pd.Series(ssd).rolling(window=window_size, center=True).mean()
    rmssd = np.sqrt(mssd.values)

    valid_rmssd = rmssd[~np.isnan(rmssd)]
    if valid_rmssd.size > 0:
        rmssd_to_max = (rmssd * 100) / np.max(valid_rmssd)
        rmssd = np.where(
            rmssd > (np.nanmedian(rmssd) + 4 * np.nanstd(rmssd)),
            np.nanmedian(rmssd),
            rmssd,
        )
    else:
        rmssd_to_max = rmssd

    return rmssd, rmssd_to_max


def calculate_sdnn(rr_ms):
    """Calculate SDNN from RR intervals in milliseconds."""
    if rr_ms.size == 0:
        return np.nan
    return np.std(rr_ms)


def calculate_pnn50(rr_ms):
    """Calculate pNN50 from RR intervals in milliseconds."""
    if rr_ms.size < 2:
        return np.nan

    sd = np.diff(rr_ms)
    return (np.sum(np.abs(sd) > 50) / sd.size) * 100 if sd.size > 0 else 0


def calculate_mean_hr(rr_ms):
    """Calculate mean heart rate in BPM from RR intervals in milliseconds."""
    if rr_ms.size == 0:
        return np.nan

    mean_rr_ms = np.mean(rr_ms)
    if mean_rr_ms <= 0:
        return np.nan

    return 60000.0 / mean_rr_ms


def calculate_all_hrv_metrics(hr_sp_times, n_std=3):
    """
    Calculate all HRV metrics.

    RR intervals are cleaned directly; RMSSD uses cleaned successive differences.
    """
    metrics = {}
    rr_ms = clean_rr_intervals(hr_sp_times, n_std=n_std)
    metrics['rr_intervals'] = rr_ms

    metrics['rmssd'], metrics['rmssd_to_max'] = calculate_rmssd(hr_sp_times, n_std=n_std)
    metrics['sdnn'] = calculate_sdnn(rr_ms)
    metrics['pnn50'] = calculate_pnn50(rr_ms)
    metrics['mean_hr'] = calculate_mean_hr(rr_ms)

    if rr_ms.size > 0:
        metrics['mean_rr'] = np.mean(rr_ms)
        metrics['median_rr'] = np.median(rr_ms)
        metrics['min_rr'] = np.min(rr_ms)
        metrics['max_rr'] = np.max(rr_ms)
        metrics['std_rr'] = np.std(rr_ms)
    else:
        metrics['mean_rr'] = np.nan
        metrics['median_rr'] = np.nan
        metrics['min_rr'] = np.nan
        metrics['max_rr'] = np.nan
        metrics['std_rr'] = np.nan

    return metrics


def clean_bpm_signal(inst_bpm, bpm_to_max):
    """
    Clean BPM signal by removing outliers and interpolating missing data.
    """
    inst_bpm_clean = np.asarray(inst_bpm, dtype=float).copy()
    bpm_to_max_clean = np.asarray(bpm_to_max, dtype=float).copy()

    # Zeros and NaNs come from empty beat-count windows in find_inst_bpm
    inst_bpm_clean[inst_bpm_clean <= 0] = np.nan
    inst_bpm_clean = interpolate_nan_values(inst_bpm_clean)
    bpm_to_max_clean = interpolate_nan_values(bpm_to_max_clean)

    inst_bpm_clean = remove_outliers_and_interpolate(inst_bpm_clean, n_std=3)
    bpm_to_max_clean = remove_outliers_and_interpolate(bpm_to_max_clean, n_std=3)

    # Remove sharp low drops (artifactual gaps between beats)
    median_bpm = np.nanmedian(inst_bpm_clean)
    if np.isfinite(median_bpm) and median_bpm > 0:
        low_threshold = max(30.0, median_bpm * 0.5)
        inst_bpm_clean[inst_bpm_clean < low_threshold] = np.nan
        inst_bpm_clean = interpolate_nan_values(inst_bpm_clean)

        median_scaled = np.nanmedian(bpm_to_max_clean)
        if np.isfinite(median_scaled) and median_scaled > 0:
            low_scaled = max(5.0, median_scaled * 0.5)
            bpm_to_max_clean[bpm_to_max_clean < low_scaled] = np.nan
            bpm_to_max_clean = interpolate_nan_values(bpm_to_max_clean)

    return inst_bpm_clean, bpm_to_max_clean
