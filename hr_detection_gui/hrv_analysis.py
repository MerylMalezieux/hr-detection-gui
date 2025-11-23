# -*- coding: utf-8 -*-
"""
Heart Rate Variability (HRV) Analysis Module
Contains functions for calculating various HRV metrics.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def calculate_rmssd(hr_sp_times, window_size=30):
    """
    Calculate RMSSD (Root Mean Square of Successive Differences).
    
    Parameters:
    -----------
    hr_sp_times : array
        Heart rate spike/peak times in seconds
    window_size : int, optional
        Rolling window size for smoothing (default: 30)
        
    Returns:
    --------
    rmssd : array
        RMSSD values
    rmssd_to_max : array
        RMSSD normalized to maximum (0-100)
    """
    # Calculate RR intervals
    RR = np.diff(hr_sp_times)
    RR = RR * 1000  # Convert to ms
    
    if RR.size == 0:
        return np.array([]), np.array([])
    
    # Successive differences in RR intervals
    sd = np.diff(RR)
    
    # Replace outliers (mean ± 2sd) with median value
    outlier_max = np.nanmean(sd) + np.nanstd(sd) * 2
    outlier_min = np.nanmean(sd) - np.nanstd(sd) * 2
    sd = np.where((sd > outlier_max), np.median(sd), sd)
    sd = np.where((sd < outlier_min), np.median(sd), sd)
    
    # Square of successive differences
    ssd = np.square(sd)
    
    # Mean square of successive differences
    mssd = pd.Series(ssd).rolling(window=window_size, center=True).mean()
    
    # Root mean square of successive differences
    rmssd = np.sqrt(mssd.values)
    
    # Normalize to max (0-100)
    rmssd_to_max = (rmssd * 100) / np.max(rmssd[~np.isnan(rmssd)]) if np.any(~np.isnan(rmssd)) else rmssd
    
    # Replace outliers
    rmssd = np.where((rmssd > (np.nanmedian(rmssd) + 4*np.nanstd(rmssd))),
                     np.nanmedian(rmssd), rmssd)
    
    return rmssd, rmssd_to_max


def calculate_sdnn(hr_sp_times):
    """
    Calculate SDNN (Standard Deviation of NN intervals).
    
    Parameters:
    -----------
    hr_sp_times : array
        Heart rate spike/peak times in seconds
        
    Returns:
    --------
    sdnn : float
        SDNN value in ms
    """
    RR = np.diff(hr_sp_times)
    RR = RR * 1000  # Convert to ms
    
    if RR.size == 0:
        return np.nan
    
    return np.std(RR)


def calculate_pnn50(hr_sp_times):
    """
    Calculate pNN50 (Percentage of NN intervals that differ by more than 50ms).
    
    Parameters:
    -----------
    hr_sp_times : array
        Heart rate spike/peak times in seconds
        
    Returns:
    --------
    pnn50 : float
        pNN50 percentage
    """
    RR = np.diff(hr_sp_times)
    RR = RR * 1000  # Convert to ms
    
    if RR.size == 0:
        return np.nan
    
    # Calculate successive differences
    sd = np.diff(RR)
    
    # Count differences > 50ms
    pnn50 = (np.sum(np.abs(sd) > 50) / sd.size) * 100 if sd.size > 0 else 0
    
    return pnn50


def calculate_mean_hr(hr_sp_times):
    """
    Calculate mean heart rate.
    
    Parameters:
    -----------
    hr_sp_times : array
        Heart rate spike/peak times in seconds
        
    Returns:
    --------
    mean_hr : float
        Mean heart rate in BPM
    """
    if hr_sp_times.size < 2:
        return np.nan
    
    RR = np.diff(hr_sp_times)
    mean_rr = np.mean(RR)
    
    if mean_rr == 0:
        return np.nan
    
    mean_hr = 60 / mean_rr
    return mean_hr


def calculate_all_hrv_metrics(hr_sp_times):
    """
    Calculate all HRV metrics.
    
    Parameters:
    -----------
    hr_sp_times : array
        Heart rate spike/peak times in seconds
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all HRV metrics
    """
    metrics = {}
    
    metrics['rmssd'], metrics['rmssd_to_max'] = calculate_rmssd(hr_sp_times)
    metrics['sdnn'] = calculate_sdnn(hr_sp_times)
    metrics['pnn50'] = calculate_pnn50(hr_sp_times)
    metrics['mean_hr'] = calculate_mean_hr(hr_sp_times)
    
    # Additional metrics
    RR = np.diff(hr_sp_times) * 1000  # Convert to ms
    if RR.size > 0:
        metrics['mean_rr'] = np.mean(RR)
        metrics['median_rr'] = np.median(RR)
        metrics['min_rr'] = np.min(RR)
        metrics['max_rr'] = np.max(RR)
        metrics['std_rr'] = np.std(RR)
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
    
    Parameters:
    -----------
    inst_bpm : array
        Instantaneous BPM
    bpm_to_max : array
        BPM normalized to maximum
        
    Returns:
    --------
    inst_bpm_clean : array
        Cleaned instantaneous BPM
    bpm_to_max_clean : array
        Cleaned BPM normalized to maximum
    """
    # Replace outliers in inst_bpm
    inst_bpm_clean = np.where((inst_bpm < (np.nanmedian(inst_bpm) - 4*np.nanstd(inst_bpm))),
                              np.nanmedian(inst_bpm), inst_bpm)
    inst_bpm_clean = np.where((inst_bpm_clean > (np.nanmedian(inst_bpm_clean) + 4*np.nanstd(inst_bpm_clean))),
                              np.nanmedian(inst_bpm_clean), inst_bpm_clean)
    
    # Replace outliers in bpm_to_max
    bpm_to_max_clean = np.where((bpm_to_max < (np.nanmedian(bpm_to_max) - 4*np.nanstd(bpm_to_max))),
                                np.nanmedian(bpm_to_max), bpm_to_max)
    bpm_to_max_clean = np.where((bpm_to_max_clean > (np.nanmedian(bpm_to_max_clean) + 4*np.nanstd(bpm_to_max_clean))),
                                np.nanmedian(bpm_to_max_clean), bpm_to_max_clean)
    
    # Replace values below 10
    bpm_to_max_clean = np.where((bpm_to_max_clean < 10),
                                np.nanmedian(bpm_to_max_clean), bpm_to_max_clean)
    
    # Find and interpolate drop regions
    drop_start = np.where(np.ediff1d(bpm_to_max_clean) > 5)[0]
    drop_stop = np.where(np.ediff1d(bpm_to_max_clean) < -5)[0]
    
    if drop_start.size == drop_stop.size:
        for k in np.arange(drop_start.size):
            start_idx = max(0, drop_start[k] - 1000)
            stop_idx = min(bpm_to_max_clean.size, drop_stop[k] + 1000)
            bpm_to_max_clean[start_idx:stop_idx] = 0
    elif (drop_start.size + 1) == drop_stop.size:
        for k in np.arange(drop_start.size):
            start_idx = max(0, drop_start[k] - 1000)
            stop_idx = min(bpm_to_max_clean.size, drop_stop[k] + 1000)
            bpm_to_max_clean[start_idx:stop_idx] = 0
    elif drop_start.size == 1 and drop_stop.size == 0:
        start_idx = max(0, drop_start[0] - 1000)
        bpm_to_max_clean[start_idx:-1] = 0
    
    # Interpolate missing data
    y = bpm_to_max_clean
    x = np.arange(len(y))
    idx = np.where(y != 0)[0]
    
    if idx.size > 1:
        f = interp1d(x[idx], y[idx], kind='linear', fill_value='extrapolate')
        bpm_to_max_clean = f(x)
    
    return inst_bpm_clean, bpm_to_max_clean

