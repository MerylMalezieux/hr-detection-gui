# -*- coding: utf-8 -*-
"""
Heart Rate Detection Module
Contains functions for loading ABF files and detecting heart rate peaks.
"""

import os
import numpy as np
import pandas as pd
from neo import io
from scipy import signal
from scipy.signal import savgol_filter


def downsample(ts, signal_data, ds_factor):
    """
    Downsample signal by averaging over ds_factor samples.
    
    Parameters:
    -----------
    ts : array
        Time stamps
    signal_data : array
        Signal to downsample
    ds_factor : int
        Downsampling factor
        
    Returns:
    --------
    ds_ts : array
        Downsampled time stamps
    signal_ds : array
        Downsampled signal
    """
    signal_ds = np.mean(np.resize(signal_data,
                        (int(np.floor(signal_data.size/ds_factor)), ds_factor)), 1)
    ds_ts = ts[np.arange(int(np.round(ds_factor/2)), ts.size, ds_factor)]
    # trim off last time stamp if necessary
    ds_ts = ds_ts[0:signal_ds.size]
    return ds_ts, signal_ds


def load_abf_file(file_path, channel_name='IN2', channel_index=None, 
                  start_time=0, stop_time=None, downsample_factor=10):
    """
    Load HR data from ABF file.
    
    Parameters:
    -----------
    file_path : str
        Path to ABF file
    channel_name : str, optional
        Name of channel to load (default: 'IN2')
    channel_index : int, optional
        Index of channel if channel_name not found (default: 2)
    start_time : float, optional
        Start time in seconds (default: 0)
    stop_time : float, optional
        Stop time in seconds (default: None, uses full recording)
    downsample_factor : int, optional
        Downsampling factor (default: 10)
        
    Returns:
    --------
    hr : array
        Heart rate signal
    hr_ts : array
        Time stamps for heart rate signal
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load ABF file
    r = io.AxonIO(filename=file_path)
    bl = r.read_block(signal_group_mode='split-all')
    
    # Get list of channel names
    channel_list = []
    for asig in bl.segments[0].analogsignals:
        channel_list.append(asig.name)
    
    full_ts = np.copy(bl.segments[0].analogsignals[0].times)
    
    # Find channel
    if channel_name in channel_list:
        ind = channel_list.index(channel_name)
        hr = np.copy(bl.segments[0].analogsignals[ind].data)
    else:
        # Use default channel index if channel_name not found
        if channel_index is None:
            channel_index = 2
        hr = np.copy(bl.segments[0].analogsignals[channel_index].data)
    
    # Apply time window
    if stop_time is None:
        stop_time = full_ts[-1]
    
    mask = (full_ts >= start_time) & (full_ts < stop_time)
    hr = hr[mask]
    hr_ts = full_ts[mask]
    hr = np.squeeze(hr)
    
    # Downsample
    if downsample_factor > 1:
        hr_ts, hr = downsample(hr_ts, hr, downsample_factor)
    
    # Smooth the data
    hr = savgol_filter(hr, 21, 2)
    
    return hr, hr_ts


def find_hr_peaks(ts, hr, thresh, refrac, min_dur, highpass=None, use_abs=False):
    """
    Find HR peak indices using threshold detection on derivative.
    
    Parameters:
    -----------
    ts : array
        Time stamps
    hr : array
        Heart rate signal
    thresh : float
        Threshold for derivative detection
    refrac : float
        Refractory period in ms
    min_dur : float
        Minimum duration in samples
    highpass : float, optional
        Highpass filter frequency in Hz (default: None)
    use_abs : bool, optional
        Use absolute value of signal (default: False)
        
    Returns:
    --------
    dVdt_thresh : array
        Thresholded derivative signal
    hr_sp_ind : array
        Indices of detected peaks
    """
    # Apply highpass filter if specified
    if highpass and highpass > 0:
        samp_freq = 1/(ts[1] - ts[0])
        nyq = samp_freq/2
        b, a = signal.butter(4, highpass/nyq, "high", analog=False)
        hr = signal.filtfilt(b, a, hr)
    
    # Use absolute value if specified
    if use_abs:
        hr = np.abs(hr)
    
    # Calculate derivative
    hr_diff = np.reshape(hr, (1, hr.size))
    dVdt = np.diff(hr_diff)
    dVdt = np.reshape(dVdt, (dVdt.size,))
    
    # Apply threshold
    dVdt_thresh = np.array(dVdt > thresh, float)
    
    if sum(dVdt_thresh) == 0:
        # No spikes detected
        hr_sp_ind = np.empty(shape=0)
    elif sum(dVdt_thresh) == 1:
        hr_sp_ind = np.where(np.diff(dVdt_thresh) == 1)[0]
        # Remove spikes that are too short
        stop = np.where(np.diff(dVdt_thresh) == -1)[0]
        if stop.size > 0:
            difference = stop - hr_sp_ind
            hr_sp_ind = hr_sp_ind[np.where(difference > min_dur)]
    else:
        # Keep just the first index per spike
        hr_sp_ind = np.where(np.diff(dVdt_thresh) == 1)[0]
        # Remove spikes that are too short
        stop = np.where(np.diff(dVdt_thresh) == -1)[0]
        if stop.size > 0:
            if stop[0] < hr_sp_ind[0]:
                stop = stop[1:]
            if stop.size > 0 and stop[-1] < hr_sp_ind[-1]:
                hr_sp_ind = hr_sp_ind[:-1]
            if stop.size > 0 and hr_sp_ind.size > 0:
                difference = stop[:hr_sp_ind.size] - hr_sp_ind
                hr_sp_ind = hr_sp_ind[np.where(difference > min_dur)]
        
        # Remove duplicates within refractory period
        if hr_sp_ind.size > 0:
            samp_rate = np.round(1/(ts[1]-ts[0]))
            hr_sp_ind = hr_sp_ind[np.ediff1d(hr_sp_ind,
                            to_begin=int(samp_rate*refrac/1000+1)) > samp_rate*refrac/1000]
    
    return dVdt_thresh, hr_sp_ind


def find_inst_bpm(hr, sp_times, ts):
    """
    Calculate instantaneous BPM.
    
    Parameters:
    -----------
    hr : array
        Heart rate signal
    sp_times : array
        Spike/peak times
    ts : array
        Time stamps
        
    Returns:
    --------
    inst_bpm : array
        Instantaneous BPM
    """
    inst_bpm = np.zeros(hr.size)
    for j in np.arange(inst_bpm.size):
        if j+int(ts.size/ts[-1]) < inst_bpm.size:
            # Count beats in 200 timepoints
            inst_bpm[j] = sp_times[(sp_times > ts[j]) & (sp_times < ts[j+ 200])].size
            # Multiply by 60 to get bpm
            inst_bpm[j] = inst_bpm[j] * 600
        else:
            # Replace last values where measurement is not possible
            inst_bpm[j] = inst_bpm[j-1] if j > 0 else 0
    
    # Filter the data
    inst_bpm = savgol_filter(inst_bpm, 1001, 2)
    # Smooth with rolling window
    inst_bpm = pd.Series(inst_bpm).rolling(window=1000, center=True).mean()
    return inst_bpm.values

