# -*- coding: utf-8 -*-
"""
Heart Rate Detection Module
Contains functions for loading ABF files and detecting heart rate peaks.
"""

import os
import json
import numpy as np
import pandas as pd
from neo import io
from scipy import signal
from scipy.signal import savgol_filter
from scipy.io import loadmat, wavfile


class SamplingRateRequiredError(Exception):
    """Exception raised when sampling rate is required but not provided."""
    pass


class ColumnSelectionRequiredError(Exception):
    """Exception raised when column selection is required (e.g., for MATLAB files with labels)."""
    def __init__(self, message, available_columns=None, labels=None):
        super().__init__(message)
        self.available_columns = available_columns
        self.labels = labels


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
    Calculate instantaneous BPM using a time-based window.
    
    Parameters:
    -----------
    hr : array
        Heart rate signal
    sp_times : array
        Spike/peak times (in seconds)
    ts : array
        Time stamps (in seconds)
        
    Returns:
    --------
    inst_bpm : array
        Instantaneous BPM
    """
    inst_bpm = np.zeros(hr.size)
    
    # Use a 1-second window for BPM calculation
    window_duration = 1.0  # seconds
    
    for j in np.arange(inst_bpm.size):
        current_time = ts[j]
        window_end_time = current_time + window_duration
        
        # Find the index corresponding to window_end_time
        # Use binary search for efficiency if array is large
        if window_end_time <= ts[-1]:
            # Count beats in the time window [current_time, window_end_time]
            beats_in_window = np.sum((sp_times > current_time) & (sp_times <= window_end_time))
            
            # Convert to BPM: beats per second * 60 seconds per minute
            inst_bpm[j] = (beats_in_window / window_duration) * 60.0
        else:
            # For the last part of the signal, use a shorter window or extrapolate
            if j > 0:
                # Use the last valid BPM value
                inst_bpm[j] = inst_bpm[j-1]
            else:
                # If this is the very first point and we're past the end, use a backward window
                window_start_time = current_time - window_duration
                if window_start_time >= ts[0]:
                    beats_in_window = np.sum((sp_times >= window_start_time) & (sp_times <= current_time))
                    inst_bpm[j] = (beats_in_window / window_duration) * 60.0
                else:
                    inst_bpm[j] = 0
    
    # Filter the data (only if we have enough points)
    if len(inst_bpm) > 1001:
        inst_bpm = savgol_filter(inst_bpm, 1001, 2)
    elif len(inst_bpm) > 5:
        window_length = len(inst_bpm) if len(inst_bpm) % 2 == 1 else len(inst_bpm) - 1
        inst_bpm = savgol_filter(inst_bpm, window_length, min(2, window_length // 2))
    
    # Smooth with rolling window (only if we have enough points)
    if len(inst_bpm) > 1000:
        inst_bpm = pd.Series(inst_bpm).rolling(window=1000, center=True).mean()
    elif len(inst_bpm) > 10:
        window_size = min(1000, len(inst_bpm) // 2)
        inst_bpm = pd.Series(inst_bpm).rolling(window=window_size, center=True).mean()
    
    return inst_bpm.values


def load_csv_file(file_path, signal_column=0, time_column=None, 
                  sampling_rate=None, delimiter=None, has_header=False, 
                  downsample_factor=10):
    """
    Load ECG signal from CSV or TXT file.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV/TXT file
    signal_column : int, optional
        Column index containing the signal (default: 0)
    time_column : int, optional
        Column index containing time stamps (default: None, generates from sampling_rate)
    sampling_rate : float, optional
        Sampling rate in Hz (required if time_column is None)
    delimiter : str, optional
        Delimiter character (default: None, auto-detect)
    has_header : bool, optional
        Whether file has a header row (default: False)
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
    
    # Try to auto-detect delimiter
    if delimiter is None:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            if ',' in first_line:
                delimiter = ','
            elif '\t' in first_line:
                delimiter = '\t'
            else:
                delimiter = None  # Let pandas handle it (whitespace)
    
    # Load data
    try:
        # Try with header first if has_header is True
        if has_header:
            data = pd.read_csv(file_path, delimiter=delimiter, header=0, 
                              skipinitialspace=True, engine='python')
        else:
            data = pd.read_csv(file_path, delimiter=delimiter, header=None, 
                              skipinitialspace=True, engine='python')
    except Exception as e:
        # Try without header if first attempt failed
        try:
            data = pd.read_csv(file_path, delimiter=delimiter, header=None, 
                              skipinitialspace=True, engine='python')
        except Exception as e2:
            raise ValueError(f"Failed to parse CSV/TXT file: {str(e2)}")
    
    if data.empty:
        raise ValueError("File is empty or could not be parsed")
    
    # Extract signal
    if signal_column >= data.shape[1]:
        raise ValueError(f"Signal column {signal_column} does not exist. File has {data.shape[1]} columns.")
    
    hr = pd.to_numeric(data.iloc[:, signal_column], errors='coerce').values
    
    # Extract or generate time stamps
    if time_column is not None:
        if time_column >= data.shape[1]:
            raise ValueError(f"Time column {time_column} does not exist. File has {data.shape[1]} columns.")
        hr_ts = pd.to_numeric(data.iloc[:, time_column], errors='coerce').values
    else:
        if sampling_rate is None:
            raise SamplingRateRequiredError("Sampling rate is required for this file format. Please provide sampling_rate parameter.")
        hr_ts = np.arange(len(hr)) / sampling_rate
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(hr) | np.isnan(hr_ts))
    hr = hr[valid_mask]
    hr_ts = hr_ts[valid_mask]
    
    if len(hr) == 0:
        raise ValueError("No valid data points found in file")
    
    # Ensure time stamps are monotonically increasing
    if not np.all(np.diff(hr_ts) > 0):
        # Sort by time if needed
        sort_idx = np.argsort(hr_ts)
        hr_ts = hr_ts[sort_idx]
        hr = hr[sort_idx]
    
    # Downsample
    if downsample_factor > 1:
        hr_ts, hr = downsample(hr_ts, hr, downsample_factor)
    
    # Smooth the data (only if we have enough points)
    if len(hr) > 21:
        window_length = min(21, len(hr) if len(hr) % 2 == 1 else len(hr) - 1)
        hr = savgol_filter(hr, window_length, 2)
    elif len(hr) > 5:
        window_length = len(hr) if len(hr) % 2 == 1 else len(hr) - 1
        hr = savgol_filter(hr, window_length, min(2, window_length // 2))
    
    return hr, hr_ts


def load_mat_file(file_path, signal_key=None, time_key=None, 
                  sampling_rate=None, signal_column=None, downsample_factor=10):
    """
    Load ECG signal from MATLAB .mat file.
    
    Parameters:
    -----------
    file_path : str
        Path to .mat file
    signal_key : str, optional
        Key name for signal data (default: None, tries common names or uses 'data')
    time_key : str, optional
        Key name for time data (default: None, generates from sampling_rate)
    sampling_rate : float, optional
        Sampling rate in Hz (required if time_key is None)
    signal_column : int, optional
        Column index in 'data' array if 'data' is 2D (default: None, uses first column or auto-detects from labels)
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
    
    try:
        mat_data = loadmat(file_path)
    except Exception as e:
        raise ValueError(f"Failed to load MATLAB file: {str(e)}")
    
    # Remove MATLAB metadata keys
    mat_data_clean = {k: v for k, v in mat_data.items() if not k.startswith('__')}
    
    if len(mat_data_clean) == 0:
        raise ValueError("MATLAB file contains no data")
    
    # Check for labels
    labels = None
    if 'labels' in mat_data_clean:
        labels_data = mat_data_clean['labels']
        # Handle different label formats (cell array, string array, etc.)
        if isinstance(labels_data, np.ndarray):
            # Try to extract labels from cell array or structured array
            try:
                if labels_data.dtype.names:
                    # Structured array
                    labels = [str(labels_data[name][0]) if len(labels_data[name]) > 0 else '' 
                             for name in labels_data.dtype.names]
                else:
                    # Handle MATLAB cell arrays (nested numpy arrays)
                    labels_flat = labels_data.flatten()
                    labels = []
                    for item in labels_flat:
                        if isinstance(item, np.ndarray):
                            # Cell array element - try to extract string
                            if item.dtype.kind == 'U' or item.dtype.kind == 'S':
                                # String array
                                label_str = ''.join(item.astype(str).flatten())
                            elif item.size == 1:
                                # Single element - try to convert
                                try:
                                    label_str = str(item.item())
                                except:
                                    label_str = str(item)
                            elif item.size == 0:
                                label_str = ''
                            else:
                                # Multiple elements - join them
                                try:
                                    label_str = ' '.join([str(x) for x in item.flatten()])
                                except:
                                    label_str = str(item)
                            labels.append(label_str.strip())
                        else:
                            labels.append(str(item).strip())
            except Exception as e:
                # If extraction fails, try simpler approach
                try:
                    # Try direct string conversion
                    if labels_data.dtype.kind == 'U' or labels_data.dtype.kind == 'S':
                        labels = [str(item).strip() for item in labels_data.flatten()]
                    else:
                        labels = [str(item).strip() for item in labels_data.flatten()]
                except:
                    labels = None
    
    # Find signal data
    if signal_key is not None:
        if signal_key not in mat_data_clean:
            raise ValueError(f"Signal key '{signal_key}' not found in file. Available keys: {list(mat_data_clean.keys())}")
        data_array = np.array(mat_data_clean[signal_key])
    else:
        # Check if 'data' key exists (common in MATLAB files)
        if 'data' in mat_data_clean:
            data_array = np.array(mat_data_clean['data'])
        else:
            # Try common signal names
            common_names = ['signal', 'ecg', 'hr', 'y', 'values']
            data_array = None
            for name in common_names:
                if name in mat_data_clean:
                    data_array = np.array(mat_data_clean[name])
                    signal_key = name
                    break
            
            if data_array is None:
                # Use the first numeric array
                for key, value in mat_data_clean.items():
                    if isinstance(value, np.ndarray) and value.size > 1:
                        data_array = value
                        signal_key = key
                        break
            
            if data_array is None:
                raise ValueError(f"Could not find signal data. Available keys: {list(mat_data_clean.keys())}")
    
    # Handle 2D data array (multiple channels/columns)
    if len(data_array.shape) == 2:
        if signal_column is not None:
            if signal_column >= data_array.shape[1]:
                raise ValueError(f"Column {signal_column} does not exist. Data has {data_array.shape[1]} columns.")
            hr = data_array[:, signal_column].flatten()
        elif labels is not None and len(labels) > 0:
            # Try to find ECG column from labels (case-insensitive search)
            ecg_column = None
            label_upper = [str(l).upper() for l in labels]
            
            # Search for ECG in labels (check various patterns)
            for i, label_upper_str in enumerate(label_upper):
                if label_upper_str and ('ECG' in label_upper_str):
                    ecg_column = i
                    break
            
            # If found and valid, use it
            if ecg_column is not None and ecg_column < data_array.shape[1]:
                hr = data_array[:, ecg_column].flatten()
            else:
                # Labels found but no clear ECG match, or mismatch in length
                # Show selection dialog
                if len(labels) == data_array.shape[1]:
                    # Labels match columns - show selection dialog
                    raise ColumnSelectionRequiredError(
                        f"Multiple columns found. Please select ECG column.",
                        available_columns=list(range(data_array.shape[1])),
                        labels=labels
                    )
                else:
                    # Labels don't match column count - still show dialog but with column numbers
                    raise ColumnSelectionRequiredError(
                        f"Data has {data_array.shape[1]} columns. Please select ECG column.",
                        available_columns=list(range(data_array.shape[1])),
                        labels=None  # Don't show labels if count doesn't match
                    )
        else:
            # No labels, no column specified - need user input
            if data_array.shape[1] > 1:
                raise ColumnSelectionRequiredError(
                    f"Data has {data_array.shape[1]} columns. Please select which column contains ECG signal.",
                    available_columns=list(range(data_array.shape[1])),
                    labels=None
                )
            else:
                hr = data_array[:, 0].flatten()
    else:
        # 1D array
        hr = data_array.flatten()
    
    # Find or generate time data
    if time_key is not None:
        if time_key not in mat_data_clean:
            raise ValueError(f"Time key '{time_key}' not found in file")
        hr_ts = np.array(mat_data_clean[time_key]).flatten()
    else:
        if sampling_rate is None:
            raise SamplingRateRequiredError("Sampling rate is required for this MATLAB file. Please provide sampling_rate parameter.")
        hr_ts = np.arange(len(hr)) / sampling_rate
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(hr) | np.isnan(hr_ts))
    hr = hr[valid_mask]
    hr_ts = hr_ts[valid_mask]
    
    if len(hr) == 0:
        raise ValueError("No valid data points found in file")
    
    # Downsample
    if downsample_factor > 1:
        hr_ts, hr = downsample(hr_ts, hr, downsample_factor)
    
    # Smooth the data (only if we have enough points)
    if len(hr) > 21:
        window_length = min(21, len(hr) if len(hr) % 2 == 1 else len(hr) - 1)
        hr = savgol_filter(hr, window_length, 2)
    elif len(hr) > 5:
        window_length = len(hr) if len(hr) % 2 == 1 else len(hr) - 1
        hr = savgol_filter(hr, window_length, min(2, window_length // 2))
    
    return hr, hr_ts


def load_wav_file(file_path, channel=0, downsample_factor=10):
    """
    Load ECG signal from WAV audio file.
    
    Parameters:
    -----------
    file_path : str
        Path to .wav file
    channel : int, optional
        Audio channel to use (default: 0)
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
    
    try:
        sampling_rate, audio_data = wavfile.read(file_path)
    except Exception as e:
        raise ValueError(f"Failed to load WAV file: {str(e)}")
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        if channel >= audio_data.shape[1]:
            raise ValueError(f"Channel {channel} does not exist. File has {audio_data.shape[1]} channels.")
        hr = audio_data[:, channel].astype(float)
    else:
        hr = audio_data.astype(float)
    
    # Normalize to typical ECG range (optional, but helpful)
    hr = hr / np.max(np.abs(hr)) if np.max(np.abs(hr)) > 0 else hr
    
    # Generate time stamps
    hr_ts = np.arange(len(hr)) / sampling_rate
    
    # Downsample
    if downsample_factor > 1:
        hr_ts, hr = downsample(hr_ts, hr, downsample_factor)
    
    # Smooth the data (only if we have enough points)
    if len(hr) > 21:
        window_length = min(21, len(hr) if len(hr) % 2 == 1 else len(hr) - 1)
        hr = savgol_filter(hr, window_length, 2)
    elif len(hr) > 5:
        window_length = len(hr) if len(hr) % 2 == 1 else len(hr) - 1
        hr = savgol_filter(hr, window_length, min(2, window_length // 2))
    
    return hr, hr_ts


def _extract_open_ephys_metadata(dat_file_path):
    """
    Extract Open Ephys metadata from structure.oebin if available.

    Returns:
    --------
    dict
        Metadata containing sample_rate, num_channels, labels, and bit_volts.
        Returns empty dict if metadata cannot be found or parsed.
    """
    dat_dir = os.path.dirname(os.path.abspath(dat_file_path))

    # Typical Open Ephys layout places structure.oebin a few levels above continuous.dat.
    candidate_paths = []
    current_dir = dat_dir
    for _ in range(6):
        candidate_paths.append(os.path.join(current_dir, "structure.oebin"))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    oebin_path = None
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            oebin_path = candidate
            break

    if oebin_path is None:
        return {}

    try:
        with open(oebin_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return {}

    continuous_entries = metadata.get("continuous", [])
    if not isinstance(continuous_entries, list) or len(continuous_entries) == 0:
        return {}

    # Try to match by folder name first.
    dat_path_norm = os.path.normpath(os.path.abspath(dat_file_path)).lower()
    matched_entry = None
    for entry in continuous_entries:
        folder_name = entry.get("folder_name", "")
        if folder_name and folder_name.lower() in dat_path_norm:
            matched_entry = entry
            break

    # Fallback to first continuous entry.
    if matched_entry is None:
        matched_entry = continuous_entries[0]

    channels = matched_entry.get("channels", [])
    labels = []
    bit_volts = []
    for i, channel_info in enumerate(channels):
        channel_name = channel_info.get("channel_name") or channel_info.get("name") or f"Channel {i}"
        labels.append(str(channel_name))
        bit_volts_value = channel_info.get("bit_volts")
        try:
            bit_volts.append(float(bit_volts_value) if bit_volts_value is not None else None)
        except (TypeError, ValueError):
            bit_volts.append(None)

    sample_rate = matched_entry.get("sample_rate")
    num_channels = matched_entry.get("num_channels")
    if num_channels is None and len(channels) > 0:
        num_channels = len(channels)

    result = {}
    try:
        if sample_rate is not None:
            result["sample_rate"] = float(sample_rate)
    except (TypeError, ValueError):
        pass

    try:
        if num_channels is not None:
            result["num_channels"] = int(num_channels)
    except (TypeError, ValueError):
        pass

    if labels:
        result["labels"] = labels
    if bit_volts:
        result["bit_volts"] = bit_volts

    return result


def load_open_ephys_dat_file(file_path, channel_index=None, signal_column=None, num_channels=None,
                             sampling_rate=None, downsample_factor=10):
    """
    Load ECG signal from Open Ephys .dat file.

    Parameters:
    -----------
    file_path : str
        Path to .dat file
    channel_index : int, optional
        Channel index containing ECG in interleaved data
    signal_column : int, optional
        Alias of channel_index for compatibility with generic GUI loader flow
    num_channels : int, optional
        Number of interleaved channels
    sampling_rate : float, optional
        Sampling rate in Hz (auto-detected from structure.oebin when available)
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

    metadata = _extract_open_ephys_metadata(file_path)

    if sampling_rate is None:
        sampling_rate = metadata.get("sample_rate")
    if num_channels is None:
        num_channels = metadata.get("num_channels")

    labels = metadata.get("labels")
    bit_volts = metadata.get("bit_volts")

    # Open Ephys continuous.dat uses int16 little-endian samples.
    raw_data = np.fromfile(file_path, dtype="<i2")
    if raw_data.size == 0:
        raise ValueError("DAT file is empty or could not be read")

    if num_channels is None:
        # Fallback: assume single channel when metadata is unavailable.
        num_channels = 1

    if num_channels < 1:
        raise ValueError("num_channels must be >= 1")

    # Keep compatibility with generic loader parameters used by GUI retry dialogs.
    if channel_index is None and signal_column is not None:
        channel_index = signal_column

    if num_channels > 1 and channel_index is None:
        raise ColumnSelectionRequiredError(
            f"Open Ephys DAT appears to contain {num_channels} channels. Please select ECG channel.",
            available_columns=list(range(num_channels)),
            labels=labels if labels and len(labels) == num_channels else None
        )

    if channel_index is None:
        channel_index = 0

    if channel_index < 0 or channel_index >= num_channels:
        raise ValueError(f"channel_index must be in [0, {num_channels - 1}]")

    if num_channels == 1:
        hr = raw_data.astype(float)
    else:
        usable_samples = (raw_data.size // num_channels) * num_channels
        if usable_samples == 0:
            raise ValueError("DAT file does not contain enough samples for declared channel count")
        if usable_samples < raw_data.size:
            raw_data = raw_data[:usable_samples]
        reshaped_data = raw_data.reshape(-1, num_channels)
        hr = reshaped_data[:, channel_index].astype(float)

    # Convert to physical units when Open Ephys bit_volts metadata is available.
    if bit_volts and channel_index < len(bit_volts) and bit_volts[channel_index] is not None:
        hr = hr * bit_volts[channel_index]

    if sampling_rate is None:
        raise SamplingRateRequiredError(
            "Sampling rate is required for DAT file. Could not infer from structure.oebin; please provide sampling_rate."
        )

    hr_ts = np.arange(len(hr)) / float(sampling_rate)

    # Downsample
    if downsample_factor > 1:
        hr_ts, hr = downsample(hr_ts, hr, downsample_factor)

    # Smooth the data (only if we have enough points)
    if len(hr) > 21:
        window_length = min(21, len(hr) if len(hr) % 2 == 1 else len(hr) - 1)
        hr = savgol_filter(hr, window_length, 2)
    elif len(hr) > 5:
        window_length = len(hr) if len(hr) % 2 == 1 else len(hr) - 1
        hr = savgol_filter(hr, window_length, min(2, window_length // 2))

    return hr, hr_ts


def load_ecg_file(file_path, **kwargs):
    """
    Generic ECG file loader that auto-detects file format.
    
    Parameters:
    -----------
    file_path : str
        Path to ECG file
    **kwargs : dict
        Additional arguments passed to format-specific loaders
        
    Returns:
    --------
    hr : array
        Heart rate signal
    hr_ts : array
        Time stamps for heart rate signal
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file extension
    _, ext = os.path.splitext(file_path.lower())
    
    # Route to appropriate loader
    if ext == '.abf':
        return load_abf_file(file_path, **kwargs)
    elif ext in ['.csv', '.txt']:
        return load_csv_file(file_path, **kwargs)
    elif ext == '.mat':
        return load_mat_file(file_path, **kwargs)
    elif ext == '.wav':
        return load_wav_file(file_path, **kwargs)
    elif ext == '.dat':
        return load_open_ephys_dat_file(file_path, **kwargs)
    else:
        # Try CSV as fallback for unknown extensions
        try:
            return load_csv_file(file_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: .abf, .csv, .txt, .mat, .wav, .dat. Error: {str(e)}")





