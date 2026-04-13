# BPM Calculation Fix

## Problem

The BPM calculation was showing incorrect values (800-1000 BPM instead of ~60 BPM for human ECG) because it used a fixed sample-based window instead of a time-based window.

## Root Cause

The original `find_inst_bpm` function had two issues:

1. **Fixed sample window**: It counted beats in a window of 200 samples, not a time-based window
2. **Hardcoded multiplier**: It multiplied by 600, which assumed a specific sampling rate

### Original Code (Incorrect)
```python
# Count beats in 200 timepoints
inst_bpm[j] = sp_times[(sp_times > ts[j]) & (sp_times < ts[j+ 200])].size
# Multiply by 60 to get bpm
inst_bpm[j] = inst_bpm[j] * 600
```

**Problem**: This assumes:
- A specific sampling rate (the 200 samples and 600 multiplier were calibrated for one rate)
- Time stamps are in a specific format
- The window size is appropriate for all sampling rates

## Solution

The fix uses a **time-based window** (1 second) instead of a sample-based window:

### Fixed Code (Correct)
```python
# Use a 1-second window for BPM calculation
window_duration = 1.0  # seconds

# Count beats in the time window [current_time, window_end_time]
beats_in_window = np.sum((sp_times > current_time) & (sp_times <= window_end_time))

# Convert to BPM: beats per second * 60 seconds per minute
inst_bpm[j] = (beats_in_window / window_duration) * 60.0
```

**Benefits**:
- Works correctly regardless of sampling rate
- Uses actual time stamps (in seconds)
- Proper conversion: beats/second × 60 = BPM
- Handles edge cases (beginning and end of signal)

## How It Works Now

1. For each time point in the signal:
   - Define a 1-second window starting at that time point
   - Count how many peaks (beats) occur within that window
   - Convert to BPM: `(beats / 1 second) × 60 = BPM`

2. Example:
   - If 1 beat occurs in a 1-second window: `(1 / 1) × 60 = 60 BPM` ✓
   - If 2 beats occur in a 1-second window: `(2 / 1) × 60 = 120 BPM` ✓

## Testing

After this fix:
- Human ECG (~60 BPM) should show correct values
- The calculation is independent of sampling rate
- Works correctly for any file format (ABF, CSV, MATLAB, WAV)

## Additional Improvements

- Better handling of edge cases (beginning/end of signal)
- Adaptive filtering based on signal length
- More robust smoothing for short signals


