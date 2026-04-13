# Multi-Format ECG File Support

## Overview

The HR Detection GUI now supports multiple ECG file formats beyond ABF files. The application automatically detects the file format and loads the data accordingly.

## Supported Formats

### 1. ABF Files (Axon Binary Format)
- **Extension**: `.abf`
- **Description**: Original format, fully supported with channel selection
- **Usage**: Works exactly as before

### 2. CSV/TXT Files
- **Extensions**: `.csv`, `.txt`
- **Description**: Comma, tab, or space-separated text files
- **Features**:
  - Auto-detects delimiter (comma, tab, or whitespace)
  - Supports header rows
  - Configurable signal and time columns
- **Requirements**:
  - Signal data in one column
  - Either time stamps in another column OR sampling rate must be provided
- **Example formats**:
  ```
  time,signal
  0.0,1.23
  0.001,1.45
  0.002,1.67
  ```
  Or:
  ```
  1.23
  1.45
  1.67
  ```
  (with sampling_rate parameter)

### 3. MATLAB Files
- **Extension**: `.mat`
- **Description**: MATLAB binary format files
- **Features**:
  - Auto-detects signal data (tries common names: 'signal', 'ecg', 'hr', 'data', 'y', 'values')
  - Falls back to first numeric array if no common name found
  - Supports custom signal and time keys
- **Requirements**:
  - Signal data as a numeric array
  - Either time data as a separate array OR sampling rate must be provided

### 4. WAV Files
- **Extension**: `.wav`
- **Description**: Audio format (sometimes used for ECG recordings)
- **Features**:
  - Extracts audio data as ECG signal
  - Supports multi-channel audio (selectable channel)
  - Auto-normalizes signal
  - Uses file's sampling rate for time stamps

## Usage

### Basic Usage (GUI)
1. Click "Load ECG File" button
2. Select your file (any supported format)
3. The application automatically detects the format and loads the data

### Programmatic Usage

```python
from hr_detection_gui.hr_detection import load_ecg_file

# Auto-detect format
hr, hr_ts = load_ecg_file("data.csv", sampling_rate=1000)

# For CSV with time column
hr, hr_ts = load_ecg_file("data.csv", time_column=0, signal_column=1)

# For MATLAB files
hr, hr_ts = load_ecg_file("data.mat", signal_key="ecg", sampling_rate=1000)

# For WAV files
hr, hr_ts = load_ecg_file("data.wav", channel=0)
```

## Format-Specific Parameters

### CSV/TXT Files
- `signal_column`: Column index for signal data (default: 0)
- `time_column`: Column index for time stamps (optional)
- `sampling_rate`: Sampling rate in Hz (required if time_column not provided)
- `delimiter`: Delimiter character (auto-detected if not specified)
- `has_header`: Whether file has header row (default: False)

### MATLAB Files
- `signal_key`: Key name for signal data (auto-detected if not specified)
- `time_key`: Key name for time data (optional)
- `sampling_rate`: Sampling rate in Hz (required if time_key not provided)

### WAV Files
- `channel`: Audio channel to use (default: 0)

### ABF Files
- All original parameters still supported (channel_name, channel_index, etc.)

## Implementation Details

### Auto-Detection
The `load_ecg_file()` function automatically detects the file format based on the file extension:
- `.abf` → ABF loader
- `.csv`, `.txt` → CSV/TXT loader
- `.mat` → MATLAB loader
- `.wav` → WAV loader
- Unknown extensions → Tries CSV loader as fallback

### Data Processing
All formats go through the same post-processing:
1. Downsampling (default factor: 10)
2. Smoothing with Savitzky-Golay filter
3. Time stamp generation/validation

### Error Handling
- Clear error messages for unsupported formats
- Validation of data columns/keys
- Handling of missing or invalid data points

## Backward Compatibility

- The original `load_abf_file()` function is still available and unchanged
- Existing code using `load_abf_file()` will continue to work
- The GUI now uses `load_ecg_file()` which internally calls `load_abf_file()` for ABF files

## Testing

To test the new functionality:
1. Create test files in different formats
2. Load them through the GUI
3. Verify that peak detection and analysis work correctly

## Notes

- CSV/TXT files should contain numeric data only
- MATLAB files should contain numeric arrays (not cell arrays or structures)
- WAV files are normalized to typical ECG range
- All formats require either explicit time data or a sampling rate

## Future Enhancements

Potential additions:
- EDF (European Data Format) support
- HDF5 support
- Excel file support (already have openpyxl)
- More robust header detection for CSV files
- Support for multi-channel selection in GUI


