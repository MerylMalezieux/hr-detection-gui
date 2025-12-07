
# Heart Rate Detection and Analysis GUI

A Python GUI application for detecting and analyzing heart rate from ABF (Axon Binary Format) files. This tool provides an interactive interface for peak detection, manual editing, and comprehensive HRV (Heart Rate Variability) analysis.

## Features

- **File Loading**: Load ABF files (pClamp) from folder
- **Interactive Peak Detection**: Adjustable parameters for threshold-based peak detection based on the find_peaks function of scipy 
  - Threshold adjustment
  - Refractory period control
  - Minimum duration filtering
  - Highpass filtering option
  - Absolute value option
- **Manual Editing**: Interactive event editor
  - Left-click to add peaks
  - Right-click to remove peaks
  - Real-time visualization
- **BPM Calculation**: Instantaneous heart rate (BPM) computation
- **HRV Analysis**: Comprehensive heart rate variability metrics
  - RMSSD (Root Mean Square of Successive Differences)
  - SDNN (Standard Deviation of NN intervals)
  - pNN50 (Percentage of NN intervals differing by >50ms)
  - Mean, median, min, max RR intervals
  - Additional statistical metrics
- **Data Export**: Save results to NumPy (.npy) dictionnary format for further analysis

## Installation

### Option 1: Using the Setup Script (Recommended)

**Windows (PowerShell):**
```powershell
.\setup_env.ps1
```

**Windows (Command Prompt):**
```cmd
setup_env.bat
```

### Option 2: Manual Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:

   **Windows (PowerShell):**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   **Windows (Command Prompt):**
   ```cmd
   venv\Scripts\activate.bat
   ```

   **Linux/Mac:**
   ```bash
   source venv/bin/activate
   ```

3. Upgrade pip:
```bash
python -m pip install --upgrade pip
```

4. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Activate the virtual environment** (if not already activated):
   **Windows (PowerShell):**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   **Windows (Command Prompt):**
   ```cmd
   venv\Scripts\activate.bat
   ```

2. **Run the application**:
   ```bash
   python main.py
   ```

### Workflow

1. **Load File**: Click "Load ABF File" and select your ABF file
2. **Adjust Parameters**: Use the sliders to adjust detection parameters
   - **Threshold**: Sensitivity of peak detection
   - **Refractory Period**: Minimum time between peaks (ms)
   - **Min Duration**: Minimum duration of detected events
   - **Highpass Filter**: Optional highpass filter frequency (Hz)
   - **Use Absolute Value**: Toggle to use absolute value of signal
3. **Detect Peaks**: Click "Detect Peaks" to run the detection algorithm
4. **Manual Editing**: 
   - Left-click on the plot to add a peak
   - Right-click near a peak to remove it
5. **Compute Metrics**: Click "Compute BPM/HRV" to calculate heart rate and variability metrics
6. **Save Results**: Click "Save Results" to export data to .npy format

## Output Format

The saved .npy file contains a dictionary with the following keys:

- `hr`: Raw heart rate signal
- `hr_ts`: Time stamps for the signal
- `R_start`: Detected peak times
- `hr_sp_ind`: Indices of detected peaks
- `inst_bpm`: Instantaneous BPM
- `inst_bpm_ts`: Time stamps for BPM
- `bpm_to_max`: BPM normalized to maximum (0-100)
- `hrv_metrics`: Dictionary of all HRV metrics
- `rmssd`: RMSSD values
- `rmssd_to_max`: RMSSD normalized to maximum
- `sdnn`: SDNN value
- `pnn50`: pNN50 percentage
- `mean_hr`: Mean heart rate
- `mean_rr`, `median_rr`, `min_rr`, `max_rr`, `std_rr`: RR interval statistics
- `source_file`: Path to original ABF file
- `detection_params`: Parameters used for detection

## Project Structure

```
hr-detection-gui/
├── hr_detection_gui/
│   ├── __init__.py
│   ├── hr_detection.py      # Core HR detection functions
│   ├── hrv_analysis.py       # HRV metrics calculation
│   ├── event_editor.py       # Interactive event editing
│   └── main_gui.py          # Main GUI application
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
├── setup_env.bat           # Windows batch setup script
├── setup_env.ps1           # PowerShell setup script
└── README.md               # This file
```

## Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scipy**: Signal processing and filtering
- **matplotlib**: Plotting and visualization
- **neo**: ABF file reading
- **openpyxl**: Excel file support (if needed)

## Notes

- The application automatically downsamples large datasets for display performance
- Peak detection uses derivative-based thresholding
- BPM calculation uses a rolling window approach
- HRV metrics follow standard physiological analysis methods

## Deactivating the Virtual Environment

When you're done working, you can deactivate the virtual environment:
```bash
deactivate
```


## Author

Meryl Malezieux
