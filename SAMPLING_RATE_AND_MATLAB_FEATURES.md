# Sampling Rate Input and Enhanced MATLAB Support

## Overview

The HR Detection GUI now includes interactive dialogs for user input when required information is missing from files, and enhanced MATLAB file support with automatic label reading and column selection.

## New Features

### 1. Sampling Rate Input Dialog

When loading files that don't contain time information (CSV, TXT, MATLAB files without time data), a dialog box will automatically appear asking for the sampling rate.

**Features:**
- Appears automatically when sampling rate is needed
- Default value: 1000 Hz (user can change)
- Input validation (must be positive number)
- Cancel option to abort file loading
- Enter key support for quick input

**When it appears:**
- CSV/TXT files without time column
- MATLAB files without time data
- Any format where time information cannot be determined

### 2. MATLAB Column Selection Dialog

When loading MATLAB files with multiple columns, a dialog appears to let you select which column contains the ECG signal.

**Features:**
- Shows all available columns
- Displays labels if available (from 'labels' key in MATLAB file)
- Auto-detects ECG column from labels if possible
- Double-click or OK button to select
- Cancel option to abort file loading

**When it appears:**
- MATLAB files with 2D 'data' array (multiple columns)
- When labels don't clearly identify ECG column
- When no labels are present but multiple columns exist

### 3. Enhanced MATLAB Label Reading

The MATLAB loader now:
- Automatically reads 'labels' from MATLAB files
- Searches for ECG-related labels (case-insensitive)
- Auto-selects ECG column if found in labels
- Handles various MATLAB label formats (cell arrays, string arrays, etc.)

**Label formats supported:**
- MATLAB cell arrays
- String arrays
- Structured arrays
- Nested numpy arrays (from scipy.io.loadmat)

**Example:**
If your MATLAB file has:
```matlab
data = [column1, column2, column3, column4, column5];
labels = {'Signal1', 'Signal2', 'ECG, Y, RSPEC-R', 'Signal4', 'Signal5'};
```

The loader will:
1. Read the labels
2. Detect that column 2 (index 2) contains 'ECG'
3. Automatically use that column
4. If detection fails, show selection dialog with labels displayed

## Usage Examples

### Example 1: CSV File Without Time Column

1. Click "Load ECG File"
2. Select a CSV file with only signal data (no time column)
3. Dialog appears: "Sampling rate not found in file. Please enter the sampling rate (Hz):"
4. Enter sampling rate (e.g., 1000)
5. Click OK
6. File loads successfully

### Example 2: MATLAB File with Labels

1. Click "Load ECG File"
2. Select a MATLAB file with 'data' and 'labels' keys
3. If labels contain 'ECG', column is auto-selected
4. If multiple columns and no clear ECG match, selection dialog appears:
   - Shows: "Column 0: Signal1"
   - Shows: "Column 1: Signal2"
   - Shows: "Column 2: ECG, Y, RSPEC-R" ← Select this
   - Shows: "Column 3: Signal4"
   - Shows: "Column 4: Signal5"
5. Select column 2 (or double-click it)
6. If sampling rate needed, sampling rate dialog appears
7. Enter sampling rate
8. File loads successfully

### Example 3: MATLAB File Without Labels

1. Click "Load ECG File"
2. Select a MATLAB file with 2D 'data' array but no labels
3. Selection dialog appears:
   - Shows: "Column 0"
   - Shows: "Column 1"
   - Shows: "Column 2"
   - Shows: "Column 3"
   - Shows: "Column 4"
4. Select the column containing ECG (e.g., column 2)
5. Sampling rate dialog appears
6. Enter sampling rate
7. File loads successfully

## Technical Details

### Exception Handling

The loaders use custom exceptions to signal when user input is needed:

- `SamplingRateRequiredError`: Raised when sampling rate is needed
- `ColumnSelectionRequiredError`: Raised when column selection is needed (includes available columns and labels)

### Dialog Implementation

- **Sampling Rate Dialog**: Modal dialog with entry field and validation
- **Column Selection Dialog**: Modal dialog with scrollable listbox showing columns and labels

### Retry Logic

The `load_file()` method uses a retry loop:
1. Attempts to load file
2. If exception occurs (sampling rate or column needed), shows dialog
3. Retries with user-provided parameters
4. Maximum 5 retries to prevent infinite loops

## Backward Compatibility

- All existing functionality remains unchanged
- ABF files work exactly as before (no dialogs needed)
- Files with complete information load without dialogs
- Only files missing required information trigger dialogs

## Error Handling

- Clear error messages if user cancels dialogs
- Validation of sampling rate input (must be positive)
- Column index validation
- Graceful handling of malformed labels

## Future Enhancements

Potential improvements:
- Remember last used sampling rate
- Support for time column selection in CSV files
- Batch file loading with same parameters
- Save/load parameter presets


