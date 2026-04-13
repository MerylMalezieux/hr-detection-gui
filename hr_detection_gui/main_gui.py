# -*- coding: utf-8 -*-
"""
Main GUI Application for Heart Rate Detection
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os

from .hr_detection import (load_ecg_file, find_hr_peaks, find_inst_bpm,
                          SamplingRateRequiredError, ColumnSelectionRequiredError)
from .hrv_analysis import calculate_all_hrv_metrics, clean_bpm_signal
from .event_editor import EventEditor


class HRDetectionGUI:
    """Main GUI application for heart rate detection and analysis."""
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Heart Rate Detection and Analysis")
        self.root.geometry("1400x900")
        
        # Data storage
        self.hr = None
        self.hr_ts = None
        self.hr_sp_ind = None
        self.hr_sp_times = None  # Original detected peaks (before manual editing)
        self.inst_bpm = None  # Cleaned BPM signal (after outlier removal, for saving)
        self.inst_bpm_original = None  # BPM from original detected peaks (for display/comparison)
        self.inst_bpm_from_cleaned_peaks = None  # BPM computed from manually edited peaks (before signal cleaning)
        self.bpm_to_max = None  # Cleaned BPM normalized (for saving)
        self.bpm_to_max_original = None  # Original BPM normalized (for display)
        self.bpm_to_max_from_cleaned_peaks = None  # BPM from cleaned peaks normalized (before signal cleaning)
        self.hrv_metrics = None
        self.file_path = None
        self.mouse_id = ""
        self.hr_highpass = np.array([])
        self.bpm_plot_needs_update = False  # Flag to track if BPM plot needs updating
        self.bpm_lines = {}  # Cache for BPM plot lines
        self.bpm_window = None  # Separate window for BPM plot
        self.bpm_fig = None  # Figure for BPM window
        self.bpm_ax = None  # Axes for BPM window
        self.bpm_canvas = None  # Canvas for BPM window
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Top frame: File loading and parameters
        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # File loading section
        file_frame = ttk.LabelFrame(top_frame, text="File Loading", padding="5")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(file_frame, text="Load ECG File", command=self.load_file).grid(row=0, column=0, padx=5)
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.grid(row=0, column=1, padx=5)
        
        # Mouse ID input
        ttk.Label(file_frame, text="Mouse ID:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=(5, 0))
        self.mouse_id_entry = ttk.Entry(file_frame, width=15)
        self.mouse_id_entry.grid(row=1, column=1, padx=5, pady=(5, 0), sticky=tk.W)
        
        # Parameters section
        params_frame = ttk.LabelFrame(top_frame, text="Detection Parameters", padding="5")
        params_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Threshold
        ttk.Label(params_frame, text="Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.thresh_var = tk.DoubleVar(value=0.04)
        ttk.Scale(params_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL, 
                 variable=self.thresh_var, length=150).grid(row=0, column=1, padx=5)
        self.thresh_entry = ttk.Entry(params_frame, width=8)
        self.thresh_entry.insert(0, "0.04")
        self.thresh_entry.grid(row=0, column=2, padx=5)
        self.thresh_entry.bind('<Return>', lambda e: self.update_thresh_from_entry())
        self.thresh_entry.bind('<FocusOut>', lambda e: self.update_thresh_from_entry())
        self.thresh_var.trace('w', lambda *args: self.update_thresh_entry())
        
        # Refractory period
        ttk.Label(params_frame, text="Refractory (ms):").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.refrac_var = tk.DoubleVar(value=30.0)
        ttk.Scale(params_frame, from_=10, to=1000, orient=tk.HORIZONTAL,
                 variable=self.refrac_var, length=150).grid(row=1, column=1, padx=5)
        self.refrac_entry = ttk.Entry(params_frame, width=8)
        self.refrac_entry.insert(0, "30.0")
        self.refrac_entry.grid(row=1, column=2, padx=5)
        self.refrac_entry.bind('<Return>', lambda e: self.update_refrac_from_entry())
        self.refrac_entry.bind('<FocusOut>', lambda e: self.update_refrac_from_entry())
        self.refrac_var.trace('w', lambda *args: self.update_refrac_entry())
        
        # Min duration
        ttk.Label(params_frame, text="Min Duration:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.min_dur_var = tk.IntVar(value=1)
        ttk.Scale(params_frame, from_=1, to=50, orient=tk.HORIZONTAL,
                 variable=self.min_dur_var, length=150).grid(row=2, column=1, padx=5)
        self.min_dur_entry = ttk.Entry(params_frame, width=8)
        self.min_dur_entry.insert(0, "1")
        self.min_dur_entry.grid(row=2, column=2, padx=5)
        self.min_dur_entry.bind('<Return>', lambda e: self.update_min_dur_from_entry())
        self.min_dur_entry.bind('<FocusOut>', lambda e: self.update_min_dur_from_entry())
        self.min_dur_var.trace('w', lambda *args: self.update_min_dur_entry())
        
        # Highpass filter
        ttk.Label(params_frame, text="Highpass (Hz):").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.highpass_var = tk.DoubleVar(value=0.0)
        ttk.Scale(params_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                 variable=self.highpass_var, length=150).grid(row=3, column=1, padx=5)
        self.highpass_label = ttk.Label(params_frame, text="0.0")
        self.highpass_label.grid(row=3, column=2, padx=5)
        self.highpass_var.trace('w', lambda *args: self.update_highpass_label())
        
        # Use absolute value
        self.use_abs_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Use Absolute Value", 
                       variable=self.use_abs_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5)
        
        # Buttons section
        buttons_frame = ttk.Frame(top_frame)
        buttons_frame.grid(row=0, column=2, sticky=(tk.W, tk.E))
        
        ttk.Button(buttons_frame, text="Detect Peaks", command=self.detect_peaks).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(buttons_frame, text="Compute BPM/HRV", command=self.compute_metrics).grid(row=1, column=0, padx=5, pady=2)
        ttk.Button(buttons_frame, text="Show BPM Plot", command=self.show_bpm_window).grid(row=2, column=0, padx=5, pady=2)
        ttk.Button(buttons_frame, text="Save Results", command=self.save_results).grid(row=3, column=0, padx=5, pady=2)
        
        # Main plot area (only HR signal, BPM will be in separate window)
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure with single subplot for HR signal
        self.fig = Figure(figsize=(12, 6))
        # Single subplot for HR signal and peaks
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Heart Rate Signal and Detected Peaks')
        self.ax.grid(True, alpha=0.3)
        # Adjust spacing
        self.fig.tight_layout(pad=2.0)
        
        # Canvas frame for grid layout
        canvas_frame = ttk.Frame(plot_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Toolbar in separate frame that uses pack
        toolbar_frame = tk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Event editor (will be created when needed)
        self.event_editor = None
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def update_thresh_entry(self):
        """Update threshold entry from slider."""
        self.thresh_entry.delete(0, tk.END)
        self.thresh_entry.insert(0, f"{self.thresh_var.get():.3f}")
    
    def update_thresh_from_entry(self):
        """Update threshold slider from entry."""
        try:
            value = float(self.thresh_entry.get())
            if 0.01 <= value <= 1.0:
                self.thresh_var.set(value)
            else:
                self.thresh_entry.delete(0, tk.END)
                self.thresh_entry.insert(0, f"{self.thresh_var.get():.3f}")
        except ValueError:
            self.thresh_entry.delete(0, tk.END)
            self.thresh_entry.insert(0, f"{self.thresh_var.get():.3f}")
    
    def update_refrac_entry(self):
        """Update refractory period entry from slider."""
        self.refrac_entry.delete(0, tk.END)
        self.refrac_entry.insert(0, f"{self.refrac_var.get():.1f}")
    
    def update_refrac_from_entry(self):
        """Update refractory period slider from entry."""
        try:
            value = float(self.refrac_entry.get())
            if 10 <= value <= 1000:
                self.refrac_var.set(value)
            else:
                self.refrac_entry.delete(0, tk.END)
                self.refrac_entry.insert(0, f"{self.refrac_var.get():.1f}")
        except ValueError:
            self.refrac_entry.delete(0, tk.END)
            self.refrac_entry.insert(0, f"{self.refrac_var.get():.1f}")
    
    def update_min_dur_entry(self):
        """Update min duration entry from slider."""
        self.min_dur_entry.delete(0, tk.END)
        self.min_dur_entry.insert(0, f"{self.min_dur_var.get()}")
    
    def update_min_dur_from_entry(self):
        """Update min duration slider from entry."""
        try:
            value = int(self.min_dur_entry.get())
            if 1 <= value <= 50:
                self.min_dur_var.set(value)
            else:
                self.min_dur_entry.delete(0, tk.END)
                self.min_dur_entry.insert(0, f"{self.min_dur_var.get()}")
        except ValueError:
            self.min_dur_entry.delete(0, tk.END)
            self.min_dur_entry.insert(0, f"{self.min_dur_var.get()}")
    
    def update_highpass_label(self):
        """Update highpass label."""
        self.highpass_label.config(text=f"{self.highpass_var.get():.1f}")
    
    def ask_sampling_rate(self, default_value=1000.0):
        """
        Show dialog to ask user for sampling rate.
        
        Parameters:
        -----------
        default_value : float, optional
            Default sampling rate value (default: 1000.0)
            
        Returns:
        --------
        float or None
            Sampling rate in Hz, or None if cancelled
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Sampling Rate Required")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        result = {'value': None}
        
        # Label
        label = ttk.Label(dialog, text="Sampling rate not found in file.\nPlease enter the sampling rate (Hz):", 
                         justify=tk.CENTER)
        label.pack(pady=10)
        
        # Entry frame
        entry_frame = ttk.Frame(dialog)
        entry_frame.pack(pady=10)
        
        ttk.Label(entry_frame, text="Sampling Rate (Hz):").pack(side=tk.LEFT, padx=5)
        entry = ttk.Entry(entry_frame, width=15)
        entry.insert(0, str(default_value))
        entry.pack(side=tk.LEFT, padx=5)
        entry.focus()
        entry.select_range(0, tk.END)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def ok_clicked():
            try:
                value = float(entry.get())
                if value > 0:
                    result['value'] = value
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", "Sampling rate must be positive.")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number.")
        
        def cancel_clicked():
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=ok_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key
        entry.bind('<Return>', lambda e: ok_clicked())
        dialog.bind('<Escape>', lambda e: cancel_clicked())
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result['value']
    
    def ask_matlab_column(self, available_columns, labels=None):
        """
        Show dialog to ask user to select MATLAB column.
        
        Parameters:
        -----------
        available_columns : list
            List of column indices available
        labels : list, optional
            List of labels for each column (default: None)
            
        Returns:
        --------
        int or None
            Selected column index, or None if cancelled
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Select ECG Column")
        dialog.geometry("500x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        result = {'value': None}
        
        # Label
        if labels:
            label_text = "Multiple columns found. Please select the column containing ECG signal:"
        else:
            label_text = f"Data has {len(available_columns)} columns. Please select which column contains ECG signal:"
        
        label = ttk.Label(dialog, text=label_text, justify=tk.CENTER)
        label.pack(pady=10)
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(dialog)
        list_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=8)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Populate listbox
        for i, col_idx in enumerate(available_columns):
            if labels and i < len(labels):
                display_text = f"Column {col_idx}: {labels[i]}"
            else:
                display_text = f"Column {col_idx}"
            listbox.insert(tk.END, display_text)
        
        # Select first item by default
        if listbox.size() > 0:
            listbox.selection_set(0)
            listbox.activate(0)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def ok_clicked():
            selection = listbox.curselection()
            if selection:
                result['value'] = available_columns[selection[0]]
                dialog.destroy()
            else:
                messagebox.showwarning("Warning", "Please select a column.")
        
        def cancel_clicked():
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=ok_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT, padx=5)
        
        # Bind double-click
        listbox.bind('<Double-Button-1>', lambda e: ok_clicked())
        dialog.bind('<Escape>', lambda e: cancel_clicked())
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result['value']
    
    def load_file(self):
        """Load ECG file (supports multiple formats)."""
        file_path = filedialog.askopenfilename(
            title="Select ECG File",
            filetypes=[
                ("All Supported Formats", "*.abf;*.csv;*.txt;*.mat;*.wav"),
                ("ABF files", "*.abf"),
                ("CSV files", "*.csv"),
                ("TXT files", "*.txt"),
                ("MATLAB files", "*.mat"),
                ("WAV files", "*.wav"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.status_var.set("Loading file...")
            self.root.update()
            
            # Try to load file - may need user input for sampling rate or column selection
            load_params = {}
            max_retries = 5  # Prevent infinite loops
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Load file using generic loader (auto-detects format)
                    self.hr, self.hr_ts = load_ecg_file(file_path, **load_params)
                    self.file_path = file_path
                    break  # Success, exit retry loop
                    
                except SamplingRateRequiredError as e:
                    # Ask user for sampling rate
                    sampling_rate = self.ask_sampling_rate()
                    if sampling_rate is None:
                        # User cancelled
                        self.status_var.set("File loading cancelled")
                        return
                    load_params['sampling_rate'] = sampling_rate
                    retry_count += 1
                    
                except ColumnSelectionRequiredError as e:
                    # Ask user to select column
                    column = self.ask_matlab_column(e.available_columns, e.labels)
                    if column is None:
                        # User cancelled
                        self.status_var.set("File loading cancelled")
                        return
                    load_params['signal_column'] = column
                    # Also need sampling rate if not provided
                    if 'sampling_rate' not in load_params:
                        sampling_rate = self.ask_sampling_rate()
                        if sampling_rate is None:
                            self.status_var.set("File loading cancelled")
                            return
                        load_params['sampling_rate'] = sampling_rate
                    retry_count += 1
                    
                except Exception as e:
                    # Other error - re-raise it
                    raise
            
            if retry_count >= max_retries:
                raise RuntimeError("Maximum retry attempts reached. Please check file format and try again.")
            
            self.file_path = file_path
            
            # Update file label
            filename = os.path.basename(file_path)
            self.file_label.config(text=filename)
            
            # Reset detection results completely
            self.hr_sp_ind = None
            self.hr_sp_times = None
            self.inst_bpm = None
            self.inst_bpm_original = None
            self.inst_bpm_from_cleaned_peaks = None
            self.bpm_to_max = None
            self.bpm_to_max_original = None
            self.bpm_to_max_from_cleaned_peaks = None
            self.hrv_metrics = None
            self.hr_highpass = np.array([])
            
            # Clear/destroy event editor if it exists (new file = fresh start)
            if self.event_editor is not None:
                self.event_editor = None
            
            # Try to extract mouse ID from filename if possible
            # (e.g., if filename contains mouse ID pattern)
            # For now, leave it empty for user to enter
            
            # Plot signal (without any peaks since we cleared everything)
            self.plot_signal()
            
            self.status_var.set(f"File loaded: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.status_var.set("Error loading file")
    
    def show_bpm_window(self):
        """Show or create the BPM plot window."""
        try:
            window_exists = self.bpm_window is not None and self.bpm_window.winfo_exists()
        except:
            window_exists = False
        
        if not window_exists:
            # Create new window
            self.bpm_window = tk.Toplevel(self.root)
            self.bpm_window.title("BPM Plot - Original vs. Cleaned")
            self.bpm_window.geometry("1200x400")
            
            # Create figure and canvas
            self.bpm_fig = Figure(figsize=(12, 4))
            self.bpm_ax = self.bpm_fig.add_subplot(111)
            self.bpm_ax.set_xlabel('Time (s)')
            self.bpm_ax.set_ylabel('BPM')
            self.bpm_ax.set_title('BPM: Original Detection vs. Cleaned Peaks (with signal cleaning)')
            self.bpm_ax.grid(True, alpha=0.3)
            
            # Canvas
            self.bpm_canvas = FigureCanvasTkAgg(self.bpm_fig, self.bpm_window)
            self.bpm_canvas.draw()
            self.bpm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Toolbar
            bpm_toolbar = NavigationToolbar2Tk(self.bpm_canvas, self.bpm_window)
            bpm_toolbar.update()
            
            # Handle window close
            self.bpm_window.protocol("WM_DELETE_WINDOW", self.close_bpm_window)
        
        # Update the plot
        self.update_bpm_window()
        # Bring window to front
        self.bpm_window.lift()
        self.bpm_window.focus_force()
    
    def close_bpm_window(self):
        """Close the BPM window."""
        if self.bpm_window is not None:
            self.bpm_window.destroy()
            self.bpm_window = None
            self.bpm_fig = None
            self.bpm_ax = None
            self.bpm_canvas = None
    
    def update_bpm_window(self):
        """Update the BPM plot in the separate window."""
        try:
            if self.bpm_window is None or not self.bpm_window.winfo_exists():
                return
        except:
            return
        
        if self.hr_ts is None or len(self.hr_ts) == 0:
            return
        
        # Check if we have peaks
        has_peaks = False
        if self.event_editor is not None:
            current_events = self.event_editor.get_events()
            has_peaks = len(current_events) > 0
        elif self.hr_sp_times is not None:
            has_peaks = len(self.hr_sp_times) > 0
        
        if not has_peaks:
            return
        
        # Clear BPM axes
        self.bpm_ax.clear()
        
        # Ensure original BPM is computed (from original detection, before manual editing)
        if self.inst_bpm_original is None and self.hr_sp_times is not None and len(self.hr_sp_times) >= 2:
            self.inst_bpm_original = find_inst_bpm(self.hr, self.hr_sp_times, self.hr_ts)
        
        # Subsample BPM data for faster plotting (plot every Nth point)
        subsample_factor = max(1, len(self.hr_ts) // 10000)  # Limit to ~10k points
        
        # Plot original BPM (from original detection, before manual editing)
        if self.inst_bpm_original is not None:
            ts_subsampled = self.hr_ts[::subsample_factor]
            bpm_subsampled = self.inst_bpm_original[::subsample_factor]
            self.bpm_ax.plot(ts_subsampled, bpm_subsampled, 'orange', 
                          linewidth=1.5, label='BPM (from original peaks)', alpha=0.8)
        
        # Plot cleaned BPM if available (from cleaned peaks, after signal cleaning)
        if self.inst_bpm is not None:
            ts_subsampled = self.hr_ts[::subsample_factor]
            bpm_clean_subsampled = self.inst_bpm[::subsample_factor]
            self.bpm_ax.plot(ts_subsampled, bpm_clean_subsampled, 'g--', 
                          linewidth=2, label='BPM (from cleaned peaks, signal cleaned)', alpha=0.9)
        
        # Set labels
        self.bpm_ax.set_xlabel('Time (s)')
        self.bpm_ax.set_ylabel('BPM')
        self.bpm_ax.set_title('BPM: Original Detection vs. Cleaned Peaks (with signal cleaning)')
        self.bpm_ax.grid(True, alpha=0.3)
        self.bpm_ax.legend()
        
        # Auto-scale BPM axis to show both if available
        bpm_min = None
        bpm_max = None
        if self.inst_bpm_original is not None:
            valid_bpm = self.inst_bpm_original[~np.isnan(self.inst_bpm_original)]
            if len(valid_bpm) > 0:
                bpm_min = np.min(valid_bpm) - 10
                bpm_max = np.max(valid_bpm) + 10
        if self.inst_bpm is not None:
            valid_bpm = self.inst_bpm[~np.isnan(self.inst_bpm)]
            if len(valid_bpm) > 0:
                if bpm_min is None:
                    bpm_min = np.min(valid_bpm) - 10
                    bpm_max = np.max(valid_bpm) + 10
                else:
                    bpm_min = min(bpm_min, np.min(valid_bpm) - 10)
                    bpm_max = max(bpm_max, np.max(valid_bpm) + 10)
        
        if bpm_min is not None and bpm_max is not None:
            self.bpm_ax.set_ylim([max(0, bpm_min), bpm_max])
        
        # Auto-scale x-axis to full range
        if self.hr_ts is not None and len(self.hr_ts) > 0:
            self.bpm_ax.set_xlim([self.hr_ts[0], self.hr_ts[-1]])
        
        # Draw canvas
        self.bpm_canvas.draw()
        
        # Mark that BPM plot is up to date
        self.bpm_plot_needs_update = False
    
    def plot_signal(self):
        """Plot the HR signal and BPM."""
        if self.hr is None:
            return
        
        # Get current events from editor if available
        if self.event_editor is not None:
            self.hr_sp_times = self.event_editor.get_events()
        
        # Clear bottom subplot (HR signal)
        self.ax.clear()
        
        # Plot HR signal in bottom subplot
        self.ax.plot(self.hr_ts, self.hr, 'b-', linewidth=0.5, label='HR Signal', zorder=0)
        
        # Plot peaks - let event_editor handle plotting if it exists to avoid duplicates
        if self.event_editor is not None:
            # Event editor will handle all event plotting via draw_events()
            # Don't plot here to avoid duplicate legend entries
            pass
        elif self.hr_sp_times is not None and len(self.hr_sp_times) > 0:
            # Fallback: just plot all peaks in red if no event editor
            peak_ys = np.interp(self.hr_sp_times, self.hr_ts, self.hr)
            self.ax.scatter(self.hr_sp_times, peak_ys, color='r', marker='o', 
                          s=30, zorder=2, label='Detected Peaks')
        
        # Set labels for bottom subplot
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Heart Rate Signal and Detected Peaks')
        self.ax.grid(True, alpha=0.3)
        
        # Update event editor if it exists - this will plot events and update legend
        if self.event_editor is not None:
            self.event_editor.draw_events()
            # draw_events() will handle legend creation to include all event types
        else:
            # Only show legend if we plotted something (peaks or signal)
            self.ax.legend()
        
        # Update BPM plot in separate window if it needs updating
        if self.bpm_plot_needs_update:
            self.update_bpm_window()
        
        # Auto-zoom to first 5 seconds on x-axis
        x_min = self.hr_ts[0] if self.hr_ts is not None and len(self.hr_ts) > 0 else 0
        x_max = min(5.0, self.hr_ts[-1]) if self.hr_ts is not None and len(self.hr_ts) > 0 else 5.0
        if self.hr_ts is not None and len(self.hr_ts) > 0:
            self.ax.set_xlim([x_min, x_max])
        
        # Check if we have peaks for y-axis zoom
        has_peaks = False
        if self.event_editor is not None:
            events = self.event_editor.get_events()
            has_peaks = len(events) > 0
        elif self.hr_sp_times is not None:
            has_peaks = len(self.hr_sp_times) > 0
        
        # Draw canvas first
        self.canvas.draw()
        
        # Then set y-axis zoom after drawing (to prevent matplotlib from overriding)
        if has_peaks:
            # Use fixed range [-2, +2] for y-axis zoom (bottom subplot)
            self.ax.set_ylim([-2, 2])
            # Disable auto-scaling for y-axis to preserve our zoom
            self.ax.set_autoscaley_on(False)
            # Force redraw with new y-limits
            self.canvas.draw()
        else:
            # Re-enable auto-scaling when no peaks detected
            self.ax.set_autoscaley_on(True)
    
    def detect_peaks(self):
        """Detect HR peaks using current parameters."""
        if self.hr is None:
            messagebox.showwarning("Warning", "Please load a file first.")
            return
        
        try:
            # Clear any previous detection results first
            self.hr_sp_ind = None
            self.hr_sp_times = None
            self.inst_bpm = None
            self.inst_bpm_original = None
            self.inst_bpm_from_cleaned_peaks = None
            self.bpm_to_max = None
            self.bpm_to_max_original = None
            self.bpm_to_max_from_cleaned_peaks = None
            self.hrv_metrics = None
            
            self.status_var.set("Detecting peaks...")
            self.root.update()
            
            # Get current parameters from GUI (always read fresh values)
            thresh = self.thresh_var.get()
            refrac = self.refrac_var.get()
            min_dur = self.min_dur_var.get()
            highpass_val = self.highpass_var.get()
            highpass = highpass_val if highpass_val > 0 else None
            use_abs = self.use_abs_var.get()
            
            # Show parameters being used in status
            param_str = f"thresh={thresh:.3f}, refrac={refrac:.1f}ms, min_dur={min_dur}"
            if highpass:
                param_str += f", highpass={highpass:.1f}Hz"
            if use_abs:
                param_str += ", abs=True"
            self.status_var.set(f"Detecting peaks with: {param_str}...")
            self.root.update()
            
            # Detect peaks with current parameters
            dVdt_thresh, self.hr_sp_ind = find_hr_peaks(
                self.hr_ts, self.hr, thresh, refrac, min_dur, highpass, use_abs
            )
            
            # Store highpass filtered signal if used
            if highpass and highpass > 0:
                from scipy import signal
                samp_freq = 1/(self.hr_ts[1] - self.hr_ts[0])
                nyq = samp_freq/2
                b, a = signal.butter(4, highpass/nyq, "high", analog=False)
                self.hr_highpass = signal.filtfilt(b, a, self.hr)
            else:
                self.hr_highpass = np.array([])
            
            # Convert indices to times
            if self.hr_sp_ind.size > 0:
                self.hr_sp_times = self.hr_ts[self.hr_sp_ind]
            else:
                self.hr_sp_times = np.array([])
            
            # Update or create event editor with new detection results
            # This will overwrite any previous detection
            if self.event_editor is None:
                self.event_editor = EventEditor(self.root, self.hr_ts, self.hr, 
                                                self.hr_sp_times.tolist(), self.ax, self.canvas)
            else:
                # Reset event editor with new detection results (overwrites previous)
                self.event_editor.update_events(self.hr_sp_times.tolist() if len(self.hr_sp_times) > 0 else [])
            
            # Compute original BPM immediately after detection for visual inspection
            if len(self.hr_sp_times) >= 2:
                self.inst_bpm_original = find_inst_bpm(self.hr, self.hr_sp_times, self.hr_ts)
                # Mark that BPM plot needs updating
                self.bpm_plot_needs_update = True
                # Auto-open BPM window if it doesn't exist
                try:
                    window_exists = self.bpm_window is not None and self.bpm_window.winfo_exists()
                except:
                    window_exists = False
                if not window_exists:
                    self.show_bpm_window()
            
            # Plot results with new detection (will show original BPM in top subplot)
            self.plot_signal()
            
            num_peaks = len(self.hr_sp_times)
            self.status_var.set(f"Detected {num_peaks} peaks with: {param_str}. Use left/right click to edit.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect peaks:\n{str(e)}")
            self.status_var.set("Error detecting peaks")
    
    def compute_metrics(self):
        """Compute BPM and HRV metrics from cleaned peaks (manually edited)."""
        # Check if we have original detection
        if self.hr_sp_times is None or len(self.hr_sp_times) == 0:
            messagebox.showwarning("Warning", "Please detect peaks first.")
            return
        
        try:
            self.status_var.set("Computing metrics...")
            self.root.update()
            
            # Get cleaned events from editor (includes manually added, excludes manually removed)
            cleaned_events = None
            if self.event_editor is not None:
                cleaned_events = self.event_editor.get_events()
            else:
                # Fallback to original if no editor (shouldn't happen, but be safe)
                cleaned_events = self.hr_sp_times
            
            if len(cleaned_events) < 2:
                messagebox.showwarning("Warning", "Need at least 2 peaks to compute metrics.")
                return
            
            # Ensure we have original BPM computed (from original detection, before manual editing)
            if self.inst_bpm_original is None:
                self.inst_bpm_original = find_inst_bpm(self.hr, self.hr_sp_times, self.hr_ts)
                if self.inst_bpm_original is not None:
                    self.bpm_to_max_original = (self.inst_bpm_original * 100) / np.max(self.inst_bpm_original[~np.isnan(self.inst_bpm_original)])
            
            # Compute BPM from cleaned peaks (manually edited peaks)
            self.inst_bpm_from_cleaned_peaks = find_inst_bpm(self.hr, cleaned_events, self.hr_ts)
            
            # Compute BPM normalized to max (from cleaned peaks)
            if self.inst_bpm_from_cleaned_peaks is not None:
                valid_bpm = self.inst_bpm_from_cleaned_peaks[~np.isnan(self.inst_bpm_from_cleaned_peaks)]
                if len(valid_bpm) > 0:
                    self.bpm_to_max_from_cleaned_peaks = (self.inst_bpm_from_cleaned_peaks * 100) / np.max(valid_bpm)
                else:
                    self.bpm_to_max_from_cleaned_peaks = np.full_like(self.inst_bpm_from_cleaned_peaks, np.nan)
            
            # Clean BPM signals (outlier removal, interpolation) - this is the final cleaned version
            self.inst_bpm, self.bpm_to_max = clean_bpm_signal(
                self.inst_bpm_from_cleaned_peaks.copy(), 
                self.bpm_to_max_from_cleaned_peaks.copy()
            )
            
            # Compute HRV metrics from cleaned peaks (manually edited)
            self.hrv_metrics = calculate_all_hrv_metrics(cleaned_events)
            
            # Mark that BPM plot needs updating to show cleaned BPM
            self.bpm_plot_needs_update = True
            # Update BPM window if it exists
            try:
                if self.bpm_window is not None and self.bpm_window.winfo_exists():
                    self.update_bpm_window()
            except:
                pass
            
            # Show metrics visualization window
            self.show_metrics_window()
            
            # Update plot to show cleaned BPM overlay on top of original BPM (in top subplot)
            self.plot_signal()
            
            self.status_var.set("Metrics computed successfully from cleaned peaks. Cleaned BPM overlaid on original.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute metrics:\n{str(e)}")
            self.status_var.set("Error computing metrics")
    
    
    def show_metrics_window(self):
        """Show metrics visualization window."""
        if self.hrv_metrics is None or self.inst_bpm is None:
            return
        
        # Get cleaned events (from editor if available, otherwise original)
        if self.event_editor is not None:
            cleaned_events = self.event_editor.get_events()
        else:
            cleaned_events = self.hr_sp_times
        
        # Create new window
        metrics_window = tk.Toplevel(self.root)
        metrics_window.title("HRV Metrics Visualization")
        metrics_window.geometry("1200x800")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(metrics_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Summary Statistics
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Summary")
        
        summary_text = tk.Text(summary_frame, wrap=tk.WORD, font=('Courier', 11))
        summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        summary_content = "HRV METRICS SUMMARY\n"
        summary_content += "=" * 50 + "\n\n"
        summary_content += f"Mean Heart Rate (from RR): {self.hrv_metrics['mean_hr']:.2f} BPM\n"
        if self.inst_bpm is not None:
            summary_content += f"Mean BPM (cleaned):       {np.nanmean(self.inst_bpm):.2f} BPM\n"
            summary_content += f"Median BPM (cleaned):     {np.nanmedian(self.inst_bpm):.2f} BPM\n"
            summary_content += f"Std BPM (cleaned):        {np.nanstd(self.inst_bpm):.2f} BPM\n"
        summary_content += f"SDNN:                   {self.hrv_metrics['sdnn']:.2f} ms\n"
        summary_content += f"RMSSD (mean):           {np.nanmean(self.hrv_metrics['rmssd']):.2f} ms\n"
        summary_content += f"RMSSD (median):         {np.nanmedian(self.hrv_metrics['rmssd']):.2f} ms\n"
        summary_content += f"pNN50:                  {self.hrv_metrics['pnn50']:.2f}%\n\n"
        summary_content += "RR INTERVAL STATISTICS\n"
        summary_content += "=" * 50 + "\n\n"
        summary_content += f"Mean RR:                {self.hrv_metrics['mean_rr']:.2f} ms\n"
        summary_content += f"Median RR:              {self.hrv_metrics['median_rr']:.2f} ms\n"
        summary_content += f"Min RR:                 {self.hrv_metrics['min_rr']:.2f} ms\n"
        summary_content += f"Max RR:                 {self.hrv_metrics['max_rr']:.2f} ms\n"
        summary_content += f"Std RR:                 {self.hrv_metrics['std_rr']:.2f} ms\n"
        
        summary_text.insert('1.0', summary_content)
        summary_text.config(state=tk.DISABLED)
        
        # Tab 2: BPM over time (show both original and cleaned)
        bpm_frame = ttk.Frame(notebook)
        notebook.add(bpm_frame, text="BPM Over Time")
        
        fig_bpm = Figure(figsize=(10, 5))
        ax_bpm = fig_bpm.add_subplot(111)
        
        # Plot original BPM if available (from original detection, before manual editing)
        if self.inst_bpm_original is not None:
            ax_bpm.plot(self.hr_ts, self.inst_bpm_original, 'orange', linewidth=1, 
                       alpha=0.7, label='BPM (from original peaks)')
            ax_bpm.axhline(y=np.nanmean(self.inst_bpm_original), color='orange', 
                          linestyle=':', alpha=0.5, linewidth=1)
        
        # Plot cleaned BPM (from cleaned peaks, after signal cleaning - this is the main one)
        if self.inst_bpm is not None:
            ax_bpm.plot(self.hr_ts, self.inst_bpm, 'g-', linewidth=2, 
                       label='BPM (from cleaned peaks, signal cleaned)')
            ax_bpm.axhline(y=np.nanmean(self.inst_bpm), color='r', linestyle='--', 
                          label=f'Mean (cleaned): {np.nanmean(self.inst_bpm):.2f} BPM')
        ax_bpm.set_xlabel('Time (s)')
        ax_bpm.set_ylabel('BPM')
        ax_bpm.set_title('Instantaneous Heart Rate - Original vs. Cleaned Peaks (with signal cleaning)')
        ax_bpm.grid(True, alpha=0.3)
        ax_bpm.legend()
        
        canvas_bpm = FigureCanvasTkAgg(fig_bpm, bpm_frame)
        canvas_bpm.draw()
        canvas_bpm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: RMSSD over time
        if len(self.hrv_metrics['rmssd']) > 0:
            rmssd_frame = ttk.Frame(notebook)
            notebook.add(rmssd_frame, text="RMSSD Over Time")
            
            fig_rmssd = Figure(figsize=(10, 5))
            ax_rmssd = fig_rmssd.add_subplot(111)
            
            # Calculate time points for RMSSD (one less than RR intervals)
            if len(cleaned_events) > 1:
                rmssd_times = cleaned_events[1:-1]  # Approximate times for RMSSD
                if len(rmssd_times) > len(self.hrv_metrics['rmssd']):
                    rmssd_times = rmssd_times[:len(self.hrv_metrics['rmssd'])]
                elif len(rmssd_times) < len(self.hrv_metrics['rmssd']):
                    # Create time array
                    rmssd_times = np.linspace(cleaned_events[1], cleaned_events[-1], 
                                            len(self.hrv_metrics['rmssd']))
                
                valid_mask = ~np.isnan(self.hrv_metrics['rmssd'])
                if np.any(valid_mask):
                    ax_rmssd.plot(rmssd_times[valid_mask], 
                                 self.hrv_metrics['rmssd'][valid_mask], 
                                 'b-', linewidth=1)
                    ax_rmssd.axhline(y=np.nanmean(self.hrv_metrics['rmssd']), 
                                    color='r', linestyle='--',
                                    label=f'Mean: {np.nanmean(self.hrv_metrics["rmssd"]):.2f} ms')
            
            ax_rmssd.set_xlabel('Time (s)')
            ax_rmssd.set_ylabel('RMSSD (ms)')
            ax_rmssd.set_title('RMSSD Over Time')
            ax_rmssd.grid(True, alpha=0.3)
            ax_rmssd.legend()
            
            canvas_rmssd = FigureCanvasTkAgg(fig_rmssd, rmssd_frame)
            canvas_rmssd.draw()
            canvas_rmssd.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 4: RR Intervals
        if len(cleaned_events) > 1:
            rr_frame = ttk.Frame(notebook)
            notebook.add(rr_frame, text="RR Intervals")
            
            fig_rr = Figure(figsize=(10, 5))
            ax_rr = fig_rr.add_subplot(111)
            
            RR = np.diff(cleaned_events) * 1000  # Convert to ms
            rr_times = cleaned_events[1:]  # Times for RR intervals
            
            ax_rr.plot(rr_times, RR, 'o-', markersize=3, linewidth=1)
            ax_rr.axhline(y=np.nanmean(RR), color='r', linestyle='--',
                         label=f'Mean: {np.nanmean(RR):.2f} ms')
            ax_rr.set_xlabel('Time (s)')
            ax_rr.set_ylabel('RR Interval (ms)')
            ax_rr.set_title('RR Intervals Over Time')
            ax_rr.grid(True, alpha=0.3)
            ax_rr.legend()
            
            canvas_rr = FigureCanvasTkAgg(fig_rr, rr_frame)
            canvas_rr.draw()
            canvas_rr.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def save_results(self):
        """Save results to .npy file."""
        if self.hr is None:
            messagebox.showwarning("Warning", "No data to save.")
            return
        
        # Get events from editor if available
        if self.event_editor is not None:
            self.hr_sp_times = self.event_editor.get_events()
        
        if self.hr_sp_times is None or len(self.hr_sp_times) == 0:
            messagebox.showwarning("Warning", "No peaks detected. Please detect peaks first.")
            return
        
        # Ask for save location
        if self.file_path:
            default_name = os.path.splitext(os.path.basename(self.file_path))[0] + "_hr_analysis.npy"
        else:
            default_name = "hr_analysis.npy"
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
            initialfile=default_name
        )
        
        if not file_path:
            return
        
        try:
            self.status_var.set("Saving results...")
            self.root.update()
            
            # Get mouse ID
            mouse_id = self.mouse_id_entry.get().strip()
            if not mouse_id:
                mouse_id = "unknown"
            
            # Prepare data dictionary matching original script format
            data = {
                'mouse_id': mouse_id,
                'hr': self.hr,
                'hr_ts': self.hr_ts,
                'R_start': self.hr_sp_times,
                'hr_sp_ind': self.hr_sp_ind if self.hr_sp_ind is not None else np.array([]),
            }
            
            # Add BPM data if available
            if self.inst_bpm is not None:
                data['inst_bpm'] = self.inst_bpm
                data['inst_bpm_ts'] = self.hr_ts
                data['bpm_to_max'] = self.bpm_to_max
            
            # Add HRV metrics directly (matching original script format)
            if self.hrv_metrics is not None:
                data['rmssd'] = self.hrv_metrics['rmssd']
                data['rmssd_to_max'] = self.hrv_metrics['rmssd_to_max']
                # Add other metrics if needed
                data['sdnn'] = self.hrv_metrics['sdnn']
                data['pnn50'] = self.hrv_metrics['pnn50']
                data['mean_hr'] = self.hrv_metrics['mean_hr']
                data['mean_rr'] = self.hrv_metrics['mean_rr']
                data['median_rr'] = self.hrv_metrics['median_rr']
                data['min_rr'] = self.hrv_metrics['min_rr']
                data['max_rr'] = self.hrv_metrics['max_rr']
                data['std_rr'] = self.hrv_metrics['std_rr']
            
            # Add highpass filtered signal if available
            if self.hr_highpass.size > 0:
                data['hr_highpass'] = self.hr_highpass
            else:
                data['hr_highpass'] = np.array([])
            
            # Add metadata (optional, for reference)
            data['source_file'] = self.file_path if self.file_path else ""
            data['detection_params'] = {
                'threshold': self.thresh_var.get(),
                'refractory': self.refrac_var.get(),
                'min_duration': self.min_dur_var.get(),
                'highpass': self.highpass_var.get(),
                'use_abs': self.use_abs_var.get()
            }
            
            # Save
            np.save(file_path, data, allow_pickle=True)
            
            self.status_var.set(f"Results saved to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Results saved successfully to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")
            self.status_var.set("Error saving results")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = HRDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

