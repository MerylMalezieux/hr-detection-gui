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

from .hr_detection import load_abf_file, find_hr_peaks, find_inst_bpm
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
        self.hr_sp_times = None
        self.inst_bpm = None
        self.bpm_to_max = None
        self.hrv_metrics = None
        self.file_path = None
        self.mouse_id = ""
        self.hr_highpass = np.array([])
        
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
        
        ttk.Button(file_frame, text="Load ABF File", command=self.load_file).grid(row=0, column=0, padx=5)
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
        ttk.Button(buttons_frame, text="Save Results", command=self.save_results).grid(row=2, column=0, padx=5, pady=2)
        
        # Main plot area
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Heart Rate Signal and Detected Peaks')
        self.ax.grid(True, alpha=0.3)
        
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
    
    def load_file(self):
        """Load ABF file."""
        file_path = filedialog.askopenfilename(
            title="Select ABF File",
            filetypes=[("ABF files", "*.abf"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.status_var.set("Loading file...")
            self.root.update()
            
            # Load file
            self.hr, self.hr_ts = load_abf_file(file_path)
            self.file_path = file_path
            
            # Update file label
            filename = os.path.basename(file_path)
            self.file_label.config(text=filename)
            
            # Reset detection results
            self.hr_sp_ind = None
            self.hr_sp_times = None
            self.inst_bpm = None
            self.bpm_to_max = None
            self.hrv_metrics = None
            self.hr_highpass = np.array([])
            
            # Try to extract mouse ID from filename if possible
            # (e.g., if filename contains mouse ID pattern)
            # For now, leave it empty for user to enter
            
            # Plot signal
            self.plot_signal()
            
            self.status_var.set(f"File loaded: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.status_var.set("Error loading file")
    
    def plot_signal(self):
        """Plot the HR signal."""
        if self.hr is None:
            return
        
        # Get current events from editor if available
        if self.event_editor is not None:
            self.hr_sp_times = self.event_editor.get_events()
        
        self.ax.clear()
        self.ax.plot(self.hr_ts, self.hr, 'b-', linewidth=0.5, label='HR Signal', zorder=0)
        
        # Plot peaks with different colors if event editor exists
        if self.event_editor is not None:
            # Get all event types
            original_active = [e for e in self.event_editor.original_events 
                             if e not in self.event_editor.removed_events]
            added_active = [e for e in self.event_editor.added_events 
                          if e not in self.event_editor.removed_events]
            removed = list(self.event_editor.removed_events)
            
            # Plot original detected peaks (red)
            if original_active:
                peak_ys = np.interp(original_active, self.hr_ts, self.hr)
                self.ax.scatter(original_active, peak_ys, color='r', marker='o', 
                              s=30, zorder=3, label='Detected peaks', edgecolors='darkred')
            
            # Plot manually added peaks (green)
            if added_active:
                peak_ys = np.interp(added_active, self.hr_ts, self.hr)
                self.ax.scatter(added_active, peak_ys, color='g', marker='o', 
                              s=30, zorder=3, label='Manually added', edgecolors='darkgreen')
            
            # Plot removed peaks (grey)
            if removed:
                peak_ys = np.interp(removed, self.hr_ts, self.hr)
                self.ax.scatter(removed, peak_ys, color='grey', marker='o', 
                              s=30, zorder=2, label='Removed (excluded)', 
                              edgecolors='darkgrey', alpha=0.6)
        elif self.hr_sp_times is not None and len(self.hr_sp_times) > 0:
            # Fallback: just plot all peaks in red if no event editor
            peak_ys = np.interp(self.hr_sp_times, self.hr_ts, self.hr)
            self.ax.scatter(self.hr_sp_times, peak_ys, color='r', marker='o', 
                          s=30, zorder=2, label='Detected Peaks')
        
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Heart Rate Signal and Detected Peaks')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Auto-zoom to first 100 seconds
        if self.hr_ts is not None and len(self.hr_ts) > 0:
            max_time = min(5.0, self.hr_ts[-1])  # Show 5s or full signal if shorter
            self.ax.set_xlim([self.hr_ts[0], max_time])
        
        self.canvas.draw()
        
        # Update event editor if it exists
        if self.event_editor is not None:
            self.event_editor.draw_events()
    
    def detect_peaks(self):
        """Detect HR peaks using current parameters."""
        if self.hr is None:
            messagebox.showwarning("Warning", "Please load a file first.")
            return
        
        try:
            self.status_var.set("Detecting peaks...")
            self.root.update()
            
            # Get parameters
            thresh = self.thresh_var.get()
            refrac = self.refrac_var.get()
            min_dur = self.min_dur_var.get()
            highpass = self.highpass_var.get() if self.highpass_var.get() > 0 else None
            use_abs = self.use_abs_var.get()
            
            # Detect peaks
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
            
            # Plot results
            self.plot_signal()
            
            # Create/update event editor
            if self.event_editor is None:
                self.event_editor = EventEditor(self.root, self.hr_ts, self.hr, 
                                                self.hr_sp_times.tolist(), self.ax, self.canvas)
            else:
                self.event_editor.update_events(self.hr_sp_times)
                self.plot_signal()  # Redraw to show updated events
            
            num_peaks = len(self.hr_sp_times)
            self.status_var.set(f"Detected {num_peaks} peaks. Use left/right click to edit.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect peaks:\n{str(e)}")
            self.status_var.set("Error detecting peaks")
    
    def compute_metrics(self):
        """Compute BPM and HRV metrics."""
        if self.hr_sp_times is None or len(self.hr_sp_times) == 0:
            messagebox.showwarning("Warning", "Please detect peaks first.")
            return
        
        try:
            self.status_var.set("Computing metrics...")
            self.root.update()
            
            # Get events from editor if available
            if self.event_editor is not None:
                self.hr_sp_times = self.event_editor.get_events()
                if len(self.hr_sp_times) > 0:
                    # Convert back to indices for consistency
                    self.hr_sp_ind = np.searchsorted(self.hr_ts, self.hr_sp_times)
                else:
                    self.hr_sp_ind = np.array([])
            
            if len(self.hr_sp_times) < 2:
                messagebox.showwarning("Warning", "Need at least 2 peaks to compute metrics.")
                return
            
            # Compute instantaneous BPM
            self.inst_bpm = find_inst_bpm(self.hr, self.hr_sp_times, self.hr_ts)
            
            # Compute BPM normalized to max
            self.bpm_to_max = (self.inst_bpm * 100) / np.max(self.inst_bpm[~np.isnan(self.inst_bpm)])
            
            # Clean BPM signals
            self.inst_bpm, self.bpm_to_max = clean_bpm_signal(self.inst_bpm, self.bpm_to_max)
            
            # Compute HRV metrics
            self.hrv_metrics = calculate_all_hrv_metrics(self.hr_sp_times)
            
            # Show metrics visualization window
            self.show_metrics_window()
            
            # Update plot to show BPM
            self.plot_bpm()
            
            self.status_var.set("Metrics computed successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute metrics:\n{str(e)}")
            self.status_var.set("Error computing metrics")
    
    def plot_bpm(self):
        """Plot BPM signal."""
        if self.inst_bpm is None:
            return
        
        self.ax.clear()
        self.ax.plot(self.hr_ts, self.inst_bpm, 'g-', linewidth=1, label='Instantaneous BPM')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('BPM')
        self.ax.set_title('Heart Rate (BPM)')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.canvas.draw()
    
    def show_metrics_window(self):
        """Show metrics visualization window."""
        if self.hrv_metrics is None or self.inst_bpm is None:
            return
        
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
        summary_content += f"Mean Heart Rate:        {self.hrv_metrics['mean_hr']:.2f} BPM\n"
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
        
        # Tab 2: BPM over time
        bpm_frame = ttk.Frame(notebook)
        notebook.add(bpm_frame, text="BPM Over Time")
        
        fig_bpm = Figure(figsize=(10, 5))
        ax_bpm = fig_bpm.add_subplot(111)
        ax_bpm.plot(self.hr_ts, self.inst_bpm, 'g-', linewidth=1)
        ax_bpm.axhline(y=np.nanmean(self.inst_bpm), color='r', linestyle='--', 
                      label=f'Mean: {np.nanmean(self.inst_bpm):.2f} BPM')
        ax_bpm.set_xlabel('Time (s)')
        ax_bpm.set_ylabel('BPM')
        ax_bpm.set_title('Instantaneous Heart Rate (BPM)')
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
            if len(self.hr_sp_times) > 1:
                rmssd_times = self.hr_sp_times[1:-1]  # Approximate times for RMSSD
                if len(rmssd_times) > len(self.hrv_metrics['rmssd']):
                    rmssd_times = rmssd_times[:len(self.hrv_metrics['rmssd'])]
                elif len(rmssd_times) < len(self.hrv_metrics['rmssd']):
                    # Create time array
                    rmssd_times = np.linspace(self.hr_sp_times[1], self.hr_sp_times[-1], 
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
        if len(self.hr_sp_times) > 1:
            rr_frame = ttk.Frame(notebook)
            notebook.add(rr_frame, text="RR Intervals")
            
            fig_rr = Figure(figsize=(10, 5))
            ax_rr = fig_rr.add_subplot(111)
            
            RR = np.diff(self.hr_sp_times) * 1000  # Convert to ms
            rr_times = self.hr_sp_times[1:]  # Times for RR intervals
            
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

