# -*- coding: utf-8 -*-
"""
Event Editor Module
Interactive GUI for manually editing detected heart rate peaks.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
matplotlib.use('TkAgg')


class EventEditor:
    """
    Interactive event editor for heart rate peaks.
    Left click to add event, right click to remove event.
    """
    
    def __init__(self, parent, ts, time_series, events, ax=None, canvas=None):
        """
        Initialize EventEditor.
        
        Parameters:
        -----------
        parent : tkinter widget
            Parent widget
        ts : array
            Time stamps
        time_series : array
            Time series data
        events : list
            List of event times
        ax : matplotlib axes, optional
            Existing axes to use (default: None, creates new)
        canvas : matplotlib canvas, optional
            Existing canvas to use for drawing (default: None)
        """
        self.parent = parent
        self.ts = ts
        self.time_series = time_series
        # Track original detected events (red)
        self.original_events = set(events) if events is not None else set()
        # Track manually added events (green)
        self.added_events = set()
        # Track removed events (grey, excluded from analysis)
        self.removed_events = set()
        self.ax = ax
        self.canvas = canvas
        self.event_plots = []
        self.subsample_factor = max(1, len(ts) // 50000)  # Subsample for large datasets
        
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 4))
            self.canvas = None
        else:
            self.fig = self.ax.figure
            if self.canvas is None:
                # Try to get canvas from figure
                self.canvas = self.fig.canvas
        
        self.create_plot()
        self.connect_events()
    
    def create_plot(self):
        """Create the initial plot."""
        # Only create plot if axes is new (not provided)
        if self.ax is not None and len(self.ax.lines) == 0:
            # Subsample for display if needed
            ts_display = self.ts[::self.subsample_factor]
            signal_display = self.time_series[::self.subsample_factor]
            
            self.ax.plot(ts_display, signal_display, 'b-', linewidth=0.5, zorder=0)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title('Heart Rate Detection - Left click to add (green), Right click to remove (grey)')
            self.ax.grid(True, alpha=0.3)
        
        self.draw_events()
    
    def connect_events(self):
        """Connect mouse events."""
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    
    def on_click(self, event):
        """Handle mouse clicks."""
        if event.inaxes != self.ax:
            return
        
        # Check if plot is in pan or zoom mode
        if self.ax.get_navigate_mode() in {'PAN', 'ZOOM'}:
            return
        
        # Left button: add event
        if event.button == 1:
            self.add_event(event.xdata)
        # Right button: remove event
        elif event.button == 3:
            self.remove_event_by_position(event.xdata)
    
    def add_event(self, x):
        """Add event at x position (green dot)."""
        # Check if event already exists (within tolerance)
        all_events = list(self.original_events | self.added_events)
        for event in all_events:
            if abs(event - x) < 0.001:
                return
        
        # If it was removed, un-remove it
        if x in self.removed_events:
            self.removed_events.remove(x)
        
        # Add as manually added event (green)
        self.added_events.add(x)
        self.draw_events()
    
    def remove_event_by_position(self, x):
        """Mark event as removed (grey dot, excluded from analysis)."""
        # Find closest event from all events
        all_events = list(self.original_events | self.added_events)
        if not all_events:
            return
        
        distances = [abs(event - x) for event in all_events]
        index = np.argmin(distances)
        closest_event = all_events[index]
        
        # Check if within reasonable distance (0.5 seconds)
        if distances[index] < 0.5:
            # Mark as removed
            self.removed_events.add(closest_event)
            # Remove from added events if it was manually added
            self.added_events.discard(closest_event)
            self.draw_events()
    
    def draw_events(self):
        """Draw events on plot with different colors."""
        # Clear existing event plots
        for plot in self.event_plots:
            try:
                if hasattr(plot, 'remove'):
                    plot.remove()
                elif plot in self.ax.collections:
                    self.ax.collections.remove(plot)
            except:
                pass
        self.event_plots = []
        
        # Get all events
        all_events = list(self.original_events | self.added_events | self.removed_events)
        if not all_events:
            if self.canvas:
                self.canvas.draw_idle()
            return
        
        # Separate events by type
        original_active = [e for e in self.original_events if e not in self.removed_events]
        added_active = [e for e in self.added_events if e not in self.removed_events]
        removed = list(self.removed_events)
        
        # Plot original detected events (red)
        if original_active:
            event_xs = np.array(original_active)
            event_ys = np.interp(event_xs, self.ts, self.time_series)
            scatter_red = self.ax.scatter(event_xs, event_ys, marker='o', color='r', 
                                         zorder=3, s=50, edgecolors='darkred', linewidths=1,
                                         label='Detected peaks')
            self.event_plots.append(scatter_red)
        
        # Plot manually added events (green)
        if added_active:
            event_xs = np.array(added_active)
            event_ys = np.interp(event_xs, self.ts, self.time_series)
            scatter_green = self.ax.scatter(event_xs, event_ys, marker='o', color='g', 
                                           zorder=3, s=50, edgecolors='darkgreen', linewidths=1,
                                           label='Manually added')
            self.event_plots.append(scatter_green)
        
        # Plot removed events (grey)
        if removed:
            event_xs = np.array(removed)
            event_ys = np.interp(event_xs, self.ts, self.time_series)
            scatter_grey = self.ax.scatter(event_xs, event_ys, marker='o', color='grey', 
                                          zorder=2, s=50, edgecolors='darkgrey', linewidths=1,
                                          alpha=0.6, label='Removed (excluded)')
            self.event_plots.append(scatter_grey)
        
        # Update legend if there are multiple types
        if len([x for x in [original_active, added_active, removed] if x]) > 1:
            self.ax.legend(loc='upper right', fontsize=8)
        
        if self.canvas:
            self.canvas.draw_idle()
    
    def get_events(self):
        """Get current list of active events (excluding removed ones)."""
        # Return only events that are not removed
        active_events = (self.original_events | self.added_events) - self.removed_events
        return np.array(sorted(active_events))
    
    def update_events(self, events):
        """Update events list (resets to new detected events)."""
        # Reset all tracking
        self.original_events = set(events) if events is not None else set()
        self.added_events = set()
        self.removed_events = set()
        self.draw_events()

