# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:26:09 2023

@author: meryl_malezieux
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg'

class EventEditor:

    def __init__(self, ts, time_series, events):
        self.ts = ts
        self.ts = ts[::10]
        self.time_series = time_series[::10]  # subsample by factor of 10
        self.events = events  
        self.fig, self.ax =  plt.subplots(figsize=(20, 5)) 
        self.ax.set_xlim([0, 20])
        self.ax.set_ylim([-1, 1])
        self.event_plots = []
        self.create_gui()
        self.draw_events()
        plt.show()
    
    def create_gui(self):
        # Plot time-series
        self.ax.plot(self.ts, self.time_series, zorder=0)

        # Connect event handler for mouse clicks
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Check if the plot is in pan or zoom mode
        if self.ax.get_navigate_mode() in {'PAN', 'ZOOM'}:
            return

        # Check if left or right button was clicked
        if event.button == 1:  # Left button clicked
            self.add_event(event.xdata)
        elif event.button == 3:  # Right button clicked
            self.remove_event_by_position(event.xdata)
    def add_event(self, x):
        # Check if event already exists at x position
        for event in self.events:
            if abs(event - x) < 0.001:
                return

        # Add event and redraw plot
        self.events.append(x)
        self.draw_events()

    def remove_event_by_position(self, x):
        # Find index of closest event to x position
        distances = [abs(event - x) for event in self.events]
        if not distances:
            return

        index = np.argmin(distances)

        # Check if event is within distance threshold
        if distances[index] < 50:
            # Remove event and redraw plot
            self.events.pop(index)
            self.draw_events()

    def draw_events(self):
        # Clear existing event plots
        for plot in self.event_plots:
            plot.remove()
        self.event_plots = []
    
        # Plot events
        event_xs = np.array(self.events)
        event_ys = np.zeros_like(event_xs)
        event_plots = self.ax.scatter(event_xs, event_ys, marker='o', color='r', zorder=1, s=80)
    
        self.event_plots.append(event_plots)
        self.fig.canvas.draw_idle()
        
    def remove_event(self, event):
        # Get index of clicked event
        index = self.event_plots.index(event.artist)
        self.remove_event_by_index(index)

    def remove_event_by_index(self, index):
        if index >= len(self.events):
            return

        # Remove event and redraw plot
        self.events.pop(index)
        self.draw_events()
        
        
# if __name__ == '__main__':
#     time_series = hr.astype(np.float16)
#     events = hr_sp_ind.tolist()
#     # Example usage
    # editor = EventEditor(time_series[:20000], events[:100])