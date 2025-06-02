import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from models.virtual_elevation import VirtualElevation
from ui.map_widget import (MapWidget, MapMode)


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in Qt"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class GPSGateResult(QMainWindow):
    """Window for displaying GPS gate one-way analysis results"""

    def __init__(self, fit_file, settings, selected_laps, params):
        super().__init__()
        self.fit_file = fit_file
        self.settings = settings
        self.selected_laps = selected_laps
        self.params = params
        self.result_dir = settings.result_dir
        self.detected_sections = []
        self.section_ve_profiles = []

        # List of gate sets (each gate set has A and B)
        self.gate_sets = []
        self.active_set_index = -1  # Currently active gate set

        # Initialize these attributes BEFORE calling prepare_merged_data
        self.trim_start = 0
        self.trim_end = 0
        self.calibration_end = 0  # End of calibration lap

        # Prepare merged lap data
        self.prepare_merged_data()

        # Create VE calculator
        self.ve_calculator = VirtualElevation(self.merged_data, self.params)

        # Get lap combination ID for settings
        self.lap_combo_id = "_".join(map(str, sorted(self.selected_laps)))
        self.settings_key = f"GATE_lap_{self.lap_combo_id}"

        # Try to load saved trim values for this lap combination
        file_settings = self.settings.get_file_settings(self.fit_file.filename)
        trim_settings = file_settings.get("trim_settings", {})
        saved_trim = trim_settings.get(self.settings_key, {})

        if saved_trim and "trim_start" in saved_trim and "trim_end" in saved_trim:
            # Use saved trim values if available
            self.trim_start = saved_trim["trim_start"]
            self.trim_end = saved_trim["trim_end"]

            # Load saved gates if available
            if "gate_sets" in saved_trim:
                # Create clean gate sets without previous sections
                self.gate_sets = []
                for gate_set in saved_trim["gate_sets"]:
                    clean_gate = {
                        "gate_a_pos": gate_set["gate_a_pos"],
                        "gate_b_pos": gate_set["gate_b_pos"],
                        "direction": gate_set.get("direction"),
                        "calibration_point": None,  # Will be set during detection
                        "sections": [],  # Empty sections list that will be filled during detection
                    }
                    self.gate_sets.append(clean_gate)

                self.active_set_index = min(len(self.gate_sets) - 1, 0)

        if not self.gate_sets:
            self.add_gate_set()
            self.active_set_index = min(len(self.gate_sets) - 1, 0)

        # Initialize values for CdA and Crr
        self.current_cda = self.params.get("cda")
        self.current_crr = self.params.get("crr")

        # If CdA or Crr are None (to be optimized), set initial values to middle of range
        if self.current_cda is None:
            if saved_trim and "cda" in saved_trim:
                self.current_cda = saved_trim["cda"]
            else:
                self.current_cda = (
                    self.params.get("cda_min", 0.15) + self.params.get("cda_max", 0.5)
                ) / 2

        if self.current_crr is None:
            if saved_trim and "crr" in saved_trim:
                self.current_crr = saved_trim["crr"]
            else:
                self.current_crr = (
                    self.params.get("crr_min", 0.001) + self.params.get("crr_max", 0.03)
                ) / 2

        # Setup UI
        self.initUI()

        # Update the calibration end
        self.update_calibration_end()

        # Detect sections based on gates
        self.detect_sections()

        # Calculate and plot initial VE
        self.calculate_ve()
        self.update_plots()

    def add_gate_set(self):
        """Add a new gate set with default positions"""
        # Set default gate positions
        if not self.gate_sets:
            # First gate set
            gate_a_pos = self.trim_start + int((self.trim_end - self.trim_start) * 0.25)
            gate_b_pos = self.trim_start + int((self.trim_end - self.trim_start) * 0.75)
        else:
            # Subsequent gate sets - position after the last one
            last_gate_b = self.gate_sets[-1]["gate_b_pos"]
            remaining_range = self.trim_end - last_gate_b

            if remaining_range > 30:
                gate_a_pos = last_gate_b + int(remaining_range * 0.33)
                gate_b_pos = last_gate_b + int(remaining_range * 0.67)
            else:
                # Not enough room for another gate
                return False

        # Create new gate set
        new_gate = {
            "gate_a_pos": gate_a_pos,
            "gate_b_pos": gate_b_pos,
            "direction": None,  # Will be set on first passing
            "calibration_point": None,  # Will be used to track calibration pass
            "sections": [],  # Will hold detected sections
        }

        self.gate_sets.append(new_gate)
        self.active_set_index = len(self.gate_sets) - 1
        return True

    def prepare_merged_data(self):
        """Extract and merge data for selected laps"""
        # Get records for selected laps
        self.merged_data = self.fit_file.get_records_for_laps(self.selected_laps)

        # Check if we have enough data
        if len(self.merged_data) < 30:
            raise ValueError("Not enough data points (less than 30 seconds)")

        # Get lap info for display
        self.lap_info = []
        all_laps = self.fit_file.get_lap_data()

        for lap in all_laps:
            if lap["lap_number"] in self.selected_laps:
                self.lap_info.append(lap)

        # Calculate distance, duration, etc. for the merged lap
        self.total_distance = sum(lap["distance"] for lap in self.lap_info)
        self.total_duration = sum(lap["duration"] for lap in self.lap_info)
        self.avg_power = np.mean(self.merged_data["power"].dropna())
        self.avg_speed = (
            (self.total_distance / self.total_duration) * 3600
            if self.total_duration > 0
            else 0
        )

        # Extract GPS coordinates for map
        if (
            "position_lat" in self.merged_data.columns
            and "position_long" in self.merged_data.columns
        ):
            self.has_gps = True
            # Filter out missing coordinates
            valid_coords = self.merged_data.dropna(
                subset=["position_lat", "position_long"]
            )
            if not valid_coords.empty:
                self.start_lat = valid_coords["position_lat"].iloc[0]
                self.start_lon = valid_coords["position_long"].iloc[0]
                self.end_lat = valid_coords["position_lat"].iloc[-1]
                self.end_lon = valid_coords["position_long"].iloc[-1]

                # Extract all route points
                self.route_points = list(
                    zip(valid_coords["position_lat"], valid_coords["position_long"])
                )

                # Store the timestamps to ensure correct mapping of trim indices to route points
                self.route_timestamps = valid_coords["timestamp"].tolist()

                # Initial trim values should correspond to the valid coordinates
                if self.trim_start == 0 and self.trim_end == 0:
                    self.trim_start = 0
                    self.trim_end = len(valid_coords) - 1
            else:
                self.has_gps = False
        else:
            self.has_gps = False

    def initUI(self):
        """Initialize the UI components"""
        self.setWindowTitle(
            f'GPS Gate Analysis - Laps {", ".join(map(str, self.selected_laps))}'
        )
        self.setGeometry(50, 50, 1200, 800)

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Create a splitter for adjustable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left side - Map and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Map
        self.map_widget = MapWidget(MapMode.MARKER_GATE_SETS, self.merged_data, self.params)
        if self.map_widget.has_gps:
            self.map_widget.set_trim_start(self.trim_start)
            self.map_widget.set_trim_end(self.trim_end)
            self.map_widget.set_gate_sets(self.gate_sets, self.detected_sections)
            self.map_widget.update()
            left_layout.addWidget(self.map_widget, 2)
        else:
            no_gps_label = QLabel("No GPS data available")
            no_gps_label.setAlignment(Qt.AlignCenter)
            left_layout.addWidget(no_gps_label, 2)

        # Detected sections table
        section_table_group = QGroupBox("Detected Sections")
        section_table_layout = QVBoxLayout()

        self.section_table = QTableWidget()
        self.section_table.setColumnCount(5)
        self.section_table.setHorizontalHeaderLabels(
            ["Select", "Section", "Gate Set", "Duration", "Distance"]
        )
        section_table_layout.addWidget(self.section_table)

        section_table_group.setLayout(section_table_layout)
        left_layout.addWidget(section_table_group, 1)

        # Parameter display
        param_group = QGroupBox("Analysis Parameters")
        param_layout = QFormLayout()

        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.update_config_text()
        param_layout.addRow("Configuration:", self.config_text)

        # Configuration name input
        self.config_name = QLineEdit("GPS Gate Test")
        param_layout.addRow("Save as:", self.config_name)

        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group, 1)

        # Control buttons
        button_layout = QHBoxLayout()

        self.back_button = QPushButton("Back to Lap Selection")
        self.back_button.clicked.connect(self.back_to_selection)

        self.close_button = QPushButton("Close App")
        self.close_button.clicked.connect(self.close)

        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setStyleSheet(f"background-color: #4363d8; color: white;")

        button_layout.addWidget(self.back_button)
        button_layout.addWidget(self.close_button)
        button_layout.addWidget(self.save_button)

        left_layout.addLayout(button_layout)

        # Right side - Plots and sliders
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Plot area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        # Create plots
        self.fig_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        plot_layout.addWidget(self.fig_canvas)

        right_layout.addWidget(plot_widget, 3)

        # Sliders
        slider_container = QScrollArea()
        slider_container.setWidgetResizable(True)
        slider_content = QWidget()
        slider_content_layout = QVBoxLayout(slider_content)

        # Main trim sliders
        trim_group = QGroupBox("Trim Settings")
        trim_layout = QFormLayout()

        # Trim start slider
        self.trim_start_slider = QSlider(Qt.Horizontal)
        self.trim_start_slider.setMinimum(0)
        self.trim_start_slider.setMaximum(len(self.merged_data) - 30)
        self.trim_start_slider.setValue(self.trim_start)
        self.trim_start_slider.valueChanged.connect(self.on_trim_start_changed)

        self.trim_start_label = QLabel(f"{self.trim_start} s")
        trim_start_layout = QHBoxLayout()
        trim_start_layout.addWidget(self.trim_start_slider)
        trim_start_layout.addWidget(self.trim_start_label)

        trim_layout.addRow("Trim Start:", trim_start_layout)

        # Trim end slider
        self.trim_end_slider = QSlider(Qt.Horizontal)
        self.trim_end_slider.setMinimum(30)
        self.trim_end_slider.setMaximum(len(self.merged_data))
        self.trim_end_slider.setValue(self.trim_end)
        self.trim_end_slider.valueChanged.connect(self.on_trim_end_changed)

        self.trim_end_label = QLabel(f"{self.trim_end} s")
        trim_end_layout = QHBoxLayout()
        trim_end_layout.addWidget(self.trim_end_slider)
        trim_end_layout.addWidget(self.trim_end_label)

        trim_layout.addRow("Trim End:", trim_end_layout)

        trim_group.setLayout(trim_layout)
        slider_content_layout.addWidget(trim_group)

        # Create gate set controls
        self.gate_controls = []
        for i, gate_set in enumerate(self.gate_sets):
            gate_control = self.create_gate_control_group(i)
            slider_content_layout.addWidget(gate_control)
            self.gate_controls.append(gate_control)

        # Add gate set buttons
        gate_buttons_layout = QHBoxLayout()

        self.add_gate_button = QPushButton("Add Gate Set")
        self.add_gate_button.clicked.connect(self.on_add_gate_set)
        gate_buttons_layout.addWidget(self.add_gate_button)

        self.remove_gate_button = QPushButton("Remove Last Gate Set")
        self.remove_gate_button.clicked.connect(self.remove_last_gate_set)
        gate_buttons_layout.addWidget(self.remove_gate_button)

        slider_content_layout.addLayout(gate_buttons_layout)

        # Then make sure to initialize the remove button state
        self.update_remove_gate_button()

        # Update the enabled state of the add button
        self.update_add_gate_button()

        # CdA and Crr sliders
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()

        # CdA slider
        self.cda_slider = QSlider(Qt.Horizontal)
        self.cda_slider.setMinimum(int(self.params.get("cda_min", 0.15) * 1000))
        self.cda_slider.setMaximum(int(self.params.get("cda_max", 0.5) * 1000))
        self.cda_slider.setValue(int(self.current_cda * 1000))
        self.cda_slider.valueChanged.connect(self.on_cda_changed)
        self.cda_slider.setEnabled(self.params.get("cda") is None)

        self.cda_label = QLabel(f"{self.current_cda:.3f}")
        cda_layout = QHBoxLayout()
        cda_layout.addWidget(self.cda_slider)
        cda_layout.addWidget(self.cda_label)

        params_layout.addRow("CdA:", cda_layout)

        # Crr slider
        self.crr_slider = QSlider(Qt.Horizontal)
        self.crr_slider.setMinimum(int(self.params.get("crr_min", 0.001) * 10000))
        self.crr_slider.setMaximum(int(self.params.get("crr_max", 0.03) * 10000))
        self.crr_slider.setValue(int(self.current_crr * 10000))
        self.crr_slider.valueChanged.connect(self.on_crr_changed)
        self.crr_slider.setEnabled(self.params.get("crr") is None)

        self.crr_label = QLabel(f"{self.current_crr:.4f}")
        crr_layout = QHBoxLayout()
        crr_layout.addWidget(self.crr_slider)
        crr_layout.addWidget(self.crr_label)

        params_layout.addRow("Crr:", crr_layout)

        params_group.setLayout(params_layout)
        slider_content_layout.addWidget(params_group)

        # Add a stretch at the end to push everything up
        slider_content_layout.addStretch()

        # Set the content widget for the scroll area
        slider_container.setWidget(slider_content)
        right_layout.addWidget(slider_container, 1)

        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 800])  # Set initial sizes

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_gate_control_group(self, gate_index):
        """Create a control group for a gate set"""
        gate_set = self.gate_sets[gate_index]

        group = QGroupBox(f"Gate Set {gate_index + 1}")
        layout = QFormLayout()

        # Gate A slider
        gate_a_slider = QSlider(Qt.Horizontal)
        gate_a_slider.setMinimum(self.trim_start)
        gate_a_slider.setMaximum(self.trim_end)
        gate_a_slider.setValue(gate_set["gate_a_pos"])
        gate_a_slider.valueChanged.connect(
            lambda value: self.on_gate_a_changed(gate_index, value)
        )

        gate_a_label = QLabel(f"{gate_set['gate_a_pos']} s")
        gate_a_layout = QHBoxLayout()
        gate_a_layout.addWidget(gate_a_slider)
        gate_a_layout.addWidget(gate_a_label)

        layout.addRow(f"Gate {gate_index + 1}A:", gate_a_layout)

        # Gate B slider
        gate_b_slider = QSlider(Qt.Horizontal)
        gate_b_slider.setMinimum(gate_set["gate_a_pos"])
        gate_b_slider.setMaximum(self.trim_end)
        gate_b_slider.setValue(gate_set["gate_b_pos"])
        gate_b_slider.valueChanged.connect(
            lambda value: self.on_gate_b_changed(gate_index, value)
        )

        gate_b_label = QLabel(f"{gate_set['gate_b_pos']} s")
        gate_b_layout = QHBoxLayout()
        gate_b_layout.addWidget(gate_b_slider)
        gate_b_layout.addWidget(gate_b_label)

        layout.addRow(f"Gate {gate_index + 1}B:", gate_b_layout)

        group.setLayout(layout)

        # Store the sliders and labels for later access
        group.gate_a_slider = gate_a_slider
        group.gate_a_label = gate_a_label
        group.gate_b_slider = gate_b_slider
        group.gate_b_label = gate_b_label

        return group

    def update_gate_controls(self):
        """Update all gate controls' states and values"""
        # First, ensure we have the right number of controls
        while len(self.gate_controls) < len(self.gate_sets):
            # Add new control
            index = len(self.gate_controls)
            gate_control = self.create_gate_control_group(index)
            self.gate_controls.append(gate_control)

            # Add to the layout, before the "Add Gate Set" button
            layout = self.add_gate_button.parentWidget().layout()
            layout.insertWidget(
                layout.count() - 2, gate_control
            )  # Insert before the add button and stretch

        # Update minimums, maximums, and values
        for i, gate_control in enumerate(self.gate_controls):
            gate_set = self.gate_sets[i]

            # Update Gate A slider
            min_a = self.trim_start
            if i > 0:
                # A cannot be before previous B
                min_a = max(min_a, self.gate_sets[i - 1]["gate_b_pos"])

            gate_control.gate_a_slider.setMinimum(min_a)
            gate_control.gate_a_slider.setMaximum(self.trim_end - 1)  # Leave room for B
            gate_control.gate_a_slider.setValue(gate_set["gate_a_pos"])
            gate_control.gate_a_label.setText(f"{gate_set['gate_a_pos']} s")

            # Update Gate B slider
            gate_control.gate_b_slider.setMinimum(
                gate_set["gate_a_pos"] + 1
            )  # B must be after A
            gate_control.gate_b_slider.setMaximum(self.trim_end)
            gate_control.gate_b_slider.setValue(gate_set["gate_b_pos"])
            gate_control.gate_b_label.setText(f"{gate_set['gate_b_pos']} s")

        # Update the add gate button state
        self.update_add_gate_button()

    def remove_last_gate_set(self):
        """Remove the last gate set if there are more than one"""
        if len(self.gate_sets) > 1:
            # Remove the last gate set
            self.gate_sets.pop()

            # Update active set index to the new last set
            self.active_set_index = len(self.gate_sets) - 1

            # Remove the last control from the UI
            if self.gate_controls:
                last_control = self.gate_controls.pop()
                last_control.setParent(None)  # Remove from layout
                last_control.deleteLater()  # Schedule for deletion

            # Update the UI controls
            self.update_remove_gate_button()

            # Re-detect sections with the updated gate sets
            self.detect_sections()
            self.calculate_ve()
            self.update_plots()

            self.map_widget.set_gate_sets(self.gate_sets, self.detected_sections)
            self.map_widget.update()

            return True
        return False

    def update_remove_gate_button(self):
        """Update the enabled state of the remove gate button"""
        # Enable only if there's more than one gate set
        self.remove_gate_button.setEnabled(len(self.gate_sets) > 1)

    def update_add_gate_button(self):
        """Update the enabled state of the add gate button"""
        # Enable only if there's room for another gate set
        if not self.gate_sets:
            self.add_gate_button.setEnabled(True)
            return

        last_gate_b = self.gate_sets[-1]["gate_b_pos"]
        remaining_range = self.trim_end - last_gate_b

        # Need at least 30 seconds for a new gate set
        self.add_gate_button.setEnabled(remaining_range >= 30)

        # Also update the remove button
        self.update_remove_gate_button()

    def on_add_gate_set(self):
        """Handle adding a new gate set"""
        if self.add_gate_set():
            self.update_gate_controls()
            self.detect_sections()
            self.calculate_ve()
            self.update_plots()
            self.map_widget.set_gate_sets(self.gate_sets, self.detected_sections)
            self.map_widget.update()

    def update_calibration_end(self):
        """Update the calibration end point based on gate A's first passing"""
        if not self.gate_sets:
            self.calibration_end = self.trim_end
            return

        # Calibration end is when gate A is passed again in the same direction
        # or trim_end if that doesn't happen
        self.calibration_end = self.trim_end

        # Detect when gate A is passed again with same direction
        gate_a_passings = self.detect_gate_passings(
            self.gate_sets[0]["gate_a_pos"], self.trim_start, self.trim_end
        )

        # Need at least two passings in the same direction to end calibration early
        if len(gate_a_passings) >= 2:
            # The first passing defines the reference direction
            ref_direction = gate_a_passings[0]["direction"]

            # Look for next passing in same direction
            for passing in gate_a_passings[1:]:
                # Check if direction is similar (within 30 degrees)
                dir_diff = abs(passing["direction"] - ref_direction)
                dir_diff = min(dir_diff, 360 - dir_diff)

                if dir_diff < 30:  # Similar direction
                    self.calibration_end = passing["index"]
                    break

    def update_config_text(self):
        """Update the configuration text display"""
        lap_str = ", ".join(map(str, self.selected_laps))
        distance_str = f"{self.total_distance:.2f} km"
        duration_str = f"{self.total_duration:.0f} s"
        power_str = f"{self.avg_power:.0f} W"
        # Convert speed from m/s to km/h for display
        speed_str = f"{self.avg_speed:.2f} km/h"

        config_text = f"Selected Laps: {lap_str}\n"
        config_text += f"Distance: {distance_str}\n"
        config_text += f"Duration: {duration_str}\n"
        config_text += f"Avg Power: {power_str}\n"
        config_text += f"Avg Speed: {speed_str}\n"
        config_text += f"System Mass: {self.params.get('system_mass', 90)} kg\n"
        config_text += f"Rho (air density): {self.params.get('rho', 1.2)} kg/m³\n"
        config_text += f"Eta (drivetrain eff.): {self.params.get('eta', 0.98)}\n"
        config_text += f"Current CdA: {self.current_cda:.3f}\n"
        config_text += f"Current Crr: {self.current_crr:.4f}\n"

        if self.params.get("wind_speed") not in [None, 0]:
            config_text += f"Wind Speed: {self.params.get('wind_speed')} m/s\n"

        if self.params.get("wind_direction") is not None:
            config_text += f"Wind Direction: {self.params.get('wind_direction')}°"

        self.config_text.setText(config_text)

    def detect_gate_passings(self, gate_pos, start_time, end_time):
        """
        Detect passings of a gate within a time range.

        Returns a list of passings with index, timestamp, distance and direction.
        """
        if not self.has_gps:
            return []

        # Get valid GPS coordinates
        valid_coords = self.merged_data.dropna(subset=["position_lat", "position_long"])
        if valid_coords.empty:
            return []

        # Get the gate position
        if gate_pos < 0 or gate_pos >= len(valid_coords):
            return []

        gate_lat = valid_coords.iloc[gate_pos]["position_lat"]
        gate_lon = valid_coords.iloc[gate_pos]["position_long"]

        # Calculate distance from each point to the gate
        from math import atan2, cos, degrees, radians, sin, sqrt

        def haversine(lat1, lon1, lat2, lon2):
            """Calculate the great circle distance between two points in meters"""
            R = 6371000  # Earth radius in meters
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c

        # Calculate bearing between two points
        def calculate_bearing(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            y = sin(lon2 - lon1) * cos(lat2)
            x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1)
            bearing = atan2(y, x)
            return (degrees(bearing) + 360) % 360

        # Define a threshold for being "at" the gate (e.g., 20 meters)
        threshold = 20  # meters

        # Calculate smoothed bearings for the entire track
        bearings = []
        window_size = 5  # Points to use for smoother bearing calculation

        for i in range(len(valid_coords)):
            if i < window_size:
                # For the first points, use forward-looking window
                if i + window_size < len(valid_coords):
                    start_lat = valid_coords.iloc[i]["position_lat"]
                    start_lon = valid_coords.iloc[i]["position_long"]
                    end_lat = valid_coords.iloc[i + window_size]["position_lat"]
                    end_lon = valid_coords.iloc[i + window_size]["position_long"]
                    bearings.append(
                        calculate_bearing(start_lat, start_lon, end_lat, end_lon)
                    )
                else:
                    # Fallback if we don't have enough points ahead
                    if i + 1 < len(valid_coords):
                        start_lat = valid_coords.iloc[i]["position_lat"]
                        start_lon = valid_coords.iloc[i]["position_long"]
                        end_lat = valid_coords.iloc[i + 1]["position_lat"]
                        end_lon = valid_coords.iloc[i + 1]["position_long"]
                        bearings.append(
                            calculate_bearing(start_lat, start_lon, end_lat, end_lon)
                        )
                    else:
                        # Just copy the previous bearing if we're at the last point
                        bearings.append(bearings[-1] if bearings else 0)
            elif i >= len(valid_coords) - window_size:
                # For the last points, use backward-looking window
                start_lat = valid_coords.iloc[i - window_size]["position_lat"]
                start_lon = valid_coords.iloc[i - window_size]["position_long"]
                end_lat = valid_coords.iloc[i]["position_lat"]
                end_lon = valid_coords.iloc[i]["position_long"]
                bearings.append(
                    calculate_bearing(start_lat, start_lon, end_lat, end_lon)
                )
            else:
                # For middle points, use points before and after
                start_lat = valid_coords.iloc[i - window_size // 2]["position_lat"]
                start_lon = valid_coords.iloc[i - window_size // 2]["position_long"]
                end_lat = valid_coords.iloc[i + window_size // 2]["position_lat"]
                end_lon = valid_coords.iloc[i + window_size // 2]["position_long"]
                bearings.append(
                    calculate_bearing(start_lat, start_lon, end_lat, end_lon)
                )

        # Find all points where we pass near the gate within the specified time range
        passings = []
        for i in range(start_time, min(end_time + 1, len(valid_coords))):
            if i >= len(valid_coords):
                continue

            point_lat = valid_coords.iloc[i]["position_lat"]
            point_lon = valid_coords.iloc[i]["position_long"]

            distance = haversine(gate_lat, gate_lon, point_lat, point_lon)

            if distance < threshold:
                direction = bearings[i] if i < len(bearings) else 0

                passings.append(
                    {
                        "index": i,
                        "distance": distance,
                        "direction": direction,
                        "timestamp": valid_coords.iloc[i]["timestamp"],
                    }
                )

        # Group nearby passings and keep only the closest point for each group
        grouped_passings = []
        current_group = []

        for passing in passings:
            if (
                not current_group or passing["index"] - current_group[-1]["index"] <= 5
            ):  # Points within 5 seconds are grouped
                current_group.append(passing)
            else:
                # Find the closest point in the group
                if current_group:
                    closest = min(current_group, key=lambda x: x["distance"])
                    grouped_passings.append(closest)
                current_group = [passing]

        # Add the last group if it exists
        if current_group:
            closest = min(current_group, key=lambda x: x["distance"])
            grouped_passings.append(closest)

        # Sort by time index
        grouped_passings.sort(key=lambda x: x["index"])

        return grouped_passings

    def detect_sections(self):
        """Detect sections based on gate passings with directional awareness"""
        self.detected_sections = []

        if not self.has_gps or not self.gate_sets:
            return

        # Update the calibration end first
        self.update_calibration_end()

        # Process each gate set
        for gate_set_idx, gate_set in enumerate(self.gate_sets):
            # Get gate positions
            gate_a_pos = gate_set["gate_a_pos"]
            gate_b_pos = gate_set["gate_b_pos"]

            # Determine time bounds for this gate set
            if gate_set_idx == 0:
                # First gate set uses trim_start to calibration_end
                start_bound = self.trim_start
            else:
                # Subsequent gate sets use the previous gate B to calibration_end
                prev_gate_b = self.gate_sets[gate_set_idx - 1]["gate_b_pos"]
                start_bound = prev_gate_b
            end_bound = self.trim_end

            # Get passings of gate A in this range
            gate_a_passings = self.detect_gate_passings(
                gate_a_pos, start_bound, end_bound
            )

            # If no gate A passings, skip this gate set
            if not gate_a_passings:
                continue

            # First passing of gate A defines the direction
            first_a_passing = gate_a_passings[0]
            ref_direction_a = first_a_passing["direction"]

            # Store reference direction in gate set
            gate_set["direction"] = ref_direction_a
            gate_set["calibration_point"] = first_a_passing["index"]

            # Track sections from this gate set
            gate_sections = []

            # Look for gate B passings after each valid A passing
            for a_passing in gate_a_passings:
                # Check if this A passing is in the reference direction
                a_dir_diff = abs(a_passing["direction"] - ref_direction_a)
                a_dir_diff = min(a_dir_diff, 360 - a_dir_diff)

                if a_dir_diff > 45:  # Not same direction, skip
                    continue

                # Get gate B passings after this A passing
                a_time = a_passing["index"]
                gate_b_passings = self.detect_gate_passings(
                    gate_b_pos, a_time, end_bound
                )

                # Find the first B passing in the same direction
                for b_passing in gate_b_passings:
                    first_b_passing = gate_b_passings[0]
                    ref_direction_b = first_b_passing["direction"]

                    b_dir_diff = abs(b_passing["direction"] - ref_direction_b)
                    b_dir_diff = min(b_dir_diff, 360 - b_dir_diff)

                    if b_dir_diff <= 45:  # Same direction
                        # Found valid A->B section
                        section = {
                            "gate_set": gate_set_idx,
                            "section_id": f"{gate_set_idx+1}",
                            "start_idx": a_passing["index"],
                            "end_idx": b_passing["index"],
                            "start_time": a_passing["timestamp"],
                            "end_time": b_passing["timestamp"],
                            "start_direction": a_passing["direction"],
                            "end_direction": b_passing["direction"],
                        }

                        # Calculate duration and distance for this section
                        section["duration"] = (
                            section["end_time"] - section["start_time"]
                        ).total_seconds()

                        # Get distance between points
                        distance = 0
                        try:
                            distance = (
                                self.merged_data.iloc[b_passing["index"]]["distance"]
                                - self.merged_data.iloc[a_passing["index"]]["distance"]
                            ) / 1000  # Convert to km
                        except (KeyError, IndexError):
                            # Fallback if distance field isn't available
                            # Use speed * time approximation
                            avg_speed = 0
                            if "speed" in self.merged_data.columns:
                                speed_values = self.merged_data.iloc[
                                    a_passing["index"] : b_passing["index"] + 1
                                ]["speed"].values
                                if len(speed_values) > 0:
                                    avg_speed = np.mean(speed_values)

                            distance = (
                                avg_speed * section["duration"]
                            ) / 1000  # m/s * s / 1000 = km

                        section["distance"] = max(0, distance)  # Ensure non-negative

                        # Add to detected sections
                        gate_sections.append(section)
                        break  # Only use the first valid B passing

            # Store sections in the gate set
            gate_set["sections"] = gate_sections

            # Add all sections to the master list
            self.detected_sections.extend(gate_sections)

        # Update the section table
        self.update_section_table()

    def update_section_table(self):
        """Update the section table with detected sections"""
        self.section_table.setRowCount(len(self.detected_sections))

        for row, section in enumerate(self.detected_sections):
            # Checkbox for selection
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # All sections selected by default
            checkbox.stateChanged.connect(self.update_plots)
            self.section_table.setCellWidget(row, 0, checkbox)

            # Section number
            self.section_table.setItem(
                row, 1, QTableWidgetItem(str(section["section_id"]))
            )

            # Gate set
            self.section_table.setItem(
                row, 2, QTableWidgetItem(f"Gate {section['gate_set']+1}")
            )

            # Duration
            duration_mins = int(section["duration"] // 60)
            duration_secs = int(section["duration"] % 60)
            self.section_table.setItem(
                row, 3, QTableWidgetItem(f"{duration_mins:02d}:{duration_secs:02d}")
            )

            # Distance
            self.section_table.setItem(
                row, 4, QTableWidgetItem(f"{section['distance']:.2f} km")
            )

            # Make cells read-only
            for col in range(1, 5):
                item = self.section_table.item(row, col)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

        # Resize columns to fit content
        header = self.section_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, 5):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

    def get_selected_section_indices(self):
        """Get indices of selected sections in the table"""
        selected_indices = []

        for row in range(self.section_table.rowCount()):
            checkbox = self.section_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                selected_indices.append(row)

        return selected_indices

    def calculate_ve(self):
        """Calculate virtual elevation for each detected section"""
        self.section_ve_profiles = []
        self.section_distances = []

        # Store actual elevation profiles for mean calculation
        self.all_actual_elevations = (
            {}
        )  # Key = gate_set_idx, Value = list of (distances, elevations)
        self.mean_actual_elevations = (
            {}
        )  # Key = gate_set_idx, Value = (distances, mean_elevation)

        if not self.detected_sections:
            return

        # Check if we have actual elevation data
        has_elevation = (
            "altitude" in self.merged_data.columns
            and not self.merged_data["altitude"].isna().all()
        )

        # For each detected section, calculate VE
        for section_idx, section in enumerate(self.detected_sections):
            gate_set_idx = section["gate_set"]
            start_idx = section["start_idx"]
            end_idx = section["end_idx"]

            # Skip invalid indices
            if (
                start_idx >= end_idx
                or start_idx < 0
                or end_idx >= len(self.merged_data)
            ):
                continue

            # Extract section data
            section_data = self.merged_data.iloc[start_idx : end_idx + 1].copy()

            # Create VE calculator for this section
            ve_calculator = VirtualElevation(section_data, self.params)

            # Calculate VE
            ve = ve_calculator.calculate_ve(self.current_cda, self.current_crr)

            # Get distance data
            if "distance" in section_data.columns:
                # Use distance directly from FIT file (in meters)
                distances = section_data["distance"].values

                # Make distances relative to start
                start_distance = distances[0]
                distances = distances - start_distance

                # Convert to kilometers
                distances = distances / 1000
            else:
                # Fallback if no distance data
                distances = np.linspace(0, section["distance"], len(ve))

            # Store section data
            self.section_ve_profiles.append(ve)
            self.section_distances.append(distances)

            # If we have elevation data, collect it for mean calculation
            if has_elevation:
                # Extract actual elevation for this section
                actual_elevation = section_data["altitude"].values

                # Store by gate set index for mean calculation
                if gate_set_idx not in self.all_actual_elevations:
                    self.all_actual_elevations[gate_set_idx] = []

                self.all_actual_elevations[gate_set_idx].append(
                    (distances, actual_elevation)
                )

        # Calculate mean actual elevation profile for each gate set
        if has_elevation:
            for gate_set_idx, elevations in self.all_actual_elevations.items():
                if not elevations:
                    continue

                # Find max distance for this gate set
                max_dist = 0
                for distances, _ in elevations:
                    if len(distances) > 0:
                        max_dist = max(max_dist, distances[-1])

                # Create reference distance array (1m intervals)
                ref_distance_m = np.arange(0, int(max_dist * 1000) + 1, 1)
                ref_distance_km = ref_distance_m / 1000

                # Arrays for accumulating elevation values
                elevation_sum = np.zeros_like(ref_distance_m, dtype=float)
                elevation_count = np.zeros_like(ref_distance_m, dtype=int)

                # Interpolate each elevation profile onto reference distance
                for distances, elevations in elevations:
                    # Convert distances to meters for indexing
                    distances_m = distances * 1000

                    # Interpolate onto reference distances
                    interp_elevation = np.interp(
                        ref_distance_m,
                        distances_m,
                        elevations,
                        left=np.nan,
                        right=np.nan,
                    )

                    # Add to sum and count non-NaN values
                    valid_mask = ~np.isnan(interp_elevation)
                    elevation_sum[valid_mask] += interp_elevation[valid_mask]
                    elevation_count[valid_mask] += 1

                # Calculate mean elevation
                mean_elevation = np.zeros_like(ref_distance_m, dtype=float)
                valid_points = elevation_count > 0
                mean_elevation[valid_points] = (
                    elevation_sum[valid_points] / elevation_count[valid_points]
                )

                # Store mean elevation profile
                self.mean_actual_elevations[gate_set_idx] = (
                    ref_distance_km,
                    mean_elevation,
                )

    def update_plots(self):
        """Update the virtual elevation plots"""
        # Clear previous plots
        self.fig_canvas.axes.clear()

        # Create figure with two subplots
        self.fig_canvas.fig.clear()
        gs = self.fig_canvas.fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax1 = self.fig_canvas.fig.add_subplot(gs[0])
        ax2 = self.fig_canvas.fig.add_subplot(gs[1])

        # Get selected section indices
        selected_indices = self.get_selected_section_indices()

        if not selected_indices or not self.section_ve_profiles:
            # No sections selected or detected
            ax1.text(
                0.5,
                0.5,
                "No gate sections detected or selected",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax2.text(
                0.5,
                0.5,
                "No residuals to display",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
        else:
            # Define a color palette for different gate sets
            gate_colors = {
                0: "#1f77b4",  # Blue
                1: "#ff7f0e",  # Orange
                2: "#2ca02c",  # Green
                3: "#d62728",  # Red
                4: "#9467bd",  # Purple
                5: "#8c564b",  # Brown
            }

            # Organize sections by gate set
            gate_set_sections = {}
            for i, section_idx in enumerate(selected_indices):
                if section_idx < len(self.detected_sections):
                    section = self.detected_sections[section_idx]
                    gate_set_idx = section["gate_set"]

                    if gate_set_idx not in gate_set_sections:
                        gate_set_sections[gate_set_idx] = []

                    gate_set_sections[gate_set_idx].append(section_idx)

            # Variables for plotting
            legend_handles = []
            legend_labels = []

            # Variables for error calculation
            total_error = 0

            # Calculate section offsets for plotting consecutive sections
            section_offsets = {}
            current_offset = 0

            # First calculate offsets for each gate set
            for gate_set_idx in sorted(gate_set_sections.keys()):
                section_offsets[gate_set_idx] = current_offset

                # Get the max distance for this gate set's sections
                max_dist = 0
                for section_idx in gate_set_sections[gate_set_idx]:
                    if section_idx < len(self.section_distances):
                        distances = self.section_distances[section_idx]
                        if len(distances) > 0:
                            max_dist = max(max_dist, distances[-1])

                # Update offset for next gate set
                current_offset += max_dist

            # Plot mean actual elevation profiles first
            for gate_set_idx, offset in section_offsets.items():
                if gate_set_idx in self.mean_actual_elevations:
                    distances, elevations = self.mean_actual_elevations[gate_set_idx]

                    # Offset distances
                    offset_distances = distances + offset

                    # Plot mean elevation
                    ax1.plot(
                        offset_distances,
                        elevations,
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                        label=(
                            f"Mean Actual - Gate {gate_set_idx+1}"
                            if gate_set_idx == 0
                            else "_nolegend_"
                        ),
                    )

            # Now plot each section's VE
            all_residuals = []

            for gate_set_idx, section_indices in gate_set_sections.items():
                offset = section_offsets[gate_set_idx]
                color = gate_colors.get(gate_set_idx, "#1f77b4")  # Default blue

                for i, section_idx in enumerate(section_indices):
                    if section_idx >= len(
                        self.section_ve_profiles
                    ) or section_idx >= len(self.section_distances):
                        continue

                    ve = self.section_ve_profiles[section_idx]
                    distances = self.section_distances[section_idx]

                    # Skip if no data
                    if len(ve) == 0 or len(distances) == 0:
                        continue

                    # Offset distances for consecutive plotting
                    offset_distances = distances + offset

                    # Calibrate VE if we have mean actual elevation
                    if gate_set_idx in self.mean_actual_elevations:
                        mean_distances, mean_elevations = self.mean_actual_elevations[
                            gate_set_idx
                        ]

                        # Get elevation at start (gate A)
                        if len(mean_elevations) > 0:
                            start_elevation = mean_elevations[0]

                            # Calibrate VE to match actual at start
                            calibrated_ve = ve - ve[0] + start_elevation
                        else:
                            calibrated_ve = ve
                    else:
                        calibrated_ve = ve

                    # Plot calibrated VE
                    line = ax1.plot(
                        offset_distances,
                        calibrated_ve,
                        color=color,
                        linewidth=4,
                        alpha=(
                            0.8 if i == 0 else 0.6
                        ),  # First passing of each gate set is more opaque
                    )[0]

                    # Add to legend only for first passing of each gate set
                    if i == 0:
                        legend_handles.append(line)
                        legend_labels.append(
                            f"Gate {gate_set_idx+1}: {self.detected_sections[section_idx]['section_id']}"
                        )

                    # Calculate residuals if we have mean elevation
                    if gate_set_idx in self.mean_actual_elevations:
                        mean_distances, mean_elevations = self.mean_actual_elevations[
                            gate_set_idx
                        ]

                        # Interpolate mean elevation to section distances
                        reference_elevation = np.interp(
                            distances,
                            mean_distances,
                            mean_elevations,
                            left=np.nan,
                            right=np.nan,
                        )

                        # Calculate residuals
                        valid_mask = ~np.isnan(reference_elevation)
                        if np.any(valid_mask):
                            residual_distances = offset_distances[valid_mask]
                            residuals = (
                                calibrated_ve[valid_mask]
                                - reference_elevation[valid_mask]
                            )

                            # Store for plotting
                            all_residuals.append(
                                (
                                    residual_distances,
                                    residuals,
                                    color,
                                    0.8 if i == 0 else 0.6,
                                )
                            )

                            # Calculate error (end point vs. reference)
                            if len(calibrated_ve) > 0 and len(reference_elevation) > 0:
                                end_error = abs(
                                    calibrated_ve[valid_mask][-1]
                                    - reference_elevation[valid_mask][-1]
                                )
                                total_error += end_error

            # Plot all residuals
            for residual_distances, residuals, color, alpha in all_residuals:
                ax2.plot(
                    residual_distances, residuals, color=color, linewidth=3.5, alpha=alpha
                )

            # Add zero line in residuals plot
            ax2.axhline(y=0, color="black", linestyle="-")

            # Set axis limits
            if section_offsets:
                max_offset = max(offset for offset in section_offsets.values())

                # Add max distance of last gate set
                last_gate = max(section_offsets.keys())
                max_dist_last = 0
                for section_idx in gate_set_sections.get(last_gate, []):
                    if section_idx < len(self.section_distances):
                        distances = self.section_distances[section_idx]
                        if len(distances) > 0:
                            max_dist_last = max(max_dist_last, distances[-1])

                plot_width = max_offset + max_dist_last

                # Set axis limits with 2% margin
                margin = plot_width * 0.02
                ax1.set_xlim(-margin, plot_width + margin)
                ax2.set_xlim(-margin, plot_width + margin)

            # Add subtle grid lines to both plots
            ax1.grid(True, linestyle="--", alpha=0.3, color="#cccccc")
            ax2.grid(True, linestyle="--", alpha=0.3, color="#cccccc")

            # Set titles and labels
            ax1.set_ylabel("Elevation (m)")
            ax1.set_title("GPS Gate Virtual Elevation Profiles")

            # Handle legend - use 2 columns if more than 4 sections
            if len(legend_handles) > 4:
                ax1.legend(
                    legend_handles,
                    legend_labels,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    ncol=1 + len(legend_handles) // 8,
                )
            else:
                ax1.legend(legend_handles, legend_labels)

            ax2.set_xlabel("Distance (km)")
            ax2.set_ylabel("Residuals (m)")
            ax2.set_title("Residuals (Virtual - Mean Actual Elevation)")

            # Add CdA, Crr, and error values
            cda_str = f"CdA: {self.current_cda:.3f}"
            crr_str = f"Crr: {self.current_crr:.4f}"
            error_str = f"Total Error: {total_error:.2f} m"

            self.fig_canvas.fig.text(
                0.01,
                0.99,
                cda_str + "\n" + crr_str + "\n" + error_str,
                verticalalignment="top",
                horizontalalignment="left",
                transform=self.fig_canvas.fig.transFigure,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )

        self.fig_canvas.fig.tight_layout()
        self.fig_canvas.draw()

    def on_gate_a_changed(self, gate_index, value):
        """Handle Gate A slider value change"""
        # Ensure gate A is before gate B
        if value >= self.gate_sets[gate_index]["gate_b_pos"]:
            # Move gate B as well
            self.gate_sets[gate_index]["gate_b_pos"] = value + 1
            self.gate_controls[gate_index].gate_b_slider.setValue(value + 1)
            self.gate_controls[gate_index].gate_b_label.setText(f"{value + 1} s")

        # Update gate A position
        self.gate_sets[gate_index]["gate_a_pos"] = value
        self.gate_controls[gate_index].gate_a_label.setText(f"{value} s")

        # If this is the first gate, update calibration end
        if gate_index == 0:
            self.update_calibration_end()

        # Update subsequent gates if needed
        for i in range(gate_index + 1, len(self.gate_sets)):
            if self.gate_sets[i]["gate_a_pos"] <= value:
                # Move next gate A after this gate B
                next_a_pos = self.gate_sets[gate_index]["gate_b_pos"] + 1
                self.gate_sets[i]["gate_a_pos"] = next_a_pos
                self.gate_controls[i].gate_a_slider.setValue(next_a_pos)
                self.gate_controls[i].gate_a_label.setText(f"{next_a_pos} s")

                # Move gate B if needed
                if self.gate_sets[i]["gate_b_pos"] <= next_a_pos:
                    next_b_pos = next_a_pos + 1
                    self.gate_sets[i]["gate_b_pos"] = next_b_pos
                    self.gate_controls[i].gate_b_slider.setValue(next_b_pos)
                    self.gate_controls[i].gate_b_label.setText(f"{next_b_pos} s")

        # Detect sections with new gate positions
        self.detect_sections()

        # Calculate VE for new sections
        self.calculate_ve()
        self.update_plots()

        # Update map
        self.map_widget.set_gate_sets(self.gate_sets, self.detected_sections)
        self.map_widget.update()

        # Update control min/max values
        self.update_gate_controls()

    def on_gate_b_changed(self, gate_index, value):
        """Handle Gate B slider value change"""
        # Ensure gate B is after gate A
        if value <= self.gate_sets[gate_index]["gate_a_pos"]:
            value = self.gate_sets[gate_index]["gate_a_pos"] + 1
            self.gate_controls[gate_index].gate_b_slider.setValue(value)

        # Update gate B position
        self.gate_sets[gate_index]["gate_b_pos"] = value
        self.gate_controls[gate_index].gate_b_label.setText(f"{value} s")

        # Update subsequent gates if needed
        for i in range(gate_index + 1, len(self.gate_sets)):
            if self.gate_sets[i]["gate_a_pos"] <= value:
                # Move next gate A after this gate B
                next_a_pos = value + 1
                self.gate_sets[i]["gate_a_pos"] = next_a_pos
                self.gate_controls[i].gate_a_slider.setValue(next_a_pos)
                self.gate_controls[i].gate_a_label.setText(f"{next_a_pos} s")

                # Move gate B if needed
                if self.gate_sets[i]["gate_b_pos"] <= next_a_pos:
                    next_b_pos = next_a_pos + 1
                    self.gate_sets[i]["gate_b_pos"] = next_b_pos
                    self.gate_controls[i].gate_b_slider.setValue(next_b_pos)
                    self.gate_controls[i].gate_b_label.setText(f"{next_b_pos} s")

        # Detect sections with new gate positions
        self.detect_sections()

        # Calculate VE for new sections
        self.calculate_ve()
        self.update_plots()

        # Update map
        self.map_widget.set_gate_sets(self.gate_sets, self.detected_sections)
        self.map_widget.update()

        # Update control min/max values
        self.update_gate_controls()

    def on_trim_start_changed(self, value):
        """Handle trim start slider value change"""
        # Ensure trim_start < trim_end - 30
        if value >= self.trim_end_slider.value() - 30:
            self.trim_start_slider.setValue(self.trim_end_slider.value() - 30)
            return

        self.trim_start = value
        self.trim_start_label.setText(f"{value} s")

        # Update gate A slider bounds
        for i, gate_control in enumerate(self.gate_controls):
            # If gate A is now before trim_start, move it
            if self.gate_sets[i]["gate_a_pos"] < value:
                self.gate_sets[i]["gate_a_pos"] = value
                gate_control.gate_a_slider.setValue(value)
                gate_control.gate_a_label.setText(f"{value} s")

                # If gate B is now before gate A, move it too
                if self.gate_sets[i]["gate_b_pos"] <= value:
                    self.gate_sets[i]["gate_b_pos"] = value + 1
                    gate_control.gate_b_slider.setValue(value + 1)
                    gate_control.gate_b_label.setText(f"{value + 1} s")

        # Update calibration end
        self.update_calibration_end()

        # Detect sections based on new trim values
        self.detect_sections()

        # Recalculate VE metrics and update plots
        self.calculate_ve()
        self.update_plots()

        # Update map to show trim points
        self.map_widget.set_trim_start(self.trim_start)
        self.map_widget.set_gate_sets(self.gate_sets, self.detected_sections)
        self.map_widget.update()

        # Update all gate controls
        self.update_gate_controls()

    def on_trim_end_changed(self, value):
        """Handle trim end slider value change"""
        # Ensure trim_end > trim_start + 30
        if value <= self.trim_start_slider.value() + 30:
            self.trim_end_slider.setValue(self.trim_start_slider.value() + 30)
            return

        self.trim_end = value
        self.trim_end_label.setText(f"{value} s")

        # Update gate B slider bounds
        for i, gate_control in enumerate(self.gate_controls):
            # If gate B is now after trim_end, move it
            if self.gate_sets[i]["gate_b_pos"] > value:
                self.gate_sets[i]["gate_b_pos"] = value
                gate_control.gate_b_slider.setValue(value)
                gate_control.gate_b_label.setText(f"{value} s")

                # If gate A is now after gate B, move it too
                if self.gate_sets[i]["gate_a_pos"] >= value:
                    self.gate_sets[i]["gate_a_pos"] = value - 1
                    gate_control.gate_a_slider.setValue(value - 1)
                    gate_control.gate_a_label.setText(f"{value - 1} s")

        # Update calibration end
        self.update_calibration_end()

        # Detect sections based on new trim values
        self.detect_sections()

        # Recalculate VE metrics and update plots
        self.calculate_ve()
        self.update_plots()

        # Update map to show trim points
        self.map_widget.set_trim_end(self.trim_end)
        self.map_widget.set_gate_sets(self.gate_sets, self.detected_sections)
        self.map_widget.update()

        # Update all gate controls
        self.update_gate_controls()

    def on_cda_changed(self, value):
        """Handle CdA slider value change"""
        self.current_cda = value / 1000.0
        self.cda_label.setText(f"{self.current_cda:.3f}")

        # Recalculate VE and update plots
        self.calculate_ve()
        self.update_plots()
        self.update_config_text()

    def on_crr_changed(self, value):
        """Handle Crr slider value change"""
        self.current_crr = value / 10000.0
        self.crr_label.setText(f"{self.current_crr:.4f}")

        # Recalculate VE and update plots
        self.calculate_ve()
        self.update_plots()
        self.update_config_text()

    def make_json_serializable(self, obj):
        """Convert objects that aren't JSON serializable to appropriate formats"""
        if isinstance(obj, dict):
            return {
                key: self.make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self.make_json_serializable(item) for item in obj]
        elif hasattr(obj, "isoformat"):  # For datetime and Pandas Timestamp objects
            return obj.isoformat()
        elif pd.isna(obj):  # Handle NaN values
            return None
        else:
            return obj

    def save_results(self):
        """Save analysis results"""
        # Get config name
        config_name = self.config_name.text().strip()
        if not config_name:
            QMessageBox.warning(
                self, "Missing Information", "Please enter a configuration name."
            )
            return

        # Create results directory if it doesn't exist
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # Get file basename
        file_basename = Path(self.fit_file.filename).stem

        # Create CSV file path
        csv_path = os.path.join(
            self.result_dir, f"{file_basename}_gps_gate_results.csv"
        )

        # Prepare data for CSV
        lap_str = "_".join(map(str, sorted(self.selected_laps)))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        selected_section_indices = self.get_selected_section_indices()

        # Calculate total error for selected sections
        total_error = 0
        for i in selected_section_indices:
            if i < len(self.detected_sections):
                section = self.detected_sections[i]
                gate_set_idx = section["gate_set"]
                section_idx = i

                if section_idx < len(self.section_ve_profiles):
                    ve = self.section_ve_profiles[section_idx]

                    if gate_set_idx in self.mean_actual_elevations:
                        # Calculate error vs mean elevation
                        mean_distances, mean_elevations = self.mean_actual_elevations[
                            gate_set_idx
                        ]

                        if len(ve) > 0 and len(mean_elevations) > 0:
                            # Get end point error
                            section_distances = self.section_distances[section_idx]
                            end_dist = (
                                section_distances[-1]
                                if len(section_distances) > 0
                                else 0
                            )

                            # Get elevation at end point
                            end_elev_ref = np.interp(
                                end_dist, mean_distances, mean_elevations
                            )

                            # Calibrate VE
                            ve_cal = ve - ve[0] + mean_elevations[0]

                            # Calculate error
                            end_error = abs(ve_cal[-1] - end_elev_ref)
                            total_error += end_error

        # Then modify part of the save_results method where we save gate_sets data:
        # When creating gate_sets_data in save_results method:
        gate_sets_data = []
        for i, gate_set in enumerate(self.gate_sets):
            # Create a clean copy without Timestamp objects
            clean_gate_set = {
                "index": i,
                "gate_a_pos": gate_set["gate_a_pos"],
                "gate_b_pos": gate_set["gate_b_pos"],
                "direction": gate_set.get("direction"),
            }
            # Don't include calibration_point or sections which might have Timestamp objects
            gate_sets_data.append(clean_gate_set)

        result_row = {
            "timestamp": timestamp,
            "laps": lap_str,
            "config_name": config_name,
            "cda": self.current_cda,
            "crr": self.current_crr,
            "system_mass": self.params.get("system_mass", 90),
            "rho": self.params.get("rho", 1.2),
            "eta": self.params.get("eta", 0.98),
            "wind_speed": self.params.get("wind_speed", 0),
            "wind_direction": self.params.get("wind_direction", 0),
            "trim_start": self.trim_start,
            "trim_end": self.trim_end,
            "gate_sets": json.dumps(gate_sets_data),
            "detected_sections": len(self.detected_sections),
            "selected_sections": len(selected_section_indices),
            "total_error": total_error,
        }

        # Check if file exists
        file_exists = os.path.isfile(csv_path)

        # If file exists, read existing data
        existing_data = []
        if file_exists:
            with open(csv_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                existing_data = list(reader)

        # Remove existing row with same lap selection and config name if present
        existing_data = [
            row
            for row in existing_data
            if not (
                row.get("laps") == lap_str and row.get("config_name") == config_name
            )
        ]

        # Add new row
        existing_data.append(result_row)

        # Sort by lap selection
        existing_data.sort(key=lambda x: (x.get("laps", ""), x.get("config_name", "")))

        # Write all data back to CSV
        header = list(result_row.keys())
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(existing_data)

        # Save plot
        plot_dir = os.path.join(self.result_dir, "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plot_filename = f"{file_basename}_gps_gate_laps_{lap_str}_{config_name}.png"
        plot_path = os.path.join(plot_dir, plot_filename)

        self.fig_canvas.fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Save trim and gate positions to file settings
        file_settings = self.settings.get_file_settings(self.fit_file.filename)

        # Initialize trim_settings if it doesn't exist
        if "trim_settings" not in file_settings:
            file_settings["trim_settings"] = {}

        clean_gate_sets = []
        for gate_set in self.gate_sets:
            clean_gate_set = {
                "gate_a_pos": gate_set["gate_a_pos"],
                "gate_b_pos": gate_set["gate_b_pos"],
                "direction": gate_set.get("direction"),
            }
            clean_gate_sets.append(clean_gate_set)

        # Save trim settings for this lap combination
        file_settings["trim_settings"][self.settings_key] = {
            "trim_start": self.trim_start,
            "trim_end": self.trim_end,
            "gate_sets": clean_gate_sets,
            "cda": self.current_cda,
            "crr": self.current_crr,
        }

        # Save updated file settings
        self.settings.save_file_settings(self.fit_file.filename, file_settings)

        # Show success message
        QMessageBox.information(
            self,
            "Results Saved",
            f"Analysis results saved successfully to:\n{csv_path}\n\nPlot saved to:\n{plot_path}",
        )

    def back_to_selection(self):
        """Return to lap selection window"""
        from ui.analysis_window import AnalysisWindow

        self.analysis_window = AnalysisWindow(self.fit_file, self.settings)
        self.analysis_window.show()
        self.close()
