import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QApplication,
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
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from models.virtual_elevation import VirtualElevation
from ui.async_worker import AsyncWorker
from ui.map_widget import MapMode, MapWidget
from ui.slider_textbox import SliderTextBox
from ui.ve_plot import VEFigure, VEPlotLabel, VEPlotSaver


class VEWorker(AsyncWorker):
    INPUT_KEYS = [
        "trim_start",
        "trim_end",
        "current_cda",
        "current_crr",
        "gps_marker_pos",
        "selected_lap_indices",
        "plot_size_info",
        "wind_source",  # Still needed to determine constant vs FIT mode
    ]

    RESULT_KEYS = [
        "lap_actual_elevation",
        "lap_ve_profiles",
        "lap_distances",
        "detected_laps",
        "mean_actual_elevation",
        "mean_actual_distance_km",
        "fig_res",
    ]

    def __init__(self, merged_data, params):
        super(VEWorker, self).__init__()
        self.merged_data = merged_data
        self.params = params
        self.ve_valid = False
        self.detected_laps = []
        self.mean_actual_distance_km = 0

    def _process_value(self, values: dict):
        for key in VEWorker.INPUT_KEYS:
            if key in values:
                setattr(self, key, values[key])

        detect_laps = False
        # Detect laps based on GPS marker
        if values["detect_laps"] or not self.detected_laps:
            self.detect_laps()
            detect_laps = True

        # Recalculate VE metrics and update plots
        if values["update_ve"] or not self.ve_valid:
            self.calculate_ve()
            self.ve_valid = True

        self.update_plots()

        out_values = {}
        for key in VEWorker.RESULT_KEYS:
            out_values[key] = getattr(self, key)
        out_values["detect_laps"] = detect_laps

        return out_values

    def detect_laps(self):
        """Detect laps based on GPS marker with improved directional detection and overlap prevention"""
        self.detected_laps = []

        # Get valid GPS coordinates
        valid_coords = self.merged_data.dropna(subset=["position_lat", "position_long"])
        if valid_coords.empty:
            return

        # Get the GPS marker position
        marker_idx = self.gps_marker_pos
        if marker_idx < 0 or marker_idx >= len(valid_coords):
            return

        marker_lat = valid_coords.iloc[marker_idx]["position_lat"]
        marker_lon = valid_coords.iloc[marker_idx]["position_long"]

        # Calculate distance from each point to the marker
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

        # Define a threshold for being "at" the marker (e.g., 20 meters)
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

        # Find all points where we pass near the marker within the trimmed region
        passings = []
        for i in range(self.trim_start, min(self.trim_end + 1, len(valid_coords))):
            if i >= len(valid_coords):
                continue

            point_lat = valid_coords.iloc[i]["position_lat"]
            point_lon = valid_coords.iloc[i]["position_long"]

            distance = haversine(marker_lat, marker_lon, point_lat, point_lon)

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

        # Find first reference passing direction
        reference_direction = None
        if grouped_passings:
            reference_direction = grouped_passings[0]["direction"]

        # Use a smaller angle threshold (30 degrees) for more precise direction matching
        angle_threshold = 30

        # Process consecutive passings in order
        # This prevents overlapping laps by ensuring a lap must end before a new one can start
        if len(grouped_passings) >= 2:
            # Sort passings by index
            grouped_passings.sort(key=lambda x: x["index"])

            # Start with the first passing
            lap_start = 0

            # Process all remaining passings as potential lap ends
            while lap_start < len(grouped_passings) - 1:
                lap_end = None

                # Find the next passing in the same direction
                for j in range(lap_start + 1, len(grouped_passings)):
                    # Calculate directional difference to the current passing
                    dir_diff = abs(
                        grouped_passings[lap_start]["direction"]
                        - grouped_passings[j]["direction"]
                    )
                    dir_diff = min(dir_diff, 360 - dir_diff)

                    # Check if it's in the same direction
                    if dir_diff < angle_threshold:
                        # Found a valid lap end
                        lap_end = j
                        break

                # If we found a valid lap end, create the lap
                if lap_end is not None:
                    start_passing = grouped_passings[lap_start]
                    end_passing = grouped_passings[lap_end]

                    lap = {
                        "start_idx": start_passing["index"],
                        "end_idx": end_passing["index"],
                        "start_time": start_passing["timestamp"],
                        "end_time": end_passing["timestamp"],
                        "start_direction": start_passing["direction"],
                        "end_direction": end_passing["direction"],
                    }

                    # Calculate duration and distance for this lap
                    duration = (lap["end_time"] - lap["start_time"]).total_seconds()

                    # Sum distance between all points in the lap
                    distance = 0
                    distance = (
                        self.merged_data.iloc[end_passing["index"]]["distance"]
                        - self.merged_data.iloc[start_passing["index"]]["distance"]
                    )

                    lap["duration"] = duration
                    lap["distance"] = distance / 1000  # Convert to km

                    # Add the compass direction as a human-readable value
                    direction_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                    direction_idx = round(lap["start_direction"] / 45) % 8
                    lap["direction_name"] = direction_names[direction_idx]

                    self.detected_laps.append(lap)

                    # Move to the next lap start
                    lap_start = lap_end
                else:
                    # No more valid laps to be found
                    break

    def calculate_ve(self):
        """Calculate virtual elevation for each detected lap"""
        self.lap_ve_profiles = []
        self.lap_distances = []

        # New: Store actual elevation profiles for each lap and mean elevation
        self.lap_actual_elevation = []
        self.mean_actual_elevation = None


        if not self.detected_laps:
            return

        # Check if we have actual elevation data
        has_elevation = (
            "altitude" in self.merged_data.columns
            and not self.merged_data["altitude"].isna().all()
        )

        # Create a common distance array for resampling all laps
        max_lap_distance = 0
        count = 0
        for lap in self.detected_laps:
            if "distance" in lap:
                max_lap_distance = max(max_lap_distance, lap["distance"])

        # Create a reference distance array with 1-meter intervals
        if max_lap_distance > 0:
            reference_distance = np.linspace(
                0, max_lap_distance * 1000, int(max_lap_distance * 1000) + 1
            )
            reference_distance_km = (
                reference_distance / 1000
            )  # Store in km for plotting
        else:
            # Fallback if no distance data
            reference_distance = np.array([0])
            reference_distance_km = np.array([0])

        # For each detected lap, calculate VE and actual elevation profile
        for lap in self.detected_laps:
            start_idx = lap["start_idx"]
            end_idx = lap["end_idx"]

            # Extract lap data
            if (
                start_idx >= end_idx
                or start_idx < 0
                or end_idx >= len(self.merged_data)
            ):
                continue

            lap_data = self.merged_data.iloc[start_idx : end_idx + 1].copy()

            # Get current parameters
            lap_params = self.params.copy()

            # Set comparison mode flag
            lap_params["comparison_mode"] = self.wind_source == "compare"

            # For constant wind mode, remove air_speed/wind_speed columns so VE uses constant wind
            if self.wind_source == "constant":
                if "air_speed" in lap_data.columns:
                    lap_data = lap_data.drop(columns=["air_speed"])
                if "wind_speed" in lap_data.columns:
                    lap_data = lap_data.drop(columns=["wind_speed"])

            # Create a VE calculator for this lap
            # Wind/air speed data is automatically handled from DataFrame columns
            lap_ve_calculator = VirtualElevation(lap_data, lap_params)

            # Calculate VE
            ve = lap_ve_calculator.calculate_ve(self.current_cda, self.current_crr)

            # Get or calculate distances
            if "distance" in lap_data.columns:
                # Use distance directly from the FIT file (in meters)
                distances = lap_data["distance"].values

                # Make distances cumulative and relative to the start of this lap
                start_distance = distances[0]
                distances = distances - start_distance

                # Convert to kilometers
                distances = distances / 1000
            else:
                # Fallback if distance field isn't available
                distances = [0]  # Start at 0

                # Use speed data if available
                if "speed" in lap_data.columns:
                    speeds = lap_data["speed"].values  # m/s
                    for i in range(1, len(speeds)):
                        # Calculate distance as speed * time (assuming 1 second intervals)
                        distance = speeds[i] * 1
                        distances.append(distances[-1] + distance)
                else:
                    # Use lap distance and divide evenly
                    distances = np.linspace(0, lap["distance"] * 1000, len(ve))

                # Convert to kilometers for plotting
                distances = np.array(distances) / 1000

            # Store the original VE and distances
            self.lap_ve_profiles.append(ve)
            self.lap_distances.append(distances)

            # If we have elevation data, extract the actual elevation profile
            if has_elevation and not self.params["velodrome"]:
                # Extract actual elevation profile for this lap
                actual_elevation = lap_data["altitude"].values

                # Store the actual elevation profile with its distances
                self.lap_actual_elevation.append((distances, actual_elevation))
            elif self.params["velodrome"]:
                actual_elevation = np.zeros_like(lap_data["altitude"].values)
                self.lap_actual_elevation.append((distances, actual_elevation))

        # Calculate mean actual elevation profile if we have elevation data
        if has_elevation and self.lap_actual_elevation and not self.params["velodrome"]:
            # Initialize array to accumulate elevation values
            elevation_sum = np.zeros_like(reference_distance)
            elevation_count = np.zeros_like(reference_distance)

            # For each lap's actual elevation profile
            for distances, elevations in self.lap_actual_elevation:
                # Interpolate this lap's elevation onto the reference distance array
                interpolated_elevation = np.interp(
                    reference_distance,
                    distances * 1000,  # Convert km to m
                    elevations,
                    left=np.nan,
                    right=np.nan,
                )

                # Add to the sum and count non-NaN values
                valid_mask = ~np.isnan(interpolated_elevation)
                elevation_sum[valid_mask] += interpolated_elevation[valid_mask]
                elevation_count[valid_mask] += 1

            # Calculate mean, avoiding division by zero
            mean_elevation = np.zeros_like(reference_distance)
            valid_points = elevation_count > 0
            mean_elevation[valid_points] = (
                elevation_sum[valid_points] / elevation_count[valid_points]
            )

            # Store the mean actual elevation profile
            self.mean_actual_elevation = mean_elevation
            self.mean_actual_distance_km = reference_distance_km
        elif self.params["velodrome"]:
            self.mean_actual_elevation = np.zeros_like(reference_distance_km)
            self.mean_actual_distance_km = reference_distance_km

    def update_plots(self):
        """Update the virtual elevation plots"""
        ve_fig = VEFigure(self.plot_size_info)
        fig, ax1, ax2 = ve_fig.get_fig_axes()

        # Get selected lap indices
        selected_indices = self.selected_lap_indices

        if not selected_indices or not self.lap_ve_profiles:
            # No laps selected or detected
            ax1.text(
                0.5,
                0.5,
                "No laps detected or selected",
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
            # Define a muted color palette - soft colors that are distinguishable but not flashy
            muted_colors = [
                "#4363d8",  # strong blue
                "#ffe119",  # bright yellow
                "#800000",  # dark maroon
                "#f58231",  # bright orange
                "#000075",  # dark navy blue
                "#dcbeff",  # light purple
                "#a9a9a9",  # medium gray
            ]

            # Determine maximum length for x-axis alignment
            max_dist = 0
            for i in selected_indices:
                if i < len(self.lap_ve_profiles) and i < len(self.lap_distances):
                    max_dist = max(max_dist, self.lap_distances[i][-1])

            # Plot mean actual elevation if available
            if (
                hasattr(self, "mean_actual_elevation")
                and self.mean_actual_elevation is not None
            ):
                # Plot only up to the max distance we need
                valid_indices = np.where(self.mean_actual_distance_km <= max_dist)[0]
                if len(valid_indices) > 0:
                    ax1.plot(
                        self.mean_actual_distance_km[valid_indices],
                        self.mean_actual_elevation[valid_indices],
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        label="Mean Actual Elevation",
                    )

            legend_handles = []
            legend_labels = []

            for i, lap_idx in enumerate(selected_indices):
                if lap_idx < len(self.lap_ve_profiles) and lap_idx < len(
                    self.lap_distances
                ):
                    ve = self.lap_ve_profiles[lap_idx]
                    distances = self.lap_distances[lap_idx]
                    color = muted_colors[i % len(muted_colors)]

                    # If we have mean actual elevation, calibrate VE to match at the start
                    if (
                        hasattr(self, "mean_actual_elevation")
                        and self.mean_actual_elevation is not None
                    ):
                        # Find the starting elevation value by interpolating the mean profile
                        start_elevation = self.mean_actual_elevation[0]

                        # Calibrate VE to match this starting point
                        calibrated_ve = ve - ve[0] + start_elevation
                    elif self.params["velodrome"]:
                        calibrated_ve = np.zeros_like(ve)
                    else:
                        # No calibration needed
                        calibrated_ve = ve

                    # Plot VE
                    (line1,) = ax1.plot(
                        distances, calibrated_ve, color=color, linewidth=4
                    )
                    legend_handles.append(line1)
                    legend_labels.append(f"Lap {lap_idx+1}")

                    # Plot residuals - difference from mean elevation profile
                    if (
                        hasattr(self, "mean_actual_elevation")
                        and self.mean_actual_elevation is not None
                    ):
                        # Interpolate mean elevation to match this lap's distance points
                        reference_elevation = np.interp(
                            distances,
                            self.mean_actual_distance_km,
                            self.mean_actual_elevation,
                            left=np.nan,
                            right=np.nan,
                        )

                        # Calculate residuals only for valid points
                        valid_mask = ~np.isnan(reference_elevation)
                        if np.any(valid_mask):
                            residual_distances = distances[valid_mask]
                            residuals = (
                                calibrated_ve[valid_mask]
                                - reference_elevation[valid_mask]
                            )
                            ax2.plot(
                                residual_distances,
                                residuals,
                                color=color,
                                linewidth=3.5,
                            )
                    else:
                        # If no reference elevation, use the original VE as residuals
                        ax2.plot(distances, calibrated_ve, color=color, linewidth=3.5)

            # Set axis limits
            ax1.set_xlim(0, max_dist)
            ax2.set_xlim(0, max_dist)

            # Add subtle grid lines to both plots
            ax1.grid(True, linestyle="--", alpha=0.3, color="#cccccc")
            ax2.grid(True, linestyle="--", alpha=0.3, color="#cccccc")

            # Add 0 line in residuals plot
            ax2.axhline(y=0, color="black", linestyle="-")

            # Calculate total absolute error - now relative to mean elevation profile
            total_error = 0
            for i in selected_indices:
                if i < len(self.lap_ve_profiles) and i < len(self.lap_distances):
                    ve = self.lap_ve_profiles[i]
                    total_error += abs(ve[-1] - ve[0])

            # Set titles and labels
            ax1.set_ylabel("Elevation (m)")
            ax1.set_title("Virtual Elevation Profiles")

            # Handle legend - use 2 columns if more than 5 laps
            if len(legend_handles) > 5:
                # Place legend outside the plot to the right
                ax1.legend(
                    legend_handles,
                    legend_labels,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    ncol=1
                    + len(legend_handles) // 10,  # Add a column for every 10 laps
                )
            else:
                ax1.legend(legend_handles, legend_labels)

            ax2.set_xlabel("Distance (km)")
            ax2.set_ylabel("Elevation (m)")

            # Update residuals title to reflect comparison with mean elevation
            if (
                hasattr(self, "mean_actual_elevation")
                and self.mean_actual_elevation is not None
            ):
                ax2.set_title("Residuals (Virtual - Mean Actual Elevation)")
            else:
                ax2.set_title("Residuals (should be zero at end of lap)")

            # Add CdA, Crr, and error values
            cda_str = f"CdA: {self.current_cda:.3f}"
            crr_str = f"Crr: {self.current_crr:.4f}"
            error_str = f"Total Error: {total_error:.2f} m"

            fig.text(
                0.01,
                0.99,
                cda_str + "\n" + crr_str + "\n" + error_str,
                verticalalignment="top",
                horizontalalignment="left",
                transform=fig.transFigure,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )

        self.fig_res = ve_fig.draw()


class GPSLapResult(QMainWindow):
    """Window for displaying GPS lap split analysis results"""

    def __init__(self, parent, fit_file, settings, selected_laps, params):
        super().__init__()
        self.parent = parent
        self.fit_file = fit_file
        self.settings = settings
        self.selected_laps = selected_laps
        self.params = params
        self.result_dir = settings.result_dir
        self.detected_laps = []
        self.lap_ve_profiles = []

        # Initialize these attributes BEFORE calling prepare_merged_data
        self.trim_start = 0
        self.trim_end = 0
        self.gps_marker_pos = 0

        # Prepare merged lap data
        self.prepare_merged_data()

        # Wind source settings
        self.has_fit_wind = self.fit_file.has_wind_speed_data()
        self.has_fit_air = self.fit_file.has_air_speed_data()
        self.wind_source = "constant"  # Default to constant wind

        # Ensure valid wind source if no FIT wind or air data
        if not (self.has_fit_wind or self.has_fit_air):
            self.wind_source = "constant"

        self.ve_worker = VEWorker(self.merged_data, self.get_current_params())
        self.ve_thread = QThread()
        self.ve_worker.moveToThread(self.ve_thread)
        self.ve_worker.resultReady.connect(self.on_ve_result_ready)
        self.ve_plot_saver = VEPlotSaver(
            VEWorker(self.merged_data, self.get_current_params()), self.ve_thread
        )
        self.ve_thread.start()
        QApplication.instance().aboutToQuit.connect(self.join_threads)

        # Get lap combination ID for settings
        self.lap_combo_id = "_".join(map(str, sorted(self.selected_laps)))
        self.settings_key = f"GPS_lap_{self.lap_combo_id}"

        # Try to load saved trim values for this lap combination
        file_settings = self.settings.get_file_settings(self.fit_file.filename)
        trim_settings = file_settings.get("trim_settings", {})
        saved_trim = trim_settings.get(self.settings_key, {})

        # Initialize UI values
        if saved_trim and "trim_start" in saved_trim and "trim_end" in saved_trim:
            # Use saved trim values if available
            self.trim_start = saved_trim["trim_start"]
            self.trim_end = saved_trim["trim_end"]
            # Use saved GPS marker position if available, otherwise force to 0s (trim_start)
            self.gps_marker_pos = saved_trim.get(
                "gps_marker_pos", self.trim_start
            )
        else:
            # Use defaults
            self.trim_start = 0
            self.trim_end = len(self.merged_data) - 1
            # Force GPS marker to start at 0s (trim_start)
            self.gps_marker_pos = self.trim_start

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

        self.async_update()
        self.ve_plot.sizeChanged.connect(self.on_plot_size_changed)

    def prepare_merged_data(self):
        """Extract and merge data for selected laps"""
        # Get records for selected laps
        self.merged_data = self.fit_file.get_records_for_laps(self.selected_laps)

        # Check if we have enough data
        if len(self.merged_data) < 30:
            error_msg = f"Not enough data points for analysis: {len(self.merged_data)} data points found (minimum 30 required)"
            if hasattr(self, "selected_laps") and self.selected_laps:
                error_msg += f"\nSelected laps: {self.selected_laps}"
            raise ValueError(error_msg)

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
            f'GPS Lap Split Analysis - Laps {", ".join(map(str, self.selected_laps))}'
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
        self.map_widget = MapWidget(MapMode.MARKER, self.merged_data, self.params)
        if self.map_widget.has_gps():
            self.map_widget.set_marker_pos(self.gps_marker_pos)
            self.map_widget.set_trim_start(self.trim_start)
            self.map_widget.set_trim_end(self.trim_end)
            self.map_widget.update()
            left_layout.addWidget(self.map_widget, 2)
        else:
            no_gps_label = QLabel("No GPS data available")
            no_gps_label.setAlignment(Qt.AlignCenter)
            left_layout.addWidget(no_gps_label, 2)

        # Detected laps table
        lap_table_group = QGroupBox("Detected Laps")
        lap_table_layout = QVBoxLayout()

        self.lap_table = QTableWidget()
        self.lap_table.setColumnCount(4)
        self.lap_table.setHorizontalHeaderLabels(
            ["Select", "Lap", "Duration", "Distance"]
        )
        lap_table_layout.addWidget(self.lap_table)
        
        # Add select/deselect all button
        self.select_all_laps_button = QPushButton("Select All Laps")
        self.select_all_laps_button.clicked.connect(self.toggle_all_laps_selection)
        lap_table_layout.addWidget(self.select_all_laps_button)

        lap_table_group.setLayout(lap_table_layout)
        left_layout.addWidget(lap_table_group, 1)

        # Parameter display
        param_group = QGroupBox("Analysis Parameters")
        param_layout = QFormLayout()

        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.update_config_text()
        param_layout.addRow("Configuration:", self.config_text)

        # Configuration name input
        self.config_name = QLineEdit("GPS Lap Test")
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
        self.ve_plot = VEPlotLabel(self.screen())
        plot_layout.addWidget(self.ve_plot)

        right_layout.addWidget(plot_widget, 3)
        self.plot_size_info = self.ve_plot.get_size_info()

        # Sliders
        slider_group = QGroupBox("Adjust Parameters")
        slider_layout = QFormLayout()

        # Trim start slider
        self.trim_start_slider = SliderTextBox(
            minimum=0,
            maximum=len(self.merged_data) - 30,
            value=self.trim_start,
            is_float=False
        )
        self.trim_start_slider.valueChanged.connect(self.on_trim_start_changed)

        slider_layout.addRow("Trim Start (s):", self.trim_start_slider)

        # Trim end slider
        self.trim_end_slider = SliderTextBox(
            minimum=30,
            maximum=len(self.merged_data),
            value=self.trim_end,
            is_float=False
        )
        self.trim_end_slider.valueChanged.connect(self.on_trim_end_changed)

        slider_layout.addRow("Trim End (s):", self.trim_end_slider)

        # GPS Marker slider
        self.gps_marker_slider = SliderTextBox(
            minimum=self.trim_start,
            maximum=self.trim_end,
            value=self.gps_marker_pos,
            is_float=False
        )
        self.gps_marker_slider.valueChanged.connect(self.on_gps_marker_changed)

        slider_layout.addRow("GPS Marker (s):", self.gps_marker_slider)

        # CdA slider
        self.cda_slider = SliderTextBox(
            minimum=self.params.get("cda_min", 0.15),
            maximum=self.params.get("cda_max", 0.5),
            value=self.current_cda,
            is_float=True,
            decimals=3
        )
        self.cda_slider.valueChanged.connect(self.on_cda_changed)
        self.cda_slider.set_enabled(self.params.get("cda") is None)

        slider_layout.addRow("CdA:", self.cda_slider)

        # Crr slider
        self.crr_slider = SliderTextBox(
            minimum=self.params.get("crr_min", 0.001),
            maximum=self.params.get("crr_max", 0.03),
            value=self.current_crr,
            is_float=True,
            decimals=4
        )
        self.crr_slider.valueChanged.connect(self.on_crr_changed)
        self.crr_slider.set_enabled(self.params.get("crr") is None)

        slider_layout.addRow("Crr:", self.crr_slider)

        # Wind source controls (only show if FIT wind or air data is available)
        if self.has_fit_wind or self.has_fit_air:
            # Wind source radio buttons
            wind_layout = QVBoxLayout()

            self.wind_constant_radio = QRadioButton("Use constant wind settings")
            self.wind_fit_radio = QRadioButton("Use FIT file wind speed")
            self.wind_compare_radio = QRadioButton("Compare both methods")

            # Set initial selection
            if self.wind_source == "constant":
                self.wind_constant_radio.setChecked(True)
            elif self.wind_source == "fit":
                self.wind_fit_radio.setChecked(True)
            elif self.wind_source == "compare":
                self.wind_compare_radio.setChecked(True)

            # Connect signals
            self.wind_constant_radio.toggled.connect(self.on_wind_source_changed)
            self.wind_fit_radio.toggled.connect(self.on_wind_source_changed)
            self.wind_compare_radio.toggled.connect(self.on_wind_source_changed)

            wind_layout.addWidget(self.wind_constant_radio)
            wind_layout.addWidget(self.wind_fit_radio)
            wind_layout.addWidget(self.wind_compare_radio)

            slider_layout.addRow("Wind Source:", wind_layout)

        slider_group.setLayout(slider_layout)
        right_layout.addWidget(slider_group, 1)

        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 800])  # Set initial sizes

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Maximize the window
        self.showMaximized()

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

    def update_lap_table(self):
        """Update the lap table with detected laps"""
        self.lap_table.setRowCount(len(self.detected_laps))

        for row, lap in enumerate(self.detected_laps):
            # Checkbox for selection
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # All laps selected by default
            checkbox.stateChanged.connect(lambda: self.async_update(detect_laps=False))
            self.lap_table.setCellWidget(row, 0, checkbox)

            # Lap number
            self.lap_table.setItem(row, 1, QTableWidgetItem(str(row + 1)))

            # Duration
            duration_mins = int(lap["duration"] // 60)
            duration_secs = int(lap["duration"] % 60)
            self.lap_table.setItem(
                row, 2, QTableWidgetItem(f"{duration_mins:02d}:{duration_secs:02d}")
            )

            # Distance
            self.lap_table.setItem(
                row, 3, QTableWidgetItem(f"{lap['distance']:.2f} km")
            )

            # Make cells read-only
            for col in range(1, 4):
                item = self.lap_table.item(row, col)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

        # Resize columns to fit content
        header = self.lap_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, 4):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        # Update button text based on current selection (all items are selected by default)
        if len(self.detected_laps) > 0:
            self.select_all_laps_button.setText("Deselect All Laps")

        self.selected_lap_indices = self.get_selected_lap_indices()
        # Chicken-Egg situation here, selected_lap_indices can be found after a
        # background calculation and UI query
        if not hasattr(self, "_triggered_lap_update") and self.selected_lap_indices:
            self._triggered_lap_update = True
            self.async_update()

    def get_selected_lap_indices(self):
        """Get indices of selected laps in the table"""
        selected_indices = []

        for row in range(self.lap_table.rowCount()):
            checkbox = self.lap_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                selected_indices.append(row)

        return selected_indices

    def get_current_params(self):
        """Get current parameters including wind data based on wind source selection"""
        current_params = self.params.copy()

        # NOTE: For GPS lap analysis, we don't pass global wind/air speed data here
        # because each detected lap will extract its own portion of the data.
        # Wind/air speed data will be handled per-lap in calculate_ve().

        # Set comparison mode flag
        current_params["comparison_mode"] = self.wind_source == "compare"

        return current_params

    def toggle_all_laps_selection(self):
        """Toggle selection of all laps based on current state"""
        # Check current state - count selected checkboxes
        selected_count = 0
        total_count = self.lap_table.rowCount()
        
        for row in range(total_count):
            checkbox = self.lap_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                selected_count += 1
        
        # Determine action: if any are selected, deselect all; if none selected, select all
        select_all = (selected_count == 0)
        
        # Apply the action
        for row in range(total_count):
            checkbox = self.lap_table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(select_all)
        
        # Update button text
        if select_all:
            self.select_all_laps_button.setText("Deselect All Laps")
        else:
            self.select_all_laps_button.setText("Select All Laps")

    def on_trim_start_changed(self, value):
        """Handle trim start slider value change"""
        # Ensure trim_start < trim_end - 30
        if value >= self.trim_end_slider.get_value() - 30:
            self.trim_start_slider.set_value(self.trim_end_slider.get_value() - 30)
            return

        self.trim_start = int(value)

        # Update GPS marker slider bounds
        self.gps_marker_slider.set_range(value, self.gps_marker_slider.maximum)

        # Ensure GPS marker is within bounds
        if self.gps_marker_pos < value:
            self.gps_marker_slider.set_value(value)
            self.gps_marker_pos = value
            self.map_widget.set_marker_pos(value)

        self.async_update(True)

        # Update map to show trim points
        self.map_widget.set_trim_start(self.trim_start)
        self.map_widget.update()

    def on_trim_end_changed(self, value):
        """Handle trim end slider value change"""
        # Ensure trim_end > trim_start + 30
        if value <= self.trim_start_slider.get_value() + 30:
            self.trim_end_slider.set_value(self.trim_start_slider.get_value() + 30)
            return

        self.trim_end = int(value)

        # Update GPS marker slider bounds
        self.gps_marker_slider.set_range(self.gps_marker_slider.minimum, value)

        # Ensure GPS marker is within bounds
        if self.gps_marker_pos > value:
            self.gps_marker_slider.set_value(value)
            self.gps_marker_pos = value
            self.map_widget.set_marker_pos(value)

        self.async_update(True)

        self.map_widget.set_trim_end(self.trim_end)
        self.map_widget.update()

    def on_gps_marker_changed(self, value):
        """Handle GPS marker slider value change"""
        self.gps_marker_pos = int(value)

        self.async_update(True)

        # Update map to show GPS marker
        self.map_widget.set_marker_pos(value)
        self.map_widget.update()

    def on_cda_changed(self, value):
        """Handle CdA slider value change"""
        self.current_cda = value

        self.async_update(detect_laps=False)

        self.update_config_text()

    def on_crr_changed(self, value):
        """Handle Crr slider value change"""
        self.current_crr = value

        self.async_update(detect_laps=False)

        self.update_config_text()

    def on_wind_source_changed(self):
        """Handle wind source radio button changes"""
        # Determine the new wind source
        new_wind_source = "constant"  # default
        if (
            hasattr(self, "wind_constant_radio")
            and self.wind_constant_radio.isChecked()
        ):
            new_wind_source = "constant"
        elif hasattr(self, "wind_fit_radio") and self.wind_fit_radio.isChecked():
            new_wind_source = "fit"
        elif (
            hasattr(self, "wind_compare_radio") and self.wind_compare_radio.isChecked()
        ):
            new_wind_source = "compare"

        # Only update if the source actually changed
        if new_wind_source != self.wind_source:
            self.wind_source = new_wind_source

            # Recreate VE worker with new parameters
            if hasattr(self, 've_worker') and self.ve_worker:
                try:
                    self.ve_worker.resultReady.disconnect()
                except (RuntimeError, TypeError):
                    pass  # Ignore if already disconnected or connection doesn't exist

            self.ve_worker = VEWorker(self.merged_data, self.get_current_params())
            self.ve_worker.moveToThread(self.ve_thread)
            self.ve_worker.resultReady.connect(self.on_ve_result_ready)

            self.async_update(detect_laps=False)

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
        csv_path = os.path.join(self.result_dir, f"{file_basename}_gps_lap_results.csv")

        # Prepare data for CSV
        lap_str = "_".join(map(str, sorted(self.selected_laps)))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        selected_lap_indices = self.selected_lap_indices

        # Calculate total error for selected laps
        total_error = 0
        for i in selected_lap_indices:
            if i < len(self.lap_ve_profiles):
                total_error += abs(self.lap_ve_profiles[i][-1])

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
            "gps_marker_pos": self.gps_marker_pos,
            "detected_laps": len(self.detected_laps),
            "selected_laps": len(selected_lap_indices),
            "total_error": total_error,
        }

        # Save to CSV
        header = list(result_row.keys())

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
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(existing_data)

        # Save plot
        plot_dir = os.path.join(self.result_dir, "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plot_filename = f"{file_basename}_gps_laps_{lap_str}_{config_name}.png"
        plot_path = os.path.join(plot_dir, plot_filename)

        self.ve_plot_saver.save(self.ve_worker.get_values(), plot_path)

        # Save trim and GPS marker settings to file settings
        file_settings = self.settings.get_file_settings(self.fit_file.filename)

        # Initialize trim_settings if it doesn't exist
        if "trim_settings" not in file_settings:
            file_settings["trim_settings"] = {}

        # Save trim settings for this lap combination
        file_settings["trim_settings"][self.settings_key] = {
            "trim_start": self.trim_start,
            "trim_end": self.trim_end,
            "gps_marker_pos": self.gps_marker_pos,
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
        self.parent.show()
        self.close()

    def on_plot_size_changed(self):
        self.plot_size_info = self.ve_plot.get_size_info()
        self.async_update(update_ve=False)

    def async_update(self, detect_laps=True, update_ve=True):
        self.selected_lap_indices = self.get_selected_lap_indices()

        values = {}
        for key in VEWorker.INPUT_KEYS:
            values[key] = getattr(self, key)
        values["detect_laps"] = detect_laps and self.has_gps
        values["update_ve"] = update_ve

        self.ve_worker.set_values(values)

    def on_ve_result_ready(self, res):
        for key in VEWorker.RESULT_KEYS:
            if key in res:
                setattr(self, key, res[key])

        # Update the lap table
        if res["detect_laps"]:
            self.update_lap_table()

        self.ve_plot.set_fig(self.fig_res)
        self.fig_res = None

    def join_threads(self):
        if self.map_widget:
            self.map_widget.close()
            self.map_widget = None
        if self.ve_thread:
            self.ve_thread.quit()
            self.ve_thread.wait()
            self.ve_thread = None

    def closeEvent(self, event):
        self.join_threads()
        event.accept()
