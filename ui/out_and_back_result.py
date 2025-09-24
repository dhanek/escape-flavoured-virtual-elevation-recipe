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
        "gps_marker_a_pos",
        "gps_marker_b_pos",
        "selected_section_indices",
        "plot_size_info",
    ]

    RESULT_KEYS = [
        "all_actual_elevations",
        "detected_sections",
        "mean_actual_distance_km",
        "mean_actual_elevation",
        "section_distances",
        "section_ve_profiles",
        "fig_res",
    ]

    def __init__(self, merged_data, params):
        super(VEWorker, self).__init__()
        self.merged_data = merged_data
        self.params = params
        self.ve_valid = False
        self.detected_sections = []
        self.mean_actual_distance_km = 0

    def _process_value(self, values: dict):
        for key in VEWorker.INPUT_KEYS:
            if key in values:
                setattr(self, key, values[key])

        detect_sections = False
        # Detect sections with new gate positions
        if values["detect_sections"] or not self.detected_sections:
            self.detect_sections()
            detect_sections = True

        # Recalculate VE metrics and update plots
        if values["update_ve"] or not self.ve_valid:
            self.calculate_ve()
            self.ve_valid = True

        self.update_plots()

        out_values = {}
        for key in VEWorker.RESULT_KEYS:
            out_values[key] = getattr(self, key)
        out_values["detect_sections"] = detect_sections

        return out_values

    def detect_sections(self):
        """
        Detect out-and-back sections based on GPS markers A and B

        This method identifies sections where:
        1. Marker A is passed in one direction (outbound start)
        2. Marker B is passed (outbound end)
        3. Marker B is passed again in opposite direction (inbound start)
        4. Marker A is passed again in opposite direction (inbound end)
        """
        self.detected_sections = []

        # Get valid GPS coordinates
        valid_coords = self.merged_data.dropna(subset=["position_lat", "position_long"])
        if valid_coords.empty:
            return

        # Get the GPS marker positions
        marker_a_idx = self.gps_marker_a_pos
        marker_b_idx = self.gps_marker_b_pos

        if (
            marker_a_idx < 0
            or marker_a_idx >= len(valid_coords)
            or marker_b_idx < 0
            or marker_b_idx >= len(valid_coords)
        ):
            return

        marker_a_lat = valid_coords.iloc[marker_a_idx]["position_lat"]
        marker_a_lon = valid_coords.iloc[marker_a_idx]["position_long"]
        marker_b_lat = valid_coords.iloc[marker_b_idx]["position_lat"]
        marker_b_lon = valid_coords.iloc[marker_b_idx]["position_long"]

        # Calculate distance from each point to the markers
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

        # Find all points where we pass near marker A and B within the trimmed region
        marker_a_passings = []
        marker_b_passings = []

        for i in range(self.trim_start, min(self.trim_end + 1, len(valid_coords))):
            if i >= len(valid_coords):
                continue

            point_lat = valid_coords.iloc[i]["position_lat"]
            point_lon = valid_coords.iloc[i]["position_long"]

            # Distance to marker A
            distance_a = haversine(marker_a_lat, marker_a_lon, point_lat, point_lon)
            # Distance to marker B
            distance_b = haversine(marker_b_lat, marker_b_lon, point_lat, point_lon)

            if distance_a < threshold:
                direction = bearings[i] if i < len(bearings) else 0
                marker_a_passings.append(
                    {
                        "index": i,
                        "distance": distance_a,
                        "direction": direction,
                        "timestamp": valid_coords.iloc[i]["timestamp"],
                    }
                )

            if distance_b < threshold:
                direction = bearings[i] if i < len(bearings) else 0
                marker_b_passings.append(
                    {
                        "index": i,
                        "distance": distance_b,
                        "direction": direction,
                        "timestamp": valid_coords.iloc[i]["timestamp"],
                    }
                )

        # Group nearby passings and keep only the closest point for each group
        def group_passings(passings):
            grouped = []
            current_group = []

            for passing in passings:
                if (
                    not current_group
                    or passing["index"] - current_group[-1]["index"] <= 5
                ):  # Points within 5 seconds
                    current_group.append(passing)
                else:
                    # Find the closest point in the group
                    if current_group:
                        closest = min(current_group, key=lambda x: x["distance"])
                        grouped.append(closest)
                    current_group = [passing]

            # Add the last group if it exists
            if current_group:
                closest = min(current_group, key=lambda x: x["distance"])
                grouped.append(closest)

            return grouped

        grouped_a_passings = group_passings(marker_a_passings)
        grouped_b_passings = group_passings(marker_b_passings)

        # Sort passings by index
        grouped_a_passings.sort(key=lambda x: x["index"])
        grouped_b_passings.sort(key=lambda x: x["index"])

        # Angle threshold for determining opposite direction (120-180 degrees difference)
        angle_threshold = 90

        # State machine to find out-and-back sections
        outbound_started = False
        outbound_ended = False
        inbound_started = False

        current_section = {}

        # Merge both passing lists and sort by timestamp
        all_passings = []
        for p in grouped_a_passings:
            all_passings.append({"marker": "A", **p})
        for p in grouped_b_passings:
            all_passings.append({"marker": "B", **p})

        all_passings.sort(key=lambda x: x["index"])

        # Process all passings to find out-and-back sections
        for passing in all_passings:
            marker = passing["marker"]
            idx = passing["index"]
            direction = passing["direction"]

            if not outbound_started:
                # Looking for first A
                if marker == "A":
                    current_section = {
                        "outbound_start_idx": idx,
                        "outbound_start_direction": direction,
                    }
                    outbound_started = True

            elif outbound_started and not outbound_ended:
                # Looking for first B after A
                if marker == "B":
                    # Store B passing data
                    current_section["outbound_end_idx"] = idx
                    current_section["outbound_end_direction"] = direction
                    outbound_ended = True

            elif outbound_ended and not inbound_started:
                # Looking for second B in opposite direction
                if marker == "B":
                    # Check if direction is sufficiently different (opposite direction)
                    dir_diff = abs(
                        direction - current_section["outbound_end_direction"]
                    )
                    dir_diff = min(dir_diff, 360 - dir_diff)

                    if dir_diff > angle_threshold:
                        # We found the start of inbound journey
                        current_section["inbound_start_idx"] = idx
                        current_section["inbound_start_direction"] = direction
                        inbound_started = True

            elif inbound_started:
                # Looking for second A in opposite direction to complete the section
                if marker == "A":
                    # Check if direction is sufficiently different (opposite direction)
                    dir_diff = abs(
                        direction - current_section["outbound_start_direction"]
                    )
                    dir_diff = min(dir_diff, 360 - dir_diff)

                    if dir_diff > angle_threshold:
                        # We found the end of inbound journey - complete section
                        current_section["inbound_end_idx"] = idx
                        current_section["inbound_end_direction"] = direction

                        # Calculate duration and distance for this section
                        outbound_duration = (
                            valid_coords.iloc[current_section["outbound_end_idx"]][
                                "timestamp"
                            ]
                            - valid_coords.iloc[current_section["outbound_start_idx"]][
                                "timestamp"
                            ]
                        ).total_seconds()

                        inbound_duration = (
                            valid_coords.iloc[current_section["inbound_end_idx"]][
                                "timestamp"
                            ]
                            - valid_coords.iloc[current_section["inbound_start_idx"]][
                                "timestamp"
                            ]
                        ).total_seconds()

                        total_duration = outbound_duration + inbound_duration

                        # Distance calculation
                        outbound_distance = (
                            valid_coords.iloc[current_section["outbound_end_idx"]][
                                "distance"
                            ]
                            - valid_coords.iloc[current_section["outbound_start_idx"]][
                                "distance"
                            ]
                        ) / 1000  # Convert to km

                        inbound_distance = (
                            valid_coords.iloc[current_section["inbound_end_idx"]][
                                "distance"
                            ]
                            - valid_coords.iloc[current_section["inbound_start_idx"]][
                                "distance"
                            ]
                        ) / 1000  # Convert to km

                        total_distance = outbound_distance + inbound_distance

                        current_section["outbound_duration"] = outbound_duration
                        current_section["inbound_duration"] = inbound_duration
                        current_section["total_duration"] = total_duration
                        current_section["outbound_distance"] = outbound_distance
                        current_section["inbound_distance"] = inbound_distance
                        current_section["total_distance"] = total_distance

                        # Add section to detected sections
                        self.detected_sections.append(current_section)

                        # Reset for next section
                        outbound_started = False
                        outbound_ended = False
                        inbound_started = False
                        current_section = {}

    # Update in calculate_ve method to create a single mean elevation profile
    def calculate_ve(self):
        """Calculate virtual elevation for each detected section, separate for outbound and inbound segments"""
        self.section_ve_profiles = []
        self.section_distances = []

        # Store actual elevation profiles for each section
        self.all_actual_elevations = []
        self.mean_actual_elevation = None

        if not self.detected_sections:
            return

        # Check if we have actual elevation data
        has_elevation = (
            "altitude" in self.merged_data.columns
            and not self.merged_data["altitude"].isna().all()
        )

        # For each detected section, calculate VE separately for outbound and inbound segments
        for section in self.detected_sections:
            section_data = {
                "outbound_ve": None,
                "outbound_distance": None,
                "inbound_ve": None,
                "inbound_distance": None,
            }

            # Process outbound segment (A to B)
            if "outbound_start_idx" in section and "outbound_end_idx" in section:
                start_idx = section["outbound_start_idx"]
                end_idx = section["outbound_end_idx"]

                if (
                    start_idx < end_idx
                    and start_idx >= 0
                    and end_idx < len(self.merged_data)
                ):
                    # Extract outbound segment data
                    outbound_data = self.merged_data.iloc[
                        start_idx : end_idx + 1
                    ].copy()

                    # Create VE calculator for this segment
                    outbound_ve_calculator = VirtualElevation(
                        outbound_data, self.params
                    )

                    # Calculate VE
                    outbound_ve = outbound_ve_calculator.calculate_ve(
                        self.current_cda, self.current_crr
                    )

                    # Get distance data
                    if "distance" in outbound_data.columns:
                        # Use distance directly from FIT file (in meters)
                        distances = outbound_data["distance"].values

                        # Make distances relative to start
                        start_distance = distances[0]
                        distances = distances - start_distance

                        # Convert to kilometers
                        distances = distances / 1000
                    else:
                        # Fallback if no distance data
                        distances = np.linspace(
                            0, section["outbound_distance"], len(outbound_ve)
                        )

                    # Store results
                    section_data["outbound_ve"] = outbound_ve
                    section_data["outbound_distance"] = distances

                    # Store actual elevation if available
                    if has_elevation and not self.params["velodrome"]:
                        actual_elevation = outbound_data["altitude"].values
                        self.all_actual_elevations.append(
                            (distances, actual_elevation, "outbound")
                        )
                    elif self.params["velodrome"]:
                        actual_elevation = np.zeros_like(outbound_data["altitude"])
                        self.all_actual_elevations.append((distances, actual_elevation, "outbound"))

            # Process inbound segment (B to A)
            if "inbound_start_idx" in section and "inbound_end_idx" in section:
                start_idx = section["inbound_start_idx"]
                end_idx = section["inbound_end_idx"]

                if (
                    start_idx < end_idx
                    and start_idx >= 0
                    and end_idx < len(self.merged_data)
                ):
                    # Extract inbound segment data
                    inbound_data = self.merged_data.iloc[start_idx : end_idx + 1].copy()

                    # Create VE calculator for this segment
                    inbound_ve_calculator = VirtualElevation(inbound_data, self.params)

                    # Calculate VE
                    inbound_ve = inbound_ve_calculator.calculate_ve(
                        self.current_cda, self.current_crr
                    )

                    # Get distance data
                    if "distance" in inbound_data.columns:
                        # Use distance directly from FIT file (in meters)
                        distances = inbound_data["distance"].values

                        # Make distances relative to start
                        start_distance = distances[0]
                        distances = distances - start_distance

                        # Convert to kilometers
                        distances = distances / 1000
                    else:
                        # Fallback if no distance data
                        distances = np.linspace(
                            0, section["inbound_distance"], len(inbound_ve)
                        )

                    # Store results
                    section_data["inbound_ve"] = inbound_ve
                    section_data["inbound_distance"] = distances

                    # Store actual elevation if available
                    if has_elevation and not self.params["velodrome"]:
                        actual_elevation = inbound_data["altitude"].values
                        self.all_actual_elevations.append(
                            (distances, actual_elevation, "inbound")
                        )
                    elif self.params["velodrome"]:
                        actual_elevation = np.zeros_like(inbound_data["altitude"])
                        self.all_actual_elevations.append((distances, actual_elevation, "inbound"))

            # Add section data
            self.section_ve_profiles.append(section_data)

        # Calculate single mean elevation profile if we have actual elevation data
        if has_elevation and self.all_actual_elevations and not self.params["velodrome"]:
            # Find max distance to create reference distance array
            max_distance = 0
            for distances, _, direction in self.all_actual_elevations:
                if len(distances) > 0:
                    max_distance = max(max_distance, distances[-1])

            # Create reference distance array (1m intervals)
            reference_distance = np.linspace(
                0, max_distance * 1000, int(max_distance * 1000) + 1
            )
            reference_distance_km = reference_distance / 1000

            # Initialize arrays for accumulating elevation values
            elevation_sum = np.zeros_like(reference_distance)
            elevation_count = np.zeros_like(reference_distance)

            # Process all elevation profiles
            for distances, elevations, direction in self.all_actual_elevations:
                # For inbound segments, mirror the distances
                if direction == "inbound":
                    max_dist = distances[-1]
                    distances = max_dist - distances

                # Interpolate elevation onto the reference distance array
                interpolated_elevation = np.interp(
                    reference_distance,
                    distances * 1000,  # Convert km to m
                    elevations,
                    left=np.nan,
                    right=np.nan,
                )

                # Add to sum and count non-NaN values
                valid_mask = ~np.isnan(interpolated_elevation)
                elevation_sum[valid_mask] += interpolated_elevation[valid_mask]
                elevation_count[valid_mask] += 1

            # Calculate mean, avoiding division by zero
            mean_elevation = np.zeros_like(reference_distance)
            valid_points = elevation_count > 0
            mean_elevation[valid_points] = (
                elevation_sum[valid_points] / elevation_count[valid_points]
            )

            # Store mean elevation profile
            self.mean_actual_elevation = mean_elevation
            self.mean_actual_distance_km = reference_distance_km
        elif self.params["velodrome"]:
            max_distance = 0
            for distances, _, direction in self.all_actual_elevations:
                if len(distances) > 0:
                    max_distance = max(max_distance, distances[-1])

            # Create reference distance array (1m intervals)
            reference_distance = np.linspace(
                0, max_distance * 1000, int(max_distance * 1000) + 1
            )
            reference_distance_km = reference_distance / 1000

            self.mean_actual_elevation = np.zeros_like(reference_distance_km)
            self.mean_actual_distance_km = reference_distance_km

    def update_plots(self):
        """Update the virtual elevation plots for out-and-back analysis"""
        ve_fig = VEFigure(self.plot_size_info)
        fig, ax1, ax2 = ve_fig.get_fig_axes()

        # Get selected section indices
        selected_indices = self.selected_section_indices

        if not selected_indices or not self.section_ve_profiles:
            # No sections selected or detected
            ax1.text(
                0.5,
                0.5,
                "No out-and-back laps detected or selected",
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
            # Define a muted color palette from GpsLapResults
            muted_colors = [
                "#4363d8",  # strong blue
                "#ffe119",  # bright yellow
                "#800000",  # dark maroon
                "#f58231",  # bright orange
                "#000075",  # dark navy blue
                "#dcbeff",  # light purple
                "#a9a9a9",  # medium gray
            ]

            # Determine maximum distance for x-axis alignment
            max_outbound_dist = 0
            max_inbound_dist = 0

            for i in selected_indices:
                if i < len(self.section_ve_profiles):
                    section_data = self.section_ve_profiles[i]

                    if (
                        section_data["outbound_distance"] is not None
                        and len(section_data["outbound_distance"]) > 0
                    ):
                        max_outbound_dist = max(
                            max_outbound_dist, section_data["outbound_distance"][-1]
                        )

                    if (
                        section_data["inbound_distance"] is not None
                        and len(section_data["inbound_distance"]) > 0
                    ):
                        max_inbound_dist = max(
                            max_inbound_dist, section_data["inbound_distance"][-1]
                        )

            # Plot mean actual elevation profile if available
            if (
                hasattr(self, "mean_actual_elevation")
                and self.mean_actual_elevation is not None
            ):
                # Plot mean elevation for outbound direction
                valid_indices = np.where(
                    self.mean_actual_distance_km <= max_outbound_dist
                )[0]
                if len(valid_indices) > 0:
                    ax1.plot(
                        self.mean_actual_distance_km[valid_indices],
                        self.mean_actual_elevation[valid_indices],
                        color="black",
                        # linestyle="--",
                        linewidth=1,
                        label="Mean Actual Elevation",
                    )

            # Lists to store legend handles and labels
            legend_handles = []
            legend_labels = []

            # Variables to accumulate residuals
            all_outbound_residuals = []
            all_inbound_residuals = []

            # Variable to store total elevation error
            total_error = 0
            sum_of_both_abs = 0

            # Plot each selected section
            for idx, section_idx in enumerate(selected_indices):
                if section_idx < len(self.section_ve_profiles):
                    section_data = self.section_ve_profiles[section_idx]
                    # Use one color per section from muted colors
                    section_color = muted_colors[idx % len(muted_colors)]

                    # Values to store elevation gains for error calculation
                    outbound_gain = None
                    inbound_gain = None

                    # Plot outbound segment (A→B)
                    if (
                        section_data["outbound_ve"] is not None
                        and section_data["outbound_distance"] is not None
                    ):
                        outbound_ve = section_data["outbound_ve"]
                        outbound_distances = section_data["outbound_distance"]

                        # Calibrate outbound VE if we have mean elevation
                        if (
                            hasattr(self, "mean_actual_elevation")
                            and self.mean_actual_elevation is not None
                        ):

                            # Calibrate VE to match this starting point
                            calibrated_outbound_ve = (
                                outbound_ve
                                - outbound_ve[0]
                                + self.mean_actual_elevation[0]
                            )
                        else:
                            # No calibration needed
                            calibrated_outbound_ve = outbound_ve

                        # Calculate elevation gain for outbound segment
                        if len(calibrated_outbound_ve) > 0:
                            outbound_gain = (
                                calibrated_outbound_ve[-1] - calibrated_outbound_ve[0]
                            )
                            outbound_error = (
                                calibrated_outbound_ve[-1]
                                - self.mean_actual_elevation[-1]
                            )

                        # Plot calibrated outbound VE
                        (line1,) = ax1.plot(
                            outbound_distances,
                            calibrated_outbound_ve,
                            color=section_color,
                            linewidth=4,
                            linestyle="-",
                        )
                        legend_handles.append(line1)
                        legend_labels.append(f"Lap {section_idx+1} (A→B)")

                        # Calculate and plot residuals if we have mean elevation
                        if (
                            hasattr(self, "mean_actual_elevation")
                            and self.mean_actual_elevation is not None
                        ):
                            # Interpolate mean elevation to match this segment's distance points
                            reference_elevation = np.interp(
                                outbound_distances,
                                self.mean_actual_distance_km,
                                self.mean_actual_elevation,
                                left=np.nan,
                                right=np.nan,
                            )

                            # Calculate residuals only for valid points
                            valid_mask = ~np.isnan(reference_elevation)
                            if np.any(valid_mask):
                                residual_distances = outbound_distances[valid_mask]
                                residuals = (
                                    calibrated_outbound_ve[valid_mask]
                                    - reference_elevation[valid_mask]
                                )
                                # Store distances and residuals for later plotting
                                all_outbound_residuals.append(
                                    (residual_distances, residuals, section_color)
                                )

                    # Plot inbound segment (B→A) - with mirroring on x-axis
                    if (
                        section_data["inbound_ve"] is not None
                        and section_data["inbound_distance"] is not None
                    ):
                        inbound_ve = section_data["inbound_ve"]
                        inbound_distances = section_data["inbound_distance"]

                        # Mirror the inbound distances
                        max_dist = inbound_distances[-1]
                        mirrored_inbound_distances = max_dist - inbound_distances

                        # Calibrate inbound VE if we have mean elevation
                        if (
                            hasattr(self, "mean_actual_elevation")
                            and self.mean_actual_elevation is not None
                        ):
                            # Calibrate VE to match this starting point
                            calibrated_inbound_ve = (
                                inbound_ve
                                - inbound_ve[0]
                                + self.mean_actual_elevation[-1]
                            )
                        else:
                            # No calibration needed
                            calibrated_inbound_ve = inbound_ve

                        # Calculate elevation gain for inbound segment
                        if len(calibrated_inbound_ve) > 0:
                            inbound_gain = (
                                calibrated_inbound_ve[-1] - calibrated_inbound_ve[0]
                            )
                            inbound_error = (
                                calibrated_inbound_ve[-1]
                                - self.mean_actual_elevation[0]
                            )

                        # Plot calibrated inbound VE with mirrored distances
                        (line2,) = ax1.plot(
                            mirrored_inbound_distances,
                            calibrated_inbound_ve,
                            color=section_color,  # Same color as outbound
                            linewidth=4,
                            linestyle="--",  # Dashed line for inbound
                        )
                        legend_handles.append(line2)
                        legend_labels.append(f"Lap {section_idx+1} (B→A)")

                        # Calculate and plot residuals if we have mean elevation
                        if (
                            hasattr(self, "mean_actual_elevation")
                            and self.mean_actual_elevation is not None
                        ):
                            # Interpolate mean elevation to match this segment's distance points
                            reference_elevation = np.interp(
                                inbound_distances,
                                self.mean_actual_distance_km,
                                self.mean_actual_elevation,
                                left=np.nan,
                                right=np.nan,
                            )

                            # invert reference_elevation for mirrored distances
                            reference_elevation = np.flip(reference_elevation)

                            # Calculate residuals only for valid points
                            valid_mask = ~np.isnan(reference_elevation)
                            if np.any(valid_mask):
                                # Use mirrored distances for residuals too
                                residual_distances = mirrored_inbound_distances[
                                    valid_mask
                                ]
                                residuals = (
                                    calibrated_inbound_ve[valid_mask]
                                    - reference_elevation[valid_mask]
                                )
                                # Store distances and residuals for later plotting
                                all_inbound_residuals.append(
                                    (residual_distances, residuals, section_color)
                                )

                    # Calculate the error for this section based on elevation gain difference
                    if outbound_gain is not None and inbound_gain is not None:
                        section_error = abs(outbound_gain + inbound_gain)
                        sum_of_both_abs = abs(outbound_error) + abs(inbound_error)
                        total_error += section_error

            # Plot all residuals at once
            # First outbound residuals (solid lines)
            for residual_distances, residuals, color in all_outbound_residuals:
                ax2.plot(
                    residual_distances,
                    residuals,
                    color=color,
                    linewidth=3.5,
                    linestyle="-",
                )

            # Then inbound residuals (dashed lines)
            for residual_distances, residuals, color in all_inbound_residuals:
                ax2.plot(
                    residual_distances,
                    residuals,
                    color=color,
                    linewidth=3.5,
                    linestyle="--",
                )

            # Add zero line in residuals plot
            ax2.axhline(y=0, color="black", linestyle="-")

            # Set axis limits
            max_dist = max(max_outbound_dist, max_inbound_dist)
            ax1.set_xlim(0, max_dist)
            ax2.set_xlim(0, max_dist)

            # Add subtle grid lines to both plots
            ax1.grid(True, linestyle="--", alpha=0.3, color="#cccccc")
            ax2.grid(True, linestyle="--", alpha=0.3, color="#cccccc")

            # Set titles and labels
            ax1.set_ylabel("Elevation (m)")
            ax1.set_title("Out-and-Back Virtual Elevation Profiles")

            # Handle legend - use 2 columns if more than 4 sections
            if len(legend_handles) > 4:
                # Place legend outside the plot to the right
                ax1.legend(
                    legend_handles,
                    legend_labels,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    ncol=1
                    + len(legend_handles) // 8,  # Add a column for every 8 entries
                )
            else:
                ax1.legend(legend_handles, legend_labels)

            ax2.set_xlabel("Distance (km)")
            ax2.set_ylabel("Elevation (m)")
            ax2.set_title("Residuals (Virtual - Mean Actual Elevation)")

            # Add CdA, Crr, and error values
            cda_str = f"CdA: {self.current_cda:.3f}"
            crr_str = f"Crr: {self.current_crr:.4f}"
            error_str = (
                f"Total Error: {total_error:.2f} m\n"
                + f"Diff to actual: {sum_of_both_abs:.2f} m"
            )

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


class OutAndBackResult(QMainWindow):
    """Window for displaying Out-and-Back analysis results"""

    def __init__(self, parent, fit_file, settings, selected_laps, params):
        super().__init__()
        self.parent = parent
        self.fit_file = fit_file
        self.settings = settings
        self.selected_laps = selected_laps
        self.params = params
        self.result_dir = settings.result_dir
        self.detected_sections = []
        self.section_ve_profiles = []

        # Initialize these attributes BEFORE calling prepare_merged_data
        self.trim_start = 0
        self.trim_end = 0
        self.gps_marker_a_pos = 0
        self.gps_marker_b_pos = 0

        # Prepare merged lap data
        self.prepare_merged_data()

        self.ve_worker = VEWorker(self.merged_data, self.params)
        self.ve_thread = QThread()
        self.ve_worker.moveToThread(self.ve_thread)
        self.ve_worker.resultReady.connect(self.on_ve_result_ready)
        self.ve_plot_saver = VEPlotSaver(
            VEWorker(self.merged_data, self.params), self.ve_thread
        )
        self.ve_thread.start()
        QApplication.instance().aboutToQuit.connect(self.join_threads)

        # Get lap combination ID for settings
        self.lap_combo_id = "_".join(map(str, sorted(self.selected_laps)))
        self.settings_key = f"OUTBACK_lap_{self.lap_combo_id}"

        # Try to load saved trim values for this lap combination
        file_settings = self.settings.get_file_settings(self.fit_file.filename)
        trim_settings = file_settings.get("trim_settings", {})
        saved_trim = trim_settings.get(self.settings_key, {})

        # Initialize UI values
        if saved_trim and "trim_start" in saved_trim and "trim_end" in saved_trim:
            # Use saved trim values if available
            self.trim_start = saved_trim["trim_start"]
            self.trim_end = saved_trim["trim_end"]
            # Use saved GPS marker positions if available, otherwise force A=0s, B=A+30s
            self.gps_marker_a_pos = saved_trim.get(
                "gps_marker_a_pos",
                self.trim_start,
            )
            self.gps_marker_b_pos = saved_trim.get(
                "gps_marker_b_pos",
                self.trim_start + 30,
            )
        else:
            # Use defaults
            self.trim_start = 0
            self.trim_end = len(self.merged_data) - 1
            # Force GPS marker A to start at 0s (trim_start) and marker B at A + 30s
            self.gps_marker_a_pos = self.trim_start
            self.gps_marker_b_pos = self.trim_start + 30

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
            f'Out-and-Back Analysis - Laps {", ".join(map(str, self.selected_laps))}'
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
        self.map_widget = MapWidget(MapMode.MARKER_AB, self.merged_data, self.params)
        if self.map_widget:
            self.map_widget.set_marker_a_pos(self.gps_marker_a_pos)
            self.map_widget.set_marker_b_pos(self.gps_marker_b_pos)
            self.map_widget.set_trim_start(self.trim_start)
            self.map_widget.set_trim_end(self.trim_end)
            self.map_widget.update()
            left_layout.addWidget(self.map_widget, 2)
        else:
            no_gps_label = QLabel("No GPS data available")
            no_gps_label.setAlignment(Qt.AlignCenter)
            left_layout.addWidget(no_gps_label, 2)

        # Detected sections table
        section_table_group = QGroupBox("Detected Out-and-Back Laps")
        section_table_layout = QVBoxLayout()

        self.section_table = QTableWidget()
        self.section_table.setColumnCount(4)
        self.section_table.setHorizontalHeaderLabels(
            ["Select", "Lap", "Duration", "Distance"]
        )
        section_table_layout.addWidget(self.section_table)
        
        # Add select/deselect all button
        self.select_all_sections_button = QPushButton("Select All Sections")
        self.select_all_sections_button.clicked.connect(self.toggle_all_sections_selection)
        section_table_layout.addWidget(self.select_all_sections_button)

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
        self.config_name = QLineEdit("Out and Back Test")
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

        # GPS Marker A slider
        self.gps_marker_a_slider = SliderTextBox(
            minimum=self.trim_start,
            maximum=self.trim_end,
            value=self.gps_marker_a_pos,
            is_float=False
        )
        self.gps_marker_a_slider.valueChanged.connect(self.on_gps_marker_a_changed)

        slider_layout.addRow("GPS Marker A (s):", self.gps_marker_a_slider)

        # GPS Marker B slider
        self.gps_marker_b_slider = SliderTextBox(
            minimum=self.trim_start,
            maximum=self.trim_end,
            value=self.gps_marker_b_pos,
            is_float=False
        )
        self.gps_marker_b_slider.valueChanged.connect(self.on_gps_marker_b_changed)

        slider_layout.addRow("GPS Marker B (s):", self.gps_marker_b_slider)

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

    def update_section_table(self):
        """Update the section table with detected out-and-back sections"""
        self.section_table.setRowCount(len(self.detected_sections))

        for row, section in enumerate(self.detected_sections):
            # Checkbox for selection
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # All sections selected by default
            checkbox.stateChanged.connect(
                lambda: self.async_update(detect_sections=False)
            )
            self.section_table.setCellWidget(row, 0, checkbox)

            # Section number
            self.section_table.setItem(row, 1, QTableWidgetItem(str(row + 1)))

            # Duration
            duration_mins = int(section["total_duration"] // 60)
            duration_secs = int(section["total_duration"] % 60)
            self.section_table.setItem(
                row, 2, QTableWidgetItem(f"{duration_mins:02d}:{duration_secs:02d}")
            )

            # Distance
            self.section_table.setItem(
                row, 3, QTableWidgetItem(f"{section['total_distance']:.2f} km")
            )

            # Make cells read-only
            for col in range(1, 4):
                item = self.section_table.item(row, col)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

        # Resize columns to fit content
        header = self.section_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, 4):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        # Update button text based on current selection (all items are selected by default)
        if len(self.detected_sections) > 0:
            self.select_all_sections_button.setText("Deselect All Sections")

        self.selected_section_indices = self.get_selected_section_indices()
        # Chicken-Egg situation here, selected_lap_indices can be found after a
        # background calculation and UI query
        if (
            not hasattr(self, "_triggered_section_update")
            and self.selected_section_indices
        ):
            self._triggered_section_update = True
            self.async_update()

    def get_selected_section_indices(self):
        """Get indices of selected sections in the table"""
        selected_indices = []

        for row in range(self.section_table.rowCount()):
            checkbox = self.section_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                selected_indices.append(row)

        return selected_indices

    def toggle_all_sections_selection(self):
        """Toggle selection of all sections based on current state"""
        # Check current state - count selected checkboxes
        selected_count = 0
        total_count = self.section_table.rowCount()
        
        for row in range(total_count):
            checkbox = self.section_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                selected_count += 1
        
        # Determine action: if any are selected, deselect all; if none selected, select all
        select_all = (selected_count == 0)
        
        # Apply the action
        for row in range(total_count):
            checkbox = self.section_table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(select_all)
        
        # Update button text
        if select_all:
            self.select_all_sections_button.setText("Deselect All Sections")
        else:
            self.select_all_sections_button.setText("Select All Sections")

    def on_trim_start_changed(self, value):
        """Handle trim start slider value change"""
        # Ensure trim_start < trim_end - 30
        if value >= self.trim_end_slider.get_value() - 30:
            self.trim_start_slider.set_value(self.trim_end_slider.get_value() - 30)
            return

        self.trim_start = int(value)

        # Update GPS marker slider bounds
        self.gps_marker_a_slider.set_range(value, self.gps_marker_a_slider.maximum)
        self.gps_marker_b_slider.set_range(value, self.gps_marker_b_slider.maximum)

        # Ensure GPS markers are within bounds
        if self.gps_marker_a_pos < value:
            self.gps_marker_a_slider.set_value(value)
            self.gps_marker_a_pos = value
            self.map_widget.set_marker_a_pos(self.gps_marker_a_pos)

        if self.gps_marker_b_pos < value:
            self.gps_marker_b_slider.set_value(value)
            self.gps_marker_b_pos = value
            self.map_widget.set_marker_b_pos(self.gps_marker_b_pos)

        self.async_update()

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
        self.gps_marker_a_slider.set_range(self.gps_marker_a_slider.minimum, value)
        self.gps_marker_b_slider.set_range(self.gps_marker_b_slider.minimum, value)

        # Ensure GPS markers are within bounds
        if self.gps_marker_a_pos > value:
            self.gps_marker_a_slider.set_value(value)
            self.gps_marker_a_pos = value
            self.map_widget.set_marker_a_pos(self.gps_marker_a_pos)

        if self.gps_marker_b_pos > value:
            self.gps_marker_b_slider.set_value(value)
            self.gps_marker_b_pos = value
            self.map_widget.set_marker_b_pos(self.gps_marker_b_pos)

        self.async_update()

        # Update map to show trim points
        self.map_widget.set_trim_end(self.trim_end)
        self.map_widget.update()

    def on_gps_marker_a_changed(self, value):
        """Handle GPS marker A slider value change"""
        self.gps_marker_a_pos = int(value)
        self.map_widget.set_marker_a_pos(self.gps_marker_a_pos)

        self.async_update()

        # Update map to show markers
        self.map_widget.update()

    def on_gps_marker_b_changed(self, value):
        """Handle GPS marker B slider value change"""
        self.gps_marker_b_pos = int(value)
        self.map_widget.set_marker_b_pos(self.gps_marker_b_pos)

        self.async_update()

        # Update map to show markers
        self.map_widget.update()

    def on_cda_changed(self, value):
        """Handle CdA slider value change"""
        self.current_cda = value

        self.async_update(detect_sections=False)
        self.update_config_text()

    def on_crr_changed(self, value):
        """Handle Crr slider value change"""
        self.current_crr = value

        self.async_update(detect_sections=False)
        self.update_config_text()

    def save_results(self):
        """Save analysis results with the new error calculation"""
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
        csv_path = os.path.join(self.result_dir, f"{file_basename}_outback_results.csv")

        # Prepare data for CSV
        lap_str = "_".join(map(str, sorted(self.selected_laps)))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        selected_section_indices = self.selected_section_indices

        # Calculate total error based on elevation gain difference
        total_error = 0
        sum_of_both_abs = 0

        for i in selected_section_indices:
            if i < len(self.section_ve_profiles):
                section_data = self.section_ve_profiles[i]

                # Calculate outbound gain
                outbound_gain = None
                if (
                    section_data["outbound_ve"] is not None
                    and len(section_data["outbound_ve"]) > 0
                ):
                    outbound_ve = section_data["outbound_ve"]
                    if (
                        hasattr(self, "mean_actual_elevation")
                        and self.mean_actual_elevation is not None
                    ):
                        # Calibrate against mean elevation
                        calibrated_ve = (
                            outbound_ve - outbound_ve[0] + self.mean_actual_elevation[0]
                        )
                    else:
                        calibrated_ve = outbound_ve

                    outbound_gain = calibrated_ve[-1] - calibrated_ve[0]
                    outbound_error = calibrated_ve[-1] - self.mean_actual_elevation[-1]

                # Calculate inbound gain
                inbound_gain = None
                if (
                    section_data["inbound_ve"] is not None
                    and len(section_data["inbound_ve"]) > 0
                ):
                    inbound_ve = section_data["inbound_ve"]
                    if (
                        hasattr(self, "mean_actual_elevation")
                        and self.mean_actual_elevation is not None
                    ):
                        # Calibrate against mean elevation
                        calibrated_ve = (
                            inbound_ve - inbound_ve[0] + self.mean_actual_elevation[-1]
                        )
                    else:
                        calibrated_ve = inbound_ve

                    inbound_gain = calibrated_ve[-1] - calibrated_ve[0]
                    inbound_error = calibrated_ve[-1] - self.mean_actual_elevation[0]

                # Calculate error for this section
                if outbound_gain is not None and inbound_gain is not None:
                    section_error = abs(outbound_gain + inbound_gain)
                    sum_of_both_abs += abs(outbound_error) + abs(inbound_error)
                    total_error += section_error

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
            "gps_marker_a_pos": self.gps_marker_a_pos,
            "gps_marker_b_pos": self.gps_marker_b_pos,
            "detected_sections": len(self.detected_sections),
            "selected_sections": len(selected_section_indices),
            "total_error": total_error,
            "sum_of_both_abs": sum_of_both_abs,
        }

        # Rest of the save_results method remains unchanged
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

        plot_filename = f"{file_basename}_outback_laps_{lap_str}_{config_name}.png"
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
            "gps_marker_a_pos": self.gps_marker_a_pos,
            "gps_marker_b_pos": self.gps_marker_b_pos,
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
        self.async_update(update_ve=False, detect_sections=False)

    def async_update(self, update_ve=True, detect_sections=True):
        self.selected_section_indices = self.get_selected_section_indices()

        values = {}
        for key in VEWorker.INPUT_KEYS:
            values[key] = getattr(self, key)
        values["detect_sections"] = detect_sections and self.has_gps
        values["update_ve"] = update_ve
        self.ve_worker.set_values(values)

    def on_ve_result_ready(self, res):
        for key in VEWorker.RESULT_KEYS:
            if key in res:
                setattr(self, key, res[key])

        if res["detect_sections"]:
            # Update the section table
            self.update_section_table()

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
