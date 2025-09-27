import csv
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from models.virtual_elevation import VirtualElevation
from ui.async_worker import AsyncWorker
from ui.map_widget import MapMode, MapWidget
from ui.slider_textbox import SliderTextBox
from ui.ve_plot import VEFigure, VEPlotLabel, VEPlotSaver

logger = logging.getLogger(__name__)


class VEWorker(AsyncWorker):
    INPUT_KEYS = [
        "trim_start",
        "trim_end",
        "current_cda",
        "current_crr",
        "plot_size_info",
        "wind_source",
    ]

    RESULT_KEYS = [
        "actual_elevation",
        "actual_elevation_diff",
        "distance",
        "r2",
        "rmse",
        "ve_elevation_diff",
        "virtual_elevation_calibrated",
        "virtual_elevation",
        "fig_res",
        # Comparison mode results
        "virtual_elevation_constant_wind",
        "virtual_elevation_fit_wind",
        "virtual_elevation_calibrated_constant_wind",
        "virtual_elevation_calibrated_fit_wind",
        "r2_constant_wind",
        "r2_fit_wind",
        "rmse_constant_wind",
        "rmse_fit_wind",
        "is_comparison_mode",
        # Wind speed plot data
        "wind_plot_fig",
    ]

    def __init__(self, merged_data, params):
        super(VEWorker, self).__init__()
        self.merged_data = merged_data
        self.params = params
        self.ve_calculator = None
        self.ve_valid = False
        self.current_wind_source = None

    def _process_value(self, value: dict):
        for key in VEWorker.INPUT_KEYS:
            setattr(self, key, value[key])

        # Check if we need to recreate VE calculator due to wind source change
        if self.current_wind_source != self.wind_source:
            self._create_ve_calculator()
            self.current_wind_source = self.wind_source
            self.ve_valid = False

        if value["update_ve"] or not self.ve_valid:
            self.calculate_ve()
            self.prepare_plots()
            self.ve_valid = True

        out_values = {}
        if not self.ve_valid:
            return out_values

        self.update_plots()

        # Generate wind plot if wind analysis is available (constant wind or FIT wind)
        has_constant_wind = (
            self.params.get("wind_speed")
            and self.params.get("wind_direction") is not None
        )
        has_fit_wind = self.params.get("fit_wind_data_for_plot") is not None
        has_fit_air = self.params.get("fit_air_data_for_plot") is not None

        if has_constant_wind or has_fit_wind or has_fit_air:
            self.create_wind_plot()
        else:
            self.wind_plot_fig = None

        for key in VEWorker.RESULT_KEYS:
            out_values[key] = getattr(self, key)

        return out_values

    def _create_ve_calculator(self):
        """Create VE calculator with proper data based on wind source"""
        # Get the data for VE calculation
        ve_data = self.merged_data.copy()

        # For constant wind mode, remove air_speed/wind_speed columns so VE uses constant wind
        if self.wind_source == "constant":
            if "air_speed" in ve_data.columns:
                ve_data = ve_data.drop(columns=["air_speed"])
            if "wind_speed" in ve_data.columns:
                ve_data = ve_data.drop(columns=["wind_speed"])

        # Create VE calculator with the prepared data
        self.ve_calculator = VirtualElevation(ve_data, self.params)

    def calculate_ve(self):
        """Calculate virtual elevation with current parameters"""
        # Ensure VE calculator is created
        if self.ve_calculator is None:
            self._create_ve_calculator()

        # Check if we're in comparison mode
        self.is_comparison_mode = self.params.get("comparison_mode", False)

        # Extract actual elevation if available
        self.actual_elevation = None
        if (
            "altitude" in self.merged_data.columns
            and not self.merged_data["altitude"].isna().all()
        ):
            self.actual_elevation = self.merged_data["altitude"].values

        if self.params["velodrome"] and self.actual_elevation is not None:
            self.actual_elevation = np.zeros_like(self.actual_elevation)

        if self.is_comparison_mode:
            # Calculate both constant wind and FIT wind profiles
            self._calculate_comparison_ve()
        else:
            # Calculate single virtual elevation profile
            self._calculate_single_ve()

    def _calculate_single_ve(self):
        """Calculate single virtual elevation profile"""
        # Calculate virtual elevation
        self.virtual_elevation = self.ve_calculator.calculate_ve(
            self.current_cda, self.current_crr
        )

        # Set comparison mode results to None
        self.virtual_elevation_constant_wind = None
        self.virtual_elevation_fit_wind = None
        self.virtual_elevation_calibrated_constant_wind = None
        self.virtual_elevation_calibrated_fit_wind = None
        self.r2_constant_wind = None
        self.r2_fit_wind = None
        self.rmse_constant_wind = None
        self.rmse_fit_wind = None

        self._calculate_metrics_and_calibration(
            self.virtual_elevation,
            "r2",
            "rmse",
            "ve_elevation_diff",
            "virtual_elevation_calibrated",
        )

    def _calculate_comparison_ve(self):
        """Calculate both constant wind and FIT wind virtual elevation profiles"""
        from models.virtual_elevation import VirtualElevation

        # Prepare data for constant wind (remove wind columns so it uses constant wind)
        constant_wind_data = self.merged_data.copy()
        if "air_speed" in constant_wind_data.columns:
            constant_wind_data = constant_wind_data.drop(columns=["air_speed"])
        if "wind_speed" in constant_wind_data.columns:
            constant_wind_data = constant_wind_data.drop(columns=["wind_speed"])

        # Prepare data for FIT wind (keep wind columns)
        fit_wind_data = self.merged_data.copy()

        # Initialize results to None
        self.virtual_elevation_constant_wind = None
        self.virtual_elevation_calibrated_constant_wind = None
        self.r2_constant_wind = None
        self.rmse_constant_wind = None

        try:
            # Create VE calculator for constant wind
            ve_calc_constant = VirtualElevation(constant_wind_data, self.params)
            self.virtual_elevation_constant_wind = ve_calc_constant.calculate_ve(
                self.current_cda, self.current_crr
            )

            # Calculate metrics for constant wind
            if self.virtual_elevation_constant_wind is not None:
                self._calculate_metrics_and_calibration(
                    self.virtual_elevation_constant_wind,
                    "r2_constant_wind",
                    "rmse_constant_wind",
                    "ve_elevation_diff_constant_wind",
                    "virtual_elevation_calibrated_constant_wind",
                )
        except Exception as e:
            logger.error(f"Error calculating constant wind VE: {e}")

        try:
            # Create VE calculator for FIT wind (with wind columns)
            ve_calc_fit = VirtualElevation(fit_wind_data, self.params)
            self.virtual_elevation_fit_wind = ve_calc_fit.calculate_ve(
                self.current_cda, self.current_crr
            )

            # For primary results, use FIT wind results
            self.virtual_elevation = self.virtual_elevation_fit_wind

            # Calculate metrics for FIT wind
            if self.virtual_elevation_fit_wind is not None:
                self._calculate_metrics_and_calibration(
                    self.virtual_elevation_fit_wind,
                    "r2_fit_wind",
                    "rmse_fit_wind",
                    "ve_elevation_diff_fit_wind",
                    "virtual_elevation_calibrated_fit_wind",
                )

                # Set primary results to FIT wind results
                self.r2 = self.r2_fit_wind
                self.rmse = self.rmse_fit_wind
                self.ve_elevation_diff = self.ve_elevation_diff_fit_wind
                self.virtual_elevation_calibrated = self.virtual_elevation_calibrated_fit_wind
        except Exception as e:
            logger.error(f"Error calculating FIT wind VE: {e}")

    def _calculate_metrics_and_calibration(
        self, ve_data, r2_attr, rmse_attr, diff_attr, calibrated_attr
    ):
        """Calculate metrics and calibration for a virtual elevation profile"""
        if self.actual_elevation is not None:
            # Ensure same length
            min_len = min(len(ve_data), len(self.actual_elevation))
            ve_trim = ve_data[:min_len]
            elev_trim = self.actual_elevation[:min_len]

            # Calibrate to match at trim start
            trim_start_idx = self.trim_start
            if trim_start_idx < min_len:
                # Calculate offset to make virtual elevation match actual at trim start
                offset = elev_trim[trim_start_idx] - ve_trim[trim_start_idx]
                ve_calibrated = ve_trim + offset
                setattr(self, calibrated_attr, ve_calibrated)
            else:
                setattr(self, calibrated_attr, ve_trim)

            # Calculate metrics in trimmed region
            trim_indices = np.where(
                (np.arange(len(ve_trim)) >= self.trim_start)
                & (np.arange(len(ve_trim)) <= self.trim_end)
            )[0]

            if len(trim_indices) > 2:  # Need at least 3 points for correlation
                ve_trim_region = getattr(self, calibrated_attr)[trim_indices]
                elev_trim_region = elev_trim[trim_indices]

                # R² calculation
                if np.std(elev_trim_region) == 0 or np.std(ve_trim_region) == 0:
                    setattr(self, r2_attr, 0.0)
                else:
                    corr = np.corrcoef(ve_trim_region, elev_trim_region)[0, 1]
                    setattr(self, r2_attr, corr**2)

                # RMSE calculation
                setattr(
                    self,
                    rmse_attr,
                    np.sqrt(np.mean((ve_trim_region - elev_trim_region) ** 2)),
                )

                # Calculate elevation difference
                safe_trim_end = min(self.trim_end, len(ve_trim) - 1)
                safe_trim_start = min(self.trim_start, safe_trim_end)

                ve_calibrated = getattr(self, calibrated_attr)
                setattr(
                    self,
                    diff_attr,
                    ve_calibrated[safe_trim_end] - ve_calibrated[safe_trim_start],
                )

                if not hasattr(self, "actual_elevation_diff"):
                    self.actual_elevation_diff = (
                        elev_trim[safe_trim_end] - elev_trim[safe_trim_start]
                    )
            else:
                setattr(self, r2_attr, 0)
                setattr(self, rmse_attr, 0)
                setattr(self, diff_attr, 0)
                if not hasattr(self, "actual_elevation_diff"):
                    self.actual_elevation_diff = 0
        else:
            # If no actual elevation data, still create a calibrated version
            setattr(self, calibrated_attr, ve_data.copy())
            safe_trim_end = min(self.trim_end, len(ve_data) - 1)
            safe_trim_start = min(self.trim_start, safe_trim_end)
            setattr(self, diff_attr, ve_data[safe_trim_end] - ve_data[safe_trim_start])
            setattr(self, r2_attr, 0)
            setattr(self, rmse_attr, 0)

    def prepare_plots(self):
        # Use recorded distance from FIT file (convert to km) and reset to start from 0
        if (
            "distance" in self.merged_data.columns
            and not self.merged_data["distance"].isna().all()
        ):
            # Use recorded distance from FIT file (in meters), reset to start from 0, convert to km
            distance_raw = self.merged_data["distance"].values
            self.distance = (
                distance_raw - distance_raw[0]
            ) / 1000  # Reset to 0 and convert to km
        elif hasattr(self.ve_calculator, "df") and "v" in self.ve_calculator.df.columns:
            # Fallback: calculate cumulative distance from speed (v is in m/s, dt=1s)
            distance_m = np.cumsum(
                self.ve_calculator.df["v"].values * self.ve_calculator.dt
            )
            self.distance = distance_m / 1000  # Convert to km
        else:
            # Final fallback to time-based if no distance or speed data
            self.distance = np.arange(len(self.virtual_elevation)) / 1000

    def update_plots(self):
        """Update the virtual elevation plots"""
        ve_fig = VEFigure(self.plot_size_info)
        fig, ax1, ax2 = ve_fig.get_fig_axes()

        distance = self.distance
        trim_start = self.trim_start

        if (
            self.is_comparison_mode
            and hasattr(self, "virtual_elevation_calibrated_constant_wind")
            and hasattr(self, "virtual_elevation_calibrated_fit_wind")
            and self.virtual_elevation_calibrated_constant_wind is not None
            and self.virtual_elevation_calibrated_fit_wind is not None
        ):
            # Comparison mode: plot both constant wind and FIT wind profiles
            trim_end = min(
                self.trim_end,
                len(self.virtual_elevation_calibrated_constant_wind) - 1,
                len(self.virtual_elevation_calibrated_fit_wind) - 1,
            )

            # Plot constant wind profile
            ax1.plot(
                distance,
                self.virtual_elevation_calibrated_constant_wind,
                color="#000000",
                alpha=0.3,
                linewidth=3,
                label="_nolegend_",
            )
            if trim_start <= trim_end:
                trim_distance = distance[trim_start : trim_end + 1]
                trim_ve_constant = self.virtual_elevation_calibrated_constant_wind[
                    trim_start : trim_end + 1
                ]
                ax1.plot(
                    trim_distance,
                    trim_ve_constant,
                    color="#000000",
                    alpha=1.0,
                    linewidth=4,
                    label="VE (Constant Wind)",
                )

            # Plot FIT wind profile
            ax1.plot(
                distance,
                self.virtual_elevation_calibrated_fit_wind,
                color="#4363d8",
                alpha=0.3,
                linewidth=3,
                label="_nolegend_",
            )
            if trim_start <= trim_end:
                trim_distance = distance[trim_start : trim_end + 1]
                trim_ve_fit = self.virtual_elevation_calibrated_fit_wind[
                    trim_start : trim_end + 1
                ]
                ax1.plot(
                    trim_distance,
                    trim_ve_fit,
                    color="#4363d8",
                    alpha=1.0,
                    linewidth=4,
                    label="VE (FIT Wind)",
                )

        else:
            # Single profile mode: plot virtual elevation with FULL OPACITY in trimmed region, REDUCED OPACITY elsewhere
            trim_end = min(self.trim_end, len(self.virtual_elevation_calibrated) - 1)

            # First plot full curve with reduced opacity
            ax1.plot(
                distance,
                self.virtual_elevation_calibrated,
                color="#4363d8",
                alpha=0.3,
                linewidth=3,
                label="_nolegend_",
            )

            # Then plot just the trimmed region with full opacity
            if trim_start <= trim_end:
                trim_distance = distance[trim_start : trim_end + 1]
                trim_ve = self.virtual_elevation_calibrated[trim_start : trim_end + 1]
                ax1.plot(
                    trim_distance,
                    trim_ve,
                    color="#4363d8",
                    alpha=1.0,
                    linewidth=4,
                    label="Virtual Elevation",
                )

        # Mark trimmed region with higher opacity - use lower opacity (0.1) for excluded regions
        if len(distance) > 0:
            ax1.axvspan(
                0,
                distance[self.trim_start] if self.trim_start < len(distance) else 0,
                alpha=0.1,
                color="gray",
            )
            ax1.axvspan(
                (
                    distance[self.trim_end]
                    if self.trim_end < len(distance)
                    else distance[-1]
                ),
                distance[-1],
                alpha=0.1,
                color="gray",
            )

        # Add vertical lines at trim points WITHOUT adding to legend
        if self.trim_start < len(distance):
            ax1.axvline(
                x=distance[self.trim_start],
                color="green",
                linestyle="--",
                label="_nolegend_",
            )
        if self.trim_end < len(distance):
            ax1.axvline(
                x=distance[self.trim_end],
                color="red",
                linestyle="--",
                label="_nolegend_",
            )

        # Add grid lines
        ax1.grid(True, linestyle="--", alpha=0.3)

        # Plot actual elevation if available
        if self.actual_elevation is not None:
            # Ensure same length
            min_len = min(
                len(self.virtual_elevation_calibrated),
                len(self.actual_elevation),
                len(distance),
            )
            distance_trim = distance[:min_len]
            ve_trim = self.virtual_elevation_calibrated[:min_len]
            elev_trim = self.actual_elevation[:min_len]

            # Plot actual elevation with REDUCED OPACITY outside trim region
            # First plot full curve with reduced opacity
            ax1.plot(
                distance_trim,
                elev_trim,
                color="black",
                alpha=0.3,
                linewidth=2,
                label="_nolegend_",
            )

            # Then plot just the trimmed region with full opacity
            trim_end_safe = min(trim_end, min_len - 1)

            # Ensure we have a valid range
            if trim_start <= trim_end_safe:
                trim_distance = distance_trim[trim_start : trim_end_safe + 1]
                trim_elev = elev_trim[trim_start : trim_end_safe + 1]
                ax1.plot(
                    trim_distance,
                    trim_elev,
                    color="black",
                    alpha=1.0,
                    linewidth=2,
                    label="Actual Elevation",
                )

            # Plot residuals in the second subplot
            residuals = ve_trim - elev_trim

            # First plot full residuals with reduced opacity
            ax2.plot(distance_trim, residuals, color="#a9a9a9", alpha=0.3, linewidth=3)

            # Then plot just the trimmed region with full opacity
            if trim_start <= trim_end_safe:
                trim_distance = distance_trim[trim_start : trim_end_safe + 1]
                trim_residuals = residuals[trim_start : trim_end_safe + 1]
                ax2.plot(
                    trim_distance,
                    trim_residuals,
                    color="#a9a9a9",
                    alpha=1.0,
                    linewidth=4,
                )

            ax2.axhline(y=0, color="black", linestyle="-")

            # Mark trimmed region in residuals - use lower opacity (0.1) for excluded regions
            if len(distance_trim) > 0:
                ax2.axvspan(
                    0,
                    (
                        distance_trim[self.trim_start]
                        if self.trim_start < len(distance_trim)
                        else 0
                    ),
                    alpha=0.1,
                    color="gray",
                )
                ax2.axvspan(
                    (
                        distance_trim[trim_end_safe]
                        if trim_end_safe < len(distance_trim)
                        else distance_trim[-1]
                    ),
                    distance_trim[-1],
                    alpha=0.1,
                    color="gray",
                )

            # Add vertical lines at trim points
            if self.trim_start < len(distance_trim):
                ax2.axvline(
                    x=distance_trim[self.trim_start], color="green", linestyle="--"
                )
            if trim_end_safe < len(distance_trim):
                ax2.axvline(x=distance_trim[trim_end_safe], color="red", linestyle="--")

            # Add grid to residuals plot
            ax2.grid(True, linestyle="--", alpha=0.3)

            # Set titles and labels
            ax2.set_xlabel("Distance (km)")
            ax2.set_ylabel("Residuals (m)")
            ax2.set_title("Residuals (Virtual - Actual)")

        # Set titles and labels for the main plot
        ax1.set_ylabel("Elevation (m)")
        ax1.set_title("Virtual Elevation Profile")
        ax1.legend()

        # Add text with CdA and Crr values - positioned completely outside plot area
        cda_str = f"CdA: {self.current_cda:.3f}"
        crr_str = f"Crr: {self.current_crr:.4f}"
        fig.text(
            0.01,
            0.99,
            cda_str + "\n" + crr_str,
            verticalalignment="top",
            horizontalalignment="left",
            transform=fig.transFigure,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        # Add R², RMSE and elevation gain if calculated
        if hasattr(self, "r2") and hasattr(self, "rmse"):
            r2_str = f"R²: {self.r2:.3f}"
            rmse_str = f"RMSE: {self.rmse:.3f} m"

            # Add elevation gain differences
            if hasattr(self, "ve_elevation_diff") and hasattr(
                self, "actual_elevation_diff"
            ):
                ve_gain_str = f"VE Gain: {self.ve_elevation_diff:.1f} m"
                actual_gain_str = f"Actual Gain: {self.actual_elevation_diff:.1f} m"
                diff_str = f"Gain Diff: {self.ve_elevation_diff - self.actual_elevation_diff:.1f} m"

                metrics_text = f"{r2_str}\n{rmse_str}\n{ve_gain_str}\n{actual_gain_str}\n{diff_str}"
            else:
                metrics_text = f"{r2_str}\n{rmse_str}"

            fig.text(
                0.99,
                0.99,
                metrics_text,
                verticalalignment="top",
                horizontalalignment="right",
                transform=fig.transFigure,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )

        self.fig_res = ve_fig.draw()

    def create_wind_plot(self):
        """Create wind speed analysis plot using EXACT same logic as VE plot"""
        import numpy as np

        # EXACT same VEFigure creation as update_plots
        ve_fig = VEFigure(self.plot_size_info)
        fig, ax1, ax2 = ve_fig.get_fig_axes()

        distance = self.distance
        trim_start = self.trim_start
        trim_end = min(self.trim_end, len(distance) - 1)

        # Prepare wind speed data
        ground_speed = self.merged_data["speed"].values * 3.6  # Convert m/s to km/h

        # Constant wind apparent speed
        if (
            self.params.get("wind_speed")
            and self.params.get("wind_direction") is not None
        ):
            from models.virtual_elevation import VirtualElevation

            ve_temp = VirtualElevation(self.merged_data, self.params)
            effective_wind = ve_temp.calculate_effective_wind() * 3.6
            ground_plus_constant = ground_speed + effective_wind
        else:
            ground_plus_constant = ground_speed

        # FIT air speed or wind apparent speed
        fit_air_data = self.params.get("fit_air_data_for_plot")
        fit_wind_data = self.params.get("fit_wind_data_for_plot")

        if fit_air_data is not None:
            # Use air speed directly (already apparent wind velocity)
            min_len = min(len(ground_speed), len(fit_air_data))
            fit_air_kmh = np.nan_to_num(fit_air_data[:min_len] * 3.6, nan=0.0)
            ground_plus_fit = fit_air_kmh[:min_len]
        elif fit_wind_data is not None:
            # Fall back to wind speed calculation
            min_len = min(len(ground_speed), len(fit_wind_data))
            fit_wind_kmh = np.nan_to_num(fit_wind_data[:min_len] * 3.6, nan=0.0)
            ground_plus_fit = ground_speed[:min_len] + fit_wind_kmh
        else:
            ground_plus_fit = ground_speed

        # EXACT same plotting pattern as VE plot - full curves with reduced opacity
        ax1.plot(
            distance,
            ground_speed,
            color="#a9a9a9",
            alpha=0.3,
            linewidth=3,
            label="_nolegend_",
        )
        ax1.plot(
            distance,
            ground_plus_constant,
            color="#000000",
            alpha=0.3,
            linewidth=3,
            label="_nolegend_",
        )

        # Only plot FIT air/wind if available
        if fit_air_data is not None or fit_wind_data is not None:
            min_len = min(len(distance), len(ground_plus_fit))
            ax1.plot(
                distance[:min_len],
                ground_plus_fit[:min_len],
                color="#4363d8",
                alpha=0.3,
                linewidth=3,
                label="_nolegend_",
            )

        # Trimmed regions with full opacity - EXACT same as VE plot
        if trim_start <= trim_end:
            trim_distance = distance[trim_start : trim_end + 1]
            trim_ground = ground_speed[trim_start : trim_end + 1]
            trim_constant = ground_plus_constant[trim_start : trim_end + 1]

            # Only plot FIT air/wind trimmed region if FIT air/wind data is available
            if fit_air_data is not None or fit_wind_data is not None:
                fit_trim_len = min(
                    len(trim_distance), len(ground_plus_fit) - trim_start
                )
                if fit_trim_len > 0:
                    trim_fit = ground_plus_fit[trim_start : trim_start + fit_trim_len]
                    ax1.plot(
                        trim_distance[:fit_trim_len],
                        trim_fit,
                        color="#4363d8",
                        alpha=1.0,
                        linewidth=4,
                        label="Apparent (FIT Air)" if fit_air_data is not None else "Apparent (FIT Wind)",
                    )

            ax1.plot(
                trim_distance,
                trim_ground,
                color="#a9a9a9",
                alpha=1.0,
                linewidth=4,
                label="Ground Speed",
            )
            ax1.plot(
                trim_distance,
                trim_constant,
                color="#000000",
                alpha=1.0,
                linewidth=4,
                label="Apparent (Constant Wind)",
            )

        # Mark trimmed region - EXACT same as VE plot
        if len(distance) > 0:
            ax1.axvspan(
                0,
                distance[self.trim_start] if self.trim_start < len(distance) else 0,
                alpha=0.1,
                color="gray",
            )
            ax1.axvspan(
                (
                    distance[self.trim_end]
                    if self.trim_end < len(distance)
                    else distance[-1]
                ),
                distance[-1],
                alpha=0.1,
                color="gray",
            )

        # Vertical lines - EXACT same as VE plot
        if self.trim_start < len(distance):
            ax1.axvline(
                x=distance[self.trim_start],
                color="green",
                linestyle="--",
                label="_nolegend_",
            )
        if self.trim_end < len(distance):
            ax1.axvline(
                x=distance[self.trim_end],
                color="red",
                linestyle="--",
                label="_nolegend_",
            )

        # Formatting
        ax1.set_xlabel("Distance (km)")
        ax1.set_ylabel("Speed (km/h)")
        ax1.set_title("Wind Speed Analysis")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Hide second subplot
        ax2.set_visible(False)

        # EXACT same draw method as VE plot
        self.wind_plot_fig = ve_fig.draw()


class AnalysisResult(QMainWindow):
    """Window for displaying virtual elevation analysis results"""

    def __init__(self, parent, fit_file, settings, selected_laps, params):
        super().__init__()
        self.parent = parent
        self.fit_file = fit_file
        self.settings = settings
        self.selected_laps = selected_laps
        self.params = params
        self.result_dir = settings.result_dir

        # Prepare merged lap data
        self.prepare_merged_data()

        # Get lap combination ID for settings
        self.lap_combo_id = "_".join(map(str, sorted(self.selected_laps)))

        # Try to load saved trim values for this lap combination
        file_settings = self.settings.get_file_settings(self.fit_file.filename)
        trim_settings = file_settings.get("trim_settings", {})
        saved_trim = trim_settings.get(self.lap_combo_id, {})

        # Wind source settings - initialize before creating VE worker
        self.has_fit_wind = self.fit_file.has_wind_speed_data()
        self.has_fit_air = self.fit_file.has_air_speed_data()
        self.wind_source = saved_trim.get(
            "wind_source", "constant"
        )  # "constant", "fit", "compare"

        # Ensure valid wind source if no FIT wind or air data
        if not (self.has_fit_wind or self.has_fit_air) and self.wind_source in ["fit", "compare"]:
            self.wind_source = "constant"

        # Initialize UI values
        if saved_trim and "trim_start" in saved_trim and "trim_end" in saved_trim:
            # Use saved trim values if available
            self.trim_start = saved_trim["trim_start"]
            self.trim_end = saved_trim["trim_end"]
        else:
            # Use defaults
            self.trim_start = 0
            self.trim_end = len(self.merged_data)

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

        # Initialize debounce timer for wind source changes
        self.wind_source_timer = QTimer()
        self.wind_source_timer.setSingleShot(True)
        self.wind_source_timer.timeout.connect(self._apply_wind_source_change)
        self.pending_wind_source = None

        # Create VE worker after UI initialization so plot_size_info is available
        self.ve_worker = VEWorker(self.merged_data, self.get_current_params())
        self.ve_thread = QThread()
        self.ve_worker.moveToThread(self.ve_thread)
        self.ve_worker.resultReady.connect(self.on_ve_result_ready)
        self.ve_plot_saver = VEPlotSaver(
            VEWorker(self.merged_data, self.get_current_params()), self.ve_thread
        )
        self.ve_thread.start()
        QApplication.instance().aboutToQuit.connect(self.join_threads)

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
            self.total_distance / self.total_duration
        ) * 3600  # Convert to km/h

    def initUI(self):
        """Initialize the UI components"""
        self.setWindowTitle(
            f"Virtual Elevation Analysis - Laps {', '.join(map(str, self.selected_laps))}"
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
        self.map_widget = MapWidget(MapMode.TRIM, self.merged_data, self.params)
        if self.map_widget.has_gps():
            self.map_widget.set_trim_start(self.trim_start)
            self.map_widget.set_trim_end(self.trim_end)
            self.map_widget.update()
            left_layout.addWidget(self.map_widget, 2)
        else:
            no_gps_label = QLabel("No GPS data available")
            no_gps_label.setAlignment(Qt.AlignCenter)
            left_layout.addWidget(no_gps_label, 2)

        # Parameter display
        param_group = QGroupBox("Analysis Parameters")
        param_layout = QFormLayout()

        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.update_config_text()
        param_layout.addRow("Configuration:", self.config_text)

        # Configuration name input
        self.config_name = QLineEdit("Test")
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

        # Create tabbed plot area
        self.plot_tabs = QTabWidget()

        # Virtual Elevation tab
        ve_tab = QWidget()
        ve_layout = QVBoxLayout(ve_tab)
        self.ve_plot = VEPlotLabel(self.screen())
        ve_layout.addWidget(self.ve_plot)
        self.plot_tabs.addTab(ve_tab, "Virtual Elevation")

        # Wind Speed Analysis tab (if FIT wind data is available OR constant wind is configured)
        has_constant_wind = (
            self.params.get("wind_speed")
            and self.params.get("wind_direction") is not None
        )
        if self.has_fit_wind or self.has_fit_air or has_constant_wind:
            wind_tab = QWidget()
            wind_layout = QVBoxLayout(wind_tab)
            self.wind_plot = VEPlotLabel(self.screen())
            wind_layout.addWidget(self.wind_plot)
            self.plot_tabs.addTab(wind_tab, "Wind Speed Analysis")
        else:
            self.wind_plot = None

        right_layout.addWidget(self.plot_tabs, 3)
        self.plot_size_info = self.ve_plot.get_size_info()

        # Sliders
        slider_group = QGroupBox("Adjust Parameters")
        slider_layout = QFormLayout()

        # Trim start slider
        self.trim_start_slider = SliderTextBox(
            minimum=0,
            maximum=len(self.merged_data) - 30,
            value=self.trim_start,
            is_float=False,
        )
        self.trim_start_slider.valueChanged.connect(self.on_trim_start_changed)

        slider_layout.addRow("Trim Start (s):", self.trim_start_slider)

        # Trim end slider
        self.trim_end_slider = SliderTextBox(
            minimum=30,
            maximum=len(self.merged_data),
            value=self.trim_end,
            is_float=False,
        )
        self.trim_end_slider.valueChanged.connect(self.on_trim_end_changed)

        slider_layout.addRow("Trim End (s):", self.trim_end_slider)

        # CdA slider
        self.cda_slider = SliderTextBox(
            minimum=self.params.get("cda_min", 0.15),
            maximum=self.params.get("cda_max", 0.5),
            value=self.current_cda,
            is_float=True,
            decimals=3,
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
            decimals=4,
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

    def on_trim_start_changed(self, value):
        """Handle trim start slider value change"""
        # Ensure trim_start < trim_end - 30
        if value >= self.trim_end_slider.get_value() - 30:
            self.trim_start_slider.set_value(self.trim_end_slider.get_value() - 30)
            return

        self.trim_start = int(value)

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

        self.async_update()

        self.map_widget.set_trim_end(self.trim_end)
        self.map_widget.update()

    def on_cda_changed(self, value):
        """Handle CdA slider value change"""
        self.current_cda = value

        self.async_update()

        self.update_config_text()

    def on_crr_changed(self, value):
        """Handle Crr slider value change"""
        self.current_crr = value

        self.async_update()

        self.update_config_text()

    def on_wind_source_changed(self):
        """Handle wind source radio button changes with debouncing"""
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

        # Store the pending change and start/restart the debounce timer
        self.pending_wind_source = new_wind_source
        self.wind_source_timer.start(300)  # 300ms debounce delay

    def _apply_wind_source_change(self):
        """Apply the wind source change after debounce delay"""
        if self.pending_wind_source and self.pending_wind_source != self.wind_source:
            self.wind_source = self.pending_wind_source

            # Recreate VE worker with new parameters (AsyncWorker doesn't have stop method)
            # Disconnect the old worker first
            if hasattr(self, 've_worker') and self.ve_worker:
                try:
                    self.ve_worker.resultReady.disconnect()
                except (RuntimeError, TypeError):
                    pass  # Ignore if already disconnected or connection doesn't exist

            self.ve_worker = VEWorker(self.merged_data, self.get_current_params())
            self.ve_worker.moveToThread(self.ve_thread)
            self.ve_worker.resultReady.connect(self.on_ve_result_ready)

            self.async_update()
            self.update_config_text()

        self.pending_wind_source = None

    def get_current_params(self):
        """Get current parameters including wind data based on wind source selection"""
        current_params = self.params.copy()

        # Set comparison mode flag
        current_params["comparison_mode"] = self.wind_source == "compare"

        # Add FIT air/wind data separately for wind plot generation (doesn't affect VE calculation)
        if self.has_fit_air and "air_speed" in self.merged_data.columns:
            # Prioritize air speed data for plotting (already filtered for selected laps)
            air_for_laps = self.merged_data["air_speed"].values
            if not np.isnan(air_for_laps).all():
                current_params["fit_air_data_for_plot"] = air_for_laps
        elif self.has_fit_wind and "wind_speed" in self.merged_data.columns:
            # Fall back to wind speed data for plotting (already filtered for selected laps)
            wind_for_laps = self.merged_data["wind_speed"].values
            if not np.isnan(wind_for_laps).all():
                current_params["fit_wind_data_for_plot"] = wind_for_laps

        # Add plot size info for high-quality rendering - always get fresh size info
        if hasattr(self, "wind_plot") and self.wind_plot:
            # Use wind plot's current size for wind plot rendering
            current_params["wind_plot_size_info"] = self.wind_plot.get_size_info()

        # Use VE plot size info as default
        if hasattr(self, "plot_size_info"):
            current_params["plot_size_info"] = (
                self.ve_plot.get_size_info()
            )  # Always fresh

        return current_params

    def _fit_map_to_full_route(self, m):
        """Helper to fit map to full route bounds"""
        if self.route_points:
            lats = [p[0] for p in self.route_points]
            lons = [p[1] for p in self.route_points]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)

            # Add some padding (5%)
            lat_padding = (max_lat - min_lat) * 0.05
            lon_padding = (max_lon - min_lon) * 0.05
            bounds = [
                [min_lat - lat_padding, min_lon - lon_padding],
                [max_lat + lat_padding, max_lon + lon_padding],
            ]
            m.fit_bounds(bounds)

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
        csv_path = os.path.join(self.result_dir, f"{file_basename}_ve_results.csv")

        # Prepare data for CSV
        lap_str = "_".join(map(str, sorted(self.selected_laps)))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

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
            "duration": self.total_duration,
            "distance": self.total_distance,
            "avg_power": self.avg_power,
            "avg_speed": self.avg_speed * 3.6,  # Convert to km/h
        }

        # Add R² and RMSE if available
        if hasattr(self, "r2"):
            result_row["r2"] = self.r2

        if hasattr(self, "rmse"):
            result_row["rmse"] = self.rmse

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

        # Remove existing row with same lap selection if present
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

        plot_filename = f"{file_basename}_laps_{lap_str}_{config_name}.png"
        plot_path = os.path.join(plot_dir, plot_filename)

        self.ve_plot_saver.save(self.ve_worker.get_values(), plot_path)

        # Save trim settings to file settings
        file_settings = self.settings.get_file_settings(self.fit_file.filename)

        # Initialize trim_settings if it doesn't exist
        if "trim_settings" not in file_settings:
            file_settings["trim_settings"] = {}

        # Save trim settings for this lap combination
        file_settings["trim_settings"][self.lap_combo_id] = {
            "trim_start": self.trim_start,
            "trim_end": self.trim_end,
            "cda": self.current_cda,
            "crr": self.current_crr,
            "wind_source": self.wind_source,
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

    def async_update(self, update_ve=True):
        values = {}
        for key in VEWorker.INPUT_KEYS:
            values[key] = getattr(self, key)
        values["update_ve"] = update_ve
        self.ve_worker.set_values(values)

    def on_ve_result_ready(self, res):
        for key in VEWorker.RESULT_KEYS:
            if key in res:
                setattr(self, key, res[key])

        self.ve_plot.set_fig(self.fig_res)
        self.fig_res = None

        # Update wind plot if it exists and we have wind plot data
        if (
            hasattr(self, "wind_plot")
            and hasattr(self, "wind_plot_fig")
            and self.wind_plot_fig
        ):
            self.wind_plot.set_fig(self.wind_plot_fig)
            self.wind_plot_fig = None

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
