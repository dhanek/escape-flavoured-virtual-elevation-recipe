import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class VirtualElevation:
    """
    Virtual Elevation calculator class based on Robert Chung's methodology.
    Ported from R to Python based on R code by Robert Chung (REChung@gmail.com).
    """

    def __init__(self, data, params):
        """
        Initialize the Virtual Elevation calculator.

        Parameters:
        -----------
        data : pandas.DataFrame
            Data with timestamps, speed, power, etc.
        params : dict
            Parameters for the calculation (mass, rho, cda, crr, etc.)
        """
        self.data = data
        self.params = params

        # Extract parameters
        self.kg = params.get("system_mass", 90)
        self.rho = params.get("rho", 1.2)
        self.eta = params.get("eta", 0.98)
        self.dt = 1.0  # Assume 1 second intervals, can be adjusted
        self.cda = params.get("cda")
        self.crr = params.get("crr")
        self.cda_min = params.get("cda_min", 0.15)
        self.cda_max = params.get("cda_max", 0.5)
        self.crr_min = params.get("crr_min", 0.001)
        self.crr_max = params.get("crr_max", 0.03)
        self.velodrome = bool(params.get("velodrome", False))

        # Ensure wind_speed is never None
        self.wind_speed = params.get("wind_speed", 0)
        if self.wind_speed is None:
            self.wind_speed = 0

        self.wind_direction = params.get("wind_direction")

        # FIT file wind speed data (array of wind speeds per timestamp)
        self.fit_wind_speed = params.get("fit_wind_speed")

        # Process data
        self.prepare_data()

    def prepare_data(self):
        """
        Prepare the data for virtual elevation calculation.
        - Calculate acceleration
        - Adjust for zero speed values
        - Compute effective wind speed considering rider direction
        """
        # Make a copy to avoid modifying the original data
        self.df = self.data.copy()
        initial_count = len(self.df)
        logger.debug(f"VirtualElevation: Initial data count: {initial_count}")

        # Ensure speed is properly extracted (always in m/s from FIT files)
        if "speed" in self.df.columns:
            self.df["v"] = self.df["speed"]
        else:
            raise ValueError("Speed data is required")

        # Ensure power is present, create zeros if missing
        if "power" not in self.df.columns:
            logger.info("No power data found, using zeros for all power values")
            self.df["power"] = 0.0
        elif self.df["power"].isna().all():
            logger.info("All power data is NaN, using zeros for all power values")
            self.df["power"] = 0.0

        self.df["watts"] = self.df["power"]

        # Handle altitude/elevation data
        if "altitude" in self.df.columns:
            self.df["elevation"] = self.df["altitude"]

        if self.velodrome:
            self.df.loc[:, "elevation"] = 0
            if "altitude" in self.df.columns:
                self.df.loc[:, "altitude"] = 0

        # Log data quality before processing
        zero_speed_count = (self.df["v"] == 0).sum()
        nan_speed_count = self.df["v"].isna().sum()
        nan_power_count = self.df["watts"].isna().sum()
        logger.debug(
            f"VirtualElevation: Zero speed: {zero_speed_count}, NaN speed: {nan_speed_count}, NaN power: {nan_power_count}"
        )

        # Handle zero speed (set to 0.001 as in the R code)
        self.df.loc[self.df["v"] == 0, "v"] = 0.001

        # Calculate acceleration
        self.calculate_acceleration()

        # Calculate effective wind based on wind source
        if self.fit_wind_speed is not None:
            # Use FIT file wind speed data (already relative to rider direction)
            self.apply_fit_wind_speed()
        elif self.wind_speed != 0 and self.wind_direction is not None:
            # Use constant wind with direction calculation
            self.calculate_effective_wind()
        else:
            # Set a default value of 0 for wind speed
            self.df["vw"] = 0

        logger.debug(
            f"VirtualElevation: Data preparation complete, final count: {len(self.df)}"
        )

    def virtual_slope(self, cda=None, crr=None):
        """
        Calculate virtual slope with improved wind direction handling.

        Parameters:
        -----------
        cda : float or None
            Drag coefficient * frontal area. If None, use instance value.
        crr : float or None
            Coefficient of rolling resistance. If None, use instance value.

        Returns:
        --------
        numpy.ndarray
            Virtual slope values
        """
        import numpy as np

        if cda is None:
            cda = self.cda
        if crr is None:
            crr = self.crr

        if cda is None or crr is None:
            raise ValueError("CdA and Crr must be provided")

        # Extract values for calculation
        w = self.df["watts"].values * self.eta
        vg = self.df["v"].values
        acc = self.df["a"].values

        # Use effective wind from dataframe (calculated in prepare_data)
        if "vw" in self.df.columns:
            effective_wind = self.df["vw"].values
        else:
            # Fallback: calculate effective wind if not in dataframe
            if self.wind_speed != 0 and self.wind_direction is not None:
                effective_wind = self.calculate_effective_wind()
            else:
                effective_wind = np.full_like(
                    vg, self.wind_speed if self.wind_speed is not None else 0
                )

        # Apparent velocity is ground velocity + effective wind
        va = vg + effective_wind

        # Initialize result array with zeros (default slope)
        slope = np.zeros_like(vg, dtype=float)

        # Filter out zero or very low velocities to avoid division by zero
        valid_idx = np.where(vg > 0.001)[0]
        logger.debug(
            f"VirtualElevation: virtual_slope filtering - Total points: {len(vg)}, Valid points: {len(valid_idx)}, Filtered out: {len(vg) - len(valid_idx)}"
        )

        if len(valid_idx) > 0:
            # Only process entries with valid velocity
            valid_vg = vg[valid_idx]
            valid_w = w[valid_idx]
            valid_acc = acc[valid_idx]
            valid_va = va[valid_idx]

            # Virtual slope calculation
            valid_slope = (
                (valid_w / (valid_vg * self.kg * 9.807))
                - (cda * self.rho * valid_va**2 / (2 * self.kg * 9.807))
                - crr
                - valid_acc / 9.807
            )

            # Assign results back to full array
            slope[valid_idx] = valid_slope

        # Handle NaNs
        slope[np.isnan(slope)] = 0

        return slope

    def delta_ve(self, cda=None, crr=None):
        """
        Calculate change in virtual elevation.

        Parameters:
        -----------
        cda : float or None
            Drag coefficient * frontal area. If None, use instance value.
        crr : float or None
            Coefficient of rolling resistance. If None, use instance value.

        Returns:
        --------
        numpy.ndarray
            Change in virtual elevation
        """
        if cda is None:
            cda = self.cda
        if crr is None:
            crr = self.crr

        # Calculate virtual slope
        slope = self.virtual_slope(cda, crr)

        # Get ground velocity
        vg = self.df["v"].values

        # Calculate elevation change
        delta_elev = vg * self.dt * np.sin(np.arctan(slope))

        return delta_elev

    def calculate_ve(self, cda=None, crr=None):
        """
        Calculate virtual elevation profile.

        Parameters:
        -----------
        cda : float or None
            Drag coefficient * frontal area. If None, use instance value.
        crr : float or None
            Coefficient of rolling resistance. If None, use instance value.

        Returns:
        --------
        numpy.ndarray
            Cumulative virtual elevation
        """
        delta_elev = self.delta_ve(cda, crr)

        # Cumulative sum to get elevation profile
        ve = np.cumsum(delta_elev)

        return ve

    def plot_virtual_elevation(
        self, cda=None, crr=None, actual_elevation=None, trim_start=0, trim_end=None
    ):
        """
        Plot virtual elevation profile.

        Parameters:
        -----------
        cda : float or None
            Drag coefficient * frontal area. If None, use instance value.
        crr : float or None
            Coefficient of rolling resistance. If None, use instance value.
        actual_elevation : numpy.ndarray or None
            Actual elevation profile for comparison.
        trim_start : int
            Index to start from (default 0).
        trim_end : int or None
            Index to end at (default None for end of data).

        Returns:
        --------
        tuple
            (fig, ax) matplotlib figure and axes
        """
        if cda is None:
            cda = self.cda
        if crr is None:
            crr = self.crr

        # Calculate virtual elevation
        ve = self.calculate_ve(cda, crr)

        # Create time array (assuming 1s intervals)
        time = np.arange(len(ve))

        # Set trim_end to the end if not specified
        if trim_end is None:
            trim_end = len(ve)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Plot virtual elevation
        ax1.plot(time, ve, color="blue", label="Virtual Elevation")

        # Mark trimmed region with higher opacity
        ax1.axvspan(0, trim_start, alpha=0.2, color="gray")
        ax1.axvspan(trim_end, len(ve), alpha=0.2, color="gray")

        # Add vertical lines at trim points
        ax1.axvline(x=trim_start, color="green", linestyle="--", label="Trim Start")
        ax1.axvline(x=trim_end, color="red", linestyle="--", label="Trim End")

        # Plot actual elevation if provided
        if actual_elevation is not None:
            # Ensure same length
            min_len = min(len(ve), len(actual_elevation))
            time_trim = time[:min_len]
            ve_trim = ve[:min_len]
            elev_trim = actual_elevation[:min_len]

            # Normalize to same starting point
            ve_norm = ve_trim - ve_trim[0]
            elev_norm = elev_trim - elev_trim[0]

            ax1.plot(time_trim, elev_norm, color="black", label="Actual Elevation")

            # Plot residuals in the second subplot
            residuals = ve_norm - elev_norm
            ax2.plot(time_trim, residuals, color="gray")
            ax2.axhline(y=0, color="black", linestyle="-")

            # Mark trimmed region in residuals
            ax2.axvspan(0, trim_start, alpha=0.2, color="gray")
            ax2.axvspan(trim_end, len(time_trim), alpha=0.2, color="gray")

            # Add vertical lines at trim points
            ax2.axvline(x=trim_start, color="green", linestyle="--")
            ax2.axvline(x=trim_end, color="red", linestyle="--")

            # Set titles and labels
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Residuals (m)")
            ax2.set_title("Residuals (Virtual - Actual)")

        # Set titles and labels for the main plot
        ax1.set_ylabel("Elevation (m)")
        ax1.set_title("Virtual Elevation Profile")
        ax1.legend()

        # Add text with CdA and Crr values
        cda_str = f"CdA: {cda:.4f}"
        crr_str = f"Crr: {crr:.5f}"
        ax1.text(
            0.02,
            0.95,
            cda_str + "\n" + crr_str,
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Add R² and RMSE if actual elevation is provided
        if actual_elevation is not None:
            # Calculate in trimmed region only
            trim_indices = np.where(
                (time_trim >= trim_start) & (time_trim <= trim_end)
            )[0]
            if len(trim_indices) > 2:  # Need at least 3 points for correlation
                ve_trim_region = ve_norm[trim_indices]
                elev_trim_region = elev_norm[trim_indices]

                # R² calculation
                # Handle case where elevation is constant (velodrome mode)
                if np.std(elev_trim_region) == 0 or np.std(ve_trim_region) == 0:
                    r2 = 0.0  # No correlation possible with constant data
                else:
                    corr = np.corrcoef(ve_trim_region, elev_trim_region)[0, 1]
                    r2 = corr**2

                # RMSE calculation
                rmse = np.sqrt(np.mean((ve_trim_region - elev_trim_region) ** 2))

                r2_str = f"R²: {r2:.3f}"
                rmse_str = f"RMSE: {rmse:.3f} m"
                ax1.text(
                    0.78,
                    0.95,
                    r2_str + "\n" + rmse_str,
                    transform=ax1.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        plt.tight_layout()

        return fig, (ax1, ax2)

    def calculate_acceleration(self):
        """
        Calculate acceleration using method from R code:
        a = diff(v^2)/(2*v[-1]*dt)
        """
        v = self.df["v"].values
        # Initialize with zeros
        a = np.zeros_like(v)

        # Calculate acceleration for points after the first
        for i in range(1, len(v)):
            if v[i] > 0:  # Avoid division by zero
                a[i] = (v[i] ** 2 - v[i - 1] ** 2) / (2 * v[i] * self.dt)

        # Replace NaN and infinite values with 0
        a[np.isnan(a)] = 0
        a[np.isinf(a)] = 0

        self.df["a"] = a

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calculate the bearing between two points.

        Args:
            lat1, lon1: Coordinates of first point (in degrees)
            lat2, lon2: Coordinates of second point (in degrees)

        Returns:
            float: Bearing in degrees (0-360)
        """
        import math

        # Convert to radians
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        # Calculate bearing
        y = math.sin(lon2 - lon1) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
            lat2
        ) * math.cos(lon2 - lon1)
        bearing = math.atan2(y, x)

        # Convert to degrees and normalize to 0-360
        bearing = (math.degrees(bearing) + 360) % 360

        return bearing

    def calculate_rider_directions_smoothed(self, window_size=5):
        """
        Calculate smoothed direction of travel for each point.

        Args:
            window_size (int): Size of window for smoothing directions

        Returns:
            numpy.ndarray: Smoothed direction of travel at each point (in degrees, 0-360)
        """
        import numpy as np

        if (
            "position_lat" not in self.df.columns
            or "position_long" not in self.df.columns
        ):
            return np.zeros(len(self.df))

        # Get lat/lon values
        lat = self.df["position_lat"].values
        lon = self.df["position_long"].values

        # Calculate bearing for each segment
        n = len(self.df)
        directions = np.zeros(n)

        # Calculate directions between points
        for i in range(1, n):
            # Skip points with missing data
            if (
                np.isnan(lat[i - 1])
                or np.isnan(lon[i - 1])
                or np.isnan(lat[i])
                or np.isnan(lon[i])
            ):
                continue
            directions[i - 1] = self.calculate_bearing(
                lat[i - 1], lon[i - 1], lat[i], lon[i]
            )

        # Last point gets same direction as second-to-last point
        if n > 1:
            directions[n - 1] = directions[n - 2]

        # Convert directions to x,y components to handle the circular nature of angles
        x_comp = np.cos(np.radians(directions))
        y_comp = np.sin(np.radians(directions))

        # Smooth the x,y components
        window_size = min(window_size, len(x_comp) // 3)
        if window_size >= 3:
            try:
                from scipy.signal import savgol_filter

                x_smooth = savgol_filter(x_comp, window_size, 2)
                y_smooth = savgol_filter(y_comp, window_size, 2)
            except:
                # Fall back to simple moving average if savgol fails
                kernel = np.ones(window_size) / window_size
                x_smooth = np.convolve(x_comp, kernel, mode="same")
                y_smooth = np.convolve(y_comp, kernel, mode="same")
        else:
            x_smooth = x_comp
            y_smooth = y_comp

        # Convert back to angles
        smoothed_directions = (np.degrees(np.arctan2(y_smooth, x_smooth)) + 360) % 360

        return smoothed_directions

    def calculate_effective_wind(self):
        """
        Calculate effective wind velocity based on wind direction and rider's direction of travel.

        Returns:
            numpy.ndarray: Effective wind velocity at each point (positive=headwind, negative=tailwind)
        """
        import numpy as np

        # If no wind direction specified, just return constant wind velocity
        if (
            self.wind_direction is None
            or "position_lat" not in self.df.columns
            or "position_long" not in self.df.columns
        ):
            return np.full(len(self.df), self.wind_speed)

        # Calculate rider directions (smoothed)
        rider_directions = self.calculate_rider_directions_smoothed()

        # Calculate the angle between wind and rider direction
        # Wind direction is where wind is coming FROM (meteorological convention)
        wind_direction = self.wind_direction

        # Calculate relative angle between wind direction and rider direction
        angle_diff = np.abs(wind_direction - rider_directions)

        # Ensure angle is <= 180 degrees (shortest angle between directions)
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)

        # Calculate effective wind component (projection of wind vector onto rider direction)
        # When angle_diff = 0, it's a direct headwind (cos(0°) = 1)
        # When angle_diff = 180, it's a direct tailwind (cos(180°) = -1)
        effective_wind = self.wind_speed * np.cos(np.radians(angle_diff))

        # Correct the sign based on whether wind is coming from front or behind
        is_from_behind = (angle_diff > 90) & (angle_diff < 270)
        effective_wind[is_from_behind] = -np.abs(effective_wind[is_from_behind])

        # Smooth the effective wind values to avoid abrupt changes
        window_size = min(
            11, len(effective_wind) // 10
        )  # Use at most 10% of points but at least 1
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
        window_size = max(1, window_size)  # Ensure at least 1

        if window_size > 1:
            try:
                from scipy.signal import savgol_filter

                effective_wind = savgol_filter(effective_wind, window_size, 2)
            except:
                # If savgol filter fails, use simple moving average
                kernel = np.ones(window_size) / window_size
                effective_wind = np.convolve(effective_wind, kernel, mode="same")

        return effective_wind

    def apply_fit_wind_speed(self):
        """
        Apply FIT file wind speed data directly to the dataframe.
        FIT wind_speed is already relative to rider direction (positive=headwind, negative=tailwind).
        """
        import numpy as np

        if self.fit_wind_speed is None:
            self.df["vw"] = 0
            return

        # Ensure we have the same number of data points
        min_len = min(len(self.df), len(self.fit_wind_speed))

        if min_len < len(self.df):
            # Pad with zeros if FIT wind data is shorter
            wind_data = np.zeros(len(self.df))
            wind_data[:min_len] = self.fit_wind_speed[:min_len]
        else:
            # Trim if FIT wind data is longer
            wind_data = self.fit_wind_speed[: len(self.df)]

        self.df["vw"] = wind_data
