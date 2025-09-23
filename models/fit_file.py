import logging
import threading

import numpy as np
import pandas as pd
import rasterio
from fitparse import FitFile as FitParser
from pyproj import Transformer
from rasterio.windows import Window

logger = logging.getLogger(__name__)


class CancelledError(Exception):
    """Raised when an operation is cancelled by the user."""

    pass


class AltitudeLookup:
    TILE_SIZE = 512

    def __init__(self, dem_path):
        self.dataset = rasterio.open(dem_path)
        self.transformer = Transformer.from_crs(
            "EPSG:4326", self.dataset.crs, always_xy=True
        )

    def batch_lookup(self, lats, lons):
        # Transform all coordinates
        xs, ys = self.transformer.transform(lons, lats)

        # Find which cell (row, col) in the raster matches each point
        index_func = np.vectorize(self.dataset.index)
        rows, cols = index_func(xs, ys)

        # Create dataframe for easier grouping
        df = pd.DataFrame(
            {
                "lat": lats,
                "lon": lons,
                "row": rows,
                "col": cols,
            }
        )

        # Assign tile group
        df["tile_row"] = df["row"] // self.TILE_SIZE
        df["tile_col"] = df["col"] // self.TILE_SIZE

        altitudes = np.full(len(df), np.nan)

        for (tile_r, tile_c), group in df.groupby(["tile_row", "tile_col"]):
            window = Window(
                tile_c * self.TILE_SIZE,
                tile_r * self.TILE_SIZE,
                self.TILE_SIZE,
                self.TILE_SIZE,
            )
            try:
                tile = self.dataset.read(1, window=window)
            except Exception as e:
                print(f"Tile read error: {e}")
                continue

            for idx, row in group.iterrows():
                local_r = int(row["row"] - tile_r * self.TILE_SIZE)
                local_c = int(row["col"] - tile_c * self.TILE_SIZE)

                if (0 <= local_r < tile.shape[0]) and (0 <= local_c < tile.shape[1]):
                    value = tile[local_r, local_c]
                    if value != self.dataset.nodata:
                        altitudes[idx] = value

        return altitudes

    def close(self):
        self.dataset.close()


class FitFile:
    def __init__(self, filename, dem_filename):
        """Load and parse a FIT file"""
        self.elevation = AltitudeLookup(dem_filename) if dem_filename else None
        self.elevation_error_rate = 0

        self.filename = filename
        self.fit_parser = FitParser(filename)

        # Extract data
        self.laps = []
        self.location = []
        self.speed = []
        self.power = []
        self.timestamps = []
        self.cancel_event = threading.Event()

    def parse(self):
        self.parse_data()
        self.resample_data()

    def cancel(self):
        self.cancel_event.set()

    def check_canceled(self):
        if self.cancel_event.is_set():
            raise CancelledError("cancelled")

    def parse_data(self):
        """Parse data from FIT file"""
        # Temporary storage for records
        records = []
        lap_records = []

        # Process lap messages
        for message in self.fit_parser.get_messages("lap"):
            lap_data = {}
            for data in message:
                # Include enhanced_avg_speed in data collection
                if data.name in [
                    "start_time",
                    "timestamp",
                    "total_elapsed_time",
                    "total_distance",
                    "avg_power",
                    "avg_speed",
                    "enhanced_avg_speed",  # Added enhanced_avg_speed
                    "max_speed",
                    "enhanced_max_speed",  # Added enhanced_max_speed
                    "start_position_lat",
                    "start_position_long",
                ]:
                    lap_data[data.name] = data.value

            # Prefer enhanced values over non-enhanced ones
            if "enhanced_avg_speed" in lap_data:
                lap_data["avg_speed"] = lap_data["enhanced_avg_speed"]

            if "enhanced_max_speed" in lap_data:
                lap_data["max_speed"] = lap_data["enhanced_max_speed"]

            if (
                "start_time" in lap_data
                and "timestamp" in lap_data
                and "total_elapsed_time" in lap_data
            ):
                # In FIT files, 'timestamp' is usually the end time of the lap
                lap_data["end_time"] = lap_data["timestamp"]

                # Verify times make sense - if timestamp equals start_time, calculate proper end_time
                if lap_data["end_time"] <= lap_data["start_time"]:
                    # Calculate end time from start time and duration
                    if (
                        "total_elapsed_time" in lap_data
                        and lap_data["total_elapsed_time"] > 0
                    ):
                        lap_data["end_time"] = lap_data["start_time"] + pd.Timedelta(
                            seconds=lap_data["total_elapsed_time"]
                        )

                # Convert start position from semicircles to degrees if available
                if (
                    "start_position_lat" in lap_data
                    and "start_position_long" in lap_data
                ):
                    if (
                        lap_data["start_position_lat"] is not None
                        and lap_data["start_position_long"] is not None
                    ):
                        if isinstance(
                            lap_data["start_position_lat"], (int, np.int32, np.int64)
                        ):
                            lap_data["start_position_lat"] = lap_data[
                                "start_position_lat"
                            ] * (180 / 2**31)
                            lap_data["start_position_long"] = lap_data[
                                "start_position_long"
                            ] * (180 / 2**31)
            lap_records.append(lap_data)

        self.check_canceled()

        # Process record messages
        for message in self.fit_parser.get_messages("record"):
            record_data = {}
            for data in message:
                if data.name in [
                    "timestamp",
                    "distance",
                    "position_lat",
                    "position_long",
                    "speed",
                    "enhanced_speed",  # Added enhanced_speed
                    "power",
                    "altitude",
                    "enhanced_altitude",  # Added enhanced_altitude
                    "wind_speed",  # Added wind_speed
                ]:
                    record_data[data.name] = data.value

            # Prefer enhanced values over non-enhanced ones
            if "enhanced_speed" in record_data:
                record_data["speed"] = record_data["enhanced_speed"]

            if "enhanced_altitude" in record_data:
                record_data["altitude"] = record_data["enhanced_altitude"]

            # Convert wind_speed from m/s * 1000 to m/s
            if "wind_speed" in record_data and record_data["wind_speed"] is not None:
                record_data["wind_speed"] = record_data["wind_speed"] / 1000.0

            if "timestamp" in record_data:
                records.append(record_data)

        self.check_canceled()

        # Convert to pandas dataframes
        self.records_df = pd.DataFrame(records)
        initial_record_count = len(self.records_df)
        logger.debug(f"Initial record count: {initial_record_count}")

        # Check if power data exists, if not create column with zeros
        if "power" not in self.records_df.columns or self.records_df["power"].isna().all():
            logger.info("No power data found in FIT file, using zeros for all power values")
            self.records_df["power"] = 0.0

        # drop any row from records_df that has a NaN value in timestamp or speed columns
        self.records_df.dropna(subset=["timestamp", "speed"], inplace=True)
        after_dropna_count = len(self.records_df)
        logger.debug(
            f"After dropping NaN (timestamp/speed): {after_dropna_count} (removed {initial_record_count - after_dropna_count})"
        )

        self.laps_df = pd.DataFrame(lap_records)
        logger.debug(f"Total laps found: {len(self.laps_df)}")

        # Check if we have required data
        if len(self.records_df) == 0:
            raise ValueError("No record data found in FIT file")

        if (
            "position_lat" not in self.records_df.columns
            or "position_long" not in self.records_df.columns
        ):
            self.has_gps = False
        else:
            self.has_gps = True
            self.elevation_error_count = 0
            # Convert semicircles to degrees for lat/long
            if self.records_df["position_lat"].dtype in [
                np.int32,
                np.int64,
                np.float64,
            ]:
                self.records_df["position_lat"] = self.records_df["position_lat"] * (
                    180 / 2**31
                )
                self.records_df["position_long"] = self.records_df["position_long"] * (
                    180 / 2**31
                )

                if self.elevation:
                    lat_col = self.records_df["position_lat"].values
                    lon_col = self.records_df["position_long"].values

                    self.check_canceled()

                    alts = self.elevation.batch_lookup(lat_col, lon_col)

                    # Fallback to original alt if DEM fails
                    alt_col = self.records_df["altitude"].values
                    invalid_alt_mask = np.isnan(alts)
                    corrected_alt = np.where(invalid_alt_mask, alt_col, alts)

                    self.records_df["altitude"] = corrected_alt
                    self.elevation_error_rate = invalid_alt_mask.sum() / len(alts)

        self.check_canceled()

    def resample_data(self):
        """Resample data to 1s intervals"""
        if "timestamp" not in self.records_df.columns:
            raise ValueError("No timestamp data found in FIT file")

        # Fill NaN power values with zeros before resampling
        if "power" in self.records_df.columns:
            nan_power_count = self.records_df["power"].isna().sum()
            if nan_power_count > 0:
                logger.debug(f"Filling {nan_power_count} NaN power values with zeros")
                self.records_df["power"] = self.records_df["power"].fillna(0.0)

        # Set timestamp as index
        self.records_df.set_index("timestamp", inplace=True)
        logger.debug(f"Records before resampling: {len(self.records_df)}")

        # Resample to 1s intervals (use 's' instead of 'S' to avoid FutureWarning)
        self.resampled_df = self.records_df.resample("1s").interpolate(method="linear")
        logger.debug(f"Records after resampling: {len(self.resampled_df)}")

        # Fill any power NaN values created during resampling with zeros
        if "power" in self.resampled_df.columns:
            resampled_nan_power_count = self.resampled_df["power"].isna().sum()
            if resampled_nan_power_count > 0:
                logger.debug(f"Filling {resampled_nan_power_count} NaN power values created during resampling with zeros")
                self.resampled_df["power"] = self.resampled_df["power"].fillna(0.0)

        # Reset index to have timestamp as a column
        self.resampled_df.reset_index(inplace=True)

    def get_lap_data(self):
        """Return processed lap data with calculated fields"""
        lap_data = []

        for i, lap in self.laps_df.iterrows():
            start_time = lap["start_time"]

            # Use 'end_time' if available, otherwise fallback to 'timestamp'
            if "end_time" in lap:
                end_time = lap["end_time"]
            else:
                # Calculate end time from start time and duration
                if "total_elapsed_time" in lap and lap["total_elapsed_time"] > 0:
                    end_time = start_time + pd.Timedelta(
                        seconds=lap["total_elapsed_time"]
                    )
                else:
                    end_time = lap["timestamp"]  # Fallback to timestamp

            # Ensure end time is after start time
            if (
                end_time <= start_time
                and "total_elapsed_time" in lap
                and lap["total_elapsed_time"] > 0
            ):
                end_time = start_time + pd.Timedelta(seconds=lap["total_elapsed_time"])

            # Filter records within this lap
            lap_records = self.resampled_df[
                (self.resampled_df["timestamp"] >= start_time)
                & (self.resampled_df["timestamp"] <= end_time)
            ]
            logger.debug(
                f"Lap {i + 1}: {len(lap_records)} records between {start_time} and {end_time}"
            )

            # Calculate distance and duration
            duration_seconds = lap["total_elapsed_time"]
            distance_km = lap.get("total_distance", 0) / 1000  # Convert m to km

            # Get average power from lap data or calculate from records
            if "avg_power" in lap and lap["avg_power"] is not None:
                avg_power = lap["avg_power"]
            elif "power" in lap_records.columns and not lap_records.empty:
                avg_power = lap_records["power"].mean()
            else:
                avg_power = 0

            # Get average speed directly from lap data - NO recalculation
            avg_speed_kmh = 0
            if "avg_speed" in lap and lap["avg_speed"] is not None:
                # Convert from m/s to km/h
                avg_speed_kmh = lap["avg_speed"] * 3.6

            lap_info = {
                "lap_number": i + 1,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration_seconds,
                "distance": distance_km,
                "avg_power": avg_power,
                "avg_speed": avg_speed_kmh,
            }

            # Get position data
            if "start_position_lat" in lap and "start_position_long" in lap:
                if pd.notna(lap["start_position_lat"]) and pd.notna(
                    lap["start_position_long"]
                ):
                    lap_info["start_lat"] = lap["start_position_lat"]
                    lap_info["start_lon"] = lap["start_position_long"]
            elif (
                self.has_gps
                and not lap_records.empty
                and "position_lat" in lap_records.columns
                and "position_long" in lap_records.columns
            ):
                # Find first record with valid position data
                valid_positions = lap_records.dropna(
                    subset=["position_lat", "position_long"]
                )
                if not valid_positions.empty:
                    first_record = valid_positions.iloc[0]
                    lap_info["start_lat"] = first_record["position_lat"]
                    lap_info["start_lon"] = first_record["position_long"]

            lap_data.append(lap_info)

        return lap_data

    def get_records_for_laps(self, lap_numbers):
        """Return all records for the specified laps"""
        if not self.laps_df.empty:
            # Filter records for selected laps
            selected_records = pd.DataFrame()

            for lap_num in lap_numbers:
                if lap_num <= len(self.laps_df):
                    lap = self.laps_df.iloc[lap_num - 1]
                    start_time = lap["start_time"]

                    logger.debug(
                        f"Lap {lap_num} raw data: start_time={lap.get('start_time')}, end_time={lap.get('end_time')}, timestamp={lap.get('timestamp')}, total_elapsed_time={lap.get('total_elapsed_time')}"
                    )

                    # Use end_time instead of timestamp
                    if "end_time" in lap:
                        end_time = lap["end_time"]
                        logger.debug(f"Lap {lap_num}: Using end_time = {end_time}")
                    else:
                        # Calculate end time from start time and duration
                        if (
                            "total_elapsed_time" in lap
                            and lap["total_elapsed_time"] > 0
                        ):
                            end_time = start_time + pd.Timedelta(
                                seconds=lap["total_elapsed_time"]
                            )
                            logger.debug(
                                f"Lap {lap_num}: Calculated end_time from duration = {end_time}"
                            )
                        else:
                            end_time = lap["timestamp"]  # Fallback to timestamp
                            logger.debug(
                                f"Lap {lap_num}: Using timestamp fallback = {end_time}"
                            )

                    lap_records = self.resampled_df[
                        (self.resampled_df["timestamp"] >= start_time)
                        & (self.resampled_df["timestamp"] <= end_time)
                    ]
                    logger.debug(
                        f"Selected lap {lap_num}: {len(lap_records)} records between {start_time} and {end_time}"
                    )

                    selected_records = pd.concat([selected_records, lap_records])

            logger.debug(
                f"Total selected records for analysis: {len(selected_records)}"
            )

            # Log data quality issues
            if len(selected_records) > 0:
                nan_speed_count = selected_records["speed"].isna().sum()
                nan_power_count = selected_records["power"].isna().sum()
                zero_speed_count = (selected_records["speed"] == 0).sum()
                logger.debug(
                    f"Data quality - NaN speed: {nan_speed_count}, NaN power: {nan_power_count}, Zero speed: {zero_speed_count}"
                )

            # Ensure we have enough data points
            if len(selected_records) < 30:
                logger.error(
                    f"Not enough data points for analysis: {len(selected_records)} < 30"
                )
                logger.error(f"Selected lap numbers: {lap_numbers}")
                logger.error(f"Resampled data total records: {len(self.resampled_df)}")
                raise ValueError(
                    f"Not enough data points ({len(selected_records)} < 30)"
                )

            return selected_records

        return self.resampled_df  # Return all records if no laps defined

    def get_elevation_error_rate(self):
        return self.elevation_error_rate

    def has_wind_speed_data(self):
        """Check if the FIT file contains wind speed data"""
        if not hasattr(self, "resampled_df") or self.resampled_df is None:
            return False

        if "wind_speed" not in self.resampled_df.columns:
            return False

        # Check if there's any non-null wind speed data
        wind_data = self.resampled_df["wind_speed"]
        return not wind_data.isna().all()

    def get_wind_speed_data(self):
        """Get wind speed data if available, otherwise return None"""
        if not self.has_wind_speed_data():
            return None
        return self.resampled_df["wind_speed"].values
