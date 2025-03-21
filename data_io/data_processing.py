import numpy as np
import pandas as pd


def resample_data(df, resample_freq="1s"):
    """
    Resample data to a constant time interval.

    Args:
        df (pandas.DataFrame): DataFrame containing time series data
        resample_freq (str): Resampling frequency (e.g., '1s' for 1 second)

    Returns:
        pandas.DataFrame: Resampled DataFrame
    """
    # Ensure the DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index for resampling")

    # Resample to constant time interval
    resampled_df = df.resample(resample_freq).interpolate(method="linear")

    # Remove any NaN values
    resampled_df = resampled_df.dropna()

    return resampled_df


def trim_data_by_distance(
    df, distance, trim_distance=0, trim_start=None, trim_end=None
):
    """
    Trim data from the start and end based on distance.

    Args:
        df (pandas.DataFrame): DataFrame containing cycling data
        distance (numpy.ndarray): Distance data in meters
        trim_distance (float): Distance in meters to trim from both start and end (if trim_start/trim_end not provided)
        trim_start (float): Distance in meters to trim from start (overrides trim_distance for start)
        trim_end (float): Distance in meters to trim from end (overrides trim_distance for end)

    Returns:
        pandas.DataFrame: Trimmed DataFrame
    """
    # Use trim_start and trim_end if provided, otherwise fall back to trim_distance
    start_trim = trim_start if trim_start is not None else trim_distance
    end_trim = trim_end if trim_end is not None else trim_distance

    if start_trim <= 0 and end_trim <= 0:
        return df

    # Find indices where distance is greater than start_trim from start
    start_trim_idx = 0
    if start_trim > 0:
        start_idx = np.where(distance >= start_trim)[0]
        if len(start_idx) == 0:
            print(f"Warning: Cannot trim {start_trim}m from start - not enough data")
        else:
            start_trim_idx = start_idx[0]

    # Find indices where distance is less than end_trim from end
    end_trim_idx = len(distance) - 1
    if end_trim > 0:
        total_distance = distance[-1]
        end_idx = np.where(distance <= (total_distance - end_trim))[0]
        if len(end_idx) == 0:
            print(f"Warning: Cannot trim {end_trim}m from end - not enough data")
        else:
            end_trim_idx = end_idx[-1]

    # Apply trimming if possible
    if start_trim_idx >= end_trim_idx:
        print(
            f"Warning: Cannot trim {start_trim}m from start and {end_trim}m from end - keeping original data"
        )
        return df

    # Return trimmed DataFrame
    return df.iloc[start_trim_idx : end_trim_idx + 1].copy()


def calculate_metrics(df, dt=1):
    """
    Calculate various metrics from cycling data.

    Args:
        df (pandas.DataFrame): DataFrame containing cycling data
        dt (float): Time interval in seconds

    Returns:
        dict: Dictionary of calculated metrics
    """
    metrics = {}

    # Basic statistics
    metrics["distance_meters"] = np.sum(df["v"].values * dt)
    metrics["duration_seconds"] = len(df) * dt

    # Power metrics
    if "watts" in df.columns:
        metrics["avg_power"] = df["watts"].mean()
        metrics["max_power"] = df["watts"].max()
        metrics["normalized_power"] = calculate_np(df["watts"].values)

    # Speed metrics
    metrics["avg_speed"] = df["v"].mean()
    metrics["max_speed"] = df["v"].max()

    # Elevation metrics
    if "elevation" in df.columns:
        elevation_diff = np.diff(df["elevation"].values)
        metrics["elevation_gain"] = np.sum(elevation_diff[elevation_diff > 0])
        metrics["elevation_loss"] = abs(np.sum(elevation_diff[elevation_diff < 0]))

    return metrics


def calculate_np(power_values, window_size=30):
    """
    Calculate Normalized Power® from power data.

    Args:
        power_values (numpy.ndarray): Array of power values in watts
        window_size (int): Rolling average window size in seconds

    Returns:
        float: Normalized Power value
    """
    # Apply 30-second rolling average
    rolling_avg = np.convolve(
        power_values, np.ones(window_size) / window_size, mode="valid"
    )

    # Raise to 4th power
    rolling_avg_4 = np.power(rolling_avg, 4)

    # Take average
    avg_power_4 = np.mean(rolling_avg_4)

    # Take 4th root
    np_value = np.power(avg_power_4, 0.25)

    return np_value
