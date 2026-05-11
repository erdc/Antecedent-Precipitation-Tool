import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from core.precip_downloader import (
    build_composite_precip_series,
)

logger = logging.getLogger(__name__)


def value_list_to_water_year_table(
    dates: pd.DatetimeIndex, values: pd.Series
) -> np.ndarray:
    """Convert time series into array of water years (Oct 1 – Sep 30)."""
    list_of_arrays = []
    try:
        # Find first October 1st
        first_oct_1 = (
            dates.to_series().apply(lambda x: x.month == 10 and x.day == 1).idxmax()
        )
        start_index = dates.get_loc(first_oct_1)
    except (ValueError, KeyError):
        return np.array([])

    current_row = []
    for i in range(start_index, len(dates)):
        date = dates[i]
        value = values.iloc[i]

        # Start new water year on Oct 1
        if date.month == 10 and date.day == 1 and current_row:
            if len(current_row) >= 365:
                list_of_arrays.append(np.array(current_row[:365]))
            current_row = []

        # Skip Feb 29 for consistent 365-day years
        if not (date.month == 2 and date.day == 29) and len(current_row) < 365:
            current_row.append(value)

    # Add final year if complete
    if len(current_row) == 365:
        list_of_arrays.append(np.array(current_row))

    return np.array(list_of_arrays)


def calc_normal_values(dates: pd.DatetimeIndex, values_array: np.ndarray):
    """Calculate 30th and 70th percentile normals for each day of the water year."""
    if values_array.size == 0:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")

    daily_values = values_array.T
    normal_low = np.percentile(daily_values, 30, axis=1)
    normal_high = np.percentile(daily_values, 70, axis=1)

    generic_wy_dates = pd.date_range(start="2022-10-01", periods=365, freq="D")

    low_series = pd.Series(normal_low, index=generic_wy_dates)
    high_series = pd.Series(normal_high, index=generic_wy_dates)

    # Map to target dates
    df = pd.DataFrame(index=dates)
    df["map_key"] = df.index.strftime("%m-%d")

    low_map = low_series.set_axis(low_series.index.strftime("%m-%d"))
    high_map = high_series.set_axis(high_series.index.strftime("%m-%d"))

    normal_low_series = df["map_key"].map(low_map).interpolate(method="time")
    normal_high_series = df["map_key"].map(high_map).interpolate(method="time")

    return normal_low_series, normal_high_series


def compute_rolling_precip_analysis(
    precip_series: pd.Series,
    analysis_date: datetime,
    precip_unit: str = "tenths_mm",
) -> dict:
    """Compute rolling 30-day totals and climatological normals."""
    if precip_unit == "tenths_mm":
        final_df = (precip_series / 10.0) * 0.0393701
    elif precip_unit == "mm":
        final_df = precip_series * 0.0393701
    else:
        final_df = precip_series.copy()

    rolling_30day = final_df.rolling(window=30, min_periods=1).sum()

    # Water year window for graphing
    wy_start = (
        datetime(analysis_date.year, 10, 1)
        if analysis_date.month >= 10
        else datetime(analysis_date.year - 1, 10, 1)
    )
    graph_start = wy_start - timedelta(days=365)
    graph_end = wy_start + timedelta(days=364)

    # Period for computing normals
    norm_start = datetime(analysis_date.year - 31, 10, 1)
    norm_end = datetime(analysis_date.year - 1, 9, 30)

    stats_rolling = rolling_30day[norm_start:norm_end]

    all_days_array = value_list_to_water_year_table(stats_rolling.index, stats_rolling)
    normal_low, normal_high = calc_normal_values(
        pd.date_range(graph_start, graph_end), all_days_array
    )

    return {
        "daily_precip": final_df,
        "rolling_total": rolling_30day,
        "normal_low": normal_low,
        "normal_high": normal_high,
        "graph_start": graph_start,
        "graph_end": graph_end,
        "obs_date": analysis_date,
    }


def run_full_precip_analysis(
    lat: float,
    lon: float,
    analysis_date: datetime,
    data_dir: str,
    output_dir: str,
    use_gridded: bool = False,
) -> dict:
    """
    Complete precipitation analysis pipeline.
    Returns rich dict ready for storage/plotting.
    """
    logger.info(
        f"Starting precip analysis for {lat:.4f}, {lon:.4f} on {analysis_date.date()}"
    )

    start_date = analysis_date - timedelta(days=365 * 32)

    precip_series, stations_info, elevation = build_composite_precip_series(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=analysis_date,
        use_gridded=use_gridded,
    )

    if use_gridded:
        precip_unit = "mm"
    else:
        precip_unit = "tenths_mm"

    if precip_series.empty:
        logger.error("Failed to build precipitation series.")
        return {"error": "No precipitation data available"}

    analysis = compute_rolling_precip_analysis(
        precip_series, analysis_date, precip_unit
    )

    result = {
        "analyzed_data": True,
        "data_source": "Gridded" if use_gridded else "GHCN",
        "precip_data": analysis["daily_precip"],  # kept for backward compatibility
        **analysis,
        "elevation": float(elevation),
        "local_stations_info": stations_info,
        "lat": lat,
        "lon": lon,
        "output_dir": output_dir,
        "data_dir": data_dir,
    }

    logger.info(f"Precip analysis completed successfully for {analysis_date.date()}")
    return result


if __name__ == "__main__":
    end_date = datetime.strptime("2023-11-11", "%Y-%m-%d")
    start_date = end_date - timedelta(days=365)

    series, info = build_composite_precip_series_gridded(
        lat=30.0, lon=-90.0, start_date=start_date, end_date=end_date
    )

    print(f"Extracted {len(series)} days")
    print(series.head())
    print(series.tail())
