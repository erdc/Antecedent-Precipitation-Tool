import logging
from datetime import datetime, timedelta
from config import get_base_url, load_manifest
import pandas as pd
import xarray as xr
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _get_nclimgrid_url(year: int, month: int) -> str:
    """Construct URL for nClimGrid-Daily netCDF file on NCEI THREDDS server."""
    return (
        f"{get_base_url("nclimgrid_daily")}"
        f"/{year}/ncdd-{year}{month:02d}-grd-scaled.nc"
    )


def get_gridded_precip_timeseries(
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime,
) -> pd.Series:
    """
    Extract daily precipitation time series for a point by fetching
    nClimGrid-Daily files directly from NCEI THREDDS.
    Returns Series with precipitation in mm.
    """
    logger.info(
        f"Fetching nClimGrid precip for ({lat:.4f}, {lon:.4f}) from {start_date.date()} to {end_date.date()}"
    )

    df_list = []
    periods = pd.period_range(start=start_date, end=end_date, freq="M")

    for p in tqdm(periods, desc="Fetching monthly gridded files"):
        year = p.year
        month = p.month
        url = _get_nclimgrid_url(year, month)

        try:
            with xr.open_dataset(url, engine="netcdf4") as ds:
                # Use 'prcp' as in the working version
                precip_var = next(
                    (v for v in ["prcp", "precip", "precipitation"] if v in ds), None
                )
                if not precip_var:
                    logger.error(f"Precipitation variable not found in {url}")
                    continue

                # Select nearest grid point
                subset = ds.sel(lat=lat, lon=lon, method="nearest")
                precip_data = subset[precip_var].compute()

                df_month = precip_data.to_dataframe().reset_index()
                df_list.append(df_month)

        except Exception as e:
            logger.warning(f"Failed to fetch/process {url}: {e}")

    if not df_list:
        logger.warning("No gridded data could be retrieved.")
        return pd.Series(dtype="float64")

    # Combine monthly data
    df = pd.concat(df_list, ignore_index=True)

    # Filter to requested date range
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    df = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]

    if df.empty:
        logger.warning("No data in the requested date range.")
        return pd.Series(dtype="float64")

    # Final cleanup - match expected output format
    df = (
        df[["time", precip_var]]
        .rename(columns={precip_var: "precipitation_mm"})
        .set_index("time")
        .sort_index()
    )

    series = df["precipitation_mm"]

    logger.info(f"Successfully extracted {len(series)} days of gridded precip data")
    return series


def build_composite_precip_series_gridded(
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime,
) -> tuple[pd.Series, list[dict]]:
    """
    Build composite series exactly matching the structure expected by the rest of the system.
    """
    load_manifest()
    precip_series = get_gridded_precip_timeseries(lat, lon, start_date, end_date)

    # Match data_downloader processing: reindex + interpolate + fillna(0)
    if not precip_series.empty:
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        precip_series = (
            precip_series.reindex(date_range).interpolate(method="time").fillna(0)
        )
    else:
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        precip_series = pd.Series(0.0, index=date_range, dtype="float64")

    stations_info = [
        {
            "name": "nClimGrid-Daily (Gridded)",
            "id": "GRIDDED",
            "distance": 0.0,
            "weighted_diff": 0.0,
            "days_normal": "All",
            "days_antecedent": "All",
            "latitude": lat,
            "longitude": lon,
        }
    ]

    return precip_series, stations_info


if __name__ == "__main__":
    end_date = datetime.strptime("2023-11-11", "%Y-%m-%d")
    start_date = end_date - timedelta(days=365)

    series, info = build_composite_precip_series_gridded(
        lat=30.0, lon=-90.0, start_date=start_date, end_date=end_date
    )

    print(f"Extracted {len(series)} days")
    print(series.head())
    print(series.tail())
