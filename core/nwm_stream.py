"""
NWM Streamflow Analysis
"""

import logging
import multiprocessing
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from config import get_base_url
import fsspec
import xarray as xr
import netCDF4
import dask
from dask.diagnostics import ProgressBar
from geopy.distance import great_circle
from shapely.geometry import Point
from simplekml import Kml
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region helper
# ---------------------------------------------------------------------------
def get_special_region(
    lat: float,
    lon: float,
    state_shapefile: str,
) -> str | int:
    """
    Return 'CONUS', a special-region code (AK/HI/PR/...), or -1 on failure.
    state_shapefile must point to tl_2023_us_state.shp (or equivalent URI).
    """
    try:
        usa_gdf = gpd.read_file(state_shapefile)
    except Exception as e:
        logger.error(f"Failed to read State shapefile from {state_shapefile}: {e}")
        return -1

    point = Point(lon, lat)
    joined = gpd.sjoin(
        gpd.GeoDataFrame(geometry=[point], crs=usa_gdf.crs),
        usa_gdf,
        how="left",
        predicate="intersects",
    )

    if joined.empty or pd.isna(joined["STUSPS"].iloc[0]):
        return -1

    stusps = joined["STUSPS"].iloc[0]
    special = {"AK", "HI", "PR", "VI", "GU", "MP", "AS"}
    return stusps if stusps in special else "CONUS"


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def calc_daily_percentiles(
    flow_df: pd.DataFrame, period_end: str = "2020-12-31"
) -> tuple[pd.DataFrame, str]:
    """Day-of-year percentiles from a long discharge series."""
    df = flow_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"] <= pd.to_datetime(period_end)]

    if df.empty:
        return pd.DataFrame(), ""

    hist_por = f"{df['time'].min().date()}->{df['time'].max().date()}"
    df["day"] = df["time"].dt.strftime("%m-%d")
    grouped = df.groupby("day")["discharge"]

    stats = grouped.agg(
        min="min",
        p05=lambda x: np.percentile(x, 5),
        p10=lambda x: np.percentile(x, 10),
        p25=lambda x: np.percentile(x, 25),
        p50="median",
        p75=lambda x: np.percentile(x, 75),
        p90=lambda x: np.percentile(x, 90),
        p95=lambda x: np.percentile(x, 95),
        max="max",
    ).round(3)

    return stats.reset_index(), hist_por


def calc_percentile(current_flow: float, hist_df: pd.DataFrame, date) -> float:
    """Percentile rank of current_flow on the same calendar day. -1 if insufficient data."""
    date = pd.to_datetime(date)
    day = date.strftime("%m-%d")
    same_day = hist_df[hist_df["time"].dt.strftime("%m-%d") == day]["discharge"]

    if same_day.nunique() <= 1:
        return -1.0

    rank = (same_day <= current_flow).mean() * 100
    return round(float(rank), 1)


def condition_from_percentile(perc: float) -> str:
    if perc > 75:
        return "Wet/Above Normal"
    if perc >= 25:
        return "Normal"
    return "Dry/Below Normal"


# ---------------------------------------------------------------------------
# Spatial lookup
# ---------------------------------------------------------------------------
def find_closest_comids(
    comid_shapefile: str,
    lat: float,
    lon: float,
    max_dist: float = 0,
) -> pd.DataFrame:
    """
    Return nearby NWM reaches sorted by distance.
    comid_shapefile must point to nwm_COMID_APTv3.0.shp (or equivalent URI).
    """
    bbox = (lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5)
    try:
        gdf = gpd.read_file(comid_shapefile, bbox=bbox)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read COMID shapefile from {comid_shapefile}: {e}"
        )

    filtered = gdf.assign(
        dist=gdf.geometry.apply(
            lambda geom: great_circle((geom.y, geom.x), (lat, lon)).miles
        )
    ).reset_index(drop=True)

    if max_dist > 0:
        filtered = filtered[filtered["dist"] <= max_dist].copy()

    sorted_df = filtered.sort_values("dist").reset_index(drop=True)
    sorted_df["lat"] = sorted_df.geometry.y
    sorted_df["lon"] = sorted_df.geometry.x

    return sorted_df[["COMID", "dist", "lat", "lon"]]


# ---------------------------------------------------------------------------
# Current-day flow download (analysis_assim NetCDF)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def download_nwm_flow(date: datetime, STUSPS: str, cache_dir: str) -> int:
    """
    Download (or reuse) the analysis_assim NetCDF for the given date/region.
    Files are written under cache_dir.
    Returns 0 on success, HTTP status or -1 on failure.
    """
    date_str = date.strftime("%Y%m%d")
    os.makedirs(cache_dir, exist_ok=True)

    nc_filepath = os.path.join(cache_dir, f"nwm_data_{date_str}_{STUSPS}.nc")
    if os.path.exists(nc_filepath):
        return 0

    base_url = f"{get_base_url('nwm_analysis_assim')}/nwm.{date_str}/analysis_assim"
    if STUSPS == "CONUS":
        file_name = "/nwm.t00z.analysis_assim.channel_rt.tm02.conus.nc"
    elif STUSPS == "HI":
        file_name = "_hawaii/nwm.t00z.analysis_assim.channel_rt.tm0245.hawaii.nc"
    elif STUSPS == "PR":
        file_name = "_puertorico/nwm.t00z.analysis_assim.channel_rt.tm02.puertorico.nc"
    elif STUSPS == "AK":
        file_name = "_alaska/nwm.t00z.analysis_assim.channel_rt.tm02.alaska.nc"
    else:
        logger.error(f"Unsupported STUSPS for analysis_assim: {STUSPS}")
        return -1

    url = f"{base_url}{file_name}"
    logger.debug(url)

    try:
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            return response.status_code

        with open(nc_filepath, "wb") as f:
            f.write(response.content)
        return 0

    except Exception as e:
        logger.error(f"Error retrieving NWM flow on {date}: {e}")
        return -1


def get_nwm_flow(
    comid_list: pd.DataFrame,
    date: datetime,
    STUSPS: str,
    cache_dir: str,
) -> dict | int:
    """
    Ensure the analysis_assim file is present, then extract streamflow for the
    nearest COMIDs.  Returns a dict keyed by COMID string, or -1 on failure.
    """
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    for attempt in range(3):
        ret = download_nwm_flow(date, STUSPS=STUSPS, cache_dir=cache_dir)
        if ret == 0:
            break
        if ret != -1:
            logger.debug(f"download attempt returned code: {ret}")
    else:
        logger.warning("NWM flow data download attempts exceeded 3")
        return -1

    date_str = date.strftime("%Y%m%d")
    nc_filepath = os.path.join(cache_dir, f"nwm_data_{date_str}_{STUSPS}.nc")
    if not os.path.exists(nc_filepath):
        logger.error(f"Expected NetCDF not found: {nc_filepath}")
        return -1

    ds = netCDF4.Dataset(nc_filepath)
    streamflow = ds.variables["streamflow"]
    feature_id = ds.variables["feature_id"]

    points = {}
    for _, row in comid_list.iterrows():
        comid = int(row["COMID"])
        dist = row["dist"]

        matches = np.where(feature_id[:] == comid)[0]
        if len(matches) == 0:
            continue

        flow = streamflow[matches[0]]
        if getattr(flow, "mask", False):
            continue

        actual_flow = round(float(flow) * 35.3147, 4)  # m³/s → cfs
        points[str(comid)] = {
            "flow": actual_flow,
            "flow units": "cfs",
            "COMID": str(comid),
            "distance": round(float(dist), 3),
            "distance unit": "mi",
            "date": date.strftime("%Y-%m-%d"),
        }
        if len(points) >= 10:
            break

    ds.close()
    return points if points else -1


# ---------------------------------------------------------------------------
# Historic (retrospective) download
# ---------------------------------------------------------------------------
def download_nwm_subprocess(comid_list: list, nwm_uri: str, cache_dir: str) -> int:
    """Worker that pulls a slice of the retrospective Zarr and writes a CSV."""
    csv_path = os.path.join(cache_dir, f"nwm_historic_{comid_list[0]}.csv")
    if os.path.exists(csv_path):
        return 0

    os.makedirs(cache_dir, exist_ok=True)

    fs = fsspec.get_mapper(nwm_uri, anon=True, use_ssl=False)
    ds = xr.open_zarr(fs)

    ds_filtered = ds.sel(feature_id=comid_list, time=slice("1990-01-01", "2020-12-31"))
    streamflow_data = ds_filtered["streamflow"]

    logger.info("Downloading NWM retrospective data (this may take a while)...")
    with ProgressBar():
        streamflow_data = streamflow_data.compute()

    df = streamflow_data.to_dataframe().reset_index()
    pivoted = (
        df.pivot(index="time", columns="feature_id", values="streamflow")
        .resample("D")
        .mean()
    )
    pivoted = pivoted * 35.3147  # m³/s → cfs

    pivoted.to_csv(csv_path)
    logger.info(f"NWM historic data cached: {csv_path}")
    return 0


def download_nwm_historic(
    comid_list: list,
    STUSPS: str = "CONUS",
    cache_dir: str = "data",
    timeout: int = 1800,
) -> int:
    """
    Download retrospective streamflow for the given COMIDs into cache_dir.
    Runs in a subprocess so it can be killed on timeout.
    """
    uri_dict = {
        "CONUS": get_base_url("nwm_retro_conus"),
        "AK": get_base_url("nwm_retro_ak"),
        "HI": get_base_url("nwm_retro_hi"),
        "PR": get_base_url("nwm_retro_pr"),
    }
    nwm_uri = uri_dict.get(STUSPS)
    if nwm_uri is None:
        logger.warning(f"Invalid STUSPS code {STUSPS} for retrospective download")
        return -1

    logger.info(f"Starting NWM retrospective download (timeout={timeout // 60} min)")

    process = multiprocessing.Process(
        target=download_nwm_subprocess,
        args=(comid_list, nwm_uri, cache_dir),
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join(5)
        logger.error("NWM retrospective download timed out")
        return -1

    return process.exitcode if process.exitcode is not None else -1


# ---------------------------------------------------------------------------
# Optional side-product writers
# ---------------------------------------------------------------------------
def write_nwm_csv(result_df: pd.DataFrame, output_dir: str, date_obj: datetime) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"NWM_STATS_{date_obj.date()}.csv")
    result_df.to_csv(path, index=False)
    logger.info(f"Wrote NWM CSV: {path}")
    return path


def write_nwm_kml(
    result_df: pd.DataFrame,
    local_comids: pd.DataFrame,
    lat: float,
    lon: float,
    date_obj: datetime,
    output_dir: str,
) -> str:
    """Write a simple KML with one placemark per COMID plus the query point."""
    os.makedirs(output_dir, exist_ok=True)
    kml = Kml()

    for _, row in result_df.iterrows():
        comid = int(row["COMID"])
        match = local_comids[local_comids["COMID"] == comid]
        if match.empty:
            continue

        loc = (float(match["lon"].iloc[0]), float(match["lat"].iloc[0]))
        perc = row["percentile"]
        desc = (
            f"distance = {row['distance_mi']:.2f} mi\n"
            f"flow = {row['flow_cfs']} cfs\n"
            f"percentile = {perc:.1f}%\n"
            f"condition = {row['condition']}"
        )

        p = kml.newpoint(name=str(comid), coords=[loc], description=desc)
        if perc > 90:
            p.style.iconstyle.icon.href = (
                f"{get_base_url('google_maps_kml_paddle')}/blu-blank.png"
            )
        elif perc > 75:
            p.style.iconstyle.icon.href = (
                f"{get_base_url('google_maps_kml_paddle')}/ltblu-blank.png"
            )
        elif perc >= 25:
            p.style.iconstyle.icon.href = (
                f"{get_base_url('google_maps_kml_paddle')}/grn-blank.png"
            )
        elif perc >= 10:
            p.style.iconstyle.icon.href = (
                f"{get_base_url('google_maps_kml_paddle')}/ylw-blank.png"
            )
        else:
            p.style.iconstyle.icon.href = (
                f"{get_base_url('google_maps_kml_paddle')}/red-blank.png"
            )

    qp = kml.newpoint(name="QUERY POINT", coords=[(lon, lat)])
    qp.style.iconstyle.icon.href = (
        f"{get_base_url('google_maps_kml_paddle')}/purple-stars.png"
    )

    path = os.path.join(output_dir, f"NWM_STATS_{date_obj.date()}.kml")
    kml.save(path)
    logger.info(f"Wrote NWM KML: {path}")
    return path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def analyze_nwm(
    lat: float,
    lon: float,
    analysis_date,
    *,
    data_dir: str,
    cache_dir: str,
    output_dir: str | None = None,
    write_kml: bool = False,
    write_csv: bool = False,
    historic_timeout: int = 1800,
) -> pd.DataFrame | None:
    """
    Full NWM streamflow analysis.
    """
    if isinstance(analysis_date, str):
        date_obj = datetime.strptime(analysis_date, "%Y-%m-%d")
    else:
        date_obj = analysis_date

    if date_obj < datetime(2023, 7, 21):
        logger.warning("NWM analysis-assimilation data is sparse before 2023-07-21")
        return None

    if (write_kml or write_csv) and not output_dir:
        raise ValueError("output_dir is required when write_kml or write_csv is True")

    os.makedirs(cache_dir, exist_ok=True)

    # Establish Shapefile Paths / URIs based on data_dir
    # Target the zipped shapefiles using the full zip:// URI scheme
    state_zip_path = Path(data_dir, "usa_shp.zip").as_posix()
    state_shapefile = f"zip://{state_zip_path}!tl_2023_us_state.shp"
    logger.debug(f"usa_shp zip resolved to {state_zip_path}")

    comid_zip_path = Path(data_dir, "COMID_points.zip").as_posix()
    comid_shapefile = f"zip://{comid_zip_path}!nwm_COMID_APTv3.0.shp"

    # --- spatial lookup ---
    try:
        local_comids = find_closest_comids(comid_shapefile, lat, lon)
    except Exception as e:
        logger.error(str(e))
        return None

    if local_comids.empty:
        logger.error("No NWM reaches near the query point")
        return None

    STUSPS = get_special_region(lat, lon, state_shapefile)
    if STUSPS == -1:
        logger.error("Could not determine region")
        return None

    # --- current-day flows ---
    flow_points = get_nwm_flow(
        local_comids, date_obj, STUSPS=STUSPS, cache_dir=cache_dir
    )
    if flow_points == -1 or not flow_points:
        return None

    comid_ids = [int(p["COMID"]) for p in flow_points.values()]

    # --- historic series ---
    hist_csv = os.path.join(cache_dir, f"nwm_historic_{comid_ids[0]}.csv")
    if not os.path.exists(hist_csv):
        ret = download_nwm_historic(
            comid_ids,
            STUSPS=STUSPS,
            cache_dir=cache_dir,
            timeout=historic_timeout,
        )
        if ret != 0:
            logger.error("Historic NWM download failed or timed out")
            return None

    hist = pd.read_csv(hist_csv, parse_dates=[0])
    hist = hist.rename(columns={hist.columns[0]: "time"})

    # --- percentiles ---
    results = []
    for col in hist.columns[1:]:
        comid_str = str(col)
        if comid_str not in flow_points:
            continue

        series = hist[["time", col]].rename(columns={col: "discharge"})
        flow = flow_points[comid_str]["flow"]
        perc = calc_percentile(flow, series, date_obj)
        if perc < 0:
            continue

        dist = local_comids.loc[local_comids["COMID"] == int(col), "dist"]
        dist_val = float(dist.iloc[0]) if not dist.empty else -1.0

        results.append(
            {
                "COMID": comid_str,
                "flow_cfs": flow,
                "percentile": perc,
                "condition": condition_from_percentile(perc),
                "distance_mi": round(dist_val, 2),
            }
        )

    if not results:
        logger.error("No usable NWM reaches after percentile calculation")
        return None

    result_df = pd.DataFrame(results).sort_values("distance_mi").reset_index(drop=True)

    # --- optional side products ---
    coord_str = f"{float(lat):.6f}_{float(lon):.6f}"
    coord_dir = os.path.join(output_dir, coord_str) if output_dir else None

    if write_csv and coord_dir:
        write_nwm_csv(result_df, coord_dir, date_obj)
    if write_kml and coord_dir:
        write_nwm_kml(result_df, local_comids, lat, lon, date_obj, coord_dir)

    logger.info(f"NWM analysis complete – {len(result_df)} reaches")
    return result_df


# Backwards-compatible alias
def local_norm_nwm(lat, lon, date, **kwargs):
    """Deprecated wrapper – prefer analyze_nwm with explicit paths."""
    df = analyze_nwm(lat, lon, date, **kwargs)
    return 0 if df is not None else -1


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="NWM streamflow analysis")
    parser.add_argument("--lat", type=float, default=30.0)
    parser.add_argument("--lon", type=float, default=-90.0)
    parser.add_argument("--date", type=str, default="2023-11-11")
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory for NetCDF / historic CSV caches",
    )
    parser.add_argument(
        "--state-shp",
        type=str,
        required=True,
        help="Path to tl_2023_us_state.shp",
    )
    parser.add_argument(
        "--comid-shp",
        type=str,
        required=True,
        help="Path to nwm_COMID_APTv3.0.shp",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for optional CSV / KML side products",
    )
    parser.add_argument("--write-kml", action="store_true")
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Seconds allowed for retrospective download",
    )
    args = parser.parse_args()

    df = analyze_nwm(
        lat=args.lat,
        lon=args.lon,
        analysis_date=args.date,
        cache_dir=args.cache_dir,
        state_shapefile=args.state_shp,
        comid_shapefile=args.comid_shp,
        output_dir=args.output_dir,
        write_kml=args.write_kml,
        write_csv=args.write_csv,
        historic_timeout=args.timeout,
    )

    if df is not None:
        print(df.round(2).to_string(index=False))
        print(f"\nClosest reach condition: {df.iloc[0]['condition']}")
    else:
        print("Analysis failed")
