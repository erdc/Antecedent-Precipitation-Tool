# usgs_stream.py
"""
Simple Imperative USGS Streamflow Analysis with Caching
"""

import csv
import io
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from config import get_base_url
from geopy.distance import great_circle
from shapely.geometry import Point

logger = logging.getLogger(__name__)

NWIS_CACHE_DIR = "nwis_cache"


def get_cache_path(gage_id: str, data_dir: str = "data") -> Path:
    """Return path to cached streamflow JSON for a gage."""
    cache_dir = Path(data_dir) / NWIS_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{gage_id}.json"


def load_cache(gage_id: str, data_dir: str = "data") -> dict | None:
    """Load cached data if it exists."""
    cache_path = get_cache_path(gage_id, data_dir)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to read cache for {gage_id}: {e}")
        return None


def save_cache(
    gage_id: str, df: pd.DataFrame, download_date: datetime, data_dir: str = "data"
) -> None:
    """Save streamflow data to cache as JSON."""
    if df.empty:
        return

    cache_path = get_cache_path(gage_id, data_dir)

    cache_data = {
        "gage_id": gage_id,
        "por_start": df["time"].min().strftime("%Y-%m-%d"),
        "por_end": df["time"].max().strftime("%Y-%m-%d"),
        "download_date": download_date.isoformat(),
        "data": df.to_dict(orient="records"),
    }

    try:
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2, default=str)
        logger.debug(f"Cached USGS data for {gage_id} ({len(df)} records)")
    except Exception as e:
        logger.error(f"Failed to write cache for {gage_id}: {e}")


def has_sufficient_por(
    cache: dict, required_start: datetime, required_end: datetime, min_years: int = 5
) -> bool:
    """Check if cache has enough historical data, even with some gaps."""
    if not cache or "data" not in cache:
        return False

    df = pd.DataFrame(cache["data"])
    if df.empty:
        return False

    df["time"] = pd.to_datetime(df["time"])
    mask = (df["time"] >= required_start) & (df["time"] <= required_end)
    available = df[mask]

    if len(available) < 30:  # At minimum need ~1 month
        return False

    years_covered = (available["time"].max() - available["time"].min()).days / 365.25
    return years_covered >= min_years


def download_usgs_flow(
    gage_id: str,
    start_date: str,
    end_date: str,
    data_dir: str = "data",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download daily streamflow with intelligent caching and gap detection."""
    required_start = datetime.strptime(start_date, "%Y-%m-%d")
    required_end = datetime.strptime(end_date, "%Y-%m-%d")

    cache = load_cache(gage_id, data_dir)
    now = datetime.now()

    # Case 1: Good cache exists
    if not force_refresh and cache and is_cache_fresh(cache):
        df = pd.DataFrame(cache.get("data", []))
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"])
            requested = df[
                (df["time"] >= required_start) & (df["time"] <= required_end)
            ]

            # Only use cache if we have decent coverage for the analysis date
            if not requested.empty or has_sufficient_por(
                cache, required_start, required_end
            ):
                logger.debug(
                    f"Using cached data for gage {gage_id} ({len(requested)} records in period)"
                )
                return requested if not requested.empty else df

    # Determine what to download
    download_start = required_start
    if cache and not force_refresh and is_cache_fresh(cache):
        cached_df = pd.DataFrame(cache["data"])
        cached_df["time"] = pd.to_datetime(cached_df["time"])

        # Find gaps in the required period
        all_dates = pd.date_range(required_start, required_end, freq="D")
        cached_dates = cached_df["time"].dt.normalize()
        missing_dates = all_dates[~all_dates.isin(cached_dates)]

        if len(missing_dates) == 0:
            return cached_df[
                (cached_df["time"] >= required_start)
                & (cached_df["time"] <= required_end)
            ]

        # Download only missing data + some buffer
        download_start = missing_dates.min() - timedelta(days=30)  # buffer
    else:
        download_start = datetime(1950, 1, 1)

    # === DOWNLOAD NEW DATA ===
    url = get_base_url("usgs_nwis_dv")
    params = {
        "format": "rdb",
        "sites": gage_id,
        "startDT": download_start.strftime("%Y-%m-%d"),
        "endDT": end_date,
        "parameterCd": "00060",
    }

    try:
        logger.debug(
            f"Downloading USGS flow for {gage_id} from {download_start.date()}"
        )
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()

        reader = csv.reader(io.StringIO(resp.text), delimiter="\t")
        new_data = []
        for row in reader:
            if row and len(row) > 3 and row[0] == "USGS" and not row[0].startswith("#"):
                try:
                    new_data.append(
                        {
                            "time": pd.to_datetime(row[2]).normalize(),
                            "discharge": float(row[3]),
                        }
                    )
                except (ValueError, IndexError):
                    continue

        new_df = pd.DataFrame(new_data)

        if not new_df.empty:
            download_date = now

            # Merge with existing cache
            if cache and cache.get("data"):
                old_df = pd.DataFrame(cache["data"])
                old_df["time"] = pd.to_datetime(old_df["time"]).dt.normalize()
                combined = pd.concat([old_df, new_df], ignore_index=True)
            else:
                combined = new_df

            # Remove duplicates and sort
            combined = combined.drop_duplicates(subset=["time"]).sort_values("time")

            save_cache(gage_id, combined, download_date, data_dir)
            logger.debug(
                f"Updated cache for {gage_id} with {len(combined)} total records"
            )

            return combined[
                (combined["time"] >= required_start)
                & (combined["time"] <= required_end)
            ]

    except Exception as e:
        logger.debug(f"Failed to download USGS data for {gage_id}: {e}")

    # Fallback to cache even if incomplete
    if cache and cache.get("data"):
        df = pd.DataFrame(cache["data"])
        df["time"] = pd.to_datetime(df["time"])
        return df[(df["time"] >= required_start) & (df["time"] <= required_end)]

    return pd.DataFrame()


def is_cache_fresh(cache: dict) -> bool:
    """Check if cache is less than 30 days old."""
    if not cache or "download_date" not in cache:
        return False
    download_date = datetime.fromisoformat(
        cache["download_date"].replace("Z", "+00:00")
    )
    return (datetime.now() - download_date) <= timedelta(days=30)


def get_special_region(lat: float, lon: float, shapefile_path: str):
    """Determine if location is in CONUS or a special region (AK, HI, PR, etc.)."""
    try:
        usa_gdf = gpd.read_file(shapefile_path)
    except Exception as e:
        logger.error(f"Shapefile not found or failed to read: {shapefile_path} - {e}")
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


def get_local_gages(
    lat: float, lon: float, date_str: str, data_dir: str, max_dist: int = 100
):
    """Find nearby USGS gages."""
    # Use pathlib to join and standardize to forward slashes (.as_posix())
    state_zip_path = Path(data_dir, "usa_shp.zip").as_posix()
    shapefile_path = f"zip://{state_zip_path}!tl_2023_us_state.shp"
    logger.debug(f"usa_shp zip resolved to {state_zip_path}")

    gage_data_path = Path(data_dir, "gage_data.csv")
    region = get_special_region(lat, lon, shapefile_path)

    if region == "CONUS":
        if not gage_data_path.exists():
            logger.error(f"gage_data.csv not found in {data_dir}")
            return pd.DataFrame()
        df = pd.read_csv(gage_data_path, dtype={"GAGEID": str})
    elif region != -1:
        # For AK/HI/PR - fetch on the fly
        df = _get_region_gages(region, date_str)
    else:
        return pd.DataFrame()

    if df.empty:
        return df

    if "GagesII" not in df.columns:
        df["GagesII"] = None

    # Calculate distance
    df["dist"] = df.apply(
        lambda row: great_circle((row["LatSite"], row["LonSite"]), (lat, lon)).miles,
        axis=1,
    )

    df = df[df["dist"] <= max_dist].copy()
    df = df.sort_values("dist").head(30)  # Top n closest

    return df.rename(
        columns={
            "STATION_NM": "name",
            "GAGEID": "gage_id",
            "LatSite": "lat",
            "LonSite": "lon",
            "GagesII": "gagesii",
        }
    )[["name", "gage_id", "lat", "lon", "dist", "gagesii"]]


def _get_region_gages(stusps: str, date_str: str) -> pd.DataFrame:
    """Helper to get gages for non-CONUS regions."""
    url = get_base_url("usgs_nwis_site")
    start_date = (
        datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=30 * 365)
    ).strftime("%Y-%m-%d")

    params = {
        "stateCd": stusps,
        "siteStatus": "active",
        "startDT": start_date,
        "endDT": date_str,
        "format": "rdb",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        reader = csv.reader(io.StringIO(resp.text), delimiter="\t")
        data = [
            {
                "name": row[2],
                "gage_id": row[1],
                "LatSite": float(row[4]),
                "LonSite": float(row[5]),
                "GagesII": None,
            }
            for row in reader
            if row and len(row) > 5 and row[0] == "USGS" and not row[0].startswith("#")
        ]
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Optional side-product writers (mirrors NWM)
# ---------------------------------------------------------------------------
def write_usgs_csv(result_df: pd.DataFrame, output_dir: str, date_obj: datetime) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"USGS_STATS_{date_obj.date()}.csv")
    result_df.to_csv(path, index=False)
    logger.info(f"Wrote USGS CSV: {path}")
    return path


def write_usgs_kml(
    result_df: pd.DataFrame,
    local_gages: pd.DataFrame,
    lat: float,
    lon: float,
    date_obj: datetime,
    output_dir: str,
) -> str:
    """Write a simple KML with one placemark per gage plus the query point."""
    from simplekml import Kml

    os.makedirs(output_dir, exist_ok=True)
    kml = Kml()

    for _, row in result_df.iterrows():
        gage_id = str(row["gage_id"])
        match = local_gages[local_gages["gage_id"] == gage_id]
        if match.empty:
            continue

        loc = (float(match["lon"].iloc[0]), float(match["lat"].iloc[0]))
        perc = row["percentile"]
        desc = (
            f"name = {row['name']}\n"
            f"distance = {row['distance_mi']:.2f} mi\n"
            f"flow = {row['flow_cfs']} cfs\n"
            f"percentile = {perc:.1f}%\n"
            f"condition = {row['condition']}"
        )

        p = kml.newpoint(name=gage_id, coords=[loc], description=desc)
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

    path = os.path.join(output_dir, f"USGS_STATS_{date_obj.date()}.kml")
    kml.save(path)
    logger.info(f"Wrote USGS KML: {path}")
    return path


def analyze_usgs(
    lat: float,
    lon: float,
    analysis_date: str,
    data_dir: str = "data",
    *,
    output_dir: str | None = None,
    write_kml: bool = False,
    write_csv: bool = False,
):
    """Main function: Run full USGS analysis for a location and date."""
    logger.info(f"Analyzing USGS streamflow for ({lat}, {lon}) on {analysis_date}")

    if isinstance(analysis_date, datetime):
        date_obj = analysis_date
        date_str = analysis_date.strftime("%Y-%m-%d")
    else:
        date_str = str(analysis_date)
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    if (write_kml or write_csv) and not output_dir:
        raise ValueError("output_dir is required when write_kml or write_csv is True")

    gages = get_local_gages(lat, lon, date_str, data_dir, max_dist=100)
    if gages.empty:
        logger.warning("No nearby gages found.")
        return None

    results = []
    end_dt = pd.to_datetime(date_str)

    for _, gage in gages.iterrows():
        flow_df = download_usgs_flow(
            gage["gage_id"], "1950-01-01", date_str, data_dir=data_dir
        )
        if flow_df.empty:
            continue

        current_flow_df = flow_df[flow_df["time"] == end_dt]
        if current_flow_df.empty:
            continue

        current_flow = current_flow_df["discharge"].iloc[0]

        # Calculate percentile
        day_of_year = end_dt.strftime("%m-%d")
        same_day = flow_df[flow_df["time"].dt.strftime("%m-%d") == day_of_year]
        if len(same_day) < 5:  # Not enough historical data
            continue

        percentile = round((same_day["discharge"] <= current_flow).mean() * 100, 1)
        condition = (
            "Wet/Above Normal"
            if percentile > 75
            else "Normal" if percentile >= 25 else "Dry/Below Normal"
        )

        results.append(
            {
                "gage_id": gage["gage_id"],
                "name": gage["name"][:40],
                "gagesii": gage.get("gagesii", None),
                "distance_mi": round(gage["dist"], 2),
                "flow_cfs": round(current_flow, 1),
                "percentile": percentile,
                "condition": condition,
                "lat": float(gage["lat"]) if pd.notna(gage["lat"]) else None,
                "lon": float(gage["lon"]) if pd.notna(gage["lon"]) else None,
            }
        )

        # Stop iterating once we have found 10 gages with valid data
        if len(results) >= 10:
            break

    if not results:
        logger.warning("Could not compute valid flow percentiles.")
        return None

    result_df = pd.DataFrame(results)
    logger.info(f"USGS analysis complete. Found {len(result_df)} valid gages.")

    # Optional side products
    if output_dir:
        coord_str = f"{float(lat):.6f}_{float(lon):.6f}"
        coord_dir = os.path.join(output_dir, coord_str)
        if write_csv:
            write_usgs_csv(result_df, coord_dir, date_obj)
        if write_kml:
            write_usgs_kml(result_df, gages, lat, lon, date_obj, coord_dir)

    return result_df
