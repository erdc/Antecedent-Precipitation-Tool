# usgs_simple.py
"""
Simple Imperative USGS Streamflow Analysis
No classes, no OOP - just straightforward functions.
"""

import csv
import io
import logging
import os
from datetime import datetime, timedelta

import geopandas as gpd
import pandas as pd
import requests
from geopy.distance import great_circle
from shapely.geometry import Point

logger = logging.getLogger(__name__)

USGS_DATA_SUBDIR = "usgs_flow"


def get_special_region(lat: float, lon: float, shapefile_path: str):
    """Determine if location is in CONUS or a special region (AK, HI, PR, etc.)."""
    if not os.path.exists(shapefile_path):
        logger.error(f"Shapefile not found: {shapefile_path}")
        return -1

    usa_gdf = gpd.read_file(shapefile_path)
    point = Point(lon, lat)

    joined = gpd.sjoin(
        gpd.GeoDataFrame(geometry=[point], crs=usa_gdf.crs),
        usa_gdf,
        how="left",
        predicate="intersects",
    )

    if joined.empty:
        return -1

    stusps = joined["STUSPS"].iloc[0]
    if pd.isna(stusps):
        return -1

    special = {"AK", "HI", "PR", "VI", "GU", "MP", "AS"}
    return stusps if stusps in special else "CONUS"


def download_usgs_flow(gage_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily streamflow (discharge) from USGS."""
    url = "https://waterservices.usgs.gov/nwis/dv"
    params = {
        "format": "rdb",
        "sites": gage_id,
        "startDT": start_date,
        "endDT": end_date,
        "parameterCd": "00060",
    }

    # TODO: This try needs a simple wait and retry loop
    #       currently it pings the servers too fast and
    #       the usgs drop stuff
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()

        reader = csv.reader(io.StringIO(resp.text), delimiter="\t")
        data = []
        for row in reader:
            if row and len(row) > 3 and row[0] == "USGS" and not row[0].startswith("#"):
                try:
                    data.append(
                        {"time": pd.to_datetime(row[2]), "discharge": float(row[3])}
                    )
                except (ValueError, IndexError):
                    continue

        return pd.DataFrame(data)

    except Exception as e:
        # This message is common and a non issue so hiding it at debug level
        logger.debug(f"Failed to download USGS data for {gage_id}: {e}")
        return pd.DataFrame()


def get_local_gages(
    lat: float, lon: float, date_str: str, data_dir: str, max_dist: int = 100
):
    """Find nearby USGS gages."""
    shapefile_path = os.path.join(data_dir, "usa_shp", "tl_2023_us_state.shp")
    gage_data_path = os.path.join(data_dir, "gage_data.csv")

    region = get_special_region(lat, lon, shapefile_path)

    if region == "CONUS":
        if not os.path.exists(gage_data_path):
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
        }
    )[["name", "gage_id", "lat", "lon", "dist"]]


def _get_region_gages(stusps: str, date_str: str) -> pd.DataFrame:
    """Helper to get gages for non-CONUS regions."""
    url = "https://waterservices.usgs.gov/nwis/site"
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
            }
            for row in reader
            if row and len(row) > 5 and row[0] == "USGS" and not row[0].startswith("#")
        ]
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def analyze_usgs(lat: float, lon: float, analysis_date: str, data_dir: str = "data"):
    """Main function: Run full USGS analysis for a location and date."""
    logger.info(f"Analyzing USGS streamflow for ({lat}, {lon}) on {analysis_date}")

    gages = get_local_gages(lat, lon, analysis_date, data_dir, max_dist=100)
    if gages.empty:
        logger.warning("No nearby gages found.")
        return None

    results = []
    end_dt = pd.to_datetime(analysis_date)

    for _, gage in gages.iterrows():
        flow_df = download_usgs_flow(gage["gage_id"], "1950-01-01", analysis_date)
        if flow_df.empty:
            continue

        # Get flow on analysis date
        current_flow = flow_df[flow_df["time"] == end_dt]
        if current_flow.empty:
            continue

        current_flow = current_flow["discharge"].iloc[0]

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
                "distance_mi": round(gage["dist"], 2),
                "flow_cfs": round(current_flow, 1),
                "percentile": percentile,
                "condition": condition,
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
    return result_df


# ====================== Quick Test ======================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import glob

    search_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
    result = next(glob.iglob(f"{search_dir}/**/usa_shp/", recursive=True), None)
    if result is None:
        raise FileNotFoundError(
            f"Could not find 'usa_shp' directory under {search_dir}"
        )
    result = os.path.abspath(os.path.join(result, ".."))
    print(result)

    df = analyze_usgs(
        lat=30.0,
        lon=-90.0,
        analysis_date="2023-11-11",
        data_dir=result,
    )

    if df is not None:
        print(df.round(2))
        print(f"\nClosest gage condition: {df.iloc[0]['condition']}")
