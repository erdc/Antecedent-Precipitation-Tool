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
from config import get_base_url
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
    url = get_base_url("usgs_nwis_dv")
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
        flow_df = download_usgs_flow(gage["gage_id"], "1950-01-01", date_str)
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

    # Optional side products
    if output_dir:
        coord_str = f"{lat}_{lon}"
        coord_dir = os.path.join(output_dir, coord_str)
        if write_csv:
            write_usgs_csv(result_df, coord_dir, date_obj)
        if write_kml:
            write_usgs_kml(result_df, gages, lat, lon, date_obj, coord_dir)

    return result_df
