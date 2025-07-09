#  This software was developed by United States Army Corps of Engineers (USACE)
#  employees in the course of their official duties.  USACE used copyrighted,
#  open source code to develop this software, as such this software
#  (per 17 USC § 101) is considered "joint work."  Pursuant to 17 USC § 105,
#  portions of the software developed by USACE employees in the course of their
#  official duties are not subject to copyright protection and are in the public
#  domain.
#
#  USACE assumes no responsibility whatsoever for the use of this software by
#  other parties, and makes no guarantees, expressed or implied, about its
#  quality, reliability, or any other characteristic.
#
#  The software is provided "as is," without warranty of any kind, express or
#  implied, including but not limited to the warranties of merchantability,
#  fitness for a particular purpose, and noninfringement.  In no event shall the
#  authors or U.S. Government be liable for any claim, damages or other
#  liability, whether in an action of contract, tort or otherwise, arising from,
#  out of or in connection with the software or the use or other dealings in the
#  software.
#
#  Public domain portions of this software can be redistributed and/or modified
#  freely, provided that any derivative works bear some notice that they are
#  derived from it, and any modified versions bear some notice that they have
#  been modified.
#
#  Copyrighted portions of the software are annotated within the source code.
#  Open Source Licenses, included in the source code, apply to the applicable
#  copyrighted portions.  Copyrighted portions of the software are not in the
#  public domain.

######################################
##  ------------------------------- ##
##           local_norm.py          ##
##  ------------------------------- ##
##     Written by: Chris French     ##
##  ------------------------------- ##
##    Last Edited on:  2025-07-09   ##
##  ------------------------------- ##
######################################

# Standard Libraries
import csv
import glob
import io
import logging
import multiprocessing
import os
import time
import traceback
from datetime import datetime, timedelta
from functools import lru_cache

import dask
import fsspec

# External Libraries
import geopandas as gpd

# from shapely.geometry import Point
import netCDF4

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from dask.diagnostics import ProgressBar
from geopy.distance import great_circle
from shapely.geometry import Point
from simplekml import Kml
from tqdm import tqdm

# Internal Imports
try:
    from arc.manifest import Manifest
    from arc.utils import find_file_or_dir, ini_config, setup_logger
except:
    from manifest import Manifest
    from utils import find_file_or_dir, ini_config, setup_logger

if __name__ == "__main__":
    setup_logger()

logger = logging.getLogger(__name__)

default = {
    "local_norm": {
        "CACHE_SIZE": 32,
        "NUM_LOCAL_GAGES": 10,
        "MAX_SEARCH_RANGE": 100,
        "REF_GAGES_ONLY": 0,
        "ENABLE_NWM_CALC": 1,
        "ENABLE_USGS_CALC": 1,
        "NWM_TIMEOUT_MINS": 60,
        "MTF_CONVERSION_FAC": 35.3147,
    },
    "version": {"VERSION": "v3"},
}

config = ini_config(default)
CACHE_SIZE = config.getint("local_norm", "CACHE_SIZE")
NUM_LOCAL_GAGES = config.getint("local_norm", "NUM_LOCAL_GAGES")
MAX_SEARCH_RANGE = config.getint("local_norm", "MAX_SEARCH_RANGE")
REF_GAGES_ONLY = config.getboolean("local_norm", "REF_GAGES_ONLY")
ENABLE_NWM_CALC = config.getint("local_norm", "ENABLE_NWM_CALC")
ENABLE_USGS_CALC = config.getint("local_norm", "ENABLE_USGS_CALC")
MTF_CONVERSION_FAC = config.getfloat("local_norm", "MTF_CONVERSION_FAC")
NWM_TIMEOUT_MINS = config.getint("local_norm", "NWM_TIMEOUT_MINS")
VERSION = config.get("version", "VERSION")

from osgeo import gdal

logger.debug(f"gdal install is: {gdal.VersionInfo()}")
run_manifest = Manifest()

# ### GENERAL FUNCTIONS ###


def linear_interpolation(x, x0, y0, x1, y1):
    """
    Perform linear interpolation between two points.

    Parameters:
    - x (float): The x-coordinate for which to find the corresponding y-coordinate.
    - x0, y0 (float): The coordinates of the first point.
    - x1, y1 (float): The coordinates of the second point.

    Returns:
    - float: The interpolated y-coordinate for the given x-coordinate.
    """
    y = float(y0 + (x - x0) * ((y1 - y0) / (x1 - x0)))
    return y


def cleanup_stats_df(stats_df):
    clean_df = pd.DataFrame(
        columns=[
            "min",
            "p05",
            "p10",
            "p20",
            "p25",
            "p50",
            "p75",
            "p80",
            "p90",
            "p95",
            "max",
        ]
    )

    clean_df["min"] = stats_df[("discharge", "min")]
    clean_df["p10"] = stats_df[("discharge", "p10")]
    clean_df["p05"] = stats_df[("discharge", "p05")]
    clean_df["p20"] = stats_df[("discharge", "p20")]
    clean_df["p25"] = stats_df[("discharge", "p25")]
    clean_df["p50"] = stats_df[("discharge", "p50")]
    clean_df["p75"] = stats_df[("discharge", "p75")]
    clean_df["p80"] = stats_df[("discharge", "p80")]
    clean_df["p90"] = stats_df[("discharge", "p90")]
    clean_df["p95"] = stats_df[("discharge", "p95")]
    clean_df["max"] = stats_df[("discharge", "max")]
    clean_df["day"] = stats_df[("day", "")]

    clean_df = clean_df.round(3)

    return clean_df


def calc_stats(flow_df, period_end="2022-12-31"):
    """
    Calculate daily discharge statistics for a given period.

    Parameters:
    - flow_df (DataFrame): A pandas DataFrame containing the flow data.
    - period_end (str): The end date of the period for which to calculate statistics, default is "2010-12-31".

    Returns:
    - DataFrame: A pandas DataFrame containing the daily discharge statistics.
    """
    flow_df["time"] = pd.to_datetime(flow_df["time"])
    filtered_df = flow_df[flow_df["time"] <= pd.to_datetime(period_end)]

    earliest_date = filtered_df["time"].min().strftime("%Y-%m-%d")
    latest_date = filtered_df["time"].max().strftime("%Y-%m-%d")
    hist_por = f"{earliest_date}->{latest_date}"

    # group by day and calculate the percentile values
    stats_df = (
        filtered_df.groupby(filtered_df["time"].dt.strftime("%m-%d"))
        .agg(
            {
                "discharge": [
                    "min",
                    lambda x: np.percentile(x, 5),  # p05
                    lambda x: np.percentile(x, 10),  # p10
                    lambda x: np.percentile(x, 20),  # p20
                    lambda x: np.percentile(x, 25),  # p25
                    "median",
                    lambda x: np.percentile(x, 75),  # p75
                    lambda x: np.percentile(x, 80),  # p80
                    lambda x: np.percentile(x, 90),  # p90
                    lambda x: np.percentile(x, 95),  # p95
                    "max",
                ]
            }
        )
        .rename(
            columns={
                "min": "min",
                "<lambda_0>": "p05",
                "<lambda_1>": "p10",
                "<lambda_2>": "p20",
                "<lambda_3>": "p25",
                "median": "p50",
                "<lambda_4>": "p75",
                "<lambda_5>": "p80",
                "<lambda_6>": "p90",
                "<lambda_7>": "p95",
                "max": "max",
            }
        )
    )

    # rename the 'time' column and return
    stats_df.reset_index(inplace=True)
    stats_df.rename(columns={"time": "day"}, inplace=True)
    clean_df = cleanup_stats_df(stats_df)
    return clean_df, hist_por


def calc_percentile(flow, df, date):
    # Ensure the DataFrame is sorted by time
    df = df.sort_values(by="time")
    df["time"] = pd.to_datetime(df["time"])

    # Filter df
    date = pd.to_datetime(date)
    day_of_year = date.strftime("%m-%d")
    filtered_df = df[df["time"].dt.strftime("%m-%d") == day_of_year]

    # Skip bad data
    if filtered_df["discharge"].nunique() <= 1:
        return -1

    # Calculate the percentile
    percentile_rank = (
        filtered_df[filtered_df["discharge"] <= flow].shape[0] / filtered_df.shape[0]
    ) * 100

    return round(percentile_rank, 3)


def get_special_region(lat, lon):
    # Find and read the shapefile
    usa_path = find_file_or_dir(os.getcwd(), "tl_2023_us_state.shp")
    usa_gdf = gpd.read_file(usa_path)

    # Perform spatial join
    point = Point(lon, lat)
    joined = gpd.sjoin(
        gpd.GeoDataFrame(geometry=[point], crs=usa_gdf.crs),
        usa_gdf,
        how="left",
        predicate="intersects",
    )

    # Check if a state was found
    if not joined.empty:
        value = joined["STUSPS"].iloc[0]
        special_regions = ["AK", "PR", "VI", "GU", "MP", "AS", "HI"]

        if pd.isna(value):
            return -1
        elif value in special_regions:
            return value
        return "CONUS"
    else:
        return -1  # No state found


# ### USGS FUNCTIONS ###


@lru_cache(maxsize=CACHE_SIZE)
def get_usgs_flow(gage_id, date):
    base_url = "http://waterservices.usgs.gov/nwis/dv"
    try:
        date = datetime.strptime(date, "%Y-%m-%d")
        params = {
            "format": "rdb",
            "sites": gage_id,
            # "startDT": start_date.strftime('%Y-%m-%d'),
            "startDT": "1800-01-01",
            "endDT": date.strftime("%Y-%m-%d"),
            "parameterCd": "00060",
        }

        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()

        reader = csv.reader(io.StringIO(response.text), delimiter="\t")

        data = []
        for row in reader:
            if (
                row[0].startswith("#")
                or row[0].startswith("5")
                or row[0].startswith("a")
            ):
                continue
            if len(row) < 4:
                continue

            time_str = row[2]
            try:
                discharge = float(row[3])
            except ValueError:
                discharge = np.nan
            data.append({"time": time_str, "discharge": discharge})

        df = pd.DataFrame(data)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"])  # Convert 'time' to datetime
        return df

    except Exception as e:
        logger.error(f"Error retrieving flow for gage ID {gage_id}: {str(e)}")
        return pd.DataFrame()


@lru_cache(maxsize=CACHE_SIZE)
def get_region_gages(STUSPS, date):
    base_url = "http://waterservices.usgs.gov/nwis/site"
    date = datetime.strptime(date, "%Y-%m-%d")
    start_date = date - timedelta(days=30 * 365 + 1)
    try:
        params = {
            "stateCd": f"{STUSPS}",
            "siteStatus": "active",
            "startDT": start_date.strftime("%Y-%m-%d"),
            "endDT": date.strftime("%Y-%m-%d"),
        }

        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()

        reader = csv.reader(io.StringIO(response.text), delimiter="\t")

        data = []
        error_count = 0
        for row in reader:
            if (
                row[0].startswith("#")
                or row[0].startswith("5")
                or row[0].startswith("a")
            ):
                continue
            if len(row) < 4:
                continue
            gage_id = row[1]
            station_name = row[2]
            try:
                lat = round(float(row[4]), 3)
                lon = round(float(row[5]), 3)
                data.append(
                    {
                        "STATION_NM": station_name,
                        "GAGEID": gage_id,
                        "LatSite": lat,
                        "LonSite": lon,
                    }
                )
            except:
                if error_count <= 10:
                    error_count += 1
                    logger.debug(f"gage id {gage_id} in {STUSPS} is missing data.")
                continue

        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logger.error(f"Error retrieving gage list for {STUSPS}: {str(e)}")
        return pd.DataFrame()


def get_local_gages(lat, lon, date, max_dist=0):
    # Open gage_dat.csv
    data_path = find_file_or_dir(os.getcwd(), "gage_data.csv")
    usps_code = get_special_region(lat=lat, lon=lon)

    if usps_code == "CONUS":
        df = pd.read_csv(data_path, dtype={"GAGEID": str})
        if REF_GAGES_ONLY:
            filtered_df = df[df["GagesII"] == "Ref"].copy()
        else:
            filtered_df = df
    elif usps_code != -1:
        filtered_df = get_region_gages(usps_code, date)
    else:
        return pd.DataFrame()

    # Calculate distances for each gage_id
    filtered_df["dist"] = filtered_df.apply(
        lambda row: great_circle((row["LatSite"], row["LonSite"]), (lat, lon)).miles,
        axis=1,
    )
    if max_dist > 0:
        filtered_df = filtered_df[filtered_df["dist"] <= max_dist].copy()

    # Sort by distance
    sorted_df = filtered_df.sort_values("dist")
    sorted_df = sorted_df[["STATION_NM", "GAGEID", "LatSite", "LonSite", "dist"]]
    sorted_df.columns = ["STATION_NM", "GAGEID", "lat", "lon", "dist"]
    sorted_df = sorted_df.head(500)

    # Return the sorted DataFrame
    return sorted_df


def local_norm_usgs(lat, lon, date, save_path=None):
    try:
        logger.info("Initializing USGS norm script...")
        run_manifest.reset()
        run_manifest.update(key="analysis_type", value="USGS_norm")
        run_manifest.update(key="analysis_date", value=date)
        run_manifest.update(key="analysis_pos", value=f"{lat},{lon}")
        run_manifest.update(key="apt_version", value=VERSION)
        run_manifest.update(key="data_source", value="http://waterservices.usgs.gov")
        local_gage_df = get_local_gages(
            lat=lat, lon=lon, date=date, max_dist=MAX_SEARCH_RANGE
        )
        gage_id_list = local_gage_df["GAGEID"].tolist()
        logger.debug(f"gage ids local to point: {gage_id_list[:NUM_LOCAL_GAGES]}")

        kml = Kml()
        num_found = 0
        summary_df = pd.DataFrame()
        # Assign tqdm to a variable
        progress_bar = tqdm(
            total=NUM_LOCAL_GAGES,
            unit="gage",
            ascii=r" \|/-~",
            desc="Processing",
            dynamic_ncols=True,
        )
        for gage_id in gage_id_list:
            loc_tuple = (
                local_gage_df.loc[local_gage_df["GAGEID"] == gage_id, "lon"].values[0],
                local_gage_df.loc[local_gage_df["GAGEID"] == gage_id, "lat"].values[0],
            )

            flow_df = get_usgs_flow(gage_id, date)
            if flow_df.empty:
                logger.debug(f"skipping gage {gage_id}, server returned no data")
                continue
            flow = flow_df.loc[
                flow_df["time"] == pd.to_datetime(date), "discharge"
            ].values
            if len(flow) != 1:
                logger.debug(
                    f"skipping gage {gage_id}, data on {date} is incorrect or missing"
                )
                continue
            flow = flow[0]

            previous_day = (
                datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)
            ).strftime("%Y-%m-%d")

            stat_df, hist_por = calc_stats(flow_df, previous_day)

            day_of_year = pd.to_datetime(date).strftime("%m-%d")
            percentile_stats = stat_df.loc[stat_df["day"] == day_of_year].copy()
            percentile_stats["gageid"] = gage_id
            percentile_stats["flow (cfs)"] = flow
            percentile_stats["distance (mi)"] = (
                f"{local_gage_df.loc[local_gage_df['GAGEID'] == gage_id, 'dist'].values[0]:.2f}"
            )
            perc = calc_percentile(flow, flow_df, date)

            # skip bad data
            if perc == -1:
                continue

            if perc > 75:
                normalacy = "Wet/Above Normal"
            elif perc >= 25:
                normalacy = "Normal"
            else:
                normalacy = "Dry/Below Normal"

            percentile_stats["flow percentile"] = perc
            percentile_stats["condition"] = normalacy
            percentile_stats["hist period of record"] = hist_por
            summary_df = pd.concat([summary_df, percentile_stats], ignore_index=True)
            if perc is not None:
                progress_bar.update()
                description = f"{local_gage_df.loc[local_gage_df['GAGEID'] == gage_id, 'STATION_NM'].values[0]}\n"
                description += f"distance from point = {local_gage_df.loc[local_gage_df['GAGEID'] == gage_id, 'dist'].values[0]:.2f} mi\n"
                description += f"\nCONDITION ON {date}:\n"
                description += f"flow = {flow} cfs\n"
                description += f"flow percentile= {perc:.1f}%\n"
                description += f"flow is {normalacy} \n"
                description += f"Historic period of record {hist_por} \n"

                point = kml.newpoint(
                    name=gage_id, coords=[loc_tuple], description=description
                )
                if perc > 80.0:
                    point.style.iconstyle.icon.href = (
                        "http://maps.google.com/mapfiles/kml/paddle/blu-blank.png"
                    )
                elif perc > 60.0:
                    point.style.iconstyle.icon.href = (
                        "http://maps.google.com/mapfiles/kml/paddle/ltblu-blank.png"
                    )
                elif perc > 40.0:
                    point.style.iconstyle.icon.href = (
                        "http://maps.google.com/mapfiles/kml/paddle/grn-blank.png"
                    )
                elif perc > 20.0:
                    point.style.iconstyle.icon.href = (
                        "http://maps.google.com/mapfiles/kml/paddle/ylw-blank.png"
                    )
                else:
                    point.style.iconstyle.icon.href = (
                        "http://maps.google.com/mapfiles/kml/paddle/red-blank.png"
                    )
                num_found += 1
                if num_found >= NUM_LOCAL_GAGES:
                    # Update the progress bar to its maximum value
                    break

        # Ensure the progress bar is closed properly
        remaining = NUM_LOCAL_GAGES - num_found
        if remaining > 0:
            progress_bar.update(remaining)
        progress_bar.close()

        point = kml.newpoint(name="QUERY POINT", coords=[(lon, lat)])
        point.style.iconstyle.icon.href = (
            "http://maps.google.com/mapfiles/kml/paddle/purple-stars.png"
        )
        if summary_df.empty:
            logger.warning(
                f"{lat}, {lon} on {date} could not find enough data to complete a USGS analysis."
            )
            return -1

        if save_path is not None:
            logger.info("Outputing processed data...")
            kml.save(os.path.join(save_path, f"USGS_STATS_{date}.kml"))

            summary_df = summary_df[
                ["gageid", "day", "flow (cfs)", "flow percentile", "condition"]
                + [
                    col
                    for col in summary_df.columns
                    if col
                    not in [
                        "day",
                        "gageid",
                        "flow (cfs)",
                        "flow percentile",
                        "condition",
                    ]
                ]
            ]
            summary_df.to_csv(
                os.path.join(save_path, f"USGS_STATS_{date}.csv"), index=False
            )
            data_dir = find_file_or_dir(os.getcwd(), "data")
            run_manifest.write_to_file(os.path.join(data_dir, "manifest.json"))
        logger.info("USGS norm script complete")
        return 1
    except Exception as e:
        logger.error(f"Error at {lat}, {lon} on {date}: {str(e)}.")
        tb = traceback.format_exc()
        logger.error(f"traceback:\n{tb}")
        return None


# ### NWM FUNCTIONS ###


def find_closest_comids(shapefile_path, lat, lon, max_dist=0):
    # Define the bounding box around the passed point
    bbox = (lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5)

    # Load the shapefile and filter by the bounding box
    gdf = gpd.read_file(shapefile_path, bbox=bbox)

    # Calculate the distance using great_circle directly within apply
    filtered_df = gdf.assign(
        dist=gdf.geometry.apply(
            lambda point: great_circle((point.y, point.x), (lat, lon)).miles
        )
    ).reset_index(drop=True)

    # Filter by maximum distance if specified
    if max_dist > 0:
        filtered_df = filtered_df[filtered_df["dist"] <= max_dist].copy()

    # Sort by distance
    sorted_df = filtered_df.sort_values("dist").reset_index(drop=True)

    # Add lat and lon columns to the output DataFrame
    sorted_df["lat"] = sorted_df.geometry.y
    sorted_df["lon"] = sorted_df.geometry.x

    return sorted_df[["COMID", "dist", "lat", "lon"]]


@lru_cache(maxsize=CACHE_SIZE)
def download_nwm_flow(date, STUSPS, data_dir="data"):
    date_str = date.strftime("%Y%m%d")

    if not os.path.exists(data_dir):
        print(
            "This should not happen, but the data directory was not found by download process."
        )
        return -1

    nc_filepath = os.path.join(data_dir, f"nwm_data_{date_str}_{STUSPS}.nc")
    if os.path.exists(nc_filepath):
        return 0

    # Construct the URL for the Analysis and Assimilation data
    # https://storage.googleapis.com/national-water-model/nwm.20230920/analysis_assim
    base_url = f"https://storage.googleapis.com/national-water-model/nwm.{date_str}/analysis_assim"
    if STUSPS == "CONUS":
        file_name = f"/nwm.t00z.analysis_assim.channel_rt.tm02.conus.nc"
    elif STUSPS == "HI":
        file_name = f"_hawaii/nwm.t00z.analysis_assim.channel_rt.tm0245.hawaii.nc"
    elif STUSPS == "PR":
        file_name = f"_puertorico/nwm.t00z.analysis_assim.channel_rt.tm02.puertorico.nc"
    elif STUSPS == "AK":
        file_name = f"_alaska/nwm.t00z.analysis_assim.channel_rt.tm02.alaska.nc"
    url = f"{base_url}{file_name}"
    logger.debug(url)

    try:
        # Download the NetCDF file
        response = requests.get(url)
        if response.status_code != 200:
            return response.status_code

        # Save the NetCDF file locally
        with open(nc_filepath, "wb") as f:
            f.write(response.content)

        return 0

    except Exception as e:
        logger.error(f"Error retrieving nwm flow on {date}: {str(e)}")
        return -1


def download_nwm_subprocess(comid_list, nwm_uri, data_dir="data"):
    # file named by closest point, so if already exist no need to download
    if os.path.exists(os.path.join(data_dir, f"nwm_historic_{comid_list[0]}.csv")):
        return 0

    # open the dataset
    ds = xr.open_zarr(fsspec.get_mapper(nwm_uri, anon=True))

    # filter down to POR and local comids streamflow
    ds_filtered = ds.sel(feature_id=comid_list, time=slice("1990-01-01", "2020-12-31"))
    streamflow_data = ds_filtered["streamflow"]

    logger.info("Downloading NWM data, this may take a while...")

    with ProgressBar():
        streamflow_data = streamflow_data.compute()

    df = streamflow_data.to_dataframe()
    df = df.reset_index()

    logger.info("NWM data loaded into local memory.")

    # pivot the DataFrame to have each feature_id as a column
    pivoted_df = df.pivot(index="time", columns="feature_id", values="streamflow")
    pivoted_df = pivoted_df.resample("D").mean()

    # convert streamflow from m3/s to ft3/s
    conversion_factor = MTF_CONVERSION_FAC
    pivoted_df = pivoted_df * conversion_factor

    # save the data
    csv_path = os.path.join(data_dir, f"nwm_historic_{comid_list[0]}.csv")
    logger.debug(f"Attempting to save nwm historical data to {csv_path}")
    pivoted_df.to_csv(csv_path)

    logger.info("NWM data cached to disk.")

    return 0


def download_nwm_historic(comid_list, STUSPS="CONUS", data_dir="data", timeout=6000):
    logger.info(f"Downloading NWM data will begin shortly.")
    logger.info(f"Timeout is currently set to {int(timeout/60)} minutes.")

    # setup data source URL
    uri_dict = {
        "CONUS": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
        "AK": "s3://noaa-nwm-retrospective-3-0-pds/Alaska/zarr/chrtout.zarr",
        "HI": "s3://noaa-nwm-retrospective-3-0-pds/Hawaii/zarr/chrtout.zarr",
        "PR": "s3://noaa-nwm-retrospective-3-0-pds/PR/zarr/chrtout.zarr",
    }
    nwm_uri = uri_dict.get(STUSPS, None)

    if nwm_uri == None:
        logger.warning(
            f"Invalid STUSPS code {STUSPS} in retrospective download process"
        )
        return -1

    # Create a Process object
    process = multiprocessing.Process(
        target=download_nwm_subprocess, args=(comid_list, nwm_uri, data_dir)
    )

    # Start the process
    process.start()

    # Wait for the process to finish or for the timeout to expire
    process.join(timeout)

    # Check if the process is still alive
    if process.is_alive():
        # If the process is still running, terminate it and return -1
        process.terminate()
        return -1
    else:
        # If the process has finished, return its exit code
        return process.exitcode


def get_nwm_flow(comid_list, date, STUSPS, data_dir="data"):
    if isinstance(date, str):
        try:
            # Assuming the date string is in the format 'YYYY-MM-DD'
            date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Incorrect date format, should be YYYY-MM-DD")

    download_attempt = 0
    while download_attempt < 3:
        ret = download_nwm_flow(date, STUSPS=STUSPS, data_dir=data_dir)
        if ret == 0:
            break
        elif ret != -1:
            logger.warning(f"download attempt returned code: {ret}")
        download_attempt += 1

    if download_attempt >= 3:
        logger.warning("nwm flow data download attempts exceeded 3")

    date_str = date.strftime("%Y%m%d")
    try:
        nc_filepath = find_file_or_dir(os.getcwd(), f"nwm_data_{date_str}_{STUSPS}.nc")
    except FileNotFoundError:
        print("NWM analysis does not have enough data to run")
        return -1

    ds = netCDF4.Dataset(nc_filepath)
    streamflow = ds.variables["streamflow"]
    feature_id = ds.variables["feature_id"]

    # source_unit = str(streamflow.units)
    # print(source_unit)

    points = {}
    num_found = 0
    for _, row in comid_list.iterrows():
        comid = int(row["COMID"])
        dist = row["dist"]

        # Get the flow by feature id
        flow = streamflow[np.where(feature_id[:] == comid)[0]]

        # Check if flow data is available and not missing
        if not flow.mask:
            # check if the conversion factor was meters to feet
            actual_flow = round(float(flow) * MTF_CONVERSION_FAC, 4)
            if abs(MTF_CONVERSION_FAC - 35.3147) < 0.001:
                unit = "cfs"
            elif abs(MTF_CONVERSION_FAC - 1.0) < 0.001:
                unit = "m3/s"
            else:
                unit = ""
                logger.warning(
                    "Conversion factor detected that is neither meters to feet or 1.0."
                )

            # save data in point dict
            point = {
                "flow": actual_flow,
                "flow units": unit,
                "COMID": str(comid),
                "distance": round(float(dist), 3),
                "distance unit": "mi",
                "date": date.strftime("%Y-%m-%d"),
            }
            points[str(comid)] = point
            num_found += 1
            if num_found >= NUM_LOCAL_GAGES:
                break

    return points


def update_nwm_manifest(date, lat, lon, STUSPS):
    # record basic info
    run_manifest.reset()
    run_manifest.update(key="analysis_type", value="NWM_norm")
    run_manifest.update(key="analysis_date", value=date)
    run_manifest.update(key="analysis_pos", value=f"{lat},{lon}")
    run_manifest.update(key="apt_version", value=VERSION)

    # derive and record historic source
    uri_dict = {
        "CONUS": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
        "AK": "s3://noaa-nwm-retrospective-3-0-pds/Alaska/zarr/chrtout.zarr",
        "HI": "s3://noaa-nwm-retrospective-3-0-pds/Hawaii/zarr/chrtout.zarr",
        "PR": "s3://noaa-nwm-retrospective-3-0-pds/PR/zarr/chrtout.zarr",
    }
    nwm_uri = uri_dict.get(STUSPS, None)
    run_manifest.update(key="NWM_historic_data", value=nwm_uri)

    # derive analysis assim source
    if isinstance(date, str):
        date_obj = datetime.strptime(date, "%Y-%m-%d")
    else:
        date_obj = date
    date_str = date_obj.strftime("%Y%m%d")
    base_url = f"https://storage.googleapis.com/national-water-model/nwm.{date_str}/analysis_assim"
    if STUSPS == "CONUS":
        file_name = f"/nwm.t00z.analysis_assim.channel_rt.tm02.conus.nc"
    elif STUSPS == "HI":
        file_name = f"_hawaii/nwm.t00z.analysis_assim.channel_rt.tm0245.hawaii.nc"
    elif STUSPS == "PR":
        file_name = f"_puertorico/nwm.t00z.analysis_assim.channel_rt.tm02.puertorico.nc"
    elif STUSPS == "AK":
        file_name = f"_alaska/nwm.t00z.analysis_assim.channel_rt.tm02.alaska.nc"
    url = f"{base_url}{file_name}"
    run_manifest.update(key="NWM_analysis_data", value=url)


def local_norm_nwm(lat, lon, date, save_path=None):
    # setup data dependancies and find local points with data
    logger.info("Initializing nwm norm script...")

    if isinstance(date, str):
        date_obj = datetime.strptime(date, "%Y-%m-%d")
    else:
        date_obj = date

    if date_obj < datetime(2023, 7, 21):
        print(
            "2023-07-21 earliest NWM analysis can be run, limited data before 2023-07-21"
        )
        return -1

    data_dir = find_file_or_dir(os.getcwd(), "data")
    shapefile_path = find_file_or_dir(os.getcwd(), "nwm_COMID_APTv3.0.shp")
    local_comids = find_closest_comids(shapefile_path, lat, lon)
    if len(local_comids) == 0:
        logger.error(
            "\nNWM does not have enough data near point to complete an analysis."
        )
        return -1

    # get flow on observation date at local points
    STUSPS = get_special_region(lat=lat, lon=lon)
    update_nwm_manifest(date=date, lat=lat, lon=lon, STUSPS=STUSPS)
    flow_points = get_nwm_flow(local_comids, date, STUSPS=STUSPS, data_dir=data_dir)
    if flow_points == -1:
        return -1
    comid_list = [int(point["COMID"]) for point in flow_points.values()]
    logger.debug(f"local comid: {comid_list}")

    # download historic (1990-2020) nwm flow data
    if not os.path.exists(os.path.join(data_dir, f"nwm_historic_{comid_list[0]}.csv")):
        ret_code = download_nwm_historic(
            comid_list, STUSPS=STUSPS, data_dir=data_dir, timeout=NWM_TIMEOUT_MINS * 60
        )
        if ret_code != 0:
            logger.error(
                "\nNWM data could not be downloaded. Please try again in a few minutes."
            )
            return -1

    # open downloaded nwm data
    csv_file = os.path.join(data_dir, f"nwm_historic_{comid_list[0]}.csv")
    df = pd.read_csv(csv_file)

    # loop through the points and run calculations on the data
    logger.info("Processing downloaded nwm data...")
    kml = Kml()
    summary_df = pd.DataFrame()
    for column in tqdm(
        df.columns[1:],
        unit="COMID",
        ascii=r" \|/-~",
        desc="Processing",
        dynamic_ncols=True,
    ):
        if str(column) not in flow_points:
            logger.debug(
                f"'{str(column)}' not found. Available: {sorted(flow_points.keys())}"
            )
            continue
        dataset_df = df[[df.columns[0], column]].copy()
        flow = flow_points[str(column)]["flow"]
        dataset_df.rename(
            columns={df.columns[0]: "time", column: "discharge"}, inplace=True
        )
        stat_df, hist_por = calc_stats(dataset_df)
        # perc = calc_percentile(flow, stat_df, date)
        perc = calc_percentile(flow, dataset_df, date)

        # skip bad data
        if perc == -1:
            continue

        if perc > 75:
            normalacy = "Wet/Above Normal"
        elif perc >= 25:
            normalacy = "Normal"
        else:
            normalacy = "Dry/Below Normal"

        day_of_year = pd.to_datetime(date).strftime("%m-%d")
        percentile_stats = stat_df.loc[stat_df["day"] == day_of_year].copy()
        percentile_stats["COMID"] = str(column)
        percentile_stats["flow (cfs)"] = flow
        percentile_stats["flow percentile"] = perc
        percentile_stats["condition"] = normalacy

        local_dist = local_comids.loc[
            local_comids["COMID"] == int(column), "dist"
        ].values
        if len(local_dist) >= 1:
            percentile_stats["distance (mi)"] = f"{local_dist[0]:.2f}"
        else:
            percentile_stats["distance (mi)"] = "-1"
        percentile_stats["hist period of record"] = hist_por
        summary_df = pd.concat([summary_df, percentile_stats], ignore_index=True)

        # try to get the location of the COMID, if not just use the observation point
        matching_comid = local_comids[local_comids["COMID"] == int(column)]
        if not matching_comid.empty:
            loc_tuple = (
                matching_comid["lon"].values[0],
                matching_comid["lat"].values[0],
            )
            description = (
                f"distance from point = {matching_comid['dist'].values[0]:.2f} mi\n"
            )
            description += f"\nCONDITION ON {date}:\n"
            description += f"flow = {flow} cfs\n"
            description += f"flow percentile= {perc:.1f}%\n"
            description += f"flow is {normalacy} \n"
            description += f"Historic period of record {hist_por} \n"
        else:
            loc_tuple = (lon, lat)
            description = "POINT HAD AN ERROR."

        point = kml.newpoint(
            name=str(column), coords=[loc_tuple], description=description
        )
        if perc > 80.0:
            point.style.iconstyle.icon.href = (
                "http://maps.google.com/mapfiles/kml/paddle/blu-blank.png"
            )
        elif perc > 60.0:
            point.style.iconstyle.icon.href = (
                "http://maps.google.com/mapfiles/kml/paddle/ltblu-blank.png"
            )
        elif perc > 40.0:
            point.style.iconstyle.icon.href = (
                "http://maps.google.com/mapfiles/kml/paddle/grn-blank.png"
            )
        elif perc > 20.0:
            point.style.iconstyle.icon.href = (
                "http://maps.google.com/mapfiles/kml/paddle/ylw-blank.png"
            )
        else:
            point.style.iconstyle.icon.href = (
                "http://maps.google.com/mapfiles/kml/paddle/red-blank.png"
            )

    point = kml.newpoint(name="QUERY POINT", coords=[(lon, lat)])
    point.style.iconstyle.icon.href = (
        "http://maps.google.com/mapfiles/kml/paddle/purple-stars.png"
    )

    if len(summary_df) == 0:
        logger.error(
            "\nNWM does not have enough data near point to complete an analysis."
        )
        return -1

    if save_path is not None:
        logger.info("Outputing processed data...")
        kml.save(os.path.join(save_path, f"NWM_STATS_{date}.kml"))

        summary_df = summary_df[
            ["COMID", "day", "flow (cfs)", "flow percentile", "condition"]
            + [
                col
                for col in summary_df.columns
                if col
                not in ["day", "COMID", "flow (cfs)", "flow percentile", "condition"]
            ]
        ]
        run_manifest.write_to_file(os.path.join(data_dir, "manifest.json"))
        summary_df.to_csv(os.path.join(save_path, f"NWM_STATS_{date}.csv"), index=False)

    # job done
    logger.info("NWM script complete")
    return 0


# ### MAIN ###


def local_norm(lat, lon, date, save_path=None):
    usps_code = get_special_region(lat=lat, lon=lon)

    if usps_code in ["VI"]:
        logger.warning(f"{usps_code} is not available")
        return

    if ENABLE_USGS_CALC and (usps_code != -1):
        local_norm_usgs(lat=lat, lon=lon, date=date, save_path=save_path)
    if ENABLE_NWM_CALC and (usps_code != -1):
        if usps_code in ["CONUS", "PR", "HI", "AK"]:
            local_norm_nwm(lat, lon, date, save_path)


def test():
    points = [
        (30, -90),
        (61.1925, -149.8200),
        (21.2904, -157.7299),
        (18.217, -66.289),
        (13.4091, 144.7808),
        (18.4014, -65.8110),
        (18.3291, -64.8973),
        (29.331518, -94.800046),
        (55, -105),
    ]
    date = "2025-06-01"
    try:
        data_dir = find_file_or_dir(os.getcwd(), "data")
    except:
        data_dir = os.path.join(os.getcwd(), "data")

    for point in points:
        lat, lon = point
        usps_code = get_special_region(lat=lat, lon=lon)
        special_save = os.path.join(data_dir, f"{usps_code}", f"{lat}, {lon}")
        os.makedirs(special_save, exist_ok=True)
        logger.info(f"[{usps_code}] {lat}, {lon}")
        local_norm(lat=lat, lon=lon, date=date, save_path=special_save)


def generate_conus_grid(interval=10):
    # Convert interval from miles to degrees (approximately)
    # One degree of latitude is approximately  69.172 miles
    # One degree of longitude is approximately  53.633 miles at the equator
    lat_interval = interval / 69.172
    lon_interval = interval / 53.633

    min_lat, max_lat = 24.7433195, 49.3457868
    min_lon, max_lon = -124.7844079, -66.9513812

    lats = np.arange(min_lat, max_lat + lat_interval, lat_interval)
    lons = np.arange(min_lon, max_lon + lon_interval, lon_interval)
    grid_lats, grid_lons = np.meshgrid(lats, lons)
    coordinates = [
        (lat, lon) for lat, lon in zip(grid_lats.flatten(), grid_lons.flatten())
    ]
    df = pd.DataFrame(coordinates, columns=["Latitude", "Longitude"])
    coords_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude)
    )
    coords_gdf.crs = "EPSG:4326"

    usa_path = next(
        glob.iglob(f"{os.getcwd()}/**/tl_2023_us_state.shp", recursive=True), None
    )
    if usa_path == None:
        raise FileNotFoundError("Need tl_2012_us_state.shp")
    usa_gdf = gpd.read_file(usa_path)
    if usa_gdf.crs != "EPSG:4326":
        try:
            usa_gdf = usa_gdf.to_crs("EPSG:4326")
        except:
            usa_gdf.crs = "EPSG:4326"

    joined_gdf = gpd.sjoin(coords_gdf, usa_gdf, predicate="within")
    coords_list = []
    for _, row in joined_gdf.iterrows():
        # Get the geometry of the current row, which is a Point object
        point = row["geometry"]

        # Extract the x and y coordinates from the Point object
        x, y = point.x, point.y

        # Append the coordinates as a tuple to the list
        coords_list.append((y, x))

    return coords_list


def test_grid():
    avg_time = 0.0
    runs = 0

    coordinates = generate_conus_grid(175)
    logger.info(f"testing {len(coordinates)} points.")
    for lat, lon in coordinates:
        start_time = time.time()
        local_norm(lat, lon, "2022-02-20")
        end_time = time.time()
        elapsed_time = end_time - start_time
        avg_time = ((avg_time * runs) + elapsed_time) / (runs + 1)
        runs += 1

        if not (runs % 10):
            logger.info(f"avg {avg_time:.3f} seconds, {runs}/{len(coordinates)}")

    logger.info("==== GRIDED TEST COMPLETE ====")
    logger.info(f"avg {avg_time:.3f} seconds, {runs}/{len(coordinates)}")


if __name__ == "__main__":
    start_time = time.time()
    logger.info(f"SCRIPT START")
    test()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"elapsed time: {elapsed_time:.1f} seconds\n")
