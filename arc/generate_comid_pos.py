import requests
import os
from datetime import datetime
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np
from tqdm import tqdm

tqdm.pandas(ascii=r" \|/-~")
import warnings

warnings.filterwarnings("ignore")

try:
    from arc.utils import setup_logger
except:
    from utils import setup_logger
logger = setup_logger()


def get_netcdf(date):
    # Construct the URL for the Analysis and Assimilation data
    base_url = "https://noaanwm.blob.core.windows.net/nwm"
    date_str = date.strftime("%Y%m%d")
    file_name = (
        f"nwm.{date_str}/short_range/nwm.t00z.short_range.channel_rt.f001.conus.nc"
    )
    # file_name = f"nwm.{date_str}/analysis_assim/nwm.t00z.analysis_assim.channel_rt.tm00.conus.nc"
    url = f"{base_url}/{file_name}"

    try:
        # Download the NetCDF file
        response = requests.get(url)
        if response.status_code != 200:
            return -1

        # Save the NetCDF file locally
        with open(f"nwm_data_{date_str}.nc", "wb") as f:
            f.write(response.content)

        return 0

    except:
        return -1


def download_and_save_gdf(file_path):
    """
    Download the GeoDataFrame from the Parquet file and save it to a local file.
    If the file already exists, load the saved copy instead of downloading a new one.
    """
    if os.path.exists(file_path):
        logger.info(f"loading saved gdf from {file_path}...")
        gdf = gpd.read_parquet(file_path)
    else:
        logger.info(f"downloading gdf...")
        gdf = gpd.read_parquet(
            "az://hydrofabric/nwm_reaches_conus.parquet",
            storage_options={"account_name": "noaanwm"},
        )
        logger.info(f"saving gdf to {file_path}...")
        gdf.to_parquet(file_path)
    return gdf


def get_first_point(row):
    """
    Extract the first point from a geometry.
    If the geometry is a MultiPolygon or MultiLineString, it extracts the first point from the first part.
    """
    if row.geometry.geom_type.startswith("Multi"):
        # Extract the first point from the first part of the geometry
        first_point = np.array(list(row.geometry.geoms[0].coords))[0]
    else:
        # If it's not a multi-part geometry, extract the first point directly
        first_point = np.array(list(row.geometry.coords))[0]
    return Point(first_point)


def get_coord_data(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)

    # Load the GeoDataFrame from the Parquet file
    logger.info(f"starting record download...")
    gdf_filepath = os.path.join(output_dir, "raw_gdf.bin")
    gdf = download_and_save_gdf(gdf_filepath).set_index("ID")

    # Apply the get_first_point function to each row to get the first point of each geometry
    # Use tqdm.progress_apply for progress tracking
    logger.info(f"processing data...")
    modified_df = gdf.progress_apply(
        lambda row: pd.Series({"comid": row.name, "geometry": get_first_point(row)}),
        axis=1,
    )

    # Convert the modified DataFrame back to a GeoDataFrame
    modified_gdf = gpd.GeoDataFrame(modified_df, geometry="geometry")

    # Write the modified GeoDataFrame to a shapefile
    logger.info(f"saving data to disk as shapefile...")
    # ID will not exist here, even if I reset index
    shp_path = os.path.join(output_dir, "comid_pos.shp")
    modified_gdf.to_file(shp_path)

    gdf = gpd.read_file(shp_path)
    gdf = gdf.reset_index(drop=True)
    # magically ID exist now
    gdf = gdf.drop(columns=["ID"])
    gdf.to_file(shp_path)

    logger.info(f"comid dict generated and saved as shapefile.")


if __name__ == "__main__":
    # # Example usage
    # date = datetime(2023, 1, 23)
    # get_netcdf(date)
    get_coord_data()

    # gdf = download_and_save_gdf("raw_gdf.bin")
    # print(gdf["ID"])

    # lat = 40.7128  # Example latitude
    # lon = -74.0060  # Example longitude
    # file_path = "nwm_data_20230123.nc"
