# core/pdsi.py
import os
import zipfile
import sys
import logging
from typing import Tuple
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import requests
from config import get_base_url

# Setup detailed logging to see the script's actions
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
logger = logging.getLogger(__name__)

# Constants for NOAA data fetching
BASE_URL = get_base_url("pdsi_base")
MIN_FILE_SIZE = 4_000_000  # 4MB rough size check for a valid pdsidv file


def classify_pdsi(value: float) -> Tuple[str, str]:
    """Return a human-readable PDSI classification and a corresponding color."""
    light_blue = (0.4, 0.5, 0.8)
    light_red = (0.8, 0.5, 0.5)
    white = (1, 1, 1)

    if value is None or pd.isna(value) or value == -99.99:
        return "Not Available", white
    if value >= 4.0:
        return "Extreme Wetness", light_blue
    if value >= 3.0:
        return "Severe Wetness", light_blue
    if value >= 2.0:
        return "Moderate Wetness", light_blue
    if value >= 1.0:
        return "Mild Wetness", light_blue
    if value >= 0.5:
        return "Incipient Wetness", light_blue
    if value > -0.5:
        return "Normal", white
    if value > -1.0:
        return "Incipient Drought", light_red
    if value > -2.0:
        return "Mild Drought", light_red
    if value > -3.0:
        return "Moderate Drought", light_red
    if value > -4.0:
        return "Severe Drought", light_red
    return "Extreme Drought", light_red


def _get_climdiv_from_point(lat: float, lon: float, shapefile_path: str) -> str | None:
    """Find the 4-digit climate division code for a lat/lon point using local shapefile."""
    if not os.path.exists(shapefile_path):
        logger.error(f"Shapefile not found at {shapefile_path}")
        return None
    try:
        gdf = gpd.read_file(shapefile_path)
        logger.debug(f"Shapefile loaded. Native CRS is {gdf.crs}")
        point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
        if gdf.crs is not None and gdf.crs != point_gdf.crs:
            logger.debug(f"Reprojecting target point from {point_gdf.crs} to {gdf.crs}")
            point_gdf = point_gdf.to_crs(gdf.crs)
        target_point = point_gdf.geometry.iloc[0]
        containing_poly = gdf[gdf.geometry.contains(target_point)]
        if not containing_poly.empty:
            raw_code = str(containing_poly.iloc[0]["CLIMDIV"])
            clean_code = raw_code.zfill(4)
            logger.debug(
                f"Found Climate Division: {clean_code} for coordinates ({lat}, {lon})"
            )
            return clean_code
        else:
            logger.warning(
                f"Coordinates ({lat}, {lon}) do not fall within any climate division polygon."
            )
            return None
    except Exception as e:
        logger.error(
            f"Error reading shapefile or processing geometry: {e}", exc_info=True
        )
        return None


def _find_valid_cached_pdsidv(cache_dir: Path) -> Path | None:
    """Return the newest valid (size-checked) cached pdsidv file, or None."""
    candidates = [
        f
        for f in cache_dir.glob("climdiv-pdsidv-v1.0.0-*")
        if f.is_file() and f.stat().st_size >= MIN_FILE_SIZE
    ]
    if not candidates:
        return None
    # Prefer the file with the newest name (YYYYMMDD suffix) then fall back to mtime
    return max(candidates, key=lambda p: (p.name, p.stat().st_mtime))


def _get_latest_pdsidv_filepath(
    cache_dir: Path, force_download: bool = False
) -> Path | None:
    """
    Ensure a usable PDSI file is available.

    Logic:
      1. If a valid cached file already exists and force_download=False → use it
         (no network call).
      2. Otherwise contact NOAA for the current procdate, download if needed,
         then return the new file.
      3. On network failure fall back to any existing valid cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Fast path: use existing cache when possible ---
    if not force_download:
        cached = _find_valid_cached_pdsidv(cache_dir)
        if cached is not None:
            logger.debug(f"Using existing cached PDSI file: {cached.name}")
            return cached
        logger.debug("No valid cached PDSI file found – will attempt download.")

    # --- Network path ---
    try:
        logger.debug("Fetching latest processing date from NOAA...")
        procdate = requests.get(f"{BASE_URL}/procdate.txt", timeout=15).text.strip()
        filename = f"climdiv-pdsidv-v1.0.0-{procdate}"
        local_path = cache_dir / filename

        needs_download = (
            force_download
            or not local_path.exists()
            or local_path.stat().st_size < MIN_FILE_SIZE
        )

        if needs_download:
            url = f"{BASE_URL}/{filename}"
            logger.info(f"Downloading latest PDSI data: {filename}...")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.debug("Download complete.")
        else:
            logger.debug(f"Cached file already matches latest: {filename}")

        # Clean up older versions
        for old_file in cache_dir.glob("climdiv-pdsidv-v1.0.0-*"):
            if old_file.name != filename:
                logger.debug(f"Removing old cache file: {old_file.name}")
                old_file.unlink(missing_ok=True)

        return local_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to contact NOAA for PDSI data: {e}")
        # Fallback to any existing valid cache
        cached = _find_valid_cached_pdsidv(cache_dir)
        if cached is not None:
            logger.warning(f"Falling back to cached file: {cached.name}")
            return cached
        return None


def _get_pdsi_from_file(
    clim_div: str, year: int, month: int, pdsi_data_path: Path
) -> float:
    """Return the PDSI value from the fixed-width text file."""
    key_to_find = f"{clim_div}05{year}"
    logger.debug(f"Scanning {pdsi_data_path.name} for key: '{key_to_find}'")
    try:
        with open(pdsi_data_path) as f:
            for line in f:
                if line.startswith(key_to_find):
                    monthly_values_str = [line[i : i + 7] for i in range(10, 94, 7)]
                    monthly_values = [float(v) for v in monthly_values_str]
                    value = monthly_values[month - 1]
                    logger.debug(f"Raw PDSI value for month {month}: {value}")
                    if value == -99.99 and month > 1:
                        fallback = monthly_values[month - 2]
                        logger.warning(
                            f"Month {month} is -99.99. Falling back to month {month-1}: {fallback}"
                        )
                        return fallback
                    return value
        logger.warning(
            f"Key '{key_to_find}' was NOT FOUND in {pdsi_data_path.name}. Is year {year} available?"
        )
        return -99.99
    except Exception as e:
        logger.error(f"Error parsing PDSI text file: {e}", exc_info=True)
        return -99.99


def get_pdsi_for_location(
    lat: float,
    lon: float,
    year: int,
    month: int,
    data_dir: str,
    force_download: bool = False,
) -> tuple[float, str, str]:
    """Main function to get PDSI value and classification for a location."""
    logger.info(
        f"Starting PDSI analysis for lat: {lat}, lon: {lon}, Date: {year}-{month:02d}"
    )

    # --- Step 1: Find Climate Division using LOCAL shapefile ---
    climdiv_shape_dir = Path(data_dir) / "climdiv"
    shapefile_path = climdiv_shape_dir / "GIS.OFFICIAL_CLIM_DIVISIONS.shp"
    zip_path = Path(data_dir) / "climdiv.zip"

    if not shapefile_path.exists():
        if not zip_path.exists():
            logger.error(f"Missing required file: {zip_path}")
            return -99.99, "Not Available", "#FFFFFF"
        try:
            logger.info(f"Shapefile not found. Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            logger.debug("Extraction complete.")
        except Exception as e:
            logger.error(f"Failed to extract zip file: {e}", exc_info=True)
            return -99.99, "Not Available", "#FFFFFF"

    clim_div_code = _get_climdiv_from_point(lat, lon, str(shapefile_path))
    if not clim_div_code:
        return -99.99, "Not Available", "#FFFFFF"

    # --- Step 2: Obtain PDSI data file (cache-first) ---
    pdsi_cache_dir = Path(data_dir) / "pdsi_cache"
    pdsi_data_file = _get_latest_pdsidv_filepath(
        pdsi_cache_dir, force_download=force_download
    )
    if not pdsi_data_file:
        logger.error("Could not download or find a cached PDSI data file.")
        return -99.99, "Not Available", "#FFFFFF"

    # --- Step 3: Extract Value and Classify ---
    value = _get_pdsi_from_file(clim_div_code, year, month, pdsi_data_file)
    classification, color = classify_pdsi(value)
    logger.info(f"Final PDSI Result: {value} -> {classification}")
    return value, classification, color


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdsi.py <path_to_data_dir>")
    else:
        data_directory = sys.argv[1]
        print("\n--- Running Test Case: Louisiana, Nov 2025 ---")
        result = get_pdsi_for_location(
            lat=30.00, lon=-90.00, year=2025, month=11, data_dir=data_directory
        )

        print("\n--- Running Test Case: California, July 2024 ---")
        result_future = get_pdsi_for_location(
            lat=37.77, lon=-122.41, year=2024, month=7, data_dir=data_directory
        )
