import os
import sys
import random
import logging
import zipfile
import requests
import geopandas as gpd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Configurable global constants for the random generator
MAX_CONSECUTIVE_FAILURES = 1500
SPACING_DECREMENT_MILES = 0.25
MIN_SPACING_MILES = 2


def normalize_dir_path(path: str) -> str:
    """Expands user tilde, normalizes path separators, and resolves to absolute path."""
    return os.path.abspath(os.path.expanduser(path))


def get_local_huc2(lat: float, lon: float, data_dir: str) -> str:
    """
    Reads the HUC2 shapefile directly from a local WBD.zip archive
    to find the precise HUC2 code for the given coordinates.
    """
    abs_data_dir = normalize_dir_path(data_dir)

    if not os.path.exists(abs_data_dir):
        raise NotADirectoryError(
            f"Data directory does not exist. Resolved absolute path: '{abs_data_dir}'"
        )

    wbd_zip_path = os.path.join(abs_data_dir, "WBD.zip")
    if not os.path.exists(wbd_zip_path):
        raise FileNotFoundError(
            f"Required base dataset 'WBD.zip' not found. Looked at absolute path: '{wbd_zip_path}'"
        )

    # Format spatial engine virtual directory read to read directly from the zip
    normalized_zip = wbd_zip_path.replace("\\", "/")
    huc2_uri = f"zip://{normalized_zip}!HUC2.shp"
    logger.debug(f"Loading local HUC2 boundaries from {wbd_zip_path}...")

    try:
        huc2_gdf = gpd.read_file(huc2_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to read HUC2.shp from '{wbd_zip_path}': {e}")

    input_point = Point(lon, lat)

    # Align CRS if the shapefile is not using WGS84 (EPSG:4326)
    if huc2_gdf.crs and huc2_gdf.crs != "EPSG:4326":
        point_series = gpd.GeoSeries([input_point], crs="EPSG:4326").to_crs(
            huc2_gdf.crs
        )
        input_point = point_series.iloc[0]

    # Perform point-in-polygon spatial lookup
    spatial_match = huc2_gdf[huc2_gdf.geometry.contains(input_point)]
    if spatial_match.empty:
        raise ValueError(
            f"Coordinates ({lat}, {lon}) do not fall within any known US HUC2 region "
            f"in dataset '{wbd_zip_path}'."
        )

    matched_feature = spatial_match.iloc[0]

    # Check common casing for the HUC2 column name
    huc2_code = matched_feature.get("huc2") or matched_feature.get("HUC2")
    if not huc2_code:
        raise KeyError(
            "The shapefile is missing a recognizable 'HUC2' attribute column."
        )

    return str(huc2_code)


def download_and_cache_huc2(huc2_code: str, data_dir: str) -> str:
    """
    Checks the local cache directory for the required HUC2 shapefile ZIP.
    If missing, downloads it directly from the USGS National Map repository.
    """
    abs_data_dir = normalize_dir_path(data_dir)
    cache_dir = os.path.join(abs_data_dir, "huc_cache")

    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as e:
        raise PermissionError(
            f"Failed to create cache directory at absolute path '{cache_dir}': {e}"
        )

    zip_filename = f"WBD_{huc2_code}_HU2_Shape.zip"
    local_zip_path = os.path.join(cache_dir, zip_filename)

    if os.path.exists(local_zip_path):
        logger.debug(f"Found cached dataset: {local_zip_path}")
        return local_zip_path

    # Public USGS AWS S3 distribution point for the Watershed Boundary Dataset
    url = f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/{zip_filename}"
    logger.debug(f"Target ZIP not in cache. Downloading from USGS: {url}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except Exception as e:
        status = (
            response.status_code
            if "response" in locals() and hasattr(response, "status_code")
            else "Unknown"
        )
        logger.error(f"Failed to initiate download from USGS: {e}")
        raise ConnectionError(
            f"Failed to fetch dataset from USGS ({url}). HTTP status: {status}. "
            f"Please verify that HUC2 code '{huc2_code}' is valid."
        )

    with open(local_zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    logger.info(f"Successfully downloaded and cached: {local_zip_path}")
    return local_zip_path


def get_layer_path_from_zip(zip_path: str, huc_level: int) -> str:
    """
    Inspects the ZIP archive to find the relative path of the target HUC shapefile,
    returning a virtual URI that GeoPandas can read directly.
    """
    target_pattern = f"WBDHU{huc_level}.shp"
    with zipfile.ZipFile(zip_path, "r") as z:
        for file_info in z.infolist():
            if file_info.filename.endswith(target_pattern):
                normalized_zip = os.path.abspath(zip_path).replace("\\", "/")
                return f"zip://{normalized_zip}!{file_info.filename}"

    raise FileNotFoundError(
        f"Could not find shapefile matching level HUC{huc_level} inside ZIP: '{os.path.abspath(zip_path)}'."
    )


def generate_sampling_points(
    polygon, spacing_miles: float, max_points: int = 100, random_seed: int = 2236
) -> list:
    """
    Generates semi-random points inside the polygon boundaries using distance-spacing constraints.
    Calculations are performed in an Equal-Area projection (EPSG:5070) for metric precision.
    """
    rng = random.Random(random_seed)
    projected_polygon = (
        gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(epsg=5070).iloc[0]
    )
    minx, miny, maxx, maxy = projected_polygon.bounds

    accepted_points_projected = []
    consecutive_failures = 0
    current_spacing_meters = spacing_miles * 1609.344

    logger.debug(
        f"Starting point sampling with bounds: X({minx:.1f}, {maxx:.1f}), Y({miny:.1f}, {maxy:.1f})"
    )

    while len(accepted_points_projected) < max_points:
        test_point = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))

        if projected_polygon.contains(test_point):
            properly_spaced = True
            for accepted_pt in accepted_points_projected:
                if test_point.distance(accepted_pt) < current_spacing_meters:
                    properly_spaced = False
                    break

            if properly_spaced:
                accepted_points_projected.append(test_point)
                consecutive_failures = 0
                logger.debug(
                    f"Accepted point {len(accepted_points_projected)}: ({test_point.x:.1f}, {test_point.y:.1f})"
                )
                continue

        consecutive_failures += 1

        if consecutive_failures > MAX_CONSECUTIVE_FAILURES:
            if spacing_miles > MIN_SPACING_MILES:
                spacing_miles = round(spacing_miles - SPACING_DECREMENT_MILES, 2)
                current_spacing_meters = spacing_miles * 1609.344
                logger.debug(
                    f"Spacing constraints relaxed. Lowering spacing to: {spacing_miles} miles."
                )
                consecutive_failures = 0
            else:
                logger.debug(
                    f"Hit absolute density limit. Terminating generation with {len(accepted_points_projected)} points."
                )
                break

    if not accepted_points_projected:
        return []

    points_series = gpd.GeoSeries(accepted_points_projected, crs="EPSG:5070").to_crs(
        epsg=4326
    )
    return [(point.y, point.x) for point in points_series]


def dynamic_watershed_sampler(
    lat: float, lon: float, huc_level: int, data_dir: str, random_seed: int = 2236
) -> dict:
    """
    High-level entry point.
    Identifies the correct localized HUC polygon and generates sampling points.
    """
    if huc_level not in [2, 8, 10, 12]:
        raise ValueError(
            "Invalid HUC level. This program supports levels: 2, 8, 10, or 12."
        )

    # Normalize base directory early
    normalized_data_dir = normalize_dir_path(data_dir)

    input_point = Point(lon, lat)

    # Step 1: Perform local spatial lookup to get the top-level HUC2 code
    logger.info(f"Target data directory: '{normalized_data_dir}'")
    logger.info("Executing local spatial lookup for target coordinate region...")
    huc2_code = get_local_huc2(lat, lon, normalized_data_dir)
    logger.info(f"Coordinates mapped to Water Resource Region: HUC2 '{huc2_code}'")

    # Step 2: Download, cache, and open the specific local zip file
    zip_path = download_and_cache_huc2(huc2_code, normalized_data_dir)
    layer_uri = get_layer_path_from_zip(zip_path, huc_level)

    # Step 3: Load the specific HUC level dataset
    logger.debug(f"Reading layer: WBDHU{huc_level}")
    gdf = gpd.read_file(layer_uri)

    # Step 4: Perform a vectorized spatial lookup
    spatial_match = gdf[gdf.geometry.contains(input_point)]
    if spatial_match.empty:
        if gdf.crs and gdf.crs != "EPSG:4326":
            point_series = gpd.GeoSeries([input_point], crs="EPSG:4326").to_crs(gdf.crs)
            spatial_match = gdf[gdf.geometry.contains(point_series.iloc[0])]

        if spatial_match.empty:
            raise ValueError(
                f"Spatial mismatch: input point ({lat}, {lon}) not found inside HUC{huc_level} layer from '{zip_path}'."
            )

    matched_feature = spatial_match.iloc[0]
    huc_field_name = f"huc{huc_level}"

    # Extract structural metadata
    huc_id = matched_feature.get(huc_field_name) or matched_feature.get(
        huc_field_name.upper()
    )
    huc_name = matched_feature.get("name") or matched_feature.get("NAME", "Unknown")
    target_polygon = matched_feature.geometry

    # Calculate geographic area
    projected_geometry = (
        gpd.GeoSeries([target_polygon], crs="EPSG:4326").to_crs(epsg=5070).iloc[0]
    )
    area_sq_miles = round(projected_geometry.area / 2589988.11, 2)

    logger.info(
        f'Target identified: [HUC{huc_level}] {huc_id} - "{huc_name}" ({area_sq_miles} sq miles)'
    )

    # Step 5: Generate sampling coordinates inside target boundary
    logger.debug("Spawning randomized sampling points inside polygon boundaries...")
    points = generate_sampling_points(
        target_polygon, spacing_miles=3.75, max_points=20, random_seed=random_seed
    )

    return {
        "huc_id": huc_id,
        "name": huc_name,
        "area_sq_miles": area_sq_miles,
        "points": points,
    }


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: python huc_sampling.py <path_to_data_dir>")
    else:
        import time

        start = time.time()
        DATA_CACHE_DIR = sys.argv[1]
        TEST_LAT = 30
        TEST_LON = -90
        TARGET_HUC_LEVEL = 12

        print("==================================================")
        print("   Running Watershed Query & Point Sampler Core   ")
        print("==================================================")

        try:
            result = dynamic_watershed_sampler(
                lat=TEST_LAT,
                lon=TEST_LON,
                huc_level=TARGET_HUC_LEVEL,
                data_dir=DATA_CACHE_DIR,
            )
            print("================== Results ==================")
            print(f"HUC Code:     {result['huc_id']}")
            print(f"HUC Name:     {result['name']}")
            print(f"Total Area:   {result['area_sq_miles']} sq miles")
            print(f"Points Found: {len(result['points'])}")
            print("---------------------------------------------")
            print("Sample Coordinates (Lat, Lon):")
            for pt in result["points"][:10]:
                print(f"  {pt[0]:.6f}, {pt[1]:.6f}")
            if len(result["points"]) > 10:
                print("  ...")
            print("=============================================")
            print(f"Process complete in {time.time() - start:.2f} seconds.")
        except Exception as ex:
            logger.exception(f"An error occurred during program execution: {ex}")
