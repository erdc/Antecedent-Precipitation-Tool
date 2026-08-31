# core/area_sampling.py
"""
Area / watershed sampling for APT-style analyses.

Supports:
  - HUC-2 / 8 / 10 / 12 (via local WBD.zip + on-demand HU2 downloads)
  - Custom polygon (shapefile or GeoJSON)

Returns a uniform dict:
  {
    "huc_id"      : str,          # real HUC code or custom name / "CUSTOM"
    "name"        : str,
    "area_sq_miles": float,
    "points"      : list[tuple[float, float]] | list[dict],  # (lat, lon)
  }
"""

from __future__ import annotations

import logging
import os
import random
import zipfile
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import requests
from config import get_base_url
from shapely.geometry import Point, shape
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables (match official APT density behaviour as closely as practical)
# ---------------------------------------------------------------------------
MAX_CONSECUTIVE_FAILURES = 1500
SPACING_DECREMENT_MILES = 0.25
MIN_SPACING_MILES = 2.0
DEFAULT_SPACING_MILES = 3.75
DEFAULT_MAX_POINTS = 20
DEFAULT_RANDOM_SEED = 2236


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def normalize_dir_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


# ---------------------------------------------------------------------------
# HUC-2 lookup (local WBD.zip)
# ---------------------------------------------------------------------------
def get_local_huc2(lat: float, lon: float, data_dir: str) -> str:
    abs_data_dir = normalize_dir_path(data_dir)
    wbd_zip_path = os.path.join(abs_data_dir, "WBD.zip")

    if not os.path.exists(wbd_zip_path):
        raise FileNotFoundError(
            f"Required base dataset 'WBD.zip' not found at '{wbd_zip_path}'"
        )

    normalized_zip = wbd_zip_path.replace("\\", "/")
    huc2_uri = f"zip://{normalized_zip}!HUC2.shp"
    logger.debug("Loading local HUC2 boundaries from %s", wbd_zip_path)

    huc2_gdf = gpd.read_file(huc2_uri)
    input_point = Point(lon, lat)

    if huc2_gdf.crs and huc2_gdf.crs != "EPSG:4326":
        input_point = (
            gpd.GeoSeries([input_point], crs="EPSG:4326").to_crs(huc2_gdf.crs).iloc[0]
        )

    spatial_match = huc2_gdf[huc2_gdf.geometry.contains(input_point)]
    if spatial_match.empty:
        raise ValueError(
            f"Coordinates ({lat}, {lon}) do not fall inside any HUC2 in WBD.zip"
        )

    matched = spatial_match.iloc[0]
    huc2_code = matched.get("huc2") or matched.get("HUC2")
    if not huc2_code:
        raise KeyError("HUC2 shapefile is missing a 'huc2' / 'HUC2' attribute")
    return str(huc2_code)


# ---------------------------------------------------------------------------
# On-demand HU2 download + cache
# ---------------------------------------------------------------------------
def download_and_cache_huc2(huc2_code: str, data_dir: str) -> str:
    abs_data_dir = normalize_dir_path(data_dir)
    cache_dir = os.path.join(abs_data_dir, "huc_cache")
    os.makedirs(cache_dir, exist_ok=True)

    zip_filename = f"WBD_{huc2_code}_HU2_Shape.zip"
    local_zip_path = os.path.join(cache_dir, zip_filename)

    if os.path.exists(local_zip_path):
        logger.debug("Using cached HU2 dataset: %s", local_zip_path)
        return local_zip_path

    url = f"{get_base_url('wbd_hu2_shape')}/" f"{zip_filename}"
    logger.info("Downloading HU2 dataset from USGS: %s", url)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(local_zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    logger.info("Cached: %s", local_zip_path)
    return local_zip_path


def get_layer_path_from_zip(zip_path: str, huc_level: int) -> str:
    target_pattern = f"WBDHU{huc_level}.shp"
    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            if info.filename.endswith(target_pattern):
                normalized = os.path.abspath(zip_path).replace("\\", "/")
                return f"zip://{normalized}!{info.filename}"
    raise FileNotFoundError(f"Could not find WBDHU{huc_level}.shp inside '{zip_path}'")


# ---------------------------------------------------------------------------
# Point generation (shared by HUC and custom polygon)
# ---------------------------------------------------------------------------
def generate_sampling_points(
    polygon,
    spacing_miles: float = DEFAULT_SPACING_MILES,
    max_points: int = DEFAULT_MAX_POINTS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> List[Tuple[float, float]]:
    """
    Semi-random points inside *polygon* with a minimum spacing constraint.
    Work is done in EPSG:5070 (equal-area) then returned as (lat, lon).
    """
    rng = random.Random(random_seed)
    projected = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(epsg=5070).iloc[0]
    minx, miny, maxx, maxy = projected.bounds

    accepted = []
    consecutive_failures = 0
    current_spacing_m = spacing_miles * 1609.344

    while len(accepted) < max_points:
        test = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))

        if projected.contains(test):
            if all(test.distance(p) >= current_spacing_m for p in accepted):
                accepted.append(test)
                consecutive_failures = 0
                continue

        consecutive_failures += 1
        if consecutive_failures > MAX_CONSECUTIVE_FAILURES:
            if spacing_miles > MIN_SPACING_MILES:
                spacing_miles = round(spacing_miles - SPACING_DECREMENT_MILES, 2)
                current_spacing_m = spacing_miles * 1609.344
                consecutive_failures = 0
                logger.debug("Relaxed spacing to %.2f mi", spacing_miles)
            else:
                logger.debug(
                    "Density limit reached – returning %d points", len(accepted)
                )
                break

    if not accepted:
        return []

    pts = gpd.GeoSeries(accepted, crs="EPSG:5070").to_crs(epsg=4326)
    return [(p.y, p.x) for p in pts]


# ---------------------------------------------------------------------------
# Custom polygon loader
# ---------------------------------------------------------------------------
def _load_polygon(path: str):
    """Load a shapefile or GeoJSON and return a single (multi)polygon in EPSG:4326."""
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"No features found in '{path}'")

    if gdf.crs is None:
        logger.warning("CRS missing on %s – assuming EPSG:4326", path)
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Dissolve everything into one geometry
    geom = unary_union(gdf.geometry)
    if geom.is_empty:
        raise ValueError(f"Geometry is empty after dissolve: '{path}'")
    return geom


def sample_custom_polygon(
    lat: float,
    lon: float,
    polygon_path: str,
    name: str = "CUSTOM",
    spacing_miles: float = DEFAULT_SPACING_MILES,
    max_points: int = DEFAULT_MAX_POINTS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    require_point_inside: bool = True,
) -> Dict[str, Any]:
    """
    Sample points inside a user-supplied polygon.
    """
    polygon = _load_polygon(polygon_path)
    input_pt = Point(lon, lat)

    if require_point_inside and not polygon.contains(input_pt):
        # soft warning – still sample the polygon
        logger.warning(
            "Input point (%.5f, %.5f) is outside the supplied polygon; "
            "sampling will still proceed inside the polygon.",
            lat,
            lon,
        )

    # Area in sq miles
    projected = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(epsg=5070).iloc[0]
    area_sq_miles = round(projected.area / 2589988.11, 2)

    points = generate_sampling_points(
        polygon,
        spacing_miles=spacing_miles,
        max_points=max_points,
        random_seed=random_seed,
    )

    # Guarantee at least the original point is present if the polygon is tiny
    if not points:
        points = [(round(lat, 6), round(lon, 6))]

    return {
        "huc_id": name or "CUSTOM",
        "name": name or os.path.splitext(os.path.basename(polygon_path))[0],
        "area_sq_miles": area_sq_miles,
        "points": points,
    }


# ---------------------------------------------------------------------------
# HUC sampler (original logic, cleaned)
# ---------------------------------------------------------------------------
def sample_huc(
    lat: float,
    lon: float,
    huc_level: int,
    data_dir: str,
    spacing_miles: float = DEFAULT_SPACING_MILES,
    max_points: int = DEFAULT_MAX_POINTS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    if huc_level not in (2, 8, 10, 12):
        raise ValueError("huc_level must be 2, 8, 10 or 12")

    data_dir = normalize_dir_path(data_dir)
    input_point = Point(lon, lat)

    logger.info("Looking up HUC2 for (%.5f, %.5f) …", lat, lon)
    huc2_code = get_local_huc2(lat, lon, data_dir)

    zip_path = download_and_cache_huc2(huc2_code, data_dir)
    layer_uri = get_layer_path_from_zip(zip_path, huc_level)

    gdf = gpd.read_file(layer_uri)
    spatial_match = gdf[gdf.geometry.contains(input_point)]

    if spatial_match.empty and gdf.crs and gdf.crs != "EPSG:4326":
        pt = gpd.GeoSeries([input_point], crs="EPSG:4326").to_crs(gdf.crs).iloc[0]
        spatial_match = gdf[gdf.geometry.contains(pt)]

    if spatial_match.empty:
        raise ValueError(
            f"Point ({lat}, {lon}) not found inside any HUC{huc_level} feature"
        )

    feat = spatial_match.iloc[0]
    field = f"huc{huc_level}"
    huc_id = feat.get(field) or feat.get(field.upper())
    name = feat.get("name") or feat.get("NAME") or "Unknown"
    polygon = feat.geometry

    projected = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(epsg=5070).iloc[0]
    area_sq_miles = round(projected.area / 2589988.11, 2)

    logger.info(
        'Target: HUC%d %s – "%s" (%.1f sq mi)', huc_level, huc_id, name, area_sq_miles
    )

    points = generate_sampling_points(
        polygon,
        spacing_miles=spacing_miles,
        max_points=max_points,
        random_seed=random_seed,
    )

    return {
        "huc_id": str(huc_id),
        "name": str(name),
        "area_sq_miles": area_sq_miles,
        "points": points,
    }


# ---------------------------------------------------------------------------
# Unified entry point used by the adapter
# ---------------------------------------------------------------------------
def dynamic_area_sampler(
    lat: float,
    lon: float,
    *,
    huc_level: Optional[int] = None,
    data_dir: str = "data",
    custom_polygon: Optional[str] = None,
    custom_name: Optional[str] = None,
    spacing_miles: float = DEFAULT_SPACING_MILES,
    max_points: int = DEFAULT_MAX_POINTS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Single entry point for both HUC and custom-polygon sampling.

    Exactly one of *huc_level* or *custom_polygon* must be supplied.
    """
    if custom_polygon:
        return sample_custom_polygon(
            lat=lat,
            lon=lon,
            polygon_path=custom_polygon,
            name=custom_name or "CUSTOM",
            spacing_miles=spacing_miles,
            max_points=max_points,
            random_seed=random_seed,
        )

    if huc_level is None:
        raise ValueError("Either huc_level or custom_polygon must be provided")

    return sample_huc(
        lat=lat,
        lon=lon,
        huc_level=huc_level,
        data_dir=data_dir,
        spacing_miles=spacing_miles,
        max_points=max_points,
        random_seed=random_seed,
    )
