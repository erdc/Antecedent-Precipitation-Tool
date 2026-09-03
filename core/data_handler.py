# data_handler.py
"""
Handles data processing, analysis, and storage.
This script takes raw data, performs calculations, and saves the results
as JSON and CSV files, ready for plotting.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COORD_DECIMALS = 6


class NpEncoder(json.JSONEncoder):
    """JSON serializer for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ====================== UTILITY FUNCTIONS ======================


def dict_to_series(d: Dict) -> pd.Series:
    """Converts a dictionary with date strings as keys to a pandas Series."""
    if not d:
        return pd.Series(dtype=float)
    s = pd.Series(d)
    s.index = pd.to_datetime(s.index)
    return s


def _compute_precip_condition(precip_data: Dict) -> (str, int):
    """Recomputes the wetness result string and score from stored series."""
    rolling_total = dict_to_series(precip_data.get("rolling_total"))
    normal_low = dict_to_series(precip_data.get("normal_low"))
    normal_high = dict_to_series(precip_data.get("normal_high"))

    if "obs_date" not in precip_data or not precip_data["obs_date"]:
        return "Error", 0

    obs_date = datetime.strptime(precip_data["obs_date"], "%Y-%m-%d")

    total_score = 0
    for days_prior, weight in [(0, 3), (30, 2), (60, 1)]:
        p_date = obs_date - timedelta(days=days_prior)
        obs_val = rolling_total.get(p_date)
        low_val = normal_low.get(p_date)
        high_val = normal_high.get(p_date)

        if any(pd.isna(v) or v is None for v in [obs_val, low_val, high_val]):
            c_val = 0
        elif obs_val > high_val:
            c_val = 3
        elif obs_val < low_val:
            c_val = 1
        else:
            c_val = 2
        total_score += c_val * weight

    if total_score < 10:
        condition = "Drier than Normal"
    elif total_score <= 14:
        condition = "Normal Conditions"
    else:
        condition = "Wetter than Normal"

    return condition, total_score


def _write_merged_stations_csv(precip_df: pd.DataFrame, output_dir: str, date_str: str):
    """
    Writes the merged station data to two CSV files: one in 10*mm and one in inches.
    """
    if precip_df.empty:
        return

    df = precip_df.reset_index()
    df.columns = ["Date", "precip_10mm"]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    csv_path_mm = os.path.join(output_dir, f"merged_stations_{date_str}.csv")
    df.to_csv(csv_path_mm, index=False)
    logger.info(f"Wrote merged stations CSV: {csv_path_mm}")

    df_in = df.copy()
    df_in["precip_in"] = df_in["precip_10mm"] / 254.0
    df_in = df_in[["Date", "precip_in"]]
    df_in.columns = ["Date", "PrecipitationInches"]

    csv_path_in = os.path.join(
        output_dir, f"merged_stations_converted_to_in_{date_str}.csv"
    )
    df_in.to_csv(csv_path_in, index=False)
    logger.info(f"Wrote merged stations (inches) CSV: {csv_path_in}")


def _coord_str(lat: float, lon: float) -> str:
    """Stable directory segment shared by store + PDF paths."""
    return f"{float(lat):.{COORD_DECIMALS}f}_{float(lon):.{COORD_DECIMALS}f}"


def _get_data_path(
    output_dir: str, lat: float, lon: float, date: datetime, suffix: str
) -> str:
    """Constructs the full path for storing a data file."""
    coord_directory = _coord_str(lat, lon)
    data_dir = os.path.join(output_dir, coord_directory, "data")
    os.makedirs(data_dir, exist_ok=True)
    date_str = date.strftime("%Y-%m-%d")
    return os.path.join(data_dir, f"{date_str}-{suffix}.json")


def _first_present(d: Dict[str, Any], keys):
    """Return the first present value from a dict for any key in keys."""
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return None


def _normalize_usgs_sites_for_json(usgs_sites):
    """
    Ensure each stored USGS site includes lat/lon if available under alternate keys.

    Common possibilities:
      - lat / lon
      - latitude / longitude
      - dec_lat_va / dec_long_va
      - nested fields from upstream APIs

    This version logs enough detail to identify the real upstream field names.
    """
    normalized = []

    for i, site in enumerate(usgs_sites or []):
        if not isinstance(site, dict):
            logger.warning(
                "USGS site %s is not a dict during JSON normalization: type=%s value=%r",
                i,
                type(site).__name__,
                site,
            )
            normalized.append(site)
            continue

        row = dict(site)

        lat_source = _first_present(
            row, ["lat", "latitude", "dec_lat_va", "site_lat", "y"]
        )
        lon_source = _first_present(
            row, ["lon", "longitude", "dec_long_va", "site_lon", "x"]
        )

        lat = (
            float(v)
            if pd.notna(v := pd.to_numeric(lat_source, errors="coerce"))
            else None
        )
        lon = (
            float(v)
            if pd.notna(v := pd.to_numeric(lon_source, errors="coerce"))
            else None
        )

        row["lat"] = lat
        row["lon"] = lon
        normalized.append(row)

    logger.debug("USGS JSON normalization complete: total=%s", len(normalized))

    return normalized


def _normalize_nwm_reaches_for_json(nwm_reaches):
    """
    Ensure each stored NWM reach includes lat/lon if available under alternate keys.

    Common possibilities:
      - lat / lon
      - latitude / longitude
      - reach_lat / reach_lon
      - y / x
      - nested geometry/location fields

    This version logs enough detail to identify the real upstream field names.
    """
    normalized = []

    for i, reach in enumerate(nwm_reaches or []):
        if not isinstance(reach, dict):
            logger.warning(
                "NWM reach %s is not a dict during JSON normalization: type=%s value=%r",
                i,
                type(reach).__name__,
                reach,
            )
            normalized.append(reach)
            continue

        row = dict(reach)

        lat_source = _first_present(row, ["lat", "latitude", "reach_lat", "y"])
        lon_source = _first_present(row, ["lon", "longitude", "reach_lon", "x"])

        lat = (
            float(v)
            if pd.notna(v := pd.to_numeric(lat_source, errors="coerce"))
            else None
        )
        lon = (
            float(v)
            if pd.notna(v := pd.to_numeric(lon_source, errors="coerce"))
            else None
        )

        row["lat"] = lat
        row["lon"] = lon
        normalized.append(row)

    logger.debug("NWM JSON normalization complete: total=%s", len(normalized))

    return normalized


# ====================== DATA STORAGE ======================


def store_precip_data(data: Dict[str, Any]):
    """Store precipitation analysis results as JSON and CSV."""
    lat = data["lat"]
    lon = data["lon"]
    obs_date = data["obs_date"]

    stations = data.get("local_stations_info", [])
    is_gridded = stations and stations[0].get("id") == "GRIDDED"
    suffix = "Gridded" if is_gridded else "GHCN"

    def series_to_dict(s):
        if s is None or (isinstance(s, pd.Series) and s.empty):
            return {}
        if isinstance(s, pd.Series):
            return {k.strftime("%Y-%m-%d"): float(v) for k, v in s.dropna().items()}
        return s

    payload = {
        "data_type": "precip",
        "daily_precip": series_to_dict(data.get("daily_precip")),
        "rolling_total": series_to_dict(data.get("rolling_total")),
        "normal_low": series_to_dict(data.get("normal_low")),
        "normal_high": series_to_dict(data.get("normal_high")),
        "graph_start": (
            data.get("graph_start").strftime("%Y-%m-%d")
            if data.get("graph_start")
            else None
        ),
        "graph_end": (
            data.get("graph_end").strftime("%Y-%m-%d")
            if data.get("graph_end")
            else None
        ),
        "obs_date": obs_date.strftime("%Y-%m-%d"),
        "lat": lat,
        "lon": lon,
        "elev": float(data.get("elevation", 0.0)),
        "local_stations_info": data.get("local_stations_info"),
        "data_source": data.get("data_source"),
    }

    condition, score = _compute_precip_condition(payload)
    payload["antecedent_score_summary"] = {"total_score": score, "condition": condition}

    if not is_gridded:
        daily_precip_series = data.get("daily_precip")
        if daily_precip_series is not None and not daily_precip_series.empty:
            coord_directory = _coord_str(lat, lon)
            station_data_dir = os.path.join(
                data["output_dir"], coord_directory, "Station Data"
            )
            os.makedirs(station_data_dir, exist_ok=True)
            precip_df = daily_precip_series.to_frame()
            _write_merged_stations_csv(
                precip_df, station_data_dir, obs_date.strftime("%Y-%m-%d")
            )

    path = _get_data_path(data["output_dir"], lat, lon, obs_date, suffix)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4, cls=NpEncoder)
    logger.info(f"Stored precip data: {path}")


def store_pdsi_data(data: Dict[str, Any]):
    """Stores PDSI analysis results as JSON."""
    path = _get_data_path(
        data["output_dir"], data["lat"], data["lon"], data["obs_date"], "PDSI"
    )
    payload = {
        "palmer_value": data.get("palmer_value"),
        "palmer_class": data.get("palmer_class"),
        "palmer_color": data.get("palmer_color"),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=4, cls=NpEncoder)
    logger.info(f"Stored PDSI data: {path}")


def store_usgs_data(data: Dict[str, Any]):
    """Stores USGS analysis results as JSON."""
    path = _get_data_path(
        data["output_dir"], data["lat"], data["lon"], data["obs_date"], "USGS"
    )

    normalized_sites = _normalize_usgs_sites_for_json(data.get("usgs_sites", []))

    payload = {
        "usgs_condition": data.get("usgs_condition"),
        "usgs_sites": normalized_sites,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=4, cls=NpEncoder)

    logger.info(f"Stored USGS data: {path}")


def store_nwm_data(data: Dict[str, Any]):
    """Stores NWM analysis results as JSON."""
    path = _get_data_path(
        data["output_dir"], data["lat"], data["lon"], data["obs_date"], "NWM"
    )

    normalized_reaches = _normalize_nwm_reaches_for_json(data.get("nwm_reaches", []))

    payload = {
        "nwm_condition": data.get("nwm_condition"),
        "nwm_reaches": normalized_reaches,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=4, cls=NpEncoder)

    logger.info(f"Stored NWM data: {path}")


def store_wimp_data(data: Dict[str, Any]):
    """Stores the single-line WIMP conclusion as JSON."""
    path = _get_data_path(
        data["output_dir"], data["lat"], data["lon"], data["obs_date"], "WIMP"
    )
    payload = {
        "wimp_condition": data.get("wimp_condition"),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=4, cls=NpEncoder)
    logger.info(f"Stored WIMP data: {path}")


def store_analysis_data(message: Dict[str, Any]):
    """Route storage to correct handler based on data_type."""
    data_type = message.get("data_type")
    if data_type in ("precip", "ghcn", "gridded"):
        store_precip_data(message)
    elif data_type == "pdsi":
        store_pdsi_data(message)
    elif data_type == "usgs":
        store_usgs_data(message)
    elif data_type == "nwm":
        store_nwm_data(message)
    elif data_type == "wimp":
        store_wimp_data(message)
    else:
        logger.warning(f"Unknown data_type for storage: {data_type}")
