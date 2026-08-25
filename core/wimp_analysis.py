# core/wimp_analysis.py
import xarray as xr
import numpy as np
import os
from typing import Dict, Any


def get_wimp_data_from_nc(
    nc_filepath: str, target_lat: float, target_lon: float, target_month_num: int = None
) -> Dict[str, Any]:
    """
    Loads WIMP data from a NetCDF file, finds the closest point, calculates
    seasonal conditions, and returns a dictionary with results.
    """
    if not os.path.exists(nc_filepath):
        return {"status": "ERROR", "message": f"NetCDF file not found: {nc_filepath}"}

    try:
        ds = xr.open_dataset(nc_filepath)
    except Exception as e:
        return {"status": "ERROR", "message": f"Failed to open NetCDF file: {e}"}

    lat_dist = ds.latitude - target_lat
    lon_dist = ds.longitude - target_lon
    distance_sq = lat_dist**2 + lon_dist**2
    closest_station_idx = np.argmin(distance_sq.data)

    station_data = ds.isel(station=closest_station_idx)
    found_lat = station_data.latitude.item()
    found_lon = station_data.longitude.item()

    log_lines = [
        f"Target coordinates: ({target_lat:.4f}, {target_lon:.4f})",
        f"Closest point found: ({found_lat:.4f}, {found_lon:.4f}) at distance ~{np.sqrt(distance_sq.min().item()):.2f} degrees",
    ]

    attrs = ds["condition"].attrs
    code_to_name = {
        val: name.replace("_", " ")
        for val, name in zip(
            attrs.get("flag_values", []), attrs.get("flag_meanings", "").split()
        )
    }
    fill_value = attrs.get("_FillValue", 0)
    code_to_name[fill_value] = "No Data"

    first_condition_code = station_data["condition"].values[0]
    if np.all(station_data["condition"].values == first_condition_code):
        condition_name = code_to_name.get(first_condition_code, "Unknown Condition")
        if condition_name in ["LARGE WATER BODY", "PERMANENT SNOW COVER"]:
            ds.close()
            return {
                "status": "SPECIAL_CONDITION",
                "message": condition_name,
                "selected_month_conclusion": condition_name,
                "full_table_string": "\n".join(log_lines)
                + f"\nConclusion for this point: {condition_name}",
            }

    table_header = [
        "\nTerms:",
        "DIFF is the rainfall and estimated snowmelt minus the adjusted potential evapotranspiration (mm/month).",
        "DST is the estimated change in soil moisture from the end of the previous month to the end of the current month (mm/month).",
        "DEF is the estimated deficit or unmet atmospheric demand for moisture (mm/month).",
        "   ______________________________________ ",
        "  | Mon | DIFF | DST  | DEF | Conclusion |",
        "  |--------------------------------------|",
    ]
    table_rows = []
    selected_season_conclusion = "No Data"

    for month_idx, month_name in enumerate(ds.month.values):
        diff = station_data["val_1"].values[month_idx]
        dst = station_data["val_2"].values[month_idx]
        def_val = station_data["val_3"].values[month_idx]

        season = "Wet"
        if not np.isnan(diff) and diff < 0 and not np.isnan(dst) and dst < 0:
            season = "Dry"
        if not np.isnan(def_val) and def_val > 0:
            season = "Dry"

        diff_str = f"{diff:4.0f}" if not np.isnan(diff) else " " * 4
        dst_str = f"{dst:4.0f}" if not np.isnan(dst) else " " * 4
        def_str = f"{def_val:3.0f}" if not np.isnan(def_val) else " " * 3

        marker = ""
        current_month_num = month_idx + 1
        if target_month_num is not None and current_month_num == target_month_num:
            marker = " <---Selected Month"
            selected_season_conclusion = f"{season} Season"

        table_rows.append(
            f"  | {month_name} | {(diff_str+'    ')[:4]} | {(dst_str+'    ')[:4]} | {(def_str+'   ')[:3]} | {season} Season |{marker}"
        )

    table_footer = ["   -------------------------------------- "]
    full_table_string = "\n".join(log_lines + table_header + table_rows + table_footer)

    ds.close()

    return {
        "status": "OK",
        "selected_month_conclusion": selected_season_conclusion,
        "full_table_string": full_table_string,
        "message": "Analysis complete",
    }
