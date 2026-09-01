# core/plotting.py
"""
Pure functions for data storage and PDF report generation.
No dispatcher, GUI, or message knowledge.
"""

import glob
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter

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


# ====================== DATA STORAGE ======================


def _coord_str(lat: float, lon: float) -> str:
    """Stable directory segment shared by store + PDF paths."""
    return f"{float(lat):.{COORD_DECIMALS}f}_{float(lon):.{COORD_DECIMALS}f}"


def _as_rgb(c, default=(1.0, 1.0, 1.0)):
    """Force a matplotlib-safe RGB(A) tuple of length 3 or 4."""
    if c is None:
        return default
    try:
        if isinstance(c, (list, tuple)):
            t = tuple(float(x) for x in c)
        else:
            t = tuple(c) if hasattr(c, "__iter__") else default
        if len(t) in (3, 4) and all(0.0 <= x <= 1.0 for x in t[:3]):
            return t
    except (TypeError, ValueError):
        pass
    return default


def _get_data_path(
    output_dir: str, lat: float, lon: float, date: datetime, suffix: str
) -> str:
    coord_str = _coord_str(lat, lon)
    data_dir = os.path.join(output_dir, coord_str, "data")
    os.makedirs(data_dir, exist_ok=True)
    date_str = date.strftime("%Y-%m-%d")
    return os.path.join(data_dir, f"{date_str}-{suffix}.json")


def store_precip_data(data: Dict[str, Any]):
    """Store precipitation analysis results as JSON."""
    lat = data["lat"]
    lon = data["lon"]
    obs_date = data["obs_date"]

    # Determine suffix
    stations = data.get("local_stations_info", [])
    suffix = "Gridded" if (stations and stations[0].get("id") == "GRIDDED") else "GHCN"

    path = _get_data_path(data["output_dir"], lat, lon, obs_date, suffix)

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

    with open(path, "w") as f:
        json.dump(payload, f, indent=4, cls=NpEncoder)

    logger.info(f"Stored precip data: {path}")


def store_pdsi_data(data: Dict[str, Any]):
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
    path = _get_data_path(
        data["output_dir"], data["lat"], data["lon"], data["obs_date"], "USGS"
    )
    payload = {
        "usgs_condition": data.get("usgs_condition"),
        "usgs_sites": data.get("usgs_sites", []),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=4, cls=NpEncoder)
    logger.info(f"Stored USGS data: {path}")


def store_nwm_data(data: Dict[str, Any]):
    path = _get_data_path(
        data["output_dir"], data["lat"], data["lon"], data["obs_date"], "NWM"
    )
    payload = {
        "nwm_condition": data.get("nwm_condition"),
        "nwm_reaches": data.get("nwm_reaches", []),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=4, cls=NpEncoder)
    logger.info(f"Stored NWM data: {path}")


def store_wimp_data(data: Dict[str, Any]):
    """Stores the single-line WIMP conclusion."""
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


# ====================== PDF GENERATION ======================


def _load_json_if_exists(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def generate_daily_pdf(
    lat: float,
    lon: float,
    analysis_date: datetime,
    output_dir: str,
    data_dir: str = "data",
):
    """Generate a single daily PDF report. Pages in order: precip → streamflow."""
    coord_str = _coord_str(lat, lon)
    pdf_dir = os.path.join(output_dir, coord_str)
    data_path = os.path.join(pdf_dir, "data")  # per-point stored JSON
    os.makedirs(pdf_dir, exist_ok=True)

    date_str = analysis_date.strftime("%Y-%m-%d")
    pdf_path = os.path.join(pdf_dir, f"{date_str}.pdf")

    pdsi_data = _load_json_if_exists(os.path.join(data_path, f"{date_str}-PDSI.json"))
    usgs_data = _load_json_if_exists(os.path.join(data_path, f"{date_str}-USGS.json"))
    nwm_data = _load_json_if_exists(os.path.join(data_path, f"{date_str}-NWM.json"))
    wimp_data = _load_json_if_exists(os.path.join(data_path, f"{date_str}-WIMP.json"))

    search_pattern = os.path.join(data_path, f"{date_str}-*.json")
    json_files = glob.glob(search_pattern)
    precip_files = [
        f
        for f in json_files
        if not any(
            x in f for x in ["-PDSI.json", "-USGS.json", "-NWM.json", "-WIMP.json"]
        )
    ]

    if not precip_files and not usgs_data and not nwm_data and not wimp_data:
        logger.warning(f"No data files found for {date_str}")
        return

    try:
        with PdfPages(pdf_path) as pdf:
            # 1. Precip page(s)
            for jfile in sorted(precip_files):
                with open(jfile) as f:
                    precip_data = json.load(f)
                _plot_precip_page(
                    precip_data,
                    pdsi_data,
                    usgs_data,
                    nwm_data,
                    wimp_data,
                    pdf,
                    data_dir=data_dir,
                )

            # 2. Combined USGS + NWM page (if either exists)
            if usgs_data or nwm_data:
                meta = _extract_meta(
                    precip_files, usgs_data, nwm_data, lat, lon, analysis_date
                )
                _plot_streamflow_page(usgs_data, nwm_data, pdsi_data, meta, pdf)

        logger.info(f"Generated PDF: {pdf_path}")
    except Exception as e:
        logger.error(f"Failed to generate PDF for {date_str}", exc_info=True)


def _compute_precip_condition(precip_data: Dict) -> str:
    """Recompute the wetness result string from stored series (same logic as plot)."""

    def dict_to_series(d):
        if not d:
            return pd.Series(dtype=float)
        s = pd.Series(d)
        s.index = pd.to_datetime(s.index)
        return s

    rolling_total = dict_to_series(precip_data.get("rolling_total"))
    normal_low = dict_to_series(precip_data.get("normal_low"))
    normal_high = dict_to_series(precip_data.get("normal_high"))
    obs_date = datetime.strptime(precip_data["obs_date"], "%Y-%m-%d")

    total_score = 0
    for days_prior, weight in [(0, 3), (30, 2), (60, 1)]:
        p_date = obs_date - timedelta(days=days_prior)
        date_str_val = p_date.strftime("%Y-%m-%d")
        try:
            obs_val = rolling_total.get(date_str_val)
            low_val = normal_low.get(date_str_val)
            high_val = normal_high.get(date_str_val)
            if any(pd.isna(v) for v in [obs_val, low_val, high_val]):
                c_val = 0
            elif obs_val > high_val:
                c_val = 3
            elif obs_val < low_val:
                c_val = 1
            else:
                c_val = 2
            total_score += c_val * weight
        except (KeyError, TypeError):
            pass

    if total_score < 10:
        return f"Drier than Normal ({total_score})"
    if total_score <= 14:
        return f"Normal Conditions ({total_score})"
    return f"Wetter than Normal ({total_score})"


def _extract_meta(precip_files, usgs_data, nwm_data, lat, lon, analysis_date):
    """Pull common metadata for the streamflow page."""
    meta = {
        "lat": lat,
        "lon": lon,
        "elev": 0.0,
        "obs_date": analysis_date,
        "precip_condition": None,
        "usgs_condition": usgs_data.get("usgs_condition"),
        "nwm_condition": nwm_data.get("nwm_condition"),
    }
    if precip_files:
        with open(precip_files[0]) as f:
            p = json.load(f)
        meta["lat"] = p.get("lat", lat)
        meta["lon"] = p.get("lon", lon)
        meta["elev"] = p.get("elev", 0.0)
        meta["obs_date"] = datetime.strptime(p["obs_date"], "%Y-%m-%d")
        meta["precip_condition"] = _compute_precip_condition(p)
    return meta


def _plot_precip_page(
    precip_data: Dict,
    pdsi_data: Dict,
    usgs_data: Dict,
    nwm_data: Dict,
    wimp_data: Dict,
    pdf,
    data_dir: str = "data",
):
    """Original precip page layout (graph + rain table + stations + description)."""

    def dict_to_series(d):
        if not d:
            return pd.Series(dtype=float)
        s = pd.Series(d)
        s.index = pd.to_datetime(s.index)
        return s

    daily_precip = dict_to_series(precip_data.get("daily_precip"))
    rolling_total = dict_to_series(precip_data.get("rolling_total"))
    normal_low = dict_to_series(precip_data.get("normal_low"))
    normal_high = dict_to_series(precip_data.get("normal_high"))

    obs_date = datetime.strptime(precip_data["obs_date"], "%Y-%m-%d")
    graph_end = datetime.strptime(precip_data["graph_end"], "%Y-%m-%d")

    # Calculate the start of the full preceding water year.
    # A water year runs from Oct 1 to Sept 30 and is numbered by the year it ends.
    if obs_date.month >= 10:
        current_wy_start = datetime(obs_date.year, 10, 1)
    else:
        current_wy_start = datetime(obs_date.year - 1, 10, 1)

    graph_start = datetime(current_wy_start.year - 1, 10, 1)

    # Ensure start date is within the bounds of our data
    if not daily_precip.empty and graph_start < daily_precip.index[0]:
        graph_start = daily_precip.index[0]

    lat = precip_data["lat"]
    lon = precip_data["lon"]
    elev = precip_data.get("elev", 0.0)
    stations = precip_data.get("local_stations_info", [])
    is_gridded = stations and stations[0].get("id") == "GRIDDED"

    light_green = (0.5, 0.8, 0.5)
    light_blue = (0.4, 0.5, 0.8)
    light_red = (0.8, 0.5, 0.5)
    light_grey = (0.85, 0.85, 0.85)
    white = (1, 1, 1)

    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"

    fig = plt.figure(figsize=(17, 11), dpi=140)
    fig.set_facecolor("0.77")

    ax1 = plt.subplot2grid((9, 10), (0, 0), colspan=10, rowspan=6)
    ax2 = plt.subplot2grid((9, 10), (6, 3), colspan=7, rowspan=2)
    ax3 = plt.subplot2grid((9, 10), (6, 0), colspan=3, rowspan=2)
    ax4 = plt.subplot2grid((9, 10), (8, 3), colspan=7, rowspan=1)

    for ax in [ax2, ax3, ax4]:
        ax.axis("off")
        ax.axis("tight")

    # Logo – use the data_dir passed from the caller (GUI / dispatcher)
    try:
        logo_file = os.path.join(data_dir, "RD_3_9.png")
        logo = plt.imread(logo_file)

        # Scale factor
        scale = 1.4

        # simple nearest-neighbor upscale with numpy
        if scale != 1.0:
            h, w = logo.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            y_idx = (np.arange(new_h) / scale).astype(int)
            x_idx = (np.arange(new_w) / scale).astype(int)
            logo = logo[y_idx][:, x_idx]

        fig.figimage(X=logo, xo=150, yo=16)
    except Exception as e:
        logger.warning(f"Could not load logo from {logo_file}: {e}")

    # ----- Main graph -----
    # Using major ticks for months and showing tick marks
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    ax1.xaxis.set_minor_formatter(ticker.NullFormatter())

    # Tick mark styling updated to visibly show the monthly tick marks
    ax1.tick_params(axis="x", which="major", bottom=True, length=5, labelsize=10)
    ax1.tick_params(axis="x", which="minor", bottom=False)

    for label in ax1.get_xticklabels(which="major"):
        label.set_horizontalalignment("center")

    plot_daily = daily_precip[graph_start:graph_end]
    plot_rolling = rolling_total[graph_start:graph_end]

    plot_low = normal_low[graph_start:graph_end]
    plot_high = normal_high[graph_start:graph_end]

    ax1.plot(
        plot_daily.index,
        plot_daily.values,
        color="black",
        linewidth=1,
        drawstyle="steps-post",
        label="Daily Total",
    )
    ax1.plot(
        plot_rolling.index,
        plot_rolling.values,
        linewidth=1.2,
        label="30-Day Rolling Total",
        color="blue",
    )
    ax1.fill_between(
        plot_low.index,
        plot_low.values,
        plot_high.values,
        color="orange",
        label="30-Year Normal Range",
        alpha=0.5,
    )

    rolling_max = plot_rolling.max() if not plot_rolling.empty else 0
    normal_max = plot_high.max() if not plot_high.empty else 0
    y_max_val = (
        max(rolling_max, normal_max) * 1.03 if (rolling_max or normal_max) else 1.0
    )
    ax1.set_ylim(ymin=0, ymax=y_max_val)
    ax1.set_xlim([pd.Timestamp(graph_start), pd.Timestamp(graph_end)])

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="upper right")
    ax1.set_ylabel("Rainfall (Inches)", fontsize=20)

    if is_gridded:
        ax1.set_title(
            "Antecedent Precipitation vs Normal Range based on NOAA's nClimGrid-Daily Precipitation Data",
            fontsize=20,
        )
    else:
        ax1.set_title(
            "Antecedent Precipitation vs Normal Range based on NOAA's Daily Global Historical Climatology Network",
            fontsize=20,
        )

    # ----- Rain condition table (ax2) -----
    total_score = 0
    rain_table_vals = [
        [
            "30 Days Ending",
            r"30$^{th}$ %ile (in)",
            r"70$^{th}$ %ile (in)",
            "Observed (in)",
            "Wetness Condition",
            "Condition Value",
            "Month Weight",
            "Product",
        ]
    ]
    rain_colors = [[light_grey] * 8]

    for days_prior, weight in [(0, 3), (30, 2), (60, 1)]:
        p_date = obs_date - timedelta(days=days_prior)
        date_str_val = p_date.strftime("%Y-%m-%d")
        obs_val, low_val, high_val = None, None, None
        condition, c_val, prod = "Error", 0, 0

        try:
            obs_val = rolling_total.get(date_str_val)
            low_val = normal_low.get(date_str_val)
            high_val = normal_high.get(date_str_val)

            if any(pd.isna(v) for v in [obs_val, low_val, high_val]):
                condition, c_val = "Data Missing", 0
            else:
                if obs_val > high_val:
                    condition, c_val = "Wet", 3
                elif obs_val < low_val:
                    condition, c_val = "Dry", 1
                else:
                    condition, c_val = "Normal", 2

                offset = (10, -25) if obs_val > (rolling_max * 0.85) else (15, 30)
                ax1.annotate(
                    date_str_val,
                    xy=(pd.Timestamp(date_str_val), obs_val),
                    xycoords="data",
                    xytext=offset,
                    textcoords="offset points",
                    size=13,
                    arrowprops=dict(
                        arrowstyle="simple",
                        fc="0.4",
                        ec="none",
                        connectionstyle="arc3,rad=0.5",
                    ),
                )

            prod = c_val * weight
            total_score += prod
            rain_table_vals.append(
                [
                    date_str_val,
                    f"{low_val:.2f}" if low_val is not None else "N/A",
                    f"{high_val:.2f}" if high_val is not None else "N/A",
                    f"{obs_val:.2f}" if obs_val is not None else "N/A",
                    condition,
                    c_val,
                    weight,
                    prod,
                ]
            )
        except (KeyError, TypeError):
            rain_table_vals.append(
                [date_str_val, "N/A", "N/A", "N/A", "Error", 0, weight, 0]
            )
        rain_colors.append([white] * 8)

    if total_score < 10:
        result = "Drier than Normal"
        final_cell_color = light_red
    elif 10 <= total_score <= 14:
        result = "Normal Conditions"
        final_cell_color = light_green
    else:
        result = "Wetter than Normal"
        final_cell_color = light_blue

    rain_table_vals.append(
        ["Result", "", "", "", "", "", "", f"{result} - {total_score}"]
    )
    rain_colors.append([white] * 7 + [final_cell_color])

    the_table = ax2.table(
        cellText=rain_table_vals,
        cellColours=rain_colors,
        colWidths=[0.112, 0.111, 0.111, 0.120, 0.135, 0.114, 0.1, 0.18],
        loc="center",
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    # ----- Description table (ax3) – shortened streamflow one-liners -----
    desc_vals = [
        ["Coordinates", f"{lat:.4f}, {lon:.4f}"],
        ["Observation Date", obs_date.strftime("%Y-%m-%d")],
        ["Elevation (ft)", f"{elev:.2f}"],
    ]
    desc_colors = [
        [light_grey, white],
        [light_grey, white],
        [light_grey, white],
    ]

    palmer_value = pdsi_data.get("palmer_value")
    palmer_class = pdsi_data.get("palmer_class")
    palmer_color = pdsi_data.get("palmer_color")
    if palmer_value is not None and palmer_class is not None:
        display_text = (
            "Not Available"
            if palmer_value == -99.99
            else f"{palmer_value} ({palmer_class})"
        )
        desc_vals.append(["Drought Index (PDSI)", display_text])
        color_tuple = _as_rgb(palmer_color, white)
        desc_colors.append([light_grey, color_tuple])

    usgs_condition = usgs_data.get("usgs_condition")
    if usgs_condition is not None:
        desc_vals.append(["USGS Streamflow", usgs_condition])
        desc_colors.append([light_grey, white])

    nwm_condition = nwm_data.get("nwm_condition")
    if nwm_condition is not None:
        desc_vals.append(["NWM Streamflow", nwm_condition])
        desc_colors.append([light_grey, white])

    wimp_condition = wimp_data.get("wimp_condition")
    if wimp_condition is not None:
        desc_vals.append(["WIMP Condition", wimp_condition])
        desc_colors.append([light_grey, white])

    description_table = ax3.table(
        cellText=desc_vals,
        colWidths=[0.42, 0.54],
        cellColours=desc_colors,
        loc="center left",
    )
    description_table.auto_set_font_size(False)
    description_table.set_fontsize(9)

    # ----- Bottom station table (ax4) -----
    if is_gridded:
        station_table_vals = [
            ["", "Station Count Summary"],
            ["nClimGrid-Daily Data Used", "Yes"],
        ]
        station_colors = [[light_grey, light_grey], [white, white]]

        stations_table = ax4.table(
            cellText=station_table_vals,
            cellColours=station_colors,
            colWidths=[0.4, 0.3],
            loc="center",
        )
    else:
        station_table_vals = [
            [
                "Weather Station Name",
                "Coordinates",
                "Elevation (ft)",
                "Distance (mi)",
                r"Elevation $\Delta$",
                r"Weighted $\Delta$",
                "Days Normal",
                "Days Antecedent",
            ]
        ]
        station_colors = [[light_grey] * 8]

        for s in stations[:5]:
            s_lat = s.get("latitude", 0.0)
            s_lon = s.get("longitude", 0.0)
            s_elev_m = s.get("elevation", 0.0)
            s_elev_ft = s_elev_m * 3.28084 if pd.notnull(s_elev_m) else 0.0
            e_diff = abs(elev - s_elev_ft) if elev else 0.0
            days_norm = s.get("days_normal", 0)
            days_ante = s.get("days_antecedent", 0)

            station_table_vals.append(
                [
                    s.get("name", "")[:28],
                    f"{s_lat:.4f}, {s_lon:.4f}",
                    f"{s_elev_ft:.1f}",
                    f"{s.get('distance', 0):.1f}",
                    f"{e_diff:.1f}",
                    f"{s.get('weighted_diff', 0):.1f}",
                    str(days_norm),
                    str(days_ante),
                ]
            )
            station_colors.append([white] * 8)

        stations_table = ax4.table(
            cellText=station_table_vals,
            cellColours=station_colors,
            colWidths=[0.25, 0.15, 0.095, 0.097, 0.087, 0.087, 0.104, 0.132],
            loc="center",
        )
    stations_table.auto_set_font_size(False)
    stations_table.set_fontsize(10)

    plt.subplots_adjust(
        wspace=0.00, hspace=0.08, left=0.047, bottom=0.08, top=0.968, right=0.99
    )
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_streamflow_page(
    usgs_data: Dict,
    nwm_data: Dict,
    pdsi_data: Dict,
    meta: Dict,
    pdf,
    max_rows: int = 10,
):
    """
    Combined USGS + NWM page.

    Vertical layout:
      - USGS table (top)
      - NWM table (middle)
      - Methods / sources (bottom)

    Args:
        usgs_data: Stored USGS analysis payload.
        nwm_data: Stored NWM analysis payload.
        pdsi_data: Stored PDSI payload (currently unused here).
        meta: Metadata used for titles/labels.
        pdf: PdfPages handle.
        max_rows: Maximum number of USGS gages / NWM reaches to show per table.
    """
    light_grey = (0.85, 0.85, 0.85)
    white = (1, 1, 1)

    fig = plt.figure(figsize=(17, 11), dpi=140)
    fig.set_facecolor("0.77")

    # Vertical layout: USGS top, NWM middle, Notes bottom
    gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 1.15, 0.70])

    ax_usgs = fig.add_subplot(gs[0, 0])
    ax_nwm = fig.add_subplot(gs[1, 0])
    ax_notes = fig.add_subplot(gs[2, 0])

    for ax in [ax_usgs, ax_nwm, ax_notes]:
        ax.axis("off")
        ax.axis("tight")

    # Normalize max_rows
    max_rows = max(1, int(max_rows))

    # ----- USGS table (top) -----
    sites = (usgs_data or {}).get("usgs_sites") or []
    usgs_header = [
        "Gage Name",
        "ID",
        "GagesII",
        "Dist (mi)",
        "Flow (cfs)",
        "%ile",
        "Condition",
    ]
    usgs_vals = [usgs_header]
    usgs_colors = [[light_grey] * len(usgs_header)]

    for s in sites[:max_rows]:
        usgs_vals.append(
            [
                str(s.get("name", ""))[:26],
                str(s.get("gage_id", "")),
                "" if s.get("gagesii") is None else str(s.get("gagesii")),
                f"{s.get('distance_mi', 0):.1f}",
                f"{s.get('flow_cfs', 0):.1f}",
                f"{s.get('percentile', 0):.0f}",
                str(s.get("condition", "")),
            ]
        )
        usgs_colors.append([white] * len(usgs_header))

    if len(usgs_vals) == 1:
        usgs_vals.append(["No valid USGS gages", "", "", "", "", "", ""])
        usgs_colors.append([white] * len(usgs_header))

    t_usgs = ax_usgs.table(
        cellText=usgs_vals,
        cellColours=usgs_colors,
        colWidths=[0.26, 0.11, 0.11, 0.10, 0.12, 0.08, 0.18],
        loc="upper center",
    )
    t_usgs.auto_set_font_size(False)
    t_usgs.set_fontsize(9)
    t_usgs.scale(1, 1.35)

    ax_usgs.set_title(
        f"USGS Streamflow  –  {meta.get('usgs_condition') or 'No Data'}",
        fontsize=13,
        pad=8,
    )

    # ----- NWM table (middle) -----
    reaches = (nwm_data or {}).get("nwm_reaches") or []
    nwm_header = ["COMID", "Dist (mi)", "Flow (cfs)", "%ile", "Condition"]
    nwm_vals = [nwm_header]
    nwm_colors = [[light_grey] * len(nwm_header)]

    for r in reaches[:max_rows]:
        nwm_vals.append(
            [
                str(r.get("COMID", "")),
                f"{r.get('distance_mi', 0):.1f}",
                f"{r.get('flow_cfs', 0):.1f}",
                f"{r.get('percentile', 0):.0f}",
                str(r.get("condition", "")),
            ]
        )
        nwm_colors.append([white] * len(nwm_header))

    if len(nwm_vals) == 1:
        nwm_vals.append(["No valid NWM reaches", "", "", "", ""])
        nwm_colors.append([white] * len(nwm_header))

    t_nwm = ax_nwm.table(
        cellText=nwm_vals,
        cellColours=nwm_colors,
        colWidths=[0.18, 0.16, 0.18, 0.12, 0.24],
        loc="upper center",
    )
    t_nwm.auto_set_font_size(False)
    t_nwm.set_fontsize(9)
    t_nwm.scale(1, 1.30)

    ax_nwm.set_title(
        f"NWM Streamflow  –  {meta.get('nwm_condition') or 'No Data'}",
        fontsize=13,
        pad=8,
    )

    # ----- Methods / Sources table (bottom) -----
    note_vals = [
        ["USGS Source", "NWIS Daily Values (00060)"],
        ["USGS Method", "Same-day percentile rank vs historic record"],
        [
            "USGS GagesII",
            "Ref / Non-Ref values pulled from gage_data.csv when available",
        ],
        ["NWM Source", "Analysis-assim + retrospective (1990-2020)"],
        ["NWM Method", "Same-day percentile rank vs 1990-2020"],
    ]
    note_colors = [[light_grey, white] for _ in note_vals]

    t_notes = ax_notes.table(
        cellText=note_vals,
        cellColours=note_colors,
        colWidths=[0.24, 0.72],
        loc="upper center",
    )
    t_notes.auto_set_font_size(False)
    t_notes.set_fontsize(10)
    t_notes.scale(1, 1.20)

    ax_notes.set_title("Data Sources & Methods", fontsize=13, pad=8)

    plt.subplots_adjust(
        hspace=0.34,
        left=0.04,
        bottom=0.05,
        top=0.90,
        right=0.97,
    )
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


# ====================== BATCH MERGING ======================


def merge_daily_pdfs(
    lat: float, lon: float, start_date: datetime, end_date: datetime, output_dir: str
):
    """Merge all daily PDFs into one Batch_Results.pdf."""
    coord_str = _coord_str(lat, lon)
    pdf_dir = os.path.join(output_dir, coord_str)

    pdf_files = []
    current = start_date
    while current <= end_date:
        path = os.path.join(pdf_dir, f"{current.strftime('%Y-%m-%d')}.pdf")
        if os.path.exists(path):
            pdf_files.append(path)
        current += timedelta(days=1)

    if not pdf_files:
        logger.warning("No PDFs found to merge.")
        return

    merger = PdfWriter()
    for pdf in pdf_files:
        merger.append(pdf)

    output_path = os.path.join(pdf_dir, "Batch_Results.pdf")
    try:
        merger.write(output_path)
        merger.close()
        logger.info(f"Merged {len(pdf_files)} PDFs into {output_path}")
    except Exception as e:
        logger.error("PDF merge failed", exc_info=True)


# ====================== HIGH-LEVEL ENTRY POINTS ======================


def generate_pdf(message: Dict[str, Any]):
    """Entry point called by adapter."""
    generate_daily_pdf(
        lat=message["lat"],
        lon=message["lon"],
        analysis_date=message["analysis_date"],
        output_dir=message["output_dir"],
        data_dir=message.get("data_dir", "data"),
    )


def merge_pdfs(message: Dict[str, Any]):
    """Entry point for batch merge."""
    merge_daily_pdfs(
        lat=message["lat"],
        lon=message["lon"],
        start_date=message["start_date"],
        end_date=message["end_date"],
        output_dir=message["output_dir"],
    )


def merge_huc_batch_pdfs(output_dirs: list, base_output_dir: str, huc_id: str):
    """
    Collects generated PDFs from each sampled point directory and merges them
    into a single consolidated HUC Batch report.
    """
    writer = PdfWriter()
    pdfs_merged = 0

    logging.info(
        f"Starting PDF merge for HUC {huc_id} across {len(output_dirs)} directories."
    )

    for folder in output_dirs:
        if not os.path.exists(folder):
            logging.warning(
                f"Expected output directory does not exist, skipping: {folder}"
            )
            continue

        # Prefer dated daily PDFs; fall back to any PDF if none match
        pdf_files = sorted(glob.glob(os.path.join(folder, "????-??-??.pdf")))
        if not pdf_files:
            pdf_files = sorted(
                f
                for f in glob.glob(os.path.join(folder, "*.pdf"))
                if not os.path.basename(f).startswith("HUC_")
                and os.path.basename(f) != "Batch_Results.pdf"
            )

        for pdf_path in pdf_files:
            try:
                logging.info(f"Appending PDF to batch: {pdf_path}")
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    writer.add_page(page)
                pdfs_merged += 1
            except Exception as e:
                logging.error(f"Failed to read/append PDF {pdf_path}: {e}")

    if pdfs_merged > 0:
        batch_folder = os.path.join(base_output_dir, f"{huc_id}-batch")
        os.makedirs(batch_folder, exist_ok=True)
        merged_output_path = os.path.join(
            batch_folder, f"HUC_{huc_id}_Batch_Report.pdf"
        )
        try:
            with open(merged_output_path, "wb") as f_out:
                writer.write(f_out)
            logging.info(
                f"Successfully created consolidated HUC report: {merged_output_path}"
            )
            return merged_output_path
        except Exception as e:
            logging.error(f"Failed to write merged PDF output: {e}")
    else:
        logging.error("No valid PDF files were found to merge.")

    return None
