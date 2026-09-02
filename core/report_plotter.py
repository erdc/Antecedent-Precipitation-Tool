# report_plotter.py
"""
Handles the generation of PDF reports from pre-processed data.
This script reads analysis data from JSON files and uses matplotlib to create
plots and tables, which are then assembled into single- or multi-page PDFs.
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

from core.data_handler import _compute_precip_condition, _coord_str, dict_to_series

logger = logging.getLogger(__name__)


# ====================== UTILITY FUNCTIONS ======================


def _load_json_if_exists(path: str) -> Dict:
    """Safely loads a JSON file if it exists, otherwise returns an empty dict."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


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
        condition, _ = _compute_precip_condition(p)
        meta["precip_condition"] = condition
    return meta


# ====================== PDF PLOTTING ======================


def _plot_precip_page(
    precip_data: Dict,
    pdsi_data: Dict,
    usgs_data: Dict,
    nwm_data: Dict,
    wimp_data: Dict,
    pdf: PdfPages,
    data_dir: str = "data",
):
    """Creates the precipitation page for the PDF report."""
    daily_precip = dict_to_series(precip_data.get("daily_precip"))
    rolling_total = dict_to_series(precip_data.get("rolling_total"))
    normal_low = dict_to_series(precip_data.get("normal_low"))
    normal_high = dict_to_series(precip_data.get("normal_high"))

    obs_date = datetime.strptime(precip_data["obs_date"], "%Y-%m-%d")
    graph_end = datetime.strptime(precip_data["graph_end"], "%Y-%m-%d")

    if obs_date.month >= 10:
        current_wy_start = datetime(obs_date.year, 10, 1)
    else:
        current_wy_start = datetime(obs_date.year - 1, 10, 1)
    graph_start = datetime(current_wy_start.year - 1, 10, 1)

    if not daily_precip.empty and graph_start < daily_precip.index[0]:
        graph_start = daily_precip.index[0].to_pydatetime()

    lat = precip_data["lat"]
    lon = precip_data["lon"]
    elev = precip_data.get("elev", 0.0)
    stations = precip_data.get("local_stations_info", [])
    is_gridded = stations and stations[0].get("id") == "GRIDDED"

    # Color definitions
    light_green = (0.5, 0.8, 0.5)
    light_blue = (0.4, 0.5, 0.8)
    light_red = (0.8, 0.5, 0.5)
    light_grey = (0.85, 0.85, 0.85)
    white = (1, 1, 1)

    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"

    fig = plt.figure(figsize=(17, 11), dpi=140)
    fig.set_facecolor("0.77")

    # Define layout
    ax1 = plt.subplot2grid((9, 10), (0, 0), colspan=10, rowspan=6)  # Main graph
    ax2 = plt.subplot2grid((9, 10), (6, 3), colspan=7, rowspan=2)  # Rain table
    ax3 = plt.subplot2grid((9, 10), (6, 0), colspan=3, rowspan=2)  # Description table
    ax4 = plt.subplot2grid((9, 10), (8, 3), colspan=7, rowspan=1)  # Station table

    for ax in [ax2, ax3, ax4]:
        ax.axis("off")

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

    # Main graph (ax1)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.plot(
        daily_precip[graph_start:graph_end].index,
        daily_precip[graph_start:graph_end].values,
        color="black",
        linewidth=1,
        drawstyle="steps-post",
        label="Daily Total",
    )
    ax1.plot(
        rolling_total[graph_start:graph_end].index,
        rolling_total[graph_start:graph_end].values,
        linewidth=1.2,
        label="30-Day Rolling Total",
        color="blue",
    )
    ax1.fill_between(
        normal_low[graph_start:graph_end].index,
        normal_low[graph_start:graph_end].values,
        normal_high[graph_start:graph_end].values,
        color="orange",
        label="30-Year Normal Range",
        alpha=0.5,
    )
    y_max_val = (
        max(
            rolling_total[graph_start:graph_end].max(),
            normal_high[graph_start:graph_end].max(),
        )
        * 1.03
    )
    ax1.set_ylim(ymin=0, ymax=y_max_val)
    ax1.set_xlim([pd.Timestamp(graph_start), pd.Timestamp(graph_end)])
    ax1.legend(loc="upper right")
    ax1.set_ylabel("Rainfall (Inches)", fontsize=20)
    title = (
        "nClimGrid-Daily Data"
        if is_gridded
        else "Daily Global Historical Climatology Network"
    )
    ax1.set_title(
        f"Antecedent Precipitation vs Normal Range based on {title}", fontsize=20
    )

    # Rain table (ax2)
    summary = precip_data.get("antecedent_score_summary", {})
    result, total_score = summary.get("condition"), summary.get("total_score")
    if result is None:  # Fallback for older data
        result, total_score = _compute_precip_condition(precip_data)

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
        obs_val = rolling_total.get(p_date)
        low_val = normal_low.get(p_date)
        high_val = normal_high.get(p_date)

        if any(pd.isna(v) or v is None for v in [obs_val, low_val, high_val]):
            condition, c_val = "Data Missing", 0
        elif obs_val > high_val:
            condition, c_val = "Wet", 3
        elif obs_val < low_val:
            condition, c_val = "Dry", 1
        else:
            condition, c_val = "Normal", 2

        prod = c_val * weight
        rain_table_vals.append(
            [
                p_date.strftime("%Y-%m-%d"),
                f"{low_val:.2f}",
                f"{high_val:.2f}",
                f"{obs_val:.2f}",
                condition,
                c_val,
                weight,
                prod,
            ]
        )
        rain_colors.append([white] * 8)

    final_color = {
        "Drier than Normal": light_red,
        "Normal Conditions": light_green,
        "Wetter than Normal": light_blue,
    }.get(result, white)
    rain_table_vals.append(
        ["Result", "", "", "", "", "", "", f"{result} - {total_score}"]
    )
    rain_colors.append([white] * 7 + [final_color])

    ax2.table(
        cellText=rain_table_vals,
        cellColours=rain_colors,
        colWidths=[0.112, 0.111, 0.111, 0.120, 0.135, 0.114, 0.1, 0.18],
        loc="center",
    ).set_fontsize(10)

    # Description table (ax3)
    desc_vals = [
        ["Coordinates", f"{lat:.4f}, {lon:.4f}"],
        ["Observation Date", obs_date.strftime("%Y-%m-%d")],
        ["Elevation (ft)", f"{elev:.2f}"],
    ]
    desc_colors = [[light_grey, white]] * 3
    if pdsi_data:
        text = f"{pdsi_data['palmer_value']} ({pdsi_data['palmer_class']})"
        desc_vals.append(["Drought Index (PDSI)", text])
        desc_colors.append([light_grey, _as_rgb(pdsi_data["palmer_color"], white)])
    if usgs_data:
        desc_vals.append(["USGS Streamflow", usgs_data["usgs_condition"]])
        desc_colors.append([light_grey, white])
    if nwm_data:
        desc_vals.append(["NWM Streamflow", nwm_data["nwm_condition"]])
        desc_colors.append([light_grey, white])
    if wimp_data:
        desc_vals.append(["WIMP Condition", wimp_data["wimp_condition"]])
        desc_colors.append([light_grey, white])

    ax3.table(
        cellText=desc_vals,
        colWidths=[0.42, 0.54],
        cellColours=desc_colors,
        loc="center left",
    ).set_fontsize(9)

    # Station table (ax4)
    if not is_gridded:
        station_vals = [
            ["Weather Station Name", "Dist (mi)", "Days Normal", "Days Antecedent"]
        ]
        station_colors = [[light_grey] * 4]
        for s in stations[:5]:
            station_vals.append(
                [
                    str(s.get("name", ""))[:28],
                    f"{s.get('distance', 0):.1f}",
                    str(s.get("days_normal", 0)),
                    str(s.get("days_antecedent", 0)),
                ]
            )
            station_colors.append([white] * 4)
        ax4.table(
            cellText=station_vals,
            cellColours=station_colors,
            loc="center",
        ).set_fontsize(10)

    plt.subplots_adjust(
        wspace=0.0, hspace=0.08, left=0.047, bottom=0.08, top=0.968, right=0.99
    )
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_streamflow_page(
    usgs_data: Dict, nwm_data: Dict, meta: Dict, pdf: PdfPages, max_rows: int = 10
):
    """Creates the combined USGS + NWM streamflow page."""
    light_grey = (0.85, 0.85, 0.85)
    white = (1, 1, 1)

    fig, (ax_usgs, ax_nwm, ax_notes) = plt.subplots(
        3,
        1,
        figsize=(17, 11),
        dpi=140,
        gridspec_kw={"height_ratios": [1.35, 1.15, 0.70]},
    )
    fig.set_facecolor("0.77")

    for ax in [ax_usgs, ax_nwm, ax_notes]:
        ax.axis("off")

    # USGS table
    sites = (usgs_data or {}).get("usgs_sites", [])[:max_rows]
    usgs_header = ["Gage Name", "ID", "Dist (mi)", "Flow (cfs)", "%ile", "Condition"]
    usgs_vals = [usgs_header] + [
        [
            s.get("name", "")[:26],
            s.get("gage_id", ""),
            f"{s.get('distance_mi', 0):.1f}",
            f"{s.get('flow_cfs', 0):.1f}",
            f"{s.get('percentile', 0):.0f}",
            s.get("condition", ""),
        ]
        for s in sites
    ]
    ax_usgs.set_title(
        f"USGS Streamflow – {meta.get('usgs_condition') or 'No Data'}",
        fontsize=13,
        pad=8,
    )
    ax_usgs.table(
        cellText=usgs_vals,
        cellColours=[[light_grey] * 6] + [[white] * 6] * len(sites),
        loc="upper center",
    ).set_fontsize(9)

    # NWM table
    reaches = (nwm_data or {}).get("nwm_reaches", [])[:max_rows]
    nwm_header = ["COMID", "Dist (mi)", "Flow (cfs)", "%ile", "Condition"]
    nwm_vals = [nwm_header] + [
        [
            r.get("COMID", ""),
            f"{r.get('distance_mi', 0):.1f}",
            f"{r.get('flow_cfs', 0):.1f}",
            f"{r.get('percentile', 0):.0f}",
            r.get("condition", ""),
        ]
        for r in reaches
    ]
    ax_nwm.set_title(
        f"NWM Streamflow – {meta.get('nwm_condition') or 'No Data'}", fontsize=13, pad=8
    )
    ax_nwm.table(
        cellText=nwm_vals,
        cellColours=[[light_grey] * 5] + [[white] * 5] * len(reaches),
        loc="upper center",
    ).set_fontsize(9)

    # Methods table
    note_vals = [
        ["USGS Source", "NWIS Daily Values (00060)"],
        ["USGS Method", "Same-day percentile rank vs historic record"],
        ["NWM Source", "Analysis-assim + retrospective (1990-2020)"],
        ["NWM Method", "Same-day percentile rank vs 1990-2020"],
    ]
    ax_notes.set_title("Data Sources & Methods", fontsize=13, pad=8)
    ax_notes.table(
        cellText=note_vals, cellColours=[[light_grey, white]] * 4, loc="upper center"
    ).set_fontsize(10)

    plt.subplots_adjust(hspace=0.34, left=0.04, bottom=0.05, top=0.90, right=0.97)
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


# ====================== PDF GENERATION AND MERGING ======================


def generate_daily_pdf(
    lat: float,
    lon: float,
    analysis_date: datetime,
    output_dir: str,
    data_dir: str = "data",
    analysis_types: list = None,
):
    """Generate a single daily PDF report."""
    coord_directory = _coord_str(lat, lon)
    pdf_dir = os.path.join(output_dir, coord_directory)
    data_path = os.path.join(pdf_dir, "data")
    os.makedirs(pdf_dir, exist_ok=True)
    date_str = analysis_date.strftime("%Y-%m-%d")
    pdf_path = os.path.join(pdf_dir, f"{date_str}.pdf")

    # Load data
    pdsi_data = _load_json_if_exists(os.path.join(data_path, f"{date_str}-PDSI.json"))
    usgs_data = _load_json_if_exists(os.path.join(data_path, f"{date_str}-USGS.json"))
    nwm_data = _load_json_if_exists(os.path.join(data_path, f"{date_str}-NWM.json"))
    wimp_data = _load_json_if_exists(os.path.join(data_path, f"{date_str}-WIMP.json"))

    precip_files = [
        f
        for f in glob.glob(os.path.join(data_path, f"{date_str}-*.json"))
        if not any(x in f for x in ["-PDSI", "-USGS", "-NWM", "-WIMP"])
    ]

    if not precip_files and not usgs_data and not nwm_data:
        logger.warning(f"No data files found for {date_str} at {lat}, {lon}")
        return

    with PdfPages(pdf_path) as pdf:
        for jfile in sorted(precip_files):
            with open(jfile) as f:
                precip_data = json.load(f)
            _plot_precip_page(
                precip_data, pdsi_data, usgs_data, nwm_data, wimp_data, pdf, data_dir
            )

        if usgs_data or nwm_data:
            meta = _extract_meta(
                precip_files, usgs_data, nwm_data, lat, lon, analysis_date
            )
            _plot_streamflow_page(usgs_data, nwm_data, meta, pdf)

    logger.info(f"Generated PDF: {pdf_path}")


def merge_daily_pdfs(
    lat: float, lon: float, start_date: datetime, end_date: datetime, output_dir: str
):
    """Merge all daily PDFs for a location into one Batch_Results.pdf."""
    coord_directory = _coord_str(lat, lon)
    pdf_dir = os.path.join(output_dir, coord_directory)

    pdf_files = [
        os.path.join(
            pdf_dir, f"{(start_date + timedelta(days=d)).strftime('%Y-%m-%d')}.pdf"
        )
        for d in range((end_date - start_date).days + 1)
    ]
    pdf_files = [f for f in pdf_files if os.path.exists(f)]

    if not pdf_files:
        logger.warning("No PDFs found to merge in the date range.")
        return

    merger = PdfWriter()
    for pdf in pdf_files:
        merger.append(pdf)

    output_path = os.path.join(pdf_dir, "Batch_Results.pdf")
    merger.write(output_path)
    merger.close()
    logger.info(f"Merged {len(pdf_files)} PDFs into {output_path}")


def merge_huc_batch_pdfs(output_dirs: list, base_output_dir: str, huc_id: str):
    """Merges all PDFs from a HUC analysis into a single report."""
    writer = PdfWriter()
    for folder in output_dirs:
        pdf_files = glob.glob(os.path.join(folder, "*.pdf"))
        for pdf_path in sorted(pdf_files):
            if "HUC" in os.path.basename(
                pdf_path
            ) or "Batch_Results" in os.path.basename(pdf_path):
                continue
            writer.append(pdf_path)

    if len(writer.pages) > 0:
        batch_folder = os.path.join(base_output_dir, f"{huc_id}-batch")
        os.makedirs(batch_folder, exist_ok=True)
        output_path = os.path.join(batch_folder, f"HUC_{huc_id}_Batch_Report.pdf")
        with open(output_path, "wb") as f_out:
            writer.write(f_out)
        logger.info(f"Created consolidated HUC report: {output_path}")


# ====================== HIGH-LEVEL ENTRY POINTS ======================


def generate_pdf(message: Dict[str, Any]):
    """High-level entry point for generating a single PDF."""
    generate_daily_pdf(
        lat=message["lat"],
        lon=message["lon"],
        analysis_date=message["analysis_date"],
        output_dir=message["output_dir"],
        data_dir=message.get("data_dir", "data"),
        analysis_types=message.get("analysis_types"),
    )


def merge_pdfs(message: Dict[str, Any]):
    """High-level entry point for merging PDFs."""
    merge_daily_pdfs(
        lat=message["lat"],
        lon=message["lon"],
        start_date=message["start_date"],
        end_date=message["end_date"],
        output_dir=message["output_dir"],
    )
