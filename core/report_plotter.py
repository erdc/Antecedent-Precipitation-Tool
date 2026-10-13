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
import io
from datetime import datetime, timedelta
from typing import Any, Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfWriter

from core.data_handler import _compute_precip_condition, _coord_str, dict_to_series

logger = logging.getLogger(__name__)
_CONDITION_COLORS = {
    "Drier than Normal": (0.8, 0.5, 0.5),
    "Normal Conditions": (0.5, 0.8, 0.5),
    "Wetter than Normal": (0.4, 0.5, 0.8),
}
_LIGHT_GREY = (0.85, 0.85, 0.85)
_WHITE = (1.0, 1.0, 1.0)
_PAGE_BG = (0.77, 0.77, 0.77)

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


def _condition_from_score(score):
    """Map antecedent total_score to label + RGB, matching precip page logic."""
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "No Data", _WHITE
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "No Data", _WHITE
    if score < 10:
        return "Drier than Normal", _CONDITION_COLORS["Drier than Normal"]
    if score <= 14:
        return "Normal Conditions", _CONDITION_COLORS["Normal Conditions"]
    return "Wetter than Normal", _CONDITION_COLORS["Wetter than Normal"]


def _load_precip_summary(json_path: str) -> Dict[str, Any]:
    """Pull score/condition/coords/date from one precip JSON (GHCN or Gridded)."""
    with open(json_path) as f:
        payload = json.load(f)

    summary = payload.get("antecedent_score_summary") or {}
    score = summary.get("total_score")
    condition = summary.get("condition")
    if condition is None:
        condition, _ = _condition_from_score(score)
        if score is None:
            # last-resort recompute from series if present
            try:
                condition, score = _compute_precip_condition(payload)
            except Exception:
                condition, score = "No Data", None

    stations = payload.get("local_stations_info") or []
    is_gridded = bool(stations and stations[0].get("id") == "GRIDDED")
    source = "Gridded" if is_gridded else "GHCN"
    if "-Gridded" in os.path.basename(json_path):
        source = "Gridded"
    elif "-GHCN" in os.path.basename(json_path):
        source = "GHCN"

    return {
        "lat": payload.get("lat"),
        "lon": payload.get("lon"),
        "obs_date": payload.get("obs_date"),
        "score": score,
        "condition": condition,
        "source": source,
        "elev": payload.get("elev"),
        "path": json_path,
    }


def _collect_summaries_from_folder(
    folder: str,
    want_precip: bool = True,
    obs_date: str = None,
) -> list:
    """Precip-point summaries under folder/data.

    If ``obs_date`` is given (YYYY-MM-DD), only that day is collected.
    """
    if not want_precip:
        return []
    data_path = os.path.join(folder, "data")
    if not os.path.isdir(data_path):
        return []

    rows = []
    # Prefer GHCN over Gridded when both exist for the same date
    by_date = {}
    for f in glob.glob(os.path.join(data_path, "????-??-??-*.json")):
        name = os.path.basename(f)
        if any(tag in name for tag in ("-PDSI", "-USGS", "-NWM", "-WIMP")):
            continue
        date_part = name[:10]
        try:
            datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError:
            continue
        if obs_date is not None and date_part != obs_date:
            continue
        is_ghcn = "-GHCN" in name
        is_gridded = "-Gridded" in name
        prev = by_date.get(date_part)
        if prev is None:
            by_date[date_part] = f
        elif is_ghcn:
            by_date[date_part] = f
        elif is_gridded and "-GHCN" not in os.path.basename(prev):
            by_date[date_part] = f

    for date_part in sorted(by_date):
        try:
            rows.append(_load_precip_summary(by_date[date_part]))
        except Exception as e:
            logger.warning(
                "Failed to load precip summary %s: %s", by_date[date_part], e
            )
    return rows


def _aggregate_batch_stats(rows: list) -> Dict[str, Any]:
    """Average score, preliminary determination, pie slices, breakdown table."""
    scores = []
    condition_counts = {
        "Drier than Normal": 0,
        "Normal Conditions": 0,
        "Wetter than Normal": 0,
    }
    table_vals = [["Lat", "Lon", "Date", "Score", "Condition", "Source"]]
    table_colors = [[_LIGHT_GREY] * 6]

    for r in rows:
        score = r.get("score")
        condition = r.get("condition") or "No Data"
        if score is not None:
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                pass
        if condition in condition_counts:
            condition_counts[condition] += 1

        color = _CONDITION_COLORS.get(condition, _WHITE)
        table_vals.append(
            [
                f"{r.get('lat'):.4f}" if r.get("lat") is not None else "—",
                f"{r.get('lon'):.4f}" if r.get("lon") is not None else "—",
                str(r.get("obs_date") or "—"),
                f"{float(score):.0f}" if score is not None else "—",
                condition,
                r.get("source") or "—",
            ]
        )
        table_colors.append([_WHITE, _WHITE, _WHITE, _WHITE, color, _WHITE])

    avg_score = float(np.mean(scores)) if scores else None
    prelim, prelim_color = _condition_from_score(avg_score)

    pie_labels, pie_sizes, pie_colors = [], [], []
    for label in ("Drier than Normal", "Normal Conditions", "Wetter than Normal"):
        n = condition_counts[label]
        if n > 0:
            pie_labels.append(f"{label} ({n})")
            pie_sizes.append(n)
            pie_colors.append(_CONDITION_COLORS[label])

    return {
        "avg_score": avg_score,
        "prelim": prelim,
        "prelim_color": prelim_color,
        "pie_labels": pie_labels,
        "pie_sizes": pie_sizes,
        "pie_colors": pie_colors,
        "table_vals": table_vals,
        "table_colors": table_colors,
        "n_points": len(rows),
        "n_scored": len(scores),
    }


# ====================== PDF PLOTTING ======================


def _plot_precip_page(
    precip_data: Dict,
    pdsi_data: Dict,
    usgs_data: Dict,
    nwm_data: Dict,
    wimp_data: Dict,
    pdf: PdfPages,
    data_dir: str = "data",
    debug_behavior: bool = False,
):
    """Creates the precipitation page for the PDF report."""
    if not debug_behavior:
        usgs_data = None
        nwm_data = None

    daily_precip = dict_to_series(precip_data.get("daily_precip"))
    rolling_total = dict_to_series(precip_data.get("rolling_total"))
    normal_low = dict_to_series(precip_data.get("normal_low"))
    normal_high = dict_to_series(precip_data.get("normal_high"))

    obs_date = datetime.strptime(precip_data["obs_date"], "%Y-%m-%d")
    stored_graph_end = datetime.strptime(precip_data["graph_end"], "%Y-%m-%d")

    if not debug_behavior:
        if obs_date.month >= 10:
            current_wy_start = datetime(obs_date.year, 10, 1)
        else:
            current_wy_start = datetime(obs_date.year - 1, 10, 1)
        graph_start = datetime(current_wy_start.year - 1, 10, 1)
        graph_end = stored_graph_end
    else:
        graph_start = (
            pd.Timestamp(obs_date) - pd.DateOffset(months=3, days=15)
        ).to_pydatetime()
        graph_end = (pd.Timestamp(obs_date) + pd.DateOffset(days=5)).to_pydatetime()

    if not daily_precip.empty and graph_start < daily_precip.index[0]:
        graph_start = daily_precip.index[0].to_pydatetime()

    lat = precip_data["lat"]
    lon = precip_data["lon"]
    elev = precip_data.get("elev", 0.0)
    stations = precip_data.get("local_stations_info", []) or []
    is_gridded = bool(stations and stations[0].get("id") == "GRIDDED")

    light_green = (0.5, 0.8, 0.5)
    light_blue = (0.4, 0.5, 0.8)
    light_red = (0.8, 0.5, 0.5)
    light_grey = (0.85, 0.85, 0.85)
    white = (1, 1, 1)

    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"

    fig = plt.figure(figsize=(17, 11), dpi=140)
    fig.set_facecolor("0.77")

    # ------------------------------------------------------------------
    # Figure-fraction layout — graph-first; lower band centered on page
    #   Graph:       y 0.38–0.97
    #   Middle band: y 0.24–0.34   description | rain
    #   Bottom band: y 0.012–0.24  logo        | stations
    #   Lower band left/right margins equal (0.04)
    # ------------------------------------------------------------------
    _m = 0.04  # page margin each side for lower band
    _left_w = 0.29  # description + logo column
    _gap = 0.02  # column gutter
    _right_w = 1.0 - 2 * _m - _left_w - _gap  # → 0.61
    _right_x = _m + _left_w + _gap  # → 0.35

    ax1 = fig.add_axes([0.05, 0.38, 0.935, 0.59])
    ax3 = fig.add_axes([_m, 0.24, _left_w, 0.10])
    ax2 = fig.add_axes([_right_x, 0.24, _right_w, 0.10])
    ax_logo = fig.add_axes([_m, 0.012, _left_w, 0.228])
    ax4 = fig.add_axes([_right_x, 0.012, _right_w, 0.228])

    for ax in (ax2, ax3, ax4, ax_logo):
        ax.axis("off")
        ax.set_facecolor(fig.get_facecolor())

    # ---- Main graph ----
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.tick_params(axis="x", which="major", colors="black", labelsize=9)
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
    ax1.grid(True, which="major", axis="x", linestyle="-", color="lightgrey", zorder=0)
    y_max_val = (
        max(
            rolling_total[graph_start:graph_end].max(),
            normal_high[graph_start:graph_end].max(),
        )
        * 1.03
    )
    ax1.set_ylim(ymin=0, ymax=y_max_val)
    ax1.set_xlim([pd.Timestamp(graph_start), pd.Timestamp(graph_end)])

    date_range_days = (graph_end - graph_start).days
    for days_prior in [0, 30, 60]:
        arrow_date = obs_date - timedelta(days=days_prior)
        if not (graph_start <= arrow_date <= graph_end):
            continue
        y_value = rolling_total.get(arrow_date)
        if y_value is not None and pd.notna(y_value):
            vertical_offset = -30 if y_value > y_max_val * 0.85 else 30
            days_from_start = (arrow_date - graph_start).days
            position_ratio = (
                days_from_start / date_range_days if date_range_days > 0 else 0.5
            )
            if position_ratio < 0.2:
                horizontal_align, horizontal_offset = "left", 15
            elif position_ratio > 0.8:
                horizontal_align, horizontal_offset = "right", -15
            else:
                horizontal_align, horizontal_offset = "center", 0
            ax1.annotate(
                arrow_date.strftime("%Y-%m-%d"),
                xy=(arrow_date, y_value),
                xycoords="data",
                xytext=(horizontal_offset, vertical_offset),
                textcoords="offset points",
                ha=horizontal_align,
                size=13,
                arrowprops=dict(
                    arrowstyle="simple",
                    fc="0.4",
                    ec="none",
                    connectionstyle=f"arc3,rad={-0.3 if horizontal_offset < 0 else 0.3}",
                ),
            )

    ax1.legend(loc="upper right")
    ax1.set_ylabel("Rainfall (Inches)", fontsize=18)
    title = (
        "nClimGrid-Daily Data"
        if is_gridded
        else "Daily Global Historical Climatology Network"
    )
    ax1.set_title(
        f"Antecedent Precipitation vs Normal Range based on {title}", fontsize=18
    )

    # ---- Rain table ----
    summary = precip_data.get("antecedent_score_summary", {})
    result, total_score = summary.get("condition"), summary.get("total_score")
    if result is None:
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

    t2 = ax2.table(
        cellText=rain_table_vals,
        cellColours=rain_colors,
        cellLoc="center",
        loc="center",
        colWidths=[0.12, 0.11, 0.11, 0.11, 0.14, 0.12, 0.11, 0.18],
    )
    t2.set_fontsize(9.5)
    t2.auto_set_font_size(False)
    t2.scale(1.0, 1.3)

    # ---- Description table ----
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

    t3 = ax3.table(
        cellText=desc_vals,
        colWidths=[0.45, 0.55],
        cellColours=desc_colors,
        cellLoc="center",
        loc="center",
    )
    t3.set_fontsize(9)
    t3.auto_set_font_size(False)
    t3.scale(1.0, 1.3)

    # ---- Station table: ALL stations ----
    # Short lists → natural row height, pinned to top of ax4 (no tall stretched cells).
    # Long lists  → bbox-locked inside ax4 so they cannot climb into the rain table.
    if not is_gridded and stations:
        station_vals = [
            ["Weather Station Name", "Dist (mi)", "Days Normal", "Days Antecedent"]
        ]
        station_colors = [[light_grey] * 4]
        for s in stations:
            station_vals.append(
                [
                    str(s.get("name", ""))[:32],
                    f"{s.get('distance', 0):.1f}",
                    str(s.get("days_normal", 0)),
                    str(s.get("days_antecedent", 0)),
                ]
            )
            station_colors.append([white] * 4)

        n_rows = len(station_vals)  # header + data

        if n_rows <= 7:
            # 1–6 stations: compact table at top of the cell
            t4 = ax4.table(
                cellText=station_vals,
                cellColours=station_colors,
                cellLoc="center",
                loc="upper center",
                colWidths=[0.42, 0.16, 0.21, 0.21],
            )
            t4.set_fontsize(9.0)
            t4.auto_set_font_size(False)
            t4.scale(1.0, 1.35)
        elif n_rows <= 11:
            t4 = ax4.table(
                cellText=station_vals,
                cellColours=station_colors,
                cellLoc="center",
                loc="upper center",
                colWidths=[0.42, 0.16, 0.21, 0.21],
            )
            t4.set_fontsize(8.5)
            t4.auto_set_font_size(False)
            t4.scale(1.0, 1.15)
        else:
            # Long list: lock to axes so it cannot overlap the rain table
            fs = 7.0 if n_rows <= 15 else (6.0 if n_rows <= 18 else 5.0)
            t4 = ax4.table(
                cellText=station_vals,
                cellColours=station_colors,
                cellLoc="center",
                bbox=[0.0, 0.0, 1.0, 1.0],
                colWidths=[0.42, 0.16, 0.21, 0.21],
            )
            t4.set_fontsize(fs)
            t4.auto_set_font_size(False)

    # ---- Logo ----
    try:
        logo_file = os.path.join(data_dir, "RD_3_9.png")
        logo = plt.imread(logo_file)
        ax_logo.imshow(logo, interpolation="bilinear", aspect="equal")
        ax_logo.set_anchor("C")
        ax_logo.set_xticks([])
        ax_logo.set_yticks([])
        for spine in ax_logo.spines.values():
            spine.set_visible(False)
    except Exception as e:
        logger.warning(
            "Could not load logo from %s: %s",
            os.path.join(data_dir, "RD_3_9.png"),
            e,
        )

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_streamflow_page(
    usgs_data: Dict,
    nwm_data: Dict,
    meta: Dict,
    pdf: PdfPages,
    max_rows: int = 10,
):
    """
    Page layout with full-width tables (maps removed).
    """
    light_grey = (0.85, 0.85, 0.85)
    white = (1.0, 1.0, 1.0)
    page_bg = (0.77, 0.77, 0.77)

    fig = plt.figure(figsize=(17.0, 11.3), dpi=140, facecolor=page_bg)

    # ------------------------------------------------------------------
    # Fixed layout regions – tables span nearly the full page width
    # ------------------------------------------------------------------
    ax_usgs_table = fig.add_axes([0.035, 0.575, 0.930, 0.350])
    ax_nwm_table = fig.add_axes([0.035, 0.210, 0.930, 0.320])
    ax_notes = fig.add_axes([0.035, 0.055, 0.930, 0.130])

    for ax in (ax_usgs_table, ax_nwm_table, ax_notes):
        ax.axis("off")

    table_fontsize = 9.7
    title_fontsize = 14.0

    # ===================== USGS Table =====================
    sites = (usgs_data or {}).get("usgs_sites", [])[:max_rows]
    usgs_header = ["Gage Name", "ID", "Dist (mi)", "Flow (cfs)", "%ile", "Condition"]
    usgs_vals = [usgs_header] + [
        [
            (s.get("name") or "")[:29],
            s.get("gage_id", ""),
            f"{s.get('distance_mi', 0):.1f}",
            f"{s.get('flow_cfs', 0):.1f}",
            f"{s.get('percentile', 0):.0f}",
            s.get("condition", "No Data"),
        ]
        for s in sites
    ]
    if len(usgs_vals) == 1:
        usgs_vals.append(["—", "—", "—", "—", "—", "No Data"])

    ax_usgs_table.set_title(
        f"USGS Streamflow - {meta.get('usgs_condition') or 'No Data'}",
        fontsize=title_fontsize,
        pad=10,
        fontweight="bold",
    )
    t1 = ax_usgs_table.table(
        cellText=usgs_vals,
        cellColours=[[light_grey] * 6] + [[white] * 6] * (len(usgs_vals) - 1),
        loc="upper center",
        colWidths=[0.34, 0.13, 0.10, 0.13, 0.10, 0.19],
    )
    t1.set_fontsize(table_fontsize)
    t1.auto_set_font_size(False)

    # ===================== NWM Table =====================
    reaches = (nwm_data or {}).get("nwm_reaches", [])[:max_rows]
    nwm_header = ["COMID", "Dist (mi)", "Flow (cfs)", "%ile", "Condition"]
    nwm_vals = [nwm_header] + [
        [
            str(r.get("COMID", ""))[:9],
            f"{r.get('distance_mi', 0):.1f}",
            f"{r.get('flow_cfs', 0):.1f}",
            f"{r.get('percentile', 0):.0f}",
            r.get("condition", "No Data"),
        ]
        for r in reaches
    ]
    if len(nwm_vals) == 1:
        nwm_vals.append(["—", "—", "—", "—", "No Data"])

    ax_nwm_table.set_title(
        f"NWM Streamflow - {meta.get('nwm_condition') or 'No Data'}",
        fontsize=title_fontsize,
        pad=10,
        fontweight="bold",
    )
    t2 = ax_nwm_table.table(
        cellText=nwm_vals,
        cellColours=[[light_grey] * 5] + [[white] * 5] * (len(nwm_vals) - 1),
        loc="upper center",
        colWidths=[0.23, 0.13, 0.15, 0.13, 0.36],
    )
    t2.set_fontsize(table_fontsize)
    t2.auto_set_font_size(False)

    # ===================== Notes =====================
    note_vals = [
        ["USGS Source", "NWIS Daily Values (00060)"],
        ["USGS Method", "Same-day percentile rank vs historic record"],
        ["NWM Source", "Analysis-assim + retrospective (1990-2020)"],
        ["NWM Method", "Same-day percentile rank vs 1990-2020"],
    ]
    ax_notes.set_title(
        "Data Sources & Methods", fontsize=13.5, pad=8, fontweight="bold"
    )
    t3 = ax_notes.table(
        cellText=note_vals,
        cellColours=[[light_grey, white]] * 4,
        loc="center",
        colWidths=[0.37, 0.63],
    )
    t3.set_fontsize(10.0)
    t3.auto_set_font_size(False)

    # Footnote: clarify what "Normal" means for the Condition column
    fig.text(
        0.50,
        0.018,
        "Note: “Normal” streamflow conditions correspond to the 25th-75th percentile "
        "of the historic same-day record.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.35",
        style="italic",
    )

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_batch_summary_page(
    rows: list,
    meta: Dict[str, Any],
    out_path: str,
    data_dir: str = "data",
):
    """
    One-page watershed / batch summary PDF written to out_path.

    meta keys (all optional):
      title, site_lat, site_lon, observation_date, geographic_scope,
      huc_id, huc_size, used_gridded
    """
    stats = _aggregate_batch_stats(rows)
    light_grey, white = _LIGHT_GREY, _WHITE

    fig = plt.figure(figsize=(17.0, 11.0), dpi=140, facecolor=_PAGE_BG)

    # Layout: left column tables, right pie, bottom breakdown
    ax_inputs = fig.add_axes([0.04, 0.72, 0.42, 0.20])
    ax_intermediate = fig.add_axes([0.04, 0.52, 0.42, 0.18])
    ax_prelim = fig.add_axes([0.04, 0.36, 0.42, 0.14])
    ax_breakdown = fig.add_axes([0.04, 0.04, 0.92, 0.30])
    ax_pie = fig.add_axes([0.52, 0.42, 0.42, 0.48])

    for ax in (ax_inputs, ax_intermediate, ax_prelim, ax_breakdown, ax_pie):
        ax.axis("off")

    title = meta.get("title") or "Antecedent Precipitation - Batch / Watershed Summary"
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.97)
    fig.text(
        0.97,
        0.97,
        f"Generated {datetime.today().strftime('%Y-%m-%d')}",
        ha="right",
        va="top",
        fontsize=10,
        color="0.3",
    )

    # ---- User inputs ----
    ax_inputs.set_title(
        "User Inputs", fontsize=13, fontweight="bold", loc="left", pad=6
    )
    lat = meta.get("site_lat")
    lon = meta.get("site_lon")
    coord_str = (
        f"{float(lat):.6f}, {float(lon):.6f}"
        if lat is not None and lon is not None
        else "—"
    )
    inputs_vals = [
        ["Coordinates", coord_str],
        ["Observation Date", str(meta.get("observation_date") or "—")],
        ["Geographic Scope", str(meta.get("geographic_scope") or "—")],
        ["Used Gridded Precipitation", str(meta.get("used_gridded", "—"))],
    ]
    t1 = ax_inputs.table(
        cellText=inputs_vals,
        cellColours=[[light_grey, white]] * len(inputs_vals),
        colWidths=[0.48, 0.52],
        loc="upper center",
    )
    t1.set_fontsize(10)
    t1.auto_set_font_size(False)

    # ---- Intermediate ----
    ax_intermediate.set_title(
        "Intermediate Data", fontsize=13, fontweight="bold", loc="left", pad=6
    )
    huc = meta.get("huc_id")
    try:
        float(huc)
        watershed_label = "Hydrologic Unit Code"
    except (TypeError, ValueError):
        watershed_label = "Custom Watershed Name" if huc else "Watershed ID"

    huc_size = meta.get("huc_size")
    size_str = f"{huc_size} mi²" if huc_size is not None else "—"
    inter_vals = [
        [watershed_label, str(huc) if huc is not None else "—"],
        ["Watershed Size", size_str],
        ["Sampling Points / Days", str(stats["n_points"])],
        ["With Valid Score", str(stats["n_scored"])],
    ]
    t2 = ax_intermediate.table(
        cellText=inter_vals,
        cellColours=[[light_grey, white]] * len(inter_vals),
        colWidths=[0.48, 0.52],
        loc="upper center",
    )
    t2.set_fontsize(10)
    t2.auto_set_font_size(False)

    # ---- Preliminary result ----
    ax_prelim.set_title(
        "Preliminary Result", fontsize=13, fontweight="bold", loc="left", pad=6
    )
    avg_disp = f"{stats['avg_score']:.2f}" if stats["avg_score"] is not None else "—"
    prelim_vals = [
        ["Average Antecedent Precipitation Score", avg_disp],
        ["Preliminary Determination", stats["prelim"]],
    ]
    prelim_colors = [
        [light_grey, white],
        [light_grey, stats["prelim_color"]],
    ]
    t3 = ax_prelim.table(
        cellText=prelim_vals,
        cellColours=prelim_colors,
        colWidths=[0.62, 0.38],
        loc="upper center",
    )
    t3.set_fontsize(10)
    t3.auto_set_font_size(False)

    # ---- Pie ----
    ax_pie.set_title("Condition Distribution", fontsize=13, fontweight="bold", pad=8)
    if stats["pie_sizes"]:
        wedges, texts, autotexts = ax_pie.pie(
            stats["pie_sizes"],
            colors=stats["pie_colors"],
            labels=stats["pie_labels"],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontweight("bold")
        ax_pie.axis("equal")
    else:
        ax_pie.text(0.5, 0.5, "No scored points", ha="center", va="center", fontsize=12)

    # ---- Breakdown table (cap rows so it fits) ----
    ax_breakdown.set_title(
        "Sampling Point / Day Breakdown",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    max_rows = 18  # header + 17 data rows
    vals = stats["table_vals"][:max_rows]
    cols = stats["table_colors"][:max_rows]
    if len(stats["table_vals"]) > max_rows:
        vals = vals + [
            ["…", "…", "…", "…", f"+{len(stats['table_vals']) - max_rows} more", "…"]
        ]
        cols = cols + [[white] * 6]

    t4 = ax_breakdown.table(
        cellText=vals,
        cellColours=cols,
        colWidths=[0.14, 0.14, 0.14, 0.10, 0.28, 0.12],
        loc="upper center",
    )
    t4.set_fontsize(9)
    t4.auto_set_font_size(False)

    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Wrote batch summary page: %s (%s rows)", out_path, stats["n_points"])


def _append_summary_if_any(
    writer: PdfWriter, rows: list, meta: dict, data_dir: str, dest_dir: str
):
    """Build summary PDF and append to an open PdfWriter. No-op if rows empty."""
    if not rows:
        logger.info("No precip summaries available; skipping batch summary page")
        return
    summary_path = os.path.join(dest_dir, "_batch_summary_tmp.pdf")
    try:
        _plot_batch_summary_page(rows, meta or {}, summary_path, data_dir=data_dir)
        if os.path.exists(summary_path):
            writer.append(summary_path)
    finally:
        if os.path.exists(summary_path):
            try:
                os.remove(summary_path)
            except OSError:
                pass


# ====================== PDF GENERATION AND MERGING ======================


def generate_daily_pdf(
    lat: float,
    lon: float,
    analysis_date: datetime,
    output_dir: str,
    data_dir: str = "data",
    analysis_types: list = None,
    debug_behavior: bool = False,
):
    """Generate a single daily PDF report.

    Only sources named in ``analysis_types`` are loaded.  ``None`` keeps the
    legacy “load whatever files exist” behavior.  An explicit list is a
    whitelist (case-insensitive): precip, pdsi, usgs, nwm, wimp.

    Precip files are chosen in order: GHCN if present, otherwise Gridded.
    Missing precip is a warning, not a hard stop, so streamflow-only PDFs
    can still be written.
    """
    coord_directory = _coord_str(lat, lon)
    pdf_dir = os.path.join(output_dir, coord_directory)
    data_path = os.path.join(pdf_dir, "data")
    os.makedirs(pdf_dir, exist_ok=True)
    date_str = analysis_date.strftime("%Y-%m-%d")
    pdf_path = os.path.join(pdf_dir, f"{date_str}.pdf")

    known_types = {"precip", "pdsi", "usgs", "nwm", "wimp"}
    if analysis_types is None:
        requested = set(known_types)
    else:
        requested = {str(t).lower() for t in analysis_types}

    # Load only requested sources.  Skipped types stay {} so downstream
    # `if data:` checks and .get() calls keep working.
    pdsi_data = (
        _load_json_if_exists(os.path.join(data_path, f"{date_str}-PDSI.json"))
        if "pdsi" in requested
        else {}
    )
    usgs_data = (
        _load_json_if_exists(os.path.join(data_path, f"{date_str}-USGS.json"))
        if "usgs" in requested
        else {}
    )
    nwm_data = (
        _load_json_if_exists(os.path.join(data_path, f"{date_str}-NWM.json"))
        if "nwm" in requested
        else {}
    )
    wimp_data = (
        _load_json_if_exists(os.path.join(data_path, f"{date_str}-WIMP.json"))
        if "wimp" in requested
        else {}
    )

    precip_files = []
    if "precip" in requested:
        ghcn_path = os.path.join(data_path, f"{date_str}-GHCN.json")
        gridded_path = os.path.join(data_path, f"{date_str}-Gridded.json")
        if os.path.exists(ghcn_path):
            precip_files = [ghcn_path]
        elif os.path.exists(gridded_path):
            precip_files = [gridded_path]
        else:
            logger.warning(
                f"No GHCN or Gridded precip data found for {date_str} at {lat}, {lon}"
            )

    if not precip_files and not usgs_data and not nwm_data:
        logger.warning(f"No data files found for {date_str} at {lat}, {lon}")
        return

    with PdfPages(pdf_path) as pdf:
        for jfile in sorted(precip_files):
            with open(jfile) as f:
                precip_data = json.load(f)

            _plot_precip_page(
                precip_data=precip_data,
                pdsi_data=pdsi_data,
                usgs_data=usgs_data,
                nwm_data=nwm_data,
                wimp_data=wimp_data,
                pdf=pdf,
                data_dir=data_dir,
                debug_behavior=debug_behavior,
            )

        if usgs_data or nwm_data:
            meta = _extract_meta(
                precip_files, usgs_data, nwm_data, lat, lon, analysis_date
            )
            _plot_streamflow_page(usgs_data, nwm_data, meta, pdf)

    logger.info(f"Generated PDF: {pdf_path}")


def merge_daily_pdfs(
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime,
    output_dir: str,
    data_dir: str = "data",
    analysis_types: list = None,
    debug_behavior: bool = False,
    summary_meta: dict = None,
):
    """Rebuild each daily PDF from JSON under analysis_types, then merge + summary."""
    coord_directory = _coord_str(lat, lon)
    pdf_dir = os.path.join(output_dir, coord_directory)
    os.makedirs(pdf_dir, exist_ok=True)

    known = {"precip", "pdsi", "usgs", "nwm", "wimp"}
    requested = (
        set(known)
        if analysis_types is None
        else {str(t).lower() for t in analysis_types}
    )
    want_precip = "precip" in requested

    pdf_files = []
    day_count = (end_date - start_date).days + 1
    for d in range(day_count):
        day = start_date + timedelta(days=d)
        generate_daily_pdf(
            lat=lat,
            lon=lon,
            analysis_date=day,
            output_dir=output_dir,
            data_dir=data_dir,
            analysis_types=analysis_types,
            debug_behavior=debug_behavior,
        )
        candidate = os.path.join(pdf_dir, f"{day.strftime('%Y-%m-%d')}.pdf")
        if os.path.exists(candidate):
            pdf_files.append(candidate)

    if not pdf_files:
        logger.warning("No PDFs found to merge in the date range.")
        return

    merger = PdfWriter()
    for pdf in pdf_files:
        merger.append(pdf)

    # Summary page from this location's precip JSON
    rows = _collect_summaries_from_folder(pdf_dir, want_precip=want_precip)
    meta = {
        "title": "Antecedent Precipitation - Location Batch Summary",
        "site_lat": lat,
        "site_lon": lon,
        "observation_date": (
            f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}"
        ),
        "geographic_scope": "Single location (date range)",
        "huc_id": None,
        "huc_size": None,
        "used_gridded": (
            "Yes"
            if any(r.get("source") == "Gridded" for r in rows)
            else ("No" if rows else "—")
        ),
    }
    if summary_meta:
        meta.update(summary_meta)
    _append_summary_if_any(merger, rows, meta, data_dir, pdf_dir)

    output_path = os.path.join(pdf_dir, "Batch_Results.pdf")
    merger.write(output_path)
    merger.close()
    logger.info(f"Merged {len(pdf_files)} PDFs (+ summary) into {output_path}")


def merge_huc_batch_pdfs(
    output_dirs: list,
    base_output_dir: str,
    huc_id: str,
    analysis_date: datetime,
    data_dir: str = "data",
    analysis_types: list = None,
    debug_behavior: bool = False,
    summary_meta: dict = None,
):
    """Regenerate per-point PDFs for a single analysis_date, merge, then summary.

    Only the given date is processed (HUC / area sampling is a same-day product).
    Output filename includes the date: ``{YYYY-MM-DD}-HUC_{huc_id}_Batch_Report.pdf``.
    """
    if analysis_date is None:
        raise ValueError("merge_huc_batch_pdfs requires analysis_date")
    if not isinstance(analysis_date, datetime):
        analysis_date = datetime.strptime(str(analysis_date), "%Y-%m-%d")
    date_str = analysis_date.strftime("%Y-%m-%d")

    writer = PdfWriter()
    regenerated = 0
    fallback_appended = 0
    all_rows = []

    known = {"precip", "pdsi", "usgs", "nwm", "wimp"}
    requested = (
        set(known)
        if analysis_types is None
        else {str(t).lower() for t in analysis_types}
    )
    want_precip = "precip" in requested

    for folder in output_dirs:
        folder = os.path.normpath(folder)
        basename = os.path.basename(folder)

        try:
            lat_str, lon_str = basename.split("_", 1)
            lat, lon = float(lat_str), float(lon_str)
        except (ValueError, AttributeError):
            logger.warning(
                "Could not parse lat/lon from folder name %r; "
                "appending existing daily PDF for %s without regeneration",
                basename,
                date_str,
            )
            candidate = os.path.join(folder, f"{date_str}.pdf")
            if os.path.exists(candidate):
                writer.append(candidate)
                fallback_appended += 1
            continue

        parent_output = os.path.dirname(folder)
        generate_daily_pdf(
            lat=lat,
            lon=lon,
            analysis_date=analysis_date,
            output_dir=parent_output,
            data_dir=data_dir,
            analysis_types=analysis_types,
            debug_behavior=debug_behavior,
        )
        pdf_path = os.path.join(folder, f"{date_str}.pdf")
        if os.path.exists(pdf_path):
            writer.append(pdf_path)
            regenerated += 1
        else:
            logger.warning(
                "No PDF for %s at %s after generate_daily_pdf", date_str, basename
            )

        all_rows.extend(
            _collect_summaries_from_folder(
                folder, want_precip=want_precip, obs_date=date_str
            )
        )

    batch_folder = os.path.join(base_output_dir, f"{huc_id}-batch")
    os.makedirs(batch_folder, exist_ok=True)

    meta = {
        "title": f"Antecedent Precipitation – HUC {huc_id} Watershed Sampling Summary",
        "huc_id": huc_id,
        "observation_date": date_str,
        "geographic_scope": "HUC / watershed sampling",
        "used_gridded": (
            "Yes"
            if any(r.get("source") == "Gridded" for r in all_rows)
            else ("No" if all_rows else "—")
        ),
    }
    if all_rows:
        meta.setdefault("site_lat", all_rows[0].get("lat"))
        meta.setdefault("site_lon", all_rows[0].get("lon"))
    if summary_meta:
        meta.update(summary_meta)
        # Keep the enforced analysis date authoritative
        meta["observation_date"] = date_str

    _append_summary_if_any(writer, all_rows, meta, data_dir, batch_folder)

    if len(writer.pages) > 0:
        output_path = os.path.join(
            batch_folder, f"{date_str}-HUC_{huc_id}_Batch_Report.pdf"
        )
        with open(output_path, "wb") as f_out:
            writer.write(f_out)
        logger.info(f"Created consolidated HUC report: {output_path}")
        logger.debug(
            "regenerated=%s, fallback_appended=%s, summary_rows=%s, "
            "analysis_types=%s, analysis_date=%s",
            regenerated,
            fallback_appended,
            len(all_rows),
            analysis_types,
            date_str,
        )
    else:
        logger.warning("No PDF pages to write for HUC %s on %s", huc_id, date_str)


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
        debug_behavior=message.get("debug_behavior", False),
    )


def merge_pdfs(message: Dict[str, Any]):
    """High-level entry point for merging PDFs."""
    merge_daily_pdfs(
        lat=message["lat"],
        lon=message["lon"],
        start_date=message["start_date"],
        end_date=message["end_date"],
        output_dir=message["output_dir"],
        data_dir=message.get("data_dir", "data"),
        analysis_types=message.get("analysis_types"),
        debug_behavior=message.get("debug_behavior", False),
        summary_meta=message.get("summary_meta"),
    )
