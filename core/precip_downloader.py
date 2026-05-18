# downloaders.py
import logging
import os
from datetime import datetime, timedelta
from config import get_base_url, load_manifest
import pandas as pd
import requests
from geopy.distance import great_circle
from io import StringIO
import time

logger = logging.getLogger(__name__)


# ====================== ELEVATION ======================
def get_elevation(lat: float, lon: float, unit: str = "feet") -> float:
    """Try multiple elevation APIs with fallbacks."""
    apis = [
        ("USGS EPQS", lambda: _get_epqs_elevation(lat, lon, unit)),
        ("OpenTopo", lambda: _get_opentopo_elevation(lat, lon, unit)),
        ("OpenElevation", lambda: _get_open_elevation(lat, lon, unit)),
    ]

    for name, func in apis:
        try:
            elev = func()
            if elev is not None and elev > -1000:
                logger.debug(f"Elevation from {name}: {elev:.1f} ft")
                return elev
        except Exception as e:
            logger.debug(f"{name} failed: {e}")
        time.sleep(0.3)

    logger.warning(f"All elevation APIs failed for ({lat}, {lon})")
    return -1.0


def _get_epqs_elevation(lat, lon, unit="feet"):
    url = get_base_url("usgs_epqs")
    params = {"x": lon, "y": lat, "units": unit}
    r = requests.get(url, params=params, timeout=10)
    return r.json()["value"] if r.ok else None


def _get_opentopo_elevation(lat, lon, unit="feet"):
    url = get_base_url("opentopo_elevation")
    r = requests.get(url, params={"locations": f"{lat},{lon}"}, timeout=10)
    if r.ok:
        elev_m = r.json()["results"][0]["elevation"]
        return elev_m * 3.28084 if unit == "feet" else elev_m
    return None


def _get_open_elevation(lat, lon, unit="feet"):
    url = get_base_url("open_elevation")
    r = requests.get(url, params={"locations": f"{lat},{lon}"}, timeout=10)
    if r.ok:
        elev_m = r.json()["results"][0]["elevation"]
        return elev_m * 3.28084 if unit == "feet" else elev_m
    return None


# ====================== GHCN CORE ======================
def get_ghcn_timeseries_data(
    lat: float, lon: float, start_date_str: str, end_date_str: str
):
    """
    Matches original anteProcess.py logic as closely as possible.
    """

    def _get_station_list():
        stations_url = get_base_url("ghcn_stations")

        logger.info("Downloading GHCN station list...")
        response = requests.get(stations_url, timeout=30)
        response.raise_for_status()
        col_specs = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71)]
        names = ["id", "latitude", "longitude", "elevation", "state", "name"]
        df = pd.read_fwf(
            StringIO(response.text), colspecs=col_specs, names=names, index_col="id"
        )
        df["elevation"] = pd.to_numeric(df["elevation"], errors="coerce")
        logger.info(f"Loaded {len(df)} GHCN stations.")
        return df

    def _parse_dly_file(file_content: str) -> pd.Series | None:
        records = []
        for line in file_content.splitlines():
            if line[17:21] != "PRCP":
                continue
            year = int(line[11:15])
            month = int(line[15:17])
            idx = 21
            for d in range(1, 32):
                val_str = line[idx : idx + 5]
                idx += 8
                if val_str.strip() == "-9999":
                    continue
                try:
                    date = pd.to_datetime(f"{year}-{month:02d}-{d:02d}")
                    records.append((date, float(val_str)))
                except ValueError:
                    continue
        return pd.Series(dict(records)) if records else None

    def _fetch_station_data(station_id: str) -> pd.Series | None:
        url = f"{get_base_url("ghcn_daily")}/all/{station_id}.dly"
        try:
            resp = requests.get(url, timeout=25)
            resp.raise_for_status()
            return _parse_dly_file(resp.text)
        except Exception as e:
            logger.warning(f"Failed to fetch {station_id}: {e}")
            return None

    # ====================== Primary Station Logic ======================
    def _get_primary_station(stations_with_data: list[dict], end_date: datetime):
        min_antecedent_rows = int(90 * 0.75)
        huge_record = None
        huge_lowest = 10000
        medium_record = None
        medium_lowest = 10000
        minimum_record = None
        minimum_lowest = 10000
        tolerable = 0.75

        for st in stations_with_data:
            act = st.get("actual_rows", 0)
            ant = st.get("antecedent_rows", 0)
            wdiff = st.get("weighted_diff", 10000)

            if ant >= min_antecedent_rows:
                if act > 10000 and wdiff < huge_lowest:
                    huge_lowest, huge_record = wdiff, st
                elif act > 8000 and wdiff < medium_lowest:
                    medium_lowest, medium_record = wdiff, st
                elif act > 6000 and wdiff < minimum_lowest:
                    minimum_lowest, minimum_record = wdiff, st

        best = minimum_record
        if medium_record and (
            not best
            or medium_record["weighted_diff"] < best["weighted_diff"] + 2 * tolerable
        ):
            best = medium_record
        if huge_record and (
            not best
            or huge_record["weighted_diff"] < best["weighted_diff"] + 4 * tolerable
        ):
            if best == minimum_record and medium_record:
                if (
                    huge_record["weighted_diff"]
                    < medium_record["weighted_diff"] + 2 * tolerable
                ):
                    best = huge_record
            else:
                best = huge_record

        if best is None:
            # Fallback: any station with data, sorted by weighted diff
            candidates = [
                s
                for s in stations_with_data
                if s.get("data") is not None and not s["data"].empty
            ]
            if candidates:
                return sorted(candidates, key=lambda x: x.get("weighted_diff", 10000))[
                    0
                ]
            return stations_with_data[0] if stations_with_data else None

        return best

    # ====================== Main Logic ======================
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    logger.info(f"Building GHCN composite for ({lat:.4f}, {lon:.4f})")

    target_elev_ft = get_elevation(lat, lon, "feet")

    # Get candidate stations
    all_stations = _get_station_list().dropna(
        subset=["latitude", "longitude", "elevation"]
    )
    local_stations = []
    for sid, row in all_stations.iterrows():
        dist = great_circle((lat, lon), (row["latitude"], row["longitude"])).miles
        if dist > 60:
            continue
        station_elev_ft = (row["elevation"] or 0) * 3.28084
        elev_diff = abs((target_elev_ft or 0) - station_elev_ft)
        wdiff = dist * ((elev_diff / 1000) + 0.45)

        local_stations.append(
            {
                **row.to_dict(),
                "id": sid,
                "distance": round(dist, 3),
                "weighted_diff": round(wdiff, 4),
                "elevation_ft": station_elev_ft,
            }
        )

    local_stations = sorted(local_stations, key=lambda x: x["weighted_diff"])[:30]

    logger.info(f"Fetching data for {len(local_stations)} stations...")

    stations_with_data = []
    norm_start = start_date
    norm_end = datetime(end_date.year - 1, 9, 30)
    ante_start = end_date - timedelta(days=90)
    ante_end = end_date

    for st in local_stations:
        series = _fetch_station_data(st["id"])
        st["data"] = series
        if series is not None and not series.empty:
            st["actual_rows"] = (
                series.loc[norm_start:norm_end].notna().sum()
            )  # full normal period
            st["antecedent_rows"] = series.loc[ante_start:ante_end].notna().sum()
        else:
            st["actual_rows"] = 0
            st["antecedent_rows"] = 0
        stations_with_data.append(st)

    primary = _get_primary_station(stations_with_data, end_date)
    if not primary:
        logger.error("No usable primary station found.")
        return None, [], target_elev_ft or -1

    logger.info(
        f"Selected primary: {primary['id']} / {primary.get('name')} (wdiff={primary['weighted_diff']:.3f})"
    )

    # Re-sort others relative to primary
    other_stations = [s for s in stations_with_data if s["id"] != primary["id"]]
    for st in other_stations:
        dist = great_circle(
            (primary["latitude"], primary["longitude"]),
            (st["latitude"], st["longitude"]),
        ).miles
        elev_diff = abs((primary.get("elevation_ft", 0) - st.get("elevation_ft", 0)))
        st["weighted_diff"] = round(dist * ((elev_diff / 1000) + 0.45), 4)

    sorted_stations = [primary] + sorted(
        other_stations, key=lambda x: x["weighted_diff"]
    )

    # === COMPOSITE + STATION INFO ===
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    composite = pd.Series(index=date_range, dtype="float64")
    final_used = []

    for i, st in enumerate(sorted_stations):
        if composite.isna().sum() == 0:
            break

        data = st.get("data")
        if data is not None and not data.empty:
            before = composite.isna().sum()
            # Use the station's data to fill gaps (original style)
            composite = composite.fillna(data.reindex(date_range))
            filled = before - composite.isna().sum()

            if filled > 0 or i == 0:  # always include primary
                st_info = {
                    k: v
                    for k, v in st.items()
                    if k not in ["data", "actual_rows", "antecedent_rows"]
                }

                # Better match to old "Days Normal" / "Days Antecedent"
                st_info.update(
                    {
                        "days_normal": int(
                            st.get("actual_rows", 0)
                        ),  # true historical coverage
                        "days_antecedent": int(st.get("antecedent_rows", 0)),
                    }
                )
                final_used.append(st_info)

    # Final cleanup
    composite = composite.interpolate(method="time", limit_direction="both").fillna(0.0)

    logger.info(f"Composite built using {len(final_used)} stations")
    return composite, final_used, target_elev_ft or -1.0


# ====================== Gridded (unchanged) ======================
def build_composite_precip_series_gridded(lat, lon, start_date, end_date):
    try:
        from core.gridded_downloader import (
            build_composite_precip_series_gridded as impl,
        )
    except ImportError:
        from gridded_downloader import build_composite_precip_series_gridded as impl
    return impl(lat, lon, start_date, end_date)


# ====================== Public API ======================
def build_composite_precip_series(
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime,
    use_gridded: bool = False,
):
    logger.info(f"build_composite_precip_series called - gridded={use_gridded}")

    if use_gridded:
        series, stations = build_composite_precip_series_gridded(
            lat, lon, start_date, end_date
        )
        return series, stations, -1.0

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    series, stations_info, elevation = get_ghcn_timeseries_data(
        lat, lon, start_str, end_str
    )

    if series is None:
        series = pd.Series(dtype="float64")
        stations_info = []

    # Ensure full date range
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    series = series.reindex(full_range).fillna(0)

    return (
        series,
        stations_info or [],
        float(elevation) if elevation is not None else -1.0,
    )


if __name__ == "__main__":
    end_date = datetime.strptime("2023-11-11", "%Y-%m-%d")
    start_date = end_date - timedelta(days=365 * 2)

    series, info, elev = build_composite_precip_series(
        30.0, -90.0, start_date, end_date
    )
    print(f"GHCN: {len(series)} days, elevation={elev:.1f}")
    print(series.head(5))
    print("...")
    print(series.tail(5))
