# adapters/huc_adapter.py
import logging
import os
import glob
import json

import pandas as pd

from engine import EventDispatcher
from core.data_handler import _compute_precip_condition

logger = logging.getLogger(__name__)

COORD_DECIMALS = 6


def _get_sampler():
    """Prefer the new area_sampling module; fall back to old name."""
    try:
        from core.area_sampling import dynamic_area_sampler

        return dynamic_area_sampler
    except ImportError:
        try:
            from core.area_sampling import dynamic_watershed_sampler

            return dynamic_watershed_sampler
        except ImportError:
            logger.warning("No sampling module found – using single-point stub")
            return None


def _stub_sampler(lat, lon, huc_level=None, **_):
    return {
        "huc_id": f"STUB{huc_level or 'CUSTOM'}",
        "name": "Stub",
        "area_sq_miles": 0.0,
        "points": [(lat, lon)],
    }


def _parse_point(point):
    if isinstance(point, (tuple, list)) and len(point) >= 2:
        lat, lon = float(point[0]), float(point[1])
    elif isinstance(point, dict):
        if point.get("lat") is None or point.get("lon") is None:
            return None, None
        lat, lon = float(point["lat"]), float(point["lon"])
    else:
        return None, None
    return round(lat, COORD_DECIMALS), round(lon, COORD_DECIMALS)


def _coord_dir_name(lat: float, lon: float) -> str:
    return f"{lat:.{COORD_DECIMALS}f}_{lon:.{COORD_DECIMALS}f}"


class AreaAdapter:
    def register_handlers(self, dispatcher: EventDispatcher):
        dispatcher.register("huc_analysis", self.handle_huc_analysis)
        dispatcher.register("aggregate_huc_results", self.handle_aggregate_huc_results)

    def handle_huc_analysis(self, message: dict):
        """
        Sample points inside a HUC or a custom polygon, then queue
        per-point analyses + PDF generation and a final merge.
        """
        lat = message.get("lat")
        lon = message.get("lon")
        huc_level = message.get("huc_level")
        data_dir = message.get("data_dir", "data")
        base_output_dir = message.get("output_dir", "output")
        analysis_types = message.get("analysis_types") or []
        analysis_date = message.get("analysis_date")
        custom_polygon = message.get("custom_polygon")
        custom_name = message.get("custom_name") or "CUSTOM"

        if not analysis_types:
            logger.error("HUC/area analysis: no analysis_types selected")
            return None

        sampler = _get_sampler()
        try:
            if sampler is not None:
                logger.info(
                    "Running area sampler  lat=%.5f lon=%.5f  "
                    "huc_level=%s  custom=%s",
                    lat,
                    lon,
                    huc_level,
                    bool(custom_polygon),
                )
                sampler_result = sampler(
                    lat=lat,
                    lon=lon,
                    huc_level=huc_level,
                    data_dir=data_dir,
                    custom_polygon=custom_polygon,
                    custom_name=custom_name,
                )
            else:
                sampler_result = _stub_sampler(lat, lon, huc_level)
        except Exception as e:
            logger.error("Area sampling failed: %s", e, exc_info=True)
            return {
                "message_type": "warning",
                "warning_type": "huc_sampling_failed",
                "error": str(e),
                "original_message": message,
            }

        huc_id = sampler_result.get("huc_id")
        points = sampler_result.get("points") or []

        if not huc_id or not points:
            logger.error("Sampler returned no ID or points – aborting")
            return None

        batch_root = os.path.join(base_output_dir, f"{huc_id}-batch")
        os.makedirs(batch_root, exist_ok=True)

        follow_ups = []
        generated_output_dirs = []

        for point in points:
            sub_lat, sub_lon = _parse_point(point)
            if sub_lat is None or sub_lon is None:
                logger.warning("Skipping unparseable point: %r", point)
                continue

            point_dir = os.path.join(batch_root, _coord_dir_name(sub_lat, sub_lon))
            generated_output_dirs.append(point_dir)

            bulk = {
                "lat": sub_lat,
                "lon": sub_lon,
                "analysis_date": analysis_date,
                "output_dir": batch_root,  # keep all points under the same batch root
                "data_dir": data_dir,
            }
            if message.get("gridded"):
                bulk["gridded"] = True

            # pass through optional keys the other adapters understand
            for optional in (
                "cache_dir",
                "state_shapefile",
                "comid_shapefile",
                "write_kml",
                "write_csv",
                "historic_timeout",
            ):
                if optional in message:
                    bulk[optional] = message[optional]

            for analysis_type in analysis_types:
                follow_ups.append({"message_type": f"{analysis_type}_analysis", **bulk})
            follow_ups.append(
                {
                    "message_type": "generate_pdf",
                    "analysis_types": analysis_types,
                    **bulk,
                }
            )

        if not generated_output_dirs:
            logger.error("No valid sample points after parsing – aborting")
            return None

        # Final merge step
        follow_ups.append(
            {
                "message_type": "merge_huc_pdfs",
                "output_dirs": generated_output_dirs,
                "base_output_dir": base_output_dir,
                "huc_id": huc_id,
            }
        )

        follow_ups.append(
            {
                "message_type": "aggregate_huc_results",
                "output_dirs": generated_output_dirs,
                "base_output_dir": batch_root,  # Correct directory for the output file
                "huc_id": huc_id,
                "analysis_date": analysis_date,
            }
        )

        logger.info(
            "Queued %d points in %s (%d messages)",
            len(generated_output_dirs),
            huc_id,
            len(follow_ups),
        )

        return {"message_type": "_followups", "messages": follow_ups}

    def handle_aggregate_huc_results(self, message: dict):
        """
        Collects results from all sampled points in a HUC analysis and writes
        the final Sampling Results.csv file.
        """
        logger.info(f"Aggregating results for HUC {message['huc_id']}")
        results = []

        def _load_json_if_exists(path: str) -> dict:
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
            return {}

        for point_dir in message["output_dirs"]:
            try:
                # Extract lat/lon from directory name
                lat_str, lon_str = os.path.basename(point_dir).split("_")
                lat, lon = float(lat_str), float(lon_str)
                date_str = message["analysis_date"].strftime("%Y-%m-%d")

                data_path = os.path.join(point_dir, "data")

                # Find the main precip file (GHCN or Gridded)
                precip_json_path = next(
                    iter(glob.glob(os.path.join(data_path, f"{date_str}-G*.json"))),
                    None,
                )
                if not precip_json_path:
                    logger.warning(
                        f"No precipitation data file found in {data_path}, skipping."
                    )
                    continue

                precip_data = _load_json_if_exists(precip_json_path)
                pdsi_data = _load_json_if_exists(
                    os.path.join(data_path, f"{date_str}-PDSI.json")
                )
                wimp_data = _load_json_if_exists(
                    os.path.join(data_path, f"{date_str}-WIMP.json")
                )

                # Use the authoritative function from the data handler
                condition, score = _compute_precip_condition(precip_data)

                results.append(
                    {
                        "Latitude": lat,
                        "Longitude": lon,
                        "Date": date_str,
                        "PDSI Value": pdsi_data.get("palmer_value"),
                        "PDSI Class": pdsi_data.get("palmer_class"),
                        "Season": wimp_data.get("wimp_condition"),
                        "Antecedent Precipitation Score": (
                            score if condition != "Error" else None
                        ),
                        "Antecedent Precipitation Condition": (
                            condition if condition != "Error" else "Data Missing"
                        ),
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Could not process point directory {point_dir}: {e}", exc_info=True
                )

        if not results:
            logger.error(f"No results to aggregate for HUC {message['huc_id']}")
            return None

        # Write to CSV
        df = pd.DataFrame(results)
        huc_name = message["huc_id"]
        date_str = message["analysis_date"].strftime("%Y-%m-%d")
        csv_filename = f"{date_str}-{huc_name}- Sampling Results.csv"

        output_csv_path = os.path.join(message["base_output_dir"], csv_filename)

        df.to_csv(output_csv_path, index=False)
        logger.info(f"Wrote watershed sampling results to {output_csv_path}")
