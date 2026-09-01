# adapters/huc_adapter.py
import logging
import os

from engine import EventDispatcher

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

        logger.info(
            "Queued %d points in %s (%d messages)",
            len(generated_output_dirs),
            huc_id,
            len(follow_ups),
        )

        return {"message_type": "_followups", "messages": follow_ups}
