import logging
import os

from engine import EventDispatcher

logger = logging.getLogger(__name__)

COORD_DECIMALS = 6


def _get_sampler():
    """Import sampler; allow missing module during development."""
    try:
        from core.huc_sampling import dynamic_watershed_sampler

        return dynamic_watershed_sampler
    except ImportError:
        logger.warning("core.huc_sampling not found; using single-point HUC stub")
        return None


def _stub_sampler(lat, lon, huc_level, data_dir):
    """Fallback: one point at the requested location so the pipeline can run."""
    huc_id = f"STUB{huc_level}"
    return {
        "huc_id": huc_id,
        "points": [{"lat": lat, "lon": lon}],
    }


def _parse_point(point):
    """
    Normalize a sampler point to (lat, lon) rounded to COORD_DECIMALS.
    Supports (lat, lon) / [lat, lon] tuples and {"lat", "lon"} dicts.
    """
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
    """Stable folder name matching plotting path helpers."""
    return f"{lat:.{COORD_DECIMALS}f}_{lon:.{COORD_DECIMALS}f}"


class HucAdapter:
    def register_handlers(self, dispatcher: EventDispatcher):
        dispatcher.register("huc_analysis", self.handle_huc_analysis)

    def handle_huc_analysis(self, message: dict):
        """
        Sample points in the HUC, queue analyses + PDF per point, then merge.

        Layout (same as single-point mode under a batch root):
          {base}/{huc_id}-batch/{lat:.6f}_{lon:.6f}/data/*.json
          {base}/{huc_id}-batch/{lat:.6f}_{lon:.6f}/{date}.pdf
        """
        lat = message.get("lat")
        lon = message.get("lon")
        huc_level = message.get("huc_level", 8)
        data_dir = message.get("data_dir", "data")
        base_output_dir = message.get("output_dir", "output")
        analysis_types = message.get("analysis_types") or []
        analysis_date = message.get("analysis_date")

        if not analysis_types:
            logger.error("HUC analysis: no analysis_types selected")
            return None

        sampler = _get_sampler()
        try:
            if sampler is not None:
                logger.info(
                    f"Running dynamic_watershed_sampler for lat={lat}, "
                    f"lon={lon}, huc_level={huc_level}"
                )
                sampler_result = sampler(lat, lon, huc_level, data_dir)
            else:
                sampler_result = _stub_sampler(lat, lon, huc_level, data_dir)
        except Exception as e:
            logger.error(f"HUC sampling failed: {e}", exc_info=True)
            return {
                "message_type": "warning",
                "warning_type": "huc_sampling_failed",
                "error": str(e),
                "original_message": message,
            }

        huc_id = sampler_result.get("huc_id")
        points = sampler_result.get("points") or []

        if not huc_id or not points:
            logger.error("No HUC ID or points returned from the sampler. Aborting.")
            return None

        batch_root = os.path.join(base_output_dir, f"{huc_id}-batch")
        os.makedirs(batch_root, exist_ok=True)

        follow_ups = []
        generated_output_dirs = []

        for point in points:
            sub_lat, sub_lon = _parse_point(point)
            if sub_lat is None or sub_lon is None:
                logger.warning(f"Skipping unparseable sample point: {point!r}")
                continue

            point_dir = os.path.join(batch_root, _coord_dir_name(sub_lat, sub_lon))
            generated_output_dirs.append(point_dir)

            bulk = {
                "lat": sub_lat,
                "lon": sub_lon,
                "analysis_date": analysis_date,
                "output_dir": batch_root,
                "data_dir": data_dir,
            }
            if message.get("gridded"):
                bulk["gridded"] = True
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

            follow_ups.append({"message_type": "generate_pdf", **bulk})

        if not generated_output_dirs:
            logger.error("No valid sample points after parsing. Aborting.")
            return None

        follow_ups.append(
            {
                "message_type": "merge_huc_pdfs",
                "output_dirs": generated_output_dirs,
                "base_output_dir": base_output_dir,
                "huc_id": huc_id,
            }
        )

        logger.info(
            f"HUC analysis queued for {len(generated_output_dirs)} points in "
            f"HUC {huc_id} ({len(follow_ups)} messages)."
        )

        return {"message_type": "_followups", "messages": follow_ups}
