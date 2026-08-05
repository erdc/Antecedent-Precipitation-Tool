# adapters/nwm_adapter.py
"""
Thin adapter for NWM Streamflow Analysis.
Connects the pure core/nwm_stream.py functions to the event dispatcher.
"""

import logging
import os
from datetime import datetime

from core.nwm_stream import analyze_nwm
from engine import EventDispatcher

logger = logging.getLogger(__name__)


class NwmAdapter:
    """Adapter that translates dispatcher messages to pure NWM analysis functions."""

    def register_handlers(self, dispatcher: EventDispatcher):
        """Register all NWM-related handlers."""
        dispatcher.register("nwm_analysis", self.handle_nwm_analysis)

    def handle_nwm_analysis(self, message: dict):
        """
        Run NWM streamflow analysis for a location/date.
        Returns a store_data message so the plotter can pick it up.
        """
        try:
            required = ["lat", "lon", "analysis_date", "data_dir", "output_dir"]
            if not all(k in message for k in required):
                logger.error("Missing required fields for NWM analysis")
                return None

            # analysis_date may arrive as datetime or string
            analysis_date = message["analysis_date"]
            if isinstance(analysis_date, datetime):
                date_str = analysis_date.strftime("%Y-%m-%d")
                date_obj = analysis_date
            else:
                date_str = str(analysis_date)
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")

            data_dir = message["data_dir"]
            cache_dir = message.get("cache_dir")
            if cache_dir == None:
                cache_dir = os.path.join(message.get("output_dir"), "data")
            state_shapefile = message.get(
                "state_shapefile",
                os.path.join(data_dir, "usa_shp", "tl_2023_us_state.shp"),
            )
            comid_shapefile = message.get(
                "comid_shapefile",
                os.path.join(data_dir, "nwm_COMID_APTv3.0.shp"),
            )

            result_df = analyze_nwm(
                lat=message["lat"],
                lon=message["lon"],
                analysis_date=date_obj,
                cache_dir=cache_dir,
                state_shapefile=state_shapefile,
                comid_shapefile=comid_shapefile,
                output_dir=message.get("output_dir"),
                write_kml=message.get("write_kml", True),
                write_csv=message.get("write_csv", True),
                historic_timeout=message.get("historic_timeout", 1800),
            )

            if result_df is None or result_df.empty:
                logger.warning("No valid NWM reaches found for this location/date")
                nwm_condition = "No Data"
                nwm_reaches = []
            else:
                # Use the closest reach (already sorted by distance)
                closest = result_df.iloc[0]
                nwm_condition = (
                    f"{closest['condition']} "
                    f"({closest['percentile']:.0f}%ile, "
                    f"{closest['distance_mi']:.1f} mi)"
                )
                logger.info(
                    f"NWM closest reach COMID {closest['COMID']}: {nwm_condition}"
                )

                # Persist top reaches for the PDF table
                nwm_reaches = result_df.head(8).to_dict(orient="records")

            return {
                "message_type": "store_data",
                "data_type": "nwm",
                "nwm_condition": nwm_condition,
                "nwm_reaches": nwm_reaches,
                "lat": message["lat"],
                "lon": message["lon"],
                "obs_date": date_obj,
                "output_dir": message["output_dir"],
            }

        except Exception as e:
            logger.error(f"NWM analysis failed: {e}", exc_info=True)
            return {
                "message_type": "warning",
                "warning_type": "nwm_analysis_failed",
                "error": str(e),
                "original_message": message,
            }
