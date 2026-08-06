# adapters/usgs_adapter.py
"""
Thin adapter for USGS Streamflow Analysis.
Connects the pure core/usgs_stream.py functions to the event dispatcher.
"""

import logging
from datetime import datetime

from core.usgs_stream import analyze_usgs
from engine import EventDispatcher

logger = logging.getLogger(__name__)


class UsgsAdapter:
    """Adapter that translates dispatcher messages to pure USGS analysis functions."""

    def register_handlers(self, dispatcher: EventDispatcher):
        """Register all USGS-related handlers."""
        dispatcher.register("usgs_analysis", self.handle_usgs_analysis)

    def handle_usgs_analysis(self, message: dict):
        """
        Run USGS streamflow analysis for a location/date.
        Returns a store_data message so the plotter can pick it up.
        """
        try:
            required = ["lat", "lon", "analysis_date", "data_dir"]
            if not all(k in message for k in required):
                logger.error("Missing required fields for USGS analysis")
                return None

            # analysis_date may arrive as datetime or string
            analysis_date = message["analysis_date"]
            if isinstance(analysis_date, datetime):
                date_str = analysis_date.strftime("%Y-%m-%d")
                date_obj = analysis_date
            else:
                date_str = str(analysis_date)
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")

            result_df = analyze_usgs(
                lat=message["lat"],
                lon=message["lon"],
                analysis_date=date_str,
                data_dir=message["data_dir"],
                output_dir=message.get("output_dir"),
                write_kml=message.get("write_kml", True),
                write_csv=message.get("write_csv", True),
            )

            if result_df is None or result_df.empty:
                logger.warning("No valid USGS gages found for this location/date")
                usgs_condition = "No Data"
                usgs_sites = []
            else:
                # Use the closest gage (already sorted by distance)
                closest = result_df.iloc[0]
                usgs_condition = (
                    f"{closest['condition']} "
                    f"({closest['percentile']:.0f}%ile, "
                    f"{closest['distance_mi']:.1f} mi)"
                )
                logger.info(f"USGS closest gage: {closest['name']} - {usgs_condition}")

                # Persist top sites for the PDF table (limit to 8 to keep layout clean)
                usgs_sites = result_df.head(8).to_dict(orient="records")

            return {
                "message_type": "store_data",
                "data_type": "usgs",
                "usgs_condition": usgs_condition,
                "usgs_sites": usgs_sites,
                "lat": message["lat"],
                "lon": message["lon"],
                "obs_date": date_obj,
                "output_dir": message["output_dir"],
            }

        except Exception as e:
            logger.error(f"USGS analysis failed: {e}", exc_info=True)
            return {
                "message_type": "warning",
                "warning_type": "usgs_analysis_failed",
                "error": str(e),
                "original_message": message,
            }
