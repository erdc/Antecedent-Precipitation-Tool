# adapters/wimp_adapter.py
import logging
import os
from datetime import datetime

from core.wimp_analysis import get_wimp_data_from_nc
from engine import EventDispatcher

logger = logging.getLogger(__name__)


class WimpAdapter:
    """Adapter for WIMP analysis using the local NetCDF file."""

    def register_handlers(self, dispatcher: EventDispatcher):
        """Register the WIMP analysis handler."""
        dispatcher.register("wimp_analysis", self.handle_wimp_analysis)

    def handle_wimp_analysis(self, message: dict):
        """
        Run WIMP analysis, log the full output table, and return the
        single-month result for storage.
        """
        try:
            required = ["lat", "lon", "analysis_date", "data_dir"]
            if not all(k in message for k in required):
                logger.error("Missing required fields for WIMP analysis")
                return None

            analysis_date = message["analysis_date"]
            if not isinstance(analysis_date, datetime):
                analysis_date = datetime.strptime(str(analysis_date), "%Y-%m-%d")

            nc_file_path = os.path.join(message["data_dir"], "wimp_data.nc")

            wimp_result = get_wimp_data_from_nc(
                nc_filepath=nc_file_path,
                target_lat=message["lat"],
                target_lon=message["lon"],
                target_month_num=analysis_date.month,
            )

            if wimp_result.get("ERROR"):
                logger.warning("WebWIMP analysis could not be completed")
                if wimp_result.get("message"):
                    logger.warning(wimp_result["message"])

            # Log the detailed table provided by the core function
            if wimp_result.get("full_table_string"):
                logger.info(
                    "WIMP Analysis Full Table:\n" + wimp_result["full_table_string"]
                )

            # Determine the condition to pass to the plotter
            if wimp_result["status"] == "SPECIAL_CONDITION":
                wimp_condition = wimp_result["message"]
            elif wimp_result["status"] == "OK":
                wimp_condition = wimp_result["selected_month_conclusion"]
            else:
                wimp_condition = "Error"

            return {
                "message_type": "store_data",
                "data_type": "wimp",
                "wimp_condition": wimp_condition,
                "lat": message["lat"],
                "lon": message["lon"],
                "obs_date": analysis_date,
                "output_dir": message["output_dir"],
            }

        except Exception as e:
            logger.error(f"WIMP analysis failed: {e}", exc_info=True)
            return {
                "message_type": "warning",
                "warning_type": "wimp_analysis_failed",
                "error": str(e),
                "original_message": message,
            }
