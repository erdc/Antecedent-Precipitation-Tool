# adapters/precip_adapter.py
import logging
from datetime import datetime

from core.precip import run_full_precip_analysis
from engine import EventDispatcher

logger = logging.getLogger(__name__)


class PrecipAdapter:
    def register_handlers(self, dispatcher: EventDispatcher):
        dispatcher.register("precip_analysis", self.handle_precip_analysis)
        dispatcher.register("batch_prefetch", self.handle_prefetch)

    def handle_precip_analysis(self, message: dict):
        try:
            result = run_full_precip_analysis(
                lat=message["lat"],
                lon=message["lon"],
                analysis_date=message["analysis_date"],
                data_dir=message["data_dir"],
                output_dir=message["output_dir"],
                use_gridded=message.get("gridded", False),
            )

            return {
                "message_type": "store_data",
                "data_type": "precip",
                **result,
            }

        except Exception as e:
            logger.error(f"Precip analysis failed: {e}", exc_info=True)
            # Trigger download flow
            manifest_key = (
                "nclimgrid_data" if message.get("gridded") else "ghcn_station_list"
            )
            return {
                "message_type": "warning",
                "warning_type": "download_request",
                "manifest_key": manifest_key,
                "original_message": message,
            }

    def handle_prefetch(self, message: dict):
        if not message.get("ghcn") and not message.get("gridded"):
            return
        # Add prefetch logic later
        logger.info("Precip prefetch requested (stub)")
