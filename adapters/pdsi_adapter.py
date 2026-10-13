# adapters/pdsi_adapter.py
import logging
from datetime import datetime

from core.pdsi import get_pdsi_for_location
from engine import EventDispatcher

logger = logging.getLogger(__name__)


class PdsiAdapter:
    """Adapter for PDSI analysis."""

    def register_handlers(self, dispatcher: EventDispatcher):
        """Register the PDSI analysis handler."""
        dispatcher.register("pdsi_analysis", self.handle_pdsi_analysis)

    def handle_pdsi_analysis(self, message: dict):
        """
        Run PDSI analysis and return the result for storage.
        """
        try:
            required = ["lat", "lon", "analysis_date", "data_dir", "output_dir"]
            if not all(k in message for k in required):
                logger.error("Missing required fields for PDSI analysis")
                return None

            analysis_date = message["analysis_date"]
            if not isinstance(analysis_date, datetime):
                analysis_date = datetime.strptime(str(analysis_date), "%Y-%m-%d")

            pdsi_value, pdsi_class, pdsi_color = get_pdsi_for_location(
                lat=message["lat"],
                lon=message["lon"],
                year=analysis_date.year,
                month=analysis_date.month,
                data_dir=message["data_dir"],
            )

            logger.info(
                f"PDSI for {analysis_date.strftime('%Y-%m')}: "
                f"{pdsi_value:.2f} ({pdsi_class})"
            )

            return {
                "message_type": "store_data",
                "data_type": "pdsi",
                "palmer_value": pdsi_value,
                "palmer_class": pdsi_class,
                "palmer_color": pdsi_color,
                "lat": message["lat"],
                "lon": message["lon"],
                "obs_date": analysis_date,
                "output_dir": message["output_dir"],
            }

        except Exception as e:
            logger.error(f"PDSI analysis failed: {e}", exc_info=True)
            return {
                "message_type": "warning",
                "warning_type": "pdsi_analysis_failed",
                "error": str(e),
                "original_message": message,
            }
