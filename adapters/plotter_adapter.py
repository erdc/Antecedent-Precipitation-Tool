# adapters/plotter_adapter.py
"""
Thin adapter for the PDF Plotter.
Connects the pure core/plotting.py functions to the event dispatcher.
"""

import logging

from core.plotting import (
    generate_pdf,
    merge_pdfs,
    merge_huc_batch_pdfs,
    store_analysis_data,
)
from engine import EventDispatcher

logger = logging.getLogger(__name__)


class PlotterAdapter:
    """Adapter that translates dispatcher messages to pure plotting functions."""

    def register_handlers(self, dispatcher: EventDispatcher):
        """Register all plotting-related handlers."""
        dispatcher.register("store_data", self.handle_store_data)
        dispatcher.register("generate_pdf", self.handle_generate_pdf)
        dispatcher.register("merge_pdfs", self.handle_merge_pdfs)
        dispatcher.register("merge_huc_pdfs", self.handle_merge_huc_pdfs)

    def handle_store_data(self, message: dict):
        """
        Store analysis results (precip, pdsi, usgs, nwm) as JSON.
        Called after each analysis completes.
        """
        try:
            store_analysis_data(message)
            return None
        except Exception as e:
            logger.error(f"Failed to store data: {e}", exc_info=True)
            return {
                "message_type": "warning",
                "warning_type": "storage_failed",
                "error": str(e),
                "original_message": message,
            }

    def handle_generate_pdf(self, message: dict):
        """
        Generate a single daily PDF report.
        Triggered after all data for a date has been stored.
        """
        try:
            required = ["lat", "lon", "analysis_date", "output_dir"]
            if not all(k in message for k in required):
                logger.error("Missing required fields for PDF generation")
                return None

            generate_pdf(message)
            logger.debug(
                f"PDF generation request completed for {message.get('analysis_date')}"
            )
            return None

        except Exception as e:
            logger.error(f"PDF generation failed: {e}", exc_info=True)
            return {
                "message_type": "warning",
                "warning_type": "pdf_generation_failed",
                "error": str(e),
                "original_message": message,
            }

    def handle_merge_pdfs(self, message: dict):
        """
        Merge multiple daily PDFs into a single Batch_Results.pdf.
        Used in batch mode.
        """
        try:
            required = ["lat", "lon", "start_date", "end_date", "output_dir"]
            if not all(k in message for k in required):
                logger.error("Missing required fields for PDF merge")
                return None

            merge_pdfs(message)
            logger.info("Batch PDF merge completed successfully.")
            return None

        except Exception as e:
            logger.error(f"PDF merge failed: {e}", exc_info=True)
            return {
                "message_type": "warning",
                "warning_type": "pdf_merge_failed",
                "error": str(e),
                "original_message": message,
            }

    def handle_merge_huc_pdfs(self, message: dict):
        """
        Merge per-point daily PDFs into one HUC batch report.
        """
        try:
            required = ["output_dirs", "base_output_dir", "huc_id"]
            if not all(k in message for k in required):
                logger.error("Missing required fields for HUC PDF merge")
                return None

            path = merge_huc_batch_pdfs(
                output_dirs=message["output_dirs"],
                base_output_dir=message["base_output_dir"],
                huc_id=message["huc_id"],
            )
            if path:
                logger.info(f"HUC PDF merge completed: {path}")
            else:
                logger.warning("HUC PDF merge produced no output")
            return None

        except Exception as e:
            logger.error(f"HUC PDF merge failed: {e}", exc_info=True)
            return {
                "message_type": "warning",
                "warning_type": "huc_pdf_merge_failed",
                "error": str(e),
                "original_message": message,
            }
