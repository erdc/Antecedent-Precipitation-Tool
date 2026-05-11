import os
import tkinter as tk
from engine import EventDispatcher
import logging
from logging.handlers import RotatingFileHandler

from gui import PrecipGUI
from config import load_manifest
from adapters.precip_adapter import PrecipAdapter
from adapters.plotter_adapter import PlotterAdapter


def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    long_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    short_format = logging.Formatter("%(name)s - %(message)s")

    # 1. Create the rolling file handler
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "log.log"), maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(long_format)

    # 2. Create the console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(short_format)

    # 3. Set the ROOT logger to WARNING and add handlers.
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    # 4. Explicitly set YOUR local modules to DEBUG
    local_modules = [
        "__main__",
        "engine",
        "gui",
        "core.precip_downloader",
        "core.gridded_downloader",
        "core.precip",
        "core.plotting",
        "config",
    ]
    for module_name in local_modules:
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.DEBUG)


if __name__ == "__main__":
    setup_logging()
    load_manifest()

    dispatcher = EventDispatcher()

    # New precip pipeline
    PrecipAdapter().register_handlers(dispatcher)
    PlotterAdapter().register_handlers(dispatcher)

    # TODO: Add other adapters later

    root = tk.Tk()
    app = PrecipGUI(root, dispatcher)
    root.mainloop()
