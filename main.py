# main.py
import os
import sys
import multiprocessing
import tkinter as tk
import logging
from logging.handlers import RotatingFileHandler

# Tell Matplotlib to use the pre-compiled cache we bundle in the .spec file
if getattr(sys, "frozen", False):
    # Running in PyInstaller bundle
    base_dir = sys._MEIPASS
else:
    # Running in normal Python environment
    base_dir = os.path.dirname(os.path.abspath(__file__))

os.environ["MPLCONFIGDIR"] = os.path.join(base_dir, "mpl_cache")

# App-specific imports
from engine import EventDispatcher
from gui import PrecipGUI
from config import load_manifest, find_data_dir
from adapters.precip_adapter import PrecipAdapter
from adapters.plotter_adapter import PlotterAdapter
from adapters.usgs_adapter import UsgsAdapter
from adapters.nwm_adapter import NwmAdapter
from adapters.wimp_adapter import WimpAdapter
from adapters.pdsi_adapter import PdsiAdapter
from adapters.area_adapter import AreaAdapter

TITLE = r"""
    ++++  +++  ++++                     +++  ++++  +++                     _                    _            _
    hNNN +NNNy hNNm                    yNNN+ mNNd oNNN+        /\         | |                  | |          | |
    hMMMhhMMMmymMMN                    hMMMhyNMMmyhMMM+       /  \   _ __ | |_ _ __  ___ _ __  _| | __ _ _ __ | |_
    sNMMMMMMMMMMMMd   syyo syyy  yyys  sNMMMMMMMMMMMMm       / /\ \ | '_ \| __/ _ \/ __/ _ \/ _` |/ _ \ '_ \| __|
     +mMMMMMMMMMMs    NMMh mMMMo+MMMN   +dMMMMMMMMMMh       / ____ \| | | | ||  __/ (_|  __/ (_| |  __/ | | | |_
      dMMMm++MMMM+    NMMNNMMMMNNMMMN    yMMMMo+dMMMs     _/_/   \_\_| |_|\__\___|\___\___|\__,_|\___|_| |_|\__|
      dMMMm  MMMM+    yNMMMMMMMMMMMmy    yMMMM+ dMMMs    |  __ \             (_)     (_) |      | | (_)
      dMMMm  MMMM+     sMMMMMMMMMMm+     yMMMM+ dMMMs    | |__) | __ _  ___ _ _ __  _| |_ __ _ | |_ _  ___  _ __
      dMMMmooMMMMyyyyyyhMMMMMMMMMMmyyyyyydMMMMsodMMMs    |  ___/ '__/ _ \/ __| | '_ \| | __/ _` | __| |/ _ \| '_ \
      dMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMs    | |   | | |  __/ (__| | |_) | | || (_| | |_| | (_) | | | |
      dMMMMMMMMMMMMMMMMMMMNhysshmMMMMMMMMMMMMMMMMMMMs    |_|   |_|  \___|\___|_| .__/|_|\__\__,_|\__|_|\___/|_| |_|
      dMMMNyyMMMMMMyymMMMh+      hMMMNyyMMMMMMhymMMMs                       | |
      dMMMm  MMMMMM  dMMN         NMMN  NMMMMM+ dMMMs                       |_|   Concept by: Jason C. Deters
      dMMMm  MMMMMM+ dMMm         mMMN  NMMMMM+ dMMMs                             Developed by: Christopher E. 
    +dMMMMm++MMMMMMddNMMm         mMMMddMMMMMMo+dMMMNh                            French, Stephen W. Brown,
    hMMMMMMNNMMMMMMMMMMMm         mMMMMMMMMMMMNNNMMMMMo                           Chase O. Hamilton, Joseph L. 
    hMMMMMMMMMMMMMMMMMMMNhhhhhhhhhNMMMMMMMMMMMMMMMMMMMo                           Gutenson, and Jason C. Deters
    ymmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm+                              
"""


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

    # 4. Explicitly set local modules to DEBUG
    local_modules = [
        "__main__",
        "engine",
        "gui",
        "core.precip_downloader",
        "core.gridded_downloader",
        "core.precip",
        "core.plotting",
        "core.nwm_stream",
        "core.usgs_stream",
        "core.wimp_analysis",
        "core.pdsi",
        "core.area_sampling",
        "config",
        "adapters.area_adapter",
    ]
    for module_name in local_modules:
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.DEBUG)


if __name__ == "__main__":
    # 1. This must be the very first execution step in the main block
    # prevents a bug in the NWM download code
    multiprocessing.freeze_support()

    print(TITLE)

    # 2. System level setup
    setup_logging()
    load_manifest()
    data_dir = find_data_dir()

    # 3. Load core modules
    dispatcher = EventDispatcher()
    PrecipAdapter().register_handlers(dispatcher)
    PlotterAdapter().register_handlers(dispatcher)
    UsgsAdapter().register_handlers(dispatcher)
    NwmAdapter().register_handlers(dispatcher)
    WimpAdapter().register_handlers(dispatcher)
    PdsiAdapter().register_handlers(dispatcher)
    AreaAdapter().register_handlers(dispatcher)

    # 4. Start GUI
    root = tk.Tk()
    app = PrecipGUI(root, dispatcher, data_dir=data_dir)
    root.mainloop()
