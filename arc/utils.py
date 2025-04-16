#  This software was developed by United States Army Corps of Engineers (USACE)
#  employees in the course of their official duties.  USACE used copyrighted,
#  open source code to develop this software, as such this software
#  (per 17 USC § 101) is considered "joint work."  Pursuant to 17 USC § 105,
#  portions of the software developed by USACE employees in the course of their
#  official duties are not subject to copyright protection and are in the public
#  domain.
#
#  USACE assumes no responsibility whatsoever for the use of this software by
#  other parties, and makes no guarantees, expressed or implied, about its
#  quality, reliability, or any other characteristic.
#
#  The software is provided "as is," without warranty of any kind, express or
#  implied, including but not limited to the warranties of merchantability,
#  fitness for a particular purpose, and noninfringement.  In no event shall the
#  authors or U.S. Government be liable for any claim, damages or other
#  liability, whether in an action of contract, tort or otherwise, arising from,
#  out of or in connection with the software or the use or other dealings in the
#  software.
#
#  Public domain portions of this software can be redistributed and/or modified
#  freely, provided that any derivative works bear some notice that they are
#  derived from it, and any modified versions bear some notice that they have
#  been modified.
#
#  Copyrighted portions of the software are annotated within the source code.
#  Open Source Licenses, included in the source code, apply to the applicable
#  copyrighted portions.  Copyrighted portions of the software are not in the
#  public domain.

######################################
##  ------------------------------- ##
##            utils.py              ##
##  ------------------------------- ##
##     Written by: Chris French     ##
##      Edited by: Gene Kloss       ##
##      Edited by: Chris French     ##
##  ------------------------------- ##
##    Last Edited on:  2025-04-02   ##
##  ------------------------------- ##
######################################

import configparser
import glob
import logging
import os
import pkgutil
from logging.handlers import TimedRotatingFileHandler

logger = logging.getLogger(__name__)


def find_file_or_dir(search_dir, pattern, num_searches=2):
    """
    Helper function to find a file or directory based on a pattern.
    Raises FileNotFoundError with a custom message if not found.
    """
    for _ in range(num_searches):
        result = next(glob.iglob(f"{search_dir}/**/{pattern}", recursive=True), None)
        if result is not None:
            return os.path.abspath(result)
        search_dir = os.path.dirname(search_dir)
    raise FileNotFoundError(
        f"Pattern '{pattern}' not found after {num_searches} searches."
    )


def ini_config(default, data_dir="data"):
    # find file
    try:
        data_dir = find_file_or_dir(os.getcwd(), data_dir)
    except:
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
    config_path = os.path.join(data_dir, "config.ini")
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        # create a new one with the default values
        for section, options in default.items():
            config[section] = options
        with open(config_path, "w") as configfile:
            config.write(configfile)
    else:
        # if found, load it
        config.read(config_path)
        # if the default do not exist add them
        for section, options in default.items():
            if not config.has_section(section):
                config.add_section(section)
            for option, value in options.items():
                if not config.has_option(section, option):
                    config.set(section, option, str(value))
        # save the updated config file
        with open(config_path, "w") as configfile:
            config.write(configfile)
    return config


def filter_strings(strings, forbidden):
    """
    Filter out strings that contain any of the forbidden substrings.

    Args:
        strings (list): List of strings to filter
        forbidden (list): List of forbidden substrings

    Returns:
        list: Filtered list of strings that don't contain any forbidden substrings
    """
    # Handle None values and empty lists
    if strings is None or forbidden is None:
        return []
    if not isinstance(strings, list) or not isinstance(forbidden, list):
        return []

    # Use list comprehension to filter out strings containing forbidden substrings
    return [s for s in strings if not any(f in s for f in forbidden)]


def setup_logger(log_path: os.PathLike = None):
    """
    Set up root logger with both console and file handlers and optionally blacklist modules.

    Args:
        log_path: Path to save log files
        blacklist_modules: List of module names to exclude from logging

    Returns:
        Module-specific logger instance
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.propagate = False

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Handle log path configuration
    if log_path is None:
        try:
            log_path = find_file_or_dir(os.getcwd(), "Logs")
        except:
            log_path = os.path.join(os.getcwd(), "Logs")
        os.makedirs(log_path, exist_ok=True)
        log_path = os.path.join(log_path, "log.log")
    if not os.path.exists(log_path):
        with open(log_path, "w") as outfile:
            outfile.write("")

    # Create handlers
    r_handler = TimedRotatingFileHandler(
        log_path, when="midnight", interval=1, backupCount=3
    )
    c_handler = logging.StreamHandler()

    # Set level of handlers
    r_handler.setLevel(logging.DEBUG)
    c_handler.setLevel(logging.INFO)

    # Create formatters and add to handlers
    verbose_format = logging.Formatter(
        "%(asctime)s[%(filename)s]:%(levelname)s- %(message)s",
        datefmt="%H:%M:%S",
    )
    short_format = logging.Formatter("%(message)s")
    r_handler.setFormatter(verbose_format)
    c_handler.setFormatter(short_format)

    # Add handlers to root logger
    root_logger.addHandler(r_handler)
    root_logger.addHandler(c_handler)

    # Disable all non local loggers (set them to warn)
    all_loggers = get_all_loggers()
    local_modules = get_local_modules()
    local_modules += ["tqdm", "__main__", "apt", "arc"]
    loggers_to_disable = filter_strings(all_loggers, local_modules)
    blacklist_loggers(loggers_to_disable)

    logger.debug(f"Logger initialized, disabled loggers: {loggers_to_disable}")


def get_local_modules():
    current_dir = os.path.dirname(__file__)
    return [module.name for module in pkgutil.iter_modules([current_dir])]


def get_all_loggers():
    root_logger = logging.getLogger()
    return list(root_logger.manager.loggerDict.keys())


def blacklist_loggers(loggers_to_blacklist):
    """
    Sets the logging level to WARNING for specified loggers.
    """
    for logger_name in loggers_to_blacklist:
        if (
            ("tqdm" not in logger_name)
            and ("__main__" not in logger_name)
            and ("arc" not in logger_name)
        ):
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
