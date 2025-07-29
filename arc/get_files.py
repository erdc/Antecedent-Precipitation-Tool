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
##          get_items.py            ##
##  ------------------------------- ##
##     Written by: Jason Deters     ##
##      Edited by: Chase Hamilton   ##
##  ------------------------------- ##
##    Last Edited on: 2022-11-10    ##
##  ------------------------------- ##
######################################

# Import Standard Libraries
import os
import sys
import time
import zipfile
from datetime import datetime

import requests

# Find module path
MODULE_PATH = os.path.dirname(os.path.realpath(__file__))
# Find ROOT folder
ROOT = os.path.split(MODULE_PATH)[0]


# Import Custom Libraries
try:
    # Frozen Application Method
    from .utilities import JLog
except Exception:
    # Reverse compatibility method - add utilities folder to path directly
    PYTHON_SCRIPTS_FOLDER = os.path.join(ROOT, "Python Scripts")
    TEST = os.path.exists(PYTHON_SCRIPTS_FOLDER)
    if TEST:
        UTILITIES_FOLDER = os.path.join(PYTHON_SCRIPTS_FOLDER, "utilities")
        sys.path.append(UTILITIES_FOLDER)
    else:
        ARC_FOLDER = os.path.join(ROOT, "arc")
        UTILITIES_FOLDER = os.path.join(ARC_FOLDER, "utilities")
        sys.path.append(UTILITIES_FOLDER)
    import JLog


def extract_to_folder(zip_file, output_folder, pwd=None):
    #    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    #        if pwd is None:
    #            zip_ref.extractall(output_folder)
    #        else:
    #            zip_ref.extractall(output_folder, pwd=pwd.encode())
    with zipfile.ZipFile(zip_file) as zip:
        for zip_info in zip.infolist():
            if pwd is None:
                try:
                    zip.extract(zip_info, output_folder)
                except Exception:
                    pass
            else:
                try:
                    zip.extract(zip_info, output_folder, pwd=pwd.encode())
                except Exception:
                    pass


def sizeof_fmt(num, suffix="B"):
    for unit in [" ", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return "{:6.2f} {}{}".format(num, unit, suffix)
        num /= 1024.0
    return "{:6.2f} {}{}".format(num, "Y", suffix)


def ensure_file_exists(
    file_url,
    local_file_path,
    local_check_file=None,
    version_url=None,
    version_local_path=None,
    minimum_size=None,
    extract_path=None,
    extract_pwd=None,
):
    """Checks for file, downloads if necessary"""
    download = False
    local_version = 0
    log = JLog.PrintLog()
    download_dir, file_name = os.path.split(local_file_path)
    # Check for local file
    local_file_exists = os.path.exists(local_file_path)
    if not local_file_exists:
        download = True
    else:
        if minimum_size is not None:
            local_file_size = os.path.getsize(local_file_path)
            if local_file_size < minimum_size:
                log.Wrap("    {} corrupt. Deleting...".format(local_file_path))
                os.remove(local_file_path)
                download = True
    if download is True:
        # Ensure download directory exists
        try:
            os.makedirs(download_dir)
        except Exception:
            pass
        dl_start = datetime.now()
        # Streaming with requests module
        num_bytes = 0
        count = 0
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(local_file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        num_bytes += 8192
                        count += 1
                        if count > 25:
                            formatted_bytes = sizeof_fmt(num_bytes)
                            log.print_status_message(
                                "    Downloading {}... ({})".format(
                                    file_name, formatted_bytes
                                )
                            )
                            count = 0
        formatted_bytes = sizeof_fmt(num_bytes)
        log.Wrap("    {} Downloaded ({})".format(file_name, formatted_bytes))
        sys.stdout.flush()
        time.sleep(0.1)
        # Extract compressed package if selected
        if extract_path is None:
            extracted = ""
        else:
            extracted = "and Extracting "
            log.Wrap("    Extracting package to target directory...")
            extract_to_folder(
                zip_file=local_file_path, output_folder=extract_path, pwd=extract_pwd
            )
            log.Wrap("     Extraction complete. Deleting zip file...")
            # Remove zip file after extraction
            os.remove(local_file_path)
        log.Time(dl_start, "Downloading {}{}".format(extracted, file_name))
        log.Wrap("")
    return
