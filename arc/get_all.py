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
##           get_all.py             ##
##  ------------------------------- ##
##     Written by: Jason Deters     ##
##     Edited by: Joseph Gutenson   ##
##  ------------------------------- ##
##    Last Edited on: 2023-04-26    ##
##  ------------------------------- ##
######################################

import os
from pathlib import Path

try:
    from . import get_files
except Exception:
    import get_files

MODULE_FOLDER = os.path.dirname(os.path.realpath(__file__))
ROOT_FOLDER = os.path.split(MODULE_FOLDER)[0]


def ensure_version_file():
    local_file_path = os.path.join(ROOT_FOLDER, "version")
    file_url = (
        "https://github.com/erdc/Antecedent-Precipitation-Tool/raw/master/version"
    )
    get_files.ensure_file_exists(file_url=file_url, local_file_path=local_file_path)


def ensure_wbd_folder():
    file_url = "https://github.com/erdc/Antecedent-Precipitation-Tool/releases/download/2.9.9/WBD.zip"
    gis_folder = os.path.join(ROOT_FOLDER, "GIS")
    local_file_path = os.path.join(gis_folder, "WBD.zip")
    wbd_folder = os.path.join(gis_folder, "WBD")
    wbd_Exists = os.path.exists(wbd_folder)
    version_folder = os.path.join(ROOT_FOLDER, "v")
    local_version_file = os.path.join(version_folder, "wbd")
    web_version_url = (
        "https://github.com/erdc/Antecedent-Precipitation-Tool/raw/master/v/wbd"
    )
    get_files.get_only_newer_version(
        file_url=file_url,
        local_file_path=local_file_path,
        local_check_file=wbd_folder,
        version_url=web_version_url,
        version_local_path=local_version_file,
        extract_path=wbd_folder,
    )


def ensure_us_shp_folder():
    gis_folder = os.path.join(ROOT_FOLDER, "GIS")
    us_shp_folder = os.path.join(gis_folder, "us_shp")
    us_shp_folder_exists = os.path.exists(us_shp_folder)
    if not us_shp_folder_exists:
        local_file_path = os.path.join(gis_folder, "us_shp.zip")
        try:
            os.remove(local_file_path)
        except Exception:
            pass
        try:
            file_url = "https://github.com/erdc/Antecedent-Precipitation-Tool/releases/download/2.9.9/us_shp.zip"
            get_files.ensure_file_exists(
                file_url=file_url,
                local_file_path=local_file_path,
                extract_path=us_shp_folder,
            )
        except:
            file_url = "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_nation_5m.zip"
            get_files.ensure_file_exists(
                file_url=file_url,
                local_file_path=local_file_path,
                extract_path=us_shp_folder,
            )


def ensure_climdiv_folder():
    gis_folder = os.path.join(ROOT_FOLDER, "GIS")
    climdiv_folder = os.path.join(gis_folder, "climdiv")
    climdiv_folder_exists = os.path.exists(climdiv_folder)
    if not climdiv_folder_exists:
        local_file_path = os.path.join(gis_folder, "climdiv.zip")
        try:
            os.remove(local_file_path)
        except Exception:
            pass
        file_url = "https://github.com/erdc/Antecedent-Precipitation-Tool/releases/download/2.9.9/climdiv.zip"
        get_files.ensure_file_exists(
            file_url=file_url, local_file_path=local_file_path, extract_path=gis_folder
        )


if __name__ == "__main__":
    ensure_wbd_folder()
    ensure_us_shp_folder()
    ensure_climdiv_folder()
