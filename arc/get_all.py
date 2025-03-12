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

import os
import zipfile

try:
    from .utils import find_file_or_dir
except Exception:
    from utils import find_file_or_dir

MODULE_FOLDER = os.path.dirname(os.path.realpath(__file__))
ROOT_FOLDER = os.path.split(MODULE_FOLDER)[0]


def extract_local_archive(zip_path, extract_path):
    """
    Extract contents of a ZIP archive to a specified directory.
    """
    # Create extraction directory if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)

    # Extract ZIP contents
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)


def ensure_wbd_folder():
    # setup paths
    gis_folder = os.path.join(ROOT_FOLDER, "GIS")
    wbd_folder = os.path.join(gis_folder, "WBD")
    if os.path.exists(wbd_folder):
        return

    # find local zip, raises if not found
    local_file_path = find_file_or_dir(ROOT_FOLDER, "WBD.zip", num_searches=1)

    # create folder
    extract_local_archive(zip_path=local_file_path, extract_path=wbd_folder)


def ensure_us_shp_folder():
    # setup paths
    gis_folder = os.path.join(ROOT_FOLDER, "GIS")
    us_shp_folder = os.path.join(gis_folder, "us_shp")
    if os.path.exists(us_shp_folder):
        return

    # find local zip, raises if not found
    local_file_path = find_file_or_dir(ROOT_FOLDER, "us_shp.zip", num_searches=1)

    # create folder
    extract_local_archive(zip_path=local_file_path, extract_path=us_shp_folder)


def ensure_climdiv_folder():
    gis_folder = os.path.join(ROOT_FOLDER, "GIS")
    climdiv_folder = os.path.join(gis_folder, "climdiv")

    if os.path.exists(climdiv_folder):
        return

    # find local zip, raises if not found
    local_file_path = find_file_or_dir(ROOT_FOLDER, "climdiv.zip", num_searches=1)

    # create folder
    extract_local_archive(zip_path=local_file_path, extract_path=climdiv_folder)


if __name__ == "__main__":
    ensure_wbd_folder()
    ensure_us_shp_folder()
    ensure_climdiv_folder()
