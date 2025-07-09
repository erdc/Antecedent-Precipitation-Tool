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
##          check_usa.py            ##
##  ------------------------------- ##
##    Written by: Jason Deters      ##
##     Edited by: Chase Hamilton    ##
##     Edited by: Chris French      ##
##  ------------------------------- ##
##    Last Edited on:  2025-07-09   ##
##  ------------------------------- ##
######################################

import logging
import time

import requests

logger = logging.getLogger(__name__)

RETRY_LIMIT = 3


def get_epqs_elevation(lat, long, unit="feet"):
    url = "https://epqs.nationalmap.gov/v1/json?"
    params = {"x": long, "y": lat, "units": unit}
    attempt = 0
    while attempt < RETRY_LIMIT:
        response = None  # response will be undefined if get fails
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200 and response.text.strip():
                # can still return 200 on a failure bc proper errors are hard apparently
                data = response.json()
                elevation = data["value"]
                # if attempt > 0:
                #     logger.debug(f"epqs success attempt: {attempt+1}")
                logger.debug(f"epqs success attempt: {attempt+1}")
                return elevation
            else:
                logger.error(f"National Map EPQS Error: {response.status_code}")
        except Exception as e:
            logger.error(f"Exception occurred: {str(e)}")
        attempt += 1
        logger.debug(f"WAITING {2**attempt} seconds")
        time.sleep(int(2**attempt))
        url = "http://epqs.nationalmap.gov/v1/json?"

    logger.debug(f"epqs failure after {attempt} attempts")
    return -1


def get_open_elevation(lat, long, unit="feet"):
    url = "http://api.open-elevation.com/api/v1/lookup"
    params = {"locations": f"{lat},{long}"}
    attempt = 0
    while attempt < RETRY_LIMIT:
        response = None  # response will be undefined if get fails
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                elevation = (
                    response.json().get("results", [{}])[0].get("elevation", None)
                )
                if (unit == "feet") | (
                    unit == "ft"
                ):  # open-elevation only returns meters
                    elevation *= 3.28084
                logger.debug(f"open elevation success attempt: {attempt+1}")
                return elevation
            else:
                logger.error(f"Open Elevation Error: {response.status_code}")
        except Exception as e:
            logger.error(f"Exception occurred: {str(e)}")
        attempt += 1
        logger.debug(f"WAITING {2**attempt} seconds")
        time.sleep(int(2**attempt))

    logger.debug(f"open elevation failure after {attempt} attempts")
    return -1


def get_elevation(lat, long, unit="feet"):
    elevation = get_epqs_elevation(lat, long, unit)

    if elevation == -1:
        elevation = get_open_elevation(lat, long, unit)

    return elevation


def batch(list_of_coords, units="Feet"):
    sampling_point_elevations = dict()
    for coord in list_of_coords:
        lat, long = coord
        elevation = get_elevation(lat, long, units)
        dict_key = "{},{}".format(lat, long)
        sampling_point_elevations[dict_key] = elevation
    logger.info(f"Sampling point elevations: {sampling_point_elevations}")
    return sampling_point_elevations
