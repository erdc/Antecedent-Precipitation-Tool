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
##       web_wimp_scraper.py        ##
##  ------------------------------- ##
##      Writen by: Jason Deters     ##
##      Edited by: Joseph Gutenson  ##
##  ------------------------------- ##
##    Last Edited on:  2023-06-13   ##
##  ------------------------------- ##
######################################

"""
Web-Scrapes WebWIMP:
The Web-based, Water-Budget, Interactive, Modeling Program using
Selenium to operate the JavaScript forms at "http://cyclops.deos.udel.edu/wimp/public_html/index.html"

Parses WebWIMP data for specific monthly values and determine Wet or Dry
season using instructions from the Regional Supplement copied below:

Excerpt from ERDC/EL TR-08-28
Regional Supplement to the Corps of Engineers Wetland Delineation Manual
Arid West Region (Version 2.0)
Section 5 - Difficult Wetland Situations in the Arid West
Wetlands that periodically lack indicators of wetland hydrology:
    "...
    3. Use one or more of the following approaches to determine whether wetland
    hydrology is present and the site is a wetland. In the remarks section
    of the data form or in the delineation report, explain the rationale for
    concluding that wetland hydrology is present even though indicators of
    wetland hydrology described in Chapter 4 were not observed.
    a. Site visits during the dry season. Determine whether the site visit
    occurred during the normal annual dry season. The dry season, as
    used in this supplement, is the period of the year when soil moisture is
    normally being depleted and water tables are falling to low levels in
    response to decreased precipitation and/or increased evapotranspiration,
    usually during late spring and summer. It also includes the
    beginning of the recovery period in late summer or fall. The Web-
    Based Water-Budget Interactive Modeling Program (WebWIMP) is
    one source for approximate dates of wet and dry seasons for any
    terrestrial location based on average monthly precipitation and estimated
    evapotranspiration (http://cyclops.deos.udel.edu/wimp/public_html/index.html).
    In general, the dry season in a typical year is indicated when potential
    evapotranspiration exceeds precipitation (indicated by negative values of
    DIFF in the WebWIMP output), resulting in drawdown of soil moisture storage
    (negative values of DST) and/or a moisture deficit (positive values of
    DEF, also called the unmet atmospheric demand for moisture). Actual
    dates for the dry season vary by locale and year.
    ..."
"""

import glob
import json
import os
import subprocess
import time
import traceback
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

try:
    from . import JLog
except Exception:
    import JLog


class WebWimpChecker:
    def __init__(self):
        self.log = JLog.PrintLog()
        self.driver = None
        wimp_path = next(
            glob.iglob(f"{os.getcwd()}/**/wimp_dict.json", recursive=True), None
        )
        if wimp_path is not None:
            with open(wimp_path, "r") as wimp_file:
                self.wimp_dict = json.load(wimp_file)
        else:
            print("Cached webwimp not found.")
            self.wimp_dict = None

    def _open_browser(self):
        try:
            self.chrome_driver_path = ChromeDriverManager().install()
            self._open_browser()

            chrome_options = Options()
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--show-capture=no")
            chrome_options.add_argument("--log-level=3")

            service = Service(self.chrome_driver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception as e:
            self.log.Wrap(f"Error opening browser: {e}")

    def _close_browser(self):
        if self.driver:
            self.driver.quit()
            self.driver = None

    def check_wimp(self, lat, lon, output_folder=None, watershed_analysis=False):
        """
        Queries WebWIMP:
        The Web-based, Water-Budget, Interactive, Modeling Program using
        Selenium to operate the JavaScript forms at
        "http://cyclops.deos.udel.edu/wimp/public_html/index.html"
        """
        # Create PrintLog
        log = JLog.PrintLog(Indent=2)

        try:
            lat = round(lat, 1)
            lon = round(lon, 1)

            if self.wimp_dict != None:
                error_messages = ["LARGE WATER BODY", "PERMANENT SNOW COVER", "ERROR"]
                coordinates_to_check = [
                    (lat, lon),
                    (lat + 0.1, lon),
                    (lat - 0.1, lon),
                    (lat, lon + 0.1),
                    (lat, lon - 0.1),
                ]

                for lat_check, lon_check in coordinates_to_check:
                    rows = self.wimp_dict.get(
                        f"{lat_check:.1f},{lon_check:.1f}", "ERROR"
                    )
                    if rows not in error_messages:
                        return rows

            url = "http://cyclops.deos.udel.edu/wimp/public_html/index.html"
            log.Wrap("Scraping Page ({})...".format(url))

            for x in range(4):
                try:
                    if self.driver is None:
                        self.open_browser()
                    self.driver.get(url)
                    break
                except Exception:
                    if x == 0:
                        time.sleep(0.1)
                        self.close_browser()
                        time.sleep(1)
                    elif x == 1:
                        time.sleep(0.1)
                        self.close_browser()
                        time.sleep(2)
                    elif x == 2:
                        time.sleep(0.1)
                        self.close_browser()
                        time.sleep(3)
                    elif x == 3:
                        return "ERROR"
            # Time buffer (Occasionally helps fewer warning messages post, but not mandatory)
            time.sleep(0.1)
            # Get Project Title input Box
            #            log.Wrap("  Supplying today's date as the 'Project Title' and hitting 'Return'")
            title_element = self.driver.find_element("name", "yname")
            # Type Current Date in Input Box
            current_date = datetime.now()
            current_date_string = current_date.strftime("%Y-%m-%d")
            title_string = "ARC Request - {}".format(current_date_string)
            title_element.clear()
            title_element.send_keys(title_string)
            # Submit Project Title by pressing return key
            title_element.send_keys(Keys.RETURN)
            # Allow for page to load
            time.sleep(0.1)
            # Find Latitude input box, clear it, then type selected Latitude
            #            log.Wrap("  Inputting latitude and longitude in respective boxes and hitting 'Return'")
            lat_element = self.driver.find_element("name", "lati")
            lat_element.clear()
            lat_element.send_keys("{}".format(lat))
            # Find Longitude input box, clear it, then type selected longitude
            lon_element = self.driver.find_element("name", "long")
            lon_element.clear()
            lon_element.send_keys("{}".format(lon))
            # Submit inputs by pressing return key
            lon_element.send_keys(Keys.RETURN)
            # Switch tabs (because it opened a new one and apparently it doesn't automatically switch)
            self.driver.switch_to.window(self.driver.window_handles[-1])
            # Allow for page to load
            time.sleep(0.1)
            # Check to see if point on Large Body of Water error occurred
            waterbody_error = False
            for elem in self.driver.find_elements(By.XPATH, "//span"):
                if "large body of water." in elem.text:
                    waterbody_error = True
            if waterbody_error:
                # Try to clear the error, which often goes away if you just click Revise lat/lon
                for element in self.driver.find_elements(By.XPATH, "//input"):
                    if (
                        element.get_attribute("value")
                        == "Revise Longitude and Latitude"
                    ):
                        revise_element = element
                revise_element.click()
                # Allow for page to load
                time.sleep(0.1)
                # Check to see if point on Large Body of Water error occurred
                waterbody_error = False
                for elem in self.driver.find_elements(By.XPATH, "//span"):
                    if "large body of water." in elem.text:
                        waterbody_error = True
                if waterbody_error:
                    lat_element = self.driver.find_element("name", "Latitude")
                    lat_element.clear()
                    updated_lat = lat + 0.01
                    lat_element.send_keys("{}".format(updated_lat))
                    # Test if the above cleared the error (Click revise lat/lon)
                    for element in self.driver.find_elements(By.XPATH, "//input"):
                        if (
                            element.get_attribute("value")
                            == "Revise Longitude and Latitude"
                        ):
                            revise_element = element
                    revise_element.click()
                    # Allow for page to load
                    time.sleep(0.1)
                    # Check to see if point on Large Body of Water error occurred
                    waterbody_error = False
                    for elem in self.driver.find_elements(By.XPATH, "//span"):
                        if "large body of water." in elem.text:
                            waterbody_error = True
                    if waterbody_error:
                        lat_element = self.driver.find_element("name", "Latitude")
                        lat_element.clear()
                        lat_element.send_keys("{}".format(lat))
                        # Find Longitude input box, clear it, then type selected longitude
                        lon_element = self.driver.find_element("name", "Longitude")
                        lon_element.clear()
                        updated_lon = lon + 0.01
                        lon_element.send_keys("{}".format(updated_lon))
                        # Test if the above cleared the error (Click revise lat/lon)
                        for element in self.driver.find_elements(By.XPATH, "//input"):
                            if (
                                element.get_attribute("value")
                                == "Revise Longitude and Latitude"
                            ):
                                revise_element = element
                        revise_element.click()
                        time.sleep(0.1)
                        # Check to see if point on Large Body of Water error occurred
                        for elem in self.driver.find_elements(By.XPATH, "//span"):
                            if "large body of water." in elem.text:
                                log.Wrap(
                                    "WebWIMP ERROR: - This location falls on a large body of water."
                                )
                                if not watershed_analysis:
                                    # Close browser
                                    self.driver.stop_client()
                                    self.driver.quit()
                                return "LARGE WATER BODY"
            # Find "Calculate Monthly Water Balance" button and click it
            for element in self.driver.find_elements(By.XPATH, "//input"):
                #            if element.get_attribute('name') == 'height':
                #                elevation = element.get_attribute('value')
                if element.get_attribute("value") == "Water Balance":
                    water_balance_element = element
            water_balance_element.click()
            # Switch tabs (because it opened a new one and apparently it doesn't automatically switch)
            self.driver.switch_to.window(self.driver.window_handles[-1])
            # Allow for page to load THIS STEP LAGS SO GIVE IT EXTRA TIME
            time.sleep(4)
            # Check for permanent snow cover error
            for elem in self.driver.find_elements(By.XPATH, "//span"):
                if "A permanent snow cover exists" in elem.text:
                    log.Wrap("WebWIMP ERROR: - {}".format(elem.text))
                    if not watershed_analysis:
                        # Close browser
                        self.driver.stop_client()
                        self.driver.quit()
                    return "PERMANENT SNOW COVER"
            # Create empty lists
            rows = []
            row = []
            jan_count = 0
            month_text = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            raw_list = []
            for elem in self.driver.find_elements(By.XPATH, "//td"):
                raw_list.append(elem.text)
            for text in raw_list:
                if text == "Total":
                    rows.append(row)
                    break
                if text == "Jan":
                    jan_count += 1
                if jan_count > 2:
                    if text in month_text:
                        if row:
                            rows.append(row)
                            row = []
                    row.append(text)
            # Get Graph
            if not watershed_analysis:
                if not output_folder is None:
                    try:
                        for elem in self.driver.find_elements(By.XPATH, "//a"):
                            if (
                                elem.text
                                == "Monthly and annual climatic water balance graph"
                            ):
                                graph_link_element = elem
                        graph_link_element.click()
                        # Switch tabs (because it opened a new one and apparently it doesn't automatically switch)
                        self.driver.switch_to.window(self.driver.window_handles[-1])
                        # Save a screenshot of the Water Balance Graph
                        log.Wrap("Getting Web Wimp Water Balance Graph screenshot...")
                        web_wimp_screenshot_path = os.path.join(
                            output_folder, "Web WIMP Water Balance Graph.png"
                        )
                        for numerator in range(10):
                            numerator += 1
                            # Test if screenshot exists
                            screenshot_exists = os.path.exists(web_wimp_screenshot_path)
                            if not screenshot_exists:
                                log.Wrap(
                                    "     ---Attempt {} of 10---".format(numerator)
                                )
                                # Wait for page to load further
                                time.sleep(numerator / 7)
                                # Attempt to save Screenshot
                                log.Wrap("  Saving screenshot of graph...")
                                self.driver.save_screenshot(web_wimp_screenshot_path)
                                screenshot_exists = os.path.exists(
                                    web_wimp_screenshot_path
                                )
                            if screenshot_exists:
                                # Test screenshot size (successful = > 40000)
                                screenshot_size = os.path.getsize(
                                    web_wimp_screenshot_path
                                )
                                if screenshot_size > 40000:
                                    log.Wrap(
                                        "  Opening saved screenshot in sub-process..."
                                    )
                                    subprocess.Popen(
                                        web_wimp_screenshot_path, shell=True
                                    )
                                    break
                                else:
                                    log.Wrap(
                                        "  Screenshot incomplete (Size < 40000 bytes)"
                                    )
                                    try:
                                        log.Wrap("  Deleting screenshot...")
                                        os.remove(web_wimp_screenshot_path)
                                    except Exception:
                                        pass
                    except Exception:
                        log.Wrap(traceback.format_exc())
                    self.close_browser()
            return rows
        except Exception:
            log.Wrap(traceback.format_exc())
            try:
                self.close_browser()
                time.sleep(1)
                self.open_browser()
            except Exception:
                pass


def make_length(number, length):
    num_str = str(number)
    while len(num_str) < length:
        num_str = num_str + " "
    return num_str


def calculate_wet_dry_table(wimp_rows, output_folder=None):
    """
    Parses WebWIMP table for specific monthly values and determine Wet or Dry season
    using instructions from the Regional Supplement.

    Excerpt from ERDC/EL TR-08-28
    Regional Supplement to the Corps of Engineers Wetland Delineation Manual
    Arid West Region (Version 2.0)
    Section 5 - Difficult Wetland Situations in the Arid West
    Wetlands that periodically lack indicators of wetland hydrology:
        "...
        3. Use one or more of the following approaches to determine whether wetland
        hydrology is present and the site is a wetland. In the remarks section
        of the data form or in the delineation report, explain the rationale for
        concluding that wetland hydrology is present even though indicators of
        wetland hydrology described in Chapter 4 were not observed.
        a. Site visits during the dry season. Determine whether the site visit
        occurred during the normal annual dry season. The dry season, as
        used in this supplement, is the period of the year when soil moisture is
        normally being depleted and water tables are falling to low levels in
        response to decreased precipitation and/or increased evapotranspiration,
        usually during late spring and summer. It also includes the
        beginning of the recovery period in late summer or fall. The Web-
        Based Water-Budget Interactive Modeling Program (WebWIMP) is
        one source for approximate dates of wet and dry seasons for any
        terrestrial location based on average monthly precipitation and estimated
        evapotranspiration (http://cyclops.deos.udel.edu/wimp/public_html/index.html). In general, the
        dry season in a typical year is indicated when potential evapotranspiration
        exceeds precipitation (indicated by negative values of DIFF in the
        WebWIMP output), resulting in drawdown of soil moisture storage
        (negative values of DST) and/or a moisture deficit (positive values of
        DEF, also called the unmet atmospheric demand for moisture). Actual
        dates for the dry season vary by locale and year.
        ..."
    """
    csv_rows = []
    # Create Main Printlog
    log = JLog.PrintLog(Indent=2)
    log.Wrap("Parsing scraped WebWimp values and Calculating Wet/Dry Season...")
    try:
        for num in range(12):
            # Parse Values
            row = wimp_rows[num]
            mon = row[0]
            diff = row[5]
            dst = row[7]
            def_val = row[9]
            # Correct interpreation (Confirmed by ERDC) - (Includes first part of recovery period IF it doesn't immediately eleminate deficit (DEF))
            season = "Wet"  # Starting value
            if float(diff) < 0:
                if float(dst) < 0:
                    season = "Dry"
            if float(def_val) > 0:
                season = "Dry"

            # Convert WebWIMP Scraped wimp_rows to smaller subset saved to CSV
            csv_rows.append([mon, diff, dst, def_val, season])
    except Exception:
        raise
    return csv_rows


def get_season_from_rows(rows, month=None):
    """
    Reads and reports cached WebWIMP results table and returns Wet/Dry Season
    """
    if month is None:
        month = 1
    # Create Printlogs
    log = JLog.PrintLog(Indent=2)
    # Print Terms and start of table
    log.Wrap("")
    log.Wrap("Terms:")
    log.Wrap(
        "DIFF is the rainfall and estimated snowmelt minus the adjusted potential evapotranspiration (mm/month)."
    )
    log.Wrap(
        "DST is the estimated change in soil moisture from the end of the previous month to the end of the current month (mm/month)."
    )
    log.Wrap(
        "DEF is the estimated deficit or unmet atmospheric demand for moisture (mm/month)."
    )
    log.Wrap("   ______________________________________ ")
    log.Wrap("  | Mon | DIFF | DST  | DEF | Conclusion |")
    log.Wrap("  |--------------------------------------|")
    num = 1
    for row in rows:
        # Get current row values
        mon = row[0]
        diff = row[1]
        dst = row[2]
        def_val = row[3]
        season = row[4]
        diff_four = make_length(diff, 4)
        dst_four = make_length(dst, 4)
        def_val_three = make_length(def_val, 3)
        # Save selected month value, if provided
        if num == month:
            selected_season = "{} Season".format(season)
            # Write the current Table row AND Selected Month Marker
            log.Wrap(
                "  | {} | {} | {} | {} | {} Season | <---Selected Month".format(
                    mon, diff_four, dst_four, def_val_three, season
                )
            )
        else:
            # Write the current Table row
            log.Wrap(
                "  | {} | {} | {} | {} | {} Season |".format(
                    mon, diff_four, dst_four, def_val_three, season
                )
            )
        num += 1
    # Print bottom line of table
    log.Wrap("   -------------------------------------- ")
    if month is None:
        selected_season = None
    return selected_season


class WimpScraper(object):
    """Manages the scraping of Web WIMP"""

    def __init__(self, watershed_analysis=False):
        self.log = JLog.PrintLog()
        self.rows = []
        self.wimp_checker_instance = None
        self.wimp_checker_executions = 0

    def get_season(self, lat, lon, month=None):
        """
        Checks for local copies before scraping WebWIMP and calculating
        the Dry/Wet Season per the instructions in the Regional Supplement.
        """
        self.log.print_section(
            "Web WIMP - Web-based Water-Budget Interactive Modeling Program"
        )
        lat = round(lat, 1)
        lon = round(lon, 1)
        self.log.Wrap("Scraping WebWIMP at {},{}...".format(lat, lon))
        try:
            # Create wimp_checker_instance
            if self.wimp_checker_instance is None:
                self.wimp_checker_instance = WebWimpChecker()
            # pull the data directly from WebWimp
            self.log.Wrap(" Calling check_wimp() function...")
            wimp_rows = self.wimp_checker_instance.check_wimp(lat, lon)
            if wimp_rows == "ERROR":
                return "ERROR"
            if wimp_rows == "LARGE WATER BODY":
                self.rows = wimp_rows
                self.log.print_separator_line()
                self.log.Write("")
                return "ERROR"
            if wimp_rows == "PERMANENT SNOW COVER":
                self.rows = wimp_rows
                self.log.print_separator_line()
                self.log.Write("")
                return "ERROR"
            try:
                self.rows = calculate_wet_dry_table(wimp_rows)
            except:
                self.rows = wimp_rows
            season = get_season_from_rows(self.rows, month)
            self.log.print_separator_line()
            self.log.Write("")
            return season
        except Exception:
            self.log.Wrap(traceback.format_exc())
            self.log.Write("")
            return "ERROR"


if __name__ == "__main__":
    WIMP_SCRAPER = WimpScraper()
    # WIMP_SCRAPER.get_season(lat=33.2098, lon=-87.5692, month=10)
    # WIMP_SCRAPER.get_season(lat=33.2098, lon=-87.6692, month=10)
    WIMP_SCRAPER.get_season(lat=30, lon=-90, month=10)
