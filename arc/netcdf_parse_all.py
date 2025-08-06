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
##         netcdf_parse_all.py      ##
##  ------------------------------- ##
##      Writen by: Jason Deters     ##
##      Edited by: Joseph Gutenson  ##
##      Edited by: Chase Hamilton   ##
##      Edited by: Chris French     ##
##  ------------------------------- ##
##    Last Edited on:  2025-07-09   ##
##  ------------------------------- ##
######################################

import math
import multiprocessing
import os
import traceback
from datetime import datetime, timedelta

import netCDF4
import numpy
import pandas
import requests
from dateutil import relativedelta

RECENT_CHECK = False


class _NetCDFFileHandler:
    def __init__(self, url):
        # Extract filename from URL (everything before last '/')
        self.url = url
        self.filename = os.path.basename(url.rstrip("/")) + ".tmp"
        self.file_path = None

    def __enter__(self):
        # Download file
        response = requests.get(self.url)
        if not response.ok:
            raise Exception(f"Failed to download file: {response.status_code}")

        # Write to temporary file
        with open(self.filename, "wb") as f:
            f.write(response.content)
        self.file_path = os.path.abspath(self.filename)

        # Open NetCDF dataset
        self.dataset = netCDF4.Dataset(self.file_path, "r")
        return self.dataset

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close dataset
        if hasattr(self, "dataset"):
            self.dataset.close()

        # Delete temporary file
        if self.file_path and os.path.exists(self.file_path):
            os.remove(self.file_path)


def tunnel_fast(latvals, lonvals, lat0, lon0):
    """
    Find closest point in a set of (lat,lon) points to specified point
    latvar - 2D latitude variable from an open netCDF dataset
    lonvar - 2D longitude variable from an open netCDF dataset
    lat0,lon0 - query point
    Returns iy,ix such that the square of the tunnel distance
    between (latval[it,ix],lonval[iy,ix]) and (lat0,lon0)
    is minimum.
    """
    rad_factor = math.pi / 180.0  # for trignometry, need angles in radians
    # Read latitude and longitude from file into numpy arrays
    latvals = latvals * rad_factor
    lonvals = lonvals * rad_factor
    # ny, nx = latvals.shape
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    # Compute numpy arrays for all values, no loops
    clat, clon = numpy.cos(latvals), numpy.cos(lonvals)
    slat, slon = numpy.sin(latvals), numpy.sin(lonvals)
    delX = numpy.cos(lat0_rad) * numpy.cos(lon0_rad) - clat * clon
    delY = numpy.cos(lat0_rad) * numpy.sin(lon0_rad) - clat * slon
    delZ = numpy.sin(lat0_rad) - slat
    dist_sq = delX**2 + delY**2 + delZ**2
    minindex_1d = dist_sq.argmin()  # 1D index of minimum element
    iy_min, ix_min = numpy.unravel_index(minindex_1d, latvals.shape)
    x = numpy.unravel_index(minindex_1d, latvals.shape)
    return iy_min, ix_min


def get_closest_coordinates_numpy(dataset, lat0, lon0):
    print("Creating complete list of coordinates...")
    lat_variable = dataset.variables["lat"]
    lon_variable = dataset.variables["lon"]
    lat_degree_vals = lat_variable[:]
    lon_degree_vals = lon_variable[:]
    lat_array_list = []
    lon_array_list = []
    for dataset_lat in lat_degree_vals:
        for dataset_lon in lon_degree_vals:
            lat_array_list.append(dataset_lat)
            lon_array_list.append(dataset_lon)
    lat_degree_vals_numpy = numpy.array(lat_array_list)
    lon_degree_vals_numpy = numpy.array(lon_array_list)
    print("Locating closest coordinate pair...")
    rad_factor = math.pi / 180.0  # for trignometry, need angles in radians

    # Read latitude from file into numpy arrays
    latvals = lat_degree_vals_numpy * rad_factor
    lat0_rad = lat0 * rad_factor

    # Find nearest grid cell centroid using Haversine Distance
    r = 6371  # radius of the earth in km
    dlat = rad_factor * (lat_degree_vals_numpy - lat0)
    dlon = rad_factor * (lon_degree_vals_numpy - lon0)
    a = (
        numpy.sin(dlat / 2) ** 2
        + numpy.cos(lat0_rad) * numpy.cos(latvals) * numpy.sin(dlon / 2) ** 2
    )
    c = 2 * numpy.arctan2(numpy.sqrt(a), numpy.sqrt(1 - a))
    distance = c * r  # in units of km
    distance_min = numpy.amin(distance) * 0.621371
    minindex_1d = distance.argmin()  # 1D index of minimum element
    closest_lat = lat_degree_vals_numpy[minindex_1d]
    closest_lon = lon_degree_vals_numpy[minindex_1d]

    # Convert to positions
    n = 0
    for lat_degree_val in lat_degree_vals:
        if lat_degree_val == closest_lat:
            lat_index = n
        n += 1
    n = 0
    for lon_degree_val in lon_degree_vals:
        if lon_degree_val == closest_lon:
            lon_index = n
        n += 1
    return (
        closest_lat,
        lat_index,
        closest_lon,
        lon_index,
        distance_min,
        lat_array_list,
        lon_array_list,
    )


### Check if connectivity exists to the NOAA THREDDS server
### Only check once every three minutes, to make sure we're not overloading them


def check_thredds_status():
    global RECENT_CHECK

    if RECENT_CHECK:
        if datetime.now() - RECENT_CHECK[0] < timedelta(minutes=5):
            return RECENT_CHECK[1]

    url = "https://www.ncei.noaa.gov/"

    try:
        response = requests.get(url, allow_redirects=True, stream=True)
        response.raise_for_status()
        good = True
    except requests.exceptions.RequestException as e:
        print(f"Error checking THREDDS status: {e}")
        good = False

    RECENT_CHECK = (datetime.now(), good)
    return good


def get_nc_files(
    netcdf_precip_url,
    netcdf_station_count_url,
    normal_period_data_start_date,
    actual_data_end_date,
):
    nc_dates_and_files = []
    # create a unique list of month-day pairs to use when downloading gridded data
    months = []
    query_dates = []
    while normal_period_data_start_date <= actual_data_end_date:
        if [
            normal_period_data_start_date.year,
            normal_period_data_start_date.month,
        ] not in query_dates:
            months.append(normal_period_data_start_date.month)
            query_dates.append(
                [
                    normal_period_data_start_date.year,
                    normal_period_data_start_date.month,
                ]
            )
        normal_period_data_start_date += timedelta(days=1)

    # loop through the days to create the inputs that will feed workers in the multiprocessing step
    for query_date in query_dates:
        year = str(query_date[0])
        month = str(query_date[1])
        month = month.zfill(2)
        date = "{0}{1}".format(year, month)

        station_count_file = f"ncddsupp-{date}-obcounts.nc"
        station_count_file_path = (
            f"{netcdf_station_count_url}/{year}/{station_count_file}"
        )

        # create the paths to the precip netcdfs
        currentMonth = datetime.now().month
        currentYear = datetime.now().year
        currentYearMonth = datetime(currentYear, currentMonth, 1)
        testYearMonth = datetime(int(year), int(month), 1)
        # if the month is within two months of the current month, file name will be different
        delta = relativedelta.relativedelta(currentYearMonth, testYearMonth)
        delta_months = delta.months + (delta.years * 12)
        # if within 2 months, the preliminary grid may still apply
        if delta_months <= 2:
            prcp_file = f"ncdd-{date}-grd-prelim.nc"
        else:
            prcp_file = f"ncdd-{date}-grd-scaled.nc"
        prcp_file_path = f"{netcdf_precip_url}/{year}/{prcp_file}"

        nc_dates_and_files.append([date, prcp_file_path, station_count_file_path])

    nc_dates_and_files.sort(key=lambda x: x[0], reverse=False)
    return nc_dates_and_files


def _process_precip_data(args):
    """Helper function to process data from a single precip path"""
    # Split input tuple into constituents
    nc_file = args[0]
    lat_index = args[1]
    lon_index = args[2]
    precip_path = args[3]

    # Get precip file date from nc_file list
    file_date = nc_file[0]

    # Instantiate empty values
    total_rows = 0
    data_rows = 0
    blank_rows = 0

    prcp_values = []
    timestamps = []
    station_count_values = []

    with _NetCDFFileHandler(url=precip_path) as prcp_dataset:
        with _NetCDFFileHandler(url=nc_file[2]) as station_count_dataset:
            prcp = prcp_dataset.variables["prcp"]
            timevar = prcp_dataset.variables["time"]
            timeunits = timevar.units
            times = timevar[:]

            # Open station count dataset
            station_count = station_count_dataset.variables["cntp"]

            # Pull relevant data subset from precip & station count datasets
            prcp_vals = prcp[:, lat_index, lon_index]
            station_count_vals = station_count[:, lat_index, lon_index]

            total_rows = 0
            data_rows = 0
            blank_rows = 0
            prcp_values = []
            timestamps = []
            station_count_values = []

            # Process data and update variables
            for x_time in range(len(times)):
                total_rows += 1
                prcp_val = prcp_vals[x_time]
                if str(prcp_val) != "--":
                    data_rows += 1
                    prcp_values.append(prcp_val)
                else:
                    blank_rows += 1

                time_val = netCDF4.num2date(times[x_time], timeunits)
                timestamp = pandas.Timestamp(str(time_val))
                timestamps.append(timestamp)

                station_count_val = station_count_vals[x_time]
                station_count_values.append(station_count_val)

            return (
                file_date,
                prcp_values,
                timestamps,
                station_count_values,
                total_rows,
                data_rows,
                blank_rows,
            )


def nc_file_worker(args):
    # Split input tuple into constituents
    nc_file = args[0]
    lat_index = args[1]
    lon_index = args[2]
    precip_paths = [
        nc_file[1],
        nc_file[1][:-9] + "prelim.nc",
        nc_file[1][:-9] + "scaled.nc",
    ]
    # test if the file exist within NOAA's naming convention
    for precip_path in precip_paths:
        try:
            p_args = (nc_file, lat_index, lon_index, precip_path)
            return _process_precip_data(p_args)
        except Exception as e:
            print(f"Error processing {precip_path}: {str(e)}")
    print(
        "It appears the nClimGrid-Daily THREDDS data service is experiencing issues, please try again...\n"
    )
    print(
        "If the problems persists, please contact the nClimGrid-Daily team at ncei.grids@noaa.gov\n"
    )


class get_point_history(object):

    def __init__(self, lat, lon, normal_period_data_start_date, actual_data_end_date):
        self.lat = lat
        self.lon = lon
        self.closest_lat = None
        self.closest_lon = None
        self.lat_index = None
        self.lon_index = None
        self.nc_files = None
        self.csv_export_path = None
        self.prcp_data = []
        self.timestamps = []
        self.prcp_values = []
        self.station_count_values = []
        self.total_rows = 0
        self.blank_rows = 0
        self.data_rows = 0
        self.entire_precip_ts = None
        self.entire_station_count_ts = None
        self.distance = 0
        self.normal_period_data_start_date = normal_period_data_start_date
        self.actual_data_end_date = actual_data_end_date

    def __call__(self):
        print(
            "Getting complete PRCP history for ({}, {})...".format(self.lat, self.lon)
        )
        netcdf_precip_url = (
            r"https://www.ncei.noaa.gov/thredds/fileServer/nclimgrid-daily"
        )
        netcdf_station_count_url = (
            r"https://www.ncei.noaa.gov/thredds/fileServer/nclimgrid-daily-auxiliary"
        )
        self.nc_files = get_nc_files(
            netcdf_precip_url,
            netcdf_station_count_url,
            self.normal_period_data_start_date,
            self.actual_data_end_date,
        )

        # Open first dataset and get variables / basic info
        nc_file = self.nc_files[0]

        try:
            with _NetCDFFileHandler(url=nc_file[1]) as prcp_dataset:
                (
                    self.closest_lat,
                    self.lat_index,
                    self.closest_lon,
                    self.lon_index,
                    self.distance,
                    lat_array_list,
                    lon_array_list,
                ) = get_closest_coordinates_numpy(prcp_dataset, self.lat, self.lon)

        except Exception as e:
            error_message = (
                f"Error: {str(e)}\n\nDetailed Error:\n{traceback.format_exc()}"
            )
            print(error_message)

        # Create list of all tasks
        process_queue = []

        for nc_file in self.nc_files:
            process_queue.append((nc_file, self.lat_index, self.lon_index))

        process_number = min(multiprocessing.cpu_count() - 2, 12)
        chunksize = process_number

        # Create empty results dictionary
        mp_results = {}

        # Chunk and loop through chunks, necessary to not run into blocking issues on Windows
        print("Downloading gridded time series data...\n")

        for i in range(0, len(process_queue), chunksize):
            chunk_start = datetime.now()

            start_index = i
            end_index = min(i + chunksize, len(process_queue))

            partial_queue = process_queue[start_index:end_index]

            mp_pool = multiprocessing.Pool(processes=process_number)

            for result in mp_pool.map(nc_file_worker, partial_queue):
                (
                    file_date,
                    prcp_values,
                    timestamps,
                    station_count_values,
                    total_rows,
                    data_rows,
                    blank_rows,
                ) = result
                mp_results[file_date] = (
                    prcp_values,
                    timestamps,
                    station_count_values,
                    total_rows,
                    data_rows,
                    blank_rows,
                )

                self.prcp_values += prcp_values
                self.timestamps += timestamps
                self.station_count_values += station_count_values
                self.total_rows += total_rows
                self.data_rows += data_rows
                self.blank_rows += blank_rows

            mp_pool.close()
            mp_pool.join()

            start_date = partial_queue[0][0][0]
            start_date = start_date[:4] + "-" + start_date[4:]
            end_date = partial_queue[-1][0][0]
            end_date = end_date[:4] + "-" + end_date[4:]

            print(
                "{} through {}: {}".format(
                    start_date, end_date, datetime.now() - chunk_start
                )
            )

        # remove "masked" returns from station count results
        self.station_count_values = [x for x in self.station_count_values if x >= 0]

        print("----------")
        print("")
        print("All datasets processed.")
        print("Converting data to TimeSeries format...")
        self.entire_precip_ts = pandas.Series(
            data=self.prcp_values, index=self.timestamps, dtype="float64", name="value"
        )
        self.entire_station_count_ts = pandas.Series(
            data=self.station_count_values,
            index=self.timestamps,
            dtype="float64",
            name="value",
        )
        print("Conversion complete.")
        print("")
        print("-" * 119)
        print("Total rows = {}".format(self.total_rows))
        print("Rows with data = {}".format(self.data_rows))
        print("Blank rows = {}".format(self.blank_rows))
        print("-" * 119)
        return


if __name__ == "__main__":
    check_thredds_status()

    start_date = datetime.strptime("2016-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2019-12-31", "%Y-%m-%d")
    tester = get_point_history(
        lat=30,
        lon=-90,
        normal_period_data_start_date=start_date,
        actual_data_end_date=end_date,
    )
    tester()
