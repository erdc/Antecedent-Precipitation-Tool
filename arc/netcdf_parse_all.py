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
##  ------------------------------- ##
##    Last Edited on: 2021-12-28    ##
##  ------------------------------- ##
######################################

# Import Standard Libraries
import os
import math
import time
from datetime import datetime


# Import 3rd Party Libraries
import netCDF4
import numpy
from geopy.distance import great_circle
import pandas
from dask.distributed import Client


def tunnel_fast(latvals, lonvals, lat0, lon0):
    '''
    Find closest point in a set of (lat,lon) points to specified point
    latvar - 2D latitude variable from an open netCDF dataset
    lonvar - 2D longitude variable from an open netCDF dataset
    lat0,lon0 - query point
    Returns iy,ix such that the square of the tunnel distance
    between (latval[it,ix],lonval[iy,ix]) and (lat0,lon0)
    is minimum.
    '''
    rad_factor = math.pi/180.0 # for trignometry, need angles in radians
    # Read latitude and longitude from file into numpy arrays
    latvals = latvals * rad_factor
    lonvals = lonvals * rad_factor
    #ny, nx = latvals.shape
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    # Compute numpy arrays for all values, no loops
    clat,clon = numpy.cos(latvals),numpy.cos(lonvals)
    slat,slon = numpy.sin(latvals),numpy.sin(lonvals)
    delX = numpy.cos(lat0_rad)*numpy.cos(lon0_rad) - clat*clon
    delY = numpy.cos(lat0_rad)*numpy.sin(lon0_rad) - clat*slon
    delZ = numpy.sin(lat0_rad) - slat;
    dist_sq = delX**2 + delY**2 + delZ**2
    minindex_1d = dist_sq.argmin()  # 1D index of minimum element
    iy_min, ix_min = numpy.unravel_index(minindex_1d, latvals.shape)
    x = numpy.unravel_index(minindex_1d, latvals.shape)
    return iy_min, ix_min


def get_closest_coordinates(dataset, lat, lon):
    print('Finding closest coordinates...')
    start_time = time.clock()
    latvals = dataset.variables['lat'][:]
    lonvals = dataset.variables['lon'][:]
    test_coords = (lat, lon)
    lowest_distance = 9999999
    for dataset_lat in latvals:
        for dataset_lon in lonvals:
            dataset_coords = (dataset_lat, dataset_lon)
            distance = great_circle(test_coords, dataset_coords).miles
            if distance < lowest_distance:
                lowest_distance = distance
                closest_lat = dataset_lat
                closest_lon = dataset_lon
    time_taken = time.clock() - start_time
    print('Found closest coordinates in {} seconds'.format(time_taken))
    print('Closest = {}, {}'.format(closest_lat, closest_lon))
    return closest_lat, closest_lon

#LAT, LON = get_closest_coordinates(DATASET, 38.5, -121.5)

def get_closest_coordinates_numpy(dataset, lat0, lon0):
    print('Creating complete list of coordinates...')
    lat_variable = dataset.variables['lat']
    lon_variable = dataset.variables['lon']
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
    print('Locating closest coordinate pair...')
    calc_start = time.clock()
    rad_factor = math.pi/180.0 # for trignometry, need angles in radians
    # Read latitude and longitude from file into numpy arrays
    latvals = lat_degree_vals_numpy * rad_factor
    lonvals = lon_degree_vals_numpy * rad_factor
    #ny, nx = latvals.shape
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    # Find nearest grid cell centroid using Haversine Distance
    r = 6371 #radius of the earth in km
    clat,clon = numpy.cos(latvals), numpy.cos(lonvals)
    slat,slon = numpy.sin(latvals), numpy.sin(lonvals)
    dlat = rad_factor * (lat_degree_vals_numpy - lat0)
    dlon = rad_factor * (lon_degree_vals_numpy - lon0)
    a = numpy.sin(dlat/2)**2 + numpy.cos(lat0_rad) * numpy.cos(latvals) * numpy.sin(dlon/2)**2
    # c = 2 * numpy.arcsin(numpy.sqrt(a))
    c = 2 * numpy.arctan2(numpy.sqrt(a), numpy.sqrt(1-a))
    distance = c * r # in units of km
    distance_min = numpy.amin(distance)*0.621371
    minindex_1d = distance.argmin()  # 1D index of minimum element
    closest_lat = lat_degree_vals_numpy[minindex_1d]
    closest_lon = lon_degree_vals_numpy[minindex_1d]
    time_taken = time.clock() - calc_start
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
    return closest_lat, lat_index, closest_lon, lon_index, distance_min, lat_array_list, lon_array_list

def get_nc_files(prcp_netcdf_folder, station_count_netcdf_folder):
    # netcdf_folder = r'\\coe-spknv001sac.spk.ds.usace.army.mil\EGIS_GEOMATICS\Regulatory\BaseData\Climatology\nclimdivd\nclimdivd-alpha-nc'
    nc_dates_and_files = []
    # years = os.listdir(netcdf_folder)
    first_year = datetime(1951,1,1)
    current_year = datetime.now()
    years = pandas.date_range(first_year,current_year,freq='AS').year.tolist()
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    for year in years:
        year = str(year)
        for month in months:
            date = '{0}{1}'.format(year,month)
            # create the paths to the precip netcdfs
            prcp_file = 'ncdd-{0}-grd-scaled.nc'.format(date)
            prcp_file_path = '{0}/{1}/{2}'.format(prcp_netcdf_folder,year,prcp_file)
            # create the paths to the station count netcdfs
            station_count_file = 'ncddsupp-{0}-obcounts.nc'.format(date)
            station_count_file_path = '{0}/{1}/{2}'.format(station_count_netcdf_folder,year,station_count_file)
            nc_dates_and_files.append([date, prcp_file_path, station_count_file_path])
                    # if pre == 'prcp-':
                    #     if post == '-grd-scaled.nc':
                    #         file_path = os.path.join(root, file_name)
                    #         nc_dates_and_files.append([date, file_path])
    nc_dates_and_files.sort(key=lambda x: x[0], reverse=False)
    # nc_files = []
    # for nc_date_and_file in nc_dates_and_files:
    #     nc_files.append(nc_date_and_file[1])
    return nc_dates_and_files

class get_point_history(object):

    def __init__(self, lat, lon):
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

    def __call__(self):
        # this was Joseph trying to summarize the station count data
        # list_of_monthly_minimum_station_counts = []
        # list_of_monthly_median_station_counts = []
        # list_of_monthly_mean_station_counts = []
        # list_of_monthly_maximum_station_counts = []
        print('Getting complete PRCP history for ({}, {})...'.format(self.lat, self.lon))
        netcdf_precip_folder = r'https://www.ncei.noaa.gov/thredds/dodsC/nclimgrid-daily'
        netcdf_station_count_folder = r'https://www.ncei.noaa.gov/thredds/dodsC/nclimgrid-daily-auxiliary'
        self.nc_files = get_nc_files(netcdf_precip_folder, netcdf_station_count_folder)
        # with Client() as client:
        # client = Client()
        num_datasets = len(self.nc_files)
        current_dataset = 0
        for nc_file in self.nc_files:
            print(nc_file)
            current_dataset += 1
            try:
                # Open dataset and get variables / basic info
                prcp_dataset = netCDF4.Dataset(nc_file[1], 'r')
                prcp = prcp_dataset.variables['prcp']
                timevar = prcp_dataset.variables['time']
                timeunits = timevar.units
                times = timevar[:]
                # Open dataset and get variables / basic info
                station_count_dataset = netCDF4.Dataset(nc_file[2], 'r')
                station_count = station_count_dataset.variables['cntp']

                # Find closest Lat/Lon and set export path
                if self.closest_lat is None or self.closest_lon is None:
                    self.closest_lat, self.lat_index, self.closest_lon, self.lon_index, self.distance, lat_array_list, lon_array_list = get_closest_coordinates_numpy(prcp_dataset, self.lat, self.lon)
                    print('Closest coordinates in dataset = {}, {}'.format(self.closest_lat, self.closest_lon))
                    query_coords = (self.lat, self.lon)
                    grid_coords = (self.closest_lat, self.closest_lon)
                    # distance = great_circle(query_coords, grid_coords).miles # calculations in get_closest_coordinates_numpy() seem to replicate this function
                    print('Distance to center of grid = {} miles'.format(self.distance))
                    print('Reading values from {} netCDF datasets...'.format(num_datasets))
                # Collect/print/write relevant values
                for x_time in range(len(times)):
                    prcp_val = prcp[x_time, self.lat_index, self.lon_index]
                    self.total_rows += 1
                    if str(prcp_val) != '--':
                        self.data_rows += 1
                        self.prcp_values.append(prcp_val)
                        time_val = netCDF4.num2date(times[x_time], timeunits)
                        t_stamp = pandas.Timestamp(time_val)
                        self.timestamps.append(t_stamp)
                    else:
                        self.blank_rows += 1
                    station_count_val = station_count[x_time, self.lat_index, self.lon_index]
                    self.station_count_values.append(station_count_val)
                print(prcp_val)
                print(station_count_val)
            except Exception as F:
                pass
                # print('----------')
                # print('----EXCEPTION!!!------')
                # print('----------')
                # print('Error processing dataset {} of {}'.format(current_dataset, num_datasets))
                # dataset_name = os.path.split(nc_file)[1]
                # print('Dataset name = {}'.format(dataset_name))
                # print(str(F))
                # print('----------')
                # print('----EXCEPTION!!!------')
            # # Open dataset and get variables / basic info
            # station_count_nc_file = os.path.join(netcdf_station_count_folder,"ncddsupp-{0}-obcounts.nc".format(nc_file[0]))
            # dataset = netCDF4.Dataset(station_count_nc_file, 'r')
            # station_count = dataset.variables['cntp']
            # # Collect/print/write relevant values
            # for x_time in range(len(times)):
            #     station_count_val = station_count[x_time, self.lat_index, self.lon_index]
            #     self.station_count_values.append(station_count_val)
        # this was Joseph tring to summarize the station count data
            # list_of_monthly_station_counts = []
            # vals = station_count[:].tolist()
            # for val in vals:
            #     for i in val:
            #         vals_filtered = list(filter(lambda j: j is not None, i))
            #         list_of_monthly_station_counts.extend(vals_filtered)
            #
            # list_of_monthly_station_counts = sorted(list_of_monthly_station_counts)
            #
            # minimum_monthly_station_count = min(list_of_monthly_station_counts)
            # list_of_monthly_minimum_station_counts.append(minimum_monthly_station_count)
            #
            # median_monthly_station_count = numpy.median(list_of_monthly_station_counts)
            # list_of_monthly_median_station_counts.append(median_monthly_station_count)
            #
            # mean_monthly_station_count = sum(list_of_monthly_station_counts)/len(list_of_monthly_station_counts)
            # list_of_monthly_mean_station_counts.append(mean_monthly_station_count)
            #
            # max_monthly_station_count = max(list_of_monthly_station_counts)
            # list_of_monthly_maximum_station_counts.append(max_monthly_station_count)
            # print(nc_file)
            # for val in vals:
            #     if isinstance(val, int):
            #         vals_filtered.append(val)
            #     else:
            #         pass
                # print(vals_filtered)


                    # if str(val) != "--":
                    #     print(val)
        # date_range = pandas.date_range(start='1/1/1989', end='1/31/2022', freq='MS')
        # print(date_range)
        # data = {'date': date_range,
        #         'Minimum': list_of_monthly_minimum_station_counts,
        #         'Median': list_of_monthly_median_station_counts,
        #         'Mean': list_of_monthly_mean_station_counts,
        #         'Maximum': list_of_monthly_maximum_station_counts}
        # df = pandas.DataFrame(data)
        # df = df.set_index('date')
        #
        # # getting a line plot of the montlhy data
        # output_path = r"C:\Users\RDCHLJLG\Desktop\line_plot.tif"
        # line_plot = df.plot(figsize=(7,5), kind='line', lw=2, colormap='jet')
        # line_plot.set_xlabel("")
        # line_plot.set_ylabel("Number of GHCN Stations")
        # line_plot.figure.savefig(output_path)
        print('----------')
        print('')
        print('All datasets processed.')
        # Convert to pandas dataframe
        print('Converting data to TimeSeries format...')
        self.entire_precip_ts = pandas.Series(data=self.prcp_values,
                                       index=self.timestamps,
                                       dtype="float64",
                                       name='value')
        self.entire_station_count_ts = pandas.Series(data=self.station_count_values,
                                       index=self.timestamps,
                                       dtype="float64",
                                       name='value')
        print('Conversion complete.')
        # Print and log summary stats
        print('')
        print('-'*119)
        print('Total rows = {}'.format(self.total_rows))
        print('Rows with data = {}'.format(self.data_rows))
        print('Blank rows = {}'.format(self.blank_rows))
        print('-'*119)
        return


if __name__ == '__main__':
    COORD_LIST = []
    COORD_LIST.append([36.5, -121.5])
##    COORD_LIST.append([39.1122, -119.7603])
##    COORD_LIST.append([46.925686, -117.683027])
##    COORD_LIST.append([46.925718, -117.682981])
##    COORD_LIST.append([37.197776, -117.813483])
##    COORD_LIST.append([35.1919, -80.6567])
##    COORD_LIST.append([49.354, -95.0625]) # Complete History
##    COORD_LIST.append([38.790289, -120.798243]) # Kelsey
##    COORD_LIST.append([38.5, -121.5]) # Empty in previous tests, complete 03/12/2018
##    COORD_LIST.append([36.5, -121.5]) # Sparse Stations
    list_of_instances = []
    for COORDS in COORD_LIST:
        instance = get_point_history(COORDS[0], COORDS[1])
        instance()
        list_of_instances.append(instance)
    input("STALLING...")
