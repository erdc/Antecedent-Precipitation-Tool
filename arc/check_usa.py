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
##     Copyright: Jason Deters      ##
##                                  ##
##    Written by: Jason Deters      ##
##     Edited by: Chase Hamilton    ##
##  Rewritten by: Chris French      ##
##     Edited by: Stephen Brown     ##
##  ------------------------------- ##
##    Last Edited on:  2025-07-09   ##
##  ------------------------------- ##
######################################

"""Script facilitates testing a point against a shapefile of the USA boundary"""

# Import Standard Libraries
import os

# External Libraries
import geopandas as gpd

# Internal Imports
try:
    from arc.utils import find_file_or_dir
except:
    from utils import find_file_or_dir


def main(lat, lon):
    # Load the shapefile
    usa_path = find_file_or_dir(os.getcwd(), "tl_2023_us_state.shp")
    usa_gdf = gpd.read_file(usa_path)

    # Create a GeoSeries from the latitude and longitude
    point_geo = gpd.points_from_xy([lon], [lat], crs="EPSG:4326")

    # Check if the point intersects with any of the states
    is_within_usa = point_geo.intersects(usa_gdf.geometry).any()

    return is_within_usa


if __name__ == "__main__":
    # CA
    LAT = 38.544418
    LON = -120.812989
    print("California")
    print(main(lat=LAT, lon=LON))
    # AK
    LAT = 67.261448
    LON = -153.100011
    print("Alaska")
    print(main(lat=LAT, lon=LON))
    # HUC12 Test
    LAT = 38.4008283
    LON = -120.8286800
    print("HUC Sample Point")
    print(main(lat=LAT, lon=LON))
    LAT = 60
    LON = -106
    print("Canada")
    print(main(lat=LAT, lon=LON))

    """     
    # Locations used in 2024 Gutenson et al JAWRA Article  https://onlinelibrary.wiley.com/doi/full/10.1111/1752-1688.13189
    #  
    # USGS CHOPTANK RIVER NEAR GREENSBORO, MD  https://waterdata.usgs.gov/monitoring-location/01491000/
    LAT = 38.99719444
    LON = -75.7858056
    print("CHOPTANK RIVER NEAR GREENSBORO, MD")
    print(main(lat=LAT, lon=LON))

    # USGS CEDAR CREEK NEAR CEDARVILLE, IN  https://waterdata.usgs.gov/monitoring-location/04180000/
    LAT = 41.218938
    LON = -85.0763589
    print("CEDAR CREEK NEAR CEDARVILLE, IN")
    print(main(lat=LAT, lon=LON))

    # USGS ROARING FORK RIVER AT GLENWOOD SPRINGS, CO.  https://waterdata.usgs.gov/monitoring-location/09085000/
    LAT = 39.54666667
    LON = -107.3308333
    print("ROARING FORK RIVER AT GLENWOOD SPRINGS, CO.")
    print(main(lat=LAT, lon=LON))

    # USGS DONNER UND BLITZEN RIVER NR FRENCHGLEN OR  https://waterdata.usgs.gov/monitoring-location/09085000/
    LAT = 42.7908333
    LON = -118.8675
    print("DONNER UND BLITZEN RIVER NR FRENCHGLEN OR")
    print(main(lat=LAT, lon=LON)) 
    
    """
