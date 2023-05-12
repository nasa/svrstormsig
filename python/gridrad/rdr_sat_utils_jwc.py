#+
# Name:
#     rdr_sat_utils_jwc.py
# Purpose:
#     This is a script to with functions for matching up radar and satellite data. 
# Calling sequence:
#     from rdr_sat_utils_jwc import *
#     rdr_sat_utils_jwc.rdr_sat_utils_jwc()
# Input:
#     None.
# Functions:
#     rdr_sat_utils_jwc                             : Main function to update the radar csv files with satellite data.
#     dir_maker                                     : Makes a new directory and parent directories if they do not already exist.
#     make_gif                                      : Makes a GIF from multiple image file inputs.
#     plot_pic                                      : Plots satellite imagery with the option to overlay AACP locations from radar data.
#     vincenty_dist_calc                            : This function leverages geopy's vincenty distance.  It is vectorized as "vfunc" so that it can be used in numpy matrix calls.
#     geodesic_dist_calc                            : This function leverages geopy's geodesic distance.  It is vectorized as "vfunc" so that it can be used in numpy matrix calls.
#     sat_rdr_marry                                 : Takes a satellite data file and a lat long from radar data and finds the closest pixel lat / long in the satellite data to the 
#                                                     radar lat / long coordinate.
#     match_sat_data_to_datetime                    : Finds satellite file closest to a specific radar date time.
#     get_sat_files                                 : Reads all satellite imagery in a directory, adds each filename to a pandas dataframe, and creates a column in the dataframe with an extracted date and time 
#                                                     from the filename.
#     sat_row_op                                    : A function used for error handling for apply calls to sat_rdr_marry() for satellite imagery.
#     rdr_parallax_correct                          : Reads radar data from a CSV file and then finds closest match for satellite data files stored in sat_dir.
#     rdr_csv_update2                               : Updates radar csv file with original radar CSV data as well as matched satellite data files for each line and the index and lat / long on the 
#                                                     sat image(s) closest to each radar point of interest along with distances between the closest sat point and the radar point of interest  
#     get_plume_idx_list                            : Gets a list of plume indices for a specified NetCDF satellite imagery file.
#     mult_pic_plot                                 : Plots all satellite pictures in a directory.
#     gridrad_read_sparse_eth_ncdf                  : Function to read sparsely stored ETH file.
#     gridrad_read_calc_zrel                        : Calls function that reads ETH and tropopause and calculates tropopause-relative altitude for a given reflectivity threshold
#     convert_longitude                             : Function to convert longitude -180 to 180 into 0-360 or visa versa.
#     index_of_nearest                              : Find the nearest index to value in the array of interest
#     extract_us_lat_lon_region                     : Uses switch in order to return the latitude and longitude bounds of US states and regions
#     extract_us_lat_lon_region_parallax_correction : Uses switch in order to return the latitude and longitude parallax corrections of US states and regions
# Output:
#     Creates updated radar csv file (provided by Cameron Homeyer) with radar data matched with satellite data.
# Keywords:
#     date_str        : STRING specifying date to update the csv file. (YearMonthDay format)
#     sector          : STRING specifying sector to update with GOES data.
#                       DEFAULT = 'M1' -> mesoscale sector 1.
#     rdr_inroot      : STRING path to radar data directories. Inout radar csv file and output location of updated csv file
#                       DEFAULT = '../../../misc-data0/GridRad/NEXRAD/'
#     sat_inroot      : STRING path to the combined netCDF GLM, GOES visible and GOES IR data directories
#                       DEFAULT = '../../../goes-data/combined_nc_dir/'
#     date_str        : STRING date to update CSV file. DEFAULT = '20190430-0501'
#     sector          : STRING GOES scan data sector to update CSV file. DEFAULT = 'M1' -> mesoscale sector 1. (ex. 'M2', 'C', 'F')
#     lat_correction  : FLOAT value to parallax correct radar data latitudes to match with satellite data. DEFAULT = 0.12
#     lon_correction  : FLOAT value to parallax correct radar data longitudes to match with satellite data. DEFAULT = 0.05
#     write_gcs       : IF keyword set (True), write everything to the google cloud platform.
#                       DEFAULT = True
#     use_local       : IF keyword set (True), use files stored in local directory.
#                       DEFAULT = False -> use files stored in GCP.
#     del_local       : IF keyword set (True), delete local copies of files after writing them to google cloud. Only does anything if write_gcs = True.
#                       DEFAULT = False
#     rdr_bucket      : Google cloud storage bucket to read the radar csv files. 
#                       DEFAULT = 'misc-data0'
#     sat_bucket      : Google cloud storage bucket to read combined IR/GLM/VIS netCDF files.
#                       DEFAULT = 'ir-vis-sandwhich'
#     verbose         : BOOL keyword to specify whether or not to print verbose informational messages.
#                       DEFAULT = True which implies to print verbose informational messages
# Author and history:
#     caliles                  2018-04-18.
#     John W. Cooney           2021-07-16. (Adapted code from rdr_sat_utils.py)
#
#-

#### Environment Setup ###
# Package imports
from scipy.io import netcdf
from netCDF4 import Dataset
from geopy.distance import geodesic
import numpy as np 
import time
import pandas as pd
import glob
import multiprocessing as mp
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle
from ast import literal_eval
import imageio
import os
import re
from netCDF4 import Dataset
from xarray import open_dataset
import sys 
#sys.path.insert(1, '../')
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.getcwd()))
from new_model.gcs_processing import write_to_gcs, load_csv_gcs, download_ncdf_gcs, list_gcs

def rdr_sat_utils_jwc(date_str       = '20190430-0501', 
                      sector         = 'M1',
                      lat_correction = None, #0.12, # This needs to change based on the geographic focus of the satellite.  0.12 for most datasets, 0.09 for 5 Apr 2017
                      lon_correction = None, #0.05, # This needs to change based on the geographic focus of the satellite.  0.05 for most datasets, -0.02 for 5 Apr 2017
                      rdr_inroot     = '../../../misc-data0/GridRad/NEXRAD/',
                      sat_inroot     = '../../../goes-data/combined_nc_dir/', 
                      rdr_bucket     = 'misc-data0',
                      sat_bucket     = 'ir-vis-sandwhich', 
                      use_local      = False, write_gcs = True, del_local = True, 
                      verbose        = True):
  '''
  Main function to update the radar csv files with satellite data.
  Args:
    date_str        : STRING specifying date to update the csv file. (YearMonthDay format)
    sector          : STRING specifying sector to update with GOES data.
                      DEFAULT = 'M1' -> mesoscale sector 1.
    rdr_inroot      : STRING path to radar data directories. Inout radar csv file and output location of updated csv file
                      DEFAULT = '../../../misc-data0/GridRad/NEXRAD/'
    sat_inroot      : STRING path to the combined netCDF GLM, GOES visible and GOES IR data directories
                      DEFAULT = '../../../goes-data/combined_nc_dir/'
    date_str        : STRING date to update CSV file. DEFAULT = '20190430-0501'
    sector          : STRING GOES scan data sector to update CSV file. DEFAULT = 'M1' -> mesoscale sector 1. (ex. 'M2', 'C', 'F')
    lat_correction  : FLOAT value to parallax correct radar data latitudes to match with satellite data. DEFAULT = 0.12
    lon_correction  : FLOAT value to parallax correct radar data longitudes to match with satellite data. DEFAULT = 0.05
    write_gcs       : IF keyword set (True), write everything to the google cloud platform.
                      DEFAULT = True
    use_local       : IF keyword set (True), use files stored in local directory.
                      DEFAULT = False -> use files stored in GCP.
    del_local       : IF keyword set (True), delete local copies of files after writing them to google cloud. Only does anything if write_gcs = True.
                      DEFAULT = False
    rdr_bucket      : Google cloud storage bucket to read the radar csv files. 
                      DEFAULT = 'misc-data0'
    sat_bucket      : Google cloud storage bucket to read combined IR/GLM/VIS netCDF files.
                      DEFAULT = 'ir-vis-sandwhich'
    verbose         : BOOL keyword to specify whether or not to print verbose informational messages.
                      DEFAULT = True which implies to print verbose informational messages
  '''
  
  start_time = time.time()  
  rdr_inroot = os.path.realpath(rdr_inroot)                                                                                                #Create link to real path so compatible with Mac
  sat_inroot = os.path.realpath(sat_inroot)                                                                                                #Create link to real path so compatible with Mac
  sat_dir    = os.path.join(sat_inroot, date_str)                                                                                          #Add date string to satellite root path directory
  if use_local == False:
    dir_maker(sat_dir)                                                                                                                     #Create output directory for satellite data if it does not already exist
    sat_files = list_gcs(comb_bucket_name, 'combined_nc_dir/' + date_str + '/', [sector + '_COMBINED_'], delimiter = '*/')                 #Extract list of satellite files from GCP storage bucket
    for g in sat_files: download_ncdf_gcs(sat_bucket, g, sat_dir)                                                                          #Download IR/VIS/GLM combined netCDF files from GCP bucket

  # Find domain bounds
  if lat_correction == None or lon_correction == None:
    sat_files = sorted(glob.glob(sat_dir + '/*_' + sector + '_*.nc'))                                                                      #Extract names of all of the GOES combined netCDF data files
    with Dataset(sat_files[0]) as f:       
      domain = [np.nanmin(f.variables['longitude'][0,:]), np.nanmin(f.variables['latitude'][0,:]), np.nanmax(f.variables['longitude'][0,:]), np.nanmax(f.variables['latitude'][0,:])]
    pc = estimate_sector_parallax(domain = domain)
    lon_correction = pc[0]
    lat_correction = pc[1]
 
  # Marry up IR and vis imagery with radar data.
  rdr_df = rdr_csv_update2(os.path.join(rdr_inroot, date_str + '.csv'), sat_dir, lat_correction, lon_correction, sector,                   #Update csv file
                           rdr_bucket = rdr_bucket, use_local = use_local)
  
#   rdr_df = rdr_csv_update('../../../data/'+cal_date+'/'+cal_date+'_tracks_subset_of_fields_Fixed.csv', '../../../data/'+cal_date+'/'+jul_date+'/IRNCDF/', '../../../data/'+cal_date+'/'+jul_date+'/VISNCDF/', lat_correction, lon_correction)
#   rdr_df.to_csv('../../../data/'+cal_date+'/update_rdr_data.csv', index=False)
  dir_maker(os.path.join(rdr_inroot, date_str, sector))                                                                                    #Create output directory for satellite data if it does not already exist
  rdr_df.to_csv(os.path.join(rdr_inroot, date_str, sector, 'update_rdr_data.csv'), index=False)
  
  if write_gcs == True and use_local == False:
      pref = re.split('misc-data0/', rdr_inroot)[1]
      write_to_gcs(rdr_bucket, os.path.join(pref, date_str, sector, 'update_rdr_data.csv'), os.path.join(rdr_inroot, date_str, sector, 'update_rdr_data.csv'), del_local = del_local)
  
  print("--- %s seconds ---" % (time.time() - start_time))
  
  # Uncomment to reload rdr_df data with tuples handled correctly.
  #rdr_df = pd.read_csv('../../data/' + cal_date + '/update_rdr_data.csv', converters={'IR_Min_Index': literal_eval, 'VIS_Min_Index': literal_eval})
  
  # Plot all satellite imagery with radar data AACPs superimposed on top.
  #mult_pic_plot('../../data/'+cal_date+'/'+jul_date+'/IRNCDF/', rdr_df, 'IR', '../../output/'+cal_date+'_IR/', 1, sector)
  #mult_pic_plot('../../data/'+cal_date+'/'+jul_date+'/VISNCDF/', rdr_df, 'VIS', '../../output/'+cal_date+'_VIS/', 1, sector)
  #print("--- %s seconds ---" % (time.time() - start_time))
  
  # Build GIF files for IR and visible data.

  #ir_files = []
  #vis_files = []
  #for n in range(50,60):
  #  ir_files.append('../../output/IR/IR_2017138_18'+str(n)+'26.nc.png')
  #  vis_files.append('../../output/VIS/VIS_2017138_18'+str(n)+'26.nc.png')
  #for n in range(0,10):
  #  ir_files.append('../../output/IR/IR_2017138_190'+str(n)+'26.nc.png')
  #  vis_files.append('../../output/VIS/VIS_2017138_190'+str(n)+'26.nc.png')
  #for n in range(10,27):
  #  ir_files.append('../../output/IR/IR_2017138_19'+str(n)+'26.nc.png')
  #  vis_files.append('../../output/VIS/VIS_2017138_19'+str(n)+'26.nc.png') 
  #make_gif(ir_files, 'ir.gif')
  #make_gif(vis_files, 'vis.gif')
  #print("--- %s seconds ---" % (time.time() - start_time))



def dir_maker(new_dir):
  '''
  Makes a new directory and parent directories if they do not already exist.
  Args:
    new_dir: a string representing the path for the new directory
  '''
  if not os.path.exists(new_dir):
    os.makedirs(new_dir)

def make_gif(files, out_file):
  '''
  Makes a GIF from multiple image file inputs.
  Args:
    files:     a list of all of the image files which will make up the GIF
    out_file:  the filename for the output GIF
  '''
  images = []
  for n in files:
    images.append(imageio.imread(n))
  imageio.mimsave(out_file, images)

def plot_pic(in_file, min_inds, overlay_tgt, out_file, pic_title, vmn=0, vmx=2700):
  '''
  TODO: fix the vmin, vmax stuff
  Plots satellite imagery with the option to overlay AACP locations from radar data.
  Args:
    in_file:     the input satellite imagery
    min_inds:    the indices of AACP locations on the imagery 
    overlay_tgt: boolean indicating user's desire to have AACP locations overlaid on imagery 
    out_file:    the file name for the output image
    pic_title:   the title to be added to the Matplotlib image
  '''
  f = Dataset(in_file, 'r')
  img_data = np.copy(np.asarray(f.variables['data'][0,:,:]))
  fig,ax = plt.subplots(1)
  img = ax.imshow(img_data, cmap=cm.gist_rainbow, vmin=vmn, vmax=vmx)
  fig.colorbar(img)
  ax.title.set_text(pic_title)
  if overlay_tgt:
    rad_size = np.floor(img_data.shape[0] / 10.0) # Adjust the size of the target radius based off of the pixel size of the input sat data.
    for n in min_inds:
      circ = Circle((n[1], n[0]),rad_size,fill=False,edgecolor='black')
      ax.add_patch(circ)
      x_array = np.array([n[1],n[1],n[1]-rad_size,n[1]+rad_size])
      y_array = np.array([n[0]-rad_size,n[0]+rad_size,n[0],n[0]])
      ax.plot(x_array[0:2], y_array[0:2], 'black')
      ax.plot(x_array[2:4], y_array[2:4], 'black')
  plt.savefig(out_file, bbox_inches='tight', dpi=250)
  plt.close()
  f.close()

def geodesic_dist_calc(img_lat, img_long, rdr_lat, rdr_long):
  '''
  This function leverages geopy's geodesic distance.  It is vectorized as "vfunc" so that it can be
  used in numpy matrix calls. Vincenty has been removed from geopy so use geodesic instead.
  Args:
    img_lat:   latitude at point on sat image (matrix of points)
    img_long:  longitude at point on sat image (matrix of points)
    rdr_lat:   latitude of point of interest for radar reading (single point)
    rdr_long:  longitude of point of interest for radar reading (single point)
  Returns:
    matrix of distances between input image point matrices and the input radar lat / long in kilometers
  '''
  data = np.zeros(img_lat.shape)
  for iy, la in enumerate(img_lat[:, 0]):
    for ix, lo in enumerate(img_long[0, :]):
      data[iy, ix] = geodesic((la, lo), (rdr_lat, rdr_long)).kilometers 
  return(data)
  

def vincenty_dist_calc(img_lat, img_long, rdr_lat, rdr_long):
  '''
  This function leverages geopy's vincenty distance.  It is vectorized as "vfunc" so that it can be
  used in numpy matrix calls.
  Args:
    img_lat:   latitude at point on sat image (matrix of points)
    img_long:  longitude at point on sat image (matrix of points)
    rdr_lat:   latitude of point of interest for radar reading (single point)
    rdr_long:  longitude of point of interest for radar reading (single point)
  Returns:
    matrix of distances between input image point matrices and the input radar lat / long in kilometers
  '''
  return vincenty((img_lat, img_long), (rdr_lat, rdr_long)).kilometers  

#vfunc = np.vectorize(vincenty_dist_calc) # Vectorize the geopy vincenty calculation formula for numpy ops.

def sat_rdr_marry(in_file, lat, long):
  '''
  Takes a satellite data file and a lat long from radar data and
  finds the closest pixel lat / long in the satellite data to the 
  radar lat / long coordinate.
  Args:
    in_file:  the netCDF file holding the satellite data
    lat:      the latitude specified in the radar file
    long:     the longitude specified in the radar file
  Returns:
    Tuple including:
      1) tuple of the index on the sat image which is closest to the lat and long input args
      2) distance (km) between the sat image point at min index and the lat and long input args
      3) image latitude of min index point specified above
      4) image longitude of min index point specified above 
  '''
  f = Dataset(in_file, 'r')  
 
  # Find longitudinal indices.
  north_long_idx = np.searchsorted(f.variables['longitude'][0,:], long)
  south_long_idx = np.searchsorted(f.variables['longitude'][-1,:], long)
  west_long_ind  = np.max([np.min([north_long_idx, south_long_idx]) - 1, 0])
  east_long_ind  = np.min([np.max([north_long_idx, south_long_idx]) + 1, f.variables['longitude'].shape[0]])
  # Find latitudinal indices.
  west_lat_idx  = np.searchsorted(np.negative(f.variables['latitude'][:,west_long_ind]), -lat)
  east_lat_idx  = np.searchsorted(np.negative(f.variables['latitude'][:,np.min([east_long_ind, f.variables['longitude'].shape[1]-1])]), -lat)
  north_lat_ind = np.max([np.min([west_lat_idx, east_lat_idx]) - 1, 0])
  south_lat_ind = np.min([np.max([west_lat_idx, east_lat_idx]) + 1, f.variables['latitude'].shape[1]])
  # Pre-allocate matrices for holding the radar lat and long.
  #TODO: can we get rid of the following four lines
  rdr_lat  = np.ones(f.variables['latitude'][north_lat_ind:south_lat_ind,west_long_ind:east_long_ind].shape)
  rdr_lat  = np.multiply(rdr_lat, lat)
  rdr_long = np.ones(f.variables['latitude'][north_lat_ind:south_lat_ind,west_long_ind:east_long_ind].shape)
  rdr_long = np.multiply(rdr_long, long)
  
  # Get the distances between the satellite and radar locations in our subset of indices.  We will find the minimum distance to locate the index of the 
  dists   = geodesic_dist_calc(np.asarray(f.variables['latitude'][north_lat_ind:south_lat_ind,west_long_ind:east_long_ind]),np.asarray(f.variables['longitude'][north_lat_ind:south_lat_ind,west_long_ind:east_long_ind]),lat,long)
  min_ind = np.unravel_index(dists.argmin(), dists.shape)
  min_ind = (north_lat_ind + min_ind[0], west_long_ind + min_ind[1]) # Need to account for the shrinking of the window for the Geodesic matrix op.
  close_sat_lat = f.variables['latitude'][min_ind]
  close_sat_long = f.variables['longitude'][min_ind]
  f.close
  return (min_ind, np.min(dists), close_sat_lat, close_sat_long)

def match_sat_data_to_datetime(sat_df, rdr_dt, sector):
  '''
  Finds satellite file closest to a specific radar date time.
  Args:
    sat_df: a pandas dataframe containing filenames for satellite imagery and their associated date times
    rdr_dt: a specific date time from the radar csv which we are trying to match up with satellite data.
    sector: String specifying satellite domain sector.
  Returns:
    the filename of the satellite imagery with the closest date time to rdr_dt; if no satellite imagery was
      captured within a minute of the specified rdr_dt date time, returns NO_VALID_FILE
  '''
  if sector.lower() == 'c':
    mins = 2
    secs = 30
  else:
    mins = 1
    secs = 0
  keys0 = list(sat_df.keys())                                                                #Determine column headers in DataFrame
  if 'filename' in keys0: 
    sat_df['dif'] = abs(sat_df['date_time'] - rdr_dt) 
    exit() 
    if sat_df['dif'].min() > timedelta(minutes = mins, seconds = secs): 
      return 'NO_VALID_FILE' 
    else: 
      min_dt = sat_df['dif'].idxmin() 
      return(sat_df.iloc[min_dt]['filename']) 
  elif 'ir_files' in keys0: 
    times = pd.Series([pd.Timestamp(x) for x in sat_df['date_time'].squeeze()])              #Need to have series of Time stamps
    tdif = abs(times - rdr_dt)                                                               #Series of timestamps - timestamp
    if 'dif' in keys0:
      sat_df['dif'] = np.minimum(tdif, sat_df['dif'])                                        #Series of timestamps - timestamp
    else:
      sat_df['dif'] = tdif                                                                   #Series of timestamps - timestamp
   
#    if sat_df['dif'].min() > timedelta(minutes=1):
    if tdif.min() > timedelta(minutes = mins, seconds = secs):
      return 'NO_VALID_FILE'
    else:
#      min_dt = sat_df['dif'].idxmin()
      min_dt = tdif.idxmin()
#      sat_df.iloc[min_dt]['dif'] = tdif.min()
      return(sat_df.iloc[min_dt]['ir_files'])
  else:
    print('Keys not found???')
    exit()

def get_sat_files(in_dir, img_type, sector):
  '''
  Reads all satellite imagery in a directory, adds each filename to a pandas dataframe, and creates a column in the dataframe with an extracted date and time 
  from the filename.
  Args:
    in_dir  : directory containing all of the satellite image files
    img_type: string specifying IR or VIS for IR and visible satellite imagery respectively
    sector  : STRING specifying data sector of GOES scan (ex. M1 or M2 for mesoscale domain sector 1 and 2, respectively)
  Returns:
    dataframe containing all filenames in the specified in_dir and extracted date times from each filename.
  '''
#  files = [f for f in listdir(in_dir) if ((isfile(join(in_dir, f)) and (not (f.startswith('.')))))]
  files = [os.path.basename(f) for f in sorted(glob.glob(in_dir + '/*_' + sector + '_*.nc'))]
  df    = pd.DataFrame(files, columns = ['filename'])
  if (img_type == 'IR'):
    df['date_time'] = df['filename'].apply(lambda x: datetime.strptime(x,'IR_%Y%j_%H%M%S.nc'))
  elif (img_type == 'VIS'):
    df['date_time'] = df['filename'].apply(lambda x: datetime.strptime(x,'VIS_%Y%j_%H%M%S.nc'))
  elif (img_type == 'COMB'):
    df['date_time'] = df['filename'].apply(lambda x: datetime.strptime(re.split('_|s', x)[6][0:-1],'%Y%j%H%M%S'))
  elif (img_type == 'ENTLN'):
    df['date_time'] = df['filename'].apply(lambda x: datetime.strptime(x,'ENTLN_1min_binned_data_%Y_%m%d_%H%M.nc').replace(second=0))
  else:
    print('Please specify a valid argument for img_type: IR or VIS')
  return(df)

def sat_row_op(row):
  '''
  A function used for error handling for apply calls to sat_rdr_marry() for satellite imagery.
  Args:
    row:    a dataframe row containing radar csv data
  Returns:  
    pandas series containing either all of the return args from sat_rdr_marry() or dummy values of 300s in the case where no valid satellite imagery
      exists for the timestamp specified in the radar csv file.
  '''
  if row['sat_Data_File'] == 'NO_VALID_FILE':
    return pd.Series(((300,300), 300, 300, 300))
  else:
    return pd.Series(sat_rdr_marry(os.path.join(row['sat_dir'], row['sat_Data_File']), row['Parallax Corrected Latitude'], row['Parallax Corrected Longitude']))
  
def rdr_parallax_correct(rdr_df, lat_correction, long_correction):
  '''
  Reads radar data from a CSV file and then finds closest match for satellite data files stored in sat_dir.
  Args:
    rdr_df         : pandas dataframe of radar data read from a CSV
    lat_correction : Float value contaning the latitude parallax correction to apply in order to match the radar data with the satellite data
    lon_correction : Float value contaning the latitude parallax correction to apply in order to match the radar data with the satellite data
  Returns:
    rdr_df: pandas dataframe with corrected radar lat/long for plumes
  '''
#  rdr_df.columns = [c.replace(' ', '') for c in rdr_df.columns]
  rdr_df['Parallax Corrected Latitude']  = rdr_df['Latitude'] + lat_correction
#  rdr_df['Parallax Corrected Longitude'] = rdr_df['Longitude'] - 360 - long_correction 
  rdr_df['Parallax Corrected Longitude'] = convert_longitude(rdr_df['Longitude']) - long_correction 
  return(rdr_df)

def rdr_csv_update2(rdr_csv, comb_dir, lat_correction, lon_correction, sector, 
                    rdr_bucket = 'misc-data0', use_local = False):
  '''
  Reads radar data from a CSV file and then finds closest match for satellite data files stored in sat_dir.
  Args:
    rdr_csv        : CSV file containing radar readings at time and expert labels of AACP plumes
    comb_dir       : Directory containing the IR, VIS, GLM combined data files.
    lat_correction : Float value contaning the latitude parallax correction to apply in order to match the radar data with the satellite data
    lon_correction : Float value contaning the latitude parallax correction to apply in order to match the radar data with the satellite data
    sector         : STRING specifying data sector of GOES scan (ex. M1 or M2 for mesoscale domain sector 1 and 2, respectively)
  Keywords:
    rdr_bucket     : Google cloud storage bucket to read the radar csv files. 
                      DEFAULT = 'misc-data0'
    use_local      : IF keyword set (True), use files stored in local directory.
                     DEFAULT = False -> use files stored in GCP.
  Returns:
    rdr_df: pandas dataframe containing original radar CSV data as well as matched satellite data files for each line and the index and lat / long on the 
      sat image(s) closest to each radar point of interest along with distances between the closest sat point and the radar point of interest  
  '''
  if use_local == True:
    rdr_df = pd.read_csv(rdr_csv, skiprows = [1])                                                                                                  #Read radar csv file into pandas dataframe
  else:
    pref   = re.split('misc-data0/', rdr_csv)[1]                                                                                                   #Extract path after misc-data0 to get GCP bucket path
    rdr_df = load_csv_gcs(rdr_bucket, pref, skiprows = [1])                                                                                        #Read radar csv file into pandas dataframe fromGCP

  rdr_df.fillna(0.0, inplace=True)                                                                                                                 #Fill data frame of NaNs with zeros
  rdr_df.replace(to_replace='#NAME?', value=0, inplace=True)                                                                                       #Replace data frame #NAME?' values with zeros
  rdr_df.columns = rdr_df.columns.str.strip()
  rdr_parallax_correct(rdr_df, lat_correction, lon_correction)                                                                                     #Correct for parallax differences between satellite and radar  
  rdr_df['date_time'] = rdr_df['Time'].str.strip().apply(lambda x: datetime.strptime(x,'%Y%m%dT%H%M%SZ'))
 
  # Add sat data columns.
  rdr_df['sat_Data_File'] = ''
  rdr_df['sat_dir']       = os.path.relpath(comb_dir)
  sat_df = get_sat_files(comb_dir, 'COMB', sector)
  rdr_df['sat_Data_File'] = rdr_df['date_time'].apply(lambda x: match_sat_data_to_datetime(sat_df, x))                                             #Match satellite data in time to radar data
  sat_res = rdr_df.apply(sat_row_op, axis=1).apply(pd.Series)                                                                                      # pd.Series(
  sat_res.columns = ['sat_Min_Index','sat_dist_delta','sat_closest_lat','sat_closest_long']                                                        #Set the column labels of the IR dataFrame
  rdr_df  = pd.concat([rdr_df, sat_res], axis=1)                                                                                                   #Concatenate the radar dataFrame with the IR dataFrame
  return(rdr_df)

def get_plume_idx_list(ncdf_file, rdr_df, img_type):
  '''
  Gets a list of plume indices for a specified NetCDF satellite imagery file.
  Args:
    ncdf_file: the satellite NetCDF file to which we want to find all labeled radar plumes 
    rdr_df   : pandas dataframe containing all radar CSV data after it has been matched up with satellite data from the rdr_csv_update() function
    img_type : specifies satellite imagery type; should either be 'IR' or 'VIS'
  Returns:
    plumes   :    list containing tuples of indices for plumes associated with ncdf_file
  '''
  plumes = []
  rdr_df = rdr_df.loc[rdr_df[img_type+'_Data_File'] == ncdf_file]
  for _, row in rdr_df.iterrows():
    plumes.append(row[img_type+'_Min_Index'])
  return plumes
  
def mult_pic_plot(pic_dir, rdr_df, img_type, out_dir, overlay_tgt, sector):
  '''
  Plots all satellite pictures in a directory.
  Args:
    pic_dir    : directory containing the NetCDF satellite data
    rdr_df     : pandas dataframe containing all radar CSV data after it has been matched up with satellite data from the rdr_csv_update() function
    img_type   : specifies satellite imagery type; should either be 'IR' or 'VIS'
    out_dir    : directory to which all of the satellite images will be saved
    overlay_tgt: boolean indicating user's desire to have AACP locations overlaid on imagery 
    sector  : STRING specifying data sector of GOES scan (ex. M1 or M2 for mesoscale domain sector 1 and 2, respectively)
  '''
  # Get list of all the image files you will need to plot.
  pic_df = get_sat_files(pic_dir, img_type, sector)
  rdr_df = rdr_df.loc[rdr_df['Plume'] == 1.0] # Subset only on confirmed AACPs. 
  pic_df['min_inds'] = pic_df['filename'].apply(lambda x: get_plume_idx_list(x, rdr_df, img_type))
  dir_maker(out_dir)
  pic_df.apply(lambda row: plot_pic(pic_dir+row['filename'], row['min_inds'], overlay_tgt, out_dir+row['filename'].split(',')[0]+'.png', row['filename'].split(',')[0]), axis=1)

def convert_longitude(x, to_model = False):
  '''
  This is a function to read echo-top height and tropopause file and calculated the tropopause-relative altitudes in km
  Args:
   x   : Numpy array of longitudes
  Keywords: 
   to_model : if set, input values are assumed to be in navigation convention
              and are converted to model convention. If not set, input values
              are assumed to be in model convention and are converted to
              navigation convention.
              DEFAULT = False
  Returns:
    Tropopause-relative altitude for dbz_thresh, GridRad longitude values, and GridRad latitude values
  '''  
  if to_model == True: 
    return((x + 360.0) % 360.0)                                                                                                                                                 #Convert from navigation to model convention
  else:
    return(x - (x.astype(int)/180).astype(int)*360.0)                                                                                                                           #Convert from model to navigation convention


def gridrad_read_sparse_eth_ncdf(eth_file, dbz_thresh = 10.0):
  '''
  This is a function to read sparse stored echo-top height netCDF files and reshape to [latitude, longitude] arrays
  Args:
   eth_file   : Echo-top height path and netCDF file name to read
  Keywords: 
   dbz_thresh : FLOAT horizontally polarized radar reflectivity threshold value at which echo-top height was calculated
                DEFAULT = 10.0 (dBZ)
  Returns:
    Echo-top altitude for dbz_thresh, GridRad longitude values, and GridRad latitude values (km)
  '''  
#Read ETH file
  with open_dataset(eth_file) as eth:
    z_e0       = np.copy(eth.z_e.values)
    eth_thresh = np.copy(eth.threshold.values)                                                                                                                                  #Extract reflectivity threshold from file
    ind        = list(np.where(eth_thresh == dbz_thresh)[0])                                                                                                                    #Extract index where echo-top altitude threshold matches the one specified by the user    
    if len(ind) == 0 or len(ind) > 1:
      print('dbz_thresh not found in file or too many matches??')
      print(ind)
      exit()
    
    x_nex = np.copy(eth.Longitude.values)                                                                                                                                       #Read Longitudes
    y_nex = np.copy(eth.Latitude.values)                                                                                                                                        #Read Latitudes
    z_e   = np.zeros([eth.Longitude.shape[0], eth.Latitude.shape[0], eth.threshold.shape[0]])*np.nan						                                                           #Create values array (# elements of dbz_thresh)
    einds = eth.index.values                                                                                                                                                    #Read sparse storage indices
    if len(einds) > 1:
      xIndc, yIndc, zIndc = np.unravel_index(einds, z_e.shape, order = 'F')
      z_e[xIndc, yIndc, zIndc] = z_e0                                                                                                                                           #Extract echo-top heights for all reflectivity thresholds
    z_e = z_e[:, :, ind[0]]                                                                                                                                                     #Extract echo-top heights at specified reflectivity threshold
    if eth['z_e'].units == 'm': z_e = z_e*0.001                                                                                                                                 #Convert echo-top altitudes in meters into kilometers
  
  z_e = np.transpose(z_e)

  return(z_e, x_nex, y_nex)


def gridrad_read_sparse_refl_ncdf(refl_file, verbose = True):
  '''
  This is a function to read sparse stored GridRad reflectivity netCDF files and reshape to [altitude, latitude, longitude] arrays
  Args:
    refl_file : GridRad reflectivity file path and netCDF file name to read
  Keywords: 
    verbose : IF keyword set (True), print verbose informational messages
              DEFAULT = True
  Returns:
    Reflectivity data on GridRad lon, lat, and altitude grid. Also returns, GridRad longitudes, latitudes, and altitudes, respectively (dBZ)
  '''  
#Read reflectivity file
  with open_dataset(refl_file) as refl:
    z_h0 = np.copy(refl.Reflectivity.values)
    z    = np.copy(refl.Altitude.values)                                                                                                                                        #Extract reflectivity altitudes from file
    
    x_nex = np.copy(refl.Longitude.values)                                                                                                                                      #Read Longitudes
    y_nex = np.copy(refl.Latitude.values)                                                                                                                                       #Read Latitudes
    z_h   = np.zeros([refl.Longitude.shape[0], refl.Latitude.shape[0], z.shape[0]])*np.nan                                                                                      #Create values array filled with NaNs
    einds = refl.index.values                                                                                                                                                   #Read sparse storage indices
    if len(einds) > 1:
      xIndc, yIndc, zIndc = np.unravel_index(einds, z_h.shape, order = 'F')
      z_h[xIndc, yIndc, zIndc] = z_h0                                                                                                                                           #Extract GridRad reflectivities
    z_h = z_h[:, :, :]                                                                                                                                                          #Extract GridRad reflectivities
  
  z_h   = np.transpose(z_h)                                                                                                                                                     #Extract reflectivity data in altitude x latitude x longitude
  x_nex = convert_longitude(x_nex, to_model = False)                                                                                                                            #Convert 0°-360° to -180° to 180°


  return(z_h, x_nex, y_nex, z)

def gridrad_read_calc_zrel(eth_file, trop_file, dbz_thresh = 10.0):
  '''
  This is a function to read echo-top height and tropopause file and calculated the tropopause-relative altitudes in km
  Args:
   eth_file   : Echo-top height path and netCDF file name to read
   trop_file  : Tropopause height path and netCDF file name to read
  Keywords: 
   dbz_thresh : FLOAT horizontally polarized radar reflectivity threshold value at which echo-top height was calculated
                DEFAULT = 10.0 (dBZ)
  Returns:
    Tropopause-relative altitude for dbz_thresh, GridRad longitude values, and GridRad latitude values
  '''  

  z_e, x_nex, y_nex = gridrad_read_sparse_eth_ncdf(eth_file, dbz_thresh = dbz_thresh)                                                                                           #Read ETH file
  
  with Dataset(trop_file) as trop0:                                                                                                                                             #Read tropopause file   
    z_t = np.copy(np.asarray(trop0.variables['Z_trop'][0, :, :]))                                                                                                               #Read tropopause altitudes (m)
    if trop0['Z_trop'].units == 'm': z_t = z_t*0.001                                                                                                                            #Convert tropopause altitudes in meters into kilometers
 #   z_t = z_t*0.001

  z_r   = np.copy(z_e - z_t)                                                                                                                                                    #Calculate tropopause-relative altitude (km)
  x_nex = convert_longitude(x_nex, to_model = False)
  return(z_r, z_e, z_t, x_nex, y_nex)

def sat_time_intervals(sat, sector = None):
    '''
    Returns the time interval between satellite scans for a specified scan sector in real-time runs. This function makes sure the code does not re-run the same date by waiting
    for interval to pass before starting next job.
    Args:
        sat : STRING specifying satellite name (ex. 'goes-16')
    Keywords:
        sector : Satellite scan sector. 
                 DEFAULT = None -> must be set for GOES but some satellites may not have scan sectors like GOES so default is None.
    Returns:
        Time in seconds for specified satellite to scan specified sector.
    '''  
    sat = sat.replace(' ', '').replace('_', '').replace('-', '').lower()
    if sat.lower() == 'goes16' or sat.lower() == 'goes17' or sat.lower() == 'goes18':
        if sector == None:
            print('Sector must be specified for GOES satellite in sat_time_intervals function!!')
            exit()
        if sector[0].lower() == 'm':
            t_interval = 60.0                                                                                                                                                   #60 seconds between mesoscale scans for GOES-16 and GOES-17
        elif sector[0].lower() == 'c':
            t_interval = 300.0                                                                                                                                                  #5 mins between conus scans for GOES-16 and GOES-17
        elif sector[0].lower() == 'f':
            t_interval = 600.0                                                                                                                                                  #10 mins between full disk scans for GOES-16 and GOES-17
        else:
          print('sector specified not found??')
          print('Specified sector = ' + str(sector))
          print('Sectors for GOES satellite are Full, CONUS, Mesoscale.')
          exit()
    else:
        print('sat_time_intervals function not set up to handle specified satellite!!')
        print('Exiting program.')
        exit()
    
    return(t_interval)
   
def index_of_nearest(array, value):
    '''
    Find the nearest index to value in the array of interest
    Args:
        array : Array to search for nearest index
        value : Value to find nearest index in array point
    Keywords:
        None.
    Returns:
        idx : Index of nearest point in array to value
    '''
    array = np.asarray(array)
    idx   = (np.abs(array - value)).argmin()
    return(idx)

def extract_us_lat_lon_region(region, exact = False):
    '''
    Uses switch in order to return the latitude and longitude bounds of US states and regions
    Args:
        region : State or sector of US to get latitude and longitude data for
    Keywords:
        exact : IF keyword set (True), use the exact longitude and latitude boundaries.
                DEFAULT = False -> round up to nearest integer for x1 and y1 and down to nearest integer for x0 and y0
    Returns:
        domain : Longitude and latitude [x0, y0, x1, y1]
    '''
    region = region.lower()
    domain = {
              'alabama'                                      : [-88.473227, 30.223334, -84.88908, 35.008028],
              'alaska'                                       : [-179.148909, 51.214183, -140.0, 71.365162],
              'american samoa'                               : [-171.089874, -14.548699, -168.1433, -11.046934],
              'samoa'                                        : [-171.089874, -14.548699, -168.1433, -11.046934],
              'arizona'                                      : [-114.81651, 31.332177, -109.045223, 37.00426],
              'arkansas'                                     : [-94.617919, 33.004106, -89.644395, 36.4996],
              'california'                                   : [-124.409591, 32.534156, -114.131211, 42.009518],
              'colorado'                                     : [-109.060253, 36.992426, -102.041524, 41.003444],
              'commonwealth of the northern mariana islands' : [144.886331, 14.110472, 146.064818, 20.5538],
              'connecticut'                                  : [-73.727775, 40.980144, -71.786994, 42.050587],
              'delaware'                                     : [-75.788658, 38.451013, -75.048939, 39.839007],
              'district of columbia'                         : [-77.119759, 38.791645, -76.909395, 38.99511],
              'washington dc'                                : [-77.119759, 38.791645, -76.909395, 38.99511],
              'florida'                                      : [-87.634938, 24.523096, -80.031362, 31.000888],
              'georgia'                                      : [-85.605165, 30.357851, -80.839729, 35.000659],
              'guam'                                         : [144.618068, 13.234189, 144.956712, 13.654383],
              'hawaii'                                       : [-178.334698, 18.910361, -154.806773, 28.402123],
              'idaho'                                        : [-117.243027, 41.988057, -111.043564, 49.001146],
              'illinois'                                     : [-91.513079, 36.970298, -87.494756, 42.508481],
              'indiana'                                      : [-88.09776, 37.771742, -84.784579, 41.760592],
              'iowa'                                         : [-96.639704, 40.375501, -90.140061, 43.501196],
              'kansas'                                       : [-102.051744, 36.993016, -94.588413, 40.003162],
              'kentucky'                                     : [-89.571509, 36.497129, -81.964971, 39.147458],
              'louisiana'                                    : [-94.043147, 28.928609, -88.817017, 33.019457],
              'maine'                                        : [-71.083924, 42.977764, -66.949895, 47.459686],
              'maryland'                                     : [-79.487651, 37.911717, -75.048939, 39.723043],
              'massachusetts'                                : [-73.508142, 41.237964, -69.928393, 42.886589],
              'michigan'                                     : [-90.418136, 41.696118, -82.413474, 48.2388],
              'minnesota'                                    : [-97.239209, 43.499356, -89.491739, 49.384358],
              'mississippi'                                  : [-91.655009, 30.173943, -88.097888, 34.996052],
              'missouri'                                     : [-95.774704, 35.995683, -89.098843, 40.61364],
              'montana'                                      : [-116.050003, 44.358221, -104.039138, 49.00139],
              'nebraska'                                     : [-104.053514, 39.999998, -95.30829, 43.001708],
              'nevada'                                       : [-120.005746, 35.001857, -114.039648, 42.002207],
              'new hampshire'                                : [-72.557247, 42.69699, -70.610621, 45.305476],
              'new jersey'                                   : [-75.559614, 38.928519, -73.893979, 41.357423],
              'new mexico'                                   : [-109.050173, 31.332301, -103.001964, 37.000232],
              'new york'                                     : [-79.762152, 40.496103, -71.856214, 45.01585],
              'north carolina'                               : [-84.321869, 33.842316, -75.460621, 36.588117],
              'north dakota'                                 : [-104.0489, 45.935054, -96.554507, 49.000574],
              'ohio'                                         : [-84.820159, 38.403202, -80.518693, 41.977523],
              'oklahoma'                                     : [-103.002565, 33.615833, -94.430662, 37.002206],
              'oregon'                                       : [-124.566244, 41.991794, -116.463504, 46.292035],
              'pennsylvania'                                 : [-80.519891, 39.7198, -74.689516, 42.26986],
              'puerto rico'                                  : [-67.945404, 17.88328, -65.220703, 18.515683],
              'rhode island'                                 : [-71.862772, 41.146339, -71.12057, 42.018798],
              'south carolina'                               : [-83.35391, 32.0346, -78.54203, 35.215402],
              'south dakota'                                 : [-104.057698, 42.479635, -96.436589, 45.94545],
              'tennessee'                                    : [-90.310298, 34.982972, -81.6469, 36.678118],
              'texas'                                        : [-106.645646, 25.837377, -93.508292, 36.500704],
              'united states virgin islands'                 : [-65.085452, 17.673976, -64.564907, 18.412655],
              'virgin islands'                               : [-65.085452, 17.673976, -64.564907, 18.412655],
              'utah'                                         : [-114.052962, 36.997968, -109.041058, 42.001567],
              'vermont'                                      : [-73.43774, 42.726853, -71.464555, 45.016659],
              'virginia'                                     : [-83.675395, 36.540738, -75.242266, 39.466012],
              'washington'                                   : [-124.763068, 45.543541, -116.915989, 49.002494],
              'west virginia'                                : [-82.644739, 37.201483, -77.719519, 40.638801],
              'wisconsin'                                    : [-92.888114, 42.491983, -86.805415, 47.080621],
              'wyoming'                                      : [-111.056888, 40.994746, -104.05216, 45.005904],
              'southeast'                                    : [-95.0, 30.0, -75.0, 37.5],
              'southeast2'                                   : [-95.0, 30.0, -85.0, 37.0], #Covers west of Georgia
              'northeast'                                    : [-85.0, 40.0, -68.0, 50.0],
              'southwest'                                    : [-117.0, 29.0, -104.0, 39.0],
              'midwest'                                      : [-105.5, 36.15, -79.0, 49.5],
              'great plains'                                 : [-100.0, 36.0, -86.0, 42.0],
              'custom1'                                      : [-97.5, 28.0, -105.0, 36.0],#Custom region
              'custom2'                                      : [-105.0, 33.5, -94.0, 41.0],
              '20211210_outbreak'                            : [-94.5, 33.0, -84.0, 42.0], #Also called custom3
              '20211215_outbreak'                            : [-102.0, 37.002206, -86.805415, 47.080621], 
              'hurricane_sam21'                              : [-52.0, 12.0, -45.0, 16.0], 
              'hurricane_dorian19'                           : [-80.0, 23.0, -72.0, 30.0], 
              'dcotss'                                       : [-97.0, 38.0, -85.0, 48.0], #Custom region
              'dcotss_20210726'                              : [-105.0, 25.0, -91.0, 41.0], #Custom region
              'dcotss_20210814'                              : [-100.0, 31.0, -80.0, 45.0], #Custom region
              'dcotss_20210819'                              : [-115.0, 19.0, -100.0, 39.0], #Custom region
              'central texas'                                : [-104.0, 28.0, -98.0, 36.0], 
              '2019146-147_region'                           : [-106.0, 31.093575, -91.10014, 45.30275], 
              '2019146-147_region2'                          : [-106.0, 37.002206, -94.430662, 43.001708], #Focused around Kansas and Nebraska
              '2019146-147_region3'                          : [-106.0, 37.002206, -97.0, 42.0], #Narrow region ever further around western 2/3rds of 2019146-147_region2 and southern half of Nebraska
              '2019120-121_region'                           : [-103.002565, 28.928609, -92.0, 40.003162], 
              '2019125-126_m1_region'                        : [-103.5, 33.0, -94.617919, 43.001708], 
              '2019125-126_m2_region'                        : [-106.645646, 28.0, -98.0, 39.5], 
              '2019127-128_region'                           : [-106.0, 30.0, -93.508292, 38.5],
              '2019137-138_region'                           : [-107.0, 37.0, -95.30829, 46.0], 
              '2019140-141_region'                           : [-103.002565, 31.0, -94.0, 39.8], 
              '2020134-135_region'                           : [-103.25 , 28.0, -98.0, 36.500704], 
              'argentina'                                    : [-73.5600329, -55.1850761, -53.6374515, -21.781168],
              '2018103-2018104_C_region'                     : [-101.5, 27.0, -86.0, 43.5],
              '2018104-2018105_C_region'                     : [-94.0,  26.0, -84.0, 37.0],
              '2018152-2018153_C_region'                     : [-111.0, 25.0, -77.0, 50.0],
              '2018157-2018158_C_region'                     : [-113.5, 32.5, -92.5, 50.0],
              '2018169-2018170_C_region'                     : [-112.0, 24.0, -66.0, 50.0],
              '2018170-2018171_C_region'                     : [-109.0, 26.0, -83.0, 50.0],
              '2018208-2018209_C_region'                     : [-125.0, 24.0, -66.0, 50.0],
              '2018239-2018240_C_region'                     : [-103.0, 38.5, -84.5, 49.5],
              '2018284-2018285_C_region'                     : [-83.0,  32.0, -73.0, 42.0],
              '2019019-2019020_C_region'                     : [-93.0,  27.0, -83.0, 37.0],
              '2019062-2019063_C_region'                     : [-90.0,  27.0, -79.0, 37.0],
              '2019123-2019124_C_region'                     : [-110.0, 24.0, -77.0, 49.0],
              '2019126-2019127_C_region'                     : [-106.0, 27.5, -86.5, 46.5],
              '2019170-2019171_C_region'                     : [-107.5, 27.5, -83.5, 47.5],
              '2019174-2019175_C_region'                     : [-109.0, 25.0, -83.0, 47.0],
              '2019211-2019212_C_region'                     : [-125.0, 24.0, -66.5, 50.0],
              '2019254-2019255_C_region'                     : [-107.0, 31.5, -75.5, 46.5],
              '2019267-2019268_C_region'                     : [-117.0, 24.0, -69.0, 50.0],
              '2019304-2019305_C_region'                     : [-84.0,  31.0, -73.0, 43.0],
              '2019350-2019351_C_region'                     : [-95.0,  27.0, -83.0, 39.0], 
              'central_europe'                               : [5.0, 40.0, 20.0, 56.0],
              'central_europe2'                              : [5.0, 44.0, 16.0, 52.0],
              'conus'                                        : [-151.85506521266967, 14.56418656871904, -52.92544258725, 56.73345476748197], 
              'aruba'                                        : [-70.2809842, 12.1702998, -69.6409842, 12.8102998], 
              'afghanistan'                                  : [60.5176034, 29.3772, 74.889862, 38.4910682], 
              'angola'                                       : [11.4609793, -18.038945, 24.0878856, -4.3880634], 
              'anguilla'                                     : [-63.6391992, 18.0615454, -62.7125449, 18.7951194], 
              'aland islands'                                : [19.0832098, 59.4541578, 21.3456556, 60.87665], 
              'albania'                                      : [-72.6598913, 11.081041, -72.4039128, 11.408791], 
              'andorra'                                      : [1.4135781, 42.4288238, 1.7863837, 42.6559357], 
              'netherlands antilles'                         : [-68.940593, 12.1544542, -68.9403518, 12.1547472], 
              'united arab emirates'                         : [51.498, 22.6444, 56.3834, 26.2822], 
              'argentina'                                    : [-73.5600329, -55.1850761, -53.6374515, -21.781168], 
              'armenia'                                      : [-75.7854388, 4.3951997, -75.6325211, 4.5895768], 
              'antarctica'                                   : [-180, -85.0511287, 180, -60], 
              'french southern territories'                  : [39.4138676, -50.2187169, 77.8494974, -11.3139928], 
              'antigua and barbuda'                          : [-62.5536517, 16.7573901, -61.447857, 17.929], 
              'australia'                                    : [72.2460938, -55.3228175, 168.2249543, -9.0882278], 
              'austria'                                      : [9.5307487, 46.3722761, 17.160776, 49.0205305], 
              'azerbaijan'                                   : [44.7633701, 38.3929551, 51.0090302, 41.9502947], 
              'burundi'                                      : [29.0007401, -4.4693155, 30.8498462, -2.3096796], 
              'belgium'                                      : [2.3889137, 49.4969821, 6.408097, 51.5516667], 
              'benin'                                        : [5.4621058, 6.1730586, 5.7821058, 6.4930586], 
              'burkina faso'                                 : [-5.5132416, 9.4104718, 2.4089717, 15.084], 
              'bangladesh'                                   : [88.0075306, 20.3756582, 92.6804979, 26.6382534], 
              'bulgaria'                                     : [22.3571459, 41.2353929, 28.8875409, 44.2167064], 
              'bahrain'                                      : [50.414458, 25.815088, 50.695428, 26.342619], 
              'bahamas'                                      : [-80.7001941, 20.7059846, -72.4477521, 27.4734551], 
              'bosnia and herzegovina'                       : [15.7287433, 42.5553114, 19.6237311, 45.2764135], 
              'saint barthelemy'                             : [-63.06639, 17.670931, -62.5844019, 18.1375569], 
              'belarus'                                      : [23.1783344, 51.2575982, 32.7627809, 56.17218], 
              'belize'                                       : [-89.2262083, 15.8857286, -87.3098494, 18.496001], 
              'bermuda'                                      : [-65.1232222, 32.0469651, -64.4109842, 32.5913693], 
              'bolivia'                                      : [-69.6450073, -22.8982742, -57.453, -9.6689438], 
              'brazil'                                       : [-73.9830625, -33.8689056, -28.6341164, 5.2842873], 
              'barbados'                                     : [-59.8562115, 12.845, -59.2147175, 13.535], 
              'brunei darussalam'                            : [114.0758734, 4.002508, 115.3635623, 5.1011857], 
              'brunei'                                       : [114.0758734, 4.002508, 115.3635623, 5.1011857], 
              'bhutan'                                       : [88.7464724, 26.702016, 92.1252321, 28.246987], 
              'bouvet island'                                : [2.9345531, -54.654, 3.7791099, -54.187], 
              'botswana'                                     : [19.9986474, -26.9059669, 29.375304, -17.778137], 
              'central african republic'                     : [55.1742999, 25.217277, 55.1762608, 25.2194356], 
              'canada'                                       : [-141.00275, 41.6765556, -52.3231981, 83.3362128], 
              'cocos islands'                                : [96.612524, -12.4055983, 97.1357343, -11.6213132], 
              'keeling islands'                              : [96.612524, -12.4055983, 97.1357343, -11.6213132], 
              'cocos keeling islands'                        : [96.612524, -12.4055983, 97.1357343, -11.6213132], 
              'switzerland'                                  : [5.9559113, 45.817995, 10.4922941, 47.8084648], 
              'chile'                                        : [-109.6795789, -56.725, -66.0753474, -17.4983998], 
              'china'                                        : [-99.4612273, 25.0671341, -98.421576, 25.9161824], 
              'cote divoire'                                 : [-8.601725, 4.1621205, -2.493031, 10.740197], 
              'ivory coast'                                  : [-8.601725, 4.1621205, -2.493031, 10.740197], 
              'cameroon'                                     : [8.3822176, 1.6546659, 16.1921476, 13.083333], 
              'the democratic republic of the congo'         : [15.2829578, -4.2726671, 15.283136, -4.2723703], 
              'democratic republic of the congo'             : [15.2829578, -4.2726671, 15.283136, -4.2723703], 
              'congo'                                        : [11.0048205, -5.149089, 18.643611, 3.713056], 
              'cook islands'                                 : [-166.0856468, -22.15807, -157.1089329, -8.7168792], 
              'colombia'                                     : [-82.1243666, -4.2316872, -66.8511907, 16.0571269], 
              'comoros'                                      : [43.025305, -12.621, 44.7451922, -11.165], 
              'cape verde'                                   : [-25.3609478, 14.8031546, -22.6673416, 17.2053108], 
              'costa rica'                                   : [-87.2722647, 5.3329698, -82.5060208, 11.2195684], 
              'cuba'                                         : [-85.1679702, 19.6275294, -73.9190004, 23.4816972], 
              'christmas island'                             : [105.5336422, -10.5698515, 105.7130159, -10.4123553], 
              'cayman islands'                               : [-81.6313748, 19.0620619, -79.5110954, 19.9573759], 
              'cyprus'                                       : [25.6451285, 27.4823018, 40.6451285, 42.4823018], 
              'czech republic'                               : [12.0905901, 48.5518083, 18.859216, 51.0557036], 
              'germany'                                      : [5.8663153, 47.2701114, 15.0419319, 55.099161], 
              'djibouti'                                     : [41.7713139, 10.9149547, 43.6579046, 12.7923081], 
              'dominica'                                     : [-61.6869184, 15.0074207, -61.0329895, 15.7872222], 
              'denmark'                                      : [7.7153255, 54.4516667, 15.5530641, 57.9524297], 
              'dominican republic'                           : [-72.0574706, 17.2701708, -68.1101463, 21.303433], 
              'algeria'                                      : [-8.668908, 18.968147, 11.997337, 37.2962055], 
              'ecuador'                                      : [-92.2072392, -5.0159314, -75.192504, 1.8835964], 
              'egypt'                                        : [24.6499112, 22, 37.1153517, 31.8330854], 
              'eritrea'                                      : [55.1720522, 25.2197993, 55.1736716, 25.2220008], 
              'western sahara'                               : [-13.7867848, 24.1597324, -13.7467848, 24.1997324], 
              'spain'                                        : [-18.3936845, 27.4335426, 4.5918885, 43.9933088], 
              'estonia'                                      : [21.3826069, 57.5092997, 28.2100175, 59.9383754], 
              'ethiopia'                                     : [32.9975838, 3.397448, 47.9823797, 14.8940537], 
              'finland'                                      : [19.0832098, 59.4541578, 31.5867071, 70.0922939], 
              'fiji'                                         : [-180, -21.9434274, 180, -12.2613866], 
              'falkland islands (malvinas)'                  : [-61.7726772, -53.1186766, -57.3662367, -50.7973007], 
              'falkland islands'                             : [-61.7726772, -53.1186766, -57.3662367, -50.7973007], 
              'malvinas'                                     : [-61.7726772, -53.1186766, -57.3662367, -50.7973007], 
              'france'                                       : [-5.4534286, 41.2632185, 9.8678344, 51.268318], 
              'faroe islands'                                : [-7.6882939, 61.3915553, -6.2565525, 62.3942991], 
              'federated states of micronesia'               : [137.2234512, 0.827, 163.2364054, 10.291], 
              'micronesia'                                   : [137.2234512, 0.827, 163.2364054, 10.291], 
              'gabon'                                        : [8.5002246, -4.1012261, 14.539444, 2.3182171], 
              'united kingdom'                               : [-10.664378, 49.922384, 1.574392, 59.231738], 
              'uk'                                           : [-10.664378, 49.922384, 1.574392, 59.231738], 
              'guernsey'                                     : [-2.6751703, 49.4155331, -2.501814, 49.5090776], 
              'ghana'                                        : [-3.260786, 4.5392525, 1.2732942, 11.1748562], 
              'gibraltar'                                    : [-5.3941295, 36.100807, -5.3141295, 36.180807], 
              'guinea'                                       : [-77.4591124, 38.1240893, -77.4191124, 38.1640893], 
              'guadeloupe'                                   : [-61.809764, 15.8320085, -61.0003663, 16.5144664], 
              'gambia'                                       : [-17.0288254, 13.061, -13.797778, 13.8253137], 
              'guinea-bissau'                                : [-16.894523, 10.6514215, -13.6348777, 12.6862384], 
              'equatorial guinea'                            : [5.4172943, -1.6732196, 11.3598628, 3.989], 
              'greece'                                       : [19.2477876, 34.7006096, 29.7296986, 41.7488862], 
              'grenada'                                      : [-62.0065868, 11.786, -61.1732143, 12.5966532], 
              'greenland'                                    : [-70.907562, 43.0015105, -70.805801, 43.0770931], 
              'guatemala'                                    : [-92.3105242, 13.6345804, -88.1755849, 17.8165947], 
              'french guiana'                                : [-54.60278, 2.112222, -51.6346139, 5.7507111], 
              'guam'                                         : [144.563426, 13.182335, 145.009167, 13.706179], 
              'guyana'                                       : [-61.414905, 1.1710017, -56.4689543, 8.6038842], 
              'hong kong'                                    : [114.0028131, 22.1193278, 114.3228131, 22.4393278], 
              'heard island and mcdonald islands'            : [72.2460938, -53.394741, 74.1988754, -52.7030677], 
              'honduras'                                     : [-89.3568207, 12.9808485, -82.1729621, 17.619526], 
              'croatia'                                      : [13.2104814, 42.1765993, 19.4470842, 46.555029], 
              'haiti'                                        : [-78.1147805, 21.7470269, -78.0747805, 21.7870269], 
              'hungary'                                      : [16.1138867, 45.737128, 22.8977094, 48.585257], 
              'indonesia'                                    : [94.7717124, -11.2085669, 141.0194444, 6.2744496], 
              'isle of man'                                  : [-4.7946845, 54.0539576, -4.3076853, 54.4178705], 
              'india'                                        : [68.1113787, 6.5546079, 97.395561, 35.6745457], 
              'british indian ocean territory'               : [71.036504, -7.6454079, 72.7020157, -5.037066], 
              'ireland'                                      : [-11.0133788, 51.222, -5.6582363, 55.636], 
              'islamic republic of iran'                     : [44.0318908, 24.8465103, 63.3332704, 39.7816502], 
              'iran'                                         : [44.0318908, 24.8465103, 63.3332704, 39.7816502], 
              'iraq'                                         : [38.7936719, 29.0585661, 48.8412702, 37.380932], 
              'iceland'                                      : [0.5475318, 51.3863696, 0.5479162, 51.3868275], 
              'israel'                                       : [34.2674994, 29.4533796, 35.8950234, 33.3356317], 
              'italy'                                        : [6.6272658, 35.2889616, 18.7844746, 47.0921462], 
              'jamaica'                                      : [-78.5782366, 16.5899443, -75.7541143, 18.7256394], 
              'jersey'                                       : [-2.254512, 49.1625179, -2.0104193, 49.2621288], 
              'jordan'                                       : [122.4078085, 10.5378127, 122.6356296, 10.6910595], 
              'japan'                                        : [122.7141754, 20.2145811, 154.205541, 45.7112046], 
              'kazakhstan'                                   : [46.4932179, 40.5686476, 87.3156316, 55.4421701], 
              'kenya'                                        : [33.9098987, -4.8995204, 41.899578, 4.62], 
              'kyrgyzstan'                                   : [69.2649523, 39.1728437, 80.2295793, 43.2667971], 
              'cambodia'                                     : [102.3338282, 9.4752639, 107.6276788, 14.6904224], 
              'kiribati'                                     : [-179.1645388, -7.0516717, -164.1645388, 7.9483283], 
              'saint kitts and nevis'                        : [-63.051129, 16.895, -62.3303519, 17.6158146], 
              'republic of korea'                            : [126.89575, 37.52806, 126.89585, 37.52816], 
              'south korea'                                  : [126.89575, 37.52806, 126.89585, 37.52816], 
              'kuwait'                                       : [-72.8801102, -12.0494672, -72.8401102, -12.0094672], 
              'lao peoples democratic republic'              : [100.0843247, 13.9096752, 107.6349989, 22.5086717], 
              'lebanon'                                      : [-76.4508668, 40.3199193, -76.391965, 40.355802], 
              'liberia'                                      : [-11.6080764, 4.1555907, -7.367323, 8.5519861], 
              'libyan arab jamahiriya'                       : [9.391081, 19.5008138, 25.3770629, 33.3545898], 
              'saint lucia'                                  : [-61.2853867, 13.508, -60.6669363, 14.2725], 
              'liechtenstein'                                : [9.4716736, 47.0484291, 9.6357143, 47.270581], 
              'sri lanka'                                    : [55.1813071, 25.2287738, 55.1828523, 25.2303051], 
              'lesotho'                                      : [27.0114632, -30.6772773, 29.4557099, -28.570615], 
              'lithuania'                                    : [20.653783, 53.8967893, 26.8355198, 56.4504213], 
              'luxembourg'                                   : [4.9684415, 49.4969821, 6.0344254, 50.430377], 
              'latvia'                                       : [20.6715407, 55.6746505, 28.2414904, 58.0855688], 
              'macao'                                        : [55.1803834, 25.2349583, 55.1821684, 25.2368385], 
              'saint martin'                                 : [-63.3605643, 17.8963535, -62.7644063, 18.1902778], 
              'morocco'                                      : [-17.2551456, 21.3365321, -0.998429, 36.0505269], 
              'monaco'                                       : [7.4090279, 43.7247599, 7.4398704, 43.7519311], 
              'republic of moldova'                          : [29.7521496, 46.6744934, 29.7522496, 46.6745934], 
              'moldova'                                      : [29.7521496, 46.6744934, 29.7522496, 46.6745934], 
              'madagascar'                                   : [43.2202072, -25.6071002, 50.4862553, -11.9519693], 
              'maldives'                                     : [55.1823199, 25.2252959, 55.1837469, 25.2263769], 
              'mexico'                                       : [-99.2933416, 19.2726009, -98.9733416, 19.5926009], 
              'marshall islands'                             : [163.4985095, -0.5481258, 178.4985095, 14.4518742], 
              'the former yugoslav republic of macedonia'    : [20.4529023, 40.8536596, 23.034051, 42.3735359], 
              'macedonia'                                    : [20.4529023, 40.8536596, 23.034051, 42.3735359], 
              'mali'                                         : [-12.2402835, 10.147811, 4.2673828, 25.001084], 
              'malta'                                        : [13.9324226, 35.6029696, 14.8267966, 36.2852706], 
              'myanmar'                                      : [92.1719423, 9.4399432, 101.1700796, 28.547835], 
              'montenegro'                                   : [-51.662, -29.8446455, -51.3498193, -29.5743342], 
              'mongolia'                                     : [87.73762, 41.5800276, 119.931949, 52.1496], 
              'northern mariana islands'                     : [144.813338, 14.036565, 146.154418, 20.616556], 
              'mozambique'                                   : [-96.274167, 19.038889, -96.234167, 19.078889], 
              'mauritania'                                   : [55.1665334, 25.2143602, 55.1690669, 25.2163939], 
              'montserrat'                                   : [-62.450667, 16.475, -61.9353818, 17.0152978], 
              'martinique'                                   : [-61.2290815, 14.3948596, -60.8095833, 14.8787029], 
              'mauritius'                                    : [56.3825151, -20.725, 63.7151319, -10.138], 
              'malawi'                                       : [32.6703616, -17.1296031, 35.9185731, -9.3683261], 
              'malaysia'                                     : [105.3471939, -5.1076241, 120.3471939, 9.8923759], 
              'mayotte'                                      : [45.0183298, -13.0210119, 45.2999917, -12.6365902], 
              'namibia'                                      : [11.5280384, -28.96945, 25.2617671, -16.9634855], 
              'new caledonia'                                : [-92.6876556, 33.0351363, -92.6476556, 33.0751363], 
              'niger'                                        : [0.1689653, 11.693756, 15.996667, 23.517178], 
              'norfolk island'                               : [167.6873878, -29.333, 168.2249543, -28.796], 
              'nigeria'                                      : [2.676932, 4.0690959, 14.678014, 13.885645], 
              'nicaragua'                                    : [-87.901532, 10.7076565, -82.6227023, 15.0331183], 
              'niue'                                         : [-169.9383436, -19.0840514, -169.9129672, -19.0739661], 
              'netherlands'                                  : [-68.6255319, 11.825, 7.2274985, 53.7253321], 
              'norway'                                       : [4.0875274, 57.7590052, 31.7614911, 71.3848787], 
              'nepal'                                        : [80.0586226, 26.3477581, 88.2015257, 30.446945], 
              'nauru'                                        : [166.9091794, -0.5541334, 166.9589235, -0.5025906], 
              'new zealand'                                  : [-179.059153, -52.8213687, 179.3643594, -29.0303303], 
              'oman'                                         : [55.1759911, 25.2268538, 55.1773672, 25.2291352], 
              'pakistan'                                     : [60.872855, 23.5393916, 77.1203914, 37.084107], 
              'panama'                                       : [-119.0767707, 35.246906, -119.0367707, 35.286906], 
              'pitcairn'                                     : [-130.8049862, -25.1306736, -124.717534, -23.8655769], 
              'peru'                                         : [-89.164781, 41.309511, -89.105652, 41.38197], 
              'philippines'                                  : [114.0952145, 4.2158064, 126.8072562, 21.3217806], 
              'palau'                                        : [2.9351868, 42.5481153, 2.9840518, 42.5844284], 
              'papua new guinea'                             : [136.7489081, -13.1816069, 151.7489081, 1.8183931], 
              'poland'                                       : [14.1229707, 49.0020468, 24.145783, 55.0336963], 
              'puerto rico'                                  : [-67.271492, 17.9268695, -65.5897525, 18.5159789], 
              'democratic peoples republic of korea'         : [124.0913902, 37.5867855, 130.924647, 43.0089642], 
              'north korea'                                  : [124.0913902, 37.5867855, 130.924647, 43.0089642], 
              'portugal'                                     : [-31.5575303, 29.8288021, -6.1891593, 42.1543112], 
              'paraguay'                                     : [-62.6442036, -27.6063935, -54.258, -19.2876472], 
              'french polynesia'                             : [-154.9360599, -28.0990232, -134.244799, -7.6592173], 
              'qatar'                                        : [55.1726803, 25.2271767, 55.1741485, 25.2289841], 
              'reunion'                                      : [55.2164268, -21.3897308, 55.8366924, -20.8717136], 
              'romania'                                      : [20.2619773, 43.618682, 30.0454257, 48.2653964], 
              'russian federation'                           : [19.6389, 41.1850968, 180, 82.0586232], 
              'russia'                                       : [19.6389, 41.1850968, 180, 82.0586232], 
              'rwanda'                                       : [28.8617546, -2.8389804, 30.8990738, -1.0474083], 
              'saudi arabia'                                 : [121.0177023, 14.4840316, 121.0265441, 14.4846897], 
              'sudan'                                        : [6.48, 10.88, 6.52, 10.92], 
              'senegal'                                      : [-17.7862419, 12.2372838, -11.3458996, 16.6919712], 
              'singapore'                                    : [103.6920359, 1.1304753, 104.0120359, 1.4504753], 
              'south georgia and the south sandwich islands' : [-42.354739, -59.684, -25.8468303, -53.3500755], 
              'ascension and tristan da cunha saint helena'  : [-5.9973424, -16.23, -5.4234153, -15.704], 
              'saint helena'                                 : [-5.9973424, -16.23, -5.4234153, -15.704], 
              'svalbard and jan mayen'                       : [-9.6848146, 70.6260825, 34.6891253, 81.028076], 
              'solomon islands'                              : [155.3190556, -13.2424298, 170.3964667, -4.81085], 
              'sierra leone'                                 : [-13.5003389, 6.755, -10.271683, 9.999973], 
              'el salvador'                                  : [-90.1790975, 12.976046, -87.6351394, 14.4510488], 
              'san marino'                                   : [12.4033246, 43.8937002, 12.5160665, 43.992093], 
              'somalia'                                      : [15.246667, -13.200556, 15.286667, -13.160556], 
              'saint pierre and miquelon'                    : [-56.6972961, 46.5507173, -55.9033333, 47.365], 
              'serbia'                                       : [18.8142875, 42.2322435, 23.006309, 46.1900524], 
              'sao tome and principe'                        : [6.260642, -0.2135137, 7.6704783, 1.9257601], 
              'suriname'                                     : [-58.070833, 1.8312802, -53.8433358, 6.225], 
              'slovakia'                                     : [16.8331891, 47.7314286, 22.56571, 49.6138162], 
              'slovenia'                                     : [13.3754696, 45.4214242, 16.5967702, 46.8766816], 
              'sweden'                                       : [10.5930952, 55.1331192, 24.1776819, 69.0599699], 
              'swaziland'                                    : [55.1807318, 25.2156672, 55.18286, 25.217255], 
              'seychelles'                                   : [55.1816849, 25.2237725, 55.184065, 25.2250726], 
              'syrian arab republic'                         : [31.5494106, 27.1401861, 46.5494106, 42.1401861], 
              'syria'                                        : [31.5494106, 27.1401861, 46.5494106, 42.1401861], 
              'turks and caicos islands'                     : [-72.6799046, 20.9553418, -70.8643591, 22.1630989], 
              'chad'                                         : [13.47348, 7.44107, 24, 23.4975], 
              'togo'                                         : [-0.1439746, 5.926547, 1.8087605, 11.1395102], 
              'thailand'                                     : [97.3438072, 5.612851, 105.636812, 20.4648337], 
              'tajikistan'                                   : [67.3332775, 36.6711153, 75.1539563, 41.0450935], 
              'tokelau'                                      : [-172.7213673, -9.6442499, -170.9797586, -8.3328631], 
              'turkmenistan'                                 : [52.335076, 35.129093, 66.6895177, 42.7975571], 
              'timor-leste'                                  : [106.9549541, -6.3294157, 106.9551755, -6.3284839], 
              'tonga'                                        : [-179.3866055, -24.1034499, -173.5295458, -15.3655722], 
              'trinidad and tobago'                          : [-62.083056, 9.8732106, -60.2895848, 11.5628372], 
              'tunisia'                                      : [7.5219807, 30.230236, 11.8801133, 37.7612052], 
              'turkey'                                       : [25.6212891, 35.8076804, 44.8176638, 42.297], 
              'tuvalu'                                       : [-180, -10.9939389, 180, -5.4369611], 
              'taiwan'                                       : [6.0296969, 50.8832129, 6.0297969, 50.8833129], 
              'united republic of tanzania'                  : [39.2895223, -6.8127308, 39.2897105, -6.812538], 
              'tanzania'                                     : [39.2895223, -6.8127308, 39.2897105, -6.812538], 
              'uganda'                                       : [29.573433, -1.4823179, 35.000308, 4.2340766], 
              'ukraine'                                      : [22.137059, 44.184598, 40.2275801, 52.3791473], 
              'united states minor outlying islands'         : [-162.6816297, 6.1779744, -162.1339885, 6.6514388], 
              'uruguay'                                      : [-58.4948438, -35.7824481, -53.0755833, -30.0853962], 
              'united states'                                : [-125.0011, 24.9493, -66.9326, 49.5904], 
              'usa'                                          : [-125.0011, 24.9493, -66.9326, 49.5904], 
              'u.s.a'                                        : [-125.0011, 24.9493, -66.9326, 49.5904], 
              'united states of america'                     : [-125.0011, 24.9493, -66.9326, 49.5904], 
              'america'                                      : [-125.0011, 24.9493, -66.9326, 49.5904], 
              'uzbekistan'                                   : [55.9977865, 37.1821164, 73.1397362, 45.590118], 
              'holy see'                                     : [12.4457442, 41.9002044, 12.4583653, 41.9073912], 
              'saint vincent and the grenadines'             : [-61.6657471, 12.5166548, -60.9094146, 13.583], 
              'bolivarian republic of venezuela'             : [-73.3529632, 0.647529, -59.5427079, 15.9158431], 
              'venezuela'                                    : [-73.3529632, 0.647529, -59.5427079, 15.9158431], 
              'british virgin islands'                       : [-65.159094, 17.623468, -64.512674, 18.464984], 
              'u.s. virgin islands'                          : [-65.159094, 17.623468, -64.512674, 18.464984], 
              'us virgin islands'                            : [-65.159094, 17.623468, -64.512674, 18.464984], 
              'viet nam'                                     : [102.14441, 8.1790665, 114.3337595, 23.393395], 
              'vietnam'                                      : [102.14441, 8.1790665, 114.3337595, 23.393395], 
              'vanuatu'                                      : [166.3355255, -20.4627425, 170.449982, -12.8713777], 
              'wallis and futuna'                            : [-178.3873749, -14.5630748, -175.9190391, -12.9827961], 
              'yemen'                                        : [41.60825, 11.9084802, 54.7389375, 19], 
              'south africa'                                 : [16.3335213, -47.1788335, 38.2898954, -22.1250301], 
              'south america'                                : [-81.0, -56.0, -35.0, 11.0],
              'africa'                                       : [-20.0, -35.0, 50.0, 35.0], 
              'europe'                                       : [-10.0, 36.0, 40.0, 70.0], 
              'asia'                                         : [30.0, 10.0, 150.0, 55.0],
              'zambia'                                       : [21.9993509, -18.0765945, 33.701111, -8.2712822], 
              'zimbabwe'                                     : [25.2373, -22.4241096, 33.0683413, -15.6097033] 
             }
    
    if exact == False:
        domain[region][0] = np.floor(domain[region][0])
        domain[region][1] = np.floor(domain[region][1])
        domain[region][2] = np.ceil(domain[region][2])
        domain[region][3] = np.ceil(domain[region][3])
    
    val = domain.get(region.lower(), 'Invalid Region provided')
    if val == 'Invalid Date string':
        print(region + ' is not set up to for subsetting the domain. Please make changes to code and try again')
        exit()
    return(domain)

def extract_us_lat_lon_region_parallax_correction(region):
    '''
    Uses switch in order to return the latitude and longitude parallax corrections of US states and regions
    Args:
        region : State or sector of US to get latitude and longitude data for
    Keywords:
        None.
    Returns:
        pc : Longitude and latitude parallax corrections in degrees [x_pc, y_pc]
    '''
    region = region.lower()
    pc = {
          'alabama'                      : [0.047181483, 0.105333],
          'alaska'                       : [2.0170805, 0.4347033], 
          'arizona'                      : [0.18987493, 0.118084855],
          'arkansas'                     : [0.07514861, 0.11556531],
          'california'                   : [0.27882335, 0.13836992],
          'colorado'                     : [0.16829619, 0.14053455],
          'connecticut'                  : [0.012833186, 0.1492862],
          'delaware'                     : [0.0012274326, 0.1361893],
          'district of columbia'         : [0.008761978, 0.1347824],
          'washington dc'                : [0.008761978, 0.1347824],
          'florida'                      : [0.03155108, 0.08551337],
          'georgia'                      : [0.032714244, 0.10538049],
          'idaho'                        : [0.30530143, 0.18848412],
          'illinois'                     : [0.07278328, 0.14013281],
          'indiana'                      : [0.056634456, 0.13999222],
          'iowa'                         : [0.10164025, 0.15376815],
          'kansas'                       : [0.11870096, 0.13546413],
          'kentucky'                     : [0.049951598, 0.12975118],
          'louisiana'                    : [0.065283276, 0.098906524],
          'maine'                        : [0.037474282, 0.17274772],
          'maryland'                     : [0.0099939015, 0.13440578], 
          'massachusetts'                : [0.018591667, 0.1526565],
          'michigan'                     : [0.068071134, 0.17138793],
          'minnesota'                    : [0.12072617, 0.18353322],
          'mississippi'                  : [0.06096855, 0.105516315],
          'missouri'                     : [0.08490613, 0.13304436],
          'montana'                      : [0.27124283, 0.1946061],
          'nebraska'                     : [0.14025825, 0.15294404],
          'nevada'                       : [0.26495105, 0.14394666],
          'new hampshire'                : [0.020809742, 0.1646142],
          'new jersey'                   : [0.0027646595, 0.14150508],
          'new mexico'                   : [0.14894165, 0.11583067],
          'new york'                     : [0.011131603, 0.15659216],
          'north carolina'               : [0.020322405, 0.11667287],
          'north dakota'                 : [0.18296503, 0.194599],
          'ohio'                         : [0.03777938, 0.141969],
          'oklahoma'                     : [0.11014652, 0.11942014],
          'oregon'                       : [0.3758237, 0.18576586],
          'pennsylvania'                 : [0.01269481, 0.14629363],
          'puerto rico'                  : [0.026965572, 0.05276545],
          'rhode island'                 : [0.019530078, 0.14981595],
          'south carolina'               : [0.023903511, 0.10946053],
          'south dakota'                 : [0.15943064, 0.17051275],
          'tennessee'                    : [0.048085745, 0.119995825],
          'texas'                        : [0.10494422, 0.09970868],
          'united states virgin islands' : [0.032510683, 0.052317414],
          'virgin islands'               : [0.032510683, 0.052317414],
          'utah'                         : [0.21952337, 0.14627726],
          'vermont'                      : [0.015738694, 0.16376625],
          'virginia'                     : [0.019993756, 0.13024455],
          'washington'                   : [0.43653083, 0.21291439],
          'west virginia'                : [0.024084847, 0.13497369],
          'wisconsin'                    : [0.0894805, 0.17107269],
          'wyoming'                      : [0.21076538, 0.16590616],
          'southeast'                    : [0.041102234, 0.11009514],
          'southeast2'                   : [0.06290473, 0.10941378],
          'northeast'                    : [0.02601052, 0.17009985],
          'southwest'                    : [0.17816994, 0.116292976],
          'midwest'                      : [0.09697646, 0.1574287],
          'great plains'                 : [0.08975209, 0.13677296],
          'custom2'                      : [0.11870096, 0.13546413],
          '20211210_outbreak'            : [0.09697646, 0.1574287], 
          '20211215_outbreak'            : [0.10164025, 0.15376815],
          'hurricane_sam21'              : [0.07194763, 0.040349],
          'hurricane_dorian19'           : [0.0073492443, 0.08057583], 
          'central texas'                : [0.10494422, 0.09970868], 
          '2019146-147_region'           : [0.11864212, 0.13245392], 
          '2019146-147_region2'          : [0.11870096, 0.13546413], 
          '2019146-147_region3'          : [0.1428124, 0.14177759],
          '2020134-135_region'           : [0.15, 0.10],
          'conus'                        : [0.13978362, 0.11234573], 
          'argentina'                    : [0.055636175, 0.124635026]
          }
    val = pc.get(region.lower(), 'Invalid Region provided')
    if val == 'Invalid Date string':
        print(region + ' is not set up to for finding the parallax corrections. Please make changes to code and try again')
        exit()
    return(pc)

def estimate_sector_parallax(domain = [-105.5, 36.15, -79.0, 49.5]):
  '''
  Function to estimate the average longitude and latitude parallax correction for a given sector based upon the latitude and longitude bounds
  Args:
    domain : Longitude and latitude bounds of sector to get parallax correction for [x0, y0, x1, y1] 
  Returns:
    pc     : Longitude and latitude parallax corrections in degrees [x_pc, y_pc]
  '''
  x_mean = (domain[0] + domain[2])/2.0                                                                                                                                          #Calculate middle of domain to base parallax correction upon
  y_mean = (domain[1] + domain[3])/2.0                                                                                                                                          #Calculate middle of domain to base parallax correction upon
  if np.isnan(x_mean):
    r  = extract_us_lat_lon_region_parallax_correction('Kansas')
    pc = r['kansas']                                                                                                                                                            #Use Kansas for parallax corrections if mean values are NaNs due to meshgrid
    exit()
    
  else:
    r2   = extract_us_lat_lon_region('Texas')                                                                                                                                   #Get domain bounds of states and territory regions
    keys = list(r2.keys())                                                                                                                                                      #Get list of dictionary keys specifying states and territory regions
    pc   = None                                                                                                                                                                 #Initialize parallax correction as None 
    for i in keys:
      if ((x_mean >= r2[i][0]) and (x_mean <= r2[i][2]) and (y_mean >= r2[i][1]) and (y_mean <= r2[i][3])):                                                                     #Find states/boundaries in which x_mean and y_mean are located
        r  = extract_us_lat_lon_region_parallax_correction(i)                                                                                                                   #Extract parallax corrections for region 
        pc = r[i]                                                                                                                                                               #Copy parallax corrections for region 
        break 
  
  if pc == None:
    r2   = extract_lat_lon_region()                                                                                                                                             #Get domain bounds of states and territory regions
    keys = list(r2.keys())                                                                                                                                                      #Get list of dictionary keys specifying states and territory regions
    pc   = None                                                                                                                                                                 #Initialize parallax correction as None 
    for i in keys:
      if ((x_mean >= r2[i][0]) and (x_mean <= r2[i][2]) and (y_mean >= r2[i][1]) and (y_mean <= r2[i][3])):                                                                     #Find states/boundaries in which x_mean and y_mean are located
        r  = extract_goeseast_lat_lon_region_parallax_correction(i)                                                                                                             #Extract parallax corrections for region 
        pc = r[i]                                                                                                                                                               #Copy parallax corrections for region 
        break 
    
    if pc == None:
      print('x_mean and y_mean not within any regions???')
      print(x_mean)
      print(y_mean)
      print(domain)
      exit()
  
  return(pc)

def sat_info():
  sat = {
         'GOES-E'   : {'lon0' : 285.0, 'lat0' : 0.0, 'r0' : 35800000.0, 'r_E' : 6371009.0},
         'GOES-W'   : {'lon0' : 225.0, 'lat0' : 0.0, 'r0' : 35800000.0, 'r_E' : 6371009.0},
         'MTSAT-1R' : {'lon0' : 140.0, 'lat0' : 0.0, 'r0' : 35800000.0, 'r_E' : 6371009.0},
         }
  
  return(sat)
 
def read_goes_east_parallax_correction_ncdf(infile = '../../../goes-data/parallax_corrections/G-16.2021253.2210.plax.cldtop15.0-km.image.nc'):
  '''
  Reads the GOES east parallax correction file.
  Args:
      None.
  Keywords:
      infile : File name and path of GOES parallax correction file produced by Doug Spangenberg in McIdis. 
               DEFAULT = '../../../goes-data/parallax_corrections/G-16.2021253.2210.plax.cldtop15.0-km.image.nc'
  Returns:
      pc : Longitude and latitude parallax corrections in degrees [x_pc, y_pc]
  '''
  with Dataset(infile) as data:                                                                                                                                                 #Read GOES-E parallax correction file   
    lat   = np.copy(np.asarray(data.variables['Latitude']))                                                                                                                     #Read latitudes (deg)
    lat[lat == int(data.variables['Latitude'].missing_value)]  = np.nan
    lon   = np.copy(np.asarray(data.variables['Longitude']))                                                                                                                    #Read longitudes (deg)
    lon[lon == int(data.variables['Longitude'].missing_value)] = np.nan
    lat_c = np.copy(np.asarray(data.variables['LATCD'][0, :, :]))                                                                                                               #Read latitudes parallax correction distances (deg)
    lat_c[lat_c == int(data.variables['LATCD'].missing_value)] = np.nan
    lon_c = np.copy(np.asarray(data.variables['LONCD'][0, :, :]))                                                                                                               #Read longitudes parallax correction distances (deg)
    lon_c[lon_c == int(data.variables['LONCD'].missing_value)] = np.nan
  lon0, lat0, lon_c0, lat_c0 = lon[np.isfinite(lon) | np.isfinite(lat) | np.isfinite(lon_c) | np.isfinite(lat_c)], lat[np.isfinite(lon) | np.isfinite(lat) | np.isfinite(lon_c) | np.isfinite(lat_c)], lon_c[np.isfinite(lon) | np.isfinite(lat) | np.isfinite(lon_c) | np.isfinite(lat_c)], lat_c[np.isfinite(lon) | np.isfinite(lat) | np.isfinite(lon_c) | np.isfinite(lat_c)]  
  return(lon, lat, lon_c, lat_c)


def find_nearest_parallax_correction_on_grid(lon, lat):
  '''
  Find GOES satellite parallax corrections nearest to latitude and longitude grid points.
  Args:
      lon: Longitudes of points to find parallax corrections for (2D numpy array).
      lat: Latitudes of points to find parallax corrections for  (2D numpy array).
  Keywords:
      None.
  Returns:
      x_pc : Longitude parallax corrections in degrees. 2D array the same dimensions as lon and lat
      y_pc : Latitude parallax corrections in degrees. 2D array the same dimensions as lon and lat
  '''
  
  print('Extracting parallax for each point in domain')
  x_pc    = np.nan*np.zeros((lat.shape))                                                                                                                                        #Set x-coordinate parallax correction for entire domain to NaN values
  y_pc    = np.nan*np.zeros((lat.shape))                                                                                                                                        #Set y-coordinate parallax correction for entire domain to NaN values
  inds    = np.where((np.isfinite(lon)) & (np.isfinite(lat)))                                                                                                                   #Find lat/lon indices that ar not NaNs
  if len(inds[0]) == 0:
    print('No finite longitudes or latitudes???')
    print(lon)
    print(lat)
    exit()
  lon2    = lon[inds]                                                                                                                                                           #Extract all finite longitudes
  lat2    = lat[inds]                                                                                                                                                           #Extract all finite latitudes
  x_pc2   = np.nan*np.zeros((lon2.shape))                                                                                                                                       #Set 1D longitude parallax corrections for all finite points to NaN
  y_pc2   = np.nan*np.zeros((lat2.shape))                                                                                                                                       #Set 1D latitude parallax corrections for all finite points to NaN

#  results = [pool.apply(estimate_sector_parallax, kwds={'domain' : [lon2[row]-0.02, lat2[row]-0.02, lon2[row]+0.02, lat2[row]+0.02]}) for row in range(len(lon2))]              #Run estimate_sector_parallax function in parallel threads
  t0 = time.time()
#  counter = 0
  for q in range(5000):   
    pool    = mp.Pool(10)                                                                                                                                                        #Initialize 5 threads for multiprocessing
    if q == 0:
        start_index = 0
    else:
        start_index = end_index
    if q == 4999:
        end_index = len(lon2)    
    else:
        end_index = start_index + int(len(lon2)/5000.0)
    if end_index > len(lon2):
        end_index = len(lon2)
   # counter = counter+1
    results = [[row, pool.apply_async(estimate_sector_parallax, kwds={'domain' : [lon2[row]-0.02, lat2[row]-0.02, lon2[row]+0.02, lat2[row]+0.02]})] for row in range(start_index, end_index)]                     #Run estimate_sector_parallax function in parallel threads
#  results = [[row, pool.apply_async(estimate_sector_parallax, kwds={'domain' : [lon2[row]-0.02, lat2[row]-0.02, lon2[row]+0.02, lat2[row]+0.02]})] for row in range(len(lon2))]                     #Run estimate_sector_parallax function in parallel threads
    for idx, result in enumerate(results):
      idx0 = result[0]
      res  = result[1].get(timeout = 60)
      x_pc2[idx0] = res[0]                                                                                                                                                      #Find parallax correction for each job
      y_pc2[idx0] = res[1]                                                                                                                                                      #find parallax correction for each job
    pool.close()                                                                                                                                                                #Stop code until all parallel threads are complete
    pool.join()                                                                                                                                                                 #Stop code until all parallel threads are complete
    results = []
  x_pc[inds] = x_pc2                                                                                                                                                            #Set 2D longitude parallax corrections using 1D finite indices
  y_pc[inds] = y_pc2                                                                                                                                                            #Set 2D latitude parallax corrections using 1D finite indices
  if np.sum(np.isnan(x_pc)) != np.sum(np.isnan(lon)):
    print('Number of NaN values in each array are not the same????')
    print(np.sum(np.isnan(lon)))
    print(np.sum(np.isnan(x_pc)))
    exit()
    
  print('Parallax found for each point in domain')
  print('Time to extract pc at each point' + str(time.time() - t0))
  return(x_pc, y_pc)

def extract_us_lat_lon_region_parallax_correction2(region = None):
  r = extract_us_lat_lon_region(region = 'Texas')
  lon0, lat0, lon_c0, lat_c0 = read_goes_east_parallax_correction_ncdf()
  if region != None:
    locations = [region]
  else:  
    locations = list(r.keys())
  for l in locations:
    lon00, lat00, lon_c00, lat_c00 = lon0[( lon0 >= r[l][0] ) & ( lon0 <= r[l][2] ) & ( lat0 >= r[l][1] ) & ( lat0 <= r[l][3] )], lat0[( lon0 >= r[l][0] ) & ( lon0 <= r[l][2] ) & ( lat0 >= r[l][1] ) & ( lat0 <= r[l][3] )], lon_c0[( lon0 >= r[l][0] ) & ( lon0 <= r[l][2] ) & ( lat0 >= r[l][1] ) & ( lat0 <= r[l][3] )], lat_c0[( lon0 >= r[l][0] ) & ( lon0 <= r[l][2] ) & ( lat0 >= r[l][1] ) & ( lat0 <= r[l][3] )]
    if len(lon00) > 0:
      if (np.min(lon00) >= r[l][0] or np.max(lon00) <= r[l][2] or np.min(lat00) >= r[l][1] or np.max(lat00) <= r[l][3]):
        print(l + '[' + str(np.mean(lon_c00)) + ', ' + str(np.mean(lat_c00)) + ']')
        print()
      else:
        print('Array is less than min/max???')
        print(np.min(lon00), np.max(lon00), np.min(lat00), np.max(lat00))
        print(r[l][0], r[l][2], r[l][1], r[l][3])
        print(l)
        exit()
    
    else:
      print('No data within range for ' + l)

def read_dcotss_er2_plane(inroot      = '../../../misc-data0/DCOTSS/aircraft/', 
                          bucket_name = 'misc-data0',
                          use_local   = False, 
                          verbose     = True):
    '''
    Uses switch in order to return the latitude and longitude bounds of US states and regions
    Args:
        date_str : Date string in format YYYY-MM-DD hh:mm:ss
    Keywords:
        inroot      : STRING specifying location of ER2 airplane file
                      DEFAULT = '../../../misc-data0/DCOTSS/aircraft/'
        bucket_name : STRING specifying GCP bucket the plane ER2 csv file is stored in. 
                      DEFAULT = 'misc-data0'
        use_local   : IF keyword set (True), read plane csv file from local storage directoy
                      DEFAULT = False -> read csv file stored on GCP
        verbose     : IF keyword set (True), print verbose informational messages.              
    Returns:
        Pandas data frame containing the DCOTSS flight data. 
    '''
    inroot      = os.path.realpath(inroot)                                                                                                     #Create link to real path so compatible with Mac
    if use_local == True:
        csv_file = sorted(glob.glob(inroot + '/*.csv'))                                                                                        #Find ER2 aircraft csv file
        if len(csv_file) <= 0 or len(csv_file) > 1:
            print('No or too many ER2 aircraft csv file found????')
            print(inroot)
            print(csv_file)
            exit()
        if verbose == True: print('Reading aircraft file ' + csv_file[0])
        data = pd.read_csv(csv_file[0])                                                                                                        #Read ER2 aircraft csv file
    else:
        gcs_prefix = re.split('misc-data0/', inroot)[1]                                                                                        #Path to ER2 aircraft csv file on GCP bucket
        csv_file   = list_gcs(bucket_name, gcs_prefix + '/', ['.csv'], delimiter = '/')                                                        #Find ER2 aircraft csv file
        if len(csv_file) <= 0 or len(csv_file) > 1:
            print('No or too many ER2 aircraft csv file found????')
            print(gcs_prefix)
            print(bucket_name)
            print(inroot)
            print(csv_file)
            exit()
        data = load_csv_gcs(bucket_name, csv_file[0], skiprows = None)                                                                         #Read ER2 aircraft csv file
    
    return(data)


def doctss_read_lat_lon_alt_trajectory_particle_ncdf(yyyymmdd, index0, 
                                                     inroot      = '../../../misc-data0/DCOTSS/traj/',
                                                     bucket_name = 'misc-data0',
                                                     use_local   = False, 
                                                     verbose     = True):
    '''
    Finds and reads back trajectory netCDF file based on yyyymmdd date string.
    Args:
        yyyymmdd : Date string in format YYYYMMDD
        index0   : Index within flight to read. (ex. 00092)
    Keywords:
        inroot      : STRING specifying location of ER2 airplane file
                      DEFAULT = '../../../misc-data0/DCOTSS/traj/'
        bucket_name : STRING specifying GCP bucket the plane ER2 csv file is stored in. 
                      DEFAULT = 'misc-data0'
        use_local   : IF keyword set (True), read plane csv file from local storage directoy
                      DEFAULT = False -> read csv file stored on GCP
        verbose     : IF keyword set (True), print verbose informational messages.              
    Returns:
        Latitude, Longitude, and altitude of each particle in back trajecto 
    '''
    inroot  = os.path.realpath(inroot)                                                                                                         #Extract date string from input root
    rflight_days = [20210716, 20210720, 20210723, 20210726, 20210729, 20210802, 20210806, 20210810, 20210814, 20210817, 20210819]              #DCOTSS research flight days
    ind     = (np.asarray(rflight_days) - int(yyyymmdd)).tolist().index(0) + 1                                                                 #Determine which research flight ot read
    ind     = "{:02d}".format(ind)                                                                                                             #Extract flight number as string
    index0  = "{:05d}".format(index0)                                                                                                          #Extract index number as string
    nc_file = sorted(glob.glob(inroot + '/RF' + ind + '*/DCOTSS_ERA5_*backward_RF' + ind + '_index_' + index0 + '.nc', recursive = True))      #Find back trajectory file in local storage
    if use_local == False:
        if len(nc_file) == 0:
            gcs_prefix = re.split('misc-data0/', inroot)[1]                                                                                    #Path to ER2 aircraft csv file on GCP bucket
            nc_file    = list_gcs(bucket_name, gcs_prefix + '/', ['_index_' + index0, '.nc'], delimiter = '*/')                                #Find back trajectory netCDF file
            if len(nc_file) > 1:
                print('Too many back trajectory netCDF files file found????')
                print(gcs_prefix)
                print(bucket_name)
                print(inroot)
                print(nc_file)
                exit()
            if len(nc_file) > 0:
                indir = os.path.dirname(nc_file[0])
                os.makedirs(os.path.join(inroot, os.path.basename(indir)), exist_ok = True)                                                    #Create directory to download file to if it does not already exist
                download_ncdf_gcs(bucket_name, nc_file[0], os.path.join(inroot, os.path.basename(indir)))                                      #Download back trajectory netCDF file
                nc_file = sorted(glob.glob(inroot + '/RF' + ind + '*/DCOTSS_ERA5_*backward_RF' + ind + '_index_' + index0 + '.nc', recursive = True))  #Find back trajectory file in local storage
            else:
                return([-1], [-1], [-1], [-1])
    if len(nc_file) > 1:
        print('Too many back trajectory netCDF files file found????')
        print(inroot)
        print(nc_file)
        exit()
    if len(nc_file) == 0:
        return([-1], [-1], [-1], [-1])

    if verbose == True: print('Reading back trajectory netCDF file ' + nc_file[0])
    
    data  = Dataset(nc_file[0])                                                                                                                #Read in back trajectory file
    lon   = np.copy(data.variables['Longitude'])                                                                                               #Extract longitude of particles
    lat   = np.copy(data.variables['Latitude'])                                                                                                #Extract latitude of particles
    Z     = np.copy(data.variables['Z'])                                                                                                       #Extract geopotential altitude of particles (m)
    dates = np.copy(data.variables['Time_ISO'])                                                                                                #Extract dates of particle initializations
    dates = [datetime.strptime(''.join(dates[d].astype('U13').tolist()), "%Y-%m-%d %H:%M:%SZ") for d in range(len(dates))]
    data.close()                                                                                                                               #Close the netCDF file that was read
    lon  = convert_longitude(lon, to_model = False)                                                                                            #Extract longitudes in terms of -180° to 180°
    
    return(lon, lat, Z, dates)


def extract_lat_lon_region():
    '''
    Uses switch in order to return the latitude and longitude bounds of regions outside of US but associated with parallax corrections we know
    Args:
        None.
    Keywords:
        None.
    Returns:
        domain : Longitude and latitude [x0, y0, x1, y1]
    '''
    domain = {
              '-150_15_-145_20' : [-150, 15, -145, 20], 
              '-150_20_-145_25' : [-150, 20, -145, 25], 
              '-150_25_-145_30' : [-150, 25, -145, 30], 
              '-150_30_-145_35' : [-150, 30, -145, 35], 
              '-150_35_-145_40' : [-150, 35, -145, 40], 
              '-150_40_-145_45' : [-150, 40, -145, 45], 
              '-150_45_-145_50' : [-150, 45, -145, 50], 
              '-145_15_-140_20' : [-145, 15, -140, 20], 
              '-145_20_-140_25' : [-145, 20, -140, 25], 
              '-145_25_-140_30' : [-145, 25, -140, 30], 
              '-145_30_-140_35' : [-145, 30, -140, 35], 
              '-145_35_-140_40' : [-145, 35, -140, 40], 
              '-145_40_-140_45' : [-145, 40, -140, 45], 
              '-145_45_-140_50' : [-145, 45, -140, 50], 
              '-145_50_-140_55' : [-145, 50, -140, 55], 
              '-145_55_-140_60' : [-145, 55, -140, 60], 
              '-140_15_-135_20' : [-140, 15, -135, 20], 
              '-140_20_-135_25' : [-140, 20, -135, 25], 
              '-140_25_-135_30' : [-140, 25, -135, 30], 
              '-140_30_-135_35' : [-140, 30, -135, 35], 
              '-140_35_-135_40' : [-140, 35, -135, 40], 
              '-140_40_-135_45' : [-140, 40, -135, 45], 
              '-140_45_-135_50' : [-140, 45, -135, 50], 
              '-140_50_-135_55' : [-140, 50, -135, 55], 
              '-140_55_-135_60' : [-140, 55, -135, 60], 
              '-135_15_-130_20' : [-135, 15, -130, 20], 
              '-135_20_-130_25' : [-135, 20, -130, 25], 
              '-135_25_-130_30' : [-135, 25, -130, 30], 
              '-135_30_-130_35' : [-135, 30, -130, 35], 
              '-135_35_-130_40' : [-135, 35, -130, 40], 
              '-135_40_-130_45' : [-135, 40, -130, 45], 
              '-135_45_-130_50' : [-135, 45, -130, 50], 
              '-135_50_-130_55' : [-135, 50, -130, 55], 
              '-135_55_-130_60' : [-135, 55, -130, 60], 
              '-130_15_-125_20' : [-130, 15, -125, 20], 
              '-130_20_-125_25' : [-130, 20, -125, 25], 
              '-130_25_-125_30' : [-130, 25, -125, 30], 
              '-130_30_-125_35' : [-130, 30, -125, 35], 
              '-130_35_-125_40' : [-130, 35, -125, 40], 
              '-130_40_-125_45' : [-130, 40, -125, 45], 
              '-130_45_-125_50' : [-130, 45, -125, 50], 
              '-130_50_-125_55' : [-130, 50, -125, 55], 
              '-130_55_-125_60' : [-130, 55, -125, 60], 
              '-125_15_-120_20' : [-125, 15, -120, 20], 
              '-125_20_-120_25' : [-125, 20, -120, 25], 
              '-125_25_-120_30' : [-125, 25, -120, 30], 
              '-125_30_-120_35' : [-125, 30, -120, 35], 
              '-125_35_-120_40' : [-125, 35, -120, 40], 
              '-125_40_-120_45' : [-125, 40, -120, 45], 
              '-125_45_-120_50' : [-125, 45, -120, 50], 
              '-125_50_-120_55' : [-125, 50, -120, 55], 
              '-125_55_-120_60' : [-125, 55, -120, 60], 
              '-120_45_-115_50' : [-120, 45, -115, 50], 
              '-120_50_-115_55' : [-120, 50, -115, 55], 
              '-120_55_-115_60' : [-120, 55, -115, 60], 
              '-115_45_-110_50' : [-115, 45, -110, 50], 
              '-115_50_-110_55' : [-115, 50, -110, 55], 
              '-115_55_-110_60' : [-115, 55, -110, 60], 
              '-110_45_-105_50' : [-110, 45, -105, 50], 
              '-110_50_-105_55' : [-110, 50, -105, 55], 
              '-110_55_-105_60' : [-110, 55, -105, 60], 
              '-105_45_-100_50' : [-105, 45, -100, 50], 
              '-105_50_-100_55' : [-105, 50, -100, 55], 
              '-105_55_-100_60' : [-105, 55, -100, 60], 
              '-100_45_-95_50'  : [-100, 45, -95, 50], 
              '-100_50_-95_55'  : [-100, 50, -95, 55], 
              '-100_55_-95_60'  : [-100, 55, -95, 60], 
              '-95_45_-90_50'   : [-95, 45, -90, 50], 
              '-95_50_-90_55'   : [-95, 50, -90, 55], 
              '-95_55_-90_60'   : [-95, 55, -90, 60], 
              '-90_45_-85_50'   : [-90, 45, -85, 50], 
              '-90_50_-85_55'   : [-90, 50, -85, 55], 
              '-90_55_-85_60'   : [-90, 55, -85, 60], 
              '-85_45_-80_50'   : [-85, 45, -80, 50], 
              '-85_50_-80_55'   : [-85, 50, -80, 55], 
              '-85_55_-80_60'   : [-85, 55, -80, 60], 
              '-80_45_-75_50'   : [-80, 45, -75, 50], 
              '-80_50_-75_55'   : [-80, 50, -75, 55], 
              '-80_55_-75_60'   : [-80, 55, -75, 60], 
              '-75_45_-70_50'   : [-75, 45, -70, 50], 
              '-75_50_-70_55'   : [-75, 50, -70, 55], 
              '-75_55_-70_60'   : [-75, 55, -70, 60], 
              '-70_45_-65_50'   : [-70, 45, -65, 50], 
              '-70_50_-65_55'   : [-70, 50, -65, 55], 
              '-70_55_-65_60'   : [-70, 55, -65, 60], 
              '-85_10_-80_15'   : [-85, 10, -80, 15], 
              '-85_15_-80_20'   : [-85, 15, -80, 20], 
              '-85_20_-80_25'   : [-85, 20, -80, 25], 
              '-85_25_-80_30'   : [-85, 25, -80, 30], 
              '-85_30_-80_35'   : [-85, 30, -80, 35], 
              '-85_35_-80_40'   : [-85, 35, -80, 40], 
              '-85_40_-80_45'   : [-85, 40, -80, 45], 
              '-80_10_-75_15'   : [-80, 10, -75, 15], 
              '-80_15_-75_20'   : [-80, 15, -75, 20], 
              '-80_20_-75_25'   : [-80, 20, -75, 25], 
              '-80_25_-75_30'   : [-80, 25, -75, 30], 
              '-80_30_-75_35'   : [-80, 30, -75, 35], 
              '-80_35_-75_40'   : [-80, 35, -75, 40], 
              '-80_40_-75_45'   : [-80, 40, -75, 45], 
              '-75_10_-70_15'   : [-75, 10, -70, 15], 
              '-75_15_-70_20'   : [-75, 15, -70, 20], 
              '-75_20_-70_25'   : [-75, 20, -70, 25], 
              '-75_25_-70_30'   : [-75, 25, -70, 30], 
              '-75_30_-70_35'   : [-75, 30, -70, 35], 
              '-75_35_-70_40'   : [-75, 35, -70, 40], 
              '-75_40_-70_45'   : [-75, 40, -70, 45], 
              '-70_10_-65_15'   : [-70, 10, -65, 15], 
              '-70_15_-65_20'   : [-70, 15, -65, 20], 
              '-70_20_-65_25'   : [-70, 20, -65, 25], 
              '-70_25_-65_30'   : [-70, 25, -65, 30], 
              '-70_30_-65_35'   : [-70, 30, -65, 35], 
              '-70_35_-65_40'   : [-70, 35, -65, 40], 
              '-70_40_-65_45'   : [-70, 40, -65, 45],
              '-150_5_-145_10'  : [-150, 5, -145, 10], 
              '-150_10_-145_15' : [-150, 10, -145, 15], 
              '-145_5_-140_10'  : [-145, 5, -140, 10], 
              '-145_10_-140_15' : [-145, 10, -140, 15], 
              '-140_5_-135_10'  : [-140, 5, -135, 10], 
              '-140_10_-135_15' : [-140, 10, -135, 15], 
              '-135_5_-130_10'  : [-135, 5, -130, 10], 
              '-135_10_-130_15' : [-135, 10, -130, 15], 
              '-130_5_-125_10'  : [-130, 5, -125, 10], 
              '-130_10_-125_15' : [-130, 10, -125, 15], 
              '-125_5_-120_10'  : [-125, 5, -120, 10], 
              '-125_10_-120_15' : [-125, 10, -120, 15], 
              '-120_5_-115_10'  : [-120, 5, -115, 10], 
              '-120_10_-115_15' : [-120, 10, -115, 15], 
              '-120_15_-115_20' : [-120, 15, -115, 20], 
              '-120_20_-115_25' : [-120, 20, -115, 25], 
              '-120_25_-115_30' : [-120, 25, -115, 30], 
              '-120_30_-115_35' : [-120, 30, -115, 35], 
              '-115_5_-110_10'  : [-115, 5, -110, 10], 
              '-115_10_-110_15' : [-115, 10, -110, 15], 
              '-115_15_-110_20' : [-115, 15, -110, 20], 
              '-115_20_-110_25' : [-115, 20, -110, 25], 
              '-115_25_-110_30' : [-115, 25, -110, 30], 
              '-115_30_-110_35' : [-115, 30, -110, 35], 
              '-110_5_-105_10'  : [-110, 5, -105, 10],
              '-110_10_-105_15' : [-110, 10, -105, 15], 
              '-110_15_-105_20' : [-110, 15, -105, 20], 
              '-110_20_-105_25' : [-110, 20, -105, 25], 
              '-110_25_-105_30' : [-110, 25, -105, 30], 
              '-110_30_-105_35' : [-110, 30, -105, 35], 
              '-105_5_-100_10'  : [-105, 5, -100, 10], 
              '-105_10_-100_15' : [-105, 10, -100, 15],
              '-105_15_-100_20' : [-105, 15, -100, 20], 
              '-105_20_-100_25' : [-105, 20, -100, 25], 
              '-105_25_-100_30' : [-105, 25, -100, 30],
              '-105_30_-100_35' : [-105, 30, -100, 35], 
              '-100_5_-95_10'   : [-100, 5, -95, 10], 
              '-100_10_-95_15'  : [-100, 10, -95, 15], 
              '-100_15_-95_20'  : [-100, 15, -95, 20], 
              '-100_20_-95_25'  : [-100, 20, -95, 25], 
              '-100_25_-95_30'  : [-100, 25, -95, 30], 
              '-100_30_-95_35'  : [-100, 30, -95, 35], 
              '-95_5_-90_10'    : [-95, 5, -90, 10], 
              '-95_10_-90_15'   : [-95, 10, -90, 15], 
              '-95_15_-90_20'   : [-95, 15, -90, 20], 
              '-95_20_-90_25'   : [-95, 20, -90, 25], 
              '-95_25_-90_30'   : [-95, 25, -90, 30], 
              '-95_30_-90_35'   : [-95, 30, -90, 35], 
              '-90_5_-85_10'    : [-90, 5, -85, 10], 
              '-90_10_-85_15'   : [-90, 10, -85, 15], 
              '-90_15_-85_20'   : [-90, 15, -85, 20], 
              '-90_20_-85_25'   : [-90, 20, -85, 25], 
              '-90_25_-85_30'   : [-90, 25, -85, 30], 
              '-90_30_-85_35'   : [-90, 30, -85, 35], 
              '-85_5_-80_10'    : [-85, 5, -80, 10], 
              '-80_5_-75_10'    : [-80, 5, -75, 10], 
              '-75_5_-70_10'    : [-75, 5, -70, 10], 
              '-70_5_-65_10'    : [-70, 5, -65, 10], 
              '-155_5_-150_10'  : [-155, 5, -150, 10], 
              '-155_10_-150_15' : [-155, 10, -150, 15], 
              '-155_15_-150_20' : [-155, 15, -150, 20], 
              '-155_20_-150_25' : [-155, 20, -150, 25], 
              '-155_25_-150_30' : [-155, 25, -150, 30],
              '-155_30_-150_35' : [-155, 30, -150, 35],
              '-155_35_-150_40' : [-155, 35, -150, 40],
              '-155_40_-150_45' : [-155, 40, -150, 45],
              '-155_45_-150_50' : [-155, 45, -150, 50], 
              '-55_50_-50_55'   : [-55, 50, -50, 55], 
              '-155_25_-150_30' : [-155, 25, -150, 30]}
    
    return(domain)

def extract_goeseast_lat_lon_region_parallax_correction(region):
    '''
    Uses switch in order to return the latitude and longitude parallax corrections of regions outside of the US states
    Args:
        region : State or sector of US to get latitude and longitude data for
    Keywords:
        None.
    Returns:
        pc : Longitude and latitude parallax corrections in degrees [x_pc, y_pc]
    '''
    region = region.lower()
    pc = {
          '-150_15_-145_20' : [0.9381072,0.08593847],
          '-150_20_-145_25' : [1.0316764,0.116488084],
          '-150_25_-145_30' : [1.1500506,0.15134889],
          '-150_30_-145_35' : [1.2801032,0.19093049],
          '-150_35_-145_40' : [1.458583,0.23867424],
          '-150_40_-145_45' : [1.6968815,0.29680732],
          '-150_45_-145_50' : [1.916579,0.34794074],
          '-145_15_-140_20' : [0.5933213,0.07113823],
          '-145_20_-140_25' : [0.6456162,0.09546943],
          '-145_25_-140_30' : [0.72102636,0.12349503],
          '-145_30_-140_35' : [0.8282043,0.15700224],
          '-145_35_-140_40' : [0.9852697,0.19900917],
          '-145_40_-140_45' : [1.2180371,0.25404233],
          '-145_45_-140_50' : [1.5566466,0.32756495],
          '-145_50_-140_55' : [1.9265767,0.4136096],
          '-145_55_-140_60' : [2.2893555,0.4967882],
          '-140_15_-135_20' : [0.42521122,0.06399034],
          '-140_20_-135_25' : [0.46049118,0.08542696],
          '-140_25_-135_30' : [0.510598,0.10971929],
          '-140_30_-135_35' : [0.5812113,0.13819584],
          '-140_35_-135_40' : [0.6823653,0.17291237],
          '-140_40_-135_45' : [0.830692,0.21713124],
          '-140_45_-135_50' : [1.0595661,0.27694148],
          '-140_50_-135_55' : [1.4337256,0.36351657],
          '-140_55_-135_60' : [1.9917547,0.4844322],
          '-135_15_-130_20' : [0.3251,0.059814144],
          '-135_20_-130_25' : [0.35105428,0.07961139],
          '-135_25_-130_30' : [0.38766322,0.10184449],
          '-135_30_-130_35' : [0.43883404,0.12757073],
          '-135_35_-130_40' : [0.51124614,0.15838975],
          '-135_40_-130_45' : [0.61632365,0.19691741],
          '-135_45_-130_50' : [0.77471757,0.2474815],
          '-135_50_-130_55' : [1.0269121,0.31822535],
          '-135_55_-130_60' : [1.4667709,0.426577],
          '-130_15_-125_20' : [0.25792846,0.057118107],
          '-130_20_-125_25' : [0.27796817,0.07589109],
          '-130_25_-125_30' : [0.3060861,0.09680664],
          '-130_30_-125_35' : [0.34528744,0.12082264],
          '-130_35_-125_40' : [0.40035844,0.14931138],
          '-130_40_-125_45' : [0.4794169,0.184405],
          '-130_45_-125_50' : [0.5970817,0.22967272],
          '-130_50_-125_55' : [0.78195256,0.29163432],
          '-130_55_-125_60' : [1.0956209,0.38343647],
          '-125_15_-120_20' : [0.20903523,0.05526626],
          '-125_20_-120_25' : [0.22494596,0.07331478],
          '-125_25_-120_30' : [0.24723941,0.09335863],
          '-125_30_-120_35' : [0.27818635,0.11621889],
          '-125_35_-120_40' : [0.3214567,0.14313427],
          '-125_40_-120_45' : [0.3832712,0.17600024],
          '-125_45_-120_50' : [0.47434267,0.21783686],
          '-125_50_-120_55' : [0.6158846,0.27424777],
          '-125_55_-120_60' : [0.8514836,0.35583675],
          '-150_45_-145_50' : [1.916579,0.34794074],
          '-145_45_-140_50' : [1.5566466,0.32756495],
          '-145_50_-140_55' : [1.9265767,0.4136096],
          '-145_55_-140_60' : [2.2893555,0.4967882],
          '-140_45_-135_50' : [1.0595661,0.27694148],
          '-140_50_-135_55' : [1.4337256,0.36351657],
          '-140_55_-135_60' : [1.9917547,0.4844322],
          '-135_45_-130_50' : [0.77471757,0.2474815],
          '-135_50_-130_55' : [1.0269121,0.31822535],
          '-135_55_-130_60' : [1.4667709,0.426577],
          '-130_45_-125_50' : [0.5970817,0.22967272],
          '-130_50_-125_55' : [0.78195256,0.29163432],
          '-130_55_-125_60' : [1.0956209,0.38343647],
          '-125_45_-120_50' : [0.47434267,0.21783686],
          '-125_50_-120_55' : [0.6158846,0.27424777],
          '-125_55_-120_60' : [0.8514836,0.35583675],
          '-120_45_-115_50' : [0.38335142,0.20963757],
          '-120_50_-115_55' : [0.49446654,0.2622341],
          '-120_55_-115_60' : [0.6775049,0.33718637],
          '-115_45_-110_50' : [0.3116552,0.20359772],
          '-115_50_-110_55' : [0.40042663,0.25361466],
          '-115_55_-110_60' : [0.54502445,0.32398662],
          '-110_45_-105_50' : [0.25297335,0.19914374],
          '-110_50_-105_55' : [0.32388705,0.24721213],
          '-110_55_-105_60' : [0.43872625,0.31429932],
          '-105_45_-100_50' : [0.20307952,0.19579332],
          '-105_50_-100_55' : [0.2593708,0.24246942],
          '-105_55_-100_60' : [0.3500474,0.30712593],
          '-100_45_-95_50'  : [0.15929505,0.19327119],
          '-100_50_-95_55'  : [0.2030621,0.23889545],
          '-100_55_-95_60'  : [0.27345058,0.30184633],
          '-95_45_-90_50'   : [0.119813636,0.19142663],
          '-95_50_-90_55'   : [0.15250064,0.23626225],
          '-95_55_-90_60'   : [0.20496525,0.29792917],
          '-90_45_-85_50'   : [0.083307825,0.19011222],
          '-90_50_-85_55'   : [0.10591379,0.23442978],
          '-90_55_-85_60'   : [0.14218965,0.29523197],
          '-85_45_-80_50'   : [0.048722923,0.18929753],
          '-85_50_-80_55'   : [0.061936524,0.23329712],
          '-85_55_-80_60'   : [0.083041765,0.29344872],
          '-80_45_-75_50'   : [0.01536194,0.18888433],
          '-80_50_-75_55'   : [0.019429997,0.23269527],
          '-80_55_-75_60'   : [0.026170788,0.29283312],
          '-75_45_-70_50'   : [0.017860444,0.18890247],
          '-75_50_-70_55'   : [0.022595257,0.23272246],
          '-75_55_-70_60'   : [0.030374035,0.2928997],
          '-70_45_-65_50'   : [0.051438767,0.18934684],
          '-70_50_-65_55'   : [0.06540811,0.2333724],
          '-70_55_-65_60'   : [0.087654926,0.29353023],
          '-85_10_-80_15'   : [0.021431686,0.035331015],
          '-85_15_-80_20'   : [0.022571664,0.050467353],
          '-85_20_-80_25'   : [0.024204656,0.06670728],
          '-85_25_-80_30'   : [0.026463944,0.0845305],
          '-85_30_-80_35'   : [0.029594263,0.10456991],
          '-85_35_-80_40'   : [0.03391231,0.12770154],
          '-85_40_-80_45'   : [0.039973162,0.15524082],
          '-85_45_-80_50'   : [0.048722923,0.18929753],
          '-85_50_-80_55'   : [0.061936524,0.23329712],
          '-85_55_-80_60'   : [0.083041765,0.29344872],
          '-80_10_-75_15'   : [0.006752557,0.035259776],
          '-80_15_-75_20'   : [0.0071062744,0.050390486],
          '-80_20_-75_25'   : [0.0076196496,0.0666199],
          '-80_25_-75_30'   : [0.0083352765,0.0843937],
          '-80_30_-75_35'   : [0.0093174465,0.1043601],
          '-80_35_-75_40'   : [0.010649883,0.12747629],
          '-80_40_-75_45'   : [0.01261364,0.15500768],
          '-80_45_-75_50'   : [0.01536194,0.18888433],
          '-80_50_-75_55'   : [0.019429997,0.23269527],
          '-80_55_-75_60'   : [0.026170788,0.29283312],
          '-75_10_-70_15'   : [0.007867145,0.035262663],
          '-75_15_-70_20'   : [0.00828018,0.05039258],
          '-75_20_-70_25'   : [0.008878607,0.06662453],
          '-75_25_-70_30'   : [0.009707192,0.08439996],
          '-75_30_-70_35'   : [0.0108491285,0.104366906],
          '-75_35_-70_40'   : [0.012401547,0.1274848],
          '-75_40_-70_45'   : [0.014673563,0.15502071],
          '-75_45_-70_50'   : [0.017860444,0.18890247],
          '-75_50_-70_55'   : [0.022595257,0.23272246],
          '-75_55_-70_60'   : [0.030374035,0.2928997],
          '-70_10_-65_15'   : [0.022624593,0.035336427],
          '-70_15_-65_20'   : [0.023826621,0.050475463],
          '-70_20_-65_25'   : [0.025555022,0.066718295],
          '-70_25_-65_30'   : [0.027934464,0.08454593],
          '-70_30_-65_35'   : [0.031242657,0.104592144],
          '-70_35_-65_40'   : [0.035798926,0.12773016],
          '-70_40_-65_45'   : [0.042200986,0.15527412],
          '-70_45_-65_50'   : [0.051438767,0.18934684],
          '-70_50_-65_55'   : [0.06540811,0.2333724],
          '-70_55_-65_60'   : [0.087654926,0.29353023],
          '-150_5_-145_10'  : [0.8377926,0.034614906],
          '-150_10_-145_15' : [0.8765185,0.059129797],
          '-150_25_-145_30' : [1.1500506,0.15134889],
          '-150_30_-145_35' : [1.2801032,0.19093049],
          '-145_5_-140_10'  : [0.5355992,0.028979842],
          '-145_10_-140_15' : [0.55767965,0.04928179],
          '-145_15_-140_20' : [0.5933213,0.07113823],
          '-145_20_-140_25' : [0.6456162,0.09546943],
          '-145_25_-140_30' : [0.72102636,0.12349503],
          '-145_30_-140_35' : [0.8282043,0.15700224],
          '-140_5_-135_10'  : [0.38606745,0.026216345],
          '-140_10_-135_15' : [0.40113318,0.044472564],
          '-140_15_-135_20' : [0.42521122,0.06399034],
          '-140_20_-135_25' : [0.46049118,0.08542696],
          '-140_25_-135_30' : [0.510598,0.10971929],
          '-140_30_-135_35' : [0.5812113,0.13819584],
          '-135_5_-130_10'  : [0.29629472,0.024596399],
          '-135_10_-130_15' : [0.30740663,0.04166561],
          '-135_15_-130_20' : [0.3251,0.059814144],
          '-135_20_-130_25' : [0.35105428,0.07961139],
          '-135_25_-130_30' : [0.38766322,0.10184449],
          '-135_30_-130_35' : [0.43883404,0.12757073],
          '-130_5_-125_10'  : [0.23560938,0.023544377],
          '-130_10_-125_15' : [0.24422826,0.039850008],
          '-130_15_-125_20' : [0.25792846,0.057118107],
          '-130_20_-125_25' : [0.27796817,0.07589109],
          '-130_25_-125_30' : [0.3060861,0.09680664],
          '-130_30_-125_35' : [0.34528744,0.12082264],
          '-125_5_-120_10'  : [0.19122456,0.02281108],
          '-125_10_-120_15' : [0.19809963,0.038589638],
          '-125_15_-120_20' : [0.20903523,0.05526626],
          '-125_20_-120_25' : [0.22494596,0.07331478],
          '-125_25_-120_30' : [0.24723941,0.09335863],
          '-125_30_-120_35' : [0.27818635,0.11621889],
          '-120_5_-115_10'  : [0.15685378,0.022286166],
          '-120_10_-115_15' : [0.16241395,0.037673216],
          '-120_15_-115_20' : [0.1712638,0.053922247],
          '-120_20_-115_25' : [0.18411176,0.07146126],
          '-120_25_-115_30' : [0.20206982,0.09086654],
          '-120_30_-115_35' : [0.2269737,0.112926684],
          '-115_5_-110_10'  : [0.12901258,0.02189486],
          '-115_10_-110_15' : [0.13353461,0.037004225],
          '-115_15_-110_20' : [0.14072904,0.05292419],
          '-115_20_-110_25' : [0.15117967,0.07008824],
          '-115_25_-110_30' : [0.1657633,0.08903099],
          '-115_30_-110_35' : [0.18594845,0.11049442],
          '-110_5_-105_10'  : [0.10560392,0.021591661],
          '-110_10_-105_15' : [0.10927845,0.03648679],
          '-110_15_-105_20' : [0.11512808,0.052169736],
          '-110_20_-105_25' : [0.123596445,0.06904493],
          '-110_25_-105_30' : [0.13541971,0.08764622],
          '-110_30_-105_35' : [0.15175398,0.10866437],
          '-105_5_-100_10'  : [0.085317075,0.021368539],
          '-105_10_-100_15' : [0.08827098,0.03609591],
          '-105_15_-100_20' : [0.09296337,0.051593054],
          '-105_20_-100_25' : [0.099761724,0.06825996],
          '-105_25_-100_30' : [0.1092369,0.086591],
          '-105_30_-100_35' : [0.122317806,0.10727524],
          '-100_5_-95_10'   : [0.06725853,0.021190558],
          '-100_10_-95_15'  : [0.06956073,0.035795953],
          '-100_15_-95_20'  : [0.07324643,0.05116195],
          '-100_20_-95_25'  : [0.07857913,0.06766124],
          '-100_25_-95_30'  : [0.0860135,0.08579521],
          '-100_30_-95_35'  : [0.09624533,0.106227204],
          '-95_5_-90_10'    : [0.05076277,0.021059405],
          '-95_10_-90_15'   : [0.05250238,0.035583045],
          '-95_15_-90_20'   : [0.055275656,0.05083564],
          '-95_20_-90_25'   : [0.059288252,0.067215964],
          '-95_25_-90_30'   : [0.064873666,0.08520353],
          '-95_30_-90_35'   : [0.07255397,0.10545226],
          '-90_5_-85_10'    : [0.035382338,0.020987848],
          '-90_10_-85_15'   : [0.0365943,0.03542479],
          '-90_15_-85_20'   : [0.038521904,0.050610077],
          '-90_20_-85_25'   : [0.041315548,0.06690503],
          '-90_25_-85_30'   : [0.045190953,0.0847925],
          '-90_30_-85_35'   : [0.050529946,0.10490358],
          '-85_5_-80_10'    : [0.020734398,0.020913927],
          '-85_10_-80_15'   : [0.021431686,0.035331015],
          '-85_15_-80_20'   : [0.022571664,0.050467353],
          '-85_20_-80_25'   : [0.024204656,0.06670728],
          '-85_25_-80_30'   : [0.026463944,0.0845305],
          '-85_30_-80_35'   : [0.029594263,0.10456991],
          '-80_5_-75_10'    : [0.0065271948,0.02085913],
          '-80_10_-75_15'   : [0.006752557,0.035259776],
          '-80_15_-75_20'   : [0.0071062744,0.050390486],
          '-80_20_-75_25'   : [0.0076196496,0.0666199],
          '-80_25_-75_30'   : [0.0083352765,0.0843937],
          '-80_30_-75_35'   : [0.0093174465,0.1043601],
          '-75_5_-70_10'    : [0.007608674,0.02086145],
          '-75_10_-70_15'   : [0.007867145,0.035262663],
          '-75_15_-70_20'   : [0.00828018,0.05039258],
          '-75_20_-70_25'   : [0.008878607,0.06662453],
          '-75_25_-70_30'   : [0.009707192,0.08439996],
          '-75_30_-70_35'   : [0.0108491285,0.104366906],
          '-70_5_-65_10'    : [0.02188849,0.02091826],
          '-70_10_-65_15'   : [0.022624593,0.035336427],
          '-70_15_-65_20'   : [0.023826621,0.050475463],
          '-70_20_-65_25'   : [0.025555022,0.066718295],
          '-70_25_-65_30'   : [0.027934464,0.08454593],
          '-70_30_-65_35'   : [0.031242657,0.104592144],
          '-155_5_-150_10'  : [1.2795843,0.042537063],
          '-155_10_-150_15' : [1.3247126,0.072284535],
          '-155_15_-150_20' : [1.3899313,0.103894316],
          '-155_20_-150_25' : [1.4793016,0.13663463],
          '-155_25_-150_30' : [1.5603842,0.16275406],
          '-155_30_-150_35' : [1.5603842,0.104366906],
          '-155_35_-150_40' : [1.5603842,0.1274848],
          '-155_40_-150_45' : [1.5603842,0.29680732],
          '-155_45_-150_50' : [1.5603842,0.18890247],
          '-150_5_-145_10'  : [0.8377926,0.034614906],
          '-150_10_-145_15' : [0.8765185,0.059129797],
          '-150_25_-145_30' : [1.1500506,0.15134889],
          '-150_30_-145_35' : [1.2801032,0.19093049],
          '-150_35_-145_40' : [1.458583,0.23867424],
          '-150_40_-145_45' : [1.6968815,0.29680732],
          '-55_50_-50_55'   : [0.20731361,0.2391383], 
          '-150_45_-145_50' : [1.916579,0.34794074]}
    val = pc.get(region.lower(), 'Invalid Region provided')
    if val == 'Invalid Date string':
        print(region + ' is not set up to for finding the parallax corrections. Please make changes to code and try again')
        exit()
    return(pc)
  
def main():
  rdr_sat_utils_jwc()

  
if __name__ == '__main__':
  main()
  