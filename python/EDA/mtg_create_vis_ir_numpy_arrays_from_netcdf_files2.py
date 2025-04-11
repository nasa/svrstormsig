#+
# Name:
#     mtg_create_vis_ir_numpy_arrays_from_netcdf_files2.py
# Purpose:
#     MAIN FUNCTION. This is a script to run (by importing) programs that create numpy files for IR, VIS, solar zenith angle, and GLM data.
#     Creates IR, VIS, and GLM numpy arrays for each day. Also creates corresponding csv files which yields the date, filepath/name to the 
#     original GOES netCDF files, and whether the scan is a night or day time scan.
#     These are the files that get fed into the model.
# Calling sequence:
#     import mtg_create_vis_ir_numpy_arrays_from_netcdf_files2
#     mtg_create_vis_ir_numpy_arrays_from_netcdf_files2.mtg_create_vis_ir_numpy_arrays_from_netcdf_files2()
# Input:
#     None.
# Functions:
#     fetch_convert_ir     : Clips edges of temperature data in degrees kelvin from min_value to max_value and returns normalized BT data
#     fetch_convert_vis    : Clips edges of reflectance data. Normalizes the VIS data by the solar zenith angle and returns normalized VIS data
#     fetch_convert_glm    : Clips edges of flash extent density data. Normalizes the VIS data by the solar zenith angle and returns normalized GLM data
#     fetch_convert_irdiff : Clips edges of 6.3 - 10.2 micron BTs. Normalizes the data and returns normalized BT_difference data
#     merge_csv_files      : Merges ALL csv files in the json root directory. Contains label csv file and VIS/IR/GLM numpy csv file
# Output:
#     Creates numpy files that are normalized for model input.
# Keywords:
#      inroot           : STRING specifying input root directory containing original IR/VIS/GLM netCDF files
#                         DEFAULT = '../../../mtg-data/20241029/'
#      layered_root     : STRING specifying directory containing the combined IR/VIS/GLM netCDF file (created by combine_ir_glm_vis.py)
#                         DEFAULT = '../../../mtg-data/combined_nc_dir/'
#      outroot          : STRING specifying directory to send the VIS, IR, and GLM numpy files as well as corresponding csv files
#                         DEFAULT = '../../../mtg-data/labelled/'
#      json_root        : STRING specifying root directory to read the json labeled mask csv file (created by labelme_seg_mask2.py)
#                         DEFAULT = '../../../mtg-data/labelled/'
#      date_range       : List containing start date and end date in YYYY-MM-DD hh:mm:ss format ("%Y-%m-%d %H:%M:%S") to run over. (ex. date_range = ['2021-04-30 19:00:00', '2021-05-01 04:00:00'])
#                         DEFAULT = [] -> follow start_index keyword or do all files in VIS/IR directories
#      meso_sector      : LONG integer specifying the mesoscale domain sector to use to create maps (= 1 or 2). DEFAULT = 2 (sector 2)                
#      domain_sector    : STRING specifying the satellite domain sector to use to create maps (ex. 'Full', 'CONUS'). DEFAULT = None -> use meso_sector               
#      ir_min_value     : Minimum value used to clip the IR brightness temperatures. All values below min are set to min. DEFAULT = 180 K
#      ir_max_value     : Maximum value used to clip the IR brightness temperatures. All values above max are set to max. DEFAULT = 230 K
#      no_write_ir      : IF keyword set (True), do not write the IR numpy data arrays. DEFAULT = False
#      no_write_vis     : IF keyword set (True), do not write the VIS numpy data arrays. DEFAULT = False
#      no_write_irdiff  : IF keyword set (True), do not write the IR differnece (11 micron - 6.3 micron numpy file). DEFAULT = True.
#      no_write_glm     : IF keyword set (True), do not write the GLM numpy data arrays. DEFAULT = False
#      no_write_csv     : IF keyword set (True), do not write the csv numpy data arrays. no_write_ir cannot be set if this is False.
#      no_write_sza     : IF keyword set (True), do not write the Solar zenith angle numpy data arrays (degrees). DEFAULT = False.
#      run_gcs          : IF keyword set (True), read and write everything directly from the google cloud platform.
#                         DEFAULT = False
#      use_local        : IF keyword set (True), read locally stored files.                  
#                         DEFAULT = False (read from GCP.)
#      real_time        : IF keyword set (True), run the code in real time by only grabbing the most recent file to image and write. 
#                         Files are also output to a real time directory.
#                         DEFAULT = False
#      del_local        : IF keyword set (True), delete local copies of files after writing them to google cloud. Only does anything if run_gcs = True.
#                         DEFAULT = False
#      og_bucket_name   : Google cloud storage bucket to read raw IR/GLM/VIS files.
#                         DEFAULT = 'mtg-data'
#      comb_bucket_name : Google cloud storage bucket to read raw IR/GLM/VIS files.
#                         DEFAULT = 'ir-vis-sandwhich'
#      proc_bucket_name : Google cloud storage bucket to write normalized numpy files used for model input and csv files containing each day information
#                         DEFAULT = 'aacp-proc-data'
#      use_native_ir    : IF keyword set (True), write files for native IR satellite resolution.
#                         DEFAULT = False -> use satellite VIS resolution
#      verbose          : IF keyword set (True), print verbose informational messages to terminal.
# Author and history:
#     John W. Cooney           2025-01-03. (Adapted from create_vis_ir_numpy_arrays_from_netcdf_files2 but specific to MTG)
#
#-

from netCDF4 import Dataset
import numpy as np
import glob
import math
import os
from os import listdir
from os.path import isfile, join, basename, exists
import re
import cv2
import csv
import pandas as pd
from datetime import datetime
import scipy.ndimage as ndimage
from threading import Thread
import sys 
#sys.path.insert(1, '../')
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.getcwd()))
from new_model.gcs_processing import write_to_gcs, download_ncdf_gcs, list_gcs, load_csv_gcs
from gridrad.rdr_sat_utils_jwc import match_sat_data_to_datetime
from glm_gridder.run_create_image_from_three_modalities import sort_goes_comb_files, sort_mtg_irvis_files
#import matplotlib.pyplot as plt

def fetch_convert_ir(_combined_nc, lon_shape, lat_shape, min_value = 180.0, max_value = 230.0):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Clips edges of temperature data in degrees kelvin from min_value to max_value". 
    Previously was 190.0 K and 233.15 K. Values closer to min_value are masked closer to 1.
  
    Args:
      _combined_nc: Numpy file contents
      lon_shape   : Shape of numpy 2D longitude array
      lat_shape   : Shape of numpy 2D latitude array
    Keywords:
      min_value: Minimum temperature value to cut off for input BT data. All BT < min_value = min_value. DEFAULT = 180.0  
      max_value: Maximum temperature value to cut off for input BT data. All BT > max_value = max_value. DEFAULT = 230.0   
    Returns:    
      ir_dat: 2000 x 2000 IR data numpy array after IR range conversion. Array is resized using cv2.resize with cv2.INTER_NEAREST interpolation
    '''
    ir_dat = np.copy(np.asarray(_combined_nc.variables['ir_brightness_temperature'][:], dtype = np.float32))[0, :, :]                          #Copy IR data into a numpy array                        

#Test of SEVIRI file
#     ir_dat = np.copy(np.asarray(_combined_nc.variables['channel_9'][:], dtype = np.float32))[:, :]                                             #Copy IR data into a numpy array                        
#     dd = np.load('/Users/jwcooney/Downloads/bt_c9_230121T1300.npz')
#     lon    = np.copy(np.asarray(_combined_nc.variables['lon']))                                                                                #Copy array of longitudes into lon variable
#     lat    = np.copy(np.asarray(_combined_nc.variables['lat']))                                                                                #Copy array of latitudes into lat variable
#     print(lon.shape)
#     print(lat.shape)
#     ir_dat = dd['bt']
#     print(ir_dat.shape)
#     if ir_dat.shape[0] != lon_shape[0] or ir_dat.shape[1] != lat_shape[0]:
#         ir_dat = cv2.resize(ir_dat, (lon_shape[0], lat_shape[0]), interpolation=cv2.INTER_NEAREST)                                             #Upscale the ir data array to VIS resolution
    if ir_dat.shape[0] != lon_shape[0] or ir_dat.shape[1] != lon_shape[1]:
        ir_dat = cv2.resize(ir_dat, (lon_shape[0], lon_shape[1]), interpolation=cv2.INTER_NEAREST)                                             #Upscale the ir data array to VIS resolution
    na  = (ir_dat < 0)
    na2 = ((ir_dat > max_value) & (ir_dat <= 260))
    ir_dat[ir_dat < min_value] = min_value                                                                                                     #Clip all BT below min value to min value
    ir_dat[ir_dat > max_value] = max_value                                                                                                     #Clip all BT above max value to max value
    ir_dat = np.true_divide(ir_dat - min_value, min_value - max_value)                                                                         #Normalize the IR BT data by the max and min values
    ir_dat += 1
    if (np.amax(ir_dat) > 1 or np.amin(ir_dat) < 0):                                                                                           #Check to make sure IR data is properly normalized between 0 and 1
        print('IR data is not normalized properly between 0 and 1??')
        exit()
    ir_dat[na] = -1                                                                                                                            #Set NaN values to -1 so they can later be recognized and the OT/AACP model removes any chance that they could be valid detections. Set to max weight when input into model though to avoid faulty detections along borders of no data regions
#    ir_dat[na2] = -2                                                                                                                            #Set NaN values to -1 so they can later be recognized and the OT/AACP model removes any chance that they could be valid detections. Set to max weight when input into model though to avoid faulty detections along borders of no data regions
   
    return(ir_dat)

def fetch_convert_trop(_combined_nc, lon_shape, lat_shape, min_value = -15.0, max_value = 20.0):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Clips edges of temperature data in degrees kelvin from min_value to max_value". 
    Values closer to min_value are masked closer to 1.
  
    Args:
      _combined_nc: Numpy file contents
      lon_shape   : Shape of numpy 2D longitude array
      lat_shape   : Shape of numpy 2D latitude array
    Keywords:
      min_value: Minimum temperature value to cut off for input BT-tropT data. All BT - tropT < min_value = min_value. DEFAULT = -15.0
      max_value: Maximum temperature value to cut off for input BT-tropT data. All BT - tropT > max_value = max_value. DEFAULT = 20.0   
    Returns:    
      ir_dat: 2000 x 2000 IR data numpy array after IR range conversion. Array is resized using cv2.resize with cv2.INTER_NEAREST interpolation
    '''
    ir_dat = np.copy(np.asarray(_combined_nc.variables['ir_brightness_temperature'][:], dtype = np.float32))[0, :, :]                          #Copy IR data into a numpy array                        
    tr_dat = np.copy(np.asarray(_combined_nc.variables['tropopause_temperature'][:], dtype = np.float32))[0, :, :]                             #Copy tropopapuse data into a numpy array                        
    if ir_dat.shape[0] != lon_shape[0] or ir_dat.shape[1] != lon_shape[1]:
        ir_dat = cv2.resize(ir_dat, (lon_shape[0], lon_shape[1]), interpolation=cv2.INTER_NEAREST)                                             #Upscale the ir data array to VIS resolution
        tr_dat = cv2.resize(tr_dat, (lon_shape[0], lon_shape[1]), interpolation=cv2.INTER_NEAREST)                                             #Upscale the ir data array to VIS resolution
    d_dat = ir_dat - tr_dat
    na = (ir_dat < 0)
    d_dat[d_dat < min_value] = min_value                                                                                                       #Clip all BT-tropT below min value to min value
    d_dat[d_dat > max_value] = max_value                                                                                                       #Clip all BT-tropT above max value to max value
    d_dat = np.true_divide(d_dat - min_value, min_value - max_value)                                                                           #Normalize the IR BT-tropT data by the max and min values
    d_dat += 1
    if (np.amax(d_dat) > 1 or np.amin(d_dat) < 0):                                                                                             #Check to make sure IR-tropT data is properly normalized between 0 and 1
        print('IR data is not normalized properly between 0 and 1??')
        exit()
    d_dat[na] = -1                                                                                                                             #Set NaN values to -1 so they can later be recognized and the OT/AACP model removes any chance that they could be valid detections. Set to max weight when input into model though to avoid faulty detections along borders of no data regions
    d_dat[(d_dat > max_value)] = -1                                                                                                                            #Set NaN values to -2 so they can later be recognized and the OT/AACP model removes any chance that they could be valid detections. Set to max weight when input into model though to avoid faulty detections along borders of no data regions
    return(d_dat)

def fetch_convert_vis(_combined_nc, min_value = 0.0, max_value = 1.0, no_write_vis = False):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Clips edges of reflectance data. Normalizes the VIS data by the solar zenith angle.
  
    Args:
      _combined_nc: Numpy file contents
    Keywords:
      min_value    : Minimum temperature value to cut off for input BT data. All BT < min_value = min_value. DEFAULT = 0.0  
      max_value    : Maximum temperature value to cut off for input BT data. All BT > max_value = max_value. DEFAULT = 1.0   
      no_write_vis : IF keyword set (True), do not write the VIS numpy data arrays. (returns vis data array full of zeroes). DEFAULT = False.
    Returns:    
      vis_dat: 2000 x 2000 VIS data numpy array
      tod    : STRING specifying if the file is considered a night time file or day time file
      zen    : 2000 x 2000 solar zenith angle data numpy arrayy
    '''
    if no_write_vis == False:
        vis_dat = np.copy(np.asarray(_combined_nc.variables['visible_reflectance'][:], dtype = np.float32))[0, :, :]                           #Copy VIS data into a numpy array                        
    zen = np.copy(np.asarray(_combined_nc.variables['solar_zenith_angle'][:], dtype = np.float32))[0, :, :]                                    #Copy solar zenith angle into variable
    if _combined_nc.variables['solar_zenith_angle'].units == 'radians': 
        zen = math.degrees(zen)                                                                                                                #Calculate solar zenith angle in degrees
    
#    mid_zen = zen[int(vis_dat.shape[0]/2), int(vis_dat.shape[1]/2)]                                                                           #Calculate solar zenith angle for mid point of domain (degrees)
#    mid_zen = np.nanmax(zen)                                                                                                                   #Find maximum solar zenith angle within domain. Needed to set up this way due to huge conus and full disk regions.
    mid_zen = np.sum(zen > 85.0)/(zen.shape[0]*zen.shape[1])
    if no_write_vis == False:
        vis_dat[vis_dat < min_value] = min_value                                                                                               #Clip all reflectance below min value to min value
        vis_dat[vis_dat > max_value] = max_value                                                                                               #Clip all reflectance above max value to max value
        if (np.amax(vis_dat) > 1 or np.amin(vis_dat) < 0):                                                                                     #Check to make sure IR data is properly normalized between 0 and 1
            print('VIS data is not normalized properly between 0 and 1??')
            exit()
        if min_value != 0.0 or max_value != 1.0:
            vis_dat = np.true_divide(vis_dat - min_value, max_value - min_value)                                                               #Normalize the IR BT data by the max and min values
        if (np.amax(vis_dat) > 1 or np.amin(vis_dat) < 0):                                                                                     #Check to make sure IR data is properly normalized between 0 and 1
            print('VIS data is not normalized properly between 0 and 1??')
            exit()
    else:
        vis_dat = zen*0.0
#    tod = 'day' if (mid_zen < 85.0) else 'night'                                                                                              #Set tod variable to night or day depending on the zenith angle in the middle of the scene
    tod = 'day' if (mid_zen < 0.05) else 'night'                                                                                               #Set tod variable to night or day depending on the zenith angle in the middle of the scene
    return(vis_dat, tod, zen)

def fetch_convert_snowice(_combined_nc, min_value = 0.0, max_value = 1.0):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Clips edges of reflectance data. Normalizes the VIS data by the solar zenith angle.
  
    Args:
      _combined_nc: Numpy file contents
    Keywords:
      min_value: Minimum reflectance value to cut off for input snow/ice channel data. All reflectance < min_value = min_value. DEFAULT = 0.0  
      max_value: Maximum reflectance value to cut off for input snow/ice channel data. All reflectance > max_value = max_value. DEFAULT = 1.0   
    Returns:    
      snowice_dat: 2000 x 2000 VIS data numpy array
      tod       : STRING specifying if the file is considered a night time file or day time file
      zen       : 2000 x 2000 solar zenith angle data numpy arrayy
    '''
    snowice_dat = np.copy(np.asarray(_combined_nc.variables['snowice_reflectance'][:], dtype = np.float32))[0, :, :]                           #Copy snowice data into a numpy array                        
    snowice_dat[snowice_dat < min_value] = min_value                                                                                           #Clip all reflectance below min value to min value
    snowice_dat[snowice_dat > max_value] = max_value                                                                                           #Clip all reflectance above max value to max value
    if (np.amax(snowice_dat) > 1 or np.amin(snowice_dat) < 0):                                                                                 #Check to make sure snow/ice channel data is properly normalized between 0 and 1
        print('snowice data is not normalized properly between 0 and 1??')
        exit()
#    tod = 'day' if (mid_zen < 85.0) else 'night'                                                                                               #Set tod variable to night or day depending on the zenith angle in the middle of the scene
#    tod = 'day' if (np.nanmax(zen) < 85.0) else 'night'                                                                                       #Set tod variable to night or day depending on the zenith angle in the middle of the scene

    return(snowice_dat)#, tod, zen)

def fetch_convert_cirrus(_combined_nc, min_value = 0.0, max_value = 1.0):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Clips edges of reflectance data. Normalizes the VIS data by the solar zenith angle.
  
    Args:
      _combined_nc: Numpy file contents
    Keywords:
      min_value: Minimum reflectance value to cut off for input snow/ice channel data. All reflectance < min_value = min_value. DEFAULT = 0.0  
      max_value: Maximum reflectance value to cut off for input snow/ice channel data. All reflectance > max_value = max_value. DEFAULT = 1.0   
    Returns:    
      snowice_dat: 2000 x 2000 VIS data numpy array
      tod       : STRING specifying if the file is considered a night time file or day time file
      zen       : 2000 x 2000 solar zenith angle data numpy arrayy
    '''
    cirrus_dat = np.copy(np.asarray(_combined_nc.variables['cirrus_reflectance'][:], dtype = np.float32))[0, :, :]                             #Copy cirrus data into a numpy array                        
    cirrus_dat[cirrus_dat < min_value] = min_value                                                                                             #Clip all reflectance below min value to min value
    cirrus_dat[cirrus_dat > max_value] = max_value                                                                                             #Clip all reflectance above max value to max value
    if (np.amax(cirrus_dat) > 1 or np.amin(cirrus_dat) < 0):                                                                                   #Check to make sure cirrus channel data is properly normalized between 0 and 1
        print('cirrus data is not normalized properly between 0 and 1??')
        exit()
#    tod = 'day' if (mid_zen < 85.0) else 'night'                                                                                               #Set tod variable to night or day depending on the zenith angle in the middle of the scene
#    tod = 'day' if (np.nanmax(zen) < 85.0) else 'night'                                                                                       #Set tod variable to night or day depending on the zenith angle in the middle of the scene

    return(cirrus_dat)#, tod, zen)

def fetch_convert_dirtyirdiff(_combined_nc, lon_shape, lat_shape, min_value = -1.0, max_value = 2.0):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Clips edges of temperature data in degrees kelvin from min_value to max_value". 
    Values closer to min_value are masked closer to 0.
  
    Args:
      _combined_nc: Numpy file contents
      lon_shape   : Shape of numpy 2D longitude array
      lat_shape   : Shape of numpy 2D latitude array
    Keywords:
      min_value: Minimum temperature value to cut off for input BT data. All BT < min_value = min_value. DEFAULT = -1.0  
      max_value: Maximum temperature value to cut off for input BT data. All BT > max_value = max_value. DEFAULT = 2.0   
    Returns:    
      ir_dat: 2000 x 2000 IR data numpy array after IR range conversion. Array is resized using cv2.resize with cv2.INTER_NEAREST interpolation
    '''
    ir_dat = np.copy(np.asarray(_combined_nc.variables['dirtyir_brightness_temperature_diff'][:], dtype = np.float32))[0, :, :]                #Copy IR data into a numpy array                        
    if ir_dat.shape[0] != lon_shape[0] or ir_dat.shape[1] != lon_shape[1]:
        ir_dat = cv2.resize(ir_dat, (lon_shape[0], lon_shape[1]), interpolation=cv2.INTER_NEAREST)                                             #Upscale the ir data array to VIS resolution
    ir_dat[ir_dat < min_value] = min_value                                                                                                     #Clip all BT below min value to min value
    ir_dat[ir_dat > max_value] = max_value                                                                                                     #Clip all BT above max value to max value
    ir_dat = np.true_divide(ir_dat - min_value, max_value - min_value)                                                                         #Normalize the IR BT data by the max and min values
    if (np.amax(ir_dat) > 1 or np.amin(ir_dat) < 0):                                                                                           #Check to make sure IR data is properly normalized between 0 and 1
        print('IR data is not normalized properly between 0 and 1??')
        exit()
    return(ir_dat)

def fetch_convert_glm(_combined_nc, lon_shape, lat_shape, min_value = 0.0, max_value = 20.0):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Normalizes GLM flash extent density data. 
  
    Args:
      _combined_nc: Numpy file contents
      lon_shape   : Shape of numpy 2D longitude array
      lat_shape   : Shape of numpy 2D latitude array
    Keywords:
      min_value: Minimum flash extent density counts value to cut off for input GLM data. All counts < min_value = min_value. DEFAULT = 0.0  
      max_value: Maximum flash extent density counts value to cut off for input GLM data. All counts > max_value = max_value DEFAULT = 20.0   
    Returns:    
      glm_dat: 2000 x 2000 GLM data numpy array after GLM range conversion. Array is resized using cv2.resize with cv2.INTER_NEAREST interpolation
    '''
    glm_dat = np.copy(np.asarray(_combined_nc['glm_flash_extent_density'][:], dtype = np.float32))[0, :, :]                                    #Copy GLM flash extent density data into a numpy array  
    if glm_dat.shape[0] != lon_shape[0] or glm_dat.shape[1] != lon_shape[1]:
        glm_dat = cv2.resize(glm_dat, (lon_shape[0], lon_shape[1]), interpolation=cv2.INTER_NEAREST)                                           #Upscale the GLM data array to VIS resolution
        glm_dat = ndimage.gaussian_filter(glm_dat, sigma=1.0, order=0)                                                                         #Smooth the GLM data using a Gaussian filter (higher sigma = more blurriness)
    glm_dat[glm_dat < min_value] = min_value                                                                                                   #Clip all reflectance below min value to min value
    glm_dat[glm_dat > max_value] = max_value                                                                                                   #Clip all reflectance above max value to max value
    glm_dat = np.true_divide(glm_dat - min_value, max_value - min_value)                                                                       #Normalize the GLM data by the max and min values
    if (np.amax(glm_dat) > 1 or np.amin(glm_dat) < 0):                                                                                         #Check to make sure IR data is properly normalized between 0 and 1
        print('VIS data is not normalized properly between 0 and 1??')
        exit()
    
    return(glm_dat)

def fetch_convert_irdiff(_combined_nc, lon_shape, lat_shape, min_value = -20.0, max_value = 10.0):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Clips edges of temperature data in degrees kelvin from min_value to max_value". 
    Values closer to max_value are masked closer to 1.
    Args:
      _combined_nc: Numpy file contents
      lon_shape   : Shape of numpy 2D longitude array
      lat_shape   : Shape of numpy 2D latitude array
    Keywords:
      min_value: Minimum brightness temperature difference value to cut off for input BT data. All BT_diff < min_value = min_value. DEFAULT = -20.0
      max_value: Maximum brightness temperature difference value to cut off for input BT data. All BT_diff > max_value = max_value. DEFAULT = 10.0  
    Returns:    
      ir_dat: 2000 x 2000 IR data numpy array after IR range conversion. Array is resized using cv2.resize with cv2.INTER_NEAREST interpolation
    '''
#    ir_img = upscale_img_to_fit(ir_raw_img, vis_img)                                                                                          #Resize IR image to fit Visible data image  
    ir_dat = np.copy(np.asarray(_combined_nc.variables['ir_brightness_temperature_diff'][:], dtype = np.float32))[0, :, :]                     #Copy IR data into a numpy array                        
    if ir_dat.shape[0] != lon_shape[0] or ir_dat.shape[1] != lon_shape[1]:
        ir_dat = cv2.resize(ir_dat, (lon_shape[0], lon_shape[1]), interpolation=cv2.INTER_NEAREST)                                             #Upscale the ir data array to VIS resolution
#       ir_dat = zoom(ir_dat, 4, order=2)                                                                                                      #Upscale the ir data array to VIS resolution   (OLD)                  
    ir_dat[ir_dat < min_value] = min_value                                                                                                     #Clip all BT below min value to min value
    ir_dat[ir_dat > max_value] = max_value                                                                                                     #Clip all BT above max value to max value
    ir_dat = np.true_divide(ir_dat - min_value, max_value - min_value)                                                                         #Normalize the IR BT data by the max and min values
    if (np.amax(ir_dat) > 1 or np.amin(ir_dat) < 0):                                                                                           #Check to make sure IR data is properly normalized between 0 and 1
        print('IR data is not normalized properly between 0 and 1??')
        exit()
    return(ir_dat)
    
def merge_csv_files(json_root, run_gcs = False, proc_bucket_name = 'aacp-proc-data'):
    '''
    Merges ALL csv files in the json root directory. Contains label csv file and VIS/IR/GLM numpy csv file
  
    Args:
      json_root: STRING specifying root directory. (ex. '/Users/jwcooney/python/code/goes-data/labelled/2020134/M2/')
    Keywords:
      run_gcs          : Read files from GCP. DEFAULT = False -> read local files.
      proc_bucket_name : GCP bucket to read processed data. DEFAULT = 'aacp-proc-data 
    Output:    
      Writes merged csv file 
    Returns:    
      Filename and path of merged csv file
    '''
    json_root = os.path.realpath(json_root)
    sector = os.path.basename(json_root)
    rm = 0
    if run_gcs == True:
        prefix   = os.path.join(os.path.basename(re.split(os.sep + 'labelled' + os.sep, json_root)[0]), 'labelled', re.split(os.sep + 'labelled' + os.sep, json_root)[1])
        json_csv = list_gcs(proc_bucket_name, prefix, ['.csv'], delimiter = '*/')
        if os.path.join(prefix, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv') in json_csv: 
            json_csv.remove(os.path.join(prefix, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                                #Do not merge previously merged ir, vis, glm, and json data frame
        fb       = [os.path.basename(x) for x in json_csv]
        fb2      = [x[-7:] for x in fb]
        if 'old.csv' in fb2: 
            json_csv = [json_csv[i] for i, j in enumerate(fb2) if j != 'old.csv']                                                              #Need to remove any csv files that have old in the name
        fb       = [os.path.basename(x) for x in json_csv]
        dir0     = [os.path.dirname(x) for x in json_csv]
        if 'zrel_ncdf_filenames_with_npy_files.csv' in fb: 
#            json_csv.remove(os.path.join(dir0[fb.index('zrel_ncdf_filenames_with_npy_files.csv')], 'zrel_ncdf_filenames_with_npy_files.csv'))  #Need to merge radar csv file in special way in order to match up nearest times
            json_csv = [json_csv[i] for i, j in enumerate(fb) if j != 'zrel_ncdf_filenames_with_npy_files.csv']                                #Need to remove any csv files that have old in the name
            rm = 1
    else:
        json_csv = glob.glob(os.path.join(json_root, '**', '*.csv'), recursive = True)                                                          #Extract all json labeled csv file names
        if os.path.join(json_root, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv') in json_csv: 
            json_csv.remove(os.path.join(json_root, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                             #Do not merge previously merged ir, vis, glm, and json data frame
        fb       = [os.path.basename(x) for x in json_csv]
        fb2      = [x[-7:] for x in fb]
        if 'old.csv' in fb2: 
            json_csv = [json_csv[i] for i, j in enumerate(fb2) if j != 'old.csv']                                                              #Need to remove any csv files that have old in the name
        fb       = [os.path.basename(x) for x in json_csv]
        dir0     = [os.path.dirname(x) for x in json_csv]        
        if 'zrel_ncdf_filenames_with_npy_files.csv' in fb: 
#            json_csv.remove(os.path.join(dir0[fb.index('zrel_ncdf_filenames_with_npy_files.csv')], 'zrel_ncdf_filenames_with_npy_files.csv'))  #Need to merge radar csv file in special way in order to match up nearest times
            json_csv = [json_csv[i] for i, j in enumerate(fb) if j != 'zrel_ncdf_filenames_with_npy_files.csv']                                #Need to remove any csv files that have old in the name
            rm = 1
    if len(json_csv) > 0 or rm == 1:
        for j, file0 in enumerate(json_csv):
            if run_gcs == True:
                if j == 0: 
                    ir_vis_glm_json = load_csv_gcs(proc_bucket_name, json_csv[j])                                                              #Create pandas data structure containing first read in csv file
                else: 
                    j_data = load_csv_gcs(proc_bucket_name, json_csv[j])
                    ir_vis_glm_json = ir_vis_glm_json.merge(j_data, on = 'date_time', how = 'outer', sort = True)                              #Merge csv file based upon date              
            else:
                if j == 0: 
                    ir_vis_glm_json = pd.read_csv(file0)                                                                                       #Create pandas data structure containing first read in csv file
                else: 
                    j_data = pd.read_csv(file0)                                                                                                #Read specified csv file
                    ir_vis_glm_json = ir_vis_glm_json.merge(j_data, on = 'date_time', how = 'outer', sort = True)                              #Merge csv file based upon date
        ir_vis_glm_json.reset_index(drop=True, inplace=True)                                                                                   #Drop the index values between csv files. Retain only specific ones
        if rm == 1:
            if run_gcs == True:
                j_data = load_csv_gcs(proc_bucket_name, os.path.join(dir0[fb.index('zrel_ncdf_filenames_with_npy_files.csv')], 'zrel_ncdf_filenames_with_npy_files.csv'))   #Read specified radar data csv file
            else:
                j_data = pd.read_csv(os.path.join(dir0[fb.index('zrel_ncdf_filenames_with_npy_files.csv')], 'zrel_ncdf_filenames_with_npy_files.csv'))                      #Read specified radar data csv file
            j_data['ir_files'] = ''
            j_data['ir_files'] = j_data['date_time'].apply(lambda x: match_sat_data_to_datetime(ir_vis_glm_json, pd.Timestamp(x), sector)).astype(str) #Match satellite data in time to radar data
            j_data.drop(columns = 'date_time', inplace = True)
            ir_vis_glm_json = ir_vis_glm_json.merge(j_data,  on = 'ir_files', how = 'outer', sort = True)
            ir_vis_glm_json.dropna(subset = ['ir_index'], inplace = True)
        ir_vis_glm_json.reset_index(drop=True, inplace=True)                                                                                   #Drop the index values between csv files. Retain only specific ones
        os.makedirs(json_root, exist_ok = True)
        ir_vis_glm_json.to_csv(join(json_root, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                                  #Write output file
        return(join(json_root, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))
    else:
        return(-1)

def mtg_create_vis_ir_numpy_arrays_from_netcdf_files2(inroot          = os.path.join('..', '..', '..', 'mtg-data', '20241029'), 
                                                     layered_root     = os.path.join('..', '..', '..', 'mtg-data', 'combined_nc_dir'),
                                                     outroot          = os.path.join('..', '..', '..', 'mtg-data', 'labelled'),
                                                     json_root        = os.path.join('..', '..', '..', 'mtg-data', 'labelled'),  
                                                     date_range       = [], 
                                                     domain_sector    = None, 
                                                     satellite        = 'MTG1',
                                                     ir_min_value     = 180, ir_max_value   = 230, 
                                                     no_write_ir      = False, no_write_vis = False, no_write_irdiff = True, 
                                                     no_write_glm     = False, no_write_sza = False, no_write_csv = False,
                                                     no_write_cirrus  = True, no_write_snowice = True, no_write_dirtyirdiff = True, no_write_trop = True, 
                                                     run_gcs          = False, use_local = False, real_time = False, del_local = False,
                                                     og_bucket_name   = 'mtg-data', comb_bucket_name = 'ir-vis-sandwhich', proc_bucket_name = 'aacp-proc-data', 
                                                     use_native_ir    = False, 
                                                     verbose          = True):
    '''
    MAIN FUNCTION. This is a script to run (by importing) programs that create numpy files for IR, VIS, solar zenith angle, and GLM data.
    Creates IR, VIS, and GLM numpy arrays for each day. Also creates corresponding csv files which yields the date, filepath/name to the 
    original MTG netCDF files, and whether the scan is a night or day time scan.
    These are the files that get fed into the model.
  
    Calling sequence:
        import mtg_create_vis_ir_numpy_arrays_from_netcdf_files2
        mtg_create_vis_ir_numpy_arrays_from_netcdf_files2.mtg_create_vis_ir_numpy_arrays_from_netcdf_files2()
    Args:
        None.
    Functions:
        fetch_convert_ir     : Clips edges of temperature data in degrees kelvin from min_value to max_value and returns normalized BT data
        fetch_convert_vis    : Clips edges of reflectance data. Normalizes the VIS data by the solar zenith angle and returns normalized VIS data
        fetch_convert_glm    : Clips edges of flash extent density data. Normalizes the VIS data by the solar zenith angle and returns normalized GLM data
        fetch_convert_irdiff : Clips edges of 6.3 - 10.2 micron BTs. Normalizes the data and returns normalized BT_difference data
        merge_csv_files      : Merges ALL csv files in the json root directory. Contains label csv file and VIS/IR/GLM numpy csv file
    Output:
         Creates numpy files that are normalized for model input.
    Keywords:
         inroot               : STRING specifying input root directory containing original IR/VIS/GLM netCDF files
                                DEFAULT = '../../../mtg-data/20241029/'
         layered_root         : STRING specifying directory containing the combined IR/VIS/GLM netCDF file (created by combine_ir_glm_vis.py)
                                DEFAULT = '../../../mtg-data/combined_nc_dir/'
         outroot              : STRING specifying directory to send the VIS, IR, and GLM numpy files as well as corresponding csv files
                                DEFAULT = '../../../mtg-data/labelled/'
         json_root            : STRING specifying root directory to read the json labeled mask csv file (created by labelme_seg_mask2.py)
                                DEFAULT = '../../../mtg-data/labelled/'
         date_range           : List containing start date and end date in YYYY-MM-DD hh:mm:ss format ("%Y-%m-%d %H:%M:%S") to run over. (ex. date_range = ['2021-04-30 19:00:00', '2021-05-01 04:00:00'])
                                DEFAULT = [] -> follow start_index keyword or do all files in VIS/IR directories
         domain_sector        : STRING specifying the satellite domain sector to use to create maps (ex. 'Full', 'CONUS'). DEFAULT = None -> use meso_sector               
         ir_min_value         : Minimum value used to clip the IR brightness temperatures. All values below min are set to min. DEFAULT = 180 K
         ir_max_value         : Maximum value used to clip the IR brightness temperatures. All values above max are set to max. DEFAULT = 230 K
         no_write_ir          : IF keyword set (True), do not write the IR numpy data arrays. DEFAULT = False
         no_write_vis         : IF keyword set (True), do not write the VIS numpy data arrays. DEFAULT = False
         no_write_irdiff      : IF keyword set (True), do not write the IR differnece (11 micron - 6.3 micron numpy file). DEFAULT = True.
         no_write_glm         : IF keyword set (True), do not write the GLM numpy data arrays. DEFAULT = False
         no_write_csv         : IF keyword set (True), do not write the csv numpy data arrays. no_write_ir cannot be set if this is False.
         no_write_sza         : IF keyword set (True), do not write the Solar zenith angle numpy data arrays (degrees). DEFAULT = False.
         no_write_cirrus      : IF keyword set (True), do not write the cirrus channel numpy data arrays. DEFAULT = True.
         no_write_snowice     : IF keyword set (True), do not write the Snow/Ice channel numpy data arrays. DEFAULT = True.
         no_write_dirtyirdiff : IF keyword set (True), do not write the IR BT difference (12-10 microns) numpy data arrays. DEFAULT = True.
         no_write_trop        : IF keyword set (True), do not write the IR BT -tropopause temperature difference numpy data arrays. DEFAULT = True.
         run_gcs              : IF keyword set (True), read and write everything directly from the google cloud platform.
                                DEFAULT = False
         use_local            : IF keyword set (True), read locally stored files.                  
                                DEFAULT = False (read from GCP.)
         real_time            : IF keyword set (True), run the code in real time by only grabbing the most recent file to image and write. 
                                Files are also output to a real time directory.
                                DEFAULT = False
         del_local            : IF keyword set (True), delete local copies of files after writing them to google cloud. Only does anything if run_gcs = True.
                                DEFAULT = False
         og_bucket_name       : Google cloud storage bucket to read raw MTG L1c files.
                                DEFAULT = 'mtg-data'
         comb_bucket_name     : Google cloud storage bucket to read combined channel netCDF files.
                                DEFAULT = 'ir-vis-sandwhich'
         proc_bucket_name     : Google cloud storage bucket to read raw IR/GLM/VIS files.
                                DEFAULT = 'aacp-proc-data'
         use_native_ir        : IF keyword set (True), write files for native IR satellite resolution.
                                DEFAULT = False -> use satellite VIS resolution
         verbose              : IF keyword set (True), print verbose informational messages to terminal.
    Output:    
        Writes IR, VIS, and GLM numpy arrays for each satellite scan
    Author and history:
        John W. Cooney           2025-01-03. (Adapted from create_vis_ir_numpy_arrays_from_netcdf_files2 but specific to MTG)
    '''
    
    inroot       = os.path.realpath(inroot)                                                                                                    #Create link to real path so compatible with Mac
    #Try to extract date in different formats
    try:
        #Attempt to extract as "yyyymmdd"
        date = os.path.basename(inroot)
        datetime.strptime(date, "%Y%m%d")                                                                                                      #Check if date string from input root is in valid format
    except ValueError:
        try:
            #If that fails, try extracting from "/yyyy/mm/dd/"
            parts = os.path.normpath(inroot).split(os.sep)
            date = f"{parts[-3]}{parts[-2]}{parts[-1]}"                                                                                        #Combine into "yyyymmdd"
            datetime.strptime(date, "%Y%m%d")                                                                                                  #Validate format
        except (ValueError, IndexError):
            print("Error: Could not parse date from inroot.")
            print(inroot)
            date = None
            exit()
    
    layered_root = os.path.realpath(layered_root)                                                                                              #Create link to real path so compatible with Mac
    layered_root = os.path.join(layered_root, date)
    json_root    = os.path.realpath(json_root)                                                                                                 #Create link to real path so compatible with Mac
    
    outroot    = os.path.realpath(outroot)                                                                                                     #Create link to real path so compatible with Mac
#     ir_dir     = join(inroot, 'ir')
#     vis_dir    = join(inroot, 'vis')
#     irdiff_dir = join(inroot, 'ir_diff')
#     os.makedirs(ir_dir,  exist_ok = True)
    no_write_sza = True
    keep0 = 'first'
    if no_write_vis == False:
#         os.makedirs(vis_dir, exist_ok = True)
        keep0 = 'last'
        no_write_sza = False
#     if no_write_irdiff == False:
#         os.makedirs(irdiff_dir, exist_ok = True)
#     if no_write_cirrus == False:
#         cirrus_dir = join(inroot, 'cirrus')
#         os.makedirs(cirrus_dir, exist_ok = True)
#     if no_write_snowice == False:
#         snowice_dir = join(inroot, 'snowice')
#         os.makedirs(snowice_dir, exist_ok = True)
#     if no_write_dirtyirdiff == False:
#         dirtyir_dir = join(inroot, 'dirtyirdiff')
#         os.makedirs(dirtyir_dir, exist_ok = True)
#     if no_write_trop == False:
#         trop_dir = join(inroot, 'tropdiff')
#         os.makedirs(trop_dir, exist_ok = True)

    sector = domain_sector.upper()                                                                                                             #Set domain sector string. Used for output directory and which input files to use
    
    if use_local == False:
        raise Exception("Sorry, MTG is currently not setup to be handled writing and reading from the Google Cloud Platform. Please only use local data for now.")
        ir_files   = list_gcs(og_bucket_name,   os.path.join(date, 'ir'),  ['-Rad' + sector])                                                  #Find IR data files
        if no_write_vis == False:
            vis_files  = list_gcs(og_bucket_name,   os.path.join(date, 'vis'), ['-Rad' + sector])                                              #Find VIS data files
        comb_files = list_gcs(comb_bucket_name, os.path.join('combined_nc_dir', date), [sector + '_COMBINED_'], delimiter = '*/')              #Find IR/VIS/GLM combined netCDF data files
        if real_time == True:                                                                                                                  #Only retain latest filename to download if running in real time
            ir_files   = [ir_files[-1]]
            if no_write_vis == False:
                vis_files  = [vis_files[-1]]
            comb_files = [comb_files[-1]]
        else:
            if no_write_vis == False:
                vis_files2  = sorted(glob.glob(os.path.join(vis_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))                     #Extract names of all of the GOES visible data files already in local storage
                if len(vis_files2) > 0:                                                                                                        #Only download files that are not already available on the local disk 
                    fv  = sorted([os.path.basename(g) for g in vis_files])
                    fv2 = sorted([os.path.basename(g) for g in vis_files2])
                    vis_files0 = [vis_files[ff] for ff in range(len(fv)) if fv[ff] not in fv2]                                                 #Find vis_files not already stored on local disk
                    vis_files  = vis_files0
            ir_files2   = sorted(glob.glob(os.path.join(ir_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))                          #Extract names of all of the GOES IR data files already in local storage
            comb_files2 = sorted(glob.glob(os.path.join(layered_dir, '*' + '_' + sector + '_COMBINED_' + '*.nc'), recursive = True))           #Find combined netCDF files not already stored on local disk
            if len(ir_files2) > 0:                                                                                                             #Only download files that are not already available on the local disk 
                fi  = sorted([os.path.basename(g) for g in ir_files])
                fi2 = sorted([os.path.basename(g) for g in ir_files2])
                ir_files0  = [ir_files[ff] for ff in range(len(fi)) if fi[ff] not in fi2]                                                      #Find vis_files not already stored on local disk
                ir_files   = ir_files0
            if len(comb_files2) > 0:                                                                                                           #Only download files that are not already available on the local disk 
                fc  = sorted([os.path.basename(g) for g in comb_files])
                fc2 = sorted([os.path.basename(g) for g in comb_files2])
                comb_files0 = [comb_files[ff] for ff in range(len(fc)) if fc[ff] not in fc2]                                                   #Find combined netCDF files not already stored on local disk
                comb_files  = comb_files0
        
        os.makedirs(layered_root, exist_ok = True)                                                                                             #Make layered root if is not not already exist
        for g in   ir_files: download_ncdf_gcs(og_bucket_name, g, ir_dir)                                                                      #Download IR data files from Google storage bucket
        if no_write_vis == False:
            for g in  vis_files: download_ncdf_gcs(og_bucket_name, g, vis_dir)                                                                 #Download VIS data files from Google storage bucket
        for g in comb_files: download_ncdf_gcs(comb_bucket_name, g, layered_root)                                                              #Download IR/VIS/GLM combined netCDF data files from Google storage bucket

    sec = 'FD'
    wv  = 0
    if use_native_ir:
        wv  = 1
        files = sorted(glob.glob(os.path.join(inroot, '*MTI' + satellite[-1] + '*FDHSI-' + sec + '-*-TRAIL-*.nc')), key = sort_mtg_irvis_files)  #Extract MTG data netCDF file names 
    else:
        files = sorted(glob.glob(os.path.join(inroot, '*MTI' + satellite[-1] + '*HRFI-' + sec + '-*-TRAIL-*.nc')), key = sort_mtg_irvis_files) #Extract MTG data netCDF file names 
        if not no_write_irdiff or not no_write_cirrus or not no_write_snowice or not no_write_dirtyirdiff:
            files2 = sorted(glob.glob(os.path.join(inroot, '*MTI' + satellite[-1] + '*FDHSI-' + sec + '-*-TRAIL-*.nc')), key = sort_mtg_irvis_files) #Extract MTG data netCDF file names for native IR data
            wv = 1 
    
    comb_files = sorted(glob.glob(os.path.join(layered_root, '*' + '_' + sector + '_COMBINED_*.nc'), recursive = True), key = sort_goes_comb_files) #Extract VIS/IR combined data netCDF file names    
    nfiles  = len(files)
    ncfiles = len(comb_files)
    if nfiles <= 0 or ncfiles <= 0:
        print(use_native_ir)
        print('No files found???')
        print(f"Number of TRAIL files = {nfiles}")
        print(inroot)
        print(sec)
        print(f"Number of comb_files = {ncfiles}")
        print(layered_root)
        print(sector)
        exit()
    if not no_write_irdiff or not no_write_cirrus or not no_write_snowice or not no_write_dirtyirdiff:
        nfiles2 = len(files2)
        if nfiles2 <= 0 or ncfiles <= 0:
            print('No files found???')
            print(f"Number of FDHSI TRAIL files = {nfiles}")
            print(inroot)
            print(sec)
            print(f"Number of comb_files = {ncfiles}")
            print(layered_root)
            print(sector)
            exit()
    
    if real_time == True:                                                                                                                      #Only retain latest file if running in real time
        ir_files = [files[-1]]
        if no_write_vis == False:
            vis_files = [files[-1]]
        comb_files = [comb_files[-1]]
    else:
        ir_files  = files
        vis_files = files
        if len(date_range) > 0:
            time0 = datetime.strptime(date_range[0], "%Y-%m-%d %H:%M:%S")                                                                      #Extract start date to find data within of the date range
            time1 = datetime.strptime(date_range[1], "%Y-%m-%d %H:%M:%S")                                                                      #Extract end date to find data within of the date range
            if verbose:
                print(time0)
                print(time1)
            
            #Extract subset of raw VIS netCDF files within date_range
            if not no_write_vis:
                start_index = 0
                end_index   = len(vis_files)
                for f in (range(0, len(vis_files))):
                    file_attr = re.split('_|-|,|\+', os.path.basename(vis_files[f]))                                                                   
                    date_str  = file_attr[23][8:14] 
                    if file_attr[23][0:14] >= "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(time0.year, time0.month, time0.day, time0.hour, time0.minute, time0.second) and start_index == 0:
                        if f == 0:
                            start_index = -1
                        else:    
                            start_index = f
                    if file_attr[23][0:14] >= "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(time1.year, time1.month, time1.day, time1.hour, time1.minute, time1.second) and end_index == len(vis_files):
                        if "{:02d}{:02d}{:02d}".format(time1.hour, time1.minute, time1.second) == date_str:
                            if f == (len(vis_files)-1):
                                end_index = len(vis_files)
                            else:    
                                end_index = f
                        else:  
                            if f == (len(vis_files)-1):
                                end_index = len(vis_files)
                            else:
                                end_index = f                 
                if start_index == -1:
                    start_index = 0
                vis_files = vis_files[start_index:end_index]
            
            if wv:
                if use_native_ir:
                    ir_files = files                
                else:
                    ir_files = files2
                if os.path.dirname(ir_files[0]) == os.path.dirname(vis_files[0]):
                    #Extract subset of raw IR netCDF files within date_range
                    start_index = 0
                    end_index   = len(ir_files)
                    for f in (range(0, len(ir_files))):
                        file_attr = re.split('_|-|,|\+', os.path.basename(ir_files[f]))                                                        #Split file string in order to extract date string of scan
                        date_str  = file_attr[23][8:14]                                                                                        #Split file string in order to extract date string of scan
                        if file_attr[23][0:14] >= "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(time0.year, time0.month, time0.day, time0.hour, time0.minute, time0.second) and start_index == 0:
                            if f == 0:
                                start_index = -1
                            else:    
                                start_index = f
                        if file_attr[23][0:14] >= "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(time1.year, time1.month, time1.day, time1.hour, time1.minute, time1.second) and end_index == len(ir_files):
                            if "{:02d}{:02d}{:02d}".format(time1.hour, time1.minute, time1.second) == date_str:
                                if f == (len(ir_files)-1):
                                    end_index = len(ir_files)
                                else:    
                                    end_index = f
                            else:  
                                if f == (len(ir_files)-1):
                                    end_index = len(ir_files)
                                else:
                                    end_index = f                
                    if start_index == -1:
                        start_index = 0
            ir_files = ir_files[start_index:end_index]
            
            #Extract subset of combined netCDF files within date_range
            start_index = 0
            end_index   = len(comb_files)
            for f in (range(0, len(comb_files))):
                file_attr = re.split('_s|_', os.path.basename(comb_files[f]))                                                                  #Split file string in order to extract date string of scan
                date_str  = file_attr[5][7:13]                                                                                                 #Split file string in order to extract date string of scan
                if file_attr[5][0:13] >= "{:04d}{:03d}{:02d}{:02d}{:02d}".format(time0.year, time0.timetuple().tm_yday, time0.hour, time0.minute, time0.second) and start_index == 0:
                    if f == 0:
                        start_index = -1
                    else:    
                        start_index = f
                if file_attr[5][0:13] >= "{:04d}{:03d}{:02d}{:02d}{:02d}".format(time1.year, time1.timetuple().tm_yday, time1.hour, time1.minute, time1.second) and end_index == len(comb_files):
                    if "{:02d}{:02d}{:02d}".format(time1.hour, time1.minute, time1.second) == date_str:
                        if f == (len(comb_files)-1):
                            end_index = len(comb_files)
                        else:    
                            end_index = f+1
                    else:  
                        if f == (len(comb_files)-1):
                            end_index = len(comb_files)
                        else:
                            end_index = f              
            if start_index == -1:
                start_index = 0
            comb_files = comb_files[start_index:end_index]        
    
    if no_write_vis == False:
        if len(ir_files) == 0 or len(vis_files) == 0 or len(comb_files) == 0:
            print('No VIS, IR netCDF or labeled json files found?')
            exit()
        if len(ir_files) != len(vis_files) or len(ir_files) != len(comb_files):
            print('Number of files do not match???')
            print(len(ir_files))
            print(len(vis_files))
            print(len(comb_files))
            print(ir_files[0])
            print(ir_files[-1])
            print(vis_files[0])
            print(vis_files[-1])
            print(comb_files[0])
            print(comb_files[-1])
            exit()
        df_vis  = pd.DataFrame(vis_files)                                                                                                      #Create data structure containing VIS data file names
        df_vis['date_time'] = df_vis[0].apply(lambda x: datetime.strptime(((re.split('_|-|,|\+', os.path.basename(x)))[23])[:],'%Y%m%d%H%M%S'))#Extract date of file scan and put into data structure
        df_vis.rename(columns={0:'vis_files'}, inplace = True)                                                                                 #Rename column VIS files
    else:
        if len(ir_files) == 0 or len(comb_files) == 0:
            print('No VIS, IR netCDF or labeled json files found?')
            exit()
        if len(ir_files) != len(comb_files):
            print('Number of files do not match???')
            print(len(ir_files))
            print(len(comb_files))
            print(ir_files[0])
            print(ir_files[-1])
            print(comb_files[0])
            print(comb_files[-1])
            exit()
          
    df_ir   = pd.DataFrame(ir_files)                                                                                                           #Create data structure containing IR data file names
    df_comb = pd.DataFrame(comb_files)                                                                                                         #Create data structure containing combined netCDF data file names
    df_ir['date_time']   = df_ir[0].apply( lambda x: datetime.strptime(((re.split('_|-|,|\+', os.path.basename(x)))[23])[:],'%Y%m%d%H%M%S'))   #Extract date of file scan and put into data structure
    df_comb['date_time'] = df_comb[0].apply(lambda x: datetime.strptime(((re.split('_s|_',  os.path.basename(x)))[5])[0:-1],'%Y%j%H%M%S'))     #Extract date of file scan and put into data structure
    df_ir.rename(columns={0:'ir_files'}, inplace = True)                                                                                       #Rename column IR files
    df_comb.rename(columns={0:'comb_files'}, inplace = True)                                                                                   #Rename column combined VIS/IR files
    if no_write_vis == False:
        ir_vis = df_vis.merge(df_ir, on = 'date_time', how = 'outer', sort = True)                                                             #Merge the two lists together ensuring that the largest one is kept
        ir_vis = ir_vis.merge(df_comb, on = 'date_time', how = 'outer', sort = True)                                                           #Merge the two lists together ensuring that the largest one is kept
    else:
        ir_vis = df_ir.merge(df_comb, on = 'date_time', how = 'outer', sort = True)                                                            #Merge the two lists together ensuring that the largest one is kept

    date_str0   = '1800136'                                                                                                                    #Default random date string to determine if date changed in loop   
    for f in range(ir_vis.shape[0]):    
        date_str    = datetime.strftime(ir_vis['date_time'][f], '%Y%j')                                                                        #Extract loop date string as day-number (determines if date changes)
        ir_results       = []                                                                                                                  #Initialize list to store IR data for single day
        ird_results      = []                                                                                                                  #Initialize list to store IR BT difference data for single day
        vis_results      = []                                                                                                                  #Initialize list to store VIS data for single day
        glm_results      = []                                                                                                                  #Initialize list to store GLM data for single day
        cirrus_results   = []                                                                                                                  #Initialize list to store cirrus channel data for single day
        snowice_results  = []                                                                                                                  #Initialize list to store Snow/Ice channel data for single day
        dirtyird_results = []                                                                                                                  #Initialize list to store IR BT difference (12-10 micron) data for single day
        trop_results     = []                                                                                                                  #Initialize list to store IR BT - tropopause temperature difference data for single day
        sza_results      = []                                                                                                                  #Initialize list to store solar zenith angle data for single day (°)
        if (date_str0 != date_str):    
            if f!= 0:    
                if no_write_csv == False: 
                    if verbose == True: print('Writing file containing combined netCDF file names:', join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'))
                    if no_write_vis == False:
                        df_vis2 = pd.DataFrame({'vis_files': v_files, 'vis_index':range(len(v_files))})                                        #Create data structure containing VIS data file names
                        df_vis2['date_time'] = df_vis2['vis_files'].apply(lambda x: datetime.strptime(((re.split('_|-|,|\+', os.path.basename(x)))[23])[:],'%Y%m%d%H%M%S'))  #Extract date of file scan and put into data structure
                        df_vis2.set_index('vis_index')                                                                                         #Extract visible data index and put into data structure
                    df_ir2  = pd.DataFrame({'ir_files' : i_files, 'ir_index' :range(len(i_files))})                                            #Create data structure containing IR data file names
                    df_tod2 = pd.DataFrame({'day_night': tod})                                                                                 #Create data structure containing night or day string
                    df_ir2['date_time']  = df_ir2['ir_files'].apply(lambda x: datetime.strptime(((re.split('_|-|,|\+', os.path.basename(x)))[23])[:],'%Y%m%d%H%M%S'))#Extract date of file scan and put into data structure
                    df_ir2.set_index('ir_index')                                                                                               #Extract IR data index and put into data structure
              #      if no_write_glm == False:
                    df_glm2 = pd.DataFrame({'glm_files': g_files, 'glm_index':range(len(g_files))})                                            #Create data structure containing GLM data file names
                    df_glm2['date_time'] = df_glm2['glm_files'].apply(lambda x: datetime.strptime(((re.split('_s|_', os.path.basename(x)))[5])[0:-1],'%Y%j%H%M%S'))  #Extract date of file scan and put into data structure
                    df_glm2.set_index('glm_index')                                                                                             #Extract GLM data index and put into data structure
                    df_comb = df_glm2.copy()
                    df_comb.rename(columns = {'glm_files' : 'comb_files', 'glm_index' : 'comb_index'}, inplace = True)
                    if (len(tod) != len(i_files)):
                        print('day night variable does not match number of IR files???')
                        exit()
                    df_tod2['date_time'] = df_ir2['ir_files'].apply(lambda x: datetime.strptime(((re.split('_|-|,|\+', os.path.basename(x)))[23])[:],'%Y%m%d%H%M%S'))#Extract date of file scan and put into data structure
                    if no_write_vis == False:
                        ir_vis2 = df_vis2.merge(df_ir2,  on = 'date_time', how = 'outer', sort = True)                                         #Merge the two lists together ensuring that the largest one is kept
                    else:
                        ir_vis2  = df_ir2.copy()
                    if no_write_glm == False: 
                        ir_vis2 = ir_vis2.merge(df_glm2, on = 'date_time', how = 'outer', sort = True)                                         #Merge the two lists together ensuring that the largest one is kept
                 
                    ir_vis2 = ir_vis2.merge(df_comb, on = 'date_time', how = 'outer', sort = True)                                             #Merge the two lists together ensuring that the largest one is kept
                    ir_vis2 = ir_vis2.merge(df_tod2, on = 'date_time', how = 'outer', sort = True)                                             #Merge the two lists together ensuring that the largest one is kept
                    if use_local == False:  
                        csv_exist = list_gcs(proc_bucket_name, join(pref, sector), ['vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'], delimiter = '*/')  #Check GCP to see if file exists
                        if len(csv_exist) == 1:
                            ir_vis3 = load_csv_gcs(proc_bucket_name, csv_exist[0])                                                             #Read in csv dataframe
                            ir_vis3.drop("Unnamed: 0",axis=1, inplace = True)
                            ir_vis2 = pd.concat([ir_vis3, ir_vis2], axis = 0, join = 'outer', ignore_index = True)
                    else:
                        csv_exist = glob.glob(join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'), recursive = True, delimiter = '*/')#Extract all json labeled csv file names
                        if len(csv_exist) == 1:
                            ir_vis3 = pd.read_csv(csv_exist[0])                                                                                #Read in csv dataframe
                            ir_vis3.drop("Unnamed: 0",axis=1, inplace = True)
                            ir_vis2 = pd.concat([ir_vis3, ir_vis2], axis = 0, join = 'outer', ignore_index = True)
                        else:
                            if run_gcs == True:
                                csv_exist = list_gcs(proc_bucket_name, join(pref, sector), ['vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'], delimiter = '*/')  #Check GCP to see if file exists
                                if len(csv_exist) == 1:
                                    ir_vis3 = load_csv_gcs(proc_bucket_name, csv_exist[0])                                                     #Read in csv dataframe
                                    ir_vis3.drop("Unnamed: 0",axis=1, inplace = True)
                                    ir_vis2 = pd.concat([ir_vis3, ir_vis2], axis = 0, join = 'outer', ignore_index = True)
                    ir_vis2 = ir_vis2.iloc[pd.to_datetime(ir_vis2.date_time).values.argsort()].drop_duplicates(subset = ['glm_files'], keep = keep0).reset_index().drop(columns = ['index']) #Remove duplicate date times and sort them
                    ir_vis2['ir_index']   = [idx if np.isfinite(i) else np.nan for idx, i in enumerate(ir_vis2['ir_index'])]
                    ir_vis2['comb_index'] = [idx if np.isfinite(i) else np.nan for idx, i in enumerate(ir_vis2['comb_index'])]
                    ir_vis2['glm_index']  = [idx if np.isfinite(i) else np.nan for idx, i in enumerate(ir_vis2['glm_index'])]
                    if no_write_vis == False: ir_vis2['vis_index'] = [idx if np.isfinite(i) else np.nan for idx, i in enumerate(ir_vis2['vis_index'])]
                    ir_vis2.to_csv(join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'))                              #Write IR/VIS/GLM csv file corresponding to the numpy files     
                    if run_gcs == True: write_to_gcs(proc_bucket_name, join(pref, sector), join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'), del_local = del_local)             #Write the IR/VIS/GLM with all of the labelled mask csv files for specified date to google cloud storage
                    if len(re.split('real_time', outroot)) > 1:
                        ir_vis2.to_csv(join(outdir, sector, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                     #Write IR/VIS/GLM csv file corresponding to the numpy files
                        if run_gcs == True: write_to_gcs(proc_bucket_name, join(pref, sector), join(outdir, sector, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'), del_local = del_local)    #Write the IR/VIS/GLM with all of the labelled mask csv files for specified date to google cloud storage
                    else:
                        csv_fname = merge_csv_files(join(json_root, date_str0, sector), run_gcs = run_gcs)                                     #Merge the IR/VIS/GLM with all of the labelled mask csv files for specified date
                        if run_gcs == True and csv_fname != -1: write_to_gcs(proc_bucket_name, join(pref, sector), csv_fname, del_local = del_local)   #Write the IR/VIS/GLM with all of the labelled mask csv files for specified date to google cloud storage
            
            outdir    = join(outroot, date_str)                                                                                                #Set up output directory path using file date (year + day_number)
            if run_gcs == True:
                pref  = os.path.join(os.path.basename(re.split(os.sep + 'labelled' + os.sep, outdir)[0]), 'labelled', re.split(os.sep + 'labelled' + os.sep, outdir)[1])
            os.makedirs(join(outdir, sector, 'ir'),  exist_ok = True)                                                                          #Create output directory if it does not exist
            if no_write_irdiff == False:
                os.makedirs(join(outdir, sector, 'ir_diff'), exist_ok = True)                                                                  #Create output directory if it does not exist
            if no_write_cirrus == False:
                os.makedirs(join(outdir, sector, 'cirrus'), exist_ok = True)                                                                   #Create output directory if it does not exist
            if no_write_snowice == False:
                os.makedirs(join(outdir, sector, 'snowice'), exist_ok = True)                                                                  #Create output directory if it does not exist
            if no_write_dirtyirdiff == False:
                os.makedirs(join(outdir, sector, 'dirtyirdiff'), exist_ok = True)                                                              #Create output directory if it does not exist
            if no_write_trop == False:
                os.makedirs(join(outdir, sector, 'tropdiff'), exist_ok = True)                                                                 #Create output directory if it does not exist
            if no_write_vis == False:
                os.makedirs(join(outdir, sector, 'vis'), exist_ok = True)                                                                      #Create output directory if it does not exist
            if no_write_glm == False:
                os.makedirs(join(outdir, sector, 'glm'), exist_ok = True)                                                                      #Create output directory if it does not exist
            os.makedirs(join(outdir, sector, 'sza'), exist_ok = True)                                                                          #Create output directory if it does not exist
            date_str0 = date_str                                                                                                               #Update previous loop string holder
            i_files   = []                                                                                                                     #Initialize list to store IR files for single day
            v_files   = []                                                                                                                     #Initialize list to store VIS files for single day
            g_files   = []                                                                                                                     #Initialize list to store GLM files for single day
            j_files   = []                                                                                                                     #Initialize list to store JSON for single day
            tod       = []                                                                                                                     #Initialize list to store whether or not the scene is considered 'night' or 'day'    
        
        if pd.notna(ir_vis['comb_files'][f]) == True:
            with Dataset(ir_vis['comb_files'][f]) as combined_nc_dat:                                                                          #Read combined netCDF file
                lon_shape = np.copy(np.asarray(combined_nc_dat.variables['longitude'])).shape                                                  #Copy array of longitudes into lon variable
                lat_shape = np.copy(np.asarray(combined_nc_dat.variables['latitude'])).shape                                                   #Copy array of latitudes into lat variable
    
                if pd.notna(ir_vis['ir_files'][f]) == True:
                    if (ir_vis['date_time'][f] != ir_vis['date_time'][f]):                                                                     #Add check to make sure working with same file
                        print(ir_vis['date_time'][f])
                        print(ir_vis['date_time'][f])
                        print((re.split('_|-|,|\+', os.path.basename(ir_vis['ir_files'][f]))[23]))
                        print(re.split('_s|_', os.path.basename(ir_vis['comb_files'][f]))[5])
                        print('Combined netCDF and IR netCDF dates do not match')
                        exit()
                        
                    if no_write_ir  == False: 
                        i_files.append(os.path.relpath(ir_vis['ir_files'][f]))                                                                 #Add loops IR file name to list
                        ir_results.append(fetch_convert_ir(combined_nc_dat, lon_shape, lat_shape, min_value = ir_min_value, max_value = ir_max_value))         #Add new normalized IR data result to IR list
                
                    if no_write_irdiff == False: 
                        ird_results.append(fetch_convert_irdiff(combined_nc_dat, lon_shape, lat_shape))                                        #Add new normalized IR BT difference data result to IRdiff list
                    if no_write_cirrus == False: 
                        cirrus_results.append(fetch_convert_cirrus(combined_nc_dat))                                                           #Add new normalized cirrus data result to cirrus list
                    if no_write_snowice == False: 
                        snowice_results.append(fetch_convert_snowice(combined_nc_dat))                                                         #Add new normalized snowice data result to snowice list
                    if no_write_dirtyirdiff == False: 
                        dirtyird_results.append(fetch_convert_dirtyirdiff(combined_nc_dat, lon_shape, lat_shape))                              #Add new normalized dirtyIR BT difference data result to dirtyIR list
                    if no_write_trop == False: 
                        trop_results.append(fetch_convert_trop(combined_nc_dat, lon_shape, lat_shape))                                         #Add new normalized IR BT - tropT difference data result to tropdiff list
                if no_write_vis == False: 
                    if pd.notna(ir_vis['vis_files'][f]) == True:
                        if (ir_vis['date_time'][f] != ir_vis['date_time'][f]):                                                                 #Add check to make sure working with same file
                            print('Combined netCDF and VIS netCDF dates do not match')
                            exit()
                      
                        vis, tod0, sza = fetch_convert_vis(combined_nc_dat)                                                                    #Extract new normalized VIS data result and if night or day
                        if tod0 == 'day':
                            v_files.append(os.path.relpath(ir_vis['vis_files'][f]))                                                            #Add loops VIS file name to list
                            vis_results.append(vis)                                                                                            #Add new normalized VIS data result to VIS list
                        sza_results.append(sza)                                                                                                #Add new SZA data result to SZA list
                        tod.append(tod0)                                                                                                       #Add if night or day to list
                    else:
                        tod.append(np.nan)    
                else:
                    if no_write_sza == False:
                        vis, tod0, sza = fetch_convert_vis(combined_nc_dat, no_write_vis = no_write_vis)                                       #Extract new normalized VIS data result and if night or day
                        sza_results.append(sza)                                                                                                #Add new SZA data result to SZA list
                        tod.append(tod0)                                                                                                       #Add if night or day to list
                    else:
                        tod.append('night')
                g_files.append(os.path.relpath(ir_vis['comb_files'][f]))                                                                       #Add loops GLM file name to list for combined netCDF names
                if no_write_glm == False: 
                    glm_results.append(fetch_convert_glm(combined_nc_dat, lon_shape, lat_shape))                                               #Add new normalized GLM data result to GLM list
#                     if len(ir_results) != len(glm_results):
#                         print('Number of elements in IR array does not match GLM array.')

        d_str = datetime.strftime(ir_vis['date_time'][f], '%Y%j%H%M%S')
        if no_write_ir     == False      and len(ir_results)       > 0: 
            os.makedirs(join(outdir, sector, 'ir'), exist_ok = True)
            np.save(join(outdir, sector, 'ir',          d_str + '_ir.npy'),          np.asarray(ir_results))                                   #Write IR data to numpy file
        if no_write_vis    == False      and len(vis_results)      > 0: 
            os.makedirs(join(outdir, sector, 'vis'), exist_ok = True)
            np.save(join(outdir, sector, 'vis',         d_str + '_vis.npy'),         np.asarray(vis_results))                                  #Write VIS data to numpy file
        if no_write_sza    == False      and len(sza_results)      > 0: 
            os.makedirs(join(outdir, sector, 'sza'), exist_ok = True)
            np.save(join(outdir, sector, 'sza',         d_str + '_sza.npy'),         np.asarray(sza_results))                                  #Write solar zenith angle data to numpy file
        if no_write_glm    == False      and len(glm_results)      > 0: 
            os.makedirs(join(outdir, sector, 'glm'), exist_ok = True)
            np.save(join(outdir, sector, 'glm',         d_str + '_glm.npy'),         np.asarray(glm_results))                                  #Write GLM data to numpy file
        if no_write_irdiff == False      and len(ird_results)      > 0: 
            os.makedirs(join(outdir, sector, 'ir_diff'), exist_ok = True)
            np.save(join(outdir, sector, 'ir_diff',     d_str + '_ir_diff.npy'),      np.asarray(ird_results))                                 #Write IR BT difference data to numpy file
        if no_write_cirrus == False      and len(cirrus_results)   > 0: 
            os.makedirs(join(outdir, sector, 'cirrus'), exist_ok = True)
            np.save(join(outdir, sector, 'cirrus',      d_str + '_cirrus.npy'),      np.asarray(cirrus_results))                               #Write cirrus data to numpy file
        if no_write_snowice == False     and len(snowice_results)  > 0: 
            os.makedirs(join(outdir, sector, 'snowice'), exist_ok = True)
            np.save(join(outdir, sector, 'snowice',     d_str + '_snowice.npy'),     np.asarray(snowice_results))                              #Write snowice data to numpy file
        if no_write_dirtyirdiff == False and len(dirtyird_results) > 0: 
            os.makedirs(join(outdir, sector, 'dirtyirdiff'), exist_ok = True)
            np.save(join(outdir, sector, 'dirtyirdiff', d_str + '_dirtyirdiff.npy'), np.asarray(dirtyird_results))                             #Write dirtyIR BT difference data to numpy file
        if no_write_trop == False        and len(trop_results)     > 0: 
            os.makedirs(join(outdir, sector, 'tropdiff'), exist_ok = True)
            np.save(join(outdir, sector, 'tropdiff',    d_str + '_tropdiff.npy'),    np.asarray(trop_results))                                 #Write IR BT-TropT difference data to numpy file
        if run_gcs == True and np.asarray(ir_results).shape[1] <= 2000:
            if no_write_ir  == False and len(ir_results)  > 0: 
                t = Thread(target = write_to_gcs, args = (proc_bucket_name, join(pref, sector, 'ir'), join(outdir, sector, 'ir',  d_str + '_ir.npy')), kwargs = {'del_local' : del_local})
                t.start()
            if no_write_vis == False and len(vis_results) > 0: 
                t1 = Thread(target = write_to_gcs, args = (proc_bucket_name, join(pref, sector, 'vis'), join(outdir, sector, 'vis', d_str + '_vis.npy')), kwargs = {'del_local' : del_local})
                t1.start()
            if no_write_sza == False and len(sza_results) > 0:
                t2 = Thread(target = write_to_gcs, args = (proc_bucket_name, join(pref, sector, 'sza'), join(outdir, sector, 'sza', d_str + '_sza.npy')), kwargs = {'del_local' : del_local})
                t2.start()
            if no_write_glm == False and len(glm_results) > 0: 
                t3 = Thread(target = write_to_gcs, args = (proc_bucket_name, join(pref, sector, 'glm'), join(outdir, sector, 'glm', d_str + '_glm.npy')), kwargs = {'del_local' : del_local})
                t3.start()
            if no_write_irdiff == False and len(ird_results) > 0:
                t4 = Thread(target = write_to_gcs, args = (proc_bucket_name, join(outdir, sector, 'ir_diff'), join(outdir, sector, 'ir_diff', d_str + '_ir_diff.npy')), kwargs = {'del_local' : del_local})
                t4.start()
            if no_write_cirrus == False and len(cirrus_results) > 0:
                t5 = Thread(target = write_to_gcs, args = (proc_bucket_name, join(outdir, sector, 'cirrus'), join(outdir, sector, 'cirrus', d_str + '_cirrus.npy')), kwargs = {'del_local' : del_local})
                t5.start()
            if no_write_snowice == False and len(snowice_results) > 0:
                t6 = Thread(target = write_to_gcs, args = (proc_bucket_name, join(outdir, sector, 'snowice'), join(outdir, sector, 'snowice', d_str + '_snowice.npy')), kwargs = {'del_local' : del_local})
                t6.start()
            if no_write_dirtyirdiff == False and len(dirtyird_results) > 0:
                t7 = Thread(target = write_to_gcs, args = (proc_bucket_name, join(outdir, sector, 'dirtyirdiff'), join(outdir, sector, 'dirtyirdiff', d_str + '_dirtyirdiff.npy')), kwargs = {'del_local' : del_local})
                t7.start()
            if no_write_trop == False and len(trop_results) > 0:
                t8 = Thread(target = write_to_gcs, args = (proc_bucket_name, join(outdir, sector, 'tropdiff'), join(outdir, sector, 'tropdiff', d_str + '_tropdiff.npy')), kwargs = {'del_local' : del_local})
                t8.start()
        else:
            if run_gcs == True:
                if no_write_ir  == False and len(ir_results)  > 0: 
                    write_to_gcs(proc_bucket_name, join(pref, sector, 'ir'),  join(outdir, sector, 'ir',  d_str + '_ir.npy'),  del_local = del_local)  #Write IR data to numpy file in google cloud storage bucket
                if no_write_vis == False and len(vis_results) > 0: 
                    write_to_gcs(proc_bucket_name, join(pref, sector, 'vis'), join(outdir, sector, 'vis', d_str + '_vis.npy'), del_local = del_local)  #Write VIS data to numpy filein google cloud storage bucket
                if no_write_sza == False and len(sza_results) > 0:
                    write_to_gcs(proc_bucket_name, join(pref, sector, 'sza'), join(outdir, sector, 'sza', d_str + '_sza.npy'), del_local = del_local)  #Write solar zenith angle data to numpy filein google cloud storage bucket
                if no_write_glm == False and len(glm_results) > 0: 
                    write_to_gcs(proc_bucket_name, join(pref, sector, 'glm'), join(outdir, sector, 'glm', d_str + '_glm.npy'), del_local = del_local)  #Write GLM data to numpy filein google cloud storage bucket
                if no_write_irdiff == False and len(ird_results) > 0:
                    write_to_gcs(proc_bucket_name, join(pref, sector, 'ir_diff'), join(outdir, sector, 'ir_diff', d_str + '_ir_diff.npy'), del_local = del_local)  #Write IR diff data to numpy filein google cloud storage bucket
                if no_write_cirrus == False and len(cirrus_results) > 0:
                    write_to_gcs(proc_bucket_name, join(pref, sector, 'cirrus'), join(outdir, sector, 'cirrus', d_str + '_cirrus.npy'), del_local = del_local)  #Write cirrus data to numpy filein google cloud storage bucket
                if no_write_snowice == False and len(snowice_results) > 0:
                    write_to_gcs(proc_bucket_name, join(pref, sector, 'snowice'), join(outdir, sector, 'snowice', d_str + '_snowice.npy'), del_local = del_local)  #Write snowice data to numpy filein google cloud storage bucket
                if no_write_dirtyirdiff == False and len(dirtyird_results) > 0:
                    write_to_gcs(proc_bucket_name, join(pref, sector, 'dirtyirdiff'), join(outdir, sector, 'dirtyirdiff', d_str + '_dirtyirdiff.npy'), del_local = del_local)  #Write dirtyirdiff data to numpy filein google cloud storage bucket
                if no_write_trop == False and len(trop_results) > 0:
                    write_to_gcs(proc_bucket_name, join(pref, sector, 'tropdiff'), join(outdir, sector, 'tropdiff', d_str + '_tropdiff.npy'), del_local = del_local)  #Write tropdiff data to numpy filein google cloud storage bucket
    if no_write_csv == False:
        if verbose == True: print('Writing file containing combined netCDF file names:', join(outdir, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'))
        df_ir2  = pd.DataFrame({'ir_files' : i_files,  'ir_index':range(len(i_files))})                                                        #Create data structure containing IR data file names
        df_ir2['date_time']  = df_ir2['ir_files'].apply(lambda x: datetime.strptime(((re.split('_|-|,|\+', os.path.basename(x)))[23])[:],'%Y%m%d%H%M%S'))            #Extract date of file scan and put into data structure
        df_ir2.set_index('ir_index')                                                                                                           #Extract visible data index and put into data structure
        df_tod2 = pd.DataFrame({'day_night':tod})                                                                                              #Create data structure containing IR data file names
        df_tod2['date_time'] = df_ir2['ir_files'].apply(lambda x: datetime.strptime(((re.split('_|-|,|\+', os.path.basename(x)))[23])[:],'%Y%m%d%H%M%S'))            #Extract date of file scan and put into data structure
        if no_write_vis == False:
            df_vis2 = pd.DataFrame({'vis_files': v_files, 'vis_index':range(len(v_files))})                                                    #Create data structure containing VIS data file names
            df_vis2['date_time'] = df_vis2['vis_files'].apply(lambda x: datetime.strptime(((re.split('_|-|,|\+', os.path.basename(x)))[23])[:],'%Y%m%d%H%M%S'))      #Extract date of file scan and put into data structure
            df_vis2.set_index('vis_index')                                                                                                     #Extract visible data index and put into data structure
            ir_vis2 = df_vis2.merge(df_ir2, on = 'date_time', how = 'outer', sort = True)                                                      #Merge the two lists together ensuring that the largest one is kept
        else:
            ir_vis2 = df_ir2.copy()
       
        ir_vis2 = ir_vis2.merge(df_tod2, on = 'date_time', how = 'outer', sort = True)                                                         #Merge the two lists together ensuring that the largest one is kept
  #      if no_write_glm == False:
        df_glm2 = pd.DataFrame({'glm_files': g_files, 'glm_index':range(len(g_files))})                                                        #Create data structure containing GLM data file names
        df_glm2['date_time'] = df_glm2['glm_files'].apply(lambda x: datetime.strptime(((re.split('_s|_', os.path.basename(x)))[5])[0:-1],'%Y%j%H%M%S'))      #Extract date of file scan and put into data structure
        df_glm2.set_index('glm_index')                                                                                                         #Extract GLM data index and put into data structure
        ir_vis2 = ir_vis2.merge(df_glm2, on = 'date_time', how = 'outer', sort = True)                                                         #Merge the two lists together ensuring that the largest one is kept
        df_comb = df_glm2.copy()
        df_comb.rename(columns = {'glm_files' : 'comb_files', 'glm_index' : 'comb_index'}, inplace = True)
        ir_vis2 = ir_vis2.merge(df_comb, on = 'date_time', how = 'outer', sort = True)                                                         #Merge the two lists together ensuring that the largest one is kept
        if (len(tod) != len(i_files)):
            print('day night variable does not match number of IR files???')
            exit()

        if use_local == False:  
            csv_exist = list_gcs(proc_bucket_name, join(pref, sector), ['vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'], delimiter = '*/')  #Check GCP to see if file exists
            if len(csv_exist) == 1:
                ir_vis3 = load_csv_gcs(proc_bucket_name, csv_exist[0])                                                                         #Read in csv dataframe
                ir_vis3.drop("Unnamed: 0",axis=1, inplace = True)
                ir_vis2 = pd.concat([ir_vis3, ir_vis2], axis = 0, join = 'outer', ignore_index = True)
        else:
            csv_exist = glob.glob(join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'), recursive = True)             #Extract all json labeled csv file names
            if len(csv_exist) == 1:
                ir_vis3 = pd.read_csv(csv_exist[0])                                                                                            #Read in csv dataframe
                ir_vis3.drop("Unnamed: 0",axis=1, inplace = True)
                ir_vis2 = pd.concat([ir_vis3, ir_vis2], axis = 0, join = 'outer', ignore_index = True)            
            else:
                if run_gcs == True:
                    csv_exist = list_gcs(proc_bucket_name, join(pref, sector), ['vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'], delimiter = '*/')  #Check GCP to see if file exists
                    if len(csv_exist) == 1:
                        ir_vis3 = load_csv_gcs(proc_bucket_name, csv_exist[0])                                                                 #Read in csv dataframe
                        ir_vis3.drop("Unnamed: 0",axis=1, inplace = True)
                        ir_vis2 = pd.concat([ir_vis3, ir_vis2], axis = 0, join = 'outer', ignore_index = True)
        ir_vis2 = ir_vis2.iloc[pd.to_datetime(ir_vis2.date_time).values.argsort()].drop_duplicates(subset = ['glm_files'], keep = keep0).reset_index().drop(columns = ['index'])       #Remove duplicate date times and sort them
        ir_vis2['ir_index']   = [idx if np.isfinite(i) else np.nan for idx, i in enumerate(ir_vis2['ir_index'])]
        ir_vis2['glm_index']  = [idx if np.isfinite(i) else np.nan for idx, i in enumerate(ir_vis2['glm_index'])]
        ir_vis2['comb_index'] = [idx if np.isfinite(i) else np.nan for idx, i in enumerate(ir_vis2['comb_index'])]
        if no_write_vis == False:
            ir_vis2['vis_index'] = [idx if np.isfinite(i) else np.nan for idx, i in enumerate(ir_vis2['vis_index'])]
        ir_vis2.to_csv(join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'))                                          #Write IR/VIS/GLM csv file corresponding to the numpy files  
        if run_gcs == True: write_to_gcs(proc_bucket_name, join(pref, sector), join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'), del_local = del_local)             #Write the IR/VIS/GLM with all of the labelled mask csv files for specified date to google cloud storage
        if len(re.split('real_time', outroot)) > 1:
            ir_vis2.to_csv(join(outdir, sector, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                                 #Write IR/VIS/GLM csv file corresponding to the numpy files
            if run_gcs == True: write_to_gcs(proc_bucket_name, join(pref, sector), join(outdir, sector, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'), del_local = del_local)    #Write the IR/VIS/GLM with all of the labelled mask csv files for specified date to google cloud storage
        else:
            csv_fname = merge_csv_files(join(json_root, date_str0, sector), run_gcs = run_gcs)                                                 #Merge the IR/VIS/GLM with all of the labelled mask csv files for specified date
            if run_gcs == True and csv_fname != -1: write_to_gcs(proc_bucket_name, join(pref, sector), csv_fname, del_local = del_local)       #Write the IR/VIS/GLM with all of the labelled mask csv files for specified date to google cloud storage

    return(join(outdir, sector, 'ir', 'ir.npy'), np.asarray(ir_results).shape)

def main():
    create_vis_ir_numpy_arrays_from_netcdf_files2()
    
if __name__ == '__main__':
    main()
