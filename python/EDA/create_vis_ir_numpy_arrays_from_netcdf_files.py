#+
# Name:
#     create_vis_ir_numpy_arrays_from_netcdf_files.py
# Purpose:
#     MAIN FUNCTION. This is a script to run (by importing) programs that create numpy files for IR, VIS, solar zenith angle, and GLM data.
#     Creates IR, VIS, and GLM numpy arrays for each day. Also creates corresponding csv files which yields the date, filepath/name to the 
#     original GOES netCDF files, and whether the scan is a night or day time scan.
#     These are the files that get fed into the model.
# Calling sequence:
#     import create_vis_ir_numpy_arrays_from_netcdf_files
#     create_vis_ir_numpy_arrays_from_netcdf_files.create_vis_ir_numpy_arrays_from_netcdf_files()
# Input:
#     None.
# Functions:
#     fetch_convert_ir          : Clips edges of temperature data in degrees kelvin from min_value to max_value and returns normalized BT data
#     fetch_convert_vis         : Clips edges of reflectance data. Returns normalized VIS data
#     fetch_convert_glm         : Clips edges of flash extent density data. Normalizes the VIS data by the solar zenith angle and returns normalized GLM data
#     fetch_convert_irdiff      : Clips edges of 6.3 - 10.2 micron BTs. Normalizes the data and returns normalized BT_difference data
#     fetch_convert_dirtyirdiff : Clips edges of 12.3 - 10.2 micron BTs. Normalizes the data and returns normalized BT_difference data
#     fetch_convert_cirrus      : Clips edges of cirrus channel reflectances. Normalizes the data and returns normalized cirrus data
#     fetch_convert_snowice     : Clips edges of Snow/Ice channel reflectances. Normalizes the data and returns normalized snowice data
#     merge_csv_files           : Merges ALL csv files in the json root directory. Contains label csv file and VIS/IR/GLM numpy csv file
# Output:
#     Creates netCDF file for GLM data gridded on GOES VIS grid, combined VIS, IR, and GLM netCDF file and
#     image file that gets put into labelme software to identify overshoot plumes.
# Keywords:
#      inroot           : STRING specifying input root directory containing original IR/VIS/GLM netCDF files
#                         DEFAULT = '../../../goes-data/20190517-18/'
#      layered_root     : STRING specifying directory containing the combined IR/VIS/GLM netCDF file (created by combine_ir_glm_vis.py)
#                         DEFAULT = '../../../goes-data/combined_nc_dir/'
#      outroot          : STRING specifying directory to send the VIS, IR, and GLM numpy files as well as corresponding csv files
#                         DEFAULT = '../../../goes-data/labelled/'
#      json_root        : STRING specifying root directory to read the json labeled mask csv file (created by labelme_seg_mask2.py)
#                         DEFAULT = '../../../goes-data/labelled/'
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
#                         DEFAULT = 'goes-data'
#      comb_bucket_name : Google cloud storage bucket to read raw IR/GLM/VIS files.
#                         DEFAULT = 'ir-vis-sandwhich'
#      proc_bucket_name : Google cloud storage bucket to read raw IR/GLM/VIS files.
#                         DEFAULT = 'aacp-proc-data'
#      use_native_ir    : IF keyword set (True), write files for native IR satellite resolution.
#                         DEFAULT = False -> use satellite VIS resolution
#      verbose          : IF keyword set (True), print verbose informational messages to terminal.
# Author and history:
#     John W. Cooney           2020-11-17.
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
import sys 
#sys.path.insert(1, '../')
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.getcwd()))
from new_model.gcs_processing import write_to_gcs, download_ncdf_gcs, list_gcs, load_csv_gcs
from gridrad.rdr_sat_utils_jwc import match_sat_data_to_datetime
from glm_gridder.run_create_image_from_three_modalities import sort_goes_irvis_files, sort_goes_comb_files
#import matplotlib.pyplot as plt

def fetch_convert_ir(combined_nc_file, min_value = 180.0, max_value = 230.0):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Clips edges of temperature data in degrees kelvin from min_value to max_value". 
    Previously was 190.0 K and 233.15 K. Values closer to min_value are masked closer to 1.
  
    Args:
      combined_nc_file: IR/VIS/GLM netCDF file path and name to read IR data from
    Keywords:
      min_value: Minimum temperature value to cut off for input BT data. All BT < min_value = min_value. DEFAULT = 180.0  
      max_value: Maximum temperature value to cut off for input BT data. All BT > max_value = max_value. DEFAULT = 230.0   
    Returns:    
      ir_dat: 2000 x 2000 IR data numpy array after IR range conversion. Array is resized using cv2.resize with cv2.INTER_NEAREST interpolation
    '''
    ir = Dataset(combined_nc_file)                                                                                                             #Read IR netCDF file
#    ir_img = upscale_img_to_fit(ir_raw_img, vis_img)                                                                                          #Resize IR image to fit Visible data image  
    ir_dat = np.copy(np.asarray(ir.variables['ir_brightness_temperature'][:], dtype = np.float32))[0, :, :]                                    #Copy IR data into a numpy array                        
    lon    = np.copy(np.asarray(ir.variables['longitude']))                                                                                    #Copy array of longitudes into lon variable
    lat    = np.copy(np.asarray(ir.variables['latitude']))                                                                                     #Copy array of latitudes into lat variable
    ir.close()                                                                                                                                 #Close IR data file
    if ir_dat.shape[0] != lon.shape[0] or ir_dat.shape[1] != lat.shape[1]:
        ir_dat = cv2.resize(ir_dat, (lon.shape[1], lat.shape[0]), interpolation=cv2.INTER_NEAREST)                                             #Upscale the ir data array to VIS resolution
#       ir_dat = zoom(ir_dat, 4, order=2)                                                                                                      #Upscale the ir data array to VIS resolution   (OLD)                  
    na = (ir_dat < 0)
    ir_dat[ir_dat < min_value] = min_value                                                                                                     #Clip all BT below min value to min value
    ir_dat[ir_dat > max_value] = max_value                                                                                                     #Clip all BT above max value to max value
    ir_dat = np.true_divide(ir_dat - min_value, min_value - max_value)                                                                         #Normalize the IR BT data by the max and min values
    ir_dat += 1
    if (np.amax(ir_dat) > 1 or np.amin(ir_dat) < 0):                                                                                           #Check to make sure IR data is properly normalized between 0 and 1
        print('IR data is not normalized properly between 0 and 1??')
        exit()
    ir_dat[na] = -1                                                                                                                            #Set NaN values to -1 so they can later be recognized and the OT/AACP model removes any chance that they could be valid detections. Set to max weight when input into model though to avoid faulty detections along borders of no data regions
    return(ir_dat)

def fetch_convert_vis(combined_nc_file, min_value = 0.0, max_value = 1.0, no_write_vis = False):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Clips edges of reflectance data. Normalizes the VIS data by the solar zenith angle.
  
    Args:
      combined_nc_file: IR/VIS/GLM netCDF file path and name to read solar zenith angle normalized VIS data from as well as solar zenith angle
    Keywords:
      min_value    : Minimum temperature value to cut off for input BT data. All BT < min_value = min_value. DEFAULT = 0.0  
      max_value    : Maximum temperature value to cut off for input BT data. All BT > max_value = max_value. DEFAULT = 1.0   
      no_write_vis : IF keyword set (True), do not write the VIS numpy data arrays. (returns vis data array full of zeroes). DEFAULT = False.
    Returns:    
      vis_dat: 2000 x 2000 VIS data numpy array
      tod    : STRING specifying if the file is considered a night time file or day time file
      zen    : 2000 x 2000 solar zenith angle data numpy arrayy
    '''
    vis = Dataset(combined_nc_file)                                                                                                            #Read VIS netCDF file
#    ir_img = upscale_img_to_fit(ir_raw_img, vis_img)                                                                                          #Resize VIS image to fit Visible data image  
    if no_write_vis == False:
        vis_dat = np.copy(np.asarray(vis.variables['visible_reflectance'][:], dtype = np.float32))[0, :, :]                                    #Copy VIS data into a numpy array                        
        print(vis_dat.shape)
    zen     = np.copy(np.asarray(vis.variables['solar_zenith_angle'][:], dtype = np.float32))[0, :, :]                                         #Copy VIS zenith angle into variable
#    vis_dat = zoom(ir_dat, 4, order=2)    ####zoom???   OLD (features move a little compared to what was done for PNGs)                  
    if vis.variables['solar_zenith_angle'].units == 'radians': 
        zen     = math.degrees(zen)                                                                                                            #Calculate solar zenith angle in degrees

    vis.close()                                                                                                                                #Close VIS data file
    mid_zen = zen[int(zen.shape[0]/2), int(zen.shape[1]/2)]                                                                                    #Calculate solar zenith angle for mid point of domain (degrees)
    if no_write_vis == False:
        vis_dat[vis_dat < min_value] = min_value                                                                                               #Clip all reflectance below min value to min value
        vis_dat[vis_dat > max_value] = max_value                                                                                               #Clip all reflectance above max value to max value
        if (np.amax(vis_dat) > 1 or np.amin(vis_dat) < 0):                                                                                     #Check to make sure IR data is properly normalized between 0 and 1
            print('VIS data is not normalized properly between 0 and 1??')
            exit()
    else:
        vis_dat = zen*0.0
    tod = 'day' if (mid_zen < 85.0) else 'night'                                                                                               #Set tod variable to night or day depending on the zenith angle in the middle of the scene
#    tod = 'day' if (np.nanmax(zen) < 85.0) else 'night'                                                                                       #Set tod variable to night or day depending on the zenith angle in the middle of the scene

    return(vis_dat, tod, zen)

def fetch_convert_snowice(combined_nc_file, min_value = 0.0, max_value = 1.0):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Clips edges of reflectance data. Normalizes the VIS data by the solar zenith angle.
  
    Args:
      combined_nc_file: IR/VIS/GLM netCDF file path and name to read solar zenith angle normalized VIS data from as well as solar zenith angle
    Keywords:
      min_value: Minimum reflectance value to cut off for input snow/ice channel data. All reflectance < min_value = min_value. DEFAULT = 0.0  
      max_value: Maximum reflectance value to cut off for input snow/ice channel data. All reflectance > max_value = max_value. DEFAULT = 1.0   
    Returns:    
      snowice_dat: 2000 x 2000 VIS data numpy array
      tod       : STRING specifying if the file is considered a night time file or day time file
      zen       : 2000 x 2000 solar zenith angle data numpy arrayy
    '''
    snowice = Dataset(combined_nc_file)                                                                                                        #Read snowice netCDF file
    snowice_dat = np.copy(np.asarray(snowice.variables['snowice_reflectance'][:], dtype = np.float32))[0, :, :]                                #Copy snowice data into a numpy array                        
#     zen     = np.copy(np.asarray(snowice.variables['solar_zenith_angle'][:], dtype = np.float32))[0, :, :]                                     #Copy snowice zenith angle into variable
#     if snowice.variables['solar_zenith_angle'].units == 'radians': 
#         zen     = math.degrees(zen)                                                                                                            #Calculate solar zenith angle in degrees
    
#    mid_zen = zen[int(snowice_dat.shape[0]/2), int(snowice_dat.shape[1]/2)]                                                                    #Calculate solar zenith angle for mid point of domain (degrees)
    snowice.close()                                                                                                                            #Close IR data file
    snowice_dat[snowice_dat < min_value] = min_value                                                                                           #Clip all reflectance below min value to min value
    snowice_dat[snowice_dat > max_value] = max_value                                                                                           #Clip all reflectance above max value to max value
    if (np.amax(snowice_dat) > 1 or np.amin(snowice_dat) < 0):                                                                                 #Check to make sure snow/ice channel data is properly normalized between 0 and 1
        print('snowice data is not normalized properly between 0 and 1??')
        exit()
#    tod = 'day' if (mid_zen < 85.0) else 'night'                                                                                               #Set tod variable to night or day depending on the zenith angle in the middle of the scene
#    tod = 'day' if (np.nanmax(zen) < 85.0) else 'night'                                                                                       #Set tod variable to night or day depending on the zenith angle in the middle of the scene

    return(snowice_dat)#, tod, zen)

def fetch_convert_cirrus(combined_nc_file, min_value = 0.0, max_value = 1.0):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Clips edges of reflectance data. Normalizes the VIS data by the solar zenith angle.
  
    Args:
      combined_nc_file: IR/VIS/GLM netCDF file path and name to read solar zenith angle normalized VIS data from as well as solar zenith angle
    Keywords:
      min_value: Minimum reflectance value to cut off for input snow/ice channel data. All reflectance < min_value = min_value. DEFAULT = 0.0  
      max_value: Maximum reflectance value to cut off for input snow/ice channel data. All reflectance > max_value = max_value. DEFAULT = 1.0   
    Returns:    
      snowice_dat: 2000 x 2000 VIS data numpy array
      tod       : STRING specifying if the file is considered a night time file or day time file
      zen       : 2000 x 2000 solar zenith angle data numpy arrayy
    '''
    cirrus = Dataset(combined_nc_file)                                                                                                         #Read cirrus netCDF file
    cirrus_dat = np.copy(np.asarray(cirrus.variables['cirrus_reflectance'][:], dtype = np.float32))[0, :, :]                                   #Copy cirrus data into a numpy array                        
#    zen     = np.copy(np.asarray(cirrus.variables['solar_zenith_angle'][:], dtype = np.float32))[0, :, :]                                      #Copy cirrus zenith angle into variable
#    if cirrus.variables['solar_zenith_angle'].units == 'radians': 
#        zen     = math.degrees(zen)                                                                                                            #Calculate solar zenith angle in degrees
    
#    mid_zen = zen[int(cirrus_dat.shape[0]/2), int(cirrus_dat.shape[1]/2)]                                                                      #Calculate solar zenith angle for mid point of domain (degrees)
    cirrus.close()                                                                                                                             #Close IR data file
    cirrus_dat[cirrus_dat < min_value] = min_value                                                                                             #Clip all reflectance below min value to min value
    cirrus_dat[cirrus_dat > max_value] = max_value                                                                                             #Clip all reflectance above max value to max value
    if (np.amax(cirrus_dat) > 1 or np.amin(cirrus_dat) < 0):                                                                                   #Check to make sure cirrus channel data is properly normalized between 0 and 1
        print('cirrus data is not normalized properly between 0 and 1??')
        exit()
#    tod = 'day' if (mid_zen < 85.0) else 'night'                                                                                               #Set tod variable to night or day depending on the zenith angle in the middle of the scene
#    tod = 'day' if (np.nanmax(zen) < 85.0) else 'night'                                                                                       #Set tod variable to night or day depending on the zenith angle in the middle of the scene

    return(cirrus_dat)#, tod, zen)

def fetch_convert_dirtyirdiff(combined_nc_file, min_value = -1.0, max_value = 2.0):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Clips edges of temperature data in degrees kelvin from min_value to max_value". 
    Values closer to min_value are masked closer to 0.
  
    Args:
      combined_nc_file: IR/VIS/GLM netCDF file path and name to read IR data from
    Keywords:
      min_value: Minimum temperature value to cut off for input BT data. All BT < min_value = min_value. DEFAULT = -1.0  
      max_value: Maximum temperature value to cut off for input BT data. All BT > max_value = max_value. DEFAULT = 2.0   
    Returns:    
      ir_dat: 2000 x 2000 IR data numpy array after IR range conversion. Array is resized using cv2.resize with cv2.INTER_NEAREST interpolation
    '''
    ir     = Dataset(combined_nc_file)                                                                                                         #Read IR netCDF file
    ir_dat = np.copy(np.asarray(ir.variables['dirtyir_brightness_temperature_diff'][:], dtype = np.float32))[0, :, :]                          #Copy IR data into a numpy array                        
    lon    = np.copy(np.asarray(ir.variables['longitude']))                                                                                    #Copy array of longitudes into lon variable
    lat    = np.copy(np.asarray(ir.variables['latitude']))                                                                                     #Copy array of latitudes into lat variable
    ir.close()                                                                                                                                 #Close IR data file
    if ir_dat.shape[0] != lon.shape[0] or ir_dat.shape[1] != lat.shape[1]:
        ir_dat = cv2.resize(ir_dat, (lon.shape[1], lat.shape[0]), interpolation=cv2.INTER_NEAREST)                                             #Upscale the ir data array to VIS resolution
    ir_dat[ir_dat < min_value] = min_value                                                                                                     #Clip all BT below min value to min value
    ir_dat[ir_dat > max_value] = max_value                                                                                                     #Clip all BT above max value to max value
    ir_dat = np.true_divide(ir_dat - min_value, max_value - min_value)                                                                         #Normalize the IR BT data by the max and min values
    if (np.amax(ir_dat) > 1 or np.amin(ir_dat) < 0):                                                                                           #Check to make sure IR data is properly normalized between 0 and 1
        print('IR data is not normalized properly between 0 and 1??')
        exit()
    return(ir_dat)

def fetch_convert_glm(combined_nc_file, min_value = 0.0, max_value = 20.0):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Normalizes GLM flash extent density data. 
  
    Args:
      combined_nc_file: IR/VIS/GLM netCDF file path and name to read IR data from
    Keywords:
      min_value: Minimum flash extent density counts value to cut off for input GLM data. All counts < min_value = min_value. DEFAULT = 0.0  
      max_value: Maximum flash extent density counts value to cut off for input GLM data. All counts > max_value = max_value DEFAULT = 20.0   
    Returns:    
      glm_dat: 2000 x 2000 GLM data numpy array after GLM range conversion. Array is resized using cv2.resize with cv2.INTER_NEAREST interpolation
    '''
    glm     = Dataset(combined_nc_file)                                                                                                        #Read IR/VIS/GLM netCDF file
    glm_dat = np.copy(np.asarray(glm['glm_flash_extent_density'][:], dtype = np.float32))[0, :, :]                                             #Copy GLM flash extent density data into a numpy array  
    print(glm_dat.shape)
    lon     = np.copy(np.asarray(glm.variables['longitude']))                                                                                  #Copy array of longitudes into lon variable
    lat     = np.copy(np.asarray(glm.variables['latitude']))                                                                                   #Copy array of latitudes into lat variable
    glm.close()
    if glm_dat.shape[0] != lon.shape[0] or glm_dat.shape[1] != lon.shape[1]:
        glm_dat = cv2.resize(glm_dat, (lon.shape[1], lat.shape[0]), interpolation=cv2.INTER_NEAREST)                                           #Upscale the GLM data array to VIS resolution
        glm_dat = ndimage.gaussian_filter(glm_dat, sigma=1.0, order=0)                                                                         #Smooth the GLM data using a Gaussian filter (higher sigma = more blurriness)
    glm_dat[glm_dat < min_value] = min_value                                                                                                   #Clip all reflectance below min value to min value
    glm_dat[glm_dat > max_value] = max_value                                                                                                   #Clip all reflectance above max value to max value
    glm_dat = np.true_divide(glm_dat - min_value, max_value - min_value)                                                                       #Normalize the GLM data by the max and min values
    if (np.amax(glm_dat) > 1 or np.amin(glm_dat) < 0):                                                                                         #Check to make sure IR data is properly normalized between 0 and 1
        print('VIS data is not normalized properly between 0 and 1??')
        exit()
    
    return(glm_dat)

def fetch_convert_irdiff(combined_nc_file, min_value = -20.0, max_value = 10.0):
    '''
    Reads the combined VIS/IR/GLM netCDF file. Clips edges of temperature data in degrees kelvin from min_value to max_value". 
    Values closer to max_value are masked closer to 1.
    Args:
      combined_nc_file: IR/VIS/GLM netCDF file path and name to read IR data from
    Keywords:
      min_value: Minimum brightness temperature difference value to cut off for input BT data. All BT_diff < min_value = min_value. DEFAULT = -20.0
      max_value: Maximum brightness temperature difference value to cut off for input BT data. All BT_diff > max_value = max_value. DEFAULT = 10.0  
    Returns:    
      ir_dat: 2000 x 2000 IR data numpy array after IR range conversion. Array is resized using cv2.resize with cv2.INTER_NEAREST interpolation
    '''
    ir = Dataset(combined_nc_file)                                                                                                             #Read IR netCDF file
#    ir_img = upscale_img_to_fit(ir_raw_img, vis_img)                                                                                          #Resize IR image to fit Visible data image  
    ir_dat = np.copy(np.asarray(ir.variables['ir_brightness_temperature_diff'][:], dtype = np.float32))[0, :, :]                               #Copy IR data into a numpy array                        
    lon    = np.copy(np.asarray(ir.variables['longitude']))                                                                                    #Copy array of longitudes into lon variable
    lat    = np.copy(np.asarray(ir.variables['latitude']))                                                                                     #Copy array of latitudes into lat variable
    ir.close()                                                                                                                                 #Close IR data file
    if ir_dat.shape[0] != lon.shape[0] or ir_dat.shape[1] != lon.shape[1]:
        ir_dat = cv2.resize(ir_dat, (lon.shape[1], lat.shape[0]), interpolation=cv2.INTER_NEAREST)                                             #Upscale the ir data array to VIS resolution
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
        prefix   = os.path.join(os.path.basename(re.split('/labelled/', json_root)[0]), 'labelled', re.split('/labelled/', json_root)[1]) + '/'
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
        json_csv = glob.glob(json_root + '/**/*.csv', recursive = True)                                                                        #Extract all json labeled csv file names
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
                    ir_vis_glm_json.reset_index(inplace = True)
                else: 
                    j_data = load_csv_gcs(proc_bucket_name, json_csv[j])
                    j_data.reset_index(inplace = True)
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
        ir_vis_glm_json.drop('index', inplace = True, axis = 1)
        os.makedirs(json_root, exist_ok = True)
#         ir_vis_glm_json.to_csv(join('/Users/jwcooney/python/code/goes-data/test/', 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                                  #Write output file
#         return(join('/Users/jwcooney/python/code/goes-data/test/', 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))
        ir_vis_glm_json.to_csv(join(json_root, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                                  #Write output file
        return(join(json_root, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))
    else:
        return(-1)
        
def create_vis_ir_numpy_arrays_from_netcdf_files(inroot           = '../../../goes-data/20190517-18/', 
                                                 layered_root     = '../../../goes-data/combined_nc_dir/',
                                                 outroot          = '../../../goes-data/labelled/',
                                                 json_root        = '../../../goes-data/labelled/',  
                                                 meso_sector      = 2, 
                                                 domain_sector    = None, 
                                                 ir_min_value     = 180, ir_max_value   = 230, 
                                                 no_write_ir      = False, no_write_vis = False, no_write_irdiff = True, 
                                                 no_write_cirrus  = True, no_write_snowice = True, no_write_dirtyirdiff = True, 
                                                 no_write_glm     = False, no_write_sza = False, no_write_csv = False,
                                                 run_gcs          = False, use_local = False, real_time = False, del_local = False,
                                                 og_bucket_name   = 'goes-data', comb_bucket_name = 'ir-vis-sandwhich', 
                                                 proc_bucket_name = 'aacp-proc-data', 
                                                 use_native_ir    = False, 
                                                 verbose          = True):
    '''
    MAIN FUNCTION. This is a script to run (by importing) programs that create numpy files for IR, VIS, solar zenith angle, and GLM data.
    Creates IR, VIS, and GLM numpy arrays for each day. Also creates corresponding csv files which yields the date, filepath/name to the 
    original GOES netCDF files, and whether the scan is a night or day time scan.
    These are the files that get fed into the model.
  
    Calling sequence:
        import create_vis_ir_numpy_arrays_from_netcdf_files
        create_vis_ir_numpy_arrays_from_netcdf_files.create_vis_ir_numpy_arrays_from_netcdf_files()
    Args:
      None.
    Keywords:
      inroot               : STRING specifying input root directory containing original IR/VIS/GLM netCDF files
                             DEFAULT = '../../../goes-data/20190517-18/'
      layered_root         : STRING specifying directory containing the combined IR/VIS/GLM netCDF file (created by combine_ir_glm_vis.py)
                             DEFAULT = '../../../goes-data/combined_nc_dir/'
      outroot              : STRING specifying directory to send the VIS, IR, and GLM numpy files as well as corresponding csv files
                             DEFAULT = '../../../goes-data/labelled/'
      json_root            : STRING specifying root directory to read the json labeled mask csv file (created by labelme_seg_mask2.py)
                             DEFAULT = '../../../goes-data/labelled/'
      meso_sector          : LONG integer specifying the mesoscale domain sector to use to create maps (= 1 or 2). DEFAULT = 2 (sector 2)                
      domain_sector        : STRING specifying the satellite domain sector to use to create maps (ex. 'Full', 'CONUS'). DEFAULT = None -> use meso_sector               
      ir_min_value         : Minimum value used to clip the IR brightness temperatures. All values below min are set to min. DEFAULT = 180 K
      ir_max_value         : Maximum value used to clip the IR brightness temperatures. All values above max are set to max. DEFAULT = 230 K
      no_write_ir          : IF keyword set (True), do not write the IR numpy data arrays. DEFAULT = False.
      no_write_vis         : IF keyword set (True), do not write the VIS numpy data arrays. DEFAULT = False.
      no_write_irdiff      : IF keyword set (True), do not write the IR BT difference numpy data arrays. DEFAULT = True.
      no_write_cirrus      : IF keyword set (True), do not write the cirrus channel numpy data arrays. DEFAULT = True.
      no_write_snowice     : IF keyword set (True), do not write the Snow/Ice channel numpy data arrays. DEFAULT = True.
      no_write_dirtyirdiff : IF keyword set (True), do not write the IR BT difference (12-10 microns) numpy data arrays. DEFAULT = True.
      no_write_glm         : IF keyword set (True), do not write the GLM numpy data arrays. DEFAULT = False.
      no_write_sza         : IF keyword set (True), do not write the Solar zenith angle numpy data arrays (degrees). DEFAULT = False.
      no_write_csv         : IF keyword set (True), do not write the csv numpy data arrays. no_write_ir cannot be set if this is False.
      run_gcs              : IF keyword set (True), read and write everything directly from the google cloud platform.
                             DEFAULT = False
      use_local            : IF keyword set (True), read locally stored files.                  
                             DEFAULT = False
      real_time            : IF keyword set (True), run the code in real time by only grabbing the most recent file to image and write. 
                             Files are also output to a real time directory.
                             DEFAULT = False
      del_local            : IF keyword set (True), delete local copies of files after writing them to google cloud. Only does anything if run_gcs = True.
                             DEFAULT = False
      og_bucket_name       : Google cloud storage bucket to read raw IR/GLM/VIS files.
                             DEFAULT = 'goes-data'
      comb_bucket_name     : Google cloud storage bucket to read raw IR/GLM/VIS files.
                             DEFAULT = 'ir-vis-sandwhich'
      proc_bucket_name     : Google cloud storage bucket to read raw IR/GLM/VIS files.
                             DEFAULT = 'aacp-proc-data'
      use_native_ir        : IF keyword set (True), write files for native IR satellite resolution.
                             DEFAULT = False -> use satellite VIS resolution
      verbose              : IF keyword set (True), print verbose informational messages to terminal.
    Output:    
      Writes merged csv file 
    '''
    
#     if no_write_csv == False and no_write_vis == True: 
#         print('Need to write VIS if want to write CSV file!!')
#         exit()

    inroot       = os.path.realpath(inroot)                                                                                                    #Create link to real path so compatible with Mac
    date         = os.path.basename(inroot)                                                                                                    #Extract file date from basename 
    layered_root = os.path.realpath(layered_root)                                                                                              #Create link to real path so compatible with Mac
    layered_root = os.path.join(layered_root, date)
    json_root    = os.path.realpath(json_root)                                                                                                 #Create link to real path so compatible with Mac
    
    if use_native_ir == True:
        if 'native_ir' not in layered_root:
            layered_root = join(layered_root, 'native_ir')
        outpat = 'ir_res_'
        combpat= 'native_ir/'
    else:
        outpat = '' 
        combpat= ''
    outroot    = os.path.realpath(outroot)                                                                                                     #Create link to real path so compatible with Mac
    ir_dir     = join(inroot, 'ir/')
    vis_dir    = join(inroot, 'vis/')
    irdiff_dir = join(inroot, 'ir_diff/')
    os.makedirs(ir_dir,  exist_ok = True)
    if no_write_vis == False:
        os.makedirs(vis_dir, exist_ok = True)
    if no_write_irdiff == False:
        os.makedirs(irdiff_dir, exist_ok = True)
    if domain_sector == None:
        sector = 'M' + str(meso_sector)                                                                                                        #Set mesoscale sector string. Used for output directory and which input files to use
    else:
        sector = str(domain_sector[0]).upper()                                                                                                 #Set domain sector string. Used for output directory and which input files to use
    
    if run_gcs == True and use_local == False:
        ir_files   = list_gcs(og_bucket_name,   date + '/ir/',  ['-Rad' + sector])
        if no_write_vis == False:
            vis_files  = list_gcs(og_bucket_name,   date + '/vis/', ['-Rad' + sector])
        comb_files = list_gcs(comb_bucket_name, 'combined_nc_dir/' + combpat + date + '/', [sector + '_COMBINED_'], delimiter = '*/')
        if real_time == True:
            ir_files   = [ir_files[-1]]
            if no_write_vis == False:
                vis_files  = [vis_files[-1]]
            comb_files = [comb_files[-1]]
        os.makedirs(layered_root, exist_ok = True)                                                                                             #Make layered root if is not not already exist
        for g in   ir_files: download_ncdf_gcs(og_bucket_name, g, ir_dir)
        if no_write_vis == False:
            for g in  vis_files: download_ncdf_gcs(og_bucket_name, g, vis_dir)
        for g in comb_files: download_ncdf_gcs(comb_bucket_name, g, layered_root)

    ir_files   = sorted(glob.glob(ir_dir       + '/**/*' + '-Rad' + sector + '*.nc', recursive = True), key = sort_goes_irvis_files)           #Extract IR data netCDF file names    
    if no_write_vis == False:
        vis_files  = sorted(glob.glob(vis_dir      + '/**/*' + '-Rad' + sector + '*.nc', recursive = True), key = sort_goes_irvis_files)       #Extract VIS data netCDF file names    
    comb_files = sorted(glob.glob(layered_root + '/*' + '_' + sector + '_COMBINED_*.nc', recursive = True), key = sort_goes_comb_files)        #Extract VIS/IR combined data netCDF file names    
    if real_time == True:                                                                                                                      #Only retain latest file if running in real time
        ir_files   = [ir_files[-1]]
        if no_write_vis == False:
            vis_files  = [vis_files[-1]]
        comb_files = [comb_files[-1]]
    if no_write_vis == False:
        if len(ir_files) == 0 or len(vis_files) == 0 or len(comb_files) == 0:
            print('No VIS, IR netCDF or labeled json files found?')
            exit()
        df_vis  = pd.DataFrame(vis_files)                                                                                                      #Create data structure containing VIS data file names
        df_vis['date_time']  = df_vis[0].apply(lambda x: datetime.strptime(((re.split('_s|_', os.path.basename(x)))[3])[0:-1],'%Y%j%H%M%S'))   #Extract date of file scan and put into data structure
        df_vis.rename(columns={0:'vis_files'}, inplace = True)                                                                                 #Rename column VIS files
    else:   
        if len(ir_files) == 0 or len(comb_files) == 0:
            print('No IR netCDF or labeled json files found?')
            exit()
    df_ir   = pd.DataFrame(ir_files)                                                                                                           #Create data structure containing IR data file names
    df_comb = pd.DataFrame(comb_files)                                                                                                         #Create data structure containing combined netCDF data file names
    df_ir['date_time']   = df_ir[0].apply( lambda x: datetime.strptime(((re.split('_s|_', os.path.basename(x)))[3])[0:-1],'%Y%j%H%M%S'))       #Extract date of file scan and put into data structure
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
        date_str  = datetime.strftime(ir_vis['date_time'][f], '%Y%j')                                                                          #Extract loop date string as day-number (determines if date changes)
        if (date_str0 != date_str):    
            if f!= 0:    
                if no_write_ir          == False: np.save(join(outdir, sector, outpat + 'ir.npy'),  np.asarray(ir_results))                    #Write IR data to numpy file
                if no_write_vis         == False: np.save(join(outdir, sector, outpat + 'vis.npy'), np.asarray(vis_results))                   #Write VIS data to numpy file
                if no_write_sza         == False: np.save(join(outdir, sector, outpat + 'sza.npy'), np.asarray(sza_results))                   #Write solar zenith angle data to numpy file
                if no_write_glm         == False: np.save(join(outdir, sector, outpat + 'glm.npy'), np.asarray(glm_results))                   #Write GLM data to numpy file
                if no_write_irdiff      == False: np.save(join(outdir, sector, outpat + 'irdiff.npy'), np.asarray(ird_results))                #Write IR BT difference data to numpy file
                if no_write_snowice     == False: np.save(join(outdir, sector, outpat + 'snowice.npy'), np.asarray(snowice_results))           #Write Snow/Ice channel data to numpy file
                if no_write_cirrus      == False: np.save(join(outdir, sector, outpat + 'cirrus.npy'), np.asarray(cirrus_results))             #Write cirrus channel data to numpy file
                if no_write_dirtyirdiff == False: np.save(join(outdir, sector, outpat + 'dirtyirdiff.npy'), np.asarray(dirtyird_results))      #Write IR BT difference (12-10 micron)_ data to numpy file
                if run_gcs == True:
                    pref  = os.path.join(os.path.basename(re.split('/labelled/', outdir)[0]), 'labelled', re.split('/labelled/', outdir)[1]) + '/'
                    if no_write_ir          == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'ir.npy'), del_local = del_local)         #Write IR data to numpy file in google cloud storage bucket
                    if no_write_vis         == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'vis.npy'), del_local = del_local)        #Write VIS data to numpy filein google cloud storage bucket
                    if no_write_sza         == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'sza.npy'), del_local = del_local)        #Write solar zenith angle data to numpy filein google cloud storage bucket
                    if no_write_glm         == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'glm.npy'), del_local = del_local)        #Write GLM data to numpy filein google cloud storage bucket
                    if no_write_irdiff      == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'irdiff.npy'), del_local = del_local)     #Write IR BT diff data to numpy filein google cloud storage bucket
                    if no_write_snowice     == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'snowice.npy'), del_local = del_local)    #Write Snow/Ice channel data to numpy file
                    if no_write_cirrus      == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'cirrus.npy'), del_local = del_local)     #Write cirrus channel data to numpy file
                    if no_write_dirtyirdiff == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'dirtyirdiff.npy'), del_local = del_local)#Write IR BT difference (12-10 micron) data to numpy file
                if no_write_csv == False: 
                    if verbose == True: print('Writing file containing combined netCDF file names:', join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'))
                    if no_write_vis == False: 
                        df_vis2 = pd.DataFrame({'vis_files': v_files, 'vis_index':range(len(v_files))})                                        #Create data structure containing VIS data file names
                        df_vis2['date_time'] = df_vis2['vis_files'].apply(lambda x: datetime.strptime(((re.split('_s|_', os.path.basename(x)))[3])[0:-1],'%Y%j%H%M%S'))  #Extract date of file scan and put into data structure
                        df_vis2.set_index('vis_index')                                                                                         #Extract visible data index and put into data structure
                    if no_write_glm == False:
                        df_glm2 = pd.DataFrame({'glm_files': g_files, 'glm_index':range(len(g_files))})                                        #Create data structure containing GLM data file names
                        df_glm2['date_time'] = df_glm2['glm_files'].apply(lambda x: datetime.strptime(((re.split('_s|_', os.path.basename(x)))[5])[0:-1],'%Y%j%H%M%S'))  #Extract date of file scan and put into data structure
                        df_glm2.set_index('glm_index')                                                                                         #Extract GLM data index and put into data structure
                   
                    df_tod2 = pd.DataFrame({'day_night':tod})                                                                                  #Create data structure containing IR data file names
                    df_ir2  = pd.DataFrame({'ir_files':  i_files, 'ir_index' :range(len(i_files))})                                            #Create data structure containing IR data file names
                    df_ir2['date_time']  = df_ir2['ir_files'].apply(lambda x: datetime.strptime(((re.split('_s|_', os.path.basename(x)))[3])[0:-1],'%Y%j%H%M%S'))#Extract date of file scan and put into data structure
                    df_ir2.set_index('ir_index')                                                                                               #Extract IR data index and put into data structure
                    
                    if (len(tod) != len(i_files)):
                        print('day night variable does not match number of IR files???')
                        exit()
                    df_tod2['date_time'] = df_ir2['ir_files'].apply(lambda x: datetime.strptime(((re.split('_s|_', os.path.basename(x)))[3])[0:-1],'%Y%j%H%M%S'))#Extract date of file scan and put into data structure
                    if no_write_vis == False: 
                        ir_vis2 = df_vis2.merge(df_ir2,  on = 'date_time', how = 'outer', sort = True)                                         #Merge the two lists together ensuring that the largest one is kept
                    else:
                        ir_vis2  = df_ir2.copy()
                    if no_write_glm == False: 
                        ir_vis2 = ir_vis2.merge(df_glm2, on = 'date_time', how = 'outer', sort = True)                                         #Merge the two lists together ensuring that the largest one is kept
                    ir_vis2 = ir_vis2.merge(df_tod2, on = 'date_time', how = 'outer', sort = True)                                             #Merge the two lists together ensuring that the largest one is kept
                    if real_time == True:
                        if run_gcs == True:  
                            csv_exist = list_gcs(proc_bucket_name, join(pref, sector) + '/', ['vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'], delimiter = '*/')  #Check GCP to see if file exists
                            if len(csv_exist) == 1:
                                ir_vis3 = load_csv_gcs(proc_bucket_name, csv_exist[0])                                                         #Read in csv dataframe
                                ir_vis2 = pd.concat([ir_vis3, ir_vis2], axis = 0, join = 'outer', ignore_index = True)
                        else:
                            csv_exist = glob.glob(join(outdir, sector) + '/vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv', recursive = True)   #Extract all json labeled csv file names
                            if len(csv_exist) == 1:
                                ir_vis3 = pd.read_csv(csv_exist[0])                                                                            #Read in csv dataframe
                                ir_vis2 = pd.concat([ir_vis3, ir_vis2], axis = 0, join = 'outer', ignore_index = True)                        
                    
                    ir_vis2.to_csv(join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'))                              #Write IR/VIS/GLM csv file corresponding to the numpy files     
                    if run_gcs == True: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'), del_local = del_local)             #Write the IR/VIS/GLM with all of the labelled mask csv files for specified date to google cloud storage
                    if len(re.split('real_time/', outroot)) > 1:
                        ir_vis2.to_csv(join(outdir, sector, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                     #Write IR/VIS/GLM csv file corresponding to the numpy files
                        if run_gcs == True: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'), del_local = del_local)    #Write the IR/VIS/GLM with all of the labelled mask csv files for specified date to google cloud storage
                    else:
                        csv_fname = merge_csv_files(join(json_root, date_str0, sector), run_gcs = run_gcs)                                     #Merge the IR/VIS/GLM with all of the labelled mask csv files for specified date
                        if run_gcs == True and csv_fname != -1: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', csv_fname, del_local = del_local)   #Write the IR/VIS/GLM with all of the labelled mask csv files for specified date to google cloud storage
    
            date_str0   = date_str                                                                                                             #Update previous loop string holder
            outdir      = join(outroot, date_str0)                                                                                             #Set up output directory path using file date (year + day_number)
            os.makedirs(join(outdir, sector), exist_ok = True)                                                                                 #Create output file path directory if does not already exist
            ir_results       = []                                                                                                              #Initialize list to store IR data for single day
            vis_results      = []                                                                                                              #Initialize list to store VIS data for single day
            glm_results      = []                                                                                                              #Initialize list to store GLM data for single day
            ird_results      = []                                                                                                              #Initialize list to store IR BT difference data for single day
            cirrus_results   = []                                                                                                              #Initialize list to store cirrus channel data for single day
            snowice_results  = []                                                                                                              #Initialize list to store Snow/Ice channel data for single day
            dirtyird_results = []                                                                                                              #Initialize list to store IR BT difference (12-10 micron) data for single day
            sza_results      = []                                                                                                              #Initialize list to store solar zenith angle data for single day (°)
            i_files          = []                                                                                                              #Initialize list to store IR files for single day
            v_files          = []                                                                                                              #Initialize list to store VIS files for single day
            g_files          = []                                                                                                              #Initialize list to store GLM files for single day
            j_files          = []                                                                                                              #Initialize list to store JSON for single day
            tod              = []                                                                                                              #Initialize list to store whether or not the scene is considered 'night' or 'day'    
            
        if pd.notna(ir_vis['comb_files'][f]) == True:
            if pd.notna(ir_vis['ir_files'][f]) == True:
                if (ir_vis['date_time'][f] != ir_vis['date_time'][f]) or ((re.split('_s|_', os.path.basename(ir_vis['ir_files'][f]))[3]) != (re.split('_s|_', os.path.basename(ir_vis['comb_files'][f]))[5])):  #Add check to make sure working with same file
                    print('Combined netCDF and IR netCDF dates do not match')
                    exit()
                    
                if no_write_ir  == False: 
                    i_files.append(os.path.relpath(ir_vis['ir_files'][f]))                                                                     #Add loops IR file name to list
                    ir_results.append(fetch_convert_ir(ir_vis['comb_files'][f], min_value = ir_min_value, max_value = ir_max_value))           #Add new normalized IR data result to IR list
            
                if no_write_irdiff == False: 
                    ird_results.append(fetch_convert_irdiff(ir_vis['comb_files'][f]))                                                          #Add new normalized IR BT difference data result to IRdiff list
                if no_write_dirtyirdiff == False: 
                    dirtyird_results.append(fetch_convert_dirtyirdiff(ir_vis['comb_files'][f]))                                                #Add new normalized dirty IR BT difference data result to dirtyirdiff list
            if no_write_vis == False: 
                if pd.notna(ir_vis['vis_files'][f]) == True:
                    if (ir_vis['date_time'][f] != ir_vis['date_time'][f]) or ((re.split('_s|_', os.path.basename(ir_vis['vis_files'][f]))[3]) != (re.split('_s|_', os.path.basename(ir_vis['comb_files'][f]))[5])):  #Add check to make sure working with same file
                        print('Combined netCDF and VIS netCDF dates do not match')
                        exit()
                  
                    vis, tod0, sza = fetch_convert_vis(ir_vis['comb_files'][f])                                                                #Extract new normalized VIS data result and if night or day
                    sza_results.append(sza)                                                                                                    #Add new SZA data result to SZA list
                    tod.append(tod0)                                                                                                           #Add if night or day to list
                    if tod0 == 'day':
                        v_files.append(os.path.relpath(ir_vis['vis_files'][f]))                                                                #Add loops VIS file name to list
                        vis_results.append(vis)                                                                                                #Add new normalized VIS data result to VIS list
                        if no_write_cirrus == False:
                            cirrus_results.append(fetch_convert_cirrus(ir_vis['comb_files'][f]))                                               #Add new normalized cirrus data result to cirrus list
                        if no_write_snowice == False:
                            snowice_results.append(fetch_convert_snowice(ir_vis['comb_files'][f]))                                             #Add new normalized snowice data result to snowice list
                    
                else:
                    tod.append(np.nan)    
            else:
                vis, tod0, sza = fetch_convert_vis(ir_vis['comb_files'][f], no_write_vis = no_write_vis)                                       #Extract new normalized VIS data result and if night or day
                sza_results.append(sza)                                                                                                        #Add new SZA data result to SZA list
                tod.append(tod0)                                                                                                               #Add if night or day to list
                if tod0 == 'day':
                    if no_write_cirrus == False:
                        cirrus_results.append(fetch_convert_cirrus(ir_vis['comb_files'][f]))                                                   #Add new normalized cirrus data result to cirrus list
                    if no_write_snowice == False:
                        snowice_results.append(fetch_convert_snowice(ir_vis['comb_files'][f]))                                                 #Add new normalized snowice data result to snowice list
            
            g_files.append(os.path.relpath(ir_vis['comb_files'][f]))                                                                           #Add loops GLM file name to list
            if no_write_glm == False: 
                glm_results.append(fetch_convert_glm(ir_vis['comb_files'][f]))                                                                 #Add new normalized GLM data result to GLM list
                if len(ir_results) != len(glm_results):
                    print('Number of elements in IR array does not match GLM array.')
    
    if no_write_ir          == False: np.save(join(outdir, sector, outpat + 'ir.npy'),  np.asarray(ir_results))                                #Write IR data to numpy file
    if no_write_vis         == False: np.save(join(outdir, sector, outpat + 'vis.npy'), np.asarray(vis_results))                               #Write VIS data to numpy file
    if no_write_sza         == False: np.save(join(outdir, sector, outpat + 'sza.npy'), np.asarray(sza_results))                               #Write solar zenith angle data to numpy file
    if no_write_glm         == False: np.save(join(outdir, sector, outpat + 'glm.npy'), np.asarray(glm_results))                               #Write GLM data to numpy file
    if no_write_irdiff      == False: np.save(join(outdir, sector, outpat + 'irdiff.npy'), np.asarray(ird_results))                            #Write IR BT difference data to numpy file
    if no_write_snowice     == False: np.save(join(outdir, sector, outpat + 'snowice.npy'), np.asarray(snowice_results))                       #Write Snow/Ice channel data to numpy file
    if no_write_cirrus      == False: np.save(join(outdir, sector, outpat + 'cirrus.npy'), np.asarray(cirrus_results))                         #Write cirrus channel data to numpy file
    if no_write_dirtyirdiff == False: np.save(join(outdir, sector, outpat + 'dirtyirdiff.npy'), np.asarray(dirtyird_results))                  #Write IR BT difference (12-10 micron)_ data to numpy file
    if run_gcs == True:
        pref  = os.path.join(os.path.basename(re.split('/labelled/', outdir)[0]), 'labelled', re.split('/labelled/', outdir)[1]) + '/'
        if no_write_ir          == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'ir.npy'), del_local = del_local)         #Write IR data to numpy file in google cloud storage bucket
        if no_write_vis         == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'vis.npy'), del_local = del_local)        #Write VIS data to numpy filein google cloud storage bucket
        if no_write_sza         == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'sza.npy'), del_local = del_local)        #Write VIS data to numpy filein google cloud storage bucket
        if no_write_glm         == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'glm.npy'), del_local = del_local)        #Write GLM data to numpy filein google cloud storage bucket
        if no_write_irdiff      == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'irdiff.npy'), del_local = del_local)     #Write IR BT diff data to numpy filein google cloud storage bucket
        if no_write_snowice     == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'snowice.npy'), del_local = del_local)    #Write Snow/Ice channel data to numpy file
        if no_write_cirrus      == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'cirrus.npy'), del_local = del_local)     #Write cirrus channel data to numpy file
        if no_write_dirtyirdiff == False: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, outpat + 'dirtyirdiff.npy'), del_local = del_local)#Write IR BT difference (12-10 micron) data to numpy file
    if no_write_csv == False:
        if verbose == True: print('Writing file containing combined netCDF file names:', join(outdir, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'))
        df_ir2  = pd.DataFrame({'ir_files': i_files, 'ir_index':range(len(i_files))})                                                          #Create data structure containing IR data file names
        df_ir2['date_time']  = df_ir2['ir_files'].apply(lambda x: datetime.strptime(((re.split('_s|_', os.path.basename(x)))[3])[0:-1],'%Y%j%H%M%S'))            #Extract date of file scan and put into data structure
        df_ir2.set_index('ir_index')                                                                                                           #Extract visible data index and put into data structure
        df_tod2 = pd.DataFrame({'day_night':tod})                                                                                              #Create data structure containing IR data file names
        df_tod2['date_time'] = df_ir2['ir_files'].apply(lambda x: datetime.strptime(((re.split('_s|_', os.path.basename(x)))[3])[0:-1],'%Y%j%H%M%S'))            #Extract date of file scan and put into data structure
        
        if no_write_vis == False:
            df_vis2 = pd.DataFrame({'vis_files': v_files, 'vis_index':range(len(v_files))})                                                    #Create data structure containing VIS data file names
            df_vis2['date_time'] = df_vis2['vis_files'].apply(lambda x: datetime.strptime(((re.split('_s|_', os.path.basename(x)))[3])[0:-1],'%Y%j%H%M%S'))      #Extract date of file scan and put into data structure
            df_vis2.set_index('vis_index')                                                                                                     #Extract visible data index and put into data structure
            ir_vis2 = df_vis2.merge(df_ir2, on = 'date_time', how = 'outer', sort = True)                                                      #Merge the two lists together ensuring that the largest one is kept
        else:
            ir_vis2 = df_ir2.copy()

        ir_vis2 = ir_vis2.merge(df_tod2, on = 'date_time', how = 'outer', sort = True)                                                         #Merge the two lists together ensuring that the largest one is kept
        if no_write_glm == False:
            df_glm2 = pd.DataFrame({'glm_files': g_files, 'glm_index':range(len(g_files))})                                                    #Create data structure containing GLM data file names
            df_glm2['date_time'] = df_glm2['glm_files'].apply(lambda x: datetime.strptime(((re.split('_s|_', os.path.basename(x)))[5])[0:-1],'%Y%j%H%M%S'))      #Extract date of file scan and put into data structure
            df_glm2.set_index('glm_index')                                                                                                     #Extract GLM data index and put into data structure
            ir_vis2 = ir_vis2.merge(df_glm2, on = 'date_time', how = 'outer', sort = True)                                                     #Merge the two lists together ensuring that the largest one is kept
        if (len(tod) != len(i_files)):
            print('day night variable does not match number of IR files???')
            exit()
        ir_vis2.to_csv(join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'))                                          #Write IR/VIS/GLM csv file corresponding to the numpy files  
        if run_gcs == True: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'), del_local = del_local)             #Write the IR/VIS/GLM with all of the labelled mask csv files for specified date to google cloud storage
        if len(re.split('real_time/', outroot)) > 1:
            ir_vis2.to_csv(join(outdir, sector, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                                 #Write IR/VIS/GLM csv file corresponding to the numpy files
            if run_gcs == True: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', join(outdir, sector, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'), del_local = del_local)    #Write the IR/VIS/GLM with all of the labelled mask csv files for specified date to google cloud storage
        else:
            csv_fname = merge_csv_files(join(json_root, date_str0, sector), run_gcs = run_gcs)                                                 #Merge the IR/VIS/GLM with all of the labelled mask csv files for specified date
            if run_gcs == True and csv_fname != -1: write_to_gcs(proc_bucket_name, join(pref, sector) + '/', csv_fname, del_local = del_local )#Write the IR/VIS/GLM with all of the labelled mask csv files for specified date to google cloud storage

    return(join(outdir, sector, 'ir.npy'))

def main():
    create_vis_ir_numpy_arrays_from_netcdf_files()
    
if __name__ == '__main__':
    main()
