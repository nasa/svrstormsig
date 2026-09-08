#+
# Name:
#     seviri_create_vis_ir_numpy_arrays_from_netcdf_files2.py
# Purpose:
#     MAIN FUNCTION. This is a script to run (by importing) programs that create numpy files for IR, VIS, 
#     solar zenith angle, IR differences, and Tropopause differences.
#     Creates numpy arrays for each day. Also creates corresponding csv files which yields the date, 
#     filepath/name to the original SEVIRI netCDF files, and whether the scan is a night or day time scan.
#     These are the files that get fed directly into the machine learning models.
# Calling sequence:
#     import seviri_create_vis_ir_numpy_arrays_from_netcdf_files2
#     seviri_create_vis_ir_numpy_arrays_from_netcdf_files2.seviri_create_vis_ir_numpy_arrays_from_netcdf_files2()
# Input:
#     None.
# Functions:
#     fetch_convert_ir           : Clips edges of temperature data in degrees kelvin from min_value to max_value and returns normalized BT data
#     fetch_convert_vis          : Clips edges of reflectance data. Normalizes the VIS data by the solar zenith angle and returns normalized VIS data
#     fetch_convert_irdiff       : Clips edges of 6.2 - 10.8 micron BTs. Normalizes the data and returns normalized BT_difference data
#     fetch_convert_dirtyirdiff  : Clips edges of 12.0 - 10.8 micron BTs. Normalizes the data and returns normalized BT_difference data
#     fetch_convert_trop         : Clips edges of IR BT - Tropopause BT. Normalizes data using cosine or linear weighting
#     compute_gradient_magnitude : Computes spatial gradient magnitude using Sobel operator
#     fetch_convert_ir_gradient  : Computes IR BT gradient magnitude normalized to [0, 1]
#     merge_csv_files            : Merges ALL csv files in the json root directory. Contains label csv file and VIS/IR/GLM numpy csv file
# Output:
#     Creates numpy files that are normalized for model input.
# Keywords:
#      inroot               : STRING specifying input root directory containing original SEVIRI EPCT netCDF files
#      layered_root         : STRING specifying directory containing the combined netCDF file (created by seviri_combine_ir_glm_vis.py)
#      outroot              : STRING specifying directory to send the numpy files as well as corresponding csv files
#      json_root            : STRING specifying root directory to read the json labeled mask csv file (created by labelme_seg_mask2.py)
#      date_range           : List containing start date and end date in YYYY-MM-DD hh:mm:ss format
#      domain_sector        : STRING specifying the satellite domain sector to use to create maps (ex. 'F'). 
#      satellite            : STRING specifying the satellite to use (e.g., 'Meteosat09'). DEFAULT = 'Meteosat09'
#      ir_min_value         : Minimum value used to clip the IR brightness temperatures. DEFAULT = 180 K
#      ir_max_value         : Maximum value used to clip the IR brightness temperatures. DEFAULT = 230 K
#      gradient_sigma       : Gaussian smoothing sigma applied before IR BT gradient calculation. DEFAULT = 1.0
#      no_write_ir          : IF keyword set (True), do not write the IR numpy data arrays. DEFAULT = False
#      no_write_vis         : IF keyword set (True), do not write the VIS numpy data arrays. DEFAULT = False
#      no_write_ir_gradient : IF keyword set (True), do not write the IR gradient numpy data arrays. DEFAULT = True
#      no_write_irlaplacian: IF keyword set (True), do not write the IR laplacian numpy data arrays. DEFAULT = True
#      no_write_irdiff      : IF keyword set (True), do not write the IR differnece (6.2 - 10.8 micron numpy file). DEFAULT = True.
#      no_write_dirtyirdiff : IF keyword set (True), do not write the IR BT difference (12-10.8 microns) numpy data arrays. DEFAULT = True.
#      no_write_trop        : IF keyword set (True), do not write the IR BT -tropopause temperature difference numpy data arrays. DEFAULT = True.
#      no_write_sza         : IF keyword set (True), do not write the Solar zenith angle numpy data arrays (degrees). DEFAULT = False.
#      no_write_csv         : IF keyword set (True), do not write the csv numpy data arrays.
#      no_write_glm         : Maintained for compatibility. Forced True.
#      no_write_cirrus      : Maintained for compatibility. Forced True.
#      no_write_snowice     : Maintained for compatibility. Forced True.
#      run_gcs              : IF keyword set (True), read and write everything directly from the google cloud platform. DEFAULT = False
#      use_local            : IF keyword set (True), read locally stored files. DEFAULT = True
#      real_time            : IF keyword set (True), run the code in real time by only grabbing the most recent file. DEFAULT = False
#      del_local            : IF keyword set (True), delete local copies of files. DEFAULT = False
#      use_native_ir        : Maintained for compatibility.
#      new_weighting        : BOOL keyword to specify whether or not to use the TROPDIFF new weighting scheme. DEFAULT = True
#      verbose              : IF keyword set (True), print verbose informational messages to terminal.
# Author and history:
#     John W. Cooney           2026-06-08. (Adapted for SEVIRI. Removed GLM/Cirrus/SnowIce processing. Simplified file matching).
#-

from netCDF4 import Dataset
import numpy as np
import glob
import math
import os
from os.path import join
import re
import cv2
import pandas as pd
from datetime import datetime, timedelta
import scipy.ndimage as ndimage
from threading import Thread
import sys 

sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.getcwd()))
from new_model.gcs_processing import write_to_gcs, download_ncdf_gcs, list_gcs, load_csv_gcs
from gridrad.rdr_sat_utils_jwc import match_sat_data_to_datetime
from glm_gridder.run_create_image_from_three_modalities import sort_goes_comb_files

def fetch_convert_ir(_combined_nc, lon_shape, lat_shape, min_value = 180.0, max_value = 230.0):
    '''
    Reads the combined netCDF file. Clips edges of temperature data in degrees kelvin from min_value to max_value". 
    Values closer to min_value are masked closer to 1.
    '''
    ir_dat = np.copy(np.asarray(_combined_nc.variables['ir_brightness_temperature'][:], dtype = np.float32))[0, :, :]

    if ir_dat.shape[0] != lon_shape[0] or ir_dat.shape[1] != lon_shape[1]:
        ir_dat = cv2.resize(ir_dat, (lon_shape[0], lon_shape[1]), interpolation=cv2.INTER_NEAREST)
        
    na  = (ir_dat < 0) | np.isnan(ir_dat)
    ir_dat[ir_dat < min_value] = min_value                                                                                                     
    ir_dat[ir_dat > max_value] = max_value                                                                                                     
    ir_dat = np.true_divide(ir_dat - min_value, min_value - max_value)                                                                         
    ir_dat += 1
    
    if (np.amax(ir_dat) > 1 or np.amin(ir_dat) < 0):                                                                                           
        print('IR data is not normalized properly between 0 and 1??')
        exit()
        
    ir_dat[na] = -1                                                                                                                            
    return(ir_dat)

def fetch_convert_trop(_combined_nc, lon_shape, lat_shape, new_weighting, min_value = -12.0, max_value = 25.0):
    '''
    Reads the combined netCDF file. Calculates IR BT - Tropopause BT.
    Values closer to min_value are masked closer to 1.
    '''
    if not new_weighting: 
      min_value = -15.0
      max_value = 20.0
      
    ir_dat = np.copy(np.asarray(_combined_nc.variables['ir_brightness_temperature'][:], dtype = np.float32))[0, :, :]                                              
    tr_dat = np.copy(np.asarray(_combined_nc.variables['tropopause_temperature'][:], dtype = np.float32))[0, :, :]                                                 
    ir_dat[ir_dat < 163] = np.nan
    if ir_dat.shape[0] != lon_shape[0] or ir_dat.shape[1] != lon_shape[1]:
        ir_dat = cv2.resize(ir_dat, (lon_shape[0], lon_shape[1]), interpolation=cv2.INTER_NEAREST)                                             
        tr_dat = cv2.resize(tr_dat, (lon_shape[0], lon_shape[1]), interpolation=cv2.INTER_NEAREST)                                             
        
    d_dat = ir_dat - tr_dat
    lat = np.copy(np.asarray(_combined_nc.variables['latitude'][:], dtype = np.float32))[:]
    lon = np.copy(np.asarray(_combined_nc.variables['longitude'][:], dtype = np.float32))[:]
    inds = np.where((lat > 20) & (lat < 30) & (lon > 70) & (lon < 80))

    na = (ir_dat < 0) | (ir_dat >= 250) | np.isnan(ir_dat)
    d_dat[d_dat < min_value] = min_value                                                                                                       
    d_dat[d_dat > max_value] = max_value                                                                                                       
    
    if new_weighting:
      d_dat = 0.5 * (1 + np.cos(np.pi * (d_dat - min_value) / (max_value - min_value)))
    else:
      d_dat = np.true_divide(d_dat - min_value, min_value - max_value)                                                                         
      d_dat += 1

    if (np.amax(d_dat) > 1 or np.amin(d_dat) < 0):                                                                                             
        print('TropDiff data is not normalized properly between 0 and 1??')
        exit()
        
    d_dat[na] = -1                                                                                                                             
    d_dat[(d_dat > max_value)] = -1                                                                                                                            

    return(d_dat)

def fetch_convert_vis(_combined_nc, min_value = 0.0, max_value = 1.0, no_write_vis = False):
    '''
    Reads the combined netCDF file. Clips edges of reflectance data. 
    VIS data was already normalized by SZA in the combine script.
    '''
    if no_write_vis == False:
        vis_dat = np.copy(np.asarray(_combined_nc.variables['visible_reflectance'][:], dtype = np.float32))[0, :, :]                                                  
    
    zen = np.copy(np.asarray(_combined_nc.variables['solar_zenith_angle'][:], dtype = np.float32))[0, :, :]                                    
    if _combined_nc.variables['solar_zenith_angle'].units == 'radians': 
        zen = math.degrees(zen)                                                                                                                
    
    # Calculate mid_zen based on percentage of dark pixels
    mid_zen = np.sum(zen > 85.0)/(zen.shape[0]*zen.shape[1])
    
    if no_write_vis == False:
        vis_dat[np.isnan(vis_dat)] = 0.0
        vis_dat[vis_dat < min_value] = min_value                                                                                               
        vis_dat[vis_dat > max_value] = max_value                                                                                               
        if min_value != 0.0 or max_value != 1.0:
            vis_dat = np.true_divide(vis_dat - min_value, max_value - min_value)                                                               
        if (np.amax(vis_dat) > 1 or np.amin(vis_dat) < 0):                                                                                     
            print('VIS data is not normalized properly between 0 and 1??')
            exit()
    else:
        vis_dat = zen*0.0
        
    tod = 'day' if (mid_zen < 0.05) else 'night'                                                                                               
    return(vis_dat, tod, zen)

def fetch_convert_dirtyirdiff(_combined_nc, lon_shape, lat_shape, min_value = -1.0, max_value = 2.0):
    '''
    Clips edges of Dirty IR difference (12.0 - 10.8) BT data. 
    '''
    ir_dat = np.copy(np.asarray(_combined_nc.variables['dirtyir_brightness_temperature_diff'][:], dtype = np.float32))[0, :, :]                                       
    if ir_dat.shape[0] != lon_shape[0] or ir_dat.shape[1] != lon_shape[1]:
        ir_dat = cv2.resize(ir_dat, (lon_shape[0], lon_shape[1]), interpolation=cv2.INTER_NEAREST)                                             
        
    ir_dat[np.isnan(ir_dat)] = max_value
    ir_dat[ir_dat < min_value] = min_value                                                                                                     
    ir_dat[ir_dat > max_value] = max_value                                                                                                     
    ir_dat = np.true_divide(ir_dat - min_value, max_value - min_value)                                                                         
    
    if (np.amax(ir_dat) > 1 or np.amin(ir_dat) < 0):                                                                                           
        print('Dirty IR data is not normalized properly between 0 and 1??')
        exit()
    return(ir_dat)

def fetch_convert_irdiff(_combined_nc, lon_shape, lat_shape, min_value = -20.0, max_value = 10.0):
    '''
    Clips edges of WV-IR difference (6.2 - 10.8) BT data. 
    '''
    ir_dat = np.copy(np.asarray(_combined_nc.variables['ir_brightness_temperature_diff'][:], dtype = np.float32))[0, :, :]                     
    if ir_dat.shape[0] != lon_shape[0] or ir_dat.shape[1] != lon_shape[1]:
        ir_dat = cv2.resize(ir_dat, (lon_shape[0], lon_shape[1]), interpolation=cv2.INTER_NEAREST)                                             
        
    ir_dat[np.isnan(ir_dat)] = max_value
    ir_dat[ir_dat < min_value] = min_value                                                                                                     
    ir_dat[ir_dat > max_value] = max_value                                                                                                     
    ir_dat = np.true_divide(ir_dat - min_value, max_value - min_value)                                                                         
    
    if (np.amax(ir_dat) > 1 or np.amin(ir_dat) < 0):                                                                                           
        print('IR Diff data is not normalized properly between 0 and 1??')
        exit()
    return(ir_dat)

def compute_gradient_magnitude(data, sigma=1.0, grad='raw'):
    """
    Compute gradient magnitude using Sobel operator.
    """
    if grad == 'raw':
        valid_mask = (data >= 0) if np.any(data < 0) else np.ones_like(data, dtype=bool)
    elif grad == 'btd_diff':
        valid_mask = (data > -100) & ~np.isnan(data)
    else:
        print(f'Not setup to handle grad: {grad}')
    
    data_clean = data.copy()
    data_clean[~valid_mask] = np.nan
    
    smoothed = ndimage.gaussian_filter(data_clean, sigma=sigma)
    grad_x = ndimage.sobel(smoothed, axis=1)
    grad_y = ndimage.sobel(smoothed, axis=0)
    
    grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
    
    if np.nanmax(grad_mag) > 0:
        grad_mag = grad_mag / np.nanmax(grad_mag)
    
    grad_mag[~valid_mask] = -1
    grad_mag[np.isnan(grad_mag)] = -1
    
    return(grad_mag.astype(np.float32))

def fetch_convert_ir_gradient(_combined_nc, lon_shape, lat_shape, min_value=160.0, max_value=245.0, sigma=1.0):
    """
    Compute IR BT gradient magnitude.
    """
    ir_raw = np.copy(np.asarray(_combined_nc.variables['ir_brightness_temperature'][:], dtype=np.float32))[0, :, :]

    if ir_raw.shape[0] != lon_shape[0] or ir_raw.shape[1] != lon_shape[1]:
        ir_raw = cv2.resize(ir_raw, (lon_shape[0], lon_shape[1]), interpolation=cv2.INTER_NEAREST)
    
    ir_raw[ir_raw > max_value] = max_value
    na = (ir_raw < 0) | np.isnan(ir_raw)
    
    grad_mag = compute_gradient_magnitude(ir_raw, sigma=sigma, grad='raw')
    grad_mag[na] = -1
    
    return(grad_mag)

def fetch_convert_ir_laplacian(_combined_nc, lon_shape, lat_shape, min_value=160.0, max_value=245.0, sigma=3.0):
    """
    Computes the Laplacian of Gaussian (LoG) on IR brightness temperatures.
    OTs (cold pits/bowls) will have strong positive values.
    
    Args:
      _combined_nc: combined netCDF fata
    Keywords:
      min_value: Minimum temperature for IR. DEFAULT = 180.0
      max_value: Maximum temperature for IR. DEFAULT = 250.0
      sigma: Gaussian sigma. Tunes the "size" of the bowl to look for. DEFAULT = 6.0
    Returns:
      log_norm: Laplacian normalized to [0, 1]. >0.5 are bowls, <0.5 are domes.
    """
    ir_raw = np.copy(np.asarray(_combined_nc.variables['ir_brightness_temperature'][:], dtype=np.float32))[0, :, :]
    
    if ir_raw.shape[0] != lon_shape[0] or ir_raw.shape[1] != lon_shape[1]:
        ir_raw = cv2.resize(ir_raw, (lon_shape[1], lon_shape[0]), interpolation=cv2.INTER_CUBIC)
            
    na = (ir_raw < 0)
    ir_raw[ir_raw < min_value] = min_value
    ir_raw[ir_raw > max_value] = max_value
    valid_mask = ~na
    
    # Fill invalid areas
    ir_clean = ir_raw.copy()
    if np.any(valid_mask):
        ir_clean[~valid_mask] = max_value
    
    # Compute Laplacian of Gaussian
    # Local minima (cold bowls) yield positive values.
    log_raw = ndimage.gaussian_laplace(ir_clean, sigma=sigma)
    
    # Normalize using percentiles to ignore extreme outliers
    if np.any(valid_mask):
        p1  = np.percentile(log_raw[valid_mask], 1)
        p99 = np.percentile(log_raw[valid_mask], 99)
        
        if p99 - p1 > 0:
            log_norm = (log_raw - p1) / (p99 - p1)
            log_norm = np.clip(log_norm, 0.0, 1.0)
        else:
            log_norm = np.zeros_like(log_raw) + 0.5
    else:
        log_norm = np.zeros_like(log_raw)
        
    log_norm[~valid_mask] = -1
    
    return(log_norm.astype(np.float32))


def merge_csv_files(json_root, run_gcs = False, proc_bucket_name = 'aacp-proc-data'):
    '''
    Merges ALL csv files in the json root directory.
    '''
    json_root = os.path.realpath(json_root)
    sector = os.path.basename(json_root)
    rm = 0
    if run_gcs == True:
        prefix   = os.path.join(os.path.basename(re.split(os.sep + 'labelled' + os.sep, json_root)[0]), 'labelled', re.split(os.sep + 'labelled' + os.sep, json_root)[1])
        json_csv = list_gcs(proc_bucket_name, prefix, ['.csv'], delimiter = '*/')
        if os.path.join(prefix, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv') in json_csv: 
            json_csv.remove(os.path.join(prefix, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                                
        fb       = [os.path.basename(x) for x in json_csv]
        fb2      = [x[-7:] for x in fb]
        if 'old.csv' in fb2: 
            json_csv = [json_csv[i] for i, j in enumerate(fb2) if j != 'old.csv']                                                              
        fb       = [os.path.basename(x) for x in json_csv]
        dir0     = [os.path.dirname(x) for x in json_csv]
        if 'zrel_ncdf_filenames_with_npy_files.csv' in fb: 
            json_csv = [json_csv[i] for i, j in enumerate(fb) if j != 'zrel_ncdf_filenames_with_npy_files.csv']                                
            rm = 1
    else:
        json_csv = glob.glob(os.path.join(json_root, '**', '*.csv'), recursive = True)                                                          
        if os.path.join(json_root, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv') in json_csv: 
            json_csv.remove(os.path.join(json_root, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                             
        fb       = [os.path.basename(x) for x in json_csv]
        fb2      = [x[-7:] for x in fb]
        if 'old.csv' in fb2: 
            json_csv = [json_csv[i] for i, j in enumerate(fb2) if j != 'old.csv']                                                              
        fb       = [os.path.basename(x) for x in json_csv]
        dir0     = [os.path.dirname(x) for x in json_csv]        
        if 'zrel_ncdf_filenames_with_npy_files.csv' in fb: 
            json_csv = [json_csv[i] for i, j in enumerate(fb) if j != 'zrel_ncdf_filenames_with_npy_files.csv']                                
            rm = 1
            
    if len(json_csv) > 0 or rm == 1:
        for j, file0 in enumerate(json_csv):
            if run_gcs == True:
                if j == 0: 
                    ir_vis_glm_json = load_csv_gcs(proc_bucket_name, json_csv[j])                                                              
                else: 
                    j_data = load_csv_gcs(proc_bucket_name, json_csv[j])
                    ir_vis_glm_json = ir_vis_glm_json.merge(j_data, on = 'date_time', how = 'outer', sort = True)                              
            else:
                if j == 0: 
                    ir_vis_glm_json = pd.read_csv(file0)                                                                                       
                else: 
                    j_data = pd.read_csv(file0)                                                                                                
                    ir_vis_glm_json = ir_vis_glm_json.merge(j_data, on = 'date_time', how = 'outer', sort = True)                              
        ir_vis_glm_json.reset_index(drop=True, inplace=True)                                                                                   
        
        if rm == 1:
            if run_gcs == True:
                j_data = load_csv_gcs(proc_bucket_name, os.path.join(dir0[fb.index('zrel_ncdf_filenames_with_npy_files.csv')], 'zrel_ncdf_filenames_with_npy_files.csv'))   
            else:
                j_data = pd.read_csv(os.path.join(dir0[fb.index('zrel_ncdf_filenames_with_npy_files.csv')], 'zrel_ncdf_filenames_with_npy_files.csv'))                      
            j_data['ir_files'] = ''
            j_data['ir_files'] = j_data['date_time'].apply(lambda x: match_sat_data_to_datetime(ir_vis_glm_json, pd.Timestamp(x), sector)).astype(str) 
            j_data.drop(columns = 'date_time', inplace = True)
            ir_vis_glm_json = ir_vis_glm_json.merge(j_data,  on = 'ir_files', how = 'outer', sort = True)
            ir_vis_glm_json.dropna(subset = ['ir_index'], inplace = True)
            
        ir_vis_glm_json.reset_index(drop=True, inplace=True)                                                                                   
        os.makedirs(json_root, exist_ok = True)
        ir_vis_glm_json.to_csv(join(json_root, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                                  
        return(join(json_root, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))
    else:
        return(-1)


def seviri_create_vis_ir_numpy_arrays_from_netcdf_files2(inroot          = os.path.join('..', '..', '..', 'seviri-data', '20260606'), 
                                                         layered_root     = os.path.join('..', '..', '..', 'seviri-data', 'combined_nc_dir'),
                                                         outroot          = os.path.join('..', '..', '..', 'seviri-data', 'labelled'),
                                                         json_root        = os.path.join('..', '..', '..', 'seviri-data', 'labelled'),  
                                                         date_range       = [], 
                                                         domain_sector    = 'F', 
                                                         satellite        = 'Meteosat09',
                                                         ir_min_value     = 180, ir_max_value   = 230, gradient_sigma = 1.0,
                                                         no_write_ir      = False, no_write_vis = False, no_write_ir_gradient = True, no_write_irlaplacian = True, no_write_irdiff = True, 
                                                         no_write_glm     = True, no_write_sza = False, no_write_csv = False,
                                                         no_write_cirrus  = True, no_write_snowice = True, no_write_dirtyirdiff = True, no_write_trop = True, 
                                                         run_gcs          = False, use_local = True, real_time = False, del_local = False,
                                                         og_bucket_name   = 'seviri-data', comb_bucket_name = 'ir-vis-sandwhich', proc_bucket_name = 'aacp-proc-data', 
                                                         use_native_ir    = False, 
                                                         new_weighting    = True,
                                                         verbose          = True):
    '''
    Name:
        seviri_create_vis_ir_numpy_arrays_from_netcdf_files2.py
    Purpose:
        MAIN FUNCTION. This is a script to run (by importing) programs that create numpy files for IR, VIS, 
        solar zenith angle, IR differences, and Tropopause differences.
        Creates numpy arrays for each day. Also creates corresponding csv files which yields the date, 
        filepath/name to the original SEVIRI netCDF files, and whether the scan is a night or day time scan.
        These are the files that get fed directly into the machine learning models.
    Calling sequence:
        import seviri_create_vis_ir_numpy_arrays_from_netcdf_files2
        seviri_create_vis_ir_numpy_arrays_from_netcdf_files2.seviri_create_vis_ir_numpy_arrays_from_netcdf_files2()
    Input:
        None.
    Functions:
        fetch_convert_ir           : Clips edges of temperature data in degrees kelvin from min_value to max_value and returns normalized BT data
        fetch_convert_vis          : Clips edges of reflectance data. Normalizes the VIS data by the solar zenith angle and returns normalized VIS data
        fetch_convert_irdiff       : Clips edges of 6.2 - 10.8 micron BTs. Normalizes the data and returns normalized BT_difference data
        fetch_convert_dirtyirdiff  : Clips edges of 12.0 - 10.8 micron BTs. Normalizes the data and returns normalized BT_difference data
        fetch_convert_trop         : Clips edges of IR BT - Tropopause BT. Normalizes data using cosine or linear weighting
        compute_gradient_magnitude : Computes spatial gradient magnitude using Sobel operator
        fetch_convert_ir_gradient  : Computes IR BT gradient magnitude normalized to [0, 1]
        fetch_convert_ir_laplacian : Computes the Laplacian of Gaussian (LoG) on IR brightness temperatures. OTs (cold pits/bowls) will have strong positive values.
        merge_csv_files            : Merges ALL csv files in the json root directory. Contains label csv file and VIS/IR/GLM numpy csv file
    Output:
        Creates numpy files that are normalized for model input.
    Keywords:
         inroot               : STRING specifying input root directory containing original SEVIRI EPCT netCDF files
         layered_root         : STRING specifying directory containing the combined netCDF file (created by seviri_combine_ir_glm_vis.py)
         outroot              : STRING specifying directory to send the numpy files as well as corresponding csv files
         json_root            : STRING specifying root directory to read the json labeled mask csv file (created by labelme_seg_mask2.py)
         date_range           : List containing start date and end date in YYYY-MM-DD hh:mm:ss format
         domain_sector        : STRING specifying the satellite domain sector to use to create maps (ex. 'F'). 
         satellite            : STRING specifying the satellite to use (e.g., 'Meteosat09'). DEFAULT = 'Meteosat09'
         ir_min_value         : Minimum value used to clip the IR brightness temperatures. DEFAULT = 180 K
         ir_max_value         : Maximum value used to clip the IR brightness temperatures. DEFAULT = 230 K
         gradient_sigma       : Gaussian smoothing sigma applied before IR BT gradient calculation. DEFAULT = 1.0
         no_write_ir          : IF keyword set (True), do not write the IR numpy data arrays. DEFAULT = False
         no_write_vis         : IF keyword set (True), do not write the VIS numpy data arrays. DEFAULT = False
         no_write_ir_gradient : IF keyword set (True), do not write the IR gradient numpy data arrays. DEFAULT = True
         no_write_irlaplacian: IF keyword set (True), do not write the IR laplacian numpy data arrays. DEFAULT = True
         no_write_irdiff      : IF keyword set (True), do not write the IR differnece (6.2 - 10.8 micron numpy file). DEFAULT = True.
         no_write_dirtyirdiff : IF keyword set (True), do not write the IR BT difference (12-10.8 microns) numpy data arrays. DEFAULT = True.
         no_write_trop        : IF keyword set (True), do not write the IR BT -tropopause temperature difference numpy data arrays. DEFAULT = True.
         no_write_sza         : IF keyword set (True), do not write the Solar zenith angle numpy data arrays (degrees). DEFAULT = False.
         no_write_csv         : IF keyword set (True), do not write the csv numpy data arrays.
         no_write_glm         : Maintained for compatibility. Forced True.
         no_write_cirrus      : Maintained for compatibility. Forced True.
         no_write_snowice     : Maintained for compatibility. Forced True.
         run_gcs              : IF keyword set (True), read and write everything directly from the google cloud platform. DEFAULT = False
         use_local            : IF keyword set (True), read locally stored files. DEFAULT = True
         real_time            : IF keyword set (True), run the code in real time by only grabbing the most recent file. DEFAULT = False
         del_local            : IF keyword set (True), delete local copies of files. DEFAULT = False
         use_native_ir        : Maintained for compatibility.
         new_weighting        : BOOL keyword to specify whether or not to use the TROPDIFF new weighting scheme. DEFAULT = True
         verbose              : IF keyword set (True), print verbose informational messages to terminal.
    Author and history:
        John W. Cooney           2026-06-08. (Adapted for SEVIRI. Removed GLM/Cirrus/SnowIce processing. Simplified file matching).
    
    '''
    
    inroot = os.path.realpath(inroot)                                                                                                    
    try:
        date = os.path.basename(inroot)
        datetime.strptime(date, "%Y%m%d")                                                                                                      
    except ValueError:
        try:
            parts = os.path.normpath(inroot).split(os.sep)
            date = f"{parts[-3]}{parts[-2]}{parts[-1]}"                                                                                        
            datetime.strptime(date, "%Y%m%d")                                                                                                  
        except (ValueError, IndexError):
            print("Error: Could not parse date from inroot.")
            print(inroot)
            date = None
            exit()
    
    layered_root = os.path.realpath(layered_root)                                                                                              
    layered_root = os.path.join(layered_root, date)
    json_root    = os.path.realpath(json_root)                                                                                                 
    outroot      = os.path.realpath(outroot)                                                                                                     

    no_write_sza = True
    keep0 = 'first'
    if not no_write_vis:
        keep0 = 'last'
        no_write_sza = False

    sector = domain_sector.upper()                                                                                                             
    
    if not use_local:
        raise Exception("GCP not supported for SEVIRI in this script version.")

    # ------------------------------------------------------------------
    # Universal Time Extraction matching the COMBINED writer outputs
    # ------------------------------------------------------------------
    def get_raw_start_time(filepath):
        """ Extract standard start time from raw EPCT .nc or Native .nat files """
        basename = os.path.basename(filepath)
        clean_name = basename.replace('T', '').replace('Z', '')
        matches = re.findall(r'(\d{14})', clean_name)
        sat_clean = satellite.lower().replace('-', '').replace('_', '')
        is_rss = False
        if 'rss' in sat_clean or 'rapid' in sat_clean:
            is_rss = True
        elif matches and sat_clean in ['meteosat09', 'meteosat9', 'msg2', 'msg02'] and (datetime.strptime(matches[0][:8], "%Y%m%d") >= datetime(2013, 4, 9) and datetime.strptime(matches[0][:8], "%Y%m%d") < datetime(2018, 3, 1)):
            is_rss = True
        elif matches and sat_clean in ['meteosat10', 'meteosat10', 'msg3', 'msg03'] and (datetime.strptime(matches[0][:8], "%Y%m%d") >= datetime(2018, 3, 1)):
            is_rss = True

        if filepath.endswith('.nc') and len(matches) >= 2:
            return datetime.strptime(matches[0], '%Y%m%d%H%M%S')
        elif filepath.endswith('.nat') and len(matches) >= 1:
            de = datetime.strptime(matches[0], '%Y%m%d%H%M%S')
            if is_rss:
                return de - timedelta(minutes=4, seconds=43)
            else:
                return de - timedelta(minutes=12, seconds=29)
        elif matches:
            return datetime.strptime(matches[0], '%Y%m%d%H%M%S')
        return datetime.min
        
    # Find raw files (Single file per time step for EPCT SEVIRI or Native .nat)
    nc_files  = glob.glob(os.path.join(inroot, 'HRSEVIRI_*.nc'))
    nat_files = glob.glob(os.path.join(inroot, '*.nat'))

    if len(nat_files) > 0:
        unique_files = {}
        for f in nc_files:
            unique_files[get_raw_start_time(f)] = f
            
        for f in nat_files:
            t_key = get_raw_start_time(f)
            if t_key not in unique_files:
                unique_files[t_key] = f
    
        raw_files = sorted([unique_files[k] for k in sorted(unique_files.keys())], key=get_raw_start_time)
    else:
        raw_files = sorted(nc_files, key=get_raw_start_time)

    comb_files = sorted(glob.glob(os.path.join(layered_root, '*' + '_' + sector + '_COMBINED_*.nc'), recursive=True), key=sort_goes_comb_files)    
    
    nfiles  = len(raw_files)
    ncfiles = len(comb_files)
    if nfiles <= 0 or ncfiles <= 0:
        print('No files found???')
        print(f"Number of RAW files = {nfiles} in {inroot}")
        print(f"Number of COMBINED files = {ncfiles} in {layered_root}")
        exit()
    
    if real_time:                                                                                                                              
        raw_files = [raw_files[-1]]
        comb_files = [comb_files[-1]]
    else:
        if len(date_range) > 0:
            time0 = datetime.strptime(date_range[0], "%Y-%m-%d %H:%M:%S")                                                                      
            time1 = datetime.strptime(date_range[1], "%Y-%m-%d %H:%M:%S")                                                                      
            
            # Filter raw files
            raw_filtered = []
            for f in raw_files:
                f_time = get_raw_start_time(f)
                if time0 <= f_time <= time1:
                    raw_filtered.append(f)
            raw_files = raw_filtered
            
            # Filter comb files
            comb_filtered = []
            for f in comb_files:
                basename = os.path.basename(f)
                parts = re.split('_s|_', basename)
                dt_str = parts[5][:13]
                doy = int(dt_str[4:7])
                f_time = datetime.strptime(f"{dt_str[0:4]} {doy} {dt_str[7:]}", "%Y %j %H%M%S")
                if time0 <= f_time <= time1:
                    comb_filtered.append(f)
            comb_files = comb_filtered
            
    if len(raw_files) == 0 or len(comb_files) == 0:
        print('No matching netCDF files found after filtering?')
        exit()
        
    df_raw  = pd.DataFrame(raw_files)                                                                                                          
    df_comb = pd.DataFrame(comb_files)                                                                                                         
    
        
    df_raw['date_time']  = df_raw[0].apply(get_raw_start_time)
    df_comb['date_time'] = df_comb[0].apply(get_comb_time)
    
    df_raw.rename(columns={0:'ir_files'}, inplace=True) # Map raw to 'ir_files' for downstream logic
    df_comb.rename(columns={0:'comb_files'}, inplace=True)
    
    ir_vis = df_raw.merge(df_comb, on='date_time', how='outer', sort=True)                                                             
    if not no_write_vis:
        ir_vis['vis_files'] = ir_vis['ir_files'] # For SEVIRI, VIS comes from the exact same raw file
        
    date_str0 = '1800136'                                                                                                                    
    for f in range(ir_vis.shape[0]):    
        
        if pd.isna(ir_vis['date_time'][f]): continue
        date_str = datetime.strftime(ir_vis['date_time'][f], '%Y%j')                                                                        
        
        ir_results       = []                                                                                                                  
        irg_results      = []                                                                                                                  
        irl_results      = []
        ird_results      = []                                                                                                                  
        vis_results      = []                                                                                                                  
        dirtyird_results = []                                                                                                                  
        trop_results     = []                                                                                                                  
        sza_results      = []                                                                                                                  
        
        if (date_str0 != date_str):    
            if f != 0:    
                if not no_write_csv: 
                    if verbose: print('Writing file containing combined netCDF file names:', join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'))
                    
                    df_ir2 = pd.DataFrame({'ir_files' : i_files, 'ir_index' :range(len(i_files))})                                            
                    df_tod2 = pd.DataFrame({'day_night': tod})                                                                                 
                    df_ir2['date_time'] = df_ir2['ir_files'].apply(get_raw_start_time)
                    df_ir2.set_index('ir_index')                                                                                               
                    
                    # SEVIRI has no GLM, but we maintain the dummy column for the CSV output
                    df_glm2 = pd.DataFrame({'glm_files': g_files, 'glm_index':range(len(g_files))})                                            
                    df_glm2['date_time'] = df_glm2['glm_files'].apply(get_comb_time)
                    df_glm2.set_index('glm_index')                                                                                             
                    
                    df_comb_csv = df_glm2.copy()
                    df_comb_csv.rename(columns = {'glm_files' : 'comb_files', 'glm_index' : 'comb_index'}, inplace = True)
                    
                    df_tod2['date_time'] = df_ir2['date_time']
                    
                    if not no_write_vis:
                        df_vis2 = pd.DataFrame({'vis_files': v_files, 'vis_index':range(len(v_files))})                                        
                        df_vis2['date_time'] = df_vis2['vis_files'].apply(get_raw_start_time)  
                        df_vis2.set_index('vis_index')                                                                                         
                        ir_vis2 = df_vis2.merge(df_ir2, on='date_time', how='outer', sort=True)                                         
                    else:
                        ir_vis2 = df_ir2.copy()
                        
                    ir_vis2 = ir_vis2.merge(df_glm2, on='date_time', how='outer', sort=True)                                         
                    ir_vis2 = ir_vis2.merge(df_comb_csv, on='date_time', how='outer', sort=True)                                             
                    ir_vis2 = ir_vis2.merge(df_tod2, on='date_time', how='outer', sort=True)                                             
                    
                    csv_exist = glob.glob(join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'), recursive=True)
                    if len(csv_exist) == 1:
                        ir_vis3 = pd.read_csv(csv_exist[0])                                                                                
                        if "Unnamed: 0" in ir_vis3.columns: ir_vis3.drop("Unnamed: 0", axis=1, inplace=True)
                        ir_vis2 = pd.concat([ir_vis3, ir_vis2], axis=0, join='outer', ignore_index=True)
                        
                    ir_vis2 = ir_vis2.iloc[pd.to_datetime(ir_vis2.date_time).values.argsort()].drop_duplicates(subset=['comb_files'], keep=keep0).reset_index(drop=True) 
                    
                    ir_vis2['ir_index']   = [idx if pd.notna(i) else np.nan for idx, i in enumerate(ir_vis2['ir_index'])]
                    ir_vis2['comb_index'] = [idx if pd.notna(i) else np.nan for idx, i in enumerate(ir_vis2['comb_index'])]
                    ir_vis2['glm_index']  = [idx if pd.notna(i) else np.nan for idx, i in enumerate(ir_vis2['glm_index'])]
                    if not no_write_vis: ir_vis2['vis_index'] = [idx if pd.notna(i) else np.nan for idx, i in enumerate(ir_vis2['vis_index'])]
                    
                    ir_vis2.to_csv(join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'))                              
                    
                    if len(re.split('real_time', outroot)) > 1:
                        ir_vis2.to_csv(join(outdir, sector, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                     
                    else:
                        csv_fname = merge_csv_files(join(json_root, date_str0, sector), run_gcs=run_gcs)                                     
            
            outdir = join(outroot, date_str)                                                                                                
            os.makedirs(join(outdir, sector, 'ir'), exist_ok=True)                                                                          
            if not no_write_irdiff: os.makedirs(join(outdir, sector, 'ir_diff'), exist_ok=True)                                                                  
            if not no_write_ir_gradient: os.makedirs(join(outdir, sector, 'ir_gradient'), exist_ok=True)                                                              
            if not no_write_irlaplacian: os.makedirs(join(outdir, sector, 'irlaplacian'), exist_ok=True)                                                              
            if not no_write_dirtyirdiff: os.makedirs(join(outdir, sector, 'dirtyirdiff'), exist_ok=True)                                                              
            if not no_write_trop: os.makedirs(join(outdir, sector, 'tropdiff'), exist_ok=True)                                                                 
            if not no_write_vis: os.makedirs(join(outdir, sector, 'vis'), exist_ok=True)                                                                      
            os.makedirs(join(outdir, sector, 'sza'), exist_ok=True)                                                                          
            
            date_str0 = date_str                                                                                                               
            i_files = []                                                                                                                     
            v_files = []                                                                                                                     
            g_files = []                                                                                                                     
            tod     = []                                                                                                                     
        
        if pd.notna(ir_vis['comb_files'][f]):
            with Dataset(ir_vis['comb_files'][f]) as combined_nc_dat:                                                                          
                lon_shape = np.copy(np.asarray(combined_nc_dat.variables['longitude'])).shape                                                  
                lat_shape = np.copy(np.asarray(combined_nc_dat.variables['latitude'])).shape                                                   
    
                if pd.notna(ir_vis['ir_files'][f]):
                    if not no_write_ir: 
                        i_files.append(os.path.relpath(ir_vis['ir_files'][f]))                                                                 
                        ir_results.append(fetch_convert_ir(combined_nc_dat, lon_shape, lat_shape, min_value=ir_min_value, max_value=ir_max_value))         
                    if not no_write_ir_gradient:
                        irg_results.append(fetch_convert_ir_gradient(combined_nc_dat, lon_shape, lat_shape, min_value=160.0, max_value=245.0, sigma=gradient_sigma))           
                    if not no_write_irlaplacian:
                        irl_results.append(fetch_convert_ir_laplacian(combined_nc_dat, lon_shape, lat_shape, min_value=160.0, max_value=245.0))
                    if not no_write_irdiff:
                        ird_results.append(fetch_convert_irdiff(combined_nc_dat, lon_shape, lat_shape))                                        
                    if not no_write_dirtyirdiff: 
                        dirtyird_results.append(fetch_convert_dirtyirdiff(combined_nc_dat, lon_shape, lat_shape))                              
                    if not no_write_trop: 
                        trop_results.append(fetch_convert_trop(combined_nc_dat, lon_shape, lat_shape, new_weighting))                          
                        
                if not no_write_vis:
                    if pd.notna(ir_vis['vis_files'][f]):
                        vis, tod0, sza = fetch_convert_vis(combined_nc_dat)                                                                    
                        if tod0 == 'day':
                            v_files.append(os.path.relpath(ir_vis['vis_files'][f]))                                                            
                            vis_results.append(vis)                                                                                            
                        sza_results.append(sza)                                                                                                
                        tod.append(tod0)                                                                                                       
                    else:
                        tod.append(np.nan)    
                else:
                    if not no_write_sza:
                        vis, tod0, sza = fetch_convert_vis(combined_nc_dat, no_write_vis=no_write_vis)                                       
                        sza_results.append(sza)                                                                                                
                        tod.append(tod0)                                                                                                       
                    else:
                        tod.append('night')
                        
                g_files.append(os.path.relpath(ir_vis['comb_files'][f])) # Keep comb_files list populated for the csv df_glm mapping

        d_str_out = datetime.strftime(ir_vis['date_time'][f], '%Y%j%H%M%S')
        
        # Save arrays
        if not no_write_ir and len(ir_results) > 0: 
            np.save(join(outdir, sector, 'ir', d_str_out + '_ir.npy'), np.asarray(ir_results))                                   
        if not no_write_ir_gradient and len(irg_results) > 0:
            np.save(join(outdir, sector, 'ir_gradient', d_str_out + '_ir_gradient.npy'), np.asarray(irg_results))                                  
        if not no_write_irlaplacian and len(irl_results) > 0:
            np.save(join(outdir, sector, 'irlaplacian', d_str_out + '_irlaplacian.npy'), np.asarray(irl_results))                                  
        if not no_write_vis and len(vis_results) > 0: 
            np.save(join(outdir, sector, 'vis', d_str_out + '_vis.npy'), np.asarray(vis_results))                                  
        if not no_write_sza and len(sza_results) > 0: 
            np.save(join(outdir, sector, 'sza', d_str_out + '_sza.npy'), np.asarray(sza_results))                                  
        if not no_write_irdiff and len(ird_results) > 0: 
            np.save(join(outdir, sector, 'ir_diff', d_str_out + '_ir_diff.npy'), np.asarray(ird_results))                                 
        if not no_write_dirtyirdiff and len(dirtyird_results) > 0: 
            np.save(join(outdir, sector, 'dirtyirdiff', d_str_out + '_dirtyirdiff.npy'), np.asarray(dirtyird_results))                             
        if not no_write_trop and len(trop_results) > 0: 
            np.save(join(outdir, sector, 'tropdiff', d_str_out + '_tropdiff.npy'), np.asarray(trop_results))                                 

    if not no_write_csv:
        if verbose: print('Writing file containing combined netCDF file names:', join(outdir, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'))
        
        df_ir2  = pd.DataFrame({'ir_files' : i_files,  'ir_index':range(len(i_files))})                                                        
        df_ir2['date_time'] = df_ir2['ir_files'].apply(get_raw_start_time)            
        df_ir2.set_index('ir_index')                                                                                                           
        
        df_tod2 = pd.DataFrame({'day_night':tod})                                                                                              
        df_tod2['date_time'] = df_ir2['date_time']            
        
        if not no_write_vis:
            df_vis2 = pd.DataFrame({'vis_files': v_files, 'vis_index':range(len(v_files))})                                                    
            df_vis2['date_time'] = df_vis2['vis_files'].apply(get_raw_start_time)      
            df_vis2.set_index('vis_index')                                                                                                     
            ir_vis2 = df_vis2.merge(df_ir2, on='date_time', how='outer', sort=True)                                                      
        else:
            ir_vis2 = df_ir2.copy()
       
        ir_vis2 = ir_vis2.merge(df_tod2, on='date_time', how='outer', sort=True)                                                         
        
        df_glm2 = pd.DataFrame({'glm_files': g_files, 'glm_index':range(len(g_files))})                                                        
        df_glm2['date_time'] = df_glm2['glm_files'].apply(get_comb_time)      
        df_glm2.set_index('glm_index')                                                                                                         
        
        ir_vis2 = ir_vis2.merge(df_glm2, on='date_time', how='outer', sort=True)                                                         
        
        df_comb_csv = df_glm2.copy()
        df_comb_csv.rename(columns={'glm_files': 'comb_files', 'glm_index': 'comb_index'}, inplace=True)
        ir_vis2 = ir_vis2.merge(df_comb_csv, on='date_time', how='outer', sort=True)                                                         
        
        csv_exist = glob.glob(join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'), recursive=True)             
        if len(csv_exist) == 1:
            ir_vis3 = pd.read_csv(csv_exist[0])                                                                                            
            if "Unnamed: 0" in ir_vis3.columns: ir_vis3.drop("Unnamed: 0", axis=1, inplace=True)
            ir_vis2 = pd.concat([ir_vis3, ir_vis2], axis=0, join='outer', ignore_index=True)            
            
        ir_vis2 = ir_vis2.iloc[pd.to_datetime(ir_vis2.date_time).values.argsort()].drop_duplicates(subset=['comb_files'], keep=keep0).reset_index(drop=True)       
        
        ir_vis2['ir_index']   = [idx if pd.notna(i) else np.nan for idx, i in enumerate(ir_vis2['ir_index'])]
        ir_vis2['glm_index']  = [idx if pd.notna(i) else np.nan for idx, i in enumerate(ir_vis2['glm_index'])]
        ir_vis2['comb_index'] = [idx if pd.notna(i) else np.nan for idx, i in enumerate(ir_vis2['comb_index'])]
        if not no_write_vis:
            ir_vis2['vis_index'] = [idx if pd.notna(i) else np.nan for idx, i in enumerate(ir_vis2['vis_index'])]
            
        ir_vis2.to_csv(join(outdir, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'))                                          
        
        if len(re.split('real_time', outroot)) > 1:
            ir_vis2.to_csv(join(outdir, sector, 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))                                 
        else:
            csv_fname = merge_csv_files(join(json_root, date_str0, sector), run_gcs=run_gcs)                                                 

    return(join(outdir, sector, 'ir', 'ir.npy'), np.asarray(ir_results).shape)


def get_comb_time(x):
    """ Extract exactly matching start time natively written to COMBINED NetCDF filename """
    basename = os.path.basename(x)
    dt_str = re.split('_s|_', basename)[5][:13]
    doy = int(dt_str[4:7])
    return datetime.strptime(f"{dt_str[0:4]} {doy} {dt_str[7:]}", "%Y %j %H%M%S")

def main():
    seviri_create_vis_ir_numpy_arrays_from_netcdf_files2()
    
if __name__ == '__main__':
    main()