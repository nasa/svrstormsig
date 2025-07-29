#+
# Name:
#     run_write_severe_storm_post_processing.py
# Purpose:
#     This is a script to read in the combined netCDF files and write to the same files the object ID number for each pixel, OT min - anvil mean BTD (which is the difference between minimum
#     temperature of OT object and the mean temperature of all pixels not part of an OT object that are also colder than 225 K), and the GFS tropopause temperature. 
# Calling sequence:
#     import run_write_severe_storm_post_processing
#     run_write_severe_storm_post_processing.run_write_severe_storm_post_processing()
# Input:
#     None.
# Functions:
#     run_write_severe_storm_post_processing : Main function to write the post processing data to combined netCDF file
# Output:
#     Numpy file of rewritten model results
# Keywords:
#     inroot        : STRING specifying input root directory to the vis_ir_glm_json csv files
#                     DEFAULT = '../../../goes-data/aacp_results/ir_vis_glm/plume_day_model/2021-04-28/'
#     outroot       : STRING specifying output root directory to save the 512x512 numpy files
#                     DEFAULT = None -> same as input root
#     use_local     : IF keyword set (True), use local files. If False, use files located on the google cloud. 
#                     DEFAULT = False -> use files on google cloud server. 
#     write_gcs     : IF keyword set (True), write the output files to the google cloud in addition to local storage.
#                     DEFAULT = True.
#     c_bucket_name : STRING specifying the name of the gcp bucket to read model results data files from. use_local needs to be False in order for this to matter.
#                     DEFAULT = 'aacp-results'
#     del_local     : IF keyword set (True) AND write_gcs == True, delete local copy of output file.
#                     DEFAULT = True.
#     object_type   : STRING specifying the object (OT or AACP) that you detected and want to postprocess.
#                     DEFAULT = 'AACP'
#     mod_type      : STRING specifying the model type used to make the detections that you want to postprocess. (ex. multiresunet, unet, attentionunet)
#                     DEFAULT = 'multiresunet'
#     sector        : STRING specifying the domain sector to use to create maps (= 'conus' or 'full' or 'm1' or 'm2'). DEFAULT = None -> use meso_sector
#     rewrite       : BOOL keyword to specify whether or not to rewrite the ID numbers and IR BTD, etc.
#                     DEFAULT = True -> rewrite the post-processed data
#     percent_omit  : FLOAT keyword specifying the percentage of cold and warm pixels to remove from anvil mean brightness temperature calculation
#                     DEFAULT = 20 -> 20% of the warmest and coldest anvil pixel temperatures are removed in order to prevent the contribution of noise to 
#                     the OT IR-anvil BTD calculation.
#                     NOTE: percent_omit must be between 0 and 100.
#     verbose       : BOOL keyword to specify whether or not to print verbose informational messages.
#                     DEFAULT = True which implies to print verbose informational messages
# Author and history:
#     John W. Cooney           2023-05-31.
#
#-

#### Environment Setup ###
# Package imports
import numpy as np
import glob
import os
import re
from math import ceil, floor
from scipy.ndimage import label, generate_binary_structure, convolve, generic_filter, distance_transform_edt, binary_dilation
from scipy.interpolate import interpn
import xarray as xr
from datetime import datetime, timedelta
import requests
from netCDF4 import Dataset
import pandas as pd
import pygrib
import warnings
import sys
import time
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(3, os.path.dirname(os.getcwd()))
from new_model.gcs_processing import write_to_gcs, list_gcs, download_ncdf_gcs
#from glm_gridder.dataScaler import DataScaler
from gridrad.rdr_sat_utils_jwc import gfs_interpolate_tropT_to_goes_grid, convert_longitude
#from new_model.run_write_gfs_trop_temp_to_combined_ncdf import lanczos_kernel, circle_mean_with_offset, weighted_cold_bias
#from visualize_results.gfs_contour_trop_temperature import gfs_contour_trop_temperature
def run_write_severe_storm_post_processing(inroot          = os.path.join('..', '..', '..', 'goes-data', 'combined_nc_dir', '20230516'),
                                           infile          = None,
                                           date1           = None,
                                           date2           = None,
                                           gfs_root        = os.path.join('..', '..', '..', 'gfs-data'),
                                           outroot         = None, 
                                           GFS_ANALYSIS_DT = 21600,
                                           use_local       = True, write_gcs = False, del_local = True,
                                           c_bucket_name   = 'ir-vis-sandwhich',
                                           object_type     = 'AACP',
                                           mod_type        = 'multiresunet',
                                           sector          = 'M2',
                                           pthresh         = None, 
                                           rewrite         = True,
                                           percent_omit    = 20, 
                                           verbose         = True):
    
    '''    
    Name:
        run_write_severe_storm_post_processing.py
    Purpose:
        This is a script to read in the combined netCDF files and write to the same files the object ID number for each pixel, OT min - anvil mean BTD (which is the difference between minimum
        temperature of OT object and the mean temperature of all pixels not part of an OT object that are also colder than 225 K), and the GFS tropopause temperature. 
    Calling sequence:
        import run_write_severe_storm_post_processing
        run_write_severe_storm_post_processing.run_write_severe_storm_post_processing()
    Input:
        None.
    Functions:
        run_write_severe_storm_post_processing : Main function to write the post processing data to combined netCDF file
    Output:
        Numpy file of rewritten model results
    Keywords:
        inroot          : STRING specifying input root directory to the combined netCDF files
                          DEFAULT = os.path.join('..', '..', '..', 'goes-data', 'combined_nc_dir', '20230516')
        infile          : STRING specifying input root directory to the vis_ir_glm_json csv files
                          DEFAULT = None -> loop over all files in the input root directory
        date1           : STRING specifying start date string to begin post-processing the data (YYYY-MM-DD hh:mm:ss)
                          DEFAULT = None  -> loop over all files in the input root directory
        date2           : STRING specifying start date string to complete post-processing the data (YYYY-MM-DD hh:mm:ss)
                          DEFAULT = None  -> loop over all files in the input root directory
        gfs_root        : STRING output directory path for GFS data storage
                          DEFAULT = os.path.join('..', '..', '..', 'gfs-data')
        outroot         : STRING specifying output root directory to save the netCDF output files
                          DEFAULT = None -> same as input root
        GFS_ANALYSIS_DT : FLOAT keyword specifying the time between GFS files in sec. 
                          DEFAULT = 21600 -> seconds between GFS files
        use_local       : IF keyword set (True), use local files. If False, use files located on the google cloud. 
                          DEFAULT = False -> use files on google cloud server. 
        write_gcs       : IF keyword set (True), write the output files to the google cloud in addition to local storage.
                          DEFAULT = True.
        c_bucket_name   : STRING specifying the name of the gcp bucket to read model results data files from. use_local needs to be False in order for this to matter.
                          DEFAULT = 'aacp-results'
        del_local       : IF keyword set (True) AND write_gcs == True, delete local copy of output file.
                          DEFAULT = True.
        object_type     : STRING specifying the object (OT or AACP) that you detected and want to postprocess.
                          DEFAULT = 'AACP'
        mod_type        : STRING specifying the model type used to make the detections that you want to postprocess. (ex. multiresunet, unet, attentionunet)
                          DEFAULT = 'multiresunet'
        sector          : STRING specifying the domain sector to use to create maps (= 'conus' or 'full' or 'm1' or 'm2'). DEFAULT = None -> use meso_sector
        pthresh         : FLOAT keyword to specify the optimal likelihood value to threshold the outputted model likelihoods in order for object to be OT or AACP
                          DEFAULT = None -> use the default value in file
                          NOTE: day_night optimal runs may require different pthresh scores that yield the best results. It is suggested to keep this as None for those
                          jobs.
        rewrite         : BOOL keyword to specify whether or not to rewrite the ID numbers and IR BTD, etc.
                          DEFAULT = True -> rewrite the post-processed data
        percent_omit    : FLOAT keyword specifying the percentage of cold and warm pixels to remove from anvil mean brightness temperature calculation
                          DEFAULT = 20 -> 20% of the warmest and coldest anvil pixel temperatures are removed in order to prevent the contribution of noise to 
                          the OT IR-anvil BTD calculation.
                          NOTE: percent_omit must be between 0 and 100.
        verbose         : BOOL keyword to specify whether or not to print verbose informational messages.
                          DEFAULT = True which implies to print verbose informational messages
    Author and history:
        John W. Cooney           2023-05-31.
    '''
    
    if object_type.lower() == 'plume' or object_type.replace(" ", '').lower() == 'warmplume':
        object_type = 'AACP'
   
    if object_type.lower() == 'updraft' or object_type.replace(" ", '').lower() == 'overshootingtop':
        object_type = 'OT'
    
    if mod_type.lower() != 'multiresunet' and mod_type.lower() != 'unet' and mod_type.lower() != 'attentionunet':
        print('Model specified is not available!!')
        exit()
    
    if infile is not None:
        inroot = os.path.dirname(infile)
        
    inroot = os.path.realpath(inroot)                                                                                                                  #Create link to real path so compatible with Mac
    d_str  = os.path.basename(inroot)                                                                                                                  #Extract date string from the input root directory
    yyyy   = d_str[0:4]
    if outroot == None:
        outroot = inroot                                                                                                                               #Create link to real path so compatible with Mac
    else: 
        outroot = os.path.realpath(outroot)                                                                                                            #Create link to real path so compatible with Mac
        os.makedirs(outroot, exist_ok = True)                                                                                                          #Create output directory file path if does not already exist
    
    if use_local:
        gfs_files = sorted(glob.glob(os.path.join(gfs_root, '**', '**', '*.grib2'), recursive = True))                                                 #Search for all GFS model files
    else:
        gfs_files = sorted(list_gcs(m_bucket_name, 'gfs-data', ['.grib2'], delimiter = '*/*'))                                                         #Extract names of all of the GOES visible data files from the google cloud   

    gdates = [(re.split('\.', os.path.basename(g)))[2] for g in gfs_files]
    sector = ''.join(e for e in sector if e.isalnum())                                                                                                 #Remove any special characters or spaces.
    if sector[0].lower() == 'm':
        try:
            sector    = 'M' + sector[-1]
        except:
            sector    = 'meso'
    elif sector[0].lower() == 'c':
        sector    = 'C'
    elif sector[0].lower() == 'f': 
        sector    = 'F'
    elif sector[0].lower() == 'q' or sector[0].lower() == 'h' or sector[0].lower() == 't': 
        sector    = sector.upper()
    else:
        print('Satellite sector specified is not available. Enter M1, M2, F, or C. Please try again.')
        print()
    
    if object_type.lower() == 'aacp':
        frac_max_pix = 0.10                                                                                                                            #Keep all pixels that are at least 10% of the value of the maximum likelihood score in the plume
    else:
        frac_max_pix = 0.50                                                                                                                            #Keep all pixels that are at least 50% of the value of the maximum likelihood score in the plume
    
    if infile is not None:
        nc_files = [infile]
    else:
        if use_local:    
            nc_files = sorted(glob.glob(os.path.join(inroot, '*_' + sector + '_*.nc')))                                                                #Find netCDF files to postprocess
        else:
            pref     = os.path.join(os.path.basename(os.path.dirname(inroot)), os.path.basename(inroot))                                               #Find netCDF files in GCP bucket to postprocess
            nc_files = sorted(list_gcs(c_bucket_name, pref, ['_' + sector + '_']))                                                                     #Extract names of all of the GOES visible data files from the google cloud   
        
        if date1 != None:
            # Convert to datetime objects
            dt1 = datetime.strptime(date1, '%Y-%m-%d %H:%M:%S')
            nc_files = sorted([f for f in nc_files if (dt := extract_combined_nc_datetime(f)) and dt1 <= dt])
        if date2 != None:
            dt2 = datetime.strptime(date2, '%Y-%m-%d %H:%M:%S')
            nc_files = sorted([f for f in nc_files if (dt := extract_combined_nc_datetime(f)) and dt2 >= dt])

    if len(nc_files) <= 0:
        print('No files found in specified directory???')
        print(sector)
        if use_local:
            print(inroot)
        else:
            print(c_bucket_name)
            print(pref)
        exit()
            
    native_ir = False
    
    x_sat0  = []
    y_sat0  = []
    gfsf    = 'kadshbfavbjfv'
    lanczos_kernel0 = lanczos_kernel(7, 3)                                                                                                             #Define the Lanczos kernel (kernel size = 7x7; cutoff frequency = 3)
    s       = generate_binary_structure(2,2)
    counter = 0
    for l in range(len(nc_files)):
        nc_file = nc_files[l]
        if use_local != True:
            download_ncdf_gcs(c_bucket_name, nc_file, inroot)                                                                                          #Download netCDF file from GCP

        nc_dct = {}
        with xr.open_dataset(os.path.join(inroot, os.path.basename(nc_file))).load() as f:    
            dt64 = pd.to_datetime(f['time'].values[0])                                                                                                 #Read in the date of satellite scan and convert it to Timestamp
            date = datetime(dt64.year, dt64.month, dt64.day, dt64.hour, dt64.minute, dt64.second)                                                      #Convert Timestamp to datetime structure
            x_sat = f['longitude'].values                                                                                                              #Extract latitude and longitude satellite domain
            y_sat = f['latitude'].values                                                                                                               #Extract latitude and longitude satellite domain
            var_keys = list(f.keys())                                                                                                                  #Get list of variables contained within the netCDF file
            for var_key in var_keys:
                if '_' + object_type.lower() in var_key and 'id' not in var_key and 'anvilmean_brightness_temperature_difference' not in var_key and 'bt' not in var_key:   #Check so only read object data likelihood scores of interest
                    nc_dct[var_key] = f[var_key]                                                                                                       #Read object data with likelihood scores
            try:
                xyres = f.spatial_resolution
            except:
                if 'visible_reflectance' in var_keys:
                    xyres = '0.5km at nadir'
                else:    
                    xyres = '2km at nadir'
                    native_ir = True
            bt  = f['ir_brightness_temperature']
            sat = f['imager_projection'].attrs['satellite_name']
        
        if 'mtg' in sat.lower() or 'mti' in sat.lower():
            aacp_thresh2 = 0.95
        else:
            aacp_thresh2 = 0.65
        xyres0 = np.float64(re.split('km', xyres)[0])
        anv_p  = int((15.0/xyres0)*2.0)                                                                                                                #Calculate number of pixels to be included as part of the anvil
        sigma  = int(36.0/(xyres0*2.0))                                                                                                                #Calculate the standard deviation for Gaussian kernel for smoothing tropopause temperatures on the satellite grid
        keys0  = list(nc_dct.keys())                                                                                                                   #Extract keys 
        if len(keys0) <= 0:
            if verbose:
                print('Anvil mean BT and/or tropopause temperature not found in combined netCDF file. Writing now.')
        else: 
            bt0 = bt.values[0, :, :]
            for k in range(len(keys0)):
                if keys0[k] + '_id_number' not in var_keys or rewrite:
                    data = nc_dct[keys0[k]]
                    if pthresh == None:
                        pthresh0 = data.optimal_thresh
                    else:
                        pthresh0 = pthresh    
                    if data.attrs['model_type'].lower() == mod_type.lower():                                                                           #Make sure to only loop over instances where model type in combined netCDF file matches the specified one
                        res  = data.values[0, :, :]
                        res[res < 0.05] = 0                                                                                                            #Set all results pixels lower than the mean probability threshold to 0
                        labeled_array, num_updrafts = label(res > 0, structure = s)                                                                    #Extract labels for each updraft location (make sure points at diagonals are part of same updrafts)
#                         if verbose == True:
#                             print('Number of ' + object_type + 's for time step = ' + str(num_updrafts)) 
                        for u in range(num_updrafts):
                            res0 = np.copy(res)
                            inds = (labeled_array == u+1)                                                                                              #Find locations of updraft region mask
                            if np.sum(inds == True) > 0:                                                                                               #Find if results has any overlapping indices with mask updraft. If so, then we have a match
                                max_res = np.nanmax(res[inds])                                                                                         #Find maximum confidence in the object
                                if max_res >= pthresh0:
                                    res50   = frac_max_pix*max_res                                                                                     #Calculate value that is 10% or 50% of maximum. All pixels attached to object < this value are removed from object
                                    res2    = np.copy(res0[inds])
                                    res2[res2 <  res50] = 0                                                                                            #Set pixels with confidence below 50% of max value in object to 0
                                    res2[res2 >= res50] = max_res                                                                                      #Set pixels with confidence ≥ 50% of max value in object to max value in object
                                    res[inds] = np.copy(res2)                                                                                          #Copy updated results array to res
                        
                        if object_type.lower() == 'aacp':
                            res00 = res.copy()
                            res00[res00 < pthresh0] = 0
                            convective_proxy = bt0 <= 230
                            labeled_cells0, num_cells = label(convective_proxy, structure = s) 
                            ll = np.copy(labeled_cells0).astype(float)
#                             ll[ll <= 0] = np.nan
                            labeled_array0, num_updrafts0 = label(res00 > 0, structure = s)                                                            #Extract labels for each updraft location (make sure points at diagonals are part of same updrafts)
                            for u0 in range(num_updrafts0):
                                inds0 = (labeled_array0 == u0+1)
                                if np.sum(inds0) > 0:                                                                                                  #Find if results has any overlapping indices with mask updraft. If so, then we have a match
                                    #Get AACP blob coordinates
                                    coords = np.argwhere(inds0)
                                    if np.nanmax(ll[coords[:, 0], coords[:, 1]]) != np.nanmin(ll[coords[:, 0], coords[:, 1]]):
                                        print('Coords cells do not match???')
                                        print(np.nanmax(ll[coords[:, 0], coords[:, 1]]))
                                        print(np.nanmin(ll[coords[:, 0], coords[:, 1]]))
                                        exit()
                                    
                                    idxx = ll[coords[:, 0], coords[:, 1]][0]
                                    local_bt = bt0[ll == idxx]
                                    bt_thresh = np.percentile(local_bt, 5)
                                    
#                                     r0, c0 = np.mean(coords, axis=0).astype(int)
#                                     rmin, rmax = max(r0 - 10, 0), min(r0 + 11, bt0.shape[0])
#                                     cmin, cmax = max(c0 - 10, 0), min(c0 + 11, bt0.shape[1])
                    
                                    convective_proxy2 = bt0 < bt_thresh
                                    dist_to_convection = distance_transform_edt(~convective_proxy2)
                            
                                    min_dist = np.min(dist_to_convection[inds0])
                    
                                    # Dilate the detection slightly
                                    if native_ir:
                                        dilated_inds0 = binary_dilation(inds0, iterations=3)
                                        dist_thresh   = 3
                                    else:
                                        dilated_inds0 = binary_dilation(inds0, iterations=5)
                                        dist_thresh   = 10
                                    
                                    ring_mask = np.logical_and(dilated_inds0, ~inds0)                                                                  #Only outer ring
                            
                                    #Check if max BT in ring is warm
                                    if np.any(ring_mask):
                                        max_ring_bt = np.nanmax(bt0[ring_mask])
                                    else:
                                        max_ring_bt = 0                                                                                                #Failsafe
                    
                                    if (min_dist > dist_thresh or max_ring_bt > 235) and np.nanmax(res00[inds0]) < aacp_thresh2:                    
#                                         print('Removing detection')
#                                         print(date)
                                        res[inds0] = pthresh0-0.01              

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            maxc = np.nanmax(res)
                        if maxc < pthresh0:
                            btd   = res*np.nan                                                                                                         #Set values to 0 if no objects detected for time step
                            ot_id = np.full_like(res, 0).astype('uint16')                                                                              #Set values to 0 if no objects detected for time step
                        else:
                            labeled_array2, num_updrafts2 = label(res >= pthresh0, structure = s)                                                      #Extract labels for each updraft location (make sure points at diagonals are part of same updrafts)
                            anvil_mask = (labeled_array2 <= 0)                                                                                         #Find all pixels that are not part of an OT/AACP object
                            btd = res*np.nan                                                                                                           #Initialize array to store objects detected data for time step
                            if object_type.lower() == 'ot':                                                                                            #Only calculate the anvil mean BTD if object we are identifying is OTs
                                ot_id = np.full_like(res, 0).astype('uint16')
                            else:
                                ot_id = labeled_array2.astype('uint16')
    
                            ot_num = 1
                            for u in range(num_updrafts2):
                                inds2 = (labeled_array2 == u+1)                                                                                        #Find locations of detected object region mask
                                arr   = np.where(inds2 == True)                                                                                        #Find pixel indices in object detected
                                if len(arr[0]) > 0:
                                    min_ind = np.nanargmin(bt0[inds2])                                                                                 #Find minimum brightness temperature location in object
                                    min_bt0 = bt0[inds2][min_ind]                                                                                      #Extract minimum BT within object
                                    if object_type.lower() == 'ot':                                                                                    #Only calculate the anvil mean BTD if object we are identifying is OTs
                                        i_pix = [arr[0][min_ind]-int(anv_p/2), arr[0][min_ind]+ceil(anv_p/2)]                                          #Find indices in which to calculate 15x15 pixel box to calculate the anvil mean
                                        j_pix = [arr[1][min_ind]-int(anv_p/2), arr[1][min_ind]+ceil(anv_p/2)]                                          #Find indices in which to calculate 15x15 pixel box to calculate the anvil mean
                                        i_pix[0] = 0 if i_pix[0] == 0 else i_pix[0]                                                                    #Set minimum i index to 0 if negative
                                        j_pix[0] = 0 if j_pix[0] == 0 else j_pix[0]                                                                    #Set minimum j index to 0 if negative
                                        anvil_area_mask = anvil_mask[i_pix[0]:i_pix[1], j_pix[0]:j_pix[1]]                                             #Mask area surrounding the minimum object BT
                                        anvil_bts = anvil_area_mask*bt0[i_pix[0]:i_pix[1], j_pix[0]:j_pix[1]]                                          #Calculate BTs for anvil region around OT
                                        anvil_bts[anvil_bts <= 0] = np.nan                                                                             #Set all regions that are part of an OT object to NaN
                                        anvil_bts[anvil_bts > 235] = np.nan                                                                            #Set all regions that are warmer than 230 K to NaN to avoid ground contamination
                                        anvil_bts     = np.sort(anvil_bts[np.isfinite(anvil_bts)].flatten())                                           #Sort BTs in anvil in order to remove upper and lower 10% before taking the mean
                                        if len(anvil_bts) <= 0:
                                            anvil_bt_mean = np.nan                                                                                     #Set to 0 if no pixels possible to calculate anvil mean
                                        else:
                                            tenper    = int(np.floor((percent_omit/100.0)*len(anvil_bts)))                                             #Calculate the 20% of number of pixels in anvil
                                            anvil_bts = anvil_bts[0+tenper:len(anvil_bts)-tenper]
                                            anvil_bt_mean = np.nanmean(anvil_bts)                                                                      #Calculate anvil mean BTD
                                            if np.isnan(anvil_bt_mean):
                                                print('NaN data in mean')
                                                print(tenper)
                                                print(len(anvil_bts))
                                                print(nc_file)
                                                exit()
                                            
                                        val = min_bt0 - anvil_bt_mean
                                        if val <= -1:                                      
                                           if (min_bt0 >= 225) and val >= -6:
                                               temppppp = 1
                                           else:
                                               ot_id[inds2] = ot_num
                                               btd[inds2]   = val                                                                                      #Subtract minimum brightness temperature of OT by the anvil mean brightness temperature difference (BTD)
                                               ot_num += 1
#                                         else:
#                                            ot_id[inds2] = 0
                        files_used = []            
                        if len(gfs_files) > 0 and 'tropopause_temperature' not in var_keys:
                            near_date = gfs_nearest_time(date, GFS_ANALYSIS_DT, ROUND = 'round')                                                       #Find nearest date to satellite scan
                            nd_str    = near_date.strftime("%Y%m%d%H")                                                                                 #Extract nearest date to satellite scan as a string
                            no_f = 0
                            try:
                                gfs_file = gfs_files[gdates.index(nd_str)]                                                                             #Find file that matches date string
                            except:
                                no_f = 1

                            if no_f == 1:
                                for cc in range(1, 8):
                                    near_date = gfs_nearest_time(date-timedelta(seconds = cc*GFS_ANALYSIS_DT), GFS_ANALYSIS_DT, ROUND = 'round')       #Find nearest date to satellite scan
                                    nd_str    = near_date.strftime("%Y%m%d%H")                                                                         #Extract nearest date to satellite scan as a string
                                    no_f = 0
                                    try:
                                        gfs_file = gfs_files[gdates.index(nd_str)]                                                                     #Find file that matches date string
                                    except:
                                        no_f = 1
                                    if no_f == 0:
                                        break    
                            
                            if no_f == 0:
#OLD METHOD OF INTERPOLATING GFS TROPOPAUSE TEMPERATURES
#                                 grbs  = pygrib.open( gfs_file )                                                                                        #Open Grib file to read
#                                 grb   = grbs.select(name='Temperature', typeOfLevel='tropopause')[0]                                                   #Extract tropopause temperatures from the GRIB file
#                                 tropT = grb.values
#                                 
#                                 
#                                 
#                                 lats, lons = grb.latlons()                                                                                             #Extract GFS latitudes and longitudes
#                                 x_sat = convert_longitude(x_sat, to_model = True)                                                                      #Convert longitudes to be 0-360 degrees
#                                 tropT = np.flip(tropT, axis = 0)                                                                                       #Put latitudes in ascending order in order to be interpolated onto satellite grid
#                                 lats = np.flip(lats[:, 0])                                                                                             #Put latitudes in ascending order in order to be interpolated onto satellite grid
#                                 lons = convert_longitude(lons[0, :], to_model = True)
#                                 x_sat = convert_longitude(x_sat, to_model = True)                                                                      #Convert longitudes to be 0-360 degrees
#                                 tropT = gfs_interpolate_tropT_to_goes_grid(tropT, lons, lats, x_sat, y_sat,                                            #Interpolate tropopause temperatures to GOES satellite grid points
#                                                                            no_parallax = True, 
#                                                                            x_pc        = [],
#                                                                            y_pc        = [],
#                                                                            sigma       = sigma,
#                                                                            verbose     = verbose)
#                                 grbs.close()

                                x_sat = convert_longitude(x_sat, to_model = True)                                                                      #Convert longitudes to be 0-360 degrees
                                #Check if already read in this GFS file so do not have to read same file multiple times in a row
                                if gfs_file != gfsf:
                                  grbs  = pygrib.open( gfs_file )                                                                                      #Open Grib file to read
                                  grb   = grbs.select(name='Temperature', typeOfLevel='tropopause')[0]                                                 #Extract tropopause temperatures from the GRIB file
                                  tropT = np.copy(grb.values)                                                                                          #Copy tropopause temperatures from backward date into array
                                  lats, lons = grb.latlons()                                                                                           #Extract GFS latitudes and longitudes
                                  tropT = np.flip(tropT, axis = 0)                                                                                     #Put latitudes in ascending order in order to be interpolated onto satellite grid
#                                   tropT = weighted_cold_bias(tropT, 5, weight=0.2)
                                  tropT = circle_mean_with_offset(tropT, 20, -0.6)        
                                  lats  = np.flip(lats[:, 0])                                                                                          #Put latitudes in ascending order in order to be interpolated onto satellite grid
                                  lons  = convert_longitude(lons[0, :], to_model = True)
#                                   lons  = (np.asarray(lons)).tolist()
#                                   lats  = (np.asarray(lats)).tolist()
#                                   tropT = convolve(tropT, lanczos_kernel0)                                                                             #Apply Lanczos kernel to the interpolated data 
#                                   tropT2 = interpn((lats, lons), tropT, (y_sat, x_sat), method = 'linear', bounds_error = False, fill_value = np.nan)


                                  #Longitude wraparound padding
                                  N_wrap = 10                                                                                                          #Number of grid points to duplicate on each side (tweak as needed)
                                  
                                  #Extend longitudes (pad with wraparound)
                                  lons_extended = np.concatenate((
                                      lons[-N_wrap:] - 360,                                                                                            #Left pad (negative side)
                                      lons,
                                      lons[:N_wrap] + 360                                                                                              #Right pad (beyond 360)
                                  ))
                              
                                  #Extend tropT to match padded longitudes
                                  tropT_extended = np.concatenate((
                                      tropT[:, -N_wrap:],                                                                                              #Left pad
                                      tropT,
                                      tropT[:, :N_wrap]                                                                                                #Right pad
                                  ), axis=1)
                              
                                  #Apply Lanczos kernel
                                  tropT_extended = convolve(tropT_extended, lanczos_kernel0)
                                  tropT2 = interpn((lats, lons_extended), tropT_extended, (y_sat, x_sat), method = 'linear', bounds_error = False, fill_value = np.nan)

                                  grbs.close()
                                  gfsf   = gfs_file
                                  x_sat0 = x_sat
                                  y_sat0 = y_sat
                                else:
                                  if len(x_sat0) <= 0 or len(y_sat0) <= 0:
                                    print('I think something had to go wrong here.. why is there no x_sat0 or y_sat0 values??? Should have needed to read in the GFS file first time through loop.')
                                    print(len(x_sat0))
                                    print(len(y_sat0))
                                    exit()
                                    tropT2 = interpn((lats, lons), tropT, (y_sat, x_sat), method = 'linear', bounds_error = False, fill_value = np.nan)
                                    x_sat0 = x_sat
                                    y_sat0 = y_sat
                                  else:
                                    #If interpolating to different latitudes and longitudes then need to interpolate (if not, use the same TropT2 as previous loop iteration
                                    if np.nanmax(np.abs(x_sat - x_sat0)) != 0 or np.nanmax(np.abs(y_sat - y_sat0)) != 0:
                                      tropT2 = interpn((lats, lons_extended), tropT_extended, (y_sat, x_sat), method = 'linear', bounds_error = False, fill_value = np.nan)
                                    
                                files_used = [os.path.realpath(gfs_file)]
                            else:
                                tropT2 = res*np.nan
                        else:
                          tropT2 = res*np.nan
                        
                        append_combined_ncdf_with_model_post_processing(nc_file, ot_id, btd, tropT2, data.attrs, anv_p, files_used = files_used, pthresh = pthresh, rewrite = rewrite, percent_omit = percent_omit, write_gcs = write_gcs, del_local = del_local, outroot = outroot, c_bucket_name = c_bucket_name, verbose = verbose)

def append_combined_ncdf_with_model_post_processing(nc_file, object_id, btd, tropT, mod_attrs, resolution, files_used = [], pthresh = None, rewrite = True, percent_omit = 20, write_gcs = True, del_local = True, outroot = None, c_bucket_name = 'ir-vis-sandwhich', verbose = True):
  '''
  This is a function to append the combined netCDF files with the model post-processing data. 
  Args:
      nc_file    : Filename and path to combined netCDF file that will be appended.
      object_id  : Numpy array with the model ID numbers above a set probability threshold.
      btd        : Numpy array with the model minimum brightness temperature of object minus the anvil mean brightness temperature
      tropT      : Numpy array with the GFS tropopause temperatures interpolated to each pixel in the satellite domain
      mod_attrs  : Attributes of combined netCDF variable
      resolution : FLOAT giving the number of satellite pixels in x and y space of the anvil calculation
  Keywords:
      files_used    : List of grib files used to make the tropopause temperature calculation.
      pthresh       : FLOAT keyword to specify the optimal likelihood value to threshold the outputted model likelihoods in order for object to be OT or AACP
                      DEFAULT = None -> use the default value in file
                      NOTE: day_night optimal runs may require different pthresh scores that yield the best results. It is suggested to keep this as None for those
                      jobs.
      rewrite       : BOOL keyword to specify whether or not to rewrite the ID numbers and IR BTD, etc.
                      DEFAULT = True -> rewrite the post-processed data
      percent_omit  : FLOAT keyword specifying the percentage of cold and warm pixels to remove from anvil mean brightness temperature calculation
                      DEFAULT = 20 -> 20% of the warmest and coldest anvil pixel temperatures are removed in order to prevent the contribution of noise to 
                      the OT IR-anvil BTD calculation.
                      NOTE: percent_omit must be between 0 and 100.
      write_gcs     : IF keyword set (True), write the output files to the google cloud in addition to local storage.
                      DEFAULT = True.
      del_local     : IF keyword set (True) AND run_gcs = True, delete local copy of output file.
                      DEFAULT = True.
      outroot       : STRING output directory path for results data storage
                      DEFAULT = same as nc_file
      c_bucket_name : STRING specifying the name of the gcp bucket to write combined IR, VIS, GLM files to As well as 3 modalities figures.
                      run_gcs needs to be True in order for this to matter.
                      DEFAULT = 'ir-vis-sandwhich'
      verbose       : BOOL keyword to specify whether or not to print verbose informational messages.
                      DEFAULT = True which implies to print verbose informational messages
  Output:
      Appends the combined netCDF data files with post-processing data
  '''  
  with Dataset(nc_file, 'a', format="NETCDF4") as f:                                                                                                   #Open combined netCDF file to append with model results
    chan0  = re.split('_', mod_attrs['standard_name'])[0].upper()                                                                                      #Extract satellite channels input into model
    vname  = chan0.replace('+', '_').lower() + '_' + re.split('_', mod_attrs['standard_name'])[1].lower()                                              #Extract model info to create variable name
    mod_description = chan0 + ' ' + re.split('_', mod_attrs['standard_name'])[1].upper()                                                               #Extract model info for attributes description
  #  missin = np.where(np.isnan(btd))
  #  btd[missin] = 0.0
    vnames = list(f.variables.keys())
    if vname + '_id_number' not in vnames or rewrite:
      if vname + '_id_number' in vnames:
        f[vname + '_id_number'][0, :, :] = object_id
        if pthresh != None:
          f[vname + '_id_number'].likelihood_threshold = pthresh
        if 'ot' in mod_attrs['standard_name'].lower():
          f[vname + '_anvilmean_brightness_temperature_difference'][0, :, :] = btd
          if pthresh != None:
            f[vname + '_anvilmean_brightness_temperature_difference'].likelihood_threshold = pthresh
        
        if 'tropopause_temperature' not in vnames and np.sum(np.isfinite(tropT)) > 0:
          var_mod3 = f.createVariable('tropopause_temperature', 'f4', ('time', 'Y', 'X',), zlib = True, least_significant_digit = 2, complevel = 7)#, fill_value = Scaler._FillValue)
          var_mod3.set_auto_maskandscale( False )
          var_mod3.long_name          = "Temperature of the Tropopause Retrieved from GFS Interpolated and Smoothed onto Satellite Grid"
          var_mod3.standard_name      = "GFS Tropopause"
          var_mod3.contributing_files = files_used
          var_mod3.missing_value      = np.nan
          var_mod3.units              = 'K'
          var_mod3.coordinates        = 'longitude latitude time'
          var_mod3[0, :, :]           = np.copy(tropT)
      else:
        lat = np.copy(np.asarray(f.variables['latitude'][:,:]))                                                                                        #Read latitude from combined netCDF file to make sure it is the same shape as the model results
        lon = np.copy(np.asarray(f.variables['longitude'][:,:]))                                                                                       #Read longitude from combined netCDF file to make sure it is the same shape as the model results
        if object_id.shape != lat.shape or btd.shape != lon.shape:
          print('Model results file does not match file latitude or longitude shape!!!!')
          print(lon.shape)
          print(lat.shape)
          print(object_id.shape)
          print(btd.shape)
          exit()
        #Declare variables
        var_mod = f.createVariable(vname + '_id_number', 'u2', ('time', 'Y', 'X',), zlib = True, complevel = 7)
#        var_mod = f.createVariable(vname + '_id_number', np.uint16, ('time', 'Y', 'X',), zlib = True)
        var_mod.set_auto_maskandscale( False )
        var_mod.long_name      = mod_description + ' Identification Number'
        var_mod.standard_name  = mod_description + ' ID Number'
        var_mod.model_type     = mod_attrs['model_type']
   #     var_mod.valid_range    = np.asarray([1, 65535], dtype=np.uint16)
        if pthresh == None:
          var_mod.likelihood_threshold = mod_attrs['optimal_thresh']
        else:
          var_mod.likelihood_threshold = pthresh
        
        var_mod.missing_value  = 0
        var_mod.units          = 'dimensionless'
        var_mod.coordinates    = 'longitude latitude time'
        var_mod.description    = "The object Identification Number field shows all pixels that belong to an individual object region.  The ID numbers apply uniquely to each satellite scan, i.e. ID number 1 in one scan will likely not be the same feature as ID number 1 in the next scan, and therefore cannot be used to track an object throughout its lifetime." 
        var_mod[0, :, :]       = np.copy(object_id)                                                                                                    #Write the Identification numbers to the combined netCDF file
        if 'ot' in mod_attrs['standard_name'].lower():
#          Scaler  = DataScaler( nbytes = 4, signed = True )                                                                                           #Extract data scaling and offset ability for np.int32
          var_mod2 = f.createVariable(vname + '_anvilmean_brightness_temperature_difference', 'f4', ('time', 'Y', 'X',), zlib = True, least_significant_digit = 2, complevel = 7)#, fill_value = Scaler._FillValue)
#          var_mod2 = f.createVariable(vname + '_anvilmean_brightness_temperature_difference', 'f4', ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue)
          var_mod2.set_auto_maskandscale( False )
          var_mod2.long_name      = chan0 + " Overshooting Top Minus Anvil Brightness Temperature Difference"
          var_mod2.standard_name  = chan0 + " OT - Anvil BTD"
          var_mod2.model_type     = mod_attrs['model_type']
          var_mod2.valid_range    = [-50.0, 0.0]
          var_mod2.missing_value  = np.nan
          if pthresh == None:
            var_mod2.likelihood_threshold = mod_attrs['optimal_thresh']
          else:
            var_mod2.likelihood_threshold = pthresh
          var_mod2.units          = 'K'
          var_mod2.coordinates    = 'longitude latitude time'
          var_mod2.description    = "Minimum brightness temperature within an OT Minus Anvil Brightness Temperature. The anvil is identified as all pixels, not part of an OT, within a " + str(int(resolution)) + "x" + str(int(resolution)) + " pixel region centered on the coldest BT in OT object. Temperatures > 230 K are removed and then the coldest and warmest " + str(percent_omit) + "% of anvil BT pixels are removed prior to anvil mean calculation." 
          var_mod2[0, :, :]       = np.copy(btd)
#           data2, scale_btd, offset_btd = Scaler.scaleData(btd)                                                                                       #Extract BTD data, scale factor and offsets that is scaled from Float to short
#           var_mod2.add_offset    = offset_btd                                                                                                        #Write the data offset to the combined netCDF file
#           var_mod2.scale_factor  = scale_btd                                                                                                         #Write the data scale factor to the combined netCDF file
#           var_mod2[0, :, :]      = data2
#           var_mod2.add_offset    = offset_btd                                                                                                        #Write the data offset to the combined netCDF file
#           var_mod2.scale_factor  = scale_btd                                                                                                         #Write the data scale factor to the combined netCDF file
        if np.sum(np.isfinite(tropT)) > 0:
#          Scaler  = DataScaler( nbytes = 4, signed = True )                                                                                           #Extract data scaling and offset ability for np.int32
          var_mod3 = f.createVariable('tropopause_temperature', 'f4', ('time', 'Y', 'X',), zlib = True, least_significant_digit = 2, complevel = 7)#, fill_value = Scaler._FillValue)
          var_mod3.set_auto_maskandscale( False )
          var_mod3.long_name          = "Temperature of the Tropopause Retrieved from GFS Interpolated and Smoothed onto Satellite Grid"
          var_mod3.standard_name      = "GFS Tropopause"
          var_mod3.contributing_files = files_used
     #     var_mod3.valid_range        = [160.0, 310.0]
          var_mod3.missing_value      = np.nan
          var_mod3.units              = 'K'
          var_mod3.coordinates        = 'longitude latitude time'
 #         var_mod3.description        = "Minimum brightness temperature within an OT Minus Anvil Brightness Temperature. The anvil is identified as all pixels, not part of an OT, within a " + str(int(resolution)) + "x" + str(int(resolution)) + " pixel region centered on the coldest BT in OT object. Temperatures > 230 K are removed and then the coldest and warmest " + str(percent_omit) + "% of anvil BT pixels are removed prior to anvil mean calulcation." 
          var_mod3[0, :, :]           = np.copy(tropT)
#           data2, scale_btd, offset_btd = Scaler.scaleData(btd)                                                                                       #Extract BTD data, scale factor and offsets that is scaled from Float to short
#           var_mod2.add_offset    = offset_btd                                                                                                        #Write the data offset to the combined netCDF file
#           var_mod2.scale_factor  = scale_btd                                                                                                         #Write the data scale factor to the combined netCDF file
#           var_mod2[0, :, :]      = data2
#           var_mod2.add_offset    = offset_btd                                                                                                        #Write the data offset to the combined netCDF file
#           var_mod2.scale_factor  = scale_btd                                                                                                         #Write the data scale factor to the combined netCDF file
  
    if verbose:
        print('Post-processing model output netCDF file name = ' + nc_file)
#   f.close()                                                                                                                                          #Close combined netCDF file once finished appending
  return()

def download_gfs_analysis_files(date_str1, date_str2,
                                GFS_ANALYSIS_DT = 21600,
                                outroot         = os.path.join('..', '..', '..', 'gfs-data'),
                                c_bucket_name   = 'ir-vis-sandwhich',
                                write_gcs       = False,
                                del_local       = True,
                                real_time       = False,
                                verbose         = True):
    '''
    This is a function to to download selected files with date range from rda.ucar.edu 
    Args:
        date_str1 : Start date. String containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to start downloading 
        date_str2 : End date. String containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to end downloading 
    Keywords:
        GFS_ANALYSIS_DT : FLOAT keyword specifying the time between GFS files in sec. 
                          DEFAULT = 21600 -> seconds between GFS files
        write_gcs       : IF keyword set (True), write the output files to the google cloud in addition to local storage.
                          DEFAULT = False.
        del_local       : IF keyword set (True) AND run_gcs = True, delete local copy of output file.
                          DEFAULT = True.
        outroot         : STRING output directory path for GFS data storage
                          DEFAULT = os.path.join('..', '..', '..', 'gfs-data')
        c_bucket_name   : STRING specifying the name of the gcp bucket to write combined IR, VIS, GLM files to As well as 3 modalities figures.
                          run_gcs needs to be True in order for this to matter.
                          DEFAULT = 'ir-vis-sandwhich'
        verbose         : BOOL keyword to specify whether or not to print verbose informational messages.
                        DEFAULT = True which implies to print verbose informational messages
    Output:
        Downloads GFS data files.
    Author and history:
        John W. Cooney           2023-06-14
    '''  
    class SimpleResponse:
        status_code = 404
        content = b""

    now = datetime.utcnow()
    date1 = datetime.strptime(date_str1, "%Y-%m-%d %H:%M:%S")                                                                                          #Convert start date string to datetime structure
    date2 = datetime.strptime(date_str2, "%Y-%m-%d %H:%M:%S")                                                                                          #Convert start date string to datetime structure
    if (now-date1) <= timedelta(days =1):                                                                                                              #If start date is within 1 day of the current date then latest GFS file may not be available
        date1 = date1 - timedelta(seconds = 2*GFS_ANALYSIS_DT)                                                                                         #If latest GFS file is not available then download previous dates GFS files
    date1 = gfs_nearest_time(date1, GFS_ANALYSIS_DT, ROUND = 'down')                                                                                   #Extract minimum time of GFS files required to download to encompass data date range
    date2 = gfs_nearest_time(date2, GFS_ANALYSIS_DT, ROUND = 'up')                                                                                     #Extract maximum time of GFS files required to download to encompass data date range
  #  aws s3 ls --no-sign-request s3://noaa-gfs-warmstart-pds/noaa-gfs-bdp-pds/gfs.2023030700/gfs.t00z.pgrb2.0p25.anl
    
    #New change? Check number of files to download for this.
    files = []
    total_seconds = int((date2 - date1).total_seconds())
    for offset in range(0, total_seconds + 1, GFS_ANALYSIS_DT):
        file_time = date1 + timedelta(seconds=offset)
        file_str = (
            file_time.strftime("%Y") + '/' + 
            file_time.strftime("%Y%m%d") + '/gfs.0p25.' + 
            file_time.strftime("%Y%m%d%H") + '.f000.grib2'
        )
        files.append(file_str)

#    files = [(date1 + timedelta(hours = d)).strftime("%Y") + '/' + (date1 + timedelta(hours = d)).strftime("%Y%m%d") + '/gfs.0p25.' + (date1 + timedelta(hours = d)).strftime("%Y%m%d%H") + '.f000.grib2' for d in range(int((date2-date1).days*24 + ceil((date2-date1).seconds/3600.0))+1) if (3600*(date1 + timedelta(hours = d)).hour + 60*(date1 + timedelta(hours = d)).minute + (date1 + timedelta(hours = d)).second) % GFS_ANALYSIS_DT == 0]

   # download the data file(s)
    if verbose:
        print('Downloading GFS files to extract tropopause temperatures.')
        print(files)
        print(os.path.realpath(outroot))
    for file in files:
        idx = file.rfind("/")
        if (idx > 0):
            ofile = file[idx+1:]
        else:
            ofile = file
        
        outdir = os.path.join(outroot, (os.sep).join(re.split('/', os.path.dirname(file))))
        if not os.path.exists(os.path.join(outdir, ofile)):
            #Old?            
#            response = requests.get("https://data.rda.ucar.edu/ds084.1/" + file)
            #New path?
            try:
                response = requests.get("https://data.rda.ucar.edu/d084001/" + file)
            except:
                response = SimpleResponse()
            
            if verbose:
                print(response)
                print(response.status_code)
                print(file)
            if response.status_code == 200:
                os.makedirs(outdir, exist_ok = True)
                with open(os.path.join(outdir, ofile), "wb") as f:
                    f.write(response.content)
            else:
                time.sleep(1)
                #Old?            
#                response = requests.get("https://data.rda.ucar.edu/ds084.1/" + file)
                #New path?
                try:
                    response = requests.get("https://data.rda.ucar.edu/d084001/" + file)
                except:
                    response = SimpleResponse()
                
                if verbose:
                    print('Trying again')
                    print(response)
                    print(response.status_code)
                    print(file)
                if response.status_code == 200:
                    os.makedirs(outdir, exist_ok = True)
                    with open(os.path.join(outdir, ofile), "wb") as f:
                        f.write(response.content)
                else:                
#                    dates = [(date1 + timedelta(hours = d)).strftime("%Y-%m-%d %H:%M:%S") for d in range(int((date2-date1).days*24 + ceil((date2-date1).seconds/3600.0))+1) if (3600*(date1 + timedelta(hours = d)).hour + 60*(date1 + timedelta(hours = d)).minute + (date1 + timedelta(hours = d)).second) % GFS_ANALYSIS_DT == 0]
                    print('No GFS files found in rda.ucar.edu. Trying Google Cloud.')
                    dd = os.path.basename(file).split('.')[2]
                    dates = [datetime.strptime(dd, "%Y%m%d%H").strftime("%Y-%m-%d %H:%M:%S")]
                    for dd in dates:
                        download_gfs_analysis_files_from_gcloud(dd, GFS_ANALYSIS_DT = GFS_ANALYSIS_DT, outroot = outroot, write_gcs = write_gcs, del_local = del_local, real_time = real_time, verbose = verbose)

def download_gfs_analysis_files_from_gcloud(date_str, 
                                            GFS_ANALYSIS_DT = 21600,
                                            infile          = None,
                                            outroot         = os.path.join('..', '..', '..', 'gfs-data'),
                                            gfs_bucket_name = 'global-forecast-system', 
                                            write_gcs       = False,
                                            del_local       = True,
                                            real_time       = True,
                                            verbose         = True):
    '''
    This is a function to to download the nearest GFS file to the date specified from Google Cloud. Used for real-time model runs
    Args:
        date_str : Analysis date. String containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to start downloading 
    Keywords:
        GFS_ANALYSIS_DT : FLOAT keyword specifying the time between GFS files in sec. 
                          DEFAULT = 21600 -> seconds between GFS files
        infile          : Name of GFS data file to download.
                          DEFAULT = None -> use the date strings.
        write_gcs       : IF keyword set (True), write the output files to the google cloud in addition to local storage.
                          DEFAULT = False.
        del_local       : IF keyword set (True) AND run_gcs = True, delete local copy of output file.
                          DEFAULT = True.
        outroot         : STRING output directory path for GFS data storage
                          DEFAULT = os.path.join('..', '..', '..', 'gfs-data')
        c_bucket_name   : STRING specifying the name of the gcp bucket to write combined IR, VIS, GLM files to As well as 3 modalities figures.
                          run_gcs needs to be True in order for this to matter.
                          DEFAULT = 'ir-vis-sandwhich'
        c_bucket_name   : STRING specifying the GFS gcp bucket to write combined IR, VIS, GLM files to As well as 3 modalities figures.
                          run_gcs needs to be True in order for this to matter.
                          DEFAULT = 'ir-vis-sandwhich'
        verbose         : BOOL keyword to specify whether or not to print verbose informational messages.
                        DEFAULT = True which implies to print verbose informational messages
    Output:
        Downloads GFS data files.
    Author and history:
        John W. Cooney           2024-02-20
    '''  
    if infile is None:
        date1 = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")                                                                                       #Convert start date string to datetime structure
        date1 = gfs_nearest_time(date1, GFS_ANALYSIS_DT, ROUND = 'down')                                                                               #Extract minimum time of GFS files required to download to encompass data date range
        fpath = 'gfs.' + date1.strftime('%Y%m%d') + '/' + date1.strftime('%H') + '/atmos/gfs.t' + date1.strftime('%H') + 'z.pgrb2.0p25.f000'
        file  = list_gcs(gfs_bucket_name, os.path.dirname(fpath), [os.path.basename(fpath)])
        try:
            file.remove(fpath + '.idx')
        except:
            nothing = 0
        if len(file) == 0:
            date1 = gfs_nearest_time(date1-timedelta(seconds = 30), GFS_ANALYSIS_DT, ROUND = 'down')                                                   #Extract minimum time of GFS files required to download to encompass data date range
            fpath = 'gfs.' + date1.strftime('%Y%m%d') + '/' + date1.strftime('%H') + '/atmos/gfs.t' + date1.strftime('%H') + 'z.pgrb2.0p25.f000'
            file  = list_gcs(gfs_bucket_name, os.path.dirname(fpath), [os.path.basename(fpath)])
            try:
                file.remove(fpath + '.idx')
            except:
                nothing = 0
    
            if len(file) == 0:
                date1 = gfs_nearest_time(date1-timedelta(seconds = 30), GFS_ANALYSIS_DT, ROUND = 'down')                                               #Extract minimum time of GFS files required to download to encompass data date range
                fpath = 'gfs.' + date1.strftime('%Y%m%d') + '/' + date1.strftime('%H') + '/atmos/gfs.t' + date1.strftime('%H') + 'z.pgrb2.0p25.f000'
                file  = list_gcs(gfs_bucket_name, os.path.dirname(fpath), [os.path.basename(fpath)])
                try:
                    file.remove(fpath + '.idx')
                except:
                    nothing = 0
    
        if verbose:
            print(file)    
        if len(file) == 1:
            outdir = os.path.join(os.path.realpath(outroot), date1.strftime('%Y'), date1.strftime('%Y%m%d'))
            if os.path.exists(os.path.join(outdir, os.path.basename(file[0]))) or os.path.exists(os.path.join(outdir, 'gfs.0p25.' + date1.strftime('%Y%m%d%H') + '.f000.grib2')):
                exist = True
                if verbose:
                    print('GFS file already exists so do not need to download')
            else:        
                exist = False
                os.makedirs(outdir, exist_ok = True)
                download_ncdf_gcs(gfs_bucket_name, file[0], outdir)
        else:        
            print('No GFS file found within 12 hours of date specified')
            print(date_str)
            print(fpath)
            print(gfs_bucket_name)
            print('File may not be available on Google Cloud??')
            exit()    
    else:
        print('Trying google cloud redownload')
        exist      = False
        fb         = os.path.basename(infile)
        yyyymmddhh = re.split(r'\.', fb)[2]
        date1      = datetime.strptime(yyyymmddhh, '%Y%m%d%H')
        yyyymmdd   = os.path.basename(os.path.dirname(infile))
        yyyy       = os.path.basename(os.path.dirname(os.path.dirname(infile)))
        fpath      = 'gfs.' + yyyymmdd + '/' + yyyymmddhh[-2:] + '/atmos/gfs.t' + yyyymmddhh[-2:] + 'z.pgrb2.0p25.f000'
        file       = list_gcs(gfs_bucket_name, os.path.dirname(fpath), [os.path.basename(fpath)])
        try:
            file.remove(fpath + '.idx')
        except:
            nothing = 0
        if len(file) == 1:
            outdir = os.path.join(os.path.realpath(outroot), yyyy, yyyymmdd)
            os.makedirs(outdir, exist_ok = True)
            download_ncdf_gcs(gfs_bucket_name, file[0], outdir)
        else:
            exist = True

    if not exist:
        original_path = os.path.join(outdir, os.path.basename(file[0]))
        new_filename = 'gfs.0p25.' + date1.strftime('%Y%m%d%H') + '.f000.grib2'
        new_path = os.path.join(outdir, new_filename)
        os.rename(original_path, new_path)
        if verbose:
            print(f"Renamed downloaded file {original_path} to {new_filename}")            

def gfs_nearest_time(date, dt, ROUND = 'round'):
    '''
    Name:
        GFS_NEAREST_TIME
    Purpose:
        Calculate the date and time of the GFS analysis or forecast file nearest to 'date'.
    Calling sequence:
        gfs_date = GFS_NEAREST_TIME(date, dt)
    Inputs:
        date  : Date and time {datetime}.
        dt    : Interval between analysis or forecast files (sec)
    Keywords:
        ROUND : Keyword to control rounding of date.
                'round' : find nearest analysis or forecast time
                'down'  : find previous analysis or forecast time
                'up'    : find subsequent analysis or forecast time
    Output:
        datetime date
    System variables:
        None.
    Author and history:
        John W. Cooney           2023-06-14
    
    '''
    
    f = np.double(3600*date.hour + 60*date.minute + date.second)/dt                                                                                    #Calculate where in day that the date resides within GFS analysis times
    if ROUND.lower() == 'round':
        k = round(f)
    elif ROUND.lower() == 'down':
        k = floor(f)
    elif ROUND.lower() == 'up':
        k = ceil(f)
    else:
        print('Round must be set to "round", "up", or "down"')
        print(ROUND)
        exit()
    
    return(datetime(date.year, date.month, date.day) + timedelta(seconds = k*dt))                                                                      #Return requested date and time

def circle_mean_with_offset(arr, radius, offset):
    def circle_mean(x):
        return(np.mean(x) + offset * np.std(x))
    
    return(generic_filter(arr, circle_mean, size=2*radius+1, mode='wrap'))

def lanczos_kernel(size, cutoff):
    """
    Generate a Lanczos kernel.

    Parameters:
        - size: The size of the kernel (should be an odd number).
        - cutoff: The cutoff frequency, which determines the width of the central lobe.

    Returns:
        - A 1D numpy array representing the Lanczos kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size should be an odd number")

    half_size = size // 2
    kernel = np.zeros((size, size), dtype=float)
    for x in range(-half_size, half_size + 1):
            for y in range(-half_size, half_size + 1):
                distance = np.sqrt(x**2 + y**2)
                if distance == 0:
                    kernel[x + half_size, y + half_size] = 1.0
                elif distance <= cutoff:
                    kernel[x + half_size, y + half_size] = (
                        np.sinc(x / cutoff) * np.sinc(y / cutoff)
                    )

#     for x in range(-half_size, half_size + 1):
#         if x == 0:
#             kernel[x + half_size] = 1.0
#         elif abs(x) <= cutoff:
#             kernel[x + half_size] = np.sinc(x / cutoff) * np.sinc(x / half_size)

    return(kernel / kernel.sum())

def weighted_cold_bias(arr, radius, weight=0.1):
    def func(x):
        sorted_vals = np.sort(x)
        idx = max(0, int(weight * len(sorted_vals)))  # Avoid index errors
        return(sorted_vals[idx])  # Pick a colder value
    return(generic_filter(arr, func, size=2*radius+1, mode='wrap'))


#Function to extract start datetime from file
def extract_combined_nc_datetime(filename):
    basename = os.path.basename(filename)
    match0   = re.search(r'_s(\d{4})(\d{3})(\d{6})', basename)  #sYYYYDDDHHMMSS
    if match0:
        year = int(match0.group(1))
        doy  = int(match0.group(2))
        hms  = match0.group(3)
        try:
            return(datetime.strptime(f"{year} {doy} {hms}", '%Y %j %H%M%S'))
        except:
            return(None)
    return(None)
        
def main():
    run_write_severe_storm_post_processing()
    
if __name__ == '__main__':
    main()