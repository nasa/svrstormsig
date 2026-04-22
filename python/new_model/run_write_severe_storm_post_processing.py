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
#     inroot           : STRING specifying input root directory to the vis_ir_glm_json csv files
#                        DEFAULT = '../../../goes-data/aacp_results/ir_vis_glm/plume_day_model/2021-04-28/'
#     outroot          : STRING specifying output root directory to save the 512x512 numpy files
#                        DEFAULT = None -> same as input root
#     use_local        : IF keyword set (True), use local files. If False, use files located on the google cloud. 
#                        DEFAULT = False -> use files on google cloud server. 
#     write_gcs        : IF keyword set (True), write the output files to the google cloud in addition to local storage.
#                        DEFAULT = True.
#     c_bucket_name    : STRING specifying the name of the gcp bucket to read model results data files from. use_local needs to be False in order for this to matter.
#                        DEFAULT = 'aacp-results'
#     del_local        : IF keyword set (True) AND write_gcs == True, delete local copy of output file.
#                        DEFAULT = True.
#     object_type      : STRING specifying the object (OT, AACP, or BOTH) that you detected and want to postprocess.
#                        DEFAULT = 'OT'
#     mod_type         : STRING specifying the model type used to make the detections that you want to postprocess. (ex. multiresunet, unet, attentionunet)
#                        DEFAULT = 'multiresunet'
#     sector           : STRING specifying the domain sector to use to create maps (= 'conus' or 'full' or 'm1' or 'm2'). DEFAULT = None -> use meso_sector
#     rewrite          : BOOL keyword to specify whether or not to rewrite the ID numbers and IR BTD, etc.
#                        DEFAULT = True -> rewrite the post-processed data
#     percent_omit     : FLOAT keyword specifying the percentage of cold and warm pixels to remove from anvil mean brightness temperature calculation
#                        DEFAULT = 20 -> 20% of the warmest and coldest anvil pixel temperatures are removed in order to prevent the contribution of noise to 
#                        the OT IR-anvil BTD calculation.
#                        NOTE: percent_omit must be between 0 and 100.
#     aacp_ot_max_dist : LONG keyword to specify distance of an AACP in pixels to an OT in order for it to be combined into an AACP object.
#                        ONLY applies to AACPs.
#                        DEFAULT = 40 pixels -> 20 km.
#     verbose          : BOOL keyword to specify whether or not to print verbose informational messages.
#                        DEFAULT = True which implies to print verbose informational messages
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
from scipy.ndimage import label, generate_binary_structure, convolve, uniform_filter, generic_filter, distance_transform_edt, binary_dilation, find_objects
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
from gridrad.rdr_sat_utils_jwc import gfs_interpolate_tropT_to_goes_grid, convert_longitude

def run_write_severe_storm_post_processing(inroot          = os.path.join('..', '..', '..', 'goes-data', 'combined_nc_dir', '20230516'),
                                           infile          = None,
                                           date1           = None,
                                           date2           = None,
                                           gfs_root        = os.path.join('..', '..', '..', 'gfs-data'),
                                           outroot         = None, 
                                           GFS_ANALYSIS_DT = 21600,
                                           use_local       = True, write_gcs = False, del_local = True,
                                           c_bucket_name   = 'ir-vis-sandwhich',
                                           object_type     = 'OT',
                                           mod_type        = 'multiresunet',
                                           sector          = 'M2',
                                           pthresh         = None, 
                                           rewrite         = True,
                                           percent_omit    = 20, 
                                           aacp_ot_max_dist= 10,
                                           channel_str     = None,
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
        inroot           : STRING specifying input root directory to the combined netCDF files
                           DEFAULT = os.path.join('..', '..', '..', 'goes-data', 'combined_nc_dir', '20230516')
        infile           : STRING specifying input root directory to the vis_ir_glm_json csv files
                           DEFAULT = None -> loop over all files in the input root directory
        date1            : STRING specifying start date string to begin post-processing the data (YYYY-MM-DD hh:mm:ss)
                           DEFAULT = None  -> loop over all files in the input root directory
        date2            : STRING specifying start date string to complete post-processing the data (YYYY-MM-DD hh:mm:ss)
                           DEFAULT = None  -> loop over all files in the input root directory
        gfs_root         : STRING output directory path for GFS data storage
                           DEFAULT = os.path.join('..', '..', '..', 'gfs-data')
        outroot          : STRING specifying output root directory to save the netCDF output files
                           DEFAULT = None -> same as input root
        GFS_ANALYSIS_DT  : FLOAT keyword specifying the time between GFS files in sec. 
                           DEFAULT = 21600 -> seconds between GFS files
        use_local        : IF keyword set (True), use local files. If False, use files located on the google cloud. 
                           DEFAULT = False -> use files on google cloud server. 
        write_gcs        : IF keyword set (True), write the output files to the google cloud in addition to local storage.
                           DEFAULT = True.
        c_bucket_name    : STRING specifying the name of the gcp bucket to read model results data files from. use_local needs to be False in order for this to matter.
                           DEFAULT = 'aacp-results'
        del_local        : IF keyword set (True) AND write_gcs == True, delete local copy of output file.
                           DEFAULT = True.
        object_type      : STRING specifying the object (OT, AACP, or BOTH) that you detected and want to postprocess.
                           DEFAULT = 'OT'
        mod_type         : STRING specifying the model type used to make the detections that you want to postprocess. (ex. multiresunet, unet, attentionunet)
                           DEFAULT = 'multiresunet'
        sector           : STRING specifying the domain sector to use to create maps (= 'conus' or 'full' or 'm1' or 'm2'). DEFAULT = None -> use meso_sector
        pthresh          : FLOAT keyword to specify the optimal likelihood value to threshold the outputted model likelihoods in order for object to be OT or AACP
                           DEFAULT = None -> use the default value in file
                           NOTE: day_night optimal runs may require different pthresh scores that yield the best results. It is suggested to keep this as None for those
                           jobs.
        rewrite          : BOOL keyword to specify whether or not to rewrite the ID numbers and IR BTD, etc.
                           DEFAULT = True -> rewrite the post-processed data
        percent_omit     : FLOAT keyword specifying the percentage of cold and warm pixels to remove from anvil mean brightness temperature calculation
                           DEFAULT = 20 -> 20% of the warmest and coldest anvil pixel temperatures are removed in order to prevent the contribution of noise to 
                           the OT IR-anvil BTD calculation.
                           NOTE: percent_omit must be between 0 and 100.
        aacp_ot_max_dist : LONG keyword to specify distance of an AACP in pixels to an OT in order for it to be combined into an AACP object.
                           ONLY applies to AACPs.
                           DEFAULT = 40 pixels -> 20 km.
        verbose          : BOOL keyword to specify whether or not to print verbose informational messages.
                          DEFAULT = True which implies to print verbose informational messages
    Author and history:
        John W. Cooney           2023-05-31.
                                 2025-03-27. (Adapted to add aacp_dist_thresh to ensure AACPs within some distance of an OT detection)
    '''
    # Handle multiple object types efficiently
    if isinstance(object_type, str):
        if object_type.upper() == 'BOTH':
            object_types = ['OT', 'AACP']
        else:
            obj_clean = object_type.replace(" ", "").lower()
            if obj_clean in ['plume', 'warmplume', 'aacp']:
                object_types = ['AACP']
            elif obj_clean in ['updraft', 'overshootingtop', 'ot']:
                object_types = ['OT']
            else:
                object_types = [object_type.upper()]
    elif isinstance(object_type, list):
        object_types = [obj.upper() for obj in object_type]
    
    if mod_type.lower() not in ['multiresunet', 'unet', 'attentionunet']:
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
        try: sector = 'M' + sector[-1]
        except: sector = 'meso'
    elif sector[0].lower() == 'c': sector = 'C'
    elif sector[0].lower() == 'f': sector = 'F'
    elif sector[0].lower() in ['q', 'h', 't']: sector = sector.upper()
    else:
        print('Satellite sector specified is not available. Enter M1, M2, F, or C. Please try again.')
        exit()
    
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
            
    native_ir  = False    
    ot_var_key = None

    x_sat0  = []
    y_sat0  = []
    gfsf    = 'kadshbfavbjfv'
    lanczos_kernel0 = lanczos_kernel(7, 3)                                                                                                             #Define the Lanczos kernel (kernel size = 7x7; cutoff frequency = 3)
    s       = generate_binary_structure(2,2)
    counter = 0
    for l in range(len(nc_files)):
        if len(nc_files) <= 2: time.sleep(2)
        nc_file = nc_files[l]
        if use_local != True:
            download_ncdf_gcs(c_bucket_name, nc_file, inroot)                                                                                          #Download netCDF file from GCP

        #Read combined netCDF file
        with xr.open_dataset(os.path.join(inroot, os.path.basename(nc_file))).load() as f:    
            dt64 = pd.to_datetime(f['time'].values[0])                                                                                                 #Read in the date of satellite scan and convert it to Timestamp
            date = datetime(dt64.year, dt64.month, dt64.day, dt64.hour, dt64.minute, dt64.second)                                                      #Convert Timestamp to datetime structure
            x_sat = f['longitude'].values                                                                                                              #Extract latitude and longitude satellite domain
            y_sat = f['latitude'].values                                                                                                               #Extract latitude and longitude satellite domain
            var_keys = list(f.keys())                                                                                                                  #Get list of variables contained within the netCDF file
            try: xyres = f.spatial_resolution
            except:
                if 'visible_reflectance' in var_keys: xyres = '0.5km at nadir'
                else:    
                    xyres = '2km at nadir'
                    native_ir = True
 
            bt  = f['ir_brightness_temperature']
            bt0 = bt.values[0, :, :]
            sat = f['imager_projection'].attrs['satellite_name']
            
            # Pre-load OT distance mask for AACP filtering
            ot_res_for_dist     = None
            ot_pthresh_for_dist = None
            dist_to_nearest_ot  = None
            if 'AACP' in object_types and aacp_ot_max_dist is not None:
                if counter == 0 or ot_var_key is None:
                    ot_var_keys = [k for k in var_keys if k.endswith('_ot') or (k.endswith('_ot2')) or ('_ot_' not in k and k.split('_')[-1] == 'ot' and 'id' not in k and 'anvilmean' not in k and 'bt' not in k)]
                    if len(ot_var_keys) == 0:
                        ot_var_keys = [k for k in var_keys if '_ot' in k and 'id_number' not in k and 'anvilmean' not in k and 'brightness_temperature' not in k]
                    
                    if len(ot_var_keys) > 0:
                        ot_var_key = 'tropdiff_ot' if 'tropdiff_ot' in ot_var_keys else ot_var_keys[0]

                if ot_var_key is not None and ot_var_key in f:
                    try:
                        ot_res_for_dist = f[ot_var_key].values[0, :, :]
                        ot_pthresh_for_dist = float(f[ot_var_key].attrs.get('optimal_thresh', 0.35))
                        ot_binary = (ot_res_for_dist >= max(0.05, ot_pthresh_for_dist - 0.05)).astype(bool)
                        if np.sum(ot_binary) > 0:
                            dist_to_nearest_ot = distance_transform_edt(~ot_binary)
                    except Exception as e:
                        print(f'WARNING [aacp_ot_max_dist]: {e}')
            
            # Cache the objects for processing
            nc_dct_full = {k: f[k] for k in var_keys}

        # Calculate GFS tropopause
        tropT2 = bt0 * np.nan
        files_used = []
        if len(gfs_files) > 0 and 'tropopause_temperature' not in var_keys:
            near_date = gfs_nearest_time(date, GFS_ANALYSIS_DT, ROUND = 'round')
            nd_str    = near_date.strftime("%Y%m%d%H")
            no_f = 0
            try: 
              gfs_file = gfs_files[gdates.index(nd_str)]
            except: 
              no_f = 1

            if no_f == 1:
                for cc in range(1, 8):
                    near_date = gfs_nearest_time(date-timedelta(seconds = cc*GFS_ANALYSIS_DT), GFS_ANALYSIS_DT, ROUND = 'round')
                    nd_str    = near_date.strftime("%Y%m%d%H")
                    no_f = 0
                    try: 
                      gfs_file = gfs_files[gdates.index(nd_str)]
                    except: 
                      no_f = 1
                    if no_f == 0: break    
            
            if no_f == 0:
                x_sat_conv = convert_longitude(x_sat, to_model = True)
                if gfs_file != gfsf:
                    grbs  = pygrib.open(gfs_file)
                    grb   = grbs.select(name='Temperature', typeOfLevel='tropopause')[0]
                    tropT = np.copy(grb.values)
                    lats, lons = grb.latlons()
                    tropT = np.flip(tropT, axis = 0)
                    tropT = circle_mean_with_offset(tropT, 20, -0.6)        
                    lats  = np.flip(lats[:, 0])
                    lons  = convert_longitude(lons[0, :], to_model = True)

                    N_wrap = 10
                    lons_extended = np.concatenate((lons[-N_wrap:] - 360, lons, lons[:N_wrap] + 360))
                    tropT_extended = np.concatenate((tropT[:, -N_wrap:], tropT, tropT[:, :N_wrap]), axis=1)
                    tropT_extended = convolve(tropT_extended, lanczos_kernel0)
                    
                    tropT2 = interpn((lats, lons_extended), tropT_extended, (y_sat, x_sat_conv), method='linear', bounds_error=False, fill_value=np.nan)
                    grbs.close()
                    gfsf   = gfs_file
                    x_sat0 = x_sat_conv
                    y_sat0 = y_sat
                else:
                    if np.nanmax(np.abs(x_sat_conv - x_sat0)) != 0 or np.nanmax(np.abs(y_sat - y_sat0)) != 0:
                        tropT2 = interpn((lats, lons_extended), tropT_extended, (y_sat, x_sat_conv), method='linear', bounds_error=False, fill_value=np.nan)
                files_used = [os.path.realpath(gfs_file)]

        #Process objects in memory
        output_payload = {}
        xyres0 = np.float64(re.split('km', xyres)[0])
        anv_p  = int((15.0/xyres0)*2.0)
        
        for current_obj in object_types:
            da = []
            keys0 = []
            for var_key in var_keys:
                if f'_{current_obj.lower()}' in var_key and 'id' not in var_key and 'anvilmean' not in var_key and 'bt' not in var_key:
                    if channel_str is not None and channel_str not in var_key:
                        continue
                    try: 
                        da.append(nc_dct_full[var_key].attrs['date_added'])
                    except: 
                        da.append('2025-08-08 12:00:00')
                    keys0.append(var_key)
            
            if any('2_' in fff for fff in keys0): keys0 = [fff for fff in keys0 if "2_" in fff]
            if len(keys0) > 1: keys0 = [keys0[np.argmax([datetime.strptime(ddd, "%Y-%m-%d %H:%M:%S") for ddd in da])]]
            
            if len(keys0) <= 0:
                if verbose: 
                    print(f'{current_obj} likelihood missing in {nc_file}. Skipping.')
                    print(var_keys)
                continue

            var_key = keys0[0]
            if var_key + '_id_number' not in var_keys or rewrite:
                data = nc_dct_full[var_key]
                pthresh0 = pthresh if pthresh is not None else data.attrs.get('optimal_thresh', 0.5)
                
                if data.attrs.get('model_type', '').lower() == mod_type.lower():
                    res = data.values[0, :, :].copy()
                    res[res < 0.05] = 0
                    res_safe = np.where(np.isfinite(res), res, 0)
                    labeled_array, num_updrafts = label(res > 0, structure=s)
                    res_full = res.copy()
                    if num_updrafts > 0:
                        max_per_label = np.zeros(num_updrafts + 1, dtype=res.dtype)
                        np.maximum.at(max_per_label, labeled_array.ravel(), res_safe.ravel())
                        label_max_map = max_per_label[labeled_array]
                        above_thresh = (labeled_array > 0) & (label_max_map >= pthresh0)
                        
                        if current_obj.lower() == 'aacp':
                            res10_map = 0.10 * label_max_map
                            res_full = res.copy()
                            res_full[above_thresh & (res_full < res10_map)] = 0
                            res_full[above_thresh & (res_full >= res10_map)] = label_max_map[above_thresh & (res_full >= res10_map)]
                            
                            frac_max_map = np.clip(0.8 - 1.75 * (label_max_map - 0.3), 0.1, 0.8)
                            res_dyn_map = frac_max_map * label_max_map
                            res[above_thresh & (res < res_dyn_map)] = 0
                            res[above_thresh & (res >= res_dyn_map)] = label_max_map[above_thresh & (res >= res_dyn_map)]
                        else:
                            # Standard fixed fraction for Overshooting Tops
                            res50_map = 0.50 * label_max_map
                            res[above_thresh & (res < res50_map)] = 0
                            res[above_thresh & (res >= res50_map)] = label_max_map[above_thresh & (res >= res50_map)]
                            res_full = res.copy()                 
                    
                    if current_obj == 'AACP':
                        # Use the 10% map to create labeled_array0 so morphological checks hit the true cloud edge
                        res00 = res_full.copy()
                        res00[res00 < pthresh0] = 0
                        convective_proxy = bt0 <= 230
                        labeled_cells0, _ = label(convective_proxy, structure=s)
                        ll = np.copy(labeled_cells0).astype(float)
                        
                        labeled_array0, num_updrafts0 = label(res00 > 0, structure=s)
                        slices0 = find_objects(labeled_array0)
                        
                        for u0, slc in enumerate(slices0):
                            if slc is None: continue
                            
                            # Pad the bounding box so the dilation has room to grow
                            pad  = 4 if native_ir else 15
                            pad2 = 4 if native_ir else 6
                            y_start = max(0, slc[0].start - pad)
                            y_stop  = min(bt0.shape[0], slc[0].stop + pad)
                            x_start = max(0, slc[1].start - pad)
                            x_stop  = min(bt0.shape[1], slc[1].stop + pad)
                            padded_slc = (slice(y_start, y_stop), slice(x_start, x_stop))
                            
                            y_start2 = max(0, slc[0].start - pad2)
                            y_stop2  = min(bt0.shape[0], slc[0].stop + pad2)
                            x_start2 = max(0, slc[1].start - pad2)
                            x_stop2  = min(bt0.shape[1], slc[1].stop + pad2)
                            padded_slc2 = (slice(y_start2, y_stop2), slice(x_start2, x_stop2))
                            
                            local_label = labeled_array0[padded_slc]
                            local_inds0 = (local_label == u0 + 1)
                            
                            if np.sum(local_inds0) > 0:
                                local_ll = ll[padded_slc]
                                idxx = local_ll[local_inds0][0]
                                
                                local_bt = bt0[ll == idxx]
                                bt_thresh = np.percentile(local_bt, 5)
                                
                                cell_area_pixels = len(local_bt)
                              
                                dist_to_convection = distance_transform_edt(~(bt0 < bt_thresh))
                                
                                local_dist = dist_to_convection[padded_slc]
                                min_dist = np.min(local_dist[local_inds0])
                                
                                # Fast Local Dilation
                                local_dilated = binary_dilation(local_inds0, iterations=pad)
                                dist_thresh = 3 if native_ir else 10
                                local_ring = np.logical_and(local_dilated, ~local_inds0)
                                
                                local_dilated2 = binary_dilation(local_inds0, iterations=pad2)
                                local_ring2 = np.logical_and(local_dilated2, ~local_inds0)
                               
                                local_bt0 = bt0[padded_slc]
                                max_ring_bt = np.nanmax(local_bt0[local_ring]) if np.any(local_ring) else 0
                                max_plume_bt = np.nanmax(local_bt0[local_inds0])
                                
                                max_ring_bt2 = np.nanmax(local_bt0[local_ring2]) if np.any(local_ring2) else 0
                               
                                ring_thresh = bt_thresh + 20
                                if ring_thresh > 230:
                                    ring_thresh = 230
                                
                                # Calculate the bounding box dimensions
                                box_height = slc[0].stop - slc[0].start
                                box_width  = slc[1].stop - slc[1].start
                                bbox_area  = box_height * box_width
                                
                                # Plume Area is just the number of pixels inside the object
                                plume_area_pixels = np.sum(local_inds0)
                                
                                plume_to_cell_ratio = plume_area_pixels / cell_area_pixels if cell_area_pixels > 0 else 1.0

                                # Dynamic Plume Area is the size of the final saved object
                                dynamic_plume_pixels = np.sum((local_label == u0 + 1) & (res[padded_slc] >= pthresh0))
                                
                                # Calculate Fill Ratio
                                fill_ratio = plume_area_pixels / bbox_area
                                
                                local_res_original   = res_safe[padded_slc]
                                plume_original_probs = local_res_original[local_inds0]
                                max_prob             = np.nanmax(plume_original_probs)
                                
                                core_pixel_count = np.sum(plume_original_probs >= pthresh0)
                                
                                min_ot_dist = np.inf
                                if aacp_ot_max_dist is not None and dist_to_nearest_ot is not None:
                                    local_ot_dist = dist_to_nearest_ot[padded_slc]
                                    min_ot_dist = np.min(local_ot_dist[local_inds0])
                                
                                is_textbook_plume = (max_prob > 0.8) and ((core_pixel_count >= 5)) and (fill_ratio > 0.50) and (min_ot_dist <= 5) and (plume_to_cell_ratio < 0.25) and (max_ring_bt2 <= ring_thresh)
                                if not is_textbook_plume:
                                    if (min_dist > dist_thresh) or (max_ring_bt > ring_thresh) or (dynamic_plume_pixels < 16) or ((fill_ratio < 0.25) or (fill_ratio < 0.35 and max_prob < 0.5)) or (core_pixel_count <= 5) or (plume_to_cell_ratio >= 0.75):                    
                                        res[labeled_array0 == (u0 + 1)] = 0.05
                                           
                                    # Distance to OT filter
                                    elif aacp_ot_max_dist is not None and dist_to_nearest_ot is not None:
                                        if min_ot_dist > aacp_ot_max_dist:
                                            res[labeled_array0 == (u0 + 1)] = 0.05

                    # Optimized OT BTD processing
                    maxc = np.nanmax(res) if np.any(np.isfinite(res)) else 0
                    if maxc < pthresh0:
                        btd = res * np.nan
                        ot_id = np.full_like(res, 0).astype('uint16')
                    else:
                        labeled_array2, num_updrafts2 = label(res >= pthresh0, structure=s)
                        if current_obj == 'AACP':
                            for u2 in range(1, num_updrafts2 + 1):
                                final_obj_mask = (labeled_array2 == u2)
                                if np.any(final_obj_mask):
                                    raw_max_prob = np.nanmax(res_safe[final_obj_mask])
                                    # If the raw object never authentically passed the threshold, kill it
                                    if raw_max_prob < pthresh0:
                                        labeled_array2[final_obj_mask] = 0
                                        res[final_obj_mask] = 0

                        anvil_mask = (labeled_array2 <= 0)
                        btd = res * np.nan
                        ot_id = np.full_like(res, 0).astype('uint16') if current_obj == 'OT' else labeled_array2.astype('uint16')

                        slices2 = find_objects(labeled_array2)
                        ot_num = 1
                        
                        for u, slc2 in enumerate(slices2):
                            if slc2 is None: continue
                            
                            local_label2 = labeled_array2[slc2]
                            local_inds2 = (local_label2 == u + 1)
                            
                            if np.sum(local_inds2) > 0:
                                local_bt0 = bt0[slc2]
                                plume_bts = local_bt0[local_inds2]
                                
                                # Find min BT index locally
                                min_ind_flat = np.nanargmin(plume_bts)
                                min_bt0 = plume_bts[min_ind_flat]
                                
                                if current_obj == 'OT':
                                    # Convert local min BT index to 2D local coordinates, then to global coordinates
                                    local_coords = np.argwhere(local_inds2)
                                    min_local_y, min_local_x = local_coords[min_ind_flat]
                                    global_y = min_local_y + slc2[0].start
                                    global_x = min_local_x + slc2[1].start
                                    
                                    # Bounding box for anvil using global coordinates
                                    i_pix = [max(0, global_y - int(anv_p/2)), min(bt0.shape[0], global_y + ceil(anv_p/2))]
                                    j_pix = [max(0, global_x - int(anv_p/2)), min(bt0.shape[1], global_x + ceil(anv_p/2))]
                                    
                                    anvil_area_mask = anvil_mask[i_pix[0]:i_pix[1], j_pix[0]:j_pix[1]]
                                    anvil_bts = anvil_area_mask * bt0[i_pix[0]:i_pix[1], j_pix[0]:j_pix[1]]
                                    anvil_bts[anvil_bts <= 0]  = np.nan
                                    anvil_bts[anvil_bts > 235] = np.nan
                                    anvil_bts = np.sort(anvil_bts[np.isfinite(anvil_bts)].flatten())
                                    
                                    if len(anvil_bts) > 0:
                                        tenper = int(np.floor((percent_omit/100.0)*len(anvil_bts)))
                                        anvil_bts = anvil_bts[0+tenper:len(anvil_bts)-tenper]
                                        anvil_bt_mean = np.nanmean(anvil_bts)
                                        val = min_bt0 - anvil_bt_mean
                                        if val <= -1:
                                            if not ((min_bt0 >= 225) and val >= -6):
                                                # SAFE ASSIGNMENT: Use the global array mask
                                                global_inds2 = (labeled_array2 == u + 1)
                                                ot_id[global_inds2] = ot_num
                                                btd[global_inds2]   = val
                                                ot_num += 1

                    output_payload[current_obj] = {
                        'id': ot_id,
                        'btd': btd,
                        'attrs': data.attrs,
                        'pthresh': pthresh0
                    }

        #Append combined netCDF
        if output_payload or len(files_used) > 0:
            append_combined_ncdf_with_model_post_processing(
                nc_file, output_payload, tropT2, anv_p, files_used, 
                rewrite, percent_omit, write_gcs, del_local, outroot, c_bucket_name, verbose
            )
        
        counter += 1

def append_combined_ncdf_with_model_post_processing(nc_file, output_payload, tropT, resolution, files_used, rewrite=True, percent_omit=20, write_gcs=True, del_local=True, outroot=None, c_bucket_name='ir-vis-sandwhich', verbose=True):
    with Dataset(nc_file, 'a', format="NETCDF4") as f:
        vnames = list(f.variables.keys())
        
        #Write Tropopause Temperature 
        if 'tropopause_temperature' not in vnames and np.sum(np.isfinite(tropT)) > 0:
            var_mod3 = f.createVariable('tropopause_temperature', 'f4', ('time', 'Y', 'X',), zlib=True, least_significant_digit=2, complevel=7)
            var_mod3.set_auto_maskandscale(False)
            var_mod3.long_name = "Temperature of the Tropopause Retrieved from GFS Interpolated and Smoothed onto Satellite Grid"
            var_mod3.standard_name = "GFS Tropopause"
            var_mod3.contributing_files = files_used
            var_mod3.missing_value = np.nan
            var_mod3.units = 'K'
            var_mod3.coordinates = 'longitude latitude time'
            var_mod3[0, :, :] = tropT
    
        #Write Object Outputs
        for obj, p_data in output_payload.items():
            mod_attrs = p_data['attrs']
            object_id = p_data['id']
            btd = p_data['btd']
            pthresh = p_data['pthresh']
            
            chan0 = re.split('_', mod_attrs['standard_name'])[0].upper()
            vname = chan0.replace('+', '_').lower() + '_' + re.split('_', mod_attrs['standard_name'])[1].lower()
            mod_description = chan0 + ' ' + re.split('_', mod_attrs['standard_name'])[1].upper()
            
            id_name = vname + '_id_number'
            btd_name = vname + '_anvilmean_brightness_temperature_difference'
            
            if id_name not in vnames or rewrite:
                if id_name in vnames:
                    f[id_name][0, :, :] = object_id
                    f[id_name].likelihood_threshold = pthresh
                    if 'ot' in mod_attrs['standard_name'].lower():
                        f[btd_name][0, :, :] = btd
                        f[btd_name].likelihood_threshold = pthresh
                else:
                    var_mod = f.createVariable(id_name, 'u2', ('time', 'Y', 'X',), zlib=True, complevel=7)
                    var_mod.set_auto_maskandscale(False)
                    var_mod.long_name = mod_description + ' Identification Number'
                    var_mod.standard_name = mod_description + ' ID Number'
                    var_mod.model_type = mod_attrs['model_type']
                    var_mod.likelihood_threshold = pthresh
                    var_mod.missing_value = 0
                    var_mod.units = 'dimensionless'
                    var_mod.coordinates = 'longitude latitude time'
                    var_mod[0, :, :] = object_id
    
                    if 'ot' in mod_attrs['standard_name'].lower():
                        var_mod2 = f.createVariable(btd_name, 'f4', ('time', 'Y', 'X',), zlib=True, least_significant_digit=2, complevel=7)
                        var_mod2.set_auto_maskandscale(False)
                        var_mod2.long_name = chan0 + " Overshooting Top Minus Anvil Brightness Temperature Difference"
                        var_mod2.standard_name = chan0 + " OT - Anvil BTD"
                        var_mod2.model_type = mod_attrs['model_type']
                        var_mod2.valid_range = [-50.0, 0.0]
                        var_mod2.missing_value = np.nan
                        var_mod2.likelihood_threshold = pthresh
                        var_mod2.units = 'K'
                        var_mod2.coordinates = 'longitude latitude time'
                        var_mod2[0, :, :] = btd
    
    if verbose:
        print('Post-processing model output netCDF file name = ' + nc_file)
    return()

def download_gfs_analysis_files(date_str1, date_str2,
                                GFS_ANALYSIS_DT = 21600,
                                outroot         = os.path.join('..', '..', '..', 'gfs-data'),
                                c_bucket_name   = 'ir-vis-sandwhich',
                                write_gcs       = False,
                                del_local       = True,
                                real_time       = False,
                                verbose         = True):
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
    size  = 2 * radius + 1
    arr_f = arr.astype(np.float64)
    
    mean_arr = uniform_filter(arr_f, size=size, mode='wrap')                                                                                           #Fast vectorized mean via uniform_filter
    if offset == 0: return(mean_arr)
    
    mean_sq  = uniform_filter(np.square(arr_f), size=size, mode='wrap')
    std_arr  = np.sqrt(np.maximum(mean_sq - np.square(mean_arr), 0))                                                                                   #Std = sqrt(E[x^2] - E[x]^2), also vectorizable
    return(mean_arr + offset * std_arr)

def lanczos_kernel(size, cutoff):
    if size % 2 == 0:
        raise ValueError("Kernel size should be an odd number")

    half_size = size // 2
    kernel = np.zeros((size, size), dtype=float)
    for x in range(-half_size, half_size + 1):
        for y in range(-half_size, half_size + 1):
            distance = np.sqrt(np.square(x) + np.square(y))
            if distance == 0:
                kernel[x + half_size, y + half_size] = 1.0
            elif distance <= cutoff:
                kernel[x + half_size, y + half_size] = (np.sinc(x / cutoff) * np.sinc(y / cutoff))

    return(kernel / kernel.sum())

def weighted_cold_bias(arr, radius, weight=0.1):
    def func(x):
        sorted_vals = np.sort(x)
        idx = max(0, int(weight * len(sorted_vals)))  # Avoid index errors
        return(sorted_vals[idx])  # Pick a colder value
    return(generic_filter(arr, func, size=2*radius+1, mode='wrap'))

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