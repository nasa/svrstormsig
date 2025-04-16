#+
# Name:
#     run_write_gfs_trop_temp_to_combined_ncdf.py
# Purpose:
#     This is a script to read in the combined netCDF files and write the GFS tropopause temperature to it. 
# Calling sequence:
#     import run_write_gfs_trop_temp_to_combined_ncdf
#     run_write_gfs_trop_temp_to_combined_ncdf.run_write_gfs_trop_temp_to_combined_ncdf()
# Input:
#     None.
# Functions:
#     run_write_trop_temp_to_combined_ncdf : Main function to write the GFS tropopause temperature data to combined netCDF file
# Output:
#     Combined netCDF files containing tropopause temperature data
# Keywords:
#     inroot        : STRING specifying input root directory to the vis_ir_glm_json csv files
#                     DEFAULT = '../../../goes-data/aacp_results/ir_vis_glm/plume_day_model/2021-04-28/'
#     outroot       : STRING specifying output root directory to save the 512x512 numpy files
#                     DEFAULT = None -> same as input root
#     use_local     : IF keyword set (True), use local files. If False, use files located on the google cloud. 
#                     DEFAULT = False -> use files on google cloud server. 
#     write_gcs     : IF keyword set (True), write the output files to the google cloud in addition to local storage.
#                     DEFAULT = True.
#     c_bucket_name : STRING specifying the name of the gcp bucket to read combined netCDF. use_local needs to be False in order for this to matter.
#                     DEFAULT = 'aacp-results'
#     m_bucket_name : STRING specifying the name of the gcp bucket to read GFS tropopause data files from. use_local needs to be False in order for this to matter.
#                     DEFAULT = 'misc-data0'
#     del_local     : IF keyword set (True) AND write_gcs == True, delete local copy of output file.
#                     DEFAULT = True.
#     rewrite       : BOOL keyword to specify whether or not to rewrite the ID numbers and IR BTD, etc.
#                     DEFAULT = True -> rewrite the post-processed data
#     real_time     : IF keyword set (True), only rewrite the latest combined netCDF file
#                     DEFAULT = False. 
#     verbose       : BOOL keyword to specify whether or not to print verbose informational messages.
#                     DEFAULT = True which implies to print verbose informational messages
# Author and history:
#     John W. Cooney           2023-11-03.
#
#-

#### Environment Setup ###
# Package imports
import numpy as np
import glob
import os
import re
from math import ceil, floor
from scipy.ndimage import label, generate_binary_structure
import xarray as xr
from datetime import datetime, timedelta
import requests
import time
from netCDF4 import Dataset
import pandas as pd
import pygrib
import sys
from scipy.ndimage import convolve, generic_filter
from scipy.interpolate import interpn
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(3, os.path.dirname(os.getcwd()))
from new_model.gcs_processing import write_to_gcs, list_gcs, download_ncdf_gcs
from new_model.run_write_severe_storm_post_processing import download_gfs_analysis_files_from_gcloud
#from glm_gridder.dataScaler import DataScaler
from gridrad.rdr_sat_utils_jwc import gfs_interpolate_tropT_to_goes_grid, convert_longitude
#from visualize_results.gfs_contour_trop_temperature import gfs_contour_trop_temperature
def run_write_gfs_trop_temp_to_combined_ncdf(inroot          = os.path.join('..', '..', '..', 'goes-data', 'combined_nc_dir', '20230516'),
                                             gfs_root        = os.path.join('..', '..', '..', 'gfs-data'),
                                             outroot         = None, 
                                             GFS_ANALYSIS_DT = 21600,
                                             use_local       = True, write_gcs = False, del_local = True,
                                             c_bucket_name   = 'ir-vis-sandwhich',
                                             m_bucket_name   = 'misc-data0',
                                             sector          = 'F', 
                                             rewrite         = True,
                                             real_time       = False,
                                             verbose         = True):
    '''    
    Name:
        run_write_gfs_trop_temp_to_combined_ncdf.py
    Purpose:
        This is a script to read in the combined netCDF files and write the GFS tropopause temperature to it. 
    Calling sequence:
        import run_write_gfs_trop_temp_to_combined_ncdf
        run_write_gfs_trop_temp_to_combined_ncdf.run_write_gfs_trop_temp_to_combined_ncdf()
    Input:
        None.
    Functions:
        run_write_trop_temp_to_combined_ncdf : Main function to write the GFS tropopause temperature data to combined netCDF file
    Output:
        Combined netCDF files containing tropopause temperature data
    Keywords:
        inroot        : STRING specifying input root directory to the vis_ir_glm_json csv files
                        DEFAULT = '../../../goes-data/aacp_results/ir_vis_glm/plume_day_model/2021-04-28/'
        outroot       : STRING specifying output root directory to save the 512x512 numpy files
                        DEFAULT = None -> same as input root
        use_local     : IF keyword set (True), use local files. If False, use files located on the google cloud. 
                        DEFAULT = False -> use files on google cloud server. 
        write_gcs     : IF keyword set (True), write the output files to the google cloud in addition to local storage.
                        DEFAULT = True.
        c_bucket_name : STRING specifying the name of the gcp bucket to read combined netCDF. use_local needs to be False in order for this to matter.
                        DEFAULT = 'aacp-results'
        m_bucket_name : STRING specifying the name of the gcp bucket to read GFS tropopause data files from. use_local needs to be False in order for this to matter.
                        DEFAULT = 'misc-data0'
        del_local     : IF keyword set (True) AND write_gcs == True, delete local copy of output file.
                        DEFAULT = True.
        real_time     : IF keyword set (True), only rewrite the latest combined netCDF file
                        DEFAULT = False. 
        rewrite       : BOOL keyword to specify whether or not to rewrite the ID numbers and IR BTD, etc.
                        DEFAULT = True -> rewrite the post-processed data
                        DEFAULT = True which implies to print verbose informational messages
    Author and history:
        John W. Cooney           2023-11-03.
    '''
    
    inroot   = os.path.realpath(inroot)                                                                                                                #Create link to real path so compatible with Mac
    gfs_root = os.path.realpath(gfs_root)                                                                                                              #Create link to real path so compatible with Mac
    d_str    = os.path.basename(inroot)                                                                                                                #Extract date string from the input root directory
    yyyy     = d_str[0:4]
    mm       = d_str[4:6]
    if outroot == None:
        outroot = inroot                                                                                                                               #Create link to real path so compatible with Mac
    else: 
        outroot = os.path.realpath(outroot)                                                                                                            #Create link to real path so compatible with Mac
        os.makedirs(outroot, exist_ok = True)                                                                                                          #Create output directory file path if does not already exist
    
    if real_time:        
        day_earlier = datetime.strptime(d_str, "%Y%m%d")-timedelta(seconds = 30)
        if use_local:
#             gfs_files = sorted(glob.glob(os.path.join(gfs_root, str(day_earlier.year), day_earlier.strftime("%Y%m%d"), '*pgrb2*'), recursive = True))  #Search for all GFS model files
#             gfs_files.extend(sorted(glob.glob(os.path.join(gfs_root, yyyy, d_str, '*pgrb2*'), recursive = True)))                                      #Search for all GFS model files
            gfs_files = sorted(glob.glob(os.path.join(gfs_root, str(day_earlier.year), day_earlier.strftime("%Y%m%d"), '*.grib2'), recursive = True))  #Search for all GFS model files
            gfs_files.extend(sorted(glob.glob(os.path.join(gfs_root, yyyy, d_str, '*.grib2'), recursive = True)))                                      #Search for all GFS model files
        else:
#             gfs_files = sorted(list_gcs(m_bucket_name, os.path.join('gfs-data', str(day_earlier.year), day_earlier.strftime("%Y%m%d"), ['.pgrb2'], delimiter = '*')))                                                     #Extract names of all of the GOES visible data files from the google cloud    
#             gfs_files.extend(sorted(list_gcs(m_bucket_name, os.path.join('gfs-data', yyyy, d_str), ['.pgrb2'], delimiter = '*')))                      #Extract names of all of the GOES visible data files from the google cloud    
            gfs_files = sorted(list_gcs(m_bucket_name, os.path.join('gfs-data', str(day_earlier.year), day_earlier.strftime("%Y%m%d"), ['.grib2'], delimiter = '*')))                                                     #Extract names of all of the GOES visible data files from the google cloud    
            gfs_files.extend(sorted(list_gcs(m_bucket_name, os.path.join('gfs-data', yyyy, d_str), ['.grib2'], delimiter = '*')))                      #Extract names of all of the GOES visible data files from the google cloud    
    else:
        if use_local:
            gfs_files = sorted(glob.glob(os.path.join(gfs_root, '**', '**', '*.grib2'), recursive = True))                                             #Search for all GFS model files
            if len(gfs_files) <= 0:
                gfs_files = sorted(glob.glob(os.path.join(gfs_root, '**', '**', '*.f000*'), recursive = True))                                         #Search for all GFS model files
                gfs_files = [fff for fff in gfs_files if not fff.endswith('.idx')]
        else:
            gfs_files = sorted(list_gcs(m_bucket_name, 'gfs-data', ['.grib2'], delimiter = '*/*'))                                                     #Extract names of all of the GOES visible data files from the google cloud   
   
    if len(gfs_files) <= 0:
        print('No GFS files found in specified directory???')
        print(gfs_files)
        print(d_str)
        if use_local:
            print(gfs_root)
        else:
            print(m_bucket_name)
        exit()

    if use_local:    
        nc_files = sorted(glob.glob(os.path.join(inroot, '*_' + sector.upper() + '_*.nc')))                                                            #Find netCDF files to postprocess
    else:
        if 'native_ir' in inroot:
            pref = os.path.join(os.path.basename(os.path.dirname(os.path.dirname(inroot))), os.path.basename(os.path.dirname(inroot)), os.path.basename(inroot))#Find netCDF files in GCP bucket to postprocess        
        else:
            pref = os.path.join(os.path.basename(os.path.dirname(inroot)), os.path.basename(inroot))                                                   #Find netCDF files in GCP bucket to postprocess
        nc_files = sorted(list_gcs(c_bucket_name, pref, ['_' + sector.upper() + '_', '.nc']))                                                          #Extract names of all of the GOES visible data files from the google cloud   
    
    if len(nc_files) <= 0:
        print('Combined netCDF files')
        print(nc_files)        
        print(sector)
        print('No files found in specified directory???')
        if use_local:
            print(inroot)
        else:
            print(c_bucket_name)
            print(pref)
        exit()

    if real_time:
        gfs_files = [gfs_files[-1]]
        nc_files  = [nc_files[-1]]
        if gfs_files[0].endswith('.grib2'):
            gdates    = [(re.split('\.', os.path.basename(gfs_files[0])))[2]]                                                                          #Extract date of the nearest GFS file
        else:
            gdates    = [os.path.basename(os.path.dirname(gfs_files[0])) + (re.split('\.|t|z', os.path.basename(g)))[2] for g in gfs_files]            #Extract date of the nearest GFS file
        gdate     = datetime.strptime(gdates[0], "%Y%m%d%H")
    else:
#        gdates = [(re.split('\.', os.path.basename(g)))[2] for g in gfs_files]                                                                        #Extract date of all GFS files
        gfs_files = sorted(list(set(gfs_files)))
        gdates = []
        for gg in gfs_files:
            if gg.endswith('.grib2'):
                gdates.append((re.split('\.', os.path.basename(gg)))[2])
            else:
                gdates.append(os.path.basename(os.path.dirname(gg)) + (re.split('\.|t|z', os.path.basename(gg)))[2])                                   #Extract date of all GFS files
#                gdates = [os.path.basename(os.path.dirname(g)) + (re.split('\.|t|z', os.path.basename(g)))[2] for g in gfs_files]                          #Extract date of all GFS files

    lanczos_kernel0 = lanczos_kernel(7, 3)                                                                                                             #Define the Lanczos kernel (kernel size = 7x7; cutoff frequency = 3)
    lats    = []                                                                                                                                       #Initialize array to store latitudes and longitudes from GFS. If I have read it, then I do not need to read it again
    lons    = []                                                                                                                                       #Initialize array to store latitudes and longitudes from GFS. If I have read it, then I do not need to read it again
    gfile00 = nc_files[0]                                                                                                                              #Initialize array to the latest GFS file that was read so do not need to read again
    gfile11 = nc_files[0]                                                                                                                              #Initialize array to the latest GFS file that was read so do not need to read again
    s       = generate_binary_structure(2,2)
    counter = 0
    for l in range(len(nc_files)):
        nc_file = nc_files[l]
        if use_local == False:
            os.makedirs(inroot, exist_ok = True)
            download_ncdf_gcs(c_bucket_name, nc_file, inroot)                                                                                          #Download netCDF file from GCP

        check  = 0
        nc_dct = {}
        con_files = []                                                                                                                                 #List to store contributing files to make the tropopause temperature. If contributing files are the same as before then no need to rewrite to netCDF
        with xr.open_dataset(os.path.join(inroot, os.path.basename(nc_file))).load() as f:    
            dt64  = pd.to_datetime(f['time'].values[0])                                                                                                #Read in the date of satellite scan and convert it to Timestamp
            date  = datetime(dt64.year, dt64.month, dt64.day, dt64.hour, dt64.minute, dt64.second)                                                     #Convert Timestamp to datetime structure
            x_sat = f['longitude'].values                                                                                                              #Extract latitude and longitude satellite domain
            y_sat = f['latitude'].values                                                                                                               #Extract latitude and longitude satellite domain
            var_keys = list(f.keys())                                                                                                                  #Get list of variables contained within the netCDF file
            for var_key in var_keys:
                if 'tropopause_temperature' in var_key:                                                                                                #Check to see if tropopause temperature was already written to combined netCDF file
                    check = 1                                                                                                                          #Set check to 1
                    con_files = f['tropopause_temperature'].attrs['contributing_files']
#             try:
#                 xyres = f.spatial_resolution
#             except:
#                 if 'visible_reflectance' in var_keys:
#                     xyres = '0.5km at nadir'
#                 else:    
#                     xyres = '2km at nadir'
#             if check == 0 or rewrite == True:
#                 bt = f['ir_brightness_temperature']
        
#        xyres0 = np.float64(re.split('km', xyres)[0])
        
        if check == 0 or rewrite == True:
#             bt0   = bt.values[0, :, :]                 
            x_sat = convert_longitude(x_sat, to_model = True)                                                                                          #Convert longitudes to be 0-360 degrees
            if real_time == False:
                near_date0 = gfs_nearest_time(date, GFS_ANALYSIS_DT, ROUND = 'down')                                                                   #Find nearest date to satellite scan
                near_date1 = gfs_nearest_time(date, GFS_ANALYSIS_DT, ROUND = 'up')                                                                     #Find nearest date to satellite scan
                nd_str0    = near_date0.strftime("%Y%m%d%H")                                                                                           #Extract nearest date to satellite scan as a string
                nd_str1    = near_date1.strftime("%Y%m%d%H")                                                                                           #Extract nearest date to satellite scan as a string
                no_f0 = 0
                try:
                    gfs_file0 = gfs_files[gdates.index(nd_str0)]                                                                                       #Find file that matches date string
                except:
                    no_f0 = 1
                    gfs_file0 = ''         
    
                no_f1 = 0
                try:
                    gfs_file1 = gfs_files[gdates.index(nd_str1)]                                                                                       #Find file that matches date string
                except:
                    no_f1 = 1
                    gfs_file1 = ''          
                
                if no_f0 == 1 and no_f1 == 1:
                    gdate     = datetime.strptime(gdates[-1], "%Y%m%d%H")
                    if abs(gdate - date) <= timedelta(seconds = 12*60*60):
                        gfs_file0 = gfs_files[-1]
                        gfs_file1 = gfs_files[-1]
                        no_f0 = 0
                        no_f1 = 0
                        near_date0 = gdate
                        near_date1 = gdate
                        nd_str0    = near_date0.strftime("%Y%m%d%H")                                                                                   #Extract nearest date to satellite scan as a string
                        nd_str1    = near_date1.strftime("%Y%m%d%H")                                                                                   #Extract nearest date to satellite scan as a string
                    else:    
                        print('No file found for previous or future GFS file date??')
                        print(nd_str0)
                        exit()
                
                if no_f0 == 1:
                    print('No file found for previous GFS file date??')
                    print(nd_str0)
#                    exit()
    
                if no_f1 == 1:
                    print('No file found for future GFS file date??')
                    print(nd_str1)
#                    exit()
            else:
                if abs(gdate - date) <= timedelta(seconds = 12*60*60):
                    gfs_file0 = gfs_files[0]    
                    gfs_file1 = gfs_files[0]    
                    no_f0     = 0
                else:
                    print('No GFS file within 12 hours of current date???')
                    print(date)
                    print(gfs_files[0])
                    print('Exiting program')
                    exit()
                       
            gfile0 = gfs_file0
            gfile1 = gfs_file1
       
            nowrite = False
            if check and not real_time and len(con_files) > 0:
                if len(con_files) == 1:
                     if os.path.basename(gfs_file0) == os.path.basename(con_files[0]) or os.path.basename(gfs_file1) == os.path.basename(con_files[0]):
#                         print(gfs_file0)
#                         print(gfs_file1)
#                         print(con_files)
                        nowrite = True
                else:
                    if os.path.basename(gfs_file0) == os.path.basename(con_files[0]) and os.path.basename(gfs_file1) == os.path.basename(con_files[1]):
#                         print(gfs_file0)
#                         print(gfs_file1)
#                         print(con_files)
                        nowrite = True
                     
            if not nowrite:
                if use_local == False:
                    gfile0 = os.path.join(gfs_root, os.path.basename(os.path.dirname(os.path.dirname(gfs_file0))), os.path.basename(os.path.dirname(gfs_file0)), os.path.basename(gfs_file0))
                    gfile1 = os.path.join(gfs_root, os.path.basename(os.path.dirname(os.path.dirname(gfs_file1))), os.path.basename(os.path.dirname(gfs_file1)), os.path.basename(gfs_file1))
                    if os.path.isfile(gfile0) == False:
                        os.makedirs(os.path.dirname(gfile0), exist_ok = True)
                        download_ncdf_gcs(m_bucket_name, gfs_file0, os.path.dirname(gfile0))                                                           #Download netCDF file from GCP
    
                    if os.path.isfile(gfile1) == False:
                        os.makedirs(os.path.dirname(gfile1), exist_ok = True)
                        download_ncdf_gcs(m_bucket_name, gfs_file1, os.path.dirname(gfile1))                                                           #Download netCDF file from GCP
    
                p0 = 0
                p1 = 0
                gfile0 = gfile0
                gfile1 = gfile1
                if gfile0 != '':
                    if gfile0 != gfile00:
                        if gfile0 == gfile11:
                            tropT0 = np.copy(tropT1)
                        else:
                            grbs   = pygrib.open( gfile0 )                                                                                             #Open Grib file to read
                            try:
                                grb    = grbs.select(name='Temperature', typeOfLevel='tropopause')[0]                                                  #Extract tropopause temperatures from the GRIB file
                            except:
                                grbs.close()
                                download_gfs_analysis_files('', '', infile = gfile0, verbose = True)
                                grbs   = pygrib.open( gfile0 )                                                                                         #Open Grib file to read
                                try:
                                    grb    = grbs.select(name='Temperature', typeOfLevel='tropopause')[0]                                              #Extract tropopause temperatures from the GRIB file
                                except:
                                    print('Bad GFS file. Unable to find tropopause temperature within the file.')
                                    print(os.path.realpath(gfile0))
                                    print('Exiting')
                                    exit()
                            tropT0 = np.copy(grb.values)                                                                                               #Copy tropopause temperatures from backward date into array
                            tropT0 = np.flip(tropT0, axis = 0)                                                                                         #Put latitudes in ascending order in order to be interpolated onto satellite grid
                    #        tropT0 = weighted_cold_bias(tropT0, 5, weight=0.3)
                            tropT0 = circle_mean_with_offset(tropT0, 20, -0.6)        
                            p0     = 1
                        gfile00 = gfile0
                    
                    if len(lats) <= 0:
                        lats, lons = grb.latlons()                                                                                                     #Extract GFS latitudes and longitudes
                        lats  = np.flip(lats[:, 0])                                                                                                    #Put latitudes in ascending order in order to be interpolated onto satellite grid
                        lons  = convert_longitude(lons[0, :], to_model = True)
#                         lons  = (np.asarray(lons)).tolist()
#                         lats  = (np.asarray(lats)).tolist()
                    
                if gfile1 != gfile11 and real_time == False:
                    if gfile1 != '':
                        grbs1   = pygrib.open( gfile1 )                                                                                                #Open Grib file to read
                        try:
                            grb1    = grbs1.select(name='Temperature', typeOfLevel='tropopause')[0]                                                    #Extract tropopause temperatures from the GRIB file
                        except:
                            grbs1.close()
                            download_gfs_analysis_files('', '', infile = gfile1, verbose = True)
                            grbs1   = pygrib.open( gfile1 )                                                                                            #Open Grib file to read
                            try:
                                grb1    = grbs1.select(name='Temperature', typeOfLevel='tropopause')[0]                                                 #Extract tropopause temperatures from the GRIB file
                            except:
                                print('Bad GFS file. Unable to find tropopause temperature within the file.')
                                print(os.path.realpath(gfile1))
                                print('Exiting')
                                exit()
                        
                        tropT1  = np.copy(grb1.values)                                                                                                 #Copy tropopause temperatures from forward date into array
                        tropT1  = np.flip(tropT1, axis = 0)                                                                                            #Put latitudes in ascending order in order to be interpolated onto satellite grid
             #           tropT1  = weighted_cold_bias(tropT1, 5, weight=0.3)
                        tropT1  = circle_mean_with_offset(tropT1, 20, -0.6)        
                        gfile11 = gfile1
                        p1      = 1
                    if gfile0 == '':
                        if len(lats) <= 0:
                            lats, lons = grb.latlons()                                                                                                 #Extract GFS latitudes and longitudes
                            lats  = np.flip(lats[:, 0])                                                                                                #Put latitudes in ascending order in order to be interpolated onto satellite grid
                            lons  = convert_longitude(lons[0, :], to_model = True)
                    
                    
                if real_time or gfile1 == '' or gfile0 == '':
                    if real_time or gfile1 == '':
                        tropT = tropT0
                        files_used = [os.path.realpath(gfile00)]
                    elif gfile0 == '':
                        tropT = tropT1
                        files_used = [os.path.realpath(gfile11)]
                else:    
                    files_used = [os.path.realpath(gfile00), os.path.realpath(gfile11)]
                    dt = near_date1 - near_date0                                                                                                       #Time difference between GFS data intervals
#                    if dt == 0:
                    if dt != timedelta(seconds = 0):
                        wt = (date - near_date0)/dt                                                                                                    #Calculate time weight
                        tropT = (1.0 - wt)*tropT0 + wt*tropT1                                                                                          #Interpolate data in time
                    else:
                        tropT = tropT0

                #Longitude wraparound padding
                N_wrap = 10                                                                                                                            #Number of grid points to duplicate on each side (tweak as needed)
                
                #Extend longitudes (pad with wraparound)
                lons_extended = np.concatenate((
                    lons[-N_wrap:] - 360,                                                                                                              #Left pad (negative side)
                    lons,
                    lons[:N_wrap] + 360                                                                                                                #Right pad (beyond 360)
                ))
                
                #Extend tropT to match padded longitudes
                tropT_extended = np.concatenate((
                    tropT[:, -N_wrap:],                                                                                                                #Left pad
                    tropT,
                    tropT[:, :N_wrap]                                                                                                                  #Right pad
                ), axis=1)
                
                #Apply Lanczos kernel
                tropT_fin = convolve(tropT_extended, lanczos_kernel0)
                goes_trop_data = interpn((lats, lons_extended), tropT_fin, (y_sat, x_sat), method = 'linear', bounds_error = False, fill_value = np.nan)
                if p0 == 1:
                    grbs.close()
                if p1 == 1:
                    grbs1.close()
    
                append_combined_ncdf_with_gfs_trop_temp(os.path.join(inroot, os.path.basename(nc_file)), goes_trop_data, files_used = files_used, rewrite = rewrite, verbose = verbose)
                if write_gcs:
                    if use_local:
                        if 'native_ir' in inroot:
                            pref = 'combined_nc_dir/native_ir/' + os.path.basename(os.path.dirname(nc_file))                    
                        else:
                            pref = 'combined_nc_dir/' + os.path.basename(os.path.dirname(nc_file))
                    else:
                        pref = os.path.dirname(nc_file)
    
                    write_to_gcs(c_bucket_name, pref, os.path.join(inroot, os.path.basename(nc_file)), del_local = del_local)    
        
def append_combined_ncdf_with_gfs_trop_temp(nc_file, tropT, files_used = [], rewrite = True, verbose = True):
  '''
  This is a function to append the combined netCDF files with the model post-processing data. 
  Args:
      nc_file : Filename and path to combined netCDF file that will be appended.
      tropT   : Numpy array with the GFS tropopause temperatures interpolated to each pixel in the satellite domain
  Keywords:
      files_used : List of grib files used to make the tropopause temperature calculation.
      rewrite    : BOOL keyword to specify whether or not to rewrite the ID numbers and IR BTD, etc.
                   DEFAULT = True -> rewrite the post-processed data
      verbose    : BOOL keyword to specify whether or not to print verbose informational messages.
                   DEFAULT = True which implies to print verbose informational messages
  Output:
      Appends the combined netCDF data files with post-processing data
  '''  

  with Dataset(nc_file, 'a', format="NETCDF4") as f:                                                                                                   #Open combined netCDF file to append with model results
    keys0 = f.variables.keys()
    lat   = np.copy(np.asarray(f.variables['latitude'][:,:]))                                                                                          #Read latitude from combined netCDF file to make sure it is the same shape as the model results
    lon   = np.copy(np.asarray(f.variables['longitude'][:,:]))                                                                                         #Read longitude from combined netCDF file to make sure it is the same shape as the model results
    if tropT.shape != lat.shape or tropT.shape != lon.shape:
      print('Model results file does not match file latitude or longitude shape!!!!')
      print(lon.shape)
      print(lat.shape)
      print(object_id.shape)
      print(btd.shape)
      exit()
    
    if 'tropopause_temperature' in keys0 and rewrite == True:
        f['tropopause_temperature'][0, :, :] = tropT
    else:
      #Declare variables
      if np.sum(np.isfinite(tropT)) > 0:
        var_mod3 = f.createVariable('tropopause_temperature', 'f4', ('time', 'Y', 'X',), zlib = True, least_significant_digit = 2, complevel = 7)#, fill_value = Scaler._FillValue)
        var_mod3.set_auto_maskandscale( False )
        var_mod3.long_name          = "Temperature of the Tropopause Retrieved from GFS Interpolated and Smoothed onto Satellite Grid"
        var_mod3.standard_name      = "GFS Tropopause"
        var_mod3.contributing_files = files_used
        var_mod3.missing_value      = np.nan
        var_mod3.units              = 'K'
        var_mod3.coordinates        = 'longitude latitude time'
        var_mod3[0, :, :]           = np.copy(tropT)
  
    if verbose == True:
      print('GFS tropopause temperature data written to = ' + nc_file)
    
  return()

def download_gfs_analysis_files(date_str1, date_str2,
                                infile          = None,
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
        infile          : Name of GFS data file to download.
                          DEFAULT = None -> use the date strings.
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
    if infile is None:
        date1 = datetime.strptime(date_str1, "%Y-%m-%d %H:%M:%S")                                                                                      #Convert start date string to datetime structure
        date2 = datetime.strptime(date_str2, "%Y-%m-%d %H:%M:%S")                                                                                      #Convert start date string to datetime structure
        date1 = gfs_nearest_time(date1, GFS_ANALYSIS_DT, ROUND = 'down')                                                                               #Extract minimum time of GFS files required to download to encompass data date range
        date2 = gfs_nearest_time(date2, GFS_ANALYSIS_DT, ROUND = 'up')                                                                                 #Extract maximum time of GFS files required to download to encompass data date range
  #      aws s3 ls --no-sign-request s3://noaa-gfs-warmstart-pds/noaa-gfs-bdp-pds/gfs.2023030700/gfs.t00z.pgrb2.0p25.anl
        files = [(date1 + timedelta(hours = d)).strftime("%Y") + '/' + (date1 + timedelta(hours = d)).strftime("%Y%m%d") + '/gfs.0p25.' + (date1 + timedelta(hours = d)).strftime("%Y%m%d%H") + '.f000.grib2' for d in range(int((date2-date1).days*24 + ceil((date2-date1).seconds/3600.0))+1) if (3600*(date1 + timedelta(hours = d)).hour + 60*(date1 + timedelta(hours = d)).minute + (date1 + timedelta(hours = d)).second) % GFS_ANALYSIS_DT == 0]
        files = [f for f in files if not f.endswith('.idx')]
   #     download the data file(s)
        if verbose == True:
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
            if os.path.exists(os.path.join(outdir, ofile)) == False:
                #Old?                        
#                response = requests.get("https://data.rda.ucar.edu/ds084.1/" + file)
                #New path?
                response = requests.get("https://data.rda.ucar.edu/d084001/" + file)
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
#                    response = requests.get("https://data.rda.ucar.edu/ds084.1/" + file)
                    #New path?
                    response = requests.get("https://data.rda.ucar.edu/d084001/" + file)
                    if verbose:
                        print(response)
                        print(response.status_code)
                        print(file)
                    if response.status_code == 200:
                        os.makedirs(outdir, exist_ok = True)
                        with open(os.path.join(outdir, ofile), "wb") as f:
                            f.write(response.content)
                    else:                
                        print('No GFS files found to calculate tropopause temperature')
    else:
        print('Trying to redownload bad GFS data file')
        fb       = os.path.basename(infile)
        yyyymmdd = os.path.basename(os.path.dirname(infile))
        yyyy     = os.path.basename(os.path.dirname(os.path.dirname(infile)))
        file     = yyyy + '/' + yyyymmdd + '/' + fb
        outdir   = os.path.realpath(os.path.join(outroot, yyyy, yyyymmdd))
        response = requests.get("https://data.rda.ucar.edu/d084001/" + file)
        if verbose:
            print(response)
            print(response.status_code)
            print(file)
            print(outdir)
        
        if response.status_code == 200:
            os.makedirs(outdir, exist_ok = True)
            with open(os.path.join(outdir, fb), "wb") as f:
                f.write(response.content)
        else:
            time.sleep(1)
            #Old?                        
#            response = requests.get("https://data.rda.ucar.edu/ds084.1/" + file)
            #New path?
            response = requests.get("https://data.rda.ucar.edu/d084001/" + file)
            if verbose:
                print(response)
                print(response.status_code)
                print(file)
            if response.status_code == 200:
                os.makedirs(outdir, exist_ok = True)
                with open(os.path.join(outdir, fb), "wb") as f:
                    f.write(response.content)
            else:                
               download_gfs_analysis_files_from_gcloud('',  infile = infile)
               print('No GFS files found to calculate tropopause temperature')
        
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
    
def main():
    run_write_gfs_trop_temp_to_combined_ncdf()
    
if __name__ == '__main__':
    main()