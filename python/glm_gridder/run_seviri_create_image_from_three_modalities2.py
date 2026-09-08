#+
# Name:
#     run_seviri_create_image_from_three_modalities2.py
# Purpose:
#     This is a script to run (by importing) programs that combine SEVIRI IR and 
#     VIS data into a netCDF file, and create a layered image that gets put into labelme software.
#     THIS CODE RUNS THE JOBS IN PARALLEL!!!!! Adapted for SEVIRI single-file EPCT outputs.
# Calling sequence:
#     import run_seviri_create_image_from_three_modalities2
#     run_seviri_create_image_from_three_modalities2.run_seviri_create_image_from_three_modalities2()
# Input:
#     None.
# Functions:
#     seviri_combine_ir_glm_vis_parallel                 : Combines IR and VIS data and puts into netCDF file in parallel.
#     img_from_three_modaltities                         : Creates 3-layered image that is input into labelme software
#     seviri_create_image_from_three_modalities_parallel : Function that allows for multiprocessing.
# Output:
#     Creates combined VIS and IR netCDF file and image file that gets put into labelme software 
#     to identify overshoot plumes.
# Keywords:
#     inroot               : STRING path to SEVIRI EPCT data directories
#                            DEFAULT = '../../../seviri-data/20260606/'
#     layered_dir          : STRING output directory path for netCDF file containing combined IR and VIS data
#                            DEFAULT = '../../../seviri-data/combined_nc_dir/'
#     img_out_dir          : Output directory for the IR and VIS data plot that is input into labelme software.
#                            DEFAULT = '../../../seviri-data/layered_img_dir/'
#     no_plot              : IF keyword set (True), do not plot the IR/VIS data. DEFAULT = False
#     no_write_vis         : IF keyword set, do not write the VIS data to the combined modality netCDF file.
#                            DEFAULT = False.
#     no_write_irdiff      : IF keyword set, write the difference between 10.8 micron and 6.2 micron to file.  
#                            DEFAULT = True 
#     no_write_dirtyirdiff : IF keyword set, write the difference between 12.0 and 10.8 micron to file.  
#                            DEFAULT = True
#     xy_bounds            : ARRAY of floating point values giving the domain boundaries [x0, y0, x1, y1]
#                            DEFAULT = [] -> write combined netCDF data for full scan region.
#     domain_sector        : STRING specifying the domain sector to use to create maps (ex. 'full'). DEFAULT = 'F'
#     del_combined_nc      : IF keyword set (True), delete combined netCDF file AFTER modalities image is created.
#                            DEFAULT = False 
#     ir_min_value         : LONG keyword to specify minimum IR temperature value to clip IR data image (K). DEFAULT = 170 
#     ir_max_value         : LONG keyword to specify maximum IR temperature value to clip IR data image (K). DEFAULT = 230 
#     grid_data            : If keyword set (True), grid the data onto latitude and longitude domain. DEFAULT = False 
#     start_index          : Index which to start looping over the files at. DEFAULT = 0
#     date_range           : List containing start date and end date in YYYY-MM-DD hh:mm:ss format. DEFAULT = [] 
#     colorblind           : IF keyword set (True), use colorblind friendly color table. DEFAULT = True 
#     plt_img_name         : IF keyword set (True), plot image name on this figure. DEFAULT = False
#     plt_cbar             : IF keyword set (True), plot color bar on the figure. DEFAULT = False
#     subset               : [x0, y0, x1, y1] array to give subset of image domain desired to be plotted.
#     region               : State or sector of US to plot data around. DEFAULT = None -> plot for full domain.
#     run_gcs              : IF keyword set (True), read and write everything directly from the google cloud platform. DEFAULT = False
#     write_combined_gcs   : IF keyword set (True), write combined netCDF files to the google cloud platform. DEFAULT = True.
#     real_time            : IF keyword set (True), run the code in real time by only grabbing the most recent file. DEFAULT = False
#     del_local            : IF keyword set (True), delete local copies of files. DEFAULT = False
#     pthresh              : FLOAT keyword to specify probability threshold to use. DEFAULT = 0.0
#     plt_model            : Numpy array of model results (values 0-1) to plot on the image. DEFAULT = None
#     model                : Name of model being plotted (ex. IR+VIS). DEFAULT = None.
#     chk_day_night        : Structure containing day and night model data information. DEFAULT = {}             
#     out_bucket_name      : Google cloud storage bucket to write intermediate netCDF files. DEFAULT = 'ir-vis-sandwhich'
#     replot_img           : IF keyword set (True), plot the IR/VIS sandwich image again. DEFAULT = True
#     rewrite_nc           : IF keyword set (True), rewrite the combined netCDF file. DEFAULT = False
#     append_nc            : IF keyword set (True), append the combined netCDF file. DEFAULT = True
#     verbose              : BOOL keyword to specify whether or not to print verbose informational messages. DEFAULT = True
# Author and history:
#     John W. Cooney           2026-06-08. (Adapted for SEVIRI single-file format running in parallel).
#-

#### Environment Setup ###
import glob
import os
import re
from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta
import multiprocessing as mp
import sys 

sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.getcwd()))
from glm_gridder.seviri_combine_ir_glm_vis_parallel import seviri_combine_ir_glm_vis_parallel
from glm_gridder.img_from_three_modalities2 import img_from_three_modalities2
from new_model.gcs_processing import write_to_gcs

def run_seviri_create_image_from_three_modalities2(inroot             = os.path.join('..', '..', '..', 'seviri-data', '20260606'), 
                                                   layered_dir        = os.path.join('..', '..', '..', 'seviri-data', 'combined_nc_dir'), 
                                                   img_out_dir        = os.path.join('..', '..', '..', 'seviri-data', 'layered_img_dir'), 
                                                   no_plot            = False, no_plot_irdiff = True, no_plot_cirrus = True, no_plot_tropdiff = True, no_plot_snowice = True, no_plot_dirtyirdiff = True,
                                                   no_write_vis       = False, no_write_irdiff = True, no_write_dirtyirdiff = True, 
                                                   xy_bounds          = [], 
                                                   domain_sector      = 'F', 
                                                   del_combined_nc    = False, 
                                                   ir_min_value       = 170, ir_max_value  = 230, 
                                                   grid_data          = False, start_index = 0, 
                                                   date_range         = [], 
                                                   colorblind         = True, plt_img_name = False, 
                                                   plt_cbar           = False, subset      = None,  region    = None,   
                                                   run_gcs            = False, real_time   = False, del_local = False, pthresh = 0.0,
                                                   write_combined_gcs = True, 
                                                   plt_model          = None, model = None,
                                                   satellite          = 'Meteosat09', 
                                                   chk_day_night      = {}, 
                                                   out_bucket_name    = 'ir-vis-sandwhich',
                                                   replot_img         = True, rewrite_nc = False, append_nc = True, verbose = True):
    
    if run_gcs:
        print('SEVIRI data not setup to be run using Google Cloud Platform.')
        exit()
    
    if domain_sector == None:
        sector = 'F'                                                                                                                           #Set domain sector string. Used for output directory
    else:
        sector = domain_sector.upper()                                                                                                         #Set domain sector string. Used for output directory
    
    inroot = os.path.realpath(inroot)                                                                                                          #Create link to real path so compatible with Mac
    
    #Try to extract date in different formats
    try:
        f_dates = os.path.basename(inroot)
        datetime.strptime(f_dates, "%Y%m%d")                                                                                                   #Check if date string from input root is in valid format
    except ValueError:
        try:
            parts = os.path.normpath(inroot).split(os.sep)
            f_dates = f"{parts[-3]}{parts[-2]}{parts[-1]}"                                                                                     #Combine into "yyyymmdd"
        except (ValueError, IndexError):
            print("Error: Could not parse date from inroot.")
            print(inroot)
            exit()

    layered_dir = os.path.realpath(os.path.join(layered_dir, f_dates))                                                                         #Create link to real path so compatible with Mac
    img_out_dir = os.path.realpath(img_out_dir)                                                                                                #Create link to real path so compatible with Mac
    os.makedirs(layered_dir, exist_ok = True)                                                                                                  #Create output directory file path if does not already exist
    if not no_plot: os.makedirs(img_out_dir, exist_ok = True)                                                                                  #Create image output directory file path if does not already exist

    nc_files  = glob.glob(os.path.join(inroot, 'HRSEVIRI_*.nc'))                                                                               #Extract names of all of the SEVIRI EPCT NetCDF data files
    nat_files = glob.glob(os.path.join(inroot, '*.nat'))
    
    if len(nat_files) > 0:
        # Deduplicate files by timestamp, prioritizing .nc files over .nat files
        unique_files = {}
        for f in nc_files:
            unique_files[get_time_key(f, satellite)] = f
            
        for f in nat_files:
            t_key = get_time_key(f, satellite)
            # Only add the .nat file if we don't already have a .nc file for this timestamp
            if t_key not in unique_files:
                unique_files[t_key] = f

        # Sort the unique files chronologically
        raw_files = sorted(list(unique_files.values()), key=lambda f: get_time_key(f, satellite))
    else:
        raw_files = sorted(nc_files)

    if len(raw_files) == 0:
        print(f"No SEVIRI data found in {inroot}!")
        return ([], [], None)

    if real_time:
        raw_files = [raw_files[-1]]                                                                                                            #Download only the most recent files if running code in real time
    elif len(date_range) > 0:
        time0 = datetime.strptime(date_range[0], "%Y-%m-%d %H:%M:%S")                                                                          #Extract start date to find data within of the date range
        time1 = datetime.strptime(date_range[1], "%Y-%m-%d %H:%M:%S")                                                                          #Extract end date to find data within of the date range
        filtered_files = []
        for f in raw_files:
            dt_str = get_time_key(f, satellite)
            f_time = datetime.strptime(dt_str, "%Y%m%d%H%M%S")                                                                                 #Convert string to datetime
            if time0 <= f_time <= time1:
                filtered_files.append(f)                                                                                                       #Append to list if falls within date range boundaries
        raw_files = filtered_files                                                                                                             #Replace raw_files with filtered files list

    end_index = len(raw_files)                                                                                                                 #Set end index to the length of the files to process
    if start_index == -1: start_index = 0                                                                                                      #Ensure start index is 0 if passed as -1
    if end_index == 0: return ([], [], None)                                                                                                   #Exit cleanly if no files left to process

    pool = mp.Pool(3, maxtasksperchild=30)                                                                                                     #Set up multiprocessing threads
    results = [pool.apply_async(seviri_create_image_from_three_modalities_parallel, args=(row, raw_files, layered_dir, img_out_dir, no_plot, 
                                                                                          no_write_vis, no_write_irdiff, no_write_dirtyirdiff, 
                                                                                          no_plot_irdiff, no_plot_cirrus, no_plot_tropdiff, no_plot_snowice, no_plot_dirtyirdiff,
                                                                                          xy_bounds, del_combined_nc, ir_min_value, ir_max_value, 
                                                                                          grid_data, colorblind, plt_img_name, domain_sector, satellite, 
                                                                                          plt_cbar, subset, region, pthresh, plt_model, model, chk_day_night,
                                                                                          replot_img, rewrite_nc, append_nc, verbose)) for row in range(start_index, end_index)]
    
    results2  = [result.get() for result in results]                                                                                           #Get results returned from the multiprocessing workers
    fnames    = [x[0] for x in results2]                                                                                                       #Extract the image names created
    fnames2   = [x[1] for x in results2]                                                                                                       #Extract the combined netCDF file names created
    proj_data = [x[2] for x in results2]                                                                                                       #Extract satellite projection information
    
    pool.close()                                                                                                                               #Close multiprocessing threads
    pool.join()                                                                                                                                #Wait until multi processed job finished before moving on threads
    pool = None
    if len(fnames2) != len(range(start_index, end_index)):
        print('Not all files created, missing check could be implemented here.')                                                               #Warning check to notify user if files dropped
        
    return(fnames, fnames2, proj_data[0] if proj_data else None)

def seviri_create_image_from_three_modalities_parallel(f, raw_files, layered_dir, img_out_dir, no_plot, 
                                                       no_write_vis, no_write_irdiff, no_write_dirtyirdiff, 
                                                       no_plot_irdiff, no_plot_cirrus, no_plot_tropdiff, no_plot_snowice, no_plot_dirtyirdiff,
                                                       xy_bounds, del_combined_nc, ir_min_value, ir_max_value, 
                                                       grid_data, colorblind, plt_img_name, domain_sector, satellite, 
                                                       plt_cbar, subset, region, pthresh, plt_model, model, chk_day_night,
                                                       replot_img, rewrite_nc, append_nc, verbose):
    
    raw_file = raw_files[f]                                                                                                                    #Extract target file from the multiprocessing index
    
    combined_nc_file, arr_shape = seviri_combine_ir_glm_vis_parallel(infile               = raw_file,                                          #Create netCDF file containing SEVIRI processed data
                                                                     layered_dir          = layered_dir,
                                                                     satellite            = satellite,
                                                                     domain_sector        = domain_sector,
                                                                     no_write_vis         = no_write_vis,  
                                                                     no_write_irdiff      = no_write_irdiff, 
                                                                     no_write_dirtyirdiff = no_write_dirtyirdiff, 
                                                                     rewrite_nc           = rewrite_nc, 
                                                                     append_nc            = append_nc,
                                                                     xy_bounds            = xy_bounds, 
                                                                     verbose              = verbose)
    
    fnames = None                                                                                                                              #Initialize fnames as None incase plot is not generated
    if not no_plot:
        image = img_from_three_modalities2(nc_file             = combined_nc_file,                                                             #Create layered image output
                                           out_dir             = img_out_dir, 
                                           no_plot_glm         = True, 
                                           no_plot_vis         = no_write_vis, 
                                           no_plot_irdiff      = no_plot_irdiff, 
                                           no_plot_cirrus      = no_plot_cirrus, 
                                           no_plot_tropdiff    = no_plot_tropdiff, 
                                           no_plot_snowice     = no_plot_snowice, 
                                           no_plot_dirtyirdiff = no_plot_dirtyirdiff,
                                           ir_min_value        = ir_min_value, 
                                           ir_max_value        = ir_max_value, 
                                           grid_data           = grid_data, 
                                           colorblind          = colorblind, 
                                           plt_img_name        = plt_img_name, 
                                           plt_cbar            = plt_cbar, 
                                           subset              = subset,
                                           latlon_domain       = xy_bounds,  
                                           region              = region, 
                                           pthresh             = pthresh, 
                                           proj                = None, 
                                           replot_img          = replot_img,
                                           verbose             = verbose)
        fnames = image                                                                                                                         #Store the name of the image generated

    if del_combined_nc:
        os.remove(combined_nc_file)                                                                                                            #Delete the temporary combined file to save storage space
        
    return (fnames, combined_nc_file, None)

def get_time_key(filepath, sat='msg2'):
    """
    Extracts the scanning start time from the filename for proper sorting and deduplication.
    Native EUMETSAT formats only encode the END time, so we calculate the start time dynamically.
    """
    basename = os.path.basename(filepath)
    clean_name = basename.replace('T', '').replace('Z', '')
    matches = re.findall(r'(\d{14})', clean_name)
    
    sat_clean = sat.lower().replace('-', '').replace('_', '')
    is_rss = False
    if 'rss' in sat_clean or 'rapid' in sat_clean:
        is_rss = True
    elif matches and sat_clean in ['meteosat09', 'meteosat9', 'msg2', 'msg02'] and (datetime.strptime(matches[0][:8], "%Y%m%d") >= datetime(2013, 4, 9) and datetime.strptime(matches[0][:8], "%Y%m%d") < datetime(2018, 3, 1)):
        is_rss = True
    elif matches and sat_clean in ['meteosat10', 'meteosat10', 'msg3', 'msg03'] and (datetime.strptime(matches[0][:8], "%Y%m%d") >= datetime(2018, 3, 1)):
        is_rss = True

    if filepath.endswith('.nc') and len(matches) >= 2:
        return matches[0] # EPCT files list start time first
    elif filepath.endswith('.nat') and len(matches) >= 1:
        de = datetime.strptime(matches[0], "%Y%m%d%H%M%S")
        if is_rss:
            ds = de - timedelta(minutes=4, seconds=43) # Approx 4.7 min duration for RSS
        else:
            ds = de - timedelta(minutes=12, seconds=29) # Approx 12.5 min duration for Full Disk
        return ds.strftime("%Y%m%d%H%M%S")
    elif matches:
        return matches[0]
        
    return "00000000000000"
    
if __name__ == '__main__':
    run_seviri_create_image_from_three_modalities2()