#+
# Name:
#     run_create_image_from_three_modalities.py
# Purpose:
#     This is a script to run (by importing) programs that grid the GLM data, combine IR, GLM and 
#     VIS data into a netCDF file, and create a 3 layered image that gets put into labelme software
# Calling sequence:
#     import run_create_image_from_three_modalities
#     run_create_image_from_three_modalities.run_create_image_from_three_modalities()
# Input:
#     None.
# Functions:
#     glm_gridder                : Extracts and puts the GLM data and puts onto the VIS/IR data grid
#     combine_ir_glm_vis         : Combines GLM, IR, VIS data and puts into netCDF file.
#     img_from_three_modaltities : Creates 3-layered image that is input into labelme software
#     sort_goes_irvis_files      : Pass function into sorted function in order to sort the IR and VIS files by date strings.
#     sort_goes_comb_files       : Pass function into sorted function in order to sort the combined netCDF files by date strings.
# Output:
#     Creates netCDF file for GLM data gridded on GOES VIS grid, combined VIS, IR, and GLM netCDF file and
#     image file that gets put into labelme software to identify overshoot plumes.
# Keywords:
#     inroot               : STRING path to GLM, GOES visible and GOES IR data directories
#                            DEFAULT = '../../../goes-data/20200513-14/'
#     glm_out_dir          : STRING path to output GLM data directory containing GLM data gridded on same grid as VIS data
#                            DEFAULT = '../../../goes-data/out_dir/'
#     layered_dir          : STRING output directory path for netCDF file containing combined IR, VIS, and GLM data
#                            DEFAULT = '../../../goes-data/combined_nc_dir/'
#     img_out_dir          : Output directory for the IR, GLM, VIS data plot that is input into labelme software.
#                            DEFAULT = '../../../goes-data/layered_img_dir/'
#     plane_inroot         : STRING specifying directory path to airplane csv file. ONLY matters if plt_plane keyword is set. 
#                            DEFAULT = '../../../misc-data0/DCOTSS/aircraft/'
#     no_plot_glm          : IF keyword set, do not plot the GLM data. Setting this keyword = True makes you plot only the VIS
#                            and IR data. DEFAULT is to set this keyword to True and only plot the VIS and IR data.
#     no_plot              : IF keyword set (True), do not plot the IR/VIS data. DEFAULT = True (do not plot the IR/VIS data)
#     no_write_glm         : IF keyword set, do not write the GLM data to the combined modality netCDF file. Setting this 
#                            keyword = True makes you plot only the VIS and IR data. 
#                            DEFAULT = False. True -> only write the VIS and IR data.
#     no_write_vis         : IF keyword set, do not write the VIS data to the combined modality netCDF file. Setting this 
#                            keyword = True makes you write only the IR and GLM data. 
#                            DEFAULT = False. True -> only write the IR and GLM data.
#     no_write_irdiff      : IF keyword set, write the difference between 10.3 micron and 6.2 micron to file.  
#                            DEFAULT = True -> only write the VIS and IR and glm data.
#     no_write_cirrus      : IF keyword set, write the difference between 1.37 micron to file.  
#                            DEFAULT = True -> only write the VIS and IR and glm data.
#     no_write_snowice     : IF keyword set, write the difference between 1.6 micron to file.  
#                            DEFAULT = True -> only write the VIS and IR and glm data.
#     no_write_dirtyirdiff : IF keyword set, write the difference between 12-10 micron to file.  
#                            DEFAULT = True -> only write the VIS and IR and glm data.
#     xy_bounds            : ARRAY of floating point values giving the domain boundaries the user wants to restrict the combined netCDF and output image to
#                            [x0, y0, x1, y1]
#                            DEFAULT = [] -> write combined netCDF data for full scan region.
#     domain_sector        : STRING specifying the domain sector to use to create maps (= 'conus' or 'full'). DEFAULT = None -> use meso_sector
#     meso_sector          : LONG integer specifying the mesoscale domain sector to use to create maps (= 1 or 2). DEFAULT = 2 (sector 2)
#     del_combined_nc      : IF keyword set (True), delete combined netCDF file AFTER 3 modalities image is created. This keyword saves much needed space.
#                            DEFAULT = False -> does not deletes the IR/VIS/GLM combined netCDF file.
#     universal_file       : IF keyword set (True), write combined netCDF file that can be read universally. GLM and IR data resampled onto VIS data grid. 
#                            DEFAULT = False
#     ir_min_value         : LONG keyword to specify minimum IR temperature value to clip IR data image (K).
#                            DEFAULT = 170 
#     ir_max_value         : LONG keyword to specify maximum IR temperature value to clip IR data image (K).
#                            DEFAULT = 230 
#     grid_data            : If keyword set (True), grid the data onto latitude and longitude domain. If not set,
#                            grid the data by number of lat_pixels x number of lon_pixels.
#                            DEFAULT = False (grid pixel by pixel and do not grid data onto latitude and longitude domain)
#     start_index          : Index which to start looping over the vis and IR files at. This can be used if for some reason
#                            script crashes midway through running so you do not have to restart running program. DEFAULT = 0
#     date_range           : List containing start date and end date in YYYY-MM-DD hh:mm:ss format ("%Y-%m-%d %H:%M:%S") to run over. (ex. date_range = ['2021-04-30 19:00:00', '2021-05-01 04:00:00'])
#                            DEFAULT = [] -> follow start_index keyword or do all files in VIS/IR directories
#     colorblind           : IF keyword set (True), use colorblind friendly color table.
#                            DEFAULT = True (use colorblind friendly table)
#     plt_img_name         : IF keyword set (True), do not create 1:1 plot of lat/lon pixels and output image pixels. Instead
#                            plot image name on this figure. Used for creating videos.
#     plt_cbar             : IF keyword set (True), do not create 1:1 plot of lat/lon pixels and output image pixels. Instead
#                            plot color bar on the figure. Used for creating videos.
#     subset               : [x0, y0, x1, y1] array to give subset of image domain desired to be plotted. Array could contain pixel indices or lat/lon indices
#                            DEFAULT is to plot the entire domain. Y-axis is plotted from top down so y0 index is from the top!!!!
#     xy_bounds            : ARRAY of floating point values giving the domain boundaries the user wants to restrict the combined netCDF and output image to.
#                            [x0, y0, x1, y1]
#                            DEFAULT = [] -> write combined netCDF data for full scan region.
#                            NOTE: If region keyword is set/specified then it supersedes this for the image creation but not the combined netCDF.
#     region               : State or sector of US to plot data around. 
#                            DEFAULT = None -> plot for full domain.
#     run_gcs              : IF keyword set (True), read and write everything directly from the google cloud platform.
#                            DEFAULT = False
#     write_combined_gcs   : IF keyword set (True), write combined netCDF files to the google cloud platform. 
#                            NOTE: run_gcs keyword MUST BE True for this keyword to matter
#                            DEFAULT = True.
#     real_time            : IF keyword set (True), run the code in real time by only grabbing the most recent file to image and write.
#                            DEFAULT = False
#     del_local            : IF keyword set (True), delete local copies of files after writing them to google cloud. Only does anything if run_gcs = True.
#                            DEFAULT = False
#     pthresh              : FLOAT keyword to specify probability threshold to use. 
#                            DEFAULT = 0.0
#     plt_plane            : IF keyword set (True), plot airplane location on IR+VIS maps.
#                            DEFAULT = False
#     plt_traj             : IF keyword set (True), plot parcel back trajectory on images.
#                            DEFAULT = False
#     plt_model            : 1) If real_time == False -> Numpy array of model results (values 0-1) to plot on the image.
#                            2) If real_time == True  -> String specifying name and file path of model file to load and plot
#                            DEFAULT = None -> do not plot model results on image.
#     model                : Name of model being plotted (ex. IR+VIS)
#                            DEFAULT = None.
#     chk_day_night        : Structure containing day and night model data information in order to make seemless transition in predictions of the 2.
#                            DEFAULT = {} -> do not use day_night transition                 
#     in_bucket_name       : Google cloud storage bucket to read raw IR/GLM/VIS files.
#                            DEFAULT = 'goes-data'
#     out_bucket_name      : Google cloud storage bucket to write intermediate netCDF files (IR, VIS, and GLM data on same grid) as well as IR+VIS sandwich images.
#                            DEFAULT = 'ir-vis-sandwhich'
#     plane_bucket_name    : Google cloud storage bucket to read airplane csv data from
#                            DEFAULT = 'misc-data0'
#     replot_img           : IF keyword set (True), plot the IR/VIS sandwich image again
#                            DEFAULT = True
#     rewrite_nc           : IF keyword set (True), rewrite the combined ir/vis/glm netCDF file.
#                            DEFAULT = False (do not rewrite combined netCDF file if one exists. Write if it does not exist though)
#     append_nc            : IF keyword set (True), append the combined ir/vis/glm netCDF file.
#                            DEFAULT = True-> Append existing netCDF file
#     verbose              : BOOL keyword to specify whether or not to print verbose informational messages.
#                            DEFAULT = True which implies to print verbose informational messages
# Author and history:
#     John W. Cooney           2020-09-03.
#                              2020-09-24. (Added rewrite_nc keyword and fixed bug with range for loop end index)
#                              2024-02-15. (Added no_write_cirrus, no_write_dirtyirdiff, no_write_snowice)
#
#-

#### Environment Setup ###
# Package imports
import glob
import os
import re
from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta
import psutil
import pandas as pd
import metpy
import xarray
#import cartopy
#import cartopy.crs as ccrs
import pyproj
import multiprocessing as mp
import sys 
from threading import Thread
import time
#sys.path.insert(1, '../')
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.getcwd()))
from glm_gridder.glm_gridder2 import glm_gridder2
from glm_gridder.combine_ir_glm_vis import combine_ir_glm_vis
#from glm_gridder.img_from_three_modalities2 import img_from_three_modalities2
from new_model.gcs_processing import write_to_gcs, download_ncdf_gcs, list_gcs, load_csv_gcs
from gridrad.rdr_sat_utils_jwc import read_dcotss_er2_plane, doctss_read_lat_lon_alt_trajectory_particle_ncdf

def run_create_image_from_three_modalities(inroot             = os.path.join('..', '..', '..', 'goes-data', '20200513-14'), 
                                           glm_out_dir        = os.path.join('..', '..', '..', 'goes-data', 'out_dir'), 
                                           layered_dir        = os.path.join('..', '..', '..', 'goes-data', 'combined_nc_dir'), 
                                           img_out_dir        = os.path.join('..', '..', '..', 'goes-data', 'layered_img_dir'), 
                                           plane_inroot       = os.path.join('..', '..', '..', 'misc-data0', 'DCOTSS', 'aircraft'),
                                           no_plot_glm        = True, no_plot = True, 
                                           no_write_glm       = False, no_write_vis = False, no_write_irdiff = True, no_write_cirrus = True, no_write_snowice = True, no_write_dirtyirdiff = True, 
                                           xy_bounds          = [], 
                                           domain_sector      = None, 
                                           meso_sector        = 2, del_combined_nc = False, universal_file = True, 
                                           ir_min_value       = 170, ir_max_value  = 230, 
                                           grid_data          = False, start_index = 0, 
                                           date_range         = [], 
                                           colorblind         = True, plt_img_name = False, 
                                           plt_cbar           = False, subset      = None,  region    = None,   
                                           run_gcs            = False, real_time   = False, del_local = False, pthresh = 0.0,
                                           write_combined_gcs = True, 
                                           plt_plane          = False, plt_traj    = False, 
                                           plt_model          = None, model = None, 
                                           chk_day_night      = {}, 
                                           in_bucket_name     = 'goes-data', out_bucket_name = 'ir-vis-sandwhich', plane_bucket_name = 'misc-data0',
                                           replot_img         = True, rewrite_nc = False, append_nc = True, verbose = True):
    '''
    Name:
        run_create_image_from_three_modalities.py
    Purpose:
        This is a script to run (by importing) programs that grid the GLM data, combine IR, GLM and 
        VIS data into a netCDF file, and create a 3 layered image that gets put into labelme software
    Calling sequence:
        import run_create_image_from_three_modalities
        run_create_image_from_three_modalities.run_create_image_from_three_modalities()
    Input:
        None.
    Functions:
        glm_gridder                : Extracts and puts the GLM data and puts onto the VIS/IR data grid
        combine_ir_glm_vis         : Combines GLM, IR, VIS data and puts into netCDF file.
        img_from_three_modaltities : Creates 3-layered image that is input into labelme software
        sort_goes_irvis_files      : Pass function into sorted function in order to sort the IR and VIS files by date strings.
        sort_goes_comb_files       : Pass function into sorted function in order to sort the combined netCDF files by date strings.
    Output:
        Creates netCDF file for GLM data gridded on GOES VIS grid, combined VIS, IR, and GLM netCDF file and
        image file that gets put into labelme software to identify overshoot plumes.
    Keywords:
        inroot               : STRING path to GLM, GOES visible and GOES IR data directories
                               DEFAULT = '../../../goes-data/20200513-14/'
        glm_out_dir          : STRING path to output GLM data directory containing GLM data gridded on same grid as VIS data
                               DEFAULT = '../../../goes-data/out_dir/'
        layered_dir          : STRING output directory path for netCDF file containing combined IR, VIS, and GLM data
                               DEFAULT = '../../../goes-data/combined_nc_dir/'
        img_out_dir          : Output directory for the IR, GLM, VIS data plot that is input into labelme software.
                               DEFAULT = '../../../goes-data/layered_img_dir/'
        plane_inroot         : STRING specifying directory path to airplane csv file. ONLY matters if plt_plane keyword is set. 
                               DEFAULT = '../../../misc-data0/DCOTSS/aircraft/'
        no_plot_glm          : IF keyword set, do not plot the GLM data. Setting this keyword = True makes you plot only the VIS
                               and IR data. DEFAULT is to set this keyword to True and only plot the VIS and IR data.
        no_plot              : IF keyword set (True), do not plot the IR/VIS data. DEFAULT = True (do not plot the IR/VIS data)
        no_write_vis         : IF keyword set, do not write the VIS data to the combined modality netCDF file. Setting this 
                               keyword = True makes you write only the IR and GLM data. 
                               DEFAULT = False. True -> only write the IR and GLM data.
        no_write_glm         : IF keyword set, do not write the GLM data to the combined modality netCDF file. Setting this 
                               keyword = True makes you plot only the VIS and IR data. 
                               DEFAULT = False. True -> only write the VIS and IR data.
        no_write_irdiff      : IF keyword set, write the difference between 10.3 micron and 6.2 micron to file.  
                               DEFAULT = True -> only write the VIS and IR and glm data.
        no_write_cirrus      : IF keyword set, write the difference between 1.37 micron to file.  
                               DEFAULT = True -> only write the VIS and IR and glm data.
        no_write_snowice     : IF keyword set, write the difference between 1.6 micron to file.  
                               DEFAULT = True -> only write the VIS and IR and glm data.
        no_write_dirtyirdiff : IF keyword set, write the difference between 12-10 micron to file.  
                               DEFAULT = True -> only write the VIS and IR and glm data.
        domain_sector        : STRING specifying the domain sector to use to create maps (= 'conus' or 'full'). DEFAULT = None -> use meso_sector
        meso_sector          : LONG integer specifying the mesoscale domain sector to use to create maps (= 1 or 2). DEFAULT = 2 (sector 2)
        del_combined_nc      : IF keyword set (True), delete combined netCDF file AFTER 3 modalities image is created. This keyword saves much needed space.
                               DEFAULT = False -> does not deletes the IR/VIS/GLM combined netCDF file.
        universal_file       : IF keyword set (True), write combined netCDF file that can be read universally. GLM and IR data resampled onto VIS data grid. 
                               DEFAULT = False
        ir_min_value         : LONG keyword to specify minimum IR temperature value to clip IR data image (K).
                               DEFAULT = 170 
        ir_max_value         : LONG keyword to specify maximum IR temperature value to clip IR data image (K).
                               DEFAULT = 230 
        grid_data            : If keyword set (True), grid the data onto latitude and longitude domain. If not set,
                               grid the data by number of lat_pixels x number of lon_pixels.
                               DEFAULT = False (grid pixel by pixel and do not grid data onto latitude and longitude domain)
        start_index          : Index which to start looping over the vis and IR files at. This can be used if for some reason
                               script crashes midway through running so you do not have to restart running program. DEFAULT = 0
        date_range           : List containing start date and end date in YYYY-MM-DD hh:mm:ss format ("%Y-%m-%d %H:%M:%S") to run over. (ex. date_range = ['2021-04-30 19:00:00', '2021-05-01 04:00:00'])
                               DEFAULT = [] -> follow start_index keyword or do all files in VIS/IR directories
        colorblind           : IF keyword set (True), use colorblind friendly color table.
                               DEFAULT = True (use colorblind friendly table)
        plt_img_name         : IF keyword set (True), do not create 1:1 plot of lat/lon pixels and output image pixels. Instead
                               plot image name on this figure. Used for creating videos.
        plt_cbar             : IF keyword set (True), do not create 1:1 plot of lat/lon pixels and output image pixels. Instead
                               plot color bar on the figure. Used for creating videos.
        subset               : [x0, y0, x1, y1] array to give subset of image domain desired to be plotted. Array could contain pixel indices or lat/lon indices
                               DEFAULT is to plot the entire domain. Y-axis is plotted from top down so y0 index is from the top!!!!
        xy_bounds            : ARRAY of floating point values giving the domain boundaries the user wants to restrict the combined netCDF and output image to.
                               [x0, y0, x1, y1]
                               DEFAULT = [] -> write combined netCDF data for full scan region.
                               NOTE: If region keyword is set/specified then it supersedes this for the image creation but not the combined netCDF.
        region               : State or sector of US to plot data around. 
                               DEFAULT = None -> plot for full domain.
        run_gcs              : IF keyword set (True), read and write everything directly from the google cloud platform.
                               DEFAULT = False
        write_combined_gcs   : IF keyword set (True), write combined netCDF files to the google cloud platform. 
                               NOTE: run_gcs keyword MUST BE True for this keyword to matter
                               DEFAULT = True.
        real_time            : IF keyword set (True), run the code in real time by only grabbing the most recent file to image and write.
                               DEFAULT = False
        del_local            : IF keyword set (True), delete local copies of files after writing them to google cloud. Only does anything if run_gcs = True.
                               DEFAULT = False
        pthresh              : FLOAT keyword to specify probability threshold to use. 
                               DEFAULT = 0.0
        plt_plane            : IF keyword set (True), plot airplane location on IR+VIS maps.
                               DEFAULT = False
        plt_traj             : IF keyword set (True), plot parcel back trajectory on images.
                               DEFAULT = False
        plt_model            : 1) If real_time == False -> Numpy array of model results (values 0-1) to plot on the image.
                               2) If real_time == True  -> String specifying name and file path of model file to load and plot
                               DEFAULT = None -> do not plot model results on image.
        model                : Name of model being plotted (ex. IR+VIS)
                               DEFAULT = None.
        chk_day_night        : Structure containing day and night model data information in order to make seemless transition in predictions of the 2.
                               DEFAULT = {} -> do not use day_night transition                 
        in_bucket_name       : Google cloud storage bucket to read raw IR/GLM/VIS files.
                               DEFAULT = 'goes-data'
        out_bucket_name      : Google cloud storage bucket to write intermediate netCDF files (IR, VIS, and GLM data on same grid) as well as IR+VIS sandwich images.
                               DEFAULT = 'ir-vis-sandwhich'
        plane_bucket_name    : Google cloud storage bucket to read airplane csv data from
                               DEFAULT = 'misc-data0'
        replot_img           : IF keyword set (True), plot the IR/VIS sandwich image again
                               DEFAULT = True
        rewrite_nc           : IF keyword set (True), rewrite the combined ir/vis/glm netCDF file.
                               DEFAULT = False (do not rewrite combined netCDF file if one exists. Write if it does not exist though)
        append_nc            : IF keyword set (True), append the combined ir/vis/glm netCDF file.
                               DEFAULT = True-> Append existing netCDF file
        verbose              : BOOL keyword to specify whether or not to print verbose informational messages.
                               DEFAULT = True which implies to print verbose informational messages
    Author and history:
        John W. Cooney           2020-09-03.
                                 2020-09-24. (Added rewrite_nc keyword and fixed bug with range for loop end index)
    '''
    if domain_sector == None:
        sector  = 'M' + str(meso_sector)                                                                                                       #Set mesoscale sector string. Used for output directory and which input files to use
        str_sec = 'meso'
    else:
        if domain_sector.lower() == 'conus' or domain_sector.lower() == 'c':
            sector  = 'C'                                                                                                                      #Set domain sector string. Used for output directory and which input files to use
            str_sec = 'conus'
        elif domain_sector.lower() == 'full' or domain_sector.lower() == 'f':
            sector  = 'F'                                                                                                                      #Set domain sector string. Used for output directory and which input files to use
            str_sec = 'full'
    inroot      = os.path.realpath(inroot)                                                                                                     #Create link to real path so compatible with Mac
    f_dates     = os.path.basename(inroot)                                                                                                     #Extract date string from input root
    glm_out_dir = os.path.realpath(os.path.join(glm_out_dir, f_dates, sector))                                                                 #Create link to real path so compatible with Mac
    layered_dir = os.path.realpath(os.path.join(layered_dir, f_dates))                                                                         #Create link to real path so compatible with Mac
    img_out_dir = os.path.realpath(img_out_dir)                                                                                                #Create link to real path so compatible with Mac

    vis_dir    = os.path.join(inroot, 'vis')                                                                                                   #Path to GOES Visible data files
    glm_in_dir = os.path.join(inroot, 'glm')                                                                                                   #Path to GLM data file
    ir_dir     = os.path.join(inroot, 'ir')                                                                                                    #Path to GOES IR data files
    os.makedirs(ir_dir,      exist_ok = True)
    os.makedirs(layered_dir, exist_ok = True)                                                                                                  #Create output VIS/IR/GLM directory file path if does not already exist
    if no_write_vis == False:
        os.makedirs(vis_dir,     exist_ok = True)
    if no_write_glm == False:
        os.makedirs(glm_in_dir,  exist_ok = True)
        os.makedirs(glm_out_dir, exist_ok = True)
    if no_write_irdiff == False:
        irdiff_dir = os.path.join(inroot, 'ir_diff')                                                                                           #Path to GOES IR 6.2 micron data files
        os.makedirs(irdiff_dir,  exist_ok = True)
    if no_write_cirrus == False:
        cirrus_dir = os.path.join(inroot, 'cirrus')                                                                                            #Path to GOES IR 1.37 micron data files
        os.makedirs(cirrus_dir,  exist_ok = True)
    if no_write_snowice == False:
        snowice_dir = os.path.join(inroot, 'snowice')                                                                                          #Path to GOES IR 1.6 micron data files
        os.makedirs(snowice_dir,  exist_ok = True)
    if no_write_dirtyirdiff == False:
        dirtyir_dir = os.path.join(inroot, 'dirtyir')                                                                                          #Path to GOES IR 12.3 micron data files
        os.makedirs(dirtyir_dir,  exist_ok = True)
    
    if plt_img_name == True:   
        nam = 'plt_img_name'
    else:
        nam = ''
    if plt_cbar == True:   
        cbar = 'plt_cbar'
    else:
        cbar = ''
    if real_time != True:
        if os.path.basename(img_out_dir) == 'layered_img_dir':
            if subset != None: 
                img_out_dir = os.path.join(re.split(os.path.basename(img_out_dir), img_out_dir)[0], 'subset_domain', nam, cbar, os.path.basename(img_out_dir))  #Set output image path to subset domain files
            else:
                img_out_dir = os.path.join(re.split(os.path.basename(img_out_dir), img_out_dir)[0], nam, cbar, os.path.basename(img_out_dir))  #Set output image path to pixel grid files
            if grid_data == False: 
                img_out_dir = os.path.join(img_out_dir, 'pixel_grid', sector, f_dates)                                                         #Set output image path to pixel grid files
            else:
                img_out_dir = os.path.join(img_out_dir, 'lat_lon_grid', sector, f_dates)                                                       #Set output image path to pixel grid files
        else:
            if subset != None: 
                img_out_dir = os.path.join(img_out_dir, 'subset_domain')                                                                       #Set output image path to pixel grid files
            if grid_data == False: 
                img_out_dir = os.path.join(img_out_dir, 'pixel_grid', sector, nam, cbar, f_dates)                                              #Set output image path to pixel grid files
            else:
                img_out_dir = os.path.join(img_out_dir, 'lat_lon_grid', sector, nam, cbar, f_dates)                                            #Set output image path to pixel grid files
    
    if no_plot == False: os.makedirs(os.path.dirname(img_out_dir), exist_ok = True)                                                            #Create output image directory file path if does not already exist
    if replot_img == False:     
        img_files = sorted(glob.glob(os.path.join(img_out_dir, '*.png')))                                                                      #Find all image files
        i_dstr    = [re.split('_s|_', os.path.basename(i))[-3] for i in img_files]                                                             #Extract date string of image files
    
    if run_gcs == True:
        if no_write_vis == False:
            vis_files = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'vis'), ['-Rad' + sector + '-']))                                #Extract names of all of the GOES visible data files from the google cloud
        ir_files  = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'ir'), ['-Rad' + sector + '-']))                                     #Extract names of all of the GOES infrared data files from the google cloud
        if rewrite_nc == False or append_nc == True:
            com_files = sorted(list_gcs(out_bucket_name, os.path.join('combined_nc_dir', f_dates), ['_COMBINED_', '_' + sector + '_']))        #Extract names of all of the GOES infrared data files from the google cloud
        if no_write_irdiff == False:
            ird_files = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'ir_diff'), ['-Rad' + sector + '-']))                            #Extract names of all of the GOES 6.2 micron infrared data files from the google cloud
        if no_write_cirrus == False:
            c_files = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'cirrus'), ['-Rad' + sector + '-']))                               #Extract names of all of the GOES 6.2 micron infrared data files from the google cloud
        if no_write_snowice == False:
            s_files = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'snowice'), ['-Rad' + sector + '-']))                              #Extract names of all of the GOES 6.2 micron infrared data files from the google cloud
        if no_write_dirtyirdiff == False:
            dird_files = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'dirtyir'), ['-Rad' + sector + '-']))                           #Extract names of all of the GOES 6.2 micron infrared data files from the google cloud
        if no_write_glm == False:                                                                                                              #Must download GLM files to local storage
            glm_files = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'glm'), ['-LCFA', 'GLM']))                                       #Extract names of all of the GOES GLM data files from the google cloud
            if real_time == True: 
                glm_files = glm_files[-15:]                                                                                                    #Download only the most recent GLM files if running code in real time
            for g in glm_files: download_ncdf_gcs(in_bucket_name, g, glm_in_dir)
        if real_time == True: 
            ir_files  = [ir_files[-1]]                                                                                                         #Download only the most recent IR files if running code in real time
            if no_write_vis == False:
                vis_files  = [vis_files[-1]]                                                                                                   #Download only the most recent VIS files if running code in real time
            if no_write_irdiff == False:
                ird_files  = [ird_files[-1]]                                                                                                   #Download only the most recent IR files if running code in real time
            if no_write_cirrus == False:
                c_files    = [c_files[-1]]                                                                                                     #Download only the most recent IR files if running code in real time
            if no_write_snowice == False:
                s_files    = [s_files[-1]]                                                                                                     #Download only the most recent IR files if running code in real time
            if no_write_dirtyirdiff == False:
                dird_files = [dird_files[-1]]                                                                                                  #Download only the most recent IR files if running code in real time
        else:
            ir_files2  = sorted(glob.glob(os.path.join(ir_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))                           #Extract names of all of the GOES IR data files already in local storage
            if len(ir_files2) > 0:                                                                                                             #Only download files that are not already available on the local disk 
                fi  = sorted([os.path.basename(g) for g in ir_files])
                fi2 = sorted([os.path.basename(g) for g in ir_files2])
                ir_files0  = [ir_files[ff] for ff in range(len(fi)) if fi[ff] not in fi2]                                                      #Find ir_files not already stored on local disk
                ir_files   = ir_files0
            if no_write_vis == False:
                vis_files2 = sorted(glob.glob(os.path.join(vis_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))                      #Extract names of all of the GOES visible data files already in local storage
                if len(vis_files2) > 0:                                                                                                        #Only download files that are not already available on the local disk 
                    fv  = sorted([os.path.basename(g) for g in vis_files])
                    fv2 = sorted([os.path.basename(g) for g in vis_files2])
                    vis_files0 = [vis_files[ff] for ff in range(len(fv)) if fv[ff] not in fv2]                                                 #Find vis_files not already stored on local disk
                    vis_files  = vis_files0
                for g in vis_files: download_ncdf_gcs(in_bucket_name, g, vis_dir)
   
        for g in ir_files: download_ncdf_gcs(in_bucket_name, g, ir_dir)
        if no_write_irdiff == False:
            if real_time != True:
                ird_files2 = sorted(glob.glob(os.path.join(irdiff_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))                   #Extract names of all of the GOES IR channel 8 data files
                if len(ird_files2) > 0:
                    fi  = sorted([os.path.basename(g) for g in ird_files])
                    fi2 = sorted([os.path.basename(g) for g in ird_files2])
                    ird_files0 = [ird_files[ff] for ff in range(len(fi)) if fi[ff] not in fi2]                                                 #Find IR channel 8 files not already stored on local disk
                    ird_files  = ird_files0
            for g in ird_files: download_ncdf_gcs(in_bucket_name, g, irdiff_dir)
        
        if no_write_cirrus == False:
            if real_time != True:
                c_files2 = sorted(glob.glob(os.path.join(cirrus_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))                     #Extract names of all of the GOES cirrus channel data files
                if len(c_files2) > 0:
                    fi  = sorted([os.path.basename(g) for g in c_files])
                    fi2 = sorted([os.path.basename(g) for g in c_files2])
                    c_files0 = [c_files[ff] for ff in range(len(fi)) if fi[ff] not in fi2]                                                     #Find cirrus channel files not already stored on local disk
                    c_files  = c_files0
            for g in c_files: download_ncdf_gcs(in_bucket_name, g, cirrus_dir)

        if no_write_snowice == False:
            if real_time != True:
                s_files2 = sorted(glob.glob(os.path.join(snowice_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))                    #Extract names of all of the GOES snowice channel data files
                if len(s_files2) > 0:
                    fi  = sorted([os.path.basename(g) for g in s_files])
                    fi2 = sorted([os.path.basename(g) for g in s_files2])
                    s_files0 = [s_files[ff] for ff in range(len(fi)) if fi[ff] not in fi2]                                                     #Find snowice files not already stored on local disk
                    s_files  = s_files0
            for g in s_files: download_ncdf_gcs(in_bucket_name, g, snowice_dir)

        if no_write_dirtyirdiff == False:
            if real_time != True:
                dird_files2 = sorted(glob.glob(os.path.join(dirtyir_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))                 #Extract names of all of the GOES dirty IR channel data files
                if len(dird_files2) > 0:
                    fi  = sorted([os.path.basename(g) for g in dird_files])
                    fi2 = sorted([os.path.basename(g) for g in dird_files2])
                    dird_files0 = [dird_files[ff] for ff in range(len(fi)) if fi[ff] not in fi2]                                               #Find dirty IR files not already stored on local disk
                    dird_files  = dird_files0
            for g in dird_files: download_ncdf_gcs(in_bucket_name, g, dirtyir_dir)
        
        if (rewrite_nc == False or append_nc == True) and real_time != True:
            com_files2 = sorted(glob.glob(os.path.join(layered_dir, '*' + '_' + sector + '_COMBINED_' + '*.nc'), recursive = True))            #Find combined netCDF files not already stored on local disk
            if len(com_files2) > 0:
                fi  = sorted([os.path.basename(g) for g in com_files])
                fi2 = sorted([os.path.basename(g) for g in com_files2])
                com_files0 = [com_files[ff] for ff in range(len(fi)) if fi[ff] not in fi2]                                                     #Find combined netCDF files not already stored on local disk
                com_files  = com_files0
            for g in com_files: download_ncdf_gcs(out_bucket_name, g, layered_dir)                                                             #Download combined netCDF files if exist and want them to be rewritten
#         if len(xy_bounds) > 0 and domain_sector != None:
#             dpath = os.path.join(re.split('combined_nc_dir', layered_dir)[0], 'sat_projection_files')
#             if os.path.exists(dpath) == False:
#                 sat_proj_files = sorted(list_gcs(in_bucket_name, 'sat_projection_files', ['.nc']))
#                 if len(sat_proj_files) <= 0:
#                     print('Satellite projection files not found???')
#                     print(sat_proj_files)
#                     print(in_bucket_name)
#                     exit()
#                 os.makedirs(dpath, exist_ok = True)
#                 for g in sat_proj_files: download_ncdf_gcs(in_bucket_name, g, dpath)
    
    vis_files = sorted(glob.glob(os.path.join(vis_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True), key = sort_goes_irvis_files)      #Extract names of all of the GOES visible data files
    ir_files  = sorted(glob.glob(os.path.join(ir_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True), key = sort_goes_irvis_files)       #Extract names of all of the GOES IR data files
    if no_write_vis == False:
        if (len(vis_files) != len(ir_files) or len(vis_files) <= 0):                                                                           #Check to make sure same number of IR and VIS files in directory
            print('Number of visible and infrared data files in directory do not match or no files in directory!?')
            print(len(vis_files))
            print(len(ir_files))
            if len(vis_files) == 0:
                exit()
            if abs(len(vis_files) - len(ir_files)) > 10:
                print('More than 10 files difference in size within the directories. Must abort.')
                print(ir_files[0])
                print(ir_files[-1])
                print(vis_files[0])
                print(vis_files[-1])
                exit()
            else:
                if real_time != True:
                    vis_dstr = [re.split('_s|_', os.path.basename(vis_files[f]))[3] for f in range(0, len(vis_files))]                         #Extract visibile date strings to determine missing VIS or IR data files
                    ir_dstr = [re.split('_s|_', os.path.basename(ir_files[f]))[3] for f in range(0, len(ir_files))]                            #Extract IR date strings to determine missing VIS or IR data files
                    for f in (range(0, len(vis_files))):
                        if vis_dstr[f] not in ir_dstr:
                            if run_gcs == True and del_local == True:
                                print(vis_files[f] + ' does not have a matching IR file???')
                                print('Deleting the visible data file from the local directory')
                                rm = 'y'
                            else:    
                                rm = str(input(vis_files[f] + ' does not have a matching IR data file??? Is it ok to delete this file from local directory? (y/n): '))
                            if rm[0].lower() == 'n':
                                print('User chose not to remove file. File must be removed in order for remaining code to work.')
                                exit()
                            if rm[0].lower() == 'y':
                                os.remove(vis_files[f])                                                                                        #Delete visible data file that does not have a matching IR data file scan
                    for f in (range(0, len(ir_files))):
                        if ir_dstr[f] not in vis_dstr:
                            if run_gcs == True and del_local == True:
                                print(ir_files[f] + ' does not have a matching VIS file???')
                                print('Deleting the IR data file from the local directory')
                                rm = 'y'
                            else:    
                                rm = str(input(ir_files[f] + ' does not have a matching VIS data file??? Is it ok to delete this file from local directory? (y/n): '))
                            if rm[0].lower() != 'y':
                                print('User chose not to remove file. File must be removed in order for remaining code to work.')
                                exit()
                            if rm[0].lower() == 'y':
                                os.remove(ir_files[f])                                                                                         #Delete IR data file that does not have a matching IR data file scan
                    vis_files = sorted(glob.glob(os.path.join(vis_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True), key = sort_goes_irvis_files) #Extract names of all of the GOES visible data files
                    ir_files  = sorted(glob.glob(os.path.join(ir_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True), key = sort_goes_irvis_files)  #Extract names of all of the GOES IR data files

    if real_time == True:
        if no_write_vis == False:
            vis_files = [vis_files[-1]]
        ir_files  = [ir_files[-1]]

    if len(date_range) > 0:
        time0 = datetime.strptime(date_range[0], "%Y-%m-%d %H:%M:%S")                                                                          #Extract start date to find data within of the date range
        time1 = datetime.strptime(date_range[1], "%Y-%m-%d %H:%M:%S")                                                                          #Extract end date to find data within of the date range
        #Extract subset of raw VIS netCDF files within date_range
        start_index = 0
        end_index   = len(ir_files)
        for f in (range(0, len(ir_files))):
            file_attr = re.split('_s|_', os.path.basename(ir_files[f]))                                                                   
            date_str  = file_attr[3][7:11]                                                                                                 
            if file_attr[3][0:11] >= "{:04d}{:03d}{:02d}{:02d}".format(time0.year, time0.timetuple().tm_yday, time0.hour, time0.minute) and start_index == 0:
                if f == 0:
                    start_index = -1
                else:    
                    start_index = f
            if file_attr[3][0:11] >= "{:04d}{:03d}{:02d}{:02d}".format(time1.year, time1.timetuple().tm_yday, time1.hour, time1.minute) and end_index == len(ir_files):
                if "{:02d}{:02d}".format(time1.hour, time1.minute) == date_str:
                    if f == (len(ir_files)-1):
                        end_index = len(ir_files)
                    else:    
                        end_index = f+1
                else:  
                    if f == (len(ir_files)-1):
                        end_index = len(ir_files)              
                    else:
                        end_index = f                
    else:
        end_index = len(ir_files)

    if start_index == -1:
        start_index = 0

#     if plt_plane == True:
#         if grid_data == False:
#             print('Must plot plane on Grid so grid_data keyword must be set!!!')
#             exit()
#         df = read_dcotss_er2_plane(inroot      = plane_inroot,                                                                                 #Read DCOTSS ER2 file in as pandas dataframe
#                                    bucket_name = plane_bucket_name,
#                                    use_local   = not run_gcs, 
#                                    verbose     = verbose)
#         pdates = sorted([datetime.strptime(str(df['YYYYMMDD'][d]), "%Y%m%d") + timedelta(seconds = int(df['Time_Start'][d])) for d in range(len(df))]) #Extract dates within file as datetime stucture
#         
#         if len(pdates) <= 0:
#             print('No plane dates???')
#             exit()
#         
#         if plt_traj == True:
#             if run_gcs == True:
#                 gcs_prefix = re.split('misc-data0/', plane_inroot)[1]                                                                          #Path to DCOTSS standard deviation text file on GCP bucket
#                 txt_file   = list_gcs(plane_bucket_name, gcs_prefix, ['DCOTSS_HHH_H2Obinned_by_POT_MMS.txt'], delimiter = '/')                 #Find DCOTSS standard deviation text file
#                 if len(txt_file) <= 0 or len(txt_file) > 1:
#                     print('No or too many DCOTSS standard deviation text files found????')
#                     print(gcs_prefix)
#                     print(plane_bucket_name)
#                     print(plane_inroot)
#                     print(txt_file)
#                     exit()
#                 df_std = load_csv_gcs(plane_bucket_name, txt_file[0], skiprows = None)                                                         #Load DCOTSS standard deviation text file from GCP
#             else:
#                 df_std = pd.read_csv(os.path.join(plane_inroot, 'DCOTSS_HHH_H2Obinned_by_POT_MMS.txt'))                                        #Read standard deviation file
# 
#             std_theta_bin = [int(re.split('-', s)[0]) for s in list(df_std['POT_MMS'])]                                                        #Extract bin start values
#             HHH_H2O_mean  = list(df_std['HHH_H2O mean'])                                                                                       #Extract mean HHH_H20 at each potential temperature level
#             HHH_H2O_std   = list(df_std['HHH_H2O stdev'])                                                                                      #Extract standard deviation of HHH_H20 at each potential temperature level
#             rm_dates      = []                                                                                                                 #Initialize list to store dates not to include because HHH_H20 < mean+std. dev
#             for idx, p in enumerate(pdates):
#                 theta_mms = float(df['POT_MMS'][idx])                                                                                          #Extract potential temperature from flight
#                 if theta_mms < np.min(std_theta_bin) or theta_mms > np.max(std_theta_bin)+5:
#                     rm_dates.append(p)
#                 else:
#                     if pd.isna(theta_mms):
#                        rm_dates.append(p)
#                     else:
#                        theta_bin = np.where((theta_mms - np.asarray(std_theta_bin)) > 0)[-1][-1]                                               #Find theta bin in which potential temperature falls
#                        if HHH_H2O_mean[theta_bin] >= 0.0:
#                            if (df['HHH_H2O'][idx] < (HHH_H2O_mean[theta_bin] + HHH_H2O_std[theta_bin])):
#                                rm_dates.append(p)                                                                                              #Append list of dates to skip because HHH_H20 < mean+std. dev
#                        else:
#                            rm_dates.append(p)                                                                                                  #Append list of dates to skip because HHH_H20 < mean+std. dev
    
    fnames  = []                                                                                                                               #Initialize list to store the names of the image files
    fnames2 = []                                                                                                                               #Initialize list to store the names of the combined netCDF files
    for f in range(start_index, end_index):
        file_attr = re.split('_s|_e|_', os.path.basename(ir_files[f]))                                                                         #Split file string in order to extract date string of scan
        date_str  = file_attr[3]                                                                                                               #Split file string in order to extract date string of scan
        if no_write_vis == False:
            file_attr2 = re.split('_s|_', os.path.basename(vis_files[f]))                                                                      #Split file string in order to extract date string of scan
            date_str2  = file_attr2[3]                                                                                                         #Split file string in order to extract date string of scan
            if (date_str2 != date_str):                                                                                                        #Throw error if not using matching IR and VIS files
                print(ir_files[f])
                print(vis_files[f])
                print('GOES IR and VIS file date strings do not match??')
                exit()
#         ir    = xarray.open_dataset(vis_files[f]).load()
#         lon_c = str(np.copy(ir['geospatial_lat_lon_extent'].geospatial_lon_center))
#         lat_c = str(np.copy(ir['geospatial_lat_lon_extent'].geospatial_lat_center))
#         dat   = ir.metpy.parse_cf('Rad')
#         proj  = [dat.metpy.cartopy_crs, dat.x.min(), dat.x.max(), dat.y.min(), dat.y.max()]
#         ir.close() 
#         print(proj)
        proj = None
        if no_write_glm == False:
            with Dataset(ir_files[f], 'r') as ir:                                                                                              #Read in IR file to get mesoscale domain central latitudes and longitudes
                lon_c = str(np.copy(ir.variables['geospatial_lat_lon_extent'].geospatial_lon_center))                                          #Extract longitude center of mesoscale domain as STRING
                lat_c = str(np.copy(ir.variables['geospatial_lat_lon_extent'].geospatial_lat_center))                                          #Extract latitude center of mesoscale domain as STRING
            glm_file = os.path.join(glm_out_dir, date_str + '_gridded_data.nc')                                                                #Create output GLM directory path if does not already exist
            if sector[0].lower() == 'c' or sector[0].lower() == 'f':
                date_str3 = file_attr[4]
                d_range   = [date_str, file_attr[4]]                                                                                           #Set range of dates
            else:
                date_str3 = date_str
                d_range   = []
            if real_time == True:
                t = Thread(target = glm_gridder2, kwargs = {'outfile' : glm_file, 'glm_root' : glm_in_dir, 'no_plot' : no_plot_glm, 'date_str' : date_str3, 'date_range' : [], 'ctr_lon0' : lon_c, 'ctr_lat0' : lat_c, 'sector' : str_sec, 'verbose' : verbose})
                t.start()
                if no_write_vis == False:
                    combined_nc_file, arr_shape = combine_ir_glm_vis(vis_file             = vis_files[f],                                      #Create netCDF file containing IR, GLM and VIS data
                                                                     ir_file              = ir_files[f], 
                                                                     glm_file             = glm_file, 
                                                                     layered_dir          = layered_dir,
                                                                     no_write_vis         = no_write_vis,  
                                                                     no_write_glm         = no_write_glm,  
                                                                     no_write_irdiff      = no_write_irdiff, 
                                                                     no_write_cirrus      = no_write_cirrus, 
                                                                     no_write_snowice     = no_write_snowice, 
                                                                     no_write_dirtyirdiff = no_write_dirtyirdiff, 
                                                                     universal_file       = universal_file, 
                                                                     rewrite_nc           = rewrite_nc, 
                                                                     append_nc            = append_nc,
                                                                     xy_bounds            = xy_bounds, 
                                                                     glm_thread_info      = [t], 
                                                                     verbose              = verbose)
                else:    
                    combined_nc_file, arr_shape = combine_ir_glm_vis(vis_file             = '',                                                #Create netCDF file containing IR, GLM data
                                                                     ir_file              = ir_files[f], 
                                                                     glm_file             = glm_file, 
                                                                     layered_dir          = layered_dir,
                                                                     no_write_vis         = no_write_vis,  
                                                                     no_write_glm         = no_write_glm,  
                                                                     no_write_irdiff      = no_write_irdiff, 
                                                                     no_write_cirrus      = no_write_cirrus, 
                                                                     no_write_snowice     = no_write_snowice, 
                                                                     no_write_dirtyirdiff = no_write_dirtyirdiff, 
                                                                     universal_file       = universal_file, 
                                                                     rewrite_nc           = rewrite_nc, 
                                                                     append_nc            = append_nc,
                                                                     xy_bounds            = xy_bounds, 
                                                                     glm_thread_info      = [t], 
                                                                     verbose              = verbose)
            else:    
                glm_grid = glm_gridder2(outfile = glm_file, glm_root = glm_in_dir, no_plot = no_plot_glm, date_str = date_str3, date_range = d_range, ctr_lon0 = lon_c, ctr_lat0 = lat_c, sector = str_sec, verbose = verbose)
                if no_write_vis == False:
                    combined_nc_file, arr_shape = combine_ir_glm_vis(vis_file             = vis_files[f],                                      #Create netCDF file containing IR, GLM and VIS data
                                                                     ir_file              = ir_files[f], 
                                                                     glm_file             = glm_file, 
                                                                     layered_dir          = layered_dir,
                                                                     no_write_vis         = no_write_vis,  
                                                                     no_write_glm         = no_write_glm,  
                                                                     no_write_irdiff      = no_write_irdiff, 
                                                                     no_write_cirrus      = no_write_cirrus, 
                                                                     no_write_snowice     = no_write_snowice, 
                                                                     no_write_dirtyirdiff = no_write_dirtyirdiff, 
                                                                     universal_file       = universal_file, 
                                                                     rewrite_nc           = rewrite_nc, 
                                                                     append_nc            = append_nc,
                                                                     xy_bounds            = xy_bounds, 
                                                                     verbose              = verbose)         
                else:
                    combined_nc_file, arr_shape = combine_ir_glm_vis(vis_file             = '',                                                #Create netCDF file containing IR, GLM data
                                                                     ir_file              = ir_files[f], 
                                                                     glm_file             = glm_file, 
                                                                     layered_dir          = layered_dir,
                                                                     no_write_vis         = no_write_vis,  
                                                                     no_write_glm         = no_write_glm,  
                                                                     no_write_irdiff      = no_write_irdiff, 
                                                                     no_write_cirrus      = no_write_cirrus, 
                                                                     no_write_snowice     = no_write_snowice, 
                                                                     no_write_dirtyirdiff = no_write_dirtyirdiff, 
                                                                     universal_file       = universal_file, 
                                                                     rewrite_nc           = rewrite_nc, 
                                                                     append_nc            = append_nc,
                                                                     xy_bounds            = xy_bounds, 
                                                                     verbose              = verbose)         
        else:
            if no_write_vis == False:
                combined_nc_file, arr_shape = combine_ir_glm_vis(vis_file             = vis_files[f],                                          #Create netCDF file containing IR, GLM and VIS data
                                                                 ir_file              = ir_files[f], 
                                                                 layered_dir          = layered_dir,
                                                                 no_write_vis         = no_write_vis,  
                                                                 no_write_glm         = no_write_glm,  
                                                                 no_write_irdiff      = no_write_irdiff, 
                                                                 no_write_cirrus      = no_write_cirrus, 
                                                                 no_write_snowice     = no_write_snowice, 
                                                                 no_write_dirtyirdiff = no_write_dirtyirdiff, 
                                                                 rewrite_nc           = rewrite_nc, 
                                                                 append_nc            = append_nc,
                                                                 universal_file       = universal_file, 
                                                                 xy_bounds            = xy_bounds, 
                                                                 verbose              = verbose)
            else:
                combined_nc_file, arr_shape = combine_ir_glm_vis(vis_file             = '',                                                    #Create netCDF file containing IR, GLM data
                                                                 ir_file              = ir_files[f], 
                                                                 layered_dir          = layered_dir,
                                                                 no_write_vis         = no_write_vis,  
                                                                 no_write_glm         = no_write_glm,  
                                                                 no_write_irdiff      = no_write_irdiff, 
                                                                 no_write_cirrus      = no_write_cirrus, 
                                                                 no_write_snowice     = no_write_snowice, 
                                                                 no_write_dirtyirdiff = no_write_dirtyirdiff, 
                                                                 rewrite_nc           = rewrite_nc, 
                                                                 append_nc            = append_nc,
                                                                 universal_file       = universal_file, 
                                                                 xy_bounds            = xy_bounds, 
                                                                 verbose              = verbose)
        fnames2.append(combined_nc_file)                                                                                                       #Store the names of the images in a list
        
        exist = False                                                                                                                          #Default is to plot image again
        if replot_img == False:     
            if date_str in i_dstr:
                exist = True
                image = img_files[i_dstr.index(date_str)]
        p = None
        if no_plot == False and exist == False:
            if plt_plane == True:
                fdate  = datetime.strptime(date_str[0:-1], '%Y%j%H%M%S')                                                                       #Extract date of VIS/IR data file as datetime structure
                fdate2 = datetime.strptime(date_str[0:-1], '%Y%j%H%M%S') - timedelta(seconds = 5*60)
                rm_pdates2 = np.where(fdate < np.asarray(pdates))
 #               print(pdates)
                pdates2 = np.copy(np.asarray(pdates))
                pdates2 = pdates2[rm_pdates2[0]].tolist()
          #      print(pdates2)
                p_ind  = pdates2.index(min(pdates2, key=lambda x: abs(x - fdate)))                                                             #Return index within data frame nearest to IR/VIS file scan
                p_ind2 = pdates2.index(min(pdates2, key=lambda x: abs(x - fdate2)))                                                            #Return index within data frame nearest to IR/VIS file scan
                p_ind  = np.min(rm_pdates2[0]) + p_ind
#                 p_ind  = pdates.index(min(pdates, key=lambda x: abs(x - fdate)))                                                             #Return index within data frame nearest to IR/VIS file scan
#                 p_ind2 = pdates.index(min(pdates, key=lambda x: abs(x - fdate2)))                                                            #Return index within data frame nearest to IR/VIS file scan
                min_index = np.min(np.where(np.asarray(df['YYYYMMDD']) == df['YYYYMMDD'][p_ind]))
                max_index = np.max(np.where(np.asarray(df['YYYYMMDD']) == df['YYYYMMDD'][p_ind]))
                if start_index == 60:
                  exit()
                traj_dat  = None
                if abs(pdates[p_ind] - fdate) > timedelta(seconds = 120):
                    if plt_traj == True:
                        traj_dat = {}
                        for ind0 in range(p_ind, max_index):
                            if pdates[ind0] not in rm_dates:
                                lon, lat, Z, tdates = doctss_read_lat_lon_alt_trajectory_particle_ncdf(df['YYYYMMDD'][p_ind], ind0-min_index, 
                                                                                                       inroot      = '../../../misc-data0/DCOTSS/traj/',
                                                                                                       bucket_name = plane_bucket_name,
                                                                                                       use_local   = True, 
                                                                                                       verbose     = verbose)
                                if len(lon) > 1:
                                    ir_date = datetime.strptime(date_str[0:-1], "%Y%j%H%M%S")
                                    mdate   = tdates.index(min(tdates, key=lambda x: abs(x - ir_date)))
                                    if traj_dat:
                                        traj_dat['lon'] = np.append(traj_dat['lon'], lon[mdate, :].flatten())
                                        traj_dat['lat'] = np.append(traj_dat['lat'], lat[mdate, :].flatten())
                                        traj_dat['Z']   = np.append(traj_dat['Z'], Z[mdate, :].flatten())
                                    else:
                                        traj_dat = {'lon':lon[mdate, :].flatten(), 'lat':lat[mdate, :].flatten(), 'Z':Z[mdate, :].flatten()}
                        if traj_dat == {}:
                            traj_dat =  {'lon':[-1], 'lat':[-1], 'Z':[-1]}

                    image    = img_from_three_modalities2(nc_file       = combined_nc_file,                                                    #Create 3-layered image
                                                          out_dir       = img_out_dir, 
                                                          no_plot_glm   = no_plot_glm, 
                                                          ir_min_value  = ir_min_value, 
                                                          ir_max_value  = ir_max_value, 
                                                          grid_data     = grid_data, 
                                                          colorblind    = colorblind, 
                                                          plt_img_name  = plt_img_name, 
                                                          plt_cbar      = plt_cbar, 
                                                          subset        = subset, 
                                                          latlon_domain = xy_bounds, 
                                                          region        = region, 
                                                          pthresh       = pthresh, 
                                                          plane_data    = [], 
                                                          traj_data     = traj_dat,
                                                          chk_day_night = chk_day_night,
                                                          proj          = proj, 
                                                          replot_img    = replot_img, 
                                                          verbose       = verbose)
                else:    
#                    p_data   = [df['G_LONG_MMS'][p_ind2:p_ind+1], df['G_LAT_MMS'][p_ind2:p_ind+1], df['HHH_H2O'][p_ind2:p_ind+1]]              #Extract necessary plane data
                    p_data    = [df['Longitude_ER2'][p_ind2:p_ind+1], df['Latitude_ER2'][p_ind2:p_ind+1], df['GPS_Altitude_ER2'][p_ind], df['HHH_H2O'][p_ind2:p_ind+1]] #Extract necessary plane data
                    if plt_traj == True:
                        traj_dat    = {}
                        for ind0 in range(p_ind, max_index):
                            if pdates[ind0] not in rm_dates:
                                lon, lat, Z, tdates = doctss_read_lat_lon_alt_trajectory_particle_ncdf(df['YYYYMMDD'][p_ind], ind0-min_index, 
                                                                                                       inroot      = '../../../misc-data0/DCOTSS/traj/',
                                                                                                       bucket_name = plane_bucket_name,
                                                                                                       use_local   = True, 
                                                                                                       verbose     = verbose)
                                if len(lon) > 1:
                                    mdate = tdates.index(min(tdates, key=lambda x: abs(x - pdates[p_ind])))
                                    if traj_dat:
                                        traj_dat['lon'] = np.append(traj_dat['lon'], lon[mdate, :].flatten())
                                        traj_dat['lat'] = np.append(traj_dat['lat'], lat[mdate, :].flatten())
                                        traj_dat['Z']   = np.append(traj_dat['Z'], Z[mdate, :].flatten())
                                    else:
                                        traj_dat = {'lon':lon[mdate, :].flatten(), 'lat':lat[mdate, :].flatten(), 'Z':Z[mdate, :].flatten()}
                        if traj_dat == {}:
                            traj_dat =  {'lon':[-1], 'lat':[-1], 'Z':[-1]}
                    
                    image    = img_from_three_modalities2(nc_file       = combined_nc_file,                                                    #Create 3-layered image
                                                          out_dir       = img_out_dir, 
                                                          no_plot_glm   = no_plot_glm, 
                                                          ir_min_value  = ir_min_value, 
                                                          ir_max_value  = ir_max_value, 
                                                          grid_data     = grid_data, 
                                                          colorblind    = colorblind, 
                                                          plt_img_name  = plt_img_name, 
                                                          plt_cbar      = plt_cbar, 
                                                          subset        = subset, 
                                                          latlon_domain = xy_bounds, 
                                                          region        = region, 
                                                          pthresh       = pthresh, 
                                                          plane_data    = p_data,
                                                          traj_data     = traj_dat,
                                                          proj          = proj, 
                                                          replot_img    = replot_img, 
                                                          chk_day_night = chk_day_night, 
                                                          verbose       = verbose)
            else:
                if real_time == True:
                    d_str0     = datetime.strptime(date_str[0:-1], "%Y%j%H%M%S").strftime("%Y%j%H%M%S")
                    if plt_model.endswith('.npy'):
                        plt_model2 = plt_model
                    else:
                        plt_model2 = os.path.join(plt_model, d_str0 + '_test_' + str.format('{0:.0f}', arr_shape) + '_results.npy')
                    try:
                        p = mp.Process(target = img_from_three_modalities2,  kwargs={'nc_file'      : combined_nc_file,
                                                                                     'out_dir'      : img_out_dir,
                                                                                     'no_plot_glm'  : no_plot_glm,
                                                                                     'no_plot_vis'  : no_write_vis, 
                                                                                     'ir_min_value' : ir_min_value,
                                                                                     'ir_max_value' : ir_max_value,
                                                                                     'grid_data'    : grid_data,
                                                                                     'colorblind'   : colorblind,
                                                                                     'plt_img_name' : plt_img_name,
                                                                                     'plt_cbar'     : plt_cbar,
                                                                                     'subset'       : subset,
                                                                                     'latlon_domain': xy_bounds, 
                                                                                     'region'       : region,
                                                                                     'pthresh'      : pthresh, 
                                                                                     'replot_img'   : replot_img,
                                                                                     'real_time'    : real_time, 
                                                                                     'plt_model'    : plt_model2,
                                                                                     'model'        : model, 
                                                                                     'chk_day_night': chk_day_night, 
                                                                                     'verbose'      : verbose})
                        p.start()
                    finally:
                        print('Image being created in background')
                else:
                    image        = img_from_three_modalities2(nc_file       = combined_nc_file,                                                #Create 3-layered image
                                                              out_dir       = img_out_dir, 
                                                              no_plot_glm   = no_plot_glm, 
                                                              no_plot_vis   = no_write_vis, 
                                                              ir_min_value  = ir_min_value, 
                                                              ir_max_value  = ir_max_value, 
                                                              grid_data     = grid_data, 
                                                              colorblind    = colorblind, 
                                                              plt_img_name  = plt_img_name, 
                                                              plt_cbar      = plt_cbar, 
                                                              subset        = subset,
                                                              latlon_domain = xy_bounds,  
                                                              region        = region, 
                                                              pthresh       = pthresh, 
                                                              proj          = proj, 
                                                              replot_img    = replot_img,
#                                                              plt_model     = plt_model, 
                                                              verbose       = verbose)

            if real_time == True:
                image = os.path.join(img_out_dir, date_str + '_' + re.sub('\.nc$', '', os.path.basename(combined_nc_file)) + '.png')
            fnames.append(image)                                                                                                               #Store the names of the images in a list
        else:
            if real_time == True:
                image = os.path.join(img_out_dir, date_str + '_' + re.sub('\.nc$', '', os.path.basename(combined_nc_file)) + '.png')
                fnames.append(image)                                                                                                           #Store the names of the images in a list

        if run_gcs == True:
            if write_combined_gcs == True:
                pref = os.path.join('combined_nc_dir', os.path.basename(os.path.dirname(combined_nc_file)))                                    #Extract path prefix to write files into google cloud
                write_to_gcs(out_bucket_name, pref, combined_nc_file, del_local = del_local)                                                   #Write combined netCDF file to google cloud storage bucket
            if no_plot == False and real_time == False:
                pref = os.path.join('layered_img_dir', re.split('layered_img_dir', os.path.dirname(image))[1])                                 #Extract path prefix to write files into google cloud
                write_to_gcs(out_bucket_name, pref, image, del_local = del_local)                                                              #Write sandwich image to google cloud storage bucket    
        else:
            if del_combined_nc == True:
                if verbose == True: print('Removing IR/VIS/GLM combined netCDF file')
                os.remove(combined_nc_file)
        
        if verbose == True: print()

    if no_write_glm == False and run_gcs == True and del_local == True:
        glm_files = sorted(glob.glob(os.path.join(glm_in_dir, '**', '*.nc'), recursive = True))
        [os.remove(x) for x in glm_files]                                                                                                      #Remove all downloaded GLM files
        os.rmdir(glm_in_dir)                                                                                                                   #Remove GLM data directory
   
    if run_gcs == True and del_local == True:    
        [os.remove(x) for x in ir_files]                                                                                                       #Remove all downloaded IR files
        os.rmdir(ir_dir)                                                                                                                       #Remove IR data directory
        if no_write_vis == False:
            [os.remove(x) for x in vis_files]                                                                                                  #Remove all downloaded VIS files
            os.rmdir(vis_dir)                                                                                                                  #Remove VIS data directory
        if no_write_irdiff == False:
            ird_files = sorted(glob.glob(os.path.join(irdiff_dir, '**', '*.nc'), recursive = True))
            [os.remove(x) for x in ird_files]                                                                                                  #Remove all downloaded Ch.8 IR files
            os.rmdir(irdiff_dir)                                                                                                               #Remove WV data directory
        if no_write_cirrus == False:
            c_files = sorted(glob.glob(os.path.join(cirrus_dir, '**', '*.nc'), recursive = True))
            [os.remove(x) for x in c_files]                                                                                                    #Remove all downloaded cirrus files
            os.rmdir(cirrus_dir)                                                                                                               #Remove GcirrusLM data directory
        if no_write_snowice == False:
            s_files = sorted(glob.glob(os.path.join(snowice_dir, '**', '*.nc'), recursive = True))
            [os.remove(x) for x in s_files]                                                                                                    #Remove all downloaded snowice files
            os.rmdir(snowice_dir)                                                                                                              #Remove snowice data directory
        if no_write_dirtyirdiff == False:
            dird_files = sorted(glob.glob(os.path.join(dirtyir_dir, '**', '*.nc'), recursive = True))
            [os.remove(x) for x in dird_files]                                                                                                 #Remove all downloaded dirty IR files 
            os.rmdir(dirtyir_dir)                                                                                                              #Remove dirty IR data directory
#     
    return(fnames, fnames2, proj, p)

def sort_goes_irvis_files(elem):
    d_str = re.split('_s|_', os.path.basename(elem))[3][0:11]
    return(d_str)

def sort_goes_comb_files(elem):
    d_str = re.split('_s|_', os.path.basename(elem))[5][0:11]
    return(d_str)

def main():
    run_create_image_from_three_modalities()
    
if __name__ == '__main__':
    main()
