#+
# Name:
#     run_mtg_create_image_from_three_modalities2.py
# Purpose:
#     This is a script to run (by importing) programs that grid the GLM data, combine IR, GLM and 
#     VIS data into a netCDF file, and create a 3 layered image that gets put into labelme software.
#     THIS CODE RUNS THE JOBS IN PARALLEL!!!!! If do not, want parallel processing the see run_create_image_from_three_modalities.py
# Calling sequence:
#     import run_mtg_create_image_from_three_modalities2
#     run_mtg_create_image_from_three_modalities2.run_mtg_create_image_from_three_modalities2()
# Input:
#     None.
# Functions:
#     glm_gridder                                     : Extracts and puts the GLM data and puts onto the VIS/IR data grid
#     mtg_combine_ir_glm_vis                          : Combines GLM, IR, VIS data and puts into netCDF file.
#     img_from_three_modaltities                      : Creates 3-layered image that is input into labelme software
#     mtg_create_image_from_three_modalities_parallel : Function that allows for multiprocessing.
#     find_images_not_created                         : Find images that were inexplicably skipped and rerun to make those images
# Output:
#     Creates netCDF file for GLM data gridded on MTG VIS grid, combined VIS, IR, and GLM netCDF file and
#     image file that gets put into labelme software to identify overshoot plumes.
# Keywords:
#     inroot               : STRING path to GLM, MTG visible and MTG IR data directories
#                            DEFAULT = '../../../mtg-data/20200513-14/'
#     glm_out_dir          : STRING path to output GLM data directory containing GLM data gridded on same grid as VIS data
#                            DEFAULT = '../../../mtg-data/out_dir/'
#     layered_dir          : STRING output directory path for netCDF file containing combined IR, VIS, and GLM data
#                            DEFAULT = '../../../mtg-data/combined_nc_dir/'
#     img_out_dir          : Output directory for the IR, GLM, VIS data plot that is input into labelme software.
#                            DEFAULT = '../../../mtg-data/layered_img_dir/'
#     no_plot_glm          : IF keyword set, do not plot the GLM data. Setting this keyword = True makes you plot only the VIS
#                            and IR data. DEFAULT is to set this keyword to True and only plot the VIS and IR data.
#     no_plot              : IF keyword set (True), do not plot the IR/VIS data. DEFAULT = False (plots the IR?VIS data)
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
#     domain_sector        : STRING specifying the domain sector to use to create maps (ex. 'h1', 't1', 'q4' or 'full'). DEFAULT = None -> use full
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
#     in_bucket_name       : Google cloud storage bucket to read raw IR/GLM/VIS files.
#                            DEFAULT = 'goes-data'
#     out_bucket_name      : Google cloud storage bucket to write intermediate netCDF files (IR, VIS, and GLM data on same grid) as well as IR+VIS sandwich images.
#                            DEFAULT = 'ir-vis-sandwhich'
#     replot_img           : IF keyword set (True), plot the IR/VIS sandwich image again
#                            DEFAULT = True
#     rewrite_nc           : IF keyword set (True), rewrite the combined ir/vis/glm netCDF file.
#                            DEFAULT = False (do not rewrite combined netCDF file if one exists. Write if it does not exist though)
#     append_nc            : IF keyword set (True), append the combined ir/vis/glm netCDF file.
#                            DEFAULT = True-> Append existing netCDF file
#     verbose              : BOOL keyword to specify whether or not to print verbose informational messages.
#                            DEFAULT = True which implies to print verbose informational messages
# Author and history:
#     John W. Cooney           2025-02-043. (Adapted from run_create_image_from_three_modalities2.py)
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
#import psutil
import multiprocessing as mp
import pandas as pd
#import metpy
#import xarray
import sys 
#sys.path.insert(1, '../')
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.getcwd()))
from glm_gridder.glm_gridder2 import glm_gridder2
from glm_gridder.mtg_combine_ir_glm_vis import mtg_combine_ir_glm_vis
from glm_gridder.img_from_three_modalities2 import img_from_three_modalities2
from new_model.gcs_processing import write_to_gcs, download_ncdf_gcs, list_gcs, load_csv_gcs
from glm_gridder.run_create_image_from_three_modalities import sort_mtg_irvis_files

def run_mtg_create_image_from_three_modalities2(inroot             = os.path.join('..', '..', '..', 'mtg-data', '20200513-14'), 
                                                glm_out_dir        = os.path.join('..', '..', '..', 'mtg-data', 'out_dir'), 
                                                layered_dir        = os.path.join('..', '..', '..', 'mtg-data', 'combined_nc_dir'), 
                                                img_out_dir        = os.path.join('..', '..', '..', 'mtg-data', 'layered_img_dir'), 
                                                no_plot_glm        = True, no_plot = False, 
                                                no_write_glm       = False, no_write_vis = False, no_write_irdiff = True, no_write_cirrus = True, no_write_snowice = True, no_write_dirtyirdiff = True, 
                                                xy_bounds          = [], 
                                                domain_sector      = None, 
                                                del_combined_nc    = False, universal_file = True, 
                                                ir_min_value       = 170, ir_max_value  = 230, 
                                                grid_data          = False, start_index = 0, 
                                                date_range         = [], 
                                                colorblind         = True, plt_img_name = False, 
                                                plt_cbar           = False, subset      = None,  region    = None,   
                                                run_gcs            = False, real_time   = False, del_local = False, 
                                                write_combined_gcs = True, 
                                                satellite          = 'MTG1',
                                                in_bucket_name     = 'mtg-data', out_bucket_name = 'ir-vis-sandwhich', plane_bucket_name = 'misc-data0',
                                                replot_img         = True, rewrite_nc = False, append_nc = True, verbose = True):
    '''
    Name:
        run_mtg_create_image_from_three_modalities2.py
    Purpose:
        This is a script to run (by importing) programs that grid the GLM data, combine IR, GLM and 
        VIS data into a netCDF file, and create a 3 layered image that gets put into labelme software.
        THIS CODE RUNS THE JOBS IN PARALLEL!!!!! If do not, want parallel processing the see run_create_image_from_three_modalities.py
    Calling sequence:
        import run_mtg_create_image_from_three_modalities2
        run_mtg_create_image_from_three_modalities2.run_mtg_create_image_from_three_modalities2()
    Input:
        None.
    Functions:
        glm_gridder                                     : Extracts and puts the GLM data and puts onto the VIS/IR data grid
        mtg_combine_ir_glm_vis                          : Combines GLM, IR, VIS data and puts into netCDF file.
        img_from_three_modaltities                      : Creates 3-layered image that is input into labelme software
        mtg_create_image_from_three_modalities_parallel : Function that allows for multiprocessing.
        find_images_not_created                         : Find images that were inexplicably skipped and rerun to make those images
    Output:
        Creates netCDF file for GLM data gridded on MTG VIS grid, combined VIS, IR, and GLM netCDF file and
        image file that gets put into labelme software to identify overshoot plumes.
    Keywords:
        inroot               : STRING path to GLM, MTG visible and MTG IR data directories
                               DEFAULT = '../../../mtg-data/20200513-14/'
        glm_out_dir          : STRING path to output GLM data directory containing GLM data gridded on same grid as VIS data
                               DEFAULT = '../../../mtg-data/out_dir/'
        layered_dir          : STRING output directory path for netCDF file containing combined IR, VIS, and GLM data
                               DEFAULT = '../../../mtg-data/combined_nc_dir/'
        img_out_dir          : Output directory for the IR, GLM, VIS data plot that is input into labelme software.
                               DEFAULT = '../../../mtg-data/layered_img_dir/'
        no_plot_glm          : IF keyword set, do not plot the GLM data. Setting this keyword = True makes you plot only the VIS
                               and IR data. DEFAULT is to set this keyword to True and only plot the VIS and IR data.
        no_plot              : IF keyword set (True), do not plot the IR/VIS data. DEFAULT = False (plots the IR/VIS data)
        no_write_glm         : IF keyword set, do not write the GLM data to the combined modality netCDF file. Setting this 
                               keyword = True makes you plot only the VIS and IR data. 
                               DEFAULT = False. True -> only write the VIS and IR data.
        no_write_vis         : IF keyword set, do not write the VIS data to the combined modality netCDF file. Setting this 
                               keyword = True makes you write only the IR and GLM data. 
                               DEFAULT = False. True -> only write the IR and GLM data.
        no_write_irdiff      : IF keyword set, write the difference between 10.3 micron and 6.2 micron to file.  
                               DEFAULT = True -> only write the VIS and IR and glm data.
        no_write_cirrus      : IF keyword set, write the difference between 1.37 micron to file.  
                               DEFAULT = True -> only write the VIS and IR and glm data.
        no_write_snowice     : IF keyword set, write the difference between 1.6 micron to file.  
                               DEFAULT = True -> only write the VIS and IR and glm data.
        no_write_dirtyirdiff : IF keyword set, write the difference between 12-10 micron to file.  
                               DEFAULT = True -> only write the VIS and IR and glm data.
        domain_sector        : STRING specifying the domain sector to use to create maps (ex. 'h1', 't1', 'q4' or 'full'). DEFAULT = None -> use full
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
        in_bucket_name       : Google cloud storage bucket to read raw IR/GLM/VIS files.
                               DEFAULT = 'goes-data'
        out_bucket_name      : Google cloud storage bucket to write intermediate netCDF files (IR, VIS, and GLM data on same grid) as well as IR+VIS sandwich images.
                               DEFAULT = 'ir-vis-sandwhich'
        replot_img           : IF keyword set (True), plot the IR/VIS sandwich image again
                               DEFAULT = True
        rewrite_nc           : IF keyword set (True), rewrite the combined ir/vis/glm netCDF file.
                               DEFAULT = False (do not rewrite combined netCDF file if one exists. Write if it does not exist though)
        append_nc            : IF keyword set (True), append the combined ir/vis/glm netCDF file.
                               DEFAULT = True-> Append existing netCDF file
        verbose              : BOOL keyword to specify whether or not to print verbose informational messages.
                               DEFAULT = True which implies to print verbose informational messages
    Author and history:
        John W. Cooney           2021-08-04. (Adapted from run_create_image_from_three_modalities to run code in parallel)
    '''
    if run_gcs:
        print('MTG data not setup to be run using Google Cloud Platform.')
        print('run_mtg_create_image_from_three_modalities2.py')
        exit()
    if domain_sector == None:
        sector  = 'F'                                                                                                                          #Set mesoscale sector string. Used for output directory and which input files to use
    else:
        sector = domain_sector.upper()                                                                                                         #Set domain sector string. Used for output directory and which input files to use
    str_sec = 'full'
    inroot      = os.path.realpath(inroot)                                                                                                     #Create link to real path so compatible with Mac
    #Try to extract date in different formats
    try:
        #Attempt to extract as "yyyymmdd"
        f_dates = os.path.basename(inroot)
        datetime.strptime(f_dates, "%Y%m%d")                                                                                                   #Check if date string from input root is in valid format
    except ValueError:
        try:
            #If that fails, try extracting from "/yyyy/mm/dd/"
            parts = os.path.normpath(inroot).split(os.sep)
            f_dates = f"{parts[-3]}{parts[-2]}{parts[-1]}"                                                                                     #Combine into "yyyymmdd"
            datetime.strptime(f_dates, "%Y%m%d")                                                                                               #Validate format
        except (ValueError, IndexError):
            print("Error: Could not parse date from inroot.")
            print(inroot)
            f_dates = None
            exit()
   
    glm_in_dir  = os.path.join(inroot, 'glm')                                                                                                  #Path to GLM data file
    glm_out_dir = os.path.realpath(os.path.join(glm_out_dir, f_dates, sector))                                                                 #Create link to real path so compatible with Mac
    layered_dir = os.path.realpath(os.path.join(layered_dir, f_dates))                                                                         #Create link to real path so compatible with Mac
    img_out_dir = os.path.realpath(img_out_dir)                                                                                                #Create link to real path so compatible with Mac

    if no_write_glm == False:
        os.makedirs(glm_in_dir,  exist_ok = True)
        os.makedirs(glm_out_dir, exist_ok = True)
    
    if plt_img_name == True:   
        nam = 'plt_img_name'
    else:
        nam = ''
    if plt_cbar == True:   
        cbar = 'plt_cbar'
    else:
        cbar = ''
    
    if os.path.basename(img_out_dir) == 'layered_img_dir':
        if subset != None: 
            img_out_dir = os.path.join(re.split(os.path.basename(img_out_dir), img_out_dir)[0], 'subset_domain', nam, cbar, os.path.basename(img_out_dir))  #Set output image path to subset domain files
        else:
            img_out_dir = os.path.join(re.split(os.path.basename(img_out_dir), img_out_dir)[0], nam, cbar, os.path.basename(img_out_dir))      #Set output image path to pixel grid files
        if grid_data == False: 
            img_out_dir = os.path.join(img_out_dir, 'pixel_grid', sector, f_dates)                                                             #Set output image path to pixel grid files
        else:
            img_out_dir = os.path.join(img_out_dir, 'lat_lon_grid', sector, f_dates)                                                           #Set output image path to pixel grid files
    else:
        if subset != None: 
            img_out_dir = os.path.join(img_out_dir, 'subset_domain')                                                                           #Set output image path to pixel grid files
        if grid_data == False: 
            img_out_dir = os.path.join(img_out_dir, 'pixel_grid', sector, nam, cbar, f_dates)                                                  #Set output image path to pixel grid files
        else:
            img_out_dir = os.path.join(img_out_dir, 'lat_lon_grid', sector, nam, cbar, f_dates)                                                #Set output image path to pixel grid files
    
    if no_plot == False: os.makedirs(os.path.dirname(img_out_dir), exist_ok = True)                                                            #Create output image directory file path if does not already exist
    if run_gcs == True:
        print('Not setup to for run_gcs with GCP. Changes must be made!!!')
        print('run_mtg_create_image_from_three_modalities2.py')
        exit()
        
#         if no_write_vis == False:
#             vis_files = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'vis'), ['-Rad' + sector + '-']))                                #Extract names of all of the GOES visible data files from the google cloud
#         ir_files  = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'ir'), ['-Rad' + sector + '-']))                                     #Extract names of all of the GOES infrared data files from the google cloud
#         if rewrite_nc == False or append_nc == True:
#             com_files = sorted(list_gcs(out_bucket_name, os.path.join('combined_nc_dir', f_dates), ['_COMBINED_', '_' + sector + '_']))        #Extract names of all of the GOES infrared data files from the google cloud
#         if no_write_irdiff == False:
#             ird_files = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'ir_diff'), ['-Rad' + sector + '-']))                            #Extract names of all of the GOES 6.2 micron infrared data files from the google cloud
#         if no_write_cirrus == False:
#             c_files = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'cirrus'), ['-Rad' + sector + '-']))                               #Extract names of all of the GOES 1.37 micron infrared data files from the google cloud
#         if no_write_snowice == False:
#             snowice_files = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'snowice'), ['-Rad' + sector + '-']))                        #Extract names of all of the GOES 1.6 micron infrared data files from the google cloud
#         if no_write_dirtyirdiff == False:
#             dirtyir_files = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'dirtyir'), ['-Rad' + sector + '-']))                        #Extract names of all of the GOES 12.3 micron infrared data files from the google cloud
# 
#         if no_write_glm == False:                                                                                                              #Must download GLM files to local storage
#             glm_files = sorted(list_gcs(in_bucket_name, os.path.join(f_dates, 'glm'), ['-LCFA', 'GLM']))                                       #Extract names of all of the GOES GLM data files from the google cloud
#             if real_time == True: 
#                 glm_files = glm_files[-15:]                                                                                                    #Download only the most recent GLM files if running code in real time
#             for g in glm_files: download_ncdf_gcs(in_bucket_name, g, glm_in_dir)
#         if real_time == True: 
#             ir_files  = [ir_files[-1]]                                                                                                         #Download only the most recent IR files if running code in real time
#             if no_write_vis == False:
#                 vis_files = [vis_files[-1]]                                                                                                    #Download only the most recent VIS files if running code in real time
#             if no_write_irdiff == False:
#                 ird_files = [ird_files[-1]]                                                                                                    #Download only the most recent 6.2 micron files if running code in real time
#             if no_write_cirrus == False:
#                 c_files = [c_files[-1]]                                                                                                        #Download only the most recent 1.37 micron files if running code in real time
#             if no_write_snowice == False:
#                 snowice_files = [snowice_files[-1]]                                                                                            #Download only the most recent 1.6 micron  files if running code in real time
#             if no_write_dirtyirdiff == False:
#                 dirtyir_files = [dirtyir_files[-1]]                                                                                            #Download only the most recent 12.3 micron  files if running code in real time
#         else:
#             ir_files2  = sorted(glob.glob(os.path.join(ir_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))                           #Extract names of all of the GOES IR data files already in local storage
#             if len(ir_files2) > 0:                                                                                                             #Only download files that are not already available on the local disk 
#                 fi  = sorted([os.path.basename(g) for g in ir_files])
#                 fi2 = sorted([os.path.basename(g) for g in ir_files2])
#                 ir_files0  = [ir_files[ff] for ff in range(len(fi)) if fi[ff] not in fi2]                                                      #Find ir_files not already stored on local disk
#                 ir_files   = ir_files0
#             if no_write_vis == False:
#                 vis_files2 = sorted(glob.glob(os.path.join(vis_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))                      #Extract names of all of the GOES visible data files already in local storage
#                 if len(vis_files2) > 0:                                                                                                        #Only download files that are not already available on the local disk 
#                     fv  = sorted([os.path.basename(g) for g in vis_files])
#                     fv2 = sorted([os.path.basename(g) for g in vis_files2])
#                     vis_files0 = [vis_files[ff] for ff in range(len(fv)) if fv[ff] not in fv2]                                                 #Find vis_files not already stored on local disk
#                     vis_files  = vis_files0
#                 for g in vis_files: download_ncdf_gcs(in_bucket_name, g, vis_dir)
#    
#         for g in ir_files: download_ncdf_gcs(in_bucket_name, g, ir_dir)
#         if no_write_irdiff == False:
#             if real_time != True:
#                 ird_files2 = sorted(glob.glob(os.path.join(irdiff_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))                   #Extract names of all of the GOES IR channel 8 data files
#                 if len(ird_files2) > 0:
#                     fi  = sorted([os.path.basename(g) for g in ird_files])
#                     fi2 = sorted([os.path.basename(g) for g in ird_files2])
#                     ird_files0 = [ird_files[ff] for ff in range(len(fi)) if fi[ff] not in fi2]                                                 #Find IR channel 8 files not already stored on local disk
#                     ird_files  = ird_files0
#             for g in ird_files: download_ncdf_gcs(in_bucket_name, g, irdiff_dir)                                                               #Download IR 6.2 micron channel files
#         
#         if no_write_cirrus == False:
#             if real_time != True:
#                 cirrus_files2 = sorted(glob.glob(os.path.join(cirrus_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))                #Extract names of all of the GOES IR channel 8 data files
#                 if len(cirrus_files2) > 0:
#                     fi  = sorted([os.path.basename(g) for g in c_files])
#                     fi2 = sorted([os.path.basename(g) for g in cirrus_files2])
#                     cirrus_files0 = [c_files[ff] for ff in range(len(fi)) if fi[ff] not in fi2]                                                #Find 1.37 micron files not already stored on local disk
#                     c_files       = cirrus_files0
#             for g in c_files: download_ncdf_gcs(in_bucket_name, g, cirrus_dir)                                                                 #Download IR 1.37 micron channel files
#         
#         if no_write_snowice == False:
#             if real_time != True:
#                 snowice_files2 = sorted(glob.glob(os.path.join(snowice_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))              #Extract names of all of the GOES 1.6 micron files
#                 if len(snowice_files2) > 0:
#                     fi  = sorted([os.path.basename(g) for g in snowice_files])
#                     fi2 = sorted([os.path.basename(g) for g in snowice_files2])
#                     snowice_files0 = [snowice_files[ff] for ff in range(len(fi)) if fi[ff] not in fi2]                                         #Find 1.6 micron files not already stored on local disk
#                     snowice_files  = snowice_files0
#             for g in snowice_files: download_ncdf_gcs(in_bucket_name, g, snowice_dir)                                                          #Download 1.6 micron channel files
#        
#         if no_write_dirtyirdiff == False:
#             if real_time != True:
#                 dirtyir_files2 = sorted(glob.glob(os.path.join(dirtyir_dir, '**', '*-Rad' + sector + '-*.nc'), recursive = True))              #Extract names of all of the GOES Dirty IR channel data files
#                 if len(dirtyir_files2) > 0:
#                     fi  = sorted([os.path.basename(g) for g in dirtyir_files])
#                     fi2 = sorted([os.path.basename(g) for g in dirtyir_files2])
#                     dirtyir_files0 = [dirtyir_files[ff] for ff in range(len(fi)) if fi[ff] not in fi2]                                         #Find dirty IR channel 12.3 micron files not already stored on local disk
#                     dirtyir_files  = dirtyir_files0
#             for g in dirtyir_files: download_ncdf_gcs(in_bucket_name, g, dirtyir_dir)                                                          #Download IR 12.3 micron channel files
#         
#         if rewrite_nc == False or append_nc == True:
#             com_files2 = sorted(glob.glob(os.path.join(layered_dir, '*' + '_' + sector + '_COMBINED_' + '*.nc'), recursive = True))            #Find combined netCDF files not already stored on local disk
#             if len(com_files2) > 0:
#                 fi  = sorted([os.path.basename(g) for g in com_files])
#                 fi2 = sorted([os.path.basename(g) for g in com_files2])
#                 com_files0 = [com_files[ff] for ff in range(len(fi)) if fi[ff] not in fi2]                                                     #Find combined netCDF files not already stored on local disk
#                 com_files  = com_files0
#             for g in com_files: download_ncdf_gcs(out_bucket_name, g, layered_dir)                                                             #Download combined netCDF files if exist and want them to be rewritten
# #         if len(xy_bounds) > 0 and domain_sector != None:
# #             dpath = os.path.join(re.split('combined_nc_dir', layered_dir)[0], 'sat_projection_files')
# #             if os.path.exists(dpath) == False:
# #                 sat_proj_files = sorted(list_gcs(in_bucket_name, 'sat_projection_files', ['.nc']))
# #                 if len(sat_proj_files) <= 0:
# #                     print('Satellite projection files not found???')
# #                     print(sat_proj_files)
# #                     print(in_bucket_name)
# #                     exit()
# #                 os.makedirs(dpath, exist_ok = True)  
# #                 for g in sat_proj_files: download_ncdf_gcs(in_bucket_name, g, dpath)
    
    sec = 'FD'
    vis_files  = sorted(glob.glob(os.path.join(inroot, '*MTI' + satellite[-1] + '*HRFI-' + sec + '-*-TRAIL-*.nc'), recursive = True), key = sort_mtg_irvis_files)             #Extract names of all of the MTG data files
    if no_write_vis == False:
        ir_files = vis_files
        if (len(vis_files) != len(ir_files) or len(vis_files) <= 0):                                                                           #Check to make sure same number of IR and VIS files in directory
            print('Number of visible and infrared data files in directory do not match or no files in directory!?')
            print(len(vis_files))
            print(len(ir_files))
            print(inroot)
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
                vis_dstr = [re.split('_|-|,|\+', os.path.basename(ir_files[f]))[23] for f in range(0, len(ir_files))]                          #Extract visibile date strings to determine missing VIS or IR data files
                ir_dstr = [re.split('_|-|,|\+', os.path.basename(ir_files[f]))[23] for f in range(0, len(ir_files))]                           #Extract IR date strings to determine missing VIS or IR data files
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
                            os.remove(vis_files[f])                                                                                            #Delete visible data file that does not have a matching IR data file scan
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
                            os.remove(ir_files[f])                                                                                             #Delete IR data file that does not have a matching IR data file scan
                vis_files = sorted(glob.glob(os.path.join(inroot, '*MTI' + satellite[-1] + '*HRFI-' + sec + '-*-TRAIL-*.nc'), recursive = True), key = sort_mtg_irvis_files) #Extract names of all of the MTG visible data files
                ir_files  = vis_files                                                                                                          #Extract names of all of the MTG IR data files
    else:
        ir_files  = sorted(glob.glob(os.path.join(inroot, '*MTI' + satellite[-1] + '*FDHSI-' + sec + '-*-TRAIL-*.nc'), recursive = True), key = sort_mtg_irvis_files) #Extract names of all of the MTG IR data files        
    if len(date_range) > 0:
        time0 = datetime.strptime(date_range[0], "%Y-%m-%d %H:%M:%S")                                                                          #Extract start date to find data within of the date range
        time1 = datetime.strptime(date_range[1], "%Y-%m-%d %H:%M:%S")                                                                          #Extract end date to find data within of the date range
        #Extract subset of raw VIS netCDF files within date_range
        start_index = 0
        end_index   = len(ir_files)
        for f in (range(0, len(ir_files))):
            file_attr = re.split('_|-|,|\+', os.path.basename(ir_files[f]))                                                                   
            date_str  = file_attr[23][8:14]
            if file_attr[23][0:14] >= "{:04d}{:02d}{:02d}{:02d}{:02d}".format(time0.year, time0.month, time0.day, time0.hour, time0.minute) and start_index == 0:
                if f == 0:
                    start_index = -1
                else:    
                    start_index = f
            if file_attr[23][0:14] >= "{:04d}{:02d}{:02d}{:02d}{:02d}".format(time1.year, time1.month, time1.day, time1.hour, time1.minute) and end_index == len(ir_files):
                if "{:02d}{:02d}{:02d}".format(time1.hour, time1.minute, time1.second) == date_str:
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
    if end_index == 0:
        print('No files within date range! Returning empty sets so this date is skipped.')
        return([], [], None)

    pool     = mp.Pool(2, maxtasksperchild = 2)                                                                                                #Set up multiprocessing threads
    results  = [pool.apply_async(mtg_create_image_from_three_modalities_parallel, args=(row, ir_files, vis_files, 
                                                                                        glm_in_dir, glm_out_dir, layered_dir,
                                                                                        img_out_dir, no_plot_glm, no_plot, 
                                                                                        no_write_glm, no_write_vis, no_write_irdiff, no_write_cirrus, no_write_snowice, no_write_dirtyirdiff, 
                                                                                        xy_bounds, del_combined_nc, universal_file, ir_min_value, ir_max_value, 
                                                                                        grid_data, colorblind, plt_img_name, domain_sector, satellite, 
                                                                                        plt_cbar, subset, region, run_gcs, write_combined_gcs, del_local, out_bucket_name, 
                                                                                        replot_img, rewrite_nc, append_nc, verbose)) for row in range(start_index, end_index)]
    results2  = [result.get() for result in results]
    fnames    = [x[0] for x in results2]                                                                                                       #Extract the image names created
    fnames2   = [x[1] for x in results2]                                                                                                       #Extract the combined netCDF file names created
    proj_data = [x[2] for x in results2]
    pool.close()                                                                                                                               #Close multiprocessing threads
    pool.join()                                                                                                                                #Wait until multi processed job finished before moving on threads
   
    if len(fnames2) != len(range(start_index, end_index)):
        print('Not all image files created???')
        if no_write_vis == False:
            rerun_vfiles = find_images_not_created(fnames, vis_files[start_index:end_index+1])                                                 #Find vis data files that were not created        
        else:
            rerun_vfiles = find_images_not_created(fnames, ir_files[start_index:end_index+1])                                                  #Find vis data files that were not created        
        print('VIS/IR image files not created', rerun_vfiles)
        rerun_ifiles = [ir_files[vis_files.index(rv)] for rv in rerun_vfiles]                                                                  #Find ir files corresponding to vis files
        pool         = mp.Pool(2, maxtasksperchild =2)                                                                                         #Set up multiprocessing threads
        results      = [pool.apply_async(mtg_create_image_from_three_modalities_parallel, args=(row, rerun_ifiles, rerun_vfiles, 
                                                                                                glm_in_dir, glm_out_dir, layered_dir,
                                                                                                img_out_dir, no_plot_glm, no_plot, 
                                                                                                no_write_glm, no_write_vis, no_write_irdiff, no_write_cirrus, no_write_snowice, no_write_dirtyirdiff, 
                                                                                                xy_bounds, del_combined_nc, universal_file, ir_min_value, ir_max_value, 
                                                                                                grid_data, colorblind, plt_img_name, domain_sector, satellite, 
                                                                                                plt_cbar, subset, region, run_gcs, write_combined_gcs, del_local, out_bucket_name, 
                                                                                                replot_img, rewrite_nc, append_nc, verbose)) for row in range(len(rerun_vfiles))]

        results2  = [result.get() for result in results]
        fnames.append([x[0] for x in results2])                                                                                                #Extract output image name
        fnames2.append([x[1] for x in results2])                                                                                               #Extract combined netCDF file name
        proj_data.append([x[2] for x in results2])                                                                                             #Extract satellite projection
        pool.close()                                                                                                                           #Close multiprocessing threads
        pool.join()                                                                                                                            #Wait until multi processed job finished before moving on threads
    if len(fnames2) != len(range(start_index, end_index)):
        print('Not all image files created???')
        exit()
        
    if len(fnames)  > 0 and (fnames.count(None) != len(fnames)):
        fnames = sorted(fnames)
    
    if len(fnames2) > 0:
        fnames2 = sorted(fnames2)
    if len(fnames) != len(fnames2):
        print('Number of image files do not match number of combined netCDF files created.')
        exit()
    
    
    if no_write_glm == False and run_gcs == True and del_local == True:
        print('Not setup to handle lightning data yet!!')
        exit()
        glm_files = sorted(glob.glob(os.path.join(glm_in_dir, '**', '*.nc'), recursive = True))
        [os.remove(x) for x in glm_files]                                                                                                      #Remove all downloaded GLM files
        os.rmdir(glm_in_dir)                                                                                                                   #Remove GLM data directory
   
    if run_gcs == True and del_local == True:    
        print('Not setup to handle writing to GCP')
        exit()
        [os.remove(x) for x in ir_files]                                                                                                       #Remove all downloaded IR files
        os.rmdir(ir_dir)                                                                                                                       #Remove IR data directory
        if no_write_vis == False:
            [os.remove(x) for x in vis_files]                                                                                                  #Remove all downloaded VIS files
            os.rmdir(vis_dir)                                                                                                                  #Remove VIS data directory
       
        if no_write_irdiff == False:
            ird_files = sorted(glob.glob(os.path.join(irdiff_dir, '**', '*.nc'), recursive = True))
            [os.remove(x) for x in ird_files]                                                                                                  #Remove all downloaded Ch.8 IR files
            os.rmdir(irdiff_dir)                                                                                                               #Remove Ch.8 data directory

        if no_write_cirrus == False:
            cirrus_files = sorted(glob.glob(os.path.join(cirrus_dir, '**', '*.nc'), recursive = True))
            [os.remove(x) for x in cirrus_files]                                                                                               #Remove all downloaded 1.37 micron files
            os.rmdir(cirrus_dir)                                                                                                               #Remove 1.37 micron data directory
        
        if no_write_snowice == False:
            snowice_files = sorted(glob.glob(os.path.join(snowice_dir, '**', '*.nc'), recursive = True))
            [os.remove(x) for x in snowice_files]                                                                                              #Remove all downloaded 1.6 micron files
            os.rmdir(snowice_dir)                                                                                                              #Remove 1.6 micron data directory
       
        if no_write_dirtyirdiff == False:
            dirtyir_files = sorted(glob.glob(os.path.join(dirtyir_dir, '**', '*.nc'), recursive = True))
            [os.remove(x) for x in dirtyir_files]                                                                                              #Remove all downloaded 12.3 micron files
            os.rmdir(dirtyir_dir)                                                                                                              #Remove 12.3 micron data directory

#     proc = psutil.Process()                                                                                                                  #This object concerns the current process
#     if len(proc.open_files()) > 1:        
#         for e in range(len(proc.open_files())):        
#             os.close(proc.open_files()[0][1])                                                                                                #Close all open files
#     
    results  = None
    results2 = None
    return(fnames, fnames2, proj_data[0])

def mtg_create_image_from_three_modalities_parallel(f, ir_files, vis_files, 
                                                    glm_in_dir, glm_out_dir, layered_dir,
                                                    img_out_dir, no_plot_glm, no_plot, 
                                                    no_write_glm, no_write_vis, no_write_irdiff, no_write_cirrus, no_write_snowice, no_write_dirtyirdiff, 
                                                    xy_bounds, del_combined_nc, universal_file, ir_min_value, ir_max_value, 
                                                    grid_data, colorblind, plt_img_name, domain_sector, satellite, 
                                                    plt_cbar, subset, region, run_gcs, write_combined_gcs, del_local, out_bucket_name, 
                                                    replot_img, rewrite_nc, append_nc, verbose):   
    from netCDF4 import Dataset
    ir_files  = sorted(ir_files)
    vis_files = sorted(vis_files)
    if domain_sector == None:
        sector = 'F'                                                                                                                           #Set mesoscale sector string. Used for output directory and which input files to use
    else:
        sector = domain_sector.upper()                                                                                                         #Set domain sector string. Used for output directory and which input files to use
    str_sec = 'full'

    fnames    = None                                                                                                                           #Initialize list to store the names of the image files
    fnames2   = None                                                                                                                           #Initialize list to store the names of the combined netCDF files
    file_attr = re.split('_|-|,|\+', os.path.basename(ir_files[f]))                                                                            #Split file string in order to extract date string of scan
    date_str  = file_attr[23]                                                                                                                  #Split file string in order to extract date string of scan

    if no_write_vis == False:
        file_attr2 = re.split('_|-|,|\+', os.path.basename(vis_files[f]))                                                                      #Split file string in order to extract date string of scan
        date_str2  = file_attr2[23]                                                                                                            #Split file string in order to extract date string of scan
    
        if (date_str2 != date_str):                                                                                                            #Throw error if not using matching IR and VIS files
            print('GOES IR and VIS file date strings do not match??')    
            print(date_str2)
            print(date_str)
            print(vis_files)
            print()
            print()
            print()
            print(ir_files)
            print()
            print(os.path.basename(vis_files[f]))
            print(os.path.basename(ir_files[f]))
            exit()    
#     if grid_data == True:
#         ir    = xarray.open_dataset(ir_files[f]).load()
#         lon_c = str(np.copy(ir['geospatial_lat_lon_extent'].geospatial_lon_center))
#         lat_c = str(np.copy(ir['geospatial_lat_lon_extent'].geospatial_lat_center))
#         dat   = ir.metpy.parse_cf('Rad')
#         proj  = [dat.metpy.cartopy_crs, dat.x.min(), dat.x.max(), dat.y.min(), dat.y.max()]
#         ir.close() 
#     else:
#         proj = None
    proj = None
    if no_write_glm == False:    
        print('Not setup to handle lightning data yet!!')
        exit()
        
        with Dataset(ir_files[f], 'r') as ir:                                                                                                  #Read in IR file to get mesoscale domain central latitudes and longitudes
            lon_c = str(np.copy(ir.variables['geospatial_lat_lon_extent'].geospatial_lon_center))                                              #Extract longitude center of mesoscale domain as STRING
            lat_c = str(np.copy(ir.variables['geospatial_lat_lon_extent'].geospatial_lat_center))                                              #Extract latitude center of mesoscale domain as STRING
            sat   = str(ir.getncattr('platform_ID'))
        file_attr0 = re.split('_' + sat + '_|-', os.path.basename(ir_files[f]))                                                                #Split IR file name string to create output file name
        out_nc     = os.path.join(layered_dir, file_attr0[0] + '_' + file_attr0[1] + '_' + str(file_attr0[2].split('Rad')[1]) + '_COMBINED_' + file_attr0[4])
        glm_file   = os.path.join(glm_out_dir, date_str + '_gridded_data.nc')                                                                  #Create output GLM directory path if does not already exist
        if os.path.isfile(out_nc) == False or rewrite_nc == True or append_nc == True:
            if sector[0].lower() == 'c' or sector[0].lower() == 'f':
                date_str3 = file_attr[4]
                d_range   = []#[date_str2, file_attr2[4]]                                                                                      #Set range of dates
            else:
                date_str3 = date_str
                d_range   = []
            glm_grid = glm_gridder2(outfile = glm_file, glm_root = glm_in_dir, no_plot = no_plot_glm, date_str = date_str3, date_range = d_range, ctr_lon0 = lon_c, ctr_lat0 = lat_c, sector = str_sec, verbose = verbose)
        if no_write_vis == False:
            combined_nc_file, arr_shape = mtg_combine_ir_glm_vis(infile               = ir_files[f],                                           #Create netCDF file containing IR, GLM and VIS data
                                                                 satellite            = satellite,
                                                                 domain_sector        = sector,
                                                                 layered_dir          = layered_dir,
                                                                 no_write_glm         = no_write_glm,  
                                                                 no_write_vis         = no_write_vis,
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
            combined_nc_file, arr_shape = mtg_combine_ir_glm_vis(infile               = ir_files[f],                                           #Create netCDF file containing IR, GLM and VIS data
                                                                 satellite            = satellite,
                                                                 domain_sector        = sector,
                                                                 layered_dir          = layered_dir,
                                                                 no_write_glm         = no_write_glm,  
                                                                 no_write_vis         = no_write_vis,
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
            combined_nc_file, arr_shape = mtg_combine_ir_glm_vis(infile               = ir_files[f],                                           #Create netCDF file containing IR, GLM and VIS data
                                                                 satellite            = satellite,
                                                                 domain_sector        = sector,
                                                                 layered_dir          = layered_dir,
                                                                 no_write_glm         = no_write_glm,  
                                                                 no_write_vis         = no_write_vis,  
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
            combined_nc_file, arr_shape = mtg_combine_ir_glm_vis(infile               = ir_files[f],                                           #Create netCDF file containing IR, GLM and VIS data
                                                                 satellite            = satellite,
                                                                 domain_sector        = sector,
                                                                 layered_dir          = layered_dir,
                                                                 no_write_glm         = no_write_glm,  
                                                                 no_write_vis         = no_write_vis,  
                                                                 no_write_irdiff      = no_write_irdiff, 
                                                                 no_write_cirrus      = no_write_cirrus, 
                                                                 no_write_snowice     = no_write_snowice, 
                                                                 no_write_dirtyirdiff = no_write_dirtyirdiff, 
                                                                 universal_file       = universal_file, 
                                                                 rewrite_nc           = rewrite_nc, 
                                                                 append_nc            = append_nc, 
                                                                 xy_bounds            = xy_bounds, 
                                                                 verbose              = verbose)
            
    fnames2 = combined_nc_file                                                                                                                 #Store the names of the images in a list
    if no_plot == False:
        image        = img_from_three_modalities2(nc_file       = combined_nc_file,                                                            #Create 3-layered image
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
                                                  proj          = proj, 
                                                  replot_img    = replot_img, 
                                                  verbose       = verbose)
        fnames = image                                                                                                                         #Store the names of the images in a list
    if run_gcs == True:    
        if write_combined_gcs == True:
            pref = os.path.join('combined_nc_dir', os.path.basename(os.path.dirname(combined_nc_file)))                                        #Extract path prefix to write files into google cloud
            write_to_gcs(out_bucket_name, pref, combined_nc_file, del_local = del_local)                                                       #Write combined netCDF file to google cloud storage bucket
        if no_plot == False:    
            pref = 'layered_img_dir' + os.sep + re.split('layered_img_dir' + os.sep, os.path.dirname(image))[1]                                #Extract path prefix to write files into google cloud
            write_to_gcs(out_bucket_name, pref, image, del_local = del_local)                                                                  #Write sandwich image to google cloud storage bucket    
    else:
        if del_combined_nc == True:
            if verbose == True: print('Removing IR/VIS/GLM combined netCDF file')
            os.remove(combined_nc_file)
    
    return(fnames, fnames2, proj)  

def find_images_not_created(img_files, vis_files):
    '''
    Find image files that were not created and return the visible file names that need to be re-run
  
    Args:
      img_files : List containing image file name and paths
      vis_files : List containing visible file names and paths for which we wanted to create images for.
    Keywords:
      None.
    Output:    
      Returns list of files of which we need to re-run plotting because they were not created
    '''
    idat_str = [re.split('_s|_', os.path.basename(img_files[f]))[-3] for f in img_files]                                                       #Extract date string of image files
    vdat_str = [re.split('_s|_', os.path.basename(vis_files[f]))[3]  for f in vis_files]                                                       #Extract date string of vis files
    rereun   = []
    for v in vdat_str:
        if v not in idat_str:
             rerun.append(v)                                                                                                                   #Append array with vis data file name so that know to rerun image creation
    
    if len(rerun) == len(vdat_str):
        print('All images not created???')
        print(idat_str[0])
        print(vdat_str[0])
        exit()
    return(rerun)
    
    
def main():
    run_create_image_from_three_modalities2()
    
if __name__ == '__main__':
    main()

