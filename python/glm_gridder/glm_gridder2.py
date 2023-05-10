#+
# Name:
#		glm_gridder.py
# Purpose:
#		This is a script to run (by importing) programs that grid the GLM data, combine IR, GLM and 
#     VIS data into a netCDF file, and create a 3 layered image that gets put into labelme software
# Calling sequence:
#     import glm_gridder
# Input:
#		None.
# Functions:
#     glm_gridder : Extracts and puts the GLM data and puts onto grid
# Output:
#		Creates netCDF file for all GLM data gridded together
# Keywords:
#     outfile    : STRING file name and path to gridded GLM output file
#     glm_root   : STRING path to input GLM data directory containing raw GLM data files
#     date_str   : STRING containing the date to find ± time window GLM files around.
#     date_range : List of 2 STRINGS containing the start and end date, respectively. This is required for full disk and conus scans.
#                  DEFAULT = [] -> just use date_range if supplied
#     twindow    : [minute, second] 2D LONG list containing time window surround date_str to search for GLM files
#                  DEFAULT = [2, 30] = 2 minutes and 30 seconds
#     no_plot    : IF keyword set (True), do not plot the GLM data to the screen.
#                  DEFAULT = True
#     sector     : STRING keyword used to specify domain sector.
#                  DEFAULT = 'meso'. (other options are 'conus' and 'full')
#     verbose    : BOOL keyword to specify whether or not to print verbose informational messages.
#                  DEFAULT = True which implies to print verbose informational messages
# Author and history:
#     Anna Cuddeback
#     John W. Cooney           2020-09-15. (Updated to include documentation and create as a function to 
#                                           give output file name and input directories)
#-
#

## package imports ##
import sys, os
import glmtools
from os import listdir
from glmtools.io.glm import GLMDataset
import numpy as np
from glmtools.io.imagery import open_glm_time_series
import matplotlib.pyplot as plt
#from glmtools.plot.grid import plot_glm_grid
from PIL import Image 
import argparse
from datetime import datetime, timedelta
from time import sleep
from glmtools_docs.examples.grid.make_GLM_grids import create_parser as create_grid_parser
from glmtools_docs.examples.grid.make_GLM_grids import grid_setup
from multiprocessing import freeze_support
import netCDF4
import re
import time
#print('glm_gridder.py Imports complete.')

def glm_gridder2(outfile     = None, 
                 glm_root    = '../../../data/goes-data/20200513-14/glm/', 
                 date_str    = None, 
                 date_range  = [], 
                 twindow     = [2, 30], 
                 no_plot     = True,
                 ctr_lon0    = None, 
                 ctr_lat0    = None,
                 sector      = 'meso',
                 verbose     = True):

    ## pipeline configuration ##
    # define output directory and file
    if outfile == None: outfile = '../../../data/out_dir/gridded_data.nc'                                                                   #Set default output file
    outfile = os.path.realpath(outfile)                                                                                                     #Create link to real path so compatible with Mac
    os.makedirs(os.path.dirname(outfile), exist_ok = True)	                                                                                #Make output directory if it does not exist		
    if os.path.isfile(outfile) == False:
        open(outfile, 'w').close()                                                                                                          #Create empty output file
        
        # define input file directory and generate list of filenames and paths
        if outfile == None: glm_root = '../../../data/raw_dir/'                                                                             #Cooney path = '../../../data/goes-data/20200513-14/glm_dir/'
        glm_root = os.path.realpath(glm_root)                                                                                               #Create link to real path so compatible with Mac
        
        #raw_dir = '/Users/jwcooney/python/data/goes-data/20200513-14/glm_dir/'
        files = [os.path.join(glm_root, f) for f in listdir(glm_root) if f.endswith('.nc')]                                                 #OLD will grab .DS_Store files too :os.path.join(glm_root, f) for f in listdir(glm_root)][:]                                                                       #Extract file path and name of GLM files
        if len(date_range) > 0:
            files      = sorted(files)
            date0      = datetime.strptime(date_range[0][0:-1],'%Y%j%H%M%S')
            date1      = date0 - timedelta(minutes = twindow[0], seconds = twindow[1])
            date0      = datetime.strptime(date_range[1][0:-1],'%Y%j%H%M%S')
            date2      = date0 + timedelta(minutes = twindow[0], seconds = twindow[1])
            glm_fdates = lambda x : datetime.strptime(((re.split('_s|_', os.path.basename(x)))[3])[0:-1],'%Y%j%H%M%S')
            files0     = [f for f in files if date1 <= glm_fdates(f) <= date2]
            files      = files0
        else:
            if date_str != None:
                files      = sorted(files)
                date0      = datetime.strptime(date_str[0:-1],'%Y%j%H%M%S')
                date1      = date0 - timedelta(minutes = twindow[0], seconds = twindow[1])
                date2      = date0 + timedelta(minutes = twindow[0], seconds = twindow[1])
                glm_fdates = lambda x : datetime.strptime(((re.split('_s|_', os.path.basename(x)))[3])[0:-1],'%Y%j%H%M%S')
                files0     = [f for f in files if date1 <= glm_fdates(f) <= date2]
                files      = files0
        if verbose == True: print('GLM gridder using ' + str(len(files)) + ' GLM files')
        if len(files) == 0:
            print('No GLM files found????')
            print(listdir(glm_root))
            exit()
        # define grid characteristics
#         grid_spec = ["--fixed_grid", "--split_events",
#                      "--goes_position", "east", "--goes_sector", "conus",
#                      "--dx=2.0", "--dy=2.0",
#                      # "--ctr_lat=33.5", "--ctr_lon=-101.5",
#                      ]
        if ctr_lon0 == None or ctr_lat0 == None:
            grid_spec = ["--fixed_grid", "--split_events",
                         "--goes_position", "east", "--goes_sector", sector,
                         "--dx=2.0", "--dy=2.0",
                         # "--ctr_lat=33.5", "--ctr_lon=-101.5",
                         ]
        else:
            grid_spec = ["--fixed_grid", "--split_events",
                         "--goes_position", "east", "--goes_sector", sector,
                         "--dx=2.0", "--dy=2.0",
                         "--ctr_lat=" + ctr_lat0, "--ctr_lon=" + ctr_lon0,
                         ]
        # set up arguments for gridder creation
        cmd_args = ["-o", outfile] + grid_spec + files
        grid_parser = create_grid_parser()
        grid_args = grid_parser.parse_args(cmd_args)    
        if verbose == True: print("Pipeline configured.")
        
        # create gridder for data processing
        freeze_support()
        gridder, glm_filenames, start_time, end_time, grid_kwargs = grid_setup(grid_args)
        the_grid = gridder(glm_filenames, start_time, end_time, **grid_kwargs)
        if verbose == True:
            print('start time = ' + str(start_time))
            print('end time = ' + str(end_time))
            print("Gridder declared.")
    ## gridding ##
    
    ## plotting ##
    # selection of data for plotting
#     if no_plot != True:
#         glm_data = netCDF4.Dataset(outfile, "r", "NETCDF4")
#         print("Grid created.")
#         goes_image_proj = np.copy(np.asarray(glm_data.variables['flash_extent_density'][:]))                                                #Extract Flash extent density
#         
#         # plot creation
#         plt.imshow(goes_image_proj, cmap="terrain")                                                                                         #Plot Flash extent density
#         
#         # plot styling
#         the_time_change = (end_time - start_time)                                                                                           #Calculate time difference between start and end of raw GLM data files
#         time_seconds = the_time_change.seconds % 60                                                                                         #Calculate time change in seconds
#         time_minutes = int((the_time_change.seconds - time_seconds)/60)                                                                     #Calculate time change in minutes
#         plt.title('Plot of ' + str(len(files)) + ' GLM event files spanning ' + str(time_minutes) +':' + str(time_seconds) + ' (mm:ss)')
#         plt.colorbar()
#         plt.show()                                                                                                                          #Project plot to screen
#         plt.close()                                                                                                                         #Close plot
#         
#         print("GLM image plotted.")

if __name__ == '__main__': sys.exit()

#exit()