#+
# Name:
#     run_all_plume_updraft_model_predict.py
# Purpose:
#     This is a script to run (by importing) programs to run previous models using checkpoint files for inputs that include
#     IR, VIS, GLM, and IR_DIFF channels. 
# Calling sequence:
#     import run_all_plume_updraft_model_predict
#     run_all_plume_updraft_model_predict.run_all_plume_updraft_model_predict()
# Input:
#     None.
# Functions:
#     run_all_plume_updraft_model_predict        : MAIN function that calls all of the subroutines 
#     run_tf_3_channel_plume_updraft_day_predict : Function to get tensor flow of 3 channel model run
#     run_tf_2_channel_plume_updraft_day_predict : Function to get tensor flow of 2 channel model run
#     run_tf_1_channel_plume_updraft_day_predict : Function to get tensor flow of 1 channel model run
# Output:
#     Numpy files with the model results, figure containing OT or plume locations, csv file yielding OT or plume locations, and time aggregated figures 
#     in cases of not real-time runs.
# Keywords:
#     verbose : BOOL keyword to specify whether or not to print verbose informational messages.
#               DEFAULT = True which implies to print verbose informational messages
# Notices:
#     Copyright 2023 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. 
#     All Rights Reserved.
# Disclaimers:
#     No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, 
#                  INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES 
#                  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE 
#                  WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT 
#                  DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, 
#                  HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
#                  DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT 
#                  "AS IS."
#  
# Waiver and Indemnity: RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, 
#                       AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, 
#                       EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE 
#                       OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND 
#                       SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER 
#                       SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
# Author and history:
#     John W. Cooney           2022-05-05.
#                              2023-06-05. MAJOR REVISIONS. 1) Added post-processing function that yields object ID numbers of OTs and AACPs as well as
#                                          OT IR-anvil brightness temperature differences. 2) Fixed issue that occurred in full disk and CONUS scanned
#                                          model runs. The issue occurred at the very edge of the domain where the satellite view is off in to space on the
#                                          edge of Earth. This issue was fixed prior to release for OTs but AACPs needed to be fixed in a unique way.
#                              2023-06-13. MINOR REVISIONS. 1) Fixed issue where user wants to correct the model inputs or model type and not everything 
#                                          like the correct optimal model to be called followed with it. 2) Pass optimal_threshold for a given model to be written
#                                          to output netCDF file rather than whatever is chosen by user for post-processing and plotting. The one chosen by
#                                          the user is still passed into the subroutines for plotting and post-processing and is referred to as the likelihood_threshold
#                                          in order to differentiate it from the optimal_thresh attribute in the raw model likelihood output variables. Users can
#                                          identify objects in post-processing by specifying their own likelihood thresholds, rather than being forced to use the
#                                          optimal thresholds found for a particular model. 3) Decrease size of files by improving compression by setting 
#                                          least_significant_digits keyword when creating the variables.
#                              2023-06-22. MAJOR REVISIONS. 1) Added tropopause temperature to post-processing functionality for OT model runs. This software downloads
#                                          GFS data from NCAR and then interpolates and smooths the tropopause temperature onto the satellite data grid before writing
#                                          the output to the netCDF files.
#                              2024-03-05. MAJOR REVISIONS. VERSION 2 of software!!! LARGE model performance enhancement, particularly with AACP detection. Added A LOT 
#                                          of new model input combinations as possibilities, including TROPDIFF, DIRTYIRDIFF, SNOWICE, and CIRRUS. Added the new and improved 
#                                          checkpoint files associated with those model runs. Optimal models and thresholds have been updated since the Version 1 release. 
#                                          Changed how we interpolate the GFS data onto the GOES grid. We now follow the methods outlined in Khlopenkov et al. (2021). New 
#                                          version changes how software decides between daytime and nighttime model runs. Previously used the maximum solar zenith angle within 
#                                          the domain and checked if that exceeded 85°, however, there was an issue of only nighttime models being used for CONUS domains 
#                                          due to how the satellite views the Earth and the data are gridded. Thus, now the software checks to see if more than 5% of the pixels 
#                                          in the domain have solar zenith angles that exceed 85$°. If they do, then the software acts as if the domain is nighttime and if 
#                                          not the software acts as if it is daytime within the domain. Fixed major issue with looping over post-processing dates and the month or 
#                                          year changed during a real-time or archived model run. New version speeds up post-processing for OTs. Set variables to None after using 
#                                          them in order to clear cached memory. Catch and hide RunTime warnings that the User does not need to worry about. 
#                              2024-04-23. MINOR REVISION. A bug was found when IR+GLM AACP model run using the configuration file. The bug was that the code was not using the
#                                          IR resolution data and was instead trying to run the at VIS resolution. Simple fix was made.
#                              2024-05-01. MAJOR REVISION. Added visualization product capabilities back into the software. In real-time mode, plots will be created in background.
#                                          In archived mode, plots are created at the end. NOTE, plotting will slow down the processing.       
#                              2024-05-06. MINOR REVISION. Tropopause temperature data now written for AACP runs in addition to OT runs. Will also check previous day for GFS files
#                              2024-10-07. MINOR REVISION. Setup to handle GOES mesoscale imagery at temporal frequency of 30 seconds. The code was previously setup to do 1 min
#                                          mesoscale files and checked for start scan minute. This looks at the start scan second now in filenames. Note, code is still unable to
#                                          handle M1 and M2 sectors simultaneously. This update is just to handle a single meso sector at 30 second frequency,
#                              2025-04-16. MAJOR REVISION. VERSION 3 of software!!! Major changes include making software more applicable to global applications as well as the
#                                          additions of MTG-1 and GOES-19 satellites into the software framework. Optimal models and thresholds have been updated since the Version 2
#                                          release for TROPDIFF and VIS+TROPDIFF OT models. A new default VIS+TROPDIFF OT model is included with this release. TROPDIFF and VIS+TROPDIFF
#                                          OT detection model inputs are now the default. Our team decided to change these to the default inputs because IR-only and IR+VIS was not
#                                          performing up to standards for global situations across various seasons. This change should hopefully improve model detections. We also
#                                          changed how we interpolate the GFS data onto the GOES grid. Previously, we used 10 GFS grid boxes to smooth the tropopause temperatures but
#                                          now we use 20. We chose to use additional grid boxes because we noticed that in some cases there were tropopause temperature discontinuities
#                                          which resulted in OT detections where the tropopause was warmer even though the underlying satellite imagery did not show an OT. Another change
#                                          that was added was that in the optimal run settings and not plotting the data, we delete the intermediate numpy files after using them. By
#                                          deleting these files, Users will save a lot of disk space.
#
#-

#### Environment Setup ###
# Package imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
#from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.regularizers import l1
from keras import backend
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.losses import BinaryCrossentropy
from keras.utils.vis_utils import plot_model, model_to_dot
from datetime import datetime, timedelta
import os
from os import listdir
import re
import glob
import cv2
from time import sleep
import time
from netCDF4 import Dataset
import netCDF4
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import figure
import argparse
#import cartopy
#import cartopy.crs as ccrs
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#from cartopy import feature
import pyproj                              
import pandas as pd
from PIL import Image 
from scipy.ndimage import label, generate_binary_structure
from threading import Thread
import argparse
import warnings
from multiprocessing import freeze_support
import multiprocessing as mp
import pygrib
#import metpy
import xarray
import sys 
#sys.path.insert(1, '../')
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(3, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'glmtools'))
sys.path.insert(4, os.path.dirname(os.getcwd()))
import glmtools
from glmtools.io.glm import GLMDataset
from glmtools.io.imagery import open_glm_time_series
from glmtools_docs.examples.grid.make_GLM_grids import create_parser as create_grid_parser
from glmtools_docs.examples.grid.make_GLM_grids import grid_setup
from new_model.run_2_channel_just_updraft_day_np_preprocess2 import build_imgs, build_subset_tensor, reconstruct_tensor_from_subset
from new_model.gcs_processing import write_to_gcs, download_model_chk_file, list_gcs_chkpnt, list_gcs, list_csv_gcs, load_csv_gcs, load_npy_blobs, load_npy_gcs, download_ncdf_gcs
from new_model.run_2_channel_just_updraft_day_np_preprocess2 import build_imgs, build_subset_tensor, reconstruct_tensor_from_subset, reconstruct_tensor_from_subset2, build_subset_tensor2, reconstruct_tensor_from_subset3, build_subset_tensor3
from new_model.gcs_processing import write_to_gcs, download_model_chk_file, list_gcs_chkpnt, list_gcs, list_csv_gcs, load_csv_gcs, load_npy_blobs, load_npy_gcs, download_ncdf_gcs
from new_model.run_tf_1_channel_plume_updraft_day_predict import run_tf_1_channel_plume_updraft_day_predict
from new_model.run_tf_2_channel_plume_updraft_day_predict import run_tf_2_channel_plume_updraft_day_predict
from new_model.run_tf_3_channel_plume_updraft_day_predict import run_tf_3_channel_plume_updraft_day_predict
from new_model.run_write_severe_storm_post_processing import run_write_severe_storm_post_processing, download_gfs_analysis_files
from new_model.run_write_gfs_trop_temp_to_combined_ncdf import run_write_gfs_trop_temp_to_combined_ncdf
from glm_gridder.run_download_goes_ir_vis_l1b_glm_l2_data import *
from glm_gridder.run_download_goes_ir_vis_l1b_glm_l2_data_parallel import *
from glm_gridder.run_create_image_from_three_modalities2 import *
from glm_gridder.run_create_image_from_three_modalities import * 
from glm_gridder.run_download_goes_ir_vis_l1b_glm_l2_data import *
from glm_gridder.run_download_goes_ir_vis_l1b_glm_l2_data_parallel import *
from glm_gridder import turbo_colormap
from glm_gridder.glm_data_processing import *
from glm_gridder.cpt_convert import loadCPT                                                                                        #Import the CPT convert function        
from gridrad.rdr_sat_utils_jwc import extract_us_lat_lon_region, sat_time_intervals, mtg_extract_data_sector_from_region
from glm_gridder.dataScaler import DataScaler
from EDA.create_vis_ir_numpy_arrays_from_netcdf_files import *
from EDA.unet import unet
from EDA.MultiResUnet import MultiResUnet
from EDA.create_vis_ir_numpy_arrays_from_netcdf_files2 import *
from visualize_results.visualize_time_aggregated_results import visualize_time_aggregated_results
from visualize_results.run_write_plot_model_result_predictions import write_plot_model_result_predictions
backend.clear_session()
tf.get_logger().setLevel('ERROR')
def run_all_plume_updraft_model_predict(verbose     = True,
                                        drcs        = False,
                                        essl        = False, 
                                        config_file = None):
  '''
      This is a script to run (by importing) programs to run previous models using checkpoint files for inputs that include
      IR, VIS, GLM, and IR_DIFF channels. 
  Args:
      None.
  Keywords:
      config_file        : STRING specifying path to configure file which allows user to not have to answer any questions when starting model.
                           DEFAULT = None -> answer questions to determine type of run
      drcs               : BOOL keyword to specify whether or not this will be used for DRCS (NASA Langley disaster coordinator response) purposes.
                           IF keyword set (True), use IR+VIS+GLM OT model. DRCS wants GLM data for time aggregated plotting and this is
                           easiest way to do that. So, essentially, this keyword just ensures GLM data are in output netCDF files.
                           DEFAULT = False which implies to use default OT model which is IR+VIS and no GLM data are included in output files.
      essl               : BOOL keyword to specify whether or not this will be used for ESSL purposes. ESSL has data stored in /yyyy/mm/dd/ format rather
                           software built format yyyymmdd. Because the software uses the yyyymdd to build other directories etc. off of the path
                           and ESSL may use this software as a way to download the data, this was easiest fix.
                           DEFAULT = False which implies data are stored using /yyyymmdd/. Set to True if data are stored in /yyyy/mm/dd/ format.
      verbose            : BOOL keyword to specify whether or not to print verbose informational messages.
                           DEFAULT = True which implies to print verbose informational messages
  Functions:
      run_all_plume_updraft_model_predict        : MAIN function that calls all of the subroutines 
      run_tf_3_channel_plume_updraft_day_predict : Function to get tensor flow of 3 channel model run
      run_tf_2_channel_plume_updraft_day_predict : Function to get tensor flow of 2 channel model run
      run_tf_1_channel_plume_updraft_day_predict : Function to get tensor flow of 1 channel model run
  Output:
      Model run output files
  '''  
  if sys.path[0] != '':
    os.chdir(sys.path[0])                                                                                                          #Change directory to the system path of file 

  use_native_ir = False                                                                                                            #Constant set up but will change to True if IR only model is being use
  if config_file == None:
    percent_omit = 20
    mod_check2 = 0
    counter2   = 0
    while mod_check2 == 0:
      counter2 = counter2 + 1
      if counter2 == 4:
        print('Too many failed attempts! Exiting program.')
        exit()
      rt0 = str(input('Is this a real-time run or not? (y/n): ')).replace("'", '')
      rt0 = ''.join(e for e in rt0 if e.isalnum())                                                                                 #Remove any special characters or spaces.
      if rt0.lower() == 'rt' or rt0.lower() == 'realtime' or rt0.lower() == '':
        rt0 = 'y'  
      if rt0[0].lower() == 'y':
        d_str1 = None
        d_str2 = None
        print('You have chosen to run the model in real time.')
        print()
        nhours = input('How many hours would you like the model to run for? Decimal numbers are accepted. ')
        try:
          nhours = float(nhours)
        except:
          print('Wrong type entered! Number of hours must either be set to an integer or floating point value. Please try again.')
          print('Format type entered = ' + str(type(nhours)))
          print()
          nhours = input('How many hours would you like the model to run for? Decimal numbers are accepted. ')
          try:
            nhours = float(nhours)
          except:
            print('Wrong type entered again! Number of hours must either be set to an integer or floating point value. Exiting program.')
            print('Value entered = ' + str(nhours))
            print('Format type entered = ' + str(type(nhours)))
            exit()
      else:
        print('You have chosen NOT to run the model in real time.')
        nhours = None
        d_str1 = str(input('Enter the model start date (ex. 2017-04-29 00:00:00): ')).replace("'", '').strip()
        d_str2 = str(input('Enter the model end date (ex. 2017-04-29 00:00:00): ')).replace("'", '').strip()
        if len(d_str1) <= 11:
          d_str1 = d_str1.strip()
          d_str1 = d_str1 + ' 00:00:00'
        if len(d_str2) <= 11:
          d_str2 = d_str2.strip()
          d_str2 = d_str2 + ' 00:00:00'
        if d_str2 < d_str1:
          print('End date cannot be before start date!!!! Please try again.')
          print('Start date entered = ' + d_str1)
          print('End date entered = ' + d_str2)
          print()
          d_str1 = str(input('Enter the model start date (ex. 2017-04-29 00:00:00): ')).replace("'", '').strip()
          d_str2 = str(input('Enter the model end date (ex. 2017-04-29 00:00:00): ')).replace("'", '').strip()
        if d_str2 < d_str1:
          print('End date cannot be before start date!!!! Exiting program.')
          print('Start date entered = ' + d_str1)
          print('End date entered = ' + d_str2)
          print()
          exit()
        try:                                                                                                                       #Make sure date1 is in proper format
          d1 = datetime.strptime(d_str1, "%Y-%m-%d %H:%M:%S")
        except:
          print('date1 not in correct format!!! Format must be YYYY-mm-dd HH:MM:SS. Exiting program. Please try again.') 
          print(d_str1)
          exit() 
        
        try:                                                                                                                       #Make sure date2 is in proper format
          d2 = datetime.strptime(d_str2, "%Y-%m-%d %H:%M:%S")
        except:
          print('date1 not in correct format!!! Format must be YYYY-mm-dd HH:MM:SS. Exiting program. Please try again.') 
          print(d_str1)
          exit() 
      
      counter   = 0                                                                                                                #Initialize variable to store the number of times the program attempted to ascertain information from the user 
      mod_check = 0                                                                                                                #Initialize variable to store whether or not the user entered the appropriate information 
      while mod_check == 0:
        counter = counter + 1
        if counter == 4:
          print('Too many failed attempts! Exiting program.')
          exit()
        mod_sat = str(input('Enter the satellite name (ex and default. goes-16): ')).replace("'", '')
        if mod_sat == '':
          mod_sat = 'goes-16'
          print('Using GOES-16 satellite data')
          print()
        mod_sat = ''.join(e for e in mod_sat if e.isalnum())                                                                       #Remove any special characters or spaces.
        if mod_sat[0:4].lower() == 'goes':
          mod_check = 1
          sat       = mod_sat[0:4].lower() + '-' + str(mod_sat[4:].lower())
          if sat.endswith('-'):
            mod_check = 0
            print('You must specify which goes satellite number!')
            print()
        elif mod_sat[0].lower() == 'g' and len(mod_sat) <= 3:
          mod_check = 1
          sat       = 'goes-' + str(mod_sat[1:].lower())
        elif mod_sat.lower() == 'seviri':
          mod_check = 1
          sat       = 'seviri'
        elif 'mtg' in mod_sat.lower() or 'mti' in mod_sat.lower():
          if '1' in mod_sat.lower():
              mod_check = 1
              sat       = 'mtg1'
          elif '2' in mod_sat.lower():    
              mod_check = 1
              sat       = 'mtg2'
          elif '3' in mod_sat.lower():    
              mod_check = 1
              sat       = 'mtg3'
          elif '4' in mod_sat.lower():    
              mod_check = 1
              sat       = 'mtg4'
          else:
              print('For MTG data, the software is only capable of handling data for MTG 1-4. Please try again.')
              print()
        else:
          print('Model is not currently set up to handle satellite specified. Please try again.')
          print()
      raw_data_root = os.path.realpath(os.path.join('..', '..', '..', re.split('-', sat)[0] + '-data'))                            #Set default directory for the stored raw data files. This may need to be changed if the raw data root directory differs from this.
      if sat[0:4] == 'goes':
        counter   = 0                                                                                                              #Initialize variable to store the number of times the program attempted to ascertain information from the user 
        mod_check = 0                                                                                                              #Initialize variable to store whether or not the user entered the appropriate information 
        while mod_check == 0:
          counter = counter + 1
          if counter == 4:
            print('Too many failed attempts! Exiting program.')
            exit()
          mod_sector = str(input('Enter the ' + sat + ' satellite sector to run model on (ex. M1, M2, F, C): ')).replace("'", '')
          if mod_sector != '':
            mod_sector = ''.join(e for e in mod_sector if e.isalnum())                                                             #Remove any special characters or spaces.
            if mod_sector[0].lower() == 'm':
              try:
                sector    = 'meso' + mod_sector[-1]
                if sector == 'meso1' or sector == 'meso2':
                  mod_check = 1
              except:
                sector    = 'meso'
                mod_check = 1
            elif mod_sector[0].lower() == 'c':
              sector    = 'conus'
              mod_check = 1
            elif mod_sector[0].lower() == 'f': 
              sector    = 'full'
              mod_check = 1
            else:
              print('GOES satellite sector specified is not available. Enter M1, M2, F, or C. Please try again.')
              print()
          else:
            print('You must enter satellite sector information. It cannot be left blank. Please try again.')
            print()
      elif sat.lower() == 'seviri':
        sector = 'full'
      elif 'mtg' in sat.lower():
        sector = 'full'
      else:
        sector = ''
      if (sat[0:4] == 'goes' and (sector == 'meso1' or sector == 'meso2')):
        xy_bounds = []
        region    = None
      else:  
        reg_bound = str(input('Is there a particular state or country you would like to contrain data to? If no, hit ENTER. If yes, enter the name of the state or country: ')).replace("'", '').strip()
        if reg_bound == '':
          xy_bounds = []
          region    = None
        else:  
          xy_bounds = []
          region    = extract_us_lat_lon_region('virginia')
          regions   = list(region.keys())
          if reg_bound.lower() not in regions and reg_bound.lower() != 'n' and reg_bound.lower() != 'no':
            print('State or country specified is not available???')
            print('You specified = ' + str(reg_bound.lower()))
            print('Regions possible = ' + str(sorted(regions)))
            print()
            counter   = 0                                                                                                          #Initialize variable to store the number of times the program attempted to ascertain information from the user 
            mod_check = 0                                                                                                          #Initialize variable to store whether or not the user entered the appropriate information 
            while mod_check == 0:
              counter = counter + 1
              if counter == 4:
                print('Too many failed attempts! Exiting program.')
                exit()
              reg_bound = str(input('Is there a particular state or country you would like to contrain data to? If no, hit ENTER. If yes, enter the name of the state or country: ')).replace("'", '').strip()
              if reg_bound.lower() in regions or reg_bound == '':
                mod_check = 1
                xy_bounds = []
              else:  
                print('State or country specified is not available???')
                print('You specified = ' + str(reg_bound.lower()))
                print()
          if reg_bound != '' and reg_bound.lower() != 'n':
            xy_bounds = region[reg_bound.lower()]
            region    = reg_bound.lower()
        if len(xy_bounds) == 0:
          mod_bound = str(input('Do you want to constrain data to particular longitude and latitude bounds? (y/n): ')).replace("'", '')
          if mod_bound == '':
            xy_bounds = []
            print('Data will be not be constrained to particular longitude and latitude bounds')
            print()
          else:  
            mod_bound = ''.join(e for e in mod_bound if e.isalnum())                                                               #Remove any special characters or spaces.
            if mod_bound[0].lower() == 'y':
              counter   = 0                                                                                                        #Initialize variable to store the number of times the program attempted to ascertain information from the user 
              mod_check = 0                                                                                                        #Initialize variable to store whether or not the user entered the appropriate information 
              while mod_check == 0:
                counter = counter + 1
                if counter == 4:
                  print('Too many failed attempts! Exiting program.')
                  exit()
                mod_x0 = input('Enter the minimum longitude point. Values must be between -180.0 and 180.0: ')
                try:
                  mod_x0 = float(mod_x0)
                  if mod_x0 >= -180.0 and mod_x0 <= 180.0:
                    x0        = mod_x0
                    mod_check = 1
                  else:  
                    print('Minimum longitude point must be between -180.0 and 180.0. Please try again.')
                    print()
                except:
                  print('Wrong data type entered! Please try again.')
                  print('Format type entered = ' + str(type(mod_x0)))
                  print()
              
              counter   = 0                                                                                                        #Initialize variable to store the number of times the program attempted to ascertain information from the user 
              mod_check = 0                                                                                                        #Initialize variable to store whether or not the user entered the appropriate information 
              while mod_check == 0:
                counter = counter + 1
                if counter == 4:
                  print('Too many failed attempts! Exiting program.')
                  exit()
                mod_x0 = input('Enter the maximum longitude point. Values must be between -180.0 and 180.0: ')
                try:
                  mod_x0 = float(mod_x0)
                  if mod_x0 >= -180.0 and mod_x0 <= 180.0:
                    if mod_x0 > x0:
                      x1        = mod_x0
                      mod_check = 1
                    else:
                      print('Maximum longitude point must be > minimum longitude point. Please try again.')
                      print('Minimum longitude point entered = ' + str(x0))
                      print('Maximum longitude point entered = ' + str(mod_x0))
                      print()
                  else:  
                    print('Maximum longitude point must be between minimum longitude point and 180.0. Please try again.')
                    print()
                except:
                  print('Wrong data type entered! Please try again.')
                  print('Format type entered = ' + str(type(mod_x0)))
                  print()
            
              counter   = 0                                                                                                        #Initialize variable to store the number of times the program attempted to ascertain information from the user 
              mod_check = 0                                                                                                        #Initialize variable to store whether or not the user entered the appropriate information 
              while mod_check == 0:
                counter = counter + 1
                if counter == 4:
                  print('Too many failed attempts! Exiting program.')
                  exit()
                mod_x0 = input('Enter the minimum latitude point. Values must be between -90.0 and 90.0: ')
                try:
                  mod_x0 = float(mod_x0)
                  if mod_x0 >= -90.0 and mod_x0 <= 90.0:
                    y0        = mod_x0
                    mod_check = 1
                  else:  
                    print('Minimum latitude point must be between -90.0 and 90.0. Please try again.')
                    print()
                except:
                  print('Wrong data type entered! Please try again.')
                  print('Format type entered = ' + str(type(mod_x0)))
                  print()
              
              counter   = 0                                                                                                        #Initialize variable to store the number of times the program attempted to ascertain information from the user 
              mod_check = 0                                                                                                        #Initialize variable to store whether or not the user entered the appropriate information 
              while mod_check == 0:
                counter = counter + 1
                if counter == 4:
                  print('Too many failed attempts! Exiting program.')
                  exit()
                mod_x0 = input('Enter the maximum latitude point. Values must be between -90.0 and 90.0: ')
                try:
                  mod_x0 = float(mod_x0)
                  if mod_x0 >= -90.0 and mod_x0 <= 90.0:
                    if mod_x0 > y0:
                      y1        = mod_x0
                      mod_check = 1
                    else:
                      print('Maximum latitude point must be > minimum latitude point. Please try again.')
                      print('Minimum latitude point entered = ' + str(y0))
                      print('Maximum latitude point entered = ' + str(mod_x0))
                      print()
                  else:  
                    print('Maximum latitude point must be between minimum latitude point and 90.0. Please try again.')
                except:
                  print('Wrong data type entered! Please try again.')
                  print('Format type entered = ' + str(type(mod_x0)))
                  print()
              
              xy_bounds = [x0, y0, x1, y1]
            else:
              xy_bounds = []
      
      counter   = 0                                                                                                                #Initialize variable to store the number of times the program attempted to ascertain information from the user 
      mod_check = 0                                                                                                                #Initialize variable to store whether or not the user entered the appropriate information 
      while mod_check == 0:
        counter = counter + 1
        if counter == 4:
          print('Too many failed attempts! Exiting program.')
          exit()
        mod_loc = str(input('Enter the desired severe weather indicator (OT or AACP): ')).replace("'", '')
        mod_loc = ''.join(e for e in mod_loc if e.isalnum())
        if mod_loc.lower() == 'updraft' or mod_loc.lower() == 'ot' or mod_loc.lower() == 'updrafts' or mod_loc.lower() == 'ots':
          mod_loc     = 'OT'
          use_updraft = True
          mod_check   = 1
        elif mod_loc.lower() == 'plume' or mod_loc.lower() == 'aacp' or mod_loc.lower() == 'plumes' or mod_loc.lower() == 'aacps' or mod_loc.lower() == 'warmplume' or mod_loc.lower() == 'warmplumes':
          mod_loc     = 'AACP'
          use_updraft = False
          mod_check   = 1
        else:
          print('Severe weather indicator entered is not available. Desired severe weather indicator must be OT or AACP!!!')
          print('You entered : ' + mod_loc)
          print('Please try again.')
          print()
      transition = 'n'
      day_night0 = str(input('Would you like the model to seamlessly transition between previously identified optimal models? (y/n): ')).replace("'", '')
      if day_night0 == '':
        day_night0 = 'y'
      day_night0 = ''.join(e for e in day_night0 if e.isalnum())                                                                   #Remove any special characters or spaces.
      if day_night0[0].lower() == 'y':
        print('You have chosen to transition between the best day time and night time ' + mod_loc + ' model')
        use_night  = True
        opt_params = 'y'
        transition = 'y'
        use_glm    = True
      else:
        day_night = str(input('Will this model be needed at night? (y/n): ')).replace("'", '')
        if day_night == '':
          day_night = 'y'
        day_night = ''.join(e for e in day_night if e.isalnum())                                                                   #Remove any special characters or spaces.
        if day_night[0].lower() == 'y':
          print('You have chosen to run a model through night time hours where VIS data will be unavailable.')
          use_night = True
        else:    
          print('You have chosen to run a model where results files will not exist for night time hours.')
          use_night = False
        
        print()
        opt_params = str(input('Would you like to use the optimal model and parameters for your choices thus far? (y/n): ')).replace("'", '')
        if opt_params == '':
          opt_params = 'y'
      
      opt_params = ''.join(e for e in opt_params if e.isalnum())                                                                   #Remove any special characters or spaces.
      if opt_params[0].lower() == 'y':
        if mod_loc == 'OT':
          if transition == 'y':
            day0    = best_model_checkpoint_and_thresh('best_day', mod_loc, native_ir = use_native_ir, drcs = drcs) 
            night0  = best_model_checkpoint_and_thresh('best_night', mod_loc, native_ir = use_native_ir, drcs = drcs)
            day_inp = re.split(r'\+', day0[0].lower())
            nig_inp = re.split(r'\+', night0[0].lower())
            use_trop        = False
            use_dirtyirdiff = False
            use_cirrus      = False
            use_snowice     = False
            use_glm         = False
            use_irdiff      = False
            if 'tropdiff' in day_inp:
              use_trop = True
            if 'wvirdiff' in day_inp:
              use_irdiff = True
            if 'dirtyirdiff' in day_inp:
              use_dirtyirdiff = True
            if 'cirrus' in day_inp:
              use_cirrus = True
            if 'snowice' in day_inp:
              use_snowice = True
            if 'glm' in day_inp:
              use_glm = True
            day   = {'mod_type': day0[1],   'mod_inputs': day0[0],   'use_chkpnt': os.path.realpath(day0[2]),   'pthresh': day0[3],   'use_night' : False, 'use_trop' : use_trop, 'use_dirtyirdiff' : use_dirtyirdiff, 'use_cirrus' : use_cirrus, 'use_snowice' : use_snowice, 'use_glm' : use_glm, 'use_irdiff' : use_irdiff}
            
            use_trop        = False
            use_dirtyirdiff = False
            use_cirrus      = False
            use_snowice     = False
            use_glm         = False
            use_irdiff      = False
            if 'tropdiff' in nig_inp:
              use_trop = True
            if 'wvirdiff' in nig_inp:
              use_irdiff = True
            if 'dirtyirdiff' in nig_inp:
              use_dirtyirdiff = True
            if 'cirrus' in nig_inp:
              use_cirrus = True
            if 'snowice' in nig_inp:
              use_snowice = True
            if 'glm' in nig_inp:
              use_glm = True
            night = {'mod_type': night0[1], 'mod_inputs': night0[0], 'use_chkpnt': os.path.realpath(night0[2]), 'pthresh': night0[3], 'use_night' : True, 'use_trop' : use_trop, 'use_dirtyirdiff' : use_dirtyirdiff, 'use_cirrus' : use_cirrus, 'use_snowice' : use_snowice, 'use_glm' : use_glm, 'use_irdiff' : use_irdiff}
            print(day['mod_inputs']   + ' ' + day['mod_type']   + ' OT model will be used during day time scans.')
            print(night['mod_inputs'] + ' ' + night['mod_type'] + ' OT model will be used during night time scans.')
            print()
          else:
            if use_night == False:
              day0        = best_model_checkpoint_and_thresh('best_day', mod_loc, native_ir = use_native_ir, drcs = drcs)
              mod_type    = day0[1]
              mod_inputs  = day0[0]
              pthresh     = day0[3]
              if 'GLM' in re.split(r'\+', mod_inputs):
                use_glm = True
              else:
                use_glm = False
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            else:
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh('best', mod_loc, native_ir = use_native_ir, drcs = drcs)
              mod_type    = day0[1]
              mod_inputs  = day0[0]
              pthresh     = day0[3]
              if 'GLM' in re.split(r'\+', mod_inputs):
                use_glm = True
              else:
                use_glm = False
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh

            print('You have chosen to run an ' + mod_inputs + ' ' + mod_type + ' OT model') 
        else:
          if transition == 'y':
            day0    = best_model_checkpoint_and_thresh('best_day', mod_loc, native_ir = use_native_ir, drcs = drcs) 
            night0  = best_model_checkpoint_and_thresh('best_night', mod_loc, native_ir = use_native_ir, drcs = drcs)
            day_inp = re.split(r'\+', day0[0].lower())
            nig_inp = re.split(r'\+', night0[0].lower())
            use_trop        = False
            use_dirtyirdiff = False
            use_cirrus      = False
            use_snowice     = False
            use_glm         = False
            use_irdiff      = False
            if 'tropdiff' in day_inp:
              use_trop = True
            if 'wvirdiff' in day_inp:
              use_irdiff = True
            if 'dirtyirdiff' in day_inp:
              use_dirtyirdiff = True
            if 'cirrus' in day_inp:
              use_cirrus = True
            if 'snowice' in day_inp:
              use_snowice = True
            if 'glm' in day_inp:
              use_glm = True
            day   = {'mod_type': day0[1],   'mod_inputs': day0[0],   'use_chkpnt': os.path.realpath(day0[2]),   'pthresh': day0[3],   'use_night' : False, 'use_trop' : use_trop, 'use_dirtyirdiff' : use_dirtyirdiff, 'use_cirrus' : use_cirrus, 'use_snowice' : use_snowice, 'use_glm' : use_glm, 'use_irdiff' : use_irdiff}
            
            use_trop        = False
            use_dirtyirdiff = False
            use_cirrus      = False
            use_snowice     = False
            use_glm         = False
            use_irdiff      = False
            if 'tropdiff' in nig_inp:
              use_trop = True
            if 'wvirdiff' in nig_inp:
              use_irdiff = True
            if 'dirtyirdiff' in nig_inp:
              use_dirtyirdiff = True
            if 'cirrus' in nig_inp:
              use_cirrus = True
            if 'snowice' in nig_inp:
              use_snowice = True
            if 'glm' in nig_inp:
              use_glm = True
            night = {'mod_type': night0[1], 'mod_inputs': night0[0], 'use_chkpnt': os.path.realpath(night0[2]), 'pthresh': night0[3], 'use_night' : True, 'use_trop' : use_trop, 'use_dirtyirdiff' : use_dirtyirdiff, 'use_cirrus' : use_cirrus, 'use_snowice' : use_snowice, 'use_glm' : use_glm, 'use_irdiff' : use_irdiff}
            print(day['mod_inputs']   + ' ' + day['mod_type']   + ' AACP model will be used during day time scans.')
            print(night['mod_inputs'] + ' ' + night['mod_type'] + ' AACP model will be used during night time scans.')
            print()          
          else:  
            if use_night == False:
              day0        = best_model_checkpoint_and_thresh('best_day', mod_loc, native_ir = use_native_ir, drcs = drcs)
              mod_type    = day0[1]
              mod_inputs  = day0[0]
              pthresh     = day0[3]
              if 'GLM' in re.split(r'\+', mod_inputs):
                use_glm = True
              else:
                use_glm = False
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            else:  
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh('best', mod_loc, native_ir = use_native_ir, drcs = drcs)
              mod_type    = day0[1]
              mod_inputs  = day0[0]
              pthresh     = day0[3]
              if 'GLM' in re.split(r'\+', mod_inputs):
                use_glm = True
              else:
                use_glm = False
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            print('You have chosen to run an ' + mod_inputs + ' ' + mod_type + ' AACP model') 
        print()
      else:  
        if mod_loc == 'OT':
          counter   = 0                                                                                                            #Initialize variable to store the number of times the program attempted to ascertain information from the user 
          mod_check = 0                                                                                                            #Initialize variable to store whether or not the user entered the appropriate information 
          while mod_check == 0:
            counter = counter + 1
            if counter == 4:
              print('Too many failed attempts! Exiting program.')
              print('You must enter multiresunet, unet, or attentionunet ONLY.')
              exit()
            mod_type = str(input('Enter the model inputs you would like to use. Options are multiresunet, unet, attentionunet: ')).replace("'", '')    
            mod_type = ''.join(e for e in mod_type if e.isalnum())
            if mod_type.lower() == 'multiresunet':  
              mod_check = 1
              mod_type  = 'multiresunet'
            elif mod_type.lower() == 'unet':  
              mod_check = 1
              mod_type = 'unet'
            elif mod_type.lower() == 'attentionunet':
              mod_check = 1
              mod_type  = 'attentionunet'
            else:
              print('Model type entered is not available. Options are multiresunet, unet, or attentionunet. Please try again.')
              print()
        else:
          print('multiresunet model will be used since it is the only model type available for AACP detections.')
          print()
          mod_type  = 'multiresunet'
        
        counter   = 0                                                                                                              #Initialize variable to store the number of times the program attempted to ascertain information from the user 
        mod_check = 0                                                                                                              #Initialize variable to store whether or not the user entered the appropriate information 
        while mod_check == 0:
          counter = counter + 1
          if counter == 4:
            print('Too many failed attempts! Exiting program.')
            print('You must enter IR, IR+VIS, IR+GLM, IR+IRDIFF, or IR+VIS+GLM ONLY.')
            exit()
          mod_inputs = str(input('Enter the model inputs you would like to use (ex. IR+VIS): ')).replace("'", '')
          str_list   = re.split(',| ', mod_inputs)
          while '' in str_list:
            str_list.remove('')
          
          mod_inputs = '+'.join(str_list)
          if mod_inputs.lower() == 'ir':  
            mod_inputs = 'IR'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                use_native_ir = True
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':  
                use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                pthresh     = 0.45
                opt_pthres0 = pthresh
              elif mod_type == 'attentionunet':  
                use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                print('Optimal pthresh must still be entered!!!')
                print(mod_inputs)
                exit()
            else:
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'ir+vis' or mod_inputs.lower() == 'vis+ir':  
            mod_inputs = 'IR+VIS'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                pthresh     = 0.35
                opt_pthres0 = pthresh
              elif mod_type == 'attentionunet':
                use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                pthresh     = 0.65
                opt_pthres0 = pthresh
            else:
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'ir+glm' or mod_inputs.lower() == 'glm+ir':
            mod_inputs = 'IR+GLM'
            mod_check  = 1
            use_glm    = True
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                use_native_ir = True
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_glm', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                pthresh     = 0.45
                opt_pthres0 = pthresh
              elif mod_type == 'attentionunet':
                use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_glm', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                print('Optimal pthresh must still be entered!!!')
                print(mod_inputs)
                exit()
            else:
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'ir+irdiff' or mod_inputs.lower() == 'irdiff+ir' or mod_inputs.lower() == 'wvirdiff+ir' or mod_inputs.lower() == 'ir+wvirdiff':  
            mod_inputs = 'IR+WVIRDIFF'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                use_native_ir = True
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_irdiff', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                pthresh     = 0.45
                opt_pthres0 = pthresh
              elif mod_type == 'attentionunet':
                use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_irdiff', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                print('Optimal pthresh must still be entered!!!')
                print(mod_inputs)
                exit()
            else:
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'ir+dirtyirdiff' or mod_inputs.lower() == 'dirtyirdiff+ir':  
            mod_inputs = 'IR+DIRTYIRDIFF'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                use_native_ir = True
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                print('No unet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
              elif mod_type == 'attentionunet':
                print('No attentionunet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
            else:
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'tropdiff':  
            mod_inputs = 'TROPDIFF'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                use_native_ir = True
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                print('No unet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
              elif mod_type == 'attentionunet':
                print('No attentionunet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
            else:
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'ir+snowice' or mod_inputs.lower() == 'snowice+ir':  
            mod_inputs = 'IR+SNOWICE'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                print('No unet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
              elif mod_type == 'attentionunet':
                print('No attentionunet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
            else:
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'ir+cirrus' or mod_inputs.lower() == 'cirrus+ir':  
            mod_inputs = 'IR+CIRRUS'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                print('No unet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
              elif mod_type == 'attentionunet':
                print('No attentionunet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
            else:
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'ir+tropdiff' or mod_inputs.lower() == 'tropdiff+ir':  
            mod_inputs = 'IR+TROPDIFF'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                use_native_ir = True
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                print('No unet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
              elif mod_type == 'attentionunet':
                print('No attentionunet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
            else:
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'vis+tropdiff' or mod_inputs.lower() == 'tropdiff+vis':  
            mod_inputs = 'VIS+TROPDIFF'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                print('No unet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
              elif mod_type == 'attentionunet':
                print('No attentionunet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
            else:
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'tropdiff+glm' or mod_inputs.lower() == 'glm+tropdiff':  
            mod_inputs = 'TROPDIFF+GLM'
            mod_check  = 1
            use_glm    = True
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                use_native_ir = True
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                print('No unet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
              elif mod_type == 'attentionunet':
                print('No attentionunet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
            else:
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'tropdiff+dirtyirdiff' or mod_inputs.lower() == 'dirtyirdiff+tropdiff':  
            mod_inputs = 'TROPDIFF+DIRTYIRDIFF'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                use_native_ir = True
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                print('No unet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
              elif mod_type == 'attentionunet':
                print('No attentionunet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
            else:
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif 'ir' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'tropdiff' in mod_inputs.lower(): 
            mod_inputs = 'IR+VIS+TROPDIFF'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                print('No unet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
              elif mod_type == 'attentionunet':
                print('No attentionunet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
            else:
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif 'glm' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'tropdiff' in mod_inputs.lower(): 
            mod_inputs = 'VIS+TROPDIFF+GLM'
            mod_check  = 1
            use_glm    = True
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                print('No unet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
              elif mod_type == 'attentionunet':
                print('No attentionunet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
            else:
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif 'ir' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'dirtyirdiff' in mod_inputs.lower(): 
            mod_inputs = 'IR+VIS+DIRTYIRDIFF'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                print('No unet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
              elif mod_type == 'attentionunet':
                print('No attentionunet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
            else:
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif 'dirtyirdiff' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'tropdiff' in mod_inputs.lower(): 
            mod_inputs = 'VIS+TROPDIFF+DIRTYIRDIFF'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                print('No unet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
              elif mod_type == 'attentionunet':
                print('No attentionunet model available for specified inputs!!!')
                print(mod_inputs)
                exit()
            else:
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          elif 'ir' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'glm' in mod_inputs.lower(): 
            mod_inputs = 'IR+VIS+GLM'
            mod_check  = 1
            use_glm    = True
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh     = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh
              elif mod_type == 'unet':
                use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis_glm', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                pthresh     = 0.45
                opt_pthres0 = pthresh
              elif mod_type == 'attentionunet':
                use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis_glm', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                pthresh     = 0.20
                opt_pthres0 = pthresh
            else:
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
          else:
            print('Model inputs entered are not available. Options are IR, IR+VIS, IR+GLM, IR+WVIRDIFF, TROPDIFF, IR+SNOWICE, IR+CIRRUS, IR+DIRTYIRDIFF, IR+TROPDIFF, VIS+TROPDIFF, TROPDIFF+GLM, IR+VIS+TROPDIFF, VIS+TROPDIFF+GLM, IR+VIS+DIRTYIRDIFF, TROPDIFF+DIRTYIRDIFF, VIS+TROPDIFF+DIRTYIRDIFF, and IR+VIS+GLM. Please try again.')
            print()
      if transition != 'y':
        opt_pthresh = str(input('Would you like to use the previously identified optimal ' + mod_loc + ' likelihood value for the ' + mod_inputs + ' ' + mod_type + ' model? (y/n): ')).replace("'", '')
        if opt_pthresh == '':
          opt_pthresh = 'y'
        
        opt_pthresh = ''.join(e for e in opt_pthresh if e.isalnum())
        if opt_pthresh[0].lower() != 'y':
          counter   = 0                                                                                                            #Initialize variable to store the number of times the program attempted to ascertain information from the user 
          mod_check = 0                                                                                                            #Initialize variable to store whether or not the user entered the appropriate information 
          while mod_check == 0:
            counter = counter + 1
            if counter == 4:
              print('Too many failed attempts! Exiting program.')
              exit()
            opt_pthresh = input('Enter the likelihood value you would like to use (0-1): ')
            try:
              opt_pthresh = float(opt_pthresh)
              if opt_pthresh < 0 or opt_pthresh > 1:
                print('Likelihood value must be between 0 and 1. Please try again.')
                print()
              else:
                mod_check = 1
                pthresh   = opt_pthresh
            except:
              print('Wrong type entered! Likelihood value must either be set to an integer or floating point value between 0 and 1. Please try again.')
              print('Format type entered = ' + str(type(opt_pthresh)))
              print()
        
        print()
      if sat[0:4] == 'goes':
        if d_str1 != None and d_str2 != None:
          dd = str(input('Do you need to download the satellite data files? (y/n): ')).replace("'", '')
          if dd == '':
            dd = 'y'
          dd = ''.join(e for e in dd if e.isalnum())
          if dd[0].lower() == 'y':
            no_download = False
          else:
            no_download    = True
            raw_data_root0 = str(input('Enter the root directory to the raw IR/VIS/GLM data files. Do not include subdirectories for dates or ir, glm, vis, wv (ex. and default ' + raw_data_root + '). Just hit ENTER to use default : ')).replace("'", '')
            if raw_data_root0.lower() != '':
              raw_data_root = raw_data_root0
        else:
          no_download = False
      else:
        if sat[0:3] == 'mtg' or sat[0:3] == 'mti':
          dd = str(input('Do you need to download the satellite data files? (y/n): ')).replace("'", '')
          if dd == '':
            dd = 'y'
          dd = ''.join(e for e in dd if e.isalnum())
          if dd[0].lower() == 'y':
            no_download = False
          else:
            no_download    = True
            raw_data_root0 = str(input('Enter the root directory to the raw IR/VIS/GLM data files. Do not include subdirectories for dates or ir, glm, vis, wv (ex. and default ' + raw_data_root + '). Just hit ENTER to use default : ')).replace("'", '')
            if raw_data_root0.lower() != '':
              raw_data_root = raw_data_root0        
        else:
          no_download = False
     
      raw_data_root00 = raw_data_root
      if rt0[0].lower() != 'y' and no_download == True:
        counter   = 0                                                                                                              #Initialize variable to store the number of times the program attempted to ascertain information from the user 
        mod_check = 0                                                                                                              #Initialize variable to store whether or not the user entered the appropriate information 
        while mod_check == 0:
          counter = counter + 1
          if counter == 4:
            print('Too many failed attempts! Exiting program.')
            print('Raw data file root not correct because they do not exist!!')
            print('Do not include subdirectories for dates or ir, glm, vis, wv')
            exit()
          if os.path.realpath(raw_data_root).endswith('ir') or os.path.realpath(raw_data_root).endswith('vis') or os.path.realpath(raw_data_root).endswith('glm') or os.path.realpath(raw_data_root).endswith('wv'):
            raw_data_root0 = os.path.dirname(os.path.realpath(raw_data_root))
          else:
            raw_data_root0 = os.path.realpath(raw_data_root)
          try:
            q             = int(os.path.basename(raw_data_root0))                                                                  #Checks if root path mistakenly included a date
            raw_data_root = os.path.dirname(raw_data_root0)                                                                        #Remove date string from path
          except:
            q             = ''                                                                                                     #Flag to show that path did not a include date if it check for date fails
         
          if os.path.exists(os.path.join(raw_data_root, d1.strftime('%Y%m%d'), 'ir')):
            mod_check = 1
          else:
            print('IR data directory within root path does not exist. Please enter the root directory to the raw IR/VIS/GLM data files.')
            print('Current path specified = ' + raw_data_root)
            raw_data_root = str(input('Enter the root directory to the raw IR/VIS/GLM data files (ex.  ' + raw_data_root00 + '): ')).replace("'", '')
      
      if rt0[0].lower() != 'y':
        dd = str(input('Would you like to write the model results files to the Google Cloud? (y/n): ')).replace("'", '')
        if dd == '':
          dd = 'n'
        dd = ''.join(e for e in dd if e.isalnum())
        if dd[0].lower() == 'y':
          run_gcs = True
          dd2     = str(input('Would you like to delete the locally stored copy of the files after writing them to the Google Cloud? (y/n): ')).replace("'", '')
          dd2     = ''.join(e for e in dd if e.isalnum())
          if dd2[0].lower() == 'y':
            del_local = True
            use_local = False
          else:
            del_local = False  
            use_local = True
        else:
          run_gcs   = False
          del_local = False  
          use_local = True
      else:
        run_gcs   = False
        del_local = False  
        use_local = True
        
      dd = str(input('Would you like to plot model results on top of IR/VIS sandwich images for each individual scene? Note, time to plot may cause run to be slower. (y/n): ')).replace("'", '')
      if dd == '':
        dd = 'y'
      dd = ''.join(e for e in dd if e.isalnum())
      if dd[0].lower() == 'y':
        no_plot = False
      else:
        no_plot = True
      print()
      if nhours != None:
        if nhours < 1:
          nhours2 = nhours*60.0
          units   = ' minutes'
        else:
          nhours2 = nhours
          units   = ' hours'   
      if len(xy_bounds) > 0:
        if d_str1 == None:
          if transition == 'y':
            print('Starting ' + str(nhours2) + units + ' day time real-time ' + day['mod_inputs'] + ' ' + day['mod_type'] + ' ' + mod_loc + ' model run with likelihood score = ' + str(day['pthresh']))       
            print('And night time ' + night['mod_inputs'] + ' ' + night['mod_type'] + ' ' + mod_loc + ' model run with likelihood score = ' + str(night['pthresh']))
            print('Over the region ' + str(xy_bounds) + ' for satellite ' + sat.upper() + ' and sector ' + sector.upper())       
          else:
            print('Starting ' + str(nhours2) + units + ' real-time ' + mod_inputs + ' ' + mod_type + ' ' + mod_loc + ' model run with likelihood score = ' + str(pthresh) + ' over the region ' + str(xy_bounds) + ' for satellite ' + sat.upper() + ' and sector ' + sector.upper())
        else:
          if transition == 'y':
            print('Starting ' + str(d_str1) + ' through ' + str(d_str2) + ' day time real-time ' + day['mod_inputs'] + ' ' + day['mod_type'] + ' ' + mod_loc + ' model run with likelihood score = ' + str(day['pthresh']))       
            print('And night time ' + night['mod_inputs'] + ' ' + night['mod_type'] + ' ' + mod_loc + ' model run with likelihood score = ' + str(night['pthresh']))
            print('Over the region ' + str(xy_bounds) + ' for satellite ' + sat.upper() + ' and sector ' + sector.upper())               
          else:
            print('Starting ' + str(d_str1) + ' through ' + str(d_str2) + ' ' + mod_inputs + ' ' + mod_type + ' ' + mod_loc + ' model run with likelihood score = ' + str(pthresh) + ' over the region ' + str(xy_bounds) + ' for satellite ' + sat.upper() + ' and sector ' + sector.upper())
      else:
        if d_str1 == None:
          if transition == 'y':
            print('Starting ' + str(nhours2) + units + ' day time real-time ' + day['mod_inputs'] + ' ' + day['mod_type'] + ' ' + mod_loc + ' model run with likelihood score = ' + str(day['pthresh']))       
            print('And night time ' + night['mod_inputs'] + ' ' + night['mod_type'] + ' ' + mod_loc + ' model run with likelihood score = ' + str(night['pthresh']))
            print('For satellite ' + sat.upper() + ' and sector ' + sector.upper())       
          else:
            print('Starting ' + str(nhours2) + units + ' real-time ' + mod_inputs + ' ' + mod_type + ' ' + mod_loc + ' model run with likelihood score = ' + str(pthresh) + ' for satellite ' + sat.upper() + ' and sector ' + sector.upper())
        else:
          if transition == 'y':
            print('Starting ' + str(d_str1) + ' through ' + str(d_str2) + ' day time real-time ' + day['mod_inputs'] + ' ' + day['mod_type'] + ' ' + mod_loc + ' model run with likelihood score = ' + str(day['pthresh']))       
            print('And night time ' + night['mod_inputs'] + ' ' + night['mod_type'] + ' ' + mod_loc + ' model run with likelihood score = ' + str(night['pthresh']))
            print('For satellite ' + sat.upper() + ' and sector ' + sector.upper())       
          else:  
            print('Starting ' + str(d_str1) + ' through ' + str(d_str2) + ' ' + mod_inputs + ' ' + mod_type + ' ' + mod_loc + ' model run with likelihood score = ' + str(pthresh) + ' for satellite ' + sat.upper() + ' and sector ' + sector.upper())
      
      print()
      dd = str(input('Is the information above all correct (y/n): ')).replace("'", '')
      if dd == '':
        dd = 'y'
      dd = ''.join(e for e in dd if e.isalnum())
      if dd[0].lower() == 'y':
        mod_check2 = 1
  else:
##### Use data entered in configuration file (config_file)
    use_local = True
    del_local = False
    run_gcs   = False
    with open(config_file) as f:
      contents = f.readlines()                                                                                                     #Read configuration file
    
    contents = [x.strip('\n\r') for x in contents if "#" not in x]                                                                 #Remove file lines that have #.
    contents = list(filter(None, [x.strip('\n\r') for x in contents if "#" not in x]))
    vals     = [re.split('=', x)[1].strip() for x in contents]                                                                     #Remove = sign and strip any leading or trailing blank spaces
   
    counter   = 0                                                                                                                  #Initialize variable to store the number of times the program attempted to ascertain information from the user 
    mod_check = 0                                                                                                                  #Initialize variable to store whether or not the user entered the appropriate information 
    while mod_check == 0:
      counter = counter + 1
      if counter == 4:
        print('Too many failed attempts! Exiting program.')
        exit()
      mod_sat = vals[6]                                                                                                            #Satellite to use
      mod_sat = ''.join(e for e in mod_sat if e.isalnum())                                                                         #Remove any special characters or spaces.
      if mod_sat[0:4].lower() == 'goes':
        mod_check = 1
        sat       = mod_sat[0:4].lower() + '-' + str(mod_sat[4:].lower())
        if sat.endswith('-'):
          mod_check = 0
          print('You must specify which goes satellite number!')
          print()
      elif mod_sat[0].lower() == 'g' and len(mod_sat) <= 3:
        mod_check = 1
        sat       = 'goes-' + str(mod_sat[1:].lower())
      elif mod_sat.lower() == 'seviri':
        mod_check = 1
        sat       = 'seviri'
      elif 'mtg' in mod_sat.lower() or 'mti' in mod_sat.lower():
        if '1' in mod_sat.lower():
            mod_check = 1
            sat       = 'mtg1'
        elif '2' in mod_sat.lower():    
            mod_check = 1
            sat       = 'mtg2'
        elif '3' in mod_sat.lower():    
            mod_check = 1
            sat       = 'mtg3'
        elif '4' in mod_sat.lower():    
            mod_check = 1
            sat       = 'mtg4'
        else:
            print('For MTG data, the software is only capable of handling data for MTG 1-4. Please try again.')
            print()
      else:
        print('Model is not currently set up to handle satellite specified. Please try again.')
        print()
  
    raw_data_root = os.path.realpath(os.path.join('..', '..', '..', re.split('-', sat)[0] + '-data'))                              #Set default directory for the stored raw data files. This may need to be changed if the raw data root directory differs from this.
    
    if sat[0:4] == 'goes':
      counter   = 0                                                                                                                #Initialize variable to store the number of times the program attempted to ascertain information from the user 
      mod_check = 0                                                                                                                #Initialize variable to store whether or not the user entered the appropriate information 
      while mod_check == 0:
        counter = counter + 1
        if counter == 4:
          print('Too many failed attempts! Exiting program.')
          exit()
        mod_sector = vals[7]                                                                                                       #Satellite scan sector to use
        if mod_sector != '':
          mod_sector = ''.join(e for e in mod_sector if e.isalnum())                                                               #Remove any special characters or spaces.
          if mod_sector[0].lower() == 'm':
            try:
              sector    = 'meso' + mod_sector[-1]
              if sector == 'meso1' or sector == 'meso2':
                mod_check = 1
            except:
              sector    = 'meso'
              mod_check = 1
          elif mod_sector[0].lower() == 'c':
            sector    = 'conus'
            mod_check = 1
          elif mod_sector[0].lower() == 'f': 
            sector    = 'full'
            mod_check = 1
          else:
            print('GOES satellite sector specified is not available. Enter M1, M2, F, or C. Please try again.')
            print()
        else:
          print('You must enter satellite sector information. It cannot be left blank. Please try again.')
          print()
    elif sat.lower() == 'seviri':
      sector = 'full'
    elif 'mtg' in sat.lower():
      sector = 'full'
    else:
      sector = ''

    if vals[0].lower() == 'y':                                                                                                     #Check if user wants to run a real-time model or archived model
      d_str1 = None
      d_str2 = None
      try:
        nhours = float(vals[1])                                                                                                    #Number of hours to run real-time model for
      except:
        print("Real-time run chosen but number of hours in config file is not a number??")
        print()
        nhours = input('How many hours would you like the model to run for? Decimal numbers are accepted. ')
        try:
          nhours = float(nhours)
        except:
          print('Wrong type entered! Number of hours must either be set to an integer or floating point value. Please try again.')
          print('Format type entered = ' + str(type(nhours)))
          print()
          nhours = input('How many hours would you like the model to run for? Decimal numbers are accepted. ')
          try:
            nhours = float(nhours)
          except:
            print('Wrong type entered again! Number of hours must either be set to an integer or floating point value. Exiting program.')
            print('Value entered = ' + str(nhours))
            print('Format type entered = ' + str(type(nhours)))
            exit()
      if 'mtg' not in mod_sat.lower() and 'mti' not in mod_sat.lower():
        no_download = False
      else:
        if vals[4][0].lower() == 'y':                                                                                              #User specifies if raw data files need to be downloaded
          no_download = False
        else:
          no_download = True
      
    else:
      d_str1 = vals[2].replace("'", '').strip()                                                                                    #User specified start date
      d_str2 = vals[3].replace("'", '').strip()                                                                                    #User specified end date
      nhours = None      
      if len(d_str1) <= 11:
        d_str1 = d_str1.strip()
        d_str1 = d_str1 + ' 00:00:00'
      if len(d_str2) <= 11:
        d_str2 = d_str2.strip()
        d_str2 = d_str2 + ' 00:00:00'
      if d_str2 < d_str1:
        print('End date cannot be before start date!!!! Please try again.')
        print('Start date entered = ' + d_str1)
        print('End date entered = ' + d_str2)
        print()
        d_str1 = str(input('Enter the model start date (ex. 2017-04-29 00:00:00): ')).replace("'", '').strip()
        d_str2 = str(input('Enter the model end date (ex. 2017-04-29 00:00:00): ')).replace("'", '').strip()
      if d_str2 < d_str1:
        print('End date cannot be before start date!!!! Exiting program.')
        print('Start date entered = ' + d_str1)
        print('End date entered = ' + d_str2)
        print()
        exit()
      try:                                                                                                                         #Make sure date1 is in proper format
        d1 = datetime.strptime(d_str1, "%Y-%m-%d %H:%M:%S")
      except:
        print('date1 not in correct format!!! Format must be YYYY-mm-dd HH:MM:SS. Exiting program. Please try again.') 
        print(d_str1)
        exit() 
      
      try:                                                                                                                         #Make sure date2 is in proper format
        d2 = datetime.strptime(d_str2, "%Y-%m-%d %H:%M:%S")
      except:
        print('date1 not in correct format!!! Format must be YYYY-mm-dd HH:MM:SS. Exiting program. Please try again.') 
        print(d_str1)
        exit() 
      
      if vals[4][0].lower() == 'y':                                                                                                #User specifies if raw data files need to be downloaded
        no_download = False
      else:
        no_download = True
    if no_download:
      if vals[5] != 'None':
        raw_data_root = vals[5]                                                                                                    #User specifies location of raw data files since the files do not need to be downloaded
      raw_data_root00 = raw_data_root
      counter   = 0                                                                                                                #Initialize variable to store the number of times the program attempted to ascertain information from the user 
      mod_check = 0                                                                                                                #Initialize variable to store whether or not the user entered the appropriate information 
      while mod_check == 0:
        counter = counter + 1
        if counter == 4:
          print('Too many failed attempts! Exiting program.')
          print('Raw data file root not correct because they do not exist!!')
          print('Do not include subdirectories for dates or ir, glm, vis, wv')
          exit()
        if os.path.realpath(raw_data_root).endswith('ir') or os.path.realpath(raw_data_root).endswith('vis') or os.path.realpath(raw_data_root).endswith('glm') or os.path.realpath(raw_data_root).endswith('wv'):
          raw_data_root0 = os.path.dirname(os.path.realpath(raw_data_root))
        else:
          raw_data_root0 = os.path.realpath(raw_data_root)
        try:
          q             = int(os.path.basename(raw_data_root0))                                                                    #Checks if root path mistakenly included a date
          raw_data_root = os.path.dirname(raw_data_root0)                                                                          #Remove date string from path
        except:
          q             = ''                                                                                                       #Flag to show that path did not a include date if it check for date fails
       
#         if raw_data_root and not raw_data_root.endswith('/'):                                                                      #Make sure the specified root directory path has / at the end
#           raw_data_root += '/'
   
#        if os.path.exists(os.path.join(raw_data_root, d1.strftime('%Y%m%d'), 'ir')):
        if os.path.exists(raw_data_root):
          mod_check = 1
        else:
          print('IR data directory within root path does not exist. Please enter the root directory to the raw IR/VIS/GLM data files.')
          print('Current path specified = ' + raw_data_root)
          raw_data_root = str(input('Enter the root directory to the raw IR/VIS/GLM data files (ex.  ' + raw_data_root00 + '): ')).replace("'", '')
         
    reg_bound = vals[8]                                                                                                            #If user specified state or country to confine the data to
    if reg_bound == 'None':
      xy_bounds = []
      region    = None
    else:  
      region  = extract_us_lat_lon_region('virginia')
      regions = list(region.keys())
      if reg_bound.lower() not in regions:
        print('State or country specified is not available???')
        print('You specified = ' + str(reg_bound.lower()))
        print('Regions possible = ' + str(sorted(regions)))
        print()
        counter   = 0                                                                                                              #Initialize variable to store the number of times the program attempted to ascertain information from the user 
        mod_check = 0                                                                                                              #Initialize variable to store whether or not the user entered the appropriate information 
        while mod_check == 0:
          counter = counter + 1
          if counter == 4:
            print('Too many failed attempts! Exiting program.')
            exit()
          reg_bound = str(input('Is there a particular state or country you would like to contrain data to? If no, hit ENTER. If yes, enter the name of the state or country: ')).replace("'", '').strip()
          if reg_bound.lower() in regions or reg_bound == '':
            mod_check = 1
          else:  
            print('State or country specified is not available???')
            print('You specified = ' + str(reg_bound.lower()))
            print()
      if reg_bound != '':
        xy_bounds = region[reg_bound.lower()]
        region    = reg_bound.lower()
    
    if len(xy_bounds) == 0:
      mod_bound = vals[9]                                                                                                          #If user set longitude and latitude boundaries to confine the data to
      if mod_bound == '[]' or mod_bound == '' or mod_bound == 'None':
        xy_bounds = []
      else:  
        mod_bound = mod_bound.replace(' ' , '').replace('[', '').replace(']', '')                                                  #Remove any spaces
        mod_x0    = re.split(',', mod_bound)[0]
        mod_y0    = re.split(',', mod_bound)[1]
        mod_x1    = re.split(',', mod_bound)[2]
        mod_y1    = re.split(',', mod_bound)[3]
        
        counter   = 0                                                                                                              #Initialize variable to store the number of times the program attempted to ascertain information from the user 
        mod_check = 0                                                                                                              #Initialize variable to store whether or not the user entered the appropriate information 
        while mod_check == 0:
          if counter >= 1:
            mod_x0 = input('Enter the minimum longitude point. Values must be between -180.0 and 180.0: ')
          counter = counter + 1
          if counter == 4:
            print('Too many failed attempts! Exiting program.')
            exit()
          try:
            mod_x0 = float(mod_x0)
            if mod_x0 >= -180.0 and mod_x0 <= 180.0:
              x0        = mod_x0
              mod_check = 1
            else:  
              print('Minimum longitude point must be between -180.0 and 180.0. Please try again.')
              print()
          except:
            print('Wrong data type entered! Please try again.')
            print('Format type entered = ' + str(type(mod_x0)))
            print()
          
          counter   = 0                                                                                                            #Initialize variable to store the number of times the program attempted to ascertain information from the user 
          mod_check = 0                                                                                                            #Initialize variable to store whether or not the user entered the appropriate information 
          while mod_check == 0:
            if counter >= 1:
              mod_x1 = input('Enter the maximum longitude point. Values must be between -180.0 and 180.0: ')
            counter = counter + 1
            if counter == 4:
              print('Too many failed attempts! Exiting program.')
              exit()
            try:
              mod_x1 = float(mod_x1)
              if mod_x1 >= -180.0 and mod_x1 <= 180.0:
                if mod_x1 > x0:
                  x1        = mod_x1
                  mod_check = 1
                else:
                  print('Maximum longitude point must be > minimum longitude point. Please try again.')
                  print('Minimum longitude point entered = ' + str(x0))
                  print('Maximum longitude point entered = ' + str(mod_x1))
                  print()
              else:  
                print('Maximum longitude point must be between minimum longitude point and 180.0. Please try again.')
                print()
            except:
              print('Wrong data type entered! Please try again.')
              print('Format type entered = ' + str(type(mod_x1)))
              print()
        
          counter   = 0                                                                                                            #Initialize variable to store the number of times the program attempted to ascertain information from the user 
          mod_check = 0                                                                                                            #Initialize variable to store whether or not the user entered the appropriate information 
          while mod_check == 0:
            if counter >= 1:
              mod_y0 = input('Enter the minimum latitude point. Values must be between -90.0 and 90.0: ')
            counter = counter + 1
            if counter == 4:
              print('Too many failed attempts! Exiting program.')
              exit()
            try:
              mod_y0 = float(mod_y0)
              if mod_y0 >= -90.0 and mod_y0 <= 90.0:
                y0        = mod_y0
                mod_check = 1
              else:  
                print('Minimum latitude point must be between -90.0 and 90.0. Please try again.')
                print()
            except:
              print('Wrong data type entered! Please try again.')
              print('Format type entered = ' + str(type(mod_y0)))
              print()
          
          counter   = 0                                                                                                            #Initialize variable to store the number of times the program attempted to ascertain information from the user 
          mod_check = 0                                                                                                            #Initialize variable to store whether or not the user entered the appropriate information 
          while mod_check == 0:
            if counter >= 1:
              mod_y1 = input('Enter the maximum latitude point. Values must be between -90.0 and 90.0: ')
            counter = counter + 1
            if counter == 4:
              print('Too many failed attempts! Exiting program.')
              exit()
            try:
              mod_y1 = float(mod_y1)
              if mod_y1 >= -90.0 and mod_y1 <= 90.0:
                if mod_y1 > y0:
                  y1        = mod_y1
                  mod_check = 1
                else:
                  print('Maximum latitude point must be > minimum latitude point. Please try again.')
                  print('Minimum latitude point entered = ' + str(y0))
                  print('Maximum latitude point entered = ' + str(mod_y1))
                  print()
              else:  
                print('Maximum latitude point must be between minimum latitude point and 90.0. Please try again.')
                print()
            except:
              print('Wrong data type entered! Please try again.')
              print('Format type entered = ' + str(type(mod_y1)))
              print()
          
          xy_bounds = [x0, y0, x1, y1]
     
    mod_loc = vals[10]                                                                                                             #Severe storm signature targeted by user
    counter   = 0                                                                                                                  #Initialize variable to store the number of times the program attempted to ascertain information from the user 
    mod_check = 0                                                                                                                  #Initialize variable to store whether or not the user entered the appropriate information 
    while mod_check == 0:
      if counter >= 1:
        mod_loc = str(input('Enter the desired severe weather indicator (OT or AACP): ')).replace("'", '')
      counter = counter + 1
      if counter == 4:
        print('Too many failed attempts! Exiting program.')
        exit()
      mod_loc = ''.join(e for e in mod_loc if e.isalnum())
      if mod_loc.lower() == 'updraft' or mod_loc.lower() == 'ot' or mod_loc.lower() == 'updrafts' or mod_loc.lower() == 'ots':
        mod_loc     = 'OT'
        use_updraft = True
        mod_check   = 1
      elif mod_loc.lower() == 'plume' or mod_loc.lower() == 'aacp' or mod_loc.lower() == 'plumes' or mod_loc.lower() == 'aacps' or mod_loc.lower() == 'warmplume' or mod_loc.lower() == 'warmplumes':
        mod_loc     = 'AACP'
        use_updraft = False
        mod_check   = 1
      else:
        print('Severe weather indicator entered is not available. Desired severe weather indicator must be OT or AACP!!!')
        print('You entered : ' + mod_loc)
        print('Please try again.')
        print()
    
    if mod_loc.lower() == 'ot':
      if len(vals) >= 18:
        percent_omit = vals[17]                                                                                                    #If target is OT, determine the percent of coldest and warmest anvil pixels to omit from the OT IR-anvil BTD calculation
        counter   = 0                                                                                                              #Initialize variable to store the number of times the program attempted to ascertain information from the user 
        mod_check = 0                                                                                                              #Initialize variable to store whether or not the user entered the appropriate information 
        while mod_check == 0:
          if counter >= 1:
            percent_omit = str(input('For post-processing, Enter the percent of coldest and warmest pixels to omit from the anvil calculation to remove bias in determining OT IR-anvil BTD: ')).replace("'", '')
          counter = counter + 1
          if counter == 4:
            print('Too many failed attempts! Exiting program.')
            exit()
          percent_omit = ''.join(e for e in percent_omit if e.isdigit() or e == '.' or e == '-')
          try:
            percent_omit = float(percent_omit)
            if percent_omit < 0 or percent_omit > 100:
              print('You MUST enter only a number between 0 and 100 for % of pixels of pixels to be omitted from and mean brightness temperature calculation')
              print('The default is 20')
              print('You entered : ' + str(percent_omit))
              print('Please try again.')
              print()
            else:
              mod_check = 1
          except:
            print('You MUST enter only a number between 0 and 100 for % of pixels of pixels to be omitted from and mean brightness temperature calculation')
            print('The default is 20')
            print('You entered : ' + str(percent_omit))
            print('Please try again.')
            print()
      else:
        percent_omit = 20                                                                                                          #Set default percent omit. This does not matter of AACPs but it still needs to be passed into post-processing function
    else:
      percent_omit = 20                                                                                                            #Set default percent omit. This does not matter of AACPs but it still needs to be passed into post-processing function
      
    transition = 'n'
    day_night0 = vals[11]                                                                                                          #If user wants to use optimal model for day-night seamless transition run
    day_night0 = ''.join(e for e in day_night0 if e.isalnum())                                                                     #Remove any special characters or spaces.
    if day_night0[0].lower() == 'y':
      use_night  = True
      opt_params = 'y'
      transition = 'y'
      use_glm    = True
    else:
      opt_params = 'n'
      use_night  = True    
    
    opt_params = ''.join(e for e in opt_params if e.isalnum())                                                                     #Remove any special characters or spaces.
    if opt_params[0].lower() == 'y':
      mod_type = 'multiresunet'
      if transition == 'y':
        day0    = best_model_checkpoint_and_thresh('best_day', mod_loc, native_ir = use_native_ir, drcs = drcs) 
        night0  = best_model_checkpoint_and_thresh('best_night', mod_loc, native_ir = use_native_ir, drcs = drcs)
        day_inp = re.split(r'\+', day0[0].lower())
        nig_inp = re.split(r'\+', night0[0].lower())
        use_trop        = False
        use_dirtyirdiff = False
        use_cirrus      = False
        use_snowice     = False
        use_glm         = False
        use_irdiff      = False
        if 'tropdiff' in day_inp:
          use_trop = True
        if 'wvirdiff' in day_inp:
          use_irdiff = True
        if 'dirtyirdiff' in day_inp:
          use_dirtyirdiff = True
        if 'cirrus' in day_inp:
          use_cirrus = True
        if 'snowice' in day_inp:
          use_snowice = True
        if 'glm' in day_inp:
          use_glm = True
        day   = {'mod_type': day0[1],   'mod_inputs': day0[0],   'use_chkpnt': os.path.realpath(day0[2]),   'pthresh': day0[3],   'use_night' : False, 'use_trop' : use_trop, 'use_dirtyirdiff' : use_dirtyirdiff, 'use_cirrus' : use_cirrus, 'use_snowice' : use_snowice, 'use_glm' : use_glm, 'use_irdiff' : use_irdiff}
        
        use_trop        = False
        use_dirtyirdiff = False
        use_cirrus      = False
        use_snowice     = False
        use_glm         = False
        use_irdiff      = False
        if 'tropdiff' in nig_inp:
          use_trop = True
        if 'wvirdiff' in nig_inp:
          use_irdiff = True
        if 'dirtyirdiff' in nig_inp:
          use_dirtyirdiff = True
        if 'cirrus' in nig_inp:
          use_cirrus = True
        if 'snowice' in nig_inp:
          use_snowice = True
        if 'glm' in nig_inp:
          use_glm = True
        night = {'mod_type': night0[1], 'mod_inputs': night0[0], 'use_chkpnt': os.path.realpath(night0[2]), 'pthresh': night0[3], 'use_night' : True, 'use_trop' : use_trop, 'use_dirtyirdiff' : use_dirtyirdiff, 'use_cirrus' : use_cirrus, 'use_snowice' : use_snowice, 'use_glm' : use_glm, 'use_irdiff' : use_irdiff}
    else:  
      mod_type = vals[13]                                                                                                          #Model type to be run (multiresunet, unet, attentionunet, etc.)
      if mod_loc == 'OT':
        counter   = 0                                                                                                              #Initialize variable to store the number of times the program attempted to ascertain information from the user 
        mod_check = 0                                                                                                              #Initialize variable to store whether or not the user entered the appropriate information 
        while mod_check == 0:
          if counter >= 1:  
            mod_type = str(input('Enter the model inputs you would like to use. Options are multiresunet, unet, attentionunet: ')).replace("'", '')
          counter = counter + 1
          if counter == 4:
            print('Too many failed attempts! Exiting program.')
            print('You must enter multiresunet, unet, or attentionunet ONLY.')
            exit()
          mod_type = ''.join(e for e in mod_type if e.isalnum())
          if mod_type.lower() == 'multiresunet':  
            mod_check = 1
            mod_type  = 'multiresunet'
          elif mod_type.lower() == 'unet':  
            mod_check = 1
            mod_type = 'unet'
          elif mod_type.lower() == 'attentionunet':
            mod_check = 1
            mod_type  = 'attentionunet'
          else:
            print('Model type entered is not available. Options are multiresunet, unet, or attentionunet. Please try again.')
            print()
      else:
        mod_type = 'multiresunet'
      
      mod_inputs = vals[12]                                                                                                        #Model inputs chosen by user
      counter   = 0                                                                                                                #Initialize variable to store the number of times the program attempted to ascertain information from the user 
      mod_check = 0                                                                                                                #Initialize variable to store whether or not the user entered the appropriate information 
      while mod_check == 0:
        if counter >= 1:
          mod_inputs = str(input('Enter the model inputs you would like to use (ex. IR+VIS): ')).replace("'", '')
        counter = counter + 1
        if counter == 4:
          print('Too many failed attempts! Exiting program.')
          print('You must enter IR, IR+VIS, IR+GLM, IR+IRDIFF, or IR+VIS+GLM ONLY.')
          exit()
        str_list = re.split(',| ', mod_inputs)
        while '' in str_list:
          str_list.remove('')
        
        mod_inputs = '+'.join(str_list)
        if mod_inputs.lower() == 'ir':  
          mod_inputs = 'IR'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':  
              use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
              pthresh     = 0.45
              opt_pthres0 = pthresh
            elif mod_type == 'attentionunet':  
              use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
              print('Optimal pthresh must still be entered!!!')
              print(mod_inputs)
              exit()
          else:
            use_native_ir = True
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'ir+vis' or mod_inputs.lower() == 'vis+ir':  
          mod_inputs = 'IR+VIS'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
              pthresh     = 0.35
              opt_pthres0 = pthresh
            elif mod_type == 'attentionunet':
              use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
              pthresh     = 0.65
              opt_pthres0 = pthresh
          else:
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'ir+glm' or mod_inputs.lower() == 'glm+ir':
          mod_inputs = 'IR+GLM'
          mod_check  = 1
          use_glm    = True
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_glm', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
              pthresh     = 0.45
              opt_pthres0 = pthresh
            elif mod_type == 'attentionunet':
              use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_glm', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
              print('Optimal pthresh must still be entered!!!')
              print(mod_inputs)
              exit()
          else:
            use_native_ir = True
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'ir+irdiff' or mod_inputs.lower() == 'irdiff+ir' or mod_inputs.lower() == 'wvirdiff+ir' or mod_inputs.lower() == 'ir+wvirdiff':  
          mod_inputs = 'IR+WVIRDIFF'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_irdiff', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
              pthresh     = 0.45
              opt_pthres0 = pthresh
            elif mod_type == 'attentionunet':
              use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_irdiff', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
              print('Optimal pthresh must still be entered!!!')
              print(mod_inputs)
              exit()
          else:
            use_native_ir = True
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'ir+dirtyirdiff' or mod_inputs.lower() == 'dirtyirdiff+ir':  
          mod_inputs = 'IR+DIRTYIRDIFF'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              print('No unet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
            elif mod_type == 'attentionunet':
              print('No attentionunet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
          else:
            use_native_ir = True
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'tropdiff':  
          mod_inputs = 'TROPDIFF'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              print('No unet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
            elif mod_type == 'attentionunet':
              print('No attentionunet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
          else:
            use_native_ir = True
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'ir+snowice' or mod_inputs.lower() == 'snowice+ir':  
          mod_inputs = 'IR+SNOWICE'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              print('No unet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
            elif mod_type == 'attentionunet':
              print('No attentionunet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
          else:
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'ir+cirrus' or mod_inputs.lower() == 'cirrus+ir':  
          mod_inputs = 'IR+CIRRUS'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              print('No unet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
            elif mod_type == 'attentionunet':
              print('No attentionunet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
          else:
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'ir+tropdiff' or mod_inputs.lower() == 'tropdiff+ir':  
          mod_inputs = 'IR+TROPDIFF'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              print('No unet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
            elif mod_type == 'attentionunet':
              print('No attentionunet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
          else:
            use_native_ir = True
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'vis+tropdiff' or mod_inputs.lower() == 'tropdiff+vis':  
          mod_inputs = 'VIS+TROPDIFF'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              print('No unet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
            elif mod_type == 'attentionunet':
              print('No attentionunet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
          else:
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'tropdiff+glm' or mod_inputs.lower() == 'glm+tropdiff':  
          mod_inputs = 'TROPDIFF+GLM'
          mod_check  = 1
          use_glm    = True
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              print('No unet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
            elif mod_type == 'attentionunet':
              print('No attentionunet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
          else:
            use_native_ir = True
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'tropdiff+dirtyirdiff' or mod_inputs.lower() == 'dirtyirdiff+tropdiff':  
          mod_inputs = 'TROPDIFF+DIRTYIRDIFF'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              use_native_ir = True
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              print('No unet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
            elif mod_type == 'attentionunet':
              print('No attentionunet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
          else:
            use_native_ir = True
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif 'ir' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'tropdiff' in mod_inputs.lower(): 
          mod_inputs = 'IR+VIS+TROPDIFF'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              print('No unet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
            elif mod_type == 'attentionunet':
              print('No attentionunet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
          else:
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif 'glm' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'tropdiff' in mod_inputs.lower(): 
          mod_inputs = 'VIS+TROPDIFF+GLM'
          mod_check  = 1
          use_glm    = True
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              print('No unet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
            elif mod_type == 'attentionunet':
              print('No attentionunet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
          else:
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif 'ir' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'dirtyirdiff' in mod_inputs.lower(): 
          mod_inputs = 'IR+VIS+DIRTYIRDIFF'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              print('No unet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
            elif mod_type == 'attentionunet':
              print('No attentionunet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
          else:
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        elif 'dirtyirdiff' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'tropdiff' in mod_inputs.lower(): 
          mod_inputs = 'VIS+TROPDIFF+DIRTYIRDIFF'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              print('No unet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
            elif mod_type == 'attentionunet':
              print('No attentionunet model available for specified inputs!!!')
              print(mod_inputs)
              exit()
          else:
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
            
        elif 'ir' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'glm' in mod_inputs.lower(): 
          mod_inputs = 'IR+VIS+GLM'
          mod_check  = 1
          use_glm    = True
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
              pthresh     = day0[3]
              use_chkpnt  = os.path.realpath(day0[2])
              opt_pthres0 = pthresh
            elif mod_type == 'unet':
              use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis_glm', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
              pthresh     = 0.45
              opt_pthres0 = pthresh
            elif mod_type == 'attentionunet':
              use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis_glm', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
              pthresh     = 0.20
              opt_pthres0 = pthresh
          else:
            day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
            pthresh     = day0[3]
            use_chkpnt  = os.path.realpath(day0[2])
            opt_pthres0 = pthresh
        else:
          print('Model inputs entered are not available. Options are IR, IR+VIS, IR+GLM, IR+WVIRDIFF, TROPDIFF, IR+SNOWICE, IR+CIRRUS, IR+DIRTYIRDIFF, IR+TROPDIFF, VIS+TROPDIFF, TROPDIFF+GLM, IR+VIS+TROPDIFF, VIS+TROPDIFF+GLM, IR+VIS+DIRTYIRDIFF, TROPDIFF+DIRTYIRDIFF, VIS+TROPDIFF+DIRTYIRDIFF, and IR+VIS+GLM. Please try again.')
          print()
    
      opt_pthresh = vals[15]                                                                                                       #If user wants to use optimal likelihood score for model inputs chosen
      opt_pthresh = ''.join(e for e in opt_pthresh if e.isalnum())
      if opt_pthresh[0].lower() != 'y':
        pthresh   = vals[16]                                                                                                       #Likelihood score chosen by user
        counter   = 0                                                                                                              #Initialize variable to store the number of times the program attempted to ascertain information from the user 
        mod_check = 0                                                                                                              #Initialize variable to store whether or not the user entered the appropriate information 
        while mod_check == 0:
          if counter >= 1:
            pthresh = input('Enter the likelihood value you would like to use (0-1): ')
          counter = counter + 1
          if counter == 4:
            print('Too many failed attempts! Exiting program.')
            exit()
          try:
            pthresh = float(pthresh)
            if pthresh < 0 or pthresh > 1:
              print('Likelihood value must be between 0 and 1. Please try again.')
              print()
            else:
              mod_check = 1
          except:
            print('Wrong type entered! Likelihood value must either be set to an integer or floating point value between 0 and 1. Please try again.')
            print('Format type entered = ' + str(type(pthresh)))
            print()

    dd = vals[14]                                                                                                                  #If user wants to plot the data
    dd = ''.join(e for e in dd if e.isalnum())
    if dd[0].lower() == 'y':
      no_plot = False
    else:
      no_plot = True
    
    if mod_loc == 'OT':
      use_updraft = True
    else:
      use_updraft = False    
    counter2   = 0
    mod_check2 = 0
    while mod_check2 == 0:
      counter2 = counter2 + 1
      print()
      if d_str1 == None:
        if counter2 == 0:
          print('You have chosen to run a real-time run')
        if nhours != None:
          if nhours < 1:
            nhours2 = nhours*60.0
            units   = ' minutes'
          else:
            nhours2 = nhours
            units   = ' hours'   
  
        print('(1) The model will run for ' + str(nhours2) + units)
      else:  
        if counter2 == 0:
          print('You have chosen not to run the model in real-time')
        print('(1) The start and end date for the model run = ' + d_str1 + ' through ' + d_str2)
      
      print('(2) The model will use ' + sat + ' satellite data')
      print('(3) The satellite scan domain sector is ' + str(sector))
      print('(4) You have chosen the model to identify ' + mod_loc + 's')
      if transition == 'y':
        print('(5) Multiresunet is the best model. Model type cannot be changed unless (6) is changed.')
        print('(6) You have chosen to run the optimal model based on the time of day. This model will seamlessly transition between the model that works best for day and night hours.')
      else:
        print('(5) You have chosen to run the ' + mod_type + ' model type')
        print('(6) You have chosen to use ' + mod_inputs + ' as inputs into the model')
      if no_plot == True:
        print('(7) You have chosen to not plot the individual scenes')  
      else:
        print('(7) You have chosen to plot the individual scenes (Note, this may cause a run-time slowdown)')  
      
      if region == None:
        print('(8) No state/country specified to constrain output to.')
      else:
        print('(8) You have chosen to constrain model region to = ' + region)
        if sector[0:4] != 'meso':
          if len(xy_bounds) > 0:
            print('(9) Domain corners set to ' + str(xy_bounds))
          else:
            print('(9) You have chosen not to constrain model to any lat/lon boundary points')

      if mod_loc.lower() == 'ot':
        print('(10) In post-processing, the percent of coldest and warmest anvil pixels to omit from OT IR-anvil BTD calculation is ' + str(percent_omit))
      
      print()
      dd = str(input('Is the information above all correct? If yes, hit ENTER. If no, type the number of the question you would like changed: ')).replace("'", '')
      if dd == '':
        dd = 'y'
      dd = ''.join(e for e in dd if e.isalnum())
      if dd[0].lower() == 'y':
        mod_check2 = 1
      else:
        if counter2 == 6:
          print('Too many failed attempts! Exiting program.')
          print('Max attempts = 5')
          print(counter2)
          exit()

        try:
          dd = int(dd)
        except:
          print('Wrong data type entered! If all of the information above is correct, hit ENTER. If not, type the number that of statement you would like changed.')
          print('Please try again.')
        
        if dd > 10 or dd <= 0:
          print('Invalid number entered. Numbers allowed range between 1 and 10.')
          print('You entered : ' + str(dd))
          print('Please try again.')
        else:  
          if d_str1 == None:
            if dd == 1:
              nhours = input('How many hours would you like the model to run for? Decimal numbers are accepted. ')
              try:
                nhours = float(nhours)
              except:
                print('Wrong type entered! Number of hours must either be set to an integer or floating point value. Please try again.')
                print('Format type entered = ' + str(type(nhours)))
                print()
                nhours = input('How many hours would you like the model to run for? Decimal numbers are accepted. ')
                try:
                  nhours = float(nhours)
                except:
                  print('Wrong type entered again! Number of hours must either be set to an integer or floating point value. Exiting program.')
                  print('Value entered = ' + str(nhours))
                  print('Format type entered = ' + str(type(nhours)))
                  exit()
          else:  
            d_str1 = str(input('Enter the model start date (ex. 2017-04-29 00:00:00): ')).replace("'", '').strip()
            d_str2 = str(input('Enter the model end date (ex. 2017-04-29 00:00:00): ')).replace("'", '').strip()
            if len(d_str1) <= 11:
              d_str1 = d_str1.strip()
              d_str1 = d_str1 + ' 00:00:00'
            if len(d_str2) <= 11:
              d_str2 = d_str2.strip()
              d_str2 = d_str2 + ' 00:00:00'
            if d_str2 < d_str1:
              print('End date cannot be before start date!!!! Please try again.')
              print('Start date entered = ' + d_str1)
              print('End date entered = ' + d_str2)
              print()
              d_str1 = str(input('Enter the model start date (ex. 2017-04-29 00:00:00): ')).replace("'", '').strip()
              d_str2 = str(input('Enter the model end date (ex. 2017-04-29 00:00:00): ')).replace("'", '').strip()
            if d_str2 < d_str1:
              print('End date cannot be before start date!!!! Exiting program.')
              print('Start date entered = ' + d_str1)
              print('End date entered = ' + d_str2)
              print()
              exit()
            try:                                                                                                                   #Make sure date1 is in proper format
              d1 = datetime.strptime(d_str1, "%Y-%m-%d %H:%M:%S")
            except:
              print('date1 not in correct format!!! Format must be YYYY-mm-dd HH:MM:SS. Exiting program. Please try again.') 
              print(d_str1)
              exit() 
            
            try:                                                                                                                   #Make sure date2 is in proper format
              d2 = datetime.strptime(d_str2, "%Y-%m-%d %H:%M:%S")
            except:
              print('date1 not in correct format!!! Format must be YYYY-mm-dd HH:MM:SS. Exiting program. Please try again.') 
              print(d_str1)
              exit() 
          
          if dd == 1:
            good = 1  
          elif dd == 2:
            counter   = 0                                                                                                          #Initialize variable to store the number of times the program attempted to ascertain information from the user 
            mod_check = 0                                                                                                          #Initialize variable to store whether or not the user entered the appropriate information 
            while mod_check == 0:
              counter = counter + 1
              if counter == 4:
                print('Too many failed attempts! Exiting program.')
                exit()
              mod_sat = str(input('Enter the satellite name (ex and default. goes-16): ')).replace("'", '')
              if mod_sat == '':
                mod_sat = 'goes-16'
                print('Using GOES-16 satellite data')
                print()
              mod_sat = ''.join(e for e in mod_sat if e.isalnum())                                                                 #Remove any special characters or spaces.
              if mod_sat[0:4].lower() == 'goes':
                mod_check = 1
                sat       = mod_sat[0:4].lower() + '-' + str(mod_sat[4:].lower())
                if sat.endswith('-'):
                  mod_check = 0
                  print('You must specify which goes satellite number!')
                  print()
              elif mod_sat[0].lower() == 'g' and len(mod_sat) <= 3:
                mod_check = 1
                sat       = 'goes-' + str(mod_sat[1:].lower())
              elif mod_sat.lower() == 'seviri':
                mod_check = 1
                sat       = 'seviri'
              elif 'mtg' in mod_sat.lower() or 'mti' in mod_sat.lower():
                if '1' in mod_sat.lower():
                    mod_check = 1
                    sat       = 'mtg1'
                elif '2' in mod_sat.lower():    
                    mod_check = 1
                    sat       = 'mtg2'
                elif '3' in mod_sat.lower():    
                    mod_check = 1
                    sat       = 'mtg3'
                elif '4' in mod_sat.lower():    
                    mod_check = 1
                    sat       = 'mtg4'
                else:
                    print('For MTG data, the software is only capable of handling data for MTG 1-4. Please try again.')
                    print()
              else:
                print('Model is not currently set up to handle satellite specified. Please try again.')
                print()
#            raw_data_root = os.path.realpath('../../../' + re.split('-', sat)[0] + '-data/')                                       #Set default directory for the stored raw data files. This may need to be changed if the raw data root directory differs from this.            
          elif dd == 3:
            counter   = 0                                                                                                          #Initialize variable to store the number of times the program attempted to ascertain information from the user 
            mod_check = 0                                                                                                          #Initialize variable to store whether or not the user entered the appropriate information 
            while mod_check == 0:
              counter = counter + 1
              if counter == 4:
                print('Too many failed attempts! Exiting program.')
                exit()
              mod_sector = str(input('Enter the ' + sat + ' satellite sector to run model on (ex. M1, M2, F, C): ')).replace("'", '')
              if mod_sector != '':
                mod_sector = ''.join(e for e in mod_sector if e.isalnum())                                                         #Remove any special characters or spaces.
                if mod_sector[0].lower() == 'm':
                  try:
                    sector    = 'meso' + mod_sector[-1]
                    if sector == 'meso1' or sector == 'meso2':
                      mod_check = 1
                  except:
                    sector    = 'meso'
                    mod_check = 1
                elif mod_sector[0].lower() == 'c':
                  sector    = 'conus'
                  mod_check = 1
                elif mod_sector[0].lower() == 'f': 
                  sector    = 'full'
                  mod_check = 1
                else:
                  print('GOES satellite sector specified is not available. Enter M1, M2, F, or C. Please try again.')
                  print()
              else:
                print('You must enter satellite sector information. It cannot be left blank. Please try again.')
                print()
          elif dd == 4:
            if mod_loc.lower() == 'ot':
              mod_loc = 'AACP'
            else:
              mod_loc = 'OT'  
              counter   = 0                                                                                                        #Initialize variable to store the number of times the program attempted to ascertain information from the user 
              mod_check = 0                                                                                                        #Initialize variable to store whether or not the user entered the appropriate information 
              while mod_check == 0:
                percent_omit = str(input('For post-processing, Enter the percent of coldest and warmest pixels to omit from the anvil calculation to remove bias in determining OT IR-anvil BTD: ')).replace("'", '')
                counter = counter + 1
                if counter == 4:
                  print('Too many failed attempts! Exiting program.')
                  exit()
                if type(percent_omit) == str:
                  percent_omit = ''.join(e for e in percent_omit if e.isdigit() or e == '.' or e == '-')
                try:
                  percent_omit = float(percent_omit)
                  if percent_omit < 0 or percent_omit > 100:
                    print('You MUST enter only a number between 0 and 100 for % of pixels of pixels to be omitted from and mean brightness temperature calculation')
                    print('The default is 20')
                    print('You entered : ' + str(percent_omit))
                    print('Please try again.')
                    print()
                  else:
                    mod_check = 1
                except:
                  print('You MUST enter only a number between 0 and 100 for % of pixels of pixels to be omitted from and mean brightness temperature calculation')
                  print('The default is 20')
                  print('You entered : ' + str(percent_omit))
                  print('Please try again.')
                  print()

            if mod_type.lower() == 'multiresunet':  
              mod_check = 1
              mod_type  = 'multiresunet'
              if transition != 'y':
                if mod_inputs.lower() == 'ir' or mod_inputs.lower() == 'ir+glm' or mod_inputs.lower() == 'ir+wvirdiff' or mod_inputs.lower() == 'tropdiff' or mod_inputs.lower() == 'ir+dirtyirdiff'  or mod_inputs.lower() == 'ir+tropdiff' or mod_inputs.lower() == 'tropdiff+glm' or mod_inputs.lower() == 'tropdiff+dirtyirdiff':
                  use_native_ir = True
                elif mod_inputs.lower() == 'ir+vis' or mod_inputs.lower() == 'ir+vis+glm' or mod_inputs.lower() == 'ir+snowice' or mod_inputs.lower() == 'ir+cirrus' or mod_inputs.lower() == 'vis+tropdiff' or mod_inputs.lower() == 'ir+vis+tropdiff' or mod_inputs.lower() == 'vis+tropdiff+glm' or mod_inputs.lower() == 'ir+vis+dirtyirdiff' or mod_inputs.lower() == 'vis+tropdiff+dirtyirdiff':  
                  use_native_ir = False
                else:
                  print('Inputs specified not found within if statement. Please check to make sure this is correct and not in error.')
                  print(mod_inputs)
                  print(mod_type)
                  exit()
                day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                pthresh0    = day0[3]
                use_chkpnt  = os.path.realpath(day0[2])
                opt_pthres0 = pthresh0
            elif mod_type.lower() == 'unet':  
              mod_check     = 1
              mod_type      = 'unet'
              use_native_ir = False
              if transition != 'y':
                if mod_loc == 'OT':
                  if mod_inputs.lower() == 'ir':
                    use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                    pthresh0    = 0.45
                    opt_pthres0 = pthresh0               
                  elif mod_inputs.lower() == 'ir+vis':  
                    use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                    pthresh0    = 0.35
                    opt_pthres0 = pthresh0               
                  elif mod_inputs.lower() == 'ir+glm':
                    use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_glm', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                    pthresh0    = 0.45
                    opt_pthres0 = pthresh0               
                  elif mod_inputs.lower() == 'ir+wvirdiff':
                    use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_wvirdiff', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                    pthresh0    = 0.45
                    opt_pthres0 = pthresh0               
                  elif mod_inputs.lower() == 'ir+vis+glm':
                    use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis_glm', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                    pthresh0    = 0.45  
                    opt_pthres0 = pthresh0               
                else:
                  print('AACP model not set up to use unet model.. ONLY multiresunet is available!!')
                  exit()
            elif mod_type.lower() == 'attentionunet':
              mod_check     = 1
              mod_type      = 'attentionunet'
              use_native_ir = False
              if transition != 'y':
                if mod_inputs.lower() == 'ir':
                  use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                  print('Optimal pthresh must still be entered!!!')
                  print(mod_inputs)
                  exit()
                elif mod_inputs.lower() == 'ir+vis':  
                  use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                  pthresh0    = 0.65
                  opt_pthres0 = pthresh0               
                elif mod_inputs.lower() == 'ir+glm':
                  use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_glm', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                  print('Optimal pthresh must still be entered!!!')
                  print(mod_inputs)
                  exit()
                elif mod_inputs.lower() == 'ir+wvirdiff':
                  use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_wvirdiff', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                  print('Optimal pthresh must still be entered!!!')
                  print(mod_inputs)
                  exit()
                elif mod_inputs.lower() == 'ir+vis+glm':
                  use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis_glm', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                  pthresh0    = 0.20
                  opt_pthres0 = pthresh0               
              else:
                print('AACP model not set up to use attentionunet model.. ONLY multiresunet is available!!')
                exit()
            else:
              print('Model type entered is not available. Options are multiresunet, unet, or attentionunet. Please try again.')
              print()

#             counter   = 0                                                                                                         #Initialize variable to store the number of times the program attempted to ascertain information from the user 
#             mod_check = 0                                                                                                         #Initialize variable to store whether or not the user entered the appropriate information 
#             while mod_check == 0:
#               counter = counter + 1
#               if counter == 4:
#                 print('Too many failed attempts! Exiting program.')
#                 exit()
#               mod_loc = str(input('Enter the desired severe weather indicator (OT or AACP): ')).replace("'", '')
#               mod_loc = ''.join(e for e in mod_loc if e.isalnum())
#               if mod_loc.lower() == 'updraft' or mod_loc.lower() == 'ot' or mod_loc.lower() == 'updrafts' or mod_loc.lower() == 'ots':
#                 mod_loc     = 'OT'
#                 use_updraft = True
#                 mod_check   = 1
#               elif mod_loc.lower() == 'plume' or mod_loc.lower() == 'aacp' or mod_loc.lower() == 'plumes' or mod_loc.lower() == 'aacps' or mod_loc.lower() == 'warmplume' or mod_loc.lower() == 'warmplumes':
#                 mod_loc     = 'AACP'
#                 use_updraft = False
#                 mod_check   = 1
#               else:
#                 print('Severe weather indicator entered is not available. Desired severe weather indicator must be OT or AACP!!!')
#                 print('You entered : ' + mod_loc)
#                 print('Please try again.')
#                 print()
          elif dd == 5:
            counter   = 0                                                                                                          #Initialize variable to store the number of times the program attempted to ascertain information from the user 
            mod_check = 0                                                                                                          #Initialize variable to store whether or not the user entered the appropriate information 
            while mod_check == 0:
              counter = counter + 1
              if counter == 4:
                print('Too many failed attempts! Exiting program.')
                print('You must enter multiresunet, unet, or attentionunet ONLY.')
                exit()
              mod_type = str(input('Enter the model inputs you would like to use. Options are multiresunet, unet, attentionunet: ')).replace("'", '')    
              mod_type = ''.join(e for e in mod_type if e.isalnum())
              if mod_type.lower() == 'multiresunet':  
                mod_check = 1
                mod_type  = 'multiresunet'
                if transition != 'y':
                  if mod_inputs.lower() == 'ir' or mod_inputs.lower() == 'ir+glm' or mod_inputs.lower() == 'ir+wvirdiff' or mod_inputs.lower() == 'tropdiff' or mod_inputs.lower() == 'ir+dirtyirdiff'  or mod_inputs.lower() == 'ir+tropdiff' or mod_inputs.lower() == 'tropdiff+glm' or mod_inputs.lower() == 'tropdiff+dirtyirdiff':
                    use_native_ir = True
                  elif mod_inputs.lower() == 'ir+vis' or mod_inputs.lower() == 'ir+vis+glm' or mod_inputs.lower() == 'ir+snowice' or mod_inputs.lower() == 'ir+cirrus' or mod_inputs.lower() == 'vis+tropdiff' or mod_inputs.lower() == 'ir+vis+tropdiff' or mod_inputs.lower() == 'vis+tropdiff+glm' or mod_inputs.lower() == 'ir+vis+dirtyirdiff' or mod_inputs.lower() == 'vis+tropdiff+dirtyirdiff':  
                    use_native_ir = False
                  else:
                    print('Inputs specified not found within if statement. Please check to make sure this is correct and not in error.')
                    print(mod_inputs)
                    print(mod_type)
                    exit()
                  day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                  pthresh0    = day0[3]
                  use_chkpnt  = os.path.realpath(day0[2])
                  opt_pthres0 = pthresh0
              elif mod_type.lower() == 'unet':  
                mod_check = 1
                mod_type = 'unet'
                if transition != 'y':
                  if mod_loc == 'OT':
                    if mod_inputs.lower() == 'ir':
                      use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      pthresh0    = 0.45
                      opt_pthres0 = pthresh0               
                    elif mod_inputs.lower() == 'ir+vis':  
                      use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      pthresh0    = 0.35
                      opt_pthres0 = pthresh0               
                    elif mod_inputs.lower() == 'ir+glm':
                      use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_glm', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      pthresh0    = 0.45
                      opt_pthres0 = pthresh0               
                    elif mod_inputs.lower() == 'ir+irdiff':
                      use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_irdiff', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      pthresh0    = 0.45
                      opt_pthres0 = pthresh0               
                    elif mod_inputs.lower() == 'ir+vis+glm':
                      use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis_glm', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      pthresh0    = 0.45  
                      opt_pthres0 = pthresh0               
                  else:
                    print('AACP model not set up to use unet model.. ONLY multiresunet is available!!')
                    exit()
              elif mod_type.lower() == 'attentionunet':
                mod_check = 1
                mod_type  = 'attentionunet'
                if transition != 'y':
                  if mod_inputs.lower() == 'ir':
                    use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                    print('Optimal pthresh must still be entered!!!')
                    print(mod_inputs)
                    exit()
                  elif mod_inputs.lower() == 'ir+vis':  
                    use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                    pthresh0    = 0.65
                    opt_pthres0 = pthresh0               
                  elif mod_inputs.lower() == 'ir+glm':
                    use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_glm', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                    print('Optimal pthresh must still be entered!!!')
                    print(mod_inputs)
                    exit()
                  elif mod_inputs.lower() == 'ir+irdiff':
                    use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_irdiff', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                    print('Optimal pthresh must still be entered!!!')
                    print(mod_inputs)
                    exit()
                  elif mod_inputs.lower() == 'ir+vis+glm':
                    use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis_glm', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                    pthresh0    = 0.20
                    opt_pthres0 = pthresh0               
                else:
                  print('AACP model not set up to use attentionunet model.. ONLY multiresunet is available!!')
                  exit()
                  
              else:
                print('Model type entered is not available. Options are multiresunet, unet, or attentionunet. Please try again.')
                print()
          elif dd == 6:
            if transition == 'y':
              transition = 'n'
              opt_params = 'n'
            else:
              day_night0 = str(input('Would you like the model to seamlessly transition between previously identified optimal models? (y/n): ')).replace("'", '')
              if day_night0 == '':
                day_night0 = 'y'
              day_night0 = ''.join(e for e in day_night0 if e.isalnum())                                                           #Remove any special characters or spaces.
              if day_night0.lower() == 'y':
                use_night  = True
                opt_params = 'y'
                transition = 'y'
                use_glm    = True
              else:
                opt_params = 'n'
                transition = 'n'

            opt_params = ''.join(e for e in opt_params if e.isalnum())                                                             #Remove any special characters or spaces.
            if opt_params[0].lower() == 'y':
              if transition == 'y':
                day0    = best_model_checkpoint_and_thresh('best_day', mod_loc, native_ir = use_native_ir, drcs = drcs) 
                night0  = best_model_checkpoint_and_thresh('best_night', mod_loc, native_ir = use_native_ir, drcs = drcs)
                day_inp = re.split(r'\+', day0[0].lower())
                nig_inp = re.split(r'\+', night0[0].lower())
                use_trop        = False
                use_dirtyirdiff = False
                use_cirrus      = False
                use_snowice     = False
                use_glm         = False
                use_irdiff      = False
                if 'tropdiff' in day_inp:
                  use_trop = True
                if 'wvirdiff' in day_inp:
                  use_irdiff = True
                if 'dirtyirdiff' in day_inp:
                  use_dirtyirdiff = True
                if 'cirrus' in day_inp:
                  use_cirrus = True
                if 'snowice' in day_inp:
                  use_snowice = True
                if 'glm' in day_inp:
                  use_glm = True
                day   = {'mod_type': day0[1],   'mod_inputs': day0[0],   'use_chkpnt': os.path.realpath(day0[2]),   'pthresh': day0[3],   'use_night' : False, 'use_trop' : use_trop, 'use_dirtyirdiff' : use_dirtyirdiff, 'use_cirrus' : use_cirrus, 'use_snowice' : use_snowice, 'use_glm' : use_glm, 'use_irdiff' : use_irdiff}
                
                use_trop        = False
                use_dirtyirdiff = False
                use_cirrus      = False
                use_snowice     = False
                use_glm         = False
                use_irdiff      = False
                if 'tropdiff' in nig_inp:
                  use_trop = True
                if 'wvirdiff' in nig_inp:
                  use_irdiff = True
                if 'dirtyirdiff' in nig_inp:
                  use_dirtyirdiff = True
                if 'cirrus' in nig_inp:
                  use_cirrus = True
                if 'snowice' in nig_inp:
                  use_snowice = True
                if 'glm' in nig_inp:
                  use_glm = True
                night = {'mod_type': night0[1], 'mod_inputs': night0[0], 'use_chkpnt': os.path.realpath(night0[2]), 'pthresh': night0[3], 'use_night' : True, 'use_trop' : use_trop, 'use_dirtyirdiff' : use_dirtyirdiff, 'use_cirrus' : use_cirrus, 'use_snowice' : use_snowice, 'use_glm' : use_glm, 'use_irdiff' : use_irdiff}
                print(day['mod_inputs']   + ' ' + day['mod_type']   + ' ' + mod_loc + ' model will be used during day time scans.')
                print(night['mod_inputs'] + ' ' + night['mod_type'] + ' ' + mod_loc + ' model will be used during night time scans.')
                print()
            else:  
              counter   = 0                                                                                                        #Initialize variable to store the number of times the program attempted to ascertain information from the user 
              mod_check = 0                                                                                                        #Initialize variable to store whether or not the user entered the appropriate information 
              while mod_check == 0:
                counter = counter + 1
                if counter == 4:
                  print('Too many failed attempts! Exiting program.')
                  print('You must enter IR, IR+VIS, IR+GLM, IR+IRDIFF, or IR+VIS+GLM ONLY.')
                  exit()
                mod_inputs = str(input('Enter the model inputs you would like to use (ex. IR+VIS): ')).replace("'", '')
                str_list   = re.split(',| ', mod_inputs)
                while '' in str_list:
                  str_list.remove('')
                
                mod_inputs = '+'.join(str_list)
                if mod_inputs.lower() == 'ir':  
                  mod_inputs = 'IR'
                  mod_check  = 1
                  use_night  = True
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      use_native_ir = True
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':  
                      use_chkpnt    = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      pthresh0      = 0.45
                      opt_pthres0   = pthresh0     
                      use_native_ir = False          
                    elif mod_type == 'attentionunet':  
                      use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      print('Optimal pthresh must still be entered!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    use_native_ir = True
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif mod_inputs.lower() == 'ir+vis' or mod_inputs.lower() == 'vis+ir':  
                  mod_inputs    = 'IR+VIS'
                  mod_check     = 1
                  use_night     = False
                  use_native_ir = False          
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      pthresh0    = 0.35
                      opt_pthres0 = pthresh0               
                    elif mod_type == 'attentionunet':
                      use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      pthresh0    = 0.65
                      opt_pthres0 = pthresh0               
                  else:
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif mod_inputs.lower() == 'ir+glm' or mod_inputs.lower() == 'glm+ir':
                  mod_inputs = 'IR+GLM'
                  mod_check  = 1
                  use_glm    = True
                  use_night  = True
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      use_native_ir = True
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      use_chkpnt    = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_glm', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      pthresh0      = 0.45
                      opt_pthres0   = pthresh0               
                      use_native_ir = False          
                    elif mod_type == 'attentionunet':
                      use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_glm', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      print('Optimal pthresh must still be entered!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    use_native_ir = True
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif mod_inputs.lower() == 'ir+irdiff' or mod_inputs.lower() == 'irdiff+ir' or mod_inputs.lower() == 'wvirdiff+ir' or mod_inputs.lower() == 'ir+wvirdiff':  
                  mod_inputs = 'IR+WVIRDIFF'
                  mod_check  = 1
                  use_night  = True
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      use_native_ir = True
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      use_chkpnt    = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_irdiff', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      pthresh0      = 0.45
                      opt_pthres0   = pthresh0               
                      use_native_ir = False          
                    elif mod_type == 'attentionunet':
                      use_chkpnt = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_irdiff', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      print('Optimal pthresh must still be entered!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    use_native_ir = True
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif mod_inputs.lower() == 'ir+dirtyirdiff' or mod_inputs.lower() == 'dirtyirdiff+ir':  
                  mod_inputs = 'IR+DIRTYIRDIFF'
                  mod_check  = 1
                  use_night     = True
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      use_native_ir = True
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      print('No unet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                    elif mod_type == 'attentionunet':
                      print('No attentionunet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    use_native_ir = True
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif mod_inputs.lower() == 'tropdiff':  
                  mod_inputs = 'TROPDIFF'
                  mod_check  = 1
                  use_night     = True
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      use_native_ir = True
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      print('No unet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                    elif mod_type == 'attentionunet':
                      print('No attentionunet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    use_native_ir = True
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif mod_inputs.lower() == 'ir+snowice' or mod_inputs.lower() == 'snowice+ir':  
                  mod_inputs = 'IR+SNOWICE'
                  mod_check  = 1
                  use_night     = False
                  use_native_ir = False          
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      print('No unet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                    elif mod_type == 'attentionunet':
                      print('No attentionunet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif mod_inputs.lower() == 'ir+cirrus' or mod_inputs.lower() == 'cirrus+ir':  
                  mod_inputs = 'IR+CIRRUS'
                  mod_check  = 1
                  use_night     = False
                  use_native_ir = False          
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      print('No unet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                    elif mod_type == 'attentionunet':
                      print('No attentionunet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif mod_inputs.lower() == 'ir+tropdiff' or mod_inputs.lower() == 'tropdiff+ir':  
                  mod_inputs = 'IR+TROPDIFF'
                  mod_check  = 1
                  use_night     = True
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      use_native_ir = True
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      print('No unet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                    elif mod_type == 'attentionunet':
                      print('No attentionunet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    use_native_ir = True
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif mod_inputs.lower() == 'vis+tropdiff' or mod_inputs.lower() == 'tropdiff+vis':  
                  mod_inputs = 'VIS+TROPDIFF'
                  mod_check  = 1
                  use_night     = False
                  use_native_ir = False          
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      print('No unet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                    elif mod_type == 'attentionunet':
                      print('No attentionunet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif mod_inputs.lower() == 'tropdiff+glm' or mod_inputs.lower() == 'glm+tropdiff':  
                  mod_inputs = 'TROPDIFF+GLM'
                  mod_check  = 1
                  use_glm    = True
                  use_night  = True
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      use_native_ir = True
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      print('No unet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                    elif mod_type == 'attentionunet':
                      print('No attentionunet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    use_native_ir = True
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif mod_inputs.lower() == 'tropdiff+dirtyirdiff' or mod_inputs.lower() == 'dirtyirdiff+tropdiff':  
                  mod_inputs = 'TROPDIFF+DIRTYIRDIFF'
                  mod_check  = 1
                  use_night  = True
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      use_native_ir = True
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      print('No unet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                    elif mod_type == 'attentionunet':
                      print('No attentionunet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    use_native_ir = True
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif 'ir' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'tropdiff' in mod_inputs.lower(): 
                  mod_inputs = 'IR+VIS+TROPDIFF'
                  mod_check  = 1
                  use_night     = False
                  use_native_ir = False          
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      print('No unet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                    elif mod_type == 'attentionunet':
                      print('No attentionunet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif 'glm' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'tropdiff' in mod_inputs.lower(): 
                  mod_inputs = 'VIS+TROPDIFF+GLM'
                  mod_check  = 1
                  use_glm    = True
                  use_night     = False
                  use_native_ir = False          
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      print('No unet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                    elif mod_type == 'attentionunet':
                      print('No attentionunet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif 'ir' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'dirtyirdiff' in mod_inputs.lower(): 
                  mod_inputs = 'IR+VIS+DIRTYIRDIFF'
                  mod_check  = 1
                  use_night     = False
                  use_native_ir = False          
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      print('No unet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                    elif mod_type == 'attentionunet':
                      print('No attentionunet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif 'dirtyirdiff' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'tropdiff' in mod_inputs.lower(): 
                  mod_inputs = 'VIS+TROPDIFF+DIRTYIRDIFF'
                  mod_check  = 1
                  use_night     = False
                  use_native_ir = False          
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      print('No unet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                    elif mod_type == 'attentionunet':
                      print('No attentionunet model available for specified inputs!!!')
                      print(mod_inputs)
                      exit()
                  else:
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                elif 'ir' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'glm' in mod_inputs.lower(): 
                  mod_inputs    = 'IR+VIS+GLM'
                  mod_check     = 1
                  use_glm       = True
                  use_night     = False
                  use_native_ir = False          
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                      pthresh0    = day0[3]
                      use_chkpnt  = os.path.realpath(day0[2])
                      opt_pthres0 = pthresh0
                    elif mod_type == 'unet':
                      use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis_glm', 'updraft_day_model', '2022-02-18', 'unet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      pthresh0    = 0.45
                      opt_pthres0 = pthresh0               
                    elif mod_type == 'attentionunet':
                      use_chkpnt  = os.path.realpath(os.path.join('..', '..', '..', 'aacp_results', 'ir_vis_glm', 'updraft_day_model', '2022-02-18', 'attentionunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'))
                      pthresh0    = 0.20
                      opt_pthres0 = pthresh0               
                  else:
                    day0        = best_model_checkpoint_and_thresh(mod_inputs, mod_loc, native_ir = use_native_ir, drcs = drcs)
                    pthresh0    = day0[3]
                    use_chkpnt  = os.path.realpath(day0[2])
                    opt_pthres0 = pthresh0
                else:
                  print('Model inputs entered are not available. Options are IR, IR+VIS, IR+GLM, IR+IRDIFF, and IR+VIS+GLM. Please try again.')
                  print()
                  
              opt_pthresh = str(input('Would you like to use the previously identified optimal ' + mod_loc + ' likelihood value for the ' + mod_inputs + ' ' + mod_type + ' model? (y/n): ')).replace("'", '')
              if opt_pthresh == '':
                opt_pthresh = 'y'
                pthresh     = pthresh0
              
              opt_pthresh = ''.join(e for e in opt_pthresh if e.isalnum())
              if opt_pthresh[0].lower() != 'y':
                counter   = 0                                                                                                            #Initialize variable to store the number of times the program attempted to ascertain information from the user 
                mod_check = 0                                                                                                            #Initialize variable to store whether or not the user entered the appropriate information 
                while mod_check == 0:
                  counter = counter + 1
                  if counter == 4:
                    print('Too many failed attempts! Exiting program.')
                    exit()
                  opt_pthresh = input('Enter the likelihood value you would like to use (0-1): ')
                  try:
                    opt_pthresh = float(opt_pthresh)
                    if opt_pthresh < 0 or opt_pthresh > 1:
                      print('Likelihood value must be between 0 and 1. Please try again.')
                      print()
                    else:
                      mod_check = 1
                      pthresh   = opt_pthresh
                  except:
                    print('Wrong type entered! Likelihood value must either be set to an integer or floating point value between 0 and 1. Please try again.')
                    print('Format type entered = ' + str(type(opt_pthresh)))
                    print()
              else:
                pthresh = pthresh0
              
              print()
                  
          elif dd == 7:
            if no_plot == True:
              print('This job will now output images.')
              no_plot = False
            else:
              print('This job will not output images.')
              no_plot = True  
          elif dd == 8:
            reg_bound = str(input('Is there a particular state or country you would like to contrain data to? If no, hit ENTER. If yes, enter the name of the state or country: ')).replace("'", '').strip()
            if reg_bound == '':
              xy_bounds = []
              region    = None
            else:  
              region  = extract_us_lat_lon_region('virginia')
              regions = list(region.keys())
              if reg_bound.lower() not in regions:
                print('State or country specified is not available???')
                print('You specified = ' + str(reg_bound.lower()))
                print('Regions possible = ' + str(sorted(regions)))
                print()
                counter   = 0                                                                                                      #Initialize variable to store the number of times the program attempted to ascertain information from the user 
                mod_check = 0                                                                                                      #Initialize variable to store whether or not the user entered the appropriate information 
                while mod_check == 0:
                  counter = counter + 1
                  if counter == 4:
                    print('Too many failed attempts! Exiting program.')
                    exit()
                  reg_bound = str(input('Is there a particular state or country you would like to contrain data to? If no, hit ENTER. If yes, enter the name of the state or country: ')).replace("'", '').strip()
                  if reg_bound.lower() in regions or reg_bound == '':
                    mod_check = 1
                  else:  
                    print('State or country specified is not available???')
                    print('You specified = ' + str(reg_bound.lower()))
                    print()
              if reg_bound != '':
                xy_bounds = region[reg_bound.lower()]
                region    = reg_bound.lower()
          elif dd == 9:
            mod_bound = str(input('Do you want to constrain the data to particular longitude and latitude bounds? (y/n): ')).replace("'", '')
            if mod_bound == '':
              xy_bounds = []
              print('Data will be not be constrained to longitude and latitude bounds')
              print()
            else:  
              mod_bound = ''.join(e for e in mod_bound if e.isalnum())                                                             #Remove any special characters or spaces.
              if mod_bound[0].lower() == 'y':
                counter   = 0                                                                                                      #Initialize variable to store the number of times the program attempted to ascertain information from the user 
                mod_check = 0                                                                                                      #Initialize variable to store whether or not the user entered the appropriate information 
                while mod_check == 0:
                  counter = counter + 1
                  if counter == 4:
                    print('Too many failed attempts! Exiting program.')
                    exit()
                  mod_x0 = input('Enter the minimum longitude point. Values must be between -180.0 and 180.0: ')
                  try:
                    mod_x0 = float(mod_x0)
                    if mod_x0 >= -180.0 and mod_x0 <= 180.0:
                      x0        = mod_x0
                      mod_check = 1
                    else:  
                      print('Minimum longitude point must be between -180.0 and 180.0. Please try again.')
                      print()
                  except:
                    print('Wrong data type entered! Please try again.')
                    print('Format type entered = ' + str(type(mod_x0)))
                    print()
                
                counter   = 0                                                                                                      #Initialize variable to store the number of times the program attempted to ascertain information from the user 
                mod_check = 0                                                                                                      #Initialize variable to store whether or not the user entered the appropriate information 
                while mod_check == 0:
                  counter = counter + 1
                  if counter == 4:
                    print('Too many failed attempts! Exiting program.')
                    exit()
                  mod_x0 = input('Enter the maximum longitude point. Values must be between -180.0 and 180.0: ')
                  try:
                    mod_x0 = float(mod_x0)
                    if mod_x0 >= -180.0 and mod_x0 <= 180.0:
                      if mod_x0 > x0:
                        x1        = mod_x0
                        mod_check = 1
                      else:
                        print('Maximum longitude point must be > minimum longitude point. Please try again.')
                        print('Minimum longitude point entered = ' + str(x0))
                        print('Maximum longitude point entered = ' + str(mod_x0))
                        print()
                    else:  
                      print('Maximum longitude point must be between minimum longitude point and 180.0. Please try again.')
                      print()
                  except:
                    print('Wrong data type entered! Please try again.')
                    print('Format type entered = ' + str(type(mod_x0)))
                    print()
              
                counter   = 0                                                                                                      #Initialize variable to store the number of times the program attempted to ascertain information from the user 
                mod_check = 0                                                                                                      #Initialize variable to store whether or not the user entered the appropriate information 
                while mod_check == 0:
                  counter = counter + 1
                  if counter == 4:
                    print('Too many failed attempts! Exiting program.')
                    exit()
                  mod_x0 = input('Enter the minimum latitude point. Values must be between -90.0 and 90.0: ')
                  try:
                    mod_x0 = float(mod_x0)
                    if mod_x0 >= -90.0 and mod_x0 <= 90.0:
                      y0        = mod_x0
                      mod_check = 1
                    else:  
                      print('Minimum latitude point must be between -90.0 and 90.0. Please try again.')
                      print()
                  except:
                    print('Wrong data type entered! Please try again.')
                    print('Format type entered = ' + str(type(mod_x0)))
                    print()
                
                counter   = 0                                                                                                      #Initialize variable to store the number of times the program attempted to ascertain information from the user 
                mod_check = 0                                                                                                      #Initialize variable to store whether or not the user entered the appropriate information 
                while mod_check == 0:
                  counter = counter + 1
                  if counter == 4:
                    print('Too many failed attempts! Exiting program.')
                    exit()
                  mod_x0 = input('Enter the maximum latitude point. Values must be between -90.0 and 90.0: ')
                  try:
                    mod_x0 = float(mod_x0)
                    if mod_x0 >= -90.0 and mod_x0 <= 90.0:
                      if mod_x0 > y0:
                        y1        = mod_x0
                        mod_check = 1
                      else:
                        print('Maximum latitude point must be > minimum latitude point. Please try again.')
                        print('Minimum latitude point entered = ' + str(y0))
                        print('Maximum latitude point entered = ' + str(mod_x0))
                        print()
                    else:  
                      print('Maximum latitude point must be between minimum latitude point and 90.0. Please try again.')
                  except:
                    print('Wrong data type entered! Please try again.')
                    print('Format type entered = ' + str(type(mod_x0)))
                    print()
                xy_bounds = [x0, y0, x1, y1]
              else:
                xy_bounds = []
          elif dd == 10:
            if mod_loc.lower() == 'ot':
              counter   = 0                                                                                                          #Initialize variable to store the number of times the program attempted to ascertain information from the user 
              mod_check = 0                                                                                                          #Initialize variable to store whether or not the user entered the appropriate information 
              while mod_check == 0:
                percent_omit = str(input('For post-processing, Enter the percent of coldest and warmest pixels to omit from the anvil calculation to remove bias in determining OT IR-anvil BTD: ')).replace("'", '')
                counter = counter + 1
                if counter == 4:
                  print('Too many failed attempts! Exiting program.')
                  exit()
                if type(percent_omit) == str:
                  percent_omit = ''.join(e for e in percent_omit if e.isdigit() or e == '.' or e == '-')
                try:
                  percent_omit = float(percent_omit)
                  if percent_omit < 0 or percent_omit > 100:
                    print('You MUST enter only a number between 0 and 100 for % of pixels of pixels to be omitted from and mean brightness temperature calculation')
                    print('The default is 20')
                    print('You entered : ' + str(percent_omit))
                    print('Please try again.')
                    print()
                  else:
                    mod_check = 1
                except:
                  print('You MUST enter only a number between 0 and 100 for % of pixels of pixels to be omitted from and mean brightness temperature calculation')
                  print('The default is 20')
                  print('You entered : ' + str(percent_omit))
                  print('Please try again.')
                  print()
            else:
              print('The question specified to be changed does not matter for AACPs. If you intended on detecting OTs, please change that.')
          else:
            print('Invalid number entered.')
            print('You entered : ' + str(dd))
            print('Please try again.')
  
    if mod_loc == 'OT':
      use_updraft = True
    else:
      use_updraft = False    
  
  if 'mtg' in sat.lower() and len(xy_bounds) > 0:
    sector = mtg_extract_data_sector_from_region(xy_bounds[1], xy_bounds[3], verbose = verbose)
  else:
    if 'mtg' in sat.lower():
      sector = 'F'

  if drcs:
    use_glm = True
  if nhours == None:
    tstart = int(''.join(re.split('-', d_str1))[0:8])                                                                              #Extract start date of job for real-time post processing
    tend   = int(''.join(re.split('-', d_str2))[0:8])                                                                              #Extract end date of job for real-time post processing
    ts     = d_str1
    te     = d_str2
    t0     = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    if d_str2 > t0:
      print('Waiting until end date to start archived model run.')
      print('End date chosen = ' + d_str2)
      time.sleep((datetime.strptime(d_str2, "%Y-%m-%d %H:%M:%S") - datetime.strptime(t0, "%Y-%m-%d %H:%M:%S")).total_seconds())
      
    if transition == 'y':
      if drcs:
        day['use_glm'] = True
      chk_day_night = {'day': day, 'night': night}
      pthresh = None
      mod_inp = day_inp
      mod_inp.extend(nig_inp)
      run_tf_3_channel_plume_updraft_day_predict(date1          = d_str1, date2 = d_str2, 
                                                 use_updraft    = use_updraft, 
                                                 sat            = sat,
                                                 sector         = sector,
                                                 region         = None, 
                                                 xy_bounds      = xy_bounds, 
                                                 run_gcs        = run_gcs, 
                                                 use_local      = use_local, 
                                                 del_local      = del_local, 
                                                 inroot         = os.path.join(raw_data_root, 'aacp_results'), 
                                                 outroot        = raw_data_root, 
                                                 og_bucket_name = 'goes-data',
                                                 c_bucket_name  = 'ir-vis-sandwich',
                                                 p_bucket_name  = 'aacp-proc-data',
                                                 f_bucket_name  = 'aacp-results', 
                                                 chk_day_night  = chk_day_night, 
                                                 use_night      = True, 
                                                 grid_data      = True, 
                                                 no_plot        = no_plot, 
                                                 no_download    = no_download, 
                                                 rewrite_model  = True,
                                                 essl           = essl, 
                                                 verbose        = verbose)
    else:
      mod_inp = re.split(r'\+', mod_inputs)
      if mod_inputs.count('+') == 0:
        run_tf_1_channel_plume_updraft_day_predict(date1          = d_str1, date2 = d_str2, 
                                                   use_updraft    = use_updraft, 
                                                   sat            = sat,
                                                   sector         = sector,
                                                   region         = None, 
                                                   xy_bounds      = xy_bounds, 
                                                   pthresh        = pthresh,
                                                   opt_pthresh    = opt_pthres0,
                                                   run_gcs        = run_gcs, 
                                                   use_local      = use_local, 
                                                   del_local      = del_local, 
                                                   inroot         = os.path.join(raw_data_root, 'aacp_results'), 
                                                   outroot        = raw_data_root, 
                                                   og_bucket_name = 'goes-data',
                                                   c_bucket_name  = 'ir-vis-sandwich',
                                                   p_bucket_name  = 'aacp-proc-data',
                                                   f_bucket_name  = 'aacp-results', 
                                                   use_chkpnt     = use_chkpnt, 
                                                   use_night      = use_night, 
                                                   grid_data      = True, 
                                                   no_plot        = no_plot, 
                                                   no_download    = no_download, 
                                                   rewrite_model  = True,
                                                   use_native_ir  = use_native_ir, 
                                                   essl           = essl, 
                                                   verbose        = verbose)
      elif mod_inputs.count('+') == 1:
        run_tf_2_channel_plume_updraft_day_predict(date1          = d_str1, date2 = d_str2, 
                                                   use_glm        = False,
                                                   use_updraft    = use_updraft, 
                                                   sat            = sat,
                                                   sector         = sector,
                                                   region         = None, 
                                                   xy_bounds      = xy_bounds, 
                                                   pthresh        = pthresh,
                                                   opt_pthresh    = opt_pthres0,
                                                   run_gcs        = run_gcs, 
                                                   use_local      = use_local, 
                                                   del_local      = del_local, 
                                                   inroot         = os.path.join(raw_data_root, 'aacp_results'), 
                                                   outroot        = raw_data_root, 
                                                   og_bucket_name = 'goes-data',
                                                   c_bucket_name  = 'ir-vis-sandwich',
                                                   p_bucket_name  = 'aacp-proc-data',
                                                   f_bucket_name  = 'aacp-results', 
                                                   use_chkpnt     = use_chkpnt, 
                                                   use_night      = use_night, 
                                                   grid_data      = True, 
                                                   no_plot        = no_plot, 
                                                   no_download    = no_download, 
                                                   rewrite_model  = True,
                                                   use_native_ir  = use_native_ir, 
                                                   essl           = essl, 
                                                   verbose        = verbose)
      elif mod_inputs.count('+') == 2:
        run_tf_3_channel_plume_updraft_day_predict(date1          = d_str1, date2 = d_str2, 
                                                   use_updraft    = use_updraft, 
                                                   sat            = sat,
                                                   sector         = sector,
                                                   region         = None, 
                                                   xy_bounds      = xy_bounds, 
                                                   pthresh        = pthresh,
                                                   opt_pthresh    = opt_pthres0,
                                                   run_gcs        = run_gcs, 
                                                   use_local      = use_local, 
                                                   del_local      = del_local, 
                                                   inroot         = os.path.join(raw_data_root, 'aacp_results'), 
                                                   outroot        = raw_data_root, 
                                                   og_bucket_name = 'goes-data',
                                                   c_bucket_name  = 'ir-vis-sandwich',
                                                   p_bucket_name  = 'aacp-proc-data',
                                                   f_bucket_name  = 'aacp-results', 
                                                   use_chkpnt     = use_chkpnt, 
                                                   use_night      = use_night, 
                                                   grid_data      = True, 
                                                   no_plot        = no_plot, 
                                                   no_download    = no_download, 
                                                   rewrite_model  = True,
                                                   essl           = essl, 
                                                   verbose        = verbose)
      else: 
        print('Not set up for specified model. You have encountered a bug. Exiting program.')
        exit()
  else:   
    if 'mtg' in sat.lower() or 'mti' in sat.lower():
        t_sec = sat_time_intervals(sat, sector = 'F')                                                                              #Extract time interval between satellite sector scan files (sec)    
    elif 'goes' in sat.lower():
        t_sec = sat_time_intervals(sat, sector = sector)                                                                           #Extract time interval between satellite sector scan files (sec)
    else:
        print('Something went wrong and satellite specified not found here')
        print('run_all_pliume_updraft_model_predict.py')
        exit()
    
    while datetime.utcnow().second < 5:
        sleep(1)                                                                                                                   #Wait until time has elapsed for new satellite scan file to be available
    ts     = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")                                                                       #Extract start date of job for post processing
    tstart = int(datetime.utcnow().strftime("%Y%m%d"))                                                                             #Extract start date of job for post processing
    t0     = time.time()                                                                                                           #Start clock to time the run job process
    tdiff  = 0.0                                                                                                                   #Initialize variable to calulate the amount of time that has passed
    lc     = 0                                                                                                                     #Initialize variable to store loop counter
    while (tdiff <= (nhours*3600.0)):
      if tdiff != 0.0:
        cc = 0
        while (time.time()-t0) < lc*t_sec:
          sleep(0.1)                                                                                                               #Wait until time has elapsed for new satellite scan file to be available
          if cc == 0 and verbose == True:
              print('Waiting for next model input files to come online')
              cc = 1
      lc = lc+1                                                                                                                    #Start counter to see when to start the next processing job
      if transition == 'y':
        if drcs:
          day['use_glm'] = True
        chk_day_night = {'day': day, 'night': night}
        pthresh = None
        mod_inp = day_inp
        mod_inp.extend(nig_inp)
        run_tf_3_channel_plume_updraft_day_predict(date1          = d_str1, date2 = d_str2, 
                                                   use_updraft    = use_updraft, 
                                                   sat            = sat,
                                                   sector         = sector,
                                                   region         = None, 
                                                   xy_bounds      = xy_bounds, 
                                                   run_gcs        = run_gcs, 
                                                   use_local      = use_local, 
                                                   del_local      = del_local, 
                                                   inroot         = os.path.join(raw_data_root, 'aacp_results'), 
                                                   outroot        = raw_data_root, 
                                                   og_bucket_name = 'goes-data',
                                                   c_bucket_name  = 'ir-vis-sandwich',
                                                   p_bucket_name  = 'aacp-proc-data',
                                                   f_bucket_name  = 'aacp-results', 
                                                   chk_day_night  = chk_day_night, 
                                                   use_night      = True, 
                                                   grid_data      = True, 
                                                   no_plot        = no_plot, 
                                                   no_download    = no_download, 
                                                   rewrite_model  = True,
                                                   essl           = essl, 
                                                   verbose        = verbose)
      else:
        mod_inp = re.split(r'\+', mod_inputs)
        if mod_inputs.count('+') == 0:
          run_tf_1_channel_plume_updraft_day_predict(date1          = d_str1, date2 = d_str2, 
                                                     use_updraft    = use_updraft, 
                                                     sat            = sat,
                                                     sector         = sector,
                                                     region         = None, 
                                                     xy_bounds      = xy_bounds, 
                                                     pthresh        = pthresh,
                                                     opt_pthresh    = opt_pthres0,
                                                     run_gcs        = run_gcs, 
                                                     use_local      = use_local, 
                                                     del_local      = del_local, 
                                                     inroot         = os.path.join(raw_data_root, 'aacp_results'), 
                                                     outroot        = raw_data_root, 
                                                     og_bucket_name = 'goes-data',
                                                     c_bucket_name  = 'ir-vis-sandwich',
                                                     p_bucket_name  = 'aacp-proc-data',
                                                     f_bucket_name  = 'aacp-results', 
                                                     use_chkpnt     = use_chkpnt, 
                                                     use_night      = use_night, 
                                                     grid_data      = True, 
                                                     no_plot        = no_plot, 
                                                     no_download    = no_download, 
                                                     rewrite_model  = True,
                                                     use_native_ir  = use_native_ir, 
                                                     essl           = essl, 
                                                     verbose        = verbose)
        elif mod_inputs.count('+') == 1:
          run_tf_2_channel_plume_updraft_day_predict(date1          = d_str1, date2 = d_str2, 
                                                     use_glm        = False,
                                                     use_updraft    = use_updraft, 
                                                     sat            = sat,
                                                     sector         = sector,
                                                     region         = None, 
                                                     xy_bounds      = xy_bounds, 
                                                     pthresh        = pthresh,
                                                     opt_pthresh    = opt_pthres0,
                                                     run_gcs        = run_gcs, 
                                                     use_local      = use_local, 
                                                     del_local      = del_local, 
                                                     inroot         = os.path.join(raw_data_root, 'aacp_results'), 
                                                     outroot        = raw_data_root, 
                                                     og_bucket_name = 'goes-data',
                                                     c_bucket_name  = 'ir-vis-sandwich',
                                                     p_bucket_name  = 'aacp-proc-data',
                                                     f_bucket_name  = 'aacp-results', 
                                                     use_chkpnt     = use_chkpnt, 
                                                     use_night      = use_night, 
                                                     grid_data      = True, 
                                                     no_plot        = no_plot, 
                                                     no_download    = no_download, 
                                                     rewrite_model  = True,
                                                     use_native_ir  = use_native_ir, 
                                                     essl           = essl, 
                                                     verbose        = verbose)
        elif mod_inputs.count('+') == 2:
          run_tf_3_channel_plume_updraft_day_predict(date1          = d_str1, date2 = d_str2, 
                                                     use_updraft    = use_updraft, 
                                                     sat            = sat,
                                                     sector         = sector,
                                                     region         = None, 
                                                     xy_bounds      = xy_bounds, 
                                                     pthresh        = pthresh,
                                                     opt_pthresh    = opt_pthres0,
                                                     run_gcs        = run_gcs, 
                                                     use_local      = use_local, 
                                                     del_local      = del_local, 
                                                     inroot         = os.path.join(raw_data_root, 'aacp_results'), 
                                                     outroot        = raw_data_root, 
                                                     og_bucket_name = 'goes-data',
                                                     c_bucket_name  = 'ir-vis-sandwich',
                                                     p_bucket_name  = 'aacp-proc-data',
                                                     f_bucket_name  = 'aacp-results', 
                                                     use_chkpnt     = use_chkpnt, 
                                                     use_night      = use_night, 
                                                     grid_data      = True, 
                                                     no_plot        = no_plot, 
                                                     no_download    = no_download, 
                                                     rewrite_model  = True,
                                                     essl           = essl, 
                                                     verbose        = verbose)
        else: 
          print('Not set up for specified model. You have encountered a bug. Exiting program.')
          exit()
      backend.clear_session()  
      tdiff = time.time() - t0
    
    tend = int(datetime.utcnow().strftime("%Y%m%d"))                                                                               #Extract end date of job for real-time post processing
    te   = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")                                                                         #Extract end date of job for post processing
    
  
  object_type = 'OT' if use_updraft == True else 'AACP'
  if transition == 'y':
    mod_type = day['mod_type']
  
  print('Starting post-processing of ' + object_type + ' ' + mod_type + ' job')
  if nhours != None:
    time.sleep(2)

  if 'tropdiff' not in mod_inp and 'TROPDIFF' not in mod_inp:
    download_gfs_analysis_files(ts, te,
                                GFS_ANALYSIS_DT = 21600,
                                write_gcs       = run_gcs,
                                del_local       = del_local,
                                verbose         = verbose)
  
  tstart0 = datetime.strptime(str(tstart), "%Y%m%d")
  tend0   = datetime.strptime(str(tend), "%Y%m%d")
  while tstart0 <= tend0:
    u = tstart0.strftime("%Y%m%d")
    directory0 = os.path.join(raw_data_root, 'combined_nc_dir', str(u))
    if os.path.isdir(directory0) and any(os.scandir(directory0)):
        run_write_severe_storm_post_processing(inroot        = directory0, 
                                               use_local     = use_local, write_gcs = run_gcs, del_local = del_local,
                                               c_bucket_name = 'ir-vis-sandwhich',
                                               object_type   = object_type,
                                               mod_type      = mod_type,
                                               sector        = sector,
                                               pthresh       = pthresh,
                                               percent_omit  = percent_omit,
                                               verbose       = verbose)
    tstart0 = tstart0 + timedelta(days=1)
    
def best_model_checkpoint_and_thresh(mod_input, mod_object, native_ir = False, drcs = False):
  '''
      This is a function to return the path to the best model checkpoint files as well as the likelihood threshold, given
      the model inputs (e.g. 'IR', 'IR+VIS', etc.) the User wants and the model object the User hopes to detect (e.g. 'OT' or 'AACP'). 
  Args:
      mod_input  : STRING specifying the GOES channels should be input to the model. (ex. 'IR', 'IR+VIS', etc.)
      mod_object : STRING specifying what the User wants to detect (ex. 'OT' or 'AACP')
  Keywords:
      native_ir : IF keyword set (True), return models that work on native IR data grid
      drcs      : IF keyword set (True), return models that work on native IR data grid
  Output:
      Returns the path to the checkpoint files as well as the model optimal likelihood threhsolds found in testing 
  '''  
  if mod_object.lower() == 'ot' or mod_object.lower() == 'updraft':
    
    if native_ir:
      best = {
              'IR'                   : ['IR',                   'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir',                   'updraft_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.2],
              'TROPDIFF'             : ['TROPDIFF',             'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff',             'updraft_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.4],
              'IR+GLM'               : ['IR+GLM',               'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_glm',               'updraft_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.25],
              'IR+WVIRDIFF'          : ['IR+WVIRDIFF',          'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_wvirdiff',          'updraft_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.55],
              'IR+DIRTYIRDIFF'       : ['IR+DIRTYIRDIFF',       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_dirtyirdiff',       'updraft_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.25],
              'IR+TROPDIFF'          : ['IR+TROPDIFF',          'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_tropdiff',          'updraft_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.4],
              'TROPDIFF+GLM'         : ['TROPDIFF+GLM',         'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff_glm',         'updraft_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.5], 
              'TROPDIFF+DIRTYIRDIFF' : ['TROPDIFF+DIRTYIRDIFF', 'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff_dirtyirdiff', 'updraft_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.45], 
              'BEST'                 : ['IR',                   'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir',                   'updraft_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.2]
              }
    else:
      if drcs:
        best = {
                'IR'                       : ['IR',                       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir',                       'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.4],
                'TROPDIFF'                 : ['TROPDIFF',                 'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff',                 'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.65],
                'IR+VIS'                   : ['IR+VIS',                   'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis',                   'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.25],
                'IR+GLM'                   : ['IR+GLM',                   'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_glm',                   'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.65],
                'IR+WVIRDIFF'              : ['IR+WVIRDIFF',              'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_wvirdiff',              'updraft_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.3],
                'IR+SNOWICE'               : ['IR+SNOWICE',               'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_snowice',               'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.6],
                'IR+CIRRUS'                : ['IR+CIRRUS',                'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_cirrus',                'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.5],
                'IR+DIRTYIRDIFF'           : ['IR+DIRTYIRDIFF',           'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_dirtyirdiff',           'updraft_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.45],
                'IR+TROPDIFF'              : ['IR+TROPDIFF',              'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_tropdiff',              'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.25],
                'VIS+TROPDIFF'             : ['VIS+TROPDIFF',             'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'vis_tropdiff',             'updraft_day_model', '2025-03-31', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.15],
                'TROPDIFF+GLM'             : ['TROPDIFF+GLM',             'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff_glm',             'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.55],
                'IR+VIS+GLM'               : ['IR+VIS+GLM',               'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_glm',               'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.4],
                'IR+VIS+TROPDIFF'          : ['IR+VIS+TROPDIFF',          'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_tropdiff',          'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.15],
                'VIS+TROPDIFF+GLM'         : ['VIS+TROPDIFF+GLM',         'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'vis_tropdiff_glm',         'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.4],
                'TROPDIFF+DIRTYIRDIFF'     : ['TROPDIFF+DIRTYIRDIFF',     'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff_dirtyirdiff',     'updraft_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.45],
                'IR+VIS+DIRTYIRDIFF'       : ['IR+VIS+DIRTYIRDIFF',       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_dirtyirdiff',       'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.3],
                'VIS+TROPDIFF+DIRTYIRDIFF' : ['VIS+TROPDIFF+DIRTYIRDIFF', 'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'vis_tropdiff_dirtyirdiff', 'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.45],
                'BEST_DAY'                 : ['VIS+TROPDIFF',             'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'vis_tropdiff',             'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.15],
                'BEST_NIGHT'               : ['TROPDIFF',                 'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff',                 'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.65]
#                 'BEST_DAY'                 : ['IR+VIS+GLM',               'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_glm',               'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.4],
#                 'BEST_NIGHT'               : ['IR',                       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir',                       'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.4]
                }
      else:
        best = {
                'IR'                       : ['IR',                       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir',                       'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.4],
                'TROPDIFF'                 : ['TROPDIFF',                 'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff',                 'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.65],
                'IR+VIS'                   : ['IR+VIS',                   'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis',                   'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.25],
                'IR+GLM'                   : ['IR+GLM',                   'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_glm',                   'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.65],
                'IR+WVIRDIFF'              : ['IR+WVIRDIFF',              'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_wvirdiff',              'updraft_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.3],
                'IR+SNOWICE'               : ['IR+SNOWICE',               'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_snowice',               'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.6],
                'IR+CIRRUS'                : ['IR+CIRRUS',                'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_cirrus',                'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.5],
                'IR+DIRTYIRDIFF'           : ['IR+DIRTYIRDIFF',           'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_dirtyirdiff',           'updraft_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.45],
                'IR+TROPDIFF'              : ['IR+TROPDIFF',              'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_tropdiff',              'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.25],
                'VIS+TROPDIFF'             : ['VIS+TROPDIFF',             'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'vis_tropdiff',             'updraft_day_model', '2025-03-31', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.15],
                'TROPDIFF+GLM'             : ['TROPDIFF+GLM',             'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff_glm',             'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.55],
                'IR+VIS+GLM'               : ['IR+VIS+GLM',               'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_glm',               'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.4],
                'IR+VIS+TROPDIFF'          : ['IR+VIS+TROPDIFF',          'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_tropdiff',          'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.15],
                'VIS+TROPDIFF+GLM'         : ['VIS+TROPDIFF+GLM',         'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'vis_tropdiff_glm',         'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.4],
                'TROPDIFF+DIRTYIRDIFF'     : ['TROPDIFF+DIRTYIRDIFF',     'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff_dirtyirdiff',     'updraft_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.45],
                'IR+VIS+DIRTYIRDIFF'       : ['IR+VIS+DIRTYIRDIFF',       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_dirtyirdiff',       'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.3],
                'VIS+TROPDIFF+DIRTYIRDIFF' : ['VIS+TROPDIFF+DIRTYIRDIFF', 'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'vis_tropdiff_dirtyirdiff', 'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.45],
#                'BEST_DAY'                 : ['IR+VIS',                   'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis',                   'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.25],
#                'BEST_NIGHT'               : ['IR',                       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir',                       'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.4]
                'BEST_DAY'                 : ['VIS+TROPDIFF',             'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'vis_tropdiff',             'updraft_day_model', '2025-03-31', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.15],
                'BEST_NIGHT'               : ['TROPDIFF',                 'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff',                 'updraft_day_model', '2023-09-07', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.65]
                }
  
  elif mod_object.lower() == 'aacp' or mod_object.lower() == 'plume' or mod_object.lower() == 'above anvil cirrus plume':
    if native_ir == True:
      best = {
              'IR'                   : ['IR',                   'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir',                   'plume_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.3],
              'TROPDIFF'             : ['TROPDIFF',             'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff',             'plume_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.6],
              'IR+GLM'               : ['IR+GLM',               'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_glm',               'plume_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.3],
              'IR+WVIRDIFF'          : ['IR+WVIRDIFF',          'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_wvirdiff',          'plume_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.55],
              'IR+DIRTYIRDIFF'       : ['IR+DIRTYIRDIFF',       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_dirtyirdiff',       'plume_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.15],
              'IR+TROPDIFF'          : ['IR+TROPDIFF',          'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_tropdiff',          'plume_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.2],
              'TROPDIFF+GLM'         : ['TROPDIFF+GLM',         'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff_glm',         'plume_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.5],
              'TROPDIFF+DIRTYIRDIFF' : ['TROPDIFF+DIRTYIRDIFF', 'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff_dirtyirdiff', 'plume_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.15],
              'BEST'                 : ['IR+DIRTYIRDIFF',       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_dirtyirdiff',       'plume_day_model', '2023-11-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.15]
              }
    else:
      best = {
              'IR'                       : ['IR',                       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir',                       'plume_day_model', '2023-09-05', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.75],
              'TROPDIFF'                 : ['TROPDIFF',                 'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff',                 'plume_day_model', '2023-09-05', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.8],
              'IR+VIS'                   : ['IR+VIS',                   'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis',                   'plume_day_model', '2023-09-05', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.5],
              'IR+GLM'                   : ['IR+GLM',                   'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_glm',                   'plume_day_model', '2023-09-05', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.2],
              'IR+WVIRDIFF'              : ['IR+WVIRDIFF',              'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_wvirdiff',              'plume_day_model', '2023-09-05', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.8],
              'IR+SNOWICE'               : ['IR+SNOWICE',               'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_snowice',               'plume_day_model', '2023-09-05', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.25],
              'IR+CIRRUS'                : ['IR+CIRRUS',                'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_cirrus',                'plume_day_model', '2023-09-05', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.6],
              'IR+DIRTYIRDIFF'           : ['IR+DIRTYIRDIFF',           'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_dirtyirdiff',           'plume_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.45],
              'IR+TROPDIFF'              : ['IR+TROPDIFF',              'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_tropdiff',              'plume_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.8],
              'VIS+TROPDIFF'             : ['VIS+TROPDIFF',             'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'vis_tropdiff',             'plume_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.7],
              'TROPDIFF+GLM'             : ['TROPDIFF+GLM',             'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff_glm',             'plume_day_model', '2023-09-05', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.55],
              'IR+VIS+GLM'               : ['IR+VIS+GLM',               'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_glm',               'plume_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.35],
              'IR+VIS+TROPDIFF'          : ['IR+VIS+TROPDIFF',          'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_tropdiff',          'plume_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.7],
              'VIS+TROPDIFF+GLM'         : ['VIS+TROPDIFF+GLM',         'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'vis_tropdiff_glm',         'plume_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.3],
              'TROPDIFF+DIRTYIRDIFF'     : ['TROPDIFF+DIRTYIRDIFF',     'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'tropdiff_dirtyirdiff',     'plume_day_model', '2023-09-05', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.8],
              'VIS+TROPDIFF+DIRTYIRDIFF' : ['VIS+TROPDIFF+DIRTYIRDIFF', 'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'vis_tropdiff_dirtyirdiff', 'plume_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.3],
              'IR+VIS+DIRTYIRDIFF'       : ['IR+VIS+DIRTYIRDIFF',       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_dirtyirdiff',       'plume_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.3],
#              'IR+VIS+DIRTYIRDIFF'       : ['IR+VIS+DIRTYIRDIFF',       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_dirtyirdiff',       'plume_day_model', '2023-09-06', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.45],
#              'BEST_DAY'                 : ['IR+VIS+DIRTYIRDIFF',       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_dirtyirdiff',       'plume_day_model', '2023-09-06', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.45],
#              'BEST_DAY'                 : ['IR+VIS+DIRTYIRDIFF',       'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_dirtyirdiff',       'plume_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.3],
              'BEST_DAY'                 : ['IR+DIRTYIRDIFF',           'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_dirtyirdiff',           'plume_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.45],
              'BEST_NIGHT'               : ['IR+DIRTYIRDIFF',           'multiresunet', os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_dirtyirdiff',           'plume_day_model', '2023-12-21', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 0.45]
              }
  else:
    print('Not setup forobject detection specified! Please try again.')
    print(mod_object)
    exit()
  
  
  val = best.get(mod_input.upper(), 'Invalid Model Inputs provided')
  if val == 'Invalid Model Inputs provided':
      print(mod_input + ' inputs for ' + mod_object + ' is not set up to provide checkpoint files. Please specify new model inputs and try again.')
      exit()
  return(val)


def main():
    parser = argparse.ArgumentParser(
        description = 'svrstormsig software for detecting OTs and AACPs using machine learning and written by John W. Cooney')
    
    parser.add_argument(                                                                 #If config file set
        '-c', '--config',
        type=str, 
        help   = 'Path to the config file')
  
    parser.add_argument(                                                                 #If drcs set
        '-d', '--drcs',
        action = 'store_true',
        help   = 'Argument to run IR+VIS+GLM OT model for the DRCS because they want time aggregated GLM flash extent density plots.')   

    parser.add_argument(                                                                 #If drcs set
        '-e', '--essl',
        action = 'store_true',
        help   = 'Argument to run software to look for raw MTG data in directory /yyyy/mm/dd/ rather than the typical /yyyymmdd/ directory path structure.')   

    args = parser.parse_args()
    
    run_all_plume_updraft_model_predict(config_file = args.config, drcs = args.drcs, essl = args.essl)
    
#OLD method of parsing the command line inputs
#     args = sys.argv[1:]
#     config_file = None
#     if len(args) > 0:
#       if len(args) == 2 and args[0] == '--config':
#         config_file = args[1]
#       else:
#         if len(args) != 2:
#           print('Number of arguments not correct!!')
#           print(args)
#           exit()
#         if args[0] != '--config':
#           print('Not set up to handle specified argument!!')
#           print(args)
#           exit()  
#     run_all_plume_updraft_model_predict(config_file = config_file)
    
if __name__ == '__main__':
    main()