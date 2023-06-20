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
#                              2022-06-05. MAJOR REVISIONS. 1) Added post-processing function that yields object ID numbers of OTs and AACPs as well as
#                                          OT IR-anvil brightness temperature differences. 2) Fixed issue that occurred in full disk and CONUS scanned
#                                          model runs. The issue occurred at the very edge of the domain where the satellite view is off in to space on the
#                                          edge of Earth. This issue was fixed prior to release for OTs but AACPs needed to be fixed in a unique way.
#                              2022-06-13. MINOR REVISIONS. 1) Fixed issue where user wants to correct the model inputs or model type and not everything 
#                                          like the correct optimal model to be called followed with it. 2) Pass optimal_threshold for a given model to be written
#                                          to output netCDF file rather than whatever is chosen by user for post-processing and plotting. The one chosen by
#                                          the user is still passed into the subroutines for plotting and post-processing and is referred to as the likelihood_threshold
#                                          in order to differentiate it from the optimal_thresh attribute in the raw model likelihood output variables. Users can
#                                          identify objects in post-processing by specifying their own likelihood thresholds, rather than being forced to use the
#                                          optimal thresholds found for a particular model. 3) Decrease size of files by improving compression by setting 
#                                          least_significant_digits keyword when creating the variables.
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
#import metpy
#import xarray
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
from new_model.run_write_severe_storm_post_processing import run_write_severe_storm_post_processing
from glm_gridder.run_download_goes_ir_vis_l1b_glm_l2_data import *
from glm_gridder.run_download_goes_ir_vis_l1b_glm_l2_data_parallel import *
from glm_gridder.run_create_image_from_three_modalities2 import *
from glm_gridder.run_create_image_from_three_modalities import * 
from glm_gridder.run_download_goes_ir_vis_l1b_glm_l2_data import *
from glm_gridder.run_download_goes_ir_vis_l1b_glm_l2_data_parallel import *
from glm_gridder import turbo_colormap
from glm_gridder.glm_data_processing import *
from glm_gridder.cpt_convert import loadCPT                                                                                        #Import the CPT convert function        
from gridrad.rdr_sat_utils_jwc import extract_us_lat_lon_region, sat_time_intervals
from glm_gridder.dataScaler import DataScaler
from EDA.create_vis_ir_numpy_arrays_from_netcdf_files import *
from EDA.unet import unet
from EDA.MultiResUnet import MultiResUnet
from EDA.create_vis_ir_numpy_arrays_from_netcdf_files2 import *
#from visualize_results.visualize_time_aggregated_results import visualize_time_aggregated_results
from visualize_results.run_write_plot_model_result_predictions import write_plot_model_result_predictions
backend.clear_session()
tf.get_logger().setLevel('ERROR')
def run_all_plume_updraft_model_predict(verbose              = False, 
                                        config_file          = None, 
                                        ir_ot_best           = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir', 'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 
                                        ir_ot_native_best    = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir', 'updraft_day_model', '2022-11-08', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 
                                        irvis_ot_best        = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis', 'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 
                                        irglm_ot_best        = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_glm', 'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 
                                        irglm_ot_native_best = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_glm', 'updraft_day_model', '2022-11-23', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 
                                        irirdiff_ot_best     = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_irdiff', 'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 
                                        irvisglm_ot_best     = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_glm', 'updraft_day_model', '2022-02-18', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 
                                        ir_aacp_best         = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir', 'plume_day_model', '2022-05-02', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 
                                        irvis_aacp_best      = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis', 'plume_day_model', '2022-05-02', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 
#                                         irglm_aacp_best      = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_glm', 'plume_day_model', '2021-12-02', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'),
#                                         irglm_aacp_best      = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_glm', 'plume_day_model', '2022-09-30', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'),
                                        irglm_aacp_best      = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_glm', 'plume_day_model', '2022-05-02', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp'), 
#                                         irvisglm_aacp_best   = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_glm', 'plume_day_model', '2021-12-02', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp')):
#                                         irvisglm_aacp_best   = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_glm', 'plume_day_model', '2022-09-30', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp')):
                                        irvisglm_aacp_best   = os.path.join('..', '..', 'data', 'model_checkpoints', 'ir_vis_glm', 'plume_day_model', '2022-05-02', 'multiresunet', 'chosen_indices', 'by_date', 'by_updraft', 'unet_checkpoint.cp')):
  '''
      This is a script to run (by importing) programs to run previous models using checkpoint files for inputs that include
      IR, VIS, GLM, and IR_DIFF channels. 
  Args:
      None.
  Keywords:
      config_file        : STRING specifying path to configure file which allows user to not have to answer any questions when starting model.
                           DEFAULT = None -> answer questions to determine type of run
      ir_ot_best         : STRING specifying the best OT model with IR inputs
      ir_ot_native_best  : STRING specifying the best OT model with IR at native IR resolution input
      irvis_ot_best      : STRING specifying the best OT model with IR+VIS inputs
      irglm_ot_best      : STRING specifying the best OT model with IR+GLM inputs
      irirdiff_ot_best   : STRING specifying the best OT model with IR+IRDIFF inputs
      irvisglm_ot_best   : STRING specifying the best OT model with IR+VIS+GLM inputs
      ir_aacp_best       : STRING specifying the best AACP model with IR inputs
      irvis_aacp_best    : STRING specifying the best AACP model with IR+VIS inputs
      irglm_aacp_best    : STRING specifying the best AACP model with IR+GLM inputs
      irvisglm_aacp_best : STRING specifying the best AACP model with IR+VIS+GLM inputs
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

  os.chdir(sys.path[0])                                                                                                            #Change directory to the system path of file 
  use_native_ir = False                                                                                                            #Constant set up but will change to True if IR only model is being use
  if config_file == None:
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
            day   = {'mod_type': 'multiresunet', 'mod_inputs': 'IR+VIS+GLM', 'use_chkpnt': os.path.realpath(irvisglm_ot_best), 'pthresh': 0.40, 'use_night' : False, 'use_irdiff' : False, 'use_glm' : True}
            night = {'mod_type': 'multiresunet', 'mod_inputs': 'IR', 'use_chkpnt': os.path.realpath(ir_ot_best), 'pthresh': 0.40, 'use_night' : True, 'use_irdiff' : False, 'use_glm' : False}
            print(day['mod_inputs']   + ' ' + day['mod_type']   + ' OT model will be used during day time scans.')
            print(night['mod_inputs'] + ' ' + night['mod_type'] + ' OT model will be used during night time scans.')
            print()
          else:
            if use_night == False:
              mod_type    = 'multiresunet'
              mod_inputs  = 'IR+VIS+GLM'
              use_glm     = True
              use_chkpnt  = os.path.realpath(irvisglm_ot_best)
              pthresh     = 0.40
              opt_pthres0 = pthresh
            else:  
              mod_type      = 'multiresunet'
              mod_inputs    = 'IR'
#               use_chkpnt    = os.path.realpath(ir_ot_best)
#               pthresh       = 0.40
              use_chkpnt    = os.path.realpath(ir_ot_native_best)
              pthresh       = 0.20
              opt_pthres0   = pthresh
              use_native_ir = True
            print('You have chosen to run an ' + mod_inputs + ' ' + mod_type + ' OT model') 
        else:
          if transition == 'y':
            day   = {'mod_type': 'multiresunet', 'mod_inputs': 'IR+VIS+GLM', 'use_chkpnt': os.path.realpath(irvisglm_aacp_best), 'pthresh': 0.30, 'use_night' : False, 'use_irdiff' : False, 'use_glm' : True}
            night = {'mod_type': 'multiresunet', 'mod_inputs': 'IR+GLM', 'use_chkpnt': os.path.realpath(irglm_aacp_best), 'pthresh': 0.60, 'use_night' : True, 'use_irdiff' : False, 'use_glm' : True}
            print(day['mod_inputs']   + ' ' + day['mod_type']   + ' AACP model will be used during day time scans.')
            print(night['mod_inputs'] + ' ' + night['mod_type'] + ' AACP model will be used during night time scans.')
            print()          
          else:  
            if use_night == False:
              mod_type    = 'multiresunet'
              mod_inputs  = 'IR+VIS+GLM'
              use_glm     = True
              use_chkpnt  = os.path.realpath(irvisglm_aacp_best)
              pthresh     = 0.30
              opt_pthres0 = pthresh
            else:  
              mod_type    = 'multiresunet'
              mod_inputs  = 'IR+GLM'
              use_glm     = True
              use_chkpnt  = os.path.realpath(irglm_aacp_best)
              pthresh     = 0.60
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
#                 use_chkpnt = os.path.realpath(ir_ot_best)
#                 pthresh    = 0.40
                use_chkpnt    = os.path.realpath(ir_ot_native_best)
                pthresh       = 0.20
                opt_pthres0   = pthresh
                use_native_ir = True
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
              use_chkpnt  = os.path.realpath(ir_aacp_best)
              pthresh     = 0.80
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'ir+vis' or mod_inputs.lower() == 'vis+ir':  
            mod_inputs = 'IR+VIS'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                use_chkpnt  = os.path.realpath(irvis_ot_best)
                pthresh     = 0.25
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
              use_chkpnt = os.path.realpath(irvis_aacp_best)
              pthresh     = 0.40
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'ir+glm' or mod_inputs.lower() == 'glm+ir':
            mod_inputs = 'IR+GLM'
            mod_check  = 1
            use_glm    = True
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
#                 use_chkpnt = os.path.realpath(irglm_ot_best)
#                 pthresh    = 0.65
                use_chkpnt    = os.path.realpath(irglm_ot_native_best)
                pthresh       = 0.30
                opt_pthres0   = pthresh
                use_native_ir = True
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
              use_chkpnt  = os.path.realpath(irglm_aacp_best)
              pthresh     = 0.60
              opt_pthres0 = pthresh
          elif mod_inputs.lower() == 'ir+irdiff' or mod_inputs.lower() == 'irdiff+ir':  
            mod_inputs = 'IR+IRDIFF'
            mod_check  = 1
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                use_chkpnt  = os.path.realpath(irirdiff_ot_best)
                pthresh     = 0.35
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
              print('IR+IRDIFF is not available for AACPs. Please choose different model inputs.')
              print()
              mod_check = 0
              
          elif 'ir' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'glm' in mod_inputs.lower(): 
            mod_inputs = 'IR+VIS+GLM'
            mod_check  = 1
            use_glm    = True
            if mod_loc == 'OT':
              if mod_type == 'multiresunet':
                use_chkpnt  = os.path.realpath(irvisglm_ot_best)
                pthresh     = 0.40
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
              use_chkpnt  = os.path.realpath(irvisglm_aacp_best)
              pthresh     = 0.30
              opt_pthres0 = pthresh
          else:
            print('Model inputs entered are not available. Options are IR, IR+VIS, IR+GLM, IR+IRDIFF, and IR+VIS+GLM. Please try again.')
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
        no_download = False
     
#       if raw_data_root and not raw_data_root.endswith('/'):                                                                      #Make sure the specified root directory path has / at the end
#           raw_data_root += '/'
      
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
         
#           if raw_data_root and not raw_data_root.endswith('/'):                                                                    #Make sure the specified root directory path has / at the end
#             raw_data_root += '/'
  
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
        
#       dd = str(input('Would you like to plot model results on top of IR/VIS sandwich images for each individual scene? Note, time to plot may cause run to be slower. (y/n): ')).replace("'", '')
#       if dd == '':
#         dd = 'y'
#       dd = ''.join(e for e in dd if e.isalnum())
#       if dd[0].lower() == 'y':
#         no_plot = False
#       else:
#         no_plot = True
#       print()
      no_plot = True
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
      no_download = False
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
    if no_download == True and d_str1 != None:
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
      if mod_loc == 'OT':
        if transition == 'y':
          day   = {'mod_type': 'multiresunet', 'mod_inputs': 'IR+VIS+GLM', 'use_chkpnt': os.path.realpath(irvisglm_ot_best), 'pthresh': 0.40, 'use_night' : False, 'use_irdiff' : False, 'use_glm' : True}
          night = {'mod_type': 'multiresunet', 'mod_inputs': 'IR', 'use_chkpnt': os.path.realpath(ir_ot_best), 'pthresh': 0.40, 'use_night' : True, 'use_irdiff' : False, 'use_glm' : False}
      else:
        if transition == 'y':
          day   = {'mod_type': 'multiresunet', 'mod_inputs': 'IR+VIS+GLM', 'use_chkpnt': os.path.realpath(irvisglm_aacp_best), 'pthresh': 0.30, 'use_night' : False, 'use_irdiff' : False, 'use_glm' : True}
          night = {'mod_type': 'multiresunet', 'mod_inputs': 'IR+GLM', 'use_chkpnt': os.path.realpath(irglm_aacp_best), 'pthresh': 0.60, 'use_night' : True, 'use_irdiff' : False, 'use_glm' : True}
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
#               use_chkpnt = os.path.realpath(ir_ot_best)
#               pthresh    = 0.40
              use_chkpnt    = os.path.realpath(ir_ot_native_best)
              pthresh       = 0.20
              opt_pthres0   = pthresh
              use_native_ir = True
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
            use_chkpnt  = os.path.realpath(ir_aacp_best)
            pthresh     = 0.80
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'ir+vis' or mod_inputs.lower() == 'vis+ir':  
          mod_inputs = 'IR+VIS'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              use_chkpnt  = os.path.realpath(irvis_ot_best)
              pthresh     = 0.25
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
            use_chkpnt  = os.path.realpath(irvis_aacp_best)
            pthresh     = 0.40
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'ir+glm' or mod_inputs.lower() == 'glm+ir':
          mod_inputs = 'IR+GLM'
          mod_check  = 1
          use_glm    = True
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
#               use_chkpnt = os.path.realpath(irglm_ot_best)
#               pthresh    = 0.65
              use_chkpnt    = os.path.realpath(irglm_ot_native_best)
              pthresh       = 0.30
              opt_pthres0   = pthresh
              use_native_ir = True
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
            use_chkpnt  = os.path.realpath(irglm_aacp_best)
            pthresh     = 0.60
            opt_pthres0 = pthresh
        elif mod_inputs.lower() == 'ir+irdiff' or mod_inputs.lower() == 'irdiff+ir':  
          mod_inputs = 'IR+IRDIFF'
          mod_check  = 1
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              use_chkpnt  = os.path.realpath(irirdiff_ot_best)
              pthresh     = 0.35
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
            print('IR+IRDIFF is not available for AACPs. Please choose different model inputs.')
            print()
            mod_check = 0
            
        elif 'ir' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'glm' in mod_inputs.lower(): 
          mod_inputs = 'IR+VIS+GLM'
          mod_check  = 1
          use_glm    = True
          if mod_loc == 'OT':
            if mod_type == 'multiresunet':
              use_chkpnt  = os.path.realpath(irvisglm_ot_best)
              pthresh     = 0.40
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
            use_chkpnt  = os.path.realpath(irvisglm_aacp_best)
            pthresh     = 0.30
            opt_pthres0 = pthresh
        else:
          print('Model inputs entered are not available. Options are IR, IR+VIS, IR+GLM, IR+IRDIFF, and IR+VIS+GLM. Please try again.')
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

#     dd = vals[14]                                                                                                                 #If user wants to plot the data
#     dd = ''.join(e for e in dd if e.isalnum())
#     if dd[0].lower() == 'y':
#       no_plot = False
#     else:
#       no_plot = True
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
#        print('(7) You have chosen to plot the individual scenes (Note, this may cause a run-time slowdown)')  
        print()
        print('(7) This software is currently not set up to plot images due to licensing restrictions. We apologize for the inconvenience.')
      
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
                if mod_loc == 'OT':
                  if mod_inputs.lower() == 'ir':
#                     use_chkpnt    = os.path.realpath(ir_ot_best)
#                     pthresh0      = 0.40
                    use_chkpnt    = os.path.realpath(ir_ot_native_best)
                    pthresh0      = 0.20
                    opt_pthres0   = pthresh0               
                    use_native_ir = True
                  elif mod_inputs.lower() == 'ir+vis':  
                    use_chkpnt    = os.path.realpath(irvis_ot_best)
                    pthresh0      = 0.25
                    opt_pthres0   = pthresh0
                    use_native_ir = False
                  elif mod_inputs.lower() == 'ir+glm':
#                     use_chkpnt = os.path.realpath(irglm_ot_best)
#                     pthresh0   = 0.65
                    use_chkpnt    = os.path.realpath(irglm_ot_native_best)
                    pthresh0      = 0.30
                    opt_pthres0   = pthresh0               
                    use_native_ir = True
                  elif mod_inputs.lower() == 'ir+irdiff':
                    use_chkpnt    = os.path.realpath(irirdiff_ot_best)
                    pthresh0      = 0.35
                    opt_pthres0   = pthresh0               
                    use_native_ir = False
                  elif mod_inputs.lower() == 'ir+vis+glm':
                    use_chkpnt    = os.path.realpath(irvisglm_ot_best)
                    pthresh0      = 0.40  
                    opt_pthres0   = pthresh0               
                    use_native_ir = False
                else:
                  use_native_ir = False
                  if mod_inputs.lower() == 'ir':
                    use_chkpnt  = os.path.realpath(ir_aacp_best)
                    pthresh0    = 0.80
                    opt_pthres0 = pthresh0               
                  elif mod_inputs.lower() == 'ir+vis':  
                    use_chkpnt  = os.path.realpath(irvis_aacp_best)
                    pthresh0    = 0.40
                    opt_pthres0 = pthresh0               
                  elif mod_inputs.lower() == 'ir+glm':
                    use_chkpnt  = os.path.realpath(irglm_aacp_best)
                    pthresh0    = 0.60
                    opt_pthres0 = pthresh0               
                  elif mod_inputs.lower() == 'ir+vis+glm':
                    use_chkpnt  = os.path.realpath(irvisglm_aacp_best)
                    pthresh0    = 0.30
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
                  if mod_loc == 'OT':
                    if mod_inputs.lower() == 'ir':
#                       use_chkpnt    = os.path.realpath(ir_ot_best)
#                       pthresh0      = 0.40
                      use_chkpnt    = os.path.realpath(ir_ot_native_best)
                      pthresh0      = 0.20
                      opt_pthres0   = pthresh0               
                      use_native_ir = True
                    elif mod_inputs.lower() == 'ir+vis':  
                      use_chkpnt  = os.path.realpath(irvis_ot_best)
                      pthresh0    = 0.25
                      opt_pthres0 = pthresh0               
                    elif mod_inputs.lower() == 'ir+glm':
#                       use_chkpnt = os.path.realpath(irglm_ot_best)
#                       pthresh0   = 0.65
                      use_chkpnt    = os.path.realpath(irglm_ot_native_best)
                      pthresh0      = 0.30
                      opt_pthres0   = pthresh0               
                      use_native_ir = True
                    elif mod_inputs.lower() == 'ir+irdiff':
                      use_chkpnt  = os.path.realpath(irirdiff_ot_best)
                      pthresh0    = 0.35
                      opt_pthres0 = pthresh0               
                    elif mod_inputs.lower() == 'ir+vis+glm':
                      use_chkpnt  = os.path.realpath(irvisglm_ot_best)
                      pthresh0    = 0.40  
                      opt_pthres0 = pthresh0               
                  else:
                    if mod_inputs.lower() == 'ir':
                      use_chkpnt  = os.path.realpath(ir_aacp_best)
                      pthresh0    = 0.80
                      opt_pthres0 = pthresh0               
                    elif mod_inputs.lower() == 'ir+vis':  
                      use_chkpnt  = os.path.realpath(irvis_aacp_best)
                      pthresh0    = 0.40
                      opt_pthres0 = pthresh0               
                    elif mod_inputs.lower() == 'ir+glm':
                      use_chkpnt  = os.path.realpath(irglm_aacp_best)
                      pthresh0    = 0.60
                      opt_pthres0 = pthresh0               
                    elif mod_inputs.lower() == 'ir+vis+glm':
                      use_chkpnt  = os.path.realpath(irvisglm_aacp_best)
                      pthresh0    = 0.30
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
              if mod_loc == 'OT':
                if transition == 'y':
                  day   = {'mod_type': 'multiresunet', 'mod_inputs': 'IR+VIS+GLM', 'use_chkpnt': os.path.realpath(irvisglm_ot_best), 'pthresh': 0.40, 'use_night' : False, 'use_irdiff' : False, 'use_glm' : True}
                  night = {'mod_type': 'multiresunet', 'mod_inputs': 'IR', 'use_chkpnt': os.path.realpath(ir_ot_best), 'pthresh': 0.40, 'use_night' : True, 'use_irdiff' : False, 'use_glm' : False}
                  print(day['mod_inputs']   + ' ' + day['mod_type']   + ' OT model will be used during day time scans.')
                  print(night['mod_inputs'] + ' ' + night['mod_type'] + ' OT model will be used during night time scans.')
                  print()
              else:
                if transition == 'y':
                  day   = {'mod_type': 'multiresunet', 'mod_inputs': 'IR+VIS+GLM', 'use_chkpnt': os.path.realpath(irvisglm_aacp_best), 'pthresh': 0.30, 'use_night' : False, 'use_irdiff' : False, 'use_glm' : True}
                  night = {'mod_type': 'multiresunet', 'mod_inputs': 'IR+GLM', 'use_chkpnt': os.path.realpath(irglm_aacp_best), 'pthresh': 0.60, 'use_night' : True, 'use_irdiff' : False, 'use_glm' : True}
                  print(day['mod_inputs']   + ' ' + day['mod_type']   + ' AACP model will be used during day time scans.')
                  print(night['mod_inputs'] + ' ' + night['mod_type'] + ' AACP model will be used during night time scans.')
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
#                       use_chkpnt    = os.path.realpath(ir_ot_best)
#                       pthresh       = 0.40
                      use_chkpnt    = os.path.realpath(ir_ot_native_best)
                      pthresh0      = 0.20
                      opt_pthres0   = pthresh0               
                      use_native_ir = True
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
                    use_chkpnt  = os.path.realpath(ir_aacp_best)
                    pthresh0    = 0.80
                    opt_pthres0 = pthresh0               
                elif mod_inputs.lower() == 'ir+vis' or mod_inputs.lower() == 'vis+ir':  
                  mod_inputs    = 'IR+VIS'
                  mod_check     = 1
                  use_night     = False
                  use_native_ir = False          
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      use_chkpnt  = os.path.realpath(irvis_ot_best)
                      pthresh0    = 0.25
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
                    use_chkpnt  = os.path.realpath(irvis_aacp_best)
                    pthresh0    = 0.40
                    opt_pthres0 = pthresh0               
                elif mod_inputs.lower() == 'ir+glm' or mod_inputs.lower() == 'glm+ir':
                  mod_inputs = 'IR+GLM'
                  mod_check  = 1
                  use_glm    = True
                  use_night  = True
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
#                       use_chkpnt    = os.path.realpath(irglm_ot_best)
#                       pthresh       = 0.65
                      use_chkpnt    = os.path.realpath(irglm_ot_native_best)
                      pthresh0      = 0.30
                      opt_pthres0   = pthresh0               
                      use_native_ir = True
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
                    use_chkpnt    = os.path.realpath(irglm_aacp_best)
                    pthresh0      = 0.60
                    opt_pthres0   = pthresh0               
                    use_native_ir = False          
                elif mod_inputs.lower() == 'ir+irdiff' or mod_inputs.lower() == 'irdiff+ir':  
                  mod_inputs = 'IR+IRDIFF'
                  mod_check  = 1
                  use_night  = True
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      use_chkpnt    = os.path.realpath(irirdiff_ot_best)
                      pthresh0      = 0.35
                      opt_pthres0   = pthresh0               
                      use_native_ir = False          
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
                    print('IR+IRDIFF is not available for AACPs. Please choose different model inputs.')
                    print()
                    mod_check = 0
                elif 'ir' in mod_inputs.lower() and 'vis' in mod_inputs.lower() and 'glm' in mod_inputs.lower(): 
                  mod_inputs    = 'IR+VIS+GLM'
                  mod_check     = 1
                  use_glm       = True
                  use_night     = False
                  use_native_ir = False          
                  if mod_loc == 'OT':
                    if mod_type == 'multiresunet':
                      use_chkpnt  = os.path.realpath(irvisglm_ot_best)
                      pthresh0    = 0.40
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
                    use_chkpnt  = os.path.realpath(irvisglm_aacp_best)
                    pthresh0    = 0.30
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
#              print('This job will now output images.')
#               no_plot = False
              print()
              print('This software is currently not set up to plot images due to licensing restrictions. We apologize for the inconvenience.')
              no_plot = True
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

  no_plot = True
  if nhours == None:
    tstart = int(''.join(re.split('-', d_str1))[0:8])                                                                              #Extract start date of job for real-time post processing
    tend   = int(''.join(re.split('-', d_str2))[0:8])                                                                              #Extract end date of job for real-time post processing
    t0 = time.strftime("%Y-%m-%d %H:%M:%S")
    if d_str2 > t0:
      print('Waiting until end date to start archived model run.')
      print('End date chosen = ' + d_str2)
      time.sleep((datetime.strptime(d_str2, "%Y-%m-%d %H:%M:%S") - datetime.strptime(t0, "%Y-%m-%d %H:%M:%S")).total_seconds())
      
    if transition == 'y':
      chk_day_night = {'day': day, 'night': night}
      pthresh = None
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
                                                 verbose        = verbose)
    else:
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
                                                   verbose        = verbose)
      else: 
        print('Not set up for specified model. You have encountered a bug. Exiting program.')
        exit()
  else:   
    t_sec = sat_time_intervals(sat, sector = sector)                                                                               #Extract time interval between satellite sector scan files (sec)
    while datetime.utcnow().second < 5:
        sleep(1)                                                                                                                   #Wait until time has elapsed for new satellite scan file to be available
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
      if transition == 'y':
        chk_day_night = {'day': day, 'night': night}
        pthresh = None
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
                                                   verbose        = verbose)
      else:
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
                                                     verbose        = verbose)
        else: 
          print('Not set up for specified model. You have encountered a bug. Exiting program.')
          exit()
      lc = lc+1
      backend.clear_session()  
      tdiff = time.time() - t0
    
    tend = int(datetime.utcnow().strftime("%Y%m%d"))                                                                               #Extract end date of job for real-time post processing
    
  
  object_type = 'OT' if use_updraft == True else 'AACP'
  if transition == 'y':
    mod_type = day['mod_type']
  
  print('Starting post-processing of ' + object_type + ' ' + mod_type + ' job')
  if nhours != None:
    time.sleep(2)
  for u in range(tstart, tend+1): 
    run_write_severe_storm_post_processing(inroot = os.path.join(raw_data_root, 'combined_nc_dir', str(u)), 
                                           use_local     = use_local, write_gcs = run_gcs, del_local = del_local,
                                           c_bucket_name = 'ir-vis-sandwhich',
                                           object_type   = object_type,
                                           mod_type      = mod_type,
                                           sector        = sector,
                                           pthresh       = pthresh,
                                           percent_omit  = percent_omit,
                                           verbose       = verbose)
    
def main():
    args = sys.argv[1:]
    config_file = None
    if len(args) > 0:
      if len(args) == 2 and args[0] == '--config':
        config_file = args[1]
      else:
        if len(args) != 2:
          print('Number of arguments not correct!!')
          print(args)
          exit()
        if args[0] != '--config':
          print('Not set up to handle specified argument!!')
          print(args)
          exit()  
    run_all_plume_updraft_model_predict(config_file = config_file)
    
if __name__ == '__main__':
    main()