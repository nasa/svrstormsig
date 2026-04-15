#+
# Name:
#     run_tf_N_channel_plume_updraft_day_predict.py
# Purpose:
#     This is a script to run (by importing) programs to run previous models using checkpoint files for inputs that include
#     IR, VIS, GLM, and IR_DIFF channels. Consolidates run_tf_1/2/3_channel_plume_updraft_day_predict into a single
#     function that dispatches to the correct tf_N_channel inner function based on the number of model input channels.
#     Also supports simultaneous OT+AACP detection via BOTH mode by looping over both signatures within a
#     single call, with shared pre-processing (download, combined netCDF creation, tropopause write, numpy array
#     creation) done exactly once.
# Calling sequence:
#     import run_tf_N_channel_plume_updraft_day_predict
#     run_tf_N_channel_plume_updraft_day_predict.run_tf_N_channel_plume_updraft_day_predict()
# Input:
#     None.
# Functions:
#     run_tf_N_channel_plume_updraft_day_predict : MAIN outer function. Handles data download, combined netCDF
#                                                  creation, numpy pre-processing, and dispatches to
#                                                  tf_N_channel_plume_updraft_day_predict. Supports BOTH mode
#                                                  for simultaneous OT+AACP detection with shared pre-processing.
#     tf_N_channel_plume_updraft_day_predict     : Inner prediction function. Replaces tf_1/2/3_channel versions.
#                                                  Dispatches model construction and input tensor building based
#                                                  on number of channels derived from use_chkpnt path.
#     append_combined_ncdf_with_model_results    : Function appends the combined netCDF file that is created in
#                                                  pre-processing with the model results. Imported from
#                                                  run_tf_1_channel_plume_updraft_day_predict.
# Output:
#     Numpy files with the model results, figure containing OT or plume locations, csv file yielding OT or plume
#     locations, and time aggregated figures in cases of not real-time runs.
# Keywords:
#     date1          : Start date. List containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to start downloading 
#                      DEFAULT = None -> download nearest to IR/VIS file to current date and time and nearest 15 GLM files. (ex. '2017-04-29 00:00:00')
#     date2          : End date. String containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to end downloading 
#                      DEFAULT = None -> download only files nearest to date1. (ex. '2017-04-29 00:00:00')
#     use_updraft    : IF keyword set (True), pre-process and run OT model, otherwise run plume model.
#                      DEFAULT = True. 
#                      NOTE: Keyword is superseded if checkpoint file is given. The checkpoint file path/name is used to determine if model is updraft or plume.
#                      NOTE: In BOTH mode this argument is ignored; OT and AACP are both run.
#     sat            : STRING specifying GOES satellite to download data for. 
#                      DEFAULT = 'goes-16'.
#     sector         : STRING specifying GOES sector to download. (ex. 'meso', 'conus', 'full')
#                      DEFAULT = 'meso2'
#     region         : State or sector of US to plot data around. 
#                      DEFAULT = None -> plot for full domain of the satellite sector.
#     xy_bounds      : ARRAY of floating point values giving the domain boundaries the user wants to restrict the combined netCDF and output image to
#                      [x0, y0, x1, y1]
#                      IF region set, xy_bounds taken from rdr_sat_utils_jwc function extract_us_lat_lon_region.
#                      xy_bounds is ONLY USED IF sector is not meso1 or meso2
#                      DEFAULT = [] -> write combined netCDF data for full scan region.
#     pthresh        : FLOAT keyword to specify probability threshold to use. 
#                      DEFAULT = None -> Use maximized IoU probability threshold.
#     opt_pthresh    : FLOAT keyword to specify probability threshold that was found to optimize the specified model during Cooney testing. 
#                      DEFAULT = None.
#     run_gcs        : IF keyword set (True), write the output files to the google cloud in addition to local storage.
#                      DEFAULT = True.
#     use_local      : IF keyword set (True), use local files. If False, use files located on the google cloud. 
#                      DEFAULT = False -> use files on google cloud server. 
#     del_local      : IF keyword set (True) AND run_gcs = True, delete local copy of output file.
#                      DEFAULT = True.
#     inroot         : STRING specifying input root directory to the pre-processed test, train, val files
#                      DEFAULT = '../../../goes-data/aacp_results/'
#     outroot        : STRING output directory path for results data storage
#                      DEFAULT = '../../../goes-data/'
#     og_bucket_name : STRING specifying the name of the gcp bucket to write downloaded files to. run_gcs needs to be True in order for this to matter.
#                      DEFAULT = 'goes-data'
#     c_bucket_name  : STRING specifying the name of the gcp bucket to write combined IR, VIS, GLM files to As well as 3 modalities figures.
#                      run_gcs needs to be True in order for this to matter.
#                      DEFAULT = 'ir-vis-sandwhich'
#     p_bucket_name  : STRING specifying the name of the gcp bucket to write IR/VIS/GLM numpy files to. run_gcs needs to be True in order for this to matter.
#                      DEFAULT = 'aacp-proc-data'
#     f_bucket_name  : STRING specifying the name of the gcp bucket to write model prediction results files to. run_gcs needs to be True in order for this to matter.
#                      DEFAULT = 'aacp-results'
#     use_chkpnt     : Optional STRING keyword to specify file name and path to check point file with model weights from a previous model run you want to use.
#                      Setting this keyword allows the user to skip over all the model creation stuff and go straight to loading the weights from a trained model.
#                      NOTE: Not used in BOTH mode (chk_day_night provides all checkpoint info).
#     by_date        : IF keyword set (True), use model training, validation, and testing numpy output files for separate dates.
#                      DEFAULT = True. If False, use first 75% of data for training, next 20% for validation, and last 5% for testing.
#     by_updraft     : IF keyword set (True), create model training, validation, and testing numpy output files for 64x64 pixels around each updraft by dates.
#                      NOTE: If this keyword is set, by_date will automatically also be set.
#                      DEFAULT = True. If False, use by_date 512x512 or first 75% of data for training, next 20% for validation, and last 5% for testing.
#     subset         : IF keyword set (True), use numpy files in which subset indices [x0, y0, x1, y1] were given for the model training, testing, and validation.
#                      DEFAULT = True. False implies using the 512x512 boxes created for the full domain
#     chk_day_night  : Structure containing day and night model data information in order to make seemless transition in predictions of the 2.
#                      In BOTH mode this is a nested dict: {'OT': {'day': {...}, 'night': {...}}, 'AACP': {'day': {...}, 'night': {...}}}
#                      In standard transition mode this is: {'day': {...}, 'night': {...}}
#                      DEFAULT = {} -> do not use day_night transition
#     use_night      : IF keyword set (True), use night time data. CURRENTLY ONLY SET UP TO DO IR+GLM and IR and IR_DIFF!!!!
#                      DEFAULT = None -> True for IR and False for VIS runs
#     grid_data      : IF keyword set (True), plot data on lat/lon grid.
#                      DEFAULT = True. False-> plot on 1:1 pixel grid.
#     no_plot        : IF keyword set (True), do not plot the data.        
#                      DEFAULT = False -> plot_data files.
#     no_download    : IF keyword set (True), do not download the raw IR, VIS, and GLM data files. This would only be set if re-running a failed job and the files were already downloaded.
#                      Keyword is ignored if real_time = True
#                      DEFAULT = False -> download raw vis, glm, ir files.
#     rewrite_model  : IF keyword set (True), rewrite the model numpy files. 
#                      Keyword is ignored if real_time = True
#                      DEFAULT = False -> do not rewrite model files if already exists. 
#     use_native_ir  : IF keyword set (True), run model at the native IR data resolution.
#                      DEFAULT = False -> interpolate IR resolution data to higher resolution visible data grid.
#     essl           : BOOL keyword to specify whether or not this will be used for ESSL purposes. ESSL has data stored in /yyyy/mm/dd/ format rather
#                      software built format yyyymmdd. Because the software uses the yyyymdd to build other directories etc. off of the path
#                      and ESSL may use this software as a way to download the data, this was easiest fix.
#                      DEFAULT = False which implies data are stored using /yyyymmdd/. Set to True if data are stored in /yyyy/mm/dd/ format.
#     new_weighting  : BOOL keyword to specify whether or not to use the TROPDIFF new weighting scheme that uses a cosine function rather than linear weighting
#                      DEFAULT = True -> use new weighting scheme
#     verbose        : BOOL keyword to specify whether or not to print verbose informational messages.
#                      DEFAULT = True which implies to print verbose informational messages
# Author and history:
#     John W. Cooney           2021-05-05.
#                              2026-03-30. Consolidated run_tf_1/2/3_channel_plume_updraft_day_predict into single
#                                          run_tf_N_channel_plume_updraft_day_predict function.
#                                          Added BOTH mode for simultaneous OT+AACP detection with shared
#                                          pre-processing (download, combined netCDF, tropopause, numpy arrays
#                                          all written once using the union of required channels).
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
import re
import glob
import time
from netCDF4 import Dataset
import pandas as pd
from math import ceil
from scipy.ndimage import label, generate_binary_structure, find_objects
from threading import Thread
import multiprocessing as mp
import sys 
#sys.path.insert(1, '../')
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.getcwd()))
from glm_gridder.dataScaler import DataScaler
from new_model.run_tf_1_channel_plume_updraft_day_predict import append_combined_ncdf_with_model_results
from new_model.run_2_channel_just_updraft_day_np_preprocess2 import build_imgs, build_subset_tensor, reconstruct_tensor_from_subset, reconstruct_tensor_from_subset2, build_subset_tensor2, reconstruct_tensor_from_subset3, build_subset_tensor3
from new_model.gcs_processing import write_to_gcs, download_model_chk_file, list_gcs_chkpnt, list_gcs, list_csv_gcs, load_csv_gcs, load_npy_blobs, load_npy_gcs, download_ncdf_gcs
from glm_gridder.run_create_image_from_three_modalities2 import *
from glm_gridder.run_create_image_from_three_modalities import * 
from glm_gridder.run_mtg_create_image_from_three_modalities2 import *
from glm_gridder.run_mtg_create_image_from_three_modalities import * 
from glm_gridder.run_download_goes_ir_vis_l1b_glm_l2_data import *
from glm_gridder.run_download_goes_ir_vis_l1b_glm_l2_data_parallel import *
from glm_gridder.run_download_mtg_fci_data import *
from glm_gridder.run_download_mtg_fci_data_parallel import *
from gridrad.rdr_sat_utils_jwc import extract_us_lat_lon_region
from EDA.create_vis_ir_numpy_arrays_from_netcdf_files import *
from EDA.unet import unet
from EDA.MultiResUnet import MultiResUnet
from EDA.MultiResUnet_new_UPsampling import MultiResUnet_new_UPsampling
from EDA.create_vis_ir_numpy_arrays_from_netcdf_files2 import *
from EDA.mtg_create_vis_ir_numpy_arrays_from_netcdf_files2 import *
from visualize_results.visualize_time_aggregated_results import visualize_time_aggregated_results
from visualize_results.run_write_plot_model_result_predictions import write_plot_model_result_predictions
from new_model.run_write_severe_storm_post_processing import download_gfs_analysis_files_from_gcloud, download_gfs_analysis_files
from new_model.run_write_gfs_trop_temp_to_combined_ncdf import run_write_gfs_trop_temp_to_combined_ncdf
backend.clear_session()


def run_tf_N_channel_plume_updraft_day_predict(date1          = None, date2 = None, 
                                               use_updraft    = True, 
                                               sat            = 'goes-16',
                                               sector         = 'meso2',
                                               region         = None, 
                                               xy_bounds      = [], 
                                               pthresh        = None,
                                               opt_pthresh    = None,
                                               run_gcs        = True, 
                                               use_local      = False, 
                                               del_local      = True, 
                                               inroot         = os.path.join('..', '..', '..', 'goes-data', 'aacp_results'), 
                                               outroot        = os.path.join('..', '..', '..', 'goes-data'), 
                                               og_bucket_name = 'goes-data',
                                               c_bucket_name  = 'ir-vis-sandwhich',
                                               p_bucket_name  = 'aacp-proc-data',
                                               f_bucket_name  = 'aacp-results', 
                                               use_chkpnt     = None, 
                                               by_date        = True,  by_updraft = True, subset = True, 
                                               chk_day_night  = {}, 
                                               use_night      = None, 
                                               grid_data      = True, 
                                               no_plot        = False, 
                                               no_download    = False, 
                                               rewrite_model  = False,
                                               use_native_ir  = False, 
                                               essl           = False, 
                                               new_weighting  = True,
                                               verbose        = True):
  '''
  This is a script to run (by importing) programs that predict the N-channel plume or updraft model based on the most
  recent previous model run results checkpoint weights. Consolidates the run_tf_1/2/3_channel_plume_updraft_day_predict
  functions into a single entry point. Also supports simultaneous OT+AACP detection (BOTH mode) when chk_day_night is
  a nested dict keyed by 'OT' and 'AACP'. In BOTH mode, all pre-processing (download, combined netCDF creation,
  tropopause write, numpy array creation) is performed exactly once using the union of channels required by all four
  model checkpoints (OT day, OT night, AACP day, AACP night), and both model predictions run against the results
  already held in memory.

  Args:
      None.
  Keywords:
      date1          : Start date. List containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to start downloading 
                       DEFAULT = None -> download nearest to IR/VIS file to current date and time and nearest 15 GLM files. (ex. '2017-04-29 00:00:00')
      date2          : End date. String containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to end downloading 
                       DEFAULT = None -> download only files nearest to date1. (ex. '2017-04-29 00:00:00')
      use_updraft    : IF keyword set (True), pre-process and run OT model, otherwise run plume model.
                       DEFAULT = True. 
                       NOTE: Keyword is superseded if checkpoint file is given. The checkpoint file path/name is used to determine if model is updraft or plume.
      sat            : STRING specifying GOES satellite to download data for. 
                       DEFAULT = 'goes-16'.
      sector         : STRING specifying GOES sector to download. (ex. 'meso', 'conus', 'full')
                       DEFAULT = 'meso2'
      region         : State or sector of US to plot data around. 
                       DEFAULT = None -> plot for full domain of the satellite sector.
      xy_bounds      : ARRAY of floating point values giving the domain boundaries the user wants to restrict the combined netCDF and output image to
                       [x0, y0, x1, y1]
                       IF region set, xy_bounds taken from rdr_sat_utils_jwc function extract_us_lat_lon_region.
                       xy_bounds is ONLY USED IF sector is not meso1 or meso2
                       DEFAULT = [] -> write combined netCDF data for full scan region.
      pthresh        : FLOAT keyword to specify probability threshold to use. 
                       DEFAULT = None -> Use maximized IoU probability threshold.
      opt_pthresh    : FLOAT keyword to specify probability threshold that was found to optimize the specified model during Cooney testing. 
                       DEFAULT = None.
      run_gcs        : IF keyword set (True), write the output files to the google cloud in addition to local storage.
                       DEFAULT = True.
      use_local      : IF keyword set (True), use local files. If False, use files located on the google cloud. 
                       DEFAULT = False -> use files on google cloud server. 
      del_local      : IF keyword set (True) AND run_gcs = True, delete local copy of output file.
                       DEFAULT = True.
      inroot         : STRING specifying input root directory to the pre-processed test, train, val files
                       DEFAULT = '../../../goes-data/aacp_results/'
      outroot        : STRING output directory path for results data storage
                       DEFAULT = '../../../goes-data/'
      og_bucket_name : STRING specifying the name of the gcp bucket to write downloaded files to. run_gcs needs to be True in order for this to matter.
                       DEFAULT = 'goes-data'
      c_bucket_name  : STRING specifying the name of the gcp bucket to write combined IR, VIS, GLM files to As well as 3 modalities figures.
                       run_gcs needs to be True in order for this to matter.
                       DEFAULT = 'ir-vis-sandwhich'
      p_bucket_name  : STRING specifying the name of the gcp bucket to write IR/VIS/GLM numpy files to. run_gcs needs to be True in order for this to matter.
                       DEFAULT = 'aacp-proc-data'
      f_bucket_name  : STRING specifying the name of the gcp bucket to write model prediction results files to. run_gcs needs to be True in order for this to matter.
                       DEFAULT = 'aacp-results'
      use_chkpnt     : Optional STRING keyword to specify file name and path to check point file with model weights from a previous model run you want to use.
                       Setting this keyword allows the user to skip over all the model creation stuff and go straight to loading the weights from a trained model.
      by_date        : IF keyword set (True), use model training, validation, and testing numpy output files for separate dates.
                       DEFAULT = True. If False, use first 75% of data for training, next 20% for validation, and last 5% for testing.
      by_updraft     : IF keyword set (True), create model training, validation, and testing numpy output files for 64x64 pixels around each updraft by dates.
                       NOTE: If this keyword is set, by_date will automatically also be set.
                       DEFAULT = True. If False, use by_date 512x512 or first 75% of data for training, next 20% for validation, and last 5% for testing.
      subset         : IF keyword set (True), use numpy files in which subset indices [x0, y0, x1, y1] were given for the model training, testing, and validation.
                       DEFAULT = True. False implies using the 512x512 boxes created for the full domain
      chk_day_night  : Structure containing day and night model data information in order to make seemless transition in predictions of the 2.
                       DEFAULT = {} -> do not use day_night transition                 
      use_night      : IF keyword set (True), use night time data. CURRENTLY ONLY SET UP TO DO IR+GLM!!!!
                       DEFAULT = None -> True for IR+GLM and False for VIS+GLM and IR+VIS runs
      grid_data      : IF keyword set (True), plot data on lat/lon grid.
                       DEFAULT = True. False-> plot on 1:1 pixel grid.
      no_plot        : IF keyword set (True), do not plot the data.        
                       DEFAULT = False -> plot_data files.
      no_download    : IF keyword set (True), do not download the raw IR, VIS, and GLM data files. This would only be set if re-running a failed job and the files were already downloaded.
                       Keyword is ignored if real_time = True
                       DEFAULT = False -> download raw vis, glm, ir files.
      rewrite_model  : IF keyword set (True), rewrite the model numpy files. 
                       Keyword is ignored if real_time = True
                       DEFAULT = False -> do not rewrite model files if already exists.     
      essl           : BOOL keyword to specify whether or not this will be used for ESSL purposes. ESSL has data stored in /yyyy/mm/dd/ format rather
                       software built format yyyymmdd. Because the software uses the yyyymdd to build other directories etc. off of the path
                       and ESSL may use this software as a way to download the data, this was easiest fix.
                       DEFAULT = False which implies data are stored using /yyyymmdd/. Set to True if data are stored in /yyyy/mm/dd/ format.
      new_weighting  : BOOL keyword to specify whether or not to use the TROPDIFF new weighting scheme that uses a cosine function rather than linear weighting
                       DEFAULT = True -> use new weighting scheme
      verbose        : BOOL keyword to specify whether or not to print verbose informational messages.
                       DEFAULT = True which implies to print verbose informational messages
  Returns:
      Numpy files with the model results, figure containing OT or plume locations, csv file yielding OT or plume
      locations, and time aggregated figures in cases of not real-time runs.
  '''  

  # Detect BOTH mode: chk_day_night is nested {'OT': {...}, 'AACP': {...}}
  both_mode = (isinstance(chk_day_night, dict) and
               'OT'   in chk_day_night and
               'AACP' in chk_day_night and
               isinstance(chk_day_night['OT'],   dict) and
               isinstance(chk_day_night['AACP'], dict) and
               'day'  in chk_day_night['OT'] and
               'night' in chk_day_night['OT'])

  if both_mode:
    # ------------------------------------------------------------------
    # BOTH mode: OT + AACP simultaneous detection.
    #
    # DESIGN:
    #   Pre-processing phase:
    #     - Compute union of all channel flags across OT day, OT night,
    #       AACP day, AACP night checkpoints.
    #     - Download ALL channels needed by EITHER signature using the
    #       union flags.
    #     - Write combined netCDF with ALL variables once.
    #     - Write tropopause to combined netCDF once.
    #     - Write labelled numpy files once.
    #     - Store dims, ir_file0, nc_names, img_names, proj, p
    #       No numpy files are loaded to recover this information.
    #   Prediction phase:
    #     - Loop over ['OT', 'AACP'] signatures.
    #     - For each signature call tf_N_channel_plume_updraft_day_predict
    #       using dims/nc_names/etc already
    #     - del_local = False during OT pass so labelled numpy files and
    #       combined netCDFs survive for AACP pass.
    #     - del_local = original value during AACP pass (safe to clean up).
    # ------------------------------------------------------------------
    t0 = time.time()
    if verbose:
      print('BOTH mode detected: running OT + AACP with shared pre-processing.')
      print()

    # ------------------------------------------------------------------
    # Step 1: Compute union of channel flags across all four checkpoints.
    # ------------------------------------------------------------------
    def _flags_from_chk(sig_dict):
      '''Return channel requirement flags for a single day or night model dict.'''
      chk = sig_dict['use_chkpnt']
      _has_tropdiffot = len(re.split('tropdiffot', chk)) != 1
      _has_tropdiff   = (len(re.split('tropdiff',  chk)) != 1 and not _has_tropdiffot)

      return {
        'use_glm'        : len(re.split('glm',          chk)) != 1,
        'use_irdiff'     : len(re.split('_wvirdiff',    chk)) != 1,
        'use_dirtyirdiff': len(re.split('_dirtyirdiff', chk)) != 1,
        'use_irgradient' : len(re.split('_irgradient',  chk)) != 1,
        'use_tropdiff'   : _has_tropdiff,
        'use_tropdiffot' : _has_tropdiffot,
        'use_snowice'    : len(re.split('_snowice',     chk)) != 1,
        'use_cirrus'     : len(re.split('_cirrus',      chk)) != 1,
        'use_vis'        : True,
#         'use_vis'        : (len(re.split('_vis',        chk)) != 1 or
#                             len(re.split('vis_',        chk)) != 1 or
#                             len(re.split('_snowice',    chk)) != 1 or
#                             len(re.split('_cirrus',     chk)) != 1),
      }

    all_flags = [_flags_from_chk(chk_day_night['OT']['day']),
                 _flags_from_chk(chk_day_night['OT']['night']),
                 _flags_from_chk(chk_day_night['AACP']['day']),
                 _flags_from_chk(chk_day_night['AACP']['night'])]

    union = {k: any(f[k] for f in all_flags) for k in all_flags[0]}                                                                                                        #Union: True if ANY of the four models needs this channel

    no_irdiff      = not union['use_irdiff']
    no_dirtyirdiff = not union['use_dirtyirdiff']
    no_tropdiff    = not union['use_tropdiff']
    no_snowice     = not union['use_snowice']
    no_cirrus      = not union['use_cirrus']
    no_irgradient  = not union['use_irgradient']
    use_glm0       = union['use_glm']
    no_vis         = not union['use_vis'] if not use_native_ir else True
    use_tropdiffot = union['use_tropdiffot']
    if verbose:
      print('Union of required channels across OT+AACP models:')
      print('  Use GLM         : ' + str(union['use_glm']))
      print('  Use WV-IRDIFF   : ' + str(union['use_irdiff']))
      print('  Use DIRTYIRDIFF : ' + str(union['use_dirtyirdiff']))
      print('  Use TROPDIFF    : ' + str(union['use_tropdiff']))
      print('  Use TROPDIFFOT  : ' + str(union['use_tropdiffot']))
      print('  Use IRGRADIENT  : ' + str(union['use_irgradient']))
      print('  Use SNOWICE     : ' + str(union['use_snowice']))
      print('  Use CIRRUS      : ' + str(union['use_cirrus']))
      print('  Use VIS         : ' + str(union['use_vis']))
      print('  No  VIS         : ' + str(no_vis))
      print()

    # ------------------------------------------------------------------
    # Step 2: Standard setup (date parsing, sector, real-time flag,
    # directory roots). Mirrors the top of the standard-mode body.
    # ------------------------------------------------------------------
    if date2 == None and date1 != None: date2 = date1
    if date1 == None:
      date_range = []
    else:
      d2 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
      if date2 != date1 and d2.hour == 0 and d2.minute == 0 and d2.second == 0:
        date2 = (d2 - timedelta(seconds = 1)).strftime("%Y-%m-%d %H:%M:%S")
      date_range  = [date1, date2]
      date_range2 = date_range.copy()

    date = datetime.utcnow()
    if date1 == None:
      rt  = True
      pat = 'real_time'
      os.makedirs(os.path.join(outroot, 'tmp_dir'), exist_ok = True)
    else:
      rt  = False
      pat = 'not_real_time'

    if 'mtg' in sat.lower():
      sector00 = sector.upper()
      if sector00.lower() == 'full':
        sector00 = 'F'
      if rt:
        if date.minute < 10:
          date = date - timedelta(minutes = 10)
    else:
      if 'meso' in sector.lower() or sector[0].lower() == 'm':
        m_sector = int(sector[-1])
        sector0  = None
        sector00 = 'M' + str(m_sector)
      else:
        m_sector = 2
        sector0  = sector[0].upper()
        sector00 = sector0
        if rt:
          if date.minute < 10 and (sector.lower() == 'full' or sector.lower() == 'f'):
            date = date - timedelta(minutes = 10)
          elif date.minute < 4 and sector.lower() == 'conus':
            date = date - timedelta(minutes = 5)
    d_str = date.strftime("%Y-%m-%d")

    inroot  = os.path.realpath(inroot)
    outroot = os.path.realpath(outroot)
    if ((region != None) and (len(xy_bounds) <= 0) and (sector00[0].lower() != 'm')):
      xy_bounds = extract_us_lat_lon_region(region = region)[region]

    # ------------------------------------------------------------------
    # Step 3: Download using union channel flags.
    # ------------------------------------------------------------------
    if verbose and not no_download: print('Downloading model input data (union of OT+AACP channels)')
    t00 = time.time()
    if run_gcs:
      og_bucket_name2 = og_bucket_name
    else:
      og_bucket_name2 = None

    if rt:
      if not no_download:
        if 'mtg' in sat.lower():
          og_dir, d_strs = run_download_mtg_fci_data(date1     = date1, date2 = date2,
                                                     outroot   = outroot,
                                                     sat       = sat,
                                                     sector    = sector,
                                                     no_vis    = no_vis,
                                                     no_native = no_irdiff and no_dirtyirdiff and no_cirrus and no_snowice,
                                                     essl      = essl,
                                                     verbose   = verbose)
        else:
          og_dir = run_download_goes_ir_vis_l1b_glm_l2_data(date1      = date1, date2 = date2,
                                                            outroot    = outroot,
                                                            sat        = sat,
                                                            sector     = sector,
                                                            no_glm     = not use_glm0,
                                                            no_vis     = no_vis,
                                                            no_irdiff  = no_irdiff,
                                                            no_dirtyir = no_dirtyirdiff,
                                                            no_cirrus  = no_cirrus,
                                                            no_snowice = no_snowice,
                                                            gcs_bucket = og_bucket_name2,
                                                            del_local  = False,
                                                            verbose    = verbose)
          d_strs = [os.path.basename(og_dir[0])]
      else:
        date1  = date.strftime("%Y-%m-%d %H:%M:%S")
        date01 = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
        if essl and 'mtg' in sat.lower():
          og_dir = [os.path.join(outroot, date01.strftime("%Y"), date01.strftime("%m"), date01.strftime("%d"))]
        else:
          og_dir = [os.path.join(outroot, date01.strftime("%Y%m%d"))]
        d_strs = [date01.strftime("%Y%m%d")]
      if not no_tropdiff:
        download_gfs_analysis_files_from_gcloud(date.strftime("%Y-%m-%d %H:%M:%S"),
                                                outroot         = os.path.join(os.path.dirname(outroot), 'gfs-data'),
                                                gfs_bucket_name = 'global-forecast-system',
                                                write_gcs       = False,
                                                del_local       = True,
                                                real_time       = True,
                                                verbose         = verbose)
    else:
      if not no_tropdiff:
        download_gfs_analysis_files(date1, date2,
                                    GFS_ANALYSIS_DT = 21600,
                                    outroot         = os.path.join(os.path.dirname(outroot), 'gfs-data'),
                                    c_bucket_name   = 'ir-vis-sandwhich',
                                    write_gcs       = False,
                                    del_local       = True,
                                    real_time       = False,
                                    verbose         = verbose)
      if not no_download:
        if 'mtg' in sat.lower():
          og_dir, d_strs = run_download_mtg_fci_data_parallel(date1     = date1, date2 = date2,
                                                              outroot   = outroot,
                                                              sat       = sat,
                                                              sector    = sector,
                                                              no_native = no_irdiff and no_dirtyirdiff and no_cirrus and no_snowice,
                                                              no_vis    = no_vis,
                                                              essl      = essl,
                                                              verbose   = verbose)
          if len(og_dir) <= 0:
            print('No MTG data found to download???')
            print(date1)
            print(date2)
            exit()
        else:
          og_dir = run_download_goes_ir_vis_l1b_glm_l2_data_parallel(date1      = date1, date2 = date2,
                                                                     outroot    = outroot,
                                                                     sat        = sat,
                                                                     sector     = sector,
                                                                     no_glm     = not use_glm0,
                                                                     no_vis     = no_vis,
                                                                     no_irdiff  = no_irdiff,
                                                                     no_dirtyir = no_dirtyirdiff,
                                                                     no_cirrus  = no_cirrus,
                                                                     no_snowice = no_snowice,
                                                                     gcs_bucket = og_bucket_name2,
                                                                     del_local  = False,
                                                                     verbose    = verbose)
        date02 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
        if essl and 'mtg' in sat.lower():
          if d_strs[-1] > date02.strftime("%Y%m%d"):
            og_dir = og_dir[:-1]
            d_strs = d_strs[:-1]
        else:
          if os.path.basename(og_dir[-1]) > date02.strftime("%Y%m%d"):
            og_dir = og_dir[:-1]
          d_strs = [os.path.basename(o) for o in og_dir]
      else:
        date01 = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
        date02 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
        if essl and 'mtg' in sat.lower():
          og_dir = sorted(list(set([
              os.path.join(outroot,
                           (date01 + timedelta(hours=x)).strftime("%Y"),
                           (date01 + timedelta(hours=x)).strftime("%m"),
                           (date01 + timedelta(hours=x)).strftime("%d"))
              for x in range(int((date02 - date01).days * 24 + ceil((date02 - date01).seconds / 3600.0)) + 1)
          ])))
          d_strs = sorted(list(set([
              (date01 + timedelta(hours=x)).strftime("%Y%m%d")
              for x in range(int((date02 - date01).days * 24 + ceil((date02 - date01).seconds / 3600.0)) + 1)
          ])))
        else:
          og_dir = sorted(list(set([os.path.join(outroot, (date01 + timedelta(hours = x)).strftime("%Y%m%d")) for x in range(int((date02-date01).days*24 + ceil((date02-date01).seconds/3600.0))+1)])))
          d_strs = sorted(list(set([(date01 + timedelta(hours = x)).strftime("%Y%m%d") for x in range(int((date02-date01).days*24 + ceil((date02-date01).seconds/3600.0))+1)])))
      if d_strs[-1] > date02.strftime("%Y%m%d"):
        og_dir = og_dir[:-1]
        d_strs = d_strs[:-1]

    if len(og_dir) == 0: print('No download output directory roots???')
    if verbose and not no_download:
      print('Time to download files ' + str(time.time() - t00) + ' sec')
      print('Output directory of downloaded netCDF files = ' + outroot)
      print()
    if verbose: print(d_strs)

    img_path = [datetime.strptime(o, "%Y%m%d").strftime("%Y%j") for o in d_strs]
    if len(img_path) > 1 and img_path[0] != img_path[-1]:
      img_path = datetime.strptime(d_strs[0], "%Y%m%d").strftime("%Y%j") + '-' + datetime.strptime(d_strs[-1], "%Y%m%d").strftime("%Y%j")
    else:
      img_path = datetime.strptime(d_strs[0], "%Y%m%d").strftime("%Y%j")

    # ------------------------------------------------------------------
    # Step 4: Per-date combined netCDF creation, tropopause write, and
    # numpy pre-processing. Done ONCE for both OT and AACP.
    # ------------------------------------------------------------------
    sig_accum = {
      'OT'  : {'d_range': [], 'outdate': [], 'nc_files': [], 'tods': [],
                'ir_files': [], 'res_dirs': [], 'img_names': [],
                'proj': None, 'outdir': None, 'pthresh0': None},
      'AACP': {'d_range': [], 'outdate': [], 'nc_files': [], 'tods': [],
                'ir_files': [], 'res_dirs': [], 'img_names': [],
                'proj': None, 'outdir': None, 'pthresh0': None},
    }


    for idx0, o in enumerate(og_dir):
      d_str2     = datetime.strptime(d_strs[idx0], "%Y%m%d").strftime("%Y%j")
      mod1_both  = 'day_night_optimal'
      mod0_both  = 'Day Night Optimal'
      ot_mod_pat = 'updraft_day_model'
      res_dir    = os.path.join(outroot, 'aacp_results', mod1_both, ot_mod_pat, pat, d_str)

      # --------------------------------------------------------------
      # Pre-processing for this date: combined netCDF, tropopause, numpy arrays. Done once for both OT and AACP.
      # --------------------------------------------------------------
      if verbose: print('Combining model input data into a netCDF file for date ' + d_strs[idx0])
      t00 = time.time()
      if 'mtg' in sat.lower():
        if rt:
          img_names, nc_names, proj, p = run_mtg_create_image_from_three_modalities(
                                            inroot               = o,
                                            glm_out_dir          = os.path.join(outroot, 'out_dir'),
                                            layered_dir          = os.path.join(outroot, 'combined_nc_dir'),
                                            img_out_dir          = os.path.join(outroot, 'aacp_results_imgs', mod1_both, ot_mod_pat, pat, d_str, d_str2, str(sector)),
                                            no_plot              = no_plot, grid_data = grid_data,
                                            no_write_glm         = not use_glm0,
                                            no_write_vis         = no_vis,
                                            no_write_irdiff      = no_irdiff,
                                            no_write_cirrus      = no_cirrus,
                                            no_write_snowice     = no_snowice,
                                            no_write_dirtyirdiff = no_dirtyirdiff,
                                            domain_sector        = sector,
                                            run_gcs              = run_gcs,
                                            write_combined_gcs   = False,
                                            append_nc            = True,
                                            region               = region, xy_bounds = xy_bounds,
                                            real_time            = rt, del_local = False,
                                            plt_model            = os.path.join(res_dir, d_str2, str(sector)),
                                            model                = mod0_both + ' OT Detection',
                                            satellite            = sat,
                                            chk_day_night        = chk_day_night['OT'],
                                            pthresh              = pthresh,
                                            in_bucket_name       = og_bucket_name, out_bucket_name = c_bucket_name,
                                            verbose              = verbose)
        else:
          img_names, nc_names, proj = run_mtg_create_image_from_three_modalities2(
                                            inroot               = o,
                                            glm_out_dir          = os.path.join(outroot, 'out_dir'),
                                            layered_dir          = os.path.join(outroot, 'combined_nc_dir'),
                                            date_range           = date_range,
                                            no_plot              = True, grid_data = grid_data,
                                            no_write_glm         = not use_glm0,
                                            no_write_vis         = no_vis,
                                            no_write_irdiff      = no_irdiff,
                                            no_write_cirrus      = no_cirrus,
                                            no_write_snowice     = no_snowice,
                                            no_write_dirtyirdiff = no_dirtyirdiff,
                                            domain_sector        = sector,
                                            run_gcs              = run_gcs,
                                            write_combined_gcs   = False,
                                            append_nc            = True,
                                            region               = region, xy_bounds = xy_bounds,
                                            real_time            = rt, del_local = False,
                                            satellite            = sat,
                                            in_bucket_name       = og_bucket_name, out_bucket_name = c_bucket_name,
                                            verbose              = verbose)
          p = None
      else:
        if rt:
          img_names, nc_names, proj, p = run_create_image_from_three_modalities(
                                            inroot               = o,
                                            glm_out_dir          = os.path.join(outroot, 'out_dir'),
                                            layered_dir          = os.path.join(outroot, 'combined_nc_dir'),
                                            img_out_dir          = os.path.join(outroot, 'aacp_results_imgs', mod1_both, ot_mod_pat, pat, d_str, d_str2, str(sector00)),
                                            no_plot              = no_plot, grid_data = grid_data,
                                            no_write_glm         = not use_glm0,
                                            no_write_vis         = no_vis,
                                            no_write_irdiff      = no_irdiff,
                                            no_write_cirrus      = no_cirrus,
                                            no_write_snowice     = no_snowice,
                                            no_write_dirtyirdiff = no_dirtyirdiff,
                                            meso_sector          = m_sector,
                                            domain_sector        = sector0,
                                            run_gcs              = run_gcs,
                                            write_combined_gcs   = False,
                                            append_nc            = True,
                                            region               = region, xy_bounds = xy_bounds,
                                            real_time            = rt, del_local = False,
                                            plt_model            = os.path.join(res_dir, d_str2, str(sector00)),
                                            model                = mod0_both + ' OT Detection',
                                            chk_day_night        = chk_day_night['OT'],
                                            pthresh              = pthresh,
                                            in_bucket_name       = og_bucket_name, out_bucket_name = c_bucket_name,
                                            verbose              = verbose)
        else:
          img_names, nc_names, proj = run_create_image_from_three_modalities2(
                                            inroot               = o,
                                            glm_out_dir          = os.path.join(outroot, 'out_dir'),
                                            layered_dir          = os.path.join(outroot, 'combined_nc_dir'),
                                            date_range           = date_range,
                                            no_plot              = True, grid_data = grid_data,
                                            no_write_glm         = not use_glm0,
                                            no_write_vis         = no_vis,
                                            no_write_irdiff      = no_irdiff,
                                            no_write_cirrus      = no_cirrus,
                                            no_write_snowice     = no_snowice,
                                            no_write_dirtyirdiff = no_dirtyirdiff,
                                            meso_sector          = m_sector,
                                            domain_sector        = sector0,
                                            run_gcs              = run_gcs,
                                            write_combined_gcs   = False,
                                            append_nc            = True,
                                            region               = region, xy_bounds = xy_bounds,
                                            real_time            = rt, del_local = False,
                                            in_bucket_name       = og_bucket_name, out_bucket_name = c_bucket_name,
                                            verbose              = verbose)
          p = None

      if not no_tropdiff:
        t000 = time.time()
        run_write_gfs_trop_temp_to_combined_ncdf(inroot        = os.path.join(outroot, 'combined_nc_dir', d_strs[idx0]),
                                                 gfs_root      = os.path.join(os.path.dirname(outroot), 'gfs-data'),
                                                 date1         = date1,
                                                 date2         = date2,
                                                 use_local     = True, write_gcs = run_gcs, del_local = False,
                                                 c_bucket_name = c_bucket_name,
                                                 real_time     = rt,
                                                 sector        = sector00,
                                                 verbose       = verbose)
        print('Time to write all of the tropopause data to combined netCDF files ' + str(time.time() - t000) + ' sec')
      if len(img_names) == 0:
        if verbose: print('No image names returned for date ' + d_strs[idx0] + ', skipping predictions.')
        continue                                                                                                                                                             #Nothing to predict for this date; move on

      if verbose:
        print('Data combined into a netCDF file for date ' + d_strs[idx0])
        print('Time to create combined netCDF file ' + str(time.time() - t00) + ' sec')
        print('Output directory of combined netCDF files = ' + os.path.join(outroot, 'combined_nc_dir'))
        print()
        print('Creating numpy files for date ' + d_strs[idx0])
      t00 = time.time()
      if 'mtg' in sat.lower():
        ir_file0, dims = mtg_create_vis_ir_numpy_arrays_from_netcdf_files2(
                                            inroot               = o,
                                            layered_root         = os.path.join(outroot, 'combined_nc_dir'),
                                            outroot              = os.path.join(outroot, 'labelled', pat),
                                            json_root            = os.path.join(outroot, 'labelled'),
                                            date_range           = date_range,
                                            domain_sector        = sector,
                                            use_native_ir        = use_native_ir,
                                            no_write_glm         = not union['use_glm'],
                                            no_write_vis         = no_vis,
                                            no_write_irdiff      = no_irdiff,
                                            no_write_cirrus      = no_cirrus,
                                            no_write_snowice     = no_snowice,
                                            no_write_dirtyirdiff = no_dirtyirdiff,
                                            no_write_trop        = no_tropdiff,
                                            no_write_ir_gradient = no_irgradient,
                                            run_gcs              = run_gcs, real_time = rt, del_local = False,
                                            use_local            = True,
                                            og_bucket_name       = og_bucket_name,
                                            comb_bucket_name     = c_bucket_name,
                                            proc_bucket_name     = p_bucket_name,
                                            new_weighting        = new_weighting,
                                            verbose              = verbose)
      else:
        ir_file0, dims = create_vis_ir_numpy_arrays_from_netcdf_files2(
                                            inroot               = o,
                                            layered_root         = os.path.join(outroot, 'combined_nc_dir'),
                                            outroot              = os.path.join(outroot, 'labelled', pat),
                                            json_root            = os.path.join(outroot, 'labelled'),
                                            date_range           = date_range,
                                            meso_sector          = m_sector,
                                            domain_sector        = sector0,
                                            no_write_glm         = not union['use_glm'],
                                            no_write_vis         = no_vis,
                                            no_write_irdiff      = no_irdiff,
                                            no_write_cirrus      = no_cirrus,
                                            no_write_snowice     = no_snowice,
                                            no_write_dirtyirdiff = no_dirtyirdiff,
                                            no_write_trop        = no_tropdiff,
                                            no_write_ir_gradient = no_irgradient,
                                            run_gcs              = run_gcs, real_time = rt, del_local = False,
                                            use_local            = True,
                                            og_bucket_name       = og_bucket_name,
                                            comb_bucket_name     = c_bucket_name,
                                            proc_bucket_name     = p_bucket_name,
                                            new_weighting        = new_weighting,
                                            verbose              = verbose)

      if verbose:
        print('Data numpy files created for date ' + d_strs[idx0])
        print('Time to create numpy files = ' + str(time.time() - t00) + ' sec')
        print('Output directory of numpy files = ' + os.path.join(outroot, 'labelled', pat))
        print()

      # --------------------------------------------------------------
      # Predictions for this date: OT then AACP.
      # OT uses del_local=False so files survive for AACP.
      # AACP uses the original del_local value.
      # After both complete, labelled numpy files and raw nc files
      # for THIS date are deleted before the next date begins.
      # --------------------------------------------------------------
      for sig_label, sig_use_updraft, sig_chk_dn in [('OT',   True,  chk_day_night['OT']), ('AACP', False, chk_day_night['AACP'])]:
        this_del_local = False if sig_label == 'OT' else del_local
        acc = sig_accum[sig_label]

        acc['nc_files'].extend(sorted(nc_names))
        acc['ir_files'].append(ir_file0)
        acc['outdate'].append(d_str2)
        acc['img_names'] = img_names                                                                                                                                        #Overwrite with most recent date; used only for len() > 0 guard
        acc['proj']      = proj

        if verbose: print('=== BOTH mode: running ' + sig_label + ' predictions for date ' + d_strs[idx0] + ' ===')
        if use_tropdiffot and sig_label == 'OT':
            no_write_npy = False
        else:
            no_write_npy = no_plot
        
        t00 = time.time()
        outdir, pthresh0, df = tf_N_channel_plume_updraft_day_predict(
                                  dims          = dims,
                                  use_glm       = sig_chk_dn['day']['use_glm'] or sig_chk_dn['night']['use_glm'],
                                  use_updraft   = sig_use_updraft,
                                  d_str         = d_str,
                                  date_range    = date_range,
                                  sector        = sector00,
                                  pthresh       = pthresh,
                                  opt_pthresh   = opt_pthresh,
                                  rt            = rt,
                                  run_gcs       = run_gcs,
                                  use_local     = use_local,
                                  del_local     = this_del_local,
                                  inroot        = o,
                                  outroot       = outroot,
                                  p_bucket_name = p_bucket_name,
                                  c_bucket_name = c_bucket_name,
                                  f_bucket_name = f_bucket_name,
                                  use_chkpnt    = sig_chk_dn['day']['use_chkpnt'],
                                  by_date       = by_date, by_updraft = by_updraft, subset = subset,
                                  use_night     = True,
                                  rewrite_model = rewrite_model,
                                  chk_day_night = sig_chk_dn,
                                  use_native_ir = use_native_ir,
                                  no_write_npy  = no_write_npy,
                                  verbose       = verbose)

        acc['res_dirs'].append(outdir)
        acc['outdir']   = outdir
        acc['pthresh0'] = pthresh0
        if verbose:
          print('Time to run ' + sig_label + ' prediction for date ' + d_strs[idx0] + ': ' + str(time.time() - t00) + ' sec')
          print()

        if not rt:
          if date_range2 != None and len(df) > 0:
            if date_range2[0] > str(np.min(df['date_time'])):
              date_range2[0] = str(np.min(df['date_time']))
            date_range2[1] = str(np.max(df['date_time']))
          acc['d_range'].append(date_range[0] + ';' + date_range[1])

        nc_names_out = sorted(list(df['glm_files']))
        if rt:
          if df.empty:
            tod          = []
            nc_names_out = []
          else:
            tod          = [df['day_night'][len(df)-1]]
            nc_names_out = [nc_names_out[-1]]
        else:
          tod = list(df['day_night'])
        acc['tods'].extend(tod)

        if len(nc_names_out) != len(tod):
          print('Time of day list length does not match nc_names list length??')
          print(len(nc_names_out))
          print(tod)
          exit()

        mod1_sig    = 'day_night_optimal'
        mod0_sig    = 'Day Night Optimal'
        mod00_sig   = ' OT Detection' if sig_use_updraft else ' AACP Detection'
        mod_pat_sig = 'updraft_day_model' if sig_use_updraft else 'plume_day_model'

        if len(nc_names_out) > 0:
          if rt:
            im_names, df_out = write_plot_model_result_predictions(nc_names_out, [], tod = tod, res = [], use_local = use_local, no_plot = True, res_dir = outdir, region = region, latlon_domain = xy_bounds, pthresh = pthresh0, model = mod0_sig + mod00_sig, chk_day_night = sig_chk_dn, grid_data = grid_data, proj = proj, outroot = os.path.join(outroot, 'aacp_results_imgs', mod1_sig, mod_pat_sig, pat, d_str, img_path, str(sector00).upper()), verbose = verbose)
          else:
            if not no_plot:
              im_names, df_out = write_plot_model_result_predictions(nc_names_out, [], tod = tod, res = [], use_local = use_local, no_plot = no_plot, res_dir = outdir, region = region, latlon_domain = xy_bounds, pthresh = pthresh0, model = mod0_sig + mod00_sig, chk_day_night = sig_chk_dn, grid_data = grid_data, proj = proj if len(img_names) > 0 else None, outroot = os.path.join(outroot, 'aacp_results_imgs', mod1_sig, mod_pat_sig, pat, d_str, img_path, str(sector00).upper()), verbose = verbose)
            else:
              im_names = []
          if run_gcs and not no_plot:
            pref = os.path.relpath(os.path.dirname(im_names[0]), re.split('aacp_results_imgs', os.path.dirname(im_names[0]))[0])
            for i in im_names:
              write_to_gcs(f_bucket_name, pref, i, del_local = this_del_local)
        else:
          if rt and not no_plot:
            print('No image created because model you wanted to run cannot be used for night time operations')
            if p is not None: p.kill()

      if rt and not no_plot and p is not None:
        p.join()
        p.close()
        while not os.path.exists(im_names[0]):
          time.sleep(1)
      # --------------------------------------------------------------
      # Both OT and AACP are done for this date.
      # Delete labelled numpy files and raw nc files for this date
      # now, before moving on to the next date.
      # --------------------------------------------------------------
      if no_plot:
        lbl_files = sorted(set(glob.glob(os.path.join(outroot, 'labelled', pat, d_str2, sector00, '**', '*.npy'), recursive = True)))
        if len(lbl_files) > 0:
          [os.remove(x) for x in lbl_files]
          if verbose: print('Deleted labelled numpy files for date ' + d_strs[idx0])
        if use_tropdiffot:
          if len(chk_day_night) == 0:
              lbl_files = sorted(set(glob.glob(os.path.join(outroot, 'aacp_results', 'updraft_day_model', pat, '**', d_str2, sector00, '*.npy'), recursive = True)))                                                                                            #Location of the results predictions and blob_results_positioning.csv file
          else:
              lbl_files = sorted(set(glob.glob(os.path.join(outroot, 'aacp_results', 'day_night_optimal',  'updraft_day_model', pat, '**', d_str2, sector00, '*.npy'), recursive = True)))                                                                      #Location of the results predictions and blob_results_positioning.csv file
          
          if len(lbl_files) > 0:
            [os.remove(x) for x in lbl_files]
            if verbose: print('Deleted labelled numpy files for date ' + d_strs[idx0])
      
      if del_local:
        del_files = sorted(glob.glob(os.path.join(o, '**', '*.nc'), recursive = True))
        if len(del_files) > 0:
          [os.remove(x) for x in del_files]
          if verbose: print('Deleted raw netCDF files for date ' + d_strs[idx0])

    # ------------------------------------------------------------------
    # Post-loop: dates_with_ftype csv and time-aggregated figures.
    # Runs once per signature after ALL dates are done.
    # ------------------------------------------------------------------
    for sig_label, sig_use_updraft, sig_chk_dn in [('OT',   True,  chk_day_night['OT']), ('AACP', False, chk_day_night['AACP'])]:
      acc          = sig_accum[sig_label]
      outdir       = acc['outdir']
      pthresh0     = acc['pthresh0']
      img_names    = acc['img_names']
      outdate_sig  = acc['outdate']
      d_range_sig  = acc['d_range']
      tods_sig     = acc['tods']
      ir_files_sig = acc['ir_files']
      res_dirs_sig = acc['res_dirs']
      nc_files_sig = sorted(list(set(acc['nc_files'])))
      mod1_sig     = 'day_night_optimal'
      mod_pat_sig  = 'updraft_day_model' if sig_use_updraft else 'plume_day_model'

      if outdir is None or len(img_names) == 0:
        continue

      if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(outdir)), 'dates_with_ftype.csv')) == True:
        df0 = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(outdir)), 'dates_with_ftype.csv'), index_col = 0)
      else:
        if use_local:
          df0 = pd.DataFrame()
          os.makedirs(outdir, exist_ok = True)
        else:
          pref      = os.path.join(re.split(d_str, os.path.relpath(outdir, re.split('day_night_optimal', outdir)[0]))[0], d_str)
          csv_ftype = list_csv_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), 'dates_with_ftype.csv')
          df0       = pd.DataFrame() if len(csv_ftype) == 0 else load_csv_gcs(f_bucket_name, csv_ftype[0])

      df_out2  = pd.DataFrame({'out_date'   : outdate_sig})
      df_ftype = pd.DataFrame({'ftype'      : ['test']*len(outdate_sig)})
      df_sec   = pd.DataFrame({'sector'     : [str(sector00).upper()]*len(outdate_sig)})
      df_rev   = pd.DataFrame({'reviewers'  : [None]*len(outdate_sig)})
      df_nrev  = pd.DataFrame({'nreviewers' : [0]*len(outdate_sig)})
      df_night = pd.DataFrame({'use_night'  : [True]*len(outdate_sig)})
      if len(d_range_sig) > 0:
        df_range = pd.DataFrame({'date_range' : d_range_sig})
        df_merge = pd.concat([df_out2, df_sec, df_ftype, df_rev, df_nrev, df_night, df_range], axis = 1, join = 'inner')
      else:
        df_merge = pd.concat([df_out2, df_sec, df_ftype, df_rev, df_nrev, df_night], axis = 1, join = 'inner')
      if len(df0) > 0:
        df0 = pd.concat([df0, df_merge], axis = 0, join = 'inner', sort = True)
        df0 = df0.astype({'out_date' : str})
        df0.sort_values(by = 'out_date', axis = 0, inplace = True)
        df0.drop_duplicates(inplace = True)
        df0.reset_index(inplace = True)
        header = list(df0.head(0))
        if 'Unnamed: 0' in header:
          df0.drop('Unnamed: 0', axis = 'columns', inplace = True)
        if 'index' in header:
          df0.drop(columns = 'index', axis = 'columns', inplace = True)
        df_merge = df0
        df_merge = df_merge.astype({'out_date' : str})
      df_merge.to_csv(os.path.join(os.path.dirname(os.path.dirname(outdir)), 'dates_with_ftype.csv'))
      if run_gcs:
        pref = os.path.join(re.split(d_str, os.path.relpath(outdir, re.split('day_night_optimal', outdir)[0]))[0], d_str)
        write_to_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), os.path.join(os.path.dirname(os.path.dirname(outdir)), 'dates_with_ftype.csv'))

#       if not rt and not no_plot:
#         pref0   = os.path.join(re.split(d_str, os.path.relpath(outdir, re.split('day_night_optimal', outdir)[0]))[0], d_str)
#         results = []
#         for r, res_dir in enumerate(res_dirs_sig):
#           res_dir = os.path.realpath(res_dir)
#           pref    = os.path.relpath(res_dir, re.split('day_night_optimal', res_dir)[0])
#           if use_local:
#             res_file = sorted(glob.glob(os.path.join(res_dir, '*.npy')))
#           else:
#             res_file = list_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), ['results', '.npy'], delimiter = '/')
#           if len(res_file) == 0:
#             print('No results files found for ' + sig_label + '??')
#             print(pref)
#           if len(res_file) > 1:
#             pref00  = ''
#             counter = 0
#             fb = [re.split('_s|_', os.path.basename(x))[5][0:-1] for x in nc_files_sig]
#             for idx, n in enumerate(nc_files_sig):
#               if use_local:
#                 res_file2 = sorted(glob.glob(os.path.join(res_dir, '*_' + str(dims[1]) + '_*.npy')))
#                 if len(res_file2) == 0:
#                   res0 = []
#                 elif len(res_file2) > 1:
#                   date_str = fb[idx]
#                   if counter == 0:
#                     fb2     = [re.split('_', os.path.basename(x))[0] for x in res_file2]
#                     counter = 1
#                   res0 = np.load(res_file2[fb2.index(date_str)]) if date_str in fb2 else []
#                 else:
#                   res0 = np.load(res_file2[0])
#               else:
#                 pref_gcs = os.path.join('Cooney_testing', pref)
#                 if pref_gcs != pref00:
#                   res_file2 = list_gcs(f_bucket_name, pref_gcs, ['results', '_' + str(dims[1]) + '_', '.npy'], delimiter = '/')
#                   pref00    = pref_gcs
#                 if len(res_file2) > 1:
#                   date_str = fb[idx]
#                   if counter == 0:
#                     fb2     = [re.split('_', os.path.basename(x))[0] for x in res_file2]
#                     counter = 1
#                   res0 = load_npy_gcs(f_bucket_name, res_file2[fb2.index(date_str)]) if date_str in fb2 else []
#                 else:
#                   res0 = []
#               if len(res0) > 0:
#                 res0 = res0[0, :, :, 0]
#                 if tods_sig[idx].lower() == 'night':
#                   res0[res0 < sig_chk_dn['night']['pthresh']] = 0.0
#                   pthresh0 = sig_chk_dn['night']['pthresh']
#                 elif tods_sig[idx].lower() == 'day':
#                   res0[res0 < sig_chk_dn['day']['pthresh']] = 0.0
#                   pthresh0 = sig_chk_dn['day']['pthresh']
#                 else:
#                   print('tod variable must be set to day or night if given chk_day_night dictionary.')
#                   print(tods_sig[idx])
#                   exit()
#                 if len(results) <= 0:
#                   results = np.copy(res0)
#                   res0    = []
#                 else:
#                   results[res0 > results] = res0[res0 > results]                                                                                                            #Retain only the maximum values at each index
#                   res0 = []
#           else:
#             if len(res_file) == 1:
#               if run_gcs and del_local:
#                 loaded = np.nanmax(load_npy_gcs(f_bucket_name, res_file[0])[:, :, :, 0], axis = 0)
#               else:
#                 loaded = np.nanmax(np.load(res_file[0])[:, :, :, 0], axis = 0)
#               if len(results) <= 0:
#                 results = loaded
#               else:
#                 results[loaded > results] = loaded[loaded > results]                                                                                                        #Retain only the maximum values at each index
# 
#         d1  = datetime.strptime(date_range2[0], "%Y-%m-%d %H:%M:%S").strftime("%Y%j")
#         d1j = datetime.strptime(date_range2[0], "%Y-%m-%d %H:%M:%S").strftime("%j")
#         d2  = datetime.strptime(date_range2[1], "%Y-%m-%d %H:%M:%S").strftime("%j")
#         pat3 = d1 if d1j == d2 else d1 + '-' + d2
#         write_plot_model_result_predictions(nc_files_sig, ir_files_sig, tod = tods_sig, res = results, res_dir = outdir, write_gcs = run_gcs, use_local = use_local, region = region, latlon_domain = xy_bounds, pthresh = pthresh0, chk_day_night = sig_chk_dn, date_range = date_range2, outroot = os.path.join(outroot, 'aacp_results_imgs', pref0, 'time_ag_figs', pat3, str(sector00).upper()), time_ag = True, verbose = verbose)

      if del_local:
        if len(nc_files_sig) > 0:
          [os.remove(x) for x in nc_files_sig if os.path.exists(x)]                                                                                                        #Remove any remaining combined netCDF files for this signature
    
    print('Time to run entire model prediction ' + str(time.time() - t0) + ' sec')
    return
    
    
  # -----------------------------------------------------------------------
  # Standard single-signature mode (OT only, AACP only, or day/night
  # transition). This is the generalised body that covers all channel
  # counts (1, 2, 3+) by dispatching to tf_N_channel_plume_updraft_day_predict.
  # -----------------------------------------------------------------------
  t0 = time.time()                                                                                                                                                          #Start clock to time the run job process
  #Set up input and output paths   
  if date2 == None and date1 != None: date2 = date1                                                                                                                         #Default set end date to start date
  if date1 == None:
    date_range = []
  else:
    d2 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    if date2 != date1 and d2.hour == 0 and d2.minute == 0 and d2.second == 0:
        date2 = (d2 - timedelta(seconds = 1)).strftime("%Y-%m-%d %H:%M:%S")
    date_range  = [date1, date2]                                                                                                                                            #Get range of dates for not real-time runs
    date_range2 = date_range.copy()
  
  date  = datetime.utcnow()                                                                                                                                                 #Extract real time date we are running over and date we are doing the processing
  if date1 == None:     
    rt  = True                                                                                                                                                              #Real-time download flag
    pat = 'real_time'                                                                                                                                                       #Real-time file path subdirectory name
    os.makedirs(os.path.join(outroot, 'tmp_dir'), exist_ok = True)
  else:    
    rt  = False                                                                                                                                                             #Real-time download flag
    pat = 'not_real_time' 
  
  if 'mtg' in sat.lower():
      sector00 = sector.upper()
      if sector00.lower() == 'full':
        sector00 = 'F'
      if rt:
        if date.minute < 10:
          date = date - timedelta(minutes = 10)
  else:
    if 'meso' in sector.lower() or sector[0].lower() == 'm': 
      m_sector = int(sector[-1])                                                                                                                                            #Extract meso scale sector number
      sector0  = None
      sector00 = 'M' + str(m_sector)
    else:
      m_sector = 2                                                                                                                                                          #Set default meso scale sector number
      sector0  = sector[0].upper()
      sector00 = sector0
      if rt:
        if date.minute < 10 and (sector.lower() == 'full' or sector.lower() == 'f'):
          date = date - timedelta(minutes = 10)
        elif date.minute < 4 and sector.lower() == 'conus':  
          date = date - timedelta(minutes = 5)
  d_str = date.strftime("%Y-%m-%d")                                                                                                                                         #Extract date string. Used for file directory paths

  # -----------------------------------------------------------------------
  # Parse channel/input flags from chk_day_night or use_chkpnt.
  # -----------------------------------------------------------------------
  if len(chk_day_night) == 0:
    if use_chkpnt != None:                                                                                                                                                  #Get information about model from model check point filenames and paths
      if len(re.split('glm', use_chkpnt)) != 1:
        use_glm  = True
        use_glm0 = True
      else:
        use_glm  = False
        use_glm0 = False

      if len(re.split('_vis', use_chkpnt)) != 1 and len(re.split('_snowice', use_chkpnt)) != 1 and len(re.split('_cirrus', use_chkpnt)) != 1 and len(re.split('vis_', use_chkpnt)) != 1:
        use_night = False
        no_vis    = True
      else:
        no_vis    = False
      if len(re.split('_wvirdiff', use_chkpnt)) != 1:
        no_irdiff = False
      else:
        no_irdiff = True
      if len(re.split('_dirtyirdiff', use_chkpnt)) != 1:
        no_dirtyirdiff = False
      else:
        no_dirtyirdiff = True
      if (len(re.split('tropdiff',   use_chkpnt)) != 1 and len(re.split('tropdiffot', use_chkpnt)) == 1):
        no_tropdiff = False
      else:
        no_tropdiff = True
      if len(re.split('_snowice', use_chkpnt)) != 1:
        no_snowice = False
      else:
        no_snowice = True
      if len(re.split('_cirrus', use_chkpnt)) != 1:
        no_cirrus = False
      else:
        no_cirrus = True

      if len(re.split('updraft_day_model', use_chkpnt)) != 1:
        use_updraft = True
        mod_pat     = 'updraft_day_model'    
        mod00       = ' OT Detection'
      else:
        use_updraft = False
        mod_pat     = 'plume_day_model'
        mod00       = ' AACP Detection'
    else:
      no_irdiff      = True
      no_dirtyirdiff = True
      no_tropdiff    = True
      no_cirrus      = True
      no_snowice     = True
      use_glm        = True
      use_glm0       = True
      no_vis         = False
      if use_updraft:
        mod_pat = 'updraft_day_model'
        mod00   = ' OT Detection'
      else:
        mod_pat = 'plume_day_model'
        mod00   = ' AACP Detection'
  else:
    # Standard day/night transition dict: {'day': {...}, 'night': {...}}
    day0   = chk_day_night['day']                                                                                                                                           #Extract model info to be used for day
    night0 = chk_day_night['night']                                                                                                                                         #Extract model info to be used for night
    no_vis = False
    if 'use_chkpnt' not in day0 or 'mod_type' not in day0 or 'mod_inputs' not in day0 or 'pthresh' not in day0 or 'use_night' not in day0 or 'use_irdiff' not in day0 or 'use_glm' not in day0 or 'use_dirtyirdiff' not in day0  or 'use_trop' not in day0 or 'use_cirrus' not in day0 or 'use_snowice' not in day0:
      print('Day model info is missing critical information!!!')
      print(day0)
      print('Must include: use_chkpnt, mod_type, mod_inputs, pthresh, use_night, use_irdiff, use_glm, use_trop, use_dirtyirdiff, use_cirrus, use_snowice')
      exit()
    if 'use_chkpnt' not in night0 or 'mod_type' not in night0 or 'mod_inputs' not in night0 or 'pthresh' not in night0 or 'use_night' not in night0 or 'use_irdiff' not in night0 or 'use_glm' not in night0 or 'use_dirtyirdiff' not in night0  or 'use_trop' not in night0 or 'use_cirrus' not in night0 or 'use_snowice' not in night0:
      print('Night model info is missing critical information!!!')
      print(night0)
      print('Must include: use_chkpnt, mod_type, mod_inputs, pthresh, use_night, use_irdiff, use_glm, use_trop, use_dirtyirdiff, use_cirrus, use_snowice')
      exit()
    if len(re.split('updraft_day_model', day0['use_chkpnt'])) != 1:
      use_updraft = True
      mod_pat     = 'updraft_day_model'    
      mod00       = ' OT Detection'
    else:
      use_updraft = False
      mod_pat     = 'plume_day_model'
      mod00       = ' AACP Detection'
    if day0['use_irdiff'] or night0['use_irdiff']:
      no_irdiff = False
    else:
      no_irdiff = True
    if day0['use_glm'] or night0['use_glm']:
      use_glm0 = True
    else:
      use_glm0 = False
    if len(re.split('glm', day0['use_chkpnt'])) != 1 or len(re.split('glm', night0['use_chkpnt'])) != 1:
      use_glm = True
    else:
      use_glm = False
    if len(re.split('glm', day0['use_chkpnt'])) == 1:
      day0['use_glm'] = False    
    if len(re.split('glm', night0['use_chkpnt'])) == 1:
      night0['use_glm'] = False    
    day_chk_str   = day0.get('use_chkpnt', '')
    night_chk_str = night0.get('use_chkpnt', '')
    day_real_trop   = (day0['use_trop']   and
                       len(re.split('tropdiffot', day_chk_str))   == 1)
    night_real_trop = (night0['use_trop'] and
                       len(re.split('tropdiffot', night_chk_str)) == 1)
    if day_real_trop or night_real_trop:
      no_tropdiff = False
    else:
      no_tropdiff = True
    if day0['use_dirtyirdiff'] or night0['use_dirtyirdiff']:
      no_dirtyirdiff = False
    else:
      no_dirtyirdiff = True
    if day0['use_cirrus'] or night0['use_cirrus']:
      no_cirrus = False
    else:
      no_cirrus = True
    if day0['use_snowice'] or night0['use_snowice']:
      no_snowice = False
    else:
      no_snowice = True

  if use_native_ir:
    no_vis = True

  if 'mtg' in sat.lower(): 
    use_glm  = False
    use_glm0 = False 

  inroot  = os.path.realpath(inroot)                                                                                                                                        #Create link to real input root directory path so compatible with Mac
  outroot = os.path.realpath(outroot)                                                                                                                                       #Create link to real path so compatible with Mac
  if ((region != None) and (len(xy_bounds) <= 0) and (sector00[0].lower() != 'm')):
    xy_bounds = extract_us_lat_lon_region(region = region)[region]  
   
  if verbose and not no_download: print('Downloading model input data')   
  t00 = time.time()                                                                                                                                                         #Start clock to time the download job process
  if run_gcs:
    og_bucket_name2 = og_bucket_name
  else:  
    og_bucket_name2 = None
  if rt:
    if not no_download:
      if 'mtg' in sat.lower():
        og_dir, d_strs = run_download_mtg_fci_data(date1     = date1, date2 = date2,
                                                   outroot   = outroot, 
                                                   sat       = sat, 
                                                   sector    = sector, 
                                                   no_vis    = no_vis,
                                                   no_native = no_irdiff and no_dirtyirdiff and no_cirrus and no_snowice,
                                                   essl      = essl,
                                                   verbose   = verbose)
      else:
        og_dir = run_download_goes_ir_vis_l1b_glm_l2_data(date1          = date1, date2 = date2,                                                                            #Download satellite GLM, VIS, and IR data for specified date range or real time (returns all output directories)
                                                          outroot        = outroot, 
                                                          sat            = sat, 
                                                          sector         = sector, 
                                                          no_glm         = not use_glm0, 
                                                          no_vis         = no_vis,
                                                          no_irdiff      = no_irdiff, 
                                                          no_dirtyir     = no_dirtyirdiff, 
                                                          no_cirrus      = no_cirrus, 
                                                          no_snowice     = no_snowice,
                                                          gcs_bucket     = og_bucket_name2,
                                                          del_local      = False,
                                                          verbose        = verbose)

        d_strs = [os.path.basename(og_dir[0])]
    else:
      date1  = date.strftime("%Y-%m-%d %H:%M:%S")
      date01 = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S") 
      if essl and 'mtg' in sat.lower():
        og_dir = [os.path.join(outroot, date01.strftime("%Y"), date01.strftime("%m"), date01.strftime("%d"))]
      else:
        og_dir = [os.path.join(outroot, date01.strftime("%Y%m%d"))]
      d_strs = [date01.strftime("%Y%m%d")]
    if not no_tropdiff:
      download_gfs_analysis_files_from_gcloud(date.strftime("%Y-%m-%d %H:%M:%S"), 
                                              outroot         = os.path.join(os.path.dirname(outroot), 'gfs-data'),
                                              gfs_bucket_name = 'global-forecast-system', 
                                              write_gcs       = False,
                                              del_local       = True,
                                              real_time       = True,
                                              verbose         = verbose)
  else:  
    if not no_tropdiff:
      download_gfs_analysis_files(date1, date2,
                                  GFS_ANALYSIS_DT = 21600,
                                  outroot         = os.path.join(os.path.dirname(outroot), 'gfs-data'),
                                  c_bucket_name   = 'ir-vis-sandwhich',
                                  write_gcs       = False,
                                  del_local       = True,
                                  real_time       = False,
                                  verbose         = verbose)
    
    if not no_download:
      if 'mtg' in sat.lower():
        og_dir, d_strs = run_download_mtg_fci_data_parallel(date1     = date1, date2 = date2,
                                                            outroot   = outroot, 
                                                            sat       = sat, 
                                                            sector    = sector, 
                                                            no_native = no_irdiff and no_dirtyirdiff and no_cirrus and no_snowice,
                                                            no_vis    = no_vis,
                                                            essl      = essl,
                                                            verbose   = verbose)
        if len(og_dir) <= 0:
          print('No MTG data found to download???')
          print(date1)
          print(date2)
          exit()
      else:
        og_dir = run_download_goes_ir_vis_l1b_glm_l2_data_parallel(date1          = date1, date2 = date2,                                                                   #Download satellite GLM, VIS, and IR data for specified date range or real time (returns all output directories)
                                                                   outroot        = outroot, 
                                                                   sat            = sat, 
                                                                   sector         = sector, 
                                                                   no_glm         = not use_glm0, 
                                                                   no_vis         = no_vis,
                                                                   no_irdiff      = no_irdiff, 
                                                                   no_dirtyir     = no_dirtyirdiff, 
                                                                   no_cirrus      = no_cirrus, 
                                                                   no_snowice     = no_snowice,
                                                                   gcs_bucket     = og_bucket_name2,
                                                                   del_local      = False,
                                                                   verbose        = verbose)

      date02 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")                                                                                                                #Year-month-day hour:minute:second of start time to download data
      if essl and 'mtg' in sat.lower():
        if d_strs[-1] > date02.strftime("%Y%m%d"):
          og_dir = og_dir[:-1]      
          d_strs = d_strs[:-1]    
      else:
        if os.path.basename(og_dir[-1]) > date02.strftime("%Y%m%d"):
          og_dir = og_dir[:-1]
        d_strs = [os.path.basename(o) for o in og_dir]
    else:
      date01 = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")                                                                                                                #Year-month-day hour:minute:second of start time to download data
      date02 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")                                                                                                                #Year-month-day hour:minute:second of start time to download data
      if essl and 'mtg' in sat.lower():
        og_dir = sorted(list(set([
            os.path.join(outroot, 
                         (date01 + timedelta(hours=x)).strftime("%Y"),
                         (date01 + timedelta(hours=x)).strftime("%m"),
                         (date01 + timedelta(hours=x)).strftime("%d"))
            for x in range(int((date02 - date01).days * 24 + ceil((date02 - date01).seconds / 3600.0)) + 1)
        ])))
        d_strs = sorted(list(set([
            (date01 + timedelta(hours=x)).strftime("%Y%m%d")
            for x in range(int((date02 - date01).days * 24 + ceil((date02 - date01).seconds / 3600.0)) + 1)
        ])))
      else:
        og_dir = sorted(list(set([os.path.join(outroot, (date01 + timedelta(hours = x)).strftime("%Y%m%d")) for x in range(int((date02-date01).days*24 + ceil((date02-date01).seconds/3600.0))+1)])))   #Extract all date hours between date1 and date2 based on hour of satellite scan
        d_strs = sorted(list(set([(date01 + timedelta(hours = x)).strftime("%Y%m%d") for x in range(int((date02-date01).days*24 + ceil((date02-date01).seconds/3600.0))+1)]))) 

    if d_strs[-1] > date02.strftime("%Y%m%d"):
      og_dir = og_dir[:-1]      
      d_strs = d_strs[:-1]    

  if len(og_dir) == 0: print('No download output directory roots???')
  if verbose and not no_download: 
    print('Time to download files ' + str(time.time() - t00) + ' sec')
    print('Output directory of downloaded netCDF files = ' + outroot)
    print()
  if verbose: print(d_strs)

  img_path = [datetime.strptime(o, "%Y%m%d").strftime("%Y%j") for o in d_strs]
  if len(img_path) > 1 and img_path[0] != img_path[-1]:
    img_path = datetime.strptime(d_strs[0], "%Y%m%d").strftime("%Y%j") + '-' + datetime.strptime(d_strs[-1], "%Y%m%d").strftime("%Y%j")
  else:
    img_path = datetime.strptime(d_strs[0], "%Y%m%d").strftime("%Y%j")

  d_range  = []
  outdate  = []
  nc_files = []
  tods     = []
  ir_files = []
  res_dirs = []
  for idx0, o in enumerate(og_dir):
    d_str2 = datetime.strptime(d_strs[idx0], "%Y%m%d").strftime("%Y%j")                                                                                                     #Extract date string in terms of Year and Julian date
    outdate.append(d_str2)
    if verbose: print('Combining model input data into a netCDF file for date ' + d_strs[idx0])

    # ------------------------------------------------------------------
    # Determine mod0/mod1/res_dir for image creation path naming.
    # ------------------------------------------------------------------
    if len(chk_day_night) == 0:
      if use_chkpnt == None:
        if use_night:
          mod0 = 'IR+IRDIFF+GLM'
          mod1 = 'ir_irdiff_glm'
          print('Model at night not available for 3 inputs. Try giving a checkpoint file.')
          print(mod0)
          exit()
        else:
          mod0      = 'IR+VIS+GLM'
          mod1      = 'ir_vis_glm'
          use_night = False
        res_dir = os.path.join(outroot, 'aacp_results', mod1, mod_pat, pat, d_str)  
      else:
        # Derive mod0/mod1 from use_chkpnt path (covers all 1/2/3-channel cases)
        # Order: most-specific (3-channel) to least-specific (1-channel)
        if len(re.split('ir_vis_glm', use_chkpnt)) > 1:
          mod0 = 'IR+VIS+GLM';                   mod1 = 'ir_vis_glm';                   use_night = False
        elif len(re.split('ir_dirtyirdiff_irgradient_tropdiffots', use_chkpnt)) > 1:
          mod0 = 'IR+DIRTYIRDIFF+IRGRADIENT+TROPDIFFOTS'; mod1 = 'ir_dirtyirdiff_irgradient_tropdiffots'
          if use_night == None: use_night = True
        elif len(re.split('vis_tropdiff_glm', use_chkpnt)) > 1:
          mod0 = 'VIS+TROPDIFF+GLM';             mod1 = 'vis_tropdiff_glm';             use_night = False
        elif len(re.split('ir_vis_tropdiff', use_chkpnt)) > 1:
          mod0 = 'IR+VIS+TROPDIFF';              mod1 = 'ir_vis_tropdiff';              use_night = False
        elif len(re.split('ir_vis_dirtyirdiff', use_chkpnt)) > 1:
          mod0 = 'IR+VIS+DIRTYIRDIFF';           mod1 = 'ir_vis_dirtyirdiff';           use_night = False
        elif len(re.split('vis_tropdiff_dirtyirdiff', use_chkpnt)) > 1:
          mod0 = 'VIS+TROPDIFF+DIRTYIRDIFF';     mod1 = 'vis_tropdiff_dirtyirdiff';     use_night = False
        elif len(re.split('vis_tropdiff', use_chkpnt)) > 1:
          mod0 = 'VIS+TROPDIFF';                 mod1 = 'vis_tropdiff';                 use_night = False
        elif len(re.split('ir_vis', use_chkpnt)) > 1:
          mod0 = 'IR+VIS';                       mod1 = 'ir_vis';                       use_night = False
        elif len(re.split('vis_glm', use_chkpnt)) > 1:
          mod0 = 'VIS+GLM';                      mod1 = 'vis_glm';                      use_night = False
        elif len(re.split('ir_glm', use_chkpnt)) > 1:
          mod0 = 'IR+GLM';                       mod1 = 'ir_glm';
          if use_night == None: use_night = True
        elif len(re.split('tropdiff_glm', use_chkpnt)) > 1:
          mod0 = 'TROPDIFF+GLM';                 mod1 = 'tropdiff_glm';
          if use_night == None: use_night = True
        elif len(re.split('tropdiff_dirtyirdiff', use_chkpnt)) > 1:
          mod0 = 'TROPDIFF+DIRTYIRDIFF';         mod1 = 'tropdiff_dirtyirdiff';
          if use_night == None: use_night = True
        elif len(re.split('ir_wvirdiff', use_chkpnt)) > 1:
          mod0 = 'IR+WVIRDiff';                  mod1 = 'ir_wvirdiff';
          if use_night == None: use_night = True
        elif len(re.split('ir_dirtyirdiff', use_chkpnt)) > 1:
          mod0 = 'IR+DIRTYIRDIFF';               mod1 = 'ir_dirtyirdiff';
          if use_night == None: use_night = True
        elif len(re.split('ir_cirrus', use_chkpnt)) > 1:
          mod0 = 'IR+CIRRUS';                    mod1 = 'ir_cirrus';                    use_night = False
        elif len(re.split('ir_snowice', use_chkpnt)) > 1:
          mod0 = 'IR+SNOWICE';                   mod1 = 'ir_snowice';                   use_night = False
        elif len(re.split('ir_tropdiff', use_chkpnt)) > 1:
          mod0 = 'IR+TROPDIFF';                  mod1 = 'ir_tropdiff';
          if use_night == None: use_night = True
        elif len(re.split('model_checkpoints' + os.sep + 'tropdiff', use_chkpnt)) > 1:
          mod0 = 'TROPDIFF';                     mod1 = 'tropdiff';
          if use_night == None: use_night = True
        elif len(re.split('model_checkpoints' + os.sep + 'ir', use_chkpnt)) > 1:
          mod0 = 'IR';                           mod1 = 'ir';
          if use_night == None: use_night = True
        else:
          print('Not set up to handle model specified.')
          print(use_chkpnt)
          exit()
        res_dir = os.path.join(outroot, 'aacp_results', mod1, mod_pat, pat, d_str)
    else:
      mod0    = 'Day Night Optimal'
      mod1    = 'day_night_optimal'
      res_dir = os.path.join(outroot, 'aacp_results', mod1, mod_pat, pat, d_str)

    t00 = time.time()                                                                                                                                                       #Start clock to time how long the following process takes
    if 'mtg' in sat.lower():
      if rt:
        img_names, nc_names, proj, p = run_mtg_create_image_from_three_modalities(inroot               = o,                                                                 #Create combined netCDF file (do not create image)
                                                                                  glm_out_dir          = os.path.join(outroot, 'out_dir'), 
                                                                                  layered_dir          = os.path.join(outroot, 'combined_nc_dir'), 
                                                                                  img_out_dir          = os.path.join(outroot, 'aacp_results_imgs', mod1, mod_pat, pat, d_str, d_str2, str(sector)),
                                                                                  no_plot              = no_plot, grid_data = grid_data,  
                                                                                  no_write_glm         = not use_glm0, 
                                                                                  no_write_vis         = no_vis,
                                                                                  no_write_irdiff      = no_irdiff, 
                                                                                  no_write_cirrus      = no_cirrus, 
                                                                                  no_write_snowice     = no_snowice, 
                                                                                  no_write_dirtyirdiff = no_dirtyirdiff, 
                                                                                  domain_sector        = sector, 
                                                                                  run_gcs              = run_gcs, 
                                                                                  write_combined_gcs   = False, 
                                                                                  append_nc            = True, 
                                                                                  region               = region, xy_bounds = xy_bounds, 
                                                                                  real_time            = rt, del_local = False, 
                                                                                  plt_model            = os.path.join(res_dir, d_str2, str(sector)), 
                                                                                  model                = mod0 + mod00, 
                                                                                  satellite            = sat, 
                                                                                  chk_day_night        = chk_day_night, 
                                                                                  pthresh              = pthresh, 
                                                                                  in_bucket_name       = og_bucket_name, out_bucket_name = c_bucket_name,
                                                                                  verbose              = verbose)
      else:
        img_names, nc_names, proj = run_mtg_create_image_from_three_modalities2(inroot               = o,                                                                   #Create combined netCDF file (do not create image)
                                                                                glm_out_dir          = os.path.join(outroot, 'out_dir'), 
                                                                                layered_dir          = os.path.join(outroot, 'combined_nc_dir'), 
                                                                                date_range           = date_range,
                                                                                no_plot              = True, grid_data = grid_data,  
                                                                                no_write_glm         = not use_glm0, 
                                                                                no_write_vis         = no_vis,
                                                                                no_write_irdiff      = no_irdiff, 
                                                                                no_write_cirrus      = no_cirrus, 
                                                                                no_write_snowice     = no_snowice, 
                                                                                no_write_dirtyirdiff = no_dirtyirdiff, 
                                                                                domain_sector        = sector, 
                                                                                run_gcs              = run_gcs, 
                                                                                write_combined_gcs   = False, 
                                                                                append_nc            = True, 
                                                                                region               = region, xy_bounds = xy_bounds,  
                                                                                real_time            = rt, del_local = False, 
                                                                                satellite            = sat, 
                                                                                in_bucket_name       = og_bucket_name, out_bucket_name = c_bucket_name,
                                                                                verbose              = verbose)
    else:
      if rt:
        img_names, nc_names, proj, p = run_create_image_from_three_modalities(inroot               = o,                                                                     #Create combined netCDF file (do not create image)
                                                                              glm_out_dir          = os.path.join(outroot, 'out_dir'), 
                                                                              layered_dir          = os.path.join(outroot, 'combined_nc_dir'), 
                                                                              img_out_dir          = os.path.join(outroot, 'aacp_results_imgs', mod1, mod_pat, pat, d_str, d_str2, str(sector00)),
                                                                              no_plot              = no_plot, grid_data = grid_data,  
                                                                              no_write_glm         = not use_glm0, 
                                                                              no_write_vis         = no_vis,
                                                                              no_write_irdiff      = no_irdiff, 
                                                                              no_write_cirrus      = no_cirrus, 
                                                                              no_write_snowice     = no_snowice, 
                                                                              no_write_dirtyirdiff = no_dirtyirdiff, 
                                                                              meso_sector          = m_sector, 
                                                                              domain_sector        = sector0, 
                                                                              run_gcs              = run_gcs, 
                                                                              write_combined_gcs   = False, 
                                                                              append_nc            = True, 
                                                                              region               = region, xy_bounds = xy_bounds, 
                                                                              real_time            = rt, del_local = False, 
                                                                              plt_model            = os.path.join(res_dir, d_str2, str(sector00)), 
                                                                              model                = mod0 + mod00, 
                                                                              chk_day_night        = chk_day_night, 
                                                                              pthresh              = pthresh, 
                                                                              in_bucket_name       = og_bucket_name, out_bucket_name = c_bucket_name,
                                                                              verbose              = verbose)
      else:
        img_names, nc_names, proj = run_create_image_from_three_modalities2(inroot               = o,                                                                       #Create combined netCDF file (do not create image)
                                                                            glm_out_dir          = os.path.join(outroot, 'out_dir'), 
                                                                            layered_dir          = os.path.join(outroot, 'combined_nc_dir'), 
                                                                            date_range           = date_range,
                                                                            no_plot              = True, grid_data = grid_data,  
                                                                            no_write_glm         = not use_glm0, 
                                                                            no_write_vis         = no_vis,
                                                                            no_write_irdiff      = no_irdiff, 
                                                                            no_write_cirrus      = no_cirrus, 
                                                                            no_write_snowice     = no_snowice, 
                                                                            no_write_dirtyirdiff = no_dirtyirdiff, 
                                                                            meso_sector          = m_sector, 
                                                                            domain_sector        = sector0, 
                                                                            run_gcs              = run_gcs, 
                                                                            write_combined_gcs   = False, 
                                                                            append_nc            = True, 
                                                                            region               = region, xy_bounds = xy_bounds, 
                                                                            real_time            = rt, del_local = False, 
                                                                            in_bucket_name       = og_bucket_name, out_bucket_name = c_bucket_name,
                                                                            verbose              = verbose)
     
    if not no_tropdiff:
      run_write_gfs_trop_temp_to_combined_ncdf(inroot        = os.path.join(outroot, 'combined_nc_dir', d_strs[idx0]), 
                                               gfs_root      = os.path.join(os.path.dirname(outroot), 'gfs-data'),
                                               date1         = date1,
                                               date2         = date2,
                                               use_local     = True, write_gcs = run_gcs, del_local = del_local,
                                               c_bucket_name = c_bucket_name,
                                               real_time     = rt,
                                               sector        = sector00,
                                               verbose       = verbose)
    
    if len(img_names) > 0:
      nc_files.extend(sorted(nc_names))
      if verbose: 
        print('Data combined into a netCDF file for date ' + d_strs[idx0])
        print('Time to create combined netCDF file ' + str(time.time() - t00) + ' sec')
        print('Output directory of combined netCDF files = ' + os.path.join(outroot, 'combined_nc_dir'))
        print()
        print('Creating numpy files for date ' + d_strs[idx0])
      t00 = time.time()                                                                                                                                                     #Start clock to time how long the following process takes
      if 'mtg' in sat.lower():
        ir_file0, dims = mtg_create_vis_ir_numpy_arrays_from_netcdf_files2(inroot               = o,                                                                        #Create VIS, IR, GLM and solar zenith angle numpy arrays. (returns ir.npy file name and path as well as image dimensions)
                                                                           layered_root         = os.path.join(outroot, 'combined_nc_dir'),
                                                                           outroot              = os.path.join(outroot, 'labelled', pat),
                                                                           json_root            = os.path.join(outroot, 'labelled'),  
                                                                           date_range           = date_range, 
                                                                           domain_sector        = sector, 
                                                                           use_native_ir        = use_native_ir, 
                                                                           no_write_glm         = not use_glm, 
                                                                           no_write_vis         = no_vis,
                                                                           no_write_irdiff      = no_irdiff, 
                                                                           no_write_cirrus      = no_cirrus, 
                                                                           no_write_snowice     = no_snowice, 
                                                                           no_write_dirtyirdiff = no_dirtyirdiff, 
                                                                           no_write_trop        = no_tropdiff, 
                                                                           run_gcs              = run_gcs, real_time = rt, del_local = del_local, use_local = True,
                                                                           og_bucket_name       = og_bucket_name, 
                                                                           comb_bucket_name     = c_bucket_name, 
                                                                           proc_bucket_name     = p_bucket_name, 
                                                                           new_weighting        = new_weighting,
                                                                           verbose              = verbose)
      else:
        ir_file0, dims = create_vis_ir_numpy_arrays_from_netcdf_files2(inroot               = o,                                                                            #Create VIS, IR, GLM and solar zenith angle numpy arrays. (returns ir.npy file name and path as well as image dimensions)
                                                                       layered_root         = os.path.join(outroot, 'combined_nc_dir'),
                                                                       outroot              = os.path.join(outroot, 'labelled', pat),
                                                                       json_root            = os.path.join(outroot, 'labelled'),  
                                                                       date_range           = date_range, 
                                                                       meso_sector          = m_sector, 
                                                                       domain_sector        = sector0, 
                                                                       no_write_glm         = not use_glm, 
                                                                       no_write_vis         = no_vis,
                                                                       no_write_irdiff      = no_irdiff, 
                                                                       no_write_cirrus      = no_cirrus, 
                                                                       no_write_snowice     = no_snowice, 
                                                                       no_write_dirtyirdiff = no_dirtyirdiff, 
                                                                       no_write_trop        = no_tropdiff, 
                                                                       run_gcs              = run_gcs, real_time = rt, del_local = del_local, use_local = True,
                                                                       og_bucket_name       = og_bucket_name, 
                                                                       comb_bucket_name     = c_bucket_name, 
                                                                       proc_bucket_name     = p_bucket_name, 
                                                                       new_weighting        = new_weighting,
                                                                       verbose              = verbose)
      
      ir_files.append(ir_file0)                                                                                                                                             #Append array storing IR file names
      if verbose: 
        print('Data numpy files created for date ' + d_strs[idx0])
        print('Time to create numpy files = ' + str(time.time() - t00) + ' sec')
        print('Output directory of numpy files = ' + os.path.join(outroot, 'labelled', pat))
        print()
      
      t00 = time.time()                                                                                                                                                     #Start clock to time the model predictions process

      # ------------------------------------------------------------------
      # Dispatch to tf_N_channel inner function.
      # In transition mode the day/night inner calls are made inside
      # tf_N_channel_plume_updraft_day_predict by checking channel count.
      # ------------------------------------------------------------------
      if len(chk_day_night) == 0:
        outdir, pthresh0, df = tf_N_channel_plume_updraft_day_predict(dims          = dims,
                                                                      use_glm       = use_glm,
                                                                      use_updraft   = use_updraft, 
                                                                      d_str         = d_str,
                                                                      date_range    = date_range,
                                                                      sector        = sector00,
                                                                      pthresh       = pthresh,
                                                                      opt_pthresh   = opt_pthresh,
                                                                      rt            = rt, 
                                                                      run_gcs       = run_gcs, 
                                                                      use_local     = use_local, 
                                                                      del_local     = del_local, 
                                                                      inroot        = o, 
                                                                      outroot       = outroot, 
                                                                      p_bucket_name = p_bucket_name,
                                                                      c_bucket_name = c_bucket_name, 
                                                                      f_bucket_name = f_bucket_name, 
                                                                      use_chkpnt    = use_chkpnt, 
                                                                      by_date       = by_date, by_updraft = by_updraft, subset = subset, 
                                                                      use_night     = use_night,
                                                                      rewrite_model = rewrite_model, 
                                                                      chk_day_night = chk_day_night, 
                                                                      use_native_ir = use_native_ir,
                                                                      no_write_npy  = no_plot,
                                                                      verbose       = verbose)
      else:
        # Transition mode: dispatch day and night calls by channel count
        day0      = chk_day_night['day']
        night0    = chk_day_night['night']
        day_chk   = day0['use_chkpnt']
        night_chk = night0['use_chkpnt']
        pthresh0  = None

        day_outdir, day_pthresh0, day_df = tf_N_channel_plume_updraft_day_predict(
                                                                      dims          = dims,
                                                                      use_glm       = day0['use_glm'],
                                                                      use_updraft   = use_updraft, 
                                                                      d_str         = d_str,
                                                                      date_range    = date_range,
                                                                      sector        = sector00,
                                                                      pthresh       = day0['pthresh'],
                                                                      opt_pthresh   = opt_pthresh,
                                                                      rt            = rt, 
                                                                      run_gcs       = run_gcs, 
                                                                      use_local     = use_local, 
                                                                      del_local     = del_local, 
                                                                      inroot        = o, 
                                                                      outroot       = outroot, 
                                                                      p_bucket_name = p_bucket_name,
                                                                      c_bucket_name = c_bucket_name, 
                                                                      f_bucket_name = f_bucket_name, 
                                                                      use_chkpnt    = day_chk, 
                                                                      by_date       = by_date, by_updraft = by_updraft, subset = subset, 
                                                                      use_night     = False,
                                                                      night_only    = False, 
                                                                      rewrite_model = rewrite_model, 
                                                                      chk_day_night = chk_day_night, 
                                                                      use_native_ir = use_native_ir,
                                                                      no_write_npy  = no_plot,
                                                                      verbose       = verbose)
        outdir = day_outdir

        if len(day_df) > 0 and rt:
          night_outdir = day_outdir
          night_df     = pd.DataFrame()
        else:
          night_outdir, night_pthresh0, night_df = tf_N_channel_plume_updraft_day_predict(
                                                                      dims          = dims,
                                                                      use_glm       = night0['use_glm'],
                                                                      use_updraft   = use_updraft, 
                                                                      d_str         = d_str,
                                                                      date_range    = date_range,
                                                                      sector        = sector00,
                                                                      pthresh       = night0['pthresh'],
                                                                      opt_pthresh   = opt_pthresh,
                                                                      rt            = rt, 
                                                                      run_gcs       = run_gcs, 
                                                                      use_local     = use_local, 
                                                                      del_local     = del_local, 
                                                                      inroot        = o, 
                                                                      outroot       = outroot, 
                                                                      p_bucket_name = p_bucket_name,
                                                                      c_bucket_name = c_bucket_name, 
                                                                      f_bucket_name = f_bucket_name, 
                                                                      use_chkpnt    = night_chk, 
                                                                      by_date       = by_date, by_updraft = by_updraft, subset = subset, 
                                                                      use_night     = True,
                                                                      night_only    = True, 
                                                                      rewrite_model = rewrite_model, 
                                                                      chk_day_night = chk_day_night, 
                                                                      use_native_ir = use_native_ir,
                                                                      no_write_npy  = no_plot,
                                                                      verbose       = verbose)
          outdir = night_outdir

        if len(night_df) > 0 and len(day_df) > 0:
          df = pd.concat([day_df, night_df], axis = 0, join = 'inner', sort = True)
          df.sort_values(by = 'date_time', axis = 0, inplace = True)
          df.reset_index(inplace = True)
          df.drop(columns = 'index', axis = 'columns', inplace = True)
        elif len(night_df) == 0:
          df = day_df.copy()
        elif len(day_df) == 0: 
          df = night_df.copy()

      res_dirs.append(outdir)
      if verbose: 
        print('Time to run prediction ' + str(time.time() - t00) + ' sec')
        print()
      if not rt:
        if date_range2 != None and len(df) > 0:
          if date_range2[0] > str(np.min(df['date_time'])):                                                                                                                 #Correct range of dates if necessary
            date_range2[0] = str(np.min(df['date_time']))
          date_range2[1] = str(np.max(df['date_time']))
        d_range.append(date_range[0] + ';' + date_range[1])
      else:
        d_range = []
      nc_names = sorted(list(df['glm_files']))
      if rt:
        if df.empty:
          tod      = []
          nc_names = []
        else:  
          tod      = [df['day_night'][len(df)-1]]
          nc_names = [nc_names[-1]]
      else:
        tod = list(df['day_night'])

      tods.extend(tod)
      if verbose:
        print(nc_names)
        print(ir_files)
        print(outdir)
        print(tod)
        print(len(nc_names))
        print(len(tod))
      if len(nc_names) != len(tod):
        print('Time of day list length does not match nc_names list length??')
        print(len(nc_names))
        print(tod)
        exit()

    if no_plot:
      lbl_files = sorted(set(glob.glob(os.path.join(outroot, 'labelled', pat, d_str2, sector00, '**', '*.npy'), recursive = True)))
      if len(lbl_files) > 0:     
        [os.remove(x) for x in lbl_files]                                                                                                                                   #Remove labelled numpy files if not plotting

#Plot and write locations of OTs or AACPs
    if len(nc_names) > 0:
      if rt:
        im_names, df_out = write_plot_model_result_predictions(nc_names, [], tod = tod, res = [], use_local = use_local, no_plot = True, res_dir = outdir, region = region, latlon_domain = xy_bounds, pthresh = pthresh0, model = mod0 + mod00, chk_day_night = chk_day_night, grid_data = grid_data, proj = proj, outroot = os.path.join(outroot, 'aacp_results_imgs', mod1, mod_pat, pat, d_str, img_path, str(sector00).upper()), verbose = verbose)
        if not no_plot:
          p.join()
          p.close()
          while not os.path.exists(im_names[0]):
            time.sleep(1)                                                                                                                                                   #Wait until file exists, then write to gcs
      else:    
        if not no_plot:
          if len(img_names) > 0:
            im_names, df_out = write_plot_model_result_predictions(nc_names, [], tod = tod, res = [], use_local = use_local, no_plot = no_plot, res_dir = outdir, no_plot_vis = use_native_ir, region = region, latlon_domain = xy_bounds, pthresh = pthresh0, model = mod0 + mod00, chk_day_night = chk_day_night, grid_data = grid_data, proj = proj, outroot = os.path.join(outroot, 'aacp_results_imgs', mod1, mod_pat, pat, d_str, img_path, str(sector00).upper()), verbose = verbose)
          else:
            im_names, df_out = write_plot_model_result_predictions(nc_names, [], tod = tod, res = [], use_local = use_local, no_plot = no_plot, res_dir = outdir, no_plot_vis = use_native_ir, region = region, latlon_domain = xy_bounds, pthresh = pthresh0, model = mod0 + mod00, chk_day_night = chk_day_night, grid_data = grid_data, proj = None, outroot = os.path.join(outroot, 'aacp_results_imgs', mod1, mod_pat, pat, d_str, img_path, str(sector00).upper()), verbose = verbose)
        else:
          im_names = []

      if run_gcs and not no_plot:
        pref = os.path.relpath(os.path.dirname(im_names[0]), re.split('aacp_results_imgs', os.path.dirname(im_names[0]))[0])
        for i in im_names:
          write_to_gcs(f_bucket_name, pref, i, del_local = del_local)                                                                                                       #Write results to Google Cloud Storage Bucket
    else:
      if rt and not no_plot:
        print('No image created because model you wanted to run cannot be used for night time operations')
        p.kill()

    if del_local:
      del_files = sorted(glob.glob(os.path.join(o, '**', '*.nc'), recursive = True))   
      if len(del_files) > 0:     
        [os.remove(x) for x in del_files]                                                                                                                                   #Remove all combined netCDF files from local storage

  if len(img_names) > 0:
    if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(outdir)), 'dates_with_ftype.csv')) == True:
      df0 = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(outdir)), 'dates_with_ftype.csv'), index_col = 0)                                                      #Read previous csv file
    else:
      if use_local:
        df0 = pd.DataFrame()                                                                                                                                                #Create empty data frame
        os.makedirs(outdir, exist_ok = True) 
      else:
        if 'day_night_optimal' in outdir:
          pref = os.path.join(re.split(d_str, os.path.relpath(outdir, re.split('day_night_optimal', outdir)[0]))[0], d_str)
        else:
          pref = os.path.join(re.split(d_str, os.path.relpath(outdir, re.split(mod1, outdir)[0]))[0], d_str)
        csv_ftype = list_csv_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), 'dates_with_ftype.csv')                                                               #Find dates_wtih_ftype csv file to see if exists or not
        if len(csv_ftype) == 0:
          df0 = pd.DataFrame()                                                                                                                                              #Create empty data frame
        else:
          if len(csv_ftype) > 0:
            df0 = load_csv_gcs(f_bucket_name, csv_ftype[0])                                                                                                                 #Read specified csv file
  
    df_out2  = pd.DataFrame({'out_date'   : outdate}) 
    df_ftype = pd.DataFrame({'ftype'      : ['test']*len(outdate)}) 
    df_sec   = pd.DataFrame({'sector'     : [str(sector00).upper()]*len(outdate)}) 
    df_rev   = pd.DataFrame({'reviewers'  : [None]*len(outdate)})
    df_nrev  = pd.DataFrame({'nreviewers' : [0]*len(outdate)})
    df_night = pd.DataFrame({'use_night'  : [use_night]*len(outdate)})
    if len(d_range) > 0:
      df_range = pd.DataFrame({'date_range' : d_range})
      df_merge = pd.concat([df_out2, df_sec, df_ftype, df_rev, df_nrev, df_night, df_range], axis = 1, join = 'inner')
    else:
      df_merge = pd.concat([df_out2, df_sec, df_ftype, df_rev, df_nrev, df_night], axis = 1, join = 'inner')
    if len(df0) > 0:
      df0 = pd.concat([df0, df_merge], axis = 0, join = 'inner', sort = True)                                                                                               #Concatenate the dataframes and sort by date 
      df0 = df0.astype({'out_date' : str})
      df0.sort_values(by = 'out_date', axis =0, inplace = True)
      df0.drop_duplicates(inplace = True)
      df0.reset_index(inplace = True)
      header = list(df0.head(0))                                                                                                                                            #Extract header of csv file
      if 'Unnamed: 0' in header:
        df0.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                                                            #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
      if 'index' in header:
        df0.drop(columns = 'index', axis = 'columns', inplace = True)                                                                                                       #Remove index column that was created by reset_index.
      df_merge = df0        
      df_merge = df_merge.astype({'out_date' : str})
    df_merge.to_csv(os.path.join(os.path.dirname(os.path.dirname(outdir)), 'dates_with_ftype.csv'))                                                                         #Write output file
    if run_gcs:
      if 'day_night_optimal' in outdir:
        pref = os.path.join(re.split(d_str, os.path.relpath(outdir, re.split('day_night_optimal', outdir)[0]))[0], d_str)
      else:
        pref = os.path.join(re.split(d_str, os.path.relpath(outdir, re.split(mod1, outdir)[0]))[0], d_str)
      write_to_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), os.path.join(os.path.dirname(os.path.dirname(outdir)), 'dates_with_ftype.csv'))                     #Write results csv file to Google Cloud Storage Bucket
    nc_files = sorted(list(set(nc_files)))
    if rt != True and no_plot == False:
      if 'day_night_optimal' in outdir:
        pref0 = os.path.join(re.split(d_str, os.path.relpath(outdir, re.split('day_night_optimal', outdir)[0]))[0], d_str)
      else:
        pref0 = os.path.join(re.split(d_str, os.path.relpath(outdir, re.split(mod1, outdir)[0]))[0], d_str)
      results = []
      for r, res_dir in enumerate(res_dirs):
        res_dir = os.path.realpath(res_dir)
        if 'day_night_optimal' in res_dir:
          pref = os.path.relpath(res_dir, re.split('day_night_optimal', res_dir)[0])
        else:
          pref = os.path.relpath(res_dir, re.split(mod1, res_dir)[0])
        if use_local == True:
          res_file = sorted(glob.glob(os.path.join(res_dir, '*.npy')))
        else:
          res_file = list_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), ['results', '.npy'], delimiter = '/')
        if len(res_file) ==  0:
          print('No results files found??')
          print(pref)
          print(f_bucket_name)
          print(res_file)
        if len(res_file) > 1:
          pref00  = ''
          counter = 0
          fb = [re.split('_s|_', os.path.basename(x))[5][0:-1] for x in nc_files]
          for idx, n in enumerate(nc_files): 
            if use_local == True:
              res_file = sorted(glob.glob(os.path.join(res_dir, '*_' + str(dims[1]) + '_*.npy')))                                                                           #Extract name of numpy results file
              if len(res_file) == 0:    
                print('No results file found??')    
                res0 = []  
              if len(res_file) > 1:                                                                                                                                         #If results file for each scan
                date_str = fb[idx]                                                                                                                                          #Split file string in order to extract date string of scan
                if counter == 0:
                  fb2 = [re.split('_', os.path.basename(x))[0] for x in res_file]
                  counter = 1
                if date_str in fb2:
                  res0 = np.load(res_file[fb2.index(date_str)])                                                                                                             #Load numpy results file
                else:
                  res0 = []
              else:
                res0 = np.load(res_file[0])                                                                                                                                 #Load numpy results file
            else:    
              pref = os.path.join('Cooney_testing', pref)   
              if pref != pref00:
                res_file = list_gcs(f_bucket_name, pref, ['results', '_' + str(dims[1]) + '_', '.npy'], delimiter = '/')                                                    #Extract name of model results file
                if len(res_file) == 0:    
                  print('No results file found??')
                  print(pref)
                  print(f_bucket_name)
                  res0 = []  
                pref00 = pref
              if len(res_file) > 1:    
                date_str = fb[idx]                                                                                                                                          #Split file string in order to extract date string of scan
                if counter == 0:
                  fb2 = [re.split('_', os.path.basename(x))[0] for x in res_file]
                  counter = 1
                if date_str in fb2:
                  res0 = load_npy_gcs(f_bucket_name, res_file[fb2.index(date_str)])                                                                                         #Load numpy results file
                else:
                  res0 = []          
            if len(res0) > 0:
              res0 = res0[0, :, :, 0]
              if len(chk_day_night) != 0:
                if tods[idx].lower() == 'night':
                  res0[res0 < chk_day_night['night']['pthresh']] = 0.0
                  pthresh = chk_day_night['night']['pthresh']
                elif tods[idx].lower() == 'day':
                  res0[res0 < chk_day_night['day']['pthresh']] = 0.0
                  pthresh = chk_day_night['day']['pthresh']
                else:
                  print('tod variable must be set to day or night if given chk_day_night dictionary.')
                  print(tods[idx])
                  print(chk_day_night)
                  exit()
              if len(results) <= 0:
                results = np.copy(res0)
                res0    = [] 
              else:  
                results[res0 > results] = res0[res0 > results]                                                                                                              #Retain only the maximum values at each index
                res0 = [] 
        else:
          if len(res_file) == 1:
            if run_gcs == True and del_local == True: 
              if len(results) <= 0:
                results = np.nanmax(load_npy_gcs(f_bucket_name, res_file[0])[:, :, :, 0], axis = 0)
              else:
                res0    = np.nanmax(load_npy_gcs(f_bucket_name, res_file[0])[:, :, :, 0], axis = 0)
                results[res0 > results] = res0[res0 > results]                                                                                                              #Retain only the maximum values at each index
                res0    = []
            else:
              if len(results) <= 0:
                results = np.nanmax(np.load(res_file[0])[:, :, :, 0], axis = 0)
              else:
                res0    = np.nanmax(np.load(res_file[0])[:, :, :, 0], axis = 0)
                results[res0 > results] = res0[res0 > results]                                                                                                              #Retain only the maximum values at each index
                res0 = []
     
      d1  = datetime.strptime(date_range2[0], "%Y-%m-%d %H:%M:%S").strftime("%Y%j")                                                                                         #Extract dates as string in order to put onto plots and in out filepaths
      d1j = datetime.strptime(date_range2[0], "%Y-%m-%d %H:%M:%S").strftime("%j")
      d2  = datetime.strptime(date_range2[1], "%Y-%m-%d %H:%M:%S").strftime("%j")
      if d1j == d2:
        pat3 = d1
      else:
        pat3 = d1 + '-' + d2
      write_plot_model_result_predictions(nc_files, ir_files, tod = tods, res = results, res_dir = outdir, write_gcs = run_gcs, use_local = use_local, region = region, latlon_domain = xy_bounds, pthresh = pthresh0, chk_day_night = chk_day_night, date_range = date_range2, outroot = os.path.join(outroot, 'aacp_results_imgs', pref0, 'time_ag_figs', pat3, str(sector00).upper()), time_ag = True, verbose = verbose)
    
  if del_local: 
    if len(nc_files) > 0: 
      [os.remove(x) for x in nc_files]                                                                                                                                      #Remove all combined netCDF files from local storage
  print('Time to run entire model prediction ' + str(time.time() - t0) + ' sec')
  print()


def tf_N_channel_plume_updraft_day_predict(dims           = [1, 2000, 2000], 
                                           use_glm        = False,
                                           use_updraft    = True, 
                                           d_str          = None,
                                           date_range     = [], 
                                           sector         = 'M2', 
                                           pthresh        = None,
                                           opt_pthresh    = None,
                                           rt             = True,
                                           run_gcs        = True, 
                                           use_local      = False, 
                                           del_local      = True, 
                                           inroot         = os.path.join('..', '..', '..', 'goes-data', '20190505'), 
                                           outroot        = os.path.join('..', '..', '..', 'goes-data'), 
                                           p_bucket_name  = 'aacp-proc-data',
                                           c_bucket_name  = 'ir-vis-sandwhich', 
                                           f_bucket_name  = 'aacp-results', 
                                           use_chkpnt     = None, 
                                           by_date        = True,  by_updraft = True, subset = True, 
                                           use_night      = None, night_only = False, 
                                           rewrite_model  = False, 
                                           chk_day_night  = {}, 
                                           use_native_ir  = False, 
                                           no_write_npy   = False, 
                                           verbose        = True):
  '''
  This is a function to predict the N-channel updraft or plume model based on the most recent previous model run
  results checkpoint weights. Consolidates tf_1/2/3_channel_plume_updraft_day_predict into a single function.
  The number of channels is derived automatically from the use_chkpnt path or the chk_day_night dictionary,
  and the correct model architecture and input tensor are constructed accordingly.

  Args:
      None.
  Keywords:
      dims           : Numpy array size dimensions. DEFAULT = [1, 2000, 2000, 1] which corresponds to GOES satellite mesoscale size.
      use_glm        : IF keyword set (True), pre-process and run model for GLM data rather than IR or VIS data.
                       DEFAULT = False -> pre-process using IR or VIS data.
      use_updraft    : IF keyword set (True), pre-process and run OT model, otherwise run plume model.
                       DEFAULT = True.
      d_str          : Date string of predictions. If None, d_str uses current date/time
                       DEFAULT = None.
      date_range     : 2 element List containing the start date and end date to only run the model over. DEFAULT = [].                 
      sector         : INT keyword to specifiy the mesoscale sector. DEFAULT = 'M2'
      pthresh        : FLOAT keyword to specify probability threshold to use. 
                       DEFAULT = None -> Use maximized IoU probability threshold.
      opt_pthresh    : FLOAT keyword to specify probability threshold that was found to optimize the specified model during Cooney testing. 
                       DEFAULT = None.
      rt             : Real-time flag.
                       DEFAULT = True.
      run_gcs        : IF keyword set (True), write the output files to the google cloud in addition to local storage.
                       DEFAULT = True.
      use_local      : IF keyword set (True), use local files. If False, use files located on the google cloud. 
                       DEFAULT = False -> use files on google cloud server. 
      del_local      : IF keyword set (True) AND run_gcs = True, delete local copy of output file.
                       DEFAULT = True.
      inroot         : STRING specifying input root directory to the dowloaded IR, GLM, and VIS files.
                       DEFAULT = '../../../goes-data/20190505/'
      outroot        : STRING output directory path for results data storage
                       DEFAULT = '../../../goes-data/'
      p_bucket_name  : STRING specifying the name of the gcp bucket to write IR/VIS/GLM numpy files to. run_gcs needs to be True in order for this to matter.
                       DEFAULT = 'aacp-proc-data'
      f_bucket_name  : STRING specifying the name of the gcp bucket to write model prediction results files to. run_gcs needs to be True in order for this to matter.
                       DEFAULT = 'aacp-results'
      use_chkpnt     : Optional STRING keyword to specify file name and path to check point file with model weights from a previous model run you want to use.
                       Setting this keyword allows the user to skip over all the model creation stuff and go straight to loading the weights from a trained model.
      by_date        : IF keyword set (True), use model training, validation, and testing numpy output files for separate dates.
                       DEFAULT = True. If False, use first 75% of data for training, next 20% for validation, and last 5% for testing.
      by_updraft     : IF keyword set (True), create model training, validation, and testing numpy output files for 64x64 pixels around each updraft by dates.
                       NOTE: If this keyword is set, by_date will automatically also be set.
                       DEFAULT = True. If False, use by_date 512x512 or first 75% of data for training, next 20% for validation, and last 5% for testing.
      subset         : IF keyword set (True), use numpy files in which subset indices [x0, y0, x1, y1] were given for the model training, testing, and validation.
                       DEFAULT = True. False implies using the 512x512 boxes created for the full domain
      use_night      : IF keyword set (True), use night time data. CURRENTLY ONLY SET UP TO DO IR+GLM!!!!
                       DEFAULT = None -> True for IR+GLM and False for VIS+GLM and IR+VIS runs
      night_only     : IF keyword set (True), run model for night time hours ONLY.
                       DEFAULT = False -> run for day and night(if use_night = True)
      rewrite_model  : IF keyword set (True), rewrite the model numpy files. 
                       Keyword is ignored if real_time = True
                       DEFAULT = False -> do not rewrite model files if already exists.     
      chk_day_night  : Structure containing day and night model data information in order to make seemless transition in predictions of the 2.
                       DEFAULT = {} -> do not use day_night transition                 
      use_native_ir  : IF keyword set (True), run model at the native IR data resolution.
                       DEFAULT = False -> interpolate IR resolution data to higher resolution visible data grid.
      no_write_npy   : IF keyword set (True), do not write numpy results files. Only append the combined netCDF files
                       DEFAULT = False -> write the numpy results files
      verbose        : BOOL keyword to specify whether or not to print verbose informational messages.
                       DEFAULT = True which implies to print verbose informational messages
  Returns:
      Numpy files with the model results for real time date or specified date range.
  '''  
  backend.clear_session()
  if night_only:
    use_night = True
  if d_str == None: d_str = datetime.utcnow().strftime("%Y-%m-%d")
  try:
      d_str2 = datetime.strptime(os.path.basename(os.path.realpath(inroot)), "%Y%m%d").strftime("%Y%j")                                                                     #Extract date string in terms of Year and Julian date
  except ValueError:
      try:
          parts  = os.path.normpath(os.path.realpath(inroot)).split(os.sep)
          d_str2 = datetime.strptime(f"{parts[-3]}{parts[-2]}{parts[-1]}", "%Y%m%d").strftime("%Y%j")
      except (ValueError, IndexError):
          print("Error: Could not parse date from inroot.")
          d_str2 = None
  if rt:  
    pat = 'real_time'                                                                                                                                                       #Real-time file path subdirectory name
  else: 
    pat = 'not_real_time' 
  if use_updraft:
    mod_pat  = 'updraft_day_model'
    mod_name = 'OT'
    val_set  = 0
  else:
    mod_pat  = 'plume_day_model'
    mod_name = 'AACP'
    val_set  = 0

  if use_local:
    csv_files = sorted(glob.glob(os.path.join(outroot, 'labelled', pat, d_str2, str(sector), 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'), recursive = True))   #Extract names of all of the GOES visible data files
  else:                                   
    csv_files = list_csv_gcs(p_bucket_name, os.path.join(os.path.basename(os.path.realpath(outroot)), 'labelled', pat, d_str2, str(sector)), 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv') #Extract names of all of the GOES visible data files
  
  if len(csv_files) == 0: 
    print('No csv files found??')
    exit()
  if len(csv_files) > 1:
    print('Multiple csv files found???')
    exit()
  dir_path, fb = os.path.split(csv_files[0])                                                                                                                                #Extract csv file directory path and file basename
  if use_local:  
    df = pd.read_csv(csv_files[0], index_col = 0)                                                                                                                           #Read specified csv file
  else:                                     
    df = load_csv_gcs(p_bucket_name, csv_files[0])                                                                                                                          #Read specified csv file
 
  if rt:
    df = df[-1:].copy()
  if use_night == False:
    df.dropna(subset = ['vis_files'], inplace = True)                                                                                                                       #Find night time scans and remove from model que
  if night_only == True:
    df = df[(df['day_night'] == 'night')]

  # Determine mod0/mod1 from use_chkpnt path.
  if use_chkpnt == None:
    if use_glm:
      mod0  = 'IR+GLM'
      mod1  = 'ir_glm'
      if use_night == None: use_night = True 
    else:
      mod0      = 'IR'
      mod1      = 'ir'
      if use_night == None: use_night = True

    indir   = os.path.join(outroot, 'aacp_results', mod1, mod_pat)
    dirs    = os.path.join('Cooney_testing', mod1, mod_pat)                                                                                                                 #Path to search for checkpoint file in GCP
    outdir0 = os.path.join(outroot, 'aacp_results', mod1, mod_pat, pat, d_str)                                                                                              #Location of the results predictions and blob_results_positioning.csv file
  else:
    indir   = os.path.realpath(os.path.dirname(use_chkpnt))                                                                                                                 #Real path to the checkpoint file 
    outdir  = os.path.realpath(os.path.dirname(use_chkpnt))                                                                                                                 #Location of the checkpoint file to be downloaded to 

    # 3-channel with GLM
    if len(re.split('ir_vis_glm', use_chkpnt)) > 1:
      mod0 = 'IR+VIS+GLM';                   mod1 = 'ir_vis_glm';                   use_night = False
    elif len(re.split('ir_dirtyirdiff_irgradient_tropdiffots', use_chkpnt)) > 1:
      mod0 = 'IR+DIRTYIRDIFF+IRGRADIENT+TROPDIFFOTS'; mod1 = 'ir_dirtyirdiff_irgradient_tropdiffots'
      if use_night == None: use_night = True
    elif len(re.split('vis_tropdiff_glm', use_chkpnt)) > 1:
      mod0 = 'VIS+TROPDIFF+GLM';             mod1 = 'vis_tropdiff_glm';             use_night = False
    # 3-channel without GLM
    elif len(re.split('ir_vis_tropdiff', use_chkpnt)) > 1:
      mod0 = 'IR+VIS+TROPDIFF';              mod1 = 'ir_vis_tropdiff';              use_night = False
    elif len(re.split('ir_vis_dirtyirdiff', use_chkpnt)) > 1:
      mod0 = 'IR+VIS+DIRTYIRDIFF';           mod1 = 'ir_vis_dirtyirdiff';           use_night = False
    elif len(re.split('vis_tropdiff_dirtyirdiff', use_chkpnt)) > 1:
      mod0 = 'VIS+TROPDIFF+DIRTYIRDIFF';     mod1 = 'vis_tropdiff_dirtyirdiff';     use_night = False
    # 2-channel with GLM
    elif len(re.split('vis_glm', use_chkpnt)) > 1:
      mod0 = 'VIS+GLM';                      mod1 = 'vis_glm';                      use_night = False
    elif len(re.split('tropdiff_glm', use_chkpnt)) > 1:
      mod0 = 'TROPDIFF+GLM';                 mod1 = 'tropdiff_glm';
      if use_night == None: use_night = True
    elif len(re.split('ir_glm', use_chkpnt)) > 1:
      mod0 = 'IR+GLM';                       mod1 = 'ir_glm';
      if use_night == None: use_night = True
    # 2-channel without GLM
    elif len(re.split('vis_tropdiff', use_chkpnt)) > 1:
      mod0 = 'VIS+TROPDIFF';                 mod1 = 'vis_tropdiff';                 use_night = False
    elif len(re.split('ir_vis', use_chkpnt)) > 1:
      mod0 = 'IR+VIS';                       mod1 = 'ir_vis';                       use_night = False
    elif len(re.split('tropdiff_dirtyirdiff', use_chkpnt)) > 1:
      mod0 = 'TROPDIFF+DIRTYIRDIFF';         mod1 = 'tropdiff_dirtyirdiff';
      if use_night == None: use_night = True
    elif len(re.split('ir_wvirdiff', use_chkpnt)) > 1:
      mod0 = 'IR+WVIRDiff';                  mod1 = 'ir_wvirdiff';
      if use_night == None: use_night = True
    elif len(re.split('ir_dirtyirdiff', use_chkpnt)) > 1:
      mod0 = 'IR+DIRTYIRDIFF';               mod1 = 'ir_dirtyirdiff';
      if use_night == None: use_night = True
    elif len(re.split('ir_cirrus', use_chkpnt)) > 1:
      mod0 = 'IR+CIRRUS';                    mod1 = 'ir_cirrus';                    use_night = False
    elif len(re.split('ir_snowice', use_chkpnt)) > 1:
      mod0 = 'IR+SNOWICE';                   mod1 = 'ir_snowice';                   use_night = False
    elif len(re.split('ir_tropdiff', use_chkpnt)) > 1:
      mod0 = 'IR+TROPDIFF';                  mod1 = 'ir_tropdiff';
      if use_night == None: use_night = True
    # 1-channel
    elif len(re.split('model_checkpoints' + os.sep + 'tropdiff', use_chkpnt)) > 1:
      mod0 = 'TROPDIFF';                     mod1 = 'tropdiff';
      if use_night == None: use_night = True
    elif len(re.split('model_checkpoints' + os.sep + 'ir', use_chkpnt)) > 1:
      mod0 = 'IR';                           mod1 = 'ir';
      if use_night == None: use_night = True
    else:
      print('Not set up to handle model specified.')
      print(use_chkpnt)
      exit()

    if len(chk_day_night) == 0:
      outdir0 = os.path.join(outroot, 'aacp_results', mod1, mod_pat, pat, d_str)                                                                                            #Location of the results predictions and blob_results_positioning.csv file
    else:
      outdir0 = os.path.join(outroot, 'aacp_results', 'day_night_optimal', mod_pat, pat, d_str)                                                                             #Location of the results predictions and blob_results_positioning.csv file

  # Derive the number of input channels from mod1
  n_channels = mod1.count('_') + 1                                                                                                                                          #Number of underscores + 1 gives number of input channels

  if rt and df.empty:
    return(os.path.join(outdir0, d_str2, str(sector)), pthresh, df)
  
  if len(date_range) > 0 and rt != True and df.empty == False:
    df.reset_index(inplace = True)                                                                                                                                          #Reset the index so it is 0 to number of scans within time range
    df.drop(columns = 'index', inplace = True)                                                                                                                              #Remove index column that was created by reset_index.
    df   = df[(df['date_time'] >= date_range[0]) & (df['date_time'] <= date_range[1])]                                                                                      #Extract list of dates to see which dates are within the specified date range
 
  df.reset_index(inplace = True)                                                                                                                                            #Reset the index so it is 0 to number of scans within time range
  df.drop(columns = 'index', inplace = True)                                                                                                                                #Remove index column that was created by reset_index.

  if by_updraft:
    by_date = True
  if use_chkpnt == None:
    outdir  = os.path.join(indir, d_str)                                                                                                                                    #Set up output root directory path to current date
    if subset:           
      indir  = os.path.join(indir, 'chosen_indices')           
      outdir = os.path.join(outdir, 'chosen_indices')           
    if by_date:           
      indir  = os.path.join(indir, 'by_date')           
      outdir = os.path.join(outdir, 'by_date')           
    if by_updraft:  
      indir  = os.path.join(indir, 'by_updraft')           
      outdir = os.path.join(outdir, 'by_updraft')  
  
  if use_local:  
    if os.path.exists(os.path.join(indir, 'dates_with_max_iou.csv')):  
      df_iou = pd.read_csv(os.path.join(indir, 'dates_with_max_iou.csv'), index_col = 0)                                                                                    #Read specified csv file
    else:  
      df_iou = pd.DataFrame()                                                                                                                                               #Create empty data frame
  else:  
    csv_file2 = os.path.join('Cooney_testing', os.path.relpath(indir, re.split(mod1, indir)[0]), 'dates_with_max_iou.csv')                                                  #Find max_iou csv file. Can be used for model type (Unet or MultiResUnet)
    df_iou    = load_csv_gcs(f_bucket_name, csv_file2)                                                                                                                      #Read specified csv file from GCP
    
  if not df_iou.empty:  
    mod_type = str(df_iou['model'][0])                                                                                                                                      #Extract model type from file
  else:                                                                                                                                                           
    if use_chkpnt == None:
      mod_type = 'unet'                                                                                                                                                     #If file does not exist then the model type was likely just a simple unet model
    else:
      if len(re.split('multiresunet', use_chkpnt)) != 1:  
        mod_type = 'multiresunet'
      elif len(re.split('attentionunet', use_chkpnt)) != 1:  
        mod_type = 'attentionunet'
      elif len(re.split('unet', use_chkpnt)) != 1:  
        mod_type = 'unet'     
      else:
        print('Cannot determine model type (multiresunet, unet, attentionunet) from checkpoint name???')
        print(use_chkpnt)
        exit()
  
  if verbose:
    print('Model type being used = ' + mod_type)
    print('Number of input channels = ' + str(n_channels))
    print(dims)

  # Build the model architecture
  if use_native_ir:
    if mod_type.lower() == 'unet':  
      print('Model not setup to run native IR resolution for unet model')
      exit()
    elif mod_type.lower() == 'multiresunet':  
      if dims[1] == 500 and dims[2] == 500:
        model = MultiResUnet_new_UPsampling(512, 512, n_channels, adam_optimizer = 1e-4)                                                                                    #Run MultiResUnet_new_UPsampling function with N channels
      else:
        model = MultiResUnet_new_UPsampling(512, 512, n_channels)                                                                                                           #Run MultiResUnet_new_UPsampling function with N channels
    else:  
      print('Not set up to do model specified.')  
      print(mod_type)  
      exit()  
  else:  
    if dims[1] <= 2000 and dims[2] <= 2000:
      if mod_type.lower() == 'unet':  
        model = unet(input_size = (dims[1], dims[2], n_channels))                                                                                                           #Run unet function with N channels
      elif mod_type.lower() == 'multiresunet':  
        if len(re.split('tropdiffots', use_chkpnt)) != 1:
            model = MultiResUnet(512, 512, n_channels)                                                                                                                      #Run MultiResUnet function with N channels
        else:
            model = MultiResUnet(dims[1], dims[2], n_channels)                                                                                                                  #Run MultiResUnet function with N channels
#            model = MultiResUnet(128, 128, n_channels)                                                                                                                      #Run MultiResUnet function with N channels
        
      else:  
        print('Not set up to do model specified.')  
        print(mod_type)  
        exit()  
    else:
      model = None 
  
  mod_description = mod0 + ' ' + mod_name + ' ' + mod_type.lower().capitalize()                                                                                             #Create string specifying the model description which is written into the appended combined netCDF file
  
  os.makedirs(outdir, exist_ok = True)                                                                                                                                      #Create output directory if it does not already exist
  if verbose: 
    print('Checkpoint files output directory = ' + outdir)
    print('Loading test model weights')  
  if use_chkpnt == None:  
    if not os.path.isfile(os.path.join(indir, 'unet_checkpoint.cp')):  
      chk_files = list_gcs_chkpnt(f_bucket_name, dirs, delimiter = '*/*/*/*/*/*/')                                                                                          #Search for checkpoint files in all pre-processed directories
      if len(chk_files) <= 0:
        print('No check point files found???')
        print(dirs)
        exit()
      chk_files = chk_files[-3:]                                                                                                                                            #Only retain most recent checkpoint files (there are 3 that are needed)
      [download_model_chk_file(f_bucket_name, c, outdir) for c in chk_files]                                                                                                #Download checkpoint files to local storage in order to load and read
    use_chkpnt = os.path.join(outdir, 'unet_checkpoint.cp')
  else:  
    if not os.path.isfile(use_chkpnt + '.index'):  
      dirs = os.path.join('Cooney_testing', os.path.relpath(indir, re.split(mod1, indir)[0]))  
      chk_files = list_gcs_chkpnt(f_bucket_name, dirs)                                                                                                                      #Search for checkpoint files in all pre-processed directories
      if len(chk_files) <= 0:
        print('No check point files found???')
        print(dirs)
        exit()
      [download_model_chk_file(f_bucket_name, c, outdir) for c in chk_files]                                                                                                #Download checkpoint files to local storage in order to load and read  

  if dims[1] <= 2000 and dims[2] <= 2000:
    model.load_weights(use_chkpnt).expect_partial()                                                                                                                         #Load the weights from previous checkpoint file specified by string

    if del_local:   
      chk_files = sorted(glob.glob(os.path.join(outdir, '*checkpoint*'), recursive = True))  
      [os.remove(c) for c in chk_files]
          
  os.makedirs(os.path.join(outdir0, d_str2, str(sector)), exist_ok = True)                                                                                                  #Create output directory to send model results files, if necessary
  if 'day_night_optimal' not in outdir0:
    pref = os.path.join(os.path.relpath(outdir0, re.split(mod1, outdir0)[0]), d_str2, str(sector))
  else:
    pref = os.path.join(os.path.relpath(outdir0, re.split('day_night_optimal', outdir0)[0]), d_str2, str(sector))
  if rt:
    s_ind = len(df)-1
  else:
    s_ind = 0  
  if use_local:
    p_bucket_name2 = None
  else:
    p_bucket_name2 = p_bucket_name
  counter = 0
  for d in range(s_ind, len(df)):
    d_str0 = datetime.strptime(df['date_time'][d], "%Y-%m-%d %H:%M:%S").strftime("%Y%j%H%M%S")
    ncf    = os.path.realpath(df['glm_files'][d])                                                                                                                           #Extract name of combined netCDF so that it can be appended with model results
    exist  = 0
    if not rt and not rewrite_model:
      fname = os.path.join(outdir0, d_str2, str(sector), d_str0 + '_test_' + str.format('{0:.0f}', dims[1]) + '_results.npy')                                               #Save file path and name in case want to write results to google cloud storage bucket
      if use_local:
        if os.path.exists(fname):                                                                                                                                           #Check if results file exist in google cloud storage bucket
          exist = 1
      else:
        exist = list_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), [os.path.basename(fname)], delimiter = '/')                                                   #Check if results file exist in google cloud storage bucket
        if len(exist) > 1:
          print('Multiple model files found in GCP bucket matching request???')
          print(exist)
          print(f_bucket_name)
          print(os.path.join('Cooney_testing', pref))
          exit()
        if len(exist) > 0:
          if not del_local:
            download_ncdf_gcs(f_bucket_name, exist[0], os.path.join(outdir0, d_str2, str(sector)))
          exist = 1
        else:
          exist = 0

    if exist == 0:    
      # Build the input tensor for N channels using ch_map lookup.
      ch_map = {                                                                                                                                                             #Maps mod1 -> list of (index_col, subdir, file_suffix) tuples
        'ir_vis_glm'                            : [('ir_index',  'ir',          '_ir.npy'),
                                                   ('vis_index', 'vis',         '_vis.npy'),
                                                   ('glm_index', 'glm',         '_glm.npy')],
#         'ir_dirtyirdiff_irgradient_tropdiffots' : [('ir_index',  'ir',           '_ir.npy'),
#                                                    ('ir_index',  'dirtyirdiff',  '_dirtyirdiff.npy'),
#                                                    ('ir_index',  'ir_gradient',  '_ir_gradient.npy'),
#                                                    ('ir_index',  'tropdiffots',  '_tropdiffots_results.npy')],
        'ir_dirtyirdiff_irgradient_tropdiffots' : [('ir_index',  'ir',           '_ir.npy'),
                                                   ('ir_index',  'dirtyirdiff',  '_dirtyirdiff.npy'),
                                                   ('ir_index',  'tropdiffots',  '_tropdiffots_results.npy'),
                                                   ('ir_index',  'ir_gradient',  '_ir_gradient.npy')],
        'vis_tropdiff_glm'                      : [('vis_index', 'vis',         '_vis.npy'),
                                                   ('ir_index',  'tropdiff',    '_tropdiff.npy'),
                                                   ('glm_index', 'glm',         '_glm.npy')],
        'ir_vis_tropdiff'                       : [('ir_index',  'ir',          '_ir.npy'),
                                                   ('vis_index', 'vis',         '_vis.npy'),
                                                   ('ir_index',  'tropdiff',    '_tropdiff.npy')],
        'ir_vis_dirtyirdiff'                    : [('ir_index',  'ir',          '_ir.npy'),
                                                   ('vis_index', 'vis',         '_vis.npy'),
                                                   ('ir_index',  'dirtyirdiff', '_dirtyirdiff.npy')],
        'vis_tropdiff_dirtyirdiff'              : [('vis_index', 'vis',         '_vis.npy'),
                                                   ('ir_index',  'tropdiff',    '_tropdiff.npy'),
                                                   ('ir_index',  'dirtyirdiff', '_dirtyirdiff.npy')],
        'vis_glm'                               : [('vis_index', 'vis',         '_vis.npy'),
                                                   ('glm_index', 'glm',         '_glm.npy')],
        'tropdiff_glm'                          : [('ir_index',  'tropdiff',    '_tropdiff.npy'),
                                                   ('glm_index', 'glm',         '_glm.npy')],
        'ir_glm'                                : [('ir_index',  'ir',          '_ir.npy'),
                                                   ('glm_index', 'glm',         '_glm.npy')],
        'vis_tropdiff'                          : [('vis_index', 'vis',         '_vis.npy'),
                                                   ('ir_index',  'tropdiff',    '_tropdiff.npy')],
        'ir_vis'                                : [('ir_index',  'ir',          '_ir.npy'),
                                                   ('vis_index', 'vis',         '_vis.npy')],
        'tropdiff_dirtyirdiff'                  : [('ir_index',  'tropdiff',    '_tropdiff.npy'),
                                                   ('ir_index',  'dirtyirdiff', '_dirtyirdiff.npy')],
        'ir_wvirdiff'                           : [('ir_index',  'ir',          '_ir.npy'),
                                                   ('ir_index',  'ir_diff',     '_ir_diff.npy')],
        'ir_dirtyirdiff'                        : [('ir_index',  'ir',          '_ir.npy'),
                                                   ('ir_index',  'dirtyirdiff', '_dirtyirdiff.npy')],
        'ir_cirrus'                             : [('ir_index',  'ir',          '_ir.npy'),
                                                   ('vis_index', 'cirrus',      '_cirrus.npy')],
        'ir_snowice'                            : [('ir_index',  'ir',          '_ir.npy'),
                                                   ('vis_index', 'snowice',     '_snowice.npy')],
        'ir_tropdiff'                           : [('ir_index',  'ir',          '_ir.npy'),
                                                   ('ir_index',  'tropdiff',    '_tropdiff.npy')],
        'tropdiff'                              : [('ir_index',  'tropdiff',    '_tropdiff.npy')],
        'ir'                                    : [('ir_index',  'ir',          '_ir.npy')],
      }

      if mod1 not in ch_map:
        print('Not set up to handle model inputs in ch_map:')
        print(mod1)
        exit()

      ch_list  = ch_map[mod1]
      use_tropdiffots = (len(re.split('tropdiffots', mod1)) != 1)

      if use_tropdiffots:
        # Separate the tropdiffots entry from the regular channel entries.
        regular_ch_list = [c for c in ch_list if c[1] != 'tropdiffots']
        tropdiffots_idx = next(i for i, c in enumerate(ch_list) if c[1] == 'tropdiffots')                                                                                    #Channel index position of tropdiffots in imgs

        npy_args = [(idx_col, os.path.join(dir_path, subdir, d_str0 + suffix)) for idx_col, subdir, suffix in regular_ch_list]
        imgs = build_imgs(df.loc[d].to_frame().T, npy_args,
                          dims        = [dims[1], dims[2]],
                          bucket_name = p_bucket_name2)

        # Expand imgs to accommodate the extra tropdiffots channel.
        imgs_full = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2], len(ch_list)), dtype = np.float32)
        reg_idx = 0
        for ch_idx, ch_entry in enumerate(ch_list):
            if ch_entry[1] != 'tropdiffots':
                imgs_full[:, :, :, ch_idx] = imgs[:, :, :, reg_idx]
                reg_idx += 1
        imgs = imgs_full

        # Find and load the OT results file for this scan timestamp.
        ot_res_dir  = os.path.join(outroot, 'aacp_results', 'day_night_optimal', 'updraft_day_model', pat, d_str, d_str2, str(sector))
        ot_res_glob = sorted(glob.glob(os.path.join(ot_res_dir, d_str0 + '_test_*_results.npy')))

        if len(ot_res_glob) == 0:
          print('ERROR: No OT results file found for timestamp ' + d_str0 + ' in:')
          print('       ' + ot_res_dir)
          print('       The TROPDIFFOTS channel requires a completed OT prediction for')
          print('       each scan before the AACP model can run. Exiting.')
          exit()
        if len(ot_res_glob) > 1:
          print('WARNING: Multiple OT results files found for timestamp ' + d_str0 + ':')
          print('         ' + str(ot_res_glob))
          print('         Using the most recently modified file.')
          ot_res_glob = [max(ot_res_glob, key = os.path.getmtime)]

        ot_res_path = ot_res_glob[0]
#         if verbose:
#           print('Loading OT results file for TROPDIFFOTS channel: ' + ot_res_path)

        ot_results = np.load(ot_res_path)

        # Normalise to [1, ny, nx, 1] regardless of how it was saved.
        if ot_results.ndim == 4:
          ot_prob = ot_results[0, :, :, 0]
        elif ot_results.ndim == 3:
          ot_prob = ot_results[:, :, 0]
        elif ot_results.ndim == 2:
          ot_prob = ot_results
        else:
          print('ERROR: Unexpected OT results array shape: ' + str(ot_results.shape))
          exit()

        # If OT output spatial dims differ from AACP dims, resize to match.
        if ot_prob.shape[0] != dims[1] or ot_prob.shape[1] != dims[2]:
          print('this should never happen')
          exit()
#           from scipy.ndimage import zoom as _zoom
#           zy = dims[1] / ot_prob.shape[0]
#           zx = dims[2] / ot_prob.shape[1]
#           ot_prob = _zoom(ot_prob, (zy, zx), order = 1).astype(np.float32)  # bilinear resize

        imgs[0, :, :, tropdiffots_idx] = ot_prob                                                                                                                             #Inject OT probability map as channel in tropdiffots_idx
      else:
        # Standard path: all channels come from labelled numpy files.
        npy_args = [(idx_col, os.path.join(dir_path, subdir, d_str0 + suffix)) for idx_col, subdir, suffix in ch_list]
        imgs = build_imgs(df.loc[d].to_frame().T, npy_args,
                          dims        = [dims[1], dims[2]],
                          bucket_name = p_bucket_name2)

      # NaN masking
      if mod1 == 'vis_tropdiff':
        na  = (imgs[:, :, :, 1] < 0)
        na2 = (imgs[:, :, :, 1] <= 0)
        na3 = (imgs[:, :, :, 1] == -2)
      else:
        na  = (imgs[:, :, :, 0] < 0)                                                                                                                                        #Find instances in which the IR data are NaNs so they can be removed from results
        na2 = (imgs[:, :, :, 0] <= 0)
        na3 = (imgs[:, :, :, 0] == -2)
      if sector.lower() == 'c' or sector.lower() == 'f':
        imgs[na, :] = val_set                                                                                                                                               #Set NaN regions to val_set so that edges of domain are not mistakenly identified as an OT or AACP
      else:
        imgs[na, 0] = 0                                                                                                                                                     #Set NaN regions to 0 so that edges of domain are not mistakenly identified as an OT or AACP

      # Tiling / batching logic
      tropdiffots_patched = False                                                                                                                                           #Flag: True when predict_large_image_patches is used for tropdiffots; skips normal dataset/model.predict path and reconstruction
      if len(re.split('tropdiffots', use_chkpnt)) != 1:
        patch_sz = 512                                                                                                                                                      #Tropdiffots AACP model was trained on 256x256 boxes
      elif 'plume_day_model' in use_chkpnt:
        patch_sz = 2000
      else:
        patch_sz = 512
      if use_native_ir:
        if dims[1] == 500 and dims[2] == 500:
          imgs2 = np.zeros((imgs.shape[0], 512, 512, imgs.shape[-1]))
          imgs2[:, 5:-7, 5:-7, :] = imgs
          dataset = tf.data.Dataset.from_tensor_slices(imgs2).batch(1)                                                                                                      #Creates a Dataset whose elements are slices of the given tensors
          dataset.cache
          dataset.prefetch(1)
          results = model.predict(dataset)
          results = results[:, 5:-7, 5:-7, :]
        else:
          imgs, chunks = build_subset_tensor3(imgs, 512, 512)                                                                                                               #Create subset tensor so it does not overwhelm GPU memory
          if dims[1] > 2000 or dims[2] > 2000:
            model.load_weights(use_chkpnt).expect_partial()                                                                                                                 #Load the weights from previous checkpoint file specified by string
          mnval, mxval = np.min(imgs[:, :, :, 0], axis = (1, 2)), np.max(imgs[:, :, :, 0], axis = (1, 2))
          rem    = np.where((mxval != 0) & (mnval != 1))
          imgs2  = imgs[rem[0], :, :, :] if len(rem[0]) > 0 else imgs
          dataset = tf.data.Dataset.from_tensor_slices(imgs2).batch(16)                                                                                                     #Creates a Dataset whose elements are slices of the given tensors
          dataset.cache
          dataset.prefetch(1)
          results = model.predict(dataset)
          if len(rem[0]) > 0:
            results2 = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2], results.shape[3]))
            results2[rem[0], :, :, :] = results
            results = results2
          results = reconstruct_tensor_from_subset3(results, dims[1], dims[2], chunks)
      else:
        if dims[1] > 2000 or dims[2] > 2000:
          #Tile the full image into patch_sz x patch_sz blocks
          imgs_tiled, chunks = build_subset_tensor3(imgs, patch_sz, patch_sz)
          if model is None:
            if mod_type.lower() == 'unet':
              model = unet(input_size=(patch_sz, patch_sz, n_channels))
            elif mod_type.lower() == 'multiresunet':
              model = MultiResUnet(patch_sz, patch_sz, n_channels)
            else:
              print('Not set up to do model specified.')
              print(mod_type)
              exit()
            model.load_weights(use_chkpnt).expect_partial()

          # Skip all-zero (empty) tiles
          mnval, mxval = np.min(imgs_tiled[:, :, :, 0], axis=(1, 2)), np.max(imgs_tiled[:, :, :, 0], axis=(1, 2))
          rem    = np.where((mxval != 0) & (mnval != 1))
          imgs2  = imgs_tiled[rem[0], :, :, :] if len(rem[0]) > 0 else imgs_tiled

          # Predict on non-empty tiles using patch-based overlap-averaging
#          batch_size = 10 if 'plume_day_model' in (use_chkpnt if use_chkpnt is not None else '') else 16
          batch_size = 16
          results_tiles = predict_large_image_patches(model, imgs2,
                                                      patch_size = patch_sz,
                                                      overlap    = 64,
                                                      batch_size = batch_size,
                                                      verbose    = verbose)

          # Re-insert non-empty results back into full tile array
          if len(rem[0]) > 0:
            results2 = np.zeros((imgs_tiled.shape[0], imgs_tiled.shape[1], imgs_tiled.shape[2], results_tiles.shape[3]))
            results2[rem[0], :, :, :] = results_tiles
            results_tiles = results2

          # Reconstruct full spatial domain from tiles
          results = reconstruct_tensor_from_subset3(results_tiles, dims[1], dims[2], chunks)

        else:
          #dims <= 2000
          if 'updraft_day_model' in (use_chkpnt if use_chkpnt is not None else ''):
            dataset = tf.data.Dataset.from_tensor_slices((imgs)).batch(1)                                                                                                   #Creates a Dataset whose elements are slices of the given tensors
            dataset.cache                                                                                                                                                   #Cache the elements in the dataset
            dataset.prefetch(1)                                                                                                                                             #Creates a Dataset that prefetches elements from this dataset
            results = model.predict(dataset)
          else:
            if len(re.split('tropdiffots', use_chkpnt)) == 1:
              dataset = tf.data.Dataset.from_tensor_slices((imgs)).batch(1)                                                                                                 #Creates a Dataset whose elements are slices of the given tensors
              dataset.cache                                                                                                                                                 #Cache the elements in the dataset
              dataset.prefetch(1)                                                                                                                                           #Creates a Dataset that prefetches elements from this dataset
              results = model.predict(dataset)
            else:
              results = predict_large_image_patches(model, imgs,
                                                    patch_size = patch_sz,
                                                    overlap    = 64,
                                                    batch_size = 16,
                                                    verbose    = verbose)
#       if full_pred:
#         # Edge zeroing
#         if mod_type.lower() == 'multiresunet':
#           if np.max(results[0, 64-8:64+8, results.shape[2]-64-8:results.shape[2]-64+8, :]) > 0.05:
#             results[:, 64-8:64+8, results.shape[2]-64-8:results.shape[2]-64+8, :] = 0.0
#           if np.max(results[0, 64-8:64+8, 64-8:64+8, :]) > 0.05:
#             results[:, 64-8:64+8, 64-8:64+8, :] = 0.0
#           if np.max(results[0, results.shape[1]-64-8:results.shape[1]-64+8, results.shape[2]-64-8:results.shape[2]-64+8, :]) > 0.05:
#             results[:, results.shape[1]-64-8:results.shape[1]-64+8, results.shape[2]-64-8:results.shape[2]-64+8, :] = 0.0
# #           if np.max(results[0, results.shape[1]-64-8:results.shape[1]-64+8,64-8:64+8, :]) > 0.05:
# #             results[:, results.shape[1]-64-8:results.shape[1]-64+8,64-8:64+8, :] = 0.0
#           if mod1 == 'tropdiff':                                                                                                                                            #Additional edge zeroing for 1-channel tropdiff model
#             results[:, 64-8:64+8, 64-4:64+8, :] = 0.0  
#           if mod1 == 'vis_tropdiff':                                                                                                                                        #Additional edge zeroing for vis_tropdiff model
#             results[:, 64-8:64+8, :, :] = 0.0  
#             results[:, :, results.shape[1]-64-8:results.shape[1]-64+8, :] = 0.0  

      fname = os.path.join(outdir0, d_str2, str(sector), d_str0 + '_test_' + str.format('{0:.0f}', results.shape[1]) + '_results.npy')                                      #Save file path and name in case want to write results to google cloud storage bucket
#       results4 = np.copy(results)
      results[results < 0.02] = 0
#      results[results < 0.05] = 0
#       results4 = np.copy(results)
      
#       if not no_write_npy:
#         np.save(fname, results4)                                                                                                                                            #Save model results to numpy file in local storage
#         if verbose:
#           print('Prediction file output file path/name = ' + fname)
#         if run_gcs:  
#           if rt:
#             write_to_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), fname, del_local = False)
#           else:  
#             t = Thread(target = write_to_gcs, args = (f_bucket_name, os.path.join('Cooney_testing', pref), fname), kwargs = {'del_local' : del_local})                      #Write results to Google Cloud Storage Bucket
#             t.start()

      # NaN / edge masking of results
      if np.sum(na3) > 0:
        results0 = np.copy(results) 
      if np.sum(na) > 0:
        results[na, 0] = 0                                                                                                                                                  #Set all model predictions scores to 0 in regions that there was no valid satellite data
        r0 = results[0, :, :, 0]
        r0[np.where(np.min(np.asarray([r0, np.roll(r0, 1, axis = 0), np.roll(r0, -1, axis = 0), np.roll(r0, 1, axis = 1), np.roll(r0, -1, axis = 1), 
                                       np.roll(r0, (1, 1), axis = (0, 1)), np.roll(r0, (1, 1), axis = (1, 0)), np.roll(r0, (-1, -1), axis = (0, 1)), np.roll(r0, (-1, -1), axis = (1, 0))]), axis = 0) == 0)] = 0 #Set likelihood scores near to edges of domain to 0s.
        results[0, :, :, 0] = r0
      if np.sum(na2) > 0:
        results[na2, 0] = 0
      if np.sum(na3) > 0:
        results[na3, 0] = results0[na3, 0]
        results0 = None

      binary_mask = (results[0, :, :, 0] > 0).astype(int)
      labeled_array, num_features = label(binary_mask)
      
      slices = find_objects(labeled_array)
      
      for i, slc in enumerate(slices):
          if slc is None: 
              continue
              
          # Isolate the exact bounding box for this blob
          local_label = labeled_array[slc]
          local_mask = (local_label == (i + 1))
          
          # Calculate the size (number of pixels)
          size = np.sum(local_mask)
          if size >= 50:
              continue  # Skip large, valid detections instantly
              
          # Extract probabilities just for this tiny blob
          local_probs = results[0, slc[0], slc[1], 0] * local_mask
          
          # Find the peak probability coordinates within this small box
          max_y_local, max_x_local = np.unravel_index(np.argmax(local_probs), local_probs.shape)
          
          # Check if the peak is physically sheared against the flat edge
          on_edge_y = (max_y_local == 0) or (max_y_local == local_mask.shape[0] - 1)
          on_edge_x = (max_x_local == 0) or (max_x_local == local_mask.shape[1] - 1)
          
          # IF the peak is jammed against an edge AND the blob is small (< 50 pixels) -> Delete it
          if on_edge_y or on_edge_x:
              # Use the full labeled_array to ensure the original array is modified
              results[0, labeled_array == (i + 1), 0] = 0
      
      if not no_write_npy:
        np.save(fname, results)                                                                                                                                            #Save model results to numpy file in local storage
        if verbose:
          print('Prediction file output file path/name = ' + fname)
        if run_gcs:  
          if rt:
            write_to_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), fname, del_local = False)
          else:  
            t = Thread(target = write_to_gcs, args = (f_bucket_name, os.path.join('Cooney_testing', pref), fname), kwargs = {'del_local' : del_local})                      #Write results to Google Cloud Storage Bucket
            t.start()

      if counter != 0:
        t2.join()
      if len(chk_day_night) != 0:
        dn     = df['day_night'][d]
        if dn.lower() == 'night':
          pthresh     = chk_day_night['night']['pthresh']
          opt_pthresh = pthresh
        elif dn.lower() == 'day':
          pthresh     = chk_day_night['day']['pthresh']
          opt_pthresh = pthresh
        else:
          print('dn variable must be set to day or night if given chk_day_night dictionary.')
          print(dn)
          print(chk_day_night)
          exit()
      t2 = Thread(target = append_combined_ncdf_with_model_results, args = (ncf, results, mod_description), kwargs = {'rt': rt, 'optimal_thresh' : opt_pthresh, 'write_gcs': run_gcs, 'del_local': del_local, 'outroot': None, 'c_bucket_name': c_bucket_name, 'use_chkpnt': use_chkpnt, 'verbose': verbose})     #Write results to combined netCDF file
      t2.start()
      counter = counter + 1

  # Clear memory
  results0 = None
  results4 = None
  results2 = None
  dataset  = None
  imgs     = None
  imgs2    = None
  model    = None
  na       = None
  na2      = None
  r0       = None
  mnval    = None
  mxval    = None
  if counter > 1:
    print('Waiting for output netCDF file to close')
    print(t2)
    t2.join()
  return(os.path.join(outdir0, d_str2, str(sector)), pthresh, df)

def predict_large_image_patches(model, large_images, patch_size=256, overlap=32, batch_size=4, verbose=False):
    """
    Predict on large images by breaking into overlapping patches
    
    Args:
        model: Your trained MultiResUNet model
        large_image: Input image array (height, width, channels) or (batch, height, width, channels)
        patch_size: Size of square patches to use - should match training size
        overlap: Overlap between patches in pixels
        batch_size: Batch size for prediction to maximize GPU utilization
        verbose: Whether to print progress information
    
    Returns:
        Prediction array for the large image with same spatial dimensions
    """    
    # Get dimensions
    num_images, lat_dim, lon_dim, num_channels = large_images.shape
    
#     if verbose:
#         print(f"Processing {num_images} images of size {lat_dim}x{lon_dim} with {num_channels} channels")
    
    # Initialize results array
    all_results = []
    
    # Process each image separately
    for img_idx in range(num_images):
#         if verbose:
#             print(f"Processing image {img_idx+1}/{num_images}")
        
        # Get single image: (latitude, longitude, num_channels)
        single_image = large_images[img_idx]
        h, w = lat_dim, lon_dim
        
        # Check if we need patches
        if h <= patch_size and w <= patch_size:
            # Image is small enough for direct prediction
            pred = model.predict(single_image[np.newaxis, ...], verbose=0)[0]
            all_results.append(pred)
            continue
        
        # Effective stride between patch starts
        stride = patch_size - overlap
        
        # Calculate number of patches needed
        n_patches_h = max(1, math.ceil((h - patch_size) / stride) + 1)
        n_patches_w = max(1, math.ceil((w - patch_size) / stride) + 1)
        total_patches = n_patches_h * n_patches_w
        
#         if verbose:
#             print(f"  Breaking into {total_patches} patches ({n_patches_h}x{n_patches_w}) of size {patch_size}x{patch_size}")
        
        # Create output array for this image
        prediction = np.zeros((h, w, 1))
        weight_map = np.zeros((h, w))
        
        # Prepare batch processing
        patches = []
        patch_coords = []
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch coordinates (ensuring we don't go out of bounds)
                start_lat = min(h - patch_size, max(0, i * stride))
                start_lon = min(w - patch_size, max(0, j * stride))
                end_lat = start_lat + patch_size
                end_lon = start_lon + patch_size
                
                # Extract patch: (patch_size, patch_size, num_channels)
                patch = single_image[start_lat:end_lat, start_lon:end_lon, :]
                
                # Handle edge case where patch might be smaller than patch_size
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    # Create a properly sized patch and fill with available data
                    padded_patch = np.zeros((patch_size, patch_size, num_channels))
                    padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = padded_patch
                
                # Add to batch
                patches.append(patch)
                patch_coords.append((start_lat, start_lon, end_lat, end_lon, patch.shape[0], patch.shape[1]))
                
                # Process batch when full or at the end
                if len(patches) == batch_size or (i == n_patches_h-1 and j == n_patches_w-1):
                    # Convert batch to numpy array: (batch_size, patch_size, patch_size, num_channels)
                    batch_array = np.array(patches)
                    
                    # Predict: returns (batch_size, patch_size, patch_size, 1)
                    batch_preds = model.predict(batch_array, verbose=0)
                    
#                     margin = 8
                    # Add each prediction to the output
                    for k, (start_lat, start_lon, end_lat, end_lon, orig_h, orig_w) in enumerate(patch_coords):
                        pred = batch_preds[k]
                        
#                         ps   = pred.shape[0]                                                # patch_size
#                         pred[64-margin:64+margin,    64-margin:64+margin,    :] = 0         # top-left corner
#                         pred[64-margin:64+margin,    ps-64-margin:ps-64+margin, :] = 0      # top-right corner
#                         pred[ps-64-margin:ps-64+margin, ps-64-margin:ps-64+margin, :] = 0   # bottom-right corner
#                         pred[ps-64-margin:ps-64+margin, 64-margin:64+margin,    :] = 0      # bottom-left corner
                        
                        # Only use the valid portion of the prediction (in case of padding)
                        pred_h = min(pred.shape[0], orig_h, end_lat - start_lat)
                        pred_w = min(pred.shape[1], orig_w, end_lon - start_lon)
                        
                        # Add to output with proper weighting for overlap
                        prediction[start_lat:start_lat+pred_h, start_lon:start_lon+pred_w, :] += pred[:pred_h, :pred_w, :]
                        weight_map[start_lat:start_lat+pred_h, start_lon:start_lon+pred_w] += 1
                    
                    # Clear batches
                    del batch_array
                    del batch_preds
                    batch_array  = []
                    batch_preds  = []
                    patches      = []
                    patch_coords = []
        
        # Average overlapping predictions
        prediction = prediction / np.maximum(weight_map[..., np.newaxis], 1)
        all_results.append(prediction)
    
    # Stack all results: (num_images, latitude, longitude, 1)
    return(np.array(all_results))

def main():
  run_tf_N_channel_plume_updraft_day_predict()
    
if __name__ == '__main__':
  main()