#+
# Name:
#     run_write_plot_model_result_predictions.py
# Purpose:
#     This is a script to run plotting routines over the model predicted results. 
# Calling sequence:
#     import run_write_plot_model_result_predictions
#     run_write_plot_model_result_predictions.run_write_plot_model_result_predictions()
# Input:
#     None.
# Functions:
#     run_write_plot_model_result_predictions : Main function to get tensor flow of IR/VIS channels for plumes and updrafts
#     write_plot_model_result_predictions     : Function writes and plots model reesults to give to forecasters
#     img_from_three_modalities2              : Creates individual scene maps of model results on top of IR/VIS sandwich images
#     visualize_time_aggregated_results       : Creates time aggregated maps of results and satellite data
# Output:
#     Maps of model results with satellite imagery
# Keywords:
#     inroot         : STRING specifying input root directory to the pre-processed test, train, val files
#                      DEFAULT = '../../../goes-data/aacp_results/ir_vis_glm/updraft_day_model/not_real_time/2022-02-18/multiresunet/'
#     outroot        : STRING output directory path for results data storage
#                      DEFAULT = None -> figures directory of inroot
#     region         : State or sector of US to plot data around. 
#                      DEFAULT = None -> plot for full domain.
#     latlon_domain  : [x0, y0, x1, y1] array in latitude and longitude coordinates to give subset of image domain desired to be plotted. Array must contain lat/lon values
#                      DEFAULT is to plot the entire domain. 
#                      NOTE: If region keyword is set, it supersedes this keyword
#     pthresh        : FLOAT keyword to specify probability threshold to use. 
#                      DEFAULT = None -> Use maximized IoU probability threshold.
#     use_local      : IF keyword set (True), use local files. If False, use files located on the google cloud. 
#                      DEFAULT = False -> use files on google cloud server. 
#     run_gcs        : IF keyword set (True), write the output files to the google cloud in addition to local storage.
#                      DEFAULT = True.
#     del_local      : IF keyword set (True) AND run_gcs = True, delete local copy of output file.
#                      DEFAULT = True.
#     o_bucket_name  : STRING specifying the name of the gcp bucket to write downloaded files to. run_gcs needs to be True in order for this to matter.
#                      DEFAULT = 'goes-data'
#     c_bucket_name  : STRING specifying the name of the gcp bucket to write combined IR, VIS, GLM files to As well as 3 modalities figures.
#                      run_gcs needs to be True in order for this to matter.
#                      DEFAULT = 'ir-vis-sandwhich'
#     p_bucket_name  : STRING specifying the name of the gcp bucket to write IR/VIS/GLM numpy files to. run_gcs needs to be True in order for this to matter.
#                      DEFAULT = 'aacp-proc-data'
#     f_bucket_name  : STRING specifying the name of the gcp bucket to write model prediction results files to. run_gcs needs to be True in order for this to matter.
#                      DEFAULT = 'aacp-results'
#     real_time      : IF keyword set (True), run the code in real time by only grabbing the most recent file to image and write.
#                      DEFAULT = False
#     by_date        : IF keyword set (True), use model training, validation, and testing numpy output files for separate dates.
#                      DEFAULT = False. If False, use first 75% of data for training, next 20% for validation, and last 5% for testing.
#     by_updraft     : IF keyword set (True), create model training, validation, and testing numpy output files for 64x64 pixels around each updraft by dates.
#                      NOTE: If this keyword is set, by_date will automatically also be set.
#                      DEFAULT = False. If False, use by_date 512x512 or first 75% of data for training, next 20% for validation, and last 5% for testing.
#     subset         : IF keyword set (True), use numpy files in which subset indices [x0, y0, x1, y1] were given for the model training, testing, and validation.
#                      DEFAULT = False. False implies using the 512x512 boxes created for the full domain
#     time_ag_only   : IF keyword set (True), only plot the time aggregated maps.
#                      DEFAULT = False -> plotting the individual maps as well as time aggregated maps. 
#     no_night       : IF keyword set (True), only plot day time data figures.
#                      DEFAULT = True.
#     no_plot_vis    : IF keyword set, do not plot the VIS data because it was not written to combined netCDF files. Setting this keyword = True makes you plot only IR data.
#                      DEFAULT is to set this keyword to False and plot the VIS and IR data.
#     ir_max         : FLOAT keyword specifying the maximum IR BT in color range of map (K). DEFAULT = 230 K
#     ir_min         : FLOAT keyword specifying the minimum IR BT in color range of map (K). DEFAULT = 180 K
#     verbose        : BOOL keyword to specify whether or not to print verbose informational messages.
#                      DEFAULT = True which implies to print verbose informational messages
# Author and history:
#     John W. Cooney           2021-07-22.
#
#-

#### Environment Setup ###
# Package imports
import numpy as np
from datetime import datetime
import os
import re
import glob
import pandas as pd
from scipy.ndimage import label, generate_binary_structure
import multiprocessing as mp
import metpy
import xarray
import sys 
#sys.path.insert(1, '../')
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.getcwd()))
from new_model.gcs_processing import write_to_gcs, list_gcs, list_csv_gcs, load_csv_gcs, load_npy_blobs, load_npy_gcs, download_gcp_parallel, download_ncdf_gcs
#from glm_gridder.run_create_image_from_three_modalities import *
#from visualize_results.visualize_time_aggregated_results import visualize_time_aggregated_results

def run_write_plot_model_result_predictions(inroot         = os.path.join('..', '..', '..', 'goes-data', 'aacp_results', 'ir_vis_glm', 'updraft_day_model', '2022-02-18', 'multiresunet'), 
                                            outroot        = None, 
                                            region         = None, 
                                            latlon_domain  = [], 
                                            pthresh        = None,
                                            chk_day_night  = {}, 
                                            use_local      = False, 
                                            run_gcs        = True, 
                                            del_local      = True, 
                                            o_bucket_name  = 'goes-data',
                                            c_bucket_name  = 'ir-vis-sandwhich',
                                            p_bucket_name  = 'aacp-proc-data',
                                            f_bucket_name  = 'aacp-results', 
                                            real_time      = False, 
                                            by_date        = False,  by_updraft = False, subset = False, 
                                            time_ag_only   = False, 
                                            no_night       = True,
                                            no_plot_vis    = False, 
                                            ir_max         = 230.0, ir_min = 180.0,
                                            verbose        = True):
  '''
  Name:
      run_write_plot_model_result_predictions.py
  Purpose:
      This is a script to run plotting routines over the model predicted results. 
  Calling sequence:
      import run_write_plot_model_result_predictions
      run_write_plot_model_result_predictions.run_write_plot_model_result_predictions()
  Args:
      None.
  Functions:
      run_write_plot_model_result_predictions : Main function to get tensor flow of IR/VIS channels for plumes and updrafts
      write_plot_model_result_predictions     : Function writes and plots model reesults to give to forecasters
      img_from_three_modalities2              : Creates individual scene maps of model results on top of IR/VIS sandwich images
      visualize_time_aggregated_results       : Creates time aggregated maps of results and satellite data
  Output:
      Maps of model results with satellite imagery
  Keywords:
      inroot         : STRING specifying input root directory to the pre-processed test, train, val files
                       DEFAULT = '../../../goes-data/aacp_results/ir_vis_glm/updraft_day_model/not_real_time/2022-02-18/multiresunet/'
      outroot        : STRING output directory path for results data storage
                       DEFAULT = None -> figures directory of inroot
      region         : State or sector of US to plot data around. 
                       DEFAULT = None -> plot for full domain.
      latlon_domain  : [x0, y0, x1, y1] array in latitude and longitude coordinates to give subset of image domain desired to be plotted. Array must contain lat/lon values
                       DEFAULT is to plot the entire domain. 
                       NOTE: If region keyword is set, it supersedes this keyword
      pthresh        : FLOAT keyword to specify probability threshold to use. 
                       DEFAULT = None -> Use maximized IoU probability threshold.
      chk_day_night  : Structure containing day and night model data information in order to make seemless transition in predictions of the 2.
                       DEFAULT = {} -> do not use day_night transition                 
      use_local      : IF keyword set (True), use local files. If False, use files located on the google cloud. 
                       DEFAULT = False -> use files on google cloud server. 
      run_gcs        : IF keyword set (True), write the output files to the google cloud in addition to local storage.
                       DEFAULT = True.
      del_local      : IF keyword set (True) AND run_gcs = True, delete local copy of output file.
                       DEFAULT = True.
      o_bucket_name  : STRING specifying the name of the gcp bucket to write downloaded files to. run_gcs needs to be True in order for this to matter.
                       DEFAULT = 'goes-data'
      c_bucket_name  : STRING specifying the name of the gcp bucket to write combined IR, VIS, GLM files to As well as 3 modalities figures.
                       run_gcs needs to be True in order for this to matter.
                       DEFAULT = 'ir-vis-sandwhich'
      p_bucket_name  : STRING specifying the name of the gcp bucket to write IR/VIS/GLM numpy files to. run_gcs needs to be True in order for this to matter.
                       DEFAULT = 'aacp-proc-data'
      f_bucket_name  : STRING specifying the name of the gcp bucket to write model prediction results files to. run_gcs needs to be True in order for this to matter.
                       DEFAULT = 'aacp-results'
      real_time      : IF keyword set (True), run the code in real time by only grabbing the most recent file to image and write.
                       DEFAULT = False
      by_date        : IF keyword set (True), use model training, validation, and testing numpy output files for separate dates.
                       DEFAULT = False. If False, use first 75% of data for training, next 20% for validation, and last 5% for testing.
      by_updraft     : IF keyword set (True), create model training, validation, and testing numpy output files for 64x64 pixels around each updraft by dates.
                       NOTE: If this keyword is set, by_date will automatically also be set.
                       DEFAULT = False. If False, use by_date 512x512 or first 75% of data for training, next 20% for validation, and last 5% for testing.
      subset         : IF keyword set (True), use numpy files in which subset indices [x0, y0, x1, y1] were given for the model training, testing, and validation.
                       DEFAULT = False. False implies using the 512x512 boxes created for the full domain
      time_ag_only   : IF keyword set (True), only plot the time aggregated maps.
                       DEFAULT = False -> plotting the individual maps as well as time aggregated maps. 
      no_night       : IF keyword set (True), only plot day time data figures.
                       DEFAULT = True.
      no_plot_vis    : IF keyword set, do not plot the VIS data because it was not written to combined netCDF files. Setting this keyword = True makes you plot only IR data.
                       DEFAULT is to set this keyword to False and plot the VIS and IR data.
      ir_max         : FLOAT keyword specifying the maximum IR BT in color range of map (K). DEFAULT = 230 K
      ir_min         : FLOAT keyword specifying the minimum IR BT in color range of map (K). DEFAULT = 180 K
      verbose        : BOOL keyword to specify whether or not to print verbose informational messages.
                       DEFAULT = True which implies to print verbose informational messages
  Author and history:
      John W. Cooney           2021-07-22.
  '''  
  if len(re.split('updraft_day_model', inroot)) > 1:                      
    mod00 = ' Updraft Detection'
    chek0 = 'updraft_day_model'
  else:
    mod00 = ' Plume Detection'
    chek0 = 'plume_day_model'
  
  if 'real_time' not in inroot:
    subset     = True
    by_date    = True
    by_updraft = True
    
  use_chan = os.path.basename(os.path.realpath(re.split(chek0, inroot)[0]))
  if len(chk_day_night) == 0:
    if use_chan == 'ir':                      
      mod0 = 'IR'                      
    elif use_chan == 'ir_irdiff':                                                                                                 
      mod0 = 'IR+IR_Diff'                      
    elif use_chan == 'ir_vis':                                                                                                 
      mod0 = 'IR+VIS'                      
    elif use_chan == 'vis_glm':                                                                                                 
      mod0 = 'VIS+GLM'                      
    elif use_chan == 'ir_glm':                                                                                                 
      mod0 = 'IR+GLM'                      
    elif use_chan == 'ir_vis_glm':                                                                                                 
      mod0 = 'IR+VIS+GLM'
    elif use_chan == 'ir_irdiff_vis':                                                                                                 
      mod0 = 'IR+IR_Diff+VIS'                      
    elif use_chan == 'ir_irdiff_vis':                                                                                                 
      mod0 = 'IR+IR_Diff+GLM'                      
    else:
      print('Not set up to handle model specified.')
      exit()
  else:  
    mod0 = 'Day Night Optimal'

  if outroot == None:
    outroot = os.path.join('..', '..', '..', os.path.basename(os.path.realpath(re.split('aacp_results', inroot)[0])),  'aacp_results_imgs', use_chan, os.path.relpath(inroot, re.split(chek0, inroot)[0]))
  
  inroot  = os.path.realpath(inroot)                                                                                                                                            #Create link to real input root directory path so compatible with Mac
  outroot = os.path.realpath(outroot)                                                                                                                                           #Create link to real path so compatible with Mac
  res_dir = inroot                                                                                                                                                              #Set result directory path to input root directory (has the model name in path as well)
  if 'unet' in os.path.basename(inroot).lower():                             
    inroot = os.path.dirname(res_dir)                                                                                                                                           #Change inroot to image and mask files to input root directory without model name in the path.
  
  d_str   = os.path.basename(inroot)                                                                                                                                            #Extract date the model was run/processed on
  if subset == True:    
    inroot = os.path.join(inroot, 'chosen_indices')    
  if by_date == True:    
    inroot = os.path.join(inroot, 'by_date')    
  if by_updraft == True:    
    inroot = os.path.join(inroot, 'by_updraft')    
#Extract model information based on output root directory paths
    
  if verbose == True: print('Output directory of images = ' + outroot)
  if use_local == True:
    csv_file = sorted(glob.glob(os.path.join(inroot, 'dates_with_ftype.csv'), recursive = True))
  else:
    pref     = os.path.join('Cooney_testing', re.split('aacp_results' + os.sep, inroot)[1])
    csv_file = list_csv_gcs(f_bucket_name, pref, 'dates_with_ftype.csv')
    if isinstance(csv_file, list) == False:
      csv_file = []
  if len(csv_file) == 0:
    print('Dates with ftype csv file not found???')
    exit()
  if len(csv_file) > 1:
    print('Multiple dates with ftype csv file not found???')
    exit()
  if use_local == True:
    df_dates = pd.read_csv(csv_file[0])                                                                                                                                         #Extract path to file containing dates used in pre-processing and if they were used for model training, testing, and validation
  else:    
    df_dates = load_csv_gcs(f_bucket_name, csv_file[0])                                                                                                                         #Extract path to file containing dates used in pre-processing and if they were used for model training, testing, and validation
  
  img_path = [str(df_dates['out_date'][o]) for o in range(len(df_dates['out_date']))]
  if len(img_path) > 1 and img_path[0] != img_path[-1]:
    img_path = str(df_dates['out_date'][0]) + '-' + str(df_dates['out_date'][len(img_path)-1])                                                                                  #Set up Image output path based on dates
  else:
    img_path = str(df_dates['out_date'][0])                                                                                                                                     #Set up Image output path based on dates

  results  = []
  nc_files = []                                                                                                                                                                 #Initialize list to store the combined netCDF file names
  ir_files = []                                                                                                                                                                 #Initialize list to store the raw file names
  for l in range(len(df_dates['out_date'])):    
    dat_str = str(df_dates['out_date'][l])                                                                                                                                      #Extract date string
    ftype   = str(df_dates['ftype'][l])                                                                                                                                         #Extract if date was used for training, validation, or testing
    sector  = str(df_dates['sector'][l])                                                                                                                                        #Extract if date was used for training, validation, or testing
    pref    = os.path.realpath(re.split(d_str, inroot)[0])
    if 'real_time' in pref:    
      pref = os.path.join(re.split('aacp_results', pref)[0], 'labelled', os.path.basename(pref))
      pati = 'ir'
    else:    
      pref = os.path.join(re.split('aacp_results', pref)[0], 'labelled')      
      pati = ''

    pref2   = re.split('aacp_results', inroot)[0]    
    ir_file = os.path.join(pref, dat_str, sector, pati, 'ir.npy')                                                                                                               #Extract name of IR numpy data file
    ir_files.append(ir_file)    
    if use_local == True:    
      nc_names = sorted(glob.glob(os.path.join(pref2, 'combined_nc_dir', datetime.strptime(dat_str, "%Y%j").strftime("%Y%m%d"), '*' + sector + '_COMBINED_*.nc')))              #Extract name of combined netCDF files
    else:    
      nc_names = list_gcs(c_bucket_name, os.path.join('combined_nc_dir', datetime.strptime(dat_str, "%Y%j").strftime("%Y%m%d")), ['_COMBINED_', sector], delimiter = '/')       #Extract name of combined netCDF files
      
    nc_names = [os.path.join(pref2, x) for x in nc_names]    
        
    if use_local == True:    
      pref      = os.path.join(os.path.basename(os.path.realpath(re.split('aacp_results', inroot)[0])), os.path.relpath(inroot, os.path.realpath(re.split('aacp_results', inroot)[0])))
      night_csv = os.path.join(re.split('aacp_results', inroot)[0], 'labelled', os.path.basename(os.path.realpath(re.split(d_str, pref)[0])), dat_str, sector, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv')   #Setup local storage path of night time csv file
      if os.path.isfile(night_csv):
        df_night = pd.read_csv(night_csv)
      else:
        print('Night time csv file not found??')
        print(night_csv)
        exit()
    else:
      pref = os.path.join(os.path.basename(os.path.realpath(re.split('aacp_results', inroot)[0])), os.path.relpath(inroot, os.path.realpath(re.split('aacp_results', inroot)[0])))
      pref = os.path.join(re.split('aacp_results', pref)[0], 'labelled', os.path.basename(os.path.realpath(re.split(d_str, pref)[0])), dat_str, sector)                         #Setup Google Cloud Platform storage path of night time csv file
      night_csv = list_csv_gcs(p_bucket_name, pref, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv')                                                                    #Read csv file to determine when it is night time. 
      if len(night_csv) == 0:  
        print('Night time csv file not found??')  
        print(pref)  
        exit()  
      df_night = load_csv_gcs(p_bucket_name, night_csv[0])                                                                                                                      #Extract path to file containing dates used in pre-processing
    
    if no_night == True:
      df_night.dropna(subset = ['vis_files'], inplace = True)                                                                                                                   #Remove night time scans
      df_night = df_night.reset_index().drop(columns = ['index'])
    nc_names  = [nc_names[x] for x in df_night['ir_index'].astype(int)]
    counter   = 0
    im_names2 = []
    if use_local == True:    
      res_file = sorted(glob.glob(os.path.join(res_dir, dat_str, sector, '*.npy')))                                                                                             #Extract name of numpy results file
      if len(res_file) == 0:    
        print('No results file found??')    
        res0 = []
      if len(res_file) > 1:    
        fb = [re.split('_s|_', os.path.basename(x))[5][0:-1] for x in nc_names]                                                                                                 #Split file string in order to extract date string of scan
        for idx, n in enumerate(nc_names): 
          date_str = fb[idx]                                                                                                                                                    #Extract date string of scan
          if counter == 0:
            fb2 = [re.split('_', os.path.basename(x))[0] for x in res_file]
            counter = 1
          if date_str in fb2:
            res0 = np.load(res_file[fb2.index(date_str)])                                                                                                                       #Load numpy results file
          else:
            res0 = []
          if len(res0) > 0:
            if len(results) <= 0:
              results = np.copy(res0[0, :, :, 0])
              if df_night['day_night'][idx].lower() == 'day':
                results[results < chk_day_night['day']['pthresh']] = 0.0
              else:
                results[results < chk_day_night['night']['pthresh']] = 0.0
            else:
              res00 = np.copy(res0[0, :, :, 0])
              if df_night['day_night'][idx].lower() == 'day':
                res00[res00 < chk_day_night['day']['pthresh']] = 0.0
              else:
                res00[res00 < chk_day_night['night']['pthresh']] = 0.0

              results[res00 > results] = res00[res00 > results]
              res00 = None  
            
            if time_ag_only == False and len(res0) > 0:
#               ir    = xarray.open_dataset(df_night['ir_files'][idx]).load()
#               dat   = ir.metpy.parse_cf('Rad')
#               proj  = [dat.metpy.cartopy_crs, dat.x.min(), dat.x.max(), dat.y.min(), dat.y.max()]
#               ir.close() 
              im_names, df_out = write_plot_model_result_predictions([n], ir_files, res = res0, no_plot_vis = no_plot_vis, region = region, latlon_domain = latlon_domain, pthresh = pthresh, chk_day_night = chk_day_night, model = mod0 + mod00, outroot = os.path.join(outroot, img_path, sector), write_gcs = run_gcs, del_local = del_local, ir_max = ir_max, ir_min = ir_min, verbose = verbose)
              res0 = None
              im_names2.extend(im_names)
#               if os.path.isfile(os.path.join(inroot, dat_str, sector, 'blob_results_positioning.csv')):
#                 df2    = pd.read_csv(os.path.join(outroot, 'blob_results_positioning.csv'))                                                                                     #Read ftype csv file that already exists
#                 df_out = pd.concat([df_out, df2], axis = 0, join = 'inner', sort = 'date_time')                                                                                 #Concatenate the dataframes and sort by date 
#                 df_out = df_out.astype({'date_time' : str})
#                 df_out.sort_values(by = 'date_time', axis =0, inplace = True)
#                 df_out.drop_duplicates(inplace = True, subset = ['date_time', 'centroid_index_x', 'centroid_index_y'])                                                          #Drop any duplicate OT locations
#                 df_out.reset_index(inplace = True)
#                 header = list(df_out.head(0))                                                                                                                                   #Extract header of csv file
#                 if 'Unnamed: 0' in header:
#                   df_out.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                                                   #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
#                 if 'index' in header:
#                   df_out.drop(columns = 'index', axis = 'columns', inplace = True)                                                                                              #Remove index column that was created by reset_index.
#               else:
#                 if run_gcs == True:
#                   pref     = os.path.relpath(os.path.join(inroot, dat_str, sector), re.split(use_chan, os.path.join(inroot, dat_str, sector))[0])
#                   csv_file = list_csv_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), 'blob_results_positioning.csv')                                                  #Check to see if blob_results_positioning.csv is on GCP so that they can be appended
#                   if len(csv_file) > 0:
#                     df2    = load_csv_gcs(f_bucket_name, csv_file[0])                                                                                                           #Read specified csv file
#                     df_out = pd.concat([df_out, df2], axis = 0, join = 'inner', sort = 'date_time')                                                                             #Concatenate the dataframes and sort by date 
#                     df_out = df_out.astype({'date_time' : str})
#                     df_out.sort_values(by = 'date_time', axis =0, inplace = True)
#                     df_out.drop_duplicates(inplace = True, subset = ['date_time', 'centroid_index_x', 'centroid_index_y'])                                                      #Drop any duplicate OT locations
#                     df_out.reset_index(inplace = True)
#                     header = list(df_out.head(0))                                                                                                                               #Extract header of csv file
#                     if 'Unnamed: 0' in header:
#                       df_out.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                                               #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
#                     if 'index' in header:
#                       df_out.drop(columns = 'index', axis = 'columns', inplace = True)                                                                                          #Remove index column that was created by reset_index.
# #Deprecated
# #               if os.path.isfile(os.path.join(outroot, 'blob_results_positioning.csv')):
# #                 df2 = pd.read_csv(os.path.join(outroot, 'blob_results_positioning.csv'))                                                                                      #Read ftype csv file that already exists
# #                 df2.append(df_out, ignore_index = True)                                                                                                                       #Append previous dates ftype dates csv file with new date
# #                 df2.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                                                      #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
# #                 df_out = df2    
#               
#               df_out.to_csv(os.path.join(outroot, 'blob_results_positioning.csv'))                                                                                              #Write output csv file
      else:
        res0 = np.load(res_file[0])                                                                                                                                             #Load numpy results file
        if time_ag_only == False and len(res0) > 0:
#           ir    = xarray.open_dataset(df_night['ir_files'][0]).load()
#           dat   = ir.metpy.parse_cf('Rad')
#           proj  = [dat.metpy.cartopy_crs, dat.x.min(), dat.x.max(), dat.y.min(), dat.y.max()]
#           ir.close() 
          im_names, df_out = write_plot_model_result_predictions(nc_names, ir_files, res = res0, no_plot_vis = no_plot_vis, region = region, latlon_domain = latlon_domain, pthresh = pthresh, chk_day_night = chk_day_night, model = mod0 + mod00, outroot = os.path.join(outroot, img_path, sector), write_gcs = run_gcs, del_local = del_local, ir_max = ir_max, ir_min = ir_min, verbose = verbose)
          im_names2.extend(im_names)

#           if os.path.isfile(os.path.join(inroot, dat_str, sector, 'blob_results_positioning.csv')):
#             df2    = pd.read_csv(os.path.join(inroot, dat_str, sector, 'blob_results_positioning.csv'))                                                                         #Read ftype csv file that already exists
#             df_out = pd.concat([df_out, df2], axis = 0, join = 'inner', sort = 'date_time')                                                                                     #Concatenate the dataframes and sort by date 
#             df_out = df_out.astype({'date_time' : str})
#             df_out.sort_values(by = 'date_time', axis =0, inplace = True)
#             df_out.drop_duplicates(inplace = True, subset = ['date_time', 'centroid_index_x', 'centroid_index_y'])                                                              #Drop any duplicate OT locations
#             df_out.reset_index(inplace = True)
#             header = list(df_out.head(0))                                                                                                                                       #Extract header of csv file
#             if 'Unnamed: 0' in header:
#               df_out.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                                                       #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
#             if 'index' in header:
#               df_out.drop(columns = 'index', axis = 'columns', inplace = True)                                                                                                  #Remove index column that was created by reset_index.
#           else:
#             if run_gcs == True:
#               pref     = os.path.relpath(os.path.join(inroot, dat_str, sector), re.split(use_chan, os.path.join(inroot, dat_str, sector))[0])
#               csv_file = list_csv_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), 'blob_results_positioning.csv')                                                      #Check to see if blob_results_positioning.csv is on GCP so that they can be appended
#               if len(csv_file) > 0:
#                 df2    = load_csv_gcs(f_bucket_name, csv_file[0])                                                                                                               #Read specified csv file
#                 df_out = pd.concat([df_out, df2], axis = 0, join = 'inner', sort = 'date_time')                                                                                 #Concatenate the dataframes and sort by date 
#                 df_out = df_out.astype({'date_time' : str})
#                 df_out.sort_values(by = 'date_time', axis =0, inplace = True)
#                 df_out.drop_duplicates(inplace = True, subset = ['date_time', 'centroid_index_x', 'centroid_index_y'])                                                          #Drop any duplicate OT locations
#                 df_out.reset_index(inplace = True)
#                 header = list(df_out.head(0))                                                                                                                                   #Extract header of csv file
#                 if 'Unnamed: 0' in header:
#                   df_out.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                                                   #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
#                 if 'index' in header:
#                   df_out.drop(columns = 'index', axis = 'columns', inplace = True)                                                                                              #Remove index column that was created by reset_index.
# #deprecated                   
# #           if os.path.isfile(os.path.join(outroot, 'blob_results_positioning.csv')):
# #             df2 = pd.read_csv(os.path.join(outroot, 'blob_results_positioning.csv'))                                                                                          #Read ftype csv file that already exists
# #             df2.append(df_out, ignore_index = True)                                                                                                                           #Append previous dates ftype dates csv file with new date
# #             df2.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                                                          #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
# #             df_out = df2    
#           
#           df_out.to_csv(os.path.join(outroot, 'blob_results_positioning.csv'))                                                                                                  #Write output csv file
    else:    
      pref     = os.path.join('Cooney_testing', os.path.relpath(os.path.join(inroot, dat_str, sector), re.split(use_chan, os.path.join(inroot, dat_str, sector))[0]))   
      res_file = list_gcs(f_bucket_name, pref, ['results', '.npy'], delimiter = '/')                                                                                            #Search for model results files on Google Cloud Platform
      if len(res_file) == 0:    
        print('No results file found??') 
        print(pref)
        print(f_bucket_name)
        res0 = []
      if len(res_file) > 1:    
        fb = [re.split('_s|_', os.path.basename(x))[5][0:-1] for x in nc_names]
        for idx, n in enumerate(nc_names): 
          date_str = fb[idx]                                                                                                                                                    #Split file string in order to extract date string of scan
          if counter == 0:
            fb2 = [re.split('_', os.path.basename(x))[0] for x in res_file]
            counter = 1
          if date_str in fb2:
            res0 = load_npy_gcs(f_bucket_name, res_file[fb2.index(date_str)])                                                                                                   #Load numpy results file
          else:
            res0 = []
          if len(res0) > 0:
            if len(results) <= 0:
              results = np.copy(res0[0, :, :, 0])
              if df_night['day_night'][idx].lower() == 'day':
                results[results < chk_day_night['day']['pthresh']] = 0.0
              else:
                results[results < chk_day_night['night']['pthresh']] = 0.0
            else:
              res00 = np.copy(res0[0, :, :, 0])
              if df_night['day_night'][idx].lower() == 'day':
                res00[res00 < chk_day_night['day']['pthresh']] = 0.0
              else:
                res00[res00 < chk_day_night['night']['pthresh']] = 0.0

              results[res00 > results] = res00[res00 > results] 
              res00 = None  

            if time_ag_only == False and len(res0) > 0:
#               if os.path.isfile(df_night['ir_files'][idx]) == False:
#                 os.makedirs(os.path.realpath(os.path.dirname(df_night['ir_files'][idx])), exist_ok = True)
#                 ifile = list_gcs(o_bucket_name, re.split('goes-data/', os.path.dirname(df_night['ir_files'][idx]))[1] + '/', [os.path.basename(df_night['ir_files'][idx])], delimiter = '/')
#                 if len(ifile) <= 0 or len(ifile) > 1:
#                   print('IR file for projections not found???')
#                   print(re.split('goes-data/', os.path.dirname(df_night['ir_files'][idx])[1]))
#                   exit()
#                 download_ncdf_gcs(o_bucket_name, ifile[0], os.path.dirname(os.path.realpath(df_night['ir_files'][idx])))                                                      #Download combined netCDF file
#   
#               ir    = xarray.open_dataset(df_night['ir_files'][idx]).load()
#               dat   = ir.metpy.parse_cf('Rad')
#               proj  = [dat.metpy.cartopy_crs, dat.x.min(), dat.x.max(), dat.y.min(), dat.y.max()]
#               ir.close() 
#               os.remove(df_night['ir_files'][idx])
              im_names, df_out = write_plot_model_result_predictions([n], ir_files, res = res0, no_plot_vis = no_plot_vis, region = region, latlon_domain = latlon_domain, pthresh = pthresh, chk_day_night = chk_day_night, model = mod0 + mod00, outroot = os.path.join(outroot, img_path, sector), write_gcs = run_gcs, del_local = del_local, ir_max = ir_max, ir_min = ir_min, verbose = verbose)
              im_names2.extend(im_names)
#              if os.path.isfile(os.path.join(inroot, dat_str, sector, 'blob_results_positioning.csv')):
#                df2    = pd.read_csv(os.path.join(inroot, dat_str, sector, 'blob_results_positioning.csv'))                                                                     #Read ftype csv file that already exists
#                df_out = pd.concat([df_out, df2], axis = 0, join = 'inner', sort = 'date_time')                                                                                 #Concatenate the dataframes and sort by date 
#                df_out = df_out.astype({'date_time' : str})
#                df_out.sort_values(by = 'date_time', axis =0, inplace = True)
#                df_out.drop_duplicates(inplace = True, subset = ['date_time', 'centroid_index_x', 'centroid_index_y'])                                                          #Drop any duplicate OT locations
#                df_out.reset_index(inplace = True)
#                header = list(df_out.head(0))                                                                                                                                   #Extract header of csv file
#                if 'Unnamed: 0' in header:
#                  df_out.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                                                   #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
#                if 'index' in header:
#                  df_out.drop(columns = 'index', axis = 'columns', inplace = True)                                                                                              #Remove index column that was created by reset_index.
#              else:
#                if run_gcs == True:
#                  pref     = os.path.relpath(os.path.join(inroot, dat_str, sector), re.split(use_chan, os.path.join(inroot, dat_str, sector))[0])
#                  csv_file = list_csv_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), 'blob_results_positioning.csv')                                                  #Check to see if blob_results_positioning.csv is on GCP so that they can be appended
#                  if len(csv_file) > 0:
#                    df2    = load_csv_gcs(f_bucket_name, csv_file[0])                                                                                                           #Read specified csv file
#                    df_out = pd.concat([df_out, df2], axis = 0, join = 'inner', sort = 'date_time')                                                                             #Concatenate the dataframes and sort by date 
#                    df_out = df_out.astype({'date_time' : str})
#                    df_out.sort_values(by = 'date_time', axis =0, inplace = True)
#                    df_out.drop_duplicates(inplace = True, subset = ['date_time', 'centroid_index_x', 'centroid_index_y'])                                                      #Drop any duplicate OT locations
#                    df_out.reset_index(inplace = True)
#                    header = list(df_out.head(0))                                                                                                                               #Extract header of csv file
#                    if 'Unnamed: 0' in header:
#                      df_out.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                                               #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
#                    if 'index' in header:
#                      df_out.drop(columns = 'index', axis = 'columns', inplace = True)                                                                                          #Remove index column that was created by reset_index.
##deprecated               
##               if os.path.isfile(os.path.join(outroot, 'blob_results_positioning.csv')):
##                 df2 = pd.read_csv(os.path.join(outroot, 'blob_results_positioning.csv'))                                                                                      #Read ftype csv file that already exists
##                 df2.append(df_out, ignore_index = True)                                                                                                                       #Append previous dates ftype dates csv file with new date
##                 df2.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                                                      #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
##                 df_out = df2    
#              
#              df_out.to_csv(os.path.join(inroot, dat_str, sector, 'blob_results_positioning.csv'))                                                                              #Write output csv file
      else:
        res0 = load_npy_gcs(f_bucket_name, res_file[0])                                                                                                                         #Load numpy results file
        if len(results) <= 0:
          results = np.nanmax(res0, axis = 0)
        else:
          max_res = np.nanmax(res0, axis = 0)
          results[max_res > results] = max_res[max_res > results]
          max_res = None
        if time_ag_only == False and len(res0) > 0:
#           ifile = list_gcs(c_bucket_name, re.split('goes-data/', os.path.dirname(df_night['glm_files'][0]))[1] + '/', [os.path.basename(df_night['glm_files'][0])], delimiter = '/')  
#           if len(ifile) <= 0 or len(ifile) > 1:
#             print('IR file for projections not found???')
#             print(re.split('goes-data/', os.path.dirname(df_night['glm_files'][0])[1]))
#             exit()
#           download_ncdf_gcs(c_bucket_name, ifile[0], os.path.dirname(os.path.realpath(df_night['glm_files'][0])))                                                             #Download combined netCDF file
          im_names, df_out = write_plot_model_result_predictions(nc_names, ir_files, res = res0, no_plot_vis = no_plot_vis, region = region, latlon_domain = latlon_domain, pthresh = pthresh, chk_day_night = chk_day_night, model = mod0 + mod00, outroot = os.path.join(outroot, img_path, sector), write_gcs = run_gcs, del_local = del_local, ir_max = ir_max, ir_min = ir_min, verbose = verbose)
#          im_names2.extend(im_names)
#          if os.path.isfile(os.path.join(inroot, dat_str, sector, 'blob_results_positioning.csv')):
#            df2    = pd.read_csv(os.path.join(inroot, dat_str, sector, 'blob_results_positioning.csv'))                                                                         #Read ftype csv file that already exists
#            df_out = pd.concat([df_out, df2], axis = 0, join = 'inner', sort = 'date_time')                                                                                     #Concatenate the dataframes and sort by date 
#            df_out = df_out.astype({'date_time' : str})
#            df_out.sort_values(by = 'date_time', axis =0, inplace = True)
#            df_out.drop_duplicates(inplace = True, subset = ['date_time', 'centroid_index_x', 'centroid_index_y'])                                                              #Drop any duplicate OT locations
#            df_out.reset_index(inplace = True)
#            header = list(df_out.head(0))                                                                                                                                       #Extract header of csv file
#            if 'Unnamed: 0' in header:
#              df_out.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                                                       #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
#            if 'index' in header:
#              df_out.drop(columns = 'index', axis = 'columns', inplace = True)                                                                                                  #Remove index column that was created by reset_index.
#          else:
#            if run_gcs == True:
#              pref     = os.path.relpath(os.path.join(inroot, dat_str, sector), re.split(use_chan, os.path.join(inroot, dat_str, sector))[0])
#              csv_file = list_csv_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), 'blob_results_positioning.csv')                                                      #Check to see if blob_results_positioning.csv is on GCP so that they can be appended
#              if len(csv_file) > 0:
#                df2    = load_csv_gcs(f_bucket_name, csv_file[0])                                                                                                               #Read specified csv file
#                df_out = pd.concat([df_out, df2], axis = 0, join = 'inner', sort = 'date_time')                                                                                 #Concatenate the dataframes and sort by date 
#                df_out = df_out.astype({'date_time' : str})
#                df_out.sort_values(by = 'date_time', axis =0, inplace = True)
#                df_out.drop_duplicates(inplace = True, subset = ['date_time', 'centroid_index_x', 'centroid_index_y'])                                                          #Drop any duplicate OT locations
#                df_out.reset_index(inplace = True)
#                header = list(df_out.head(0))                                                                                                                                   #Extract header of csv file
#                if 'Unnamed: 0' in header:
#                  df_out.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                                                   #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
#                if 'index' in header:
#                  df_out.drop(columns = 'index', axis = 'columns', inplace = True)                                                                                              #Remove index column that was created by reset_index.
#
##deprecated                     
##           if os.path.isfile(os.path.join(outroot, 'blob_results_positioning.csv')):
##             df2 = pd.read_csv(os.path.join(outroot, 'blob_results_positioning.csv'))                                                                                          #Read ftype csv file that already exists
##             df2.append(df_out, ignore_index = True)                                                                                                                           #Append previous dates ftype dates csv file with new date
##             df2.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                                                          #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
##             df_out = df2     
#           
#          df_out.to_csv(os.path.join(inroot, dat_str, sector, 'blob_results_positioning.csv'))                                                                                  #Write output csv file

#      res0     = load_npy_gcs(f_bucket_name, res_file[0])                                                                                                                      #Load numpy results file
#      pref     = re.split(d_str + '/', inroot)[0]    
#    nc_names = nc_names[0:res_shape[0]]
    
    
    if use_local == False:
      os.makedirs(os.path.realpath(os.path.dirname(nc_names[0])), exist_ok = True)
      nc_names2  = list_gcs(c_bucket_name, os.path.join('combined_nc_dir', re.split('combined_nc_dir' + os.sep, nc_names)[1]), ['.nc'], delimiter = '/')
      pool       = mp.Pool(3)
      results2   = [pool.apply_async(download_gcp_parallel, args=(row, nc_names2, c_bucket_name, os.path.realpath(os.path.dirname(nc_names[0])))) for row in range(len(nc_names2))]
      pool.close()
      pool.join()
    
    nc_files.extend(nc_names)
    if time_ag_only == False and run_gcs == True:   
      pref = os.path.relpath(outroot, re.split('aacp_results_imgs', outroot)[0])
      write_to_gcs(f_bucket_name, os.path.join('Cooney_testing', pref), os.path.join(inroot, dat_str, sector, 'blob_results_positioning.csv'))                                  #Write results csv file to Google Cloud Storage Bucket
      pref = os.path.relpath(os.path.dirname(im_names2[0]), re.split('aacp_results_imgs', os.path.dirname(im_names2[0]))[0])
      for i in im_names2:
        write_to_gcs(f_bucket_name, pref, i, del_local = del_local)                                                                                                             #Write results object positions to Google Cloud Storage Bucket

#     if l == 0:
#       results = np.copy(res0)
#     else:
##      results[res0 > results] = res0[res0 > results]
#      results = np.append(results, np.copy(res0), axis = 0)
#    res0 = None
    
#   pref0 = re.split(d_str + '/', re.split('aacp_results/', inroot)[1])[0] + d_str + '/'
  
  if time_ag_only == True and run_gcs == True:
    ifile = list_gcs(c_bucket_name, os.path.join('combined_nc_dir', os.path.basename(os.path.dirname(df_night['glm_files'][0]))), [os.path.basename(df_night['glm_files'][0])], delimiter = '/')  
    if len(ifile) <= 0 or len(ifile) > 1:
      print('IR file for projections not found???')
      print(os.path.join('combined_nc_dir', os.path.basename(os.path.dirname(df_night['glm_files'][0]))))
      print(c_bucket_name)
      exit()
    download_ncdf_gcs(c_bucket_name, ifile[0], os.path.dirname(os.path.realpath(df_night['glm_files'][0])))                                                                     #Download combined netCDF file

#Set up out file path from dates
  d1    = str(np.min(df_dates['out_date']))
  d1j   = datetime.strptime(d1, "%Y%j").strftime("%j")
  d2    = datetime.strptime(str(np.max(df_dates['out_date'])), "%Y%j").strftime("%j")
  if d1j == d2:
    pat3 = d1
  else:
    pat3 = d1 + '-' + d2
#   if time_ag_only == True:
#     ifile = list_gcs(o_bucket_name, re.split('goes-data/', os.path.dirname(df_night['ir_files'][0]))[1] + '/', [os.path.basename(df_night['ir_files'][0])], delimiter = '/')  
#     if len(ifile) <= 0 or len(ifile) > 1:
#       print('IR file for projections not found???')
#       print(re.split('goes-data/', os.path.dirname(df_night['ir_files'][0])[1]))
#       exit()
#     download_ncdf_gcs(o_bucket_name, ifile[0], os.path.dirname(os.path.realpath(df_night['ir_files'][0])))                                                                    #Download combined netCDF file
#     ir    = xarray.open_dataset(df_night['ir_files'][0]).load()
#     dat   = ir.metpy.parse_cf('Rad')
#     proj  = [dat.metpy.cartopy_crs, dat.x.min(), dat.x.max(), dat.y.min(), dat.y.max()]
#     ir.close() 
  write_plot_model_result_predictions(nc_files, ir_files, res = results, region = region, latlon_domain = latlon_domain, pthresh = pthresh, chk_day_night = chk_day_night, outroot = os.path.join(outroot, 'time_ag_figs', pat3, sector), time_ag = True, proj = None, use_local = use_local, write_gcs = run_gcs, del_local = del_local, ir_max = ir_max, ir_min = ir_min, verbose = verbose)
  
  if use_local == False and del_local == True:
    [os.remove(x) for x in nc_files if os.path.isfile(x)]                                                                                                                       #Remove all downloaded combined netCDF files

def write_plot_model_result_predictions(nc_names, ir_files, tod = [], res = [], region = None, latlon_domain = [], pthresh = 0.5, 
                                        res_dir          = os.path.join('..', '..', '..', 'aacp_results', 'ir_vis_glm', 'updraft_day_model', 'not_real_time', '2022-02-18', 'multiresunet'),
                                        outroot          = os.path.join('..', '..', '..', 'aacp_results_imgs'), 
                                        res_bucket_name  = 'aacp-results',
                                        proc_bucket_name = 'aacp-proc-data', 
                                        write_gcs        = True, 
                                        del_local        = True,
                                        use_local        = False, 
                                        model            = None,
                                        date_range       = None, 
                                        grid_data        = True, 
                                        proj             = None, 
                                        chk_day_night    = {}, 
                                        time_ag          = False, 
                                        no_plot          = False, no_plot_vis = False, 
                                        ir_max           = 230.0, ir_min = 180.0,
                                        verbose          = True):
  ''' 
  This is a function to plot and write the model results to provide to forecasters 
  Args: 
    nc_names  : STRING list specifying the name and file path of the combined netCDF files 
    ir_files  : STRING list specifying the name and file path of the IR numpy files 
  Keywords:
    res             : Numpy array of model predictions (values range from 0-1). DEFAULT = [] -> read results data file
    tod             : List containing strings specifying if model is day or night run. DEFAULT = [] -> don't use day or night pthresh difference.
                      DEFAULT = None -> plot for full domain of satellite scan.
    region          : State or sector of US to plot data around. 
    latlon_domain   : [x0, y0, x1, y1] array in latitude and longitude coordinates to give subset of image domain desired to be plotted. Array must contain lat/lon values
                      DEFAULT is to plot the entire domain. 
                      NOTE: If region keyword is set, it supersedes this keyword
    pthresh         : FLOAT with the probability threshold to threshold the results by. DEFAULT = 0.5
    res_dir         : STRING output directory path for results data storage 
                      DEFAULT = '../../../aacp_results/ir_vis_glm/updraft_day_model/not_real_time/2022-02-18/multiresunet/' 
    outroot         : STRING output directory path for image root
                      DEFAULT = '../../../aacp_results_imgs/' 
    res_bucket_name : STRING results data directory
                      DEFAULT = 'aacp-results'
    proc_bucket_name: STRING processed data directory 
                      DEFAULT = 'aacp-proc-data'
    write_gcs       : IF keyword set (True), write the images to google cloud storage.
                      DEFAULT = True.                  
    del_local       : IF keyword set (True), delete locally stored files after writing them to the GCP
                      DEFAULT = True.                  
    use_local       : IF keyword set (True), use local files. If False, use files located on the google cloud. 
                      DEFAULT = False -> use files on google cloud server. 
    model           : Name of model being plotted (ex. IR+VIS)
                      DEFAULT = None.
    date_range      : List containing start date and end date of predictions [start_date, end_date]. (FORMAT = %Y-%m-%d %H:%M:%S) (ex. '2017-05-01 00:00:00') 
                      DEFAULT = None -> use nc_names to extract date range
    grid_data       : IF keyword set (True), plot data on lat/lon grid.
                      DEFAULT = True. False-> plot on 1:1 pixel grid.
    proj            : Projection of gridded data onto the map. DEFAULT = None -> no projection.
                      Projections are obtained from using xarray and metpy reading of raw GOES netCDF files.
    chk_day_night   : Structure containing day and night model data information in order to make seemless transition in predictions of the 2.
                      DEFAULT = {} -> do not use day_night transition                 
    time_ag         : IF set, plot time aggregated image.
                      DEFAULT = False -> plot and write individual image scenes
    no_plot         : IF keyword set (True), do not plot the data and only write the csv file of where the OTs are located for specified time.        
    no_plot_vis     : IF keyword set, do not plot the VIS data because it was not written to combined netCDF files. Setting this keyword = True makes you plot only IR data.
                      DEFAULT is to set this keyword to False and plot the VIS and IR data.
    ir_max          : FLOAT keyword specifying the maximum IR BT in color range of map (K). DEFAULT = 230 K
    ir_min          : FLOAT keyword specifying the minimum IR BT in color range of map (K). DEFAULT = 180 K
    verbose         : BOOL keyword to specify whether or not to print verbose informational messages. 
                      DEFAULT = True which implies to print verbose informational messages 
  Output: 
    Saves files containing the locations of the updraft identifications as well as each of the plots. 
  '''   
  if verbose == True:
    print('Results images output location = ' + outroot)
    if len(chk_day_night) == 0:
      print('Results using pthresh = ' + str(pthresh))
    else:  
      print('Results using model likelihood scores for day   >= ' + str(chk_day_night['day']['pthresh']))
      print('Results using model likelihood scores for night >= ' + str(chk_day_night['night']['pthresh']))
  if len(res) > 0 and len(chk_day_night) == 0:
    res[res < pthresh] = 0.0
  else:
    fb = [re.split('_s|_', os.path.basename(x))[5][0:-1] for x in nc_names]
  
  if time_ag == False:
    s               = generate_binary_structure(2,2) 
    im_names        = []                                                                                                                                                        #Initialize list store the output image names
    centroid_inds_x = [] 
    centroid_inds_y = [] 
    date_time2      = [] 
    max_pthresh     = []
    res0            = []
    pref0   = ''
    counter = 0
    for idx, n in enumerate(nc_names): 
      date_str = fb[idx]                                                                                                                                                        #Split file string in order to extract date string of scan
      if no_plot == False:
        if len(res) == 0:
          if date_range != None:
            d0 = datetime.strptime(date_range[0], "%Y-%m-%d %H:%M:%S").strftime("%Y%j%H%M%S")                                                                                   #Extract date start time as string in saem formate as nc_names files are saved
            d1 = datetime.strptime(date_range[1], "%Y-%m-%d %H:%M:%S").strftime("%Y%j%H%M%S")                                                                                   #Extract end start time as string in saem formate as nc_names files are saved
            if date_str[0:-2] >= d0[0:-2] and date_str[0:-2] <= d1[0:-2]:                                                                                                       #Make sure date of file is within date range specified
              cont = 1
            else:
              cont = 0
              res0 = []
          else:
            cont = 1      
          if cont == 1:
            res_dir = os.path.realpath(res_dir)                                                                                                                                 #Create link to real input root directory path so compatible with Mac
            if use_local == True: 
              res_file = sorted(glob.glob(os.path.join(res_dir, '*.npy')))                                                                                                      #Extract name of numpy results file
              if len(res_file) == 0:     
                print('No results file found??')     
                res0 = []   
              if len(res_file) > 1:                                                                                                                                             #If results file for each scan
                if counter == 0:
                  fb2 = [re.split('_', os.path.basename(x))[0] for x in res_file]
                  counter = 1
                if date_str in fb2:
                  res0 = np.load(res_file[fb2.index(date_str)])                                                                                                                 #Load numpy results file
                else:  
                  res0 = []  
              else:  
                res0 = np.load(res_file[0])                                                                                                                                     #Load numpy results file
            else:      
              pref = os.path.join('Cooney_testing', re.split('aacp_results' + os.sep, res_dir)[1])
              if pref != pref0:  
                res_file = list_gcs(res_bucket_name, pref, ['results', '.npy'], delimiter = '/')                                                                                #Extract name of model results file
                if len(res_file) == 0:      
                  print('No results file found??')  
                  res0 = []    
                pref0 = pref  
              if len(res_file) > 1:      
                if counter == 0:  
                  fb2 = [re.split('_', os.path.basename(x))[0] for x in res_file]  
                  counter = 1  
                if date_str in fb2:  
                  res0 = load_npy_gcs(res_bucket_name, res_file[fb2.index(date_str)])                                                                                           #Load numpy results file
                else:  
                  res0 = []            
              else:  
                res0 = load_npy_gcs(res_bucket_name, res_file[0])                                                                                                               #Load numpy results file
            if len(res0) > 0:  
              res0 = np.copy(res0[0, :, :, 0])  
              if len(chk_day_night) == 0:
                res0[res0 < pthresh] = 0.0
              else:
                if tod[idx].lower() == 'night':
                  res0[res0 < chk_day_night['night']['pthresh']] = 0.0
                  pthresh = chk_day_night['night']['pthresh']
                elif tod[idx].lower() == 'day':
                  res0[res0 < chk_day_night['day']['pthresh']] = 0.0
                  pthresh = chk_day_night['day']['pthresh']
                else:
                  print('tod variable must be set to day or night if given chk_day_night dictionary.')
                  print(tod)
                  print(chk_day_night)
                  exit()
        else:    
          res0 = np.copy(res[idx, :, :, 0])  
      if len(res0) > 0 or no_plot == True:  
        if no_plot == True:
          im_name = os.path.join(outroot, date_str + '_' + re.sub('\.nc$', '', os.path.basename(n)) + '.png')
        else:
          im_name = img_from_three_modalities2(nc_file       = n,                                                                                                               #Create IR/VIS sandwich image with model detections plotted on top
                                               out_dir       = outroot,   
                                               plt_model     = res0,    
                                               model         = model,   
                                               pthresh       = pthresh, 
                                               no_plot_glm   = True,    
                                               chk_day_night = chk_day_night, 
                                               grid_data     = grid_data, proj = proj, plt_img_name = False, plt_cbar = False, 
                                               subset        = None, region = region, latlon_domain = latlon_domain, no_plot_vis = no_plot_vis,
                                               plt_ir_max    = ir_max, plt_ir_min = ir_min)   
       
        res0 = None
        im_names.append(im_name)                                                                                                                                                #Add image output file name to list
#         file_attr = re.split('_s|_', os.path.basename(n))                                                                                                                       #Split file string in order to extract date string of scan
#         d_str     = file_attr[5]                                                                                                                                                #Split file string in order to extract date string of scan
#         d_str     = d_str[0:7] + ' ' + d_str[7:-1]  
#         labeled_array, num_updrafts = label(res0 > 0, structure = s)                                                                                                            #Extract labels for each updraft location (make sure points at diagonals are part of same updrafts)
#         for u in range(num_updrafts):                                                 
#           inds   = np.where(labeled_array == u+1)                                                                                                                               #Find locations of updraft region mask
#           print('Number of pixels in updraft = ' + str(len(inds[0])))
#           x_arr  = list(inds[0])                                                                                                                                                #Find x indices of updraft
#           y_arr  = list(inds[1])                                                                                                                                                #Find y indices of updraft
#           x_arr0 = int(round(np.mean(x_arr)))                                                                                                                                   #Round the mean point to get center location
#           y_arr0 = int(round(np.mean(y_arr)))                                                                                                                                   #Round the mean point to get center location
#           max_pthresh.append(np.nanmax(res0[inds]))
#           centroid_inds_x.append(x_arr0)                                                                                                                                        #Save 1D centroid index to list
#           centroid_inds_y.append(y_arr0)                                                                                                                                        #Save 1D centroid index to list
#           date_time2.append(datetime.strftime(datetime.strptime(d_str, "%Y%j %H%M%S"), "%Y-%m-%d %H:%M:%S"))                                                                    #Save date of updraft to list
      
    if len(centroid_inds_x) > 0:  
#       dim    = [None]*len(centroid_inds_x)                                                                                                                                      #List of None values equal to number of elements in centroid_inds_x
#       df_out = pd.DataFrame({'date_time': date_time2, 'max_likelihood': max_pthresh, 'centroid_index_x': centroid_inds_x, 'centroid_index_y': centroid_inds_y, 'box_size': dim})#Create dataframe of all the centroid of each object along with the dates and size of the dimensions
#       df_out = df_out.astype({'date_time' : str})
#       df_out.sort_values(by='date_time', inplace = True)  
      df_out = pd.DataFrame()                                                                                                                                                   #Create empty data frame if no OTs found

      print(str(len(centroid_inds_x)) + ' OTs identified within domain')  
    else:  
      df_out = pd.DataFrame()                                                                                                                                                   #Create empty data frame if no OTs found
#      print('No OTs identified within domain')  
    res = None
    return(sorted(im_names), df_out)                                                                                                                                            #Return the image names as well as the pandas data frame
  else:  
    if date_range == None:  
      file_attr = re.split('_s|_', os.path.basename(sorted(nc_names)[0]))                                                                                                       #Split file string in order to extract date string of scan
      d_str     = file_attr[5]                                                                                                                                                  #Split file string in order to extract date string of scan
      d_str     = d_str[0:7] + ' ' + d_str[7:-1]    
      d1        = datetime.strftime(datetime.strptime(d_str, "%Y%j %H%M%S"), "%Y-%m-%d %H:%M:%S")                                                                               #Save start date
      file_attr = re.split('_s|_', os.path.basename(sorted(nc_names)[-1]))                                                                                                      #Split file string in order to extract date string of scan
      d_str     = file_attr[5]                                                                                                                                                  #Split file string in order to extract date string of scan
      d_str     = d_str[0:7] + ' ' + d_str[7:-1]    
      d2        = datetime.strftime(datetime.strptime(d_str, "%Y%j %H%M%S"), "%Y-%m-%d %H:%M:%S")                                                                               #Save end date
    else:  
      d1 = date_range[0]  
      d2 = date_range[1]  
    
    if len(res) == 0:
      res2 = []                                                                                                                                                                 #Initialize list to store maximum model results values for each lat/lon of domain
      for idx, n in enumerate(nc_names):   
        res_dir = os.path.realpath(res_dir)                                                                                                                                     #Create link to real input root directory path so compatible with Mac
        if use_local == True:  
          res_file = sorted(glob.glob(res_dir + os.sep + '*.npy'))                                                                                                                      #Extract name of numpy results file
          if len(res_file) == 0:      
            print('No results file found??')      
          if len(res_file) > 1:                                                                                                                                                 #If results file for each scan
            date_str = fb[idx]                                                                                                                                                  #Split file string in order to extract date string of scan
            if counter == 0:  
              fb2 = [re.split('_', os.path.basename(x))[0] for x in res_file]  
              counter = 1  
            if date_str in fb2:  
              if len(res2) <= 0:  
                res2 = np.load(res_file[fb2.index(date_str)])[:, :, :, 0]                                                                                                       #Load numpy results file
                if len(chk_day_night) != 0:
                  if tod[idx].lower() == 'night':
                    res2[res0 < chk_day_night['night']['pthresh']] = 0.0
                  elif tod[idx].lower() == 'day':
                    res2[res2 < chk_day_night['day']['pthresh']] = 0.0
                  else:
                    print('tod variable must be set to day or night if given chk_day_night dictionary.')
                    print(tod)
                    print(chk_day_night)
                    exit()
              else:  
                res0 = np.load(res_file[fb2.index(date_str)])[:, :, :, 0]                                                                                                       #Load numpy results file
                if len(chk_day_night) != 0:
                  if tod[idx].lower() == 'night':
                    res0[res0 < chk_day_night['night']['pthresh']] = 0.0
                  elif tod[idx].lower() == 'day':
                    res0[res0 < chk_day_night['day']['pthresh']] = 0.0
                  else:
                    print('tod variable must be set to day or night if given chk_day_night dictionary.')
                    print(tod)
                    print(chk_day_night)
                    exit()
                res2[res0 > res2] = res0[res0 > res2]                                                                                                                           #Retain only the maximum values at each index
                res0 = []  
        else:      
          pref = os.path.join('Cooney_testing', re.split('aacp_results' + os.sep, res_dir)[1] )     
          if pref != pref0:  
            res_file = list_gcs(res_bucket_name, pref, ['results', '.npy'], delimiter = '/')                                                                                    #Extract name of model results file
            if len(res_file) == 0:      
              print('No results file found??')  
            pref0 = pref  
          if len(res_file) > 1:      
            date_str = fb[idx]                                                                                                                                                  #Split file string in order to extract date string of scan
            if counter == 0:  
              fb2 = [re.split('_', os.path.basename(x))[0] for x in res_file]  
              counter = 1  
            if date_str in fb2:  
              if len(res2) <= 0:  
                res2 = np.load(res_file[fb2.index(date_str)])[:, :, :, 0]                                                                                                       #Load numpy results file
                if len(chk_day_night) != 0:
                  if tod[idx].lower() == 'night':
                    res2[res0 < chk_day_night['night']['pthresh']] = 0.0
                  elif tod[idx].lower() == 'day':
                    res2[res2 < chk_day_night['day']['pthresh']] = 0.0
                  else:
                    print('tod variable must be set to day or night if given chk_day_night dictionary.')
                    print(tod)
                    print(chk_day_night)
                    exit()
              else:  
                res0 = np.load(res_file[fb2.index(date_str)])[:, :, :, 0]                                                                                                       #Load numpy results file
                if len(chk_day_night) != 0:
                  if tod[idx].lower() == 'night':
                    res0[res0 < chk_day_night['night']['pthresh']] = 0.0
                  elif tod[idx].lower() == 'day':
                    res0[res0 < chk_day_night['day']['pthresh']] = 0.0
                  else:
                    print('tod variable must be set to day or night if given chk_day_night dictionary.')
                    print(tod)
                    print(chk_day_night)
                    exit()
                res2[res0 > res2] = res0[res0 > res2]                                                                                                                           #Retain only the maximum values at each index
                res0 = []  
      res  = np.copy(res2)                                                                                                                                                      #Extract input data for full domain
      res2 = None  
      if len(chk_day_night) != 0:
        res[res < pthresh] = 0.0  
      res[res == 0.0] = np.nan  
    else:  
      res   = res[:, :]                                                                                                                                                         #Extract input data for full domain
      if len(chk_day_night) == 0:
        res[res < pthresh] = 0.0
      res[res == 0.0] = np.nan
#     visualize_time_aggregated_results(res, nc_names, ir_files,                                                                                                                  #Create time aggregated plot of model results and satellite data
#                                       outroot          = outroot,
#                                       date_range       = [d1, d2], 
#                                       region           = region,
#                                       latlon_domain    = latlon_domain, 
#                                       pthresh          = pthresh, 
#                                       use_local        = use_local, write_gcs = write_gcs, del_local = del_local, 
#                                       proc_bucket_name = proc_bucket_name,
#                                       res_bucket_name  = res_bucket_name,
#                                       ir_max           = ir_max, ir_min = ir_min,
#                                       verbose          = verbose)

def main():
  run_write_plot_model_result_predictions()
    
if __name__ == '__main__':
  main()

