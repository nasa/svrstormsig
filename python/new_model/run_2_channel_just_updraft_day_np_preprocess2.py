#+
# Name:
#     run_2_channel_just_updraft_day_np_preprocess2.py
# Purpose:
#     This is a script to pre-process, by subsetting 2000x2000 matrices into 64x64 matrices around the updraft centers or 512x512 matrices for just by_date, for ML training. 
# Calling sequence:
#     import run_2_channel_just_updraft_day_np_preprocess2
#     run_2_channel_just_updraft_day_np_preprocess2.run_2_channel_just_updraft_day_np_preprocess2()
# Input:
#     None.
# Functions:
#     run_2_channel_just_updraft_day_np_preprocess2 : Main function to subset the matrices
#     build_imgs                                    : Builds image files with the specified channels in the original picture shape of the image.
#     build_masks                                   : Builds masks which have been hand-labeled by SMEs for AACP detection.
#     build_subset_tensor                           : Reshapes a tensor into smaller dimensions and more rows for easier ML training (not overwhelming the GPU in matrix size).
#     extract_subset_region                         : Uses switch in order to return the subset region hard-coded indices picked from reviewing images
# Output:
#     64x64 numpy files for updraft training, testing, and validation that are input into the ML algorithm
# Keywords:
#     inroot          : STRING specifying input root directory to the vis_ir_glm_json csv files
#                       DEFAULT = '../../../goes-data/labelled/'
#     outroot         : STRING specifying output root directory to save the 512x512 numpy files
#                       DEFAULT = '../../../goes-data/labelled/'
#     batch_run       : IF keyword set (True), do not require the user to specify the number of dates to use for training, testing, and validating the model.
#                       Instead, 50% of dates are used for training, 25% for validation, and the remainder are used for testing
#                       DEFAULT = False.
#     use_local       : IF keyword set (True), use local files. If False, use files located on the google cloud. 
#                       DEFAULT = False -> use files on google cloud server. 
#     write_gcs       : IF keyword set (True), write the output files to the google cloud in addition to local storage.
#                       DEFAULT = True.
#     in_bucket_name  : STRING specifying the name of the gcp bucket to read files from. use_local needs to be False in order for this to matter.
#                       DEFAULT = 'aacp-proc-data'
#     out_bucket_name : STRING specifying the name of the gcp bucket to write files to. write_gcs needs to be True in order for this to matter.
#                       DEFAULT = 'aacp-results'
#     del_local       : IF keyword set (True) AND write_gcs == True, delete local copy of output file.
#                       DEFAULT = True.
#     reviewers       : STRING array specifying which reviewers to look for in the vis_ir_glm_json csv files
#                       DEFAULT = ['cooney', 'bedka', 'AE', 'KJ', 'LR', 'TD', 'irmod', 'irvismod', 'irglmmod', 'irvisglmmod']
#     nreviewers      : LONG specifying the number of reviewers required for their manual identifications to be included in the mask. If bedka is the reviewer then only he is required.
#                       DEFAULT = 2. For non-bedka reviewed dates, we require 2 or more reviewers in agreement in order for a pixel to be included in the mask. 
#     use_glm         : IF keyword set (True), pre-process the IR and GLM data rather than IR and VIS data. This is used to test models that were trained
#                       using only the IR and GLM data. 
#                       DEFAULT = False -> pre-process using IR and VIS data.
#     use_irdiff      : IF keyword set (True), pre-process the IR and IR_diff data rather than IR and VIS data. This is used to test models that were trained
#                       using only the IR and IR_diff data. 
#                       DEFAULT = False -> pre-process using IR and VIS data.
#     full_domain     : IF keyword set (True), output numpy file mask and images at the full domain resolution (2000x2000) in addition to at the 512x512 or 64x64 resolution
#                       DEFAULT = True. False -> Only out put numpy file mask and images at 64x64 (for by_updraft = True) or 512x512 (for by_updraft = False)
#     proc_date_str   : Optional STRING (yyyyddd) keyword that specifies date string to run pre-processing for. (ex. '2019138')
#                       DEFAULT = None -> pre-process all dates.
#     proc_sector     : Optional STRING keyword that specifies the GOES sector you want to pre-process. (examples: 'C', 'F' 'M2', 'M1')
#                       Must be set to a string if proc_date_str is set to a date.
#                       DEFAULT = None.
#     process_type    : Optional STRING keyword that specifies the pre-process file type of training, validation, or testing. (ex. 'test', 'train', 'val')
#                       DEFAULT = 'test' -> Only create test pre-processed numpy output file. 
#     use_night       : IF keyword set (True), use night time data IN ADDITION TO day time data as model input.
#                       DEFAULT = False -> only use day time data.
#     by_date         : IF keyword set (True), create model training, validation, and testing numpy output files for separate dates.
#                       User gets to choose how many dates go into each portion of model development.
#                       DEFAULT = True. If False, use first 75% of data for training, next 20% for validation, and last 5% for testing.
#     by_updraft      : IF keyword set (True), create model training, validation, and testing numpy output files for 64x64 pixels around each updraft by dates.
#                       NOTE: If this keyword is set, by_date will automatically also be set.
#                       DEFAULT = True. If False, use by_date 512x512 or first 75% of data for training, next 20% for validation, and last 5% for testing.
#     subset          : IF keyword set (True), use [x0, y0, x1, y1] indices from extract_subset_region array to give subset of image domain 
#                       desired to be trained and validated by model.
#                       DEFAULT = True. False implies to create 512x512 boxes across the full domain
#     img_size        : Size of images to use as input into the model. 64x64 regions are default if by_updraft is set. 512x512 regions default otherwise.
#     min_obj_size    : Minimum size allowed to be part of an object to be trained on. IF set to an integer value, only objects >= min_obj_size are included in the pre-processing.
#                       DEFAULT = None. (ex. 20)
#     verbose         : BOOL keyword to specify whether or not to print verbose informational messages.
#                       DEFAULT = True which implies to print verbose informational messages
# IMPORTANT NOTE:
#     Only training dates use bootstrap reviewers like irmod, irvismod, etc. Any date listed as a test or val date DOES NOT use those bootstrapped files in the mask creation.
# Author and history:
#     John W. Cooney           2021-02-24. (Rewritten from 2-channel-just-plume-day-np-preprocess.ipynb)
#
#-

#### Environment Setup ###
# Package imports
import numpy as np
import pandas as pd
from math import ceil
import glob
import os
import re
import math
from datetime import datetime
from scipy.ndimage import label, generate_binary_structure
from new_model.OverlapChunks import OverlapChunks
import sys 
#sys.path.insert(1, '../')
sys.path.insert(1, os.path.dirname(__file__))
from new_model.gcs_processing import write_to_gcs, load_npy_gcs, load_csv_gcs, list_csv_gcs

def run_2_channel_just_updraft_day_np_preprocess2(inroot         = '../../../goes-data/labelled/', 
                                                  outroot        = '../../../goes-data/labelled/', 
                                                  batch_run      = False, 
                                                  use_local      = False, write_gcs = True,
                                                  in_bucket_name = 'aacp-proc-data', out_bucket_name = 'aacp-results', 
                                                  del_local      = True,
                                                  reviewers      = ['cooney', 'bedka', 'AE', 'KJ', 'LR', 'TD', 'irmod', 'irvismod', 'irglmmod', 'irvisglmmod'], nreviewers = 2, 
                                                  use_glm        = False, use_irdiff = False, full_domain = True, 
                                                  proc_date_str  = None,  proc_sector = None, process_type = 'test',
                                                  use_night      = False, 
                                                  by_date        = True, subset = True, by_updraft = True,  img_size = None, 
                                                  min_obj_size   = None, 
                                                  use_native_ir  = False, 
                                                  verbose        = True):
    if img_size == None:
        if by_updraft == True: 
            img_size = 64                                                                                                                              #Use 64x64 pixel regions, centered on plume center location to pre-process the training, val, and test datasets    
        else:                                                                                                                            
            img_size = 512                                                                                                                             #Use 512x512 pixel regions to pre-process the training, val, and test datasets    
    if use_native_ir == True:
        dims   = [500, 500]
        outpat = 'ir_res_'                                                                                                                             #Add to name of output file to distinguish that it is a file at the native IR resolution, rather than the default visible data resolution
    else:
        dims   = [2000, 2000]   
        outpat = ''                                                                                                                                    #Add to name of output file to distinguish that it is a file at the native IR resolution, rather than the default visible data resolution
    inroot   = os.path.realpath(inroot)                                                                                                                #Create link to real path so compatible with Mac
    outroot  = os.path.realpath(outroot)                                                                                                               #Create link to real path so compatible with Mac
    if by_updraft == True:        
        by_date = True        
    if use_glm == True:         
        outroot = os.path.join(outroot, 'ir_glm')        
    else:        
        if use_irdiff == True:
            outroot = os.path.join(outroot, 'ir_irdiff')        
        else:
            outroot = os.path.join(outroot, 'ir_vis')        
            if use_night == True:
                print('Not completely set up to use night time variables with VIS data!')
                exit()
            
    outroot  = os.path.join(outroot, 'updraft_day_model')        
        
    now_time = datetime.now()                                                                                                                          #Extract current date
    outroot  = os.path.join(outroot, now_time.strftime("%Y-%m-%d/"))                                                                                   #Directory path to model results files
    if full_domain == True: outroot0 = outroot                                                                                                         #Store output directory for full domain files
    if subset == True:        
        outroot = os.path.join(outroot, 'chosen_indices')    
    if by_date == True:    
        outroot = os.path.join(outroot, 'by_date')    
    if by_updraft == True:    
        outroot = os.path.join(outroot, 'by_updraft')    
        s = generate_binary_structure(2,2)    
    if use_local == True:    
        if proc_date_str == None:    
            csv_files = sorted(glob.glob(inroot + '/**/vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv', recursive = True))                 #Extract names of all of the GOES visible data files
        else:
            if proc_sector == None:
                print('Processed data dector must be specified as C, M1, M2, or F!!')
                exit()
            proc_sector.upper()
            csv_files = sorted(glob.glob(inroot + '/' + proc_date_str + '/' + proc_sector + '/vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv'))  #Extract names of all of the GOES visible data files
    else: 
        if proc_date_str == None:
            csv_files = list_csv_gcs(in_bucket_name, 'goes-data/labelled/', 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv')              #Extract names of all of the GOES visible data files
        else:
            if proc_sector == None:
                print('Processed data dector must be specified as C, M1, M2, or F!!')
                exit()
            proc_sector.upper()
            csv_files = list_csv_gcs(in_bucket_name, 'goes-data/labelled/' + proc_date_str + '/' + proc_sector + '/', 'vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv')  #Extract names of all of the GOES visible data files
            print('Pre-processing ' + proc_date_str + ' for sector ' + proc_sector + ' as a ' + process_type)
    
    if len(csv_files) == 0: 
        print('No csv files found??')
        exit()
    
    if proc_date_str == None:
        if by_date == True:
            n_trdates = 0                                                                                                                              #Initialize variable to store number of dates to use for training the model
            n_vdates  = 0                                                                                                                              #Initialize variable to store number of dates to use for validating the model
            n_tedates = 0                                                                                                                              #Initialize variable to store number of dates to use for testing the model
            tot       = n_trdates + n_vdates + n_tedates        
            if batch_run == False:        
                while tot != len(csv_files):        
                    print('There are ' + str(len(csv_files)) + ' dates available. Please choose number of training dates: ')        
                    n_trdates = int(input())                                                                                                           #Prompt user for number of dates to use for training the model
                    print('There are ' + str(len(csv_files)-n_trdates) + ' dates available. Please choose number of validation dates: ')        
                    n_vdates  = int(input())                                                                                                           #Prompt user for number of dates to use for validating the model
                    print('There are ' + str(len(csv_files)-n_trdates-n_vdates) + ' dates available. Please choose number of testing dates: ')        
                    n_tedates = int(input())                                                                                                           #Prompt user for number of dates to use for testing the model
                    tot       = n_trdates + n_vdates + n_tedates        
                    if tot != len(csv_files):                                                                                                          #Number of dates entered by user must match number of dates available for input into model!!!
                        print('Total number of dates input does not match what is available. Try again.')        
            else:        
                n_trdates = round(len(csv_files)/2.0)        
                n_vdates  = round(len(csv_files)/4.0)        
                n_tedates = len(csv_files) - n_trdates - n_vdates        
                if n_tedates <= 0:        
                    print('Number of test dates has to be larger than zero!!')        
                    exit()        
                if tot != len(csv_files):                                                                                                              #Number of dates entered by user must match number of dates available for input into model!!!
                    print('Total number of dates input does not match what is available. Try again.')        
        else:        
            n_trdates = round(len(csv_files)/2.0)        
            n_vdates  = round(len(csv_files)/4.0)        
            n_tedates = len(csv_files) - n_trdates - n_vdates        
            if n_tedates <= 0:        
                print('Number of test dates has to be larger than zero!!')        
                exit()        
            if tot != len(csv_files):                                                                                                                  #Number of dates entered by user must match number of dates available for input into model!!!
                print('Total number of dates input does not match what is available. Try again.')        
        ptype = []
        ptype.extend(['train']*n_trdates)
        ptype.extend(['val']*n_vdates)
        ptype.extend(['test']*n_tedates)
    else:        
        process_type.lower()        
        if process_type == 'test':        
            ptype = ['test']
            n_trdates, n_vdates, n_tedates = 0, 0, 1                                                                                                   #Initialize variable to store number of dates to use for each pre-processed file type the model
        elif process_type == 'val':         
            ptype = ['val']
            n_trdates, n_vdates, n_tedates = 0, 1, 0                                                                                                   #Initialize variable to store number of dates to use for each pre-processed file type the model
        elif process_type == 'train':         
            ptype = ['train']
            n_trdates, n_vdates, n_tedates = 1, 0, 0                                                                                                   #Initialize variable to store number of dates to use for each pre-processed file type the model
        
    outdate = []                                                                                                                                       #Initialize array to store date string
    ftype   = []                                                                                                                                       #Initialize array to store train, val, or test for specific date
    sector  = []                                                                                                                                       #Initialize array to store sector (mesoscale 1, 2)
    index0  = []                                                                                                                                       #Initialize list to store the number of values for a particular day
    rev0    = []                                                                                                                                       #Initialize list store reviewers included in pre-processing
    for f in range(len(csv_files)):        
        lp_chk  = 0                                                                                                                                    #Loop check to determine if reviewers were used and went through loop
        if use_local == True:        
            df = pd.read_csv(csv_files[f])                                                                                                             #Read specified csv file
        else:        
            df = load_csv_gcs(in_bucket_name, csv_files[f])                                                                                            #Read specified csv file
        header = list(df.head(0))                                                                                                                      #Create list of header names (used to determine the reviewers of the file)
       
        reviewers0 = reviewers.copy()                                                                                                                  #Remove any bootstrapped labels from pre-processing
        if ptype[f] == 'test' or ptype[f] == 'val':
            try:
                reviewers0.remove('irmod')
            except ValueError:
                pass
            try:
                reviewers0.remove('irvismod')
            except ValueError:
                pass
            try:
                reviewers0.remove('irglmmod')
            except ValueError:
                pass
            try:
                reviewers0.remove('irvisglmmod')
            except ValueError:
                pass

        rev    = []                                                                                                                                    #Initialize list to store reviewers of file
        for r in reviewers0:        
            if (r + '_files') in header:        
                rev.append(header[header.index(r + '_files')])                                                                                         #Append to list if the reviewer listed is in the file
            else:    
                if verbose == True: print(r + ' not found in csv file')    
    
        if len(rev) == 0:                                                                                                                              #Check to make sure at least 1 of reviewers specified at start of program exist in file          
            print('None of the reviewers listed below are in the file??')    
            print(reviewers0)    
        else:          
            dir_path, fb = os.path.split(csv_files[f])                                                                                                 #Extract directory path and file basename
            if use_local == True:    
                bname = None                                                                                                                           #if use local then bucket_name is passed through as None so that files are read locally
            else:    
                bname = in_bucket_name                                                                                                                 #if use local not set then bucket_name is passed through so files are read from google cloud storage
            if use_night == False:    
                df.dropna(subset = ['vis_files'], inplace = True)                                                                                      #Remove missing values  
            rev00 = []
            for l in range(len(rev)):     
                sp = re.split('_', rev[l])                                                                                                             #Split name in order to get reviewer only
                rev00.append(sp[0])
                if rev[l] == 'bedka_files':    
                    df.dropna(subset = [rev[l]], inplace = True)                                                                                       #Remove missing values
                    masks_train = build_masks(df, sp[0] + '_index',                                                                                    #Build array masks to train the updraft labelled data
                                              [os.path.join(dir_path, sp[0], outpat + 'updraft_f.npy')], dims = dims, bucket_name = bname)#,    
                                               #'../../../data/goes-data/labelled/2019137/bedka/updraft_f.npy'])    
                else:    
                    if l == 0:     
                        df.dropna(subset = rev, thresh = nreviewers, inplace = True)                                                                   #Remove missing values from reviewers (must be 2 or more valid reviewers for specified time)
                        if df.empty == False:    
                            masks_train = build_masks(df, sp[0] + '_index',                                                                            #Build array masks to train the updraft labelled data
                                                      [os.path.join(dir_path, sp[0], outpat + 'updraft_f.npy')], dims = dims, bucket_name = bname)#,
                                                       #'../../../data/goes-data/labelled/2019137/bedka/updraft_f.npy'])
                    else:
                        if df.empty == False:
                            masks_train = masks_train + build_masks(df, sp[0] + '_index',                                                              #Build array masks to train the updraft labelled data
                                                                    [os.path.join(dir_path, sp[0], outpat + 'updraft_f.npy')], dims = dims, bucket_name = bname)#,
                                                                   #'../../../data/goes-data/labelled/2019137/bedka/updraft_f.npy'])
                            if (l == (len(rev)-1)):
                                masks_train[masks_train <  nreviewers] = 0                                                                             #Set all mask pixels to zero if at least 2 reviewers do not agree
                                masks_train[masks_train >= nreviewers] = 1                                                                             #Set all mask pixels to 1 if 2 or more reviewers agree
            if df.empty == False:                    
                lp_chk  = 1                                                                                                                            #Set loop check to 1 to show if reviewers were used and went through loop
                masks_train = np.expand_dims(masks_train, axis=3)
                date_time   = list(df['date_time'])                                                                                                    #Extract list of dates used to be input into model
                if use_glm == True:
                    tens = build_imgs(df, [('ir_index', os.path.join(dir_path, outpat + 'ir.npy')),                                                             #Build 2000x2000 tensor array that will be subsetted
                                           ('glm_index',os.path.join(dir_path, outpat + 'glm.npy'))], dims = dims, bucket_name = bname)
                else:
                    if use_irdiff == True:
                        tens = build_imgs(df, [('ir_index', os.path.join(dir_path, outpat + 'ir.npy')),                                                         #Build 2000x2000 tensor array that will be subsetted
                                               ('irdiff_index',os.path.join(dir_path, outpat + 'irdiff.npy'))], dims = dims, bucket_name = bname)
                    
                    else:
                        tens = build_imgs(df, [('ir_index', os.path.join(dir_path, outpat + 'ir.npy')),                                                         #Build 2000x2000 tensor array that will be subsetted
                                               ('vis_index',os.path.join(dir_path, outpat + 'vis.npy'))], dims = dims, bucket_name = bname)
                if verbose == True:
                    print('Tensor shape = ' + str(tens.shape))
                    print('Tensor max   = ' + str(np.max(tens)))
                    print('Masks shape = ' + str(masks_train.shape))
                img_size2 = str(tens.shape[1])
                print('Size of image tensor = ' + img_size2)
                if (masks_train.shape[0] != tens.shape[0]):
                    print('0th dimension array sizes do not match!!?? Exiting program.')
                    exit()
                if (by_updraft == True):   
                    dat_str = os.path.basename(os.path.split(dir_path)[0])                                                                             #Extract date string. Used to output files by_date
                    outdate.append(dat_str)
                    sector0 = os.path.basename(dir_path)
                    sector.append(sector0)
                    rev0.append(', '.join(rev00))
                    counter         = 0
                    centroid_inds_x = []                                                                                                               #Initialize list to store the centroid indices
                    centroid_inds_y = []                                                                                                               #Initialize list to store the centroid indices
                    date_time2      = []                                                                                                               #Initialize list to store the date times of each centroid
                    if min_obj_size != None:
                        for m in range(masks_train.shape[0]):
                            masks_train0 = masks_train[m, :, :]                                                                                        #Extract mask for specified time in day
                            if (np.max(masks_train0) > 0):
                                labeled_array, num_updrafts = label(masks_train0[:, :, 0], structure = s)                                              #Extract labels for each updraft location (make sure points at diagonals are part of same updrafts)
                                for u in range(num_updrafts):
                                    if np.sum(labeled_array == u+1) < min_obj_size:
                                        masks_train[m, (labeled_array == u+1), :] = 0
                    m = 0
                    for m in range(masks_train.shape[0]):
                        masks_train0 = masks_train[m, :, :]                                                                                            #Extract mask for specified time in day
                        if (np.max(masks_train0) > 0):
                            labeled_array, num_updrafts = label(masks_train0[:, :, 0], structure = s)                                                  #Extract labels for each updraft location (make sure points at diagonals are part of same updrafts)
                            for u in range(num_updrafts):
                                inds   = np.where(labeled_array == u+1)                                                                                #Find locations of updraft region mask
                                x_arr  = list(inds[0])                                                                                                 #Find x indices of updraft
                                y_arr  = list(inds[1])                                                                                                 #Find y indices of updraft
                                x_arr0 = int(round(np.mean(x_arr)))                                                                                    #Round the mean point to get center location
                                y_arr0 = int(round(np.mean(y_arr)))                                                                                    #Round the mean point to get center location
                                if x_arr0 > (int(img_size/2)+1) and y_arr0 > (int(img_size/2)+1) and x_arr0 < (masks_train0.shape[0]-(int(img_size/2)+1)) and y_arr0 < (masks_train0.shape[1]-(int(img_size/2)+1)): 
                                    centroid_inds_x.append(x_arr0)                                                                                     #Save 1D centroid index to list
                                    centroid_inds_y.append(y_arr0)                                                                                     #Save 1D centroid index to list
                                    date_time2.append(date_time[m])                                                                                    #Save date of updraft to list
                                    if counter == 0:
                                        imgs    =        tens[m, x_arr0-int(img_size/2):x_arr0+int(img_size/2), y_arr0-int(img_size/2):y_arr0+int(img_size/2), :]  #Extract tensor for subset region
                                        mask    = masks_train[m, x_arr0-int(img_size/2):x_arr0+int(img_size/2), y_arr0-int(img_size/2):y_arr0+int(img_size/2), :]  #Extract mask training for subset region
                                        imgs    = np.reshape(imgs, (1, imgs.shape[0], imgs.shape[1], imgs.shape[2]))
                                        mask    = np.reshape(mask, (1, mask.shape[0], mask.shape[1], mask.shape[2]))
                                        counter = counter+1
                                    else:     
                                        tens0 = tens[m, x_arr0-int(img_size/2):x_arr0+int(img_size/2), y_arr0-int(img_size/2):y_arr0+int(img_size/2), :]
                                        mask0 = masks_train[m, x_arr0-int(img_size/2):x_arr0+int(img_size/2), y_arr0-int(img_size/2):y_arr0+int(img_size/2), :]
                                        imgs = np.append(imgs, np.reshape(tens0, (1, tens0.shape[0], tens0.shape[1], tens0.shape[2])), axis = 0)       #Subset 2000x2000 domain into 64x64 regions around the updrafts for training, validation, and testing
                                        mask = np.append(mask, np.reshape(mask0, (1, mask0.shape[0], mask0.shape[1], mask0.shape[2])), axis = 0)       #Subset 2000x2000 domain into 64x64 regions around the updrafts for training, validation, and testing
        
                    if verbose == True:
                        print('Subset Tensor shape = ' + str(imgs.shape))
                        print('Subset Tensor max   = ' + str(np.max(imgs)))
                        print('Subset Mask shape   = ' + str(mask.shape))
                        print('Subset Mask max     = ' + str(np.max(mask)))
                       
                    os.makedirs(os.path.join(outroot, dat_str, sector0), exist_ok = True)                                                              #Create output directory if it does not already exist
                    dim = [img_size]*len(centroid_inds_x)                                                                                              #List to store the image size the pre-processing was done on
                    df_out = pd.DataFrame({'date_time': date_time2, 'centroid_index_x': centroid_inds_x, 'centroid_index_y': centroid_inds_y, 'box_size': dim})#Create dataframe of all the centroid of each object along with the dates and size of the dimensions
                    df_out.to_csv(os.path.join(outroot, dat_str, sector0, 'blob_mask_positioning.csv'))                                                #Write output csv file
                    if write_gcs == True:                        
                        pref = re.split('labelled/', os.path.join(outroot, dat_str, sector0))[1]
                        write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'blob_mask_positioning.csv'),  del_local = del_local)
                    
                    if (f < n_trdates):                                     
                        ftype.append('train')
                        np.save(os.path.join(outroot, dat_str, sector0, 'imgs_train_' + str(img_size) + '.npy'),  imgs)                                #Save model training of images to numpy files
                        np.save(os.path.join(outroot, dat_str, sector0, 'masks_train_' + str(img_size) + '.npy'), mask)                                #Save model training of mask to numpy files
                        if write_gcs == True:
                            pref = re.split('labelled/', os.path.join(outroot, dat_str, sector0))[1]
                            write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'imgs_train_' + str(img_size) + '.npy'),  del_local = del_local)
                            write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'masks_train_' + str(img_size) + '.npy'), del_local = del_local)
                        if full_domain == True: 
                            os.makedirs(os.path.join(outroot0, dat_str, sector0), exist_ok = True)                                                     #Create output directory if it does not already exist
                            np.save(os.path.join(outroot0, dat_str, sector0, 'imgs_train_' + str(img_size2) + '.npy'),  tens)                          #Save model training of images to numpy files
                            np.save(os.path.join(outroot0, dat_str, sector0, 'masks_train_' + str(img_size2) + '.npy'), masks_train)                   #Save model training of mask to numpy files
                            if write_gcs == True:
                                pref = re.split('labelled/', os.path.join(outroot0, dat_str, sector0))[1]
                                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot0, dat_str, sector0, 'imgs_train_' + str(img_size2) + '.npy'),  del_local = del_local)
                                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot0, dat_str, sector0, 'masks_train_' + str(img_size2) + '.npy'), del_local = del_local)
                    elif (f < (n_trdates+n_vdates)):                                                  
                        ftype.append('val')
                        np.save(os.path.join(outroot, dat_str, sector0, 'imgs_val_' + str(img_size) + '.npy'),  imgs)                                  #Save model validation of images to numpy files
                        np.save(os.path.join(outroot, dat_str, sector0, 'masks_val_' + str(img_size) + '.npy'), mask)                                  #Save model validation of mask to numpy files
                        if write_gcs == True:
                            pref = re.split('labelled/', os.path.join(outroot, dat_str, sector0))[1]
                            write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'imgs_val_' + str(img_size) + '.npy'),  del_local = del_local)
                            write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'masks_val_' + str(img_size) + '.npy'), del_local = del_local)
                        if full_domain == True: 
                            os.makedirs(os.path.join(outroot0, dat_str, sector0), exist_ok = True)                                                     #Create output directory if it does not already exist
                            np.save(os.path.join(outroot0, dat_str, sector0, 'imgs_val_' + str(img_size2) + '.npy'),  tens)                            #Save model training of images to numpy files
                            np.save(os.path.join(outroot0, dat_str, sector0, 'masks_val_' + str(img_size2) + '.npy'), masks_train)                     #Save model training of mask to numpy files
                            if write_gcs == True:
                                pref = re.split('labelled/', os.path.join(outroot0, dat_str, sector0))[1]
                                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot0, dat_str, sector0, 'imgs_val_' + str(img_size2) + '.npy'),  del_local = del_local)
                                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot0, dat_str, sector0, 'masks_val_' + str(img_size2) + '.npy'), del_local = del_local)
                    else:                                                     
                        ftype.append('test')
                        np.save(os.path.join(outroot, dat_str, sector0, 'imgs_test_' + str(img_size) + '.npy'),  imgs)                                 #Save model testing of images to numpy files
                        np.save(os.path.join(outroot, dat_str, sector0, 'masks_test_' + str(img_size) + '.npy'), mask)                                 #Save model testing of mask to numpy files
                        if write_gcs == True:
                            pref = re.split('labelled/', os.path.join(outroot, dat_str, sector0))[1]
                            write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'imgs_test_' + str(img_size) + '.npy'),  del_local = del_local)
                            write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'masks_test_' + str(img_size) + '.npy'), del_local = del_local)
                        if full_domain == True: 
                            os.makedirs(os.path.join(outroot0, dat_str, sector0), exist_ok = True)                                                     #Create output directory if it does not already exist
                            np.save(os.path.join(outroot0, dat_str, sector0, 'imgs_test_' + str(img_size2) + '.npy'),  tens)                           #Save model training of images to numpy files
                            np.save(os.path.join(outroot0, dat_str, sector0, 'masks_test_' + str(img_size2) + '.npy'), masks_train)                    #Save model training of mask to numpy files
                            if write_gcs == True:
                                pref = re.split('labelled/', os.path.join(outroot0, dat_str, sector0))[1]
                                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot0, dat_str, sector0, 'imgs_test_' + str(img_size2) + '.npy'),  del_local = del_local)
                                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot0, dat_str, sector0, 'masks_test_' + str(img_size2) + '.npy'), del_local = del_local)
                else:
                    index0.append(tens.shape[0])                                                                                                       #Append list by storing the number of cases reviewed for specified date
                    if subset == False:
                        if f == 0 or by_date == True:
                            imgs = tens
                            mask = masks_train
                        else:
                            imgs = np.append(imgs, tens, axis = 0)
                            mask = np.append(mask, masks_train, axis = 0)
                    else:
                        dat_str = os.path.basename(os.path.split(dir_path)[0])
                        outdate.append(dat_str)
                        sector0 = os.path.basename(dir_path)
                        sector.append(sector0)
                        rev0.append(', '.join(rev00))
                        inds    = extract_subset_region(dat_str, sector = sector0)                                                                     #Extract x and y indices at which to subset 2000x2000 element domain for specified date
                        inds0   = inds[dat_str]
                        if verbose == True:
                            print('Date    = ' + dat_str)
                            print('Indices = ' + str(inds0))
                        if f == 0 or by_date == True:
                            imgs = tens[:, inds0[0]:inds0[2], inds0[1]:inds0[3], :]                                                                    #Extract tensor for subset region
                            mask = masks_train[:, inds0[0]:inds0[2], inds0[1]:inds0[3], :]                                                             #Extract mask training for subset region
                        else: 
                            imgs = np.append(imgs,        tens[:, inds0[0]:inds0[2], inds0[1]:inds0[3], :], axis = 0)                                  #Extract tensor for subset region and append to images array
                            mask = np.append(mask, masks_train[:, inds0[0]:inds0[2], inds0[1]:inds0[3], :], axis = 0)                                  #Extract mask training for subset region and append to images array
                    
                    if by_date == True:
                        if subset == False:
                            imgs_512 = build_subset_tensor(imgs_train, img_size)                                                                       #Subset using provided indices for training and validation
                            mask_512 = build_subset_tensor(imgs_val,   img_size)                                                                       #Subset using provided indices for training and validation
                        else:
                            imgs_512 = imgs                                                                                                            #Subset entire 2000x2000 domain for training and validation
                            mask_512 = mask                                                                                                            #Subset entire 2000x2000 domain for training and validation
                        
                        if verbose == True:
                            print('Subset Tensor shape = ' + str(imgs_512.shape))
                            print('Subset Tensor max   = ' + str(np.max(imgs_512)))
                            print('Subset Mask shape   = ' + str(mask_512.shape))
                            print('Subset Mask max     = ' + str(np.max(mask_512)))
                       
                        os.makedirs(os.path.join(outroot, dat_str, sector0), exist_ok = True)                                                          #Create output directory if it does not already exist
                        if (f < n_trdates):                                     
                            ftype.append('train')
                            np.save(os.path.join(outroot, dat_str, sector0, 'imgs_train_' + str(img_size) + '.npy'), imgs_512)                         #Save model training of images to numpy files
                            np.save(os.path.join(outroot, dat_str, sector0, 'masks_train_' + str(img_size) + '.npy'), mask_512)                        #Save model training of mask to numpy files
                            if write_gcs == True:
                                pref = re.split('labelled/', os.path.join(outroot, dat_str, sector0))[1]
                                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'imgs_train_' + str(img_size) + '.npy'),  del_local = del_local)
                                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'masks_train_' + str(img_size) + '.npy'), del_local = del_local)
                        elif (f < (n_trdates+n_vdates)):                                     
                            ftype.append('val')
                            np.save(os.path.join(outroot, dat_str, sector0, 'imgs_val_' + str(img_size) + '.npy'), imgs_512)                           #Save model validation of images to numpy files
                            np.save(os.path.join(outroot, dat_str, sector0, 'masks_val_' + str(img_size) + '.npy'), mask_512)                          #Save model validation of mask to numpy files
                            if write_gcs == True:
                                pref = re.split('labelled/', os.path.join(outroot, dat_str, sector0))[1]
                                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'imgs_val_' + str(img_size) + '.npy'),  del_local = del_local)
                                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'masks_val_' + str(img_size) + '.npy'), del_local = del_local)
                        else:                                        
                            ftype.append('test')
                            np.save(os.path.join(outroot, dat_str, sector0, 'imgs_test_' + str(img_size) + '.npy'), imgs_512)                          #Save model testing of images to numpy files
                            np.save(os.path.join(outroot, dat_str, sector0, 'masks_test_' + str(img_size) + '.npy'), mask_512)                         #Save model testing of mask to numpy files
                            if write_gcs == True:
                                pref = re.split('labelled/', os.path.join(outroot, dat_str, sector0))[1]
                                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'imgs_test_' + str(img_size) + '.npy'),  del_local = del_local)
                                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'masks_test_' + str(img_size) + '.npy'), del_local = del_local)
           
                #reset variables in order to save RAM
                if by_updraft == False:
                    imgs        = None
                    masks_train = None
                    tens        = None
                    imgs_512    = None
                    mask_512    = None
                    tens0       = None
                    mask0       = None
                    mask        = None
    if ((proc_date_str == None) or (proc_date_str != None and lp_chk == 1)):
        df_out   = pd.DataFrame({'out_date'   : outdate}) 
        df_ftype = pd.DataFrame({'ftype'      : ftype}) 
        df_sec   = pd.DataFrame({'sector'     : sector}) 
        df_rev   = pd.DataFrame({'reviewers'  : rev0})
        df_nrev  = pd.DataFrame({'nreviewers' : [nreviewers]})
        df_night = pd.DataFrame({'use_night'  : [use_night]})
        df_merge = pd.concat([df_out, df_sec, df_ftype, df_rev, df_nrev, df_night], axis = 1, join = 'inner')
        if proc_date_str != None:
            if use_local == True:
                if os.path.isfile(os.path.join(outroot, 'dates_with_ftype.csv')):
                    df2 = pd.read_csv(csv_files[f])                                                                                                    #Read ftype csv file that already exists
                    if dat_str in list(df2['out_date']):    
                        if sector0 in df2['sector'][list(df2['out_date']).index(dat_str)]:    
                            print('Date and sector already written to csv file')    
                        else:    
                            df2 = pd.concat([df2, df_merge], axis = 0, join = 'inner', sort = 'out_date')                                              #Concatenate the dataframes and sort by date 
                            df2 = df2.astype({'out_date' : str})
                            df2.sort_values(by = 'out_date', axis =0, inplace = True)
                            df2.drop_duplicates(inplace = True)
                            df2.reset_index(inplace = True)
                            header = list(df2.head(0))                                                                                                 #Extract header of csv file
                            if 'Unnamed: 0' in header:
                              df2.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                 #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
                            if 'index' in header:
                              df2.drop(columns = 'index', axis = 'columns', inplace = True)                                                            #Remove index column that was created by reset_index.
#deprecated
#                            df2 = df2.append(df_merge, ignore_index = True)                                                                            #Append previous dates ftype dates csv file with new date
#                            df2.drop('Unnamed: 0', axis = 'columns', inplace = True)    
                            df_merge = df2        
                    else:    
                        df2 = pd.concat([df2, df_merge], axis = 0, join = 'inner', sort = 'out_date')                                                  #Concatenate the dataframes and sort by date 
                        df2 = df2.astype({'out_date' : str})
                        df2.sort_values(by = 'out_date', axis =0, inplace = True)
                        df2.drop_duplicates(inplace = True)
                        df2.reset_index(inplace = True)
                        header = list(df2.head(0))                                                                                                     #Extract header of csv file
                        if 'Unnamed: 0' in header:
                          df2.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                     #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
                        if 'index' in header:
                          df2.drop(columns = 'index', axis = 'columns', inplace = True)                                                                #Remove index column that was created by reset_index.
#deprecated
#                        df2 = df2.append(df_merge, ignore_index = True)                                                                                #Append previous dates ftype dates csv file with new date
#                        df2.drop('Unnamed: 0', axis = 'columns', inplace = True)    
                        df_merge = df2        
            else:        
                pref  = re.split('labelled/', outroot)[1]        
                exist = list_csv_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', 'dates_with_ftype.csv')        
                if len(exist) == 1:        
                    df2 = load_csv_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/dates_with_ftype.csv')                                            #Read specified csv file
                    if dat_str in list(df2['out_date']):    
                        if sector0 in df2['sector'][list(df2['out_date']).index(dat_str)]:    
                            print('Date and sector already written to csv file. Not writing again.')    
                        else:    
                            df2 = pd.concat([df2, df_merge], axis = 0, join = 'inner', sort = 'out_date')                                              #Concatenate the dataframes and sort by date 
                            df2 = df2.astype({'out_date' : str})
                            df2.sort_values(by = 'out_date', axis =0, inplace = True)
                            df2.drop_duplicates(inplace = True)
                            df2.reset_index(inplace = True)
                            header = list(df2.head(0))                                                                                                 #Extract header of csv file
                            if 'Unnamed: 0' in header:
                              df2.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                 #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
                            if 'index' in header:
                              df2.drop(columns = 'index', axis = 'columns', inplace = True)                                                            #Remove index column that was created by reset_index.
#deprecated
#                            df2 = df2.append(df_merge, ignore_index = True)                                                                            #Append previous dates ftype dates csv file with new date
#                            df2.drop('Unnamed: 0', axis = 'columns', inplace = True)    
                            df_merge = df2        
                    else:    
                        df2 = pd.concat([df2, df_merge], axis = 0, join = 'inner', sort = 'out_date')                                                  #Concatenate the dataframes and sort by date 
                        df2 = df2.astype({'out_date' : str})
                        df2.sort_values(by = 'out_date', axis =0, inplace = True)
                        df2.drop_duplicates(inplace = True)
                        df2.reset_index(inplace = True)
                        header = list(df2.head(0))                                                                                                     #Extract header of csv file
                        if 'Unnamed: 0' in header:
                          df2.drop('Unnamed: 0', axis = 'columns', inplace = True)                                                                     #Remove 'Unnamed: 0' column from dataframe. This is created when appending data frames.
                        if 'index' in header:
                          df2.drop(columns = 'index', axis = 'columns', inplace = True)                                                                #Remove index column that was created by reset_index.
#deprecated
#                        df2 = df2.append(df_merge, ignore_index = True)                                                                                #Append previous dates ftype dates csv file with new date
#                        df2.drop('Unnamed: 0', axis = 'columns', inplace = True)    
                        df_merge = df2        
        df_merge.to_csv(os.path.join(outroot, 'dates_with_ftype.csv'))                                                                                 #Write output file
        if write_gcs == True:         
            pref = re.split('labelled/', outroot)[1]         
            write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, 'dates_with_ftype.csv'))         
             
    if subset == True and by_updraft == False:         
        if len(index0) != len(csv_files):         
            print('Size of index0 variable not the same as number of variables??')         
            exit()         
#STILL NEEDS WORK!!!                
#         imgs_val   = imgs[int(sum(int(index0[:len(index0)/2)])):,:,:,:]                                                                              #Extract 20% for model validation
#         imgs_train = imgs[:int(sum(int(index0[:len(index0)/2)])),:,:,:]                                                                              #Extract first half of days for training
#         imgs_train_512 = build_subset_tensor(imgs_train, 512)                                                                                        #Build 
#         imgs_val_512   = build_subset_tensor(imgs_val, 512)           
#         masks_val = masks_train[int(masks_train.shape[0] * 0.75):,:,:,:]         
#         masks_train = masks_train[:int(masks_train.shape[0] * 0.75),:,:,:]         
#         masks_val_512 = build_subset_tensor(masks_val, 512)         
#         masks_train_512 = build_subset_tensor(masks_train, 512)         
#         if verbose == True:         
#             print('Training image shape: ' + str(imgs_train_512.shape))         
#             print('Training mask shape: ' + str(masks_train_512.shape))         
#             print('Validation image shape: ' + str(imgs_val_512.shape))         
#             print('Validation mask shape: ' + str(masks_val_512.shape))         
         
    else:         
        if by_date == False:         
            _75 = math.floor(imgs.shape[0] * 0.75)         
            _95 = math.floor(imgs.shape[0] * 0.95)         
            imgs_train  = imgs[:_75, :, :, :]                                                                                                          #Extract first 75% for model training of images
            masks_train = mask[:_75, :, :, :]                                                                                                          #Extract first 75% for model training of updraft mask
            imgs_val    = imgs[_75:_95, :, :, :]                                                                                                       #Extract next 20% for model validation of images
            masks_val   = mask[_75:_95, :, :, :]                                                                                                       #Extract next 20% for model validation of mask
            imgs_test   = imgs[_95:, :, :, :]                                                                                                          #Extract last 5% for model testing of images
            masks_test  = mask[_95:, :, :, :]                                                                                                          #Extract last 5% for model testing of mask
            if (imgs_train.shape[0] + imgs_val.shape[0] + imgs_test.shape[0]) != imgs.shape[0]:         
                print('Subscripting uszing floor did not work as planned!')         
            if verbose == True:         
                print('Training image shape: ' + str(imgs_train.shape))                        
                print('Training mask shape: ' + str(masks_train.shape))         
                print('Validation image shape: ' + str(imgs_val.shape))         
                print('Validation mask shape: ' + str(masks_val.shape))         
                print('Testing image shape: ' + str(imgs_test.shape))         
                print('Testing mask shape: ' + str(masks_test.shape))               
            os.makedirs(os.path.join(outroot), exist_ok = True)                                                                                        #Create output directory if it does not already exist
            np.save(os.path.join(outroot, 'imgs_val_' + str(img_size) + '.npy'),   imgs_val)                                                           #Save model validation of images to numpy files
            np.save(os.path.join(outroot, 'imgs_train_' + str(img_size) + '.npy'), imgs_train)                                                         #Save model training of images to numpy files
            np.save(os.path.join(outroot, 'imgs_test_' + str(img_size) + '.npy'),  imgs_test)                                                          #Save model testing of images to numpy files
            np.save(os.path.join(outroot, 'masks_train_' + str(img_size) + '.npy'), masks_train)                                                       #Save model testing of masks to numpy files
            np.save(os.path.join(outroot, 'masks_val_' + str(img_size) + '.npy'), masks_val)                                                           #Save model testing of masks to numpy files
            np.save(os.path.join(outroot, 'masks_test_' + str(img_size) + '.npy'), masks_test)                                                         #Save model testing of masks to numpy files
            if write_gcs == True:
                pref = re.split('labelled/', os.path.join(outroot, dat_str, sector0))[1]
                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'imgs_val_' + str(img_size) + '.npy'),    del_local = del_local)
                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'imgs_train_' + str(img_size) + '.npy'),  del_local = del_local)
                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'imgs_test_' + str(img_size) + '.npy'),   del_local = del_local)
                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'masks_train_' + str(img_size) + '.npy'), del_local = del_local)
                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'masks_val_' + str(img_size) + '.npy'),   del_local = del_local)
                write_to_gcs(out_bucket_name, 'Cooney_testing/' + pref + '/', os.path.join(outroot, dat_str, sector0, 'masks_test_' + str(img_size) + '.npy'),  del_local = del_local)
                    
def build_imgs(df, chan_list, dims = [2000, 2000], bucket_name = None):
    '''
    Builds image files with the specified channels in the original picture shape of the image.
    Args:
        df       : dataframe from control file with indices to the numpy matrices in the original imagery
        chan_list: the list of channels over which we iterate to build the matrices (i.e. vis, ir, glm, etc.)
    Keywords:
        dims        : x and y dimensions of array to build images for. DEFAULT = [2000, 2000]
        bucket_name : STRING GCP bucket name to load numpy files from.
                      DEFAULT = None -> use locally stored data.    
    Returns:
        imgs: the image matrix with the desired channels
    '''
    imgs = np.zeros([df.shape[0], dims[0], dims[1], len(chan_list)], dtype=np.float32)
    for idx, n in enumerate(chan_list):
        if bucket_name == None:
            if len(df) == 1:
                imgs[:,:,:,idx] = np.load(n[1])[:,:,:].astype(np.float32)
            else:   
                imgs[:,:,:,idx] = np.load(n[1])[df[n[0]].astype(int).values,:,:].astype(np.float32)
        else:
            if len(df) == 1:
                imgs[:,:,:,idx] = load_npy_gcs(bucket_name, n[1])[:,:,:].astype(np.float32)
            else:
                imgs[:,:,:,idx] = load_npy_gcs(bucket_name, n[1])[df[n[0]].astype(int).values,:,:].astype(np.float32)

    return(imgs)
    
def build_masks(df, mask_idx, masks_list, dims = [2000, 2000], bucket_name = None):
    '''
    Builds masks which have been hand-labeled by SMEs for AACP detection.
    This function will combine different masks that are listed in the masks_list.
    Args:
        df: dataframe from control file with indices to the SME labels of interest in stored numpy files
        mask_idx: the name of the labeling SME's column index in df
        masks_list: a list of npy files which will be almagated into a single mask
    Returns:
        masks: the mask matrix with combined mask labels
    '''
    df2 = df
    df2.reset_index(inplace=True)
    df2.drop('index', axis = 'columns', inplace = True)
    df2.reset_index(inplace=True)
    masks = np.zeros([df.shape[0], dims[0], dims[1]], dtype=np.float32)
    temp_masks = np.zeros([df.shape[0],  dims[0], dims[1]], dtype=np.float32)
    for n in masks_list:
#        temp_masks = np.load(n).astype(np.float32)[df[mask_idx].astype(int).values,:,:]
        if bucket_name == None:
            temp_masks[df2.dropna(subset = [mask_idx])['index'].astype(int).values] = np.load(n).astype(np.float32)[df2.dropna(subset = [mask_idx])[mask_idx].astype(int).values, :, :]
#            temp_masks[df.dropna(subset = [mask_idx])['Unnamed: 0'].astype(int).values - df.dropna(subset = [mask_idx])['Unnamed: 0'].astype(int).values[0]] = np.load(n).astype(np.float32)[df.dropna(subset = [mask_idx])[mask_idx].astype(int).values, :, :]
        else:
            temp_masks[df2.dropna(subset = [mask_idx])['index'].astype(int).values] = load_npy_gcs(bucket_name, n).astype(np.float32)[df2.dropna(subset = [mask_idx])[mask_idx].astype(int).values, :, :]
#            temp_masks[df.dropna(subset = [mask_idx])['Unnamed: 0'].astype(int).values - df.dropna(subset = [mask_idx])['Unnamed: 0'].astype(int).values[0]] = load_npy_gcs(bucket_name, n).astype(np.float32)[df.dropna(subset = [mask_idx])[mask_idx].astype(int).values, :, :]
        masks = np.maximum(masks, temp_masks)
    
    df.drop('index', axis = 'columns', inplace = True)
    return(masks)

# def build_subset_tensor(orig_tens, new_dim):
#     '''
#     Reshapes a tensor into smaller dimensions and more rows for easier ML training (not overwhelming the GPU in matrix size).
#     NOTE: this function will possibly have overlap in the original imagery if the new dimension size is not fully divisible into the original image matrix size.
#     Args:
#         orig_tens: the tensor which will be reshaped
#         new_dim: the new tensor x and y matrix dimension
#     Returns:
#         new_tens: the reshaped tensor
#     '''
#     iter_cnt = ceil(orig_tens.shape[1] / new_dim)
#     new_tens = np.zeros([orig_tens.shape[0] * iter_cnt**2,new_dim,new_dim,orig_tens.shape[3]], dtype=np.float32)
#     vert_idx = 0
#     for n in range(iter_cnt):
#         for m in range(iter_cnt):
#             #print(str(n) + ',' + str(m))
#             new_tens[vert_idx*orig_tens.shape[0]:
#                      vert_idx*orig_tens.shape[0] + orig_tens.shape[0],:,:,:] = orig_tens[:,min((n+1)*new_dim,orig_tens.shape[1])-new_dim:min((n+1)*new_dim,orig_tens.shape[1]),
#                                                                                          min((m+1)*new_dim,orig_tens.shape[1])-new_dim:min((m+1)*new_dim,orig_tens.shape[1]),:]
#             vert_idx += 1
#     return(new_tens)
def build_subset_tensor(orig_tens, new_dim_x, new_dim_y):
    '''
    Reshapes a tensor into smaller dimensions and more rows for easier ML training (not overwhelming the GPU in matrix size).
    NOTE: this function will possibly have overlap in the original imagery if the new dimension size is not fully divisible into the original image matrix size.
    Args:
        orig_tens: the tensor which will be reshaped
        new_dim_x: the new tensor x matrix dimension
        new_dim_y: the new tensor y matrix dimension
    Returns:
        new_tens: the reshaped tensor
    '''
 #   iter_cnt = ceil(orig_tens.shape[1] / new_dim)
    
    iter_cnt_x = ceil(orig_tens.shape[1]/new_dim_x)
    iter_cnt_y = ceil(orig_tens.shape[2]/new_dim_y)
    new_tens = np.zeros([orig_tens.shape[0] * (iter_cnt_x*iter_cnt_y),new_dim_x,new_dim_y,orig_tens.shape[3]], dtype=np.float32)
    vert_idx = 0
    for n in range(iter_cnt_x):
        for m in range(iter_cnt_y):
            #print(str(n) + ',' + str(m))
            new_tens[vert_idx*orig_tens.shape[0]:
                     vert_idx*orig_tens.shape[0] + orig_tens.shape[0],:,:,:] = orig_tens[:,min((n+1)*new_dim_x,orig_tens.shape[1])-new_dim_x:min((n+1)*new_dim_x,orig_tens.shape[1]),
                                                                                         min((m+1)*new_dim_y,orig_tens.shape[2])-new_dim_y:min((m+1)*new_dim_y,orig_tens.shape[2]),:]
            vert_idx += 1
    return(new_tens)

def build_subset_tensor2(orig_tens, new_dim_x, new_dim_y):
    '''
    Reshapes a tensor into smaller dimensions and more rows for easier ML training (not overwhelming the GPU in matrix size).
    NOTE: this function will have overlap in the original imagery and each box is shifted by 128 pixels with only the inner 364x364 pixels retained. 
    Args:
        orig_tens: the tensor which will be reshaped
        new_dim_x: the new tensor x matrix dimension
        new_dim_y: the new tensor y matrix dimension
    Returns:
        new_tens: the reshaped tensor
    '''
    orig_tens2 = np.zeros([orig_tens.shape[0], orig_tens.shape[1]+256, orig_tens.shape[2]+256, orig_tens.shape[3]])
    for n in range(orig_tens.shape[0]):
      for m in range(orig_tens.shape[3]):
        orig_tens2[n, :, :, m] = np.pad(orig_tens[n, :, :, m], 128, mode = 'reflect')
    orig_tens  = orig_tens2
    iter_cnt_x = ceil(orig_tens.shape[1]/128)
    iter_cnt_y = ceil(orig_tens.shape[2]/128)
    new_tens = np.zeros([orig_tens.shape[0] * (iter_cnt_x*iter_cnt_y),new_dim_x,new_dim_y,orig_tens.shape[3]], dtype=np.float32)
    vert_idx = 0
    for n in range(iter_cnt_x):
        for m in range(iter_cnt_y):
            new_tens[vert_idx*orig_tens.shape[0]:vert_idx*orig_tens.shape[0] + orig_tens.shape[0],:,:,:] = orig_tens[:,max(0, min((n+1)*128,orig_tens.shape[1])-new_dim_x):max(0, min((n+1)*128,orig_tens.shape[1])-new_dim_x) + new_dim_x, max(0, min((m+1)*128,orig_tens.shape[2])-new_dim_y):max(0, min((m+1)*128,orig_tens.shape[2])-new_dim_y) + new_dim_y, :]
            vert_idx += 1
    return(new_tens)

def build_subset_tensor3(orig_tens, new_dim_x, new_dim_y):
    '''
    Reshapes a tensor into smaller dimensions and more rows for easier ML training (not overwhelming the GPU in matrix size).
    NOTE: this function will have overlap in the original imagery and each box is shifted by 128 pixels with only the inner 364x364 pixels retained. 
    Args:
        orig_tens: the tensor which will be reshaped
        new_dim_x: the new tensor x matrix dimension
        new_dim_y: the new tensor y matrix dimension
    Returns:
        new_tens: the reshaped tensor
    '''
    chunks = OverlapChunks( orig_tens, [orig_tens.shape[0], new_dim_x, new_dim_y, orig_tens.shape[3]], [0, 64, 64, 0] )

    for c, chunk in enumerate(chunks.split()):
      if c == 0:
        new_tens = np.empty( (np.product(chunks.nChunks), *chunk.shape[1:]), dtype=chunk.dtype )
      new_tens[c] = chunk
    
    return(new_tens, chunks)

def reconstruct_tensor_from_subset(tens, og_dim_x, og_dim_y):
    '''
    Reconstruct a tensor that was made into smaller dimensions and more rows for easier ML training (not overwhelming the GPU in matrix size).
    NOTE: this function will need to account for the possibility of the sunset tensor overlapped the original imagery if the new dimension size was 
          not fully divisible into the original image matrix size.
    Args:
        tens     : the tensor which will be reshaped
        og_dim_x : the old tensor x matrix dimension
        og_dim_y : the old tensor y matrix dimension
    Returns:
        new_tens: the reconstructed tensor
    '''
    iter_cnt_x = ceil(og_dim_x/tens.shape[1])
    iter_cnt_y = ceil(og_dim_y/tens.shape[2])
    new_tens = np.zeros([tens.shape[0] // (iter_cnt_x*iter_cnt_y), og_dim_x, og_dim_y, tens.shape[3]], dtype=np.float32)                #Create an array to store the reconstructed tensor
    vert_idx = 0                                                                                                                        #Initialize index counter
    for n in range(iter_cnt_x):
        for m in range(iter_cnt_y):
            new_tens[:,min((n+1)*tens.shape[1],og_dim_x)-tens.shape[1]:min((n+1)*tens.shape[1],og_dim_x),
                     min((m+1)*tens.shape[2],og_dim_y)-tens.shape[1]:min((m+1)*tens.shape[2],og_dim_y),:] = tens[vert_idx * tens.shape[0] // (iter_cnt_x*iter_cnt_y):(vert_idx * tens.shape[0] // (iter_cnt_x*iter_cnt_y)) + tens.shape[0] // (iter_cnt_x*iter_cnt_y),:,:,:]
            vert_idx += 1                                                                                                               #Iterate index by 1

    return(new_tens)

def reconstruct_tensor_from_subset2(tens, og_dim_x, og_dim_y):
    '''
    Reconstruct a tensor that was made into smaller dimensions and more rows for easier ML training (not overwhelming the GPU in matrix size).
    NOTE: this function accounts for the sunset tensor overlapped in the original imagery every 128 pixels
    Args:
        tens     : the tensor which will be reshaped
        og_dim_x : the old tensor padded by 128 pixels in each x matrix dimension
        og_dim_y : the old tensor padded by 128 pixels in each y matrix dimension
    Returns:
        new_tens: the reconstructed tensor
    '''
    iter_cnt_x = ceil(og_dim_x/128)
    iter_cnt_y = ceil(og_dim_y/128)
    new_tens = np.zeros([tens.shape[0] // (iter_cnt_x*iter_cnt_y), og_dim_x, og_dim_y, tens.shape[3]], dtype=np.float32)                #Create an array to store the reconstructed tensor
    vert_idx = 0                                                                                                                        #Initialize index counter
    for n in range(iter_cnt_x):
        for m in range(iter_cnt_y):
            new_tens[:, max(128, min(n*128,og_dim_x)-(tens.shape[1]-128)):max(128, min((n+1)*128,og_dim_x)-tens.shape[1]) + (tens.shape[1]-128), 
            max(128, min(m*128,og_dim_y)-(tens.shape[2]-128)):max(128, min((m+1)*128,og_dim_y)-tens.shape[2]) + (tens.shape[2]-128), :] = tens[vert_idx * tens.shape[0] // (iter_cnt_x*iter_cnt_y):(vert_idx * tens.shape[0] // (iter_cnt_x*iter_cnt_y)) + tens.shape[0] // (iter_cnt_x*iter_cnt_y), 64:-64, 64:-64, :]
                        
            vert_idx += 1                                                                                                               #Iterate index by 1
    
    new_tens = new_tens[:, 128:-128, 128:-128, :]
    return(new_tens)

def reconstruct_tensor_from_subset3(tens, og_dim_x, og_dim_y, chunks):
    '''
    Reconstruct a tensor that was made into smaller dimensions and more rows for easier ML training (not overwhelming the GPU in matrix size).
    NOTE: this function accounts for the sunset tensor overlapped in the original imagery every 128 pixels
    Args:
        tens     : the tensor which will be reshaped
        og_dim_x : the old tensor padded by 128 pixels in each x matrix dimension
        og_dim_y : the old tensor padded by 128 pixels in each y matrix dimension
    Returns:
        new_tens: the reconstructed tensor
    '''
#     print(tens.shape)
#     print(chunks.array.shape)
#     print(chunks.subset)
#     print(chunks.padding)
#     print(chunks.overlap)
#     print(chunks.window)
 #   chunk = chunks.split()
#     chunks.join(tens)
#     new_tens = chunks.array
#    e = 0
#     print(chunks.nChunks)
#     print(len(chunks.nChunks))
#     print(np.product(chunks.nChunks[:len(chunks.nChunks)]))
#    tens = np.reshape(tens, (np.product(chunks.nChunks[:len(chunks.nChunks)]), chunks.nChunks[len(chunks.nChunks)], *tens.shape[1:]))
#     tens = np.append(tens, tens, axis = 3)
#     tens = np.reshape(tens, (100, 1, 512, 512, 2))
    for e, chunk in enumerate(chunks):
      chunks.join(np.expand_dims(tens[e, :, :, :], axis = 0))
    
    new_tens = chunks.array

    return(new_tens)

def extract_subset_region(d_str, sector = 'M2'):
    '''
    Uses switch in order to return the subset region hard-coded indices picked from reviewing images
    Args:
        d_str: Date String given as year + day number (ex. 2019137)
    Keywords:
        sector: Specify if mesoscale sector 1 or 2 ('M1', 'M2', respectively)
    Returns:
        inds : Indices [x0, y0, x1, y1] for the subset domain region. 
               NOTE: Y-axis indices from top down so y0 index is from the top!!!!
    '''
    if sector == 'M2':
        inds = {
                '2019125': [ 276, 1323,  788, 1835],
                '2019126': [ 276, 1323,  788, 1835],
                '2019137': [ 728,  825, 1240, 1337],
                '2019138': [ 728,  825, 1240, 1337],
                '2019146': [ 554,  612, 1066, 1124],
                '2019147': [ 554,  612, 1066, 1124],
                '2020134': [1290,  353, 1802,  865],
                '2020135': [1290,  353, 1802,  865]
               }
    elif sector == 'M1':           
        inds = {
                '2019120': [1268,  312, 1780,  824],
                '2019121': [1268,  312, 1780,  824],
                '2019125': [ 603,  683, 1115, 1195],
                '2019126': [ 603,  683, 1115, 1195],
                '2019127': [1223, 1434, 1735, 1946],
                '2019128': [1223, 1434, 1735, 1946],
                '2019140': [ 667,  510, 1180, 1022],
                '2019141': [ 667,  510, 1180, 1022], 
                '2021344': [1050, 700, 1562, 1212], 
                '2021345': [1050, 700, 1562, 1212]
               }
    else:
        print('Sector specified not found. Please input sector.')
        exit()
        
    val = inds.get(d_str, 'Invalid Date string')
    if val == 'Invalid Date string':
        print(d_str + ' is not set up to for subsetting to 512x512 domain region. Please make changes to code and try again')
        exit()
    return(inds)

def main():
    run_2_channel_just_updraft_day_np_preprocess2()
    
if __name__ == '__main__':
    main()
    
    
