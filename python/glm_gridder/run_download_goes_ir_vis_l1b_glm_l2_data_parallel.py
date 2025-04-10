#+
# Name:
#     run_download_goes_ir_vis_l1b_glm_l2_data_parallel.py
# Purpose:
#     This is a script to download GOES IR/VIS/GLM data for a specified date range
# Calling sequence:
#     import run_download_goes_ir_vis_l1b_glm_l2_data_parallel
#     run_download_goes_ir_vis_l1b_glm_l2_data_parallel.run_download_goes_ir_vis_l1b_glm_l2_data_parallel()
# Input:
#     None.
# Functions:
#     run_download_goes_ir_vis_l1b_glm_l2_data_parallel : Main function to download GOES data
#     download_goes_ir_vis_l1b_glm_l2_data_parallel     : Runs loop over dates between range (date2 and date1) in parallel so that multiple dates can be downloaded simultaneously
#     download_ncdf_gcs                                 : Downloads the files to to local storage. Output file name is returned.
#     write_to_gcs                                      : Write locally stored file that was just downloaded to a different storage cloud bucket.
# Output:
#     Downloads L1b IR and VIS files in addition to L2 GLM files for date range specified. 
# Keywords:
#     date1        : Start date. List containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to start downloading 
#                    DEFAULT = None -> download nearest to IR/VIS file to current time and nearest 5 GLM files. (ex. '2017-04-29 00:00:00)
#     date2        : End date. String containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to end downloading 
#                    DEFAULT = None -> download only files nearest to date1. (ex. '2017-04-29 00:00:00)
#     outroot      : STRING specifying output root directory to save the downloaded GOES files
#                    DEFAULT = '../../../goes-data/'
#     sat          : STRING specifying GOES satellite to download data for. 
#                    DEFAULT = 'goes-16.
#     sector       : STRING specifying GOES sector to download. (ex. 'meso', 'conus', 'full')
#                    DEFAULT = 'meso'
#     no_glm       : IF keyword set (True), use do not download GLM data files. 
#                    DEFAULT = False -> download GLM data files for date range. 
#     no_vis       : IF keyword set (True), use do not download VIS data files. 
#                    DEFAULT = False -> download VIS data files for date range. 
#     no_ir        : IF keyword set (True), use do not download IR data files. 
#                    DEFAULT = False -> download IR data files for date range. 
#     no_irdiff    : IF keyword set (True), use do not download IR data files. 
#                    DEFAULT = True -> do not download 6.2 µm IR data files for date range.  (channel 8)
#     no_cirrus    : IF keyword set (True), use do not download 1.37 micron channel data files.           
#                    DEFAULT = True -> do not download 1.37 micron data files for date range.  (channel 4)
#     no_snowice   : IF keyword set (True), use do not download 1.6 micron channel data files.           
#                    DEFAULT = True -> do not download 1.6 micron data files for date range.  (channel 5)
#     no_dirtyir   : IF keyword set (True), use do not download 12.3 micron channel data files.           
#                    DEFAULT = True -> do not download 12.3 micron data files for date range.  (channel 15)
#     no_shortwave : IF keyword set (True), use do not download 3.9 micron channel data files.           
#                    DEFAULT = True -> do not download 3.9 micron data files for date range.  (channel 7)
#     gcs_bucket   : STRING google cloud storage bucket name to write downloaded files to in addition to local storage.
#                    DEFAULT = None -> Do not write to a google cloud storage bucket.
#     del_local    : IF keyword set (True) AND gcs_bucket != None, delete local copy of output file.
#                    DEFAULT = False.
#     verbose      : BOOL keyword to specify whether or not to print verbose informational messages.
#                    DEFAULT = True which implies to print verbose informational messages
# Author and history:
#     John W. Cooney           2021-04-14. 
#
#-

#### Environment Setup ###
# Package imports
import numpy as np
import os
import re
from datetime import datetime, timedelta
import multiprocessing as mp
import time
from math import ceil
import sys 
#sys.path.insert(1, '../')
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.getcwd()))
from new_model.gcs_processing import write_to_gcs, download_ncdf_gcs, list_gcs

def run_download_goes_ir_vis_l1b_glm_l2_data_parallel(date1        = None, date2 = None, 
                                                      outroot      = '../../../goes-data/', 
                                                      sat          = 'goes-16', 
                                                      sector       = 'meso2', 
                                                      no_glm       = False, 
                                                      no_vis       = False, 
                                                      no_ir        = False,
                                                      no_irdiff    = True, 
                                                      no_cirrus    = True, 
                                                      no_snowice   = True, 
                                                      no_dirtyir   = True, 
                                                      no_shortwave = True, 
                                                      gcs_bucket   = None,
                                                      del_local    = False,
                                                      verbose      = True):
    '''
    Name:
        run_download_goes_ir_vis_l1b_glm_l2_data_parallel.py
    Purpose:
        This is a script to download GOES IR/VIS/GLM data for a specified date in parallel (does not do real-time because parallel not necessary)
    Calling sequence:
        import run_download_goes_ir_vis_l1b_glm_l2_data_parallel
        run_download_goes_ir_vis_l1b_glm_l2_data_parallel.run_download_goes_ir_vis_l1b_glm_l2_data_parallel()
    Input:
        None.
    Functions:
        run_download_goes_ir_vis_l1b_glm_l2_data_parallel : Main function to download GOES data
        download_goes_ir_vis_l1b_glm_l2_data_parallel     : Runs loop over dates between range (date2 and date1) in parallel so that multiple dates can be downloaded simultaneously
        download_ncdf_gcs                                 : Downloads the files to to local storage. Output file name is returned.
        write_to_gcs                                      : Write locally stored file that was just downloaded to a different storage cloud bucket.
    Output:
        Downloads L1b IR and VIS files in addition to L2 GLM files for date range specified. 
    Keywords:
        date1        : Start date. List containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to start downloading 
                       DEFAULT = None -> download nearest to IR/VIS file to current time and nearest 5 GLM files. (ex. '2017-04-29 00:00:00')
        date2        : End date. String containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to end downloading 
                       DEFAULT = None -> download only files nearest to date1. (ex. '2017-04-30 04:00:00')
        outroot      : STRING specifying output root directory to save the downloaded GOES files
                       DEFAULT = '../../../goes-data/'
        sat          : STRING specifying GOES satellite to download data for. 
                       DEFAULT = 'goes-16.
        sector       : STRING specifying GOES sector to download. (ex. 'meso', 'conus', 'full')
                       DEFAULT = 'meso2'
        no_glm       : IF keyword set (True), use do not download GLM data files. 
                       DEFAULT = False -> download GLM data files for date range. 
        no_vis       : IF keyword set (True), use do not download VIS data files. 
                       DEFAULT = False -> download VIS data files for date range. 
        no_ir        : IF keyword set (True), use do not download IR data files. 
                       DEFAULT = False -> download IR data files for date range. 
        no_irdiff    : IF keyword set (True), use do not download 6.2 micron IR data files (Upper-Level Tropospheric Water Vapor; ULTWV). 
                       DEFAULT = True -> do not download 6.2 micron IR data files for date range.  (channel 8)
        no_cirrus    : IF keyword set (True), use do not download 1.37 micron channel data files.           
                       DEFAULT = True -> do not download 1.37 micron data files for date range.  (channel 4)
        no_snowice   : IF keyword set (True), use do not download 1.6 micron channel data files.           
                       DEFAULT = True -> do not download 1.6 micron data files for date range.  (channel 5)
        no_dirtyir   : IF keyword set (True), use do not download 12.3 micron channel data files.           
                       DEFAULT = True -> do not download 12.3 micron data files for date range.  (channel 15)
        no_shortwave : IF keyword set (True), use do not download 3.9 micron channel data files.           
                       DEFAULT = True -> do not download 3.9 micron data files for date range.  (channel 7)
        gcs_bucket   : STRING google cloud storage bucket name to write downloaded files to in addition to local storage.
                       DEFAULT = None -> Do not write to a google cloud storage bucket.
        del_local    : IF keyword set (True) AND gcs_bucket != None, delete local copy of output file.
                       DEFAULT = False.
        verbose      : BOOL keyword to specify whether or not to print verbose informational messages.
                       DEFAULT = True which implies to print verbose informational messages 
    Author and history:
        John W. Cooney           2021-04-14. 
    
    '''
    t0 = time.time()

    
    outroot     = os.path.realpath(outroot)                                                                                                                                 #Create link to real path so compatible with Mac
    bucket_name = 'gcp-public-data-' + sat.lower()                                                                                                                          #Set google cloud bucket name for desired satellite to download
    ir_vis_prod = 'ABI-L1b-Rad{}'.format(sector[0].upper())                                                                                                                 #Set google cloud path name using the satellite scan sector provided      
    if sector == 'meso1':
        sector0 =  sector[0].upper() + '1'
    elif sector == 'meso2':
        sector0 =  sector[0].upper() + '2'
    elif sector == 'conus':
        sector0 =  'C'
    elif sector == 'full':
        sector0 =  'F'
    else:
        print('Not yet set up to do specified sector')
        print(sector)
        exit()
    glm_prod    = 'GLM-L2-LCFA'
    if date2 == None and date1 != None: date2 = date1                                                                                                                       #Default set end date to start date
    if date1 == None:
        print('Date1 must be specified!!')
        exit()
    else:
        date1 = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")                                                                                                               #Year-month-day hour:minute:second of start time to download data
        if date2 == None:
            date2 = date1
        else:
            date2 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")                                                                                                           #Year-month-day hour:minute:second of start time to download data
        if verbose == True: print('Downloading dates : ' + date1.strftime("%Y-%m-%d-%H-%M-%S") + ' - ' + date2.strftime("%Y-%m-%d-%H-%M-%S"))
    
    if sat.lower() == 'goes-16':
        if date2 >= datetime(2025, 4, 7, 15):
            print('GOES-16 stopped operating on 2025-04-07 at 15Z. You likely want to use GOES-19')
            exit()
    
    if verbose == True: print('Files downloaded to outroot ' + outroot)
    date      = date1
    date_list = [date + timedelta(hours = x) for x in range(int((date2-date1).days*24 + ceil((date2-date1).seconds/3600.0))+1)]                                             #Extract all dates between date1 and date2 based on hour of satellite scan
    nloops    = len(date_list)
    outroot0  = []
    for d in date_list:
        outroot0.append(os.path.join(outroot, d.strftime("%Y%m%d")))                                                                                                        #Store all output storage directories of the downloaded data
   
    pool      = mp.Pool(4)                                                                                                                                                  #Set up parallel multiprocessing threads
    results   = [pool.apply_async(download_goes_ir_vis_l1b_glm_l2_data_parallel, args=(row, date_list, date1, date2, ir_vis_prod, glm_prod, sector0,  
                                                                                       no_ir, no_vis, no_glm, no_irdiff, no_cirrus, no_snowice, no_dirtyir, no_shortwave, outroot, bucket_name, gcs_bucket, del_local, verbose)) for row in range(nloops)]
    pool.close()                                                                                                                                                            #Close the multiprocessing threads
    pool.join()
    if len(outroot0) <= 0:
        print('No output root directories to return???')
        exit()
    outroot0  = sorted(list(set(outroot0)))
    if verbose == True: print('Time to download files ' + str(time.time() - t0))
    
    return(outroot0)  

def download_goes_ir_vis_l1b_glm_l2_data_parallel(idx, date_list, date1, date2,
                                                  ir_vis_prod, glm_prod, sector0, 
                                                  no_ir, no_vis, no_glm, no_irdiff, no_cirrus, no_snowice, no_dirtyir, no_shortwave,
                                                  outroot, bucket_name, gcs_bucket, del_local, verbose):

    date      = date_list[idx]                                                                                                                                              #Extract date to download from list of dates
    day_num   = date.timetuple().tm_yday                                                                                                                                    #Extract the day number for specified date
    iv_prefix = '{}/{}/{:03d}/{:02d}/'.format(ir_vis_prod, date.year, day_num, date.hour)                                                                                   #Extract the IR/VIS prefix to pass into list_gcs function for specified date
    g_prefix  = '{}/{}/{:03d}/{:02d}/'.format(glm_prod, date.year, day_num, date.hour)                                                                                      #Extract the IR/VIS prefix to pass into list_gcs function for specified date
        
    if no_ir == False:    
        files  = list_gcs(bucket_name, iv_prefix, ['C13', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                                #Extract list of IR files that match product and date
        if len(files) == 0:
            print('Waiting for IR C13 files to be available online')
            while True:    
                files  = list_gcs(bucket_name, iv_prefix, ['C13', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                        #Extract list of IR files that match product and date
                if len(files) > 0:    
                    break    
            
        if len(files) > 0:    
            outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'ir')     
            os.makedirs(outdir, exist_ok = True)                                                                                                                            #Create output directory if it does not already exist
            for i in files:    
                fdate = datetime.strptime(re.split('_s|_', os.path.basename(i))[3][0:-3], "%Y%j%H%M")
                if ((fdate >= date1) and (fdate <= date2)):
                    outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                     #Download IR file to outdir
                    if gcs_bucket != None and outfile != -1:    
                        pref = os.path.join(date.strftime("%Y%m%d"), 'ir')
                        write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                      #Write locally stored files to a google cloud storage bucket
        else:    
            print('No IR files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))    
   
    if no_dirtyir == False:    
        files  = list_gcs(bucket_name, iv_prefix, ['C15', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                                #Extract list of IR files that match product and date
        if len(files) == 0:    
            print('Waiting for Dirty IR C15 files to be available online')
            while True:    
                files  = list_gcs(bucket_name, iv_prefix, ['C15', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                        #Extract list of IR files that match product and date
                if len(files) > 0:    
                    break    
            
        if len(files) > 0:    
            outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'dirtyir')     
            os.makedirs(outdir, exist_ok = True)                                                                                                                            #Create output directory if it does not already exist
            for i in files:    
                fdate = datetime.strptime(re.split('_s|_', os.path.basename(i))[3][0:-3], "%Y%j%H%M")
                if ((fdate >= date1) and (fdate <= date2)):
                    outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                     #Download Dirty IR C15 file to outdir
                    if gcs_bucket != None and outfile != -1:    
                        pref = os.path.join(date.strftime("%Y%m%d"), 'dirtyir')
                        write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                      #Write locally stored files to a google cloud storage bucket
        else:    
            print('No Dirty IR C15 files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))    
   
    if no_snowice == False:    
        files  = list_gcs(bucket_name, iv_prefix, ['C05', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                                #Extract list of IR files that match product and date
        if len(files) == 0:    
            print('Waiting for Snow/Ice C05 files to be available online')
            while True:    
                files  = list_gcs(bucket_name, iv_prefix, ['C05', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                        #Extract list of IR files that match product and date
                if len(files) > 0:    
                    break    
            
        if len(files) > 0:    
            outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'snowice')     
            os.makedirs(outdir, exist_ok = True)                                                                                                                            #Create output directory if it does not already exist
            for i in files:    
                fdate = datetime.strptime(re.split('_s|_', os.path.basename(i))[3][0:-3], "%Y%j%H%M")
                if ((fdate >= date1) and (fdate <= date2)):
                    outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                     #Download snowice C05 file to outdir
                    if gcs_bucket != None and outfile != -1:    
                        pref = os.path.join(date.strftime("%Y%m%d"), 'snowice')
                        write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                      #Write locally stored files to a google cloud storage bucket
        else:    
            print('No Snow/Ice C05 files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))    
   
    if no_cirrus == False:    
        files  = list_gcs(bucket_name, iv_prefix, ['C04', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                                #Extract list of IR files that match product and date
        if len(files) == 0:    
            print('Waiting for Cirrus C04 files to be available online')
            while True:    
                files  = list_gcs(bucket_name, iv_prefix, ['C04', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                        #Extract list of IR files that match product and date
                if len(files) > 0:    
                    break    
            
        if len(files) > 0:    
            outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'cirrus')     
            os.makedirs(outdir, exist_ok = True)                                                                                                                            #Create output directory if it does not already exist
            for i in files:    
                fdate = datetime.strptime(re.split('_s|_', os.path.basename(i))[3][0:-3], "%Y%j%H%M")
                if ((fdate >= date1) and (fdate <= date2)):
                    outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                     #Download cirrus C04 file to outdir
                    if gcs_bucket != None and outfile != -1:    
                        pref = os.path.join(date.strftime("%Y%m%d"), 'cirrus')
                        write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                      #Write locally stored files to a google cloud storage bucket
        else:    
            print('No Cirrus C04 files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))    
        
    if no_vis == False:    
        files  = list_gcs(bucket_name, iv_prefix, ['C02', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                                #Extract list of VIS files that match product and date
        if len(files) > 0:    
            outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'vis')     
            os.makedirs(outdir, exist_ok = True)                                                                                                                            #Create output directory if it does not already exist
            for i in files:    
                fdate = datetime.strptime(re.split('_s|_', os.path.basename(i))[3][0:-3], "%Y%j%H%M")
                if ((fdate >= date1) and (fdate <= date2)):
                    outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                     #Download VIS file to outdir
                    if gcs_bucket != None and outfile != -1:    
                        pref = os.path.join(date.strftime("%Y%m%d"), 'vis')
                        write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                      #Write locally stored files to a google cloud storage bucket
        else:    
            print('No VIS files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))    

    if no_irdiff == False:
        files  = list_gcs(bucket_name, iv_prefix, ['C08', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                                #Extract list of 6.2 micron files that match product and date
        if len(files) > 0:
            outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'ir_diff') 
            os.makedirs(outdir, exist_ok = True)                                                                                                                            #Create output directory if it does not already exist
            for i in files:
                fdate = datetime.strptime(re.split('_s|_', os.path.basename(i))[3][0:-3], "%Y%j%H%M")
                if ((fdate >= date1) and (fdate <= date2)):
                    outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                     #Download 6.2 micron C08 file to outdir
                    if gcs_bucket != None and outfile != -1:
                        pref = os.path.join(date.strftime("%Y%m%d"), 'ir_diff')
                        write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                      #Write locally stored files to a google cloud storage bucket
        else:
            print('No 6.2 micron IR files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))
        
    if no_shortwave == False:
        files  = list_gcs(bucket_name, iv_prefix, ['C07', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                                #Extract list of 6.2 micron files that match product and date
        if len(files) > 0:
            outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'shortwave') 
            os.makedirs(outdir, exist_ok = True)                                                                                                                            #Create output directory if it does not already exist
            for i in files:
                fdate = datetime.strptime(re.split('_s|_', os.path.basename(i))[3][0:-3], "%Y%j%H%M")
                if ((fdate >= date1) and (fdate <= date2)):
                    outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                     #Download 3.9 micron C07 file to outdir
                    if gcs_bucket != None and outfile != -1:
                        pref = os.path.join(date.strftime("%Y%m%d"), 'shortwave')
                        write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                      #Write locally stored files to a google cloud storage bucket
        else:
            print('No 3.9 micron IR files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))

    if no_glm == False:    
        date01 = date1 - timedelta(minutes = 10)
        date02 = date2 + timedelta(minutes = 10)
        files  = list_gcs(bucket_name, g_prefix, ['GLM', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour)])                                                          #Extract list of GLM files that match product and date
        if len(files) > 0:    
            outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'glm')     
            os.makedirs(outdir, exist_ok = True)                                                                                                                            #Create output directory if it does not already exist
            for i in files:    
                fdate = datetime.strptime(re.split('_s|_', os.path.basename(i))[3][0:-3], "%Y%j%H%M")
                if ((fdate >= date01) and (fdate <= date02)):
                    outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                     #Download GLM file to outdir
                    if gcs_bucket != None and outfile != -1:    
                        pref = os.path.join(date.strftime("%Y%m%d"), 'glm')
                        write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                      #Write locally stored files to a google cloud storage bucket
        else:
            print('No GLM files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))
        if date.minute < 10.0:
            date0     = date - timedelta(hours = 1)
            day_num0  = date0.timetuple().tm_yday                                                                                                                           #Extract the day number for specified date
            g_prefix0 = '{}/{}/{:03d}/{:02d}/'.format(glm_prod, date0.year, day_num0, date0.hour)                                                                           #Extract the IR/VIS prefix to pass into list_gcs function for specified date
            files     = list_gcs(bucket_name, g_prefix0, ['GLM', 's{}{:03d}{:02d}'.format(date0.year, day_num0, date0.hour)])                                               #Extract list of GLM files that match product and date
            if len(files) > 0:    
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'glm')     
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                for i in files:    
                    fdate = datetime.strptime(re.split('_s|_', os.path.basename(i))[3][0:-3], "%Y%j%H%M")
                    if ((fdate >= date01) and (fdate <= date02)):
                        outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                 #Download GLM file to outdir
                        if gcs_bucket != None and outfile != -1:    
                            pref = os.path.join(date.strftime("%Y%m%d"), 'glm')
                            write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                  #Write locally stored files to a google cloud storage bucket
        if date.minute > 50.0:
            date0     = date + timedelta(hours = 1)
            day_num0  = date0.timetuple().tm_yday                                                                                                                           #Extract the day number for specified date
            g_prefix0 = '{}/{}/{:03d}/{:02d}/'.format(glm_prod, date0.year, day_num0, date0.hour)                                                                           #Extract the IR/VIS prefix to pass into list_gcs function for specified date
            files     = list_gcs(bucket_name, g_prefix0, ['GLM', 's{}{:03d}{:02d}'.format(date0.year, day_num0, date0.hour)])                                               #Extract list of GLM files that match product and date
            if len(files) > 0:    
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'glm')     
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                for i in files:    
                    fdate = datetime.strptime(re.split('_s|_', os.path.basename(i))[3][0:-3], "%Y%j%H%M")
                    if ((fdate >= date01) and (fdate <= date02)):
                        outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                 #Download GLM file to outdir
                        if gcs_bucket != None and outfile != -1:    
                            pref = os.path.join(date.strftime("%Y%m%d"), 'glm')
                            write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                  #Write locally stored files to a google cloud storage bucket
            
def main():
    run_download_goes_ir_vis_l1b_glm_l2_data_parallel()
    
if __name__ == '__main__':
    main()
