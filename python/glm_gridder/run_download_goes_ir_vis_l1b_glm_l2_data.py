#+
# Name:
#     run_download_goes_ir_vis_l1b_glm_l2_data.py
# Purpose:
#     This is a script to download GOES IR/VIS/GLM data for a specified date or in real time (most recent data files on cloud storage)
# Calling sequence:
#     import run_download_goes_ir_vis_l1b_glm_l2_data
#     run_download_goes_ir_vis_l1b_glm_l2_data.run_download_goes_ir_vis_l1b_glm_l2_data()
# Input:
#     None.
# Functions:
#     run_download_goes_ir_vis_l1b_glm_l2_data : Main function to download GOES data
#     list_gcs                                 : Returns list of all files in storage bucket and within hour of specified date/time.
#     download_ncdf_gcs                        : Downloads the files to to local storage. Output file name is returned.
#     write_to_gcs                             : Write locally stored file that was just downloaded to a different storage cloud bucket.
# Output:
#     Downloads L1b IR and VIS files in addition to L2 GLM files for date range specified. Will also download in "real-time" as a default.
#     This means that the function will download the nearest IR and VIS files to the current time and will also download 15 nearest GLM files.
#     The 15 nearest GLM files are downloaded because they are often available at a higher frequency that IR/VIS.
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
#                    DEFAULT = True -> do not download 6.2 µm IR data files for date range. 
#     no_cirrus    : IF keyword set (True), use do not download 1.37 micron channel data files.           
#                    DEFAULT = True -> do not download 1.37 micron data files for date range.  (channel 4)
#     no_snowice   : IF keyword set (True), use do not download 1.6 micron channel data files.           
#                    DEFAULT = True -> do not download 1.6 micron data files for date range.  (channel 5)
#     no_dirtyir   : IF keyword set (True), use do not download 12.3 micron channel data files.           
#                    DEFAULT = True -> do not download 12.3 micron data files for date range.  (channel 15)
#     no_shortwave : IF keyword set (True), use do not download 3.9 micron channel data files.           
#                    DEFAULT = True -> do not download 3.9 micron data files for date range.  (channel 7)
#     no_wv        : IF keyword set (True), use do not download 6.9 micron channel data files.           
#                    DEFAULT = True -> do not download 6.9 micron data files for date range.  (channel 9)
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
import sys 
#sys.path.insert(1, '../')
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.getcwd()))
from new_model.gcs_processing import write_to_gcs, download_ncdf_gcs, list_gcs, download_gcp_parallel, copy_blob_gcs

def run_download_goes_ir_vis_l1b_glm_l2_data(date1        = None, date2 = None, 
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
                                             no_wv        = True,
                                             no_llwv      = True,
                                             no_veggie    = True,
                                             no_blue      = True,
                                             gcs_bucket   = None,
                                             del_local    = False,
                                             verbose      = True):
                                             
    '''
    Name:
        run_download_goes_ir_vis_l1b_glm_l2_data.py
    Purpose:
        This is a script to download GOES IR/VIS/GLM data for a specified date or in real time (most recent data files on cloud storage)
    Calling sequence:
        import run_download_goes_ir_vis_l1b_glm_l2_data
        run_download_goes_ir_vis_l1b_glm_l2_data.run_download_goes_ir_vis_l1b_glm_l2_data()
    Input:
        None.
    Functions:
        run_download_goes_ir_vis_l1b_glm_l2_data : Main function to download GOES data
        list_gcs                                 : Returns list of all files in storage bucket and within hour of specified date/time.
        download_ncdf_gcs                        : Downloads the files to to local storage. Output file name is returned.
        write_to_gcs                             : Write locally stored file that was just downloaded to a different storage cloud bucket.
    Output:
        Downloads L1b IR and VIS files in addition to L2 GLM files for date range specified. Will also download in "real-time" as a default.
        This means that the function will download the nearest IR and VIS files to the current time and will also download 15 nearest GLM files.
        The 15 nearest GLM files are downloaded because they are often available at a higher frequency that IR/VIS.
    Keywords:
        date1        : Start date. List containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to start downloading 
                       DEFAULT = None -> download nearest to IR/VIS file to current time and nearest 5 GLM files. (ex. '2017-04-29 00:00:00)
        date2        : End date. String containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to end downloading 
                       DEFAULT = None -> download only files nearest to date1. (ex. '2017-04-29 00:00:00)
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
        no_irdiff    : IF keyword set (True), use do not download 6.2 micron IR data files. 
                       DEFAULT = True -> do not download 6.2 micron IR data files for date range. 
        no_cirrus    : IF keyword set (True), use do not download 1.37 micron channel data files.           
                       DEFAULT = True -> do not download 1.37 micron data files for date range.  (channel 4)
        no_snowice   : IF keyword set (True), use do not download 1.6 micron channel data files.           
                       DEFAULT = True -> do not download 1.6 micron data files for date range.  (channel 5)
        no_dirtyir   : IF keyword set (True), use do not download 12.3 micron channel data files.           
                       DEFAULT = True -> do not download 12.3 micron data files for date range.  (channel 15)
        no_shortwave : IF keyword set (True), use do not download 3.9 micron channel data files.           
                       DEFAULT = True -> do not download 3.9 micron data files for date range.  (channel 7)
        no_wv        : IF keyword set (True), use do not download 6.9 micron channel data files.           
                       DEFAULT = True -> do not download 6.9 micron data files for date range.  (channel 9)
        no_llwv      :IF keyword set (True), use do not download 7.3 micron channel data files.           
                       DEFAULT = True -> do not download 7.3 micron data files for date range.  (channel 10)
        gcs_bucket   : STRING google cloud storage bucket name to write downloaded files to in addition to local storage.
                       DEFAULT = None -> Do not write to a google cloud storage bucket.
        del_local    : IF keyword set (True) AND gcs_bucket != None, delete local copy of output file.
                       DEFAULT = False.
        verbose      : BOOL keyword to specify whether or not to print verbose informational messages.
                       DEFAULT = True which implies to print verbose informational messages
    Author and history:
        John W. Cooney           2021-04-14. 
    '''
    
    outroot     = os.path.realpath(outroot)                                                                                                                                 #Create link to real path so compatible with Mac
    bucket_name = 'gcp-public-data-' + sat.lower()                                                                                                                          #Set google cloud bucket name for desired satellite to download
    ir_vis_prod = 'ABI-L1b-Rad{}'.format(sector[0].upper())                                                                                                                 #Set google cloud path name using the satellite scan sector provided   
    if sector == 'meso1':
        sector0 =  sector[0].upper() + '1'
    elif sector == 'meso2':
        sector0 =  sector[0].upper() + '2'
    elif sector == 'conus':
        sector0 =  'C'
    elif sector == 'full' or sector.lower() == 'f':
        sector0 =  'F'
    else:
        print('Not yet set up to do specified sector')
        print(sector)
        exit()
        
    glm_prod    = 'GLM-L2-LCFA'
    if date2 == None and date1 != None: date2 = date1                                                                                                                       #Default set end date to start date
    if date1 == None:
        date1 = datetime.utcnow()                                                                                                                                           #Extract current date (UTC)
        if sector0 == 'F' or sector0 == 'C':
            if date1.minute < 10 and sector0 == 'F':
                date1 = date1 - timedelta(minutes = 10)
            elif date1.minute < 4 and sector0 == 'C':  
                date1 = date1 - timedelta(minutes = 5)
        date2 = date1                                                                                                                                                       #Default set end date to start date
        rt    = True                                                                                                                                                        #Real-time download flag
 #       if verbose == True: print('Downloading in real-time: ' + date1.strftime("%Y-%m-%d-%H-%M-%S/"))
    
    else:
        date1 = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")                                                                                                               #Year-month-day hour:minute:second of start time to download data
        rt    = False                                                                                                                                                       #Real-time download flag
        if date2 == None:
            date2 = date1
        else:
            date2 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")                                                                                                           #Year-month-day hour:minute:second of end time to download data
        if verbose == True: print('Downloading dates : ' + date1.strftime("%Y-%m-%d-%H-%M-%S") + ' - ' + date2.strftime("%Y-%m-%d-%H-%M-%S"))
    if sat.lower() == 'goes-16':
        if date2 >= datetime(2025, 4, 7, 15):
            print('GOES-16 stopped operating on 2025-04-07 at 15Z. You likely want to use GOES-19')
            exit()
    if verbose == True: print('Files downloaded to outroot ' + outroot)
    outroot0 = []
    date     = date1
    while date2 >= date:
        day_num   = date.timetuple().tm_yday                                                                                                                                #Extract the day number for specified date
        iv_prefix = '{}/{}/{:03d}/{:02d}/'.format(ir_vis_prod, date.year, day_num, date.hour)                                                                               #Extract the IR/VIS prefix to pass into list_gcs function for specified date
        g_prefix  = '{}/{}/{:03d}/{:02d}/'.format(glm_prod, date.year, day_num, date.hour)                                                                                  #Extract the IR/VIS prefix to pass into list_gcs function for specified date
        
        if no_ir == False:
            files  = list_gcs(bucket_name, iv_prefix, ['C13', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                            #Extract list of IR files that match product and date
            if len(files) == 0 and rt == True:
                print('Waiting for IR C13 files to be available online')
                while True:
                    files  = list_gcs(bucket_name, iv_prefix, ['C13', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                    #Extract list of VIS files that match product and date
                    if len(files) > 0:
                        break
            
            if len(files) > 0:
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'ir') 
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                if rt == True: 
                    files = [files[-1]]                                                                                                                                     #Download only the last IR file
                for i in files:
                    if del_local == False:
                        outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                 #Download IR file to outdir
                    if gcs_bucket != None:
                        pref = os.path.join(date.strftime("%Y%m%d"), 'ir')
                        copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))
#                        write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                      #Write locally stored files to a google cloud storage bucket
            else:
                print('No IR files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))

        if no_vis == False:
            if rt == True and no_ir == False:
                fb     = re.split('_s|_', os.path.basename(files[0]))[3]
                files  = list_gcs(bucket_name, iv_prefix, ['C02', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                                    #Extract list of VIS files that match product and date
                if len(files) == 0:
                    while True:
                        files  = list_gcs(bucket_name, iv_prefix, ['C02', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                            #Extract list of VIS files that match product and date
                        if len(files) > 0:
                            break
            else:
                files  = list_gcs(bucket_name, iv_prefix, ['C02', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                        #Extract list of VIS files that match product and date
            if len(files) > 0:
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'vis') 
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                if rt == True: files = [files[-1]]                                                                                                                          #Download only the last VIS file
                for i in files:
                    if del_local == False:
                        outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                 #Download VIS file to outdir
                    if gcs_bucket != None:
                        pref = os.path.join(date.strftime("%Y%m%d"), 'vis')
                        copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))
#                        write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                     #Write locally stored files to a google cloud storage bucket
            else:
                print('No VIS files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))
        
        if no_blue == False:
            files  = list_gcs(bucket_name, iv_prefix, ['C01', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                            #Extract list of IR files that match product and date
            if len(files) == 0 and rt == True:
                print('Waiting for VIS C01 files to be available online')
                while True:
                    files  = list_gcs(bucket_name, iv_prefix, ['C01', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                    #Extract list of VIS files that match product and date
                    if len(files) > 0:
                        break
            
            if len(files) > 0:
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'blue') 
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                if rt == True: 
                    files = [files[-1]]                                                                                                                                     #Download only the last IR file
                for i in files:
                    if del_local == False:
                        outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                 #Download IR file to outdir
                    if gcs_bucket != None:
                        pref = os.path.join(date.strftime("%Y%m%d"), 'blue')
                        copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))
#                        write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                      #Write locally stored files to a google cloud storage bucket
            else:
                print('No Channel 1 files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))

        if no_veggie == False:
            files  = list_gcs(bucket_name, iv_prefix, ['C03', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                            #Extract list of IR files that match product and date
            if len(files) == 0 and rt == True:
                print('Waiting for VIS C03 files to be available online')
                while True:
                    files  = list_gcs(bucket_name, iv_prefix, ['C03', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                    #Extract list of VIS files that match product and date
                    if len(files) > 0:
                        break
            
            if len(files) > 0:
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'veggie') 
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                if rt == True: 
                    files = [files[-1]]                                                                                                                                     #Download only the last IR file
                for i in files:
                    if del_local == False:
                        outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                 #Download IR file to outdir
                    if gcs_bucket != None:
                        pref = os.path.join(date.strftime("%Y%m%d"), 'veggie')
                        copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))
#                        write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                      #Write locally stored files to a google cloud storage bucket
            else:
                print('No Channel 3 files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))

        if no_dirtyir == False:    
            if rt == True and no_ir == False:
                fb     = re.split('_s|_', os.path.basename(files[0]))[3]
                files  = list_gcs(bucket_name, iv_prefix, ['C15', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                                    #Extract list of dirtyIR files that match product and date
                if len(files) == 0:    
                    print('Waiting for Dirty IR C15 files to be available online')
                    while True:    
                        files  = list_gcs(bucket_name, iv_prefix, ['C15', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                            #Extract list of dirtyIR files that match product and date
                        if len(files) > 0:    
                            break    
            else:    
                files  = list_gcs(bucket_name, iv_prefix, ['C15', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                        #Extract list of dirtyIR files that match product and date
            if len(files) > 0:    
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'dirtyir')     
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                if rt == True: files = [files[-1]]                                                                                                                          #Download only the last dirtyIR file
                for i in files:    
                    if del_local == False:
                        outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                 #Download dirtyIR file to outdir
                    if gcs_bucket != None:
                        pref = os.path.join(date.strftime("%Y%m%d"), 'dirtyir')
                        copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))
            else:    
                print('No Dirty IR C15 files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))    
   
        if no_snowice == False:    
            if rt == True and no_ir == False:
                fb     = re.split('_s|_', os.path.basename(files[0]))[3]
                files  = list_gcs(bucket_name, iv_prefix, ['C05', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                                    #Extract list of IR files that match product and date
                if len(files) == 0:    
                    print('Waiting for Snow/Ice C05 files to be available online')
                    while True:    
                        files  = list_gcs(bucket_name, iv_prefix, ['C05', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                            #Extract list of snowice files that match product and date
                        if len(files) > 0:    
                            break    
            else:    
                files  = list_gcs(bucket_name, iv_prefix, ['C05', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                        #Extract list of snowice files that match product and date
            if len(files) > 0:    
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'snowice')     
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                if rt == True: files = [files[-1]]
                for i in files:    
                    if del_local == False:
                        outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                 #Download snowice file to outdir
                    if gcs_bucket != None:
                        pref = os.path.join(date.strftime("%Y%m%d"), 'snowice')
                        copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))
            else:    
                print('No Snow/Ice C05 files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))    
   
        if no_cirrus == False:    
            if rt == True and no_ir == False:
                fb     = re.split('_s|_', os.path.basename(files[0]))[3]
                files  = list_gcs(bucket_name, iv_prefix, ['C04', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                                    #Extract list of cirrus files that match product and date
                if len(files) == 0:    
                    print('Waiting for Cirrus C04 files to be available online')
                    while True:    
                        files  = list_gcs(bucket_name, iv_prefix, ['C04', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                            #Extract list of cirrus files that match product and date
                        if len(files) > 0:    
                            break    
            else:    
                files  = list_gcs(bucket_name, iv_prefix, ['C04', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                        #Extract list of cirrus files that match product and date
            if len(files) > 0:    
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'cirrus')     
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                if rt == True: files = [files[-1]]
                for i in files:    
                    if del_local == False:
                        outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                 #Download cirrus file to outdir
                    if gcs_bucket != None:
                        pref = os.path.join(date.strftime("%Y%m%d"), 'cirrus')
                        copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))
            else:    
                print('No Cirrus C04 files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))    
        
        if no_shortwave == False:
            if rt == True and no_ir == False:
                fb     = re.split('_s|_', os.path.basename(files[0]))[3]
                files  = list_gcs(bucket_name, iv_prefix, ['C07', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                                    #Extract list of 3.9 micron files that match product and date
                if len(files) == 0:    
                    print('Waiting for 3.9 micron C07 files to be available online')
                    while True:    
                        files  = list_gcs(bucket_name, iv_prefix, ['C07', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                            #Extract list of cirrus files that match product and date
                        if len(files) > 0:    
                            break    
            else:
                files  = list_gcs(bucket_name, iv_prefix, ['C07', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                        #Extract list of 3.9 micron files that match product and date
            if len(files) > 0:
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'shortwave') 
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                if rt == True: files = [files[-1]]
                for i in files:
                    if del_local == False:
                        outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                 #Download shortwave file to outdir
                    if gcs_bucket != None:
                        pref = os.path.join(date.strftime("%Y%m%d"), 'shortwave')
                        copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))
            else:
                print('No 3.9 micron C07 files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))

        if no_wv == False:
            if rt == True and no_ir == False:
                fb     = re.split('_s|_', os.path.basename(files[0]))[3]
                files  = list_gcs(bucket_name, iv_prefix, ['C09', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                                    #Extract list of 3.9 micron files that match product and date
                if len(files) == 0:    
                    print('Waiting for 6.9 micron C09 files to be available online')
                    while True:    
                        files  = list_gcs(bucket_name, iv_prefix, ['C09', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                            #Extract list of cirrus files that match product and date
                        if len(files) > 0:    
                            break    
            else:
                files  = list_gcs(bucket_name, iv_prefix, ['C09', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                        #Extract list of 3.9 micron files that match product and date
            if len(files) > 0:
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'wv') 
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                if rt == True: files = [files[-1]]
                for i in files:
                    if del_local == False:
                        outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                 #Download shortwave file to outdir
                    if gcs_bucket != None:
                        pref = os.path.join(date.strftime("%Y%m%d"), 'wv')
                        copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))
            else:
                print('No 6.9 micron C09 files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))

        if not no_llwv:
            if rt and not no_ir:
                fb     = re.split('_s|_', os.path.basename(files[0]))[3]
                files  = list_gcs(bucket_name, iv_prefix, ['C10', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                                    #Extract list of 3.9 micron files that match product and date
                if len(files) == 0:    
                    print('Waiting for 7.3 micron C10 files to be available online')
                    while True:    
                        files  = list_gcs(bucket_name, iv_prefix, ['C10', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                            #Extract list of cirrus files that match product and date
                        if len(files) > 0:    
                            break    
            else:
                files  = list_gcs(bucket_name, iv_prefix, ['C10', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                        #Extract list of 3.9 micron files that match product and date
            
            if len(files) > 0:
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'llwv') 
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                if rt: files = [files[-1]]
                for i in files:
                    if not del_local:
                        outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                 #Download shortwave file to outdir
                    if gcs_bucket != None:
                        pref = os.path.join(date.strftime("%Y%m%d"), 'llwv')
                        copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))
            else:
                print('No 7.3 micron C10 files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))


        if no_irdiff == False:
            if rt == True and no_ir == False:
                fb     = re.split('_s|_', os.path.basename(files[0]))[3]
                files  = list_gcs(bucket_name, iv_prefix, ['C08', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                                    #Extract list of 6.2 micron files that match product and date
                if len(files) == 0:
                    print('Waiting for IR C08 files to be available online')
                    while True:
                        files  = list_gcs(bucket_name, iv_prefix, ['C08', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0, fb])                            #Extract list of 6.2 micron files that match product and date
                        if len(files) > 0:
                            break
            else:
                files  = list_gcs(bucket_name, iv_prefix, ['C08', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour), sector0])                                        #Extract list of 6.2 micron files that match product and date
            if len(files) > 0:
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'ir_diff') 
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                if rt == True: files = [files[-1]]                                                                                                                          #Download only the last VIS file
                for i in files:
                    if del_local == False:
                        outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                                 #Download VIS file to outdir
                    if gcs_bucket != None:    
                        pref = os.path.join(date.strftime("%Y%m%d"), 'ir_diff')
                        copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))                       
#                     if gcs_bucket != None and outfile != -1:
#                         pref = re.split('goes-data/', outdir)[1]
#                         write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                    #Write locally stored files to a google cloud storage bucket
            else:
                print('No 6.2 micron IR files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))

        if no_glm == False:
            files  = list_gcs(bucket_name, g_prefix, ['GLM', 's{}{:03d}{:02d}'.format(date.year, day_num, date.hour)])                                                      #Extract list of GLM files that match product and date
            if len(files) > 0:
                outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'glm') 
                os.makedirs(outdir, exist_ok = True)                                                                                                                        #Create output directory if it does not already exist
                if rt == True: 
                    if sector0 == 'F':
                        files = files[-30:]                                                                                                                                 #Download only the last 30 GLM files
                    elif sector0 == 'C':
                        files = files[-15:]                                                                                                                                 #Download only the last 15 GLM files
                    else:
                        files = files[-10:]                                                                                                                                 #Download only the last 10 GLM files
                else:
                    for i in files:
                        if del_local == False:
                            outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                             #Download GLM file to outdir
                        if gcs_bucket != None:    
                            pref = os.path.join(date.strftime("%Y%m%d"), 'glm')
                            copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))                       
#                         if gcs_bucket != None and outfile != -1:
#                             pref = re.split('goes-data/', outdir)[1]
#                             write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                #Write locally stored files to a google cloud storage bucket
            else:
                print('No GLM files found for {}'.format(date.strftime("%Y-%m-%d-%H-%M-%S")))
            minute = date.minute
            if minute < 3.0:
                date0     = date - timedelta(hours = 1)
                day_num0  = date0.timetuple().tm_yday                                                                                                                       #Extract the day number for specified date
                g_prefix0 = '{}/{}/{:03d}/{:02d}/'.format(glm_prod, date0.year, day_num0, date0.hour)                                                                       #Extract the IR/VIS prefix to pass into list_gcs function for specified date
                files0     = list_gcs(bucket_name, g_prefix0, ['GLM', 's{}{:03d}{:02d}'.format(date0.year, day_num0, date0.hour)])                                          #Extract list of GLM files that match product and date
                if len(files0) > 0:    
                    outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'glm')     
                    os.makedirs(outdir, exist_ok = True)                                                                                                                    #Create output directory if it does not already exist
                    if rt == True:
                        val    = int(-8+(minute*3))
                        files0 = files0[-val:]
                        files.extend(files0)
                    else:
                        files = files0[-8:]
                        for i in files:    
                            if del_local == False:
                                outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                         #Download GLM file to outdir
                            if gcs_bucket != None:    
                                pref = os.path.join(date.strftime("%Y%m%d"), 'glm')
                                copy_blob_gcs(bucket_name, i, gcs_bucket, pref + os.path.basename(outfile))
#                                write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                             #Write locally stored files to a google cloud storage bucket
            if rt == True:
                if del_local == False:
                    pool    = mp.Pool(5)
                    results = [pool.apply_async(download_gcp_parallel, args=(row, files, bucket_name, outdir)) for row in range(len(files))]
                    pool.close()
                    pool.join()
                if gcs_bucket != None:    
                    pref = os.path.join(date.strftime("%Y%m%d"), 'glm')
                    for i in files:    
                        copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))                       

            if minute > 55.0 and rt == False:
                date0     = date + timedelta(hours = 1)
                day_num0  = date0.timetuple().tm_yday                                                                                                                       #Extract the day number for specified date
                g_prefix0 = '{}/{}/{:03d}/{:02d}/'.format(glm_prod, date0.year, day_num0, date0.hour)                                                                       #Extract the IR/VIS prefix to pass into list_gcs function for specified date
                files     = list_gcs(bucket_name, g_prefix0, ['GLM', 's{}{:03d}{:02d}'.format(date0.year, day_num0, date0.hour)])                                           #Extract list of GLM files that match product and date
                if len(files) > 0:    
                    outdir = os.path.join(outroot, date.strftime("%Y%m%d"), 'glm')     
                    os.makedirs(outdir, exist_ok = True)                                                                                                                    #Create output directory if it does not already exist
                    files = files[:8]
                    for i in files:    
                        if del_local == False:
                            outfile = download_ncdf_gcs(bucket_name, i, outdir)                                                                                             #Download GLM file to outdir
                        if gcs_bucket != None:    
                            pref = os.path.join(date.strftime("%Y%m%d"), 'glm')
                            copy_blob_gcs(bucket_name, i, gcs_bucket, os.path.join(pref, os.path.basename(i)))
#                            write_to_gcs(gcs_bucket, pref, outfile, del_local = del_local)                                                                                  #Write locally stored files to a google cloud storage bucket
    
        if len(outroot0) == 0:
            outroot0.append(os.path.join(outroot, date.strftime("%Y%m%d")))
        else:
            if date.strftime("%Y%m%d") != os.path.basename(outroot0[-1]): outroot0.append(os.path.join(outroot, date.strftime("%Y%m%d"))) 
        date = date + timedelta(hours = 1)                                                                                                                                  #Increment date by 1 hour
    
    return(outroot0)   

def main():
    run_download_goes_ir_vis_l1b_glm_l2_data()
    
if __name__ == '__main__':
    main()
