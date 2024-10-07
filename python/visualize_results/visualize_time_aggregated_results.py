#+
# Name:
#     visualize_time_aggregated_results.py
# Purpose:
#     This is a script to visualize the time aggregated masks and model results
# Calling sequence:
#     import visualize_time_aggregated_results
#     visualize_time_aggregated_results.visualize_time_aggregated_results()
# Input:
#     res      : Numpy array containing model results
#     nc_files : List containing combined netCDF files names
#     ir_files : List containing raw IR data file names
# Functions:
#     visualize_time_aggregated_results : Main function
#     setup_2_gridded_axes              : Sets up 2 gridded axes side by side
#     setup_3_gridded_axes              : Sets up 3 gridded axes side by side
#     setup_4_gridded_axes              : Sets up 4 gridded axes as 2x2
#     read_spc_reports                  : Read SPC reports to plot
# Output:
#     Creates PNG image of AACP results
# Keywords:
#     outroot          : STRING output directory for the overlaid images
#                        DEFUALT = '../../../goes-data/aacp_results_imgs/ir_vis_glm/updraft_day_model/2021-07-08/time_ag_figs/2021195-196/M2/'
#     date_range       : List containing start date and end date of predictions [start_date, end_date]. (FORMAT = %Y-%m-%d %H:%M:%S) (ex. '2017-05-01 00:00:00') 
#                        DEFAULT = None -> use nc_names to extract date range
#     pthresh          : Optional FLOAT value to use to threshold the probability of the results. result probabilities >= pthresh -> 1 otherwise they -> 0.
#                        IF set, model precision and recall are both calculated and printed to the terminal at the conclusion of the program.
#                        DEFAULT = None. This implies to not plot the thresholded results or calculate precision and recall
#     latlon_domain    : [x0, y0, x1, y1] array in latitude and longitude coordinates to give subset of image domain desired to be plotted. Array must contain lat/lon values
#                        DEFAULT is to plot the entire domain. 
#                        NOTE: If region keyword is set, it supersedes this keyword
#     region           : State or sector of US to plot data around. 
#                        DEFAULT = None -> plot for full domain.
#     use_local        : IF keyword set (True), use local files. If False, use files located on the google cloud. 
#                        DEFAULT = False -> use files on google cloud server. 
#     write_gcs        : IF keyword set (True), write the output files to the google cloud in addition to local storage.
#                        DEFAULT = True.
#     del_local        : IF keyword set (True) AND write_gcs = True, delete local copy of output file.
#                        DEFAULT = True.
#     res_bucket_name  : STRING specifying the name of the gcp bucket to write images to. write_gcs needs to be True in order for this to matter.
#                        DEFAULT = 'aacp-results'
#     proc_bucket_name : STRING specifying the name of the gcp bucket to write files to. write_gcs needs to be True in order for this to matter.
#                        DEFAULT = 'aacp-proc-data'
#     plt_spc          : IF keyword set (True), plot SPC reports for date range of files.
#                        DEFAULT = True.
#     plt_glm          : IF keyword set (True), plot GLM lightning flash extent density data, rather than IR BT.
#                        DEFAULT = False -> plot IR BT
#     ir_max           : FLOAT keyword specifying the maximum IR BT in color range of map (K). DEFAULT = 230 K
#     ir_min           : FLOAT keyword specifying the minimum IR BT in color range of map (K). DEFAULT = 180 K
#     verbose          : BOOL keyword to specify whether or not to print verbose informational messages.
#                        DEFAULT = False which implies to not print verbose informational messages to terminal
# Author and history:
#     John W. Cooney           2021-07-20.
#
#-

#### Environment Setup ###
# Package imports

import numpy as np
import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy import feature
from math import ceil
from netCDF4 import Dataset
import pynimbus as pyn
import time
from datetime import datetime, timedelta
import scipy.interpolate as interp
import sys 
#sys.path.insert(1, '../')
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.getcwd()))
from new_model.gcs_processing import write_to_gcs, load_npy_blobs, load_csv_gcs, list_gcs
from glm_gridder import turbo_colormap
from gridrad.rdr_sat_utils_jwc import extract_us_lat_lon_region, extract_us_lat_lon_region_parallax_correction

def visualize_time_aggregated_results(res, nc_files, ir_files, 
                                      outroot           = os.path.join('..', '..', '..', 'goes-data', 'aacp_results_imgs', 'ir_vis_glm', 'updraft_day_model', '2021-07-08', 'time_ag_figs', '2021195-196', 'M2'),
                                      date_range        = None, 
                                      latlon_domain     = [], 
                                      region            = None,
                                      pthresh           = None, 
                                      use_local         = False, write_gcs = True, del_local = True, 
                                      res_bucket_name   = 'aacp-results',
                                      proc_bucket_name  = 'aacp-proc-data', 
                                      plt_spc           = True,
                                      plt_glm           = False,
                                      ir_max            = 230.0, ir_min = 180.0,
                                      verbose           = True):
    '''
    This function creates figures that visualize the time aggregated masks and model results

    Calling sequence:
        import create_vis_ir_numpy_arrays_from_netcdf_files2
        create_vis_ir_numpy_arrays_from_netcdf_files2.create_vis_ir_numpy_arrays_from_netcdf_files2()
    Args:
        res      : Numpy array containing model results
        nc_files : List containing combined netCDF files names
        ir_files : List containing raw IR data file names
    Functions:
        visualize_time_aggregated_results : Main function
        setup_2_gridded_axes              : Sets up 2 gridded axes side by side
        setup_3_gridded_axes              : Sets up 3 gridded axes side by side
        setup_4_gridded_axes              : Sets up 4 gridded axes as 2x2
        read_spc_reports                  : Read SPC reports to plot
    Output:
        Creates PNG image of AACP results
    Keywords:
        outroot          : STRING output directory for the overlaid images
                           DEFUALT = '../../../goes-data/aacp_results_imgs/ir_vis_glm/updraft_day_model/2021-07-08/time_ag_figs/2021195-196/M2/'
        date_range       : List containing start date and end date of predictions [start_date, end_date]. (FORMAT = %Y-%m-%d %H:%M:%S) (ex. '2017-05-01 00:00:00') 
                           DEFAULT = None -> use nc_names to extract date range
        pthresh          : Optional FLOAT value to use to threshold the probability of the results. result probabilities >= pthresh -> 1 otherwise they -> 0.
                           IF set, model precision and recall are both calculated and printed to the terminal at the conclusion of the program.
                           DEFAULT = None. This implies to not plot the thresholded results or calculate precision and recall
        latlon_domain    : [x0, y0, x1, y1] array in latitude and longitude coordinates to give subset of image domain desired to be plotted. Array must contain lat/lon values
                           DEFAULT is to plot the entire domain. 
                           NOTE: If region keyword is set, it supersedes this keyword
        region           : State or sector of US to plot data around. 
                           DEFAULT = None -> plot for full domain.
        use_local        : IF keyword set (True), use local files. If False, use files located on the google cloud. 
                           DEFAULT = False -> use files on google cloud server. 
        write_gcs        : IF keyword set (True), write the output files to the google cloud in addition to local storage.
                           DEFAULT = True.
        del_local        : IF keyword set (True) AND write_gcs = True, delete local copy of output file.
                           DEFAULT = True.
        res_bucket_name  : STRING specifying the name of the gcp bucket to write images to. write_gcs needs to be True in order for this to matter.
                           DEFAULT = 'aacp-results'
        proc_bucket_name : STRING specifying the name of the gcp bucket to write files to. write_gcs needs to be True in order for this to matter.
                           DEFAULT = 'aacp-proc-data'
        plt_spc          : IF keyword set (True), plot SPC reports for date range of files.
                           DEFAULT = True.
        plt_glm          : IF keyword set (True), plot GLM lightning flash extent density data, rather than IR BT.
                           DEFAULT = False -> plot IR BT
        ir_max           : FLOAT keyword specifying the maximum IR BT in color range of map (K). DEFAULT = 230 K
        ir_min           : FLOAT keyword specifying the minimum IR BT in color range of map (K). DEFAULT = 180 K
        verbose          : BOOL keyword to specify whether or not to print verbose informational messages.
                           DEFAULT = False which implies to not print verbose informational messages to terminal
    Returns:
        None
    Author and history:
        John W. Cooney           2021-07-20.
    '''
#     if pthresh == None:
#         print('pthresh cannot equal None!!!')
    
    outroot   = os.path.realpath(outroot)                                                                                               #Create link to real path so compatible with Mac
    dirs      = re.split('aacp_results_imgs' + os.sep, outroot)[1]                                                                      #Extract model input parameters, plume or updraft model, and date of pre-processing
    use_chan  = re.split(os.sep, dirs)[0]                                                                                               #Extract name of channels used by model
    model     = '+'.join(re.split('_', use_chan)).upper()                                                                               #Extract name of model used
    if len(re.split('updraft_day_model', outroot)) > 1:    
        mod0 = 'OT'    
        pat  = '_ot_'    
    else:    
        mod0 = 'Plume'    
        pat  = '_plume_'    
    if region == None:
      outdir  = os.path.join(outroot, 'figures', 'full_domain')
    else:
      outdir  = os.path.join(outroot, 'figures', '_'.join(re.split(' ', region.lower())))    
       
    if verbose == True:        
        print('Output root directory = ' + outroot)    

    if date_range == None:                                                                                                              #Create image title
        plt_title = model + ';'  + mod0 + ';'
        dregion   = re.split('_s|_', os.path.basename(comb_files[0]))[5][0:7] + '_' + re.split('_', os.path.basename(nc_files[0]))[3] + '_region'
    else:
        plt_title = model + ';'  + mod0 + ';' + date_range[0] + ' - ' + date_range[1]
        dregion   = date_range[0] + '-' + date_range[1] + '_' + re.split('_', os.path.basename(nc_files[0]))[3] + '_region'
    if region != None:         
        pc   = extract_us_lat_lon_region_parallax_correction(region)                                                                    #Extract parallax correction 
        pc   = pc[region.lower()]                                                                                                       #Extract the parallax correction for region for specified US state or region 
        x_pc = pc[0]                                                                                                                    #Extract parallax correction for longitudes
        y_pc = pc[1]                                                                                                                    #Extract parallax correction for latitudes
        if verbose == True:
            print('Adding ' + str(x_pc) + ' to satellite longitudes')
            print('Subtracting ' + str(y_pc) + ' from satellite latitudes')
    else:
        x_pc = 0.0
        y_pc = 0.0
#Set plotting constants      
    cpt_convert = ListedColormap(turbo_colormap.turbo_colormap_data)                                                                    #Color table to use for mapping of figures in image     
    dim         = '500'
    plt.rcParams['font.size'] = '4'    
    my_dpi      = int(dim)
    i_dat       = []
    for i, ir_file in enumerate(ir_files):
        if use_local == True:
            if os.path.basename(os.path.dirname(ir_file)) == 'ir':
                glm_file = os.path.join(os.path.dirname(os.path.dirname(ir_file)), 'glm', 'glm.npy')
                dir_path = os.path.dirname(os.path.dirname(ir_file))
            else:
                glm_file = os.path.join(os.path.dirname(ir_file), 'glm.npy')
                dir_path = os.path.dirname(ir_file)
            df = pd.read_csv(os.path.join(dir_path, 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv'))
            dates = list(df['date_time'])
            if plt_glm == True:
                f_exist = sorted(glob.glob(os.path.join(os.path.dirname(glm_file), '*glm.npy')))
            else:
                f_exist = sorted(glob.glob(os.path.join(os.path.dirname(ir_file), '*ir.npy')))
            if len(f_exist) <= 0:
                print('Could not find ir or glm numpy file???')
                print(os.path.dirname(ir_file))
                exit()
            if len(f_exist) > 1 or 'real_time' in outroot:    
                fbs = [os.path.basename(f) for f in f_exist]
                for nc_file in nc_files:
                    dd_str = re.split('_s|_', os.path.basename(nc_file))[5]
                    date = datetime.strptime(dd_str[0:-1], "%Y%j%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                    if date in dates:
                        if plt_glm == True:
                            file2 = os.path.join(os.path.dirname(glm_file), dd_str[0:-1] + '_glm.npy')
                        else:
                            file2 = os.path.join(os.path.dirname(ir_file), dd_str[0:-1] + '_ir.npy')
                        fb = os.path.basename(file2)
                        if fb in fbs:
                            if len(i_dat) <= 0:
                                i_dat = np.load(f_exist[fbs.index(fb)])[0, :, :]
                            else:    
                                i_dat0 = np.load(f_exist[fbs.index(fb)])[0, :, :]
                                i_dat[i_dat0 > i_dat] = i_dat0[i_dat0 > i_dat]                                                          #Retain only the maximum values at each index 
                                i_dat0 = None
            else:
                if len(i_dat) <= 0:
                    if plt_glm == True:
                        i_dat = np.nanmax(np.load(glm_file)[df.dropna(subset = ['ir_files'])['Unnamed: 0'].astype(int).values, :, :], axis = 0) #Read images file to get Brightness temperatures (BTs always 0th index of arrays)
                    else:
                        i_dat = np.nanmax(np.load(ir_file)[df.dropna(subset = ['ir_files'])['Unnamed: 0'].astype(int).values, :, :], axis = 0)  #Read images file to get Brightness temperatures (BTs always 0th index of arrays)
                else:
                    if plt_glm == True:
                        i_dat0 = np.nanmax(np.load(glm_file)[df.dropna(subset = ['ir_files'])['Unnamed: 0'].astype(int).values, :, :], axis = 0) #Read images file to get Brightness temperatures (BTs always 0th index of arrays)
                        i_dat = np.append(i_dat, np.load(glm_file)[df.dropna(subset = ['ir_files'])['Unnamed: 0'].astype(int).values, :, :])#Read images file to get Brightness temperatures (BTs always 0th index of arrays)
                        i_dat[i_dat0 > i_dat] = i_dat0[i_dat0 > i_dat]
                        i_dat0 = None
                    else:
                        i_dat0 = np.nanmax(np.load(ir_file)[df.dropna(subset = ['ir_files'])['Unnamed: 0'].astype(int).values, :, :], axis = 0)  #Read images file to get Brightness temperatures (BTs always 0th index of arrays)
                        i_dat[i_dat0 > i_dat] = i_dat0[i_dat0 > i_dat]
                        i_dat0 = None
        else:    
            if os.path.basename(os.path.dirname(ir_file)) == 'ir':
                glm_file = os.path.join(os.path.dirname(os.path.dirname(ir_file)), 'glm', 'glm.npy')
                pref     = os.path.join(os.path.basename(os.path.realpath(re.split('labelled', ir_file)[0])), os.path.relpath(os.path.dirname(os.path.dirname(ir_file)), re.split('labelled', ir_file)[0]))
            else:
                glm_file = os.path.join(os.path.dirname(ir_file), 'glm.npy')
                pref     = os.path.join(os.path.basename(os.path.realpath(re.split('labelled', ir_file)[0])), os.path.relpath(os.path.dirname(ir_file), re.split('labelled', ir_file)[0]))
            df    = load_csv_gcs(proc_bucket_name, pref + 'vis_ir_glm_combined_ncdf_filenames_with_npy_files.csv')
            dates = list(df['date_time'])
            if plt_glm == True:
                f_exist = list_gcs(proc_bucket_name, pref, ['glm.npy'], delimiter = '*/')
            else:
                f_exist = list_gcs(proc_bucket_name, pref, ['ir.npy'], delimiter = '*/')
            if len(f_exist) <= 0:
                print('Could not find ir or glm numpy file???')
                print(proc_bucket_name)
                print(pref)
                exit()
            if len(f_exist) > 1 or 'real_time' in outroot:
                fbs = [os.path.basename(f) for f in f_exist]
                for nc_file in nc_files:
                    dd_str = re.split('_s|_', os.path.basename(nc_file))[5]
                    date   = datetime.strptime(dd_str[0:-1], "%Y%j%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                    if date in dates:
                        if plt_glm == True:
                            file2 = os.path.join(os.path.dirname(ir_file), dd_str[0:-1] + '_glm.npy')
                        else:
                            file2 = os.path.join(os.path.dirname(ir_file), dd_str[0:-1] + '_ir.npy')
                        fb = os.path.basename(file2)
                        if fb in fbs:
                            if len(i_dat) <= 0:
                                i_dat = load_npy_blobs(proc_bucket_name, os.path.dirname(f_exist[fbs.index(fb)]), os.path.basename(f_exist[fbs.index(fb)]))[0, :, :]
                            else:    
                                i_dat0 = load_npy_blobs(proc_bucket_name, os.path.dirname(f_exist[fbs.index(fb)]), os.path.basename(f_exist[fbs.index(fb)]))[0, :, :]
                                i_dat[i_dat0 > i_dat] = i_dat0[i_dat0 > i_dat]                                                          #Retain only the maximum values at each index 
                                i_dat0 = None
            else:
                if len(i_dat) <= 0:
                    i_dat = np.nanmax(load_npy_blobs(proc_bucket_name, os.path.dirname(f_exist[0]), os.path.basename(f_exist[0]))[df.dropna(subset = ['ir_files'])['Unnamed: 0'].astype(int).values, :, :], axis = 0) #Read in input images numpy file in order to extract the number of instances to loop over for specific date
                else:
                    i_dat0 = np.nanmax(load_npy_blobs(proc_bucket_name, os.path.dirname(f_exist[0]), os.path.basename(f_exist[0]))[df.dropna(subset = ['ir_files'])['Unnamed: 0'].astype(int).values, :, :], axis = 0) #Read in input images numpy file in order to extract the number of instances to loop over for specific date
                    i_dat[i_dat0 > i_dat] = i_dat0[i_dat0 > i_dat]
                    i_dat0 = None
    
#     if i_dat.shape[0] != res.shape[0]:
#         print('Results array and IR data array shapes do not match!!')
#         print('IR      array shape = ' + str(i_dat.shape[0]))
#         print('Results array shape = ' + str(res.shape[0]))
#         exit()
#     if res.ndim == 3: res = np.expand_dims(res, axis = 3)                                                                               #This is added due to a mistake where that last dimension was flattened for some pre-processing and it should not be
    with Dataset(nc_files[0]) as data0:    
        lon      = np.copy(np.asarray(data0.variables['longitude'][:]))                                                                 #Read Longitudes
        lat      = np.copy(np.asarray(data0.variables['latitude'][:]))                                                                  #Read Latitudes
        globe    = cartopy.crs.Globe(ellipse='GRS80', semimajor_axis=data0.variables['imager_projection'].semi_major_axis, semiminor_axis = data0.variables['imager_projection'].semi_minor_axis, inverse_flattening=data0.variables['imager_projection'].inverse_flattening)
        crs      = ccrs.Geostationary(central_longitude=data0.variables['imager_projection'].longitude_of_projection_origin, satellite_height=data0.variables['imager_projection'].perspective_point_height, false_easting=0, false_northing=0, globe=globe, sweep_axis = data0.variables['imager_projection'].sweep_angle_axis)
        extent00 = re.split(',', data0.variables['imager_projection'].bounds)
        extent0  = [np.asarray(float(extent00[0])), np.asarray(float(extent00[1])), np.asarray(float(extent00[2])), np.asarray(float(extent00[3]))]
    if region != None:         
        domain = extract_us_lat_lon_region(region)                                                                                      #Extract the domain boundaries for specified US state or region
        domain = domain[region.lower()]                                                                                                 #Extract the domain boundaries for specified US state or region 
 #       extent = (0, res.shape[0], 0, res.shape[1])        
        extent = (domain[0], domain[2], domain[1], domain[3])
    else:        
        if len(latlon_domain) > 0:
            extent = (latlon_domain[0], latlon_domain[2], latlon_domain[1], latlon_domain[3])
            domain = latlon_domain                                                                                                      #Set domain to user specified lat/lon coordinates 
        else:
            domain_regions = extract_us_lat_lon_region('Texas')                                                                         #Extract the domain boundaries for specified states or regions
            if dregion in domain_regions:
                latlon_domain2 = domain_regions[dregion]                                                                                #Extract the domain boundaries for specified date
                extent = [latlon_domain2[0], latlon_domain2[2], latlon_domain2[1], latlon_domain2[3]]       
                domain = latlon_domain2                                                                                                 #Domain covers entire mesoscale sector if region not set    
            else:
                extent = (np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)) #(0, res.shape[0], 0, res.shape[1])        
                domain = [np.nanmin(lon), np.nanmin(lat), np.nanmax(lon), np.nanmax(lat)]                                               #Domain covers entire mesoscale sector if region not set
#Extract date range of files in combined netCDF directory
    if plt_spc == True:
        file_attr = re.split('_s|_', os.path.basename(nc_files[0]))                                                                     #Split file string in order to extract date string of scan
        d_str     = file_attr[5]                                                                                                        #Split file string in order to extract date string of scan
        d1        = datetime.strptime(d_str[0:4] + '-' + d_str[4:7] + 'T' + d_str[7:9] + ':' + d_str[9:11] + ':' + d_str[11:-1] + 'Z', "%Y-%jT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
        file_attr = re.split('_s|_', os.path.basename(nc_files[-1]))                                                                    #Split file string in order to extract date string of scan
        d_str     = file_attr[5]                                                                                                        #Split file string in order to extract date string of scan
        d2        = datetime.strptime(d_str[0:4] + '-' + d_str[4:7] + 'T' + d_str[7:9] + ':' + d_str[9:11] + ':' + d_str[11:-1] + 'Z', "%Y-%jT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
        print('Dates for SPC report range = ' + d1 + '_' + d2)
        hail_rprt, torn_rprt, wind_rprt = read_spc_reports(date1 = d1, date2 = d2, verbose = verbose)                                   #Read in SPC hail, tornado, and wind report over date range as a pandas DataFrame
        if hail_rprt.empty and torn_rprt.empty and wind_rprt.empty:
            plt_spc = False
        else:
            ll_lon = []
            ll_lat = []
            if hail_rprt.empty == False:
                ll_lon.extend(hail_rprt['Lon'].astype('float').tolist())
                ll_lat.extend(hail_rprt['Lat'].astype('float').tolist())
            if torn_rprt.empty == False:
                ll_lon.extend(torn_rprt['Lon'].astype('float').tolist())
                ll_lat.extend(torn_rprt['Lat'].astype('float').tolist())
            if wind_rprt.empty == False:
                ll_lon.extend(wind_rprt['Lon'].astype('float').tolist())
                ll_lat.extend(wind_rprt['Lat'].astype('float').tolist())
            ex = np.where((np.asarray(ll_lon) >= domain[0]) & (np.asarray(ll_lon) <= domain[2]) & (np.asarray(ll_lat) >= domain[1]) & (np.asarray(ll_lat) <= domain[3]))
            if len(ex[0]) <= 0:
                plt_spc = False
                
    if lon.shape[0] >= 8000:
        lw = 0.3
        s  = 0.25    
    elif lon.shape[0] >= 5000:
        lw = 0.4
        s  = 0.5    
    else:
        lw = 0.75
        s  = 1    
    if plt_spc == True:        
      #  width  = (1.60*lon.shape[0]/my_dpi)                                                                                             #Calculate width of image                 
      #  height = (0.50*lon.shape[1]/my_dpi)                                                                                             #Calculate height of image                 
      #  if height > 12 or width > 12:
      #      if height > 20 or width > 20:
      #          width  = width/3.0
      #          height = height/3.0
      #      else:
      #          width  = width/2.0
      #          height = height/2.0
      #  if width <= 2 or height <= 2:
      #      width  = width*1.50
      #      height = height*1.50
        width  = 8
        height = 6
        fig    = plt.figure(figsize=(width, height), dpi = my_dpi)                                                                      #Set output image size 
        t0 = time.time()
        ax1, ax2, ax3 = setup_3_gridded_axes(fig, extent = extent, domain = domain, region = region, linewidth = lw, verbose = verbose) #Set up axes for plot    
        print(str(time.time() -t0) + ' sec to set up the plot')
    else:
     #   width  = (1.50*lon.shape[0]/my_dpi)                                                                                             #Calculate width of image                 
     #   height = (0.50*lon.shape[1]/my_dpi)                                                                                             #Calculate height of image                 
     #   if height > 12 or width > 12:
     #       if height > 20 or width > 20:
     #           width  = width/3.0
     #           height = height/3.0
     #       else:
     #           width  = width/2.0
     #           height = height/2.0
     #   if width <= 2 or height <= 2:
     #       width  = width*1.50
     #       height = height*1.50
        width  = 8
        height = 6
        fig    = plt.figure(figsize=(width, height), dpi = my_dpi)                                                                      #Set output image size 
        ax1, ax2 = setup_2_gridded_axes(fig, extent = extent, domain = domain, region = region, linewidth = lw, verbose = verbose)      #Set up axes for plot    
   
#    res   = res[:, :, :, 0]                                                                                                            #Extract input data for full domain
    if verbose == True:    
        print(i_dat.shape)    
        print(res.shape)    
    
#    i_dat = np.amax(i_dat, axis = 0)    
    i_dat[i_dat <= 0.0] = np.nan    
#     res[res < pthresh]  = 0.0        
#     res = np.nansum(res, axis = 0)                                                                                                    #Add the 0th dimension to get time aggregation    
#     res[res > 1]    = 1                                                                                                               #Max value allowed is 1   
#     res[res == 0.0] = np.nan
#     res[res <= 0.03] = np.nan
#     print(np.nanmin(res))
#     print(np.nanmax(res))
#     print()
    p_ind = np.where(np.isfinite(res))
    if len(p_ind[0]) > 0:
        lon0  = list(lon[p_ind[0], p_ind[1]])
        lat0  = list(lat[p_ind[0], p_ind[1]])
        lon0  = (np.asarray(lon0) + x_pc).tolist()
        lat0  = (np.asarray(lat0) - y_pc).tolist()
        p0    = list(res[p_ind[0], p_ind[1]])
    else:    
        lon0  = []
        lat0  = []
        p0    = []
#    out_img3  = ax1.pcolormesh(lon, lat, res, vmin = 0.0, vmax = 1.0, cmap = cpt_convert, extent = extent, interpolation = None)       #Create image from model results data
    out_img3 = ax1.scatter(lon0,lat0,                                                                         
                  transform = ccrs.PlateCarree(),                                                                                       #Transform the cartopy projection correctly
                  c         = p0, cmap=cpt_convert.reversed(),
                  vmin      = 0.0, vmax = 1.0, 
                  marker    = 'o', 
                  s         = 0.10, linewidth = 0.10)  
    ax1.set_title('Aggregated Model ' + mod0 + ' Results')
    ir_ticks = np.arange(0.0, 1.1, 0.1, dtype = np.single)
    cbar1 = fig.colorbar(                                                                                                               #Set up the colorbar
          out_img3,                                                                                                                     #Plot a colorbar for this cartopy map
          cax          = fig.add_axes([ax1.get_position().x0, ax1.get_position().y0-0.07, ax1.get_position().x1-ax1.get_position().x0, 0.02]),    #Set the axis to plot the colorbar to
          orientation = 'horizontal',                                                                                                   #Make the colorbar horizontal, "vertical if you want it vertical
          label       = mod0 + ' Likelihood',                                                                                           #Set colorbar label
          #spacing     = 'proportional',                                                                                                    
          shrink      = 0.5,                                                                                                            #Shrink the plot to 70% the size of the subplot
          ticks       = ir_ticks,                                                                                                       #Set the colorbar ticks
          )                             
    cbar_ticks = [f'{x:.2f}' for x in ir_ticks]                      
    cbar1.set_ticklabels(cbar_ticks)                                                                                                    #Set the colorbar tick labels
    cpt_convert.set_under(turbo_colormap.turbo_colormap_data[0])
    cpt_convert.set_over(turbo_colormap.turbo_colormap_data[-1])
    if plt_glm == True:
        if np.isnan(np.sum(lon)) == True or np.isnan(np.sum(lat)) == True:
            out_img4  = ax2.pcolor(lon+x_pc, lat-y_pc, i_dat, vmin = 0.0, vmax = 1.0, shading = 'nearest', cmap = cpt_convert)#, transform = ccrs.PlateCarree())             #Create image from model results data
        else:
            out_img4  = ax2.pcolormesh(lon+x_pc, lat-y_pc, i_dat, vmin = 0.0, vmax = 1.0, shading = 'auto', cmap = cpt_convert)         #Create image from model results data

        ax2.set_title('Maximum Normalized GLM')
        ir_ticks = np.arange(0, 1.1, 0.1, dtype = np.single)
        cbar1 = fig.colorbar(                                                                                                           #Set up the colorbar
              out_img4,                                                                                                                 #Plot a colorbar for this cartopy map
              cax          = fig.add_axes([ax2.get_position().x0, ax2.get_position().y0-0.07, ax2.get_position().x1-ax2.get_position().x0, 0.02]),    #Set the axis to plot the colorbar to
              orientation = 'horizontal',                                                                                               #Make the colorbar horizontal, "vertical if you want it vertical
              label       = 'GLM Flash Extent Density',                                                                                 #Set colorbar label
              #spacing     = 'proportional',                                                                                                 
              shrink      = 0.5,                                                                                                        #Shrink the plot to 70% the size of the subplot
              ticks       = ir_ticks,                                                                                                   #Set the colorbar ticks
              )                             
        cbar_ticks = [f'{x:.2f}' for x in ir_ticks]                      
        cbar1.set_ticklabels(cbar_ticks)                                                                                                #Set the colorbar tick labels
    else:
        i_dat = (-50.0*(i_dat-1.0)) + 180.0
        if np.isnan(np.sum(lon)) == True or np.isnan(np.sum(lat)) == True:
            if x_pc == 0 and y_pc == 0:
              out_img4 = ax2.imshow(i_dat, origin = 'upper', vmin = ir_min, vmax = ir_max, transform =crs, extent = extent0, cmap = cpt_convert, interpolation = None)
            else:
              out_img4  = ax2.pcolor(lon+x_pc, lat-y_pc, i_dat, vmin = ir_min, vmax = ir_max, shading = 'auto', cmap = cpt_convert)       #Create image from model results data
        else:
            out_img4  = ax2.pcolormesh(lon+x_pc, lat-y_pc, i_dat, vmin = ir_min, vmax = ir_max, shading = 'auto', cmap = cpt_convert)   #Create image from model results data

        ax2.set_title('Mininimum BT')
        ir_ticks = np.arange(ir_min, ir_max + 5, 5, dtype = np.int32)
        if (ir_max < 230):
            extend = 'both'
        else:
            extend = 'min'        
        cbar1 = fig.colorbar(                                                                                                           #Set up the colorbar
              out_img4,                                                                                                                 #Plot a colorbar for this cartopy map
              cax          = fig.add_axes([ax2.get_position().x0, ax2.get_position().y0-0.07, ax2.get_position().x1-ax2.get_position().x0, 0.02]),    #Set the axis to plot the colorbar to
              orientation = 'horizontal',                                                                                               #Make the colorbar horizontal, "vertical if you want it vertical
              label       = 'IR Temperature (K)',                                                                                       #Set colorbar label
              #spacing     = 'proportional',                                                                                                
              shrink      = 0.5,                                                                                                        #Shrink the plot to 70% the size of the subplot
              ticks       = ir_ticks,                                                                                                   #Set the colorbar ticks
              extend      = extend,              
              )                             
        cbar_ticks = [f'{x:}' for x in ir_ticks]                      
        cbar1.set_ticklabels(cbar_ticks)                                                                                                #Set the colorbar tick labels
    if plt_spc == True:
        try:
            out_img7 = ax3.scatter(wind_rprt['Lon'].astype('float').tolist(), wind_rprt['Lat'].astype('float').tolist(),                                                                
                          transform  = ccrs.PlateCarree(),                                                                              #Transform the cartopy projection correctly
                          color      = 'blue',
                          marker     = 'o', 
                          label      = 'Wind',
                          s          = 1.0, linewidth = 1.0)  
        except KeyError:
            print('No wind data to plot')
        try:
            out_img5 = ax3.scatter(hail_rprt['Lon'].astype('float').tolist(), hail_rprt['Lat'].astype('float').tolist(),                                                                    
                          transform  = ccrs.PlateCarree(),                                                                              #Transform the cartopy projection correctly
                          color      = 'g',
                          marker     = 's', 
                          label      = 'Hail',
                          s          = 1.4, linewidth = 1.0)  
        except KeyError:
            print('No hail data to plot')
        try:
            out_img6 = ax3.scatter(torn_rprt['Lon'].astype('float').tolist(), torn_rprt['Lat'].astype('float').tolist(),                                                                
                          transform  = ccrs.PlateCarree(),                                                                              #Transform the cartopy projection correctly
                          color       = 'r',
                          marker     = '^', 
                          label      = 'Tornado',
                          s          = 1.2, linewidth = 1.0)  
        except KeyError:
            print('No tornado data to plot')
 
        ax3.legend()
        ax3.set_title('SPC Reports')
    
 #   fig.suptitle('Time Aggregated;' + plt_title)        
    fig.suptitle('Time Aggregated;' + plt_title, y = ax2.get_position().y1 + 0.07)        
        
    os.makedirs(outdir, exist_ok = True)                                                                                                #Create output directory for model validation images if it does not already exist
    out_img_name = os.path.join(outdir, use_chan + pat + 'model_' + 'time_aggregated_maps.png')        
    plt.savefig(out_img_name, dpi = my_dpi, bbox_inches = 'tight')                                                                      #Save figure
    plt.close()                                                                                                                         #Close figure window
    if write_gcs == True:        
        pref = re.split('aacp_results_imgs' + os.sep, os.path.dirname(out_img_name))[1]        
        write_to_gcs(res_bucket_name, os.path.join('aacp_results_imgs', pref), out_img_name, del_local = del_local)                     #Write image to GCP
    
    res   = None
    i_dat = None

def setup_2_gridded_axes(fig, extent = None, domain = None, region = None, label_size = 4., linewidth = 0.75, fc = None, no_grid = False, verbose = True):
    '''
    This function sets up gridded axes using a ccrs.PlateCarree() projection

    Calling sequence:
        import create_vis_ir_numpy_arrays_from_netcdf_files2
        create_vis_ir_numpy_arrays_from_netcdf_files2.create_vis_ir_numpy_arrays_from_netcdf_files2()
    Args:
        fig : Matplotlib figure from function plt.figure(figsize=(width, height))
    Functions:
        None
    Output:
        Creates PNG image of AACP results
    Keywords:
        extent     : List of floating point longitude and latitudes giving the extent of image domain [x0, x1, y0, y1]
                     DEFUALT = None.
        domain     : List of floating point longitude and latitudes giving the extent of image domain [x0, y0, x1, y1]
                     DEFAULT = None
        region     : State or sector of US to plot data around. 
                     DEFAULT = None -> plot for full domain.
        label_size : FLOAT specifying character size of longitudes and latitudes plotted on map
                     DEFAULT = 4. (ONLY matters if no_grid = False)
        linewidth  : FLOAT specifying the width of the longitude and latitude lines
                     DEFAULT = 0.75 (ONLY matters if no_grid = False)
        fc         : STRING specifying the facecolor to plot land on figure
                     DEFAULT = feature.COLORS['land']
        no_grid    : BOOL keyword to specify whether or not to plot latitude and longitude grid
                     DEFAULT = False -> do not plot lat/lon grid on map
        verbose    : BOOL keyword to specify whether or not to print verbose informational messages.
                     DEFAULT = False which implies to not print verbose informational messages to terminal
    Returns:
        Axes of the multiplot
    Author and history:
        John W. Cooney           2021-07-20.
    '''
    ax1 = fig.add_subplot(                                                                                                              #Adds subplots to a figure
                              1,2,1,                                                                                                    #Number of and position of each subpanel (i.e. 1,1,1 refers to 1 row 1 column the first subplot)
                              projection = ccrs.PlateCarree()                                                                           #Cartopy projection to make the axis mapable (i.e. a geoaxis)
                              )                       
    ax2 = fig.add_subplot(                                                                                                              #Adds subplots to a figure
                              1,2,2,                                                                                                    #Number of and position of each subpanel (i.e. 1,1,1 refers to 1 row 1 column the first subplot)
                              projection = ccrs.PlateCarree()                                                                           #Cartopy projection to make the axis mapable (i.e. a geoaxis)
                              )                       
    #Set up a map                       
    if region == None:              
        ax1.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())                       
        ax2.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())                       
    else:              
        ax1.set_extent([domain[0], domain[2], domain[1], domain[3]], crs=ccrs.PlateCarree())                       
        ax2.set_extent([domain[0], domain[2], domain[1], domain[3]], crs=ccrs.PlateCarree())                       
    if no_grid == False:
        gl = ax1.gridlines(                                                                                                             #Adds gridlines to the plot
            draw_labels = True,                                                                                                         
            crs         = ccrs.PlateCarree(),                                                                                           #Set the projection again
            #xlocs       = lon_ticks,                                                                                                   
            #ylocs       = lat_ticks,                                                                                                   
            color       = 'black',                                                                                                      #Line color
            linewidth   = 0.10,                                                                                                         #Line width
        )                                                                                                            
        gl.xlabel_style  = {'size': label_size, 'color': 'black'}                                                                       #Set the x-axis (i.e. lon label style)
        gl.ylabel_style  = {'size': label_size, 'color': 'black'}                                                                       #Set the y-axis (i.e. lat label style)
        gl.top_labels    = False                                                                                                        #Tell cartopy not to plot labels at the top of the plot
        gl.right_labels  = False                                                                                                        #Tell cartopy not to plot labels on the right side of the plot
        gl.xformatter    = LONGITUDE_FORMATTER                                                                                          #Format longitude points correctly
        gl.yformatter    = LATITUDE_FORMATTER                                                                                           #Format latitude points correctly
        gl.xlines        = True                                                                                                         #Tell cartopy to include longitude lines
        gl.ylines        = True                                                                                                         #Tell cartopy to include latitude lines
                   
#    ax1.coastlines(linewidth = linewidth)                                                                                               #Adds coastlines to the plot
#    ax1.add_feature(feature.OCEAN)                       
    ax1.set_facecolor(np.asarray([0.59375   , 0.71484375, 0.8828125 ]))
    if fc == None:
        ax1.add_feature(feature.LAND, edgecolor='lightgray', linewidth = linewidth, facecolor=feature.COLORS['land'])                   #Adds filled land to the plot
        ax1.add_feature(feature.STATES, edgecolor='lightgray', linewidth = linewidth)                       
        ax1.add_feature(feature.BORDERS, edgecolor='lightgray', linewidth = linewidth)                       
    else:
        ax1.add_feature(feature.LAND, edgecolor='black', linewidth = linewidth, facecolor=fc)                                           #Adds filled land to the plot
        ax1.add_feature(feature.STATES, edgecolor='black', linewidth = linewidth)                       
        ax1.add_feature(feature.BORDERS, edgecolor='black', linewidth = linewidth)                       
    if no_grid == False:   
        gl1 = ax2.gridlines(                                                                                                            #Adds gridlines to the plot
            draw_labels = True,                                                                                                         
            crs         = ccrs.PlateCarree(),                                                                                           #Set the projection again
            #xlocs       = lon_ticks,                                                                                                   
            #ylocs       = lat_ticks,                                                                                                   
            color       = 'black',                                                                                                      #Line color
            linewidth   = 0.10,                                                                                                         #Line width
        )                                                                                                            
        gl1.xlabel_style  = {'size': label_size, 'color': 'black'}                                                                      #Set the x-axis (i.e. lon label style)
        gl1.ylabel_style  = {'size': label_size, 'color': 'black'}                                                                      #Set the y-axis (i.e. lat label style)
        gl1.top_labels    = False                                                                                                       #Tell cartopy not to plot labels at the top of the plot
        gl1.right_labels  = False                                                                                                       #Tell cartopy not to plot labels on the right side of the plot
        gl1.xformatter    = LONGITUDE_FORMATTER                                                                                         #Format longitude points correctly
        gl1.yformatter    = LATITUDE_FORMATTER                                                                                          #Format latitude points correctly
        gl1.xlines        = True                                                                                                        #Tell cartopy to include longitude lines
        gl1.ylines        = True                                                                                                        #Tell cartopy to include latitude lines
                   
#    ax2.coastlines(linewidth = linewidth)                                                                                               #Adds coastlines to the plot
  #  ax2.add_feature(feature.OCEAN)                       
    ax2.set_facecolor(np.asarray([0.59375   , 0.71484375, 0.8828125 ]))
    if fc == None:
        ax2.add_feature(feature.LAND, edgecolor='lightgray', linewidth = linewidth, facecolor=feature.COLORS['land'])                   #Adds filled land to the plot
        ax2.add_feature(feature.STATES, edgecolor='lightgray', linewidth = linewidth)                       
        ax2.add_feature(feature.BORDERS, edgecolor='lightgray', linewidth = linewidth)                       
    else:
        ax2.add_feature(feature.LAND, edgecolor='black', linewidth = linewidth, facecolor=fc)                                           #Adds filled land to the plot
        ax2.add_feature(feature.STATES, edgecolor='black', linewidth = linewidth)                       
        ax2.add_feature(feature.BORDERS, edgecolor='black', linewidth = linewidth)                       
   
    return(ax1, ax2)

def setup_3_gridded_axes(fig, extent = None, domain = None, region = None, label_size = 4., linewidth = 0.75, fc = None, no_grid = False, verbose = True):
    '''
    This function sets up gridded axes using a ccrs.PlateCarree() projection

    Calling sequence:
        import create_vis_ir_numpy_arrays_from_netcdf_files2
        create_vis_ir_numpy_arrays_from_netcdf_files2.create_vis_ir_numpy_arrays_from_netcdf_files2()
    Args:
        fig : Matplotlib figure from function plt.figure(figsize=(width, height))
    Functions:
        None
    Output:
        Creates PNG image of AACP results
    Keywords:
        extent     : List of floating point longitude and latitudes giving the extent of image domain [x0, x1, y0, y1]
                     DEFUALT = None.
        domain     : List of floating point longitude and latitudes giving the extent of image domain [x0, y0, x1, y1]
                     DEFAULT = None
        region     : State or sector of US to plot data around. 
                     DEFAULT = None -> plot for full domain.
        label_size : FLOAT specifying character size of longitudes and latitudes plotted on map
                     DEFAULT = 4. (ONLY matters if no_grid = False)
        linewidth  : FLOAT specifying the width of the longitude and latitude lines
                     DEFAULT = 0.75 (ONLY matters if no_grid = False)
        fc         : STRING specifying the facecolor to plot land on figure
                     DEFAULT = feature.COLORS['land']
        no_grid    : BOOL keyword to specify whether or not to plot latitude and longitude grid
                     DEFAULT = False -> do not plot lat/lon grid on map
        verbose    : BOOL keyword to specify whether or not to print verbose informational messages.
                     DEFAULT = False which implies to not print verbose informational messages to terminal
    Returns:
        Axes of the multiplot
    Author and history:
        John W. Cooney           2021-07-20.
    '''
    ax1 = fig.add_subplot(                                                                                                              #Adds subplots to a figure
                              1,3,1,                                                                                                    #Number of and position of each subpanel (i.e. 1,1,1 refers to 1 row 1 column the first subplot)
                              projection = ccrs.PlateCarree()                                                                           #Cartopy projection to make the axis mapable (i.e. a geoaxis)
                              )                       
    ax2 = fig.add_subplot(                                                                                                              #Adds subplots to a figure
                              1,3,2,                                                                                                    #Number of and position of each subpanel (i.e. 1,1,1 refers to 1 row 1 column the first subplot)
                              projection = ccrs.PlateCarree()                                                                           #Cartopy projection to make the axis mapable (i.e. a geoaxis)
                              )                       
    ax3 = fig.add_subplot(                                                                                                              #Adds subplots to a figure
                              1,3,3,                                                                                                    #Number of and position of each subpanel (i.e. 1,1,1 refers to 1 row 1 column the first subplot)
                              projection = ccrs.PlateCarree()                                                                           #Cartopy projection to make the axis mapable (i.e. a geoaxis)
                              )                       
    #Set up a map                       
    if region == None:              
        ax1.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())                       
        ax2.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())                       
        ax3.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())                       
    else:              
        ax1.set_extent([domain[0], domain[2], domain[1], domain[3]], crs=ccrs.PlateCarree())                       
        ax2.set_extent([domain[0], domain[2], domain[1], domain[3]], crs=ccrs.PlateCarree())                       
        ax3.set_extent([domain[0], domain[2], domain[1], domain[3]], crs=ccrs.PlateCarree())                       
    if no_grid == False:
        gl = ax1.gridlines(                                                                                                             #Adds gridlines to the plot
            draw_labels = True,                                                                                                         
            crs         = ccrs.PlateCarree(),                                                                                           #Set the projection again
            #xlocs       = lon_ticks,                                                                                                   
            #ylocs       = lat_ticks,                                                                                                   
            color       = 'black',                                                                                                      #Line color
            linewidth   = 0.10,                                                                                                         #Line width
        )                                                                                                            
        gl.xlabel_style  = {'size': label_size, 'color': 'black'}                                                                       #Set the x-axis (i.e. lon label style)
        gl.ylabel_style  = {'size': label_size, 'color': 'black'}                                                                       #Set the y-axis (i.e. lat label style)
        gl.top_labels    = False                                                                                                        #Tell cartopy not to plot labels at the top of the plot
        gl.right_labels  = False                                                                                                        #Tell cartopy not to plot labels on the right side of the plot
        gl.xformatter    = LONGITUDE_FORMATTER                                                                                          #Format longitude points correctly
        gl.yformatter    = LATITUDE_FORMATTER                                                                                           #Format latitude points correctly
        gl.xlines        = True                                                                                                         #Tell cartopy to include longitude lines
        gl.ylines        = True                                                                                                         #Tell cartopy to include latitude lines
                   
#    ax1.coastlines(linewidth = linewidth)                                                                                               #Adds coastlines to the plot
#    ax1.add_feature(feature.OCEAN)                       
    ax1.set_facecolor(np.asarray([0.59375   , 0.71484375, 0.8828125 ]))
    if fc == None:
        ax1.add_feature(feature.LAND, edgecolor='lightgray', linewidth = linewidth, facecolor=feature.COLORS['land'])                   #Adds filled land to the plot
        ax1.add_feature(feature.STATES, edgecolor='lightgray', linewidth = linewidth)                       
  #      ax1.add_feature(feature.BORDERS, edgecolor='lightgray', linewidth = linewidth)                       
    else:
        ax1.add_feature(feature.LAND, edgecolor='black', linewidth = linewidth, facecolor=fc)                                           #Adds filled land to the plot
        ax1.add_feature(feature.STATES, edgecolor='black', linewidth = linewidth)                       
  #      ax1.add_feature(feature.BORDERS, edgecolor='black', linewidth = linewidth)                       
    if no_grid == False:
        gl1 = ax2.gridlines(                                                                                                            #Adds gridlines to the plot
            draw_labels = True,                                                                                                         
            crs         = ccrs.PlateCarree(),                                                                                           #Set the projection again
            #xlocs       = lon_ticks,                                                                                                   
            #ylocs       = lat_ticks,                                                                                                   
            color       = 'black',                                                                                                      #Line color
            linewidth   = 0.10,                                                                                                         #Line width
        )                                                                                                            
        gl1.xlabel_style  = {'size': label_size, 'color': 'black'}                                                                      #Set the x-axis (i.e. lon label style)
        gl1.ylabel_style  = {'size': label_size, 'color': 'black'}                                                                      #Set the y-axis (i.e. lat label style)
        gl1.top_labels    = False                                                                                                       #Tell cartopy not to plot labels at the top of the plot
        gl1.right_labels  = False                                                                                                       #Tell cartopy not to plot labels on the right side of the plot
        gl1.xformatter    = LONGITUDE_FORMATTER                                                                                         #Format longitude points correctly
        gl1.yformatter    = LATITUDE_FORMATTER                                                                                          #Format latitude points correctly
        gl1.xlines        = True                                                                                                        #Tell cartopy to include longitude lines
        gl1.ylines        = True                                                                                                        #Tell cartopy to include latitude lines
                   
#    ax2.coastlines(linewidth = linewidth)                                                                                               #Adds coastlines to the plot
#    ax2.add_feature(feature.OCEAN)                       
    ax2.set_facecolor(np.asarray([0.59375   , 0.71484375, 0.8828125 ]))
    if fc == None:
        ax2.add_feature(feature.LAND, edgecolor='lightgray', linewidth = linewidth, facecolor=feature.COLORS['land'])                   #Adds filled land to the plot
        ax2.add_feature(feature.STATES, edgecolor='lightgray', linewidth = linewidth)                       
  #      ax2.add_feature(feature.BORDERS, edgecolor='lightgray', linewidth = linewidth)                       
    else:
        ax2.add_feature(feature.LAND, edgecolor='black', linewidth = linewidth, facecolor=fc)                                           #Adds filled land to the plot
        ax2.add_feature(feature.STATES, edgecolor='black', linewidth = linewidth)                       
  #      ax2.add_feature(feature.BORDERS, edgecolor='black', linewidth = linewidth)                       
    if no_grid == False:
        gl2 = ax3.gridlines(                                                                                                            #Adds gridlines to the plot
            draw_labels = True,                                                                                                         
            crs         = ccrs.PlateCarree(),                                                                                           #Set the projection again
            #xlocs       = lon_ticks,                                                                                                   
            #ylocs       = lat_ticks,                                                                                                   
            color       = 'black',                                                                                                      #Line color
            linewidth   = 0.10,                                                                                                         #Line width
        )                                                                                                            
        gl2.xlabel_style  = {'size': label_size, 'color': 'black'}                                                                      #Set the x-axis (i.e. lon label style)
        gl2.ylabel_style  = {'size': label_size, 'color': 'black'}                                                                      #Set the y-axis (i.e. lat label style)
        gl2.top_labels    = False                                                                                                       #Tell cartopy not to plot labels at the top of the plot
        gl2.right_labels  = False                                                                                                       #Tell cartopy not to plot labels on the right side of the plot
        gl2.xformatter    = LONGITUDE_FORMATTER                                                                                         #Format longitude points correctly
        gl2.yformatter    = LATITUDE_FORMATTER                                                                                          #Format latitude points correctly
        gl2.xlines        = True                                                                                                        #Tell cartopy to include longitude lines
        gl2.ylines        = True                                                                                                        #Tell cartopy to include latitude lines
                   
#    ax3.coastlines(linewidth = linewidth)                                                                                               #Adds coastlines to the plot
#    ax3.add_feature(feature.OCEAN)                       
    ax3.set_facecolor(np.asarray([0.59375   , 0.71484375, 0.8828125 ]))
    if fc == None:
        ax3.add_feature(feature.LAND, edgecolor='lightgray', linewidth = linewidth, facecolor=feature.COLORS['land'])                   #Adds filled land to the plot
        ax3.add_feature(feature.STATES, edgecolor='lightgray', linewidth = linewidth)                       
 #       ax3.add_feature(feature.BORDERS, edgecolor='lightgray', linewidth = linewidth)                       
    else:
        ax3.add_feature(feature.LAND, edgecolor='black', linewidth = linewidth, facecolor=fc)                                           #Adds filled land to the plot
        ax3.add_feature(feature.STATES, edgecolor='black', linewidth = linewidth)                       
 #       ax3.add_feature(feature.BORDERS, edgecolor='black', linewidth = linewidth)                       
   
    return(ax1, ax2, ax3)

def setup_4_gridded_axes(fig, extent = None, domain = None, region = None, label_size = 4., linewidth = 0.75, fc = None, no_grid = False, verbose = True):
    '''
    This function sets up gridded axes using a ccrs.PlateCarree() projection

    Calling sequence:
        import create_vis_ir_numpy_arrays_from_netcdf_files2
        create_vis_ir_numpy_arrays_from_netcdf_files2.create_vis_ir_numpy_arrays_from_netcdf_files2()
    Args:
        fig : Matplotlib figure from function plt.figure(figsize=(width, height))
    Functions:
        None
    Output:
        Creates PNG image of AACP results
    Keywords:
        extent     : List of floating point longitude and latitudes giving the extent of image domain [x0, x1, y0, y1]
                     DEFUALT = None.
        domain     : List of floating point longitude and latitudes giving the extent of image domain [x0, y0, x1, y1]
                     DEFAULT = None
        region     : State or sector of US to plot data around. 
                     DEFAULT = None -> plot for full domain.
        label_size : FLOAT specifying character size of longitudes and latitudes plotted on map
                     DEFAULT = 4. (ONLY matters if no_grid = False)
        linewidth  : FLOAT specifying the width of the longitude and latitude lines
                     DEFAULT = 0.75 (ONLY matters if no_grid = False)
        fc         : STRING specifying the facecolor to plot land on figure
                     DEFAULT = feature.COLORS['land']
        no_grid    : BOOL keyword to specify whether or not to plot latitude and longitude grid
                     DEFAULT = False -> do not plot lat/lon grid on map
        verbose    : BOOL keyword to specify whether or not to print verbose informational messages.
                     DEFAULT = False which implies to not print verbose informational messages to terminal
    Returns:
        Axes of the multiplot
    Author and history:
        John W. Cooney           2021-07-20.
    '''
    ax1 = fig.add_subplot(                                                                                                              #Adds subplots to a figure
                              2,2,1,                                                                                                    #Number of and position of each subpanel (i.e. 1,1,1 refers to 1 row 1 column the first subplot)
                              projection = ccrs.PlateCarree()                                                                           #Cartopy projection to make the axis mapable (i.e. a geoaxis)
                              )                       
    ax2 = fig.add_subplot(                                                                                                              #Adds subplots to a figure
                              2,2,2,                                                                                                    #Number of and position of each subpanel (i.e. 1,1,1 refers to 1 row 1 column the first subplot)
                              projection = ccrs.PlateCarree()                                                                           #Cartopy projection to make the axis mapable (i.e. a geoaxis)
                              )                       
    ax3 = fig.add_subplot(                                                                                                              #Adds subplots to a figure
                              2,2,3,                                                                                                    #Number of and position of each subpanel (i.e. 1,1,1 refers to 1 row 1 column the first subplot)
                              projection = ccrs.PlateCarree()                                                                           #Cartopy projection to make the axis mapable (i.e. a geoaxis)
                              )                       
    ax4 = fig.add_subplot(                                                                                                              #Adds subplots to a figure
                              2,2,4,                                                                                                    #Number of and position of each subpanel (i.e. 1,1,1 refers to 1 row 1 column the first subplot)
                              projection = ccrs.PlateCarree()                                                                           #Cartopy projection to make the axis mapable (i.e. a geoaxis)
                              )                       
    #Set up a map                       
    if region == None:              
        ax1.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())                       
        ax2.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())                       
        ax3.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())                       
        ax4.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())                       
    else:              
        ax1.set_extent([domain[0], domain[2], domain[1], domain[3]], crs=ccrs.PlateCarree())                       
        ax2.set_extent([domain[0], domain[2], domain[1], domain[3]], crs=ccrs.PlateCarree())                       
        ax3.set_extent([domain[0], domain[2], domain[1], domain[3]], crs=ccrs.PlateCarree())                       
        ax4.set_extent([domain[0], domain[2], domain[1], domain[3]], crs=ccrs.PlateCarree())                       
#ax1    
    if no_grid == False:
        gl = ax1.gridlines(                                                                                                             #Adds gridlines to the plot
            draw_labels = True,                                                                                                         
            crs         = ccrs.PlateCarree(),                                                                                           #Set the projection again
            #xlocs       = lon_ticks,                                                                                                   
            #ylocs       = lat_ticks,                                                                                                   
            color       = 'black',                                                                                                      #Line color
            linewidth   = 0.10,                                                                                                         #Line width
        )                                                                                                            
        gl.xlabel_style  = {'size': label_size, 'color': 'black'}                                                                       #Set the x-axis (i.e. lon label style)
        gl.ylabel_style  = {'size': label_size, 'color': 'black'}                                                                       #Set the y-axis (i.e. lat label style)
        gl.top_labels    = False                                                                                                        #Tell cartopy not to plot labels at the top of the plot
        gl.right_labels  = False                                                                                                        #Tell cartopy not to plot labels on the right side of the plot
        gl.xformatter    = LONGITUDE_FORMATTER                                                                                          #Format longitude points correctly
        gl.yformatter    = LATITUDE_FORMATTER                                                                                           #Format latitude points correctly
        gl.xlines        = True                                                                                                         #Tell cartopy to include longitude lines
        gl.ylines        = True                                                                                                         #Tell cartopy to include latitude lines
                   
#    ax1.coastlines(linewidth = linewidth)                                                                                               #Adds coastlines to the plot
#    ax1.add_feature(feature.OCEAN)                       
    ax1.set_facecolor(np.asarray([0.59375   , 0.71484375, 0.8828125 ]))
    if fc == None:
        ax1.add_feature(feature.LAND, edgecolor='lightgray', linewidth = linewidth, facecolor=feature.COLORS['land'])                   #Adds filled land to the plot
        ax1.add_feature(feature.STATES, edgecolor='lightgray', linewidth = linewidth)                       
        ax1.add_feature(feature.BORDERS, edgecolor='lightgray', linewidth = linewidth)                       
    else:
        ax1.add_feature(feature.LAND, edgecolor='black', linewidth = linewidth, facecolor=fc)                                           #Adds filled land to the plot
        ax1.add_feature(feature.STATES, edgecolor='black', linewidth = linewidth)                       
        ax1.add_feature(feature.BORDERS, edgecolor='black', linewidth = linewidth)                       
#         ax1.add_feature(feature.LAND, edgecolor='white', linewidth = linewidth, facecolor=fc)                                           #Adds filled land to the plot
#         ax1.add_feature(feature.STATES, edgecolor='white', linewidth = linewidth)                       
    
#ax2                    
    if no_grid == False:
        gl1 = ax2.gridlines(                                                                                                            #Adds gridlines to the plot
            draw_labels = True,                                                                                                         
            crs         = ccrs.PlateCarree(),                                                                                           #Set the projection again
            #xlocs       = lon_ticks,                                                                                                   
            #ylocs       = lat_ticks,                                                                                                   
            color       = 'black',                                                                                                      #Line color
            linewidth   = 0.10,                                                                                                         #Line width
        )                                                                                                            
        gl1.xlabel_style  = {'size': label_size, 'color': 'black'}                                                                      #Set the x-axis (i.e. lon label style)
        gl1.ylabel_style  = {'size': label_size, 'color': 'black'}                                                                      #Set the y-axis (i.e. lat label style)
        gl1.top_labels    = False                                                                                                       #Tell cartopy not to plot labels at the top of the plot
        gl1.right_labels  = False                                                                                                       #Tell cartopy not to plot labels on the right side of the plot
        gl1.xformatter    = LONGITUDE_FORMATTER                                                                                         #Format longitude points correctly
        gl1.yformatter    = LATITUDE_FORMATTER                                                                                          #Format latitude points correctly
        gl1.xlines        = True                                                                                                        #Tell cartopy to include longitude lines
        gl1.ylines        = True                                                                                                        #Tell cartopy to include latitude lines
#    ax2.coastlines(linewidth = linewidth)                                                                                               #Adds coastlines to the plot
#    ax2.add_feature(feature.OCEAN)                       
    ax2.set_facecolor(np.asarray([0.59375   , 0.71484375, 0.8828125 ]))
    if fc == None:
        ax2.add_feature(feature.LAND, edgecolor='lightgray', linewidth = linewidth, facecolor=feature.COLORS['land'])                   #Adds filled land to the plot
        ax2.add_feature(feature.STATES, edgecolor='lightgray', linewidth = linewidth)                       
        ax2.add_feature(feature.BORDERS, edgecolor='lightgray', linewidth = linewidth)                       
    else:
        ax2.add_feature(feature.LAND, edgecolor='black', linewidth = linewidth, facecolor=fc)                                           #Adds filled land to the plot
        ax2.add_feature(feature.STATES, edgecolor='black', linewidth = linewidth)                       
        ax2.add_feature(feature.BORDERS, edgecolor='black', linewidth = linewidth)                       
#ax3                    
    if no_grid == False:
        gl2 = ax3.gridlines(                                                                                                            #Adds gridlines to the plot
            draw_labels = True,                                                                                                         
            crs         = ccrs.PlateCarree(),                                                                                           #Set the projection again
            #xlocs       = lon_ticks,                                                                                                   
            #ylocs       = lat_ticks,                                                                                                   
            color       = 'black',                                                                                                      #Line color
            linewidth   = 0.10,                                                                                                         #Line width
        )                                                                                                            
        gl2.xlabel_style  = {'size': label_size, 'color': 'black'}                                                                      #Set the x-axis (i.e. lon label style)
        gl2.ylabel_style  = {'size': label_size, 'color': 'black'}                                                                      #Set the y-axis (i.e. lat label style)
        gl2.top_labels    = False                                                                                                       #Tell cartopy not to plot labels at the top of the plot
        gl2.right_labels  = False                                                                                                       #Tell cartopy not to plot labels on the right side of the plot
        gl2.xformatter    = LONGITUDE_FORMATTER                                                                                         #Format longitude points correctly
        gl2.yformatter    = LATITUDE_FORMATTER                                                                                          #Format latitude points correctly
        gl2.xlines        = True                                                                                                        #Tell cartopy to include longitude lines
        gl2.ylines        = True                                                                                                        #Tell cartopy to include latitude lines
  #  ax3.coastlines(linewidth = linewidth)                                                                                               #Adds coastlines to the plot
#    ax3.add_feature(feature.OCEAN)                       
    ax3.set_facecolor(np.asarray([0.59375   , 0.71484375, 0.8828125 ]))
    if fc == None:
        ax3.add_feature(feature.LAND, edgecolor='lightgray', linewidth = linewidth, facecolor=feature.COLORS['land'])                   #Adds filled land to the plot
        ax3.add_feature(feature.STATES, edgecolor='lightgray', linewidth = linewidth)                       
        ax3.add_feature(feature.BORDERS, edgecolor='lightgray', linewidth = linewidth)                       
    else:
        ax3.add_feature(feature.LAND, edgecolor='black', linewidth = linewidth, facecolor=fc)                                           #Adds filled land to the plot
        ax3.add_feature(feature.STATES, edgecolor='black', linewidth = linewidth)                       
        ax3.add_feature(feature.BORDERS, edgecolor='black', linewidth = linewidth)                       
#ax4                    
    if no_grid == False:
        gl3 = ax4.gridlines(                                                                                                            #Adds gridlines to the plot
            draw_labels = True,                                                                                                         
            crs         = ccrs.PlateCarree(),                                                                                           #Set the projection again
            #xlocs       = lon_ticks,                                                                                                   
            #ylocs       = lat_ticks,                                                                                                   
            color       = 'black',                                                                                                      #Line color
            linewidth   = 0.10,                                                                                                         #Line width
        )                                                                                                            
        gl3.xlabel_style  = {'size': label_size, 'color': 'black'}                                                                      #Set the x-axis (i.e. lon label style)
        gl3.ylabel_style  = {'size': label_size, 'color': 'black'}                                                                      #Set the y-axis (i.e. lat label style)
        gl3.top_labels    = False                                                                                                       #Tell cartopy not to plot labels at the top of the plot
        gl3.right_labels  = False                                                                                                       #Tell cartopy not to plot labels on the right side of the plot
        gl3.xformatter    = LONGITUDE_FORMATTER                                                                                         #Format longitude points correctly
        gl3.yformatter    = LATITUDE_FORMATTER                                                                                          #Format latitude points correctly
        gl3.xlines        = True                                                                                                        #Tell cartopy to include longitude lines
        gl3.ylines        = True                                                                                                        #Tell cartopy to include latitude lines
                   
 #   ax4.coastlines(linewidth = linewidth)                                                                                               #Adds coastlines to the plot
#    ax4.add_feature(feature.OCEAN)                       
    ax4.set_facecolor(np.asarray([0.59375   , 0.71484375, 0.8828125 ]))
    if fc == None:
        ax4.add_feature(feature.LAND, edgecolor='lightgray', linewidth = linewidth, facecolor=feature.COLORS['land'])                   #Adds filled land to the plot
        ax4.add_feature(feature.STATES, edgecolor='lightgray', linewidth = linewidth)                       
        ax4.add_feature(feature.BORDERS, edgecolor='lightgray', linewidth = linewidth)                       
    else:
        ax4.add_feature(feature.LAND, edgecolor='black', linewidth = linewidth, facecolor=fc)                                           #Adds filled land to the plot
        ax4.add_feature(feature.STATES, edgecolor='black', linewidth = linewidth)                       
        ax4.add_feature(feature.BORDERS, edgecolor='black', linewidth = linewidth)                       
    
    return(ax1, ax2, ax3, ax4)

def read_spc_reports(date1 = None, date2 = None, verbose = True):
    '''
    This is a function to read Storm Prediction Center (SPC) reports for hail, tornado, and winds. 
    Args:
        None.
    Keywords:
        date1   : Start date. List containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to start downloading 
                  DEFAULT = None -> download nearest to IR/VIS file to current time and nearest 15 GLM files. (ex. '2017-04-29 00:00:00')
        date2   : End date. String containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to end downloading 
                  DEFAULT = None -> download only files nearest to date1. (ex. '2017-04-30 03:00:00')
        verbose : BOOL keyword to specify whether or not to print verbose informational messages.
                  DEFAULT = True which implies to print verbose informational messages
    Returns:
        Pandas DataFrames for hail reports, tornado reports, and wind reports within date range provided.
    '''  
    t0 = time.time()                                                                                                                    #Start clock to time the run job process
    #Set up input and output paths   
    if date2 == None and date1 != None: date2 = date1                                                                                   #Default set end date to start date
    if date1 == None:
        date  = datetime.utcnow()                                                                                                       #Extract real time date we are running over
        d_str = ['today']                                                                                                               #Set date string to today
    else:
        day1  = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")                                                                           #Extract datetime objects for start date
        hhmm1 = day1.strftime("%H%M")                                                                                                   #Extract hour and minute of date1 as string so we do not include data prior to that time
        yyyymmddhhss1 = day1.strftime("%Y%m%d%H%M")
        if (hhmm1 < '1200'):
            day1 = day1 - timedelta(days = 1)
        day2  = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")                                                                           #Extract datetime objects for end date
        hhmm2 = day2.strftime("%H%M")                                                                                                   #Extract hour and minute of date2 as string so we do not include data after to that time
        yyyymmddhhss2 = day2.strftime("%Y%m%d%H%M")
        if (hhmm2 < '1200'):
            day2  = day2 - timedelta(hours = day2.hour, minutes = day2.minute+1)
        dt    = day2 - day1                                                                                                             #Extract date string. Used for file directory paths
        if day2 < day1:
            print('date2 is before date1??')
            exit()
        else:
            d_str  = [(day1 + timedelta(days=x)).strftime('%y%m%d') for x in range(dt.days+1)]
            d_str2 = [(day1 + timedelta(days=x)).strftime('%Y%m%d') for x in range(dt.days+2)]
    
    hail_rprt  = pd.DataFrame()                                                                                                         #Initialize data frame to store SPC hail reports
    torn_rprt  = pd.DataFrame()                                                                                                         #Initialize data frame to store SPC tornado reports
    wind_rprt  = pd.DataFrame()                                                                                                         #Initialize data frame to store SPC wind reports
    hail_rprt0 = pd.DataFrame()
    torn_rprt0 = pd.DataFrame()
    wind_rprt0 = pd.DataFrame()
    for d_count, d in enumerate(d_str):
        link = 'https://www.spc.noaa.gov/climo/reports/' + d + '_rpts_filtered.csv'                                                     #Set up link to the SPC files
        if verbose == True:
            print('Reading SPC link ' + str(link))
        
        try:
            hail_rprt0 = pyn.get_spc_storm_reports(link, type_of_df = 'hail').df                                                        #Read the Storm Prediction Center hail report
        except ValueError:
            print('No hail data in ' + link)          
        
        try:
            torn_rprt0 = pyn.get_spc_storm_reports(link, type_of_df = 'tornado').df                                                     #Read the Storm Prediction Center tornado report
        except ValueError:
            print('No tornado data in ' + link)          
        
        try:
            wind_rprt0 = pyn.get_spc_storm_reports(link, type_of_df = 'wind').df                                                        #Read the Storm Prediction Center wind report
        except ValueError:
            print('No wind data in ' + link)          
        
        if hail_rprt0.empty == False: 
            hail_rprt0['Date'] = d_str2[d_count]
            hail_rprt0['Date'][hail_rprt0['Time'] >= '1200'] = d_str2[d_count]
            hail_rprt0['Date'][hail_rprt0['Time'] < '1200'] = d_str2[d_count+1]
        if torn_rprt0.empty == False:
            torn_rprt0['Date'] = d_str2[d_count]
            torn_rprt0['Date'][torn_rprt0['Time'] >= '1200'] = d_str2[d_count]
            torn_rprt0['Date'][torn_rprt0['Time'] < '1200'] = d_str2[d_count+1]
        if wind_rprt0.empty == False:
            wind_rprt0['Date'] = d_str2[d_count]
            wind_rprt0['Date'][wind_rprt0['Time'] >= '1200'] = d_str2[d_count]
            wind_rprt0['Date'][wind_rprt0['Time'] < '1200'] = d_str2[d_count+1]
        
        if d_count == 0:
            if hhmm1 >= '1200':
                if len(d_str) == 1 and hhmm2 > '1200':
                    if hail_rprt0.empty == False: hail_rprt0 = hail_rprt0[((d_str2[d_count] + hail_rprt0['Time'] >= yyyymmddhhss1) & (hail_rprt0['Time'] <= hhmm2))]  #Retain only SPC reports after date1 and prior to date2  
                    if torn_rprt0.empty == False: torn_rprt0 = torn_rprt0[((d_str2[d_count] + torn_rprt0['Time'] >= yyyymmddhhss1) & (torn_rprt0['Time'] <= hhmm2))]  #Retain only SPC reports after date1 and prior to date2
                    if wind_rprt0.empty == False: wind_rprt0 = wind_rprt0[((d_str2[d_count] + wind_rprt0['Time'] >= yyyymmddhhss1) & (wind_rprt0['Time'] <= hhmm2))]  #Retain only SPC reports after date1 and prior to date2                
                else:
                    if len(d_str) == 1:
                        if hail_rprt0.empty == False: hail_rprt0 = hail_rprt0[((d_str2[d_count] + hail_rprt0['Time'] >= yyyymmddhhss1) | (hail_rprt0['Time'] <= hhmm2))]  #Retain only SPC reports after date1 and prior to date2  
                        if torn_rprt0.empty == False: torn_rprt0 = torn_rprt0[((d_str2[d_count] + torn_rprt0['Time'] >= yyyymmddhhss1) | (torn_rprt0['Time'] <= hhmm2))]  #Retain only SPC reports after date1 and prior to date2
                        if wind_rprt0.empty == False: wind_rprt0 = wind_rprt0[((d_str2[d_count] + wind_rprt0['Time'] >= yyyymmddhhss1) | (wind_rprt0['Time'] <= hhmm2))]  #Retain only SPC reports after date1 and prior to date2
            else:
                if hail_rprt0.empty == False: hail_rprt0 = hail_rprt0[((hail_rprt0['Time'] > hhmm1) & (hail_rprt0['Time'] < '1200'))]   #Retain only SPC reports after date1 and prior to date2  
                if torn_rprt0.empty == False: torn_rprt0 = torn_rprt0[((torn_rprt0['Time'] > hhmm1) & (torn_rprt0['Time'] < '1200'))]   #Retain only SPC reports after date1 and prior to date2
                if wind_rprt0.empty == False: wind_rprt0 = wind_rprt0[((wind_rprt0['Time'] > hhmm1) & (wind_rprt0['Time'] < '1200'))]   #Retain only SPC reports after date1 and prior to date2
        
        elif (d_count == len(d_str)-1):
            if hhmm2 >= '1200':
                if hail_rprt0.empty == False: hail_rprt0 = hail_rprt0[((hail_rprt0['Time'] < hhmm2) & (hail_rprt0['Time'] > '1159'))]   #Retain only SPC reports after date1 and prior to date2  
                if torn_rprt0.empty == False: torn_rprt0 = torn_rprt0[((torn_rprt0['Time'] < hhmm2) & (torn_rprt0['Time'] > '1159'))]   #Retain only SPC reports after date1 and prior to date2
                if wind_rprt0.empty == False: wind_rprt0 = wind_rprt0[((wind_rprt0['Time'] < hhmm2) & (wind_rprt0['Time'] > '1159'))]   #Retain only SPC reports after date1 and prior to date2
            else:
                if hail_rprt0.empty == False: hail_rprt0 = hail_rprt0[~((hail_rprt0['Time'] > hhmm2) & (hail_rprt0['Time'] < '1200'))]  #Retain only SPC reports after date1 and prior to date2  
                if torn_rprt0.empty == False: torn_rprt0 = torn_rprt0[~((torn_rprt0['Time'] > hhmm2) & (torn_rprt0['Time'] < '1200'))]  #Retain only SPC reports after date1 and prior to date2
                if wind_rprt0.empty == False: wind_rprt0 = wind_rprt0[~((wind_rprt0['Time'] > hhmm2) & (wind_rprt0['Time'] < '1200'))]  #Retain only SPC reports after date1 and prior to date2
        if hail_rprt0.empty == False: hail_rprt0 = hail_rprt0[hail_rprt0['Lon'] != 'Lon']
        if torn_rprt0.empty == False: torn_rprt0 = torn_rprt0[torn_rprt0['Lon'] != 'Lon']
        if wind_rprt0.empty == False: wind_rprt0 = wind_rprt0[wind_rprt0['Lon'] != 'Lon']
        
#         hail_rprt  = hail_rprt.append(hail_rprt0)                                                                                       #Append dataframes (deprecated)
#         torn_rprt  = torn_rprt.append(torn_rprt0)                                                                                       #Append dataframes (deprecated)
#         wind_rprt  = wind_rprt.append(wind_rprt0)                                                                                       #Append dataframes (deprecated)
        hail_rprt = pd.concat([hail_rprt, hail_rprt0], axis = 0, join = 'outer')                                                        #Concatenate the dataframes
        torn_rprt = pd.concat([torn_rprt, torn_rprt0], axis = 0, join = 'outer')                                                        #Concatenate the dataframes
        wind_rprt = pd.concat([wind_rprt, wind_rprt0], axis = 0, join = 'outer')                                                        #Concatenate the dataframes
        hail_rprt0 = pd.DataFrame()
        torn_rprt0 = pd.DataFrame()
        wind_rprt0 = pd.DataFrame()
   
    if hail_rprt.empty == False: hail_rprt.reset_index(drop=True, inplace = True)
    if torn_rprt.empty == False: torn_rprt.reset_index(drop=True, inplace = True)
    if wind_rprt.empty == False: wind_rprt.reset_index(drop=True, inplace = True)
    
    return(hail_rprt, torn_rprt, wind_rprt)       

def main():
    visualize_time_aggregated_results()
    
if __name__ == '__main__':
    main()