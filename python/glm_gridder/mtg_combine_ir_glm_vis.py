#+
# Name:
#     mtg_combine_ir_glm_vis.py
# Purpose:
#     This is a script to create the combined netCDF file for MTG data files
# Calling sequence:
#     import mtg_combine_ir_glm_vis
# Input:
#     None.
# Functions:
#     extract_universal_vars : Extracts smoothed/resized GLM and resized IR data on VIS data grid
#     mtg_combine_ir_glm_vis : Creates and write netCDF file that combines IR, VIS, and GLM data
# Output:
#     Combined ir_vis_glm netCDF file. Also returns name and file path of combined netCDF file.
# Keywords:
#     infile               : String keyword specifying MTG data filename and path
#                            DEFAULT = '../../../mtg-data/20241029/data/W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-FD--CHK-BODY---NC4E_C_EUMT_20241029140421_IDPFI_OPE_20241029140007_20241029140017_N__C_0085_0001.nc'
#     layered_dir          : String keyword specifying directory to send the combined layer netCDF file
#                            DEFAULT = '../../../goes-data/combined_nc_dir/'
#     universal_file       : IF keyword set (True), write combined netCDF file that can be read universally. GLM and IR data resampled onto VIS data grid. 
#                            DEFAULT = False
#     rewrite_nc           : IF keyword set (True), rewrite the combined ir/vis/glm netCDF file.
#                            DEFAULT = False (do not rewrite combined netCDF file if one exists. Write if it does not exist though)
#     append_nc            : IF keyword set (True), append the combined ir/vis/glm netCDF file.
#                            DEFAULT = True-> Append existing netCDF file
#     no_write_vis         : IF keyword set, do not write the VIS data to the combined modality netCDF file. Setting this 
#                            keyword = True makes you write only the IR data. 
#                            DEFAULT = False -> write the IR and VIS data.
#     no_write_glm         : IF keyword set, do not write the GLM data to the combined modality netCDF file. Setting this 
#                            keyword = True makes you plot only the VIS and IR data. 
#                            DEFAULT = True -> only write the VIS and IR data.
#     no_write_irdiff      : IF keyword set, do not write the 6.3 micron IR data to the combined modality netCDF file. Setting this 
#                            keyword = True makes you plot only the VIS and IR data. 
#                            DEFAULT = True -> only write the VIS and IR data.
#     no_write_cirrus      : IF keyword set, write the difference between 1.37 micron to file.  
#                            DEFAULT = True -> only write the VIS and IR and glm data.
#     no_write_snowice     : IF keyword set, write the difference between 1.6 micron to file.  
#                            DEFAULT = True -> only write the VIS and IR and glm data.
#     no_write_dirtyirdiff : IF keyword set, write the difference between 12-10 micron to file.  
#                            DEFAULT = True -> only write the VIS and IR and glm data.
#     in_bucket_name       : Google cloud storage bucket to read raw IR/GLM/VIS files.
#                            DEFAULT = 'goes-data'
#     proj                 : metpy and xarray list containing object giving projection of plot, xmin, xmax, ymin, ymax. ex. [<cartopy.crs.Geostationary object at 0x7fa37f4fa130>, x.min(), x.max(), y.min(), y.max()]
#                            DEFAULT = None -> no projection to save to netCDF file.
#     xy_bounds            : ARRAY of floating point values giving the domain boundaries the user wants to restrict the combined netCDF and output image to
#                            [x0, y0, x1, y1]
#                            DEFAULT = [] -> write combined netCDF data for full scan region.
#     glm_thread_info      : GLM data thread info within list. If list is empty, then just read GLM data. If not empty, wait for thread to finish.
#     verbose              : BOOL keyword to specify whether or not to print verbose informational messages.
#                            DEFAULT = True which implies to print verbose informational messages
# Author and history:
#     John W. Cooney           2024-12-19. (Taken from combine_ir_glm.py but applied to MTG)
#-

### Environment Setup ###
# Package imports
import numpy as np
from netCDF4 import Dataset
import cv2
import glob
import netCDF4
import re
import sys
import os
import time
import datetime
import warnings
import scipy.ndimage as ndimage
import xarray as xr
#from pyorbital import astronomy
#sys.path.insert(1, '../')
from glm_gridder.dataScaler import DataScaler
from glm_gridder.glm_data_processing import *
from suncalc import get_position
import cartopy.crs as ccrs

def mtg_combine_ir_glm_vis(infile               = os.path.join('..', '..', '..', 'mtg-data', '20241029', 'W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-FD--CHK-TRAIL---NC4E_C_EUMT_20241029140421_IDPFI_OPE_20241029140007_20241029140017_N__C_0085_0001.nc'),
                           layered_dir          = os.path.join('..', '..', '..', 'mtg-data', 'combined_nc_dir', '20241029'), 
                           satellite            = 'MTG1',
                           domain_sector        = 'F',
                           universal_file       = False, 
                           rewrite_nc           = False, 
                           append_nc            = True,
                           no_write_vis         = False,                       
                           no_write_glm         = True,
                           no_write_irdiff      = True,
                           no_write_cirrus      = True, 
                           no_write_snowice     = True, 
                           no_write_dirtyirdiff = True, 
                           in_bucket_name       = 'mtg-data', 
                           proj                 = None, 
                           xy_bounds            = [], 
                           glm_thread_info      = [], 
                           verbose              = True):
    
    glm_file = 'dfs'
#    t00 = time.time()
    infile      = os.path.realpath(infile)                                                                                                  #Create link to real path so compatible with Mac
    layered_dir = os.path.realpath(layered_dir)                                                                                             #Create link to real path so compatible with Mac
       
    file_attr = re.split('_|-|,|\+', os.path.basename(infile))                                                                              #Split file name string to create output file name
    sat0      = file_attr[6]
    sat       = satellite.lower() 
    sector    = file_attr[11]
    fnum      = file_attr[-2]
    fres      = file_attr[10]                                                                                                               #Extract file data resolution from the TRAIL files (HRFI = hi-res or FDHSI = normal-res)
    fdir      = os.path.dirname(infile)
    files     = sorted(glob.glob(os.path.join(fdir, f'*MTI{satellite[-1]}*{fres}-{sector}-*-BODY-*_{fnum}_*.nc')))                          #Find the body files associated with the infile "Trail"
    if fres != 'FDHSI' and (not no_write_irdiff or not no_write_cirrus or not no_write_snowice or not no_write_dirtyirdiff):
        native_files = sorted(glob.glob(os.path.join(fdir, f'*MTI{satellite[-1]}*FDHSI-{sector}-*-BODY-*_{fnum}_*.nc')))                   #Find the body files associated with the infile "Trail"
        if len(native_files) <= 0:
            print('No FDHSI native resolution data files found???')
            print(fdir)
            print(f'*MTI{satellite[-1]}*FDHSI-{sector}-*-BODY-*_{fnum}_*.nc')
            exit()
        if len(native_files) != len(files):
            print('Number of FDHSI native resolution data files do not match number of high resolution files found???')
            print(f'Number of HRFI high resolution data files = {files}')
            print(f'Number of FDHSI native resolution data files = {native_files}')
            print(fdir)
            print(f'*MTI{satellite[-1]}*{fres}-{sector}-*-BODY-*_{fnum}_*.nc')
            print(f'*MTI{satellite[-1]}*FDHSI-{sector}-*-BODY-*_{fnum}_*.nc')
            exit()

    if sector == 'FD':
        sector = 'F'

    proc_lvl  = 'L' + file_attr[8].lower()
    ins       = file_attr[7]
    ds        = file_attr[23]
    de        = file_attr[24]

    ds0 = datetime.datetime.strptime(ds, "%Y%m%d%H%M%S")
    de0 = datetime.datetime.strptime(de, "%Y%m%d%H%M%S")
    dc0 = ds0 + (de0 - ds0)/2.0
    if not no_write_vis:
        xyres = 0.5
    else:
        xyres = 1.0
        
    #File patterns to use if high resolution data
    if domain_sector.lower() == 'q4':
        file_patterns = np.arange(29, 41) # chunks 29-40
    elif domain_sector.lower() == 'q3':
        file_patterns = np.arange(20, 31) # chunks 20-30
    elif domain_sector.lower() == 'q2':
        file_patterns = np.arange(10, 22) # chunks 10-21
    elif domain_sector.lower() == 'q1':
        file_patterns = np.arange(1, 14) # chunks 01-13
    elif domain_sector.lower() == 't3':
        file_patterns = np.arange(26, 41) # chunks 26-40
    elif domain_sector.lower() == 't2':
        file_patterns = np.arange(13, 28) # chunks 13-27
    elif domain_sector.lower() == 't1':
        file_patterns = np.arange(1, 17) # chunks 01-16
    elif domain_sector.lower() == 'h2':
        file_patterns = np.arange(20, 41) # chunks 20-40
    elif domain_sector.lower() == 'h1':
        file_patterns = np.arange(1, 22) # chunks 01-21
    elif domain_sector.lower() == 'f' or domain_sector.lower() == 'fd' or domain_sector.lower() == 'full':
        file_patterns = ["*_????_00[0-3][0-9].nc", "*_????_0040.nc", "*_????_0041.nc"] # Full disc; chunks 01-40
    else:
        print(f"Sector '{domain_sector}' is not supported.")
        exit()
    
            
    out_nc = os.path.join(layered_dir, f'OR_{ins}_{proc_lvl}_{domain_sector}_COMBINED_s{ds0.strftime("%Y%j%H%M%S0")}_e{de0.strftime("%Y%j%H%M%S0")}_c{dc0.strftime("%Y%j%H%M%S0")}.nc')#Creates output file name and path for combined ir_glm_vis file
    if (not os.path.isfile(out_nc) or rewrite_nc or append_nc):
#        t0 = time.time()
        os.makedirs(os.path.dirname(out_nc), exist_ok = True)                                                                               #Create output file path if does not already exist
         
#         # Track min/max indices for subsetting
#         min_row, max_row = 22272, 0
#         min_col, max_col = 22272, 0
        
        if no_write_vis:
            shape = (5568, 5568)
            bt = np.full(shape, np.nan)
        else:
            shape = (22272, 22272)
      
#         x = np.full(shape, np.nan)
#         y = np.full(shape, np.nan)
 
        glm_raw_img = []
        bt2         = []
        bt3         = []
        snowice_img = []
        cirrus_img  = []
        if not no_write_vis:
            bt      = np.full((11136, 11136), np.nan)
            vis_img = np.full(shape, np.nan)
            if not no_write_snowice:
                snowice_img = np.full((11136, 11136), np.nan)
            if not no_write_cirrus:
                cirrus_img = np.full((11136, 11136), np.nan)
            
        if not no_write_irdiff:
            bt2 = np.full((5568, 5568), np.nan)
        
        if not no_write_dirtyirdiff:
            bt3 = np.full((5568, 5568), np.nan)
        
        ### Dataset Creation ### 
        for ff, file in enumerate(files):            
            cont = True
            if domain_sector.lower() != 'f':
                chunk_num = int(re.split('_|-|,|\+|.nc', os.path.basename(file))[-2])
                cont = chunk_num in file_patterns
            if not cont:
                continue
                    
            if no_write_vis:
                with xr.open_dataset(file, group="data/ir_105/measured") as ir:
                    # IR datasets        
                    ir0 = np.asarray(ir['effective_radiance'].values[:, :])
                    bt0 = mtg_get_ir_bt_from_rad(ir, ir0)
                
                    sr, sc = int(ir['start_position_row']) - 1, int(ir['start_position_column']) - 1
                    er, ec = int(ir['end_position_row']), int(ir['end_position_column'])
                    
                    bt[sr:er, sc:ec] = bt0
#                     min_row, max_row = min(min_row, sr), max(max_row, er)
#                     min_col, max_col = min(min_col, sc), max(max_col, ec)
                
                    if not no_write_irdiff:
                        with xr.open_dataset(file, group="data/wv_63/measured") as ir2:
                            bt00 = mtg_get_ir_bt_from_rad(ir2, ir2['effective_radiance'].values)
                            bt2[sr:er, sc:ec] = bt00 - bt0

                    if not no_write_dirtyirdiff:
                        with xr.open_dataset(file, group="data/ir_123/measured") as ir2:
                            bt00 = mtg_get_ir_bt_from_rad(ir2, ir2['effective_radiance'].values)
                            bt3[sr:er, sc:ec] = bt00 - bt0
            else:
                with xr.open_dataset(file, group="data/vis_06_hr/measured") as vis, \
                     xr.open_dataset(file, group="state/celestial") as sun:
                    
                    sr, sc = int(vis['start_position_row']) - 1, int(vis['start_position_column']) - 1
                    er, ec = int(vis['end_position_row']), int(vis['end_position_column'])
                    vis_img[sr:er, sc:ec] = mtg_get_ref_from_rad(vis, sun)                                                                  #Calculate reflectance from radiance
         
#                     min_row, max_row = min(min_row, sr), max(max_row, er)
#                     min_col, max_col = min(min_col, sc), max(max_col, ec)
         
                    if not no_write_irdiff or not no_write_dirtyirdiff or not no_write_snowice or not no_write_cirrus:
                        chunk_num2 = int(re.split('_|-|,|\+|.nc', os.path.basename(native_files[ff]))[-2])
                        if chunk_num2 != chunk_num:
                            raise ValueError(f"Chunk mismatch:\n{file}\n{native_files[ff]}")
         
                    if not no_write_irdiff or not no_write_dirtyirdiff:
                        with xr.open_dataset(native_files[ff], group="data/ir_105/measured") as ir:
                            bt0 = mtg_get_ir_bt_from_rad(ir, ir['effective_radiance'].values)
         
                    if not no_write_irdiff:
                        with xr.open_dataset(native_files[ff], group="data/wv_63/measured") as ir2:
                            bt00 = mtg_get_ir_bt_from_rad(ir2, ir2['effective_radiance'].values)
                            sr2, sc2 = int(ir2['start_position_row']) - 1, int(ir2['start_position_column']) - 1
                            er2, ec2 = int(ir2['end_position_row']), int(ir2['end_position_column'])
                            bt2[sr2:er2, sc2:ec2] = bt00 - bt0
         
                    if not no_write_dirtyirdiff:
                        with xr.open_dataset(native_files[ff], group="data/ir_123/measured") as ir2:
                            bt00 = mtg_get_ir_bt_from_rad(ir2, ir2['effective_radiance'].values)
                            sr2, sc2 = int(ir2['start_position_row']) - 1, int(ir2['start_position_column']) - 1
                            er2, ec2 = int(ir2['end_position_row']), int(ir2['end_position_column'])
                            bt3[sr2:er2, sc2:ec2] = bt00 - bt0
         
                    if not no_write_snowice:
                        with xr.open_dataset(native_files[ff], group="data/nir_16/measured") as snowice:
                            sr2, sc2 = int(snowice['start_position_row']) - 1, int(snowice['start_position_column']) - 1
                            er2, ec2 = int(snowice['end_position_row']), int(snowice['end_position_column'])
                            snowice_img[sr2:er2, sc2:ec2] = mtg_get_ref_from_rad(snowice, sun)
         
                    if not no_write_cirrus:
                        with xr.open_dataset(native_files[ff], group="data/nir_13/measured") as cirrus:
                            sr2, sc2 = int(cirrus['start_position_row']) - 1, int(cirrus['start_position_column']) - 1
                            er2, ec2 = int(cirrus['end_position_row']), int(cirrus['end_position_column'])
                            cirrus_img[sr2:er2, sc2:ec2] = mtg_get_ref_from_rad(cirrus, sun)
                    
                    with xr.open_dataset(file, group="data/ir_105_hr/measured") as ir:
                        sr2, sc2 = int(ir['start_position_row']) - 1, int(ir['start_position_column']) - 1
                        er2, ec2 = int(ir['end_position_row']), int(ir['end_position_column'])
                        bt[sr2:er2, sc2:ec2] = mtg_get_ir_bt_from_rad(ir, ir['effective_radiance'].values[:, :])                            #Calculate IR Temperatures (K)
        
        if not no_write_vis:
            bt = upscale_img_to_fit(bt, vis_img)
            if not no_write_irdiff:
                bt2 = upscale_img_to_fit(bt2, vis_img)            
            if not no_write_dirtyirdiff:
                bt3 = upscale_img_to_fit(bt3, vis_img)            
            if not no_write_snowice:
                snowice_img = upscale_img_to_fit(snowice_img, vis_img)            
            if not no_write_cirrus:
                cirrus_img = upscale_img_to_fit(cirrus_img, vis_img)            
            
        # Vis datasets 
        if not no_write_vis:
            ### VIS Data to reflectance ###      
            pat_proj = '_vis_'
#            xyres    = 0#vis.spatial_resolution   #STOP here to provide resolution
        else:
            vis_img  = bt*0.0
            pat_proj = '_ir_'
#            xyres    = 0#ir.spatial_resolution    #STOP here to provide resolution
              
        t0 = time.time()
        ### NetCDF File Creation ###      
        # file declaration      
        if verbose: print('Writing combined VIS/IR/GLM output netCDF ' + out_nc)
        Scaler = DataScaler( nbytes = 4, signed = True )                                                                                      #Extract data scaling and offset ability for np.int32
        yesman = False
        if os.path.isfile(out_nc) and append_nc and not rewrite_nc:
            try:
              f = netCDF4.Dataset(out_nc,'a', format='NETCDF4')                                                                               #'a' stands for append (write netCDF file of combined datasets)        
              yesman = True
            except:
              print('File existed but was corrupted so am rewriting it.')
                
        if yesman:
            x_inds  = list(f.x_inds)
            y_inds  = list(f.y_inds)
            lon_arr = np.asarray(f.variables['longitude'])
            keys0   = f.variables.keys()
            keys1   = list(keys0)
            if ('solar_zenith_angle' not in keys0) and (not no_write_vis or not no_write_cirrus or not no_write_snowice):
                date   = netCDF4.num2date( ir.variables['time'][:], ir.variables['time'].units)                                             #Read in date
                date   = datetime.datetime(date.year, date.month, date.day, hour=date.hour, minute=date.minute, second=date.second, tzinfo=datetime.timezone.utc)
                lat_arr = np.asarray(f.variables['latitude'])
                with warnings.catch_warnings():
                  warnings.simplefilter("ignore", category=RuntimeWarning)
                  zen    = np.radians(90.0)-get_position(date, lon_arr, lat_arr)['altitude']
                  coszen = np.cos(zen)
            elif ('solar_zenith_angle' in keys0) and (not no_write_vis or not no_write_cirrus or not no_write_snowice):
                zen    = np.asarray(f.variables['solar_zenith_angle'])[0, :, :]
                coszen = np.cos(np.deg2rad(zen))
        else:
            keys0 = {}
            keys1 = list(keys0)
            f = netCDF4.Dataset(out_nc,'w', format='NETCDF4')                                                                               #'w' stands for write (write netCDF file of combined datasets)
            
            # latitude and longitude definition
            x_inds = []
            y_inds = []
 #           t0 = time.time()
            data = xr.open_dataset(files[0], group="data")
            proj_img = data.variables['mtg_geos_projection']
            
            # Data info
            proj_dir  = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'sat_projection_files')
            proj_file = os.path.join(proj_dir, sat + '_satellite' + pat_proj + 'full_scan_projections.nc')
            scan_mode = 'full'
            if len(xy_bounds) > 0 and not universal_file:
                print('Not set up to write subset domains if universal file is not set!!!')
                exit()
            if len(xy_bounds) > 0:
                if not os.path.exists(proj_file):
                    print('Satellite projection file for the specified indices not found????')
                    print(proj_file)
                    exit()
                with Dataset(proj_file, 'r') as scan_proj_dataset:
                    x_inds, y_inds, proj_extent_new, og_shape, lat_arr, lon_arr = get_lat_lon_subset_inds(scan_proj_dataset, xy_bounds, return_lats = True, satellite = sat, use_native_ir = no_write_vis, scan_mode = scan_mode, verbose = verbose)
#                print('Time to return subset inds = ' + str(time.time() - t0) + 'sec')
                
                if lon_arr.shape != og_shape or lat_arr.shape != og_shape:
                    print('Indexing of file shapes do not match raw data file shapes???')
                    print('Original shape for subset indices = ' + str(og_shape))
                    print('Raw data longitude array shape = ' + str(lon_arr.shape))
                    print('Raw data latitude array shape = ' + str(lat_arr.shape))
                    exit()
            else:
                with Dataset(proj_file, 'r') as scan_proj_dataset:
                    x_inds, y_inds, proj_extent, og_shape, lat_arr, lon_arr = get_lat_lon_subset_inds(scan_proj_dataset, [], return_lats = True, verbose = verbose)

            data.close()
            data = None
#             print('Lon/lat extraction = ' + str(time.time() - t0) + 'sec')
            ### Normalize VIS Data by solar zenith angle ###
            dd    = xr.open_dataset(files[0], decode_times = False)
            date = datetime.datetime.strptime(dd.attrs['time_coverage_start'], "%Y%m%d%H%M%S")
            date = datetime.datetime(date.year, date.month, date.day, hour=date.hour, minute=date.minute, second=date.second, tzinfo=datetime.timezone.utc)
            if len(xy_bounds) > 0:
                lon_arr = lon_arr[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]
                lat_arr = lat_arr[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]

#            print('Calculating solar zenith angle')
#            t0 = time.time()
            
            with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=RuntimeWarning)
              zen   = np.radians(90.0)-get_position(date, lon_arr, lat_arr)['altitude']
              coszen = np.cos(zen)

            # dimension declaration
            f.createDimension('Y',     lon_arr.shape[0])
            f.createDimension('X',     lat_arr.shape[1])
            if not universal_file:
                f.createDimension('IR_Y',   ir_raw_img.shape[0])
                f.createDimension('IR_X',   ir_raw_img.shape[1])
                # Define GLM images      
                if not no_write_glm:
                    if verbose:
                        print('Waiting for gridded GLM data file')
                    while not os.path.isfile(glm_file):
                        time.sleep(0.50)                                                                                                    #Wait until file exists, then load it as numpy array as soon as it does
                    with Dataset(glm_file, "r", "NETCDF4") as glm_data:                                                                     #Read GLM data file
                        glm_raw_img, rect = fetch_glm_to_match_ir(ir, glm_data, _rect_color='r', _label='M1')      
#                        glm_raw_img, rect = fetch_glm_to_match_ir(ir, glm_data, _rect_color='r', _label='M2')      
                    f.createDimension('GLM_Y', glm_raw_img.shape[0])
                    f.createDimension('GLM_X', glm_raw_img.shape[1])
            f.createDimension('time',      1)
            
            # variable declaration for output file
            var_lon  = f.createVariable('longitude', 'f4', ('Y','X',), zlib = True, least_significant_digit = 3)
            var_lon.long_name     = 'longitude -180 to 180 degrees east'
            var_lon.standard_name = 'longitude'
            var_lon.units         = 'degrees_east'
      
            var_lat  = f.createVariable('latitude', 'f4', ('Y', 'X',), zlib = True, least_significant_digit = 3)
            var_lat.long_name     = 'latitude -90 to 90 degrees north'
            var_lat.standard_name = 'latitude'
            var_lat.units         = 'degrees_north'
            
            var_t    = f.createVariable('time', np.float64, ('time',))
            var_t.long_name       = dd.variables['time'].attrs['long_name']
            var_t.standard_name   = dd.variables['time'].attrs['standard_name']
            var_t.units           = dd.variables['time'].attrs['units']
        if not no_write_vis:
            if 'visible_reflectance' not in keys0:
                var_vis  = f.createVariable('visible_reflectance', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue)
                var_vis.set_auto_maskandscale( False )
                var_vis.long_name     = 'Visible Reflectance Normalized by Solar Zenith Angle'
                var_vis.standard_name = 'Visible_Reflectance'
                var_vis.units         = 'reflectance_normalized_by_solar_zenith_angle'
                var_vis.coordinates   = 'longitude latitude time'
            if 'solar_zenith_angle' not in keys0:
                var_zen  = f.createVariable('solar_zenith_angle', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 2)
                var_zen.set_auto_maskandscale( False )
                var_zen.long_name     = 'Solar Zenith Angle'
                var_zen.standard_name = 'solar_zenith_angle'
                var_zen.units         = 'degrees'
        if len(keys0) <= 0:
            image_projection = f.createVariable('imager_projection', np.int32, zlib = True)        
            image_projection.long_name                      = proj_img.attrs['long_name']
            image_projection.satellite_name                 = sat0
            image_projection.grid_mapping_name              = proj_img.attrs['grid_mapping_name']
            image_projection.perspective_point_height       = proj_img.attrs['perspective_point_height']
            image_projection.semi_major_axis                = proj_img.attrs['semi_major_axis']
            image_projection.semi_minor_axis                = proj_img.attrs['semi_minor_axis']
            image_projection.inverse_flattening             = proj_img.attrs['inverse_flattening']
            image_projection.latitude_of_projection_origin  = proj_img.attrs['latitude_of_projection_origin']
            image_projection.longitude_of_projection_origin = proj_img.attrs['longitude_of_projection_origin']
            image_projection.sweep_angle_axis               = proj_img.attrs['sweep_angle_axis']
            proj4_string = (
                f"+proj=geos +lon_0={proj_img.attrs['longitude_of_projection_origin']} +h={proj_img.attrs['perspective_point_height']} "
                f"+sweep={proj_img.attrs['sweep_angle_axis']} +a={proj_img.attrs['semi_major_axis']} +b={proj_img.attrs['semi_minor_axis']} +rf={proj_img.attrs['inverse_flattening']} "
                "+ellps=GRS80"
            )
                    
            # NetCDF File Description
            f.Conventions = 'CF-1.8'                                                                                                        #Write netCDF conventions attribute
            if os.path.isfile(glm_file) and not no_write_glm: 
                if not universal_file:
                    f.description = "This file combines unscaled IR data, Visible data, and GLM data into one NetCDF file. The VIS data is normalized by the solar zenith angle. The GLM data is resampled onto the IR grid using glmtools for files ±2.5 min from VIS/IR analysis time. x_inds show 0th dimension min and max indices of subsetting region. y_inds show 1st dimension."
                else:
                    f.description = "This file combines unscaled IR data, Visible data, and GLM data into one NetCDF file. The VIS data is on its original grid but is normalized by the solar zenith angle. The GLM data is resampled onto the IR grid using glmtools for files ±2.5 min from VIS/IR analysis time. The IR and GLM data are then upscaled onto the VIS grid using cv2.resize with interpolation set as cv2.INTER_NEAREST. Lastly, the GLM data is smoothed by ndimage.gaussian_filter(glm_data, sigma=1.0, order=0). x_inds show 0th dimension min and max indices of subsetting region. y_inds show 1st dimension."
            else:
                f.description = "This file combines unscaled IR data and Visible data but no GLM data into one NetCDF file. The VIS data is normalized by the solar zenith angle. x_inds show 0th dimension min and max indices of subsetting region. y_inds show 1st dimension."
            
            f.geospatial_lat_min  = str(np.nanmin(lat_arr))                                                                                 #Write min latitude as global variable
            f.geospatial_lat_max  = str(np.nanmax(lat_arr))                                                                                 #Write max latitude as global variable
            f.geospatial_lon_min  = str(np.nanmin(lon_arr))                                                                                 #Write min longitude as global variable
            f.geospatial_lon_max  = str(np.nanmax(lon_arr))                                                                                 #Write max longitude as global variable
    
            f.x_inds  = x_inds                                                                                                              #Write min latitude as global variable
            f.y_inds  = y_inds                                                                                                              #Write max latitude as global variable
            f.spatial_resolution = str(xyres).format(".2f") + 'km at nadir'                                                                 #Write satellite spatial resolution to the file
            f.proj4_string       = proj4_string
            
            if len(xy_bounds) > 0:
                image_projection.bounds = str(proj_extent_new[0]) + ',' + str(proj_extent_new[1]) + ',' + str(proj_extent_new[2]) + ',' + str(proj_extent_new[3])
            else:
                image_projection.bounds = str(proj_extent[0]) + ',' + str(proj_extent[1]) + ',' + str(proj_extent[2]) + ',' + str(proj_extent[3])
            image_projection.bounds_units = 'm'
       
        if not universal_file:
            var_ir   = f.createVariable('ir_brightness_temperature', np.int32, ('time', 'IR_Y', 'IR_X',), zlib = True, fill_value = Scaler._FillValue)
            var_ir.set_auto_maskandscale( False )
            pat  = ''
            pat2 = ''
            if not no_write_glm:
                var_glm   = f.createVariable('glm_flash_extent_density', np.int32, ('time', 'GLM_Y', 'GLM_X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)    
                var_glm.set_auto_maskandscale( False )
                long_name = str('Flash extent density within +/- 2.5 min of time variable' + pat + pat2)
                var_glm.long_name     = long_name
                var_glm.standard_name = 'Flash_extent_density'
                var_glm.units         = 'Count per nominal 3136 microradian^2 pixel per 1.0 min'
                var_glm.coordinates   = 'longitude latitude time'
            if not no_write_irdiff: 
                var_ir2   = f.createVariable('ir_brightness_temperature_diff', np.int32, ('time', 'IR_Y', 'IR_X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)    
                var_ir2.set_auto_maskandscale( False )
                var_ir2.long_name      = '6.3 - 10.5 micron Infrared Brightness Temperature Image' + pat
                var_ir2.standard_name  = 'IR_brightness_temperature_difference'
                var_ir2.units          = 'kelvin'
                var_ir2.coordinates    = 'longitude latitude time'
        else:
            if not no_write_vis:
                pat  = ' resampled onto HRFI data grid'
            else:
                pat  = ' resampled onto FDHSI data grid'
            pat2 = ''
            if 'ir_brightness_temperature' not in keys0:
                var_ir   = f.createVariable('ir_brightness_temperature', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue)
                var_ir.set_auto_maskandscale( False )
            if not no_write_irdiff and 'ir_brightness_temperature_diff' not in keys0: 
                var_ir2   = f.createVariable('ir_brightness_temperature_diff', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)    
                var_ir2.set_auto_maskandscale( False )
                var_ir2.long_name      = '6.3 - 10.5 micron Infrared Brightness Temperature Image' + pat
                var_ir2.standard_name  = 'IR_brightness_temperature_difference'
                var_ir2.units          = 'kelvin'
                var_ir2.coordinates    = 'longitude latitude time'
            if not no_write_cirrus and 'cirrus_reflectance' not in keys0: 
                var_cirrus   = f.createVariable('cirrus_reflectance', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)    
                var_cirrus.set_auto_maskandscale( False )
                var_cirrus.long_name      = '1.3 micron Near-Infrared Reflectance Image' + pat
                var_cirrus.standard_name  = 'Cirrus_Band_Reflectance'
                var_cirrus.units          = 'reflectance_normalized_by_solar_zenith_angle'
                var_cirrus.coordinates    = 'longitude latitude time'
            if not no_write_snowice and 'snowice_reflectance' not in keys0: 
                var_snowice   = f.createVariable('snowice_reflectance', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)    
                var_snowice.set_auto_maskandscale( False )
                var_snowice.long_name      = '1.6 micron Near-Infrared Reflectance Image' + pat
                var_snowice.standard_name  = 'Snow_Ice_Band_Reflectance'
                var_snowice.units          = 'reflectance_normalized_by_solar_zenith_angle'
                var_snowice.coordinates    = 'longitude latitude time'
            if not no_write_dirtyirdiff and 'dirtyir_brightness_temperature_diff' not in keys0: 
                var_ir3   = f.createVariable('dirtyir_brightness_temperature_diff', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)    
                var_ir3.set_auto_maskandscale( False )
                var_ir3.long_name      = '12.3 - 10.5 micron Dirty Infrared Brightness Temperature Difference Image' + pat
                var_ir3.standard_name  = 'Dirty_IR_brightness_temperature_difference'
                var_ir3.units          = 'kelvin'
                var_ir3.coordinates    = 'longitude latitude time'
            if not no_write_glm and 'glm_flash_extent_density' not in keys0:
                pat2 = ' and then smoothed using Gaussian filter'
                long_name = str('Flash extent density within +/- 2.5 min of time variable' + pat + pat2)
                var_glm  = f.createVariable('glm_flash_extent_density', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)
                var_glm.long_name     = long_name
                var_glm.standard_name = 'Flash_extent_density'
                var_glm.units         = 'Count per nominal 3136 microradian^2 pixel per 1.0 min'
                var_glm.coordinates   = 'longitude latitude time'
                if verbose:
                    print('Waiting for gridded GLM data file')
                mod_check = 0
                if len(glm_thread_info) > 0:
                    glm_thread_info[0].join()
                
                with Dataset(glm_file, "r", "NETCDF4") as glm_data:                                                                         #Read GLM data file
                    glm_raw_img, rect = fetch_glm_to_match_ir(ir, glm_data, _rect_color='r', _label='M1')      
            else:
               glm_raw_img = []    
            
            with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=RuntimeWarning)
              if len(xy_bounds) > 0:
 #                 zen     = zen[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]
                  bt      = bt[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]
                  vis_img = vis_img[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]
                  if len(glm_raw_img) > 0:
                      glm_raw_img = glm_raw_img[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]
                  if len(bt2) > 0:
                      bt2 = bt2[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]
                  if len(cirrus_img) > 0:
                      cirrus_img = cirrus_img[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]/coszen                          #Calculate reflectance by normalizing by solar zenith angle
                  if len(snowice_img) > 0:
                      snowice_img = snowice_img[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]/coszen                        #Calculate reflectance by normalizing by solar zenith angle
                  if len(bt3) > 0:
                      bt3 = bt3[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]
              else:
                  if len(cirrus_img) > 0:
#                      print(np.max(cirrus_img))
                      cirrus_img  = cirrus_img/coszen                                                                                       #Calculate reflectance by normalizing by solar zenith angle
                  if len(snowice_img) > 0:
#                      print(np.max(snowice_img))
                      snowice_img = snowice_img/coszen                                                                                      #Calculate reflectance by normalizing by solar zenith angle
        if not no_write_vis and 'visible_reflectance' not in keys1:
            with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=RuntimeWarning)
              vis_img = vis_img/coszen                                                                                                      #Calculate reflectance by normalizing by solar zenith angle
            vis_img[vis_img > 2] = 2                                                                                                        #Set maximum visible reflectance normalized by solar zenith angle to be 2
            vis_img[vis_img < 0] = 0                                                                                                        #Set minimum visible reflectance normalized by solar zenith angle to be 0
        if 'ir_brightness_temperature' not in keys0:    
            var_ir.long_name      = 'Infrared Brightness Temperature Image' + pat
            var_ir.standard_name  = 'IR_brightness_temperature'
            var_ir.units          = 'kelvin'
            var_ir.coordinates    = 'longitude latitude time'
        # variable assignment
        if len(keys0) <= 0:
            var_lon[:]  = np.asarray(lon_arr)
            var_lat[:]  = np.asarray(lat_arr)
            var_t[:]    = netCDF4.date2num(date, dd.variables['time'].attrs['units'])
        bt      = np.copy(np.asarray(bt))
        rm_inds = ((bt < 163.0) | (np.isnan(bt)) | (bt > 500.0))
        if not no_write_irdiff and 'ir_brightness_temperature_diff' not in keys1: 
            bt2 = np.copy(np.asarray(bt2))
            bt2[rm_inds]   = -500.0                                                                                                         #Remove likely untrue values by setting them to -500 K
#            bt2[bt < 163.0]   = -500.0                                                                                                     #Remove NaN values by setting them to -500 K
#            bt2[np.isnan(bt)] = -500.0                                                                                                     #Remove NaN values by setting them to -500 K
        if not no_write_dirtyirdiff and 'dirtyir_brightness_temperature_diff' not in keys1: 
            bt3 = np.copy(np.asarray(bt3))
            bt3[rm_inds]   = 500.0                                                                                                          #Remove likely untrue values by setting them to -500 K
#            bt2[np.isnan(bt)] = -500.0                                                                                                     #Remove NaN values by setting them to -500 K
        if 'ir_brightness_temperature' not in keys1:
            bt[rm_inds]   = np.nan                                                                                                            #Remove likely untrue values by setting them to 500 K
#            bt[rm_inds]   = 500.0                                                                                                            #Remove likely untrue values by setting them to 500 K
#            bt[np.isnan(bt)] = 500.0                                                                                                       #Remove NaN values by setting them to 500 K
#            t0 = time.time()
            data, scale_ir, offset_ir = Scaler.scaleData(bt)                                                                                #Extract data, scale factor and offsets that is scaled from Float to short
            var_ir[0, :, :]   = data
        if not no_write_vis and 'visible_reflectance' not in keys1:
            data, scale_vis, offset_vis = Scaler.scaleData(np.copy(np.asarray(vis_img)))                                                    #Extract data, scale factor and offsets that is scaled from Float to short
            var_vis[0, :, :]  = data
            var_vis.add_offset    = offset_vis
            var_vis.scale_factor  = scale_vis
            vis.close()
            if 'solar_zenith_angle' not in keys1:
                data, scale_zen, offset_zen = Scaler.scaleData(np.copy(np.asarray(np.rad2deg(zen))))                                        #Extract data, scale factor and offsets that is scaled from Float to short
                var_zen[0, :, :]  = data
                var_zen.add_offset    = offset_zen
                var_zen.scale_factor  = scale_zen
        elif not no_write_vis:
            vis.close()
        if not no_write_glm and 'glm_flash_extent_density' not in keys1:
            data, scale_glm, offset_glm = Scaler.scaleData(glm_raw_img)
            var_glm[0, :, :]     = data    
            var_glm.add_offset   = offset_glm
            var_glm.scale_factor = scale_glm
        if not no_write_irdiff and 'ir_brightness_temperature_diff' not in keys1: 
            data, scale_ir2, offset_ir2 = Scaler.scaleData(bt2)
            var_ir2[0, :, :]     = data    
            var_ir2.add_offset   = offset_ir2
            var_ir2.scale_factor = scale_ir2
        if not no_write_dirtyirdiff and 'dirtyir_brightness_temperature_diff' not in keys1: 
            data, scale_ir3, offset_ir3 = Scaler.scaleData(bt3)
            var_ir3[0, :, :]     = data    
            var_ir3.add_offset   = offset_ir3
            var_ir3.scale_factor = scale_ir3
        if not no_write_cirrus and 'cirrus_reflectance' not in keys1: 
            data, scale_cirrus, offset_cirrus = Scaler.scaleData(cirrus_img)
            var_cirrus[0, :, :]     = data    
            var_cirrus.add_offset   = offset_cirrus
            var_cirrus.scale_factor = scale_cirrus
        if not no_write_snowice and 'snowice_reflectance' not in keys1: 
            data, scale_snowice, offset_snowice = Scaler.scaleData(snowice_img)
            var_snowice[0, :, :]     = data    
            var_snowice.add_offset   = offset_snowice
            var_snowice.scale_factor = scale_snowice
#        print('Time to scale and offset the data = '+ str(time.time() -t0) + 'sec')
        if 'ir_brightness_temperature' not in keys1:
            var_ir.add_offset     = offset_ir
            var_ir.scale_factor   = scale_ir


        shape_arr = lon_arr.shape[0]
#        t0 = time.time()
        f.close()                                                                                                                           #Close output netCDF file
        ir     = None
        vis    = None
        f      = None
        zen    = None
        coszen = None
#        print('Files closed = ' + str(time.time() - t0) + 'sec')
#        if verbose == True: print('Combined model input written to netCDF file')
#        print('Time to run = ' + str(time.time() - t00) + 'sec')
    else:
        with Dataset(out_nc, 'r') as combined_data:
            lon       = np.asarray(combined_data.variables['longitude'])                                                                    #Read longitude values to get shape (used for file name saving of model results files)
            shape_arr = lon.shape[0]                                                                                                        #Extract longitude array shape
    return(out_nc, shape_arr)    
    

def mtg_get_ir_bt_from_rad(_ir_dataset, radiance):
    # Load variables
    wavenumber = _ir_dataset.variables['radiance_to_bt_conversion_coefficient_wavenumber'].values.astype(np.float32)
    a = _ir_dataset.variables['radiance_to_bt_conversion_coefficient_a'].values.astype(np.float32)
    b = _ir_dataset.variables['radiance_to_bt_conversion_coefficient_b'].values.astype(np.float32)
    c1 = _ir_dataset.variables['radiance_to_bt_conversion_constant_c1'].values.astype(np.float32)
    c2 = _ir_dataset.variables['radiance_to_bt_conversion_constant_c2'].values.astype(np.float32)
    
    
    # Calculate brightness temperature
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        nom = wavenumber*c2
        denom = a * np.log(1 + (c1 * wavenumber ** np.float32(3.))/radiance)
        bt = (nom/denom) - (b/a)

    return(bt)

def mtg_get_ref_from_rad(_vis_dataset, sun):
    # Extract variables from the dataset
    radiance = np.asarray(_vis_dataset.variables['effective_radiance']).astype(np.float32)  # mW·m-2·sr-1·(cm-1)-1
    solar_irradiance = np.asarray(_vis_dataset.variables['channel_effective_solar_irradiance']).astype(np.float32)  # mW·m-2·(cm-1)-1 at 1 AU
    earth_sun_distance = np.asarray(sun.variables['earth_sun_distance'])  # km
#     if self.is_iqt:
#         middle_time_diff = (self.observation_end_time - self.observation_start_time) / 2
#         utc_date = self.observation_start_time + middle_time_diff
#         sun_earth_distance = sun_earth_distance_correction(utc_date)
#         logger.info(f"The value sun_earth_distance is set to {sun_earth_distance} AU.")
#     else:
#         sun_earth_distance = np.nanmean(
#             self._get_aux_data_lut_vector("earth_sun_distance")) / 149597870.7  # [AU]
    sun_earth_distance = np.nanmean(earth_sun_distance)/ 149597870.7 # AU
    
    # Calculate the Bidirectional Reflectance Factor (BRF) for each pixel
    ref_img = radiance * np.float32(np.pi) * np.float32(sun_earth_distance) ** np.float32(2) / solar_irradiance

    return(ref_img)

def get_lat_lon_from_mtg(dataset, x, y, extract_proj_coordinates = False, verbose = True):
    lat_rad_1d  = np.asarray(y)
    lon_rad_1d  = np.asarray(x)
    
    x1          = (lon_rad_1d * dataset.variables['mtg_geos_projection'].attrs['perspective_point_height']).astype('float64')
    y1          = (lat_rad_1d * dataset.variables['mtg_geos_projection'].attrs['perspective_point_height']).astype('float64')
    proj_extent = (np.min(x1), np.max(x1), np.min(y1), np.max(y1))
    proj_img    = dataset.variables['mtg_geos_projection']

    # Define the image extent
    lon_origin = proj_img.attrs['longitude_of_projection_origin']
    H          = proj_img.attrs['perspective_point_height']+proj_img.attrs['semi_major_axis']
    r_eq       = proj_img.attrs['semi_major_axis']
    r_pol      = proj_img.attrs['semi_minor_axis']
    # create meshgrid filled with radian angles
#     lon_rad,lat_rad = np.meshgrid(lon_rad_1d,lat_rad_1d)
    lon_rad = lon_rad_1d
    lat_rad = lat_rad_1d 
#     a_var = np.power(np.sin(lon_rad),2.0) + 
#     (np.power(np.cos(lon_rad),2.0)*
#     (np.power(np.cos(lat_rad),2.0)+(((np.square(r_eq))/(np.square(r_pol)))*np.power(np.sin(lat_rad),2.0))))
    
    # Use vectorized operations and reuse intermediate variables
    sin_lon_rad = np.sin(lon_rad)
    cos_lon_rad = np.cos(lon_rad)
    cos_lat_rad = np.cos(lat_rad)
    sin_lat_rad = np.sin(lat_rad)

    # Calculating terms that are reused multiple times
    r_eq_sq = r_eq ** 2
    r_pol_sq = r_pol ** 2
    sin_lat_sq = sin_lat_rad ** 2
    cos_lat_sq = cos_lat_rad ** 2

    a_var = (sin_lon_rad ** 2 + cos_lon_rad ** 2 * (cos_lat_sq + (r_eq_sq / r_pol_sq) * sin_lat_sq))
    b_var = -2.0 * H * cos_lon_rad * cos_lat_rad
    c_var = np.square(H) - r_eq_sq

    
    r_s  = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
 #   r_s   = (-1.0*b_var - np.sqrt(np.square(b_var)-(4.0*a_var*c_var)))/(2.0*a_var)
 #   print(np.max(abs(r_s - r_s2)))

    s_x = r_s * cos_lon_rad * cos_lat_rad
    s_y = -r_s * sin_lon_rad
    s_z = r_s * cos_lon_rad * sin_lat_rad

    lat_arr = np.degrees(np.arctan((r_eq_sq * s_z) / (r_pol_sq * np.sqrt(np.square(H - s_x) + np.square(s_y)))))
    lon_arr = lon_origin - np.degrees(np.arctan(s_y / (H - s_x)))

#     if verbose == True:
#       print('Longitude bounds = ' + str(np.nanmin(lon_arr)) + ', ' + str(np.nanmax(lon_arr)))
#       print('Latitude bounds = ' + str(np.nanmin(lat_arr)) + ', ' + str(np.nanmax(lat_arr)))

    if extract_proj_coordinates:
        return(lon_arr, lat_arr, proj_img, proj_extent, x1, y1)    
    else:
        return(lon_arr, lat_arr, proj_img, proj_extent)


def get_lat_lon_from_mtg2(dataset, x, y, extract_proj_coordinates = False, verbose = True):
    lat_rad_1d  = np.asarray(y)
    lon_rad_1d  = np.asarray(x)
    proj_info   = dataset.variables['mtg_geos_projection']
    x1          = (lon_rad_1d * dataset.variables['mtg_geos_projection'].attrs['perspective_point_height']).astype('float64')
    y1          = (lat_rad_1d * dataset.variables['mtg_geos_projection'].attrs['perspective_point_height']).astype('float64')
    proj_extent = (np.min(x1), np.max(x1), np.min(y1), np.max(y1))
#     proj_img    = _vis_dataset.variables['goes_imager_projection']
#     lat_rad_1d  = lat_rad_1d*proj_info.perspective_point_height
#     lon_rad_1d  = lon_rad_1d*proj_info.perspective_point_height
    lat_rad_1d  = y1
    lon_rad_1d  = x1
    try:
#         print('first try...', end='')
        globe = ccrs.Globe(semimajor_axis=proj_info.attrs['semi_major_axis'],
                           semiminor_axis=proj_info.attrs['semi_minor_axis'],
                           inverse_flattening=proj_info.attrs['inverse_flattening'])
        geos = ccrs.Geostationary(central_longitude=proj_info.attrs['longitude_of_projection_origin'], 
                                  satellite_height=proj_info.attrs['perspective_point_height'],
                                  sweep_axis=proj_info.attrs['sweep_angle_axis'],
                                  globe=globe)
    
    except:
#         print('second try')
        globe = ccrs.Globe(semimajor_axis=proj_info.attrs['semi_major_axis'][0],
                       semiminor_axis=proj_info.attrs['semi_minor_axis'][0],
                       inverse_flattening=proj_info.attrs['inverse_flattening'][0])
        geos = ccrs.Geostationary(central_longitude=proj_info.attrs['longitude_of_projection_origin'][0], 
                                  satellite_height=proj_info.attrs['perspective_point_height'][0],
                                  sweep_axis=proj_info.attrs['sweep_angle_axis'],
                                  globe=globe)   
   
    a = ccrs.PlateCarree().transform_points(geos, lon_rad_1d, lat_rad_1d)
    lons, lats, _ = a[:,:,0], a[:,:,1], a[:,:,2]
    lats[np.isinf(lats)] = np.nanmin(lats)
    lons[np.isinf(lons)] = np.nanmin(lons)
    print('Longitude bounds = ' + str(np.nanmin(lons)) + ', ' + str(np.nanmax(lons)))
    print('Latitude bounds = ' + str(np.nanmin(lats)) + ', ' + str(np.nanmax(lats)))

#     if verbose == True:
#         print('Longitude bounds = ' + str(np.nanmin(lons)) + ', ' + str(np.nanmax(lons)))
#         print('Latitude bounds = ' + str(np.nanmin(lats)) + ', ' + str(np.nanmax(lats)))
    if extract_proj_coordinates:
        return(lons, lats, proj_info, proj_extent, x1, y1)    
    else:
        return(lons, lats, proj_info, proj_extent)

def upscale_img_to_fit(_small_img, _large_img):
    #   upscales a _small_img to the dimensions of _large_img
    
    return(cv2.resize(_small_img, (_large_img.shape[1], _large_img.shape[0]), interpolation=cv2.INTER_NEAREST))


















#     def calibrate_rad_to_refl(self, radiance, key):
#         """VIS channel calibration."""
#         measured = self.get_channel_measured_group_path(key["name"])
# 
#         cesi = self.get_and_cache_npxr(measured + "/channel_effective_solar_irradiance").astype(np.float32)
# 
#         if cesi == cesi.attrs.get(
#                 "FillValue", default_fillvals.get(cesi.dtype.str[1:])):
#             logger.error(
#                 "channel effective solar irradiance set to fill value, "
#                 "cannot produce reflectance for {:s}.".format(measured))
#             return radiance * np.float32(np.nan)
#         sun_earth_distance = self._compute_sun_earth_distance
#         res = 100 * radiance * np.float32(np.pi) * np.float32(sun_earth_distance) ** np.float32(2) / solar_irradiance
#         return res
# 
#     @cached_property
#     def _compute_sun_earth_distance(self) -> float:
#         """Compute the sun_earth_distance."""
#         if self.is_iqt:
#             middle_time_diff = (self.observation_end_time - self.observation_start_time) / 2
#             utc_date = self.observation_start_time + middle_time_diff
#             from pyorbital.astronomy import sun_earth_distance_correction
#             sun_earth_distance = sun_earth_distance_correction(utc_date)
#             logger.info(f"The value sun_earth_distance is set to {sun_earth_distance} AU.")
#         else:
#             sun_earth_distance = np.nanmean(
#                 self._get_aux_data_lut_vector("earth_sun_distance")) / 149597870.7  # [AU]
#         return sun_earth_distance        
