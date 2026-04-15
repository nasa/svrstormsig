#+
# Name:
#     combine_ir_glm_vis.py
# Purpose:
#     This is a script to create the combined ir_vis_glm netCDF file. This file is used
#     to create layered image of the 3 modalities which is input into labelme
# Calling sequence:
#     import combine_ir_glm_vis
# Input:
#     None.
# Functions:
#     extract_universal_vars      : Extracts smoothed/resized GLM and resized IR data on VIS data grid
#     combine_ir_glm_vis_parallel : Creates and write netCDF file that combines IR, VIS, and GLM data
# Output:
#     Combined ir_vis_glm netCDF file. Also returns name and file path of combined netCDF file.
# Keywords:
#     vis_file             : String keyword specifying GOES VIS data filename and path
#                            DEFAULT = '../../../goes-data/vis_dir/20200513-14/OR_ABI-L1b-RadM2-M6C02_G16_s20201340000494_e20201340000552_c20201340000582.nc'
#     ir_file              : String keyword specifying GOES IR data filename and path
#                            DEFAULT = '../../../goes-data/ir_dir/20200513-14/OR_ABI-L1b-RadM2-M6C13_G16_s20201340000494_e20201340000564_c20201340001007.nc'
#     glm_file             : String keyword specifying filename and path to GLM data file
#                            DEFAULT = '../../../goes-data/out_dir/gridded_data.nc'
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
#     proj_cache           : Dictionary to cache projection file data so it is only read from disk once per unique      
#                            projection file + bounds + sector position combination per process. If None, uses the      
#                            module-level per-process cache _PROJECTION_CACHE. Do NOT pass a                            
#                            multiprocessing.Manager().dict() here as the pipe serialization overhead for large         
#                            arrays makes it slower than just reading the file. Use the default None and let each       
#                            worker process maintain its own cache.                                                     
#                            DEFAULT = None                                                                             
#     verbose              : BOOL keyword to specify whether or not to print verbose informational messages.
#                            DEFAULT = True which implies to print verbose informational messages
# Author and history:
#     Anna Cuddeback
#     John W. Cooney           2020-09-02. (Updated to include documentation)
#                              2020-09-08. (Fixed longitude and latitude arrays. Previously,
#                                           longitude array was written to latitude and visa versa)
#                              2020-09-24. (Added rewrite_nc keyword and am now writing solar zenith
#                                           angle to file as well as normalizing VIS reflectance by
#                                           solar zenith angle)
#                              2026-03-27. (Added projection file caching via proj_cache keyword to avoid     
#                                           re-reading the same static projection file for every GOES scan.   
#                                           Cache key includes mesoscale sector center lat/lon so that        
#                                           sector repositions are handled correctly. Cache stores            
#                                           pre-subsetted lat/lon arrays to avoid repeated expensive          
#                                           array slicing on every retrieval. Uses module-level               
#                                           per-process cache rather than Manager dict to avoid pipe          
#                                           serialization overhead. Also avoided re-reading lon/lat           
#                                           arrays from netCDF on append by pulling from the projection       
#                                           cache instead. Fixed UnboundLocalError by capturing               
#                                           is_append flag before f.close() writes the file.)                 
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
#from pyorbital import astronomy
from glm_gridder.glm_data_processing import *
from glm_gridder.dataScaler import DataScaler
from suncalc import get_position

_PROJECTION_CACHE = {}

def _get_projection_data(proj_file, xy_bounds, sat, no_write_vis, scan_mode, proj_cache, verbose,
                         ctr_lat=None, ctr_lon=None):
    """
    Load projection file data and cache the result so the file is only read
    from disk once per unique combination per process.

    The cache stores:
      - x_inds, y_inds   : indices used to subset data arrays every file
      - proj_extent      : written to netCDF bounds attribute
      - og_shape         : used for shape validation
      - lat_arr, lon_arr : pre-subsetted when xy_bounds non-empty so that
                           the expensive lat/lon computation is only done
                           once. These are the only arrays that are truly
                           static across files at the same sector position.

    Data arrays (bt, vis_img etc.) are NOT cached - they change every file
    and must always be read fresh from the L1b files. The x_inds/y_inds
    returned from the cache are used by the caller to subset those arrays
    exactly as before.

    Args:
        proj_file    : Full path to the satellite projection netCDF file.
        xy_bounds    : List/array of [x0, y0, x1, y1] domain bounds, or []
                       for the full domain.
        sat          : Satellite name string (e.g. 'goes16').
        no_write_vis : Boolean - if True, use native IR grid instead of VIS.
        scan_mode    : String scan mode ('conus', 'full', or 'meso').
        proj_cache   : Dict-like object to use as the cache. Should be the
                       module-level _PROJECTION_CACHE for per-process caching.
                       Do NOT use a multiprocessing.Manager().dict() as the
                       pipe serialization overhead for large numpy arrays is
                       far more expensive than each worker reading the file once.
        verbose      : Boolean - print informational messages if True.
        ctr_lat      : Float central latitude of mesoscale sector. Must be
                       supplied when scan_mode == 'meso'. DEFAULT = None.
        ctr_lon      : Float central longitude of mesoscale sector. Must be
                       supplied when scan_mode == 'meso'. DEFAULT = None.

    Returns:
        Tuple (x_inds, y_inds, proj_extent, og_shape, lat_arr, lon_arr)
        where lat_arr and lon_arr are pre-subsetted when xy_bounds is non-empty.
    """
    if scan_mode == 'meso':
        if ctr_lat is None or ctr_lon is None:
            raise ValueError(
                '_get_projection_data: ctr_lat and ctr_lon must be provided '
                'for mesoscale sectors to ensure the cache key is unique per '
                'sector position. Got ctr_lat={}, ctr_lon={}'.format(ctr_lat, ctr_lon)
            )
        ctr_lat_key = round(float(ctr_lat), 4)
        ctr_lon_key = round(float(ctr_lon), 4)
    else:
        ctr_lat_key = None
        ctr_lon_key = None

    cache_key = (proj_file, tuple(xy_bounds), sat, no_write_vis, scan_mode,
                 ctr_lat_key, ctr_lon_key)

    if cache_key in proj_cache:
        return(proj_cache[cache_key])

    # Not cached yet - read from disk
    if verbose:
        print('Reading projection file from disk and caching result: ' + proj_file)
    with Dataset(proj_file, 'r') as scan_proj_dataset:
        if len(xy_bounds) > 0:
            result = get_lat_lon_subset_inds(
                scan_proj_dataset, xy_bounds,
                return_lats   = True,
                satellite     = sat,
                use_native_ir = no_write_vis,
                scan_mode     = scan_mode,
                verbose       = verbose
            )
        else:
            result = get_lat_lon_subset_inds(
                scan_proj_dataset, [],
                return_lats = True,
                verbose     = verbose
            )

    # When xy_bounds are specified, store the pre-subsetted lat/lon arrays in the cache.
    if len(xy_bounds) > 0:
        x_inds, y_inds, proj_extent, og_shape, lat_arr, lon_arr = result
        x0, x1  = np.min(x_inds), np.max(x_inds)
        y0, y1  = np.min(y_inds), np.max(y_inds)
        lat_sub = np.array(lat_arr[x0:x1, y0:y1])                                                                                          #Pre-subsetted lat array - static across files at same sector position
        lon_sub = np.array(lon_arr[x0:x1, y0:y1])                                                                                          #Pre-subsetted lon array - static across files at same sector position
        result  = (x_inds, y_inds, proj_extent, og_shape, lat_sub, lon_sub)                                                                #Store pre-subsetted arrays - x_inds/y_inds kept for per-file data array subsetting

    proj_cache[cache_key] = result
    return(result)


def combine_ir_glm_vis_parallel(vis_file             = os.path.join('..', '..', '..', 'goes-data', 'vis', '20200513-14', 'OR_ABI-L1b-RadM2-M6C02_G16_s20201340000494_e20201340000552_c20201340000582.nc'),
                                ir_file              = os.path.join('..', '..', '..', 'goes-data', 'ir', 'goes-data', '20200513-14', 'OR_ABI-L1b-RadM2-M6C13_G16_s20201340000494_e20201340000564_c20201340001007.nc'),
                                glm_file             = os.path.join('..', '..', '..', 'goes-data', 'out_dir', 'gridded_data.nc'),
                                layered_dir          = os.path.join('..', '..', '..', 'goes-data', 'combined_nc_dir'), 
                                universal_file       = False, 
                                rewrite_nc           = False, 
                                append_nc            = True,
                                no_write_vis         = False,                       
                                no_write_glm         = True,
                                no_write_irdiff      = True,
                                no_write_cirrus      = True, 
                                no_write_snowice     = True, 
                                no_write_dirtyirdiff = True, 
                                in_bucket_name       = 'goes-data', 
                                proj                 = None, 
                                xy_bounds            = [], 
                                glm_thread_info      = [],
                                proj_cache           = None,
                                verbose              = True):

    vis_file    = os.path.realpath(vis_file)
    ir_file     = os.path.realpath(ir_file)
    glm_file    = os.path.realpath(glm_file)
    layered_dir = os.path.realpath(layered_dir)

    if proj_cache is None:
        proj_cache = _PROJECTION_CACHE

    file_attr = re.split('_|-', os.path.basename(ir_file))
    satellite = file_attr[5]
    sat       = satellite[0].lower() + 'oes' + satellite[1:]
    out_nc    = os.path.join(layered_dir, file_attr[0] + '_' + file_attr[1] + '_' + file_attr[2] + '_' + str(file_attr[3].split('Rad')[1]) + '_COMBINED_' + '_'.join(file_attr[6:]))
    if (not os.path.isfile(out_nc) or rewrite_nc or append_nc):
        t0 = time.time()
        os.makedirs(os.path.dirname(out_nc), exist_ok = True)

        ir = Dataset(ir_file, 'r')

        # Extract mesoscale sector center coordinates
        if 'RadM' in os.path.basename(ir_file):
            _ext           = ir.variables['geospatial_lat_lon_extent']
            sector_ctr_lat = float(np.copy(np.asarray(_ext.geospatial_lat_center)))
            sector_ctr_lon = float(np.copy(np.asarray(_ext.geospatial_lon_center)))
            if verbose:
                print('Mesoscale sector center: lat={:.4f}, lon={:.4f}'.format(
                      sector_ctr_lat, sector_ctr_lon))
        else:
            sector_ctr_lat = None
            sector_ctr_lon = None

        ir_raw_img = np.copy(np.asarray(ir.variables['Rad'][:,:]))
        bt         = get_ir_bt_from_rad(ir)

        if not no_write_vis:
            vis      = Dataset(vis_file, 'r')
            vis_img  = get_ref_from_rad(vis)
            pat_proj = '_vis_'
            xyres    = vis.spatial_resolution
        else:
            vis_img  = bt*0.0
            pat_proj = '_ir_'
            xyres    = ir.spatial_resolution

        if not no_write_irdiff:
            ir_dstr = re.split('_s|_', os.path.basename(ir_file))[3]
            ird_dir = os.path.join(os.path.dirname(os.path.dirname(ir_file)), 'ir_diff')
            ird_f   = sorted(glob.glob(os.path.join(ird_dir, '*s' + ir_dstr + '_*.nc')))
            if len(ird_f) != 1:
                print('Too many or not enough matching 6.2 micron IR data files.')
                print(ir_dstr)
                print(ird_f)
                exit()
            with Dataset(ird_f[0], 'r') as ir2:
                bt2 = get_ir_bt_from_rad(ir2)
                bt2 = np.copy(bt2 - bt)
        else:
            ir_dstr = None
            ird_f   = ['dfadsfasgfsag']
            bt2     = []
            
        if not no_write_dirtyirdiff:
            if ir_dstr == None:
                ir_dstr = re.split('_s|_', os.path.basename(ir_file))[3]
            dirtyir_dir = os.path.join(os.path.dirname(os.path.dirname(ir_file)), 'dirtyir')
            dirtyir_f   = sorted(glob.glob(os.path.join(dirtyir_dir, '*s' + ir_dstr + '_*.nc')))
            if len(dirtyir_f) != 1:
                print('Too many or not enough matching 12.3 micron IR data files.')
                print(ir_dstr)
                print(dirtyir_f)
                exit()
            with Dataset(dirtyir_f[0], 'r') as ir3:
                bt3 = get_ir_bt_from_rad(ir3)
                bt3 = np.copy(bt3 - bt)
        else:
            dirtyir_f = ['dfadsfasgfsag']
            bt3       = []

        if not no_write_cirrus:
            if ir_dstr == None:
                ir_dstr = re.split('_s|_', os.path.basename(ir_file))[3]
            cirrus_dir = os.path.join(os.path.dirname(os.path.dirname(ir_file)), 'cirrus')
            cirrus_f   = sorted(glob.glob(os.path.join(cirrus_dir, '*s' + ir_dstr + '_*.nc')))
            with Dataset(cirrus_f[0], 'r') as cirrus:
                cirrus_img = get_ref_from_rad(cirrus)
        else:
            cirrus_img = []

        if not no_write_snowice:   
            if ir_dstr == None:
                ir_dstr = re.split('_s|_', os.path.basename(ir_file))[3]
            snowice_dir = os.path.join(os.path.dirname(os.path.dirname(ir_file)), 'snowice')
            snowice_f   = sorted(glob.glob(os.path.join(snowice_dir, '*s' + ir_dstr + '_*.nc')))
            with Dataset(snowice_f[0], 'r') as snowice:
                snowice_img = get_ref_from_rad(snowice)
        else:
            snowice_img = []
      
        if verbose: print('Writing combined VIS/IR/GLM output netCDF ' + out_nc)
        Scaler = DataScaler( nbytes = 4, signed = True )

        is_append = os.path.isfile(out_nc) and append_nc and not rewrite_nc

        if is_append:
            f       = netCDF4.Dataset(out_nc,'a', format='NETCDF4')
            x_inds  = list(f.x_inds)
            y_inds  = list(f.y_inds)
            keys0   = f.variables.keys()
            keys1   = list(keys0)

            if ('solar_zenith_angle' not in keys0) and (not no_write_vis or not no_write_cirrus or not no_write_snowice):
                date = netCDF4.num2date(ir.variables['t'][:], ir.variables['t'].units)
                date = datetime.datetime(date.year, date.month, date.day, hour=date.hour, minute=date.minute, second=date.second, tzinfo=datetime.timezone.utc)

                # Attempt to get pre-subsetted lat/lon from the projection cache.
                got_latlon_from_cache = False
                if 'RadC' in os.path.basename(ir_file) or 'RadF' in os.path.basename(ir_file) or 'RadM' in os.path.basename(ir_file):
                    proj_dir        = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'sat_projection_files')
                    sat_cache       = sat
                    if sat_cache == 'goes18':
                        sat_cache = 'goes17'
                    if sat_cache == 'goes19':
                        sat_cache = 'goes16'
                    if 'RadC' in os.path.basename(ir_file):
                        scan_mode_cache = 'conus'
                    elif 'RadF' in os.path.basename(ir_file):
                        scan_mode_cache = 'full'
                    else:
                        scan_mode_cache = 'meso'
                    proj_file_cache = os.path.join(proj_dir, sat_cache + '_satellite' + pat_proj + scan_mode_cache + '_scan_projections.nc')

                    if scan_mode_cache == 'meso':
                        ctr_lat_key = round(sector_ctr_lat, 4) if sector_ctr_lat is not None else None
                        ctr_lon_key = round(sector_ctr_lon, 4) if sector_ctr_lon is not None else None
                    else:
                        ctr_lat_key = None
                        ctr_lon_key = None
                    cache_key = (proj_file_cache, tuple(xy_bounds), sat_cache, no_write_vis,
                                 scan_mode_cache, ctr_lat_key, ctr_lon_key)

                    if cache_key in proj_cache:
                        # Pull pre-subsetted lat/lon from cache
                        _, _, _, _, lat_arr, lon_arr = proj_cache[cache_key]                                                                #lat/lon are pre-subsetted, use directly for solar zenith angle calculation
                        got_latlon_from_cache = True
                        if verbose:
                            print('Using cached lat/lon arrays for solar zenith angle calculation (append mode)')

                if not got_latlon_from_cache:
                    # Fallback if cache miss
                    lon_arr = np.copy(np.asarray(f.variables['longitude']))
                    lat_arr = np.copy(np.asarray(f.variables['latitude']))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    zen    = np.radians(90.0)-get_position(date, lon_arr, lat_arr)['altitude']
                    coszen = np.cos(zen)

            elif ('solar_zenith_angle' in keys0) and (not no_write_vis or not no_write_cirrus or not no_write_snowice):
                zen    = np.copy(np.asarray(f.variables['solar_zenith_angle']))[0, :, :]
                coszen = np.cos(np.deg2rad(zen))

            lon_shape_0 = len(f.dimensions['Y'])

        else:
            keys0 = {}
            keys1 = list(keys0)
            f     = netCDF4.Dataset(out_nc,'w', format='NETCDF4')
#             print('Finished first part = ' + str(time.time() - t0) + 'sec')
            
            x_inds = []
            y_inds = []
#             t0     = time.time()
            if 'RadC' in os.path.basename(ir_file) or 'RadF' in os.path.basename(ir_file):
                if not no_write_vis:
                    proj_img = vis.variables['goes_imager_projection']
                else:
                    proj_img = ir.variables['goes_imager_projection']
                
                proj_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'sat_projection_files')
                if sat == 'goes18':
                    sat = 'goes17'
                if sat == 'goes19':
                    sat = 'goes16'
                if 'RadC' in os.path.basename(ir_file):
                    proj_file = os.path.join(proj_dir, sat + '_satellite' + pat_proj + 'conus_scan_projections.nc')
                    scan_mode = 'conus'
                else:
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
                    x_inds, y_inds, proj_extent_new, og_shape, lat_arr, lon_arr = _get_projection_data(
                        proj_file, xy_bounds, sat, no_write_vis, scan_mode, proj_cache, verbose,
                        ctr_lat = sector_ctr_lat,
                        ctr_lon = sector_ctr_lon)
#                     print('Time to return subset inds = ' + str(time.time() - t0) + 'sec')
                    # Note: lat_arr/lon_arr are now pre-subsetted.
                    # og_shape check skipped since lat/lon are pre-subsetted.
                else:
                    x_inds, y_inds, proj_extent, og_shape, lat_arr, lon_arr = _get_projection_data(
                        proj_file, [], sat, no_write_vis, scan_mode, proj_cache, verbose,
                        ctr_lat = sector_ctr_lat,
                        ctr_lon = sector_ctr_lon)
            else:
                if not no_write_vis:
                    lon_arr, lat_arr, proj_img, proj_extent = get_lat_lon_from_vis(vis, verbose = verbose)
                else:
                    lon_arr, lat_arr, proj_img, proj_extent = get_lat_lon_from_vis(ir, verbose = verbose)
#             print('Lon/lat extraction = ' + str(time.time() - t0) + 'sec')
            
            date = netCDF4.num2date( ir.variables['t'][:], ir.variables['t'].units)
            date = datetime.datetime(date.year, date.month, date.day, hour=date.hour, minute=date.minute, second=date.second, tzinfo=datetime.timezone.utc)

            # lon_arr/lat_arr are pre-subsetted by _get_projection_data for RadC/RadF so do NOT subset them again here.
            # For RadM, get_lat_lon_from_vis returns the full array so subsetting is still needed if xy_bounds is set.
            if len(xy_bounds) > 0 and ('RadM' in os.path.basename(ir_file)):                                                               #Only subset lon/lat for RadM - RadC/RadF lat/lon already pre-subsetted
                lon_arr = lon_arr[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]
                lat_arr = lat_arr[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]

#             t0 = time.time()
            with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=RuntimeWarning)
              zen    = np.radians(90.0)-get_position(date, lon_arr, lat_arr)['altitude']
              coszen = np.cos(zen)
#             print('sun_zenith_angle calc = ' + str(time.time() - t0) + 'sec')

#             t0 = time.time()
            f.createDimension('Y',     lon_arr.shape[0])
            f.createDimension('X',     lat_arr.shape[1])
            if not universal_file:
                f.createDimension('IR_Y',   ir_raw_img.shape[0])
                f.createDimension('IR_X',   ir_raw_img.shape[1])
                if not no_write_glm:
                    if verbose:
                        print('Waiting for gridded GLM data file')
                    while not os.path.isfile(glm_file):
                        time.sleep(0.50)
                    with Dataset(glm_file, "r", "NETCDF4") as glm_data:
                        glm_raw_img, rect = fetch_glm_to_match_ir(ir, glm_data, _rect_color='r', _label='M1')      
                    f.createDimension('GLM_Y', glm_raw_img.shape[0])
                    f.createDimension('GLM_X', glm_raw_img.shape[1])
            f.createDimension('time', 1)
            
            var_lon               = f.createVariable('longitude', 'f4', ('Y','X',), zlib = True, least_significant_digit = 3)
            var_lon.long_name     = 'longitude -180 to 180 degrees east'
            var_lon.standard_name = 'longitude'
            var_lon.units         = 'degrees_east'
            var_lat               = f.createVariable('latitude', 'f4', ('Y', 'X',), zlib = True, least_significant_digit = 3)
            var_lat.long_name     = 'latitude -90 to 90 degrees north'
            var_lat.standard_name = 'latitude'
            var_lat.units         = 'degrees_north'
            var_t                 = f.createVariable('time', np.float64, ('time',))
            var_t.long_name       = ir.variables['t'].long_name
            var_t.standard_name   = ir.variables['t'].standard_name
            var_t.units           = ir.variables['t'].units

        if not no_write_vis:
            if 'visible_reflectance' not in keys0:
                var_vis                   = f.createVariable('visible_reflectance', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue)
                var_vis.set_auto_maskandscale( False )
                var_vis.long_name         = 'Visible Reflectance Normalized by Solar Zenith Angle'
                var_vis.standard_name     = 'Visible_Reflectance'
                var_vis.units             = 'reflectance_normalized_by_solar_zenith_angle'
                var_vis.coordinates       = 'longitude latitude time'
            if 'solar_zenith_angle' not in keys0:
                var_zen                   = f.createVariable('solar_zenith_angle', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 2)
                var_zen.set_auto_maskandscale( False )
                var_zen.long_name         = 'Solar Zenith Angle'
                var_zen.standard_name     = 'solar_zenith_angle'
                var_zen.units             = 'degrees'
        if len(keys0) <= 0:
            image_projection = f.createVariable('imager_projection', np.int32, zlib = True)        
            image_projection.long_name                      = proj_img.long_name
            image_projection.satellite_name                 = satellite
            image_projection.grid_mapping_name              = proj_img.grid_mapping_name
            image_projection.perspective_point_height       = proj_img.perspective_point_height
            image_projection.semi_major_axis                = proj_img.semi_major_axis
            image_projection.semi_minor_axis                = proj_img.semi_minor_axis
            image_projection.inverse_flattening             = proj_img.inverse_flattening
            image_projection.latitude_of_projection_origin  = proj_img.latitude_of_projection_origin
            image_projection.longitude_of_projection_origin = proj_img.longitude_of_projection_origin
            image_projection.sweep_angle_axis               = proj_img.sweep_angle_axis
            proj4_string = (
                f'+proj=geos +lon_0={proj_img.longitude_of_projection_origin} +h={proj_img.perspective_point_height} '
                f'+sweep={proj_img.sweep_angle_axis} +a={proj_img.semi_major_axis} +b={proj_img.semi_minor_axis} +rf={proj_img.inverse_flattening} '
                '+ellps=GRS80'
            )
            f.Conventions        = 'CF-1.8'
            if os.path.isfile(glm_file) and not no_write_glm: 
                if not universal_file:
                    f.description = "This file combines unscaled IR data, Visible data, and GLM data into one NetCDF file. The VIS data is normalized by the solar zenith angle. The GLM data is resampled onto the IR grid using glmtools for files ±2.5 min from VIS/IR analysis time. x_inds show 0th dimension min and max indices of subsetting region. y_inds show 1st dimension."
                else:
                    f.description = "This file combines unscaled IR data, Visible data, and GLM data into one NetCDF file. The VIS data is on its original grid but is normalized by the solar zenith angle. The GLM data is resampled onto the IR grid using glmtools for files ±2.5 min from VIS/IR analysis time. The IR and GLM data are then upscaled onto the VIS grid using cv2.resize with interpolation set as cv2.INTER_NEAREST. Lastly, the GLM data is smoothed by ndimage.gaussian_filter(glm_data, sigma=1.0, order=0). x_inds show 0th dimension min and max indices of subsetting region. y_inds show 1st dimension."
            else:
                f.description = "This file combines unscaled IR data and Visible data but no GLM data into one NetCDF file. The VIS data is normalized by the solar zenith angle. x_inds show 0th dimension min and max indices of subsetting region. y_inds show 1st dimension."
            f.geospatial_lat_min  = str(np.nanmin(lat_arr))
            f.geospatial_lat_max  = str(np.nanmax(lat_arr))
            f.geospatial_lon_min  = str(np.nanmin(lon_arr))
            f.geospatial_lon_max  = str(np.nanmax(lon_arr))
            f.x_inds              = x_inds
            f.y_inds              = y_inds
            f.spatial_resolution  = xyres
            f.proj4_string        = proj4_string
            if len(xy_bounds) > 0 and ('RadC' in os.path.basename(ir_file) or 'RadF' in os.path.basename(ir_file)):
                image_projection.bounds = str(proj_extent_new[0]) + ',' + str(proj_extent_new[1]) + ',' + str(proj_extent_new[2]) + ',' + str(proj_extent_new[3])
            else:
                image_projection.bounds = str(proj_extent[0]) + ',' + str(proj_extent[1]) + ',' + str(proj_extent[2]) + ',' + str(proj_extent[3])
            image_projection.bounds_units = 'm'
       
        if not universal_file:
            var_ir = f.createVariable('ir_brightness_temperature', np.int32, ('time', 'IR_Y', 'IR_X',), zlib = True, fill_value = Scaler._FillValue)
            var_ir.set_auto_maskandscale( False )
            pat  = ''
            pat2 = ''
            if os.path.isfile(glm_file) and not no_write_glm:
                var_glm               = f.createVariable('glm_flash_extent_density', np.int32, ('time', 'GLM_Y', 'GLM_X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)    
                var_glm.set_auto_maskandscale( False )
                long_name             = str('Flash extent density within +/- 2.5 min of time variable' + pat + pat2)
                var_glm.long_name     = long_name
                var_glm.standard_name = 'Flash_extent_density'
                var_glm.units         = 'Count per nominal 3136 microradian^2 pixel per 1.0 min'
                var_glm.coordinates   = 'longitude latitude time'
            if os.path.isfile(ird_f[0]) and not no_write_irdiff: 
                var_ir2               = f.createVariable('ir_brightness_temperature_diff', np.int32, ('time', 'IR_Y', 'IR_X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)    
                var_ir2.set_auto_maskandscale( False )
                var_ir2.long_name     = '6.2 - 10.3 micron Infrared Brightness Temperature Image' + pat
                var_ir2.standard_name = 'IR_brightness_temperature_difference'
                var_ir2.units         = 'kelvin'
                var_ir2.coordinates   = 'longitude latitude time'
        else:
            if not no_write_vis:
                pat  = ' resampled onto VIS data grid'
            else:
                pat  = ' resampled onto IR data grid'
            pat2 = ''
            if 'ir_brightness_temperature' not in keys0:
                var_ir = f.createVariable('ir_brightness_temperature', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue)
                var_ir.set_auto_maskandscale( False )
            if not no_write_irdiff and 'ir_brightness_temperature_diff' not in keys0: 
                var_ir2               = f.createVariable('ir_brightness_temperature_diff', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)    
                var_ir2.set_auto_maskandscale( False )
                var_ir2.long_name     = '6.2 - 10.3 micron Infrared Brightness Temperature Image' + pat
                var_ir2.standard_name = 'IR_brightness_temperature_difference'
                var_ir2.units         = 'kelvin'
                var_ir2.coordinates   = 'longitude latitude time'
            if not no_write_cirrus and 'cirrus_reflectance' not in keys0: 
                var_cirrus               = f.createVariable('cirrus_reflectance', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)    
                var_cirrus.set_auto_maskandscale( False )
                var_cirrus.long_name     = '1.37 micron Near-Infrared Reflectance Image' + pat
                var_cirrus.standard_name = 'Cirrus_Band_Reflectance'
                var_cirrus.units         = 'reflectance_normalized_by_solar_zenith_angle'
                var_cirrus.coordinates   = 'longitude latitude time'
            if not no_write_snowice and 'snowice_reflectance' not in keys0: 
                var_snowice               = f.createVariable('snowice_reflectance', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)    
                var_snowice.set_auto_maskandscale( False )
                var_snowice.long_name     = '1.6 micron Near-Infrared Reflectance Image' + pat
                var_snowice.standard_name = 'Snow_Ice_Band_Reflectance'
                var_snowice.units         = 'reflectance_normalized_by_solar_zenith_angle'
                var_snowice.coordinates   = 'longitude latitude time'
            if not no_write_dirtyirdiff and 'dirtyir_brightness_temperature_diff' not in keys0: 
                var_ir3               = f.createVariable('dirtyir_brightness_temperature_diff', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)    
                var_ir3.set_auto_maskandscale( False )
                var_ir3.long_name     = '12.3 - 10.3 micron Dirty Infrared Brightness Temperature Difference Image' + pat
                var_ir3.standard_name = 'Dirty_IR_brightness_temperature_difference'
                var_ir3.units         = 'kelvin'
                var_ir3.coordinates   = 'longitude latitude time'
            if not no_write_glm and 'glm_flash_extent_density' not in keys0:
                pat2                  = ' and then smoothed using Gaussian filter'
                long_name             = str('Flash extent density within +/- 2.5 min of time variable' + pat + pat2)
                var_glm               = f.createVariable('glm_flash_extent_density', np.int32, ('time', 'Y', 'X',), zlib = True, fill_value = Scaler._FillValue, least_significant_digit = 4)
                var_glm.long_name     = long_name
                var_glm.standard_name = 'Flash_extent_density'
                var_glm.units         = 'Count per nominal 3136 microradian^2 pixel per 1.0 min'
                var_glm.coordinates   = 'longitude latitude time'
                if verbose:
                    print('Waiting for gridded GLM data file')
                mod_check = 0
                if len(glm_thread_info) > 0:
                    glm_thread_info[0].join()
                with Dataset(glm_file, "r", "NETCDF4") as glm_data:
                    glm_raw_img, rect = fetch_glm_to_match_ir(ir, glm_data, _rect_color='r', _label='M1')      
            else:
               glm_raw_img = []    
            
#             t00 = time.time()
            bt, glm_raw_img, bt2, cirrus_img, snowice_img, bt3 = extract_universal_vars(vis_img, bt, glm_raw_img, bt2, cirrus_img, snowice_img, bt3, verbose = verbose)
#             print('Time to run extract_universal_vars = ' + str(time.time() - t00) + 'sec')

            with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=RuntimeWarning)

              # Data arrays always come from L1b files at full resolution and must be subsetted when xy_bounds is set regardless of sector.
              if len(xy_bounds) > 0 and ('RadC' in os.path.basename(ir_file) or 'RadF' in os.path.basename(ir_file) or 'RadM' in os.path.basename(ir_file)):
                  bt      = bt[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]
                  vis_img = vis_img[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]
                  if len(glm_raw_img) > 0:
                      glm_raw_img = glm_raw_img[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]
                  if len(bt2) > 0:
                      bt2 = bt2[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]
                  if len(cirrus_img) > 0:
                      cirrus_img  = cirrus_img[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]/coszen                          #Calculate reflectance by normalizing by solar zenith angle
                  if len(snowice_img) > 0:
                      snowice_img = snowice_img[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]/coszen                         #Calculate reflectance by normalizing by solar zenith angle
                  if len(bt3) > 0:
                      bt3 = bt3[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]
              else:
                  if len(cirrus_img) > 0:
                      cirrus_img  = cirrus_img/coszen                                                                                        #Calculate reflectance by normalizing by solar zenith angle
                  if len(snowice_img) > 0:
                      snowice_img = snowice_img/coszen                                                                                       #Calculate reflectance by normalizing by solar zenith angle

        if not no_write_vis and 'visible_reflectance' not in keys1:
            with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=RuntimeWarning)
              vis_img = vis_img/coszen
            vis_img[vis_img > 2] = 2
            vis_img[vis_img < 0] = 0
        if 'ir_brightness_temperature' not in keys0:    
            var_ir.long_name      = 'Infrared Brightness Temperature Image' + pat
            var_ir.standard_name  = 'IR_brightness_temperature'
            var_ir.units          = 'kelvin'
            var_ir.coordinates    = 'longitude latitude time'
        if len(keys0) <= 0:
            var_lon[:] = np.copy(np.asarray(lon_arr))
            var_lat[:] = np.copy(np.asarray(lat_arr))
            var_t[:]   = np.copy(np.asarray(ir.variables['t'][:]))
        bt      = np.copy(np.asarray(bt))
        rm_inds = ((bt < 163.0) | (np.isnan(bt)) | (bt > 500.0))
        if not no_write_irdiff and 'ir_brightness_temperature_diff' not in keys1: 
            bt2          = np.copy(np.asarray(bt2))
            bt2[rm_inds] = -500.0
        if not no_write_dirtyirdiff and 'dirtyir_brightness_temperature_diff' not in keys1: 
            bt3          = np.copy(np.asarray(bt3))
            bt3[rm_inds] = 500.0
        if 'ir_brightness_temperature' not in keys1:
            bt[rm_inds]              = np.nan
            data, scale_ir, offset_ir = Scaler.scaleData(bt)
            var_ir[0, :, :]          = data
        if not no_write_vis and 'visible_reflectance' not in keys1:
            data, scale_vis, offset_vis = Scaler.scaleData(np.copy(np.asarray(vis_img)))
            var_vis[0, :, :]            = data
            var_vis.add_offset          = offset_vis
            var_vis.scale_factor        = scale_vis
            vis.close()
            if 'solar_zenith_angle' not in keys1:
                data, scale_zen, offset_zen = Scaler.scaleData(np.copy(np.asarray(np.rad2deg(zen))))
                var_zen[0, :, :]            = data
                var_zen.add_offset          = offset_zen
                var_zen.scale_factor        = scale_zen
        elif not no_write_vis:
            vis.close()
        if not no_write_glm and 'glm_flash_extent_density' not in keys1:
            data, scale_glm, offset_glm = Scaler.scaleData(glm_raw_img)
            var_glm[0, :, :]            = data    
            var_glm.add_offset          = offset_glm
            var_glm.scale_factor        = scale_glm
        if not no_write_irdiff and 'ir_brightness_temperature_diff' not in keys1: 
            data, scale_ir2, offset_ir2 = Scaler.scaleData(bt2)
            var_ir2[0, :, :]            = data    
            var_ir2.add_offset          = offset_ir2
            var_ir2.scale_factor        = scale_ir2
        if not no_write_dirtyirdiff and 'dirtyir_brightness_temperature_diff' not in keys1: 
            data, scale_ir3, offset_ir3 = Scaler.scaleData(bt3)
            var_ir3[0, :, :]            = data    
            var_ir3.add_offset          = offset_ir3
            var_ir3.scale_factor        = scale_ir3
        if not no_write_cirrus and 'cirrus_reflectance' not in keys1: 
            data, scale_cirrus, offset_cirrus = Scaler.scaleData(cirrus_img)
            var_cirrus[0, :, :]               = data    
            var_cirrus.add_offset             = offset_cirrus
            var_cirrus.scale_factor           = scale_cirrus
        if not no_write_snowice and 'snowice_reflectance' not in keys1: 
            data, scale_snowice, offset_snowice = Scaler.scaleData(snowice_img)
            var_snowice[0, :, :]                = data    
            var_snowice.add_offset              = offset_snowice
            var_snowice.scale_factor            = scale_snowice
        if 'ir_brightness_temperature' not in keys1:
            var_ir.add_offset   = offset_ir
            var_ir.scale_factor = scale_ir

        if is_append:
            shape_arr = lon_shape_0
        else:
            shape_arr = lon_arr.shape[0]

        ir.close()
        f.close()
        ir     = None
        vis    = None
        f      = None
        zen    = None
        coszen = None
        print('Time to write data combined netCDF file = ' + str(time.time() - t0) + 'sec')
    else:
        with Dataset(out_nc, 'r') as combined_data:
            shape_arr = len(combined_data.dimensions['Y'])
    return(out_nc, shape_arr)    
    
def extract_universal_vars(vis, ir, glm, irdiff, cirrus, snowice, irdirty, verbose = True):
    if len(glm) > 0:
        glm_data = upscale_img_to_fit(glm, vis)
        glm_data = ndimage.median_filter(glm_data, 3)
    else:
        glm_data = []
    ir_data = upscale_img_to_fit(ir, vis)
    if len(irdiff) > 0:
        irdiff_data = upscale_img_to_fit(irdiff, vis)
    else:
        irdiff_data = []
    if len(cirrus) > 0:
        cirrus_data = upscale_img_to_fit(cirrus, vis)
    else:
        cirrus_data = []
    if len(snowice) > 0:
        snowice_data = upscale_img_to_fit(snowice, vis)
    else:
        snowice_data = []
    if len(irdirty) > 0:
        irdirty_data = upscale_img_to_fit(irdirty, vis)
    else:
        irdirty_data = []
    return(ir_data, glm_data, irdiff_data, cirrus_data, snowice_data, irdirty_data)
    
if __name__ == '__main__': sys.exit()