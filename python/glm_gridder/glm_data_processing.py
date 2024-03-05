#+
# Name:
#     glm_data_processing.py
# Purpose:
#     This is a script to 
# Calling sequence:
#     from glm_data_processing import *
# Input:
#     None.
# Functions:
#     fetch_glm_to_match_ir : input IR and GLM data and returns GLM data on IR grid
#     get_ref_from_rad      : Convert radiance to reflectance and return reflectance
#     get_ir_bt_from_rad    : Convert IR radiance into brightness temperature
#     get_lat_lon_from_vis  : Extract lat and longitude for imager projection
#     upscale_img_to_fit    : Upscales a _small_img to the dimensions of _large_img
#     downscale_img_to_fit  : Downscales a _large_img to the dimensions of _small_img
#     create_layered_img    : Create the layered image of the 3 modalities (uses red, blue, and green channel)
#                             (red = IR, blue = VIS, and green = GLM)
#     min_max_scale         : returns data - min scaled
#     max_min_scale         : returns data - max scaled
# Output:
#      Returns data for particular functions that are called
# Keywords:
#     None.
# Author and history:
#     John W. Cooney.          2020-09-01.
#                              2020-09-09. (Added verbose keyword and made some deleted old excess information)
#-
#

#### Environment Setup ###
# Package imports
import numpy as np
import math
import time
import pandas as pd
import re
import os
import matplotlib.patches as patches
import cv2
import warnings
#from pysolar.solar import *
#import cartopy.crs as ccrs

def fetch_glm_to_match_ir(_ir_dataset, _glm_dataset, _northern_hemisphere=True, _rect_color='y', _label= ''):
    '''
    Note:
        for images in the northern hemisphere, the minimum value in the y,
        axis corresponds to bottom of the image and for images in the,
        southern hemisphere, the minimum value in the y axis corresponds,
        to the top of the image.,
        There is a possibility that I have not done the appropriate,
        calculations to ensure the proper use of positive values for,
        longitude as the data that I wrote this for had negative longitude.,
    '''
    # fetch IR data axes",
    ir_x = np.copy(np.asarray(_ir_dataset.variables['x']))
    ir_y = np.copy(np.asarray(_ir_dataset.variables['y']))
    # get bounds of IR data axes \n",
    ir_min_x = np.min(ir_x)
    ir_max_x = np.max(ir_x)
    ir_min_y = np.min(ir_y)
    ir_max_y = np.max(ir_y)

    # fetch GLM data axes,
    glm_x = np.copy(np.asarray(_glm_dataset.variables['x']))
    glm_y = np.copy(np.asarray(_glm_dataset.variables['y']))

    # get GLM indices matching IR data bounds\n",
    index_min_x = (np.abs(glm_x-ir_min_x)).argmin()
    index_max_x = (np.abs(glm_x-ir_max_x)).argmin()
    index_min_y = (np.abs(glm_y-ir_min_y)).argmin()
    index_max_y = (np.abs(glm_y-ir_max_y)).argmin()
    # get width and height for rectangles\n",
    width  = abs(index_max_x - index_min_x)
    height = abs(index_max_y - index_min_y)
    # get subset of GLM data matching IR data \n",
    if _northern_hemisphere:
        glm_img = np.copy(np.asarray(_glm_dataset.variables['flash_extent_density'][index_max_y:index_min_y+1,index_min_x:index_max_x+1]))
        rect= patches.Rectangle((index_min_x, index_max_y), width, height, edgecolor=_rect_color, lw=2.7, facecolor=None, label=_label)
    else:
        glm_img = np.copy(np.asarray(_glm_dataset.variables['flash_extent_density'][index_min_y:index_max_y+1,index_min_x:index_max_x+1]))
        rect= patches.Rectangle((index_min_x, index_min_y), width, height, edgecolor=_rect_color, lw=2.7, facecolor= None, label=_label)
    return(glm_img, rect)

def get_ref_from_rad(_vis_dataset):
    raw_img = np.copy(np.asarray(_vis_dataset.variables['Rad'][:,:]))
#     Esun_Ch_01 = 726.721072#AACP master
#     Esun_Ch_02 = 663.274497
#     Esun_Ch_03 = 441.868715
#     d2 = 0.3
#     # Apply the formula to convert radiance to reflectance
#     ref_img = (raw_img * np.pi * d2) / Esun_Ch_02
    kappa_factor = np.copy(np.asarray(_vis_dataset.variables['kappa0'][:]))
    ref_img = kappa_factor * raw_img
    return(ref_img)
    
    
def get_ir_bt_from_rad(_ir_dataset):
    planck_fk1 = np.copy(np.asarray(_ir_dataset.variables['planck_fk1'][:]))
    planck_fk2 = np.copy(np.asarray(_ir_dataset.variables['planck_fk2'][:]))
    planck_bc1 = np.copy(np.asarray(_ir_dataset.variables['planck_bc1'][:]))
    planck_bc2 = np.copy(np.asarray(_ir_dataset.variables['planck_bc2'][:]))
    raw_img    = np.copy(np.asarray(_ir_dataset.variables['Rad'][:,:]))
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      bt       = (planck_fk2 / (np.log( (planck_fk1/raw_img)+1 )) - planck_bc1) / planck_bc2
    return(bt)

# def get_lat_lon_from_vis2(_vis_dataset, extract_proj_coordinates = False, verbose = True):    
#     # GOES-R projection info and retrieving relevant constants look up kerchunk to get ref jsons
#     proj_info   = _vis_dataset.variables['goes_imager_projection']
#     lon_origin  = proj_info.longitude_of_projection_origin
#     lat_rad_1d  = _vis_dataset.variables['y'][:]
#     lon_rad_1d  = _vis_dataset.variables['x'][:]
#     H           = proj_info.perspective_point_height+proj_info.semi_major_axis
#     r_eq        = proj_info.semi_major_axis
#     r_pol       = proj_info.semi_minor_axis
#     x1          = (lon_rad_1d * proj_info.perspective_point_height).astype('float64')
#     y1          = (lat_rad_1d * proj_info.perspective_point_height).astype('float64')
#     proj_extent = (np.min(x1), np.max(x1), np.min(y1), np.max(y1))
#     if extract_proj_coordinates == True:
#         x2, y2 = np.meshgrid(x1,y1)
# 
# # Define the image extent
#     # Data info
# #     t0 = time.time()
#     # create meshgrid filled with radian angles
#     lon_rad,lat_rad = np.meshgrid(lon_rad_1d,lat_rad_1d)
# #     print('Time to create the meshgrid = ' + str(time.time()-t0) + ' sec')
#     # lat/lon calc routine from satellite radian angle vectors
# #     t0 = time.time()
#     
# #    lambda_0 = (lon_origin*np.pi)/180.0
# #     sin_lon = np.sin(lon_rad)
# #     cos_lon = np.cos(lon_rad)
# #     sin_lat = np.sin(lat_rad)
# #     cos_lat = np.cos(lat_rad)
# #     t0 = time.time()
# #     sin_lat2 = np.square(sin_lat)
# #     cos_lat2 = np.square(cos_lat)
# #     sin_lon2 = np.square(sin_lon)
# #     cos_lon2 = np.square(cos_lon)
# #     r_eq2    = np.square(r_eq)
# #     r_pol2   = np.square(r_pol)
# #     H2       = np.square(H)
# #     print('Time to run squaring2 = ' + str(time.time()-t0) + ' sec')
#     t0 = time.time()
#     a_var = np.power(np.sin(lon_rad),2.0) + (np.power(np.cos(lon_rad),2.0)*(np.power(np.cos(lat_rad),2.0)+(((np.square(r_eq))/(np.square(r_pol)))*np.power(np.sin(lat_rad),2.0))))
#     b_var = -2.0*H*np.cos(lon_rad)*np.cos(lat_rad)
#     c_var = np.square(H) - np.square(r_eq)
#     
#     r_s  = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
# 
#     s_x =  r_s*np.cos(lon_rad)*np.cos(lat_rad)
#     s_y = -r_s*np.sin(lon_rad)
#     s_z =  r_s*np.cos(lon_rad)*np.sin(lat_rad)
#     lat_arr = np.degrees(np.arctan(((np.square(r_eq))*s_z)/((np.square(r_pol))*np.sqrt((np.square((H-s_x))) + (np.square(s_y))))))
#     lon_arr = (lon_origin - np.degrees(np.arctan(s_y/(H-s_x))))
#     print('Time to run lat/lon equations = ' + str(time.time()-t0) + ' sec')
#     if extract_proj_coordinates == True:
#         return(lon_arr, lat_arr, proj_info, proj_extent, x2, y2)    
#     else:
#         return(lon_arr, lat_arr, proj_info, proj_extent)

#     a_var0 = np.copy(a_var)
#     b_var0 = np.copy(b_var)
#     c_var0 = np.copy(c_var)
#     r_s0   = np.copy(r_s)
#     s_y0   = np.copy(s_x)
#     s_z0   = np.copy(s_y)
#     s_x0   = np.copy(s_z)
#     lat_arr0 = np.copy(lat_arr)
#     lon_arr0 = np.copy(lon_arr)
# 
# #    a_var = np.add(sin_lon2, np.multiply(cos_lon2, np.add(cos_lon2, np.multiply(np.divide(r_eq2, r_pol2), sin_lat2))))
#     a_var = np.power(np.sin(lon_rad),2.0) + (np.power(np.cos(lon_rad),2.0)*(np.power(np.cos(lat_rad),2.0)+(((np.square(r_eq))/(np.square(r_pol)))*np.power(np.sin(lat_rad),2.0))))
#     print(np.nanmax(np.abs(np.subtract(a_var, a_var0))))
#     b_var = -2.0*H*np.cos(lon_rad)*np.cos(lat_rad)
#     c_var = np.subtract(H2, r_eq2)
#     print('Time to run var equations = ' + str(time.time()-t0) + ' sec')
#     t0 = time.time()   
#     r_s = np.divide(np.subtract(-1.0*b_var, np.sqrt(np.subtract(np.square(b_var), 4.0*np.multiply(a_var, c_var)))), 2.0*a_var)
#     print(np.nanmax(np.abs(np.subtract(r_s, r_s0))))
#     
#     s_x = np.multiply(r_s, cos_lon, cos_lat)
#     print(np.nanmax(np.abs(np.subtract(s_x, s_x0))))
#     s_y = -1.0*np.multiply(r_s,sin_lon)
#     print(np.nanmax(np.abs(np.subtract(s_y, s_y0))))
#     s_z = np.multiply(r_s, cos_lon, sin_lat)
#     print(np.nanmax(np.abs(np.subtract(s_z, s_z0))))
#     print('Time to run s equations = ' + str(time.time()-t0) + ' sec')
#  
#     t0 = time.time()
#     Hsx = np.subtract(H,s_x)
#     lat = np.degrees(np.arctan(np.multiply(np.divide(r_eq2, r_pol2), np.divide(s_z, np.sqrt(np.add(np.square(Hsx), np.square(s_y)))))))
#     lon = np.subtract(lon_origin, np.degrees(np.arctan(np.divide(s_y, Hsx))))
#     print(np.nanmax(lat_arr))
#     print(np.nanmin(lat_arr))
#     print(np.nanmax(lon_arr))
#     print(np.nanmin(lon_arr))
#     print(proj_extent)
#     exit()

#     a_var = sin_lat2 + (cos_lat2*(cos_lon2+((r_eq2/r_pol2)*sin_lon2)))
#     b_var = -2.0*H*cos_lat*cos_lon
#     c_var = H2-r_eq2
#     print('Time to run var equations = ' + str(time.time()-t0) + ' sec')
#     t0 = time.time()
#     
#     r_s = (-1.0*b_var - np.sqrt(b_var**2 - (4.0*a_var*c_var)))/(2.0*a_var)
#     
#     s_x = r_s*cos_lat*cos_lon coil-blue.larc.nasa.gov
#     s_y = - r_s*sin_lat
#     s_z = r_s*cos_lat*sin_lon
#     print('Time to run s equations = ' + str(time.time()-t0) + ' sec')
#  
# #     a_var = np.power(np.sin(lat_rad),2.0) + (np.power(np.cos(lat_rad),2.0)*(np.power(np.cos(lon_rad),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(lon_rad),2.0))))
# #     b_var = -2.0*H*np.cos(lat_rad)*np.cos(lon_rad)
# #     c_var = (np.power(H, 2))-(np.power(r_eq, 2))
# #     
# #     r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
# #     
# #     s_x = r_s*np.cos(lat_rad)*np.cos(lon_rad)
# #     s_y = - r_s*np.sin(lat_rad)
# #     s_z = r_s*np.cos(lat_rad)*np.sin(lon_rad)
#     t0 = time.time()
#     Hsx = H-s_x
#     lat = np.degrees(np.arctan((r_eq2/r_pol2)*(s_z/np.sqrt(Hsx**2 + s_y**2))))
# #    lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
#     lon = lon_origin - np.degrees(np.arctan(s_y/Hsx))
#     print('Time to run lat/lon equations = ' + str(time.time()-t0) + ' sec')
#     if extract_proj_coordinates == True:
#         return(lon, lat, proj_info, proj_extent, x2, y2)    
#     else:
#         return(lon, lat, proj_info, proj_extent)
    
def get_lat_lon_from_vis2(_vis_dataset, extract_proj_coordinates = False, verbose = True):
    lat_rad_1d  = _vis_dataset.variables['y'][:]
    lon_rad_1d  = _vis_dataset.variables['x'][:]
    proj_info   = _vis_dataset.variables['goes_imager_projection']
    x1          = (lon_rad_1d * _vis_dataset.variables['goes_imager_projection'].perspective_point_height).astype('float64')
    y1          = (lat_rad_1d * _vis_dataset.variables['goes_imager_projection'].perspective_point_height).astype('float64')
    proj_extent = (np.min(x1), np.max(x1), np.min(y1), np.max(y1))
    if extract_proj_coordinates == True:
        x2, y2 = np.meshgrid(x1,y1)
    proj_img    = _vis_dataset.variables['goes_imager_projection']
    lat_rad_1d  = lat_rad_1d*proj_info.perspective_point_height
    lon_rad_1d  = lon_rad_1d*proj_info.perspective_point_height
    try:
#         print('first try...', end='')
        globe = ccrs.Globe(semimajor_axis=proj_info.semi_major_axis,
                           semiminor_axis=proj_info.semi_minor_axis,
                           inverse_flattening=proj_info.inverse_flattening)
        geos = ccrs.Geostationary(central_longitude=proj_info.longitude_of_projection_origin, 
                                  satellite_height=proj_info.perspective_point_height,
                                  sweep_axis=proj_info.sweep_angle_axis,
                                  globe=globe)
    
    except:
#         print('second try')
        globe = ccrs.Globe(semimajor_axis=proj_info.semi_major_axis[0],
                       semiminor_axis=proj_info.semi_minor_axis[0],
                       inverse_flattening=proj_info.inverse_flattening[0])
        geos = ccrs.Geostationary(central_longitude=proj_info.longitude_of_projection_origin[0], 
                                  satellite_height=proj_info.perspective_point_height[0],
                                  sweep_axis=proj_info.sweep_angle_axis,
                                  globe=globe)   
   
    lon_arr, lat_arr = np.meshgrid(lon_rad_1d,lat_rad_1d)
    a = ccrs.PlateCarree().transform_points(geos, lon_arr, lat_arr)
    lons, lats, _ = a[:,:,0], a[:,:,1], a[:,:,2]
    lats[np.isinf(lats)] = np.nanmin(lats)
    lons[np.isinf(lons)] = np.nanmin(lons)
#     if verbose == True:
#         print('Longitude bounds = ' + str(np.nanmin(lons)) + ', ' + str(np.nanmax(lons)))
#         print('Latitude bounds = ' + str(np.nanmin(lats)) + ', ' + str(np.nanmax(lats)))
    if extract_proj_coordinates == True:
        return(lons, lats, proj_img, proj_extent, x2, y2)    
    else:
        return(lons, lats, proj_img, proj_extent)
    
def get_lat_lon_from_vis(_vis_dataset, verbose = True):
    lat_rad_1d  = _vis_dataset.variables['y'][:]
    lon_rad_1d  = _vis_dataset.variables['x'][:]
    x1          = (lon_rad_1d * _vis_dataset.variables['goes_imager_projection'].perspective_point_height).astype('float64')
    y1          = (lat_rad_1d * _vis_dataset.variables['goes_imager_projection'].perspective_point_height).astype('float64')
    proj_extent = (np.min(x1), np.max(x1), np.min(y1), np.max(y1))
    proj_img    = _vis_dataset.variables['goes_imager_projection']

# Define the image extent
    proj_info  = _vis_dataset.variables['goes_imager_projection']
    lon_origin = proj_info.longitude_of_projection_origin
    H          = proj_info.perspective_point_height+proj_info.semi_major_axis
    r_eq       = proj_info.semi_major_axis
    r_pol      = proj_info.semi_minor_axis
    # create meshgrid filled with radian angles
    lon_rad,lat_rad = np.meshgrid(lon_rad_1d,lat_rad_1d)

#    a_var0 = np.power(np.sin(lon_rad),2.0) + (np.power(np.cos(lon_rad),2.0)*(np.power(np.cos(lat_rad),2.0)+(((r_eq**2)/(r_pol**2))*np.power(np.sin(lat_rad),2.0))))
    a_var = np.power(np.sin(lon_rad),2.0) + (np.power(np.cos(lon_rad),2.0)*(np.power(np.cos(lat_rad),2.0)+(((np.square(r_eq))/(np.square(r_pol)))*np.power(np.sin(lat_rad),2.0))))
    b_var = -2.0*H*np.cos(lon_rad)*np.cos(lat_rad)
#    c_var0 = (H**2.0)-(r_eq**2.0)
    c_var = np.square(H) - np.square(r_eq)
    
    r_s  = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
 #   r_s   = (-1.0*b_var - np.sqrt(np.square(b_var)-(4.0*a_var*c_var)))/(2.0*a_var)
 #   print(np.max(abs(r_s - r_s0)))

    s_x =  r_s*np.cos(lon_rad)*np.cos(lat_rad)
    s_y = -r_s*np.sin(lon_rad)
    s_z =  r_s*np.cos(lon_rad)*np.sin(lat_rad)
#    lat_arr0 = np.degrees(np.arctan(((r_eq**2)*s_z)/((r_pol**2)*np.sqrt(((H-s_x)**2) + (s_y**2)))))
    lat_arr = np.degrees(np.arctan(((np.square(r_eq))*s_z)/((np.square(r_pol))*np.sqrt((np.square((H-s_x))) + (np.square(s_y))))))
    lon_arr = (lon_origin - np.degrees(np.arctan(s_y/(H-s_x))))
#     if verbose == True:
#         print('Longitude bounds = ' + str(np.nanmin(lon_arr)) + ', ' + str(np.nanmax(lon_arr)))
#         print('Latitude bounds = ' + str(np.nanmin(lat_arr)) + ', ' + str(np.nanmax(lat_arr)))

    return(lon_arr, lat_arr, proj_img, proj_extent)

def get_lat_lon_subset_inds(_scan_proj_dataset, xy_bounds, 
                           lat           = [], lon = [], 
                           return_lats   = False, 
                           region_csvdir = os.path.join('..', '..', 'data', 'region'), 
                           satellite     = 'goes16', scan_mode = 'full', 
                           use_native_ir = False, 
                           verbose       = True):
        
    x1  = _scan_proj_dataset.variables['x'][:]
    y1  = _scan_proj_dataset.variables['y'][:]
    if len(lat) == 0:
        lat = _scan_proj_dataset.variables['latitude'][:]
    if len(lon) == 0:
        lon = _scan_proj_dataset.variables['longitude'][:]
    
    if len(xy_bounds) > 0:        
        if satellite[0:4].lower() == 'goes':                                                                                                   #Ensure satellite chosen is possible
          mod_check = 1
          sat       = satellite[0:4].lower() + '-' + str(satellite[4:].lower())
          if sat.endswith('-'):
            mod_check = 0
            print('You must specify which goes satellite number!')
            print()
        elif satellite[0].lower() == 'g' and len(satellite) <= 3:
          mod_check = 1
          sat       = 'goes-' + str(satellite[1:].lower())
        elif satellite.lower() == 'seviri':
          mod_check = 1
          sat       = 'seviri'
        else:
          print('Model is not currently set up to handle satellite specified. Please try again.')
          print()
          print(satellite)
          exit()

        sat        = ''.join(re.split('-', sat))                                                                                               #Extract satellite name without - character
        if use_native_ir == True:
          region_csv = os.path.join(os.path.realpath(region_csvdir), sat + '_ir_region_bounds_and_scan_indices.csv')
        else:
          region_csv = os.path.join(os.path.realpath(region_csvdir), sat + '_vis_region_bounds_and_scan_indices.csv')
        if os.path.exists(region_csv):
            df = pd.read_csv(region_csv)
        else:
            df = pd.DataFrame()
        if df.empty == True:
            move_along = 1
        else:    
            minlons    = []
            minlats    = []
            maxlons    = []
            maxlats    = []
            scan_modes = []
            for r in range(len(df)):
              minlons.append("{:.7f}".format(df['min_longitude'][r]))
              minlats.append("{:.7f}".format(df['min_latitude'][r]))
              maxlons.append("{:.7f}".format(df['max_longitude'][r]))
              maxlats.append("{:.7f}".format(df['max_latitude'][r]))
              scan_modes.append(df['scan_mode'][r])
            minlons    = np.asarray(minlons)
            minlats    = np.asarray(minlats)
            maxlons    = np.asarray(maxlons)
            maxlats    = np.asarray(maxlats)
            scan_modes = np.asarray(scan_modes)
            loc = np.where((scan_modes == scan_mode) & (minlons == "{:.7f}".format(xy_bounds[0])) & (maxlons == "{:.7f}".format(xy_bounds[2])) & (minlats == "{:.7f}".format(xy_bounds[1])) & (maxlats == "{:.7f}".format(xy_bounds[3])))[0]
            if len(loc) > 0:
                x_inds     = [df['min_lon_ind'][loc[0]], df['max_lon_ind'][loc[0]]]
                y_inds     = [df['min_lat_ind'][loc[0]], df['max_lat_ind'][loc[0]]]
                move_along = 0
            else:
                move_along = 1
        if move_along == 1: 
            inds   = np.where((lon >= (xy_bounds[0]-1.0)) & (lon <= (xy_bounds[2]+1.0)) & (lat >= (xy_bounds[1]-1.0)) & (lat <= (xy_bounds[3])+1.0))
            x_inds = [np.min(inds[0]), np.max(inds[0])]
            y_inds = [np.min(inds[1]), np.max(inds[1])]
#             if ((np.max(x_inds) - np.min(x_inds)) % 2) != 0:
#                 if np.min(x_inds) > 0:
#                     x_inds = [np.min(x_inds)-1, np.max(x_inds)]
#                 else:
#                     x_inds = [np.min(x_inds)+1, np.max(x_inds)]
#             if ((np.max(y_inds) - np.min(y_inds)) % 2) != 0:
#                 if np.min(y_inds) > 0:
#                     y_inds = [np.min(y_inds)-1, np.max(y_inds)]
#                 else:
#                     y_inds = [np.min(y_inds)+1, np.max(y_inds)]
            if ((np.max(x_inds) - np.min(x_inds)) % 128) != 0:
                if np.min(x_inds) > 128:
                    x_inds2 = [np.min(x_inds)-(128 - (np.max(x_inds) - np.min(x_inds)) % 128), np.max(x_inds)]
                else:
                    x_inds2 = [np.min(x_inds)+(128 - (np.max(x_inds) - np.min(x_inds)) % 128), np.max(x_inds)]
                x_inds = x_inds2
            if ((np.max(y_inds) - np.min(y_inds)) % 128) != 0:
                if np.min(y_inds) > 128:
                    y_inds2 = [np.min(y_inds)-(128 - (np.max(y_inds) - np.min(y_inds)) % 128), np.max(y_inds)]
                else:
                    y_inds2 = [np.min(y_inds)+(128 - (np.max(y_inds) - np.min(y_inds)) % 128), np.max(y_inds)]
                y_inds = y_inds2
            
            if len(x_inds) <= 0 or len(y_inds) <=0:
                print('No longitude or latitudes found within domain region for scan.')
                print('Domain region bound = : '+ str([np.nanmin(lon), np.nanmin(lat), np.nanmax(lon), np.nanmax(lat)]))
                print('User specified domain = ' + str(xy_bounds))
                exit()
        proj_extent = (np.nanmin(x1[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]), np.nanmax(x1[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]), np.nanmin(y1[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]), np.nanmax(y1[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)]))
    else:
        x_inds      = []    
        y_inds      = []    
        proj_extent = (np.min(x1), np.max(x1), np.min(y1), np.max(y1))
#     if verbose == True:
#         print('Longitude bounds = ' + str(np.nanmin(lon[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)])) + ', ' + str(np.nanmax(lon[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)])))
#         print('Latitude bounds = ' + str(np.nanmin(lat[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)])) + ', ' + str(np.nanmax(lat[np.min(x_inds):np.max(x_inds), np.min(y_inds):np.max(y_inds)])))
     
    if return_lats == True:
        return(x_inds, y_inds, proj_extent, lon.shape, lat, lon)
    else:
        return(x_inds, y_inds, proj_extent, lon.shape)

def upscale_img_to_fit(_small_img, _large_img):
    #   upscales a _small_img to the dimensions of _large_img
    
    return(cv2.resize(_small_img, (_large_img.shape[1], _large_img.shape[0]), interpolation=cv2.INTER_NEAREST))

def downscale_img_to_fit(_small_img, _large_img):
    #   downscales a _large_img to the dimensions of _small_img
    
    return(cv2.resize(_large_img, (_small_img.shape[1], _small_img.shape[0])))#, interpolation=cv2.INTER_NEAREST))
    
def create_layered_img(_red_channel, _green_channel, _blue_channel, min_ir_temp, max_ir_temp):
    # define channels\n",
    r_img = np.copy(_red_channel)
    g_img = np.copy(_green_channel)
    b_img = np.copy(_blue_channel)
    # data scaling -- min-max scaling for now, but might need to be changed
    b_img = min_max_scale(b_img) #max_min_scale(b_img)
    if np.amin(g_img) != 0 and np.amax(g_img) != 0: 
        g_img = min_max_scale(g_img)

    r_img = ir_temp_scale(r_img, min_ir_temp, max_ir_temp)                                          #Clips IR data at set max and min values
#    r_img = min_max_scale(r_img)                                                                    #Scales data based on min and max values in the file
    return(np.dstack((r_img, g_img, b_img)))

# def calc_solar_zenith_angle(lon, lat, date):
#     altitude_deg = get_altitude(lat, lon, date)
#     zen = 90.0 - altitude_deg
#     return(math.radians(zen))

def min_max_scale(_data):
    return((_data - _data.min())/(_data.max() - _data.min()))
     
def max_min_scale(_data):
    return(abs(_data - _data.max())/(_data.max() - _data.min()))
    
def ir_temp_scale(_data, min = 170, max = 230):
    ''',
    Clips edges of temperature data in degrees kelvin from 170 to 230",
    ''',
    # assume input of data in degrees kelvin,
    temp = np.copy(_data)
    if max != None and min != None:
        temp[ (temp > max) | (temp < min) ] = np.nan     
    elif max != None:    
        temp[ (temp > max) ] = np.nan 
    elif min != None:
        temp[ (temp < min) ] = np.nan 
    elif min == None and max == None:
        temp = temp
    else:    
        if np.amax(temp) < min or np.amin(temp) > max:
            print('Error! Data temperature likely not in Kelvin')
            print('Max IR BT = ' + str(np.amax(temp)))
            print('Min IR BT = ' + str(np.amin(temp)))
        temp[ (temp > max) | (temp < min) ] = np.nan 
#     if min == None and max != None:
#         temp[ (temp > max) ] = np.nan 
#     elif min != None and max == None:
#         temp[ (temp < min) ] = np.nan 
#     elif min == None and max == None:
#         temp = temp
#     else:    
#         if np.amax(temp) < min or np.amin(temp) > max:
#             print('Error! Data temperature likely not in Kelvin')
#             print('Max IR BT = ' + str(np.amax(temp)))
#             print('Min IR BT = ' + str(np.amin(temp)))
#         temp[ (temp > max) | (temp < min) ] = np.nan 
#    return((temp - min)/(max - min))
    return(temp)

