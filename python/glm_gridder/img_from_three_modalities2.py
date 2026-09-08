#+
# Name:
#     img_from_three_modalities2.py
# Purpose:
#     This is a script to create the image input into labelme software from combined ir_vis_glm netCDF file.
#     Adapted from img_from_three_modalities
#     Supports GOES, MTG (FCI), and SEVIRI (MSG) data.
# Calling sequence:
#     import img_from_three_modalities2
# Input:
#     None.
# Functions:
#     img_from_three_modalities2 : Main function that does the plotting
# Output:
#     Image of the 3 modlaities2
# Keywords:
#     nc_file       : STRING specifying combined IR/VIS/GLM netCDF file to read
#     out_dir       : STRING specifying output directory to send the 3 modality image file
#     no_plot_vis   : IF keyword set, do not plot the VIS data because it was not written to combined netCDF files. Setting this keyword = True makes you plot only IR data.
#                     DEFAULT is to set this keyword to False and plot the VIS and IR data.
#     no_plot_glm   : IF keyword set, do not plot the GLM data. Setting this keyword = True makes you plot only the VIS
#                     and IR data. DEFAULT is to set this keyword to True and only plot the VIS and IR data.
#     ir_min_value  : LONG keyword to specify minimum IR temperature value to clip IR data image (K).
#                     DEFAULT = 170 
#     ir_max_value  : LONG keyword to specify maximum IR temperature value to clip IR data image  (K).
#                     DEFAULT = 230
#     grid_data     : IF keyword set (True), grid the data onto latitude and longitude domain. If not set,
#                     grid the data by number of lat_pixels x number of lon_pixels.
#                     DEFAULT = False (grid pixel by pixel and do not grid data onto latitude and longitude domain)
#     plt_img_name  : IF keyword set (True), do not create 1:1 plot of lat/lon pixels and output image pixels. Instead
#                     plot image name on this figure. Used for creating videos.
#                     NOTE: DO NOT USE THESE IMAGES AS INPUT INTO LABELME SOFTWARE!!!!!!!
#     plt_cbar      : IF keyword set (True), do not create 1:1 plot of lat/lon pixels and output image pixels. Instead
#                     plot color bar on the figure. Used for creating videos.
#     subset        : [x0, y0, x1, y1] array to give subset of image domain desired to be plotted. Array could contain pixel indices or lat/lon indices
#                     DEFAULT is to plot the entire domain. Y-axis is plotted from top down so y0 index is from the top!!!! y0 = 200 implies 200 pixels from top
#     latlon_domain : [x0, y0, x1, y1] array in latitude and longitude coordinates to give subset of image domain desired to be plotted. Array must contain lat/lon values
#                     DEFAULT is to plot the entire domain. 
#                     NOTE: If region keyword is set, it supersedes this keyword
#     region        : State or sector of US to plot data around. 
#                     DEFAULT = None -> plot for full domain.
#     plt_model     : 1) If real_time == False -> Numpy array of model results (values 0-1) to plot on the image.
#                     2) If real_time == True  -> String specifying name and file path of model file to load and plot
#                     DEFAULT = None -> do not plot model results on image.
#     model         : Name of model being plotted (ex. IR+VIS)
#                     DEFAULT = None.
#     plane_data    : List containing plane data in format [longitude, latitude, and HHH_H2O]. 
#                     The last dimension (HHH_H20) is optional but if set, the plane will be different colors based on the values
#                     DEFAULT = None -> Do not plot plane on image. 
#                     NOTE: FOR DCOTSS field campaign
#     traj_data     : Dictionary (Lon, Lat, Z) containing back trajectory data of particles. 
#                     NOTE: FOR DCOTSS field campaign
#     proj          : metpy and xarray list containing object giving projection of plot, xmin, xmax, ymin, ymax. ex. [<cartopy.crs.Geostationary object at 0x7fa37f4fa130>, x.min(), x.max(), y.min(), y.max()]
#                     DEFAULT = None -> no projection done
#     replot_img    : IF keyword set (True), plot the IR/VIS sandwich image again
#                     DEFAULT = True
#     real_time     : USED for model prediction plotting. IF keyword set (True), run the code in real time by waiting until results numpy file is available to plot.
#                     DEFAULT = False -> do not wait.
#     pthresh       : FLOAT keyword to specify probability threshold to use. 
#                     DEFAULT = 0.0
#     chk_day_night : Structure containing day and night model data information in order to make seemless transition in predictions of the 2.
#                     DEFAULT = {} -> do not use day_night transition                 
#     colorblind    : IF keyword set (True), use colorblind friendly color table.
#                     DEFAULT = True
#     plt_ir_max    : FLOAT keyword specifying the maximum IR BT in color range of plotted map (K). DEFAULT = 230 K
#     plt_ir_min    : FLOAT keyword specifying the minimum IR BT in color range of plotted map (K). DEFAULT = 180 K
#     verbose       : BOOL keyword to specify whether or not to print verbose informational messages.
#                     DEFAULT = True which implies to print verbose informational messages
# Author and history:
#     Anna Cuddeback
#     John W. Cooney           2020-09-02. (Updated to include documentation and create as a function)
#                              2020-09-08. (Updated to add IR color scheme transparent on VIS data. 
#                                           Figure size made to be exactly the 1:1 in terms of pixels)
#                              2020-09-24. (Plotting VIS reflectance normalized by solar zenith angle.
#                                           Added vmin and vmax to stop flickering issue between images.
#                                           Do not plot if solar zenith angle in middle of scene > 85°)
#                              2026-06-08. (Unified plotting logic to use imshow for all satellites including
#                                           SEVIRI by reading imager_projection CF metadata).
#-

#### Environment Setup ###
# Package imports

import numpy as np
from datetime import datetime
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cv2
import os
import netCDF4
import re
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy import feature
import scipy.ndimage as ndimage
from glm_gridder import turbo_colormap
from glm_gridder.glm_data_processing import upscale_img_to_fit, ir_temp_scale
from glm_gridder.cpt_convert import loadCPT                                                                                       #Import the CPT convert function        
import time
import warnings                              
import sys
#sys.path.insert(1, '../')
sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(2, os.path.dirname(os.getcwd()))
from gridrad.rdr_sat_utils_jwc import extract_us_lat_lon_region

def img_from_three_modalities2(nc_file       = os.path.join('..', '..', '..', 'goes-data', 'combined_nc_dir', 'OR_ABI_L1b_M2_COMBINED_s20201340000494_e20201340000564_c20201340001007.nc'), 
                               out_dir       = os.path.join('..', '..', '..', 'goes-data', 'layered_img_dir'),
                               no_plot_glm   = True, no_plot_vis = False, no_plot_irdiff = True, no_plot_cirrus = True, no_plot_tropdiff = True, no_plot_snowice = True, no_plot_dirtyirdiff = True,
                               ir_min_value = 170, ir_max_value = 230, 
                               grid_data     = False, plt_img_name = False, plt_cbar = False, 
                               subset        = None, latlon_domain = [], region = None, 
                               plt_model     = None, model = None, 
                               plane_data    = [],
                               traj_data     = None, 
                               proj          = None, 
                               replot_img    = True, 
                               real_time     = False,
                               pthresh       = 0.0, 
                               chk_day_night = {}, 
                               plt_ir_max    = 195.0, plt_ir_min = 225.0,
                               colorblind    = True, verbose = True):
    '''
    Name:
        img_from_three_modalities2.py
    Purpose:
        This is a script to create the image input into labelme software from combined ir_vis_glm netCDF file.
        Adapted from img_from_three_modalities
    Calling sequence:
        import img_from_three_modalities2
    Input:
        None.
    Functions:
        img_from_three_modalities2 : Main function that does the plotting
    Output:
        Image of the 3 modlaities2
    Keywords:
        nc_file       : STRING specifying combined IR/VIS/GLM netCDF file to read
        out_dir       : STRING specifying output directory to send the 3 modality image file
        no_plot_vis   : IF keyword set, do not plot the VIS data because it was not written to combined netCDF files. Setting this keyword = True makes you plot only IR data.
                        DEFAULT is to set this keyword to False and plot the VIS and IR data.
        no_plot_glm   : IF keyword set, do not plot the GLM data. Setting this keyword = True makes you plot only the VIS
                        and IR data. DEFAULT is to set this keyword to True and only plot the VIS and IR data.
        ir_min_value  : LONG keyword to specify minimum IR temperature value to clip IR data image (K).
                        DEFAULT = 170.0
        ir_max_value  : LONG keyword to specify maximum IR temperature value to clip IR data image  (K).
                        DEFAULT = 230.0
        grid_data     : IF keyword set (True), grid the data onto latitude and longitude domain. If not set,
                        grid the data by number of lat_pixels x number of lon_pixels.
                        DEFAULT = False (grid pixel by pixel and do not grid data onto latitude and longitude domain)
        plt_img_name  : IF keyword set (True), do not create 1:1 plot of lat/lon pixels and output image pixels. Instead
                        plot image name on this figure. Used for creating videos.
                        NOTE: DO NOT USE THESE IMAGES AS INPUT INTO LABELME SOFTWARE!!!!!!!
        plt_cbar      : IF keyword set (True), do not create 1:1 plot of lat/lon pixels and output image pixels. Instead
                        plot color bar on the figure. Used for creating videos.
        subset        : [x0, y0, x1, y1] array to give subset of image domain desired to be plotted. Array could contain pixel indices or lat/lon indices
                        DEFAULT is to plot the entire domain. Y-axis is plotted from top down so y0 index is from the top!!!! y0 = 200 implies 200 pixels from top
        latlon_domain : [x0, y0, x1, y1] array in latitude and longitude coordinates to give subset of image domain desired to be plotted. Array must contain lat/lon values
                        DEFAULT is to plot the entire domain. 
                        NOTE: If region keyword is set, it supersedes this keyword
        region        : State or sector of US to plot data around. 
                        DEFAULT = None -> plot for full domain.
        plt_model     : 1) If real_time == False -> Numpy array of model results (values 0-1) to plot on the image.
                        2) If real_time == True  -> String specifying name and file path of model file to load and plot
                        DEFAULT = None -> do not plot model results on image.
        model         : Name of model being plotted (ex. IR+VIS)
                        DEFAULT = None.
        plane_data    : List containing plane data in format [longitude, latitude, and HHH_H2O]. 
                        The last dimension (HHH_H20) is optional but if set, the plane will be different colors based on the values
                        DEFAULT = None -> Do not plot plane on image. 
                        NOTE: FOR DCOTSS field campaign
        traj_data     : Dictionary (Lon, Lat, Z) containing back trajectory data of particles. 
                        NOTE: FOR DCOTSS field campaign
        proj          : metpy and xarray list containing object giving projection of plot, xmin, xmax, ymin, ymax. ex. [<cartopy.crs.Geostationary object at 0x7fa37f4fa130>, x.min(), x.max(), y.min(), y.max()]
                        DEFAULT = None -> no projection done
        replot_img    : IF keyword set (True), plot the IR/VIS sandwich image again
                        DEFAULT = True
        real_time     : USED for model prediction plotting. IF keyword set (True), run the code in real time by waiting until results numpy file is available to plot.
                        DEFAULT = False -> do not wait.
        pthresh       : FLOAT keyword to specify probability threshold to use. 
                        DEFAULT = 0.0 
        chk_day_night : Structure containing day and night model data information in order to make seemless transition in predictions of the 2.
                         DEFAULT = {} -> do not use day_night transition                 
        colorblind    : IF keyword set (True), use colorblind friendly color table.
                        DEFAULT = True
        plt_ir_max    : FLOAT keyword specifying the maximum IR BT in color range of plotted map (K). DEFAULT = 230 K
        plt_ir_min    : FLOAT keyword specifying the minimum IR BT in color range of plotted map (K). DEFAULT = 180 K
        verbose       : BOOL keyword to specify whether or not to print verbose informational messages.
                        DEFAULT = True which implies to print verbose informational messages
    Author and history:
        Anna Cuddeback
        John W. Cooney           2020-09-02. (Updated to include documentation and create as a function)
                                 2020-09-08. (Updated to add IR color scheme transparent on VIS data. 
                                              Figure size made to be exactly the 1:1 in terms of pixels)
                                 2020-09-24. (Plotting VIS reflectance normalized by solar zenith angle.
                                              Added vmin and vmax to stop flickering issue between images.
                                              Do not plot if solar zenith angle in middle of scene > 85°)
                                 2024-05-05. Added dirtyirdiff and other GOES channel plotting capabilities.
                                 test
    '''
#     ir_min_value = 190
#     ir_max_value = 260
    if plt_ir_max < plt_ir_min:
        plt_ir_max, plt_ir_min = plt_ir_min, plt_ir_max
      
    warnings.filterwarnings("ignore", category=UserWarning)
    ### File Definition ###
    nc_file      = os.path.realpath(nc_file)                                                                                      #Create link to real path so compatible with Mac
    out_dir      = os.path.realpath(out_dir)                                                                                      #Create link to real path so compatible with Mac

    basename = os.path.basename(nc_file).lower()
    if 'seviri' in basename:
        file_attr = re.split('_|.nc', os.path.basename(nc_file))                                                                
        date_str  = file_attr[3]                                                                                               
        origin    = 'upper'
    else:
        file_attr = re.split('_s|_', os.path.basename(nc_file))                                                                
        date_str  = file_attr[5]                                                                                               
        origin    = 'lower' if 'fci' in basename else 'upper'
      
    if plane_data is None:
        out_img_name = os.path.join(out_dir, re.sub(r'\.nc$', '', os.path.basename(nc_file)) + '.png')                             #Output image name and path 
    else:
        if type(plt_model) == np.ndarray:
            out_img_name = os.path.join(out_dir, re.sub(r'\.nc$', '', os.path.basename(nc_file)) + '.png')                         #Output image name and path 
        else:
            out_img_name = os.path.join(out_dir, date_str[0:-1] + '_' + re.sub(r'\.nc$', '', os.path.basename(nc_file)) + '.png')  

    if not replot_img:    
        return(out_img_name)

    if verbose: print('Image output filename = ' + str(out_img_name))         
    os.makedirs(os.path.dirname(out_img_name), exist_ok=True)                                                                 
    
    # Read combined netCDF file               
    with Dataset(nc_file, 'r') as combined_data:                                                                                          
        lon           = np.copy(np.asarray(combined_data.variables['longitude']))                                                 #Copy array of longitudes into variable
        lat           = np.copy(np.asarray(combined_data.variables['latitude']))                                                  #Copy array of latitudes into variable
        ir_raw_img2   = np.copy(np.asarray(combined_data.variables['ir_brightness_temperature']))[0, :, :]                        #Copy array of GOES IR BT into variable
        if not no_plot_glm: 
            glm_raw_img = np.copy(np.asarray(combined_data.variables['glm_flash_extent_density']))[0, :, :]                       #Copy array of GLM flash extent density into variable
        elif not no_plot_snowice:
            ir_raw_img2 = np.copy(np.asarray(combined_data.variables['snowice_reflectance']))[0, :, :]                            #Copy snow/ice channel reflectance into variable
            plt_ir_min, plt_ir_max, no_plot_vis = 0.0, 0.5, True
        elif not no_plot_cirrus:
            ir_raw_img2 = np.copy(np.asarray(combined_data.variables['cirrus_reflectance']))[0, :, :]                             #Copy cirrus channel reflectance into variable
            plt_ir_min, plt_ir_max, no_plot_vis = 0.0, 1.0, True
        elif not no_plot_dirtyirdiff:
            ir_raw_img2 = np.copy(np.asarray(combined_data.variables['dirtyir_brightness_temperature_diff']))[0, :, :]            #Copy dirty IR difference (12-10) reflectance into variable
            plt_ir_min, plt_ir_max, no_plot_vis = -2.0, 2.0, True
        elif not no_plot_tropdiff:
            ir_raw_img2 = ir_raw_img2 - np.copy(np.asarray(combined_data.variables['tropopause_temperature']))[0, :, :]           #Copy dirty IR difference (12-10) reflectance into variable
            plt_ir_min, plt_ir_max, no_plot_vis = -15.0, 20.0, True
        elif not no_plot_irdiff:
            ir_raw_img2 = np.copy(np.asarray(combined_data.variables['ir_brightness_temperature_diff']))[0, :, :]                 #Copy dirty IR difference (12-10) reflectance into variable
            plt_ir_min, plt_ir_max, no_plot_vis = -20.0, 10.0, True
        
        if not no_plot_vis: 
            vis_img = np.copy(np.asarray(combined_data.variables['visible_reflectance']))[0, :, :]           #Copy visible reflectance into variable
            zen     = np.copy(np.asarray(combined_data.variables['solar_zenith_angle']))[0, :, :]                                     #Copy solar zenith angle into variable
            tod     = 'day' if (np.nanmax(zen) < 85.0) else 'night'
            if combined_data.variables['solar_zenith_angle'].units == 'radians':         
                mid_zen = math.degrees(zen[int(ir_raw_img2.shape[0]/2), int(ir_raw_img2.shape[1]/2)])                                 #Calculate solar zenith angle for mid point of image (degrees)
            else:         
                mid_zen = zen[int(ir_raw_img2.shape[0]/2), int(ir_raw_img2.shape[1]/2)]                                               #Calculate solar zenith angle for mid point of image (degrees)
        else:
            tod     = 'day'
            mid_zen = 90
        if no_plot_snowice and no_plot_cirrus and no_plot_dirtyirdiff and no_plot_tropdiff and no_plot_irdiff:
            ir_raw_img = ir_temp_scale(ir_raw_img2, min=ir_min_value, max=ir_max_value)                                           #Scale the IR BT data using the min and maximum temperatures specified
        else:
            ir_raw_img = ir_raw_img2
        if proj is None and grid_data:                                                                                    
            img_proj = combined_data.variables['imager_projection']
            globe = ccrs.Globe(ellipse='GRS80', semimajor_axis=img_proj.semi_major_axis, 
                               semiminor_axis=img_proj.semi_minor_axis, inverse_flattening=img_proj.inverse_flattening)
            crs = ccrs.Geostationary(central_longitude=img_proj.longitude_of_projection_origin, 
                                     satellite_height=img_proj.perspective_point_height, 
                                     false_easting=0, false_northing=0, globe=globe, sweep_axis=img_proj.sweep_angle_axis)
            extent00 = re.split(',', img_proj.bounds)
            extent0 = [float(extent00[0]), float(extent00[1]), float(extent00[2]), float(extent00[3])]

    ### Image Resizing ###
    # Resize IR images to fit Vis
    ir_img = ir_raw_img                                                                                                       #Resize IR image to fit Visible data image  
    if not no_plot_vis: 
        if ir_raw_img2.shape != vis_img.shape: 
            ir_img = upscale_img_to_fit(ir_raw_img, vis_img)                                                                  #Resize IR image to fit Visible data image  
    else:         
        vis_img = ir_img*0                                                                                                    #If not plotting VIS data, make VIS array all zeroes that fit visible scale
             
    # Resize GLM images to  fit Vis         
    if not no_plot_glm:          
        if not no_plot_vis and glm_raw_img.shape != vis_img.shape: 
            glm_img = upscale_img_to_fit(glm_raw_img, vis_img)                                                            
            glm_img = ndimage.median_filter(glm_img, 3) 
        else:
            glm_img = glm_raw_img
        glm_img[glm_img == 0.0] = np.nan
    else:         
        glm_img = ir_img*0                                                                                                    #If not plotting GLM data, make GLM array all zeroes that fit visible scale
    
    if region != None:
        domain = extract_us_lat_lon_region(region)                                                                            #Extract the domain boundaries for specified US state or region
        domain = domain[region.lower()]                                                                                       #Extract the domain boundaries for specified US state or region 
    elif len(latlon_domain) > 0:
        domain = latlon_domain
            
    if subset is not None:                                                                                                        
        ir_img  = ir_img[subset[1]:subset[3], subset[0]:subset[2]]         
        glm_img = glm_img[subset[1]:subset[3], subset[0]:subset[2]]         
        vis_img = vis_img[subset[1]:subset[3], subset[0]:subset[2]]         
        lon     = lon[subset[1]:subset[3], subset[0]:subset[2]]         
        lat     = lat[subset[1]:subset[3], subset[0]:subset[2]]         
        aspect  = 'auto'         
    else:         
        aspect  = 'auto'             
    if colorblind:          
        cpt_convert = ListedColormap(turbo_colormap.turbo_colormap_data)         
    else:         
        cpt = loadCPT(os.path.join(os.path.abspath(os.getcwd()), 'rainbow.cpt'))         
        cpt_convert = LinearSegmentedColormap('cpt', cpt)                                                                     #Makes a linear interpolation with the CPT file
    cpt_convert.set_under(turbo_colormap.turbo_colormap_data[0])                                                              #Set under color to be the same as the first color
    cpt_convert.set_over(turbo_colormap.turbo_colormap_data[-1])                                                              #Set over color to be the same as the last color
    
    if not no_plot_cirrus or not no_plot_snowice:
      cpt_convert = 'gray'
    multiplier = 2 if plt_cbar else 1                                                                                                        
    my_dpi = 512.0        
    width  = (multiplier*lon.shape[1]/my_dpi)         
    height = (multiplier*lon.shape[0]/my_dpi)         
    if len(plane_data) == 0 and grid_data and lon.shape[0] <= 2000 and lon.shape[1] <= 2000:
        if width <= 3.0 or height <= 3.0:
            if width <= 1.0 or height <= 1.0:
                mplier = 4
            else:  
                mplier = 1.5
            fig = plt.figure(figsize=(width*mplier, height*mplier), dpi = my_dpi)
        else:
            fig = plt.figure(figsize=(width, height), dpi = my_dpi)                                                           #Set output image size 
    else:
        if grid_data:
            fig = plt.figure(figsize=(8, 6))                                                                                  #Set output image size 
        else:
            fig = plt.figure(figsize=(width, height), dpi = my_dpi)                                                           #Set output image size 
    if grid_data:         
        extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]         
        #Set up a map         
        ax1 = fig.add_subplot(                                                                                                #Adds subplots to a figure
              1,1,1,                                                                                                          #Number of and position of each subpanel (i.e. 1,1,1 refers to 1 row 1 column the first subplot)
              projection = ccrs.PlateCarree()                                                                                 #Cartopy projection to make the axis mapable (i.e. a geoaxis)
              )         
        if plt_cbar:
            ax1.set_position([0.07, 0.14, 0.80, 0.80])
        else:
            ax1.set_position([0.10, 0.10, 0.85, 0.85])
        if region == None and len(latlon_domain) == 0:
            ax1.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())         
            domain = [extent[0], extent[2], extent[1], extent[3]]
            extent = (domain[0], domain[2], domain[1], domain[3])    
        else:
            ax1.set_extent([domain[0], domain[2], domain[1], domain[3]], crs=ccrs.PlateCarree())    
            extent = (domain[0], domain[2], domain[1], domain[3])    
        ax1.set_facecolor(np.asarray([0.59375   , 0.71484375, 0.8828125 ]))
        ax1.add_feature(feature.LAND, edgecolor='lightgray', linewidth = 0.75, facecolor=feature.COLORS['land'])              #Adds filled land to the plot
        ax1.add_feature(feature.STATES, edgecolor='lightgray', linewidth = 0.75)         
        ax1.add_feature(feature.BORDERS, edgecolor='lightgray', linewidth = 0.75)         
        if plane_data == None:
            gl = ax1.gridlines(                                                                                               #Adds gridlines to the plot
                draw_labels = True,                                                                                           
                crs         = ccrs.PlateCarree(),                                                                             #Set the projection again
                #xlocs       = lon_ticks,                                                                                     
                #ylocs       = lat_ticks,                                                                                     
                color       = 'black',                                                                                        #Line color
                linewidth   = 0.25,                                                                                           #Line width
            )                                                                                              
            gl.xlabel_style  = {'size': 6., 'color': 'black'}                                                                 #Set the x-axis (i.e. lon label style)
            gl.ylabel_style  = {'size': 6., 'color': 'black'}                                                                 #Set the y-axis (i.e. lat label style)
            gl.top_labels    = False                                                                                          #Tell cartopy not to plot labels at the top of the plot
            gl.right_labels  = False                                                                                          #Tell cartopy not to plot labels on the right side of the plot
            gl.xformatter    = LONGITUDE_FORMATTER                                                                            #Format longitude points correctly
            gl.yformatter    = LATITUDE_FORMATTER                                                                             #Format latitude points correctly
            gl.xlines        = True                                                                                           #Tell cartopy to include longitude lines
            gl.ylines        = True                                                                                           #Tell cartopy to include latitude lines
         
        ax1.add_feature(feature.COASTLINE, linewidth = 0.75, edgecolor='lightgray')                                           #Adds coastlines to the plot

        if (mid_zen < 85.0) and not no_plot_vis:         
            if proj is None:
                if no_plot_glm:
                    out_img2 = plt.imshow(ir_img, origin = origin, vmin = plt_ir_min, vmax = plt_ir_max, transform = crs, extent = extent0, cmap = cpt_convert, interpolation = None)        #Create image from 2-D layered image (IR data)
                    out_img  = plt.imshow(vis_img, origin = origin, vmin = 0.0, vmax = 1.0, transform = crs, extent = extent0, cmap= 'gray', alpha = 0.7, interpolation = None)#Create image from 2-D layered image (VIS data)
                else:
                    out_img2 = plt.imshow(glm_img, origin = origin, vmin = 0.0, vmax = 20.0, transform = crs, extent = extent0, cmap = cpt_convert, interpolation = None)        #Create image from 2-D layered image (IR data)
                
            else:
                if no_plot_glm:
                    out_img2 = plt.imshow(ir_img, origin = origin, vmin = plt_ir_min, vmax = plt_ir_max, transform =proj[0], extent = (proj[1], proj[2], proj[3], proj[4]), cmap = cpt_convert, interpolation = None)        #Create image from 2-D layered image (IR data)
                    out_img  = plt.imshow(vis_img, origin = origin, vmin = 0.0, vmax = 1.0, transform = proj[0], extent = (proj[1], proj[2], proj[3], proj[4]), cmap= 'gray', alpha = 0.7, interpolation = None)#Create image from 2-D layered image (VIS data)
                else:
                    out_img  = plt.imshow(glm_img, origin = origin, vmin = 0.0, vmax = 20.0, transform = proj[0], extent = (proj[1], proj[2], proj[3], proj[4]), cmap= cpt_convert, interpolation = None)#Create image from 2-D layered image (VIS data)
        else:                                                                                      
            if proj is None:
                if no_plot_glm:
                    out_img2 = plt.imshow(ir_img, origin = origin, vmin = plt_ir_min, vmax = plt_ir_max, transform = crs, extent = extent0, cmap = cpt_convert, interpolation = None)
                else:
                    out_img2 = plt.imshow(glm_img, origin = origin, vmin = 0.0, vmax = 20.0, transform = ccrs.Geostationary(), extent = extent, cmap = cpt_convert, interpolation = None)
            else:
                if no_plot_glm:
                    out_img2 = plt.imshow(ir_img, origin = origin, vmin = plt_ir_min, vmax = plt_ir_max, transform = proj[0], extent = (proj[1], proj[2], proj[3], proj[4]), cmap = cpt_convert, interpolation = None)
                else:
                    out_img2 = plt.imshow(glm_img, origin = origin, vmin = 0.0, vmax = 20.0, transform = proj[0], extent = (proj[1], proj[2], proj[3], proj[4]), cmap = cpt_convert, interpolation = None)

        ir_raw_img2 = None
        zen         = None
        ir_img      = None
        vis_img     = None
        glm_img     = None
        if plt_cbar:  
            if no_plot_glm and no_plot_cirrus and no_plot_tropdiff and no_plot_snowice and no_plot_dirtyirdiff and no_plot_irdiff:
                norm       = Normalize(vmin=plt_ir_min, vmax=plt_ir_max)
                ir_ticks   = np.arange(plt_ir_min, plt_ir_max+5, 5)     
                if (plt_ir_max < 230):
                   extend = 'both'
                else:
                   extend = 'min'  
                cbar_ticks = [f'{x:.1f}' for x in ir_ticks]            
                cbar1 = fig.colorbar(ScalarMappable(norm=norm, cmap=cpt_convert),                                             #Set up the colorbar
                      cax         = fig.add_axes([ax1.get_position().x0, ax1.get_position().y0-0.05, ax1.get_position().x1-ax1.get_position().x0, 0.02]),   #Set the axis to plot the colorbar to
                      orientation = 'horizontal',                                                                             #Make the colorbar horizontal, "vertical if you want it vertical
                      label       = 'IR Temperature (K)',                                                                     #Set colorbar label
                      shrink      = 1.0,                                                                                      #Shrink the plot to 70% the size of the subplot
                      extend      = extend,
                      ticks       = ir_ticks)                                                                                 #Set the colorbar ticks
            else:
                if not no_plot_glm:
                    glm_ticks = np.arange(0.0, 25.0, 5)     
                    cbar1 = fig.colorbar(                                                                                         #Set up the colorbar
                          out_img2,                                                                                               #Plot a colorbar for this cartopy map
                          cax         = fig.add_axes([ax1.get_position().x0, ax1.get_position().y0-0.05, ax1.get_position().x1-ax1.get_position().x0, 0.02]),   #Set the axis to plot the colorbar to
                          orientation = 'horizontal',                                                                             #Make the colorbar horizontal, "vertical if you want it vertical
                          label       = 'GLM Flash Extent Density',                                                               #Set colorbar label
                          shrink      = 1.0,                                                                                      #Shrink the plot to 70% the size of the subplot
                          ticks       = glm_ticks,                                                                                #Set the colorbar ticks
                          extend      = 'max',          
                          )         
                    cbar_ticks = [f'{x:.1f}' for x in glm_ticks]         
                    cbar1.set_ticklabels(glm_ticks)                                                                               #Set the colorbar tick labels
                elif not no_plot_cirrus:
                    glm_ticks = np.arange(plt_ir_min, plt_ir_max+0.1, 0.1)     
                    cbar1 = fig.colorbar(                                                                                         #Set up the colorbar
                          out_img2,                                                                                               #Plot a colorbar for this cartopy map
                          cax         = fig.add_axes([ax1.get_position().x0, ax1.get_position().y0-0.05, ax1.get_position().x1-ax1.get_position().x0, 0.02]),   #Set the axis to plot the colorbar to
                          orientation = 'horizontal',                                                                             #Make the colorbar horizontal, "vertical if you want it vertical
                          label       = 'Cirrus Reflectance',                                                                     #Set colorbar label
                          shrink      = 1.0,                                                                                      #Shrink the plot to 70% the size of the subplot
                          ticks       = glm_ticks,                                                                                #Set the colorbar ticks
                          extend      = 'max',          
                          )         
                    cbar_ticks = ['{0:1.1f}'.format(x) for x in glm_ticks] 
                    cbar1.set_ticklabels(cbar_ticks)                                                                               #Set the colorbar tick labels
                elif not no_plot_tropdiff:
                    glm_ticks = np.arange(plt_ir_min, plt_ir_max+5, 5)    
                    cbar1 = fig.colorbar(                                                                                         #Set up the colorbar
                          out_img2,                                                                                               #Plot a colorbar for this cartopy map
                          cax         = fig.add_axes([ax1.get_position().x0, ax1.get_position().y0-0.05, ax1.get_position().x1-ax1.get_position().x0, 0.02]),   #Set the axis to plot the colorbar to
                          orientation = 'horizontal',                                                                             #Make the colorbar horizontal, "vertical if you want it vertical
                          label       = 'IR BT - TropT (K)',                                                                      #Set colorbar label
                          shrink      = 1.0,                                                                                      #Shrink the plot to 70% the size of the subplot
                          ticks       = glm_ticks,                                                                                #Set the colorbar ticks
                          extend      = 'both',          
                          )         
                    cbar_ticks = [f'{x:.1f}' for x in glm_ticks]         
                    cbar1.set_ticklabels(glm_ticks)                                                                               #Set the colorbar tick labels
                elif not no_plot_snowice:
                    glm_ticks = np.arange(plt_ir_min, plt_ir_max+0.1, 0.1)     
                    cbar1 = fig.colorbar(                                                                                         #Set up the colorbar
                          out_img2,                                                                                               #Plot a colorbar for this cartopy map
                          cax         = fig.add_axes([ax1.get_position().x0, ax1.get_position().y0-0.05, ax1.get_position().x1-ax1.get_position().x0, 0.02]),   #Set the axis to plot the colorbar to
                          orientation = 'horizontal',                                                                             #Make the colorbar horizontal, "vertical if you want it vertical
                          label       = 'Snow/Ice Reflectance',                                                                   #Set colorbar label
                          shrink      = 1.0,                                                                                      #Shrink the plot to 70% the size of the subplot
                          ticks       = glm_ticks,                                                                                #Set the colorbar ticks
                          extend      = 'max',          
                          )         
                    cbar_ticks = ['{0:1.1f}'.format(x) for x in glm_ticks] 
                    cbar1.set_ticklabels(cbar_ticks)                                                                               #Set the colorbar tick labels
                elif not no_plot_dirtyirdiff:
                    glm_ticks = np.arange(plt_ir_min, plt_ir_max+0.25, 0.25)     
                    cbar1 = fig.colorbar(                                                                                         #Set up the colorbar
                          out_img2,                                                                                               #Plot a colorbar for this cartopy map
                          cax         = fig.add_axes([ax1.get_position().x0, ax1.get_position().y0-0.05, ax1.get_position().x1-ax1.get_position().x0, 0.02]),   #Set the axis to plot the colorbar to
                          orientation = 'horizontal',                                                                             #Make the colorbar horizontal, "vertical if you want it vertical
                          label       = 'DirtyIR - IR BT (K)',                                                                    #Set colorbar label
                          shrink      = 1.0,                                                                                      #Shrink the plot to 70% the size of the subplot
                          ticks       = glm_ticks,                                                                                #Set the colorbar ticks
                          extend      = 'both',          
                          )         
                    cbar_ticks = [f'{x:.1f}' for x in glm_ticks]         
                    cbar1.set_ticklabels(glm_ticks)                                                                               #Set the colorbar tick labels
                elif not no_plot_irdiff:
                    glm_ticks = np.arange(plt_ir_min, plt_ir_max+5, 5)     
                    cbar1 = fig.colorbar(                                                                                         #Set up the colorbar
                          out_img2,                                                                                               #Plot a colorbar for this cartopy map
                          cax         = fig.add_axes([ax1.get_position().x0, ax1.get_position().y0-0.05, ax1.get_position().x1-ax1.get_position().x0, 0.02]),   #Set the axis to plot the colorbar to
                          orientation = 'horizontal',                                                                             #Make the colorbar horizontal, "vertical if you want it vertical
                          label       = 'WV - IR BT (K)',                                                                         #Set colorbar label
                          shrink      = 1.0,                                                                                      #Shrink the plot to 70% the size of the subplot
                          ticks       = glm_ticks,                                                                                #Set the colorbar ticks
                          extend      = 'both',          
                          )         
                    cbar_ticks = [f'{x:.1f}' for x in glm_ticks]         
                    cbar1.set_ticklabels(glm_ticks)                                                                               #Set the colorbar tick labels
                
        d_str     = datetime.strptime(date_str[0:4] + '-' + date_str[4:7] + 'T' + date_str[7:9] + ':' + date_str[9:11] + ':' + date_str[11:-1] + 'Z', "%Y-%jT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")       
        if len(chk_day_night) > 0:
            pthresh = chk_day_night[tod]['pthresh']
            model   = chk_day_night[tod]['mod_inputs'] + re.split('Day Night Optimal', model)[1]  
        
        pat = ''
        if model != None:
            if 'mask' in model.lower():
                ax1.set_title(model + ' valid ' + d_str, fontsize = 6)   
            else:
                pat = ' (' + str(pthresh) + ')'
                ax1.set_title(str(model) + pat + ' Model Results valid ' + d_str, fontsize = 10)               
        else:
            ax1.set_title('Model Results valid ' + d_str, fontsize = 6)      

        if real_time:
            if verbose:
                print('Figure is open and waiting for model images to be created.')
                print()
            while not os.path.exists(plt_model):
                time.sleep(0.25)                                                                                              #Wait until file exists, then load it as numpy array as soon as it does

            while True:
                try:
                    plt_model0 = np.asarray(np.load(plt_model))[0, :, :, 0]                                                   #Subset model results array
                    break
                except ValueError:
                    time.sleep(0.5)                                                                                           #Wait for file to finish writing. If open too quickly then it will not have been fully written   
        else:
            plt_model0 = np.asarray(plt_model)                                                                                #If None, make into a numpy array so that we can pass into following if statement
        if plt_model0.all() != None:                                                                                          #Plot model prediction results if passed model predictions data
            if 'mask' not in model.lower():
                mask = (plt_model0 >= pthresh)                                                                                #Extract mask of locations the model exceeds the likelihood threshold
                plt_model0[~mask] = 0.0                                                                                       #Set all values below likelihood threshold to 0
                plt_model0[mask]  = 1.0                                                                                       #Set all values above likelihood threshold to 1 
                out_img3 = plt.contour(lon,lat, plt_model0,                                                                      
                               vmin = 0.0, vmax = 2.0, linewidths = 0.025*(domain[2]-domain[0]), linestyles = 'solid',                                                                           
                               transform = ccrs.PlateCarree(),                                                                #Transform the cartopy projection correctly
                               extent    = extent, 
                               colors    = 'black'),                                                                          #Set the matplotlib colormap           
            else:
                if np.nanmax(plt_model0) <= 1:
                    out_img3 = plt.contour(lon,lat, plt_model0,                                                                      
                                   vmin = 0.0, vmax = 2.0, linewidths = 0.025*(domain[2]-domain[0]), linestyles = 'solid',                                                                           
                                   transform = ccrs.PlateCarree(),                                                            #Transform the cartopy projection correctly
                                   extent    = extent, 
                                   colors    = 'black'),                                                                      #Set the matplotlib colormap           
                else:   
                    mask = (plt_model0 == 1)                                                                                  #Extract mask of locations the model exceeds the likelihood threshold
                    plt_model02 = np.copy(plt_model0)
                    
                    plt_model0[~mask] = 0.0                                                                                   #Set all values below likelihood threshold to 0
                    plt_model0[mask]  = 1.0                                                                                   #Set all values above likelihood threshold to 1 
                    
                    out_img3 = plt.contour(lon,lat, plt_model0,                                                                     
                                   vmin = 0.0, vmax = 2.0, linewidths = 0.025*(domain[2]-domain[0]), linestyles = 'solid',                                                                          
                                   transform = ccrs.PlateCarree(),                                                            #Transform the cartopy projection correctly
                                   extent    = extent, 
                                   colors    = 'black'),                                                                      #Set the matplotlib colormap           
                   
                    mask = (plt_model02 == 2)                                                                                 #Extract mask of locations the model exceeds the likelihood threshold
                    plt_model02[~mask] = 0.0                                                                                  #Set all values below likelihood threshold to 0
                    plt_model02[mask]  = 1.0                                                                                  #Set all values above likelihood threshold to 1 
                    out_img4 = plt.contour(lon,lat, plt_model02,                                                                     
                                   vmin = 0.0, vmax = 2.0, linewidths = 0.025*(domain[2]-domain[0]), linestyles = 'solid',                                                                          
                                   transform = ccrs.PlateCarree(),                                                            #Transform the cartopy projection correctly
                                   extent    = extent, 
                                   colors    = 'darkviolet'),                                                                 #Set the matplotlib colormap           
                    plt_model02 = None
                    mask = None
            plt_model0 = None
            lon = None
            lat = None
            if not real_time:
                plt_model = None
        if plt_img_name:
            ax1.set_title('IR/VIS Sandwich valid ' + d_str + pat, fontsize = 10)
    else:
        if plt_cbar != True:
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            plt.axis('off')
        else:
            ax = plt.Axes(fig, [0.03, 0.03, 0.90, 0.90])
            fig.add_axes(ax)
        
        if (mid_zen < 85.0) and not no_plot_vis: 
           if no_plot_glm:
               out_img2 = plt.imshow(ir_img, origin = origin, vmin = plt_ir_min, vmax = plt_ir_max, cmap = cpt_convert, interpolation = None, aspect = aspect)        #Create image from 2-D layered image (IR data)
               out_img  = plt.imshow(vis_img, origin = origin, vmin = 0.0, vmax = 1.0, cmap= 'gray', alpha = 0.6, interpolation = None, aspect = aspect)#Create image from 2-D layered image (VIS data)
           else:
               out_img2 = plt.imshow(glm_img, origin = origin, vmin = 0.0, vmax = 20.0, cmap = cpt_convert, interpolation = None, aspect = aspect, extent = [0, len(lon), 0, len(lat)])        #Create image from 2-D layered image (IR data)
        else:
            if no_plot_glm:
                out_img  = plt.imshow(vis_img*0.0, origin = origin, vmin = 0.0, vmax = 1.0, cmap= 'gray', interpolation = None, aspect = aspect)         #Create image from 2-D layered image (VIS data)
                out_img2 = plt.imshow(ir_img, origin = origin, vmin = plt_ir_min, vmax = plt_ir_max, cmap = cpt_convert, interpolation = None, aspect = aspect)        #Create image from 2-D layered image (IR data)
            else:
                out_img2 = plt.imshow(glm_img, origin = origin, vmin = 0.0, vmax = 20.0, cmap = cpt_convert, interpolation = None, aspect = aspect, extent = [0, len(lon), 0, len(lat)])        #Create image from 2-D layered image (IR data)
        zen     = None
        ir_img  = None
        vis_img = None
        if real_time:
            if verbose:
                print('Figure is open and waiting for model images to be created.')
                print()
            while not os.path.exists(plt_model):
                time.sleep(0.25)                                                                                              #Wait until file exists, then load it as numpy array as soon as it does

            while True:
                try:
                    plt_model0 = np.asarray(np.load(plt_model))[0, :, :, 0]                                                   #Subset model results array
                    break
                except ValueError:
                    time.sleep(0.5)                                                                                           #Wait for file to finish writing. If open too quickly then it will not have been fully written   
        else:
            plt_model0 = np.asarray(plt_model)                                                                                #If None, make into a numpy array so that we can pass into following if statement
        if plt_model0.all() != None:
            if len(chk_day_night) > 0:
                pthresh = chk_day_night[tod]['pthresh']
                model   = chk_day_night[tod]['mod_inputs'] + re.split('Day Night Optimal', model)[1]  
            mask = (plt_model0 >= pthresh)   
            plt_model0[~mask] = 0.0          
            plt_model0[mask]  = 1.0          
            if subset != None:
              plt_model0  = plt_model0[subset[1]:subset[3], subset[0]:subset[2]]         

            out_img3 = plt.contour(plt_model0,  
                          linewidths = 0.05,
                          vmin = 0.0, vmax = 2.0,   
                          colors = 'black'),                                                                                  #Set the matplotlib colormap           
            
            plt_model0 = None
            plt_model  = None
            
        if plt_cbar:  
            if no_plot_glm and no_plot_cirrus and no_plot_tropdiff and no_plot_snowice and no_plot_dirtyirdiff and no_plot_irdiff:
                ir_ticks  = np.arange(plt_ir_min, plt_ir_max+5, 5)     
                if (plt_ir_max < 230):
                   extend = 'both'
                else:
                   extend = 'min'  
                cbar1 = fig.colorbar(                                                                                          #Set up the colorbar
                      out_img2,                                                                                                #Plot a colorbar for this map
                      ax          = ax,                                                                                        #Set the axis to plot the colorbar to
                      orientation = 'horizontal',                                                                              #Make the colorbar horizontal
                      label       = 'IR Temperature (K)',                                                                      #Set colorbar label
                      shrink      = 1.0,                                                                                       #Shrink the plot
                      ticks       = ir_ticks,                                                                                  #Set the colorbar ticks
                      extend      = extend,             
                      )            
                cbar_ticks = [f'{x:.1f}' for x in ir_ticks]            
                cbar1.set_ticklabels(cbar_ticks)                                                                               #Set the colorbar tick labels
            else:
                if not no_plot_glm:
                    #If plot GLM instead of IR
                    glm_ticks = np.arange(0.0, 25.0, 5)     
                    cbar1 = fig.colorbar(                                                                                          
                          out_img2,                                                                                                
                          ax          = ax,                                                                                        
                          orientation = 'horizontal',                                                                              
                          label       = 'GLM Flash Extent Density',                                                                
                          shrink      = 1.0,                                                                                       
                          ticks       = glm_ticks,                                                                                 
                          extend      = 'max',          
                          )         
                    cbar_ticks = [f'{x:.1f}' for x in glm_ticks]         
                    cbar1.set_ticklabels(glm_ticks)                                                                                
                elif not no_plot_cirrus:
                    glm_ticks = np.arange(plt_ir_min, plt_ir_max+0.1, 0.1)     
                    cbar1 = fig.colorbar(
                          out_img2,
                          ax          = ax,
                          orientation = 'horizontal',
                          label       = 'Cirrus Reflectance',
                          shrink      = 1.0,
                          ticks       = glm_ticks,
                          extend      = 'max',          
                          )         
                    cbar_ticks = ['{0:1.1f}'.format(x) for x in glm_ticks] 
                    cbar1.set_ticklabels(cbar_ticks)
                elif not no_plot_tropdiff:
                    glm_ticks = np.arange(plt_ir_min, plt_ir_max+5, 5)    
                    cbar1 = fig.colorbar(
                          out_img2,
                          ax          = ax,
                          orientation = 'horizontal',
                          label       = 'IR BT - TropT (K)',
                          shrink      = 1.0,
                          ticks       = glm_ticks,
                          extend      = 'both',          
                          )         
                    cbar_ticks = [f'{x:.1f}' for x in glm_ticks]         
                    cbar1.set_ticklabels(glm_ticks)
                elif not no_plot_snowice:
                    glm_ticks = np.arange(plt_ir_min, plt_ir_max+0.1, 0.1)     
                    cbar1 = fig.colorbar(
                          out_img2,
                          ax          = ax,
                          orientation = 'horizontal',
                          label       = 'Snow/Ice Reflectance',
                          shrink      = 1.0,
                          ticks       = glm_ticks,
                          extend      = 'max',          
                          )         
                    cbar_ticks = ['{0:1.1f}'.format(x) for x in glm_ticks] 
                    cbar1.set_ticklabels(cbar_ticks)
                elif not no_plot_dirtyirdiff:
                    glm_ticks = np.arange(plt_ir_min, plt_ir_max+0.25, 0.25)     
                    cbar1 = fig.colorbar(
                          out_img2,
                          ax          = ax,
                          orientation = 'horizontal',
                          label       = 'DirtyIR - IR BT (K)',
                          shrink      = 1.0,
                          ticks       = glm_ticks,
                          extend      = 'both',          
                          )         
                    cbar_ticks = [f'{x:.1f}' for x in glm_ticks]         
                    cbar1.set_ticklabels(glm_ticks)
                elif not no_plot_irdiff:
                    glm_ticks = np.arange(plt_ir_min, plt_ir_max+5, 5)     
                    cbar1 = fig.colorbar(
                          out_img2,
                          ax          = ax,
                          orientation = 'horizontal',
                          label       = 'WV - IR BT (K)',
                          shrink      = 1.0,
                          ticks       = glm_ticks,
                          extend      = 'both',          
                          )         
                    cbar_ticks = [f'{x:.1f}' for x in glm_ticks]         
                    cbar1.set_ticklabels(glm_ticks)
         
        if plt_img_name:           
            d_str     = date_str                                                                                          #Split file string in order to extract date string of scan
            plt.text(50, 50, d_str[7:-1] + 'Z', fontsize = 'xx-small',color = 'cyan', ma = 'center')         
    if grid_data:
        plt.savefig(out_img_name, dpi = my_dpi, bbox_inches = 'tight')                                                        #Save figure
    else:
        plt.savefig(out_img_name, dpi = my_dpi)                                                                               #Save figure
    plt.close()                                                                                                               #Close figure window
    plt.clf()
    
    return(out_img_name)                                                                                                          #Return the name and file path of the output image. This can be used to write image into google cloud storage

if __name__ == '__main__': sys.exit()