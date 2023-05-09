'''
Created on Jun 20, 2018
@author: yhuang20 and caliles
'''

import numpy as np
import pandas as pd
import cv2
import imageio
from datetime import timedelta
from dateutil.parser import parse
from scipy.io import netcdf
from skimage.transform import resize, rotate
from ast import literal_eval
import os


def rdr_df_subset(rdr_df, bad_list):
  '''
  Take the subset of rdr_df based only on plumes where IR data file is valid and Vis data file is valid
  and dist between NC file IR pixel and the radar hit is less than 5 KM.
  Args:
    rdr_df: dataframe holding the radar info which has already been combined with the sat information]
  Returns:
    aacp_rdr_df: dataframe afer subsetting containing only AACP occurences
    non_aacp_rdr_df: dataframe afer subsetting containing only AACP occurences
  '''
  # Remove the distance delta great than 5km
  rdr_df = rdr_df[(rdr_df['IR_Data_File'] != 'NO_VALID_FILE') & (rdr_df['IR_dist_delta'] < 5) & 
                  #(rdr_df['VIS_Data_File'] != 'NO_VALID_FILE') & (rdr_df['VIS_dist_delta'] < 5) & 
                  (~rdr_df['IR_Data_File'].isin(bad_list))]
  # Only take data with label 1, AACP
  aacp_rdr_df = rdr_df[(rdr_df['Plume'] == 1)]
  # Only take data with label 0, Non-AACP never turn into AACP
  non_aacp = rdr_df.groupby('Storm Number', as_index = False).agg('sum')
  non_aacp = non_aacp[non_aacp['Plume'] == 0.0]
  non_aacp_rdr_df = rdr_df.loc[rdr_df['Storm Number'].isin(non_aacp['Storm Number'])]
  return aacp_rdr_df, non_aacp_rdr_df


def nc_img_data(filename, img_dir):
  '''
  Use netcdf to read .nc file and get image's data
  Args: 
    file_name: .nc file name
    img_dir: directory where the filename is stored
  Returns: 
    image_data: numpy array with image data from the .nc file
  '''
  f = netcdf.netcdf_file(img_dir + filename, 'r')
  img_data = np.copy(np.asarray(f.variables['data'][0,:,:]))
  f.close()
  return img_data


def rotate_aacp(img_data, plumes, sz, px_sz):
  '''
  Uniformly rotate the original image between 0.00 to 270.00 degree (two decimal) and randomly
  flip this image horizontally or vertically
  Arg:
    img_data: input a padding boarder image with twice sample pixel size
    plumes: input numpy array of aacp index
    sz: input number of non-aacp
    px_sz: input sampling pixel size
  Return:
    ret_img: output same count of non-aacp image with randomly rotated and (flip or not flip)
  '''
  ret_img = np.zeros((sz, px_sz, px_sz))
  for i in range(sz):
    # randomly generate number between 0 to 3, if 0 flip vertical, if 1 flip horizontal, otherwise no flip
    flip_ax = np.random.randint(4)
    # uniformly generate a float degree number between 0.00 to 270.00 
    #rotate_angle = np.random.uniform(low=0, high=270, size=1).astype(int)[0]
    rotate_angle = round(np.random.uniform(low=0, high=270, size=1)[0],2)
    # randomly pick a position from the aacp index
    indx = plumes[np.random.randint(plumes.shape[0])]
    M = cv2.getRotationMatrix2D((indx[0],indx[1]), rotate_angle, 1)
    rotated_img = cv2.warpAffine(img_data, M, img_data.shape)
    
    ret_img[i, :, :] = rotated_img[indx[0] + px_sz // 2: indx[0] + px_sz * 3 // 2,
                     indx[1] + px_sz // 2: indx[1] + px_sz * 3 // 2]
    
    if flip_ax < 2:
      ret_img[i] = np.flip(ret_img[i], flip_ax)
  return ret_img


def plot_image(p_type, rdr_df, img_dir, jul):
  '''
    plot AACP/Non-AACP image from VIS data and save to AACP/Non-AACP file respectively
  #Arg:
    p_type: input type, aacp or non_aacp
    rdr_df: input subset of pandas csv file
    img_dir: input path of .nc file
    jul_dt: input julian day of data
  #Return:
    none
  #Output:
    store AACP/Non-AACP image to 'temp' folder
  '''
  
  unique_dt = rdr_df['date_time'].unique()

  for dt in unique_dt:
    file_ = rdr_df.loc[rdr_df['date_time'] == dt, 'VIS_Data_File'].values[0]
    temp_img = nc_img_data(file_, img_dir)
    imageio.imwrite('../../../' + p_type + '/' + jul + '/' + file_ + \
            try_catch_datetime(dt).strftime('%Y%m%d_%H:%M:%S') + '.png', temp_img)


def sunset_subset(rdr_df, sunrise, sunset, std_off = 10):
  '''
    take a subset from rdr_df pandas csv file by remove data before sunraising and after sunset
    (set 10 minutes after sunrasing; and 10 mintutes before sunset to get intresting data image)
  #Arg:
    rdr_df: input pandas csv file; (must call after rdr_df_subset)
    sunrise: input date and time, string or datatime type
    sunset: input date and time, string or datatime type
    ** Note: for sunrise and sunset: a string if right format of '%Y-%m-%d-%H:%M:%S' or datatime type;
         otherwise call try_catch_datetime to fix the correct format
    std_off: input float or integer in minute(s), default = 10 minutes, allow the sunrise and sunset shift
  #Return:
    rdr_df: pandas csv file, subset of date time that remove before sunrise and after sunset data
  '''
  
  # turn date_time column into datetime string
  rdr_df['date_time'] = pd.to_datetime(rdr_df['date_time']).dt.strftime('%Y-%m-%d-%H:%M:%S')
  
  #check input sunrise data type
  if hasattr(sunrise, 'time'):
    raise_cor = sunrise + timedelta(minutes = std_off)
  else:
    raise_cor = parse(sunrise) + timedelta(minutes = std_off)
    
  if hasattr(sunset, 'time'):
    sunset_cor = sunset - timedelta(minutes = std_off)
  else:
    sunset_cor = parse(sunset) - timedelta(minutes = std_off)
    
  rdr_df = rdr_df[(rdr_df['date_time'] > raise_cor.strftime('%Y-%m-%d-%H:%M:%S')) &
          (rdr_df['date_time'] < sunset_cor.strftime('%Y-%m-%d-%H:%M:%S'))]
  return rdr_df
    

def try_catch_datetime(sun_date):
  '''
    try to catch the format of the input date time and convert it to datetime.datetime data type
  #Arg:
    sun_date: input a string of datetime
  #Return:
    sun_date: a corrected format of datetime datatype, i.e. '%Y/%m/%d %H:%M:%S' means (yyyy/mm/dd HH:MM:SS)
  '''
  if hasattr(sun_date, 'time'):
    return sun_date
  else:
    try:
      return parse(sun_date)
    except ValueError:
      raise ValueError
    

def aacp_img_resample(img_data_aacp):
  '''
  Takes AACP images and resamples them by rotating and flipping them.
  Args:
    img_data_aacp: AACP image data output from the ir_process_data_random() function call.
  Return:
    img_data_aacp_aug: AACP image data which has been rotated in 90 deg increments and also
      flipped, x8 images as the original AACP image data.
  '''
  img_data_aacp_aug = np.zeros([img_data_aacp.shape[0] * 8, img_data_aacp.shape[1], img_data_aacp.shape[2]])
  tgt_aacp_aug = np.ones([img_data_aacp_aug.shape[0]])
  for n in range(img_data_aacp.shape[0]):
    img_data_aacp_aug[n*8, :, :] = img_data_aacp[n, :, :]
    img_data_aacp_aug[n*8+1, :, :] = np.flip(img_data_aacp[n,:,:], 1)
    img_data_aacp_aug[n*8+2, :, :] = np.rot90(img_data_aacp[n,:,:], 1)
    img_data_aacp_aug[n*8+3, :, :] = np.flip(np.rot90(img_data_aacp[n,:,:], 1), 1)
    img_data_aacp_aug[n*8+4, :, :] = np.rot90(img_data_aacp[n,:,:], 2)
    img_data_aacp_aug[n*8+5, :, :] = np.flip(np.rot90(img_data_aacp[n,:,:], 2), 1)
    img_data_aacp_aug[n*8+6, :, :] = np.rot90(img_data_aacp[n,:,:], 3)
    img_data_aacp_aug[n*8+7, :, :] = np.flip(np.rot90(img_data_aacp[n,:,:], 3), 1)
  return img_data_aacp_aug, tgt_aacp_aug    


def normalize_and_resize_ir(rdr_df, full_pix_sz, file_, img_dir, px_sz, half_px_sz):
  '''
  Performs pixel normalization for IR data.  Also resizes image and pads edges in
  cases where AACPs are on the border of the image.
  Args:
    rdr_df: dataframe afer subsetting containing only AACP occurences
    file_: nc file with IR imagery
    img_dir: directory with nc file
    px_sz: dimension of pixel size subsetted around AACP / storm occurence
    half_px_sz: half of px_sz
  Returns:
    temp_img: resize and rebuffered image
    plumes: subset of rdr_df containing tuple indices of storm locations for the specified input file_
    aacp_matrix: A matrix which will hold boolean values for areas around AACPs (used for selecting non-AACP
      areas outside of a standoff area from AACPs
  '''
  temp_img = nc_img_data(file_, img_dir)
  # TODO HANDLE THESE: 'IR_2017179_201627.nc' and 'IR_2017180_012227.nc' contain 0 Kelvin
  # Turn the thresh kelvin greater than 360.65 to 360.65 and less than 105.65 to 105.65
  temp_img[temp_img > 360.65] = 360.65
  temp_img[temp_img < 105.65] = 105.65
  # IR image data is 500 * 500, resize to 2000 * 2000
  temp_img = resize(temp_img, (500, 500), mode = 'reflect', preserve_range = True)
  # Padding the value to the edge to take care of plume right in the edge, use reflect_101 mode
  temp_img = cv2.copyMakeBorder(temp_img, half_px_sz, half_px_sz,
                                half_px_sz, half_px_sz, cv2.BORDER_REFLECT_101)  
  plumes = rdr_df.loc[rdr_df['IR_Data_File'] == file_, 'IR_Min_Index'].values
  aacp_matrix = np.zeros([full_pix_sz + px_sz, full_pix_sz + px_sz]) # A matrix which will hold boolean values for areas around AACPs
  return temp_img, plumes, aacp_matrix
  

def ir_process_data_random(rdr_df, full_pix_sz, px_sz, standoff, img_dir):
  '''
  Sub sample multiple images to find an equal number of non-plumes as plumes
  Args:
    rdr_df: panda dataframe with married rdr and satellite data
    px_sz: dimension for both the x and the y of the desired image size around plumes / non-plumes,
      must be divisible by 2
    standoff: the minimum pixel distance in the x and y direction that a non-AACP must be from an 
      AACP (forces the randomly sample non-AACPs to be a certain distance from an actual AACP
    file_type: what type of img data we are using, can be IR or VIS
    img_dir: the directory in which the images are being stored
  return: 
    img_data_aacp: images of just AACPs
    tgt_aacp: vector with labels of 1 (AACP)
    img_data_no_aacp: images of randomly sampled non-AACPs
    tgt_no_aacp: vector with labels of  0 (non-AACP)
  '''
  # Get half size of the input pixel size and round it to int if not power of 2
  half_px_sz = int(px_sz / 2)
  img_data_aacp = np.zeros([rdr_df.shape[0], px_sz, px_sz]) # Pre-allocated images.
  tgt_aacp = np.ones([rdr_df.shape[0]]) # Pre-allocated tgt.
  img_data_no_aacp = np.zeros([rdr_df.shape[0], px_sz, px_sz]) # Pre-allocated AACP image
  tgt_no_aacp = np.zeros([rdr_df.shape[0]]) # Pre-allocated non-AACP tgt.
  unique_file = rdr_df['IR_Data_File'].unique() # Get all unique file name.
  print(str(unique_file.shape[0]) + ' unique files.')  
  aacp_idx = 0
  non_aacp_idx = 0
  for cnt, file_ in enumerate(unique_file):
    print('Instance Count: ' + str(cnt))
    temp_img, plumes, aacp_matrix = normalize_and_resize_ir(rdr_df, full_pix_sz, file_, img_dir, px_sz, half_px_sz)
    # Retrieve all sub images for AACPs.
    for indx in plumes:
      # Set AACP exclusion matrix values around AACP to one.
      aacp_matrix[max(0, indx[0]+half_px_sz-standoff):min(full_pix_sz+px_sz, indx[0]+half_px_sz+standoff),
                  max(0, indx[1]+half_px_sz-standoff):min(full_pix_sz+px_sz, indx[1]+half_px_sz+standoff)] = 1.0
      img_data_aacp[aacp_idx, :, :] = temp_img[indx[0]:indx[0]+px_sz, indx[1]:indx[1]+px_sz]
      aacp_idx += 1
    # Randomly sample within the same image for non-AACPs. 
    for indx in plumes:
      guard_ind = non_aacp_idx
      while guard_ind == non_aacp_idx:
        # Uniform randomly generate NONE plume sample index
        rand_ind = tuple(np.random.uniform(px_sz,full_pix_sz,[1,2]).astype(int)[0])
        # Check if the new randomly sampled image is within the bounds of an AACP and the user-specified standoff.
        if not aacp_matrix[rand_ind]:
          img_data_no_aacp[non_aacp_idx, :, :] = temp_img[rand_ind[0]-half_px_sz: rand_ind[0]+half_px_sz,
                                                          rand_ind[1]-half_px_sz: rand_ind[1]+half_px_sz]
          non_aacp_idx += 1
  return img_data_aacp, tgt_aacp, img_data_no_aacp, tgt_no_aacp


def get_non_aacp_storms(aacp_rdr_df, non_aacp_rdr_df, full_pix_sz, px_sz, standoff, img_dir):
  '''
  Sub sample multiple images to find an equal number of non-plumes as plumes
  Args:
    aacp_rdr_df: panda dataframe with married rdr and satellite data for AACPs
    non_aacp_rdr_df: panda dataframe with married rdr and satellite data for non-AACPs
    px_sz: dimension for both the x and the y of the desired image size around plumes / non-plumes,
      must be divisible by 2
    standoff: the minimum pixel distance in the x and y direction that a non-AACP must be from an 
      AACP (forces the randomly sample non-AACPs to be a certain distance from an actual AACP
    img_dir: the directory in which the images are being stored
  Return: 
    image_data: images of AACPs and randomly sampled non-AACPs
    tgt: vector with labels of 1 (AACP) and 0 (non-AACP) which correspond to the images in image_data
  '''
  # Get half size of the input pixel size and round it to int if not power of 2
  half_px_sz = int(px_sz / 2)
  img_data_no_aacp = np.zeros([non_aacp_rdr_df.shape[0], px_sz, px_sz]) # Pre-allocated image
  tgt_no_aacp = np.zeros([non_aacp_rdr_df.shape[0]]) # Pre-allocated non-AACP tgt.
  unique_file = non_aacp_rdr_df['IR_Data_File'].unique() # Get all unique file name.
  print(str(unique_file.shape[0]) + ' unique files.')
  non_aacp_idx = 0 # Index which we will use to subset the returned matrix for img_data_no_aacp.
  for cnt, file_ in enumerate(unique_file):
    print('Instance Count: ' + str(cnt))
    temp_img, plumes, aacp_matrix = normalize_and_resize_ir(aacp_rdr_df, full_pix_sz, file_, img_dir, px_sz, half_px_sz)
    non_plumes = non_aacp_rdr_df.loc[rdr_df['IR_Data_File'] == file_, 'IR_Min_Index'].values
    for indx in plumes:
      # Set AACP exclusion matrix values around AACP to one.
      aacp_matrix[max(0, indx[0]+half_px_sz-standoff):min(full_pix_sz+px_sz, indx[0]+half_px_sz+standoff),
                  max(0, indx[1]+half_px_sz-standoff):min(full_pix_sz+px_sz, indx[1]+half_px_sz+standoff)] = 1.0
    for indx in non_plumes:
      if not aacp_matrix[indx[0]+half_px_sz,indx[1]+half_px_sz]:
        img_data_no_aacp[non_aacp_idx, :, :] = temp_img[indx[0]: indx[0]+px_sz,
                                                        indx[1]: indx[1]+px_sz]
        non_aacp_idx += 1
  return img_data_no_aacp[:non_aacp_idx,:,:], tgt_no_aacp[:non_aacp_idx]


def rdr_df_subset_combined(rdr_df, bad_list, px_sz):
  '''
  Returns a radar data dataframe with both AACP and non-AACP storms.
  Args:
    rdr_df: dataframe holding the radar info which has already been combined with the sat information
    bad_list: a list of .NC files with known bad information (i.e. 0 deg K IR values)
    px_sz: the dimension of the picture used for subsampling
  Returns:
    rdr_df: Updated radar_df with AACP and non-AACP storms, storms with bad files have been removed
    img_data_aacp: image data matrix used as input to transfer learning modelsa
    img_tgts: numpy array with targets (plume: 1, 0) for the img_data_aacp matrix
  '''
  half_px_sz = int(px_sz / 2)
  rdr_df = rdr_df[(rdr_df['IR_Data_File'] != 'NO_VALID_FILE') & (rdr_df['IR_dist_delta'] < 5) & 
                  #(rdr_df['VIS_Data_File'] != 'NO_VALID_FILE') & (rdr_df['VIS_dist_delta'] < 5) & 
                  (~rdr_df['IR_Data_File'].isin(bad_list))]
  non_aacp = rdr_df.groupby('Storm Number', as_index = False).agg('sum')
  non_aacp = non_aacp[non_aacp['Plume'] == 0.0]
  rdr_df = rdr_df.loc[rdr_df['Storm Number'].isin(non_aacp['Storm Number']) | rdr_df['Plume'] == 1.0]
  img_data_aacp = np.zeros([rdr_df.shape[0], px_sz, px_sz]) # Pre-allocated images.
  img_tgts = np.asarray(rdr_df['Plume'])
  idx = 0
  for _, row in rdr_df.iterrows():
    temp_img = nc_img_data(row['IR_Data_File'], row['IR_dir'])
    # TODO HANDLE THESE: 'IR_2017179_201627.nc' and 'IR_2017180_012227.nc' contain 0 Kelvin
    # Turn the thresh kelvin greater than 360.65 to 360.65 and less than 105.65 to 105.65
    temp_img[temp_img > 360.65] = 360.65
    temp_img[temp_img < 105.65] = 105.65
    # IR image data is 500 * 500, resize to 2000 * 2000
    temp_img = resize(temp_img, (500, 500), mode = 'reflect', preserve_range = True)
    # Padding the value to the edge to take care of plume right in the edge, use reflect_101 mode
    temp_img = cv2.copyMakeBorder(temp_img, half_px_sz, half_px_sz,
                                  half_px_sz, half_px_sz, cv2.BORDER_REFLECT_101)
    img_data_aacp[idx,:,:] = temp_img[row['IR_Min_Index'][0]:row['IR_Min_Index'][0]+px_sz, row['IR_Min_Index'][1]:row['IR_Min_Index'][1]+px_sz]
    idx += 1
  return rdr_df, img_data_aacp, img_tgts


if __name__ == '__main__':
  dircts = ['../../../data/20170518/','../../../data/20170628/','../../../data/20170629/']
  jul_date = ['2017138', '2017179', '2017180']
  bad_list = ['IR_2017179_201627.nc', 'IR_2017180_012227.nc'] # List of files with known suspect pixel values.
  full_pix_sz = 500
  if not os.path.exists('../../../data/preprocessed_combo/'): # Create preprocessed data directory if it doesn't exist.
    os.makedirs('../../../data/preprocessed_combo/')
  # Get combined pnadas dataframe and numpy matrices for ML which will use radar data.
  for (dirct, jul_dt) in zip(dircts, jul_date):
    print('Building data for: ' + dirct + jul_dt)
    rdr_df = pd.read_csv(dirct + 'update_rdr_data.csv',
                         converters={'IR_Min_Index': literal_eval, 'VIS_Min_Index': literal_eval})
    rdr_df, img_data_aacp, img_tgts = rdr_df_subset_combined(rdr_df, bad_list, 26)
    rdr_df.to_csv('../../../data/preprocessed_combo/' + jul_dt + '.csv', index=False)
    np.save('../../../data/preprocessed_combo/IR_' + jul_dt + '_img.npy', img_data_aacp)
    np.save('../../../data/preprocessed_combo/IR_' + jul_dt + '_tgt.npy', img_tgts)
  asfda
  if not os.path.exists('../../../data/preprocessed/'): # Create preprocessed data directory if it doesn't exist.
    os.makedirs('../../../data/preprocessed/')
  # Get separated AACP and non-AACP numpy arrays.
  for (dirct, jul_dt) in zip(dircts, jul_date):
    print('Building data for: ' + dirct + jul_dt)
    rdr_df = pd.read_csv(dirct + 'update_rdr_data.csv',
                         converters={'IR_Min_Index': literal_eval, 'VIS_Min_Index': literal_eval})
    aacp_rdr_df, non_aacp_rdr_df = rdr_df_subset(rdr_df, bad_list)
    img_data_aacp, tgt_aacp, img_data_no_aacp, tgt_no_aacp = ir_process_data_random(aacp_rdr_df, full_pix_sz, 26, 13, dirct + jul_dt + '/IRNCDF/')
    np.save('../../../data/preprocessed/no_resize_all_IR_' + jul_dt + '_aacp_img.npy', img_data_aacp)
    np.save('../../../data/preprocessed/no_resize_all_IR_' + jul_dt + '_aacp_tgt.npy', tgt_aacp)
    np.save('../../../data/preprocessed/no_resize_all_IR_' + jul_dt + '_no-aacp_rand_img.npy', img_data_no_aacp)
    np.save('../../../data/preprocessed/no_resize_all_IR_' + jul_dt + '_no-aacp_rand_tgt.npy', tgt_no_aacp)
    img_data_aacp_aug, tgt_aacp_aug = aacp_img_resample(img_data_aacp)
    np.save('../../../data/preprocessed/no_resize_all_IR_' + jul_dt + '_aacp_aug_img.npy', img_data_aacp_aug)
    np.save('../../../data/preprocessed/no_resize_all_IR_' + jul_dt + '_aacp_aug_tgt.npy', tgt_aacp_aug)
    img_data_no_aacp_storm, tgt_no_aacp_storm = get_non_aacp_storms(aacp_rdr_df, non_aacp_rdr_df, full_pix_sz, 26, 13, dirct + jul_dt + '/IRNCDF/')
    np.save('../../../data/preprocessed/no_resize_all_IR_' + jul_dt + '_no_aacp_storm_img.npy', img_data_no_aacp_storm)
    np.save('../../../data/preprocessed/no_resize_all_IR_' + jul_dt + '_no_aacp_storm_tgt.npy', tgt_no_aacp_storm)
    
  
