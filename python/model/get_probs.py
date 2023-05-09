'''
Created on Jul 6, 2018

@author: caliles
'''
from keras.models import model_from_json
import numpy as np
from scipy.io import netcdf
from skimage.transform import resize
import time
from sub_samp import nc_img_data
from utility import load_model_json, kelvin2intensity, plot_pic_comp, resize_along_axes, make_up_rgb


def get_sample_pred(img_data, row, model, threshld, img_sz, px_sz, new_dim):  
  '''
  Provides probability predictions for a row of sattelite imagery.
  Args:
    img_data: 2d numpy array containing satellite image over which we want to produce a map of predicted probabilities
    row: row of img_data (minus half sub-image size) on which probabilities are being predicted
    model: the model used for prediction
    threshld: A threshold value which delineates if a pixel will be evaluated by the model (pixel's with a value greater than thrshld will not be evaluated
      and marked as having a probability of zero
    img_sz: dimensions of original satellite image
    px_sz: dimensions of sub-images which are fed into the model (prior to being resized)
    new_dim: new dimension to which sub-images must be projected prior to prediction
  Returns:
    probas: 1d numpy array with model prediction probabilities for input row 
  '''
  # First check to see if any values are below threshold.  If they are, we will evaluate them.
  # Otherwise, we will assign a value of 0 for probability
  t0 = time.time()
  half_px_sz = int(px_sz / 2)
  probas = np.zeros([img_sz-px_sz])
  search_idx = []
  for n in range(probas.shape[0]):
    if img_data[row+half_px_sz, n+half_px_sz] <= threshld:
      probas[n] = 1.0
      search_idx.append(n)
  temp = np.zeros([len(search_idx),new_dim,new_dim,3])
  for i in range(len(search_idx)):
    temp[i,:,:,:] = kelvin2intensity(make_up_rgb(resize_along_axes(np.expand_dims(img_data[row:row+px_sz,i:i+px_sz], axis=0), new_dim)))
  print('Allocation: ' + str(time.time()-t0) + ' seconds.')
  t0 = time.time()
  y_pred = model.predict(temp)
  for n in range(len(search_idx)):
    probas[search_idx[n]] = y_pred[n]
  print('Prediction: ' + str(time.time()-t0) + ' seconds.') 
  return probas


if __name__ == '__main__':
  t_0 = time.time()
  dtg = '2017136_221156'
  dtg_path = '20170516/2017136/IRNCDF/'
  img_sz = 500
  px_sz = 26 # Pixels in each dimension in sub-image over which the model will be evaluated.
  new_dim = 128
  model = load_model_json('128_img_VGG16', path = '../../../model_weights/')
  prob_map = np.zeros([img_sz-px_sz,img_sz-px_sz])
  original_img = nc_img_data('IR_' + dtg + '.nc','../../../data/' + dtg_path)
  resize_img = resize(original_img, (img_sz, img_sz), mode = 'reflect', preserve_range = True)
  
  for count, i in enumerate(range(0,img_sz-26)):
    print("Iteration: " + str(count))
    prob_map[i,:] = get_sample_pred(resize_img, i, model, 350.0, img_sz, px_sz, new_dim)
  np.save('../../../prob_maps/IR_' + dtg + '_probas.npy', prob_map)
  #plot_pic_comp(resize_img[13:500-26, 13:500-26], resize_img[13:500-26, 13:500-26], prob_map)
  print('Total Run Time: ' + str(time.time()-t_0) + ' seconds.')
  
