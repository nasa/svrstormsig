'''
Created on Sep 4, 2018

@author: caliles
'''

from keras.applications.vgg16 import VGG16
import numpy as np
import sys
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from sklearn.utils.class_weight import compute_class_weight
sys.path.append('../EDA/')
from utility import kelvin2intensity, make_up_rgb, get_x_y, resize_along_axes, get_metrics, load_model_json


def VGG16_predict(x, model_file, model_path, img_type, resize_dim):
  if img_type == 'IR':
    x = resize_along_axes(x, resize_dim)
    _, img_width, img_height = x.shape
    x = kelvin2intensity(make_up_rgb(x))
  else:
    _, img_width, img_height = x.shape
    x = make_up_rgb(batch_vis_norm(x))
  print('Define model.')
  
  model = load_model_json(model_file, model_path)
  probs = model.predict(x)  
  return probs

def main():
  img_type = 'IR'
  resize_dim = 128
  
  model_file = 'new_128_img_VGG16_storm_diff_fix'
  model_path = '../../../model_weights/'
  x = np.load('../../../data/preprocessed_combo/all_IR_2017136_img.npy')
  probs = VGG16_predict(x, model_file, model_path, img_type, resize_dim)
  np.save('../../../data/preprocessed_combo/new_all_IR_2017136_probs.npy', probs)
  
  x = np.load('../../../data/preprocessed_combo/all_IR_2017095_img.npy')
  probs = VGG16_predict(x, model_file, model_path, img_type, resize_dim)
  np.save('../../../data/preprocessed_combo/new_all_IR_2017095_probs.npy', probs)
  
if __name__ == '__main__':
  main()
  
