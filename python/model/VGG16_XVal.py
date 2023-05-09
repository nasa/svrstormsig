'''
Created on Sep 25, 2018

@author: caliles
To be used with VGG16_Model_2.
'''

import numpy as np
from VGG16_Model_2 import VGG16_model
import time
from utility import kelvin2intensity, make_up_rgb, resize_along_axes, just_metrics
from sklearn.utils.class_weight import compute_class_weight
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras.initializers import Constant, he_uniform


def main():
  start_time = time.time() 
  lyr_name = 'block2_conv1'
  img_type = 'IR'
  resize_dim = 48
  days = ['2017095', '2017136', '2017138', '2017179', '2017180']
  for idx, n in enumerate(days):
    train_x = np.concatenate((np.load('../../../data/preprocessed_combo/all_IR_' + days[(idx-1)%5] + '_img.npy'),
                              np.load('../../../data/preprocessed_combo/all_IR_' + days[(idx-3)%5] + '_img.npy'),
                              np.load('../../../data/preprocessed_combo/all_IR_' + days[(idx-4)%5] + '_img.npy')))
    train_y = np.concatenate((np.load('../../../data/preprocessed_combo/all_IR_' + days[(idx-1)%5] + '_tgt.npy'),
                              np.load('../../../data/preprocessed_combo/all_IR_' + days[(idx-3)%5] + '_tgt.npy'),
                              np.load('../../../data/preprocessed_combo/all_IR_' + days[(idx-4)%5] + '_tgt.npy')))
    
    val_x = np.load('../../../data/preprocessed_combo/all_IR_' + days[(idx-2)%5] + '_img.npy')
    val_y = np.load('../../../data/preprocessed_combo/all_IR_' + days[(idx-2)%5] + '_tgt.npy')
    
    class_wts = compute_class_weight('balanced', np.unique(train_y), train_y)
    model = VGG16_model(train_x, train_y, val_x, val_y, class_wts, img_type, resize_dim, 'XVal_Fold_' + str(idx) + '_' + lyr_name + '_' + str(resize_dim) + '_img_VGG16_storm_diff_fix', '../../../model_weights/', lyr_name)
    
    train_probs = model.predict(kelvin2intensity(make_up_rgb(resize_along_axes(train_x, resize_dim))))
    np.save('../../../data/preprocessed_combo/' + 'XVal_Fold_' + str(idx) + '_' + lyr_name + '_training_probs.npy', train_probs)
    val_probs = model.predict(kelvin2intensity(make_up_rgb(resize_along_axes(val_x, resize_dim))))
    np.save('../../../data/preprocessed_combo/' + 'XVal_Fold_' + str(idx) + '_' + lyr_name + '_validation_probs.npy', val_probs)   
    
    test_x = np.load('../../../data/preprocessed_combo/all_IR_' + n + '_img.npy')
    test_y = np.load('../../../data/preprocessed_combo/all_IR_' + n + '_tgt.npy')
        
    test_probs = model.predict(kelvin2intensity(make_up_rgb(resize_along_axes(test_x, resize_dim))))
    np.save('../../../data/preprocessed_combo/' + 'XVal_Fold_' + str(idx) + '_' + lyr_name + '_test_probs.npy', test_probs)
    
    just_metrics(test_y, test_probs, 'XVal_Fold_' + str(idx) + '_' + lyr_name + '_VGG16', '../../../results/')
    print("Fold " + str(idx) + "--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == '__main__':
  main()
    
