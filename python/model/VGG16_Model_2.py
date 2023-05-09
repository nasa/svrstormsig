'''
Created on Sep 17, 2018

@author: caliles
'''

from keras.applications.vgg16 import VGG16
import numpy as np
import sys
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras.initializers import Constant, he_uniform
from sklearn.utils.class_weight import compute_class_weight
sys.path.append('../EDA/')
from utility import kelvin2intensity, plot_confusion_matrix, roc_auc_n_conf_matrix, make_up_rgb, batch_vis_norm, get_x_y, resize_along_axes, save_model_json, get_metrics
import time



def VGG16_model(train_x, train_y, val_x, val_y, class_wts, img_type, resize_dim, weight_name, weights_path, lyr_name):
  if img_type == 'IR':
    train_x = resize_along_axes(train_x, resize_dim)
    _, img_width, img_height = train_x.shape
    train_x = kelvin2intensity(make_up_rgb(train_x))
    val_x = kelvin2intensity(make_up_rgb(resize_along_axes(val_x, resize_dim)))
  else:
    _, img_width, img_height = train_x.shape
    train_x = make_up_rgb(batch_vis_norm(train_x))
    val_x = make_up_rgb(batch_vis_norm(val_x))
  print('Define model.')
  # define base model VGG16 with our customize input image size and output with binary
  base_model = VGG16(weights = 'imagenet',
                     include_top = False,
                     input_shape = (img_width, img_height, 3))
  
  # does not updated the weight for the first few layers, only updated on the block5
  set_trainable = False
  model = Sequential()
  for idx, layer in enumerate(base_model.layers):
    #print(idx)
    #print(layer.name)
    if layer.name == lyr_name:
      set_trainable = True
      #print(layer.name)
    if set_trainable:
      layer.trainable = True
    else:
      layer.trainable = False
    model.add(layer)
    
  model.add(Flatten())
  model.add(Dense(256, activation='relu', kernel_initializer=he_uniform(seed=127), bias_initializer=Constant(value=0.1)))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))
  
  # use RMSprop optimizers with 2e-5
  opt = optimizers.RMSprop(lr = 2e-5)
  # complie the model with RMSprop optimizer and binary cross entropy loss because we want output 0 or 1
  model.compile(loss = 'binary_crossentropy',
                optimizer = opt,
                metrics = ['acc'])
  
  #checkpoint
  # only save the best weights
  checkpoint = ModelCheckpoint(weight_name + '.h5',
                 monitor='val_loss',
                 verbose=1,
                 save_best_only=True,
                 mode='min')
  callback = [checkpoint]
  history = model.fit(train_x, train_y,
            batch_size=128,
            epochs=1,
            shuffle=True,
            validation_data=(val_x, val_y),
            callbacks=callback,
            class_weight=class_wts,
            verbose=1)
  model.load_weights(weight_name + '.h5')
  save_model_json(model, weight_name, path=weights_path)
  return model

def main():
  #with tf.device('/gpu:0'):
  lyr_name = 'block4_conv1'
  start_time = time.time() 
  img_type = 'IR'
  resize_dim = 128
  train_x = np.load('../../../data/preprocessed_combo/IR_2017138_img.npy')
  train_y = np.load('../../../data/preprocessed_combo/IR_2017138_tgt.npy')
  
  val_x = np.load('../../../data/preprocessed_combo/IR_2017179_img.npy')
  val_y = np.load('../../../data/preprocessed_combo/IR_2017179_tgt.npy')
  
  test_x = np.load('../../../data/preprocessed_combo/all_IR_2017180_img.npy')
  test_y = np.load('../../../data/preprocessed_combo/all_IR_2017180_tgt.npy')
  
  #train_x, train_y = get_x_y('../../../data/preprocessed/no_resize_all_', img_type, '2017138', 'aacp', ['no_aacp_storm'])
  #print(str(train_y.shape[0]) + ' total training instances.')
  #val_x, val_y = get_x_y('../../../data/preprocessed/no_resize_all_', img_type, '2017179', 'aacp', ['no_aacp_storm'])
  
  class_wts = compute_class_weight('balanced', np.unique(train_y), train_y)
  model = VGG16_model(train_x, train_y, val_x, val_y, class_wts, img_type, resize_dim, lyr_name + '_' + str(resize_dim) + '_img_VGG16_storm_diff_fix', '../../../model_weights/', lyr_name)
  print("--- %s seconds ---" % (time.time() - start_time))
  # Initially, evaluate model on AACPs vs randomly sampled non-AACPs.
  #test_x, test_y = get_x_y('../../../data/preprocessed/no_resize_all_', img_type, '2017180', 'aacp', ['no-aacp_rand'])
  get_metrics(model, resize_dim, test_x, test_y, lyr_name + '_VGG16_random_non-aacp_180', '../../../results/', img_type)
  
  # Next, evaluate model on AACPs vs non-AACP storms.
  #test_x, test_y = get_x_y('../../../data/preprocessed/no_resize_all_', img_type, '2017180', 'aacp', ['no_aacp_storm'])
  #get_metrics(model, resize_dim, test_x, test_y, 'VGG16_storm_non-aacp', '../../../results/', img_type)
  
  train_probs = model.predict(kelvin2intensity(make_up_rgb(resize_along_axes(train_x, resize_dim))))
  np.save('../../../data/preprocessed_combo/' + lyr_name + '_IR_2017138_probs.npy', train_probs)
  val_probs = model.predict(kelvin2intensity(make_up_rgb(resize_along_axes(val_x, resize_dim))))
  np.save('../../../data/preprocessed_combo/' + lyr_name + '_IR_2017179_probs.npy', val_probs)
  test_probs = model.predict(kelvin2intensity(make_up_rgb(resize_along_axes(test_x, resize_dim))))
  np.save('../../../data/preprocessed_combo/' + lyr_name + '_all_IR_2017180_probs.npy', test_probs)
  print("--- %s seconds ---" % (time.time() - start_time))
  
  test_x = np.load('../../../data/preprocessed_combo/all_IR_2017136_img.npy')
  test_y = np.load('../../../data/preprocessed_combo/all_IR_2017136_tgt.npy')
  get_metrics(model, resize_dim, test_x, test_y, lyr_name + '_VGG16_random_non-aacp_136', '../../../results/', img_type)
  test_probs = model.predict(kelvin2intensity(make_up_rgb(resize_along_axes(test_x, resize_dim))))
  np.save('../../../data/preprocessed_combo/' + lyr_name + '_all_IR_2017136_probs.npy', test_probs)
  
  test_x = np.load('../../../data/preprocessed_combo/all_IR_2017095_img.npy')
  test_y = np.load('../../../data/preprocessed_combo/all_IR_2017095_tgt.npy')
  get_metrics(model, resize_dim, test_x, test_y, lyr_name + '_VGG16_random_non-aacp_095', '../../../results/', img_type)
  test_probs = model.predict(kelvin2intensity(make_up_rgb(resize_along_axes(test_x, resize_dim))))
  np.save('../../../data/preprocessed_combo/' + lyr_name + '_all_IR_2017095_probs.npy', test_probs)

if __name__ == '__main__':
  main()
  
