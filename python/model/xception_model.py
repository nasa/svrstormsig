'''
Created on Aug 13, 2018

@author: caliles and yhuang20
'''

from keras.applications.xception import Xception
import numpy as np
import sys
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from sklearn.utils.class_weight import compute_class_weight
sys.path.append('../EDA/')
from utility import kelvin2intensity, plot_confusion_matrix, roc_auc_n_conf_matrix, make_up_rgb, batch_vis_norm, get_x_y, resize_along_axes, save_model_json, get_metrics


def Xception_model(train_x, train_y, val_x, val_y, class_wts, img_type, resize_dim, weight_name, weights_path):
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
  base_model = Xception(weights='imagenet',
                        include_top=False,
                        input_shape = (img_width,img_height,3))
  
  # freezing all layers
  #for layer in base_model.layers:
  #  layer.trainable = False
  #base_model = Xception(include_top=False,input_shape = (img_width,img_height,3))
  x = base_model.output
  x = Flatten()(x)
  x = Dense(500, activation = 'relu')(x)
  x = Dropout(0.5)(x)
  x = Dense(200, activation = 'relu')(x)
  x = Dropout(0.5)(x)
  x = Dense(100, activation = 'relu')(x)
  output = Dense(1, activation = 'sigmoid')(x)
  model = Model(base_model.input, output)
  
  # opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  opt = optimizers.RMSprop(lr=2e-5)
  model.compile(loss='binary_crossentropy',
          optimizer=opt,
          metrics=['acc'])
  
  #checkpoint
  # only save the best weights
  checkpoint = ModelCheckpoint(weight_name + '.h5',
                 monitor='val_loss',
                 verbose=1,
                 save_best_only=True,
                 mode='min')
  callback = [checkpoint]
  history = model.fit(train_x, train_y,
            batch_size=32,
            epochs=3,
            shuffle=True,
            validation_data=(val_x, val_y),
            callbacks=callback,
            class_weight=class_wts,
            verbose=1)
  model.load_weights(weight_name + '.h5')
  save_model_json(model, weight_name, path=weights_path)
  return model

if __name__ == '__main__':
  img_type = 'IR'
  resize_dim = 128
  train_x, train_y = get_x_y('../../../data/preprocessed/', img_type, '2017138', 'aacp', ['no-aacp_rand'])
  print(str(train_y.shape[0]) + ' total training instances.')
  val_x, val_y = get_x_y('../../../data/preprocessed/', img_type, '2017180', 'aacp', ['no-aacp_rand'])
  
  class_wts = compute_class_weight('balanced', np.unique(train_y), train_y)
  model = Xception_model(train_x, train_y, val_x, val_y, class_wts, img_type, resize_dim, str(resize_dim) + '_img_Xception', '../../../model_weights/')
  
  # Initially, evaluate model on AACPs vs randomly sampled non-AACPs.
  test_x, test_y = get_x_y('../../../data/preprocessed/', img_type, '2017179', 'aacp', ['no-aacp_rand'])
  get_metrics(model, resize_dim, test_x, test_y, 'Xception_random_non-aacp', '../../../results/', img_type)
  
  # Next, evaluate model on AACPs vs non-AACP storms.
  test_x, test_y = get_x_y('../../../data/preprocessed/', img_type, '2017179', 'aacp', ['no_aacp_storm'])
  get_metrics(model, resize_dim, test_x, test_y, 'Xception_storm_non-aacp', '../../../results/', img_type)
  
