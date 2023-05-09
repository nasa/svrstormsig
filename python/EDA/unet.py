#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#+
# Name:
#     unet.py
# Purpose:
#     This is a script to run the unet model from create tensor flow keras utilities
# Calling sequence:
#     from EDA.unet import unet
#     model = unet()
# Input:
#     None.
# Functions:
#     unet : Runs the unet model
# Output:
#     Returns the model
# Keywords:
#     pretrained_weights  : Optional keyword to specify pre-trained weights to load into the model. DEAFULT = None
#     input_size          : Optional 3-element tuple which is size of model input data. DEFAULT = (256, 256, 3)
#     loss_weights        : Optional keyword acts as a coefficient for the loss. If a scalar is provided, then the 
#                           loss is simply scaled by the given value. If loss_weights is a tensor of size [batch_size], 
#                           then the total loss for each sample of the batch is rescaled by the corresponding element in 
#                           the loss_weights vector. 
#                           DEFAULT = None.
#     adam_optimizer      : Initial learning rate of model AdamOptimizer.Adam optimization is a stochastic gradient descent method that
#                           is based on adaptive estimation of first-order and second-order moments.
#                           DEFAULT = 1e-3 (this is the recommended value. Previously was 5e-5)
# Author and history:
#     rbiswas2, caliles        ?   -10-24
#     John W. Cooney           2021-03-17.
#
#-

#### Environment Setup ###
# Package imports
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1
from keras.utils.vis_utils import plot_model, model_to_dot
from tensorflow.keras.optimizers import Adam

def unet(pretrained_weights = None, input_size = (256,256,3), loss_weights = None, adam_optimizer = 1e-3):
  inputs = Input(input_size)                                                                                                                                     #Instantiate Keras tensor.
  conv1  = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)                                                        #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 64)
  conv1  = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)                                                         #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 64)
  pool1  = MaxPooling2D(pool_size=(2, 2), name='max_pool_1')(conv1)                                                                                              #Calculate the maximum, or largest, value in each patch of each feature map (the maximum output within a rectangular neighborhood)
  conv2  = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)                                                        #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 128)
  conv2  = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)                                                        #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 128)
  pool2  = MaxPooling2D(pool_size=(2, 2), name='max_pool_2')(conv2)                                                                                              #Calculate the maximum, or largest, value in each patch of each feature map (the maximum output within a rectangular neighborhood)
  conv3  = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)                                                        #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 256)
  conv3  = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)                                                        #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 256)
  pool3  = MaxPooling2D(pool_size=(2, 2), name='max_pool_3')(conv3)                                                                                              #Calculate the maximum, or largest, value in each patch of each feature map (the maximum output within a rectangular neighborhood)
  conv4  = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)                                                        #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 512)
  conv4  = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)                                                        #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 512)
  drop4  = Dropout(0.5)(conv4)                                                                                                                                   #Randomly sets input units to 0 with a frequency of rate of 0.5 at each step during training time, which helps prevent overfitting
  pool4  = MaxPooling2D(pool_size=(2, 2), name='max_pool_4')(drop4)                                                                                              #Calculate the maximum, or largest, value in each patch of each feature map (the maximum output within a rectangular neighborhood)
 
  conv5  = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)                                                       #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 1024)
  conv5  = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)                                                       #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 1024)
  drop5  = Dropout(0.5)(conv5)                                                                                                                                   #Randomly sets input units to 0 with a frequency of rate of 0.5 at each step during training time, which helps prevent overfitting

  up6    = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))                            #Create a convolution kernel (size = 2) that is convolved with the layer input to produce a tensor of outputs (output feature map = 512)
  merge6 = concatenate([drop4,up6], axis = 3)                                                                                                                    #Join arrays along axis 3
  conv6  = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)                                                       #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 512)
  conv6  = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)                                                        #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 512)

  up7    = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))                            #Create a convolution kernel (size = 2) that is convolved with the layer input to produce a tensor of outputs (output feature map = 256)
  merge7 = concatenate([conv3,up7], axis = 3)                                                                                                                    #Join arrays along axis 3
  conv7  = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)                                                       #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 256)
  conv7  = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)                                                        #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 256)

  up8    = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))                            #Create a convolution kernel (size = 2) that is convolved with the layer input to produce a tensor of outputs (output feature map = 128)
  merge8 = concatenate([conv2,up8], axis = 3)                                                                                                                    #Join arrays along axis 3
  conv8  = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)                                                       #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 128)
  conv8  = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)                                                        #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 128)

  up9    = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))                             #Create a convolution kernel (size = 2) that is convolved with the layer input to produce a tensor of outputs (output feature map = 64)
  merge9 = concatenate([conv1,up9], axis = 3)                                                                                                                    #Join arrays along axis 3
  conv9  = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)                                                        #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 64)
  conv9  = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)                                                         #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 64)
  conv9  = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)                                                          #Create a convolution kernel (size = 3) that is convolved with the layer input to produce a tensor of outputs (output feature map = 2)
  conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)                                                                                                           #Fully connected layer

  #model = Model(inputs = inputs, output = conv10)
  model = Model(inputs, conv10)
  if loss_weights:
    #model.compile(optimizer = Adam(lr = 5e-5), loss = 'binary_crossentropy', metrics = ['accuracy'], loss_weights = loss_weights)
    model.compile(optimizer = Adam(learning_rate = adam_optimizer), loss = BinaryCrossentropy(sample_weight = loss_weights), metrics = ['accuracy'])             #Configure model for training
    #options = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True))
  else:
    model.compile(optimizer = Adam(learning_rate = adam_optimizer), loss = 'binary_crossentropy', metrics = ['accuracy'])                                        #Configure model for training

  
  #model.summary()
  if(pretrained_weights):
    model.load_weights(pretrained_weights)

  return(model)

def main():
  model = unet()
  plot_model(model, to_file='../../../unet_structure.png', show_shapes=True, show_layer_names=True)
  
if __name__ == '__main__':
  main()