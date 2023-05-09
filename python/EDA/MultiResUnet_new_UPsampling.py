# Author and history:
#     @article{ibtehaz2020multiresunet,
#       title={MultiResUNet: Rethinking the U-Net architecture for multimodal biomedical image segmentation},
#       author={Ibtehaz, Nabil and Rahman, M Sohel},
#       journal={Neural Networks},
#       volume={121},
#       pages={74--87},
#       year={2020},
#       publisher={Elsevier}
#     }
#

import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add, Dropout
from keras.models import Model, model_from_json
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
#from keras.layers import ELU, LeakyReLU

from keras.utils.vis_utils import plot_model

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x


def MultiResBlock(U, inp, alpha = 1.67):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    W = alpha * U

    shortcut = inp
    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')
#1/6
    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')
#1/3
    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')
#1/2
    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')
#     v1 = int(W*0.167)
#     if v1%3 == 0:
#       v1 = v1
#     elif v1%3 == 1:
#       v1 = v1-1
#     elif v1%3 == 2:
#       v1 = v1+1
#     else:
#       print('Not mathematically possible.')
#    
#     v2 = int(W*0.333)
#     if v2%3 == 0:
#       v2 = v2
#     elif v2%3 == 1:
#       v2 = v2-1
#     elif v2%3 == 2:
#       v2 = v2+1
#     else:
#       print('Not mathematically possible.')
#  
#     v3 = int(W*0.5)
#     if v3%3 == 0:
#       v3 = v3
#     elif v3%3 == 1:
#       v3 = v3-1
#     elif v3%3 == 2:
#       v3 = v3+1
#     else:
#       print('Not mathematically possible.')
#     print(v1)
#     print(v2)
#     print(v3)
#     shortcut = conv2d_bn(shortcut, v1 + v2 +
#                          v3, 1, 1, activation=None, padding='same')
# #1/6
#     conv3x3 = conv2d_bn(inp, v1, 3, 3,
#                         activation='relu', padding='same')
# #1/3
#     conv5x5 = conv2d_bn(conv3x3, v2, 3, 3,
#                         activation='relu', padding='same')
# #1/2
#     conv7x7 = conv2d_bn(conv5x5, v3, 3, 3,
#                         activation='relu', padding='same')

#     shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
#                          int(W*0.5) + int(W), 1, 1, activation=None, padding='same')
    

# #1/6
#     conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
#                         activation='relu', padding='same')
# #1/3
#     conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
#                         activation='relu', padding='same')
# #1/2
#     conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
#                         activation='relu', padding='same')

#     conv15x15 = conv2d_bn(conv7x7, int(W), 3, 3,
#                         activation='relu', padding='same')

#     out = concatenate([conv3x3, conv5x5, conv7x7, conv15x15], axis=3)
    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)
    return out


def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)
    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def MultiResUnet_new_UPsampling(height, width, n_channels, pretrained_weights = None, loss_weights = None, adam_optimizer = 1e-3):
    '''
    MultiResUNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet model
    '''


    inputs = Input((height, width, n_channels))
#     print()
#     print(inputs.shape)
#     print()
    mresblock1 = MultiResBlock(32, inputs)
#     print(mresblock1.shape)
 #   mresblock1 = Dropout(0.5)(mresblock1)                                                                                                                                   #Randomly sets input units to 0 with a frequency of rate of 0.5 at each step during training time, which helps prevent overfitting
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)
#     print(mresblock1.shape)
#     print()

    mresblock2 = MultiResBlock(32*2, pool1)
#     print(mresblock2.shape)
#    mresblock2 = Dropout(0.5)(mresblock2)                                                                                                                                   #Randomly sets input units to 0 with a frequency of rate of 0.5 at each step during training time, which helps prevent overfitting
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)
#     print(mresblock2.shape)
#     print()

    mresblock3 = MultiResBlock(32*4, pool2)
#     print(mresblock3.shape)
#    mresblock3 = Dropout(0.5)(mresblock3)                                                                                                                                   #Randomly sets input units to 0 with a frequency of rate of 0.5 at each step during training time, which helps prevent overfitting
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)
#     print(mresblock3.shape)
#     print()
    mresblock4 = MultiResBlock(32*8, pool3)
#    mresblock4 = Dropout(0.5)(mresblock4)                                                                                                                                   #Randomly sets input units to 0 with a frequency of rate of 0.5 at each step during training time, which helps prevent overfitting
#     print(mresblock4.shape)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)
#     print(mresblock4.shape)
#     print()
    mresblock5 = MultiResBlock(32*16, pool4)
#     print(mresblock5.shape)


#3:30pm 11/17
#     up6 = concatenate([tf.image.resize(mresblock5, [4,4], method='nearest'), mresblock4], axis=3)
#     print(up6.shape)
#     mresblock6 = MultiResBlock(32*8, up6)
#     print(mresblock6.shape)
#     print()
#     
#     up7 = concatenate([tf.image.resize(mresblock6, [8,8], method='nearest'), mresblock3], axis=3)
#     print(up7.shape)
#     mresblock7 = MultiResBlock(32*4, up7)
#     print(mresblock7.shape)
#     print()
# 
#     up8 = concatenate([tf.image.resize(mresblock7, [16,16], method='nearest'), mresblock2], axis=3)
#     print(up8.shape)
#     mresblock8 = MultiResBlock(32*2, up8)
#     print(mresblock8.shape)
#     print()
# 
#     up9 = concatenate([tf.image.resize(mresblock8, [32,32], method='nearest'), mresblock1], axis=3)
#     print(up9.shape)
#     mresblock9 = MultiResBlock(32, up9)
#     print(mresblock9.shape)
#     print()

#3:00pm 11/17 (only gridding issues in corners of domain; second run got rid of them. Max IoU = 0.37, 0.36)
    up6 = concatenate([UpSampling2D(size = (2,2))(mresblock5), mresblock4], axis=3)
#     print(up6.shape)
    mresblock6 = MultiResBlock(32*8, up6)
#     print(mresblock6.shape)
#     print()
    
    up7 = concatenate([UpSampling2D(size = (2,2))(mresblock6), mresblock3], axis=3)
#     print(up7.shape)
    mresblock7 = MultiResBlock(32*4, up7)
#     print(mresblock7.shape)
#     print()

    up8 = concatenate([UpSampling2D(size = (2,2))(mresblock7), mresblock2], axis=3)
#     print(up8.shape)
    mresblock8 = MultiResBlock(32*2, up8)
#     print(mresblock8.shape)
#     print()

    up9 = concatenate([UpSampling2D(size = (2,2))(mresblock8), mresblock1], axis=3)
#     print(up9.shape)
    mresblock9 = MultiResBlock(32, up9)
#     print(mresblock9.shape)
#     print()

    
    #Conv2D(32*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')tf.image.resize_with_pad(mresblock5, 4,4, method='nearest')





# #     up6 = concatenate([tf.image.resize(mresblock5, [4,4], method='nearest'), mresblock4], axis=3)
#     up6 = concatenate([tf.image.resize_with_pad(mresblock5, 4,4, method='nearest'), mresblock4], axis=3)
# #    mresblock6 = Conv2D(32*8, (2, 2), strides=(2, 2), padding='same', use_bias=False)(up6)
# #    up6 = concatenate([Conv2D(32*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(), mresblock4], axis=3)
#     print(up6.shape)
#     mresblock6 = MultiResBlock(32*8, up6)
#     print(mresblock6.shape)
#     print()
# 
# #     up7 = concatenate([tf.image.resize(mresblock6, [8,8], method='nearest'), mresblock3], axis=3)
#     up7 = concatenate([tf.image.resize_with_pad(mresblock6, 8,8, method='nearest'), mresblock3], axis=3)
# #    mresblock7 = Conv2D(32*4, (2, 2), strides=(2, 2), padding='same', use_bias=False)(up7)
# #    up7 = concatenate([Conv2D(32*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(mresblock6)), mresblock3], axis=3)
#     print(up7.shape)
#     mresblock7 = MultiResBlock(32*4, up7)
#     print(mresblock7.shape)
#     print()
# 
# #     up8 = concatenate([tf.image.resize(mresblock7, [16,16], method='nearest'), mresblock2], axis=3)
#     up8 = concatenate([tf.image.resize_with_pad(mresblock7, 16,16, method='nearest'), mresblock2], axis=3)
# #    mresblock8 = Conv2D(32*2, (2, 2), strides=(2, 2), padding='same', use_bias=False)(up8)
# #    up8 = concatenate([Conv2D(32*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(mresblock7)), mresblock2], axis=3)
#     print(up8.shape)
#     mresblock8 = MultiResBlock(32*2, up8)
#     print(mresblock8.shape)
#     print()
# 
# #     up9 = concatenate([tf.image.resize(mresblock8, [32,32], method='nearest'), mresblock1], axis=3)
#     up9 = concatenate([tf.image.resize_with_pad(mresblock8, 32,32, method='nearest'), mresblock1], axis=3)
# #    mresblock9 = Conv2D(32, (2, 2), strides=(2, 2), padding='same', use_bias=False)(up9)
# #    up9 = concatenate([Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(mresblock8)), mresblock1], axis=3)
#     print(up9.shape)
#     mresblock9 = MultiResBlock(32, up9)
#     print(mresblock9.shape)
#     print()
    
#     up6 = concatenate([Conv2DTranspose(
#         32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
#     print(up6.shape)
#     mresblock6 = MultiResBlock(32*8, up6)
#     print(mresblock6.shape)
#     print()
# 
#     up7 = concatenate([Conv2DTranspose(
#         32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
#     print(up7.shape)
#     mresblock7 = MultiResBlock(32*4, up7)
#     print(mresblock7.shape)
#     print()
# 
#     up8 = concatenate([Conv2DTranspose(
#         32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
#     print(up8.shape)
#     mresblock8 = MultiResBlock(32*2, up8)
#     print(mresblock8.shape)
#     print()
# 
#     up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
#     print(up9.shape)
#     mresblock9 = MultiResBlock(32, up9)
#     print(mresblock9.shape)
#     print()

#     up6 = concatenate([trans_conv2d_bn(mresblock5,
#         32*8, 2, 2, strides=(2, 2), padding='same'), mresblock4], axis=3)
#     print(up6.shape)
#     mresblock6 = MultiResBlock(32*8, up6)
#     print(mresblock6.shape)
#     print()
#     up7 = concatenate([trans_conv2d_bn(mresblock6,
#         32*4, 2, 2, strides=(2, 2), padding='same'), mresblock3], axis=3)
# 
#     print(up7.shape)
#     mresblock7 = MultiResBlock(32*4, up7)
#     print(mresblock7.shape)
#     print()
#   
#     up8 = concatenate([trans_conv2d_bn(mresblock7,
#         32*2, 2, 2, strides=(2, 2), padding='same'), mresblock2], axis=3)
#     print(up8.shape)
#     mresblock8 = MultiResBlock(32*2, up8)
#     print(mresblock8.shape)
#     print()
# 
#     up9 = concatenate([trans_conv2d_bn(mresblock8,
#         32, 2, 2, strides=(2, 2), padding='same'), mresblock1], axis=3)
#     print(up9.shape)
#     mresblock9 = MultiResBlock(32, up9)
#     print(mresblock9.shape)
#     print()
    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')
    
    model = Model(inputs=[inputs], outputs=[conv10])
    if loss_weights:
        #model.compile(optimizer = Adam(lr = 5e-5), loss = 'binary_crossentropy', metrics = ['accuracy'], loss_weights = loss_weights)
        model.compile(optimizer = Adam(learning_rate = adam_optimizer), loss = BinaryCrossentropy(sample_weight = loss_weights), metrics = ['accuracy'])      #Configure model for training
        #options = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True))
    else:
        model.compile(optimizer = Adam(learning_rate = adam_optimizer), loss = 'binary_crossentropy', metrics = ['accuracy'])                                 #Configure model for training
   
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
   


def main():

    # Define the model

    model = MultiResUnet(128, 128,3)
    print(model.summary())



if __name__ == '__main__':
    main()