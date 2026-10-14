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
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add, Dropout, SpatialDropout2D
from keras.models import Model, model_from_json
#from keras.optimizers import Adam
#from tensorflow.keras.optimizers import Adam
try:
    from tensorflow.keras.optimizers.legacy import Adam
except ImportError:
    from keras.optimizers import Adam
try:
    from keras.layers.advanced_activations import ELU, LeakyReLU
except:
    from tensorflow.keras.layers import ELU, LeakyReLU

#from keras.layers import ELU, LeakyReLU

#from keras.utils.vis_utils import plot_model

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


def MultiResUnet_new_UPsampling_v2(height, width, n_channels, pretrained_weights = None, loss_weights = None, adam_optimizer = 1e-3):
    '''
    MultiResUNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet model
    '''

    if height <= 8 and width <= 8:
      psize = 1
    else:
      psize = 2  
    
    inputs = Input((height, width, n_channels))
    mresblock1 = MultiResBlock(32, inputs)
    mresblock1 = SpatialDropout2D(0.15)(mresblock1)
    pool1 = MaxPooling2D(pool_size=(psize, psize))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32*2, pool1)
    mresblock2 = SpatialDropout2D(0.15)(mresblock2)
    pool2 = MaxPooling2D(pool_size=(psize, psize))(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    mresblock3 = SpatialDropout2D(0.15)(mresblock3)
    pool3 = MaxPooling2D(pool_size=(psize, psize))(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)
    
    mresblock4 = MultiResBlock(32*8, pool3)
    mresblock4 = SpatialDropout2D(0.15)(mresblock4)
    pool4 = MaxPooling2D(pool_size=(psize, psize))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)
   
    mresblock5 = MultiResBlock(32*16, pool4)
    mresblock5 = SpatialDropout2D(0.2)(mresblock5)
    upsample5 = UpSampling2D(size=(psize, psize))(mresblock5)
    upsample5_conv = Conv2D(32*8, (2, 2), activation='relu', padding='same')(upsample5)
    up6 = concatenate([upsample5_conv, mresblock4], axis=3)

    mresblock6 = MultiResBlock(32*8, up6)
    upsample6 = UpSampling2D(size=(psize, psize))(mresblock6)
    upsample6_conv = Conv2D(32*4, (2, 2), activation='relu', padding='same')(upsample6)
    up7 = concatenate([upsample6_conv, mresblock3], axis=3)

    mresblock7 = MultiResBlock(32*4, up7)
    upsample7 = UpSampling2D(size=(psize, psize))(mresblock7)
    upsample7_conv = Conv2D(32*2, (2, 2), activation='relu', padding='same')(upsample7)
    up8 = concatenate([upsample7_conv, mresblock2], axis=3)

    mresblock8 = MultiResBlock(32*2, up8)
    upsample8 = UpSampling2D(size=(psize, psize))(mresblock8)
    upsample8_conv = Conv2D(32, (2, 2), activation='relu', padding='same')(upsample8)
    up9 = concatenate([upsample8_conv, mresblock1], axis=3)

    mresblock9 = MultiResBlock(32, up9)
    
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

    model = MultiResUnet_new_UPsampling_v2(128, 128,3)
    print(model.summary())



if __name__ == '__main__':
    main()