'''
Created on Jul 25, 2018

@author: caliles and yhuang20
'''

import pandas as pd
import numpy as np
from pysolar.solar import  get_altitude
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from skimage.transform import resize
from sub_samp import nc_img_data
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ast import literal_eval
from matplotlib.patches import Circle
import sys
sys.path.append('../EDA')
from rdr_sat_data_utils import make_gif


def get_sza(lat, long, dtg):
  '''
  Gets the solar zenith angle (SZA) for a particular latitude and longitude at a specified date and time.
  Time MUST be in UTC!
  Args:
    lat: the latitude of the location
    long: the longitude of the location
    dtg: the datetime when SZA is desired
  Returns:
    solar zenith angle for the desired location and date/time
  '''
  return 90-get_altitude(lat, long, dtg)


def get_plumes(rdr_df, filename, file_type):
  '''
  Returns all storms observed in a radar file corresponding to a specific satellite image.
  Args:
    rdr_df: pandas dataframe holding radar storm observations
    filename: name of the satellite NETCDF file to which we want to find all observed radar storms 
    file_type: IR or VIS, the type of satellite file
  Returns:
    aacps: all AACP storms observed by radar at the time of the filename input
    non_aacps: all non-AACP storms observed by radar at the time of the filename input
  '''
  aacps = rdr_df.loc[((rdr_df[file_type +'_Data_File'] == filename) & (rdr_df['Plume'] == 1)), file_type + '_Min_Index'].values
  non_aacps = rdr_df.loc[((rdr_df[file_type +'_Data_File'] == filename) & (rdr_df['Plume'] == 0)), file_type + '_Min_Index'].values
  return aacps, non_aacps


def resize_along_axes(imgs, new_dim):
  '''
  Resizes all images in a numpy array to a new, user-specified dimension.
  Args:
    imgs: numpy array containing multiple images
    new_dim: dimension to which each image will be resized
  Returns:
    new_imgs: numpy array containing resized images
  '''
  new_imgs = np.zeros([imgs.shape[0], new_dim, new_dim])
  for n in range(imgs.shape[0]):
    new_imgs[n,:,:] = resize(imgs[n,:,:], (new_dim, new_dim), mode = 'reflect', preserve_range = True)
  return new_imgs


def make_up_rgb(img_data):
  '''
  Turns 1 channel image data to 3 channels with original channel copied to all 3 channels.
  Args:
    img_data: input image data, numpy array
  Return:
    img: 3 channel image that has original channel repeated over all 3 channels
  '''
  img = np.zeros(img_data.shape + (3,))
  for i in range(3):
    img[:,:,:,i] = img_data[:,:,:]
  return img


def kelvin2intensity(img):
  '''
  Takes a satellite infrared image in deg Kelvin and normalizes it to values between -1 and 1 by centering around 233.15 deg Kelvin and dividing by 127.5 deg.
  Sign is then reversed.
  Args:
    img: input satellite infrared Kelvin image value
  Returns:
    img: numpy array with normalized values between -1 to 1
  '''
  img -= 233.15
  img /= 127.5
  img = -img
  return img


def vis_norm(img):
  '''
  Normalizes an image to values between -1 and 1.
  An initial naive means of handling visible imagery.
  Args:
    img: 2d numpy matrix with pixel values
  Return:
    img: 2d numpy matrix with normalized pixel values
  '''
  img = 2*(img - np.min(img))/(np.max(img) - np.min(img)) - 1
  return img


def batch_vis_norm(imgs):
  '''
  A naive means of normalizing multiple VIS images based only on each images min and max value
  Args:
    imgs: numpy array holding image values
  Returns:
    imgs: numpy array with normalized image values
  '''
  for n in range(imgs.shape[0]):
    imgs[n,:,:] = vis_norm(imgs[n,:,:])
  np.nan_to_num(imgs, copy=False)
  return imgs


def roc_auc_n_conf_matrix(y_true, y_pred):
  '''
  Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
  Compute True positive rate and False positive rate
  Compute confusion matrix
  Args:
    y_true: true labels for a test dataset
    y_pred: model predictions for a dataset
  Returns:
    ROC_AUC: Receiver Operating Characteristic
    TPR: True positive rate
    FPR: False positive rate
    conf_matrix: confusion matrix
  '''
  ROC_AUC = roc_auc_score(y_true, y_pred)
  conf_matrix = confusion_matrix(y_true, y_pred)
  TN, FP, FN, TP = conf_matrix.ravel()
  TPR = TP / (TP + FN)
  FPR = FP / (FP + TN)
  return ROC_AUC, TPR, FPR, conf_matrix


def plot_confusion_matrix(cm, filename, path = ''):
  '''
  Buld and save a confusion matrix before showing it.
  Args:
    cm: input confusion matrix
    filename: string file name where the confusion matrix will be saved
    path: string directory name where the confusion matrix will be saved
  Returns:
    None
  '''
  labels = ['AACP', 'Non-AACP']
  
  fig, ax = plt.subplots(dpi = 200, figsize = (8,8))
  cm = np.rot90(cm, 2)
  img = ax.imshow(cm)
  
  TP, FN, FP, TN = cm.ravel()
  TPR = round(TP / (TP + FN) * 100, 3)
  FPR = round(FP / (FP + TN) * 100, 3)
  prec = round(TP / (TP + FP) * 100, 3)
  f_score = round(2*prec*TPR/(prec+TPR), 3)
  f_half = round(1.25*prec*TPR/(0.25*prec+TPR), 3)
  
  ax.set_xlabel('Predict\nF: ' + str(f_score) + '%, F0.5: ' + str(f_half) + '%, Prec: ' + str(prec) + '%; TPR: ' + str(TPR) + '%; FPR: ' + str(FPR) + '%')
  ax.set_ylabel('Actual')
  
  ax.set_xticks(np.arange(len(labels)))
  ax.set_yticks(np.arange(len(labels)))
  
  ax.set_xticklabels(labels)
  ax.set_yticklabels(labels)
  
  cbar = ax.figure.colorbar(img, ax = ax, fraction = 0.046, pad = 0.04)
  thresh = cm.max() // 2
  plt.setp(ax.get_xticklabels(), rotation = 0, ha = 'right', rotation_mode = 'anchor')
  for i in range(len(labels)):
    for j in range(len(labels)):
      text = ax.text(j,i, cm[i, j],
               ha = 'center',
               va = 'center',
               color = ('black' if cm[i, j] > thresh else 'white'))
  ax.set_title('Confusion Matrix')
  fig.tight_layout()
  fig.savefig(path + filename + '.png')
  #plt.show()
  

def just_metrics(test_y, y_pred, res_file_name, res_file_dir):
  '''
  Takes actual and predicted values and prints TPR and FPR as well as creates a confusion matrix.
  Args:
    test_y: correct labels for the data
    y_pred: values predicted by the model
    res_file_name: filename to which metrics will be saved
    res_file_dir: directory to which res_file_name will be saved 
  '''
  ROC_AUC, TPR, FPR, conf_matrix = roc_auc_n_conf_matrix(test_y, y_pred)
  print('Mean: ' + str(np.mean(y_pred)))
  print('ROC score: %s\nTPR: %s, FPR: %s' % (ROC_AUC, TPR, FPR))
  plot_confusion_matrix(conf_matrix, res_file_name, path=res_file_dir)


def get_metrics(model, resize_dim, test_x, test_y, res_file_name, res_file_dir, img_type):
  '''
  Takes a model with a given test holdout set and provides metrics based on model prediction.
  Args:
    model: model used to provide predictions
    resize_dim: x and y dimension to which images in test_x must be resized prior to predictions
    test_x: numpy array containing images from the test set
    test_y: numpy array contain AACP labels corresponding to each image in test_x
    res_file_name: file to which the confusion matrix will be saved
    res_file_dir: directory to which the confusion matrix will be saved
    img_type: IR or VIS , used to define different image preprocessing prior to prediction
  '''
  if img_type == 'IR':
    y_pred = model.predict(kelvin2intensity(make_up_rgb(resize_along_axes(test_x, resize_dim))))
  else:
    y_pred = model.predict(test_x)
  print(y_pred)
  print(test_y)
  just_metrics(test_y, y_pred, res_file_name, res_file_dir)
    
  
def save_model_json(model, filename, path = ''):
  '''
  Save model weights to HDF5 file and model structure to JSON file
  Args:
    model: input model
    filename: file name to which JSON structure and HDF5 weights will be saved (both have different extensions but the same name)
    path: directory to which the weights will be saved
  '''
  model_json = model.to_json()
  with open(path + filename + '.json', 'w') as json_file:
    json_file.write(model_json)
  model.save_weights(path + filename + '.h5')
  

def load_model_json(filename, path = ''):
  '''
  loads a model structure and weights from json and HDF5 file respectively
  Args:
    filename: file name from which JSON structure and HDF5 weights will be loaded
    path: directory from which weights will be loaded
  Returns:
    model: returns the trained model
  '''
  # load json and create model
  json_file = open(path + filename + '.json', 'r')
  model_json = json_file.read()
  json_file.close()
  model = model_from_json(model_json)
  # load weights into new model
  model.load_weights(path + filename + '.h5')
  return model


def get_x_y(dir_pre, img_type, jul_date, aacps, non_aacps):
  '''
  Fetches both input 2d array image values for a model and their corresponding AACP/non-AACP labels.
  Args:
    dir_pre: directory in which the x and y data is located
    img_type: IR or VIS, type of satellite imagery being uploaded
    jul_date: Julian date of the satellite imagery being uploaded
    aacps: numpy files containing AACP storms
    non_aacps: list of numpy files containing non-AACP storms 
  Returns:
    x: numpy array containing images of user-specified AACP and non-AACP storms
    y: numpy array containing labels corresponding to user-specified AACP and non-AACP storms imagery in x output value  
  '''
  x = np.load(dir_pre+img_type+'_' + jul_date + '_' + aacps + '_img.npy')
  y = np.load(dir_pre+img_type+'_' + jul_date + '_' + aacps + '_tgt.npy')
  for n in non_aacps:
    x = np.concatenate((x, np.load(dir_pre+img_type+'_' + jul_date + '_' + n + '_img.npy')),axis=0)
    y = np.concatenate((y, np.load(dir_pre+img_type+'_' + jul_date + '_' + n + '_tgt.npy')))
  return x, y


def plot_subplot(fig, ax_r, img, img_label, vmin_in=None, vmax_in=None):
  '''
  Plots a subplot of either IR, VIS, or probabilities.
  Args:
    fig: matplotlib figure on which the subplot is located.
    ax_r: subplot axes
    img: 2d numpy array with values which will be plotted
    img_label: Label for the subplot
    vmin_in: (optional) if specified, provides a new minimum value to which the subplot's colormap will be projected
    vmax_in: (optional) if specified, provides a new maximum value to which the subplot's colormap will be projected
  '''
  im = ax_r.imshow(img, vmin=vmin_in, vmax=vmax_in, cmap=plt.get_cmap('coolwarm'))
  divider = make_axes_locatable(ax_r)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar(im, ax=ax_r, cax=cax)
  ax_r.set_title(img_label)


def plot_pic_comp(ir_img, vis_img, prob_img, dtg, aacps, non_aacps):
  '''
  Plots IR, VIS and probabilities as well as AACP and non-AACP storm locations in a single picture.
  Args:
    ir_img: 2d numpy array containing IR satellite imagery
    vis_img: 2d numpy array containing VIS satellite imagery 
    prob_img: 2d numpy array containing probability prediction map  
    dtg: date and time corresponding to ir_img, vis_img, and prob_img 
    aacps: indeces of AACP locations in each of the three (IR, VIS, and probability) plotted arrays
    non_aacps: indeces of non-AACP locations in each of the three (IR, VIS, and probability) plotted arrays 
  Returns:
    fig: matplotlib figure instance on which IR, VIS, and probabilities are plotted with corresponding AACP and non-AACP storms
  '''
  fig, (ax0, ax1, ax2) = plt.subplots(ncols=3)
  for n in non_aacps:
    for m in [ax0, ax1, ax2]:
      circ0 = Circle([n[1]-13, n[0]-13],
                     10,
                     linewidth = 0.25,
                     fill = False,
                     edgecolor = 'Yellow')
      m.add_patch(circ0)
  for n in aacps:
    for m in [ax0, ax1, ax2]:
      circ0 = Circle([n[1]-13, n[0]-13],
                     10,
                     linewidth = 0.5,
                     fill = False,
                     edgecolor = 'Black')
      m.add_patch(circ0)
  plot_subplot(fig, ax0, ir_img, 'IR Data')
  plot_subplot(fig, ax1, prob_img, 'Prob Predictions', vmin_in=0.0, vmax_in=1.0)
  plot_subplot(fig, ax2, vis_img, 'VIS Data')
  
  fig.suptitle(dtg, fontsize='large')
  fig.tight_layout()
  fig.subplots_adjust(top=1.5)
  #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
  #plt.show()
  return fig


if __name__ == '__main__':
  files = []
  dtgs = ['2017136_220656','2017136_220756','2017136_220856','2017136_220956','2017136_221056','2017136_221156']
  # Plot multiple plots based off of precomputed probabilities from get_probs.py
  for dtg in dtgs:
    print('Producing results for: ' + dtg)
    dtg_sub_path = '20170516/'
    dtg_path = dtg_sub_path + '2017136/'
    ir_dtg_path = dtg_path + 'IRNCDF/'
    vis_dtg_path = dtg_path + 'VISNCDF/'
    rdr_df = pd.read_csv('../../../data/' + dtg_sub_path + 'update_rdr_data.csv', 
                         converters={'IR_Min_Index': literal_eval,
                                     'VIS_Min_Index': literal_eval})
    aacps, non_aacps = get_plumes(rdr_df, 'IR_' + dtg + '.nc', 'IR')
    prob_map = np.load('../../../prob_maps/IR_' + dtg + '_probas.npy')
    ir_img = nc_img_data('IR_' + dtg + '.nc','../../../data/' + ir_dtg_path)
    ir_resize_img = resize(ir_img, (500, 500), mode = 'reflect', preserve_range = True)
    vis_img = nc_img_data('VIS_' + dtg + '.nc','../../../data/' + vis_dtg_path)
    vis_resize_img = resize(vis_img, (500, 500), mode = 'reflect', preserve_range = True)
    fig = plot_pic_comp(ir_resize_img[13:500-26, 13:500-26], vis_resize_img[13:500-26, 13:500-26], prob_map, dtg, aacps, non_aacps)
    fig.savefig('../../../plots/' + dtg + '.png', bbox_inches='tight', dpi=250)
    plt.close()
    files.append('../../../plots/' + dtg + '.png')
  # Now combine all files into a single GIF.
  make_gif(files, '../../../plots/probs.gif')
  
  # Simple code for testing the confusion matrix plotting.
  #y_true = [1,0,1,0,1,0]
  #y_pred = [1,1,1,0,1,0]
  #cm = confusion_matrix(y_true, y_pred)
  #plot_confusion_matrix(cm, 'test', path='../../../results/')
  
