'''
Created on Sep 4, 2018

@author: caliles
'''


import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from utility import roc_auc_n_conf_matrix, plot_confusion_matrix, just_metrics

def feature_builder(df_file, np_probs, skip_cols, target):
  '''
  TODO
  '''
  df = pd.read_csv(df_file) # Load training data.
  # Generate Features.
  df['VGG16_probs'] = np.load(np_probs)
  df['10dBZ-tropo'] = df['10 dBZ Echo Top Altitude'] * 1000.0 - df['NARR Tropopause Altitude']
  df['20dBZ-tropo'] = df['20 dBZ Echo Top Altitude'] * 1000.0 - df['NARR Tropopause Altitude']
  df['30dBZ-tropo'] = df['30 dBZ Echo Top Altitude'] * 1000.0 - df['NARR Tropopause Altitude']
  df['40dBZ-tropo'] = df['40 dBZ Echo Top Altitude'] * 1000.0 - df['NARR Tropopause Altitude']
  predictors = [x for x in df.columns if x not in skip_cols]
  X = df[predictors]
  y = df[target]
  return X, y

def main():
  np.random.seed(100)
  mdl_lyr = 'block5_conv1'
  data_dir = '../../../data/preprocessed_combo/'
  
  skip_cols = ['Storm Number', 'Time', 'Longitude', 'Latitude', 'Parallax Corrected Longitude', 'Parallax Corrected Latitude', 'Plume',
               'date_time', 'IR_Data_File', 'IR_dir', 'VIS_Data_File', 'VIS_dir', 'IR_Min_Index', 'IR_dist_delta', 'IR_closest_lat',
              'IR_closest_long', 'VIS_Min_Index', 'VIS_dist_delta', 'VIS_closest_lat', 'VIS_closest_long', 
              'NARR V-Component Wind at Tropopause', 'NARR U-Component Wind at Tropopause',
              'NARR Tropopause Altitude', 
              'NARR Tropopause Temperature', 
              '10 dBZ Echo Top Altitude', '20 dBZ Echo Top Altitude', '30 dBZ Echo Top Altitude', '40 dBZ Echo Top Altitude'] # Extraneous columns.
  target = 'Plume' # The column name for our dependent variable.
  
  X_train, y_train = feature_builder(data_dir + '2017138.csv', data_dir + mdl_lyr + '_IR_2017138_probs.npy', skip_cols, target)
  print(X_train)
  print(len(y_train))
  print(np.sum(y_train))
  print((len(y_train)-np.sum(y_train))/np.sum(y_train))
  X_val, y_val = feature_builder(data_dir + '2017179.csv', data_dir + mdl_lyr + '_IR_2017179_probs.npy', skip_cols, target)
  model = xgb.XGBClassifier(n_estimators=1000,
                            max_depth=5, 
                            learning_rate=0.1,
                            subsample=0.5, 
                            colsample_bytree=0.5, 
                            scale_pos_weight=(len(y_train)-np.sum(y_train))/np.sum(y_train),
                            #scale_pos_weight=250,
                            max_delta_step = 0.0)
                            #reg_alpha = 10.0,
                            #reg_lambda = 50.0)
  eval_set = [(X_val, y_val)]
  model.fit(X_train, y_train, eval_metric="logloss", eval_set=eval_set, verbose=True, early_stopping_rounds=20)
  
  # Show feature importance.
  feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=True)
  feat_imp.plot(kind='barh', title='Feature Importances')
  plt.xlabel('Feature Importance Score')
  plt.tight_layout()
  plt.savefig('../../../results/feat_imp.png')
  plt.show()
  
  # make predictions for test data
  y_train_pred = model.predict(X_train, ntree_limit=model.best_ntree_limit)
  just_metrics(y_train, y_train_pred, mdl_lyr + '_xgboost_deep_res_train_2017138', '../../../results/')
  
  y_val_pred = model.predict(X_val, ntree_limit=model.best_ntree_limit)
  just_metrics(y_val, y_val_pred, mdl_lyr + '_xgboost_deep_res_val_2017179', '../../../results/')
  
  X_test, y_test = feature_builder(data_dir + 'all_2017180.csv', data_dir + mdl_lyr + '_all_IR_2017180_probs.npy', skip_cols, target)
  y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
  just_metrics(y_test, y_pred, mdl_lyr + '_all_xgboost_deep_res_test_2017180', '../../../results/')
  
  X_test, y_test = feature_builder(data_dir + 'all_2017136.csv', data_dir + mdl_lyr + '_all_IR_2017136_probs.npy', skip_cols, target)
  y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
  just_metrics(y_test, y_pred, mdl_lyr + '_all_xgboost_deep_res_test_2017136', '../../../results/')
  
  X_test, y_test = feature_builder(data_dir + 'all_2017095.csv', data_dir + mdl_lyr + '_all_IR_2017095_probs.npy', skip_cols, target)
  y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
  just_metrics(y_test, y_pred, mdl_lyr + '_all_xgboost_deep_res_test_2017095', '../../../results/')

if __name__ == '__main__':
  main()
  
