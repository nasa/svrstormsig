#+
# Name:
#     run_download_model_chkpoint_files.py
# Purpose:
#     This is a script to download model checkpoint files.
#     This is only used if checkpoint files were updated in a new version
#     release such as between Version 1 and Version 2 releases 
# Calling sequence:
#     import run_download_model_chkpoint_files
#     run_download_model_chkpoint_files.run_download_model_chkpoint_files()
# Input:
#     None.
# Functions:
#     run_write_severe_storm_post_processing : Main function to download checkpoint files
# Output:
#     Checkpoint zip file. This file will need to replace the checkpoint files that you currently have from 
#     earlier software version releases 
# Keywords:
#     None
# Author and history:
#     John W. Cooney           2024-05-02.
#
# NOTE:
#     Users were having trouble downloading checkpoint files from website and they were
#     successful when ran these order of operations in python.
#-

#### Environment Setup ###
# Package imports

from setuptools import setup, find_packages
import os
import shutil
from setuptools.command.develop import develop
from setuptools.command.install import install
import requests
import zipfile

def run_download_model_chkpoint_files():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    r = requests.get('https://science-data.larc.nasa.gov/LaRC-SD-Publications/2023-05-05-001-JWC/data/ML_data.zip', allow_redirects=True)
    open('ML_data.zip', 'wb').write(r.content)
    try:
      with zipfile.ZipFile('ML_data.zip', 'r') as zip_ref:
        zip_ref.extractall(working_dir)
      if os.path.isdir(os.path.join(working_dir, 'data', 'model_checkpoints')):
          print('Moving old data and check points file to ' + os.path.join(working_dir, 'data_old'))
          os.makedirs(os.path.join(working_dir, 'data_old'), exist_ok = True)
          shutil.move(os.path.join(working_dir, 'data', 'model_checkpoints'), os.path.join(working_dir, 'data_old'))
      print('moving')
      shutil.move(os.path.join(working_dir, 'ML_data', 'data', 'model_checkpoints'), os.path.join(working_dir, 'data'))
      print('Removing ML_data.zip')  
      shutil.rmtree(os.path.join(working_dir, 'ML_data'))
      os.remove(os.path.join(working_dir, 'ML_data.zip'))
    except:
      print('Zip file from NASA server https://science-data.larc.nasa.gov/LaRC-SD-Publications/2023-05-05-001-JWC/data/ML_data.zip failed to download. Trying again.')
      r = requests.get('https://science-data.larc.nasa.gov/LaRC-SD-Publications/2023-05-05-001-JWC/data/ML_data.zip', allow_redirects=True)
      open('ML_data.zip', 'wb').write(r.content)
      try:
        with zipfile.ZipFile('ML_data.zip', 'r') as zip_ref:
            zip_ref.extractall(working_dir)
        
        if os.path.isdir(os.path.join(working_dir, 'data', 'model_checkpoints')):
            print('Moving old data and check points file to ' + os.path.join(working_dir, 'data_old'))
            os.makedirs(os.path.join(working_dir, 'data_old'), exist_ok = True)
            shutil.move(os.path.join(working_dir, 'data', 'model_checkpoints'), os.path.join(working_dir, 'data_old'))
        shutil.move(os.path.join(working_dir, 'ML_data', 'data', 'model_checkpoints'), os.path.join(working_dir, 'data'))
        print('Output location = ' + os.path.join(working_dir, 'data'))
        shutil.rmtree(os.path.join(working_dir, 'ML_data'))
        os.remove(os.path.join(working_dir, 'ML_data.zip'))
      except:
        print('Zip file from NASA server failed to download for second time. Try downloading the file manually.')
        print('Location of zip file is: https://science-data.larc.nasa.gov/LaRC-SD-Publications/2023-05-05-001-JWC/data/ML_data.zip')
        print('This file needs to be downloaded and data subdirectory needs to be put into svrstormsig directory')
        print('Directory paths: ')
        print('svrstormsig -> data -> model_checkpoints, region, sat_projection_files')

if __name__ == '__main__':
    run_download_model_chkpoint_files()