#+
# Name:
#     gcs_processing.py
# Purpose:
#     This is a script containing functions that do all of the gcs processing
# Calling sequence:
#     from gcs_processing import *
# Input:
#     None.
# Functions:
#     list_gcs                : Lists files within storage bucket that match specified patterns.
#     list_csv_gcs            : Lists csv files within storage bucket that match specified patterns.
#     list_gcs_chkpnt         : Lists all model checkpoint files within storage bucket within path.
#     download_model_chk_file : Downloads model checkpoint files to local directory
#     download_ncdf_gcs       : Downloads netCDF files from storage bucket to local directory
#     load_blobs              : Loads a pre-processed numpy model file from the bucket. File does not need to be downloaded.
#     load_json_gcs           : Loads json files into local memory. File does not need to be downloaded.
#     load_png_gcs            : Loads png files into local memory. File does not need to be downloaded.
#     load_npy_blobs          : Loads numpy files into local memory. File does not need to be downloaded.
#     load_npy_gcs            : Loads specific numpy files into local memory. File does not need to be downloaded.
#     load_csv_gcs            : Loads specific csv files into local memory. File does not need to be downloaded.
#     write_to_gcs            : Writes locally saved files to Google Cloud Storage bucket. Option to delete local copy of file. 
# Output:
#     Returns dependent on gcs function called. 
# Keywords:
#     None.
# Author and history:
#     John W. Cooney           2021-05-05.
#
#-

#### Environment Setup ###
# Package imports
import numpy as np
import pandas as pd
import os
from google.cloud import vision
from google.cloud import storage
from io import BytesIO
import json
#Finding file matches in GCS paths
def list_gcs(bucket_name, gcs_prefix, gcs_patterns, delimiter = '/'):
    """Lists all file names and paths given the google cloud storage bucket and patterns."""
    # bucket_name  = "your-bucket-name" = "aacp-proc-data"
    # gcs_prefix   = "storage-object-prefix" = "cooney/M2/2020134/"
    # gcs_patterns = "storage-object-file-patterns" = ['.json', 's{}{:03d}'.format(date.year, day_num)]
    
    if gcs_prefix and not gcs_prefix.endswith('/'):
        gcs_prefix += '/'
    storage_client = storage.Client()                                                                                                   #Start up storage client
    bucket = storage_client.bucket(bucket_name)                                                                                         #Access storage bucket
    blobs  = bucket.list_blobs(prefix=gcs_prefix, delimiter=delimiter)                                                                  #Extract list of blobs within storage bucket that satisfy gcs_prefix subdirectories
    fname  = []                                                                                                                         #Initialize list to store google cloud file names 
    if gcs_patterns == None or len(gcs_patterns) == 0:
       for b in blobs:
           fname.append(b.name)
    else:
       for b in blobs:
           match = True
           for pattern in gcs_patterns:
               if not pattern in b.path:
                  match = False
           if match:
               fname.append(b.name)
    
    fname2 = [fname[i] for i, j in enumerate(fname) if 'old' not in os.path.basename(j)]                                                                  #Only retain file names that do not include 'old' in the name

    return(fname2)

def list_csv_gcs(bucket_name, prefix, source_blob_name):
    """Lists all csv file names and paths given the google cloud storage bucket, prefix, and file name."""
    # bucket_name = "your-bucket-name" = "aacp-results"
    # source_blob_name = "storage-object-name" = "imgs_train_2000.npy"
    if prefix and not prefix.endswith('/'):
        prefix += '/'

#    print('Searching for ' + source_blob_name + ' in bucket ' + bucket_name)
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blobs  = list(bucket.list_blobs(fields="items(name)", prefix = prefix, delimiter = '/'))
    if (len(blobs) == 0):
        print('No blobs found in bucket???')
        print(prefix)
        return([])
    
    csv_f  = []
    for b, blob in enumerate(blobs):
        fb = os.path.basename(blob.name)
        if source_blob_name == fb:
            csv_f.append(blob.name)
    
    return(csv_f)

def list_gcs_chkpnt(bucket_name, gcs_prefix, delimiter = '/'):
    """Lists all netCDF file names and paths given the google cloud storage bucket and date string."""
    # bucket_name  = "your-bucket-name" = "aacp-results"
    # gcs_prefix   = "storage-object-prefix" = "Cooney_testing/ir_vis/2021-03-26/chosen_indices/by_date/by_updraft"
    if gcs_prefix and not gcs_prefix.endswith('/'):
        gcs_prefix += '/'
    print('Searching for checkpoint files in ' + gcs_prefix + ' in bucket ' + bucket_name)
    gcs_patterns = ['checkpoint']
    storage_client = storage.Client()                                                                                                   #Start up storage client
    bucket = storage_client.bucket(bucket_name)                                                                                         #Access storage bucket
    blobs  = bucket.list_blobs(prefix=gcs_prefix, delimiter=delimiter)                                                                  #Extract list of blobs within storage bucket that satisfy gcs_prefix subdirectories
    fname  = []                                                                                                                         #Initialize list to store google cloud file names 
    if gcs_patterns == None or len(gcs_patterns) == 0:
       for b in blobs:
           fname.append(b.name)
    else:
       for b in blobs:
           match = True
           for pattern in gcs_patterns:
               if not pattern in b.path:
                  match = False
           if match:
               fname.append(b.name)
    return(fname)

#Loading and downloading files from GCP
def download_model_chk_file(bucket_name, source_file, destination_file_path):
    """Loads a pre-processed model file from the bucket."""
    # bucket_name = "your-bucket-name" = "aacp-results"
    # source_file = "storage-object-path-name" = "Cooney_testing/ir_vis/2021-03-26/chosen_indices/by_date/by_updraft/unet_checkpoint.cp"
    # destination_file_path = "local/path/to/file" = "../../../ir_vis/2021-03-26/chosen_indices/by_date/by_updraft/"

    # bucket_name = "your-bucket-name" = "aacp-results"
    # prefix      = "storage-object-prefix" = "Cooney_testing/ir_vis/2021-03-26/chosen_indices/by_date/by_updraft"
    # source_blob_name = "storage-object-name" = "unet_checkpoint.cp"
    print('Downloading checkpoint file ' + source_file + ' in bucket ' + bucket_name + ' to ' + destination_file_path)
    if os.path.isfile(os.path.join(destination_file_path, os.path.basename(source_file))) == False:                                     #Check if file already exists before attempting to download it
        print('Searching for ' + source_file + ' in bucket ' + bucket_name)
        storage_client = storage.Client()
        
        bucket = storage_client.bucket(bucket_name)
        blob   = bucket.blob(source_file)                                                                                               #Create blob for file in google cloud
        blob.download_to_filename(os.path.join(destination_file_path, os.path.basename(source_file)))                                   #Download file from google cloud to locally stored file name and path (ex. '/mnt/disks/data/results/test.nc')

def download_ncdf_gcs(bucket_name, source_file, destination_file_path):
    """Downloads a netCDF file from the google cloud storage bucket."""
    # bucket_name = "your-bucket-name" = "goes-data"
    # source_file = "storage-object-path-name" = "ABI-L1b-RadM/2019/119/00/OR_ABI-L1b-RadM1-M6C13_G16_s20191190000320_e20191190000388_c20191190000416.nc"
    # destination_file_path = "local/path/to/file" = "../../../goes-data/aacp_results/20200517/ir/"
 
#    print('Downloading netCDF file ' + source_file + ' in bucket ' + bucket_name + ' to ' + destination_file_path)
    if os.path.isfile(os.path.join(destination_file_path, os.path.basename(source_file))) == False:                                     #Check if file already exists before attempting to download it
        storage_client = storage.Client()                                                                                               #Open google cloud storage client
        bucket = storage_client.bucket(bucket_name)                                                                                     #Extract bucket information
        blob   = bucket.blob(source_file)                                                                                               #Create blob for file in google cloud
        blob.download_to_filename(os.path.join(destination_file_path, os.path.basename(source_file)))                                   #Download file from google cloud to locally stored file name and path (ex. '/mnt/disks/data/results/test.nc')
        return(os.path.join(destination_file_path, os.path.basename(source_file)))  
    else:  
        return(os.path.join(destination_file_path, os.path.basename(source_file)))                                                      #Return file name that already exists

def load_blobs(bucket_name, prefix, source_blob_name):
    """Loads a pre-processed model file from the bucket."""
    # bucket_name = "your-bucket-name" = "aacp-results"
    # prefix      = "storage-object-prefix" = "Cooney_testing/ir_vis_glm/2019137/"
    # source_blob_name = "storage-object-name" = "imgs_train_2000.npy"
    
    if prefix and not prefix.endswith('/'):
        prefix += '/'

#    print('Reading ' + source_blob_name + ' in ' + prefix + ' with bucket ' + bucket_name)
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
#    blobs  = list(bucket.list_blobs(fields="items(name)", prefix = prefix, delimiter = '/'))
#    blobs  = list(bucket.list_blobs(prefix = prefix, delimiter = '/'))
    blobs  = list(bucket.list_blobs(prefix = prefix))
    if (len(blobs) == 0):
        print('No blobs found in bucket???')
        print(prefix)
        exit()
    c      = 0
    for b, blob in enumerate(blobs):
        fb = os.path.basename(blob.name)
        if source_blob_name == fb:
            print(blob.name)
            if c == 0:
                data0 = np.load(BytesIO(storage.blob.Blob.download_as_string(blobs[b])))
                c = c+1
            else:
                data0 = np.append(data0, np.load(BytesIO(storage.blob.Blob.download_as_string(blobs[b]))), axis = 0)
    return(data0)

def load_json_gcs(bucket_name, source_json_file):
    """Returns contents of json file within google cloud bucket."""
    # bucket_name = "your-bucket-name" = "aacp-proc-data"
    # source_csv_fullname = "storage-object-name" = "goes-data/labelled/cooney/M2/2020134/OR_ABI_L1b_M2_COMBINED_s20201341902495_e20201341902564_c20201341903017.json"
    print('Reading .json file ' + source_json_file + ' from bucket ' + bucket_name)
    storage_client = storage.Client()                                                                                                   #Initialize google cloud storage client
    bucket         = storage_client.bucket(bucket_name)                                                                                 #Initiate storage bucket that you are loading from
    blob           = bucket.get_blob(source_json_file)                                                                                  #Get json blob from json file path and name
    data0          = json.load(BytesIO(storage.blob.Blob.download_as_string(blob)))                                                     #Load json file from storage bucker
    return(data0)

def load_png_gcs(bucket_name, source_png_file):
    """Returns contents of json file within google cloud bucket."""
    # bucket_name = "your-bucket-name" = "ir-vis-sandwhich"
    # source_csv_fullname = "storage-object-name" = "layered_img_dir/pixel_grid/M2/20200513-14/OR_ABI_L1b_M2_COMBINED_s20201341902495_e20201341902564_c20201341903017.png"
    print('Reading .png file ' + source_png_file + ' from bucket ' + bucket_name)
    storage_client = storage.Client()                                                                                                   #Initialize google cloud storage client
    bucket         = storage_client.bucket(bucket_name)                                                                                 #Initiate storage bucket that you are loading from
    blob           = bucket.get_blob(source_png_file)                                                                                   #Get json blob from json file path and name
    data0          = storage.blob.Blob.download_as_string(blob)                                                                         #Load PNG file from storage bucket
    return(data0)

def load_npy_blobs(bucket_name, prefix, source_blob_name):
    """Loads a model file from the bucket. Appends all files found within the bucket + prefix."""
    # bucket_name = "your-bucket-name" = "aacp-results"
    # source_blob_name = "storage-object-name" = "imgs_train_2000.npy"
    # destination_file_name = "local/path/to/file" = "../../../goes-data/aacp_results/ir_vis_glm/"
    if prefix and not prefix.endswith('/'):
        prefix += '/'

 #   print('Reading numpy file ' + source_blob_name + ' in ' + prefix + ' in bucket ' + bucket_name)
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
#    blobs  = list(bucket.list_blobs(fields="items(name)", prefix = prefix, delimiter = '/'))
    blobs  = list(bucket.list_blobs(prefix = prefix, delimiter = '/'))
    if (len(blobs) == 0):
        print('No blobs found in bucket???')
        print(prefix)
        exit()
    c      = 0
    for b, blob in enumerate(blobs):
        fb = os.path.basename(blob.name)
        if source_blob_name == fb:
            if c == 0:
                data0 = np.load(BytesIO(storage.blob.Blob.download_as_string(blobs[b])))
                c = c+1
            else:
                data0 = np.append(data0, np.load(BytesIO(storage.blob.Blob.download_as_string(blobs[b]))), axis = 0)
    return(data0)

def load_npy_gcs(bucket_name, source_npy_fullname):
    """Gets specified numpy file read to be loaded."""
    # bucket_name = "your-bucket-name" = "goes-proc-data"
    # source_npy_fullname = "storage-object-name" = "goes-data/labelled/2019137/M2/updraft_f.npy"

#    print('Reading ' + source_npy_fullname + ' in bucket ' + bucket_name)
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob   = bucket.get_blob(source_npy_fullname)
    data0  = np.load(BytesIO(storage.blob.Blob.download_as_string(blob)))
    return(data0)

def load_csv_gcs(bucket_name, source_csv_fullname, skiprows = None):
    """Returns contents of csv file within google cloud bucket."""
    # bucket_name = "your-bucket-name" = "aacp-results"
    # source_csv_fullname = "storage-object-name" = "goes-data/labelled/2019137/M2/vis_ir_glm_json_combined_ncdf_filenames_with_npy_files.csv"

#    print('Reading ' + source_csv_fullname + ' in bucket ' + bucket_name)
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob   = bucket.get_blob(source_csv_fullname)
    if blob == None:
        return(pd.DataFrame())
    else:
        data0  = pd.read_csv(BytesIO(storage.blob.Blob.download_as_string(blob)), skiprows = skiprows, index_col = 0)
        return(data0)

def download_gcp_parallel(idx, files, bucket_name, outdir):
    """Downloads netCDF files in parallel."""
    # idx         = index of files to download
    # files       = GCP file names and prefix to download_as_string
    # bucket_name = "your-bucket-name" = "aacp-results"
    # outdir      = directory to download GCP files to
    #Example: 
    #         pool    = mp.Pool(3)
    #         results = [pool.apply_async(download_gcp_parallel, args=(row, eth_files, bucket_name, eth_dir)) for row in range(len(eth_files))]
    #         pool.close()
    #         pool.join()
    if outdir and not outdir.endswith('/'):
        outdir += '/'
        
    fdownload = download_ncdf_gcs(bucket_name, files[idx], outdir)    

#Writing to  GCP
def write_to_gcs(bucket_name, prefix, source_blob_file, del_local = False):  
    """Writes files to google cloud storage bucket."""  
    # bucket_name = "your-bucket-name" = "aacp-results"  
    # prefix      = "storage-object-prefix" = "20190507/ir/"  
    # source_blob_file = "local/path/to/file" = "../../../goes-data/20190507/glm/OR_GLM-L2-LCFA_G16_s20191210359400_e20191210400000_c20191210400029.nc"  
    if prefix and not prefix.endswith('/'):
        prefix += '/'
  
    print('Writing ' + source_blob_file + ' with path ' + prefix + ' in bucket ' + bucket_name)  
  
    storage_client = storage.Client()                                                                                                                                       #Open google cloud storage client
    bucket = storage_client.bucket(bucket_name)                                                                                                                             #Extract bucket information
    blob   = bucket.blob(os.path.join(prefix, os.path.basename(source_blob_file)))                                                                                          #Create blob for file in google cloud
    blob.upload_from_filename(source_blob_file)                                                                                                                             #Upload file to google cloud from locally stored file name and path (ex. '/mnt/disks/data/results/test.npy')
    if del_local == True: os.remove(source_blob_file)                                                                                                                       #Delete local copy of file

def copy_blob_gcs(source_bucket_name, source_blob_file, destination_bucket_name, destination_blob_file):
    """Moves a blob from one bucket to another with a new name."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The GCP storage path and name of file (without source_bucket_name)
    # blob_name = "gcp-path-to-your-source-file/source-file-basename"
    # The ID of the bucket to move the object to
    # destination_bucket_name = "destination-bucket-name"
    # The ID of your new GCS object (optional)
    # destination_blob_name = "gcp-path-to-your-destination-file/destination-file-basename"

    storage_client = storage.Client()
    source_bucket = storage_client.bucket(source_bucket_name)   
    blob_name     = source_bucket.blob(os.path.join(os.path.dirname(source_blob_file), os.path.basename(source_blob_file)))
    source_blob   = source_bucket.blob(blob_name.name)
    destination_bucket = storage_client.bucket(destination_bucket_name)

    blob_copy = source_bucket.copy_blob(
        source_blob, destination_bucket, destination_blob_file
    )
#    source_bucket.delete_blob(blob_name)

    print(
        "Blob {} in bucket {} moved to blob {} in bucket {}.".format(
            source_blob.name,
            source_bucket.name,
            blob_copy.name,
            destination_bucket.name,
        )
    )
    
def main():
    gcs_processing()
    
if __name__ == '__main__':
    main()
