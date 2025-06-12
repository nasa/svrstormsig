import os
import shutil
import fnmatch
import requests
from datetime import datetime, timedelta
import time
import eumdac

def run_download_mtg_fci_data(date1        = None, date2 = None, 
                              outroot      = '../../../mtg-data/', 
                              sat          = 'mtg1', 
                              sector       = 'full', 
                              no_glm       = True, 
                              no_vis       = False, 
                              no_native    = True,
                              gcs_bucket   = None,
                              del_local    = True,
                              essl         = False,
                              verbose      = True):
                                             
    '''
    Name:
        run_download_mtg_fci_data.py
    Purpose:
        This is a script to download MTG FCI data 
    Calling sequence:
        import run_download_mtg_fci_data
        run_download_mtg_fci_data.run_download_mtg_fci_data()
    Input:
        None.
    Functions:
        run_download_mtg_fci_data : Main function to download GOES data
        write_to_gcs              : Write locally stored file that was just downloaded to a different storage cloud bucket.
    Output:
        Downloads L1c IR and VIS FCI files
    Keywords:
        date1        : Start date. List containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to start downloading 
                       DEFAULT = None -> download nearest to IR/VIS file. (ex. '2017-04-29 00:00:00')
        date2        : End date. String containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to end downloading 
                       DEFAULT = None -> download only files nearest to date1. (ex. '2017-04-29 00:00:00')
        outroot      : STRING specifying output root directory to save the downloaded GOES files
                       DEFAULT = '../../../mtg-data/'
        sat          : STRING specifying GOES satellite to download data for. 
                       DEFAULT = 'mtg1.
        sector       : STRING specifying GOES sector to download. (ex. 'Q4', 'full')
                       DEFAULT = 'full'
        no_glm       : IF keyword set (True), use do not download GLM data files. 
                       DEFAULT = False -> download GLM data files for date range. 
        no_vis       : IF keyword set (True), use do not download VIS data files. 
                       DEFAULT = False -> download VIS data files for date range. 
        no_native    : IF keyword set (True), do not download the native MTG data files. 
                       DEFAULT = True. False -> download native MTG files
        gcs_bucket   : STRING google cloud storage bucket name to write downloaded files to in addition to local storage.
                       DEFAULT = None -> Do not write to a google cloud storage bucket.
        del_local    : IF keyword set (True) AND gcs_bucket != None, delete local copy of output file.
                       DEFAULT = False.
        essl         : BOOL keyword to specify whether or not this will be used for ESSL purposes. ESSL has data stored in /yyyy/mm/dd/ format rather
                       software built format yyyymmdd. Because the software uses the yyyymdd to build other directories etc. off of the path
                       and ESSL may use this software as a way to download the data, this was easiest fix.
                       DEFAULT = False which implies data are stored using /yyyymmdd/. Set to True if data are stored in /yyyy/mm/dd/ format.
        verbose      : BOOL keyword to specify whether or not to print verbose informational messages.
                       DEFAULT = True which implies to print verbose informational messages
    Author and history:
        John W. Cooney           2025-02-10. 
    '''
    #Consumer keys obtained from https://api.eumetsat.int/api-key/
    consumer_key    = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")
    if consumer_key is None or consumer_secret is None:
        print()
        print()
        print('If you want to download MTG data from the EUMETSAT store, you must first provide your CONSUMER_KEY and CONSUMER_SECRET!!!')
        print('CONSUMER_KEY and CONSUMER_SECRET can be obtained from https://api.eumetsat.int/api-key/')
        print('Once you have your CONSUMER_KEY and CONSUMER_SECRET, add it to your .bash_profile, .zshrc, .trsh, .bashrc or whatever it is you use.')
        print()
        print('Example lines added to your bash files are as follows:')
        print('export CONSUMER_KEY="jhbfvawkufevb98ry43ffg74g4qb"')
        print('export CONSUMER_SECRET="jdskafg320gf74vb728rg2"')
        print()
        print('Once added, open a new Terminal window and run the software there.')
        exit()

    res  = 'EO:EUM:DAT:0662'                                                                                                                                                #Normal resolution MTG data
    res2 = []
    if verbose:
        print(no_vis)
        print(no_native)
        print()
    if not no_vis:
        res = 'EO:EUM:DAT:0665'                                                                                                                                             #High resolution MTG data
    
    if not no_native:
        res2 = 'EO:EUM:DAT:0662'                                                                                                                                            #Normal resolution MTG data
    if verbose:
        print(res)
        print(res2)
        print()
    
    #file patterns to use if high resolution data
    if sector.lower() == 'q4':
        file_patterns = ["*_????_0029.nc", "*_????_003[0-9].nc", "*_????_0040.nc", "*_????_0041.nc"] # chunks 29-40 + 41
    elif sector.lower() == 'q3':
        file_patterns = ["*_????_002[0-9].nc", "*_????_0030.nc", "*_????_0041.nc"] # chunks 20-30 + 41
    elif sector.lower() == 'q2':
        file_patterns = ["*_????_001[0-9].nc", "*_????_002[0-1].nc", "*_????_0041.nc"] # chunks 10-21 + 41
    elif sector.lower() == 'q1':
        file_patterns = ["*_????_000[0-9].nc", "*_????_001[0-3].nc", "*_????_0041.nc"] # chunks 01-13 + 41
    elif sector.lower() == 't3':
        file_patterns = ["*_????_002[6-9].nc", "*_????_003[0-9].nc", "*_????_0040.nc", "*_????_0041.nc"] # chunks 26-48 + 41
    elif sector.lower() == 't2':
        file_patterns = ["*_????_001[3-9].nc", "*_????_002[0-7].nc", "*_????_0041.nc"] # chunks 13-27 + 41
    elif sector.lower() == 't1':
        file_patterns = ["*_????_000[1-9].nc", "*_????_001[0-6].nc", "*_????_0041.nc"] # chunks 01-16 + 41
    elif sector.lower() == 'h2':
        file_patterns = ["*_????_002[0-9].nc", "*_????_003[0-9].nc", "*_????_0040.nc", "*_????_0041.nc"] # chunks 20-40 + 41
    elif sector.lower() == 'h1':
        file_patterns = ["*_????_000[1-9].nc", "*_????_001[0-9].nc", "*_????_002[0-1].nc", "*_????_0041.nc"] # chunks 01-21 + 41
    elif sector.lower() == 'f' or sector.lower() == 'fd' or sector.lower() == 'full':
        file_patterns = ["*_????_00[0-3][0-9].nc", "*_????_0040.nc", "*_????_0041.nc"] # Full disc; chunks 01-40 + 41
    else:
        print('Not yet set up to do specified sector')
        print(sector)
        exit()
   
    outroot     = os.path.realpath(outroot)                                                                                                                                 #Create link to real path so compatible with Mac

    if date1 == None:
        date1 = datetime.utcnow()                                                                                                                                           #Extract current date (UTC)
        if date1.minute == 9 or date1.minute == 19 or date1.minute == 29 or date1.minute == 39 or date1.minute == 49 or date1.minute == 59:
            date1 = date1 - timedelta(minutes = 1)
        if date1.minute == 0 or date1.minute == 10 or date1.minute == 20 or date1.minute == 30 or date1.minute == 40 or date1.minute == 50:
            date1 = date1 + timedelta(minutes = 1)
        date1 = date1 - timedelta(minutes = 10)
        date2 = date1 + timedelta(minutes = 10)                                                                                                                             #Default set end date 
        rt    = True                                                                                                                                                        #Real-time download flag
        pat   = os.path.join(outroot, 'tmp_dir')       
    else:
        date1 = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")                                                                                                               #Year-month-day hour:minute:second of start time to download data
        rt    = False                                                                                                                                                       #Real-time download flag
        if date2 == None:
            date2 = date1
        else:
            date2 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")                                                                                                           #Year-month-day hour:minute:second of end time to download data
        if verbose: print('Downloading dates : ' + date1.strftime("%Y-%m-%d-%H-%M-%S") + ' - ' + date2.strftime("%Y-%m-%d-%H-%M-%S"))

    credentials = (consumer_key, consumer_secret)
    token = eumdac.AccessToken(credentials)
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(res)

    stime = time.time()
    print(date1)
    tmp_dir_path  = os.path.join(outroot, 'tmp_dir')
    tmp_file_path = os.path.join(tmp_dir_path, 'tmp_file.txt')
    if rt:
        last_processed_time = None
        if os.path.exists(tmp_file_path):
            with open(tmp_file_path, 'r') as ff:
                date_str = ff.read().strip()                                                                                                                                #Read and remove any whitespace/newlines
                try:
                    last_processed_time = datetime.strptime(date_str, '%Y%m%d%H%M')
                except Exception as e:
                   print(f"Error parsing date from tmp_file: {e}")

        #Set initial search time
        search_time = date1 - timedelta(minutes=10)            
        counter = 0
        max_att = 602                                                                                                                                                       #Maximum number of attempts before breaking out of loop
        while counter < max_att:
            # Skip if already processed
            if last_processed_time:
                if search_time <= last_processed_time+timedelta(minutes=10):
                    search_time += timedelta(minutes=10)
                    counter += 1
                    continue
            if counter == 0 and verbose:
                print(f"Searching MTG products for {search_time}")
            products = selected_collection.search(
                dtstart=search_time,
                dtend=search_time
            )
    
            if len(products) > 0:
                entries = [product.entries for product in products]
                if len(entries[0]) > 10:
                    date1 = search_time
                    date2 = search_time
                    break                                                                                                                                                   #Exit loop
            
            time.sleep(1)
            if counter == 2 and verbose:
                print(f'Waiting for MTG files for {search_time} to be available online')
                print(f'Checking for MTG files for {search_time} to be online again')
            counter+=1
        
        if counter >= max_att:
            print('No files were available to download for the time range specified?')
            print(counter)
            print(max_att)
            print(search_time)
            print(search_time + timedelta(minutes=10))
            exit()
    else:
        products = selected_collection.search(
            dtstart=date1,
            dtend=date2)
    
    outdirs   = []
    date_strs = []
    #Download all found products
    for product in products:
        date_str = product.sensing_start.strftime("%Y%m%d")
        if rt:
            print('Time spent waiting for new data files on EUMETSAT = ' + str(time.time() - stime) + ' sec')
            print('Current Time = ' + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
            d_str0 = product.sensing_start.strftime("%Y%m%d%H%M")
            print(f'Scan Being Processed Start Time = {d_str0}')
            with open(tmp_file_path, 'w') as ff:
                ff.write(d_str0)
        if essl:
            outdir = os.path.join(outroot, date_str[0:4], date_str[4:6], date_str[6:])
        else:  
            outdir = os.path.join(outroot, date_str)
        outdirs.append(outdir)
        date_strs.append(date_str)
        os.makedirs(outdir, exist_ok = True)
        for file in get_coverage(file_patterns, product.entries):
            if not os.path.exists(os.path.join(outdir, file)): 
                try:
                    with product.open(entry=file) as fsrc, \
                            open(os.path.join(outdir, fsrc.name), mode='wb') as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                        print(f'Downloaded: {fsrc.name}')
                except eumdac.product.ProductError as error:
                    print(f"Error related to the product '{selected_collection}' while trying to download it: '{error}'")
                except requests.exceptions.ConnectionError as error:
                    print(f"Error related to the connection: '{error}'")
                except requests.exceptions.RequestException as error:
                    print(f"Unexpected error: {error}")
                    print()

    if len(outdirs) <= 0 or len(date_strs) <= 0:
        print('run_download_mtg_fci_data.py')
        print('No outdirs or date_strs??? Something went wrong')
        print(credentials)
        print(token)
        print()
        print(outdirs)
        print(date_strs)
        print()
        print(products)
        print(len(products))
        print()
        print(date1)
        print(date2)
        print()
        entries = [product.entries for product in products]
        print(len(entries[0]))
        print()
        
    
    if len(res2) > 0:
        selected_collection = datastore.get_collection(res2)
        
        if rt:
            products = selected_collection.search(
                dtstart=date1,
                dtend=date1)
        else:
            products = selected_collection.search(
                dtstart=date1,
                dtend=date2)

        if len(products) == 0 and rt:
            print('Waiting for MTG files to be available online')
            while True:
                time.sleep(1)
                if verbose: print('Checking for MTG files to be online again')
                products = selected_collection.search(
                    dtstart=date1,
                    dtend=date2)
                if len(products) > 0:
                    entries = [product.entries for product in products]
                    if len(entries) > 0:
                        if len(entries[0]) > 10:
                            break
                    else:
                        print('run_download_mtg_fci_data.py')
                        print(date1)
                        print(date2)
                        print()
                        print(len(products))
                        print(len(entries))
                        print()
                        print(products)
                        print(entries)
        #Download all found products
        for product in products:
            date_str = product.sensing_start.strftime("%Y%m%d")
            if essl:
                outdir = os.path.join(outroot, date_str[0:4], date_str[4:6], date_str[6:])
            else:
                outdir = os.path.join(outroot, date_str)
            outdirs.append(outdir)
            date_strs.append(date_str)
            os.makedirs(outdir, exist_ok = True)
            for file in get_coverage(file_patterns, product.entries):
                if not os.path.exists(os.path.join(outdir, file)): 
                    os.makedirs(outdir, exist_ok = True)
                    try:
                        with product.open(entry=file) as fsrc, \
                                open(os.path.join(outdir, fsrc.name), mode='wb') as fdst:
                            shutil.copyfileobj(fsrc, fdst)
                            print(f'Downloaded: {fsrc.name}')
                    except eumdac.product.ProductError as error:
                        print(f"Error related to the product '{selected_collection}' while trying to download it: '{error}'")
                    except requests.exceptions.ConnectionError as error:
                        print(f"Error related to the connection: '{error}'")
                    except requests.exceptions.RequestException as error:
                        print(f"Unexpected error: {error}")
                        print()

    unique_outdirs = sorted(list(set(outdirs)))
    unique_dates   = sorted(list(set(date_strs)))
    return(unique_outdirs, unique_dates)

#     if del_local:
#         unique_outdirs = sorted(list(set(outdirs)))
#         for o in unique_outdirs:
#             clean_outdir(o)

# This function checks if a product entry is part of the requested coverage
def get_coverage(coverage, filenames):
    chunks = []
    for pattern in coverage:
        for file in filenames:
            if fnmatch.fnmatch(file, pattern):
                chunks.append(file)
    return(chunks)

def clean_outdir(outdir):
    """
    Deletes all non-netCDF files and empty directories within outdir.
    Parameters:
        outdir (str): Path to the output directory to clean.
    """
    if not os.path.exists(outdir):
        print(f"Directory {outdir} does not exist.")
        return
    for root, dirs, files in os.walk(outdir, topdown=False):  # Process files before directories
        # Remove non-netCDF files
        for file in files:
            if not file.endswith(".nc"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        # Remove empty directories
        for dir_ in dirs:
            dir_path = os.path.join(root, dir_)
            if not os.listdir(dir_path):  # Check if empty
                os.rmdir(dir_path)
                print(f"Removed empty directory: {dir_path}")


def main():
    run_download_mtg_fci_data()
    
if __name__ == '__main__':
    main()
