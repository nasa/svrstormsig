import os
import shutil
import fnmatch
import requests
import concurrent.futures
from datetime import datetime, timedelta
import eumdac

def download_product(product, file_patterns, outroot, essl=False, verbose=True):
    """Download files for a single product in parallel."""
    date_str = product.sensing_start.strftime("%Y%m%d")
    if essl:
        outdir = os.path.join(outroot, date_str[0:4], date_str[4:6], date_str[6:])
    else:
        outdir = os.path.join(outroot, date_str)
    
    downloaded_files = []
    for file in get_coverage(file_patterns, product.entries):
        if not os.path.exists(os.path.join(outdir, file)):
            os.makedirs(outdir, exist_ok=True)
            try:
                with product.open(entry=file) as fsrc, \
                        open(os.path.join(outdir, fsrc.name), mode='wb') as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                    downloaded_files.append(os.path.join(outdir, fsrc.name))
                    if verbose:
                        print(f"Downloaded: {fsrc.name}")
            except eumdac.product.ProductError as error:
                print(f"Product error for '{file}': {error}")
            except requests.exceptions.RequestException as error:
                print(f"Download error for '{file}': {error}")
    
    return(outdir, date_str)

def run_download_mtg_fci_data_parallel(date1        = None, date2 = None, 
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
        run_download_mtg_fci_data_parallel.py
    Purpose:
        This is a script to download MTG FCI data in parallel
    Calling sequence:
        import run_download_mtg_fci_data_parallel
        run_download_mtg_fci_data_parallel.run_download_mtg_fci_data_parallel()
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
        John W. Cooney           2025-02-11. 
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
    if not no_vis:
        res = 'EO:EUM:DAT:0665'                                                                                                                                             #High resolution MTG data
    
    if not no_native:
        res2 = 'EO:EUM:DAT:0662'                                                                                                                                            #Normal resolution MTG data
    
    if verbose:
        print(res)
        print(res2)

    # Determine file patterns based on the sector
    sector_patterns = {
        'q4'  : ["*_????_0029.nc", "*_????_003[0-9].nc", "*_????_0040.nc", "*_????_0041.nc"],
        'q3'  : ["*_????_002[0-9].nc", "*_????_0030.nc", "*_????_0041.nc"],
        'q2'  : ["*_????_001[0-9].nc", "*_????_002[0-1].nc", "*_????_0041.nc"],
        'q1'  : ["*_????_000[0-9].nc", "*_????_001[0-3].nc", "*_????_0041.nc"],
        't3'  : ["*_????_002[6-9].nc", "*_????_003[0-9].nc", "*_????_0040.nc", "*_????_0041.nc"],
        't2'  : ["*_????_001[3-9].nc", "*_????_002[0-7].nc", "*_????_0041.nc"],
        't1'  : ["*_????_000[1-9].nc", "*_????_001[0-6].nc", "*_????_0041.nc"],
        'h2'  : ["*_????_002[0-9].nc", "*_????_003[0-9].nc", "*_????_0040.nc", "*_????_0041.nc"],
        'h1'  : ["*_????_000[1-9].nc", "*_????_001[0-9].nc", "*_????_002[0-1].nc", "*_????_0041.nc"],
        'f'   : ["*_????_00[0-3][0-9].nc", "*_????_0040.nc", "*_????_0041.nc"],
        'fd'  : ["*_????_00[0-3][0-9].nc", "*_????_0040.nc", "*_????_0041.nc"],
        'full': ["*_????_00[0-3][0-9].nc", "*_????_0040.nc", "*_????_0041.nc"]
    }
    
    file_patterns = sector_patterns.get(sector.lower(), None)
    if file_patterns is None:
        print(f"Sector '{sector}' not supported.")
        return []

    outroot = os.path.realpath(outroot)

    if date2 == None and date1 != None: date2 = date1                                                                                                                       #Default set end date to start date
    if date1 == None:
        print('Date1 must be specified!!')
        exit()
   
    date1 = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S") if isinstance(date1, str) else date1
    date2 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S") if isinstance(date2, str) else date2

    if verbose:
        print(f"Downloading data from {date1.strftime('%Y-%m-%d %H:%M:%S')} to {date2.strftime('%Y-%m-%d %H:%M:%S')}")

    credentials = (consumer_key, consumer_secret)
    token = eumdac.AccessToken(credentials)
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(res)

    products = selected_collection.search(dtstart=date1, dtend=date2)

    outdirs  = set()
    date_strs = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:                                                                                                  #Adjust max_workers as needed
        futures = {executor.submit(download_product, product, file_patterns, outroot, essl, verbose): product for product in products}

        for future in concurrent.futures.as_completed(futures):
            outdir, date_str = future.result()
            outdirs.add(outdir)
            date_strs.add(date_str)
    
    if len(res2) > 0:
        selected_collection = datastore.get_collection(res2)
    
        products = selected_collection.search(dtstart=date1, dtend=date2)
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:                                                                                              #Adjust max_workers as needed
            futures = {executor.submit(download_product, product, file_patterns, outroot, essl, verbose): product for product in products}
    
            for future in concurrent.futures.as_completed(futures):
                outdir, date_str = future.result()
   
    return(sorted(outdirs), sorted(date_strs))

def get_coverage(coverage, filenames):
    """Filter filenames based on provided patterns."""
    return [file for pattern in coverage for file in filenames if fnmatch.fnmatch(file, pattern)]


def main():
    run_download_mtg_fci_data_parallel()
    
if __name__ == '__main__':
    main()
