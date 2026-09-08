import os
import shutil
import requests
import concurrent.futures
from datetime import datetime, timedelta
import time
import eumdac

def download_product_tailor(product, tailor, chain, outroot, download_format='netcdf4', essl=False, verbose=True):
    """Submit product to Data Tailor and download converted NetCDF in parallel."""
    date_str = product.sensing_start.strftime("%Y%m%d")
    if essl:
        outdir = os.path.join(outroot, date_str[0:4], date_str[4:6], date_str[6:])
    else:
        outdir = os.path.join(outroot, date_str)
    
    os.makedirs(outdir, exist_ok=True)
    
    # Check if we already downloaded a NetCDF file for this timestamp
    # EUMETSAT usually includes the timestamp in the tailor output filename
    if download_format.lower() == 'netcdf4':
        # EUMETSAT usually includes the timestamp in the tailor output filename
        existing_files = [f for f in os.listdir(outdir) if f.endswith('.nc') and product.sensing_start.strftime("%Y%m%dT%H%M") in f]
    else:
        # Native files are named exactly after the product string
        existing_files = [f for f in os.listdir(outdir) if f.endswith('.nat') and str(product) in f]
    
    if len(existing_files) > 0:
        if verbose:
            print(f"File for {product.sensing_start} already exists in {outdir}, skipping...")
        return(outdir, date_str)
        
    max_retries = 3
    for attempt in range(max_retries):
        outfile_path = None
        try:
            if download_format.lower() == 'netcdf4':
                if verbose:
                    print(f"Requesting Data Tailor NetCDF conversion for: {product} (Attempt {attempt+1}/{max_retries})")
                
                # Submit customisation to Data Tailor
                customisation = tailor.new_customisation(product, chain)
                
                # Wait for it to process on EUMETSAT servers
                status = customisation.status
                while status in ['QUEUED', 'RUNNING']:
                    time.sleep(0.5)
                    status = customisation.status
                    
                if "DONE" in status:
                    for output_file in customisation.outputs:
                        # Strip EUMETSAT's temporary workspace folder from the string
                        local_filename = os.path.basename(output_file)
                        outfile_path = os.path.join(outdir, local_filename)
                        
                        if not os.path.exists(outfile_path):
                            # We stream using the original output_file string, but save it to our clean local path
                            with customisation.stream_output(output_file) as stream, open(outfile_path, mode='wb') as fdst:
                                shutil.copyfileobj(stream, fdst)
                            if verbose:
                                print(f"Successfully downloaded: {local_filename}")
                    
                    # Delete the customisation from the server to free up the 20GB EUMETSAT quota limit
                    try: customisation.delete()
                    except: pass
                else:
                    if verbose:
                        print(f"Data Tailor customisation failed or was killed. Status: {status}")
                    raise Exception("Data Tailor failed to complete customisation.")
            else:
                if verbose: print(f"Downloading raw native file for: {product} (Attempt {attempt+1}/{max_retries})")
                
                # Products often contain multiple entries (data file + metadata).
                # In eumdac, product.entries is a list of strings (filenames).
                downloaded = False
                for entry_name in product.entries:
                    # Since entry_name is already a string, we check it directly
                    if entry_name.endswith('.nat'): 
                        outfile_path = os.path.join(outdir, entry_name)
                        
                        # Open only this specific file from the EUMETSAT product package
                        with product.open(entry=entry_name) as fsrc, open(outfile_path, mode='wb') as fdst:
                            shutil.copyfileobj(fsrc, fdst)
                            
                        if verbose:
                            print(f"Successfully downloaded native file with original name: {entry_name}")
                        
                        downloaded = True
                        break # Stop after finding the first .nat file
                        
                # Fallback just in case standard entries aren't available
                if not downloaded:
                    if verbose: print(f"No .nat entry found. Downloading whole product as .zip...")
                    outfile_path = os.path.join(outdir, str(product) + '.zip')
                    with product.open() as fsrc, open(outfile_path, mode='wb') as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                    if verbose:
                        print(f"Successfully downloaded product archive: {os.path.basename(outfile_path)}")
            
            # Download was successful!
            return(outdir, date_str)
                        
        except Exception as e:
            print(f"Error processing {product} on attempt {attempt+1}: {e}")
            if outfile_path and os.path.exists(outfile_path):
                print("Cleaning up partial file...")
                try: os.remove(outfile_path)
                except: pass
            
            if attempt < max_retries - 1:
                time.sleep(1) # Wait 1 second before retrying
            else:
                print(f"Failed to download {product} after {max_retries} attempts. Skipping.")
                return(None, None)


def run_download_seviri_iodc_data_parallel(date1           = None, date2 = None, 
                                           outroot         = '../../../seviri-data/', 
                                           sat             = 'meteosat09', 
                                           sector          = 'full', 
                                           del_local       = False,
                                           download_format = 'netcdf4',
                                           essl            = False,
                                           verbose         = True):
                                             
    '''
    Name:
        run_download_seviri_iodc_data_parallel.py
    Purpose:
        This is a script to download SEVIRI IODC (Meteosat-09) data in parallel using
        EUMETSAT's Data Tailor to convert the native formats to NetCDF4 on-the-fly.
    Calling sequence:
        import run_download_seviri_iodc_data_parallel
        run_download_seviri_iodc_data_parallel.run_download_seviri_iodc_data_parallel()
    Input:
        None.
    Functions:
        run_download_seviri_iodc_data_parallel : Main function to download SEVIRI data
        download_product_tailor                : Helper function to submit and download individual files
    Output:
        Downloads Level SEVIRI IODC files in NetCDF4 format
    Keywords:
        date1           : Start date. String containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to start downloading 
                          DEFAULT = None -> MUST BE SPECIFIED for archive parallel runs. (ex. '2022-06-05 00:00:00')
        date2           : End date. String containing year-month-day-hour-minute-second 'year-month-day hour:minute:second' to end downloading 
                          DEFAULT = None -> download only files nearest to date1. (ex. '2022-06-05 00:00:00')
        outroot         : STRING specifying output root directory to save the downloaded SEVIRI files
                          DEFAULT = '../../../seviri-data/'
        sat             : STRING specifying Satellite to download data for. 
                          DEFAULT = 'meteosat09'.
        sector          : STRING specifying sector to download. 
                          DEFAULT = 'full'
        del_local       : IF keyword set (True) AND gcs_bucket != None, delete local copy of output file.
                          DEFAULT = False.
        download_format : STRING specifying the format to download the data in. Options are 'netcdf4' or 'native'.
                          DEFAULT = 'netcdf4'
        essl            : BOOL keyword to specify whether or not this will be used for ESSL purposes. ESSL has data stored in /yyyy/mm/dd/ format rather
                          software built format yyyymmdd. Because the software uses the yyyymdd to build other directories etc. off of the path
                          and ESSL may use this software as a way to download the data, this was easiest fix.
                          DEFAULT = False which implies data are stored using /yyyymmdd/. Set to True if data are stored in /yyyy/mm/dd/ format.
        verbose         : BOOL keyword to specify whether or not to print verbose informational messages.
                          DEFAULT = True which implies to print verbose informational messages
    Author and history:
        John W. Cooney           2026-06-08. (Adapted for SEVIRI Data Tailor processing)
    '''
    #Consumer keys obtained from https://api.eumetsat.int/api-key/
    consumer_key    = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")
    if consumer_key is None or consumer_secret is None:
        print()
        print()
        print('If you want to download SEVIRI data from the EUMETSAT store, you must first provide your CONSUMER_KEY and CONSUMER_SECRET!!!')
        print('CONSUMER_KEY and CONSUMER_SECRET can be obtained from https://api.eumetsat.int/api-key/')
        print('Once you have your CONSUMER_KEY and CONSUMER_SECRET, add it to your .bash_profile, .zshrc, .trsh, .bashrc or whatever it is you use.')
        print()
        print('Example lines added to your bash files are as follows:')
        print('export CONSUMER_KEY="jhbfvawkufevb98ry43ffg74g4qb"')
        print('export CONSUMER_SECRET="jdskafg320gf74vb728rg2"')
        print()
        print('Once added, open a new Terminal window and run the software there.')
        exit()

    if date2 == None and date1 != None: date2 = date1                                                                                                                       #Default set end date to start date
    if date1 == None:
        print('Date1 must be specified for parallel archived runs!!')
        exit()
   
    date1 = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S") if isinstance(date1, str) else date1
    date2 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S") if isinstance(date2, str) else date2
    
    sat_clean = sat.lower().replace('-', '')
    if 'iodc' in sat_clean:
        collection_id = 'EO:EUM:DAT:MSG:HRSEVIRI-IODC'
    elif 'rss' in sat_clean or 'rapid' in sat_clean:
        collection_id = 'EO:EUM:DAT:MSG:MSG15-RSS'
    elif sat_clean in ['meteosat08', 'meteosat8', 'msg1', 'msg01']:
        # Meteosat-8 operated the IODC service from Oct 2016 to June 2022
        if date1 >= datetime(2016, 10, 1) and date1 < datetime(2022, 6, 1):
            collection_id = 'EO:EUM:DAT:MSG:HRSEVIRI-IODC'
        else:
            collection_id = 'EO:EUM:DAT:MSG:HRSEVIRI'
            
    elif sat_clean in ['meteosat09', 'meteosat9', 'msg2', 'msg02']:
        # Meteosat-9 took over the IODC service starting June 1, 2022
        if date1 >= datetime(2022, 6, 1):
            collection_id = 'EO:EUM:DAT:MSG:HRSEVIRI-IODC'
        elif date1 >= datetime(2013, 4, 9) and date1 < datetime(2018, 3, 1):
            collection_id = 'EO:EUM:DAT:MSG:MSG15-RSS' # Met-9 did RSS during this window
        else:
            collection_id = 'EO:EUM:DAT:MSG:HRSEVIRI'
            
    elif sat_clean in ['meteosat10', 'meteosat10', 'msg3', 'msg03']:
        # Meteosat-10 was 0-degree Prime until March 2018, then moved to Rapid Scan (RSS)
        if date1 >= datetime(2018, 3, 1):
            collection_id = 'EO:EUM:DAT:MSG:MSG15-RSS'
        else:
            collection_id = 'EO:EUM:DAT:MSG:HRSEVIRI'
            
    elif sat_clean in ['meteosat11', 'meteosat11', 'msg4', 'msg04']:
        # Meteosat-11 has been the 0-degree Prime satellite since early 2018
        collection_id = 'EO:EUM:DAT:MSG:HRSEVIRI'
    
    else:
        collection_id = 'EO:EUM:DAT:MSG:HRSEVIRI'
        
    if verbose:
        print(f"Routing request to EUMETSAT Collection: {collection_id}")    

    outroot = os.path.realpath(outroot)                                                                                                                                     #Create link to real path so compatible with Mac


    if verbose:
        print(f"Downloading data from {date1.strftime('%Y-%m-%d %H:%M:%S')} to {date2.strftime('%Y-%m-%d %H:%M:%S')}")

    # Authenticate with EUMETSAT
    credentials = (consumer_key, consumer_secret)
    token       = eumdac.AccessToken(credentials)
    datastore   = eumdac.DataStore(token)
    tailor      = eumdac.DataTailor(token)
    selected_collection = datastore.get_collection(collection_id)

    # Search for products in the archive
    products = selected_collection.search(dtstart=date1, dtend=date2)

    # Define the Data Tailor chain to convert native format to NetCDF4 based on eumdac 3.1.1
    chain = None
    if download_format.lower() == 'netcdf4':
        chain = eumdac.tailor_models.Chain(
            product='HRSEVIRI',
            format='netcdf4'
        )

    outdirs   = set()
    date_strs = set()
    
    # NOTE: max_workers is kept strictly low (e.g. 3) because EUMETSAT imposes limits 
    # on the number of concurrent Data Tailor processes per account. Exceeding this 
    # will result in rejected customisation requests.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:                                                                                                  #Adjust max_workers carefully to avoid hitting EPCT quota limits
        futures = {executor.submit(download_product_tailor, product, tailor, chain, outroot, download_format, essl, verbose): product for product in products}

        for future in concurrent.futures.as_completed(futures):
            outdir, date_str = future.result()
            # Only add to lists if the download was actually successful
            if outdir is not None:
                outdirs.add(outdir)
                date_strs.add(date_str)
   
    return(sorted(list(outdirs)), sorted(list(date_strs)))


def main():
    run_download_seviri_iodc_data_parallel()
    
if __name__ == '__main__':
    main()