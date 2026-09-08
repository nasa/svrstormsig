import os
import shutil
import requests
from datetime import datetime, timedelta
import time
import eumdac

def run_download_seviri_iodc_data(date1           = None, date2 = None, 
                                  outroot         = '../../../seviri-data/', 
                                  sat             = 'meteosat09', 
                                  sector          = 'full', 
                                  download_format = 'netcdf4',
                                  del_local       = False,
                                  essl            = False,
                                  verbose         = True):
                                             
    '''
    Name:
        run_download_seviri_iodc_data.py
    Purpose:
        This is a script to download SEVIRI data, 
        using EUMETSAT's Data Tailor to convert the native formats to NetCDF4 or downloading Native files.
    Calling sequence:
        import run_download_seviri_iodc_data
        run_download_seviri_iodc_data.run_download_seviri_iodc_data()
    Keywords:
        download_format : STRING specifying the format to download the data in. Options are 'netcdf4' or 'native'.
                          DEFAULT = 'netcdf4'
    '''
    
    # Consumer keys obtained from https://api.eumetsat.int/api-key/
    consumer_key    = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")
    if consumer_key is None or consumer_secret is None:
        print()
        print('If you want to download SEVIRI data from the EUMETSAT store, you must first provide your CONSUMER_KEY and CONSUMER_SECRET!!!')
        print('CONSUMER_KEY and CONSUMER_SECRET can be obtained from https://api.eumetsat.int/api-key/')
        print('Example lines added to your bash files are as follows:')
        print('export CONSUMER_KEY="your_key_here"')
        print('export CONSUMER_SECRET="your_secret_here"')
        exit()
 
    # Determine dates
    if date1 == None:
        date1 = datetime.utcnow()
        # SEVIRI scans every 15 minutes (00, 15, 30, 45). Find the most recent complete scan.
        minute_offset = date1.minute % 15
        date1 = date1 - timedelta(minutes=minute_offset) - timedelta(minutes=15)
        date2 = date1 + timedelta(minutes=15)
        rt    = True
        if verbose: print(f"Real-time mode. Looking for scan near {date1.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        date1 = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S") if isinstance(date1, str) else date1
        rt    = False
        if date2 == None:
            date2 = date1
        else:
            date2 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S") if isinstance(date2, str) else date2
        if verbose: print('Downloading dates : ' + date1.strftime("%Y-%m-%d %H:%M:%S") + ' - ' + date2.strftime("%Y-%m-%d %H:%M:%S"))

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

    # Authenticate with EUMETSAT
    credentials = (consumer_key, consumer_secret)
    token       = eumdac.AccessToken(credentials)
    datastore   = eumdac.DataStore(token)
    tailor      = eumdac.DataTailor(token)
    selected_collection = datastore.get_collection(collection_id)

    stime = time.time()
    tmp_dir_path  = os.path.join(outroot, 'tmp_dir')
    os.makedirs(tmp_dir_path, exist_ok=True)
    tmp_file_path = os.path.join(tmp_dir_path, 'tmp_file_seviri.txt')
    
    # Real-time search logic vs Archive search logic
    if rt:
        last_processed_time = None
        if os.path.exists(tmp_file_path):
            with open(tmp_file_path, 'r') as ff:
                date_str = ff.read().strip()
                try:
                    last_processed_time = datetime.strptime(date_str, '%Y%m%d%H%M')
                except Exception as e:
                   pass

        search_time = date1 - timedelta(minutes=15)            
        counter = 0
        max_att = 60
        while counter < max_att:
            if last_processed_time and search_time <= last_processed_time:
                search_time += timedelta(minutes=15)
                counter += 1
                continue
                
            if counter == 0 and verbose:
                print(f"Searching SEVIRI products for {search_time}")
                
            products = selected_collection.search(dtstart=search_time, dtend=search_time + timedelta(minutes=1))
    
            if len(products) > 0:
                date1 = search_time
                date2 = search_time + timedelta(minutes=1)
                break
            
            time.sleep(10)
            if counter == 2 and verbose:
                print(f'Waiting for SEVIRI files for {search_time} to be available online...')
            counter += 1
        
        if counter >= max_att:
            print('No files were available to download for the time range specified.')
            exit()
    else:
        # Search the archive for the specified window
        products = selected_collection.search(dtstart=date1, dtend=date2)
    
    outdirs   = []
    date_strs = []
    
    # If using NetCDF, set up the Data Tailor chain
    chain = None
    if download_format.lower() == 'netcdf4':
        chain = eumdac.tailor_models.Chain(
            product='HRSEVIRI',
            format='netcdf4'
        )
    
    # Process and Download all found products
    for product in products:
        date_str = product.sensing_start.strftime("%Y%m%d")
        if rt:
            d_str0 = product.sensing_start.strftime("%Y%m%d%H%M")
            if verbose: print(f'Scan Being Processed Start Time = {d_str0}')
            with open(tmp_file_path, 'w') as ff:
                ff.write(d_str0)
                
        if essl:
            outdir = os.path.join(outroot, date_str[0:4], date_str[4:6], date_str[6:])
        else:  
            outdir = os.path.join(outroot, date_str)
            
        os.makedirs(outdir, exist_ok = True)
        
        if download_format.lower() == 'netcdf4':
            # EUMETSAT usually includes the timestamp in the tailor output filename
            existing_files = [f for f in os.listdir(outdir) if f.endswith('.nc') and product.sensing_start.strftime("%Y%m%dT%H%M") in f]
        else:
            # Native files are named exactly after the product string
            existing_files = [f for f in os.listdir(outdir) if f.endswith('.nat') and str(product) in f]
            
        if len(existing_files) > 0:
            if verbose:
                print(f"File for {product} already exists in {outdir}, skipping...")
            outdirs.append(outdir)
            date_strs.append(date_str)
            continue
            
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            outfile_path = None
            try:
                if download_format.lower() == 'netcdf4':
                    if verbose: print(f"Requesting Data Tailor NetCDF conversion for: {product} (Attempt {attempt+1}/{max_retries})")
                    
                    # Submit customisation to Data Tailor
                    customisation = tailor.new_customisation(product, chain)
                    
                    # Wait for it to process
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
                        print(f"Data Tailor customisation failed or was killed. Status: {status}")
                        if hasattr(customisation, 'logfile'):
                            print(customisation.logfile)
                        raise Exception("Data Tailor failed to complete customisation.")
                
                else:
                    # NATIVE DOWNLOAD PATH (Extracting the true .nat file from the bundle)
                    if verbose: print(f"Downloading raw native file for: {product} (Attempt {attempt+1}/{max_retries})")
                    
                    downloaded = False
                    for entry_name in product.entries:
                        if entry_name.endswith('.nat'): 
                            outfile_path = os.path.join(outdir, entry_name)
                            
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
                success = True
                break
                    
            except Exception as e:
                print(f"Error processing {product} on attempt {attempt+1}: {e}")
                if outfile_path and os.path.exists(outfile_path):
                    print("Cleaning up partial file...")
                    try: os.remove(outfile_path)
                    except: pass
                
                if attempt < max_retries - 1:
                    time.sleep(1) # Wait 1 second before retrying
                    
        # Outside the retry loop, check if we eventually succeeded
        if success:
            outdirs.append(outdir)
            date_strs.append(date_str)
        else:
            print(f"Failed to download {product} after {max_retries} attempts. Skipping.")

    unique_outdirs = sorted(list(set(outdirs)))
    unique_dates   = sorted(list(set(date_strs)))
    return (unique_outdirs, unique_dates)

def main():
    run_download_seviri_iodc_data()
    
if __name__ == '__main__':
    main()