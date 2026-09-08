import os
import shutil
import requests
import zipfile

def main():
    working_dir = os.getcwd()
    
    print("Downloading updated model weights from NASA server (this may take a few minutes)...")
    url = 'https://science-data.larc.nasa.gov/LaRC-SD-Publications/2023-05-05-001-JWC/data/ML_data.zip'
    r = requests.get(url, allow_redirects=True)
    open('ML_data.zip', 'wb').write(r.content)

    print("Extracting files...")
    with zipfile.ZipFile('ML_data.zip', 'r') as zip_ref:
        zip_ref.extractall(working_dir)

    print("Merging new weights into the data directory...")
    src_data = os.path.join(working_dir, 'ML_data', 'data')
    dst_data = os.path.join(working_dir, 'data') 

    # Safely merge contents over, overwriting/adding new files without crashing
    if os.path.exists(src_data):
        shutil.copytree(src_data, dst_data, dirs_exist_ok=True)

    print("Cleaning up temporary files...")
    shutil.rmtree(os.path.join(working_dir, 'ML_data'))
    os.remove('ML_data.zip')

    print("Successfully updated model weights!")

if __name__ == '__main__':
    main()