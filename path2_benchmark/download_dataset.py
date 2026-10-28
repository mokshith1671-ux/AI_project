import os
import urllib.request
import zipfile

def download_and_extract_ujiindoorloc():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    zip_path = os.path.join(data_dir, "ujiindoorloc.zip")
    url = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
    
    print("[1/2] Downloading UJIIndoorLoc dataset from UCI Repository...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
        out_file.write(response.read())
    print("Download complete!")
    
    print("[2/2] Extracting CSV files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
        
    # Check for inner zip if present
    inner_zip = os.path.join(data_dir, "UJIndoorLoc.zip")
    if os.path.exists(inner_zip):
        with zipfile.ZipFile(inner_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
    print("\nDataset successfully downloaded and extracted into 'data/' directory!")
    print("Files ready:")
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                print(" -", os.path.join(root, file))

if __name__ == "__main__":
    download_and_extract_ujiindoorloc()