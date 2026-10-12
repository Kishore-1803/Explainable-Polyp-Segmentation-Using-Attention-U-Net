import os
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    target_dir = os.path.join(os.getcwd(), "hyper_kvasir_data")
    os.makedirs(target_dir, exist_ok=True)
    
    # Official OSF download link for Hyper-Kvasir labeled images (approx 1.5GB for just the labeled portion)
    # The full dataset is much larger, but for training the classifier, the labeled image folders are used.
    url = "https://osf.io/7qjw2/download"
    zip_path = os.path.join(target_dir, "labeled_images.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading Hyper-Kvasir labeled images to {zip_path}...")
        download_url(url, zip_path)
    else:
        print("Zip file already exists. Skipping download.")
        
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
        
    print("Download and extraction complete. The dataset is ready for Stage 03 classification!")

if __name__ == '__main__':
    main()
