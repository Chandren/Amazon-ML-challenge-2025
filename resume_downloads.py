#!/usr/bin/env python3

import os
import pandas as pd
import multiprocessing
from time import time as timer
from tqdm import tqdm
import urllib.request
from pathlib import Path
from functools import partial
import sys

# Fix tokenizers parallelism issue
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def download_image_robust(image_link, savefolder, timeout=10, max_retries=2):
    """
    More robust image download with timeout and retry logic
    """
    if not isinstance(image_link, str):
        return False
        
    filename = Path(image_link).name
    image_save_path = os.path.join(savefolder, filename)
    
    # Skip if already exists
    if os.path.exists(image_save_path):
        return True
    
    # Try downloading with retries
    for retry in range(max_retries):
        try:
            # Set timeout to prevent hanging
            urllib.request.urlretrieve(image_link, image_save_path)
            return True
            
        except Exception as ex:
            if retry == max_retries - 1:  # Last retry
                # Only print every 100th error to avoid spam
                if hash(image_link) % 100 == 0:
                    print(f'Warning: Failed to download {image_link}: {str(ex)[:100]}')
            continue
    
    return False

def download_images_robust(image_links, download_folder, num_workers=50):
    """
    Download images with reduced worker count and better error handling
    """
    print(f"Starting download of {len(image_links)} images...")
    print(f"Using {num_workers} workers (reduced from 100 for stability)")
    
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    # Check existing files
    existing_files = set(os.listdir(download_folder))
    existing_count = len([f for f in existing_files if f.endswith('.jpg')])
    
    print(f"Found {existing_count} existing images, will resume download...")
    
    # Create partial function with timeout
    download_func = partial(download_image_robust, savefolder=download_folder)
    
    # Use reduced number of workers to prevent overwhelming the server
    successful = 0
    with multiprocessing.Pool(num_workers) as pool:
        for result in tqdm(pool.imap(download_func, image_links), 
                          total=len(image_links),
                          desc="Downloading images"):
            if result:
                successful += 1
        pool.close()
        pool.join()
    
    # Final count
    final_count = len([f for f in os.listdir(download_folder) if f.endswith('.jpg')])
    success_rate = (final_count / len(image_links)) * 100
    
    print(f"\n Download complete!")
    print(f"   Downloaded: {final_count}/{len(image_links)} images")
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   New downloads: {final_count - existing_count}")
    
    return final_count

def main():
    """Resume test image downloads"""
    
    print("🔧 Resuming test image downloads...")
    
    # Check if test.csv exists
    test_csv = Path('dataset/test.csv')
    if not test_csv.exists():
        print(" Error: dataset/test.csv not found")
        return
    
    # Load test data
    print("Loading test.csv...")
    df = pd.read_csv(test_csv)
    print(f"Found {len(df)} test image URLs")
    
    # Create images directory
    images_dir = Path('images/test')
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Download images
    image_links = df['image_link'].tolist()
    download_images_robust(image_links, str(images_dir))
    
    print("\n Test image download process complete!")
   
if __name__ == "__main__":
    main()