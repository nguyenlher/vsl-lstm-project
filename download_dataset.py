from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

import os
import csv
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import threading
import time

BASE_URL = "https://qipedc.moet.gov.vn"
VIDEOS_DIR = "dataset/videos"
TEXT_DIR = "dataset/labels"
CSV_PATH = os.path.join(TEXT_DIR, "label.csv")
MAX_RETRIES = 3
RETRY_DELAY = 2
REQUEST_TIMEOUT = 30

csv_lock = threading.Lock()
video_counter = 0
counter_lock = threading.Lock()


def init_csv():
    global video_counter
    os.makedirs(TEXT_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "VIDEO", "LABEL"])
        video_counter = 0
    else:

        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            video_counter = sum(1 for _ in f) - 1


def write_to_csv(id, video, label):
    with csv_lock:
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([id, video, label])


def download_video(video_data):
    global video_counter
    video_url = video_data.get('video_url')
    label = video_data.get('label')
    filename = os.path.basename(urlparse(video_url).path)
    output_path = os.path.join(VIDEOS_DIR, filename)
    

    if os.path.exists(output_path):
        print(f"Skip: {filename}")
        return
    

    for attempt in range(MAX_RETRIES):
        try:
            print(f"Downloading: {filename}" + (f" (attempt {attempt + 1}/{MAX_RETRIES})" if attempt > 0 else ""))
            response = requests.get(video_url, stream=True, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as file:
                with tqdm(
                    desc=f"Downloading: {filename[:30]}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    ncols=100
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = file.write(chunk)
                        bar.update(size)
            

            with counter_lock:
                video_counter += 1
                write_to_csv(video_counter, filename, label)
            
            print(f"Completed: {filename}")
            return
            
        except requests.exceptions.Timeout:
            print(f"Timeout {filename}: attempt {attempt + 1}/{MAX_RETRIES}")
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error {filename}: {str(e)}")
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error {filename}: {str(e)}")
            break
        except Exception as e:
            print(f"Unexpected error {filename}: {str(e)}")
        

        if os.path.exists(output_path):
            os.remove(output_path)
        

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    
    print(f"Failed to download {filename} after {MAX_RETRIES} attempts")


def parse_page(driver):
    videos = []
    try:

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "section:nth-of-type(2) > div:nth-of-type(2)"))
        )
        

        video_elements = driver.find_elements(
            By.CSS_SELECTOR,
            "section:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(1) > a"
        )
        
        for vid_elem in video_elements:
            try:
                label = vid_elem.find_element(By.CSS_SELECTOR, "p").text.strip()
                thumbs_url = vid_elem.find_element(By.CSS_SELECTOR, "img").get_attribute("src")
                
                if not thumbs_url:
                    continue
                    
                video_id = thumbs_url.replace("https://qipedc.moet.gov.vn/thumbs/", "").replace(".png", "")
                video_url = f"{BASE_URL}/videos/{video_id}.mp4"
                
                if label and video_url:
                    videos.append({'label': label, 'video_url': video_url})
            except Exception as e:
                print(f"  Warning: Failed to parse video element: {e}")
                continue
        
        return videos
    except Exception as e:
        print(f"  Error parsing page: {e}")
        return []


def get_next_page_button(driver, current_page):
    try:

        buttons = driver.find_elements(By.CSS_SELECTOR, "button")
        

        next_page_str = str(current_page + 1)
        for button in buttons:
            if button.text.strip() == next_page_str:
                return button
        

        for button in buttons:
            if "Last" in button.text or button.text.strip() == "»":
                return None
        
        return None
    except Exception as e:
        print(f"  Error finding next page button: {e}")
        return None


def crawl_videos():
    print("\n" + "="*60)
    print("CRAWLING VSL DICTIONARY")
    print("="*60)
    

    options = Options()
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')

    
    print("Setting up ChromeDriver...")
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    all_videos = []
    current_page = 1
    
    try:
        driver.get("https://qipedc.moet.gov.vn/dictionary")
        print("Connected to VSL dictionary")
        time.sleep(3)
        

        while True:
            print(f"Crawling page {current_page}...")
            videos = parse_page(driver)
            all_videos.extend(videos)
            print(f"  Found {len(videos)} videos (Total: {len(all_videos)})")
            

            next_button = get_next_page_button(driver, current_page)
            
            if next_button is None:
                print(f"\nReached last page: {current_page}")
                break
            
            try:

                driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                time.sleep(0.5)
                next_button.click()
                time.sleep(2)
                current_page += 1
                
            except Exception as e:
                print(f"Cannot navigate to page {current_page + 1}: {e}")
                break
        
        print(f"\nTotal videos found: {len(all_videos)}")
        
    except Exception as e:
        print(f"Crawling error: {e}")
    
    finally:
        driver.quit()
    
    return all_videos


def main():
    print("\nVSL DATASET DOWNLOADER")
    print("="*60)
    

    os.makedirs(VIDEOS_DIR, exist_ok=True)
    init_csv()
    

    videos = crawl_videos()
    
    if not videos:
        print("No videos found")
        return
    

    print("\n" + "="*60)
    print("STARTING DOWNLOAD")
    print("="*60)
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_video, videos)
    
    print(f"\nDownload completed!")
    print(f"Videos: {VIDEOS_DIR}")
    print(f"Labels: {CSV_PATH}")


if __name__ == "__main__":
    main()