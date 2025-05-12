import os
import requests
import zipfile
from datetime import date, timedelta, datetime
import time
import logging
from tqdm import tqdm

# --- Configuration ---
# Adjust SYMBOLS and DATES as per your evolving needs.
# Given the scenario date is May 10, 2025.
# Data for May 9, 2025, should be the latest complete daily file available.
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL = "1m" # We need 1-minute klines
START_DATE = date(2021, 5, 9)
END_DATE = date(2025, 5, 9) # Data for this day should be available on May 10th

BASE_URL = "https://data.binance.vision/data/spot/daily/klines"
OUTPUT_DIR_BASE = "binance_market_data" # Main directory to store all data
DOWNLOAD_CHUNK_SIZE = 8192 # For downloading files
REQUEST_TIMEOUT = 30 # Seconds
DOWNLOAD_DELAY = 0.5 # Seconds to wait between downloads to be polite

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def daterange(start_date, end_date):
    """Generates a sequence of dates from start_date to end_date, inclusive."""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def download_and_extract_kline_data(symbol, interval, target_date, output_dir_symbol):
    """
    Downloads and extracts daily kline data for a given symbol, interval, and date.
    Skips if the final CSV file already exists.
    """
    year = target_date.year
    month = str(target_date.month).zfill(2)
    day = str(target_date.day).zfill(2)

    filename_base = f"{symbol}-{interval}-{year}-{month}-{day}"
    zip_filename = f"{filename_base}.zip"
    csv_filename = f"{filename_base}.csv"

    zip_filepath = os.path.join(output_dir_symbol, zip_filename)
    csv_filepath = os.path.join(output_dir_symbol, csv_filename)

    # Skip if CSV file already exists
    if os.path.exists(csv_filepath):
        logging.info(f"CSV file already exists: {csv_filepath}, skipping download.")
        return True

    # Construct URL
    # e.g., https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2023-01-01.zip
    download_url = f"{BASE_URL}/{symbol}/{interval}/{zip_filename}"

    try:
        # Download the zip file
        logging.info(f"Attempting to download: {download_url}")
        response = requests.get(download_url, stream=True, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)

        with open(zip_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
        logging.info(f"Successfully downloaded {zip_filename}")

        # Extract the CSV file from the zip
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            # Assuming the CSV inside the zip has the same name as filename_base + .csv
            # Sometimes the internal name might vary slightly, but typically it's consistent.
            # If not, one might need to inspect zip_ref.namelist()
            zip_ref.extract(csv_filename, output_dir_symbol)
        logging.info(f"Successfully extracted {csv_filename} to {output_dir_symbol}")

        # Delete the zip file after successful extraction
        os.remove(zip_filepath)
        logging.info(f"Removed zip file: {zip_filepath}")
        return True

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(f"Data not found (404) for {symbol} on {target_date}: {download_url}")
        else:
            logging.error(f"HTTP error downloading {download_url}: {e}")
        # Clean up partial zip file if it exists on error
        if os.path.exists(zip_filepath):
            os.remove(zip_filepath)
        return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error downloading {download_url}: {e}")
        if os.path.exists(zip_filepath):
            os.remove(zip_filepath)
        return False
    except zipfile.BadZipFile:
        logging.error(f"Bad zip file: {zip_filepath}. It might be corrupted or not a zip file.")
        if os.path.exists(zip_filepath): # Remove corrupted zip
             os.remove(zip_filepath)
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred for {symbol} on {target_date}: {e}")
        if os.path.exists(zip_filepath) and not os.path.exists(csv_filepath): # If zip exists but CSV wasn't extracted
            os.remove(zip_filepath)
        return False

# --- Main Script Logic ---
if __name__ == "__main__":
    logging.info("Starting Binance historical kline data download script...")

    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE)
        logging.info(f"Created base output directory: {OUTPUT_DIR_BASE}")

    for symbol in SYMBOLS:
        output_dir_symbol = os.path.join(OUTPUT_DIR_BASE, "spot", "daily", "klines_1m", symbol)
        if not os.path.exists(output_dir_symbol):
            os.makedirs(output_dir_symbol)
            logging.info(f"Created symbol directory: {output_dir_symbol}")

        logging.info(f"--- Processing symbol: {symbol} ---")
        for target_date in tqdm(daterange(START_DATE, END_DATE), desc=f"{symbol}"):
            logging.info(f"Requesting data for {symbol} on {target_date.strftime('%Y-%m-%d')}")
            success = download_and_extract_kline_data(symbol, INTERVAL, target_date, output_dir_symbol)
            if success:
                logging.info(f"Successfully processed {symbol} for {target_date.strftime('%Y-%m-%d')}")
            else:
                logging.warning(f"Failed to process {symbol} for {target_date.strftime('%Y-%m-%d')}")
            
            time.sleep(DOWNLOAD_DELAY)

    logging.info("--- Download script finished. ---")
    logging.info(f"All data is stored in subdirectories under: {os.path.join(OUTPUT_DIR_BASE, 'spot', 'daily', 'klines_1m')}")
