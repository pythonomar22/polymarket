# download_aggtrades.py
import os
import requests
import zipfile
from datetime import date, timedelta, datetime
import time
import logging
from tqdm import tqdm

# --- Configuration ---
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
START_DATE = date(2021, 5, 9) # Keep consistent with your kline data start
END_DATE = date(2025, 5, 9)   # Keep consistent with your kline data end

DATA_TYPE = "aggTrades" # Specifies the type of data to download

# Base URL for daily spot data
BASE_DATA_URL = "https://data.binance.vision/data/spot/daily"

# Main directory to store all data (remains the same)
OUTPUT_DIR_BASE = "binance_market_data"

DOWNLOAD_CHUNK_SIZE = 8192
REQUEST_TIMEOUT = 30
DOWNLOAD_DELAY = 0.5 # Seconds to wait between downloads

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def daterange(start_date, end_date):
    """Generates a sequence of dates from start_date to end_date, inclusive."""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def download_and_extract_aggtrade_data(symbol, target_date, output_dir_symbol_aggtrades):
    """
    Downloads and extracts daily aggTrade data for a given symbol and date.
    Skips if the final CSV file already exists.
    """
    year = target_date.year
    month = str(target_date.month).zfill(2)
    day = str(target_date.day).zfill(2)

    # Filename convention for aggTrades: SYMBOL-aggTrades-YYYY-MM-DD.csv
    filename_base = f"{symbol}-{DATA_TYPE}-{year}-{month}-{day}"
    zip_filename = f"{filename_base}.zip"
    csv_filename = f"{filename_base}.csv" # The CSV inside the zip should have this name

    zip_filepath = os.path.join(output_dir_symbol_aggtrades, zip_filename)
    csv_filepath = os.path.join(output_dir_symbol_aggtrades, csv_filename)

    # Skip if CSV file already exists
    if os.path.exists(csv_filepath):
        logger.info(f"CSV file already exists: {csv_filepath}, skipping download for {DATA_TYPE}.")
        return True

    # Construct URL for aggTrades
    # e.g., https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-01-01.zip
    download_url = f"{BASE_DATA_URL}/{DATA_TYPE}/{symbol}/{zip_filename}"

    try:
        logger.info(f"Attempting to download: {download_url}")
        response = requests.get(download_url, stream=True, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        with open(zip_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
        logger.info(f"Successfully downloaded {zip_filename}")

        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            # Ensure the CSV filename matches what's inside the zip.
            # For aggTrades, it should be SYMBOL-aggTrades-YYYY-MM-DD.csv
            if csv_filename not in zip_ref.namelist():
                logger.error(f"Expected CSV file '{csv_filename}' not found in zip '{zip_filename}'. Files in zip: {zip_ref.namelist()}")
                # Clean up the downloaded zip file if CSV is not found
                if os.path.exists(zip_filepath):
                    os.remove(zip_filepath)
                return False
            zip_ref.extract(csv_filename, output_dir_symbol_aggtrades)
        logger.info(f"Successfully extracted {csv_filename} to {output_dir_symbol_aggtrades}")

        os.remove(zip_filepath)
        logger.info(f"Removed zip file: {zip_filepath}")
        return True

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"{DATA_TYPE} data not found (404) for {symbol} on {target_date}: {download_url}")
        else:
            logger.error(f"HTTP error downloading {download_url}: {e}")
        if os.path.exists(zip_filepath): os.remove(zip_filepath)
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error downloading {download_url}: {e}")
        if os.path.exists(zip_filepath): os.remove(zip_filepath)
        return False
    except zipfile.BadZipFile:
        logger.error(f"Bad zip file: {zip_filepath}. It might be corrupted or not a zip file.")
        if os.path.exists(zip_filepath): os.remove(zip_filepath)
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred for {symbol} on {target_date} for {DATA_TYPE}: {e}")
        if os.path.exists(zip_filepath) and not os.path.exists(csv_filepath):
            os.remove(zip_filepath)
        return False

# --- Main Script Logic ---
if __name__ == "__main__":
    logger.info(f"Starting Binance historical {DATA_TYPE} data download script...")

    # Base output directory for all data types (e.g., spot data)
    output_dir_spot_base = os.path.join(OUTPUT_DIR_BASE, "spot", "daily")
    if not os.path.exists(output_dir_spot_base):
        os.makedirs(output_dir_spot_base)
        logger.info(f"Created base spot data directory: {output_dir_spot_base}")

    # Specific output directory for this data type (aggTrades)
    output_dir_datatype = os.path.join(output_dir_spot_base, DATA_TYPE)
    if not os.path.exists(output_dir_datatype):
        os.makedirs(output_dir_datatype)
        logger.info(f"Created base {DATA_TYPE} directory: {output_dir_datatype}")


    for symbol in SYMBOLS:
        # Directory for each symbol within the aggTrades directory
        # e.g., binance_market_data/spot/daily/aggTrades/BTCUSDT/
        output_dir_symbol_aggtrades = os.path.join(output_dir_datatype, symbol)
        if not os.path.exists(output_dir_symbol_aggtrades):
            os.makedirs(output_dir_symbol_aggtrades)
            logger.info(f"Created symbol directory for {DATA_TYPE}: {output_dir_symbol_aggtrades}")

        logger.info(f"--- Processing symbol: {symbol} for {DATA_TYPE} ---")
        dates_to_process = list(daterange(START_DATE, END_DATE)) # Create list for tqdm
        for target_date in tqdm(dates_to_process, desc=f"{symbol} {DATA_TYPE}", unit="day"):
            # logger.info(f"Requesting {DATA_TYPE} data for {symbol} on {target_date.strftime('%Y-%m-%d')}") # tqdm provides progress
            success = download_and_extract_aggtrade_data(symbol, target_date, output_dir_symbol_aggtrades)
            # if success:
            #     logger.info(f"Successfully processed {symbol} {DATA_TYPE} for {target_date.strftime('%Y-%m-%d')}")
            # else:
            #     logger.warning(f"Failed to process {symbol} {DATA_TYPE} for {target_date.strftime('%Y-%m-%d')}")
            
            time.sleep(DOWNLOAD_DELAY)

    logger.info(f"--- {DATA_TYPE} Download script finished. ---")
    logger.info(f"All {DATA_TYPE} data is stored in subdirectories under: {output_dir_datatype}")