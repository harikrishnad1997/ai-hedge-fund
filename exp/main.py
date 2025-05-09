from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import re
import pandas as pd
import logging
from datetime import datetime
import time
import os
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stock_fetcher.log")
    ]
)
logger = logging.getLogger("stock_fetcher")

app = FastAPI()
scheduler = BackgroundScheduler()

# Maximum number of retries for API requests
MAX_RETRIES = 3
# Delay between retries (in seconds)
RETRY_DELAY = 5
# Default timeout for requests (in seconds)
REQUEST_TIMEOUT = 10

def make_request_with_retry(url: str) -> requests.Response:
    """Make HTTP request with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempting request to {url} (attempt {attempt+1})")
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed for {url}")
                raise

def get_stock_details(ticker_symbol: str) -> Tuple[str, Any, Optional[int], List[str]]:
    """Get stock details from Ape Wisdom website and Yahoo Finance"""
    logger.info(f"Fetching details for {ticker_symbol}")
    
    # Initialize return values with defaults
    details_text = f"No details found for {ticker_symbol}"
    current_price = None
    positive_percentage = None
    keywords = []
    
    try:
        # Get data from Ape Wisdom
        url = f'https://apewisdom.io/stocks/{ticker_symbol}/'
        response = make_request_with_retry(url)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        details_div = soup.find('div', class_='row', style='font-size: 18px; margin-top: 20px; margin-bottom: 10px;')

        if details_div:
            details_text = details_div.get_text(separator="\n").strip()
            logger.debug(f"Found details for {ticker_symbol}")
            
            # Extract keywords
            match = re.search(r'keywords: (.*)', details_text)
            if match:
                keywords = match.group(1).split(",")
                keywords = [keyword.strip().replace('"', '') for keyword in keywords]
                logger.debug(f"Found {len(keywords)} keywords for {ticker_symbol}")
            
            # Extract positive sentiment percentage
            match = re.search(r'(\d+)% positive comments', details_text)
            if match:
                positive_percentage = int(match.group(1))
                logger.debug(f"Positive sentiment for {ticker_symbol}: {positive_percentage}%")
        else:
            logger.warning(f"Details div not found for {ticker_symbol}")
    
    except Exception as e:
        logger.error(f"Error getting Ape Wisdom data for {ticker_symbol}: {e}")
    
    # Get price data from Yahoo Finance with a separate try/except
    try:
        logger.debug(f"Fetching Yahoo Finance data for {ticker_symbol}")
        stock_data = yf.Ticker(ticker_symbol)
        hist = stock_data.history(period="1d")
        
        # Fixed the empty() check - it's a property, not a method
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            logger.debug(f"Current price for {ticker_symbol}: {current_price}")
        else:
            logger.warning(f"No price history found for {ticker_symbol}")
            current_price = None
    except Exception as e:
        logger.error(f"Error fetching price for {ticker_symbol}: {e}")
        current_price = None

    return details_text, current_price, positive_percentage, keywords

def fetch_details(ticker: str) -> pd.Series:
    """Fetch details for a single ticker and return as a pandas Series"""
    try:
        details, price, positive_pct, keywords = get_stock_details(ticker)
        return pd.Series({
            'current_price': price,
            'positive_sentiment_pct': positive_pct,
            'sentiment_details': details,
            'keywords': keywords
        })
    except Exception as e:
        logger.error(f"Error processing ticker {ticker}: {e}")
        return pd.Series({
            'current_price': None,
            'positive_sentiment_pct': None,
            'sentiment_details': f"Error: {e}",
            'keywords': []
        })

def fetch_stock_data() -> None:
    """Fetch and process stock data from Ape Wisdom API"""
    logger.info("Starting stock data fetch job")
    start_time = time.time()
    
    try:
        url = 'https://apewisdom.io/api/v1.0/filter/all-stocks'
        response = make_request_with_retry(url)
        data = response.json()
        
        if not data or 'results' not in data or not data['results']:
            logger.error("No results found in API response")
            return
        
        logger.info(f"Found {len(data['results'])} stocks in API response")
        
        # Create output directory if it doesn't exist
        output_dir = "stock_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Process in batches to avoid overloading APIs
        df = pd.DataFrame(data['results'])
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basic_filename = os.path.join(output_dir, f"stock_data_basic_{timestamp}.csv")
        
        # Save the basic data first
        df.to_csv(basic_filename, index=False)
        logger.info(f"Basic stock data saved to {basic_filename}")
        
        # Now fetch additional details
        logger.info("Fetching detailed data for each stock...")
        result_list = []
        
        for idx, row in df.iterrows():
            try:
                ticker = row['ticker']
                logger.info(f"Processing {idx+1}/{len(df)}: {ticker}")
                
                detail_series = fetch_details(ticker)
                # Combine original row data with details
                combined = pd.concat([row, detail_series])
                result_list.append(combined)
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to process row {idx}: {e}")
        
        # Create the final detailed dataframe
        if result_list:
            detailed_df = pd.DataFrame(result_list)
            detailed_filename = os.path.join(output_dir, f"stock_data_detailed_{timestamp}.csv")
            detailed_df.to_csv(detailed_filename, index=False)
            logger.info(f"Detailed stock data saved to {detailed_filename}")
        else:
            logger.error("No detailed data was collected")
            
    except Exception as e:
        logger.error(f"An error occurred during data fetching: {e}")
    
    duration = time.time() - start_time
    logger.info(f"Stock data fetch job completed in {duration:.2f} seconds")

# Initialize the scheduler with a misfire grace time
scheduler = BackgroundScheduler(misfire_grace_time=300)

# Routes
@app.get("/")
def read_root():
    return {
        "message": "Stock data fetcher is running.",
        "status": "active" if scheduler.running else "stopped"
    }

@app.get("/run-now")
def run_fetch_now():
    """Endpoint to trigger a fetch immediately"""
    try:
        logger.info("Manual fetch triggered via API")
        fetch_stock_data()
        return {"message": "Stock data fetch initiated successfully"}
    except Exception as e:
        logger.error(f"Manual fetch failed: {e}")
        return {"message": f"Error: {str(e)}"}, 500

@app.get("/status")
def get_status():
    """Endpoint to check service status"""
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "next_run_time": str(job.next_run_time) if job.next_run_time else None,
            "function": job.func.__name__
        })
    
    return {
        "scheduler_running": scheduler.running,
        "jobs": jobs
    }

# Start and shutdown events
@app.on_event("startup")
def startup_event():
    """Startup event handler for FastAPI"""
    logger.info("Starting stock data fetcher application")
    
    # Schedule the job to run daily at 10 AM
    scheduler.add_job(
        fetch_stock_data, 
        'cron', 
        hour=10, 
        minute=0, 
        id='daily_stock_fetch',
        replace_existing=True
    )
    
    # Also run once at startup for testing
    scheduler.add_job(
        fetch_stock_data,
        'date',
        run_date=datetime.now(),
        id='startup_fetch'
    )
    
    scheduler.start()
    logger.info("Scheduler started successfully")

@app.on_event("shutdown")
def shutdown_event():
    """Shutdown event handler for FastAPI"""
    logger.info("Shutting down stock data fetcher")
    scheduler.shutdown()
    logger.info("Scheduler shut down successfully")

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)