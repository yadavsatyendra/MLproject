
import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Define log file name and path
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Set up logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
