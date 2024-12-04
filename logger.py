# logger.py

import logging

# Initialize Logger
logger = logging.getLogger("trading_api")
logger.setLevel(logging.INFO)  # Set to DEBUG for more verbose output

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set to DEBUG for more verbose output

# Create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)

# Add handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(console_handler)
