import logging
from logging.handlers import RotatingFileHandler

# Configure logger
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = "app.log"  # Path to the log file

# File handler
file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.ERROR)

# Stream handler (optional: log to console too)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)

# Root logger
logger = logging.getLogger("fastapi_logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
