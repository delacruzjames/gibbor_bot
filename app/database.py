# app/database.py

import os
import time
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError
from logger import logger

# Database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Please set it in your environment.")

# Correct PostgreSQL URL if necessary
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Retry logic to wait for the database to be ready
MAX_RETRIES = 5
RETRY_INTERVAL = 5  # seconds

for attempt in range(MAX_RETRIES):
    try:
        # Attempt to connect to the database and create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database connected and tables created successfully.")
        break
    except OperationalError as oe:
        if attempt < MAX_RETRIES - 1:
            logger.warning(
                f"Database connection failed on attempt {attempt + 1}. Retrying in {RETRY_INTERVAL} seconds..."
            )
            time.sleep(RETRY_INTERVAL)
        else:
            logger.error("Max retries reached. Exiting.")
            raise oe

# Dependency to get database session
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
