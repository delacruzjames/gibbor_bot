from fastapi import FastAPI, Depends, Request, HTTPException
from sqlalchemy import Column, Integer, String, Float, create_engine, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
import time
from sqlalchemy.exc import OperationalError
import os
import json
from datetime import datetime
from collections import defaultdict

# Fetch PORT from environment or default to 8000 for local testing
PORT = int(os.getenv('PORT', 8000))

# Database URL (use environment variables in production)
# DATABASE_URL = "postgresql+psycopg2://postgres:password@db:5432/gibbor_tradingdb"
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Please set it in your environment.")



# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Retry logic to wait for the database to be ready
MAX_RETRIES = 5
RETRY_INTERVAL = 5

for attempt in range(MAX_RETRIES):
    try:
        # Attempt to connect to the database and create tables
        Base.metadata.create_all(bind=engine)
        print("Database connected and tables created successfully.")
        break
    except OperationalError:
        if attempt < MAX_RETRIES - 1:
            print(f"Database connection failed. Retrying in {RETRY_INTERVAL} seconds...")
            time.sleep(RETRY_INTERVAL)
        else:
            print("Max retries reached. Exiting.")
            raise

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# FastAPI app
app = FastAPI()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)


# Database model
class TradeRecord(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    action = Column(String)  # "buy" or "sell"
    lot_size = Column(Float)

class Price(Base):
    __tablename__ = "prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbols = Column(String, nullable=False, index=True)  # Symbol of the financial instrument
    value = Column(String, nullable=False)  # Target price
    timestamp = Column(String, nullable=False)  # When the price snapshot was recorded

# Pydantic model for trade data
class TradeData(BaseModel):
    symbols: str
    action: str  # "buy" or "sell"
    lot_size: float

class PriceData(BaseModel):
    symbols: str
    value: str
    timestamp: str

from openai import OpenAI
client = OpenAI(api_key="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")

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Query and group by symbol and date
def get_prices_grouped_by_symbol_and_date():
    session = SessionLocal()
    try:
        prices = session.query(Price).all()
        grouped_data = defaultdict(lambda: defaultdict(list))

        for price in prices:
            date = datetime.fromisoformat(price.timestamp.replace("Z", "")).date()
            grouped_data[price.symbols][date].append(float(price.value))

        # Format grouped data into the desired output
        formatted_result = []
        for symbol, dates in grouped_data.items():
            for date, values in dates.items():
                formatted_result.append(f"{symbol}: {date} Values {values}")

        return formatted_result
    finally:
        session.close()


@app.post("/chat")
async def chat_with_gpt(request: Request):
    """
    Chat with OpenAI GPT model.
    """
    try:
        # Parse the request body
        body = await request.json()

        # Call the function to get grouped prices
        grouped_prices = get_prices_grouped_by_symbol_and_date()
        logger.info(f"Grouped Prices: {grouped_prices}")

        # Check if grouped_prices is empty or None
        if not grouped_prices:
            raise HTTPException(status_code=500, detail="No data returned from get_prices_grouped_by_symbol_and_date")

        # Format the grouped prices into the desired format
        formatted_grouped_prices = "\n".join(grouped_prices)  # Join all entries with newlines

        # Generate a response using OpenAI client
        completion = client.chat.completions.create(
            model="gpt-4-turbo",  # Replace with the correct model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": formatted_grouped_prices},
                {"role": "user", "content": "Based on the EUR/USD price data, can you recommend a trade execution or a pending order"},
            ]
        )

        # Extract the AI's response
        ai_message = completion.choices[0].message

        # Log the AI's response
        logger.info(f"AI Response: {ai_message}")

        return {"status": "success", "grouped_prices": ai_message}

    except Exception as e:
        # Handle unexpected errors
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Endpoint to add a trade
@app.post("/trades")
async def add_trade(data: TradeData, db: Session = Depends(get_db)):
    """
    Endpoint to handle incoming price data.
    """
    # Debug log: Print received data
    print("Received Data:", data)
    trade_record = TradeRecord(symbol=data.symbols, action=data.action, lot_size=data.lot_size)
    db.add(trade_record)
    db.commit()
    db.refresh(trade_record)
    return {"status": "success", "trade": trade_record}


# API endpoint to add a price record
@app.post("/prices")
async def add_price(request: Request, db: Session = Depends(get_db)):
    try:
        # Read raw body
        raw_body = await request.body()

        # Decode bytes to string and parse JSON
        data = json.loads(raw_body.decode("utf-8").strip("\x00"))
        print("Received JSON Data:", data)

        # Validate required fields
        symbols = data.get("symbols")
        value = data.get("value")
        timestamp = data.get("timestamp")

        if not symbols or not value or not timestamp:
            raise HTTPException(status_code=400, detail="Missing required fields: symbols, value, or timestamp")

        # Save to database
        price_record = Price(symbols=symbols, value=value, timestamp=timestamp)
        db.add(price_record)
        db.commit()
        db.refresh(price_record)

        # Return success response
        return {
            "status": "success",
            "price": {
                "id": price_record.id,
                "symbols": symbols,
                "value": value,
                "timestamp": timestamp
            }
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        print("Error processing request:", str(e))
        raise HTTPException(status_code=422, detail="Invalid request payload")


# Endpoint to list all trades
@app.get("/trades")
async def get_trades(db: Session = Depends(get_db)):
    trades = db.query(TradeRecord).all()
    return {"trades": trades}

# @app.post("/prices")
# async def add_price(data: PriceData, db: Session = Depends(get_db)):
#     # Specify static values for each column
#     price_record = Price(
#         symbols="TEST_SYMBOL",          # Static string for the symbol
#         value=1234.56,                  # Static value for the price
#         timestamp="2024-01-01 00:00:00" # Static string for the timestamp
#     )
#     db.add(price_record)
#     db.commit()
#     db.refresh(price_record)
#     return {"status": "success", "price": price_record}

# Endpoint to list all price records
@app.get("/prices")
async def get_prices(db: Session = Depends(get_db)):
    prices = db.query(Price).all()
    return {"prices": prices}


@app.delete("/clear_prices")
async def clear_prices(db: Session = Depends(get_db)):
    try:
        # Delete all records in the prices table
        db.query(Price).delete()
        db.commit()
        return {"status": "success", "message": "All price records have been cleared."}
    except Exception as e:
        print("Error clearing price records:", str(e))
        raise HTTPException(status_code=500, detail="Failed to clear price records")

# Test endpoint
@app.get("/")
async def root():
    return {"message": "FastAPI app with PostgreSQL is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}