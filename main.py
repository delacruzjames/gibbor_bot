from fastapi import FastAPI, Depends, Request, HTTPException
from sqlalchemy import Column, Integer, String, Float, create_engine, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
import time
from sqlalchemy.exc import OperationalError
import os
from datetime import datetime

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

# Endpoint to add a trade
# @app.post("/trades")
# async def add_trade(data: TradeData, db: Session = Depends(get_db)):
#     """
#     Endpoint to handle incoming price data.
#     """
#     # Debug log: Print received data
#     print("Received Data:", data)
#     trade_record = TradeRecord(symbol=data.symbols, action=data.action, lot_size=data.lot_size)
#     db.add(trade_record)
#     db.commit()
#     db.refresh(trade_record)
#     return {"status": "success", "trade": trade_record}

# @app.post("/prices")
# async def add_price(data: PriceData, db: Session = Depends(get_db)):
#     try:
#         # Log the incoming payload
#         print("Received payload:", data.dict())
#
#         # Process the request
#         price_record = Price(
#             symbols=data.symbols,
#             value=data.value,
#             timestamp=data.timestamp
#         )
#         db.add(price_record)
#         db.commit()
#         db.refresh(price_record)
#         return {"status": "success", "price": price_record}
#     except Exception as e:
#         print("Validation or Processing Error:", str(e))
#         raise HTTPException(status_code=422, detail="Invalid request payload")


@app.post("/prices")
async def add_price(request: Request, db: Session = Depends(get_db)):
    try:
        # Read the body as a raw string
        raw_body = await request.body()
        payload = raw_body.decode("utf-8")

        # Log the received raw string payload
        print("Received raw payload:", payload)

        # Parse the raw string (you can add custom parsing logic here if needed)
        if not payload.startswith("{") or not payload.endswith("}"):
            raise HTTPException(status_code=422, detail="Payload must be a valid string containing JSON-like data.")

        # Process the string as a dictionary manually
        data = eval(payload)  # Use eval cautiously; this assumes trusted input
        symbols = data.get("symbols", "")
        value = data.get("value", "")
        timestamp = data.get("timestamp", "")

        if not (symbols and value and timestamp):
            raise HTTPException(status_code=422, detail="Missing required fields in payload.")

        # Create and save the record
        price_record = Price(
            symbols=symbols,
            value=value,
            timestamp=timestamp
        )
        db.add(price_record)
        db.commit()
        db.refresh(price_record)

        return {"status": "success", "price": price_record}
    except Exception as e:
        print("Error processing raw payload:", str(e))
        raise HTTPException(status_code=422, detail="Invalid request payload")

# Endpoint to list all trades
@app.get("/trades")
async def get_trades(db: Session = Depends(get_db)):
    trades = db.query(TradeRecord).all()
    return {"trades": trades}

@app.post("/prices")
async def add_price(data: PriceData, db: Session = Depends(get_db)):
    # Specify static values for each column
    price_record = Price(
        symbols="TEST_SYMBOL",          # Static string for the symbol
        value=1234.56,                  # Static value for the price
        timestamp="2024-01-01 00:00:00" # Static string for the timestamp
    )
    db.add(price_record)
    db.commit()
    db.refresh(price_record)
    return {"status": "success", "price": price_record}

# Endpoint to list all price records
@app.get("/prices")
async def get_prices(db: Session = Depends(get_db)):
    prices = db.query(Price).all()
    return {"prices": prices}

# @app.post("/prices")
# async def add_price(data: PriceData, db: Session = Depends(get_db)):
#     price_record = Price(symbols=data.symbols, value=data.value, timestamp=data.timestamp)
#     db.add(price_record)
#     db.commit()
#     db.refresh(price_record)
#     return {"status": "success", "price": price_record}

# Test endpoint
@app.get("/")
async def root():
    return {"message": "FastAPI app with PostgreSQL is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}