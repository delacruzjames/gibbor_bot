from fastapi import FastAPI, Depends
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
import time
from sqlalchemy.exc import OperationalError
import os

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
    value = Column(Float, nullable=False)  # Target price
    timestamp = Column(String, nullable=False)  # When the price snapshot was recorded

# Pydantic model for trade data
class TradeData(BaseModel):
    symbols: str
    action: str  # "buy" or "sell"
    lot_size: float

class PriceData(BaseModel):
    symbols: str
    value: float
    timestamp: str

# Endpoint to add a trade
@app.post("/trades")
async def add_trade(data: TradeData, db: Session = Depends(get_db)):
    trade_record = TradeRecord(symbol=data.symbols, action=data.action, lot_size=data.lot_size)
    db.add(trade_record)
    db.commit()
    db.refresh(trade_record)
    return {"status": "success", "trade": trade_record}

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