from fastapi import FastAPI, Depends
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
import time
from sqlalchemy.exc import OperationalError

# Database URL (use environment variables in production)
DATABASE_URL = "postgresql+psycopg2://postgres:password@db:5432/gibbor_tradingdb"

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL)
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

# Database model
class TradeRecord(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    action = Column(String)  # "buy" or "sell"
    lot_size = Column(Float)

# Pydantic model for trade data
class TradeData(BaseModel):
    symbol: str
    action: str  # "buy" or "sell"
    lot_size: float

# Endpoint to add a trade
@app.post("/trades")
async def add_trade(data: TradeData, db: Session = Depends(get_db)):
    trade_record = TradeRecord(symbol=data.symbol, action=data.action, lot_size=data.lot_size)
    db.add(trade_record)
    db.commit()
    db.refresh(trade_record)
    return {"status": "success", "trade": trade_record}

# Endpoint to list all trades
@app.get("/trades")
async def get_trades(db: Session = Depends(get_db)):
    trades = db.query(TradeRecord).all()
    return {"trades": trades}

# Test endpoint
@app.get("/")
async def root():
    return {"message": "FastAPI app with PostgreSQL is running"}
