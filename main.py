import os
import json
import time
import logging
from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI, Depends, Request, HTTPException
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib  # For saving and loading the model

# Technical analysis library
import ta  # Install with `pip install ta`

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch PORT from environment or default to 8000 for local testing
PORT = int(os.getenv('PORT', 8000))

# Database URL (use environment variables in production)
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
        logger.info("Database connected and tables created successfully.")
        break
    except OperationalError:
        if attempt < MAX_RETRIES - 1:
            logger.warning(f"Database connection failed. Retrying in {RETRY_INTERVAL} seconds...")
            time.sleep(RETRY_INTERVAL)
        else:
            logger.error("Max retries reached. Exiting.")
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

# Database models
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

# Pydantic models for request validation
class TradeData(BaseModel):
    symbols: str
    action: str  # "buy" or "sell"
    lot_size: float

class PriceData(BaseModel):
    symbols: str
    value: str
    timestamp: str

# Utility function to get prices from the database and return as a DataFrame
def get_price_data(symbol: str):
    session = SessionLocal()
    try:
        prices = session.query(Price).filter(Price.symbols == symbol).order_by(Price.timestamp).all()
        data = pd.DataFrame([{
            'timestamp': datetime.fromisoformat(price.timestamp.replace("Z", "")),
            'value': float(price.value)
        } for price in prices])
        return data
    finally:
        session.close()

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    data.set_index('timestamp', inplace=True)

    # Calculate RSI
    data['rsi'] = ta.momentum.RSIIndicator(close=data['value'], window=14).rsi()

    # Calculate Moving Averages
    data['ma_50'] = data['value'].rolling(window=50).mean()
    data['ma_200'] = data['value'].rolling(window=200).mean()

    # Calculate Bollinger Bands (as an example of support and resistance)
    bollinger = ta.volatility.BollingerBands(close=data['value'], window=20, window_dev=2)
    data['bb_middle'] = bollinger.bollinger_mavg()
    data['bb_upper'] = bollinger.bollinger_hband()
    data['bb_lower'] = bollinger.bollinger_lband()

    data.reset_index(inplace=True)
    return data

# Function to train the machine learning model
def train_model(data):
    # Prepare features and target
    data.dropna(inplace=True)  # Drop rows with NaN values resulting from indicator calculations

    # Shift the target variable to predict the next price
    data['target'] = data['value'].shift(-1)
    data.dropna(inplace=True)

    features = ['value', 'rsi', 'ma_50', 'ma_200', 'bb_middle', 'bb_upper', 'bb_lower']
    X = data[features]
    y = data['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'price_prediction_model.pkl')

    logger.info("Model trained and saved successfully.")
    return model

# Function to load the trained model
def load_model():
    if not os.path.exists('price_prediction_model.pkl'):
        logger.error("Model file not found. Please train the model first.")
        raise FileNotFoundError("Model file not found.")
    model = joblib.load('price_prediction_model.pkl')
    return model

# Function to generate trading signal based on predictions and indicators
def generate_trading_signal(current_price, predicted_price, indicators):
    action = 'hold'

    rsi = indicators['rsi']
    price_difference = predicted_price - current_price
    price_difference_percentage = (price_difference / current_price) * 100

    logger.info(f"Current Price: {current_price}")
    logger.info(f"Predicted Price: {predicted_price}")
    logger.info(f"Price Difference (%): {price_difference_percentage:.2f}%")
    logger.info(f"RSI: {rsi}")

    # Thresholds
    significant_move = 0.2  # Adjust this threshold as needed
    overbought = 70
    oversold = 30

    if predicted_price > current_price:
        if price_difference_percentage >= significant_move and rsi > overbought:
            action = 'buy-stop'
        elif rsi < oversold:
            action = 'buy-limit'
    elif predicted_price < current_price:
        if abs(price_difference_percentage) >= significant_move and rsi < oversold:
            action = 'sell-stop'
        elif rsi > overbought:
            action = 'sell-limit'

    logger.info(f"Generated Action: {action}")
    return action

# Endpoint to add a price record
@app.post("/prices")
async def add_price(request: Request, db: Session = Depends(get_db)):
    try:
        # Read raw body
        raw_body = await request.body()

        # Decode bytes to string and parse JSON
        data = json.loads(raw_body.decode("utf-8").strip("\x00"))
        logger.info(f"Received JSON Data: {data}")

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
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=422, detail="Invalid request payload")

# Endpoint to add a trade
@app.post("/trades")
async def add_trade(data: TradeData, db: Session = Depends(get_db)):
    logger.info(f"Received Trade Data: {data}")
    trade_record = TradeRecord(symbol=data.symbols, action=data.action, lot_size=data.lot_size)
    db.add(trade_record)
    db.commit()
    db.refresh(trade_record)
    return {"status": "success", "trade": {
        "id": trade_record.id,
        "symbol": trade_record.symbol,
        "action": trade_record.action,
        "lot_size": trade_record.lot_size
    }}

# Endpoint to list all price records
@app.get("/prices")
async def get_prices(db: Session = Depends(get_db)):
    prices = db.query(Price).all()
    price_list = [{
        "id": price.id,
        "symbols": price.symbols,
        "value": price.value,
        "timestamp": price.timestamp
    } for price in prices]
    return {"prices": price_list}

# Endpoint to list all trades
@app.get("/trades")
async def get_trades(db: Session = Depends(get_db)):
    trades = db.query(TradeRecord).all()
    trade_list = [{
        "id": trade.id,
        "symbol": trade.symbol,
        "action": trade.action,
        "lot_size": trade.lot_size
    } for trade in trades]
    return {"trades": trade_list}

# Endpoint to clear all price records
@app.delete("/clear_prices")
async def clear_prices(db: Session = Depends(get_db)):
    try:
        # Delete all records in the prices table
        db.query(Price).delete()
        db.commit()
        logger.info("All price records have been cleared.")
        return {"status": "success", "message": "All price records have been cleared."}
    except Exception as e:
        logger.error(f"Error clearing price records: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear price records")

# Endpoint to interact with the model and get trading recommendations
@app.post("/chat")
async def chat_with_model(request: Request):
    try:
        # Parse the request body
        body = await request.json()
        symbol = body.get('symbol', 'EURUSD')  # Default to 'EURUSD' if not provided

        # Fetch and preprocess data
        data = get_price_data(symbol)
        data = calculate_technical_indicators(data)

        # Load or train the model
        if not os.path.exists('price_prediction_model.pkl'):
            model = train_model(data)
        else:
            model = load_model()

        # Prepare the latest data point for prediction
        latest_data = data.iloc[-1]
        features = ['value', 'rsi', 'ma_50', 'ma_200', 'bb_middle', 'bb_upper', 'bb_lower']
        input_data = latest_data[features].values.reshape(1, -1)

        # Make prediction
        predicted_price = model.predict(input_data)[0]
        current_price = latest_data['value']

        # Generate trading signal
        indicators = latest_data.to_dict()
        action = generate_trading_signal(current_price, predicted_price, indicators)

        # Prepare Stop Loss and Take Profit
        sl = None
        tp = None
        if action in ['buy-limit', 'buy-stop']:
            sl = predicted_price * 0.995  # Stop Loss at 0.5% below entry
            tp = predicted_price * 1.005  # Take Profit at 0.5% above entry
        elif action in ['sell-limit', 'sell-stop']:
            sl = predicted_price * 1.005  # Stop Loss at 0.5% above entry
            tp = predicted_price * 0.995  # Take Profit at 0.5% below entry

        response = {
            "action": action,
            "entry": f"{predicted_price:.5f}" if action != 'hold' else None,
            "sl": f"{sl:.5f}" if sl else None,
            "tp": f"{tp:.5f}" if tp else None
        }

        return {"status": "success", "data": response}

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "FastAPI app with PostgreSQL is running"}

# Run the app with Uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
