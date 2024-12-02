import os
import time
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Request
from sqlalchemy import Column, Integer, String, Float, create_engine, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.exc import OperationalError
from fastapi.responses import JSONResponse
from logger import logger

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib  # For saving and loading the model

# Technical analysis library
import ta  # Install with `pip install ta`

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
    except OperationalError as oe:
        if attempt < MAX_RETRIES - 1:
            logger.warning(
                f"Database connection failed on attempt {attempt + 1}. Retrying in {RETRY_INTERVAL} seconds...")
            time.sleep(RETRY_INTERVAL)
        else:
            logger.error("Max retries reached. Exiting.")
            raise oe


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
    symbol = Column(String, index=True, nullable=False)
    action = Column(String, nullable=False)  # "buy" or "sell"
    lot_size = Column(Float, nullable=False)


class Price(Base):
    __tablename__ = "prices"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)  # Symbol of the financial instrument
    value = Column(Float, nullable=False)  # Changed to Float
    timestamp = Column(DateTime, nullable=False)  # Correct data type


# Pydantic models for request validation
class TradeData(BaseModel):
    symbol: str
    action: str  # "buy" or "sell"
    lot_size: float


class PriceCreate(BaseModel):
    symbol: str
    value: float
    timestamp: datetime


class ChatRequest(BaseModel):
    symbol: Optional[str] = Field(default="EURUSD")


# Utility function to get prices from the database and return as a DataFrame
def get_price_data(symbol: str) -> pd.DataFrame:
    session = SessionLocal()
    try:
        prices = session.query(Price).filter(Price.symbol == symbol).order_by(Price.timestamp).all()
        logger.debug(f"Fetched {len(prices)} price records for symbol {symbol}.")
        if not prices:
            logger.warning(f"No price data found for symbol: {symbol}")
            return pd.DataFrame()  # Return empty DataFrame if no data
        data = pd.DataFrame([{
            'symbol': price.symbol,
            'value': price.value,
            'timestamp': price.timestamp
        } for price in prices])
        logger.debug(f"Data fetched for symbol {symbol}: {data.head()}")
        return data
    finally:
        session.close()


# Function to calculate technical indicators
def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        logger.error("No data available to calculate technical indicators.")
        return data

    data.set_index('timestamp', inplace=True)
    logger.debug("Calculating technical indicators.")

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
    logger.debug("Technical indicators calculated.")
    return data


# Function to train the machine learning model
def train_model(data: pd.DataFrame) -> GradientBoostingRegressor:
    if data.empty:
        logger.error("No data available to train the model.")
        raise ValueError("Insufficient data to train the model.")

    # Prepare features and target
    initial_length = len(data)
    data.dropna(inplace=True)  # Drop rows with NaN values resulting from indicator calculations
    logger.debug(f"Data length after dropping NaNs: {len(data)} (Initial: {initial_length})")

    if 'target' in data.columns:
        data.drop(columns=['target'], inplace=True)

    # Shift the target variable to predict the next price
    data['target'] = data['value'].shift(-1)
    data.dropna(inplace=True)
    logger.debug(f"Data length after shifting target: {len(data)}")

    features = ['value', 'rsi', 'ma_50', 'ma_200', 'bb_middle', 'bb_upper', 'bb_lower']
    X = data[features]
    y = data['target']

    logger.debug(f"Feature set size: {X.shape}")
    logger.debug(f"Target set size: {y.shape}")

    # Check if there are enough samples
    if len(X) < 10:
        logger.error(f"Insufficient data for training. Only {len(X)} samples available.")
        raise ValueError("Insufficient data to train the model.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    logger.debug(f"Training set size: {X_train.shape[0]} samples")
    logger.debug(f"Test set size: {X_test.shape[0]} samples")

    if X_train.empty or y_train.empty:
        logger.error("Training set is empty after split. Cannot train model.")
        raise ValueError("Insufficient data to train the model.")

    # Train model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'price_prediction_model.pkl')

    logger.info("Model trained and saved successfully.")
    return model


# Function to load the trained model
def load_model() -> GradientBoostingRegressor:
    if not os.path.exists('price_prediction_model.pkl'):
        logger.error("Model file not found. Please train the model first.")
        raise FileNotFoundError("Model file not found.")
    model = joblib.load('price_prediction_model.pkl')
    return model


# Function to generate trading signal based on predictions and indicators
def generate_trading_signal(current_price: float, predicted_price: float, indicators: dict) -> str:
    action = 'hold'

    rsi = indicators.get('rsi')
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
        if price_difference_percentage >= significant_move and rsi and rsi > overbought:
            action = 'buy-stop'
        elif rsi and rsi < oversold:
            action = 'buy-limit'
    elif predicted_price < current_price:
        if abs(price_difference_percentage) >= significant_move and rsi and rsi < oversold:
            action = 'sell-stop'
        elif rsi and rsi > overbought:
            action = 'sell-limit'

    logger.info(f"Generated Action: {action}")
    return action


@app.middleware("http")
async def log_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)

import json

# Endpoint to add a price record
@app.post("/prices", response_model=dict)
async def add_price(price: Request, db: Session = Depends(get_db)):
    raw_body = await price.body()

    try:
        # Decode and parse JSON
        decoded_body = raw_body.decode("utf-8").strip()
        cleaned_body = decoded_body.replace("\x00", "")
        data = json.loads(cleaned_body)

        # Access dictionary keys
        symbol = data["symbol"]
        value = data["value"]
        timestamp = data["timestamp"]

        # Save to database
        price_record = Price(
            symbol=symbol,
            value=value,
            timestamp=timestamp
        )
        db.add(price_record)
        db.commit()
        db.refresh(price_record)

       
        return {
            "status": "success",
            "data": {
                "symbol": symbol,
                "value": value,
                "timestamp": timestamp
            }
        }
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except KeyError as e:
        print(f"KeyError: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Missing key in JSON: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# Endpoint to add a trade
@app.post("/trades", response_model=dict)
async def add_trade(trade: TradeData, db: Session = Depends(get_db)):
    try:
        logger.info(f"Received Trade Data: {trade}")
        trade_record = TradeRecord(
            symbol=trade.symbol,
            action=trade.action,
            lot_size=trade.lot_size
        )
        db.add(trade_record)
        db.commit()
        db.refresh(trade_record)
        return {
            "status": "success",
            "trade": {
                "id": trade_record.id,
                "symbol": trade_record.symbol,
                "action": trade_record.action,
                "lot_size": trade_record.lot_size
            }
        }
    except Exception as e:
        logger.error(f"Error processing trade: {str(e)}")
        raise HTTPException(status_code=422, detail="Invalid trade data")


# Endpoint to list all price records
@app.get("/prices", response_model=dict)
async def get_prices(db: Session = Depends(get_db)):
    try:
        prices = db.query(Price).all()
        if not prices:
            return {"prices": []}
        price_list = [{
            "id": price.id,
            "symbol": price.symbol,
            "value": price.value,
            "timestamp": price.timestamp.isoformat()
        } for price in prices]
        return {"prices": price_list}
    except Exception as e:
        logger.error(f"Error retrieving prices: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve prices")


# Endpoint to list all trades
@app.get("/trades", response_model=dict)
async def get_trades(db: Session = Depends(get_db)):
    try:
        trades = db.query(TradeRecord).all()
        if not trades:
            return {"trades": []}
        trade_list = [{
            "id": trade.id,
            "symbol": trade.symbol,
            "action": trade.action,
            "lot_size": trade.lot_size
        } for trade in trades]
        return {"trades": trade_list}
    except Exception as e:
        logger.error(f"Error retrieving trades: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trades")


# Endpoint to clear all price records
@app.delete("/clear_prices", response_model=dict)
async def clear_prices(db: Session = Depends(get_db)):
    try:
        # Delete all records in the prices table
        deleted = db.query(Price).delete()
        db.commit()
        logger.info(f"All price records have been cleared. Total deleted: {deleted}")
        return {
            "status": "success",
            "message": f"All price records have been cleared. Total deleted: {deleted}"
        }
    except Exception as e:
        logger.error(f"Error clearing price records: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear price records")


# Endpoint to interact with the model and get trading recommendations
@app.post("/chat", response_model=dict)
async def chat_with_model(chat_request: ChatRequest):
    try:
        symbol = chat_request.symbol
        logger.info(f"Received symbol: {symbol}")

        # Fetch and preprocess data
        data = get_price_data(symbol)
        if data.empty:
            logger.error(f"No price data found for symbol: {symbol}")
            raise HTTPException(status_code=404, detail=f"No price data found for symbol: {symbol}")
        logger.info(f"Fetched {len(data)} records for symbol {symbol}.")
        data = calculate_technical_indicators(data)
        logger.info("Technical indicators calculated.")

        # Load or train the model
        if not os.path.exists('price_prediction_model.pkl'):
            logger.info("Model file not found. Training model...")
            try:
                model = train_model(data)
                logger.info("Model trained successfully.")
            except ValueError as ve:
                logger.error(f"Model training failed: {str(ve)}")
                raise HTTPException(status_code=400, detail=str(ve))
        else:
            logger.info("Loading existing model...")
            model = load_model()
            logger.info("Model loaded successfully.")

        # Prepare the latest data point for prediction
        latest_data = data.iloc[-1]
        features = ['value', 'rsi', 'ma_50', 'ma_200', 'bb_middle', 'bb_upper', 'bb_lower']
        input_data = latest_data[features].values.reshape(1, -1)
        logger.debug(f"Input data for prediction: {input_data}")

        # Make prediction
        predicted_price = model.predict(input_data)[0]
        current_price = latest_data['value']
        logger.info(f"Predicted price: {predicted_price}, Current price: {current_price}")

        # Generate trading signal
        indicators = latest_data.to_dict()
        action = generate_trading_signal(current_price, predicted_price, indicators)
        logger.info(f"Generated action: {action}")

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

        logger.info(f"Response prepared: {response}")
        return {"status": "success", "data": response}

    except HTTPException as http_exc:
        raise http_exc  # Re-raise HTTP exceptions to be handled by FastAPI
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# Health check endpoint
@app.get("/health", response_model=dict)
def health_check():
    return {"status": "healthy"}


# Root endpoint
@app.get("/", response_model=dict)
async def root():
    return {"message": "FastAPI app with PostgreSQL is running"}


# Run the app with Uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
