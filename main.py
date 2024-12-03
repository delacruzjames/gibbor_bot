import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

from fastapi import FastAPI, Depends, HTTPException, Request
from sqlalchemy import Column, Integer, String, Float, create_engine, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field
from sqlalchemy.exc import OperationalError
from fastapi.responses import JSONResponse
from logger import logger
from dateutil.parser import parse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib  # For saving and loading the model

# Technical analysis library
import ta  # Install with `pip install ta`

# Fetch PORT from environment or default to 8000 for local testing
PORT = int(os.getenv('PORT', 8000))

# Database URL (use environment variables in production)
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

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
                f"Database connection failed on attempt {attempt + 1}. Retrying in {RETRY_INTERVAL} seconds..."
            )
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
    value = Column(Float, nullable=False)
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
    # Fetch data from 60 days ago
    start_date = datetime.now() - timedelta(days=60)
    logger.debug(f"Fetching price data for symbol {symbol} since {start_date}")

    session = SessionLocal()
    try:
        # Modify query for case-insensitive symbol matching
        prices = session.query(Price).filter(
            func.lower(Price.symbol) == symbol.lower(),
            Price.timestamp >= start_date
        ).order_by(Price.timestamp).all()

        logger.debug(f"Number of price records retrieved: {len(prices)}")

        if not prices:
            logger.warning(f"No price data found for symbol: {symbol} in the specified date range.")
            return pd.DataFrame()

        # Convert queried data into a DataFrame
        data = pd.DataFrame([{
            'timestamp': price.timestamp,
            'value': price.value,
            'symbol': price.symbol
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

    # Calculate EMAs
    data['ema_9'] = ta.trend.EMAIndicator(close=data['value'], window=9).ema_indicator()
    data['ema_21'] = ta.trend.EMAIndicator(close=data['value'], window=21).ema_indicator()

    # Calculate RSI for confirmation (optional)
    data['rsi'] = ta.momentum.RSIIndicator(close=data['value'], window=14).rsi()

    data.reset_index(inplace=True)
    logger.debug("Technical indicators calculated.")
    return data


# Function to train the machine learning model
def train_model(data: pd.DataFrame) -> GradientBoostingRegressor:
    if data.empty:
        logger.error("No data available to train the model.")
        raise ValueError("Insufficient data to train the model.")

    # Prepare features and target
    data.dropna(inplace=True)  # Drop rows with NaN values resulting from indicator calculations

    # Shift the target variable to predict the next price
    data['target'] = data['value'].shift(-1)
    data.dropna(inplace=True)

    features = ['value', 'ema_9', 'ema_21', 'rsi']
    X = data[features]
    y = data['target']

    # Check if there are enough samples
    if len(X) < 10:
        logger.error(f"Insufficient data for training. Only {len(X)} samples available.")
        raise ValueError("Insufficient data to train the model.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if X_train.empty or y_train.empty:
        logger.error("Training set is empty after split. Cannot train model.")
        raise ValueError("Insufficient data to train the model.")

    # Train model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Save the model with features
    model_info = {
        'model': model,
        'features': features
    }
    joblib.dump(model_info, 'price_prediction_model.pkl')
    logger.info("Model trained and saved successfully.")
    return model


# Function to load the trained model
def load_model() -> Tuple[GradientBoostingRegressor, list]:
    if not os.path.exists('price_prediction_model.pkl'):
        logger.error("Model file not found. Please train the model first.")
        raise FileNotFoundError("Model file not found.")
    model_info = joblib.load('price_prediction_model.pkl')

    # Add debug statements
    logger.debug(f"Loaded model_info type: {type(model_info)}")
    if isinstance(model_info, dict):
        logger.debug(f"Model info keys: {model_info.keys()}")
        return model_info['model'], model_info['features']
    else:
        logger.error("Loaded model_info is not a dictionary.")
        raise ValueError("Loaded model_info is not a dictionary. Please retrain the model.")


# Function to generate trading signal based on Moving Average Crossover Strategy
def generate_trading_signal(current_price: float, predicted_price: float, indicators: dict) -> str:
    action = 'hold'

    ema_9 = indicators.get('ema_9')
    ema_21 = indicators.get('ema_21')
    previous_ema_9 = indicators.get('previous_ema_9')
    previous_ema_21 = indicators.get('previous_ema_21')
    rsi = indicators.get('rsi')

    logger.info(f"Current Price: {current_price}")
    logger.info(f"Predicted Price: {predicted_price}")
    logger.info(f"EMA 9: {ema_9}, EMA 21: {ema_21}")
    logger.info(f"Previous EMA 9: {previous_ema_9}, Previous EMA 21: {previous_ema_21}")
    logger.info(f"RSI: {rsi}")

    # Identify crossover
    if previous_ema_9 is not None and previous_ema_21 is not None:
        # Buy Signal: EMA 9 crosses above EMA 21
        if previous_ema_9 < previous_ema_21 and ema_9 > ema_21:
            if rsi and rsi > 50:
                action = 'buy'
        # Sell Signal: EMA 9 crosses below EMA 21
        elif previous_ema_9 > previous_ema_21 and ema_9 < ema_21:
            if rsi and rsi < 50:
                action = 'sell'

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

        # Convert timestamp to datetime object
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

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
                "timestamp": timestamp.isoformat()
            }
        }
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except KeyError as e:
        print(f"KeyError: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Missing key in JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}", exc_info=True)
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
        price_list = []
        for price in prices:
            timestamp = price.timestamp
            # Ensure timestamp is a datetime object
            if isinstance(timestamp, str):
                try:
                    timestamp = parse(timestamp)
                except (ValueError, TypeError) as e:
                    logger.error(f"Error parsing timestamp: {e}")
                    raise HTTPException(status_code=500, detail="Error parsing timestamp from database")
            price_list.append({
                "id": price.id,
                "symbol": price.symbol,
                "value": price.value,
                "timestamp": timestamp.isoformat()
            })
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
@app.post("/open", response_model=dict)
async def chat_with_model(chat_request: Request):
    raw_body = await chat_request.body()
    print(f"Raw Body Received: {raw_body}")

    try:
        decoded_body = raw_body.decode("utf-8").strip()
        cleaned_body = decoded_body.replace("\x00", "")
        symbol = json.loads(cleaned_body)["symbol"]
        data = get_price_data(symbol)

        if data.empty:
            logger.error(f"No price data found for symbol: {symbol}")
            raise HTTPException(status_code=404, detail=f"No price data found for symbol: {symbol}")
        logger.info(f"Fetched {len(data)} records for symbol {symbol}.")
        data = calculate_technical_indicators(data)
        logger.info("Technical indicators calculated.")

        # Handle NaN values
        data.ffill(inplace=True)
        data.dropna(inplace=True)

        # Add previous EMAs for crossover detection
        data['previous_ema_9'] = data['ema_9'].shift(1)
        data['previous_ema_21'] = data['ema_21'].shift(1)
        data.dropna(inplace=True)

        # Ensure there is enough data after processing
        if data.empty or len(data) < 2:
            logger.error(f"Not enough data points available after processing. Data length: {len(data)}")
            raise HTTPException(status_code=400, detail="Not enough data points available for analysis.")

        # Prepare the latest data point for prediction
        latest_data = data.iloc[-1]
        features = ['value', 'ema_9', 'ema_21', 'rsi']

        # Load or train the model
        try:
            model, model_features = load_model()
            if model_features != features:
                logger.warning("Feature mismatch between model and current features. Retraining the model.")
                model = train_model(data)
                model_features = features  # Update model_features after retraining
        except (FileNotFoundError, ValueError) as e:
            logger.info(f"Model file not found or invalid. Training model... ({e})")
            model = train_model(data)
            model_features = features  # Update model_features after retraining

        # Convert features to numeric data types
        for feature in features:
            latest_data[feature] = pd.to_numeric(latest_data[feature], errors='coerce')

        # After conversion, check for NaNs introduced by coercion
        if latest_data[features].isnull().any():
            missing_features = latest_data[features].isnull()
            logger.error(f"Missing or invalid values in features after conversion: {missing_features[missing_features].index.tolist()}")
            raise HTTPException(status_code=400, detail="Invalid data in features after type conversion.")

        input_data = latest_data[features].values.reshape(1, -1)

        # Ensure input_data is of type float64
        input_data = input_data.astype(np.float64)

        # Now check for NaNs
        if np.isnan(input_data).any():
            logger.error("Input data for prediction contains NaN values after type conversion.")
            raise HTTPException(status_code=400, detail="Input data for prediction contains NaN values.")

        # Make prediction
        predicted_price = model.predict(input_data)[0]
        current_price = latest_data['value']
        logger.info(f"Predicted price: {predicted_price}, Current price: {current_price}")

        # Prepare indicators for signal generation
        indicators = latest_data.to_dict()
        # Include previous EMAs for crossover detection
        previous_data = data.iloc[-2] if len(data) >= 2 else latest_data
        indicators['previous_ema_9'] = previous_data['ema_9']
        indicators['previous_ema_21'] = previous_data['ema_21']

        # Generate trading signal
        action = generate_trading_signal(current_price, predicted_price, indicators)
        logger.info(f"Generated action: {action}")

        # Prepare Stop Loss and Take Profit
        sl = None
        tp = None

        # Determine recent swing highs and lows for SL and TP
        recent_high = data['value'].rolling(window=5).max().iloc[-1]
        recent_low = data['value'].rolling(window=5).min().iloc[-1]

        if action == 'buy':
            sl = recent_low  # Stop Loss below recent swing low
            tp = current_price + 2 * (current_price - sl)  # 1:2 risk-to-reward ratio
        elif action == 'sell':
            sl = recent_high  # Stop Loss above recent swing high
            tp = current_price - 2 * (sl - current_price)  # 1:2 risk-to-reward ratio

        response = {
            "action": action,
            "entry": f"{current_price:.5f}" if action != 'hold' else None,
            "sl": f"{sl:.5f}" if sl else None,
            "tp": f"{tp:.5f}" if tp else None
        }

        logger.info(f"Response prepared: {response}")
        return {"status": "success", "data": response}

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except KeyError as e:
        print(f"KeyError: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Missing key in JSON: {str(e)}")
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
