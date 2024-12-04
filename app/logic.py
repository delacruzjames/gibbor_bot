# app/logic.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List
from sqlalchemy.orm import Session
from sqlalchemy import func
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import ta
from app.models import Price
from app.database import logger
from app.schemas import OpenResponse


def get_price_data(symbol: str, db: Session) -> pd.DataFrame:
    """
    Retrieves price data for a given symbol from the database over the past 60 days.

    Args:
        symbol (str): The trading symbol (e.g., "EURUSD").
        db (Session): SQLAlchemy database session.

    Returns:
        pd.DataFrame: DataFrame containing price data.
    """
    start_date = datetime.now() - timedelta(days=60)
    logger.debug(f"Fetching price data for symbol {symbol} since {start_date}")

    # Case-insensitive symbol matching
    prices = db.query(Price).filter(
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


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators for the given price data.

    Args:
        data (pd.DataFrame): DataFrame containing price data.

    Returns:
        pd.DataFrame: DataFrame with additional technical indicator columns.
    """
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

    data.dropna(inplace=True)  # Drop rows with NaN values resulting from indicator calculations
    data.reset_index(inplace=True)
    logger.debug("Technical indicators calculated.")
    return data


def train_model(data: pd.DataFrame) -> GradientBoostingRegressor:
    """
    Trains a Gradient Boosting Regressor model on the provided data.

    Args:
        data (pd.DataFrame): DataFrame containing price data with technical indicators.

    Returns:
        GradientBoostingRegressor: Trained machine learning model.
    """
    if data.empty:
        logger.error("No data available to train the model.")
        raise ValueError("Insufficient data to train the model.")

    # Prepare features and target
    data.dropna(inplace=True)  # Ensure no NaN values
    data['target'] = data['value'].shift(-1)  # Predict the next price
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

    # Save the trained model with features
    model_info = {
        'model': model,
        'features': features
    }
    joblib.dump(model_info, 'price_prediction_model.pkl')
    logger.info("Model trained and saved successfully.")
    return model


def load_model() -> Tuple[GradientBoostingRegressor, List[str]]:
    """
    Loads the trained machine learning model from disk.

    Returns:
        Tuple[GradientBoostingRegressor, List[str]]: The model and its feature list.
    """
    model_path = 'price_prediction_model.pkl'
    if not os.path.exists(model_path):
        logger.error("Model file not found. Please train the model first.")
        raise FileNotFoundError("Model file not found.")

    model_info = joblib.load(model_path)

    if isinstance(model_info, dict):
        model = model_info.get('model')
        features = model_info.get('features')
        logger.debug(f"Loaded model with features: {features}")
        return model, features
    else:
        logger.error("Loaded model_info is not a dictionary.")
        raise ValueError("Loaded model_info is not a dictionary. Please retrain the model.")


def get_pip_size(symbol: str) -> float:
    """
    Determines the pip size for a given symbol.
    Commonly, pip size is 0.0001 for most currency pairs and 0.01 for JPY pairs.

    Args:
        symbol (str): The trading symbol (e.g., "EURUSD").

    Returns:
        float: The pip size.
    """
    if symbol.endswith("JPY"):
        return 0.01
    else:
        return 0.0001


def generate_trading_signal(current_price: float, predicted_price: float, indicators: dict) -> str:
    """
    Generates a trading signal ('buy', 'sell', or 'hold') based on technical indicators and predicted price.

    Args:
        current_price (float): The current price of the asset.
        predicted_price (float): The predicted future price of the asset.
        indicators (dict): A dictionary containing calculated technical indicators.

    Returns:
        str: 'buy', 'sell', or 'hold'
    """
    action = 'hold'  # Default action

    # Extract indicators
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
