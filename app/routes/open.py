# app/routes/open.py

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Optional
import json
import pandas as pd
import numpy as np
from app import schemas, logic
from app.database import get_db
from logger import logger

router = APIRouter(
    prefix="/open",
    tags=["open"]
)

@router.post("", response_model=schemas.APIResponse)
async def chat_with_model(chat_request: Request, db: Session = Depends(get_db)):
    """
    Processes a chat request to generate trading recommendations.
    """
    raw_body = await chat_request.body()
    logger.info(f"Raw Body Received: {raw_body}")

    try:
        decoded_body = raw_body.decode("utf-8").strip()
        cleaned_body = decoded_body.replace("\x00", "")
        data = json.loads(cleaned_body)

        # Extract symbol from the request
        symbol = data.get("symbol", "EURUSD").upper()
        logger.info(f"Processing request for symbol: {symbol}")

        # Fetch price data
        price_data = logic.get_price_data(symbol, db)

        if price_data.empty:
            logger.error(f"No price data found for symbol: {symbol}")
            raise HTTPException(status_code=404, detail=f"No price data found for symbol: {symbol}")
        logger.info(f"Fetched {len(price_data)} records for symbol {symbol}.")

        # Calculate technical indicators
        price_data = logic.calculate_technical_indicators(price_data)
        logger.info("Technical indicators calculated.")

        # Ensure there is enough data after processing
        if price_data.empty or len(price_data) < 2:
            logger.error(f"Not enough data points available after processing. Data length: {len(price_data)}")
            raise HTTPException(status_code=400, detail="Not enough data points available for analysis.")

        # Prepare the latest data point for prediction
        latest_data = price_data.iloc[-1].copy()
        features = ['value', 'ema_9', 'ema_21', 'rsi']

        # Load or train the model
        try:
            model, model_features = logic.load_model()
            if model_features != features:
                logger.warning("Feature mismatch between model and current features. Retraining the model.")
                model = logic.train_model(price_data)
                model_features = features  # Update model_features after retraining
        except (FileNotFoundError, ValueError) as e:
            logger.info(f"Model file not found or invalid. Training model... ({e})")
            model = logic.train_model(price_data)
            model_features = features  # Update model_features after retraining

        # Ensure features are numeric
        for feature in features:
            latest_data[feature] = pd.to_numeric(latest_data[feature], errors='coerce')

        # After conversion, check for NaNs introduced by coercion
        if latest_data[features].isnull().any():
            missing_features = latest_data[features].isnull()
            logger.error(
                f"Missing or invalid values in features after conversion: {missing_features[missing_features].index.tolist()}"
            )
            raise HTTPException(
                status_code=400, detail="Invalid data in features after type conversion."
            )

        # Prepare input_data as DataFrame with feature names
        input_data = pd.DataFrame([latest_data[features]])

        # Ensure input_data is of type float64
        input_data = input_data.astype(np.float64)

        # Now check for NaNs in input_data
        if input_data.isnull().any().any():
            logger.error("Input data for prediction contains NaN values after type conversion.")
            raise HTTPException(
                status_code=400, detail="Input data for prediction contains NaN values."
            )

        # Make prediction
        predicted_price = model.predict(input_data)[0]
        current_price = latest_data['value']
        logger.info(f"Predicted price: {predicted_price}, Current price: {current_price}")

        # Prepare indicators for signal generation
        indicators = latest_data.to_dict()

        # Generate trading signal
        action = logic.generate_trading_signal(current_price, predicted_price, indicators)
        logger.info(f"Generated action: {action}")

        # Define fixed TP and SL in pips
        TP_pips = 20
        SL_pips = 40

        # Determine pip size based on symbol
        pip_size = logic.get_pip_size(symbol)
        logger.info(f"Pip size for symbol {symbol}: {pip_size}")

        # Determine recent swing highs and lows for SL and TP
        recent_high = price_data['value'].rolling(window=5).max().iloc[-1]
        recent_low = price_data['value'].rolling(window=5).min().iloc[-1]

        # Calculate SL and TP based on current price, swing highs/lows, and pip size
        if action == 'buy':
            sl = recent_low - (SL_pips * pip_size)  # Stop Loss 40 pips below recent swing low
            tp = current_price + (TP_pips * pip_size)  # Take Profit 20 pips above current price
        elif action == 'sell':
            sl = recent_high + (SL_pips * pip_size)  # Stop Loss 40 pips above recent swing high
            tp = current_price - (TP_pips * pip_size)  # Take Profit 20 pips below current price
        else:
            sl = None
            tp = None

        # Validation: Ensure SL and TP are logical
        if action == 'buy':
            if sl >= current_price:
                logger.error("For BUY orders, SL must be below the current price.")
                raise HTTPException(status_code=400, detail="Invalid Stop Loss value.")
            if tp <= current_price:
                logger.error("For BUY orders, TP must be above the current price.")
                raise HTTPException(status_code=400, detail="Invalid Take Profit value.")
        elif action == 'sell':
            if sl <= current_price:
                logger.error("For SELL orders, SL must be above the current price.")
                raise HTTPException(status_code=400, detail="Invalid Stop Loss value.")
            if tp >= current_price:
                logger.error("For SELL orders, TP must be below the current price.")
                raise HTTPException(status_code=400, detail="Invalid Take Profit value.")

        # Format SL and TP to appropriate decimal places
        decimal_places = 2 if pip_size == 0.01 else 5
        response = {
            "action": action,
            "entry": f"{current_price:.{decimal_places}f}" if action != 'hold' else None,
            "sl": f"{sl:.{decimal_places}f}" if sl else None,
            "tp": f"{tp:.{decimal_places}f}" if tp else None
        }

        logger.info(f"Response prepared: {response}")
        return {"status": "success", "data": response}

    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except KeyError as e:
        logger.error(f"KeyError: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Missing key in JSON: {str(e)}")
    except HTTPException as http_exc:
        raise http_exc  # Re-raise HTTP exceptions to be handled by FastAPI
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")
