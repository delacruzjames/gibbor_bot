# app/routes/trades.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app import schemas, crud
from app.database import get_db
from logger import logger

router = APIRouter(
    prefix="/trades",
    tags=["trades"]
)

@router.post("/", response_model=schemas.APIResponse)
async def add_trade(trade: schemas.TradeData, db: Session = Depends(get_db)):
    """
    Adds a new trade record to the database.
    """
    try:
        logger.info(f"Received Trade Data: {trade}")
        trade_record = crud.create_trade(db, trade)
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
        logger.error(f"Error processing trade data: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail="Invalid trade data.")

@router.get("/", response_model=schemas.APIResponse)
async def get_trades(db: Session = Depends(get_db)):
    """
    Retrieves all trade records from the database.
    """
    try:
        logger.info("Fetching all trade records.")
        trades = crud.get_all_trades(db)
        if not trades:
            return {"status": "success", "trades": []}
        trade_list = [{
            "id": trade.id,
            "symbol": trade.symbol,
            "action": trade.action,
            "lot_size": trade.lot_size
        } for trade in trades]
        return {"status": "success", "trades": trade_list}
    except Exception as e:
        logger.error(f"Error retrieving trades: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve trades.")
