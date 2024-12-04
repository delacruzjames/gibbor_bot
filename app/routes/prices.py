# app/routes/prices.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app import schemas, crud
from app.database import get_db
from logger import logger

router = APIRouter(
    prefix="/prices",
    tags=["prices"]
)

@router.post("/", response_model=schemas.APIResponse)
async def add_price(price: schemas.PriceCreate, db: Session = Depends(get_db)):
    """
    Adds a new price record to the database.
    """
    try:
        logger.info(f"Received Price Data: {price}")
        price_record = crud.create_price(db, price)
        return {
            "status": "success",
            "data": {
                "id": price_record.id,
                "symbol": price_record.symbol,
                "value": price_record.value,
                "timestamp": price_record.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error processing price data: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail="Invalid price data.")

@router.get("/", response_model=schemas.APIResponse)
async def get_prices(db: Session = Depends(get_db)):
    """
    Retrieves all price records from the database.
    """
    try:
        logger.info("Fetching all price records.")
        prices = crud.get_all_prices(db)
        if not prices:
            return {"status": "success", "prices": []}
        price_list = [{
            "id": price.id,
            "symbol": price.symbol,
            "value": price.value,
            "timestamp": price.timestamp.isoformat()
        } for price in prices]
        return {"status": "success", "prices": price_list}
    except Exception as e:
        logger.error(f"Error retrieving prices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve prices.")

@router.delete("/clear", response_model=schemas.APIResponse)
async def clear_prices(db: Session = Depends(get_db)):
    """
    Deletes all price records from the database.
    """
    try:
        logger.info("Clearing all price records.")
        deleted = crud.clear_all_prices(db)
        logger.info(f"All price records have been cleared. Total deleted: {deleted}")
        return {
            "status": "success",
            "message": f"All price records have been cleared. Total deleted: {deleted}"
        }
    except Exception as e:
        logger.error(f"Error clearing price records: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear price records.")
