# app/schemas.py

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class TradeData(BaseModel):
    symbol: str
    action: str  # "buy" or "sell"
    lot_size: float

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "EURUSD",
                "action": "buy",
                "lot_size": 1.0
            }
        }

class PriceCreate(BaseModel):
    symbol: str
    value: float
    timestamp: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "EURUSD",
                "value": 1.23456,
                "timestamp": "2024-12-04T15:30:00"
            }
        }

class ChatRequest(BaseModel):
    symbol: Optional[str] = Field(default="EURUSD")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "EURUSD"
            }
        }

class TradeResponse(BaseModel):
    id: int
    symbol: str
    action: str
    lot_size: float

    class Config:
        from_attributes = True

class PriceResponse(BaseModel):
    id: int
    symbol: str
    value: float
    timestamp: datetime

    class Config:
        from_attributes = True

class OpenResponse(BaseModel):
    action: str
    entry: Optional[str]
    sl: Optional[str]
    tp: Optional[str]

    class Config:
        json_schema_extra = {
            "example": {
                "action": "buy",
                "entry": "1.23456",
                "sl": "1.23000",
                "tp": "1.23912"
            }
        }

class APIResponse(BaseModel):
    status: str
    data: Optional[dict] = None
    trade: Optional[TradeResponse] = None
    prices: Optional[List[PriceResponse]] = None
    trades: Optional[List[TradeResponse]] = None
    message: Optional[str] = None
    error: Optional[str] = None
