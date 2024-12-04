# app/models.py

from sqlalchemy import Column, Integer, String, Float, DateTime
from app.database import Base

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
    value = Column(Float, nullable=False)  # Close price
    timestamp = Column(DateTime, nullable=False)  # Timestamp of the price
