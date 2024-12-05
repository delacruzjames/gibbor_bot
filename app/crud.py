# app/crud.py

from sqlalchemy.orm import Session
from app.models import TradeRecord, Price
from app.schemas import TradeData, PriceCreate


def create_trade(db: Session, trade: TradeData) -> TradeRecord:
    """
    Creates a new trade record in the database.

    Args:
        db (Session): SQLAlchemy session.
        trade (TradeData): Trade data.

    Returns:
        TradeRecord: The created trade record.
    """
    trade_record = TradeRecord(
        symbol=trade.symbol,
        action=trade.action,
        lot_size=trade.lot_size
    )
    db.add(trade_record)
    db.commit()
    db.refresh(trade_record)
    return trade_record


def create_price(db: Session, price: dict) -> Price:
    """
    Creates a new price record in the database.

    Args:
        db (Session): SQLAlchemy session.
        price (PriceCreate): Price data.

    Returns:
        Price: The created price record.
    """
    price_record = Price(
        symbol=price["symbol"],
        value=price["value"],
        timestamp=price["timestamp"]
    )
    db.add(price_record)
    db.commit()
    db.refresh(price_record)
    return price_record


def get_all_prices(db: Session) -> list[Price]:
    """
    Retrieves all price records from the database.

    Args:
        db (Session): SQLAlchemy session.

    Returns:
        list[Price]: List of price records.
    """
    return db.query(Price).all()


def get_all_trades(db: Session) -> list[TradeRecord]:
    """
    Retrieves all trade records from the database.

    Args:
        db (Session): SQLAlchemy session.

    Returns:
        list[TradeRecord]: List of trade records.
    """
    return db.query(TradeRecord).all()


def clear_all_prices(db: Session) -> int:
    """
    Deletes all price records from the database.

    Args:
        db (Session): SQLAlchemy session.

    Returns:
        int: Number of records deleted.
    """
    deleted = db.query(Price).delete()
    db.commit()
    return deleted
