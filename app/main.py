# app/main.py

from fastapi import FastAPI
from app.routes import prices, trades, open
from logger import logger

# Initialize FastAPI app
app = FastAPI(
    title="Trading API",
    description="API for managing trades and price data, and generating trading signals.",
    version="1.0.0",
    strict_slashes=False
)

# Include API routers
app.include_router(prices.router)
app.include_router(trades.router)
app.include_router(open.router)

# Root endpoint
@app.get("/", response_model=dict)
async def root():
    return {"message": "FastAPI app with PostgreSQL is running"}

# Health check endpoint
@app.get("/health", response_model=dict)
def health_check():
    return {"status": "healthy"}
