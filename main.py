from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
from pydantic import BaseModel
import os

app = FastAPI()

# File paths for communication with MT4
TRADE_FILE = "trades.txt"
RESPONSE_FILE = "responses.txt"

# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Model for passing trade data
class TradeData(BaseModel):
    symbol: str
    action: str  # "buy" or "sell"
    lot_size: float

# Endpoint to send trade commands to MT4 via file
@app.post("/trade")
async def trade(data: TradeData):
    # Write trade command to file
    with open(TRADE_FILE, "w") as trade_file:
        trade_file.write(f"{data.action},{data.symbol},{data.lot_size}\n")
    return {"status": "success", "action": data.action, "symbol": data.symbol}

# WebSocket endpoint
@app.websocket("/ws/trading")
async def trading_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Wait for message from MT4
            data = await websocket.receive_text()
            print(f"Message received: {data}")

            # Process incoming data and respond
            await manager.send_message(f"Server received: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")

# Endpoint to get MT4 responses
@app.get("/responses")
async def get_responses():
    # Read responses from MT4
    if os.path.exists(RESPONSE_FILE):
        with open(RESPONSE_FILE, "r") as response_file:
            responses = response_file.readlines()
        return {"responses": responses}
    else:
        return {"responses": []}

# Test endpoint
@app.get("/")
async def root():
    return {"message": "EA Bot API is running"}
