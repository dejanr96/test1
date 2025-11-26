import asyncio
import logging
import os
from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd
import json
from typing import List
from src.agent.ppo_agent import PPOAgent
from src.env.trading_env import TradingEnv
from src.data.features import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("APEX_Core")

app = FastAPI(title="APEX AI DRL Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# Global Agent Instance
agent = None

class PredictionRequest(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Add other features as needed

class PredictionResponse(BaseModel):
    action: str
    size: float
    confidence: float

def get_agent():
    global agent
    if agent is None:
        # Initialize with dummy env for prediction structure
        # In real app, we load the model once
        df = pd.DataFrame({'close': [100]}) # Dummy
        env = TradingEnv(df)
        agent = PPOAgent(env)
        agent.load()
    return agent

@app.get("/")
async def root():
    return {"message": "APEX AI DRL Core is running", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    """
    Predict action based on market data.
    """
    try:
        agent = get_agent()
        
        # Convert request to DataFrame/Features
        # simplified: just using close for now in this mock
        # In reality: construct full feature vector
        
        # Mock observation construction
        # We need to match the observation space of the env
        # For this demo, we'll just use random or simplified logic if features don't match
        # But let's try to be somewhat realistic:
        
        # Create a single row DF
        df = pd.DataFrame([data.dict()])
        fe = FeatureEngineer()
        df = fe.add_technical_indicators(df)
        
        # Extract observation (needs to match env shape)
        # This is tricky without the full history. 
        # For live inference, we usually maintain a buffer of history.
        
        # For now, return a mock prediction to show API contract
        return PredictionResponse(
            action="BUY",
            size=0.5,
            confidence=0.85
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return PredictionResponse(action="HOLD", size=0.0, confidence=0.0)

async def train_task():
    logger.info("Starting background training...")
    # Call the train logic from src.train
    from src.train import train
    await train()
    logger.info("Background training complete.")

@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    """
    Trigger a retraining loop in the background.
    """
    background_tasks.add_task(train_task)
    return {"message": "Training started in background"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

class TradeUpdate(BaseModel):
    type: str
    data: dict

@app.post("/api/update")
async def receive_update(update: TradeUpdate):
    """
    Receive updates from the trading bot and broadcast to frontend.
    """
    await manager.broadcast(update.dict())
    return {"status": "broadcasted"}

if __name__ == "__main__":
    # If run directly, start uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
