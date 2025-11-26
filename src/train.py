import pandas as pd
from src.env.trading_env import TradingEnv
from src.agent.ppo_agent import PPOAgent
from src.data.ingestion import DataIngestionService
from src.data.features import FeatureEngineer
import asyncio

import torch

async def train():
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    if device == "cpu":
        print("WARNING: GPU not detected. Training will be slow.")
        print("To enable GPU, install PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118")

    # 1. Load Data
    print("Fetching real data from Binance...")
    service = DataIngestionService(exchange_id='binance', symbol='BTC/USDT', timeframe='1m')
    try:
        df = await service.fetch_ohlcv(limit=1000)
    finally:
        await service.close()

    if df.empty:
        print("Failed to fetch data. Exiting.")
        return
    
    # 2. Feature Engineering
    print("Engineering features...")
    fe = FeatureEngineer()
    df = fe.add_technical_indicators(df)
    
    # 3. Setup Env and Agent
    print("Setting up environment...")
    env = TradingEnv(df)
    agent = PPOAgent(env)
    
    # 4. Train
    print("Starting training...")
    agent.train(total_timesteps=1000) # Short run for verification
    print("Training complete.")
    
    # 5. Save
    agent.save()

if __name__ == "__main__":
    asyncio.run(train())
