import asyncio
import pandas as pd
import numpy as np
from src.data.ingestion import DataIngestionService
from src.data.features import FeatureEngineer
from src.agent.ppo_agent import PPOAgent
from src.env.trading_env import TradingEnv

async def verify():
    print("Fetching last 10000 candles (7 Days) for verification...")
    # Match the paper trader config
    service = DataIngestionService(exchange_id='binanceusdm', symbol='BTC/USDT:USDT', timeframe='1m')
    
    # Use the service's fetch_ohlcv which now supports pagination
    df = await service.fetch_ohlcv(limit=10000)
    await service.close()
    
    print(f"Fetched {len(df)} candles.")
    
    print("Calculating features...")
    fe = FeatureEngineer()
    df = fe.add_technical_indicators(df)
    
    # Add missing columns (zeros for backtest as we don't have hist LOB)
    for col in ['ofi', 'cvd', 'qi', 'lob_imbalance_5']:
        df[col] = 0.0
        
    env = TradingEnv(df)
    agent = PPOAgent(env)
    
    try:
        agent.load()
        print("Model loaded successfully.")
    except:
        print("Error: Could not load model. Is paper_trade.py running?")
        return

    print("Running Backtest...")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.predict(obs)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
    print(f"--------------------------------")
    print(f"Backtest Results (Last 10000 candles):")
    print(f"Final Balance: {info['balance']:.2f}")
    print(f"Profit: {info['balance'] - 10000:.2f} ({(info['balance'] - 10000)/100:.2f}%)")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"--------------------------------")

if __name__ == "__main__":
    asyncio.run(verify())
