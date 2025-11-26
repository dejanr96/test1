import asyncio
import logging
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.data.ingestion import DataIngestionService
from src.data.features import FeatureEngineer
from src.env.trading_env import TradingEnv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Backtester")

async def run_backtest():
    # 1. Fetch Data (Out-of-Sample)
    logger.info("Fetching 2,000 candles for QUICK backtest...")
    data_service = DataIngestionService(exchange_id='binanceusdm', symbol='BTC/USDT:USDT', timeframe='1m')
    await data_service.start_stream()
    df = await data_service.fetch_ohlcv(limit=2000)
    await data_service.close()
    
    # 2. Feature Engineering
    logger.info("Calculating features...")
    fe = FeatureEngineer()
    df = fe.add_advanced_microstructure_features(df)
    
    # 3. Setup Environment
    env = TradingEnv(df)
    env = DummyVecEnv([lambda: env])
    
    # Initialize fresh VecNormalize with same config as training
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    # 4. Load Model
    # Find the latest phase 2 checkpoint
    checkpoint_dir = "checkpoints"
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("ppo_apex_phase2") and f.endswith(".zip")]
    if not checkpoints:
        logger.error("No Phase 2 checkpoints found!")
        return
        
    # Sort by step count (assuming format ppo_apex_phase2_X_steps.zip)
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[3]))[-1]
    model_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    logger.info(f"Loading latest model from {model_path}...")
    model = PPO.load(model_path)
    
    # 5. Warmup Phase (Calibrate Normalization)
    logger.info("Running warmup (2000 steps) to calibrate normalization...")
    env.training = True # Update stats
    obs = env.reset()
    for _ in range(2000):
        action = [env.action_space.sample()] 
        obs, _, done, _ = env.step(action)
        if done: obs = env.reset()
        
    logger.info("Warmup complete. Switching to Eval mode.")
    env.training = False # Stop updating stats
    
    # 6. Run Backtest
    logger.info("Running backtest loop...")
    
    # Reset internal env state for clean backtest
    env.envs[0].reset() 
    obs = env.reset()
    done = False
    
    portfolio_values = []
    trades_count = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        step_info = info[0]
        portfolio_values.append(step_info['portfolio_value'])
        trades_count = step_info['trades_taken']

    # 7. Analysis
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    pnl = final_value - initial_value
    pnl_pct = (pnl / initial_value) * 100
    
    # Calculate Max Drawdown
    pv_series = pd.Series(portfolio_values)
    rolling_max = pv_series.cummax()
    drawdown = (pv_series - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    
    logger.info("="*50)
    logger.info(f"PHASE 2 BACKTEST RESULTS")
    logger.info("="*50)
    logger.info(f"Model:           {latest_checkpoint}")
    logger.info(f"Initial Balance: ${initial_value:.2f}")
    logger.info(f"Final Balance:   ${final_value:.2f}")
    logger.info(f"Net PnL:         ${pnl:.2f} ({pnl_pct:.2f}%)")
    logger.info(f"Max Drawdown:    {max_dd:.2f}%")
    logger.info(f"Trades Taken:    {trades_count}")
    logger.info("="*50)
    
    with open("backtest_results.txt", "w") as f:
        f.write("="*50 + "\n")
        f.write(f"PHASE 2 BACKTEST RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Model:           {latest_checkpoint}\n")
        f.write(f"Initial Balance: ${initial_value:.2f}\n")
        f.write(f"Final Balance:   ${final_value:.2f}\n")
        f.write(f"Net PnL:         ${pnl:.2f} ({pnl_pct:.2f}%)\n")
        f.write(f"Max Drawdown:    {max_dd:.2f}%\n")
        f.write(f"Trades Taken:    {trades_count}\n")
        f.write("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_backtest())
