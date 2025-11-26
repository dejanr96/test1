import asyncio
import logging
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.data.ingestion import DataIngestionService
from src.data.features import FeatureEngineer
from src.env.trading_env import TradingEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Backtester")

async def run_backtest():
    # 1. Fetch Data (Out-of-Sample)
    logger.info("Fetching 20,000 candles for backtest...")
    data_service = DataIngestionService(exchange_id='binanceusdm', symbol='BTC/USDT:USDT', timeframe='1m')
    await data_service.start_stream()
    df = await data_service.fetch_ohlcv(limit=20000)
    await data_service.close()
    
    # 2. Feature Engineering
    logger.info("Calculating features...")
    fe = FeatureEngineer()
    df = fe.add_advanced_microstructure_features(df)
    
    # 3. Setup Environment
    env = TradingEnv(df)
    env = DummyVecEnv([lambda: env])
    # Initialize fresh VecNormalize (since we lost the training one)
    # We need to calibrate it first
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    # 4. Load Model
    model_path = "checkpoints/ppo_apex_phase1_7200000_steps.zip"
    logger.info(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # 5. Warmup Phase (Calibrate Normalization)
    logger.info("Running warmup (2000 steps) to calibrate normalization...")
    env.training = True # Update stats
    obs = env.reset()
    for _ in range(2000):
        action = [env.action_space.sample()] # Random actions for warmup is fine, just need data flow
        obs, _, done, _ = env.step(action)
        if done: obs = env.reset()
        
    logger.info("Warmup complete. Switching to Eval mode.")
    env.training = False # Stop updating stats
    
    # 6. Run Backtest
    logger.info("Running backtest loop...")
    obs = env.reset()
    done = False
    
    portfolio_values = []
    trades = []
    
    # Reset internal env state for clean backtest
    env.envs[0].reset() 
    obs = env.reset()
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Extract info from VecEnv wrapper (it returns a list of infos)
        step_info = info[0]
        portfolio_values.append(step_info['portfolio_value'])
        
        if step_info['trades_taken'] > len(trades):
            # We don't have easy access to the exact trade details from VecEnv info list easily without modifying Env
            # But we can track portfolio value change
            pass

    # 6. Analysis
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
    logger.info(f"BACKTEST RESULTS (Phase 1 Model)")
    logger.info("="*50)
    logger.info(f"Initial Balance: ${initial_value:.2f}")
    logger.info(f"Final Balance:   ${final_value:.2f}")
    logger.info(f"Net PnL:         ${pnl:.2f} ({pnl_pct:.2f}%)")
    logger.info(f"Max Drawdown:    {max_dd:.2f}%")
    logger.info("="*50)

if __name__ == "__main__":
    asyncio.run(run_backtest())
