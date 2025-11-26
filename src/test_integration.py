import pandas as pd
import numpy as np
from src.data.features import FeatureEngineer
from src.env.trading_env import TradingEnv
from src.agent.ppo_agent import PPOAgent
from src.strategy.market_regime import MarketRegimeDetector

def test_pipeline():
    print("Starting Integration Test...")
    
    # 1. Generate Mock Data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(50000, 51000, 200),
        'high': np.random.uniform(51000, 52000, 200),
        'low': np.random.uniform(49000, 50000, 200),
        'close': np.random.uniform(50000, 51000, 200),
        'volume': np.random.uniform(100, 1000, 200)
    })
    
    # 2. Feature Engineering
    fe = FeatureEngineer()
    df = fe.add_technical_indicators(df)
    
    # Mock LOB/Tick features for the dataframe
    df['ofi'] = np.random.uniform(-10, 10, 200)
    df['cvd'] = np.random.uniform(-50, 50, 200)
    df['qi'] = np.random.uniform(-1, 1, 200)
    df['lob_imbalance_5'] = np.random.uniform(-1, 1, 200)
    
    vp_data = fe.calculate_volume_profile(df)
    print("Volume Profile Calculated:", vp_data)
    
    # 3. Regime Detection
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(df, vp_data)
    print("Detected Regime:", regime)
    
    # 4. Environment
    env = TradingEnv(df)
    obs, _ = env.reset()
    print("Environment Reset. Observation Shape:", obs.shape)
    
    # 5. Agent
    agent = PPOAgent(env)
    agent.create_model(verbose=1)
    print("Agent Model Created with Custom Policy.")
    
    # 6. Train Loop
    print("Starting Training Loop...")
    agent.train(total_timesteps=100)
    print("Training Loop Completed.")
    
    print("Integration Test PASSED.")

if __name__ == "__main__":
    test_pipeline()
