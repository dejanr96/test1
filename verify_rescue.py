import pandas as pd
import numpy as np
from src.env.trading_env import TradingEnv

def create_dummy_data(length=1000):
    dates = pd.date_range(start='2024-01-01', periods=length, freq='1min')
    df = pd.DataFrame(index=dates)
    df['open'] = 100.0
    df['high'] = 101.0
    df['low'] = 99.0
    df['close'] = 100.0 + np.random.normal(0, 0.5, length) # Random walk
    df['volume'] = 1000.0
    df['taker_buy_base_asset_volume'] = 500.0
    
    # Add required feature columns (mocked)
    df['obi'] = 0.0
    df['obi_rolling_5'] = 0.0
    df['cvd_velocity'] = 0.0
    df['cvd_acceleration'] = 0.0
    df['trade_aggressiveness'] = 0.0
    df['rvi'] = 0.0
    df['va_position'] = 0.0
    df['profile_skew'] = 0.0
    df['volatility'] = 0.1 # Default volatility
    
    return df

def test_fees():
    print("\n--- Testing Fees ---")
    df = create_dummy_data(100)
    env = TradingEnv(df, initial_balance=10000)
    env.reset()
    
    # Force Buy
    # Action: [Buy/Sell, Size, PriceOffset]
    # Buy 100% size
    action = [1.0, 1.0, 0.0] 
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"Balance after Buy: {info['balance']:.2f}")
    # Cost = 10000 / (1 + 0.0004) approx 9996
    # Fee = 9996 * 0.0004 = 3.99
    # Balance should be 10000 - 9996 - 3.99 = 0? No wait.
    # Logic: 
    # max_cost = balance / 1.0004
    # cost = min(balance * 1.0, max_cost)
    # fee = cost * 0.0004
    # balance -= (cost + fee)
    # So balance should be close to 0 (or small remainder)
    
    expected_fee_rate = 0.0004
    print(f"Trades: {env.trades}")
    if len(env.trades) > 0:
        trade = env.trades[0]
        print(f"Trade Fee: {trade['fee']:.4f}")
        assert trade['fee'] > 0, "Fee should be positive"
        print("Fee Test Passed.")
    else:
        print("Trade not executed!")

def test_stop_loss():
    print("\n--- Testing Hard Stop-Loss ---")
    df = create_dummy_data(100)
    env = TradingEnv(df, initial_balance=10000)
    env.reset()
    
    # Manually drop balance to trigger stop loss
    env.balance = 8000 # 20% drawdown (10000 -> 8000)
    
    # Step
    action = [0.0, 0.0, 0.0] # Hold
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"Stop Loss Triggered: {info.get('stop_loss_triggered')}")
    print(f"Done: {done}")
    
    assert info.get('stop_loss_triggered') == True, "Stop Loss should trigger"
    assert done == True, "Episode should end"
    print("Stop-Loss Test Passed.")

def test_regime_filter():
    print("\n--- Testing Regime Filter ---")
    df = create_dummy_data(100)
    # Set volatility to 0 to force "Choppy"
    df['volatility'] = 0.0
    
    env = TradingEnv(df, initial_balance=10000)
    env.reset()
    
    # Try to Buy
    action = [1.0, 1.0, 0.0]
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"Position after Forced Buy in Choppy Market: {env.position}")
    # Should be 0 if filter works
    assert env.position == 0, "Regime Filter should prevent buying"
    print("Regime Filter Test Passed.")

if __name__ == "__main__":
    test_fees()
    test_stop_loss()
    test_regime_filter()
