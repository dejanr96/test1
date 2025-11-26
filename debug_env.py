import asyncio
import numpy as np
import pandas as pd
from src.data.ingestion import DataIngestionService
from src.env.trading_env import TradingEnv

async def main():
    print("--- Starting Perfect Logic Sanity Check ---")

    # 1. Fetch Data (Same as training)
    print("Fetching 1000 candles...")
    data_service = DataIngestionService(exchange_id='binanceusdm', symbol='BTC/USDT:USDT', timeframe='1m')
    await data_service.start_stream()
    df = await data_service.fetch_ohlcv(limit=1000)
    await data_service.close()

    if df.empty:
        print("Error: No data fetched.")
        return

    # 2. Initialize Environment
    # overfit_test argument removed
    env = TradingEnv(df, initial_balance=10000.0)
    obs, _ = env.reset()

    total_reward = 0.0
    fee = 0.01 # The fixed penalty in env
    
    print(f"Initial Balance: {env.balance}")
    print(f"Fee (Penalty): {fee}")

    # 3. Iterate Step-by-Step
    for i in range(len(df) - 1):
        current_price = df.iloc[i]['close']
        future_price = df.iloc[i+1]['close']
        
        # Perfect Logic
        # Action: [Type, Size, Offset]
        # Type: >0.3 Buy, <-0.3 Sell
        # Size: 1.0 (Full size)
        
        action = [0.0, 0.0, 0.0] # Default Hold
        action_desc = "HOLD"

        # Calculate expected move
        price_diff = future_price - current_price
        
        # Logic: If move > 2 * Fee (heuristic) -> Trade
        # Note: Fee is in Reward units (0.01), but Price Diff is in USDT.
        # We need to check if the PnL from the move covers the fee.
        # PnL = Position * Price_Diff
        # If we buy max size (approx balance / price), Position = Balance / Price
        # PnL = (Balance / Price) * (Future - Current)
        # We want PnL * 100 (Reward Scale) > Fee? 
        # Or just PnL > 0?
        # The user said: Future_Price > Current_Price + (Fee * 2)
        # But Fee is 0.01 (scalar penalty). 
        # Let's stick to the user's heuristic but adapted:
        # If Future > Current: BUY
        # If Future < Current: SELL
        # We want to see if we make money even with fees.
        
        if future_price > current_price:
            action = [1.0, 1.0, 0.0] # BUY, Max Size
            action_desc = "BUY"
        elif future_price < current_price:
            action = [-1.0, 1.0, 0.0] # SELL, Max Size
            action_desc = "SELL"
            
        # Execute
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i}: Price {current_price:.2f} -> Future {future_price:.2f}. Action: {action_desc}. Reward: {reward:.4f}. Portfolio: {info['portfolio_value']:.2f}")

        if done:
            break

    print("-" * 30)
    print(f"Final Balance: {env.balance:.2f}")
    print(f"Final Portfolio Value: {info['portfolio_value']:.2f}")
    print(f"Total Reward: {total_reward:.4f}")
    
    profit = info['portfolio_value'] - env.initial_balance
    print(f"Total Profit: {profit:.2f}")

    if profit > 0:
        print("VERDICT: SUCCESS. The environment allows profit.")
    else:
        print("VERDICT: FAILURE. Even perfect logic lost money. Environment is broken.")

if __name__ == "__main__":
    asyncio.run(main())
