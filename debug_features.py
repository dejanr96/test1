import asyncio
import os
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from src.data.features import FeatureEngineer

# --- CONFIGURATION ---
SYMBOL = "BTC/USDT" 
TIMEFRAME = "1m"
API_KEY = os.getenv("BINANCE_DEMO_API_KEY", "JpQ1NMelCU4kfQ1N48jettZn8NcneaIPHtJXRaZZNuSMZFIXnTC4HQQlrzlsQgbj")
API_SECRET = os.getenv("BINANCE_DEMO_SECRET", "wnz2GlTNZw9skUWZhSaZ8mYcv0tfw09dTEwVJChetfxisdLqAH3CFzlItFlWev5E")

EXCHANGE_OPTIONS = {
    'defaultType': 'future',
}

async def main():
    print(f"--- 🔍 DEBUGGING FEATURES: {SYMBOL} ---")
    
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'options': EXCHANGE_OPTIONS
    })
    
    if hasattr(exchange, 'enable_demo_trading'):
        exchange.enable_demo_trading(True)
    else:
        exchange.set_sandbox_mode(True)
    
    await exchange.load_markets()
    
    print("Fetching 1000 candles...")
    ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1000)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"Data Fetched: {len(df)} rows.")
    
    fe = FeatureEngineer()
    df = fe.add_advanced_microstructure_features(df)
    
    # Inspect Last Row Features
    last_row = df.iloc[-1]
    
    obs_columns = [
        'obi', 'cvd_velocity', 'trade_aggressiveness', 
        'rvi', 'va_position', 'profile_skew'
    ]
    
    print("\n--- RAW FEATURE VALUES (Last Candle) ---")
    for col in obs_columns:
        val = last_row[col]
        print(f"{col}: {val}")
        
    # Check for Zeros
    zeros = [col for col in obs_columns if last_row[col] == 0.0]
    if zeros:
        print(f"\n⚠️ WARNING: The following features are EXACTLY 0.0: {zeros}")
    else:
        print("\n✅ All features are non-zero (Healthy).")

    await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
