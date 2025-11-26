import asyncio
import os
import time
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.data.features import FeatureEngineer
from src.env.trading_env import TradingEnv

# --- CONFIGURATION ---
MODEL_PATH = "ppo_apex_model_production"
STATS_PATH = "vec_normalize.pkl"
SYMBOL = "BTC/USDT" # CCXT Symbol
TIMEFRAME = "1m"

# BINANCE DEMO CREDENTIALS
API_KEY = os.getenv("BINANCE_DEMO_API_KEY", "JpQ1NMelCU4kfQ1N48jettZn8NcneaIPHtJXRaZZNuSMZFIXnTC4HQQlrzlsQgbj")
API_SECRET = os.getenv("BINANCE_DEMO_SECRET", "wnz2GlTNZw9skUWZhSaZ8mYcv0tfw09dTEwVJChetfxisdLqAH3CFzlItFlWev5E")

# Binance Demo URL (Futures)
EXCHANGE_OPTIONS = {
    'defaultType': 'future',
    # 'sandbox': True, # DEPRECATED
}

async def main():
    print(f"--- 🟢 STARTING LIVE PAPER TRADER: {SYMBOL} ---")
    
    # 1. Initialize Exchange
    print("Connecting to Binance Demo...")
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'options': EXCHANGE_OPTIONS
    })
    
    # ENABLE DEMO TRADING (New Binance Environment)
    if hasattr(exchange, 'enable_demo_trading'):
        exchange.enable_demo_trading(True)
        print("✅ Demo Trading Enabled.")
    else:
        print("⚠️ CCXT version might be too old for enable_demo_trading. Trying sandbox mode fallback...")
        exchange.set_sandbox_mode(True)
    
    try:
        await exchange.load_markets()
        print("✅ Connected to Exchange.")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("Please check your API Keys and Network.")
        await exchange.close()
        return

    # 2. Load Model & Stats
    print("Loading Brain...")
    # Dummy Env for loading stats
    dummy_env = DummyVecEnv([lambda: TradingEnv(pd.DataFrame(), initial_balance=10000)])
    
    # Load Normalization Stats
    env = VecNormalize.load(STATS_PATH, dummy_env)
    env.training = False
    env.norm_reward = False
    
    # Load Model
    model = PPO.load(MODEL_PATH)
    print("✅ Model Loaded.")

    # 3. Live Loop
    print("🎧 Listening for market data...")
    fe = FeatureEngineer()
    
    while True:
        try:
            # Wait for next candle close (approx)
            # For HFT, we might want to run faster, but let's stick to 1m candles for now
            # Sleep until next minute starts? Or just poll?
            # Simple polling for now
            
            # Fetch latest 100 candles (enough for features)
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=100)
            if not ohlcv:
                print("⚠️ No data received.")
                await asyncio.sleep(5)
                continue
                
            # Convert to DF
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate Features
            df = fe.add_advanced_microstructure_features(df)
            
            # Prepare Observation (Last Row)
            # We need to construct the observation exactly like TradingEnv._next_observation
            # But we don't have the 'env' state (position, entry_price) tracked perfectly here yet.
            # We need to TRACK STATE locally.
            
            # TODO: Implement State Tracking (Position, Balance)
            # For now, let's assume flat (0 position) to see if it signals BUY
            # This is a "Signal Generator" mode.
            
            # Mock Internal State
            position = 0.0 # We need to fetch this from exchange!
            unrealized_pnl = 0.0
            
            # Fetch Real Position
            balance_info = await exchange.fetch_balance()
            # Parse balance and position for SYMBOL
            # This is complex in CCXT Futures.
            # positions = await exchange.fetch_positions([SYMBOL])
            # ...
            
            # For MVP: Just print the signal based on Market Data
            # Construct Obs
            # TradingEnv.obs_columns
            obs_columns = [
                'obi', 'cvd_velocity', 'trade_aggressiveness', 
                'rvi', 'va_position', 'profile_skew'
            ]
            
            market_obs = df.iloc[-1][obs_columns].values.astype(np.float32)
            internal_obs = np.array([position, unrealized_pnl], dtype=np.float32)
            raw_obs = np.concatenate([market_obs, internal_obs])
            obs = np.tanh(raw_obs) # Normalize like env
            
            # Normalize using loaded stats
            # VecNormalize expects shape (n_envs, obs_dim)
            obs_tensor = env.normalize_obs(obs.reshape(1, -1))
            
            # Predict
            action, _ = model.predict(obs_tensor, deterministic=True)
            
            # Decode Action
            act_val = action[0][0]
            size_pct = np.clip(abs(action[0][1]), 0.01, 1.0)
            
            current_price = df.iloc[-1]['close']
            print(f"Time: {df.iloc[-1]['timestamp']} | Price: {current_price} | Action Raw: {act_val:.4f}")
            
            # FIXED SIZE FOR TESTING (Safety First)
            # 0.002 BTC is approx $180 at $90k price
            amount = 0.002 
            
            if act_val > 0.3:
                print(f"🚀 SIGNAL: BUY (Size: {amount} BTC)")
                try:
                    order = await exchange.create_market_buy_order(SYMBOL, amount)
                    print(f"✅ ORDER EXECUTED: {order['id']}")
                except Exception as e:
                    print(f"❌ ORDER FAILED: {e}")
                    
            elif act_val < -0.3:
                print(f"🔻 SIGNAL: SELL (Size: {amount} BTC)")
                try:
                    order = await exchange.create_market_sell_order(SYMBOL, amount)
                    print(f"✅ ORDER EXECUTED: {order['id']}")
                except Exception as e:
                    print(f"❌ ORDER FAILED: {e}")
            else:
                print("💤 HOLD")
            
            # Wait for next minute
            await asyncio.sleep(60)
            
        except KeyboardInterrupt:
            print("🛑 Stopped.")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            await asyncio.sleep(5)

    await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
