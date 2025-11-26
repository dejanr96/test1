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
SYMBOL = "BTC/USDT" 
TIMEFRAME = "1m"
TRADE_SIZE = 0.002  # 0.002 BTC fixed size for testing

# ⚠️ SECURITY WARNING: Use Environment Variables in production!
API_KEY = os.getenv("BINANCE_DEMO_API_KEY", "JpQ1NMelCU4kfQ1N48jettZn8NcneaIPHtJXRaZZNuSMZFIXnTC4HQQlrzlsQgbj")
API_SECRET = os.getenv("BINANCE_DEMO_SECRET", "wnz2GlTNZw9skUWZhSaZ8mYcv0tfw09dTEwVJChetfxisdLqAH3CFzlItFlWev5E")

EXCHANGE_OPTIONS = {
    'defaultType': 'future',
}

async def get_real_position(exchange, symbol):
    """Fetches the current open position size and unrealized PnL for the symbol."""
    try:
        # For Binance Futures, we fetch positions
        # Passing [symbol] might filter too strictly if CCXT symbol differs from Exchange symbol
        # Let's fetch all and filter manually to be safe, or debug what we get.
        positions = await exchange.fetch_positions() 
        
        target_found = False
        print(f"DEBUG: Checking {len(positions)} positions...")
        for pos in positions:
            # Debug: Print what we see
            print(f"DEBUG: Found Position: {pos['symbol']} Size: {pos['contracts']}")
            
            # Check for match (Exact or substring for safety)
            # e.g. symbol="BTC/USDT", pos['symbol'] might be "BTC/USDT:USDT"
            if pos['symbol'] == symbol or pos['symbol'].startswith(symbol):
                size = float(pos['contracts']) if pos['contracts'] else 0.0
                if pos['side'] == 'short':
                    size = -size
                pnl = float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0.0
                
                # Only return if size is non-zero, OR if we found the exact match
                # Actually, we want to find the entry even if size is 0 to confirm we are flat.
                return size, pnl
        
        print(f"⚠️ Warning: Position for {symbol} not found in fetch_positions()")
        return 0.0, 0.0
    except Exception as e:
        print(f"⚠️ Error fetching position: {e}")
        return 0.0, 0.0

async def main():
    print(f"--- 🟢 STARTING LIVE PAPER TRADER: {SYMBOL} ---")
    
    # 1. Initialize Exchange
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'options': EXCHANGE_OPTIONS
    })
    
    if hasattr(exchange, 'enable_demo_trading'):
        exchange.enable_demo_trading(True)
        print("✅ Demo Trading Enabled.")
    else:
        exchange.set_sandbox_mode(True)
    
    await exchange.load_markets()
    
    # Set Leverage to 1x for safety
    try:
        await exchange.set_leverage(1, SYMBOL)
        print("✅ Leverage set to 1x.")
    except:
        pass

    # 2. Load Model & Stats
    print("🧠 Loading Brain...")
    # Dummy Env required to load stats structure
    dummy_env = DummyVecEnv([lambda: TradingEnv(pd.DataFrame(), initial_balance=10000)])
    env = VecNormalize.load(STATS_PATH, dummy_env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(MODEL_PATH)
    print("✅ Model Loaded.")

    # 3. Live Loop
    print("🎧 Listening for market data...")
    fe = FeatureEngineer()
    
    # Track the last candle we processed to prevent double-firing
    last_processed_timestamp = None

    while True:
        try:
            # A. Fetch Data
            # INCREASED LIMIT TO 1000 to prevent "Cold Start" NaNs/Zeros
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1000)
            if not ohlcv:
                print("⚠️ No data received.")
                await asyncio.sleep(5)
                continue
            
            # B. Parse Data
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # C. CHECK TIMESTAMP (The Fix)
            # Get the timestamp of the *closed* candle (the second to last one is usually the last closed one)
            # But for real-time decision, we usually look at the *current* building candle or the *just closed* one.
            # Let's assume we trade on the *close* of the last completed candle (-2 in list usually, or -1 if closed).
            # Standard approach: Use the timestamp of the last row.
            current_candle_ts = ohlcv[-1][0]
            
            if last_processed_timestamp == current_candle_ts:
                # We already traded this minute. Wait.
                await asyncio.sleep(1) 
                continue
            
            # D. Prepare Features
            df = fe.add_advanced_microstructure_features(df)
            
            # E. Get REAL State (The Fix)
            real_position, real_pnl = await get_real_position(exchange, SYMBOL)
            
            # F. Construct Observation
            obs_columns = [
                'obi', 'cvd_velocity', 'trade_aggressiveness', 
                'rvi', 'va_position', 'profile_skew'
            ]
            market_obs = df.iloc[-1][obs_columns].values.astype(np.float32)
            
            # Crucial: Feed the ACTUAL wallet state to the brain
            internal_obs = np.array([real_position, real_pnl], dtype=np.float32)
            
            raw_obs = np.concatenate([market_obs, internal_obs])
            obs_normalized = np.tanh(raw_obs) # Pre-scaling (if you did this in Env)
            
            # Normalize using the saved training stats
            obs_tensor = env.normalize_obs(obs_normalized.reshape(1, -1))
            
            # --- DIAGNOSTIC PRINT (Raw Features) ---
            # Check for Zeros/NaNs that might cause "Panic" signals
            print(f"DEBUG: Raw Features: {raw_obs}")
            
            # G. Predict
            action, _ = model.predict(obs_tensor, deterministic=True)
            act_val = action[0][0]
            
            # H. Execute Logic
            current_price = df.iloc[-1]['close']
            
            # --- LIQUIDITY CHECK ---
            # fetch_ticker returned None for bid/ask, so we use order book
            orderbook = await exchange.fetch_order_book(SYMBOL, limit=5)
            bid = orderbook['bids'][0][0] if len(orderbook['bids']) > 0 else 0.0
            ask = orderbook['asks'][0][0] if len(orderbook['asks']) > 0 else 0.0
            spread = ask - bid if (ask > 0 and bid > 0) else 0.0
            
            print(f"Time: {df.iloc[-1]['timestamp']} | Last: {current_price} | Bid: {bid} | Ask: {ask} | Spread: {spread:.2f} | Action Raw: {act_val:.4f} | Pos: {real_position}")
            
            print(f"Time: {df.iloc[-1]['timestamp']} | Last: {current_price} | Bid: {bid} | Ask: {ask} | Spread: {spread:.2f} | Action Raw: {act_val:.4f} | Pos: {real_position}")

            # --- DECISION LOGIC ---
            threshold = 0.3
            trade_executed = False

            # 1. BUY SIGNAL
            if act_val > threshold:
                if real_position <= 0: # We are Flat or Short
                    print(f"🚀 SIGNAL: BUY (Close Short/Open Long)")
                    if real_position < 0:
                        print("  -> Closing Short Position")
                        try:
                            await exchange.create_market_buy_order(SYMBOL, abs(real_position))
                        except Exception as e:
                            print(f"❌ ORDER FAILED: {e}")
                            
                    if real_position == 0:
                        print(f"  -> Opening Long {TRADE_SIZE} BTC")
                        try:
                            await exchange.create_market_buy_order(SYMBOL, TRADE_SIZE)
                        except Exception as e:
                            print(f"❌ ORDER FAILED: {e}")
                    trade_executed = True
                else:
                    print("  -> Signal BUY, but already Long. Holding.")

            # 2. SELL SIGNAL
            elif act_val < -threshold:
                if real_position >= 0: # We are Flat or Long
                    print(f"🔻 SIGNAL: SELL (Close Long/Open Short)")
                    if real_position > 0:
                        print("  -> Closing Long Position")
                        try:
                            await exchange.create_market_sell_order(SYMBOL, abs(real_position))
                        except Exception as e:
                            print(f"❌ ORDER FAILED: {e}")
                            
                    if real_position == 0:
                        print(f"  -> Opening Short {TRADE_SIZE} BTC")
                        try:
                            await exchange.create_market_sell_order(SYMBOL, TRADE_SIZE)
                        except Exception as e:
                            print(f"❌ ORDER FAILED: {e}")
                    trade_executed = True
                else:
                    print("  -> Signal SELL, but already Short. Holding.")

            # 3. HOLD SIGNAL
            else:
                print("💤 SIGNAL: HOLD")

            # --- CRITICAL FIX: ALWAYS MARK CANDLE AS PROCESSED ---
            # Whether we traded, held, or were blocked by logic, 
            # we are done with this minute.
            last_processed_timestamp = current_candle_ts
            
            # Sleep to prevent CPU burn, but we are safe now because of the timestamp check
            await asyncio.sleep(1)

        except Exception as e:
            print(f"❌ Error in loop: {e}")
            await asyncio.sleep(5)

    await exchange.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program stopped.")
