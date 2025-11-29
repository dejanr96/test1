import asyncio
import os
import time
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gym
from gym import spaces

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "quant_execution_model.zip"
STATS_PATH = "quant_execution_stats.pkl"
SYMBOL = "BTC/USDT" 
TIMEFRAME = "1m"
TRADE_SIZE = 0.005 # 0.005 BTC (~$450)
MAX_POSITION = 0.02 # Max accumulated position

# ==========================================
# 2. DUMMY ENV FOR NORMALIZATION
# ==========================================
class LiveDummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Matches the observation space of the trained model
        # [obi, cvd_vel, trade_agg, dist_vwap, vol, norm_pos, norm_entry]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
    def reset(self, seed=None, options=None): return np.zeros(7, dtype=np.float32), {}
    def step(self, action): return np.zeros(7, dtype=np.float32), 0.0, False, False, {}

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
class LiveFeatureEngineer:
    def __init__(self):
        self.window_size = 1000 # Reduced from 2000 to fit within 1500 limit

    def calculate_features(self, ohlcv_df, orderbook, trades):
        """
        ohlcv_df: DataFrame with ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        orderbook: Dict with 'bids', 'asks' (top 1 is enough)
        trades: List of recent trades [{'price': ..., 'amount': ..., 'side': ...}]
        """
        # 1. Basic Data
        df = ohlcv_df.copy()
        df['mid_price'] = df['close'] # Proxy for history
        
        # Current Snapshot Overwrite (for price)
        best_bid = orderbook['bids'][0][0] if orderbook['bids'] else df['close'].iloc[-1]
        best_ask = orderbook['asks'][0][0] if orderbook['asks'] else df['close'].iloc[-1]
        current_mid = (best_bid + best_ask) / 2.0
        
        # 2. OBI (Order Book Imbalance)
        # Sum top 5 levels to avoid "dust" bias at top level
        bid_qty = sum([b[1] for b in orderbook['bids'][:5]]) if orderbook['bids'] else 0.0
        ask_qty = sum([a[1] for a in orderbook['asks'][:5]]) if orderbook['asks'] else 0.0
        obi = (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-6)
        
        # 3. CVD Velocity (Change in Net Flow)
        df['net_flow_proxy'] = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        df['cvd'] = df['net_flow_proxy'].cumsum()
        cvd_velocity = df['cvd'].diff(10).iloc[-1] # 10-period velocity
        if np.isnan(cvd_velocity): cvd_velocity = 0.0
        
        # 4. Trade Aggressiveness
        # Use LAST COMPLETED CANDLE for volume to avoid "partial candle" bias
        # But for liquidity we use current snapshot.
        current_vol = df['volume'].iloc[-2] if len(df) > 1 else df['volume'].iloc[-1]
        liquidity = bid_qty + ask_qty + 1e-6
        trade_aggressiveness = current_vol / liquidity
        
        # 5. VWAP & Dist VWAP
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['pv'] = df['tp'] * df['volume']
        
        # Use min_periods=1 to ensure we get values even if < window_size
        cum_pv = df['pv'].rolling(self.window_size, min_periods=1).sum()
        cum_vol = df['volume'].rolling(self.window_size, min_periods=1).sum()
        vwap = cum_pv / (cum_vol + 1e-6)
        
        current_vwap = vwap.iloc[-1]
        dist_vwap = (current_mid - current_vwap) / (current_vwap + 1e-6) * 1000
        
        # 6. Volatility
        df['returns'] = df['close'].pct_change()
        volatility = df['returns'].rolling(20).std().iloc[-1] * 10000
        if np.isnan(volatility): volatility = 0.0
        
        return np.array([
            obi, 
            cvd_velocity, 
            trade_aggressiveness, 
            dist_vwap, 
            volatility
        ], dtype=np.float32)

# ==========================================
# 4. PAPER TRADING SIMULATOR
# ==========================================
class PaperTrader:
    def __init__(self, initial_balance=10000.0):
        self.balance = initial_balance
        self.position = 0.0 # Positive = Long, Negative = Short
        self.entry_price = 0.0
        
    def execute(self, side, amount, price):
        cost = amount * price
        fee = cost * 0.0005 # 0.05% Taker Fee
        
        if side == 'buy':
            if self.position < 0: # Closing Short
                # PnL = (Entry - Exit) * Size
                pnl = (self.entry_price - price) * abs(self.position)
                self.balance += pnl
                self.position = 0.0
                self.entry_price = 0.0
                print(f"📉 CLOSED SHORT | PnL: ${pnl:.2f} | Bal: ${self.balance:.2f}")
            else: # Opening Long
                self.position += amount
                self.entry_price = price
                print(f"📈 OPEN LONG | Price: {price:.2f}")
                
        elif side == 'sell':
            if self.position > 0: # Closing Long
                # PnL = (Exit - Entry) * Size
                pnl = (price - self.entry_price) * abs(self.position)
                self.balance += pnl
                self.position = 0.0
                self.entry_price = 0.0
                print(f"📈 CLOSED LONG | PnL: ${pnl:.2f} | Bal: ${self.balance:.2f}")
            else: # Opening Short
                self.position -= amount
                self.entry_price = price
                print(f"📉 OPEN SHORT | Price: {price:.2f}")
                
        self.balance -= fee # Deduct fee

    def get_unrealized_pnl(self, current_price):
        if self.position == 0: return 0.0
        if self.position > 0: return (current_price - self.entry_price) * abs(self.position)
        return (self.entry_price - current_price) * abs(self.position)

# ==========================================
# 5. MAIN LOOP
# ==========================================
async def main():
    # 1. CONNECT TO REAL MAINNET (Public Data Only)
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    # No API Keys needed for public data!
    
    print("🌍 Connected to BINANCE MAINNET (Real Data)")
    print("📝 Paper Trading Mode ACTIVE (Fake Money)")
    
    # Load Model
    print("🧠 Loading Brain...")
    model = PPO.load(MODEL_PATH)
    dummy_env = DummyVecEnv([lambda: LiveDummyEnv()])
    env = VecNormalize.load(STATS_PATH, dummy_env)
    env.training = False
    env.norm_reward = False
    
    # Initialize Paper Trader
    paper = PaperTrader(initial_balance=10000.0)
    fe = LiveFeatureEngineer()
    
    print(f"💰 Initial Paper Balance: ${paper.balance:.2f}")
    print("🎧 Listening for REAL market data...")
    
    live_memory = []
    MEMORY_SIZE = 128
    last_trade_time = 0
    TRADE_SIZE = 0.005 # 0.005 BTC (~$450)
    
    while True:
        try:
            # 1. Fetch REAL Data
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1500)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            orderbook = await exchange.fetch_order_book(SYMBOL, limit=20) # Deeper book for real data
            
            current_price = df['close'].iloc[-1]
            
            # 2. Paper Position Updates
            current_pos = paper.position
            entry_price = paper.entry_price
            position_pnl = paper.get_unrealized_pnl(current_price)
            session_pnl = (paper.balance + position_pnl) - 10000.0
            
            # Features
            market_features = fe.calculate_features(df, orderbook, [])
            norm_pos = 1.0 if current_pos > 0 else (-1.0 if current_pos < 0 else 0.0)
            norm_entry = 0.0
            if current_pos != 0 and entry_price > 0:
                norm_entry = np.log(current_price / entry_price)
            
            raw_obs = np.concatenate([market_features, [norm_pos, norm_entry]])
            if np.isnan(raw_obs).any(): raw_obs = np.nan_to_num(raw_obs)
            
            # Normalize
            obs_tensor = env.normalize_obs(raw_obs.reshape(1, -1))
            if np.isnan(obs_tensor).any(): obs_tensor = np.nan_to_num(obs_tensor)

            # Predict
            action, _ = model.predict(obs_tensor, deterministic=True)
            action_int = int(action[0])
            
            # Execute
            print(f"Time: {pd.Timestamp.now()} | Price: {current_price:.2f} | U.PnL: ${position_pnl:.2f} | Sess PnL: ${session_pnl:.2f} | Act: {action_int}")
            print(f"   📊 Features: OBI={raw_obs[0]:.2f} | CVD={raw_obs[1]:.2f} | Agg={raw_obs[2]:.2f} | VWAP={raw_obs[3]:.2f} | Vol={raw_obs[4]:.2f}")
            
            # DEBUG: Check Orderbook (Sum Top 5)
            bid_q = sum([b[1] for b in orderbook['bids'][:5]]) if orderbook['bids'] else 0
            ask_q = sum([a[1] for a in orderbook['asks'][:5]]) if orderbook['asks'] else 0
            print(f"   📖 Book (Top 5): BidQty={bid_q:.4f} | AskQty={ask_q:.4f}")
            
            # --- MODEL EXECUTION (PAPER) ---
            target_pos_type = 0
            if action_int == 1: target_pos_type = 1
            elif action_int == 2: target_pos_type = -1
            
            current_pos_type = 0
            if current_pos > 0.0001: current_pos_type = 1
            elif current_pos < -0.0001: current_pos_type = -1
            
            now = time.time()
            cooldown_seconds = 60 # Faster cooldown for paper testing
            
            if target_pos_type != current_pos_type:
                is_entry = (current_pos_type == 0) and (target_pos_type != 0)
                is_flip = (current_pos_type != 0) and (target_pos_type != 0) and (current_pos_type != target_pos_type)
                
                if (is_entry or is_flip) and (now - last_trade_time < cooldown_seconds):
                    print(f"⏳ Signal {action_int} ignored (Cooldown)")
                else:
                    print(f"⚡ ACTION SIGNAL: {action_int}")
                    
                    # SIMULATE EXECUTION
                    if current_pos_type != 0: # Close existing
                        side = 'sell' if current_pos > 0 else 'buy'
                        paper.execute(side, abs(current_pos), current_price)
                        
                    if target_pos_type != 0: # Open new
                        side = 'buy' if target_pos_type == 1 else 'sell'
                        paper.execute(side, TRADE_SIZE, current_price)
                    
                    last_trade_time = now
            
            # Online Learning Collection (Keep collecting real data!)
            if len(live_memory) > 0:
                live_memory[-1]['reward'] = np.log(paper.balance / 10000.0) # Simple reward proxy
                live_memory[-1]['next_obs'] = raw_obs
                live_memory[-1]['done'] = False
            
            live_memory.append({
                'obs': raw_obs,
                'action': action_int,
                'reward': 0.0,
                'next_obs': None,
                'done': False
            })
            
            if len(live_memory) >= MEMORY_SIZE:
                print(f"🧠 Memory Full. Learning from REAL experience...")
                live_memory = [] # Reset
            
            await asyncio.sleep(60) # 1 Minute Candles
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await asyncio.sleep(5)

    await exchange.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")
