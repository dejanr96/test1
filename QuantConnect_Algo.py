# QUANTCONNECT ALGORITHM
# ------------------------------------------------------------------------------
# Copy this entire file into your QuantConnect Project (main.py)
# ------------------------------------------------------------------------------
from AlgorithmImports import *
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gym
from gym import spaces
import os

# ==========================================
# 1. FEATURE ENGINEER (ADAPTED FOR QC)
# ==========================================
class QCFeatureEngineer:
    def __init__(self, window_size=1000):
        self.window_size = window_size

    def calculate_features(self, history_df, slice_data, symbol):
        """
        history_df: DataFrame with MultiIndex (Symbol, Time) and columns [open, high, low, close, volume]
        slice_data: Current Slice object from OnData
        symbol: The Symbol object
        """
        # 1. Prepare Data
        # QC History comes with MultiIndex. We need to reset or access correctly.
        # We assume history_df is for the specific symbol.
        df = history_df.loc[symbol].copy()
        
        # Current Snapshot
        if not slice_data.ContainsKey(symbol): return None
        bar = slice_data[symbol]
        quote = slice_data.QuoteBars[symbol] if slice_data.QuoteBars.ContainsKey(symbol) else None
        
        current_price = bar.Close
        
        # 2. OBI (Order Book Imbalance)
        # QC provides QuoteBar (Best Bid/Ask). We don't have full L2 book in standard equity/crypto data usually 
        # unless we subscribe to L2. For simplicity, we use L1 (Best Bid/Ask).
        if quote:
            bid_qty = quote.BidSize
            ask_qty = quote.AskSize
            obi = (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-6)
        else:
            obi = 0.0
            
        # 3. CVD Velocity
        # Proxy: (Close - Open) * Volume
        df['net_flow'] = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        df['cvd'] = df['net_flow'].cumsum()
        cvd_velocity = df['cvd'].diff(10).iloc[-1]
        if np.isnan(cvd_velocity): cvd_velocity = 0.0
        
        # 4. Trade Aggressiveness
        # Volume / Liquidity
        current_vol = df['volume'].iloc[-1]
        liquidity = (quote.BidSize + quote.AskSize) if quote else 1.0
        trade_aggressiveness = current_vol / (liquidity + 1e-6)
        
        # 5. VWAP & Dist VWAP
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['pv'] = df['tp'] * df['volume']
        cum_pv = df['pv'].rolling(self.window_size, min_periods=1).sum()
        cum_vol = df['volume'].rolling(self.window_size, min_periods=1).sum()
        vwap = cum_pv / (cum_vol + 1e-6)
        
        current_vwap = vwap.iloc[-1]
        dist_vwap = (current_price - current_vwap) / (current_vwap + 1e-6) * 1000
        
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
# 2. DUMMY ENV (FOR STATS LOADING)
# ==========================================
class LiveDummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
    def reset(self, seed=None, options=None): return np.zeros(7, dtype=np.float32), {}
    def step(self, action): return np.zeros(7, dtype=np.float32), 0.0, False, False, {}

# ==========================================
# 3. MAIN ALGORITHM
# ==========================================
class PPO_HFT_Algo(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 1, 1)  # Set Start Date
        self.SetCash(10000)             # Set Strategy Cash
        
        # 1. Add Crypto
        self.symbol = self.AddCrypto("BTCUSDT", Resolution.Minute).Symbol
        
        # 2. AUTO-DOWNLOAD MODEL (Bypass Upload Limits)
        # ---------------------------------------------------------------
        # RAW GITHUB LINKS (Auto-Filled)
        MODEL_URL = "https://raw.githubusercontent.com/dejanr96/test1/master/quant_execution_model.zip"
        STATS_URL = "https://raw.githubusercontent.com/dejanr96/test1/master/quant_execution_stats.pkl"
        # ---------------------------------------------------------------
        
        self.model_path = "quant_execution_model.zip"
        self.stats_path = "quant_execution_stats.pkl"
        
        self.Debug("⬇️ Downloading Model & Stats...")
        try:
            # Download Model
            model_data = self.Download(MODEL_URL)
            with open(self.model_path, 'wb') as f:
                f.write(model_data.encode('latin1')) # QC Download returns string, need bytes hack or use requests if available
            
            # Download Stats
            stats_data = self.Download(STATS_URL)
            with open(self.stats_path, 'wb') as f:
                f.write(stats_data.encode('latin1'))
                
            self.Debug("✅ Download Complete.")
        except Exception as e:
            self.Debug(f"❌ Download Failed: {e}")
            self.Debug("⚠️ Please ensure URLs are correct and accessible!")
            return

        self.Debug("🧠 Loading Brain...")
        
        # Hack to load stats
        dummy_env = DummyVecEnv([lambda: LiveDummyEnv()])
        self.env = VecNormalize.load(self.stats_path, dummy_env)
        self.env.training = False
        self.env.norm_reward = False
        
        self.model = PPO.load(self.model_path)
        self.Debug("✅ Model Loaded.")
        
        self.fe = QCFeatureEngineer()
        self.SetWarmUp(1000) # Warm up for VWAP
        
        self.entry_price = 0.0
        self.last_trade_time = self.Time

    def OnData(self, data: Slice):
        if self.IsWarmingUp: return
        if not data.ContainsKey(self.symbol): return
        
        # 1. Get History
        history = self.History(self.symbol, 1000, Resolution.Minute)
        if history.empty: return
        
        # 2. Calculate Features
        raw_features = self.fe.calculate_features(history, data, self.symbol)
        if raw_features is None: return
        
        # 3. Position State
        current_pos = self.Portfolio[self.symbol].Quantity
        avg_price = self.Portfolio[self.symbol].AveragePrice
        
        norm_pos = 1.0 if current_pos > 0 else (-1.0 if current_pos < 0 else 0.0)
        norm_entry = 0.0
        current_price = data[self.symbol].Close
        
        if current_pos != 0 and avg_price > 0:
            norm_entry = np.log(current_price / avg_price)
            
        raw_obs = np.concatenate([raw_features, [norm_pos, norm_entry]])
        if np.isnan(raw_obs).any(): raw_obs = np.nan_to_num(raw_obs)
        
        # 4. Normalize & Predict
        obs_tensor = self.env.normalize_obs(raw_obs.reshape(1, -1))
        action, _ = self.model.predict(obs_tensor, deterministic=True)
        action_int = int(action[0])
        
        # 5. Execute
        # Action 0: Hold
        # Action 1: Long
        # Action 2: Short
        
        target_qty = 0.01 # Fixed size for now
        
        if action_int == 1: # GO LONG
            if current_pos <= 0:
                self.SetHoldings(self.symbol, 1.0) # Full allocation Long
                self.Debug(f"📈 BUY | Price: {current_price}")
                
        elif action_int == 2: # GO SHORT
            if current_pos >= 0:
                self.SetHoldings(self.symbol, -1.0) # Full allocation Short
                self.Debug(f"📉 SELL | Price: {current_price}")
                
        # Logging
        # self.Plot("Trade", "Price", current_price)
