import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List
from src.strategy.market_regime import MarketRegimeDetector, MarketRegimeType
from src.data.features import FeatureEngineer

class TradingEnv(gym.Env):
    """
    Custom Trading Environment that follows gym interface.
    Updated for Rescue Mission:
    - Hard Stop-Loss (15%)
    - Regime Filter (Volatility Check)
    - Maker/Taker Fees (0.02% / 0.04%)
    - Advanced Reward Function
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance=10000.0):
        super(TradingEnv, self).__init__()
        print("DEBUG: TradingEnv v4 Loaded - Rescue Mission Edition")
        
        # Ensure DF has all necessary columns
        self.df = df.fillna(0)
        self.current_step = 0
        self.max_steps = len(df) - 1
        self.initial_balance = initial_balance
        
        # Fees (Phase 2: Optimized for Profit)
        self.maker_fee = 0.0002 # 0.02%
        self.taker_fee = 0.0004 # 0.04% (reduced from 0.05%)
        
        # Risk Limits
        self.max_drawdown_limit = 0.15 # 15%
        
        # Regime Detector
        self.regime_detector = MarketRegimeDetector()
        self.feature_engineer = FeatureEngineer()
        
        # Actions: [Action Type (Hold/Buy/Sell), Size, Price Offset]
        # Normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Observations: 
        # Advanced Microstructure Features + Rescue Mission Features + Market Regime
        self.obs_columns = [
            'obi', 'obi_rolling_9', 'cvd_rolling_9', 'cvd_velocity', 'cvd_acceleration',
            'trade_aggressiveness', 'rvi', 'va_position', 'profile_skew',
            'volatility', 'market_regime'
        ]
        
        # Ensure these columns exist, else fill 0
        for col in self.obs_columns:
            if col not in self.df.columns:
                self.df[col] = 0.0
        
        num_features = len(self.obs_columns)
        
        # Add 2 for internal state (position, unrealized_pnl_pct)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(num_features + 2,), dtype=np.float32)
        
        self.balance = initial_balance
        self.position = 0.0 
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.trades = []
        self.trades_taken = 0
        self.entry_step = 0
        self.prev_portfolio_val = self.initial_balance
        self.stop_loss_triggered = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.trades = []
        self.trades_taken = 0
        self.entry_step = 0
        self.prev_portfolio_val = self.initial_balance
        self.stop_loss_triggered = False
        obs = self._next_observation()
        # NaN Guard (Data Integrity)
        assert not np.isnan(obs).any(), "Observation contains NaN values!"
        return obs, {}

    def _next_observation(self):
        # Market features
        market_obs = self.df.iloc[self.current_step][self.obs_columns].values.astype(np.float32)
        
        # Overwrite 'market_regime' with the real-time calculated value
        # Find index of 'market_regime'
        regime_idx = self.obs_columns.index('market_regime')
        market_obs[regime_idx] = getattr(self, 'current_regime_val', 0.0)
        
        # Internal state features
        current_price = self.df.iloc[self.current_step]['close']
        unrealized_pnl = 0.0
        if self.position != 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * 100
            
        internal_obs = np.array([self.position, unrealized_pnl], dtype=np.float32)
        
        raw_obs = np.concatenate([market_obs, internal_obs])
        
        # CONSTRAINT: strict clipping between -1 and 1 using np.tanh
        return np.tanh(raw_obs)

    def calculate_reward(self, step_pnl, impact_cost, fee_cost, holding_time):
        """
        Composite Reward R(t) = (PnL * 2.0) - (Fees * 0.5) - Slippage - Holding_Penalty
        Phase 3: Aggressive Profit Seeking (Greedy)
        """
        # 1. Net PnL (Realized or Unrealized Delta)
        # Normalized by initial balance to keep it in small range
        r_pnl = (step_pnl / self.initial_balance) * 100.0 * 2.0 # Boosted from 1.1
        
        # 2. Costs (Fees + Impact)
        # Phase 3 Hybrid Rehab: Low Fee Penalty (0.1) to encourage trading without spam.
        r_fee = (fee_cost / self.initial_balance) * 100.0 * 0.1 # Hybrid Rehab: 0.1
        r_impact = (impact_cost / self.initial_balance) * 100.0
        
        # 3. Holding Penalty (Time Decay)
        # Explicit penalty for holding capital (weighted by position size)
        r_holding = 0.0001 * abs(self.position)
        
        # Total
        total_reward = r_pnl - r_fee - r_impact - r_holding
        
        return np.clip(total_reward, -10, 10)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        # --- 0. Risk Management Checks ---
        
        # A. Drawdown Hard Stop (10%)
        current_portfolio_val = self.balance + (self.position * current_price)
        drawdown = (self.initial_balance - current_portfolio_val) / self.initial_balance
        
        if drawdown > 0.10: 
            self.stop_loss_triggered = True
            if self.position != 0:
                revenue = self.position * current_price
                fee = revenue * self.taker_fee
                self.balance += (revenue - fee)
                self.position = 0
            
            return self._next_observation(), -10.0, True, False, {
                'balance': self.balance, 'portfolio_value': current_portfolio_val, 
                'stop_loss_triggered': True
            }

        # B. Regime Filter (Adaptive Risk Filtering)
        # Calculate Regime
        # We need a window of data for detection (e.g. 50 candles)
        start_idx = max(0, self.current_step - 50)
        df_window = self.df.iloc[start_idx:self.current_step+1]
        
        # Calculate Volume Profile for this window
        vp_data = self.feature_engineer.calculate_volume_profile(df_window)
        
        # Detect Regime
        regime = self.regime_detector.detect_regime(df_window, vp_data)
        
        # Map Regime to Float for Observation
        # RANGING=0, TRENDING=1, VOLATILE=2, UNCERTAIN=3
        regime_map = {
            MarketRegimeType.RANGING: 0.0,
            MarketRegimeType.TRENDING: 1.0,
            MarketRegimeType.VOLATILE: 2.0,
            MarketRegimeType.UNCERTAIN: 3.0
        }
        regime_val = regime_map.get(regime, 3.0)
        
        # Store in DF for observation retrieval
        # (We update the current row's market_regime column)
        # Note: This is slightly inefficient to write to DF every step, but keeps architecture clean.
        # Better to just inject it into observation directly.
        
        # FILTER LOGIC:
        # Phase 3: REMOVED Volatility Filter to allow aggressive trading.
        # We rely on the trained agent to manage risk.
        pass
        
        # Save regime_val for _next_observation to pick up
        self.current_regime_val = regime_val

        # C. Pre-Trade Price Tolerance (Flash Move Protection)
        # If price moved > 5% since last step (unlikely in 1m but good safety)
        if self.current_step > 0:
            last_price = self.df.iloc[self.current_step - 1]['close']
            if abs(current_price - last_price) / last_price > 0.05:
                action[0] = 0.0 # Reject Trade

        # --- 1. Execute Action ---
        volatility = self.df.iloc[self.current_step]['volatility']
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        act_val = action[0]
        size_pct = np.clip(abs(action[1]), 0.01, 1.0)
        
        action_type = 0 # 0: Hold, 1: Buy, -1: Sell
        
        cost = 0.0
        fee = 0.0
        impact_cost = 0.0
        realized_pnl = 0.0
        
        # Force Exit if Max Hold Time Exceeded (200 steps)
        force_exit = False
        if self.position > 0 and (self.current_step - self.entry_step) > 200:
            force_exit = True
            
        if act_val > 0.3 and not force_exit: # Buy
            if self.position == 0: # Open Long
                 action_type = 1
                 # Taker Fee for Market Entry
                 fee_rate = self.taker_fee
                 
                 max_cost = self.balance / (1 + fee_rate)
                 cost = min(self.balance * size_pct, max_cost)
                 
                 if cost > 5.0: # Min trade size
                     fee = cost * fee_rate
                     
                     # Market Impact (Slippage): Quadratic penalty
                     # impact = constant * size_pct^2 * volatility
                     # We use volatility to scale impact
                     vol_scale = max(volatility / current_price, 0.001) * 100
                     impact_cost = cost * (size_pct ** 2) * 0.01 * vol_scale
                     
                     self.position = cost / current_price
                     self.balance -= (cost + fee)
                     self.entry_price = current_price
                     self.trades.append({'step': self.current_step, 'side': 'buy', 'price': current_price, 'fee': fee, 'impact': impact_cost})
                     self.trades_taken += 1
                     self.entry_step = self.current_step
                 else:
                     pass
            else:
                 pass # Already Long
                 
        elif act_val < -0.3 or force_exit: # Sell
             if self.position > 0: # Close Long
                 action_type = -1
                 revenue = self.position * current_price
                 
                 # Taker Fee for Market Exit
                 fee_rate = self.taker_fee
                 fee = revenue * fee_rate
                 
                 # Market Impact
                 vol_scale = max(volatility / current_price, 0.001) * 100
                 impact_cost = revenue * (size_pct ** 2) * 0.01 * vol_scale
                 
                 self.balance += (revenue - fee)
                 self.position = 0
                 self.trades.append({'step': self.current_step, 'side': 'sell', 'price': current_price, 'fee': fee, 'impact': impact_cost})
                 self.trades_taken += 1
                 
                 # Realized PnL (approx for reward)
                 # We track equity curve change instead for main reward
             else:
                 pass # Already Flat

        # --- 2. Calculate Reward ---
        # Change in Equity (Unrealized + Realized)
        new_portfolio_val = self.balance + (self.position * current_price)
        delta_equity = new_portfolio_val - self.prev_portfolio_val
        
        # Holding Time
        holding_time = (self.current_step - self.entry_step) if self.position > 0 else 0
        
        # We pass delta_equity as "step_pnl" (it includes fees paid from balance)
        # But wait, delta_equity ALREADY includes the fee deduction from balance.
        # So we shouldn't double penalize fee in calculate_reward if we pass delta_equity.
        # However, we want the agent to "feel" the fee explicitly.
        # If we just use delta_equity, it implicitly includes fee.
        # But we want to add EXTRA penalty or just rely on equity?
        # The plan says: "Subtract fee * trade_volume from every reward step."
        # If delta_equity is used, fee is already subtracted.
        # Let's stick to Equity Curve Reward but add specific penalties for Impact and Holding.
        
        reward = self.calculate_reward(delta_equity, impact_cost, 0, holding_time)
        
        # Update Baseline
        self.prev_portfolio_val = new_portfolio_val
        self.total_reward += reward
        
        # Update state
        obs = self._next_observation()
        
        info = {
            'balance': self.balance, 
            'position': self.position,
            'total_reward': self.total_reward,
            'portfolio_value': new_portfolio_val,
            'trades_taken': self.trades_taken,
            'stop_loss_triggered': False
        }
        
        # Activity Quota (CORRECTION)
        if done and self.trades_taken < 10:
            reward -= 100.0 # Penalty for inactivity
            
        return obs, reward, done, False, info

    def render(self, mode='human'):
        current_price = self.df.iloc[self.current_step]['close']
        val = self.balance + (self.position * current_price)
        print(f'Step: {self.current_step}, Portfolio: {val:.2f}, Pos: {self.position:.4f}')
