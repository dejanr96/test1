import asyncio
import logging
import pandas as pd
import numpy as np
import time
import os
import aiohttp
import json
from src.data.ingestion import DataIngestionService
from src.data.features import FeatureEngineer
from src.agent.ppo_agent import PPOAgent
from src.env.trading_env import TradingEnv
from src.strategy.adaptive_risk import AdaptiveRiskManager
from src.strategy.market_regime import MarketRegimeDetector
from src.data.recorder import ExperienceRecorder
from stable_baselines3.common.vec_env import DummyVecEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PaperTrader")

class PendingOrder:
    def __init__(self, order_type, price, quantity, leverage, take_profit, stop_loss):
        self.order_type = order_type # 1 (Buy), -1 (Sell)
        self.price = price
        self.quantity = quantity
        self.leverage = leverage
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.timestamp = time.time()

class PaperAccount:
    def __init__(self, initial_balance=100.0):
        self.balance = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.pnl = 0.0
        self.used_margin = 0.0
        self.pending_orders: List[PendingOrder] = []
        self.take_profit = 0.0
        self.stop_loss = 0.0

    def check_pending_orders(self, current_price, high, low):
        """
        Check if any pending limit orders are filled.
        """
        executed = False
        for order in self.pending_orders[:]:
            # Buy Limit: Filled if Low <= Limit Price
            if order.order_type == 1 and low <= order.price:
                self._execute_trade(1, order.quantity, order.price, order.leverage, order.take_profit, order.stop_loss)
                self.pending_orders.remove(order)
                executed = True
                logger.info(f"LIMIT BUY FILLED at {order.price}")
            
            # Sell Limit: Filled if High >= Limit Price
            elif order.order_type == -1 and high >= order.price:
                self._execute_trade(-1, order.quantity, order.price, order.leverage, order.take_profit, order.stop_loss)
                self.pending_orders.remove(order)
                executed = True
                logger.info(f"LIMIT SELL FILLED at {order.price}")
                
            # Cancel if price moves too far away (> 0.5%)
            elif abs(current_price - order.price) / order.price > 0.005:
                self.pending_orders.remove(order)
                logger.info(f"Limit Order at {order.price} CANCELLED (Price moved away).")
                
        return executed

    def check_tp_sl(self, current_price):
        """
        Check Take Profit and Stop Loss.
        """
        if self.position == 0: return

        # Long
        if self.position > 0:
            if current_price >= self.take_profit:
                self.close_position(current_price, "TAKE PROFIT")
            elif current_price <= self.stop_loss:
                self.close_position(current_price, "STOP LOSS")
        # Short
        elif self.position < 0:
            if current_price <= self.take_profit:
                self.close_position(current_price, "TAKE PROFIT")
            elif current_price >= self.stop_loss:
                self.close_position(current_price, "STOP LOSS")

    def _execute_trade(self, action_type, quantity, price, leverage, tp, sl):
        margin = (quantity * price) / leverage
        
        if action_type == 1: # Buy
            if self.position == 0: # Open Long
                if self.balance >= margin:
                    self.position = quantity
                    self.balance -= margin
                    self.entry_price = price
                    self.used_margin = margin
                    self.take_profit = tp
                    self.stop_loss = sl
                    logger.info(f"OPEN LONG: Price={price}, Size={quantity:.4f}, TP={tp}, SL={sl}")
            elif self.position < 0: # Close Short
                self.close_position(price, "REVERSAL")
                # Re-open logic could go here, but let's keep it simple: Close first.

        elif action_type == -1: # Sell
            if self.position == 0: # Open Short
                if self.balance >= margin:
                    self.position = -quantity
                    self.balance -= margin
                    self.entry_price = price
                    self.used_margin = margin
                    self.take_profit = tp
                    self.stop_loss = sl
                    logger.info(f"OPEN SHORT: Price={price}, Size={quantity:.4f}, TP={tp}, SL={sl}")
            elif self.position > 0: # Close Long
                self.close_position(price, "REVERSAL")

    def close_position(self, price, reason):
        if self.position == 0: return
        
        profit = 0
        if self.position > 0: # Close Long
            profit = (price - self.entry_price) * self.position
        else: # Close Short
            profit = (self.entry_price - price) * abs(self.position)
            
        self.balance += (self.used_margin + profit)
        self.pnl += profit
        logger.info(f"CLOSE POSITION ({reason}): Price={price}, PnL={profit:.2f}, Bal={self.balance:.2f}")
        
        self.position = 0.0
        self.used_margin = 0.0
        self.take_profit = 0.0
        self.stop_loss = 0.0

    def place_order(self, action_type, quantity, price, leverage, order_type_str, tp, sl):
        if order_type_str == "MARKET":
            self._execute_trade(action_type, quantity, price, leverage, tp, sl)
        elif order_type_str == "LIMIT":
            # Check if we already have a pending order, if so, cancel it (only 1 active for now)
            self.pending_orders = [] 
            self.pending_orders.append(PendingOrder(action_type, price, quantity, leverage, tp, sl))
            logger.info(f"PLACED LIMIT {'BUY' if action_type==1 else 'SELL'} at {price}")


class PaperTrader:
    def __init__(self):
        # Use 'binanceusdm' for Binance USDT-M Futures
        # Symbol must be 'BTC/USDT:USDT' for linear perpetuals in CCXT
        self.data_service = DataIngestionService(exchange_id='binanceusdm', symbol='BTC/USDT:USDT', timeframe='1m')
        self.fe = FeatureEngineer()
        self.account = PaperAccount()
        self.agent = None
        self.risk_manager = AdaptiveRiskManager()
        self.regime_detector = MarketRegimeDetector()
        self.recorder = ExperienceRecorder()
        
    async def initialize(self):
        logger.info("Initializing Data Service...")
        await self.data_service.start_stream()
        
        logger.info("Fetching Historical Data (10000 candles / 7 Days)...")
        # Fetch deeper history for pre-training
        df = await self.data_service.fetch_ohlcv(limit=10000)
        df = self.fe.add_technical_indicators(df)
        
        # Add missing advanced feature columns
        for col in ['ofi', 'cvd', 'qi', 'lob_imbalance_5']:
            df[col] = 0.0
        
        self.env = TradingEnv(df)
        self.agent = PPOAgent(self.env)
        
        # Check if model exists, if not, PRE-TRAIN
        model_path = "ppo_apex_model.zip"
        if os.path.exists(model_path):
            logger.info("Loading existing model...")
            self.agent.load()
            logger.info("Agent Loaded.")
        else:
            logger.info("No existing model found. Starting PRE-TRAINING (50000 steps)...")
            # Create model explicitly if not loaded
            self.agent.create_model()
            self.agent.train(total_timesteps=50000)
            logger.info("Pre-training Complete.")

    async def retrain(self):
        """
        Retrain the agent on the latest data.
        """
        logger.info("Starting Periodic Retraining...")
        # Fetch latest 1000 candles
        df = await self.data_service.fetch_ohlcv(limit=1000)
        df = self.fe.add_technical_indicators(df)
        for col in ['ofi', 'cvd', 'qi', 'lob_imbalance_5']:
            df[col] = 0.0
            
        # Update Env
        self.env = TradingEnv(df)
        self.agent.env = DummyVecEnv([lambda: self.env])
        self.agent.model.set_env(self.agent.env)
        
        # Train for a short burst
        self.agent.train(total_timesteps=2048)
        logger.info("Retraining Complete. Model Updated.")

    async def run(self):
        await self.initialize()
        logger.info("Starting Real-Time Paper Trading Loop...")
        
        step_counter = 0
        
        while True:
            try:
                # 1. Fetch Latest Data
                df = await self.data_service.fetch_ohlcv(limit=100)
                if df.empty:
                    await asyncio.sleep(1)
                    continue
                
                # Get Real-Time Data
                lob = await self.data_service.fetch_order_book()
                trades = self.data_service.get_realtime_trades()
                
                # 2. Calculate Features
                df = self.fe.add_technical_indicators(df)
                cvd = self.fe.calculate_cvd(trades)
                lob_features = self.fe.calculate_lob_features(lob)
                
                # Create a copy of the last row to modify
                current_row = df.iloc[-1].copy()
                current_row['cvd'] = cvd
                current_row['ofi'] = 0.0 
                current_row['qi'] = lob_features.get('qi', 0.0)
                current_row['lob_imbalance_5'] = lob_features.get('lob_imbalance_5', 0.0)
                
                # 3. Regime & Volume Profile
                vp_data = self.fe.calculate_volume_profile(df)
                regime = self.regime_detector.detect_regime(df, vp_data)
                
                # 4. Prepare Observation
                obs_cols = self.env.obs_columns
                market_vals = []
                for col in obs_cols:
                    if col in current_row:
                        market_vals.append(current_row[col])
                    else:
                        if col in df.columns:
                             market_vals.append(df.iloc[-1][col])
                        else:
                             market_vals.append(0.0)
                
                market_obs = np.array(market_vals, dtype=np.float32)
                
                # Internal State
                current_price = current_row['close']
                
                # Check Pending Orders & TP/SL
                self.account.check_pending_orders(current_price, current_row['high'], current_row['low'])
                self.account.check_tp_sl(current_price)
                
                unrealized_pnl = 0.0
                unrealized_pnl_pct = 0.0
                if self.account.position != 0:
                    if self.account.position > 0: # Long
                        unrealized_pnl = (current_price - self.account.entry_price) * self.account.position
                        unrealized_pnl_pct = (current_price - self.account.entry_price) / self.account.entry_price * 100
                    else: # Short
                        unrealized_pnl = (self.account.entry_price - current_price) * abs(self.account.position)
                        unrealized_pnl_pct = (self.account.entry_price - current_price) / self.account.entry_price * 100
                
                internal_obs = np.array([self.account.position, unrealized_pnl_pct], dtype=np.float32)
                full_obs = np.concatenate([market_obs, internal_obs])
                
                # 5. Predict
                prediction = self.agent.predict(full_obs)
                action = prediction['action']
                value_est = float(prediction['value'])
                log_prob = float(prediction['log_prob'])
                
                # 6. Sophisticated Execution Logic
                act_val = action[0]
                confidence = np.tanh(abs(act_val))
                
                action_type = 0 # Hold
                thought_action = "HOLD"
                
                # Only act if confidence > 0.15 AND no position
                if confidence > 0.15 and self.account.position == 0 and not self.account.pending_orders:
                    proposed_dir = 1 if act_val > 0 else -1
                    
                    # B. Confluence Checks
                    # CVD Confirmation: Buyers for Long, Sellers for Short
                    cvd_confirms = (proposed_dir == 1 and cvd > 0) or (proposed_dir == -1 and cvd < 0)
                    
                    # LOB Confirmation: Imbalance supports direction
                    imb = lob_features.get('lob_imbalance_5', 0)
                    lob_confirms = (proposed_dir == 1 and imb > -0.1) or (proposed_dir == -1 and imb < 0.1)
                    
                    if cvd_confirms and lob_confirms:
                        action_type = proposed_dir
                        thought_action = "BUY" if proposed_dir == 1 else "SELL"
                        
                        # C. Entry Logic (LVN Targeting)
                        lvn = vp_data.get('lvn', current_price)
                        dist_to_lvn = abs(current_price - lvn) / current_price
                        
                        entry_price = current_price
                        order_type_str = "MARKET"
                        
                        if dist_to_lvn < 0.003: # Within 0.3% of LVN
                            entry_price = lvn
                            order_type_str = "LIMIT"
                            
                        # D. Risk Management (TP/SL)
                        hvn = vp_data.get('hvn', current_price)
                        
                        # Dynamic Leverage
                        norm_conf = (confidence - 0.15) / (1.0 - 0.15)
                        leverage = 20.0 + (norm_conf * (125.0 - 20.0))
                        leverage = min(leverage, 125.0)
                        
                        # "Lunch Money" Protocol: Strict Risk Sizing
                        # 1. Max Risk = 1% of Balance
                        risk_amount = self.account.balance * 0.01 
                        
                        # 2. Stop Loss Distance (Tight Scalp)
                        # Default 0.5%, but can be wider if volatility allows
                        sl_dist = 0.005 
                        
                        # 3. Calculate Position Size based on Risk
                        # Risk = Position_Value * SL_Dist
                        # Position_Value = Risk / SL_Dist
                        # Quantity = Position_Value / Entry_Price
                        position_value = risk_amount / sl_dist
                        quantity = position_value / entry_price
                        
                        # 4. Margin Check
                        required_margin = position_value / leverage
                        if required_margin > self.account.balance:
                            # Cap size to max available margin if risk calculation exceeds it
                            quantity = (self.account.balance * 0.95 * leverage) / entry_price
                        
                        # TP/SL Calculation
                        # TP at HVN (if reasonable), else 1.5% (3:1 RR)
                        tp_dist = abs(hvn - entry_price) / entry_price
                        if tp_dist < 0.015: tp_dist = 0.015 # Min 1.5% target
                        
                        if action_type == 1:
                            tp = entry_price * (1 + tp_dist)
                            sl = entry_price * (1 - sl_dist)
                        else:
                            tp = entry_price * (1 - tp_dist)
                            sl = entry_price * (1 + sl_dist)
                            
                        # Execute
                        self.account.place_order(action_type, quantity, entry_price, leverage, order_type_str, tp, sl)
                        
                    else:
                        thought_action = "WAIT (No Confluence)"
                
                elif self.account.position != 0:
                    thought_action = "MANAGE POSITION"
                elif self.account.pending_orders:
                    thought_action = "PENDING ORDER"

                # 7. Log & Update
                if self.account.position != 0:
                    pos_str = f"{self.account.position:.4f} BTC"
                
                logger.info(f"MKT: {current_price:.2f} | Act: {thought_action} | Conf: {confidence:.2f} | CVD: {cvd:.2f}")

                # 8. Send Update
                await self.send_update({
                    "price": current_price,
                    "regime": regime.value,
                    "action": thought_action,
                    "confidence": float(confidence),
                    "balance": self.account.balance,
                    "position": self.account.position,
                    "entry_price": self.account.entry_price,
                    "pnl": unrealized_pnl,
                    "pnl_pct": unrealized_pnl_pct,
                    "lob_spread": lob_features.get('spread', 0),
                    "lob_imbalance": lob_features.get('lob_imbalance_5', 0),
                    "ai_value": value_est,
                    "ai_log_prob": log_prob,
                    "ai_thought": self.generate_thought(act_val, value_est, regime.value, {
                        "rsi": current_row.get('rsi', 50),
                        "lob_imbalance": lob_features.get('lob_imbalance_5', 0),
                        "cvd": cvd,
                        "lvn": vp_data.get('lvn', 0),
                        "hvn": vp_data.get('hvn', 0),
                        "close": current_price,
                        "action_type": thought_action
                    })
                })

                # 9. Periodic Retraining
                step_counter += 1
                if step_counter % 60 == 0: 
                    await self.retrain()
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in loop: {e}")
                await asyncio.sleep(5)

    async def send_update(self, data):
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "type": "market_update",
                    "data": data
                }
                async with session.post("http://localhost:8000/api/update", json=payload) as resp:
                    if resp.status != 200:
                        logger.warning(f"Failed to send update: {resp.status}")
        except Exception as e:
            # Don't crash if backend is down
            pass

            # Don't crash if backend is down
            pass

    def generate_thought(self, action_val, value, regime, features):
        thought = []
        
        # 1. Decision & Action
        action_type = features.get('action_type', 'HOLD')
        thought.append(f"Action: {action_type} (Signal {action_val:.2f}).")
        
        if action_type == "WAIT (No Confluence)":
            thought.append("Signal is strong, but waiting for CVD/LOB confirmation.")
        elif action_type == "PENDING ORDER":
            thought.append("Waiting for Limit Order execution at LVN.")
            
        # 2. Confluence Analysis
        cvd = features.get('cvd', 0)
        if cvd > 0: thought.append(f"CVD is Positive ({cvd:.2f}), supporting Bulls.")
        else: thought.append(f"CVD is Negative ({cvd:.2f}), supporting Bears.")
        
        # 3. Structural Analysis (LVN/HVN)
        close = features.get('close', 0)
        lvn = features.get('lvn', 0)
        hvn = features.get('hvn', 0)
        
        if lvn > 0:
            dist_lvn = abs(close - lvn) / close * 100
            thought.append(f"Nearest LVN at {lvn:.2f} ({dist_lvn:.2f}% away).")
            
        if hvn > 0:
             thought.append(f"Targeting HVN at {hvn:.2f} for liquidity.")

        # 4. Value & Regime
        thought.append(f"Regime: {regime}.")
        if value > 0.5:
            thought.append(f"High expected reward ({value:.2f}).")
            
        return " ".join(thought)

    async def cleanup(self):
        await self.data_service.close()

if __name__ == "__main__":
    trader = PaperTrader()
    try:
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        logger.info("Stopping Paper Trader...")
    finally:
        asyncio.run(trader.cleanup())
