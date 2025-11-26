import pandas as pd
import numpy as np
from src.env.trading_env import TradingEnv

class BacktestEngine:
    def __init__(self, env: TradingEnv, agent, maker_fee=0.0002, taker_fee=0.0004, slippage_pct=0.0001):
        self.env = env
        self.agent = agent
        self.results = []
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_pct = slippage_pct

    def run(self):
        """
        Run the backtest with friction modeling.
        """
        obs, _ = self.env.reset()
        done = False
        
        # Track detailed trade logs
        trades = []
        
        while not done:
            action = self.agent.predict(obs)
            
            # Capture state before step
            prev_balance = self.env.balance
            prev_position = self.env.position
            current_price = self.env.df.iloc[self.env.current_step]['close']
            
            obs, reward, done, truncated, info = self.env.step(action)
            
            # Calculate Frictions if a trade occurred
            # Logic: If position changed, we traded
            current_position = self.env.position
            
            if current_position != prev_position:
                trade_size = abs(current_position - prev_position)
                trade_value = trade_size * current_price
                
                # Apply Fees (Assume Taker for simplicity in aggressive strategies)
                fee_cost = trade_value * self.taker_fee
                
                # Apply Slippage
                slippage_cost = trade_value * self.slippage_pct
                
                # Deduct from balance in Env (We need to hack this in or support it in Env)
                # Ideally Env handles this, but for "Rigorous Backtest" wrapper we can adjust the result metrics
                # For now, let's adjust the 'balance' in our tracked results
                
                # Note: modifying self.env.balance directly is a bit hacky but works for this simulation wrapper
                self.env.balance -= (fee_cost + slippage_cost)
                
                trades.append({
                    'step': self.env.current_step,
                    'price': current_price,
                    'size': trade_size,
                    'fee': fee_cost,
                    'slippage': slippage_cost,
                    'action': 'BUY' if current_position > prev_position else 'SELL'
                })
            
            # Update info with friction-adjusted balance
            info['balance'] = self.env.balance
            self.results.append(info)
            
        return pd.DataFrame(self.results)

    def calculate_metrics(self, df_results: pd.DataFrame):
        """
        Calculate Sharpe, Max Drawdown, etc.
        """
        if df_results.empty:
            return {}
            
        # Assuming 'balance' is in results
        balance_curve = df_results['balance']
        returns = balance_curve.pct_change().dropna()
        
        if returns.std() == 0:
            sharpe = 0
        else:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 60) # Annualized
        
        # Max Drawdown
        peak = balance_curve.cummax()
        drawdown = (balance_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            "Final Balance": balance_curve.iloc[-1],
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown
        }
