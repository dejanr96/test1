import pandas as pd
import numpy as np
from enum import Enum

class MarketRegimeType(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNCERTAIN = "UNCERTAIN"

class MarketRegimeDetector:
    def __init__(self):
        pass

    def detect_regime(self, df: pd.DataFrame, vp_data: dict = None) -> MarketRegimeType:
        """
        Detect the current market regime based on technical indicators and Volume Profile.
        """
        if df.empty or len(df) < 50:
            return MarketRegimeType.UNCERTAIN

        current_price = df['close'].iloc[-1]

        # 1. Volatility check (ATR)
        current_atr = df['atr'].iloc[-1] if 'atr' in df else 0
        avg_atr = df['atr'].mean() if 'atr' in df else 0
        is_volatile = current_atr > (avg_atr * 2.0) # Relaxed from 1.5 for Phase 2

        if is_volatile:
            return MarketRegimeType.VOLATILE

        # 2. Volume Profile Structure Check (Primary)
        if vp_data:
            vah = vp_data.get('vah', 0)
            val = vp_data.get('val', 0)
            hvn = vp_data.get('hvn', 0)
            
            # Inside Value Area -> RANGING (Balanced)
            if val <= current_price <= vah:
                # If very close to HVN (Magnet), it's likely ranging/consolidating
                if abs(current_price - hvn) / hvn < 0.005:
                     return MarketRegimeType.RANGING
                return MarketRegimeType.RANGING
            
            # Outside Value Area -> TRENDING (Imbalanced)
            else:
                return MarketRegimeType.TRENDING

        # 3. Fallback: Trend check (SMA slope)
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        
        if abs(sma_20 - sma_50) / sma_50 > 0.02: # 2% divergence
            return MarketRegimeType.TRENDING
        
        return MarketRegimeType.RANGING
