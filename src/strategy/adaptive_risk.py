class AdaptiveRiskManager:
    def __init__(self, base_risk_per_trade: float = 0.01):
        self.base_risk = base_risk_per_trade
        self.risk_multiplier = 1.0
        self.leverage_multiplier = 1.0
        self.win_rate_history = [] # List of booleans (True=Win, False=Loss)

    def update_performance(self, is_win: bool):
        """
        Update performance history and adjust risk parameters.
        """
        self.win_rate_history.append(is_win)
        if len(self.win_rate_history) > 50:
            self.win_rate_history.pop(0)
        
        self._adjust_risk()

    def _adjust_risk(self):
        """
        Adjust risk multiplier based on recent win rate.
        """
        if not self.win_rate_history:
            return

        recent_wins = sum(self.win_rate_history[-20:])
        recent_total = len(self.win_rate_history[-20:])
        win_rate = recent_wins / recent_total if recent_total > 0 else 0

        if win_rate > 0.65:
            # Aggressive Mode
            self.risk_multiplier = 1.15
            self.leverage_multiplier = 1.05
        elif win_rate < 0.35:
            # Defensive Mode
            self.risk_multiplier = 0.5
            self.leverage_multiplier = 0.6
        else:
            # Normal Mode
            self.risk_multiplier = 1.0
            self.leverage_multiplier = 1.0

    def get_risk_parameters(self):
        return {
            "risk_per_trade": self.base_risk * self.risk_multiplier,
            "leverage": self.leverage_multiplier
        }
