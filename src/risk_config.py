"""
Risk Configuration System
=========================

10-level risk tolerance system with configurable:
- Confidence thresholds
- Position sizing
- Stop loss / profit targets
- DTE and delta ranges
- Overnight/weekend holding rules

Based on: Risk-Tolerance-Trade-Frequency-EOD-Strategies.md
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum


class TradeFrequency(Enum):
    CONSERVATIVE = "conservative"  # 85%+ confidence, 3-5 signals/week
    MODERATE = "moderate"          # 70%+ confidence, 8-15 signals/week
    AGGRESSIVE = "aggressive"      # 60%+ confidence, 15-30 signals/week


class DirectionalBias(Enum):
    BULL_ONLY = "bull_only"    # Only BUY CALL signals
    BEAR_ONLY = "bear_only"    # Only BUY PUT signals
    BOTH_SIDES = "both"        # Accept any direction
    BIDIRECTIONAL = "bidirectional"  # Trade both sides simultaneously


class EODStrategy(Enum):
    CLOSE_WINNERS = "close_winners"  # Close profitable positions by 3 PM
    FRIDAY_CLOSE = "friday_close"    # Close all positions Friday 3 PM
    INTRADAY_ONLY = "intraday_only"  # Never hold overnight
    USER_DISCRETION = "hold"          # No automated closes


@dataclass
class RiskConfig:
    """Configuration for a specific risk level."""
    
    name: str
    level: int
    
    # Signal filtering
    confidence_min: float        # Minimum AI score (0-100)
    min_indicators_agree: int    # Minimum indicators that must agree (1-4)
    
    # Position sizing
    position_size_pct: float     # % of portfolio per trade
    max_portfolio_risk: float    # Max % of portfolio at risk
    max_positions: int           # Maximum simultaneous positions
    
    # Stop loss / Profit targets
    stop_loss_pct: float         # Max loss before exit
    profit_target_pct: float     # Take profit level
    trail_stop_pct: Optional[float]  # Trailing stop %
    
    # Options parameters
    delta_min: float             # Minimum delta (0.20-0.95)
    delta_max: float             # Maximum delta
    dte_min: int                 # Minimum days to expiry
    dte_max: int                 # Maximum days to expiry
    
    # Holding rules
    hold_overnight: bool         # Allow overnight positions
    hold_weekends: bool          # Allow weekend positions
    
    # Risk filters
    avoid_earnings: bool         # Avoid earnings week
    avoid_high_iv: bool          # Avoid IV > 50 rank
    
    # Expected outcomes
    target_win_rate: float       # Expected win rate
    expected_annual_return: Tuple[float, float]  # (min%, max%)
    max_drawdown: float          # Maximum expected drawdown


# ============= 10-Level Risk Configurations =============

RISK_LEVELS: Dict[int, RiskConfig] = {
    
    1: RiskConfig(
        name="Ultra-Conservative",
        level=1,
        confidence_min=90,
        min_indicators_agree=4,
        position_size_pct=0.01,      # 1% per trade
        max_portfolio_risk=0.02,     # 2% total
        max_positions=2,
        stop_loss_pct=0.02,          # 2% stop
        profit_target_pct=0.20,      # 20% target
        trail_stop_pct=0.02,
        delta_min=0.60,
        delta_max=0.95,
        dte_min=30,
        dte_max=45,
        hold_overnight=False,
        hold_weekends=False,
        avoid_earnings=True,
        avoid_high_iv=True,
        target_win_rate=0.75,
        expected_annual_return=(0.08, 0.12),
        max_drawdown=0.08,
    ),
    
    2: RiskConfig(
        name="Conservative",
        level=2,
        confidence_min=80,
        min_indicators_agree=4,
        position_size_pct=0.025,
        max_portfolio_risk=0.05,
        max_positions=3,
        stop_loss_pct=0.03,
        profit_target_pct=0.25,
        trail_stop_pct=0.03,
        delta_min=0.50,
        delta_max=0.95,
        dte_min=28,
        dte_max=45,
        hold_overnight=True,
        hold_weekends=False,
        avoid_earnings=True,
        avoid_high_iv=True,
        target_win_rate=0.70,
        expected_annual_return=(0.12, 0.18),
        max_drawdown=0.12,
    ),
    
    3: RiskConfig(
        name="Moderate-Conservative",
        level=3,
        confidence_min=75,
        min_indicators_agree=3,
        position_size_pct=0.03,
        max_portfolio_risk=0.08,
        max_positions=4,
        stop_loss_pct=0.04,
        profit_target_pct=0.30,
        trail_stop_pct=0.04,
        delta_min=0.45,
        delta_max=0.90,
        dte_min=21,
        dte_max=45,
        hold_overnight=True,
        hold_weekends=False,
        avoid_earnings=True,
        avoid_high_iv=True,
        target_win_rate=0.65,
        expected_annual_return=(0.15, 0.22),
        max_drawdown=0.15,
    ),
    
    4: RiskConfig(
        name="Balanced-Conservative",
        level=4,
        confidence_min=70,
        min_indicators_agree=3,
        position_size_pct=0.04,
        max_portfolio_risk=0.10,
        max_positions=5,
        stop_loss_pct=0.05,
        profit_target_pct=0.35,
        trail_stop_pct=0.05,
        delta_min=0.45,
        delta_max=0.85,
        dte_min=14,
        dte_max=45,
        hold_overnight=True,
        hold_weekends=False,
        avoid_earnings=True,
        avoid_high_iv=False,
        target_win_rate=0.60,
        expected_annual_return=(0.18, 0.28),
        max_drawdown=0.18,
    ),
    
    5: RiskConfig(
        name="Moderate",
        level=5,
        confidence_min=65,
        min_indicators_agree=3,
        position_size_pct=0.05,
        max_portfolio_risk=0.12,
        max_positions=6,
        stop_loss_pct=0.06,
        profit_target_pct=0.40,
        trail_stop_pct=0.06,
        delta_min=0.40,
        delta_max=0.80,
        dte_min=14,
        dte_max=45,
        hold_overnight=True,
        hold_weekends=True,
        avoid_earnings=True,
        avoid_high_iv=False,
        target_win_rate=0.55,
        expected_annual_return=(0.20, 0.35),
        max_drawdown=0.20,
    ),
    
    6: RiskConfig(
        name="Moderate-Aggressive",
        level=6,
        confidence_min=65,
        min_indicators_agree=3,
        position_size_pct=0.06,
        max_portfolio_risk=0.15,
        max_positions=8,
        stop_loss_pct=0.07,
        profit_target_pct=0.45,
        trail_stop_pct=0.06,
        delta_min=0.35,
        delta_max=0.80,
        dte_min=7,
        dte_max=45,
        hold_overnight=True,
        hold_weekends=True,
        avoid_earnings=False,
        avoid_high_iv=False,
        target_win_rate=0.55,
        expected_annual_return=(0.25, 0.40),
        max_drawdown=0.22,
    ),
    
    7: RiskConfig(
        name="Aggressive",
        level=7,
        confidence_min=60,
        min_indicators_agree=2,
        position_size_pct=0.08,
        max_portfolio_risk=0.18,
        max_positions=10,
        stop_loss_pct=0.08,
        profit_target_pct=0.50,
        trail_stop_pct=0.07,
        delta_min=0.30,
        delta_max=0.85,
        dte_min=7,
        dte_max=45,
        hold_overnight=True,
        hold_weekends=True,
        avoid_earnings=False,
        avoid_high_iv=False,
        target_win_rate=0.52,
        expected_annual_return=(0.30, 0.50),
        max_drawdown=0.25,
    ),
    
    8: RiskConfig(
        name="Very Aggressive",
        level=8,
        confidence_min=60,
        min_indicators_agree=2,
        position_size_pct=0.10,
        max_portfolio_risk=0.20,
        max_positions=12,
        stop_loss_pct=0.09,
        profit_target_pct=0.55,
        trail_stop_pct=0.08,
        delta_min=0.25,
        delta_max=0.90,
        dte_min=1,
        dte_max=60,
        hold_overnight=True,
        hold_weekends=True,
        avoid_earnings=False,
        avoid_high_iv=False,
        target_win_rate=0.50,
        expected_annual_return=(0.35, 0.60),
        max_drawdown=0.30,
    ),
    
    9: RiskConfig(
        name="High Risk",
        level=9,
        confidence_min=55,
        min_indicators_agree=2,
        position_size_pct=0.12,
        max_portfolio_risk=0.22,
        max_positions=15,
        stop_loss_pct=0.10,
        profit_target_pct=0.60,
        trail_stop_pct=0.08,
        delta_min=0.20,
        delta_max=0.95,
        dte_min=1,
        dte_max=60,
        hold_overnight=True,
        hold_weekends=True,
        avoid_earnings=False,
        avoid_high_iv=False,
        target_win_rate=0.48,
        expected_annual_return=(0.40, 0.80),
        max_drawdown=0.35,
    ),
    
    10: RiskConfig(
        name="Maximum Aggression",
        level=10,
        confidence_min=50,
        min_indicators_agree=2,
        position_size_pct=0.15,
        max_portfolio_risk=0.25,
        max_positions=20,
        stop_loss_pct=0.10,
        profit_target_pct=0.75,
        trail_stop_pct=None,  # No trailing, hold for full move
        delta_min=0.20,
        delta_max=1.0,
        dte_min=1,
        dte_max=60,
        hold_overnight=True,
        hold_weekends=True,
        avoid_earnings=False,
        avoid_high_iv=False,
        target_win_rate=0.45,
        expected_annual_return=(0.50, 1.00),
        max_drawdown=0.40,
    ),
}


# ============= Trade Frequency Configurations =============

TRADE_FREQUENCY_CONFIG = {
    TradeFrequency.CONSERVATIVE: {
        "confidence_threshold": 85,
        "min_indicators_agree": 4,
        "require_divergence": True,
        "require_volume_confirmation": True,
        "signals_per_week_target": (3, 5),
    },
    TradeFrequency.MODERATE: {
        "confidence_threshold": 70,
        "min_indicators_agree": 3,
        "require_divergence": False,
        "require_volume_confirmation": False,
        "signals_per_week_target": (8, 15),
    },
    TradeFrequency.AGGRESSIVE: {
        "confidence_threshold": 60,
        "min_indicators_agree": 2,
        "require_divergence": False,
        "require_volume_confirmation": False,
        "signals_per_week_target": (15, 30),
    },
}


# ============= Helper Functions =============

def get_risk_config(level: int) -> RiskConfig:
    """Get risk configuration for a specific level (1-10)."""
    level = max(1, min(10, level))
    return RISK_LEVELS[level]


def get_confidence_threshold(risk_level: int, frequency: TradeFrequency) -> float:
    """Get the effective confidence threshold combining risk and frequency."""
    risk_config = get_risk_config(risk_level)
    freq_config = TRADE_FREQUENCY_CONFIG[frequency]
    
    # Use the more restrictive threshold
    return max(risk_config.confidence_min, freq_config["confidence_threshold"])


def calculate_position_size(
    risk_level: int,
    account_value: float,
    signal_score: float,
    current_portfolio_risk: float = 0.0
) -> float:
    """
    Calculate position size based on risk level and signal strength.
    
    Returns: Dollar amount to allocate to this position
    """
    config = get_risk_config(risk_level)
    
    # Base position size
    base_pct = config.position_size_pct
    
    # Scale by signal strength (higher score = larger position)
    score_multiplier = 0.8 + (signal_score / 100) * 0.4  # 0.8x to 1.2x
    
    # Reduce if approaching max portfolio risk
    available_risk = config.max_portfolio_risk - current_portfolio_risk
    if available_risk <= 0:
        return 0.0  # At max risk
    
    risk_multiplier = min(1.0, available_risk / base_pct)
    
    # Final position size
    final_pct = base_pct * score_multiplier * risk_multiplier
    position_size = account_value * final_pct
    
    return position_size


def filter_signal_by_direction(
    signal_type: str,
    directional_bias: DirectionalBias
) -> bool:
    """
    Check if a signal matches the user's directional bias.
    
    Returns: True if signal should be accepted, False to filter out
    """
    if directional_bias == DirectionalBias.BOTH_SIDES:
        return signal_type in ["BUY_CALL", "BUY_PUT"]
    
    if directional_bias == DirectionalBias.BULL_ONLY:
        return signal_type == "BUY_CALL"
    
    if directional_bias == DirectionalBias.BEAR_ONLY:
        return signal_type == "BUY_PUT"
    
    if directional_bias == DirectionalBias.BIDIRECTIONAL:
        return True  # Accept everything
    
    return False


# ============= User Preferences Dataclass =============

@dataclass
class UserPreferences:
    """Complete user trading preferences."""
    
    risk_tolerance: int = 5
    trade_frequency: TradeFrequency = TradeFrequency.MODERATE
    directional_bias: DirectionalBias = DirectionalBias.BOTH_SIDES
    eod_strategy: EODStrategy = EODStrategy.FRIDAY_CLOSE
    avoid_earnings: bool = True
    
    def get_effective_config(self) -> RiskConfig:
        """Get the effective risk config with frequency adjustments."""
        config = get_risk_config(self.risk_tolerance)
        freq_config = TRADE_FREQUENCY_CONFIG[self.trade_frequency]
        
        # Override confidence threshold if frequency is more restrictive
        effective_confidence = max(config.confidence_min, freq_config["confidence_threshold"])
        
        # Return modified config
        modified = RiskConfig(
            **{**config.__dict__}
        )
        modified.confidence_min = effective_confidence
        modified.min_indicators_agree = freq_config["min_indicators_agree"]
        
        return modified
    
    def validate(self) -> bool:
        """Validate preferences are consistent."""
        # Conservative risk shouldn't use aggressive frequency
        if self.risk_tolerance <= 2 and self.trade_frequency == TradeFrequency.AGGRESSIVE:
            return False
        
        # Conservative risk should close positions daily
        if self.risk_tolerance <= 3 and self.eod_strategy == EODStrategy.USER_DISCRETION:
            return False
        
        return True


# ============= CLI Demo =============

if __name__ == "__main__":
    print("Risk Configuration System")
    print("=" * 50)
    
    for level in [1, 5, 10]:
        config = get_risk_config(level)
        print(f"\nLevel {level}: {config.name}")
        print(f"  Confidence: {config.confidence_min}%+")
        print(f"  Position size: {config.position_size_pct*100}%")
        print(f"  Stop loss: {config.stop_loss_pct*100}%")
        print(f"  Profit target: {config.profit_target_pct*100}%")
        print(f"  Max positions: {config.max_positions}")
        print(f"  Delta range: {config.delta_min}-{config.delta_max}")
        print(f"  Hold overnight: {config.hold_overnight}")
        print(f"  Target win rate: {config.target_win_rate*100}%")
