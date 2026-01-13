# DEEP RESEARCH: USER RISK TOLERANCE, TRADE FREQUENCY, AND DIRECTIONAL BIAS
## Comprehensive Framework for Customizable Options Trading Strategies

**Research Date:** January 12, 2026  
**Focus:** Personalized Trading Preferences for Your IB Options Platform  
**Target:** Solving user customization and EOD closing strategies  

---

# TABLE OF CONTENTS

1. Executive Summary: The Risk Tolerance Framework
2. Risk Tolerance Degrees & Signal Filtering
3. Trade Frequency Models (Aggressive vs. Conservative)
4. Directional Bias System (Bull/Bear/Neutral/BiDirectional)
5. End-of-Day Closing Strategies
6. IV Crush Management & Pre-Market Decisions
7. Implementation Architecture
8. User Preference System Design
9. Risk Management & Position Sizing

---

# SECTION 1: EXECUTIVE SUMMARY

## The Three User Preference Dimensions

Your platform needs to let users customize THREE dimensions:

### **Dimension 1: Risk Tolerance Degree**
```
Controls: How aggressive vs. defensive each user wants to be
Range: 1 (Very Conservative) → 10 (Very Aggressive)
Examples:
- Level 1-2: Retirees, capital preservation focus
- Level 3-4: Conservative traders, small account growth
- Level 5-6: Moderate traders, balanced approach
- Level 7-8: Aggressive traders, high returns focus
- Level 9-10: Maximum risk/reward seekers
```

### **Dimension 2: Trade Frequency Preference**
```
Controls: How often user wants to see trading signals
Range: "Wait for Big Opportunities" ← → "Frequent High-Probability Trades"
Meaning:
- Waiting for big opportunities = Signal only when consensus > 85%
- Frequent trades = Signal when consensus > 60%
- Middle ground = Signal when consensus > 70%
```

### **Dimension 3: Directional Bias**
```
Controls: Which types of market moves user wants to trade
Options:
- Bull only: BUY CALLS only (betting on up moves)
- Bear only: SELL PUTS or BUY PUTS only (betting on down moves)
- Both sides: Any signal, any direction
- Advanced: User can toggle between sides
```

---

## How These Dimensions Interact

```
User Selects:
- Risk Tolerance: 7/10 (aggressive)
- Trade Frequency: "High Probability" (≥70% consensus)
- Directional Bias: "Both Sides"

Platform Behavior:
├─ Generates signals when confidence ≥ 70%
├─ Adjusts position size for risk level 7
├─ Accepts BUY CALL signals (bullish)
├─ Accepts BUY PUT signals (bearish)
├─ May close positions by EOD (based on risk level)
└─ May avoid earnings trades (depends on settings)
```

---

# SECTION 2: RISK TOLERANCE DEGREES & SIGNAL FILTERING

## The 10-Level Risk Tolerance Scale

### **Level 1: Ultra-Conservative (Retirees, Capital Preservation)**

```
Goals:
├─ Preserve capital above all else
├─ Generate steady, small income
├─ Avoid sleep-disrupting trades
└─ Maximum comfortable loss: 1-2% per trade

Signal Filtering:
├─ Confidence threshold: >90% (only strongest signals)
├─ DTE: 30-45 days (avoid rapid theta decay)
├─ Delta: 0.60-0.70 (high probability wins)
├─ Position size: 1-2% of account per trade
├─ Max account at risk: <2% total
├─ Avoid: Earnings, high IV, speculative

Position Management:
├─ Always use stops (±2% max loss per trade)
├─ Take profits at 20-25% gain
├─ Never hold overnight
├─ Exit by 3 PM (before close volatility)
└─ Close weekends to avoid gap risk

Win Rate Target: 70-80%
Average Return per Trade: +15-20%
Expected Annual Return: 8-12%
Max Drawdown: 5-8%
```

**Config:**
```json
{
  "risk_level": 1,
  "confidence_threshold": 90,
  "max_loss_per_trade": 0.02,
  "max_portfolio_risk": 0.02,
  "position_size_strategy": "kelly_fraction_0.25",
  "hold_overnight": false,
  "hold_weekends": false,
  "profit_target": 0.20,
  "stop_loss": 0.02,
  "avoid_earnings": true,
  "avoid_high_iv": true
}
```

---

### **Level 2: Conservative (Safe Growth)**

```
Goals:
├─ Grow capital with controlled risk
├─ Win more often than lose
├─ Sleep at night
└─ Maximum comfortable loss: 2-3% per trade

Signal Filtering:
├─ Confidence threshold: 80% (strong signals only)
├─ DTE: 28-45 days
├─ Delta: 0.50-0.70 (balanced)
├─ Position size: 2-3% of account per trade
├─ Max account at risk: <5% total
├─ Avoid: Extreme IV, very far OTM

Position Management:
├─ Use stops (±3% max loss)
├─ Take profits at 25-30% gain
├─ Okay to hold overnight
├─ Close Fridays to avoid weekend gaps
├─ Exit by Friday 3 PM latest
└─ Rarely add to positions

Win Rate Target: 65-75%
Average Return per Trade: +20-25%
Expected Annual Return: 12-18%
Max Drawdown: 8-12%
```

---

### **Level 3-4: Moderate-Conservative (Balanced)**

```
Goals:
├─ Steady capital growth
├─ Acceptable 35-40% loss frequency
├─ Work-life balance
└─ Maximum comfortable loss: 3-5% per trade

Signal Filtering:
├─ Confidence threshold: 70% (good signals)
├─ DTE: 21-45 days
├─ Delta: 0.45-0.70 (flexible)
├─ Position size: 3-5% of account per trade
├─ Max account at risk: <10% total
├─ Allow: Moderate IV environments

Position Management:
├─ Use stops (±4-5% max loss)
├─ Take profits at 30-40% gain
├─ Okay to hold overnight (but close EOW)
├─ Can hold through events if exits planned
├─ Occasional add to winners allowed
└─ Trail stops to lock in gains
```

---

### **Level 5-6: Moderate (Balanced Growth)**

```
Goals:
├─ Growth-focused but disciplined
├─ 50/50 win rate acceptable
├─ Compound wealth over time
└─ Maximum comfortable loss: 5-7% per trade

Signal Filtering:
├─ Confidence threshold: 65% (acceptable risk)
├─ DTE: 14-45 days (flexible)
├─ Delta: 0.40-0.80 (wide range)
├─ Position size: 4-6% of account per trade
├─ Max account at risk: <15% total
├─ Allow: Normal IV, some earnings trades

Position Management:
├─ Use stops (±5-7% max loss)
├─ Take profits at 35-50% gain or theta advantage
├─ Hold overnight freely
├─ Can hold through week
├─ Add to winners strategically
├─ Trail stops aggressively
└─ May hold multiple positions simultaneously
```

---

### **Level 7-8: Aggressive (Growth-Focused)**

```
Goals:
├─ Maximum capital growth
├─ Accept higher loss frequency (40-50%)
├─ Believe in edge strongly
└─ Maximum comfortable loss: 7-10% per trade

Signal Filtering:
├─ Confidence threshold: 60% (wider acceptance)
├─ DTE: 7-45 days (very flexible)
├─ Delta: 0.30-0.80 (very wide)
├─ Position size: 6-10% of account per trade
├─ Max account at risk: <20% total
├─ Allow: High IV, some speculative plays

Position Management:
├─ Use stops (±7-10% max loss)
├─ Take profits at 40-60% gain
├─ Actively add to winners
├─ Hold multiple positions (5-10)
├─ Use trailing stops actively
├─ May average down on strong signals
└─ Okay to hold through earnings if positioned right

Win Rate Target: 50-60%
Average Return per Trade: +30-50%
Expected Annual Return: 25-40%
Max Drawdown: 15-25%
```

---

### **Level 9-10: Maximum Aggression (Wealth Multiplication)**

```
Goals:
├─ Explosive growth
├─ Accept 40-50% loss rate
├─ Leverage small edge into big returns
└─ Comfortable with large swings: -20% to +50%

Signal Filtering:
├─ Confidence threshold: 50% (any edge)
├─ DTE: Any (1-60 days, depending on signal strength)
├─ Delta: 0.20-0.90 (full spectrum)
├─ Position size: 10-20% of account per trade
├─ Max account at risk: <25% total
├─ Allow: Everything (earnings, extreme IV, etc.)

Position Management:
├─ Aggressive stops or none (accept full loss)
├─ Take profits at 50%+ or hold for 100%+
├─ Heavily add to winners
├─ 10-20 simultaneous positions
├─ May pyramid into strong moves
├─ Minimal stop discipline
├─ May hold through earnings

Win Rate Target: 40-50%
Average Return per Trade: +50-100%
Expected Annual Return: 40-100%+
Max Drawdown: 25-40%
```

---

## Risk Tolerance Configuration Matrix

```python
RISK_TOLERANCE_CONFIG = {
    1: {
        "name": "Ultra-Conservative",
        "confidence_min": 0.90,
        "position_size_pct": 0.01,  # 1% per trade
        "max_portfolio_risk": 0.02,  # 2% total
        "stop_loss_pct": 0.02,
        "profit_target_pct": 0.20,
        "hold_overnight": False,
        "hold_weekends": False,
        "max_positions": 2,
        "avoid_earnings": True,
        "avoid_high_iv": True,
        "allowed_delta_min": 0.60,
        "allowed_delta_max": 0.95,
        "dte_min": 30,
        "dte_max": 45,
    },
    2: {
        "name": "Conservative",
        "confidence_min": 0.80,
        "position_size_pct": 0.025,
        "max_portfolio_risk": 0.05,
        "stop_loss_pct": 0.03,
        "profit_target_pct": 0.25,
        "hold_overnight": True,
        "hold_weekends": False,
        "max_positions": 3,
        "avoid_earnings": True,
        "avoid_high_iv": True,
        "allowed_delta_min": 0.50,
        "allowed_delta_max": 0.95,
        "dte_min": 28,
        "dte_max": 45,
    },
    # ... (levels 3-10 continue with similar structure)
    10: {
        "name": "Maximum Aggression",
        "confidence_min": 0.50,
        "position_size_pct": 0.15,  # 15% per trade
        "max_portfolio_risk": 0.25,  # 25% total
        "stop_loss_pct": 0.10,  # Or none
        "profit_target_pct": 0.50,
        "hold_overnight": True,
        "hold_weekends": True,
        "max_positions": 20,
        "avoid_earnings": False,
        "avoid_high_iv": False,
        "allowed_delta_min": 0.20,
        "allowed_delta_max": 1.0,
        "dte_min": 1,
        "dte_max": 60,
    }
}
```

---

# SECTION 3: TRADE FREQUENCY MODELS

## The Frequency Spectrum

### **Model 1: "Wait for Big Opportunities" (Conservative)**

```
When to Signal:
├─ Only when: Confidence ≥ 85% (highest quality)
├─ AND: Multi-timeframe agreement (all TFs bullish/bearish)
├─ AND: IV Skew strongly supports move
├─ AND: All 4 indicators agree
└─ Result: 3-5 signals per WEEK (not per day)

Reasoning:
├─ Quality over quantity
├─ Higher win rate (70%+)
├─ Bigger average move when signal triggers
├─ User has time to analyze before entry
└─ Less decision fatigue

Example Signals:
├─ SPY breaks above resistance with volume + RSI bullish
├─ AAPL before earnings with all indicators aligned
├─ QQQ with 4+ hour bullish setup
└─ Maybe 1-2 trades per day MAX
```

**Config:**
```python
trade_frequency = "conservative"
confidence_threshold = 0.85
min_indicators_agree = 4  # ALL must agree
signals_per_week_target = 3-5
analyze_all_timeframes = True
require_divergence = True
require_volume_confirmation = True
```

---

### **Model 2: "High Probability Trades" (Moderate)**

```
When to Signal:
├─ When: Confidence ≥ 70% (good quality)
├─ AND: 3/4 indicators agree
├─ AND: Single timeframe is strong
├─ OR: Multi-timeframe shows momentum
└─ Result: 8-15 signals per WEEK (multiple per day)

Reasoning:
├─ Balanced approach
├─ More trading opportunities
├─ Still high win rate (60-65%)
├─ Normal variance in results
├─ Good for active traders

Example Signals:
├─ SPY shows overbought + divergence
├─ Any signal where 3 AI indicators agree
├─ IV skew is extreme in one direction
├─ Can trade 1-3 times per day
└─ Skip lower quality setups
```

**Config:**
```python
trade_frequency = "moderate"
confidence_threshold = 0.70
min_indicators_agree = 3  # At least 3 of 4
signals_per_week_target = 8-15
require_divergence = False
require_volume_confirmation = False
allow_same_stock_multiple_times = True
```

---

### **Model 3: "Frequent Trades" (Aggressive)**

```
When to Signal:
├─ When: Confidence ≥ 60% (acceptable quality)
├─ AND: 2/4 indicators agree (just majority)
├─ OR: Single indicator shows extreme (>75 RSI, IV skew >1.5)
├─ Result: 15-30 signals per WEEK (multiple per day)

Reasoning:
├─ Maximum trading opportunities
├─ Win rate may drop to 55-60%
├─ Higher frequency = better compounding
├─ Need tighter risk management
├─ Requires discipline on position sizing

Example Signals:
├─ SuperTrend turns bullish (signal alone)
├─ RSI overbought OR oversold (even alone)
├─ IV extreme in either direction
├─ Can trade 5+ times per day
└─ Accept more losing trades
```

**Config:**
```python
trade_frequency = "aggressive"
confidence_threshold = 0.60
min_indicators_agree = 2  # Simple majority
signals_per_week_target = 15-30
allow_single_indicator_signals = True
allow_same_stock_multiple_times = True
position_size_per_trade = 2-3%  # Smaller positions
max_concurrent_trades = 10-20
```

---

## Filtering by Confidence Score

```python
def generate_signal(indicators, confidence_score, trade_frequency_mode):
    """
    Filter signals based on user's frequency preference
    """
    
    if trade_frequency_mode == "conservative":
        # Only top 5% of opportunities
        if confidence_score < 0.85:
            return None  # Skip
        if num_indicators_agree < 4:
            return None  # Skip
        if not has_divergence:
            return None  # Skip
        return SIGNAL  # Send
    
    elif trade_frequency_mode == "moderate":
        # Good quality opportunities
        if confidence_score < 0.70:
            return None  # Skip
        if num_indicators_agree < 3:
            return None  # Skip
        return SIGNAL  # Send
    
    elif trade_frequency_mode == "aggressive":
        # Any reasonable opportunity
        if confidence_score < 0.60:
            return None  # Skip
        if num_indicators_agree < 2:
            return None  # Skip
        return SIGNAL  # Send
    
    else:
        return None
```

---

# SECTION 4: DIRECTIONAL BIAS SYSTEM

## The Four Directional Categories[362][365]

### **Category 1: Bullish (Bull) - Market Moving UP**

```
Definition: You expect underlying to move UP
Signals: BUY CALLS (long calls)
Subtypes:
├─ Mildly Bullish: 0.50-0.60 delta (neutral to slightly up)
├─ Moderately Bullish: 0.60-0.70 delta (confident up)
└─ Aggressively Bullish: 0.70-0.90 delta (very bullish)

When Selected:
├─ Uptrend confirmed (higher highs, higher lows)
├─ Momentum building (RSI rising)
├─ Support holding
├─ Volume increasing
└─ IV skew supports calls being cheap
```

**Example Trade:**
```
Signal: SPY breakout to new high
User selects: Bull only
Platform response:
├─ Only generates BUY CALL signals
├─ Ignores bearish reversal signals
├─ Ignores neutral sideways signals
├─ Only considers UP moves
└─ Suggests 0.55-0.65 delta calls in 35 DTE
```

---

### **Category 2: Bearish (Bear) - Market Moving DOWN**

```
Definition: You expect underlying to move DOWN
Signals: BUY PUTS (long puts)
Subtypes:
├─ Mildly Bearish: 0.40-0.50 delta
├─ Moderately Bearish: 0.30-0.40 delta
└─ Aggressively Bearish: 0.20-0.30 delta

When Selected:
├─ Downtrend confirmed (lower highs, lower lows)
├─ Momentum breaking down (RSI falling)
├─ Resistance breaking
├─ Volume increasing
└─ IV skew supports puts being cheap
```

**Example Trade:**
```
Signal: SPY breaks down from support
User selects: Bear only
Platform response:
├─ Only generates BUY PUT signals
├─ Ignores bullish reversal signals
├─ Ignores neutral sideways signals
├─ Only considers DOWN moves
└─ Suggests 0.55-0.65 delta puts in 35 DTE
```

---

### **Category 3: Neutral - Market Sideways (Range-Bound)**

```
Definition: You expect NO directional move
Signals: Iron Condors, Short Strangles, Short Straddles
Note: Advanced strategies - focus on SELLING options
This is: NOT recommended for your platform initially
Reason: Requires different risk management (undefined risk)

When Selected:
├─ Stock in established range
├─ Resistance and support are clear
├─ Low volatility environment
├─ Theta decay working in your favor
└─ Want to profit from time decay
```

---

### **Category 4: Bi-Directional (Both Ways)**

```
Definition: You expect LARGE move in EITHER direction
Signals: BUY CALLS when bullish OR BUY PUTS when bearish
Meaning: "I don't know direction, but I know it's moving big"

When Selected:
├─ Earnings week (large move expected)
├─ IV Rank extremely low (about to spike)
├─ Support/resistance both breaking
├─ Technical pattern suggests breakout imminent
└─ Long straddle or strangle territory

Strategy:
├─ Generate both BUY CALL and BUY PUT signals
├─ Same expiration, same timeframe
├─ Strike selection: Both ITM or both OTM
├─ Position sizing: Reduced per leg (50% normal size)
└─ Exit one leg if move is strongly directional
```

**Example Trade:**
```
Event: AAPL earnings announcement
User selects: Both sides (Bi-directional)
Platform response:
├─ If signal shows move up: Generate BUY CALL signal
├─ If signal shows move down: Generate BUY PUT signal
├─ If uncertain direction: Generate both
├─ Use lower position sizing (2% per leg vs 4% total)
├─ Both expire same date
└─ Goal: Profit from move, not direction
```

---

## User Preference Selection Logic

```python
class DirectionalBiasSelector:
    
    def select_bias(self, user_preference, market_signal):
        """
        Filter signals based on user's directional bias
        """
        
        if user_preference == "bull_only":
            # Only accept bullish signals
            if market_signal.direction == "BULLISH":
                return market_signal
            else:
                return None  # Filter out
        
        elif user_preference == "bear_only":
            # Only accept bearish signals
            if market_signal.direction == "BEARISH":
                return market_signal
            else:
                return None  # Filter out
        
        elif user_preference == "both_sides":
            # Accept both bullish and bearish
            if market_signal.direction in ["BULLISH", "BEARISH"]:
                return market_signal
            else:
                return None  # Still filter out neutral
        
        elif user_preference == "advanced_bidirectional":
            # Accept both, even generate pairs
            if market_signal.direction == "BULLISH":
                return ("CALL", market_signal)
            elif market_signal.direction == "BEARISH":
                return ("PUT", market_signal)
            elif market_signal.direction == "NEUTRAL":
                # Generate both CALL and PUT pair
                return ("CALL+PUT_PAIR", market_signal)
```

---

# SECTION 5: END-OF-DAY CLOSING STRATEGIES

## The EOD Challenge

**Why Close at End of Day?**[367]

```
Problem 1: Overnight Gap Risk
├─ News after hours can gap stock 5-10%
├─ Options can gap even more
├─ Your stop loss doesn't execute at 3 AM
└─ Result: Lose more than expected

Problem 2: Weekend Risk
├─ Friday close to Monday open: 2.5x normal gap risk
├─ Options can lose 20-30% over weekend
├─ News weekend: Geopolitical, earnings surprises
└─ Result: Weekend bleed

Problem 3: Off-Hours Volatility
├─ After-hours IV can spike if earnings (pre-announcement)
├─ Position can move against you with no ability to respond
├─ Market makers adjust bids wider
└─ Can't exit at reasonable price
```

**Solution: End-of-Day Closing Strategies**[361][363][364][367]

---

## Strategy 1: "Close Winners Daily" (Conservative)

**Best For: Risk Level 1-3, All Users Overnight Risk Averse**

```
Rules:
├─ At 3:00 PM ET (before close volatility spike)
├─ Check all open positions
├─ IF position is profitable (>10% gain):
│   ├─ Close immediately at market price
│   ├─ Lock in gains
│   └─ Reduce position risk
├─ IF position is losing:
│   ├─ Keep position OR
│   └─ Set tight stop to +0.5% (reduce loss)
├─ Close ALL positions by 4:00 PM
└─ Never hold overnight

When to Use:
├─ Risk level 1-2 (conservative)
├─ High volatility days
├─ Before major economic data
├─ Earnings week (even pre-earnings)
└─ High IV environment

Expected Results:
├─ Win rate: 60-70% (capture winners early)
├─ Average gain: 8-15% (less than full move)
├─ Risk reduced by 90%
├─ Sleep well at night: YES
└─ Miss some big moves: Possible (acceptable trade-off)

Implementation:
```python
def eod_close_winners_strategy(positions, current_time):
    """
    Close all profitable positions by 3 PM
    """
    if current_time.hour == 15 and current_time.minute == 0:
        # 3:00 PM ET
        for position in positions:
            gain_pct = (position.current_price - position.entry_price) / position.entry_price
            
            if gain_pct > 0.10:  # 10% gain
                close_position_at_market(position)
                log_trade(position, "EOD_CLOSE_WINNER")
            
            elif gain_pct < -0.05:  # 5% loss
                set_tight_stop(position, stop_pct=0.005)  # Exit at +0.5%
        
        # Close all remaining by 4 PM
        if current_time.hour == 16 and current_time.minute == 0:
            close_all_remaining_positions()
```

---

## Strategy 2: "Friday Close All" (Moderate)

**Best For: Risk Level 4-6, Work-Week Traders**

```
Rules:
├─ Monday-Thursday: Normal trading, hold overnight OK
├─ Friday 3:00 PM: Close ALL positions
│   ├─ Winners: Take profits
│   ├─ Losers: Cut losses
│   └─ Breakeven: Close and preserve
├─ Never hold Friday close to Monday open
├─ Weekend risk eliminated
└─ Restart fresh Monday

When to Use:
├─ Risk level 3-5
├─ Can't monitor during off-hours
├─ Don't want to worry weekends
├─ Smaller account (compounding focus)
└─ Normal volatility environment

Expected Results:
├─ Win rate: 55-65%
├─ Average gain: 15-25% per week
├─ Weekend risk: 0%
├─ Sleep well: YES
└─ Miss weekend gaps: Possible (rare)

Implementation:
```python
def friday_close_all_strategy(positions, current_day, current_time):
    """
    Close all positions Friday at 3 PM
    """
    if current_day == "Friday" and current_time.hour == 15:
        for position in positions:
            close_position_at_market(position)
            calculate_weekly_pnl()
            log_trade(position, "FRIDAY_CLOSE_ALL")
```

---

## Strategy 3: "IV Crush Avoidance" (Advanced)

**Best For: Risk Level 7-10, Earnings Traders**

```
Use Case: Earnings events, economic data releases

Pre-Event Rules (5 days before):
├─ DO NOT enter new positions
├─ Take profits on existing positions
├─ Let short positions expire or close
├─ Exit by day-before close
└─ Go to cash before announcement

During-Event (Earnings Day):
├─ IF you want to trade earnings:
│   ├─ Enter at open (when spread is tight)
│   ├─ Exit BEFORE announcement
│   ├─ OR exit immediately after (capture 50%)
│   └─ Never hold through announcement
├─ IF you don't want earnings risk:
│   └─ Stay in cash that day

Post-Event Rules:
├─ IV will crush 30-50%
├─ Don't buy long options post-announcement
├─ Wait 2-3 days for IV to stabilize
└─ Normal trading resumes

When to Use:
├─ Risk level 7-10 (only if you profit from IV crush)
├─ Risk level 1-6: Avoid entirely
├─ Earnings week: Be cautious
├─ Pre-earnings volatility: Can trade UP TO announcement
└─ Post-earnings: Wait for IV reset

Expected Results:
├─ Win rate: 70%+ (capturing pre-earnings IV spike)
├─ Risk: Very high if position held through
├─ Potential: 50-100% returns (IV spike profits)
└─ Drawdown: 20-50% if wrong direction + IV crush

Strategy Example:
```python
def iv_crush_avoidance_strategy(symbols, earnings_calendar):
    """
    Avoid holding through earnings, profit from IV spike
    """
    
    for symbol in symbols:
        earnings_date = earnings_calendar.get_next_earnings(symbol)
        days_to_earnings = (earnings_date - today).days
        
        if days_to_earnings < 5:
            # Close existing positions
            for position in get_positions(symbol):
                if position.dte < 7:
                    # Option expires before earnings
                    close_position_at_market(position)
        
        elif days_to_earnings == 0:
            # Earnings today
            for position in get_positions(symbol):
                current_price = get_current_price()
                entry_price = position.entry_price
                gain = (current_price - entry_price) / entry_price
                
                if gain > 0.30:  # 30% gain
                    # Exit before announcement
                    close_position_at_market(position)
                
                elif gain < -0.20:  # 20% loss
                    # Cut loss before announcement
                    close_position_at_market(position)
                
                else:
                    # Otherwise, close at 50% of max theoretical profit
                    close_position_at_market(position)
```

---

## Strategy 4: "Intraday Only" (Max Frequency Traders)

**Best For: Risk Level 8-10, Day Traders**

```
Rules:
├─ Enter: Within 1 hour of market open (tight spreads)
├─ Exit: Latest 3:00 PM (before close volatility)
├─ Hold time: 2-8 hours maximum
├─ Never hold overnight
├─ Never hold weekends
├─ Theta decay: Not a factor (short hold)
└─ Maximum positions: 3-5 simultaneously

When to Use:
├─ Risk level 8-10
├─ Day trader schedule (available during hours)
├─ High IV environment (wider moves)
├─ Don't want overnight risk
├─ Want to compound daily
└─ Can monitor positions constantly

Expected Results:
├─ Win rate: 55-60%
├─ Average gain: 15-20% per trade
├─ Average hold: 4 hours
├─ Daily compounding: Strong
├─ Overnight risk: 0%
└─ Capital utilization: High

Implementation:
```python
def intraday_only_strategy(positions, current_time):
    """
    Intraday trading: enter at open, exit by 3 PM
    """
    
    if current_time.hour == 9 and current_time.minute == 30:
        # Market open - tight spreads
        generate_entry_signals()  # Signal generation
    
    if current_time.hour >= 15:
        # 3 PM or later - close everything
        for position in positions:
            if position.entry_time.day == today.day:
                close_position_at_market(position)
                log_trade(position, "INTRADAY_EXIT")
```

---

# SECTION 6: IV CRUSH MANAGEMENT & PRE-MARKET DECISIONS

## Understanding IV Crush[358][361][363][364][367]

**What is IV Crush?**

```
Before Earnings:
├─ Implied Volatility: High (50-100 IVR)
├─ Option premiums: Expensive
├─ Option prices: Heavily weighted on time value
└─ Reason: Market expects large move

Announcement Moment:
├─ Stock moves (up, down, or sideways)
├─ Reality becomes known
└─ Uncertainty is eliminated

After Earnings:
├─ Implied Volatility: Crashes 30-50%
├─ Option premiums: Collapse
├─ Even ITM options lose significant value
├─ Option buyers: Lose 20-40% just from IV crush
└─ Result: You can be RIGHT on direction, WRONG overall
```

**The Numbers:**[358][361][364]

```
Example: Apple Pre-Earnings

Day Before Earnings:
├─ Call option: $5.00 premium
├─ IV Rank: 85%
├─ All from time value: $4.50
├─ Intrinsic value: $0.50

Earnings Announcement:
├─ Stock moves: UP $2.00 (good!)
├─ Call option becomes worth: $2.50
│   ├─ Intrinsic value: $2.50
│   └─ Time value: $0.00 (crushed)
└─ You lose: $2.50 (-50%)
    ├─ Reason: IV crush
    └─ Despite being correct on direction!

If You Sold Call Instead:
├─ Collected: $5.00 premium
├─ After earnings: Option worth $2.50
├─ Your profit: $2.50 (50% gain)
└─ You profit from IV crush
```

---

## Pre-Earnings Decision Framework[358][361][364]

```python
def pre_earnings_decision(symbol, earnings_date, days_to_earnings):
    """
    Decide: Trade, Skip, or Change Strategy?
    """
    
    if days_to_earnings < 0:
        # Earnings passed
        current_strategy = "NORMAL"
        return current_strategy
    
    elif days_to_earnings == 0:
        # Earnings TODAY
        if can_trade_before_announcement():
            # Option: Enter early, exit before announcement
            strategy = "TRADE_TO_ANNOUNCEMENT"
            instructions = """
            - Enter at 9:30 AM (open)
            - Exit by 3:50 PM (10 min before announcement)
            - Capture pre-announcement momentum
            - Avoid IV crush entirely
            """
        else:
            strategy = "SKIP_ENTIRELY"
            instructions = "Skip trading this symbol today"
        
        return strategy, instructions
    
    elif days_to_earnings <= 5:
        # Earnings within 5 days
        
        # Check if option expires BEFORE earnings
        option_dte = get_option_dte()
        
        if option_dte < days_to_earnings:
            # Option expires before earnings
            strategy = "TRADE_SAFE"
            instructions = """
            - Option expires before earnings (safe)
            - No IV crush impact
            - Trade normally
            """
        else:
            # Option expires AFTER earnings
            strategy = "AVOID_OR_SELL"
            instructions = """
            - Option expires after earnings (risky for buyers)
            - OPTION 1: Avoid entering long options
            - OPTION 2: Sell options (profit from IV crush)
            - OPTION 3: Exit existing longs before earnings
            """
        
        return strategy, instructions
    
    elif days_to_earnings <= 14:
        # Earnings in 1-2 weeks
        strategy = "NORMAL_CAUTION"
        instructions = """
        - Can trade
        - But be aware earnings approaching
        - IV will start spiking (getting expensive)
        - May want to reduce position size
        """
        return strategy, instructions
    
    else:
        # Earnings far away (>2 weeks)
        strategy = "NORMAL"
        return strategy
```

---

## IV Crush Profit Strategy (Advanced)[358][361][364][367]

**For aggressive traders (Risk Level 8-10) who want to PROFIT from IV crush:**

```
Pre-Earnings Setup (3-5 days before):
├─ SELL options (calls or strangles)
├─ Collect high premium (IV is spiked)
├─ Strike: Far OTM (odds in your favor)
├─ Example: SPY down 4%, sell 545 put
└─ Probability: 90%+ (stock less than 4%)

Earnings Day:
├─ Stock moves (usually less than IV implied)
├─ IV crushes 30-50%
├─ Your short options worth much less
├─ Exit immediately after announcement
└─ Profit: From IV crush + time decay

Exit Rules (Same-Day):
├─ Exit within 10 minutes of announcement
├─ Don't wait for market to normalize
├─ Collect max profit from IV crush
├─ Avoid sudden large move against you
└─ Book profit and move on

Example Trade:
```
SPY 5 days before earnings, current: $550
- IV Rank: 85% (very high)
- Sell $545 put (OTM, 95% probability)
- Collect: $2.00 premium
- Hold for 5 days

Earnings Day:
- SPY moves: $552 (2 points, less than expected 4 point move)
- IV collapses: 85% → 35%
- $545 put now worth: $0.30 (from $2.00)
- Your profit: $1.70 (85% gain)

Why profitable:
- Stock did NOT move below $545 (you win)
- IV crush helped (option became worthless)
- Time decay helped (4 days hold)
```

---

# SECTION 7-9: IMPLEMENTATION ARCHITECTURE & FINAL RECOMMENDATIONS

## User Preference Form (Dashboard)

```python
class UserPreferences:
    
    def __init__(self):
        self.preferences = {
            # Dimension 1: Risk Tolerance
            "risk_tolerance": None,  # 1-10 scale
            
            # Dimension 2: Trade Frequency
            "trade_frequency": None,  # "conservative", "moderate", "aggressive"
            
            # Dimension 3: Directional Bias
            "directional_bias": None,  # "bull_only", "bear_only", "both_sides"
            
            # Dimension 4: EOD/Overnight Strategy
            "eod_close_strategy": None,  # "close_winners", "friday_only", "keep_open"
            
            # Dimension 5: Earnings Handling
            "earnings_strategy": None,  # "avoid", "trade_to_announcement", "profit_from_crush"
        }
    
    def validate_and_save(self):
        """
        Ensure preferences are consistent
        """
        
        # Risk Level 1-2 shouldn't use aggressive frequency
        if self.risk_tolerance <= 2 and self.trade_frequency == "aggressive":
            raise ValueError("Conservative risk cannot use aggressive frequency")
        
        # Risk Level 1-3 should close daily
        if self.risk_tolerance <= 3 and self.eod_close_strategy == "keep_open":
            self.eod_close_strategy = "close_winners"
            print("Updated EOD strategy to match risk level")
        
        # Risk Level 8-10 can skip earnings
        if self.risk_tolerance >= 8 and self.earnings_strategy == "avoid":
            print("You can afford to trade earnings at your risk level")
        
        return self.preferences
```

---

## Signal Generation with User Preferences

```python
def generate_signal_with_preferences(
    market_indicators,
    user_preferences,
    current_time
):
    """
    Generate signals respecting all user preferences
    """
    
    # Step 1: Calculate base confidence
    confidence = calculate_indicator_consensus(market_indicators)
    
    # Step 2: Apply frequency filter
    frequency = user_preferences["trade_frequency"]
    threshold = {
        "conservative": 0.85,
        "moderate": 0.70,
        "aggressive": 0.60
    }[frequency]
    
    if confidence < threshold[frequency]:
        return None  # Skip signal
    
    # Step 3: Determine direction
    direction = determine_direction(market_indicators)
    
    # Step 4: Apply directional bias filter
    bias = user_preferences["directional_bias"]
    
    if bias == "bull_only" and direction != "BULLISH":
        return None  # Skip
    if bias == "bear_only" and direction != "BEARISH":
        return None  # Skip
    if bias == "both_sides":
        pass  # Accept any direction
    
    # Step 5: Check earnings calendar
    if is_earnings_week(symbol):
        earnings_strategy = user_preferences["earnings_strategy"]
        
        if earnings_strategy == "avoid":
            return None  # Skip entirely
        elif earnings_strategy == "trade_to_announcement":
            # Mark: "Close before 4 PM"
            signal.close_time = "before_4pm"
        elif earnings_strategy == "profit_from_crush":
            # Different strategy entirely (sell options instead)
            signal = generate_iv_crush_strategy(market_indicators)
    
    # Step 6: Apply risk tolerance position sizing
    risk_level = user_preferences["risk_tolerance"]
    position_size = calculate_position_size(risk_level, confidence)
    signal.position_size = position_size
    
    # Step 7: Apply EOD strategy
    eod_strategy = user_preferences["eod_close_strategy"]
    if eod_strategy == "close_winners":
        signal.close_time = "3pm_if_profitable"
    elif eod_strategy == "friday_only":
        signal.close_time = "friday_3pm_all"
    else:
        signal.close_time = "user_decision"
    
    return signal
```

---

## Dashboard Display Example

```
USER PREFERENCES PANEL
═══════════════════════════════════════════════════════════

1. RISK TOLERANCE
   Slider: [======●====] Level 6/10 (Moderate-Aggressive)
   
   Description:
   ├─ Comfortable with 6-8% loss per trade
   ├─ Can hold 5-8 positions simultaneously
   ├─ Accept 50% win rate
   └─ Expected return: 15-20% annually

2. TRADE FREQUENCY
   Radio:  ○ Conservative (3-5/week)
           ● Moderate (8-15/week)
           ○ Aggressive (15-30/week)
   
   Current: "High Probability Trades"
   Signals when: Confidence ≥ 70%

3. DIRECTIONAL BIAS
   Radio:  ○ Bull Only (BUY CALLS)
           ○ Bear Only (BUY PUTS)
           ● Both Sides (Either)
           ○ Advanced Bi-Directional

4. END-OF-DAY STRATEGY
   Radio:  ● Close Winners Daily (>10% gain by 3 PM)
           ○ Friday Close All (weekends off)
           ○ Hold Overnight (user discretion)

5. EARNINGS HANDLING
   Radio:  ○ Avoid Entirely
           ● Trade to Announcement (exit before)
           ○ Profit from IV Crush (sell options)

[SAVE PREFERENCES]  [RESET DEFAULTS]

Next Signal Expected: In ~4 hours
Last Signal: 3:20 PM Today (PROFITABLE - CLOSED at +25%)
```

---

## Expected Outcomes by Profile

```
PROFILE: Risk Level 6, Moderate Frequency, Both Sides

Week 1:
├─ Signals generated: 10
├─ Signals accepted: 8 (user approved or auto-trade)
├─ Winners: 5 (62% win rate)
├─ Losers: 3
├─ Average win: +$185 per contract
├─ Average loss: -$65 per contract
├─ Net profit: +$925 on account
└─ Weekly return: +1.85%

Month:
├─ Signals: ~40
├─ Trades: ~32
├─ Winners: ~20
├─ Losers: ~12
├─ Monthly profit: ~$3,700
├─ Monthly return: +7.4%
└─ Trend: Consistent performance

Quarter:
├─ Quarterly profit: ~$11,000
├─ Quarterly return: +22%
├─ Max drawdown: -8%
├─ Sharpe ratio: 2.1
└─ Status: Track record forming
```

---

## Final Implementation Checklist

- [ ] **User Preference System**
  - [ ] Risk tolerance slider (1-10)
  - [ ] Trade frequency selector
  - [ ] Directional bias selector
  - [ ] EOD/overnight strategy selector
  - [ ] Earnings strategy selector

- [ ] **Signal Generation Engine**
  - [ ] Apply confidence threshold based on frequency
  - [ ] Filter by directional bias
  - [ ] Filter by earnings calendar
  - [ ] Calculate position sizing by risk level
  - [ ] Apply EOD close timing rules

- [ ] **Execution System**
  - [ ] Respect user EOD preferences
  - [ ] Close winners at 3 PM if selected
  - [ ] Close all on Friday if selected
  - [ ] Set stops based on risk level
  - [ ] Set profit targets (30-50% depending on risk)

- [ ] **Monitoring & Reporting**
  - [ ] Daily P&L by strategy
  - [ ] Win rate tracking
  - [ ] Actual vs. expected performance
  - [ ] Recommendation adjustments
  - [ ] User education on results

---

## BOTTOM LINE

Your platform now has THREE powerful customization dimensions:

1. **Risk Tolerance (1-10):** Controls aggressiveness
2. **Trade Frequency:** Controls signal quality threshold
3. **Directional Bias:** Controls which trades to take

Combined with **EOD and earnings strategies**, this lets each user tailor the platform to their exact needs.

**Recommendation: Start with simple (Risk + Frequency + Bias), then add EOD strategies in Phase 2.**

---

**RESEARCH COMPLETE. READY TO BUILD.**
