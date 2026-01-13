"""
IB Algo Trading Dashboard
==========================

Web-based control panel for AI signal configuration and monitoring.
Built with Flask for simplicity and real-time updates.

Features:
- Signal Settings Panel: Timeframe, thresholds, auto-execute toggle
- Live Signal Display: Real-time overbought/oversold with consensus scores
- Position Monitor: Active positions with trailing stop status
- Trade Log: Recent trades with AI signal reasoning

Endpoints:
- GET  /              - Main dashboard view
- GET  /api/settings  - Get current settings
- POST /api/settings  - Update settings
- GET  /api/signals   - Get live signals
- GET  /api/positions - Get active positions
- POST /api/approve/<trade_id> - Approve pending trade
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not installed. Run: pip install flask")

logger = logging.getLogger(__name__)


# ============= Dashboard Configuration =============

@dataclass
class DashboardSettings:
    """Configurable settings for the trading dashboard."""
    
    # Signal generation
    signal_timeframe: str = "5m"  # 5m, 15m, 1h, 4h
    min_score_threshold: int = 60  # 40-100
    
    # Auto-execution
    auto_execute_enabled: bool = True
    auto_execute_threshold: int = 85  # 70-100
    
    # Risk management
    max_position_size: int = 10  # Max contracts per trade
    max_daily_trades: int = 20
    
    # Monitoring
    refresh_interval_sec: int = 5


# Global settings instance
SETTINGS = DashboardSettings()
settings_lock = Lock()


def get_settings() -> Dict:
    """Get current settings as dictionary."""
    with settings_lock:
        return asdict(SETTINGS)


def update_settings(**kwargs) -> Dict:
    """Update settings from dashboard."""
    with settings_lock:
        for key, value in kwargs.items():
            if hasattr(SETTINGS, key):
                setattr(SETTINGS, key, value)
                logger.info(f"Setting updated: {key} = {value}")
    return get_settings()


# ============= Trade Tracking =============

@dataclass
class PendingTrade:
    """Trade awaiting approval."""
    trade_id: str
    symbol: str
    signal_type: str
    consensus_score: float
    reasons: List[str]
    timestamp: datetime
    approved: bool = False
    rejected: bool = False


@dataclass 
class ActivePosition:
    """Active trading position."""
    symbol: str
    option_type: str  # CALL or PUT
    entry_price: float
    current_price: float
    pnl_pct: float
    stop_level: float
    signal_score: float
    entry_time: datetime


# In-memory storage
pending_trades: Dict[str, PendingTrade] = {}
active_positions: List[ActivePosition] = []
trade_log: List[Dict] = []
live_signals: List[Dict] = []


# ============= Dashboard HTML Template =============

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IB Algo Trading Dashboard</title>
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #334155;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
            padding: 1rem;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }
        
        .header h1 {
            font-size: 1.5rem;
            background: linear-gradient(135deg, var(--primary), #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .status-connected { background: var(--success); color: white; }
        .status-disconnected { background: var(--danger); color: white; }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }
        
        .card {
            background: var(--bg-card);
            border-radius: 0.75rem;
            padding: 1.25rem;
            border: 1px solid var(--border);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }
        
        .card-title {
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }
        
        .setting-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border);
        }
        
        .setting-row:last-child { border-bottom: none; }
        
        .setting-label {
            font-size: 0.875rem;
            color: var(--text);
        }
        
        .setting-input {
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 0.375rem;
            padding: 0.5rem 0.75rem;
            color: var(--text);
            font-size: 0.875rem;
            width: 120px;
        }
        
        select.setting-input { cursor: pointer; }
        
        .toggle {
            position: relative;
            width: 48px;
            height: 24px;
            background: var(--border);
            border-radius: 9999px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .toggle.active { background: var(--success); }
        
        .toggle::after {
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            transition: transform 0.2s;
        }
        
        .toggle.active::after { transform: translateX(24px); }
        
        .signal-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            background: var(--bg-dark);
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .signal-symbol {
            font-weight: 600;
            font-size: 1rem;
        }
        
        .signal-type {
            font-size: 0.75rem;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
        }
        
        .signal-buy { background: rgba(16, 185, 129, 0.2); color: var(--success); }
        .signal-sell { background: rgba(239, 68, 68, 0.2); color: var(--danger); }
        
        .score-bar {
            width: 100%;
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        
        .score-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }
        
        .score-high { background: var(--success); }
        .score-medium { background: var(--warning); }
        .score-low { background: var(--danger); }
        
        .btn {
            padding: 0.5rem 1rem;
            border-radius: 0.375rem;
            font-size: 0.875rem;
            font-weight: 500;
            border: none;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn-primary { background: var(--primary); color: white; }
        .btn-primary:hover { background: var(--primary-dark); }
        .btn-success { background: var(--success); color: white; }
        .btn-danger { background: var(--danger); color: white; }
        
        .empty-state {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
        }
        
        .position-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            background: var(--bg-dark);
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .pnl-positive { color: var(--success); }
        .pnl-negative { color: var(--danger); }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .live-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
            margin-right: 0.5rem;
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 IB Algo Trading Dashboard</h1>
        <div>
            <span class="live-indicator"></span>
            <span id="connectionStatus" class="status-badge status-connected">Connected</span>
        </div>
    </div>
    
    <div class="grid">
        <!-- Signal Settings -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">⚙️ Signal Settings</span>
                <button class="btn btn-primary" onclick="saveSettings()">Save</button>
            </div>
            
            <div class="setting-row">
                <span class="setting-label">Signal Timeframe</span>
                <select id="timeframe" class="setting-input">
                    <option value="5m">5 minutes</option>
                    <option value="15m">15 minutes</option>
                    <option value="1h">1 hour</option>
                    <option value="4h">4 hours</option>
                </select>
            </div>
            
            <div class="setting-row">
                <span class="setting-label">Min Score Threshold</span>
                <input type="number" id="minScore" class="setting-input" min="40" max="100" value="60">
            </div>
            
            <div class="setting-row">
                <span class="setting-label">Auto-Execute Threshold</span>
                <input type="number" id="autoExecute" class="setting-input" min="70" max="100" value="85">
            </div>
            
            <div class="setting-row">
                <span class="setting-label">Auto-Execute Enabled</span>
                <div id="autoEnabled" class="toggle active" onclick="toggleAutoExecute()"></div>
            </div>
            
            <div class="setting-row">
                <span class="setting-label">Max Position Size</span>
                <input type="number" id="maxPosition" class="setting-input" min="1" max="50" value="10">
            </div>
        </div>
        
        <!-- Live Signals -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">📊 Live Signals</span>
                <span id="signalCount" style="color: var(--text-muted)">0 signals</span>
            </div>
            <div id="signalsContainer">
                <div class="empty-state">No active signals</div>
            </div>
        </div>
        
        <!-- Pending Approvals -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">⏳ Pending Approvals</span>
                <span id="pendingCount" style="color: var(--text-muted)">0 pending</span>
            </div>
            <div id="pendingContainer">
                <div class="empty-state">No trades awaiting approval</div>
            </div>
        </div>
        
        <!-- Active Positions -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">Active Positions</span>
                <span id="positionCount" style="color: var(--text-muted)">0 positions</span>
            </div>
            <div id="positionsContainer">
                <div class="empty-state">No active positions</div>
            </div>
        </div>
        
        <!-- Risk & EOD Settings -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">Risk & EOD Settings</span>
                <button class="btn btn-primary" onclick="saveRiskSettings()">Save</button>
            </div>
            
            <div class="setting-row">
                <span class="setting-label">Risk Tolerance Level</span>
                <select id="riskLevel" class="setting-input" onchange="updateRiskDescription()">
                    <option value="1">1 - Ultra-Conservative</option>
                    <option value="2">2 - Conservative</option>
                    <option value="3">3 - Moderate-Conservative</option>
                    <option value="4">4 - Balanced-Conservative</option>
                    <option value="5" selected>5 - Moderate</option>
                    <option value="6">6 - Moderate-Aggressive</option>
                    <option value="7">7 - Aggressive</option>
                    <option value="8">8 - Very Aggressive</option>
                    <option value="9">9 - High Risk</option>
                    <option value="10">10 - Maximum</option>
                </select>
            </div>
            
            <div id="riskDescription" style="padding: 0.5rem; background: var(--bg-dark); border-radius: 0.5rem; margin-bottom: 0.75rem; font-size: 0.75rem; color: var(--text-muted);">
                Confidence: 65% | Position: 5% | Stop: 6%
            </div>
            
            <div class="setting-row">
                <span class="setting-label">Trade Frequency</span>
                <select id="tradeFrequency" class="setting-input">
                    <option value="conservative">Conservative (85%+)</option>
                    <option value="moderate" selected>Moderate (70%+)</option>
                    <option value="aggressive">Aggressive (60%+)</option>
                </select>
            </div>
            
            <div class="setting-row">
                <span class="setting-label">Directional Bias</span>
                <select id="directionalBias" class="setting-input">
                    <option value="bull_only">Bull Only (Calls)</option>
                    <option value="bear_only">Bear Only (Puts)</option>
                    <option value="both" selected>Both Sides</option>
                </select>
            </div>
            
            <div class="setting-row">
                <span class="setting-label">EOD Strategy</span>
                <select id="eodStrategy" class="setting-input">
                    <option value="close_winners">Close Winners by 3 PM</option>
                    <option value="friday_close" selected>Friday Close All</option>
                    <option value="intraday_only">Intraday Only</option>
                    <option value="hold">User Discretion</option>
                </select>
            </div>
            
            <div class="setting-row">
                <span class="setting-label">Avoid Earnings</span>
                <div id="avoidEarnings" class="toggle active" onclick="toggleAvoidEarnings()"></div>
            </div>
        </div>
    </div>
    
    <script>
        // State
        let settings = {};
        let autoExecuteEnabled = true;
        
        // Toggle auto-execute
        function toggleAutoExecute() {
            autoExecuteEnabled = !autoExecuteEnabled;
            const toggle = document.getElementById('autoEnabled');
            toggle.classList.toggle('active', autoExecuteEnabled);
        }
        
        // Risk level descriptions
        const riskLevelInfo = {
            1: { confidence: 90, position: 1, stop: 2, name: 'Ultra-Conservative' },
            2: { confidence: 80, position: 2.5, stop: 3, name: 'Conservative' },
            3: { confidence: 75, position: 3, stop: 4, name: 'Moderate-Conservative' },
            4: { confidence: 70, position: 4, stop: 5, name: 'Balanced-Conservative' },
            5: { confidence: 65, position: 5, stop: 6, name: 'Moderate' },
            6: { confidence: 65, position: 6, stop: 7, name: 'Moderate-Aggressive' },
            7: { confidence: 60, position: 8, stop: 8, name: 'Aggressive' },
            8: { confidence: 60, position: 10, stop: 9, name: 'Very Aggressive' },
            9: { confidence: 55, position: 12, stop: 10, name: 'High Risk' },
            10: { confidence: 50, position: 15, stop: 10, name: 'Maximum' },
        };
        
        // Update risk description
        function updateRiskDescription() {
            const level = parseInt(document.getElementById('riskLevel').value);
            const info = riskLevelInfo[level];
            document.getElementById('riskDescription').textContent = 
                `Confidence: ${info.confidence}% | Position: ${info.position}% | Stop: ${info.stop}%`;
        }
        
        // Toggle avoid earnings
        let avoidEarningsEnabled = true;
        function toggleAvoidEarnings() {
            avoidEarningsEnabled = !avoidEarningsEnabled;
            const toggle = document.getElementById('avoidEarnings');
            toggle.classList.toggle('active', avoidEarningsEnabled);
        }
        
        // Save risk settings
        async function saveRiskSettings() {
            const data = {
                risk_tolerance: parseInt(document.getElementById('riskLevel').value),
                trade_frequency: document.getElementById('tradeFrequency').value,
                directional_bias: document.getElementById('directionalBias').value,
                eod_strategy: document.getElementById('eodStrategy').value,
                avoid_earnings: avoidEarningsEnabled
            };
            
            try {
                const response = await fetch('/api/risk-settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                console.log('Risk settings saved:', result);
                alert('Risk settings saved!');
            } catch (err) {
                console.error('Error saving risk settings:', err);
                alert('Failed to save risk settings');
            }
        }
        
        // Save settings
        async function saveSettings() {
            const data = {
                signal_timeframe: document.getElementById('timeframe').value,
                min_score_threshold: parseInt(document.getElementById('minScore').value),
                auto_execute_threshold: parseInt(document.getElementById('autoExecute').value),
                auto_execute_enabled: autoExecuteEnabled,
                max_position_size: parseInt(document.getElementById('maxPosition').value)
            };
            
            try {
                const response = await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                console.log('Settings saved:', result);
                alert('Settings saved successfully!');
            } catch (err) {
                console.error('Error saving settings:', err);
                alert('Failed to save settings');
            }
        }
        
        // Load settings
        async function loadSettings() {
            try {
                const response = await fetch('/api/settings');
                settings = await response.json();
                
                document.getElementById('timeframe').value = settings.signal_timeframe;
                document.getElementById('minScore').value = settings.min_score_threshold;
                document.getElementById('autoExecute').value = settings.auto_execute_threshold;
                document.getElementById('maxPosition').value = settings.max_position_size;
                
                autoExecuteEnabled = settings.auto_execute_enabled;
                document.getElementById('autoEnabled').classList.toggle('active', autoExecuteEnabled);
            } catch (err) {
                console.error('Error loading settings:', err);
            }
        }
        
        // Render signal
        function renderSignal(signal) {
            const scoreClass = signal.score >= 80 ? 'score-high' : 
                              signal.score >= 60 ? 'score-medium' : 'score-low';
            const typeClass = signal.type.includes('BUY') ? 'signal-buy' : 'signal-sell';
            
            return `
                <div class="signal-item">
                    <div>
                        <div class="signal-symbol">${signal.symbol}</div>
                        <div class="score-bar">
                            <div class="score-fill ${scoreClass}" style="width: ${signal.score}%"></div>
                        </div>
                    </div>
                    <div style="text-align: right">
                        <span class="signal-type ${typeClass}">${signal.type}</span>
                        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem">
                            Score: ${signal.score}/100
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Approve trade
        async function approveTrade(tradeId) {
            try {
                await fetch(`/api/approve/${tradeId}`, { method: 'POST' });
                loadData();
            } catch (err) {
                console.error('Error approving trade:', err);
            }
        }
        
        // Reject trade
        async function rejectTrade(tradeId) {
            try {
                await fetch(`/api/reject/${tradeId}`, { method: 'POST' });
                loadData();
            } catch (err) {
                console.error('Error rejecting trade:', err);
            }
        }
        
        // Load all data
        async function loadData() {
            try {
                // Load signals
                const signalsRes = await fetch('/api/signals');
                const signals = await signalsRes.json();
                
                const signalsContainer = document.getElementById('signalsContainer');
                if (signals.length > 0) {
                    signalsContainer.innerHTML = signals.map(renderSignal).join('');
                } else {
                    signalsContainer.innerHTML = '<div class="empty-state">No active signals</div>';
                }
                document.getElementById('signalCount').textContent = `${signals.length} signals`;
                
                // Load pending trades
                const pendingRes = await fetch('/api/pending');
                const pending = await pendingRes.json();
                
                const pendingContainer = document.getElementById('pendingContainer');
                if (pending.length > 0) {
                    pendingContainer.innerHTML = pending.map(trade => `
                        <div class="signal-item">
                            <div>
                                <div class="signal-symbol">${trade.symbol}</div>
                                <div style="font-size: 0.75rem; color: var(--text-muted)">${trade.signal_type}</div>
                            </div>
                            <div>
                                <button class="btn btn-success" onclick="approveTrade('${trade.trade_id}')">✓</button>
                                <button class="btn btn-danger" onclick="rejectTrade('${trade.trade_id}')">✗</button>
                            </div>
                        </div>
                    `).join('');
                } else {
                    pendingContainer.innerHTML = '<div class="empty-state">No trades awaiting approval</div>';
                }
                document.getElementById('pendingCount').textContent = `${pending.length} pending`;
                
                // Load positions
                const positionsRes = await fetch('/api/positions');
                const positions = await positionsRes.json();
                
                const positionsContainer = document.getElementById('positionsContainer');
                if (positions.length > 0) {
                    positionsContainer.innerHTML = positions.map(pos => `
                        <div class="position-row">
                            <div>
                                <div class="signal-symbol">${pos.symbol}</div>
                                <div style="font-size: 0.75rem; color: var(--text-muted)">${pos.option_type}</div>
                            </div>
                            <div style="text-align: right">
                                <div class="${pos.pnl_pct >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                                    ${pos.pnl_pct >= 0 ? '+' : ''}${pos.pnl_pct.toFixed(1)}%
                                </div>
                                <div style="font-size: 0.75rem; color: var(--text-muted)">
                                    Stop: $${pos.stop_level.toFixed(2)}
                                </div>
                            </div>
                        </div>
                    `).join('');
                } else {
                    positionsContainer.innerHTML = '<div class="empty-state">No active positions</div>';
                }
                document.getElementById('positionCount').textContent = `${positions.length} positions`;
                
            } catch (err) {
                console.error('Error loading data:', err);
            }
        }
        
        // Initialize
        loadSettings();
        loadData();
        
        // Auto-refresh every 5 seconds
        setInterval(loadData, 5000);
    </script>
</body>
</html>
'''


# ============= Flask Application =============

if FLASK_AVAILABLE:
    app = Flask(__name__)
    
    @app.route('/')
    def dashboard():
        """Main dashboard view."""
        return render_template_string(DASHBOARD_HTML)
    
    @app.route('/api/settings', methods=['GET'])
    def api_get_settings():
        """Get current settings."""
        return jsonify(get_settings())
    
    @app.route('/api/settings', methods=['POST'])
    def api_update_settings():
        """Update settings."""
        data = request.get_json()
        return jsonify(update_settings(**data))
    
    @app.route('/api/signals', methods=['GET'])
    def api_get_signals():
        """Get live signals."""
        # Return mock signals for demo
        return jsonify(live_signals)
    
    @app.route('/api/pending', methods=['GET'])
    def api_get_pending():
        """Get pending trades."""
        return jsonify([asdict(t) for t in pending_trades.values() 
                       if not t.approved and not t.rejected])
    
    @app.route('/api/positions', methods=['GET'])
    def api_get_positions():
        """Get active positions."""
        return jsonify([asdict(p) for p in active_positions])
    
    @app.route('/api/approve/<trade_id>', methods=['POST'])
    def api_approve_trade(trade_id):
        """Approve a pending trade."""
        if trade_id in pending_trades:
            pending_trades[trade_id].approved = True
            logger.info(f"Trade {trade_id} approved")
            return jsonify({"status": "approved"})
        return jsonify({"error": "Trade not found"}), 404
    
    @app.route('/api/reject/<trade_id>', methods=['POST'])
    def api_reject_trade(trade_id):
        """Reject a pending trade."""
        if trade_id in pending_trades:
            pending_trades[trade_id].rejected = True
            logger.info(f"Trade {trade_id} rejected")
            return jsonify({"status": "rejected"})
        return jsonify({"error": "Trade not found"}), 404
    
    # Risk settings storage
    risk_settings = {
        "risk_tolerance": 5,
        "trade_frequency": "moderate",
        "directional_bias": "both",
        "eod_strategy": "friday_close",
        "avoid_earnings": True
    }
    
    @app.route('/api/risk-settings', methods=['GET'])
    def api_get_risk_settings():
        """Get current risk settings."""
        return jsonify(risk_settings)
    
    @app.route('/api/risk-settings', methods=['POST'])
    def api_update_risk_settings():
        """Update risk settings."""
        data = request.get_json()
        for key, value in data.items():
            if key in risk_settings:
                risk_settings[key] = value
                logger.info(f"Risk setting updated: {key} = {value}")
        return jsonify(risk_settings)


# ============= Integration Functions =============

def add_signal(symbol: str, signal_type: str, score: float, reasons: List[str] = None):
    """Add a live signal to the dashboard."""
    signal = {
        "symbol": symbol,
        "type": signal_type,
        "score": score,
        "reasons": reasons or [],
        "timestamp": datetime.now().isoformat()
    }
    live_signals.insert(0, signal)
    
    # Keep only last 20 signals
    if len(live_signals) > 20:
        live_signals.pop()
    
    return signal


def add_pending_trade(
    symbol: str, 
    signal_type: str, 
    score: float,
    reasons: List[str] = None
) -> str:
    """Add a trade awaiting approval."""
    trade_id = f"{symbol}_{datetime.now().strftime('%H%M%S')}"
    trade = PendingTrade(
        trade_id=trade_id,
        symbol=symbol,
        signal_type=signal_type,
        consensus_score=score,
        reasons=reasons or [],
        timestamp=datetime.now()
    )
    pending_trades[trade_id] = trade
    logger.info(f"Pending trade added: {trade_id}")
    return trade_id


def check_trade_approved(trade_id: str) -> Optional[bool]:
    """Check if a trade has been approved/rejected."""
    if trade_id not in pending_trades:
        return None
    trade = pending_trades[trade_id]
    if trade.approved:
        return True
    if trade.rejected:
        return False
    return None  # Still pending


def add_position(
    symbol: str,
    option_type: str,
    entry_price: float,
    current_price: float,
    stop_level: float,
    signal_score: float
):
    """Add an active position."""
    pnl_pct = ((current_price - entry_price) / entry_price) * 100
    position = ActivePosition(
        symbol=symbol,
        option_type=option_type,
        entry_price=entry_price,
        current_price=current_price,
        pnl_pct=pnl_pct,
        stop_level=stop_level,
        signal_score=signal_score,
        entry_time=datetime.now()
    )
    active_positions.append(position)
    return position


# ============= Main Entry Point =============

def run_dashboard(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Run the dashboard server."""
    if not FLASK_AVAILABLE:
        print("Flask is not installed. Run: pip install flask")
        return
    
    print("=" * 60)
    print("IB ALGO TRADING DASHBOARD")
    print("=" * 60)
    print(f"\nStarting dashboard at http://localhost:{port}")
    print("Press Ctrl+C to stop\n")
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    # Add some demo data for testing
    add_signal("SPY", "BUY_CALL", 82, ["SuperTrend bullish", "RSI oversold"])
    add_signal("QQQ", "SELL_CALL", 75, ["SuperTrend bearish", "RSI overbought with divergence"])
    add_signal("AAPL", "BUY_CALL", 68, ["MFI oversold", "Volume confirmation"])
    
    add_pending_trade("TSLA", "BUY_CALL", 72, ["RSI divergence detected"])
    
    add_position("MSFT", "CALL", 4.50, 5.20, 3.80, 85)
    add_position("NVDA", "PUT", 3.20, 2.90, 2.50, 78)
    
    run_dashboard(port=5000, debug=True)
