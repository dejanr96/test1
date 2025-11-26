import React, { useEffect, useState } from 'react';
import { Activity, Wallet, TrendingUp, AlertCircle } from 'lucide-react';
import { MetricsCard } from './MetricsCard';
import { ActivePosition } from './ActivePosition';
import { PriceChart } from './PriceChart';
import { LogFeed } from './LogFeed';
import { AIInsights } from './AIInsights';

interface MarketUpdate {
    price: number;
    regime: string;
    action: string;
    confidence: number;
    balance: number;
    position: number;
    entry_price: number;
    pnl: number;
    pnl_pct: number;
    lob_spread: number;
    lob_imbalance: number;
    ai_value: number;
    ai_log_prob: number;
    ai_thought: string;
}

interface LogEntry {
    timestamp: string;
    message: string;
    type: 'info' | 'warning' | 'error' | 'success';
}

export const Dashboard: React.FC = () => {
    const [connected, setConnected] = useState(false);
    const [marketData, setMarketData] = useState<MarketUpdate | null>(null);
    const [priceHistory, setPriceHistory] = useState<{ time: string, price: number }[]>([]);
    const [logs, setLogs] = useState<LogEntry[]>([]);

    const addLog = (message: string, type: LogEntry['type'] = 'info') => {
        setLogs(prev => [...prev, {
            timestamp: new Date().toLocaleTimeString(),
            message,
            type
        }].slice(-50)); // Keep last 50 logs
    };

    useEffect(() => {
        const connect = () => {
            // Use /ws proxy if in dev, or direct if needed. 
            const ws = new WebSocket(`ws://localhost:8000/ws`);

            ws.onopen = () => {
                setConnected(true);
                addLog('System connected to Trading Core', 'success');
            };

            ws.onclose = () => {
                setConnected(false);
                addLog('Disconnected from Trading Core. Reconnecting...', 'warning');
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                try {
                    const payload = JSON.parse(event.data);
                    if (payload.type === 'market_update') {
                        const data = payload.data as MarketUpdate;
                        setMarketData(data);

                        // Update Price History (keep last 100 points)
                        setPriceHistory(prev => {
                            const newPoint = {
                                time: new Date().toLocaleTimeString(),
                                price: data.price
                            };
                            const newHistory = [...prev, newPoint];
                            if (newHistory.length > 100) return newHistory.slice(-100);
                            return newHistory;
                        });

                        // Log significant actions
                        if (data.action !== 'HOLD') {
                            const confPct = Math.min(100, data.confidence * 100).toFixed(1);
                            addLog(`Action Signal: ${data.action} (Conf: ${confPct}%)`, 'info');
                        }
                    }
                } catch (e) {
                    console.error('Parse error', e);
                }
            };

            return ws;
        };

        const ws = connect();
        return () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        };
    }, []);

    return (
        <div className="min-h-screen bg-apex-dark text-white p-6">
            {/* Header */}
            <header className="flex justify-between items-center mb-8">
                <div>
                    <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                        APEX AI TRADER
                    </h1>
                    <p className="text-slate-400 text-xs mt-1">Deep Reinforcement Learning Core</p>
                </div>
                <div className="flex items-center gap-4">
                    <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${connected ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'}`}>
                        <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
                        {connected ? 'SYSTEM ONLINE' : 'DISCONNECTED'}
                    </div>
                </div>
            </header>

            {/* Main Grid */}
            <div className="grid grid-cols-12 gap-6">
                {/* Top Metrics Row */}
                <div className="col-span-12 grid grid-cols-4 gap-6">
                    <MetricsCard
                        title="Total Balance"
                        value={marketData ? `$${marketData.balance.toFixed(2)}` : '---'}
                        icon={Wallet}
                        color="text-white"
                    />
                    <MetricsCard
                        title="Unrealized PnL"
                        value={marketData ? `$${marketData.pnl.toFixed(2)}` : '---'}
                        subValue={marketData ? `${marketData.pnl_pct.toFixed(2)}%` : ''}
                        icon={TrendingUp}
                        color={marketData && marketData.pnl >= 0 ? 'text-apex-green' : 'text-apex-red'}
                    />
                    <MetricsCard
                        title="Market Regime"
                        value={marketData ? marketData.regime : '---'}
                        icon={Activity}
                        color="text-blue-400"
                    />
                    <MetricsCard
                        title="LOB Imbalance"
                        value={marketData ? marketData.lob_imbalance.toFixed(4) : '---'}
                        icon={AlertCircle}
                        color="text-yellow-400"
                    />
                </div>

                {/* Middle Row: Chart & Position & AI Insights */}
                <div className="col-span-8">
                    <PriceChart data={priceHistory} />
                </div>
                <div className="col-span-4">
                    <div className="grid grid-rows-2 gap-6 h-full">
                        <ActivePosition
                            position={marketData?.position || 0}
                            entryPrice={marketData?.entry_price || 0}
                            currentPrice={marketData?.price || 0}
                            pnl={marketData?.pnl || 0}
                            pnlPct={marketData?.pnl_pct || 0}
                        />
                        <AIInsights
                            thought={marketData?.ai_thought || ""}
                            value={marketData?.ai_value || 0}
                            confidence={marketData?.confidence || 0}
                            regime={marketData?.regime || "Unknown"}
                        />
                    </div>
                </div>

                {/* Bottom Row: Logs */}
                <div className="col-span-12">
                    <LogFeed logs={logs} />
                </div>
            </div>
        </div>
    );
};
