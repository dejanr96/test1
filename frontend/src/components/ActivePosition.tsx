import React from 'react';
import { Clock } from 'lucide-react';

interface PositionProps {
    position: number;
    entryPrice: number;
    currentPrice: number;
    pnl: number;
    pnlPct: number;
}

export const ActivePosition: React.FC<PositionProps> = ({ position, entryPrice, currentPrice, pnl, pnlPct }) => {
    if (position === 0) {
        return (
            <div className="bg-apex-panel p-6 rounded-xl border border-slate-700 shadow-lg h-full flex flex-col justify-center items-center text-slate-500">
                <Clock className="w-12 h-12 mb-2 opacity-50" />
                <p>No Active Position</p>
            </div>
        );
    }

    const isLong = position > 0;
    const isProfit = pnl >= 0;

    return (
        <div className="bg-apex-panel p-6 rounded-xl border border-slate-700 shadow-lg h-full relative overflow-hidden">
            <div className={`absolute top-0 left-0 w-1 h-full ${isLong ? 'bg-apex-green' : 'bg-apex-red'}`} />

            <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-2">
                    <span className={`px-2 py-1 rounded text-xs font-bold ${isLong ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'}`}>
                        {isLong ? 'LONG' : 'SHORT'}
                    </span>
                    <span className="text-slate-300 font-mono text-sm">{Math.abs(position).toFixed(4)} BTC</span>
                </div>
                <div className={`text-xl font-bold ${isProfit ? 'text-apex-green' : 'text-apex-red'}`}>
                    {pnl > 0 ? '+' : ''}{pnl.toFixed(2)} USDT
                </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
                <div>
                    <p className="text-slate-500 text-xs">Entry Price</p>
                    <p className="text-white font-mono">{entryPrice.toFixed(2)}</p>
                </div>
                <div>
                    <p className="text-slate-500 text-xs">Mark Price</p>
                    <p className="text-white font-mono">{currentPrice.toFixed(2)}</p>
                </div>
                <div>
                    <p className="text-slate-500 text-xs">ROE</p>
                    <p className={`font-mono ${isProfit ? 'text-apex-green' : 'text-apex-red'}`}>
                        {pnlPct.toFixed(2)}%
                    </p>
                </div>
                <div>
                    <p className="text-slate-500 text-xs">Leverage</p>
                    <p className="text-white font-mono">Dynamic</p>
                </div>
            </div>
        </div>
    );
};
