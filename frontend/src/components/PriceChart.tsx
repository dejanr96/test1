import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface ChartData {
    time: string;
    price: number;
    action?: 'BUY' | 'SELL' | 'HOLD';
}

interface PriceChartProps {
    data: ChartData[];
}

export const PriceChart: React.FC<PriceChartProps> = ({ data }) => {
    const minPrice = Math.min(...data.map(d => d.price)) * 0.999;
    const maxPrice = Math.max(...data.map(d => d.price)) * 1.001;

    return (
        <div className="bg-apex-panel p-4 rounded-xl border border-slate-700 shadow-lg h-[400px]">
            <h3 className="text-slate-400 text-sm font-medium mb-4">BTC/USDT Real-Time Price</h3>
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data}>
                    <defs>
                        <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis
                        dataKey="time"
                        stroke="#64748b"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(val) => val.split(' ')[1] || val}
                    />
                    <YAxis
                        domain={[minPrice, maxPrice]}
                        stroke="#64748b"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(val) => val.toFixed(2)}
                    />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#fff' }}
                        itemStyle={{ color: '#fff' }}
                    />
                    <Area
                        type="monotone"
                        dataKey="price"
                        stroke="#3b82f6"
                        fillOpacity={1}
                        fill="url(#colorPrice)"
                        isAnimationActive={false}
                    />
                    {/* We could add scatter points for Buy/Sell actions here if needed */}
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
};
