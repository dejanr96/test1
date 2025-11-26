import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface MetricsCardProps {
    title: string;
    value: string | number;
    subValue?: string;
    icon: LucideIcon;
    trend?: 'up' | 'down' | 'neutral';
    color?: string;
}

export const MetricsCard: React.FC<MetricsCardProps> = ({ title, value, subValue, icon: Icon, color = 'text-white' }) => {
    return (
        <div className="bg-apex-panel p-4 rounded-xl border border-slate-700 shadow-lg">
            <div className="flex justify-between items-start">
                <div>
                    <p className="text-slate-400 text-sm font-medium">{title}</p>
                    <h3 className={`text-2xl font-bold mt-1 ${color}`}>{value}</h3>
                    {subValue && <p className="text-slate-500 text-xs mt-1">{subValue}</p>}
                </div>
                <div className="p-2 bg-slate-800 rounded-lg">
                    <Icon className="w-5 h-5 text-slate-300" />
                </div>
            </div>
        </div>
    );
};
