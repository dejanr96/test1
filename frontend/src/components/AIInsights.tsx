import React from 'react';
import { Brain, Zap, BarChart2 } from 'lucide-react';

interface AIInsightsProps {
    thought: string;
    value: number;
    confidence: number;
    regime: string;
}

export const AIInsights: React.FC<AIInsightsProps> = ({ thought, value, confidence, regime }) => {
    return (
        <div className="bg-apex-panel p-6 rounded-xl border border-slate-700 shadow-lg h-full flex flex-col">
            <div className="flex items-center gap-2 mb-6">
                <Brain className="w-6 h-6 text-purple-400" />
                <h3 className="text-lg font-bold text-white">AI Agent Insights</h3>
            </div>

            <div className="flex-1 space-y-6">
                {/* Strategy Info */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-slate-800/50 p-3 rounded-lg">
                        <p className="text-slate-400 text-xs mb-1 flex items-center gap-1">
                            <Zap className="w-3 h-3" /> Strategy
                        </p>
                        <p className="font-mono text-sm text-blue-300">PPO (Proximal Policy Optimization)</p>
                    </div>
                    <div className="bg-slate-800/50 p-3 rounded-lg">
                        <p className="text-slate-400 text-xs mb-1 flex items-center gap-1">
                            <BarChart2 className="w-3 h-3" /> Training
                        </p>
                        <p className="font-mono text-sm text-blue-300">7 Days (1m Candles)</p>
                    </div>
                </div>

                {/* Thought Process */}
                <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700/50 relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-1 h-full bg-purple-500" />
                    <p className="text-slate-400 text-xs font-bold mb-2 uppercase tracking-wider">Live Thought Process</p>
                    <div className="text-slate-200 text-sm leading-relaxed font-medium animate-pulse">
                        {thought ? (
                            thought.split('. ').map((sentence, idx) => (
                                sentence.trim() && (
                                    <p key={idx} className="mb-1 last:mb-0">
                                        {sentence.trim()}{sentence.endsWith('.') ? '' : '.'}
                                    </p>
                                )
                            ))
                        ) : (
                            "Initializing neural network..."
                        )}
                    </div>
                </div>

                {/* Metrics */}
                <div className="space-y-4">
                    <div>
                        <div className="flex justify-between text-xs mb-1">
                            <span className="text-slate-400">Value Estimate (Expected Reward)</span>
                            <span className="text-white font-mono">{value?.toFixed(4) ?? '0.0000'}</span>
                        </div>
                        <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                            {/* Visualize Value: Center is 0. Range -2 to 2 approx */}
                            <div
                                className={`h-full transition-all duration-500 ${value >= 0 ? 'bg-green-500' : 'bg-red-500'}`}
                                style={{
                                    width: `${Math.min(Math.abs(value) * 25, 50)}%`,
                                    marginLeft: value >= 0 ? '50%' : `${50 - Math.min(Math.abs(value) * 25, 50)}%`
                                }}
                            />
                        </div>
                    </div>

                    <div>
                        <div className="flex justify-between text-xs mb-1">
                            <span className="text-slate-400">Action Confidence</span>
                            <span className="text-white font-mono">{Math.min(100, confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-blue-500 transition-all duration-500"
                                style={{ width: `${Math.min(confidence * 100, 100)}%` }}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
