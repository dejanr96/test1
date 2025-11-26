import React, { useEffect, useRef } from 'react';
import { Terminal } from 'lucide-react';

interface LogEntry {
    timestamp: string;
    message: string;
    type: 'info' | 'warning' | 'error' | 'success';
}

interface LogFeedProps {
    logs: LogEntry[];
}

export const LogFeed: React.FC<LogFeedProps> = ({ logs }) => {
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    return (
        <div className="bg-apex-panel p-4 rounded-xl border border-slate-700 shadow-lg h-[300px] flex flex-col">
            <div className="flex items-center gap-2 mb-4">
                <Terminal className="w-4 h-4 text-slate-400" />
                <h3 className="text-slate-400 text-sm font-medium">System Logs</h3>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2 font-mono text-xs custom-scrollbar">
                {logs.map((log, idx) => (
                    <div key={idx} className="flex gap-2">
                        <span className="text-slate-500">[{log.timestamp}]</span>
                        <span className={`
              ${log.type === 'error' ? 'text-red-400' : ''}
              ${log.type === 'warning' ? 'text-yellow-400' : ''}
              ${log.type === 'success' ? 'text-green-400' : ''}
              ${log.type === 'info' ? 'text-slate-300' : ''}
            `}>
                            {log.message}
                        </span>
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>
        </div>
    );
};
