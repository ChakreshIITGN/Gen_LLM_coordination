import { useCallback, useEffect, useRef, useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { toPng } from 'html-to-image';
import { api } from '../lib/api';
import type { CompareEntry, TrajectorySummary } from '../lib/api';

const COLORS = ['#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#ec4899'];

const METRIC_LABELS: Record<string, string> = {
  total_reward: 'Total Reward',
  coverage_unique_positions: 'Coverage',
  net_displacement: 'Net Displacement',
  said_vs_did_rate: 'Said = Did Rate',
  reward_first_half_mean: 'Reward (1st half)',
  reward_second_half_mean: 'Reward (2nd half)',
  steps_run: 'Steps Run',
  final_position: 'Final Position',
};

function ExportButton({ targetRef, filename }: { targetRef: React.RefObject<HTMLDivElement | null>; filename: string }) {
  const handleExport = useCallback(async () => {
    if (!targetRef.current) return;
    const dataUrl = await toPng(targetRef.current, { backgroundColor: '#ffffff', pixelRatio: 2 });
    const link = document.createElement('a');
    link.download = `${filename}.png`;
    link.href = dataUrl;
    link.click();
  }, [targetRef, filename]);

  return (
    <button onClick={handleExport} className="text-xs text-gray-400 hover:text-indigo-600 transition-colors" title="Export as PNG">
      PNG &darr;
    </button>
  );
}

export default function ExperimentComparison() {
  const [searchParams] = useSearchParams();
  const runIds = (searchParams.get('runs') || '').split(',').filter(Boolean);

  const [entries, setEntries] = useState<CompareEntry[]>([]);
  const [trajectories, setTrajectories] = useState<Record<string, TrajectorySummary>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const rewardRef = useRef<HTMLDivElement>(null);
  const posRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!runIds.length) return;
    setLoading(true);
    setError(null);

    const fetchAll = async () => {
      const [compareData, ...trajData] = await Promise.all([
        api.compareRuns(runIds),
        ...runIds.map((id) => {
          const [tag, ts] = id.split('/');
          return api.getTrajectorySummary(tag, ts);
        }),
      ]);
      setEntries(compareData);
      const trajMap: Record<string, TrajectorySummary> = {};
      runIds.forEach((id, i) => { trajMap[id] = trajData[i]; });
      setTrajectories(trajMap);
    };
    fetchAll()
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [searchParams]);

  if (!runIds.length) return (
    <div className="text-center py-20">
      <p className="text-gray-500 mb-4">No runs selected for comparison.</p>
      <Link to="/experiments" className="text-indigo-600 hover:underline text-sm">
        Go to experiments &rarr;
      </Link>
    </div>
  );

  if (loading) return (
    <div className="space-y-4">
      {[1, 2, 3].map(i => <div key={i} className="h-8 bg-gray-200 rounded animate-pulse" style={{ width: `${70 + i * 10}%` }} />)}
      <div className="h-64 bg-gray-200 rounded-xl animate-pulse mt-6" />
    </div>
  );

  if (error) return (
    <div className="text-center py-20">
      <p className="text-red-500 mb-2">Failed to load comparison data</p>
      <p className="text-sm text-gray-400">{error}</p>
      <Link to="/experiments" className="text-indigo-600 hover:underline text-sm mt-4 inline-block">
        &larr; Back to experiments
      </Link>
    </div>
  );

  const maxSteps = Math.max(...Object.values(trajectories).map((t) => t.steps.length));
  const rewardOverlay = Array.from({ length: maxSteps }, (_, i) => {
    const point: Record<string, number> = { step: i };
    runIds.forEach((id) => {
      const t = trajectories[id];
      if (t && i < t.cumulative_rewards.length) point[id] = t.cumulative_rewards[i];
    });
    return point;
  });

  const positionOverlay = Array.from({ length: maxSteps }, (_, i) => {
    const point: Record<string, number> = { step: i };
    runIds.forEach((id) => {
      const t = trajectories[id];
      if (t && i < t.positions.length) point[id] = t.positions[i];
    });
    return point;
  });

  const allMetricKeys = new Set<string>();
  entries.forEach((e) => {
    if (e.metrics) Object.keys(e.metrics).forEach((k) => {
      if (typeof e.metrics![k] !== 'object') allMetricKeys.add(k);
    });
  });

  return (
    <div>
      <div className="mb-6">
        <Link to="/experiments" className="text-sm text-indigo-600 hover:underline">&larr; Back to experiments</Link>
        <h2 className="text-xl font-semibold text-gray-900 mt-2">
          Compare Experiments ({runIds.length} runs)
        </h2>
      </div>

      {/* Metrics table */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden mb-8">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100 bg-gray-50/50">
              <th className="text-left px-4 py-3 font-medium text-gray-500">Metric</th>
              {entries.map((e, i) => (
                <th key={e.id} className="text-right px-4 py-3 font-medium" style={{ color: COLORS[i % COLORS.length] }}>
                  {e.id.split('/')[0]}
                  {e.policy && <span className="text-xs text-gray-400 ml-1">({e.policy})</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from(allMetricKeys).map((key) => (
              <tr key={key} className="border-b border-gray-50">
                <td className="px-4 py-2.5 text-gray-600">{METRIC_LABELS[key] || key}</td>
                {entries.map((e) => {
                  const val = e.metrics?.[key];
                  return (
                    <td key={e.id} className="px-4 py-2.5 text-right text-gray-800 font-mono">
                      {typeof val === 'number' ? val.toFixed(4) : String(val ?? '—')}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Overlay charts */}
      <div className="grid md:grid-cols-2 gap-6">
        <div ref={rewardRef} className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-700">Cumulative Reward</h3>
            <ExportButton targetRef={rewardRef} filename="compare_reward" />
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={rewardOverlay}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="step" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              {runIds.map((id, i) => (
                <Line key={id} type="monotone" dataKey={id} name={id.split('/')[0]} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div ref={posRef} className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-700">Position Trajectory</h3>
            <ExportButton targetRef={posRef} filename="compare_position" />
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={positionOverlay}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="step" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              {runIds.map((id, i) => (
                <Line key={id} type="stepAfter" dataKey={id} name={id.split('/')[0]} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
