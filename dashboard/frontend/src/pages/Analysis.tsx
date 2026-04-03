import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
  LineChart, Line,
} from 'recharts';
import { toPng } from 'html-to-image';
import { api } from '../lib/api';
import type { AnalysisResult, TrajectorySummary } from '../lib/api';

const COLORS: Record<string, string> = {
  greedy: '#10b981',
  random: '#f59e0b',
  llm_memory: '#6366f1',
  llm_no_memory: '#8b5cf6',
};
const fallbackColor = (p: string) => COLORS[p] || '#6b7280';

const METRIC_LABELS: Record<string, string> = {
  total_reward: 'Total Reward',
  coverage_unique_positions: 'Coverage',
  net_displacement: 'Net Displacement',
  said_vs_did_rate: 'Said = Did',
  reward_first_half_mean: 'Reward (1st half)',
  reward_second_half_mean: 'Reward (2nd half)',
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

export default function Analysis() {
  const [data, setData] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [trajectories, setTrajectories] = useState<Record<string, TrajectorySummary[]>>({});
  const [trajLoading, setTrajLoading] = useState(false);

  const barRef = useRef<HTMLDivElement>(null);
  const lineRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.getAnalysis()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // Load trajectories for overlay chart
  useEffect(() => {
    if (!data) return;
    setTrajLoading(true);
    const promises: Promise<void>[] = [];
    const result: Record<string, TrajectorySummary[]> = {};

    for (const [policy, info] of Object.entries(data.policies)) {
      result[policy] = [];
      for (const run of info.runs) {
        promises.push(
          api.getTrajectorySummary(run.tag, run.timestamp)
            .then((t) => { result[policy].push(t); })
            .catch(() => {})
        );
      }
    }
    Promise.all(promises).then(() => {
      setTrajectories(result);
      setTrajLoading(false);
    });
  }, [data]);

  if (loading) return (
    <div className="space-y-4">
      <div className="h-8 bg-gray-200 rounded animate-pulse w-48" />
      <div className="h-64 bg-gray-200 rounded-xl animate-pulse" />
    </div>
  );

  if (error) return (
    <div className="text-center py-20">
      <p className="text-red-500 mb-2">Failed to load analysis</p>
      <p className="text-sm text-gray-400">{error}</p>
    </div>
  );

  if (!data || Object.keys(data.policies).length === 0) return (
    <div className="text-center py-20">
      <p className="text-gray-500 mb-4">No experiment data yet.</p>
      <Link to="/new" className="text-indigo-600 hover:underline text-sm">
        Run your first experiment &rarr;
      </Link>
    </div>
  );

  const policies = Object.keys(data.policies);

  // Build bar chart data: one bar group per metric, one bar per policy
  const metricKeys = new Set<string>();
  for (const info of Object.values(data.policies)) {
    for (const k of Object.keys(info.avg_metrics)) {
      if (METRIC_LABELS[k]) metricKeys.add(k);
    }
  }

  const barData = Array.from(metricKeys).map((key) => {
    const row: Record<string, string | number> = { metric: METRIC_LABELS[key] || key };
    for (const [policy, info] of Object.entries(data.policies)) {
      row[policy] = info.avg_metrics[key] ?? 0;
    }
    return row;
  });

  // Build average cumulative reward overlay
  const avgCumReward: Record<string, number>[] = [];
  if (!trajLoading) {
    const maxSteps = Math.max(
      ...Object.values(trajectories).flatMap((ts) => ts.map((t) => t.cumulative_rewards.length)),
      0
    );
    for (let i = 0; i < maxSteps; i++) {
      const point: Record<string, number> = { step: i };
      for (const [policy, trajs] of Object.entries(trajectories)) {
        const vals = trajs.map((t) => t.cumulative_rewards[i]).filter((v) => v !== undefined);
        if (vals.length > 0) {
          point[policy] = Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 100) / 100;
        }
      }
      avgCumReward.push(point);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Policy Analysis</h2>
          <p className="text-sm text-gray-500 mt-1">
            Average metrics across all runs, grouped by policy type.
          </p>
        </div>
        <Link
          to="/new"
          className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700"
        >
          + New Run
        </Link>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
        {policies.map((p) => (
          <div key={p} className="bg-white rounded-xl border border-gray-200 p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: fallbackColor(p) }} />
              <span className="text-sm font-semibold text-gray-800">{p}</span>
            </div>
            <p className="text-xs text-gray-400">{data.policies[p].count} run{data.policies[p].count !== 1 ? 's' : ''}</p>
            <p className="text-lg font-bold text-gray-900 mt-1">
              {data.policies[p].avg_metrics.total_reward?.toFixed(1) ?? '—'}
            </p>
            <p className="text-[10px] text-gray-400">avg total reward</p>
          </div>
        ))}
      </div>

      {/* Average metrics bar chart */}
      <div ref={barRef} className="bg-white rounded-xl border border-gray-200 p-5 mb-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-700">Average Metrics by Policy</h3>
          <ExportButton targetRef={barRef} filename="policy_metrics" />
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={barData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis dataKey="metric" type="category" tick={{ fontSize: 11 }} width={120} />
            <Tooltip />
            <Legend />
            {policies.map((p) => (
              <Bar key={p} dataKey={p} fill={fallbackColor(p)} radius={[0, 4, 4, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Average cumulative reward overlay */}
      {avgCumReward.length > 0 && (
        <div ref={lineRef} className="bg-white rounded-xl border border-gray-200 p-5 mb-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-700">
              Average Cumulative Reward
              <span className="text-xs text-gray-400 font-normal ml-2">(mean across all runs per policy)</span>
            </h3>
            <ExportButton targetRef={lineRef} filename="avg_cumulative_reward" />
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={avgCumReward}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="step" tick={{ fontSize: 11 }} label={{ value: 'Step', position: 'insideBottom', offset: -2, fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              {policies.map((p) => (
                <Line
                  key={p}
                  type="monotone"
                  dataKey={p}
                  stroke={fallbackColor(p)}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Metrics table */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100 bg-gray-50/50">
              <th className="text-left px-4 py-3 font-medium text-gray-500">Metric</th>
              {policies.map((p) => (
                <th key={p} className="text-right px-4 py-3 font-medium" style={{ color: fallbackColor(p) }}>
                  {p}
                  <span className="text-xs text-gray-400 ml-1">({data.policies[p].count})</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from(metricKeys).map((key) => (
              <tr key={key} className="border-b border-gray-50">
                <td className="px-4 py-2.5 text-gray-600">{METRIC_LABELS[key] || key}</td>
                {policies.map((p) => {
                  const val = data.policies[p].avg_metrics[key];
                  return (
                    <td key={p} className="px-4 py-2.5 text-right text-gray-800 font-mono">
                      {typeof val === 'number' ? val.toFixed(4) : '—'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Individual runs by policy */}
      <h3 className="text-sm font-semibold text-gray-700 mb-3">Individual Runs</h3>
      <div className="space-y-3">
        {policies.map((p) => (
          <details key={p} className="bg-white rounded-xl border border-gray-200">
            <summary className="px-5 py-3 cursor-pointer text-sm font-medium text-gray-800 flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: fallbackColor(p) }} />
              {p} ({data.policies[p].count} runs)
            </summary>
            <div className="px-5 pb-4">
              {data.policies[p].runs.map((r) => (
                <Link
                  key={r.run_id}
                  to={`/experiment/${r.tag}/${r.timestamp}`}
                  className="block text-xs text-indigo-600 hover:underline py-1"
                >
                  {r.run_id}
                </Link>
              ))}
            </div>
          </details>
        ))}
      </div>
    </div>
  );
}
