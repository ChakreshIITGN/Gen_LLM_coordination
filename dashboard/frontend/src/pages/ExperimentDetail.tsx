import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, PieChart, Pie, Cell,
} from 'recharts';
import { api } from '../lib/api';
import type { TrajectorySummary } from '../lib/api';

const COLORS = ['#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6'];

export default function ExperimentDetail() {
  const { tag, timestamp } = useParams<{ tag: string; timestamp: string }>();
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [traj, setTraj] = useState<TrajectorySummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!tag || !timestamp) return;
    Promise.all([
      api.getMetrics(tag, timestamp),
      api.getConfig(tag, timestamp),
      api.getTrajectorySummary(tag, timestamp),
    ])
      .then(([m, c, t]) => { setMetrics(m); setConfig(c); setTraj(t); })
      .finally(() => setLoading(false));
  }, [tag, timestamp]);

  if (loading) return (
    <div className="space-y-6">
      <div className="h-6 bg-gray-200 rounded animate-pulse w-64" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map(i => <div key={i} className="h-20 bg-gray-200 rounded-xl animate-pulse" />)}
      </div>
      <div className="grid md:grid-cols-2 gap-6">
        {[1, 2].map(i => <div key={i} className="h-64 bg-gray-200 rounded-xl animate-pulse" />)}
      </div>
    </div>
  );
  if (!metrics || !traj) return (
    <div className="text-center py-20">
      <p className="text-red-500 mb-2">Failed to load experiment data</p>
      <Link to="/experiments" className="text-indigo-600 hover:underline text-sm">&larr; Back to experiments</Link>
    </div>
  );

  // Derive chart data
  const rewardData = traj.steps.map((s, i) => ({
    step: s,
    reward: traj.rewards[i],
    cumulative: traj.cumulative_rewards[i],
  }));

  const positionData = traj.steps.map((s, i) => ({
    step: s,
    position: traj.positions[i],
  }));

  // Action distribution for pie chart
  const actionCounts: Record<string, number> = {};
  traj.actions.forEach((a) => { actionCounts[a] = (actionCounts[a] || 0) + 1; });
  const actionPieData = Object.entries(actionCounts).map(([name, value]) => ({ name, value }));

  // Said vs did — count reasons that mention the actual action taken
  const saidVsDid = traj.actions.reduce(
    (acc, action, i) => {
      const reason = (traj.reasons[i] || '').toLowerCase();
      // simple heuristic: did the agent's reason mention its action?
      const mentioned = reason.includes(action.toLowerCase());
      acc.total++;
      if (mentioned) acc.consistent++;
      return acc;
    },
    { total: 0, consistent: 0 },
  );

  const policyType = (config as Record<string, Record<string, string>>)?.policy?.type ?? '?';

  return (
    <div>
      {/* Header */}
      <div className="mb-6">
        <Link to="/experiments" className="text-sm text-indigo-600 hover:underline">&larr; All experiments</Link>
        <h2 className="text-xl font-semibold text-gray-900 mt-2">
          {(config as Record<string, string>)?.experiment_name || tag}
        </h2>
        <p className="text-sm text-gray-400 font-mono">{tag}/{timestamp}</p>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <MetricCard label="Total Reward" value={fmt(metrics.total_reward)} />
        <MetricCard label="Coverage" value={`${metrics.coverage_unique_positions ?? '—'} positions`} />
        <MetricCard label="Policy" value={policyType} />
        <MetricCard label="Said = Did" value={
          saidVsDid.total > 0
            ? `${((saidVsDid.consistent / saidVsDid.total) * 100).toFixed(0)}%`
            : 'n/a'
        } />
      </div>

      {/* Charts */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {/* Cumulative Reward */}
        <ChartCard title="Cumulative Reward">
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={rewardData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="step" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="cumulative" stroke="#6366f1" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Per-step reward */}
        <ChartCard title="Reward per Step">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={rewardData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="step" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="reward" fill="#a5b4fc" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Position trajectory */}
        <ChartCard title="Agent Position">
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={positionData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="step" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="stepAfter" dataKey="position" stroke="#10b981" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Action distribution */}
        <ChartCard title="Action Distribution">
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie
                data={actionPieData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={80}
                label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}
              >
                {actionPieData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Raw config JSON */}
      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Config</h3>
        <pre className="text-xs text-gray-600 overflow-x-auto bg-gray-50 p-4 rounded-lg">
          {JSON.stringify(config, null, 2)}
        </pre>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <p className="text-xs text-gray-400 mb-1">{label}</p>
      <p className="text-lg font-semibold text-gray-900">{value}</p>
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">{title}</h3>
      {children}
    </div>
  );
}

function fmt(v: unknown): string {
  if (typeof v === 'number') return v.toFixed(2);
  return String(v ?? '—');
}
