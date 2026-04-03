import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { api } from '../lib/api';
import type { ExperimentRun } from '../lib/api';

function policyColor(p?: string) {
  switch (p) {
    case 'greedy':        return 'bg-emerald-100 text-emerald-700';
    case 'random':        return 'bg-amber-100 text-amber-700';
    case 'llm_memory':    return 'bg-indigo-100 text-indigo-700';
    case 'llm_no_memory': return 'bg-purple-100 text-purple-700';
    default:              return 'bg-gray-100 text-gray-600';
  }
}

type SortKey = 'experiment_name' | 'policy_type' | 'total_reward' | 'coverage' | 'episode_length' | 'timestamp';
type SortDir = 'asc' | 'desc';

const PAGE_SIZES = [10, 25, 50, 0] as const; // 0 = All

function SortArrow({ active, dir }: { active: boolean; dir: SortDir }) {
  if (!active) return null;
  return <span className="ml-1 text-indigo-500">{dir === 'asc' ? '\u2191' : '\u2193'}</span>;
}

export default function ExperimentList() {
  const navigate = useNavigate();
  const location = useLocation();
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<SortKey>('timestamp');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [pageSize, setPageSize] = useState<number>(10);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api.listExperiments()
      .then(setRuns)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [location.key]);

  function toggleSort(key: SortKey) {
    if (sortBy === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortBy(key);
      setSortDir(key === 'timestamp' ? 'desc' : 'asc');
    }
  }

  const sorted = useMemo(() => {
    const copy = [...runs];
    copy.sort((a, b) => {
      const av = a[sortBy] ?? '';
      const bv = b[sortBy] ?? '';
      if (typeof av === 'number' && typeof bv === 'number') {
        return sortDir === 'asc' ? av - bv : bv - av;
      }
      const cmp = String(av).localeCompare(String(bv));
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return pageSize > 0 ? copy.slice(0, pageSize) : copy;
  }, [runs, sortBy, sortDir, pageSize]);

  function toggleSelect(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function compareSelected() {
    const ids = Array.from(selected).join(',');
    navigate(`/compare?runs=${ids}`);
  }

  if (loading) return (
    <div className="space-y-3">
      <div className="h-8 bg-gray-200 rounded animate-pulse w-48 mb-6" />
      <div className="bg-white rounded-xl border border-gray-200 p-1">
        {[1, 2, 3, 4, 5].map(i => (
          <div key={i} className="flex gap-4 px-4 py-3 border-b border-gray-50">
            <div className="w-4 h-4 bg-gray-200 rounded animate-pulse" />
            <div className="h-4 bg-gray-200 rounded animate-pulse flex-1" />
            <div className="h-4 bg-gray-200 rounded animate-pulse w-16" />
            <div className="h-4 bg-gray-200 rounded animate-pulse w-16" />
          </div>
        ))}
      </div>
    </div>
  );

  if (error) return (
    <div className="text-center py-20">
      <p className="text-red-500 mb-2">Failed to load experiments</p>
      <p className="text-sm text-gray-400">{error}</p>
    </div>
  );

  if (!runs.length) return (
    <div className="text-center py-20">
      <p className="text-gray-500 mb-4">No experiments yet.</p>
      <Link to="/new" className="text-indigo-600 hover:underline text-sm font-medium">
        Run your first experiment &rarr;
      </Link>
    </div>
  );

  const columns: { key: SortKey; label: string; align: string }[] = [
    { key: 'experiment_name', label: 'Name', align: 'text-left' },
    { key: 'policy_type', label: 'Policy', align: 'text-left' },
    { key: 'total_reward', label: 'Reward', align: 'text-right' },
    { key: 'coverage', label: 'Coverage', align: 'text-right' },
    { key: 'episode_length', label: 'Steps', align: 'text-right' },
    { key: 'timestamp', label: 'Timestamp', align: 'text-left' },
  ];

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900">Experiment Runs</h2>
        <div className="flex items-center gap-3">
          {/* Page size selector */}
          <select
            value={pageSize}
            onChange={(e) => setPageSize(Number(e.target.value))}
            className="text-xs border border-gray-200 rounded-lg px-2 py-1.5 text-gray-600 bg-white"
          >
            {PAGE_SIZES.map((s) => (
              <option key={s} value={s}>{s === 0 ? 'All' : `Show ${s}`}</option>
            ))}
          </select>
          {selected.size >= 2 && (
            <button
              onClick={compareSelected}
              className="px-4 py-2 text-sm font-medium text-indigo-600 border border-indigo-200 rounded-lg hover:bg-indigo-50 transition-colors"
            >
              Compare ({selected.size})
            </button>
          )}
          <Link
            to="/new"
            className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition-colors"
          >
            + New Run
          </Link>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100 bg-gray-50/50">
              <th className="w-10 px-4 py-3" />
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={() => toggleSort(col.key)}
                  className={`${col.align} px-4 py-3 font-medium text-gray-500 cursor-pointer hover:text-gray-700 select-none`}
                >
                  {col.label}
                  <SortArrow active={sortBy === col.key} dir={sortDir} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((run) => (
              <tr
                key={run.id}
                className="border-b border-gray-50 hover:bg-gray-50 cursor-pointer transition-colors"
                onClick={() => navigate(`/experiment/${run.tag}/${run.timestamp}`)}
              >
                <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={selected.has(run.id)}
                    onChange={() => toggleSelect(run.id)}
                    className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                  />
                </td>
                <td className="px-4 py-3 font-medium text-gray-900 truncate max-w-[200px]">
                  {run.experiment_name || run.tag}
                </td>
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${policyColor(run.policy_type)}`}>
                    {run.policy_type || '?'}
                  </span>
                </td>
                <td className="px-4 py-3 text-right text-gray-700 font-mono">
                  {run.total_reward != null ? run.total_reward.toFixed(1) : '\u2014'}
                </td>
                <td className="px-4 py-3 text-right text-gray-700">
                  {run.coverage ?? '\u2014'}
                </td>
                <td className="px-4 py-3 text-right text-gray-700">
                  {run.episode_length || '\u2014'}
                </td>
                <td className="px-4 py-3 text-gray-400 font-mono text-xs">
                  {run.timestamp}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {pageSize > 0 && runs.length > pageSize && (
        <p className="text-xs text-gray-400 mt-2 text-center">
          Showing {sorted.length} of {runs.length} runs
        </p>
      )}
    </div>
  );
}
