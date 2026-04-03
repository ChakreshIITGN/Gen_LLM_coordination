import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';

export default function WorldExplainer() {
  const navigate = useNavigate();
  const [peak, setPeak] = useState(20);
  const [decay, setDecay] = useState(0.1);
  const [initial, setInitial] = useState(10);
  const [length, setLength] = useState(50);
  const [startPos, setStartPos] = useState(30);

  // Compute reward landscape data points
  const data = useMemo(() => {
    return Array.from({ length: length + 1 }, (_, x) => ({
      position: x,
      reward: x < peak ? 0 : initial * Math.exp(-decay * (x - peak)),
    }));
  }, [peak, decay, initial, length]);

  const startReward = startPos < peak ? 0 : initial * Math.exp(-decay * (startPos - peak));

  function useSettings() {
    navigate(`/new?peak=${peak}&decay=${decay}&initial=${initial}&length=${length}`);
  }

  return (
    <div>
      <h2 className="text-xl font-semibold text-gray-900 mb-1">E. coli Chemotaxis World</h2>
      <p className="text-sm text-gray-500 mb-8">
        Interactive visualization of the 1D reward landscape. Adjust parameters to explore different configurations.
      </p>

      {/* Chart + Controls */}
      <div className="grid lg:grid-cols-3 gap-6 mb-8">
        {/* Chart — spans 2 cols */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Reward Landscape</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={data}>
              <defs>
                <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="position" tick={{ fontSize: 11 }} label={{ value: 'Position', position: 'insideBottom', offset: -2, fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} label={{ value: 'Reward', angle: -90, position: 'insideLeft', fontSize: 11 }} />
              <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
              <Area type="monotone" dataKey="reward" stroke="#6366f1" strokeWidth={2} fill="url(#rewardGradient)" />
              <ReferenceLine x={peak} stroke="#10b981" strokeDasharray="4 4" label={{ value: `Peak (${peak})`, position: 'top', fontSize: 10, fill: '#10b981' }} />
              <ReferenceLine x={startPos} stroke="#ef4444" strokeDasharray="4 4" label={{ value: `Start (${startPos})`, position: 'top', fontSize: 10, fill: '#ef4444' }} />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex gap-6 mt-2 text-xs text-gray-400">
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-emerald-500 inline-block" /> Peak</span>
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-red-500 inline-block" /> Agent start</span>
            <span>Reward at start: <span className="font-mono text-gray-600">{startReward.toFixed(4)}</span></span>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-5">
          <h3 className="text-sm font-semibold text-gray-700">Parameters</h3>

          <Slider label="Peak Position" value={peak} min={0} max={length} step={1} onChange={setPeak} />
          <Slider label="Decay Rate" value={decay} min={0.01} max={0.5} step={0.01} onChange={setDecay} />
          <Slider label="Initial Value" value={initial} min={1} max={50} step={1} onChange={setInitial} />
          <Slider label="World Length" value={length} min={20} max={200} step={10} onChange={setLength} />
          <Slider label="Start Position" value={startPos} min={0} max={length} step={1} onChange={setStartPos} />

          <button
            onClick={useSettings}
            className="w-full px-4 py-2.5 text-sm text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 font-medium mt-4"
          >
            Use these settings &rarr;
          </button>
        </div>
      </div>

      {/* Explanation cards */}
      <div className="grid md:grid-cols-3 gap-4">
        <ExplainCard
          title="What the Agent Sees"
          content="Each step the agent observes: its position on the line, the reward at its current position, and (when gradient is enabled) the reward at adjacent positions left and right. This local gradient mirrors biological chemoreceptor sensing — the agent perceives a local signal, not the global landscape."
        />
        <ExplainCard
          title="Available Actions"
          content={`The agent outputs JSON: {"type": "Left"|"Right"|"Wait", "reason": "..."}. Left moves -1, Right moves +1, Wait stays. Boundaries are blocking — the agent cannot leave [0, L]. The reason field captures the agent's stated rationale, which we compare against its actual behavior (said-vs-did analysis).`}
        />
        <ExplainCard
          title="Reward Landscape"
          content="An exponential decay function centered at the peak position. Reward is zero to the left of the peak and decays exponentially to the right. The agent starts away from the peak and must discover the gradient through observation. The key question: can the LLM learn to follow the gradient toward the peak, mimicking biological chemotaxis?"
        />
      </div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-500">{label}</span>
        <span className="font-mono text-gray-700">{value}</span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
      />
    </div>
  );
}

function ExplainCard({ title, content }: { title: string; content: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <h3 className="text-sm font-semibold text-gray-800 mb-2">{title}</h3>
      <p className="text-xs text-gray-600 leading-relaxed">{content}</p>
    </div>
  );
}
