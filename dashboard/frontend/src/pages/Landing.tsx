import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';

// ── Animated demo cards ──────────────────────────────────────────────

const DEMO_CARDS = [
  {
    title: 'Prompt Setup',
    render: (progress: number) => <PromptDemo progress={progress} />,
  },
  {
    title: 'Agent Exploring',
    render: (progress: number) => <ExploreDemo progress={progress} />,
  },
  {
    title: 'Results',
    render: (progress: number) => <ResultsDemo progress={progress} />,
  },
];

function PromptDemo({ progress }: { progress: number }) {
  const prompt = 'You are an agent on a 1D line. Your goal is to maximize total reward collected over time.';
  const obs = '{"position": 30, "reward": 3.68, "reward_left": 4.07, "reward_right": 3.33}';
  const charCount = Math.floor(progress * prompt.length);
  const obsVisible = progress > 0.6;
  const obsChars = obsVisible ? Math.floor(((progress - 0.6) / 0.4) * obs.length) : 0;

  return (
    <div className="font-mono text-[11px] leading-relaxed space-y-2">
      <p className="text-gray-700">{prompt.slice(0, charCount)}<span className="animate-pulse text-indigo-500">|</span></p>
      {obsVisible && (
        <p className="text-indigo-600 mt-2">{obs.slice(0, obsChars)}</p>
      )}
    </div>
  );
}

function ExploreDemo({ progress }: { progress: number }) {
  const worldLen = 40;
  const peak = 15;
  const positions = [30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 15, 15];
  const idx = Math.min(Math.floor(progress * positions.length), positions.length - 1);
  const pos = positions[idx];
  const reward = pos < peak ? 0 : 10 * Math.exp(-0.1 * (pos - peak));

  return (
    <div className="space-y-3">
      {/* World line */}
      <div className="relative h-3 bg-gray-100 rounded-full overflow-hidden">
        {/* Reward gradient */}
        <div
          className="absolute inset-y-0 bg-gradient-to-r from-indigo-500/40 to-transparent"
          style={{ left: `${(peak / worldLen) * 100}%`, width: `${((worldLen - peak) / worldLen) * 100}%` }}
        />
        {/* Peak marker */}
        <div className="absolute top-0 bottom-0 w-0.5 bg-emerald-400" style={{ left: `${(peak / worldLen) * 100}%` }} />
        {/* Agent dot */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-indigo-400 shadow-lg shadow-indigo-500/50 transition-all duration-300"
          style={{ left: `${(pos / worldLen) * 100}%` }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-gray-500">
        <span>0</span>
        <span className="text-emerald-400">Peak ({peak})</span>
        <span>{worldLen}</span>
      </div>
      <div className="flex gap-4 text-[11px]">
        <span className="text-gray-400">Position: <span className="text-gray-700 font-medium">{pos}</span></span>
        <span className="text-gray-400">Reward: <span className="text-indigo-600 font-medium">{reward.toFixed(2)}</span></span>
        <span className="text-gray-400">Step: <span className="text-gray-700 font-medium">{idx + 1}/{positions.length}</span></span>
      </div>
    </div>
  );
}

function ResultsDemo({ progress }: { progress: number }) {
  const totalSteps = 12;
  const visibleSteps = Math.max(1, Math.floor(progress * totalSteps));
  const rewards = [3.68, 4.07, 4.49, 4.97, 5.49, 6.07, 6.70, 7.41, 8.19, 9.05, 10.0, 10.0];
  const cumulative = rewards.reduce<number[]>((acc, r, i) => {
    acc.push(i === 0 ? r : acc[i - 1] + r);
    return acc;
  }, []);

  const maxCum = cumulative[cumulative.length - 1];
  const showMetrics = progress > 0.75;

  return (
    <div className="space-y-2">
      {/* Mini chart */}
      <div className="flex items-end gap-0.5 h-12">
        {cumulative.slice(0, visibleSteps).map((v, i) => (
          <div
            key={i}
            className="flex-1 bg-indigo-500/70 rounded-t-sm transition-all duration-200"
            style={{ height: `${(v / maxCum) * 100}%` }}
          />
        ))}
      </div>
      {showMetrics && (
        <div className="flex gap-4 text-[11px] animate-in fade-in">
          <span className="text-emerald-600">Total: {cumulative[visibleSteps - 1].toFixed(1)}</span>
          <span className="text-indigo-600">Coverage: {visibleSteps} pos</span>
          <span className="text-amber-600">Peak found</span>
        </div>
      )}
    </div>
  );
}

function DemoCarousel() {
  const [active, setActive] = useState(0);
  const [progress, setProgress] = useState(0);
  const startTime = useRef(Date.now());
  const DURATION = 4000;

  useEffect(() => {
    let raf: number;
    const tick = () => {
      const elapsed = Date.now() - startTime.current;
      const p = Math.min(elapsed / DURATION, 1);
      setProgress(p);
      if (p >= 1) {
        setActive((prev) => (prev + 1) % DEMO_CARDS.length);
        startTime.current = Date.now();
        setProgress(0);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="grid md:grid-cols-3 gap-4">
      {DEMO_CARDS.map((card, i) => (
        <div
          key={card.title}
          className={`relative bg-white rounded-xl border p-5 transition-all duration-300 ${
            i === active ? 'border-indigo-300 shadow-lg shadow-indigo-100' : 'border-gray-200'
          }`}
        >
          <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-3">{card.title}</p>
          <div className="min-h-[80px]">
            {card.render(i === active ? progress : i < active ? 1 : 0)}
          </div>
          {/* Progress bar */}
          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gray-100 rounded-b-xl overflow-hidden">
            <div
              className="h-full bg-indigo-500 transition-all duration-100"
              style={{ width: i === active ? `${progress * 100}%` : i < active ? '100%' : '0%' }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Main landing ─────────────────────────────────────────────────────

export default function Landing() {
  return (
    <div className="max-w-4xl mx-auto">
      {/* Top nav anchors */}
      <nav className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-gray-100 -mx-6 lg:-mx-8 px-6 lg:px-8 py-3 mb-8 flex gap-6">
        {['About', 'Method', 'Team'].map((s) => (
          <a
            key={s}
            href={`#${s.toLowerCase()}`}
            className="text-sm text-gray-500 hover:text-indigo-600 font-medium transition-colors"
          >
            {s}
          </a>
        ))}
      </nav>

      {/* Hero — asymmetric, card positioned left */}
      <section className="mb-12 flex">
        <div className="w-2/3 py-14 px-10 rounded-2xl bg-gradient-to-br from-indigo-50 via-white to-amber-50">
          <h1 className="text-5xl font-bold tracking-tight text-gray-900">
            LLM-ABM Lab
          </h1>
          <p className="mt-4 text-lg text-gray-500">
            Agent-Based Modeling with Large Language Models
          </p>
          <p className="mt-1 text-sm text-indigo-600 font-medium">
            Behavioral strategy inference through simulation
          </p>
          <div className="mt-10 flex gap-4">
            <Link
              to="/world"
              className="px-6 py-2.5 rounded-lg text-sm font-medium bg-indigo-600 hover:bg-indigo-700 text-white transition-colors"
            >
              Explore World &rarr;
            </Link>
            <Link
              to="/new"
              className="px-6 py-2.5 rounded-lg text-sm font-medium border border-gray-200 text-gray-600 hover:bg-gray-50 transition-colors"
            >
              Run Experiment &rarr;
            </Link>
          </div>
        </div>
      </section>

      {/* Animated demo cards */}
      <section className="mb-16">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-widest mb-4">How It Works</h3>
        <DemoCarousel />
      </section>

      {/* About */}
      <section id="about" className="mb-16 scroll-mt-16">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">About</h2>
        <div className="prose prose-gray max-w-none text-sm text-gray-600 leading-relaxed space-y-4">
          <p>
            LLM-ABM Lab is a research platform for studying emergent behavior in large language model agents.
            We treat LLMs as black-box policies &mdash; mapping observations to actions &mdash; and study their
            behavior through controlled simulation, not self-report.
          </p>
          <p>
            The platform supports configurable 1D worlds with tunable reward landscapes, multiple policy types
            (LLM with/without memory, random, greedy baselines), and rigorous experimental controls including
            gradient ablation and seed-based reproducibility.
          </p>
          <p>
            Designed for AI safety researchers, behavioral scientists, and anyone interested in understanding
            how language models make sequential decisions under uncertainty.
          </p>
        </div>
      </section>

      {/* Method */}
      <section id="method" className="mb-16 scroll-mt-16">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Method</h2>
        <div className="text-sm text-gray-600 leading-relaxed space-y-4 mb-6">
          <p>
            Our core approach is <strong>LLM-as-Policy</strong>: we treat the language model as a stateless
            (or memory-augmented) policy function. At each timestep, the agent receives an observation and
            must output a structured action. Strategy is inferred from behavior patterns, never from the
            model's stated reasoning.
          </p>
          <p className="text-gray-500 italic">
            Key insight: most LLM agent research trusts chain-of-thought explanations. We don't.
            We compare what the agent says with what it actually does (said-vs-did analysis).
          </p>
        </div>
        {/* Pipeline diagram */}
        <div className="flex items-center justify-center gap-3 py-6">
          {['Observation', 'LLM', 'Action', 'World', 'Reward'].map((step, i, arr) => (
            <div key={step} className="flex items-center gap-3">
              <div className="px-4 py-2 rounded-lg bg-white border border-gray-200 text-sm font-medium text-gray-700">
                {step}
              </div>
              {i < arr.length - 1 && (
                <span className="text-gray-300">&rarr;</span>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Team */}
      <section id="team" className="mb-16 scroll-mt-16">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Team</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {[
            { name: 'Dr. Vasudha', role: 'Principal Investigator' },
            { name: 'Dr. Jayesh', role: 'Co-Investigator' },
            { name: 'Dr. Chakresh', role: 'Lead Engineer & Researcher' },
          ].map(({ name, role }) => (
            <div key={name} className="bg-white rounded-xl border border-gray-200 p-5">
              <p className="font-semibold text-gray-900 text-sm">{name}</p>
              <p className="text-xs text-gray-500 mt-1">{role}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
