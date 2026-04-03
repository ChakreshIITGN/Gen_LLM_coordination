import { useEffect, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { api } from '../lib/api';
import type { ConfigTemplate, PromptPreview, StepData } from '../lib/api';

type Step = 'template' | 'edit' | 'review' | 'running';

const STEPS = ['Template', 'Configure', 'Review', 'Launch'] as const;

function StepIndicator({ current }: { current: Step }) {
  const idx = { template: 0, edit: 1, review: 2, running: 3 }[current];
  return (
    <div className="flex items-center gap-1 mb-8">
      {STEPS.map((label, i) => (
        <div key={label} className="flex items-center">
          <div className={`flex items-center justify-center w-7 h-7 rounded-full text-xs font-semibold transition-colors ${
            i <= idx ? 'bg-indigo-600 text-white' : 'bg-gray-200 text-gray-400'
          }`}>
            {i + 1}
          </div>
          <span className={`ml-1.5 text-xs font-medium ${
            i <= idx ? 'text-gray-900' : 'text-gray-400'
          }`}>
            {label}
          </span>
          {i < STEPS.length - 1 && (
            <div className={`w-8 h-px mx-2 ${i < idx ? 'bg-indigo-400' : 'bg-gray-200'}`} />
          )}
        </div>
      ))}
    </div>
  );
}

export default function NewExperiment() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  // Restore running job from sessionStorage if user navigated away and came back
  const saved = (() => {
    try {
      const raw = sessionStorage.getItem('running-job');
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  })();

  const [step, setStep] = useState<Step>(saved ? 'running' : 'template');
  const [templates, setTemplates] = useState<ConfigTemplate[]>([]);
  const [config, setConfig] = useState('{}');
  const [validationErr, setValidationErr] = useState<string | null>(null);
  const [preview, setPreview] = useState<PromptPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(saved?.jobId ?? null);
  const [jobStatus, setJobStatus] = useState<string>(saved ? 'running' : 'running');
  const [jobOutput, setJobOutput] = useState('');
  const [numRepeats, setNumRepeats] = useState(1);

  // Live chat state
  const [liveSteps, setLiveSteps] = useState<StepData[]>([]);
  const [showChat, setShowChat] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Batch state
  const [batchJobIds, setBatchJobIds] = useState<string[]>(saved?.batchJobIds ?? []);
  const [batchCompleted, setBatchCompleted] = useState(0);

  useEffect(() => {
    api.getTemplates().then((t) => {
      setTemplates(t);
      const peak = searchParams.get('peak');
      if (peak && t.length > 0) {
        const base = JSON.parse(JSON.stringify(t[0].config));
        base.world = {
          ...base.world,
          reward: {
            ...base.world?.reward,
            peak_position: parseFloat(searchParams.get('peak') || '20'),
            decay_rate: parseFloat(searchParams.get('decay') || '0.1'),
            initial_value: parseFloat(searchParams.get('initial') || '10'),
          },
          length: parseInt(searchParams.get('length') || '50'),
        };
        setConfig(JSON.stringify(base, null, 2));
        setStep('edit');
      }
    });
  }, []);

  // Poll live steps — also works when returning to page after navigation
  useEffect(() => {
    if (!jobId || jobStatus !== 'running') return;
    const interval = setInterval(async () => {
      try {
        const data = await api.getJobSteps(jobId);
        setLiveSteps(data.steps);
        if (data.status !== 'running') {
          setJobStatus(data.status);
          try { sessionStorage.removeItem('running-job'); } catch {}
        }
      } catch {
        try {
          const s = await api.getJobStatus(jobId);
          setJobStatus(s.status);
          setJobOutput(s.output);
          if (s.status !== 'running') {
            try { sessionStorage.removeItem('running-job'); } catch {}
          }
        } catch { /* job not found, clear */
          try { sessionStorage.removeItem('running-job'); } catch {}
        }
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [jobId, jobStatus]);

  // Poll batch status
  useEffect(() => {
    if (batchJobIds.length <= 1) return;
    const interval = setInterval(async () => {
      let done = 0;
      for (const jid of batchJobIds) {
        try {
          const s = await api.getJobStatus(jid);
          if (s.status !== 'running') done++;
        } catch { /* ignore */ }
      }
      setBatchCompleted(done);
      if (done >= batchJobIds.length) {
        setJobStatus('completed');
        clearInterval(interval);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [batchJobIds]);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [liveSteps]);

  function selectTemplate(t: ConfigTemplate) {
    setConfig(JSON.stringify(t.config, null, 2));
    setStep('edit');
  }

  async function validate(): Promise<boolean> {
    try {
      const parsed = JSON.parse(config);
      const result = await api.validateConfig(parsed);
      if (result.valid) {
        setValidationErr(null);
        return true;
      }
      setValidationErr(result.errors || 'Invalid config');
      return false;
    } catch (e) {
      setValidationErr(`Invalid JSON: ${(e as Error).message}`);
      return false;
    }
  }

  async function goToReview() {
    const ok = await validate();
    if (!ok) return;
    setPreviewLoading(true);
    try {
      const parsed = JSON.parse(config);
      const p = await api.previewPrompt(parsed);
      setPreview(p);
      setStep('review');
    } catch (e) {
      setValidationErr(`Preview failed: ${(e as Error).message}`);
    } finally {
      setPreviewLoading(false);
    }
  }

  async function launch() {
    const parsed = JSON.parse(config);
    setLiveSteps([]);
    setShowChat(true);

    if (numRepeats > 1) {
      const { job_ids } = await api.batchRun(parsed, numRepeats);
      setBatchJobIds(job_ids);
      setJobId(job_ids[0]);
      setJobStatus('running');
      setBatchCompleted(0);
      try { sessionStorage.setItem('running-job', JSON.stringify({ jobId: job_ids[0], batchJobIds: job_ids })); } catch {}
    } else {
      const { job_id } = await api.runExperiment(parsed);
      setJobId(job_id);
      setBatchJobIds([job_id]);
      setJobStatus('running');
      try { sessionStorage.setItem('running-job', JSON.stringify({ jobId: job_id, batchJobIds: [job_id] })); } catch {}
    }
    setStep('running');
  }

  function handleFinish() {
    setShowChat(false);
    setLiveSteps([]);
    try { sessionStorage.removeItem('running-job'); } catch {}
  }

  // ── Step: Template ─────────────────────────────────────────────────
  if (step === 'template') {
    return (
      <div>
        <StepIndicator current="template" />
        <h2 className="text-xl font-semibold text-gray-900 mb-2">New Experiment</h2>
        <p className="text-sm text-gray-500 mb-6">Pick a template or start from scratch.</p>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 mb-6">
          {templates.map((t) => (
            <button
              key={t.filename}
              onClick={() => selectTemplate(t)}
              className="text-left bg-white rounded-xl border border-gray-200 p-5 hover:shadow-md hover:border-indigo-200 transition-all"
            >
              <p className="font-medium text-gray-900 text-sm">{t.experiment_name}</p>
              <p className="text-xs text-gray-400 mt-1">{t.policy_type} policy</p>
            </button>
          ))}
        </div>

        <button
          onClick={() => { setConfig('{\n  \n}'); setStep('edit'); }}
          className="text-sm text-indigo-600 hover:underline"
        >
          Start from blank config &rarr;
        </button>
      </div>
    );
  }

  // ── Step: Edit ─────────────────────────────────────────────────────
  if (step === 'edit') {
    return (
      <div>
        <StepIndicator current="edit" />
        <h2 className="text-xl font-semibold text-gray-900 mb-2">Configure Experiment</h2>
        <p className="text-sm text-gray-500 mb-4">
          Edit the JSON config, then review the prompt before launching.
        </p>

        <textarea
          value={config}
          onChange={(e) => { setConfig(e.target.value); setValidationErr(null); }}
          rows={20}
          className="w-full font-mono text-sm bg-gray-50 border border-gray-200 rounded-lg p-4 focus:outline-none focus:ring-2 focus:ring-indigo-300"
        />

        {/* Repeat settings */}
        <div className="mt-4 bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-4">
            <label className="text-sm text-gray-600 font-medium">Number of repeats</label>
            <input
              type="number"
              min={1}
              max={20}
              value={numRepeats}
              onChange={(e) => setNumRepeats(Math.max(1, Math.min(20, parseInt(e.target.value) || 1)))}
              className="w-20 text-sm border border-gray-200 rounded-lg px-3 py-1.5 font-mono"
            />
          </div>
          {numRepeats > 1 && (
            <p className="text-xs text-gray-400 mt-2">
              Each repeat uses a different random seed (0 to {numRepeats - 1}) for reproducibility.
            </p>
          )}
        </div>

        {validationErr && (
          <p className="text-sm text-red-600 mt-2">{validationErr}</p>
        )}

        <div className="flex gap-3 mt-4">
          <button
            onClick={() => setStep('template')}
            className="px-4 py-2 text-sm text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50"
          >
            &larr; Back
          </button>
          <button
            onClick={validate}
            className="px-4 py-2 text-sm text-indigo-600 border border-indigo-200 rounded-lg hover:bg-indigo-50"
          >
            Validate
          </button>
          <button
            onClick={goToReview}
            disabled={previewLoading}
            className="px-4 py-2 text-sm text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:opacity-50"
          >
            {previewLoading ? 'Loading preview...' : 'Review Prompt & Launch'}
          </button>
        </div>
      </div>
    );
  }

  // ── Step: Review ───────────────────────────────────────────────────
  if (step === 'review' && preview) {
    return (
      <div>
        <StepIndicator current="review" />
        <h2 className="text-xl font-semibold text-gray-900 mb-2">Review Before Launch</h2>
        <p className="text-sm text-gray-500 mb-6">
          Verify the exact prompt and observation the LLM will receive. No hidden bias.
        </p>

        {/* Config summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          {[
            ['Policy', preview.policy_type],
            ['Model', preview.model],
            ['Temperature', String(preview.temperature)],
            ['Memory K', preview.policy_type.includes('memory') ? String(preview.memory_k) : 'n/a'],
            ['World Length', String(preview.world_length)],
            ['Start Position', String(preview.start_position)],
            ['Gradient', preview.include_gradient ? 'Yes' : 'No'],
            ['Episode', `${preview.episode_length} steps`],
            ['Repeats', String(numRepeats)],
          ].map(([label, val]) => (
            <div key={label} className="bg-white rounded-lg border border-gray-200 px-3 py-2">
              <p className="text-[10px] text-gray-400 uppercase tracking-wide">{label}</p>
              <p className="text-sm font-medium text-gray-800">{val}</p>
            </div>
          ))}
        </div>

        {/* System prompt */}
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">System Prompt</h3>
          <pre className="text-xs bg-slate-900 text-green-400 font-mono p-4 rounded-lg overflow-x-auto whitespace-pre-wrap leading-relaxed">
            {preview.system_prompt}
          </pre>
        </div>

        {/* Sample observation */}
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">
            Sample Observation (Step 0, Position {preview.start_position})
          </h3>
          <pre className="text-xs bg-slate-900 text-amber-300 font-mono p-4 rounded-lg overflow-x-auto whitespace-pre-wrap leading-relaxed">
            {JSON.stringify(preview.sample_observation, null, 2)}
          </pre>
          <p className="text-xs text-gray-400 mt-2">
            This is the exact JSON the LLM receives as user input at each step.
          </p>
        </div>

        <div className="flex gap-3">
          <button
            onClick={() => setStep('edit')}
            className="px-4 py-2 text-sm text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50"
          >
            &larr; Back to Edit
          </button>
          <button
            onClick={launch}
            className="px-5 py-2.5 text-sm text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 font-medium"
          >
            Confirm &amp; Launch {numRepeats > 1 ? `(${numRepeats} runs)` : 'Experiment'}
          </button>
        </div>
      </div>
    );
  }

  // ── Step: Running ──────────────────────────────────────────────────
  return (
    <div>
      <StepIndicator current="running" />
      <h2 className="text-xl font-semibold text-gray-900 mb-2">
        {jobStatus === 'running' ? 'Experiment Running...' : jobStatus === 'completed' ? 'Experiment Complete' : 'Experiment Failed'}
      </h2>

      {/* Batch progress */}
      {batchJobIds.length > 1 && (
        <p className="text-sm text-gray-500 mb-3">
          Run {Math.min(batchCompleted + 1, batchJobIds.length)}/{batchJobIds.length}
          {jobStatus === 'completed' && ' — All runs complete'}
        </p>
      )}

      {/* Live chat view */}
      {showChat && jobStatus === 'running' && (
        <div className="bg-gray-50 rounded-xl border border-gray-200 p-4 mb-4 max-h-96 overflow-y-auto">
          {liveSteps.length === 0 && (
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <span className="inline-block w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
              Waiting for first step...
            </div>
          )}
          {liveSteps.map((s) => {
            const moved = s.position_after !== s.position_before;
            const rewardDelta = s.step > 0 && liveSteps[s.step - 1]
              ? s.reward - liveSteps[s.step - 1].reward
              : 0;
            return (
              <div key={s.step} className="mb-3 last:mb-0">
                <div className="flex items-start gap-3">
                  <span className="text-[10px] text-gray-400 font-mono w-8 pt-1 shrink-0">#{s.step}</span>
                  <div className="flex-1">
                    <div className="bg-white rounded-lg border border-gray-200 px-3 py-2 inline-block max-w-md shadow-sm">
                      <p className="text-sm text-gray-800">
                        <span className="font-semibold text-gray-900">{s.action_type}</span>
                        {moved && (
                          <span className="text-gray-500 ml-2">
                            {s.position_before} &rarr; {s.position_after}
                          </span>
                        )}
                        <span className={`ml-2 text-xs font-mono ${
                          rewardDelta > 0 ? 'text-emerald-600' : rewardDelta < 0 ? 'text-red-500' : 'text-gray-400'
                        }`}>
                          {s.reward.toFixed(4)}
                        </span>
                      </p>
                      {s.reason && (
                        <p className="text-xs text-gray-500 italic mt-0.5">{s.reason}</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
          {/* Typing indicator */}
          {jobStatus === 'running' && liveSteps.length > 0 && (
            <div className="flex items-center gap-3 mt-2">
              <span className="w-8" />
              <div className="flex gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
      )}

      {/* Completed state */}
      {jobStatus === 'completed' && showChat && (
        <div className="bg-white border border-gray-200 rounded-xl p-5 mb-4">
          <p className="text-sm text-emerald-600 font-medium mb-1">Experiment completed successfully.</p>
          {liveSteps.length > 0 && (
            <div className="flex gap-4 text-xs text-gray-500 mt-2">
              <span>Steps: {liveSteps.length}</span>
              <span>Final position: {liveSteps[liveSteps.length - 1]?.position_after}</span>
              <span>Total reward: {liveSteps.reduce((sum, s) => sum + s.reward, 0).toFixed(2)}</span>
            </div>
          )}
          <div className="flex gap-3 mt-4">
            <button
              onClick={handleFinish}
              className="px-4 py-2 text-sm text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50"
            >
              Finish
            </button>
            <button
              onClick={() => navigate('/experiments')}
              className="px-4 py-2 text-sm text-white bg-indigo-600 rounded-lg hover:bg-indigo-700"
            >
              View Experiments
            </button>
            {batchJobIds.length > 1 && (
              <button
                onClick={() => navigate('/experiments')}
                className="px-4 py-2 text-sm text-indigo-600 border border-indigo-200 rounded-lg hover:bg-indigo-50"
              >
                Compare All Runs
              </button>
            )}
          </div>
        </div>
      )}

      {/* Failed state */}
      {jobStatus === 'failed' && (
        <p className="text-sm text-red-600 mb-4">Experiment failed. Check output below.</p>
      )}

      {/* Fallback text output (when chat is dismissed or no live steps) */}
      {!showChat && jobOutput && (
        <pre className="text-xs bg-gray-50 border border-gray-200 rounded-lg p-4 overflow-x-auto max-h-80 overflow-y-auto whitespace-pre-wrap">
          {jobOutput}
        </pre>
      )}

      {jobStatus !== 'running' && !showChat && (
        <div className="flex gap-3 mt-4">
          <button
            onClick={() => navigate('/experiments')}
            className="px-4 py-2 text-sm text-white bg-indigo-600 rounded-lg hover:bg-indigo-700"
          >
            View Experiments
          </button>
          <button
            onClick={() => { setStep('template'); setJobId(null); setJobOutput(''); setPreview(null); setLiveSteps([]); setBatchJobIds([]); setShowChat(true); }}
            className="text-sm text-indigo-600 hover:underline"
          >
            Run another experiment
          </button>
        </div>
      )}
    </div>
  );
}
