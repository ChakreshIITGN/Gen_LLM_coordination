/** Thin API client — all calls go through Vite proxy to FastAPI backend. */

const BASE = '/api';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// ── Types ────────────────────────────────────────────────────────────

export interface ExperimentRun {
  id: string;
  tag: string;
  timestamp: string;
  has_config: boolean;
  has_metrics: boolean;
  experiment_name?: string;
  policy_type?: string;
  model?: string;
  episode_length?: number;
  total_reward?: number;
  coverage?: number;
}

export interface TrajectorySummary {
  steps: number[];
  positions: number[];
  rewards: number[];
  cumulative_rewards: number[];
  actions: string[];
  reasons: string[];
}

export interface ConfigTemplate {
  filename: string;
  experiment_name: string;
  policy_type: string;
  config: Record<string, unknown>;
}

export interface SetupStatus {
  ollama_running: boolean;
  models_available: string[];
  python_version: string;
}

export interface JobStatus {
  status: 'running' | 'completed' | 'failed';
  started_at: number;
  output: string;
  run_id: string | null;
}

export interface ValidationResult {
  valid: boolean;
  errors: string | null;
}

export interface CompareEntry {
  id: string;
  metrics?: Record<string, unknown>;
  policy?: string;
  model?: string;
}

export interface ActiveJobs {
  count: number;
  job_ids: string[];
}

export interface PromptPreview {
  system_prompt: string;
  sample_observation: Record<string, unknown>;
  policy_type: string;
  memory_k: number;
  model: string;
  temperature: number;
  world_length: number;
  include_gradient: boolean;
  start_position: number;
  episode_length: number;
}

export interface BatchRunResult {
  batch_id: string;
  job_ids: string[];
  status: string;
}

export interface StepData {
  step: number;
  position_before: number;
  position_after: number;
  action_type: string;
  reason: string;
  reward: number;
  raw_llm_output?: string;
}

export interface LiveSteps {
  steps: StepData[];
  status: string;
}

export interface MultiPolicyResult {
  batch_id: string;
  policy_jobs: Record<string, string[]>;
  status: string;
}

export interface PolicyAnalysis {
  count: number;
  runs: { run_id: string; tag: string; timestamp: string }[];
  avg_metrics: Record<string, number>;
}

export interface AnalysisResult {
  policies: Record<string, PolicyAnalysis>;
}

// ── API functions ────────────────────────────────────────────────────

export const api = {
  listExperiments: () => get<ExperimentRun[]>('/experiments'),
  getConfig: (tag: string, ts: string) => get<Record<string, unknown>>(`/experiments/${tag}/${ts}/config`),
  getMetrics: (tag: string, ts: string) => get<Record<string, unknown>>(`/experiments/${tag}/${ts}/metrics`),
  getTrajectorySummary: (tag: string, ts: string) => get<TrajectorySummary>(`/experiments/${tag}/${ts}/trajectory/summary`),
  compareRuns: (ids: string[]) => get<CompareEntry[]>(`/experiments/compare?runs=${ids.join(',')}`),
  getTemplates: () => get<ConfigTemplate[]>('/configs/templates'),
  getSetupStatus: () => get<SetupStatus>('/setup/status'),
  getActiveJobs: () => get<ActiveJobs>('/experiments/jobs/active'),
  runExperiment: (config: Record<string, unknown>) => post<{ job_id: string; status: string }>('/experiments/run', { config }),
  getJobStatus: (jobId: string) => get<JobStatus>(`/experiments/job/${jobId}`),
  validateConfig: (config: Record<string, unknown>) => post<ValidationResult>('/experiments/validate-config', { config }),
  previewPrompt: (config: Record<string, unknown>) => post<PromptPreview>('/experiments/preview-prompt', { config }),
  batchRun: (config: Record<string, unknown>, numRepeats: number, seeds?: number[]) =>
    post<BatchRunResult>('/experiments/batch-run', { config, num_repeats: numRepeats, seeds }),
  getJobSteps: (jobId: string) => get<LiveSteps>(`/experiments/job/${jobId}/steps`),
  multiPolicyRun: (config: Record<string, unknown>, policies: string[], numRepeats: number) =>
    post<MultiPolicyResult>('/experiments/multi-policy-run', { config, policies, num_repeats: numRepeats }),
  getAnalysis: () => get<AnalysisResult>('/experiments/analysis'),
};
