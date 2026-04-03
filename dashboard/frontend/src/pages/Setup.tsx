import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { SetupStatus } from '../lib/api';

export default function Setup() {
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [loading, setLoading] = useState(true);

  function refresh() {
    setLoading(true);
    api.getSetupStatus().then(setStatus).finally(() => setLoading(false));
  }

  useEffect(refresh, []);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900">Environment Setup</h2>
        <button
          onClick={refresh}
          className="px-3 py-1.5 text-sm text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50"
        >
          Refresh
        </button>
      </div>

      {loading ? (
        <p className="text-sm text-gray-400">Checking...</p>
      ) : !status ? (
        <p className="text-sm text-red-500">Could not reach backend. Is it running on port 8000?</p>
      ) : (
        <div className="space-y-4 max-w-lg">
          {/* Ollama status */}
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <div className="flex items-center gap-2 mb-2">
              <span className={`inline-block w-2.5 h-2.5 rounded-full ${status.ollama_running ? 'bg-emerald-500' : 'bg-red-400'}`} />
              <h3 className="text-sm font-semibold text-gray-800">
                Ollama {status.ollama_running ? 'Running' : 'Not Running'}
              </h3>
            </div>
            {!status.ollama_running && (
              <p className="text-xs text-gray-500">
                Start Ollama with <code className="bg-gray-100 px-1.5 py-0.5 rounded">ollama serve</code> to use LLM policies.
              </p>
            )}
          </div>

          {/* Models */}
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="text-sm font-semibold text-gray-800 mb-2">Available Models</h3>
            {status.models_available.length > 0 ? (
              <ul className="space-y-1">
                {status.models_available.map((m) => (
                  <li key={m} className="text-sm text-gray-600 font-mono bg-gray-50 px-3 py-1.5 rounded">
                    {m}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-gray-400">
                {status.ollama_running
                  ? 'No models pulled yet. Run: ollama pull mistral'
                  : 'Start Ollama first.'}
              </p>
            )}
          </div>

          {/* Python */}
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="text-sm font-semibold text-gray-800 mb-2">Python</h3>
            <p className="text-sm text-gray-600 font-mono">{status.python_version}</p>
          </div>
        </div>
      )}
    </div>
  );
}
