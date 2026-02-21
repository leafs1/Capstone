const BASE = '/api';

async function fetchJSON(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${res.status}`);
  }
  return res.json();
}

// Datasets
export const getDatasetsSummary = () => fetchJSON('/datasets/summary');
export const getMarkets = (params = {}) => {
  const qs = new URLSearchParams(params).toString();
  return fetchJSON(`/datasets/markets?${qs}`);
};
export const getThemes = () => fetchJSON('/datasets/themes');
export const getEquityTickers = () => fetchJSON('/datasets/equity/tickers');
export const getTokens = (params = {}) => {
  const qs = new URLSearchParams(params).toString();
  return fetchJSON(`/datasets/tokens?${qs}`);
};

// Results
export const getResults = (params = {}) => {
  const qs = new URLSearchParams(params).toString();
  return fetchJSON(`/results?${qs}`);
};
export const getResultsStats = () => fetchJSON('/results/stats');

// Plots
export const getTimeSeries = (tokenId) => fetchJSON(`/plot/timeseries/${tokenId}`);
export const getScatterData = (tokenId) => fetchJSON(`/plot/scatter/${tokenId}`);
export const getPlotPng = (tokenId) => fetchJSON(`/plot/png/${tokenId}`);

// Analysis
export const runAnalysis = (body) =>
  fetchJSON('/analysis/run', { method: 'POST', body: JSON.stringify(body) });
export const runSingleAnalysis = (body) =>
  fetchJSON('/analysis/run_single', { method: 'POST', body: JSON.stringify(body) });
export const getJobStatus = (jobId) => fetchJSON(`/analysis/status/${jobId}`);
export const getJobs = () => fetchJSON('/analysis/jobs');

// Health
export const healthCheck = () => fetchJSON('/health');
