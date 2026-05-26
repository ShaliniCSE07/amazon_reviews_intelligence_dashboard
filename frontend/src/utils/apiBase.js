/**
 * API base URL for ReviewIntellect.
 * In dev, use '' so Vite proxies /api → http://127.0.0.1:8000 (see vite.config.js).
 */
export function getApiBase() {
  if (import.meta.env.VITE_API_URL != null && import.meta.env.VITE_API_URL !== '') {
    return import.meta.env.VITE_API_URL;
  }
  if (import.meta.env.DEV) {
    return '';
  }
  const host = window.location.hostname;
  if (host === 'localhost' || host === '127.0.0.1') {
    return `http://${host}:8000`;
  }
  return '';
}

/** Build full URL: apiPath('/api/missing-features') → '/api/missing-features' or 'http://127.0.0.1:8000/api/...' */
export function apiPath(path, apiBase) {
  const base = apiBase ?? getApiBase();
  const normalized = path.startsWith('/') ? path : `/${path}`;
  if (!base) return normalized;
  return `${base.replace(/\/$/, '')}${normalized}`;
}
