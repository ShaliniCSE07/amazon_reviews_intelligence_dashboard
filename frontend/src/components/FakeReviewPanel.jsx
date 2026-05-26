import React, { useState, useEffect, useMemo } from 'react';
import {
  ShieldAlert,
  Loader2,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  AlignLeft,
  Calendar,
  Copy,
  MessageSquareWarning,
  Zap,
  Ruler,
} from 'lucide-react';
import { apiPath } from '../utils/apiBase';

const LABEL_STYLES = {
  'Likely genuine': 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300',
  Suspicious: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300',
  'Likely fake': 'bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-300',
};

const SIGNAL_META = {
  mismatch: { label: 'Mismatch', icon: MessageSquareWarning, className: 'bg-violet-100 text-violet-700 dark:bg-violet-900/50 dark:text-violet-300' },
  burst: { label: 'Burst', icon: Calendar, className: 'bg-sky-100 text-sky-700 dark:bg-sky-900/50 dark:text-sky-300' },
  duplicate: { label: 'Duplicate', icon: Copy, className: 'bg-orange-100 text-orange-700 dark:bg-orange-900/50 dark:text-orange-300' },
  generic: { label: 'Generic', icon: AlignLeft, className: 'bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-slate-200' },
  too_short: { label: 'Too short', icon: Ruler, className: 'bg-fuchsia-100 text-fuchsia-700 dark:bg-fuchsia-900/50 dark:text-fuchsia-300' },
};

const FILTER_OPTIONS = [
  { id: 'all', label: 'All' },
  { id: 'flagged', label: 'Flagged (suspicious + fake)' },
  { id: 'fake', label: 'Likely fake only' },
];

function truncate(text, max = 100) {
  if (!text) return '';
  return text.length <= max ? text : `${text.slice(0, max)}…`;
}

const FakeReviewPanel = ({ productId, reviews, apiHost }) => {
  const [filter, setFilter] = useState('flagged');
  const [results, setResults] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    if (!productId || !reviews?.length) {
      setResults([]);
      setSummary(null);
      return;
    }

    const controller = new AbortController();
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const payload = {
          product_id: productId,
          reviews: reviews.map((r) => ({
            text: r.text ?? '',
            rating: Number(r.rating ?? 3),
            date: r.date ?? '',
          })),
        };
        const res = await fetch(apiPath('/api/fake-detection', apiHost), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `Request failed (${res.status})`);
        }
        const data = await res.json();
        setResults(data.reviews || []);
        setSummary(data.summary || null);
      } catch (e) {
        if (e.name !== 'AbortError') {
          setError(e.message || 'Failed to run fake detection');
          setResults([]);
          setSummary(null);
        }
      } finally {
        setLoading(false);
      }
    };

    run();
    return () => controller.abort();
  }, [productId, reviews, apiHost]);

  const filteredRows = useMemo(() => {
    return results
      .map((row, idx) => ({ ...row, _id: idx }))
      .filter((row) => {
        if (filter === 'flagged') {
          return row.fake_label === 'Suspicious' || row.fake_label === 'Likely fake';
        }
        if (filter === 'fake') return row.fake_label === 'Likely fake';
        return true;
      });
  }, [results, filter]);

  const topSignals = summary?.top_signals ?? [];

  return (
    <section className="glass-card p-6 border border-slate-200 dark:border-slate-800">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-lg bg-rose-100 dark:bg-rose-900/30">
          <ShieldAlert className="w-5 h-5 text-rose-600 dark:text-rose-400" />
        </div>
        <div>
          <h3 className="text-lg font-bold text-slate-900 dark:text-white">
            Fake Review Detector
          </h3>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Text, rating, date, and cross-review pattern signals
          </p>
        </div>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-slate-500 py-8 justify-center">
          <Loader2 className="w-5 h-5 animate-spin" />
          Analysing reviews…
        </div>
      )}

      {error && !loading && (
        <div className="flex items-center gap-2 text-rose-600 dark:text-rose-400 text-sm py-4">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {summary && !loading && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
          <div className="rounded-xl bg-slate-50 dark:bg-slate-900/50 p-4 border border-slate-100 dark:border-slate-800">
            <p className="text-xs uppercase tracking-wider text-slate-500 mb-1">Total analysed</p>
            <p className="text-2xl font-bold text-slate-900 dark:text-white">{summary.total}</p>
          </div>
          <div className="rounded-xl bg-amber-50 dark:bg-amber-900/20 p-4 border border-amber-100 dark:border-amber-900/40">
            <p className="text-xs uppercase tracking-wider text-amber-700 dark:text-amber-400 mb-1">
              Flagged (suspicious + fake)
            </p>
            <p className="text-2xl font-bold text-amber-800 dark:text-amber-200">
              {summary.fake_percentage}%
            </p>
            <p className="text-xs text-amber-600 dark:text-amber-500 mt-1">
              {(summary.flagged ?? summary.suspicious + summary.likely_fake)} of {summary.total} reviews
            </p>
          </div>
          <div className="rounded-xl bg-slate-50 dark:bg-slate-900/50 p-4 border border-slate-100 dark:border-slate-800">
            <p className="text-xs uppercase tracking-wider text-slate-500 mb-2 flex items-center gap-1">
              <Zap className="w-3 h-3" /> Top signals
            </p>
            {topSignals.length === 0 ? (
              <p className="text-sm text-slate-400">None fired</p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {topSignals.map((sig) => {
                  const meta = SIGNAL_META[sig] || { label: sig, className: 'bg-slate-200 text-slate-700' };
                  return (
                    <span
                      key={sig}
                      className={`text-xs font-medium px-2 py-1 rounded-full ${meta.className}`}
                    >
                      {meta.label}
                    </span>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {results.length > 0 && !loading && (
        <>
          <div className="flex flex-wrap gap-2 mb-4">
            {FILTER_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setFilter(opt.id)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  filter === opt.id
                    ? 'bg-brand-600 text-white'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
            <table className="w-full text-sm text-left">
              <thead className="bg-slate-50 dark:bg-slate-900/80 text-slate-500 uppercase text-xs">
                <tr>
                  <th className="px-4 py-3 font-semibold">Review</th>
                  <th className="px-4 py-3 font-semibold w-16">Rating</th>
                  <th className="px-4 py-3 font-semibold w-28">Label</th>
                  <th className="px-4 py-3 font-semibold w-16">Score</th>
                  <th className="px-4 py-3 font-semibold">Signals</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {filteredRows.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-4 py-8 text-center text-slate-400">
                      No reviews match this filter.
                    </td>
                  </tr>
                ) : (
                  filteredRows.map((row) => {
                    const isOpen = expandedId === row._id;
                    const signals = row.signals_fired || [];
                    return (
                      <tr key={row._id} className="hover:bg-slate-50/80 dark:hover:bg-slate-900/40">
                        <td className="px-4 py-3 align-top max-w-md">
                          <button
                            type="button"
                            className="text-left text-slate-700 dark:text-slate-200 hover:text-brand-600 dark:hover:text-brand-400"
                            onClick={() => setExpandedId(isOpen ? null : row._id)}
                          >
                            {isOpen ? row.text : truncate(row.text)}
                            {row.text?.length > 100 && (
                              <span className="inline-flex ml-1 text-slate-400">
                                {isOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                              </span>
                            )}
                          </button>
                        </td>
                        <td className="px-4 py-3 align-top font-medium">{row.rating ?? '—'}</td>
                        <td className="px-4 py-3 align-top">
                          <span
                            className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold ${
                              LABEL_STYLES[row.fake_label] || LABEL_STYLES['Likely genuine']
                            }`}
                          >
                            {row.fake_label}
                          </span>
                        </td>
                        <td className="px-4 py-3 align-top font-mono text-slate-600 dark:text-slate-300">
                          {row.fake_score}
                        </td>
                        <td className="px-4 py-3 align-top">
                          <div className="flex flex-wrap gap-1">
                            {signals.length === 0 ? (
                              <span className="text-xs text-slate-400">—</span>
                            ) : (
                              signals.map((sig) => {
                                const meta = SIGNAL_META[sig];
                                const Icon = meta?.icon;
                                return (
                                  <span
                                    key={sig}
                                    title={meta?.label || sig}
                                    className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium ${
                                      meta?.className || 'bg-slate-200 text-slate-600'
                                    }`}
                                  >
                                    {Icon && <Icon className="w-3 h-3" />}
                                    {meta?.label || sig}
                                  </span>
                                );
                              })
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
};

export default FakeReviewPanel;
