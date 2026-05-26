import React, { useState, useEffect, useMemo } from 'react';
import { Lightbulb, ChevronDown, ChevronUp, Loader2, AlertCircle } from 'lucide-react';
import { apiPath } from '../utils/apiBase';

const RATING_FILTERS = [
  { id: 'ALL', label: 'All reviews' },
  { id: 'LOW', label: '1–2 star' },
  { id: 'MID', label: '3 star' },
];

function sentimentToScore(review) {
  if (review.sentiment_confidence != null && review.sentiment) {
    const c = review.sentiment_confidence;
    if (review.sentiment === 'Negative') return Math.min(0.35, 1 - c);
    if (review.sentiment === 'Positive') return Math.max(0.7, c);
    if (review.sentiment === 'Mixed') return 0.35;
    return 0.5;
  }
  if (review.rating != null) {
    return { 1: 0.1, 2: 0.22, 3: 0.42, 4: 0.72, 5: 0.9 }[review.rating] ?? 0.5;
  }
  return undefined;
}

const MissingFeatureCard = ({ productId, reviews, apiHost, initialMissingFeatures = [] }) => {
  const [ratingFilter, setRatingFilter] = useState('ALL');
  const [missingFeatures, setMissingFeatures] = useState(initialMissingFeatures);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedFeature, setExpandedFeature] = useState(null);
  const [cached, setCached] = useState(false);
  const filteredReviews = useMemo(() => {
    return reviews.filter((r) => {
      const rating = r.rating ?? 5;
      if (ratingFilter === 'LOW') return rating <= 2;
      if (ratingFilter === 'MID') return rating === 3;
      return true;
    });
  }, [reviews, ratingFilter]);

  // Sync when analysis payload includes precomputed missing features (ALL filter)
  useEffect(() => {
    if (ratingFilter === 'ALL' && initialMissingFeatures?.length > 0) {
      setMissingFeatures(initialMissingFeatures);
      setError(null);
      setLoading(false);
    }
  }, [initialMissingFeatures, ratingFilter]);

  useEffect(() => {
    if (!productId || filteredReviews.length === 0) {
      setMissingFeatures([]);
      setError(null);
      return;
    }

    // Use analysis payload for "All reviews" — no extra network call
    if (ratingFilter === 'ALL' && initialMissingFeatures?.length > 0) {
      setMissingFeatures(initialMissingFeatures);
      setError(null);
      return;
    }

    const controller = new AbortController();
    const requestUrl = apiPath('/api/missing-features', apiHost);

    const fetchMissing = async () => {
      setLoading(true);
      setError(null);
      try {
        const payload = {
          product_id: productId,
          reviews: filteredReviews.map((r, idx) => ({
            id: r.id ?? r.review_id ?? `rev-${idx}`,
            text: r.text,
            rating: r.rating ?? null,
            date: r.date ?? null,
            sentiment_score: sentimentToScore(r),
            sentiment: r.sentiment ?? null,
          })),
        };

        const res = await fetch(requestUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          const detail = err.detail;
          const detailStr =
            typeof detail === 'string'
              ? detail
              : Array.isArray(detail)
                ? detail.map((d) => d.msg || JSON.stringify(d)).join('; ')
                : '';
          throw new Error(detailStr || `Request failed (${res.status})`);
        }

        const data = await res.json();
        setMissingFeatures(data.missing_features || []);
        setCached(!!data.cached);
        setExpandedFeature(null);
      } catch (e) {
        if (e.name !== 'AbortError') {
          const msg = e.message || 'Failed to load missing features';
          const hint =
            msg === 'Failed to fetch'
              ? ` — cannot reach API at ${requestUrl}. Start backend: cd backend && uvicorn main:app --reload --host 127.0.0.1 --port 8000. Restart frontend after code changes.`
              : '';
          setError(msg + hint);
          if (ratingFilter === 'ALL' && initialMissingFeatures?.length > 0) {
            setMissingFeatures(initialMissingFeatures);
          } else {
            setMissingFeatures([]);
          }
        }
      } finally {
        setLoading(false);
      }
    };

    fetchMissing();
    return () => controller.abort();
  }, [productId, ratingFilter, filteredReviews.length, apiHost, initialMissingFeatures]);

  const topScore = missingFeatures[0]?.score ?? 1;

  return (
    <div className="glass-card p-6 border border-slate-200/50 dark:border-slate-850">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div>
          <h3 className="font-extrabold text-lg text-slate-800 dark:text-slate-100 flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-amber-400" />
            Missing Feature Detector
          </h3>
          <p className="text-xs text-slate-400 dark:text-slate-500 mt-1 max-w-lg">
            Surfaces what customers wish the product had — mined from desire phrases in lower-rated and negative reviews.
          </p>
          {cached && (
            <span className="text-[10px] uppercase font-bold text-brand-500 tracking-wider mt-1 inline-block">
              Cached result
            </span>
          )}
        </div>

        <div className="flex rounded-xl bg-slate-100 dark:bg-slate-900 p-1 gap-1">
          {RATING_FILTERS.map((f) => (
            <button
              key={f.id}
              type="button"
              onClick={() => setRatingFilter(f.id)}
              className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all cursor-pointer ${
                ratingFilter === f.id
                  ? 'bg-white dark:bg-slate-800 text-brand-600 dark:text-brand-400 shadow-sm'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="flex items-center justify-center gap-2 py-12 text-slate-400">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm font-medium">Scanning reviews for missing features…</span>
        </div>
      )}

      {error && !loading && (
        <div className="flex items-center gap-2 text-rose-500 bg-rose-500/5 border border-rose-500/20 rounded-xl p-4 text-sm">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {!loading && !error && missingFeatures.length === 0 && (
        <p className="text-sm text-slate-400 dark:text-slate-500 py-8 text-center">
          No missing-feature requests detected for this filter. Try &quot;All reviews&quot; or add more critical reviews.
        </p>
      )}

      {!loading && !error && missingFeatures.length > 0 && (
        <ul className="space-y-4">
          {missingFeatures.map((item, idx) => {
            const isExpanded = expandedFeature === item.feature;
            const barPct = topScore > 0 ? Math.round((item.score / topScore) * 100) : 0;
            const previewExamples = isExpanded
              ? item.examples
              : item.examples.slice(0, 2);

            return (
              <li
                key={`${item.feature}-${idx}`}
                className="border border-slate-200/60 dark:border-slate-800 rounded-xl overflow-hidden bg-slate-50/50 dark:bg-slate-900/40"
              >
                <button
                  type="button"
                  onClick={() =>
                    setExpandedFeature(isExpanded ? null : item.feature)
                  }
                  className="w-full text-left p-4 hover:bg-slate-100/80 dark:hover:bg-slate-850/50 transition-colors cursor-pointer"
                >
                  <div className="flex justify-between items-start gap-4">
                    <div className="flex-1 min-w-0">
                      <h4 className="font-extrabold text-base text-slate-800 dark:text-slate-100 capitalize">
                        {item.feature}
                      </h4>
                      <p className="text-xs text-slate-400 mt-0.5">
                        <span className="font-bold text-amber-500">{item.count}</span> mention
                        {item.count !== 1 ? 's' : ''} · urgency score{' '}
                        <span className="font-bold">{item.score}</span>
                      </p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-slate-400" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-slate-400" />
                      )}
                    </div>
                  </div>

                  <div className="w-full h-1.5 bg-slate-200 dark:bg-slate-800 rounded-full mt-3 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-amber-500 to-orange-500 transition-all duration-500"
                      style={{ width: `${barPct}%` }}
                    />
                  </div>
                </button>

                <div className="px-4 pb-4 space-y-2">
                  {previewExamples.map((ex, i) => (
                    <p
                      key={i}
                      className="text-xs italic text-slate-600 dark:text-slate-300 bg-white/60 dark:bg-slate-950/50 p-3 rounded-lg border border-slate-200/50 dark:border-slate-800/60 border-l-2 border-l-amber-500/60"
                    >
                      &ldquo;{ex}&rdquo;
                    </p>
                  ))}

                  {!isExpanded && item.examples.length > 2 && (
                    <p className="text-[10px] text-slate-400 font-medium pt-0.5">
                      +{item.examples.length - 2} more — click row to expand
                    </p>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};

export default MissingFeatureCard;
