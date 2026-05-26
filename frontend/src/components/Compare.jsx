import React, { useState, useEffect } from 'react';
import { RefreshCw, ArrowLeftRight, Check, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const Compare = ({ productsList, apiHost }) => {
  const [prodId1, setProdId1] = useState('');
  const [prodId2, setProdId2] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [data1, setData1] = useState(null);
  const [data2, setData2] = useState(null);

  const handleCompare = async () => {
    if (!prodId1 || !prodId2) return;
    if (prodId1 === prodId2) {
      setError('Please select two different products to compare.');
      return;
    }

    setLoading(true);
    setError(null);
    setData1(null);
    setData2(null);

    try {
      // Fetch data for product 1
      const res1 = await fetch(`${apiHost}/api/analysis/${prodId1}`);
      if (!res1.ok) throw new Error('Failed to load first product analysis');
      const payload1 = await res1.json();
      setData1(payload1);

      // Fetch data for product 2
      const res2 = await fetch(`${apiHost}/api/analysis/${prodId2}`);
      if (!res2.ok) throw new Error('Failed to load second product analysis');
      const payload2 = await res2.json();
      setData2(payload2);
    } catch (err) {
      console.error(err);
      setError(err.message || 'Error occurred during comparative fetch.');
    } finally {
      setLoading(false);
    }
  };

  // Compile Chart data for Sentiment Comparison
  const getSentimentChartData = () => {
    if (!data1 || !data2) return [];
    
    const s1 = data1.sentiment_distribution;
    const s2 = data2.sentiment_distribution;
    const t1 = sumDist(s1);
    const t2 = sumDist(s2);

    const getPct = (count, total) => total > 0 ? Math.round((count / total) * 100) : 0;

    return ['Positive', 'Negative', 'Neutral', 'Mixed'].map(lbl => ({
      name: lbl,
      [data1.product.title]: getPct(s1[lbl] || 0, t1),
      [data2.product.title]: getPct(s2[lbl] || 0, t2)
    }));
  };

  // Compile Chart data for Emotion Comparison
  const getEmotionChartData = () => {
    if (!data1 || !data2) return [];
    const emoNames = ['frustration', 'delight', 'disappointment', 'surprise'];
    
    return emoNames.map(emo => ({
      name: emo.charAt(0).toUpperCase() + emo.slice(1),
      [data1.product.title]: data1.emotion_distribution[emo] || 0,
      [data2.product.title]: data2.emotion_distribution[emo] || 0
    }));
  };

  // Compile Common/Overlapping Feature Comparison
  const getFeatureComparisonData = () => {
    if (!data1 || !data2) return [];
    
    const f1 = data1.features;
    const f2 = data2.features;
    
    // Find matching aspect names (case insensitive)
    const map1 = {};
    f1.forEach(f => { map1[f.feature_name.toLowerCase()] = f; });
    
    const result = [];
    f2.forEach(f => {
      const match = map1[f.feature_name.toLowerCase()];
      if (match) {
        result.push({
          name: f.feature_name,
          [data1.product.title]: match.sentiment_score,
          [data2.product.title]: f.sentiment_score
        });
      }
    });

    // If no matching features, just take top 3 from each
    if (result.length === 0) {
      const topF1 = f1.slice(0, 3);
      const topF2 = f2.slice(0, 3);
      const allKeys = new Set([...topF1.map(f => f.feature_name), ...topF2.map(f => f.feature_name)]);
      
      allKeys.forEach(k => {
        const item1 = f1.find(f => f.feature_name === k);
        const item2 = f2.find(f => f.feature_name === k);
        result.push({
          name: k,
          [data1.product.title]: item1 ? item1.sentiment_score : 0,
          [data2.product.title]: item2 ? item2.sentiment_score : 0
        });
      });
    }

    return result;
  };

  const sumDist = (dist) => {
    return Object.values(dist).reduce((a, b) => a + b, 0);
  };

  return (
    <div className="space-y-8">
      {/* Selector Card */}
      <div className="glass-card p-6 border border-slate-200/50 dark:border-slate-850 shadow-md">
        <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2 mb-4">
          <ArrowLeftRight className="w-5 h-5 text-brand-500" />
          Product Comparative Analytics
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <label className="block text-xs font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-2">
              Select Product 1
            </label>
            <select
              value={prodId1}
              onChange={(e) => setProdId1(e.target.value)}
              className="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 focus:outline-none focus:ring-2 focus:ring-brand-500 text-sm dark:text-slate-300"
            >
              <option value="">-- Choose first product --</option>
              {productsList.map(p => (
                <option key={p.id} value={p.id}>{p.title} ({p.reviews_count} reviews)</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-2">
              Select Product 2
            </label>
            <select
              value={prodId2}
              onChange={(e) => setProdId2(e.target.value)}
              className="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 focus:outline-none focus:ring-2 focus:ring-brand-500 text-sm dark:text-slate-300"
            >
              <option value="">-- Choose second product --</option>
              {productsList.map(p => (
                <option key={p.id} value={p.id}>{p.title} ({p.reviews_count} reviews)</option>
              ))}
            </select>
          </div>
        </div>

        {error && (
          <div className="mb-4 p-3.5 bg-rose-50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900/40 rounded-xl flex items-center gap-2.5 text-rose-700 dark:text-rose-400 text-sm">
            <AlertCircle className="w-4 h-4" />
            {error}
          </div>
        )}

        <button
          onClick={handleCompare}
          disabled={loading || !prodId1 || !prodId2}
          className="w-full py-3 bg-brand-600 hover:bg-brand-500 text-white font-semibold rounded-xl transition-all shadow-md flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
        >
          {loading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              Loading Analyses...
            </>
          ) : (
            'Compare Products'
          )}
        </button>
      </div>

      {/* Comparison Results Dashboard */}
      {data1 && data2 && (
        <div className="space-y-8 animate-fadeIn">
          {/* Metadata Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Product 1 Details */}
            <div className="glass-card p-6 border-l-4 border-l-brand-500">
              <span className="px-2 py-0.5 bg-brand-100 dark:bg-brand-950 text-brand-700 dark:text-brand-400 text-xs font-bold rounded-md">
                Product A
              </span>
              <h4 className="font-bold text-lg mt-2 text-slate-800 dark:text-slate-100 leading-snug">
                {data1.product.title}
              </h4>
              <div className="mt-4 space-y-1.5 text-sm text-slate-600 dark:text-slate-400">
                <p><b>ASIN/ID:</b> {data1.product.id}</p>
                <p><b>Rating:</b> {data1.product.overall_rating} / 5.0</p>
                <p><b>Analyzed:</b> {data1.product.reviews_count} reviews</p>
              </div>
            </div>

            {/* Product 2 Details */}
            <div className="glass-card p-6 border-l-4 border-l-violet-500">
              <span className="px-2 py-0.5 bg-violet-100 dark:bg-violet-950 text-violet-700 dark:text-violet-400 text-xs font-bold rounded-md">
                Product B
              </span>
              <h4 className="font-bold text-lg mt-2 text-slate-800 dark:text-slate-100 leading-snug">
                {data2.product.title}
              </h4>
              <div className="mt-4 space-y-1.5 text-sm text-slate-600 dark:text-slate-400">
                <p><b>ASIN/ID:</b> {data2.product.id}</p>
                <p><b>Rating:</b> {data2.product.overall_rating} / 5.0</p>
                <p><b>Analyzed:</b> {data2.product.reviews_count} reviews</p>
              </div>
            </div>
          </div>

          {/* Chart Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Sentiment Comparison */}
            <div className="glass-card p-6">
              <h4 className="font-bold text-slate-700 dark:text-slate-350 text-sm uppercase tracking-wider mb-6">
                Sentiment Class Percentages (%)
              </h4>
              <div className="h-[280px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={getSentimentChartData()}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                    <XAxis dataKey="name" stroke="#94A3B8" fontSize={11} />
                    <YAxis stroke="#94A3B8" fontSize={11} />
                    <Tooltip cursor={{ fill: 'rgba(99, 102, 241, 0.05)' }} />
                    <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '11px' }} />
                    <Bar dataKey={data1.product.title} fill="#6366f1" radius={[4, 4, 0, 0]} />
                    <Bar dataKey={data2.product.title} fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Emotion Comparison */}
            <div className="glass-card p-6">
              <h4 className="font-bold text-slate-700 dark:text-slate-350 text-sm uppercase tracking-wider mb-6">
                Emotion Distributions (%)
              </h4>
              <div className="h-[280px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={getEmotionChartData()}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                    <XAxis dataKey="name" stroke="#94A3B8" fontSize={11} />
                    <YAxis stroke="#94A3B8" fontSize={11} />
                    <Tooltip cursor={{ fill: 'rgba(99, 102, 241, 0.05)' }} />
                    <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '11px' }} />
                    <Bar dataKey={data1.product.title} fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey={data2.product.title} fill="#ec4899" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Aspect-level Sentiment Comparison */}
          <div className="glass-card p-6">
            <h4 className="font-bold text-slate-700 dark:text-slate-355 text-sm uppercase tracking-wider mb-6">
              Aspect-level Sentiment Scores Compared
            </h4>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={getFeatureComparisonData()} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                  <XAxis type="number" domain={[0, 100]} stroke="#94A3B8" fontSize={11} />
                  <YAxis dataKey="name" type="category" stroke="#94A3B8" fontSize={11} width={80} />
                  <Tooltip />
                  <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '11px' }} />
                  <Bar dataKey={data1.product.title} fill="#6366f1" radius={[0, 4, 4, 0]} barSize={12} />
                  <Bar dataKey={data2.product.title} fill="#8b5cf6" radius={[0, 4, 4, 0]} barSize={12} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-4 text-center">
              Scores represent percentage of positive sentiment references. Higher is better.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default Compare;
