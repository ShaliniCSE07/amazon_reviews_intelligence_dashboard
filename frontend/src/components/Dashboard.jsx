import React, { useState } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, LineChart, Line, Legend } from 'recharts';
import { Download, AlertTriangle, Search, Filter, ThumbsUp, ThumbsDown, MessageSquare, Tag, Award, Sparkles, HelpCircle } from 'lucide-react';
import WordCloud from './WordCloud';
import MissingFeatureCard from './MissingFeatureCard';
import FakeReviewPanel from './FakeReviewPanel';

const Dashboard = ({ data, onBack, apiHost }) => {
  const { product, reviews, features, sentiment_distribution, emotion_distribution, trend_data, missing_features } = data;
  
  const [searchTerm, setSearchTerm] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState('ALL');
  const [activeFeatureCard, setActiveFeatureCard] = useState(null);

  // Constants for coloring
  const SENTIMENT_COLORS = {
    Positive: '#10b981', // emerald
    Negative: '#f43f5e', // rose
    Neutral: '#64748b', // slate
    Mixed: '#8b5cf6'   // violet
  };

  // Format Donut chart data
  const pieData = Object.entries(sentiment_distribution)
    .filter(entry => entry[1] > 0)
    .map(entry => ({ name: entry[0], value: entry[1] }));

  // Format Emotion chart data
  const barData = Object.entries(emotion_distribution).map(entry => ({
    name: entry[0].charAt(0).toUpperCase() + entry[0].slice(1),
    Score: entry[1]
  }));

  // Filter reviews
  const filteredReviews = reviews.filter(r => {
    const matchesSearch = r.text.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          (r.author && r.author.toLowerCase().includes(searchTerm.toLowerCase()));
    
    if (sentimentFilter === 'ALL') return matchesSearch;
    if (sentimentFilter === 'SPAM') return r.is_fake && matchesSearch;
    return r.sentiment.toUpperCase() === sentimentFilter && matchesSearch;
  });

  // Export handlers
  const handlePdfExport = () => {
    window.open(`${apiHost}/api/export/pdf/${product.id}`, '_blank');
  };

  const handleCsvExport = () => {
    window.open(`${apiHost}/api/export/csv/${product.id}`, '_blank');
  };

  // Compile words for the main word cloud (frequencies of review keywords)
  const getMainWordCloudData = () => {
    const counts = {};
    reviews.forEach(r => {
      r.keywords.forEach(kw => {
        counts[kw] = (counts[kw] || 0) + 1;
      });
    });
    return Object.entries(counts).map(([text, value]) => ({ text, value }));
  };

  // Helper to render quality color
  const getQualityColor = (score) => {
    if (score >= 70) return 'text-emerald-500 bg-emerald-50 dark:bg-emerald-950/20';
    if (score >= 40) return 'text-amber-500 bg-amber-50 dark:bg-amber-950/20';
    return 'text-rose-500 bg-rose-50 dark:bg-rose-950/20';
  };

  return (
    <div className="space-y-8 pb-16">
      {/* Header Panel */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <button 
            onClick={onBack}
            className="text-xs font-bold text-brand-600 dark:text-brand-400 uppercase tracking-widest hover:underline mb-2 cursor-pointer block"
          >
            &larr; Back to Ingestion
          </button>
          <h2 className="font-extrabold text-2xl text-slate-800 dark:text-slate-100 leading-snug">
            {product.title}
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Product Category: <b>{product.category || 'General'}</b> | Overall Rating: <b>{product.overall_rating} ★</b>
          </p>
        </div>

        {/* Action buttons */}
        <div className="flex gap-3">
          <button
            onClick={handleCsvExport}
            className="px-4 py-2.5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-850 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 font-semibold rounded-xl text-sm transition-all shadow-sm flex items-center gap-2 cursor-pointer"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
          <button
            onClick={handlePdfExport}
            className="px-4 py-2.5 bg-brand-600 hover:bg-brand-500 text-white font-semibold rounded-xl text-sm transition-all shadow-md hover:shadow-brand-500/10 flex items-center gap-2 cursor-pointer"
          >
            <Download className="w-4 h-4" />
            PDF Report
          </button>
        </div>
      </div>

      {/* Main Grid: Summary & Sentiment Donut */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Product summary Box */}
        <div className="lg:col-span-2 glass-card p-6 border-l-4 border-l-brand-500 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 flex items-center gap-2 mb-3">
              <Sparkles className="w-4 h-4 text-brand-500 animate-pulse" />
              Executive Smart Summary
            </h3>
            <p className="text-slate-700 dark:text-slate-350 text-base leading-relaxed italic">
              "{product.summary || 'A smart summarization is being prepared. Check review detail panels for specific aspect ratings.'}"
            </p>
          </div>
          <div className="border-t border-slate-200/50 dark:border-slate-800/40 pt-4 mt-6 flex justify-between text-xs text-slate-400">
            <span>Aggregated via Extractive LexRank Centrality</span>
            <span>Analyzed {reviews.length} reviews</span>
          </div>
        </div>

        {/* Sentiment Donut Card */}
        <div className="glass-card p-6 flex flex-col items-center">
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 self-start mb-2">
            Overall Sentiment Distribution
          </h3>
          <div className="w-full h-[180px] relative">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={75}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={SENTIMENT_COLORS[entry.name]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [`${value} reviews`, 'Volume']} />
              </PieChart>
            </ResponsiveContainer>
            {/* Center score */}
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none mt-2">
              <span className="text-2xl font-black text-slate-700 dark:text-slate-200">
                {Math.round((reviews.filter(r => r.sentiment === 'Positive').length / reviews.length) * 100)}%
              </span>
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Positive ratio</span>
            </div>
          </div>

          {/* Custom Legends */}
          <div className="grid grid-cols-2 gap-x-6 gap-y-2 mt-2 w-full text-xs font-semibold">
            {Object.entries(SENTIMENT_COLORS).map(([name, color]) => {
              const count = sentiment_distribution[name] || 0;
              return (
                <div key={name} className="flex items-center gap-2 justify-start">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                  <span className="text-slate-600 dark:text-slate-400 capitalize">{name}:</span>
                  <span className="text-slate-800 dark:text-slate-200 ml-auto">{count}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Analytics Row 2: Emotion Breakdown & Word Cloud */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Emotion chart */}
        <div className="glass-card p-6">
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-6">
            Emotion Profile breakdown
          </h3>
          <div className="h-[250px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                <XAxis type="number" domain={[0, 100]} stroke="#94A3B8" fontSize={11} />
                <YAxis dataKey="name" type="category" stroke="#94A3B8" fontSize={11} width={80} />
                <Tooltip formatter={(value) => [`${value}%`, 'Score']} />
                <Bar dataKey="Score" radius={[0, 4, 4, 0]} barSize={16}>
                  {barData.map((entry, index) => {
                    const colors = ['#f43f5e', '#10b981', '#3b82f6', '#f59e0b']; // mapped colors
                    return <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Word Cloud Card */}
        <div className="glass-card p-6 flex flex-col">
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-4">
            Frequent Topic Keywords
          </h3>
          <div className="flex-1 w-full min-h-[230px]">
            <WordCloud words={getMainWordCloudData()} />
          </div>
        </div>
      </div>

      {/* Trend chart over time */}
      {trend_data && trend_data.length >= 2 && (
        <div className="glass-card p-6">
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-6">
            Review Sentiment Trend over Time
          </h3>
          <div className="h-[250px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trend_data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                <XAxis dataKey="date" stroke="#94A3B8" fontSize={11} />
                <YAxis stroke="#94A3B8" fontSize={11} domain={[0, 100]} />
                <Tooltip formatter={(value) => [`${value}%`, 'Sentiment Index']} />
                <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '11px' }} />
                <Line 
                  type="monotone" 
                  dataKey="sentiment_score" 
                  name="Sentiment Index (Positivity Ratio)" 
                  stroke="#8b5cf6" 
                  strokeWidth={3} 
                  dot={{ r: 4, strokeWidth: 2 }} 
                  activeDot={{ r: 6 }} 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Missing Feature Detector */}
      <MissingFeatureCard
        productId={product.id}
        reviews={reviews}
        apiHost={apiHost}
        initialMissingFeatures={missing_features || []}
      />

      {/* Fake Review Detector */}
      <FakeReviewPanel
        productId={product.id}
        reviews={reviews}
        apiHost={apiHost}
      />

      {/* Dynamic Feature Insight Cards */}
      <div>
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-6">
          Extracted Feature Aspects ({features.length})
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feat) => {
            const isActive = activeFeatureCard === feat.id;

            return (
              <div 
                key={feat.id}
                className={`glass-card p-6 border transition-all duration-300 ${
                  isActive 
                    ? 'ring-2 ring-brand-500/40 border-brand-200 dark:border-slate-700' 
                    : 'hover:border-slate-350 dark:hover:border-slate-800'
                }`}
              >
                {/* Collapsed view summary header */}
                <div 
                  onClick={() => setActiveFeatureCard(isActive ? null : feat.id)}
                  className="flex justify-between items-center cursor-pointer"
                >
                  <div>
                    <h4 className="font-extrabold text-lg text-slate-850 dark:text-slate-100 capitalize">
                      {feat.feature_name}
                    </h4>
                    <p className="text-xs text-slate-400 mt-1">
                      Mentioned in <b>{feat.mentions} reviews</b>
                    </p>
                  </div>
                  <div className="text-right">
                    <span className={`text-2xl font-black ${
                      feat.sentiment_score >= 70 ? 'text-emerald-500' : (feat.sentiment_score >= 40 ? 'text-amber-500' : 'text-rose-500')
                    }`}>
                      {feat.sentiment_score}%
                    </span>
                    <p className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Aspect Index</p>
                  </div>
                </div>

                {/* Aspect Index slider bar */}
                <div className="w-full bg-slate-100 dark:bg-slate-850 h-2 rounded-full mt-4 overflow-hidden">
                  <div 
                    className={`h-full rounded-full ${
                      feat.sentiment_score >= 70 ? 'bg-emerald-500' : (feat.sentiment_score >= 40 ? 'bg-amber-500' : 'bg-rose-500')
                    }`}
                    style={{ width: `${feat.sentiment_score}%` }}
                  />
                </div>

                {/* Expanded Details accordion panel */}
                {isActive && (
                  <div className="mt-6 pt-6 border-t border-slate-200/50 dark:border-slate-800/40 space-y-6 animate-slideDown">
                    {/* Sample Quotes */}
                    <div className="space-y-4">
                      {/* Positive quotes */}
                      {feat.quotes_pos.length > 0 && (
                        <div className="space-y-2">
                          <h5 className="text-[10px] uppercase font-extrabold text-emerald-500 tracking-wider flex items-center gap-1">
                            <ThumbsUp className="w-3 h-3" /> Positive Snippets
                          </h5>
                          {feat.quotes_pos.map((q, idx) => (
                            <p key={idx} className="text-xs italic bg-emerald-500/5 text-emerald-700 dark:text-emerald-400 p-2.5 rounded-lg border border-emerald-500/10">
                              "{q}"
                            </p>
                          ))}
                        </div>
                      )}
                      
                      {/* Negative quotes */}
                      {feat.quotes_neg.length > 0 && (
                        <div className="space-y-2">
                          <h5 className="text-[10px] uppercase font-extrabold text-rose-500 tracking-wider flex items-center gap-1">
                            <ThumbsDown className="w-3 h-3" /> Negative Snippets
                          </h5>
                          {feat.quotes_neg.map((q, idx) => (
                            <p key={idx} className="text-xs italic bg-rose-500/5 text-rose-700 dark:text-rose-400 p-2.5 rounded-lg border border-rose-500/10">
                              "{q}"
                            </p>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Review List & Detail Table */}
      <div className="glass-card p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <h3 className="font-extrabold text-lg text-slate-800 dark:text-slate-100 flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-brand-500" />
            Review Detail Catalog
          </h3>
          
          {/* Filters and search box */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Search Input */}
            <div className="relative">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
              <input
                type="text"
                placeholder="Search reviews..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9 pr-4 py-2 border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-brand-500 w-full md:w-48"
              />
            </div>
            
            {/* Sentiment Filter Dropdown */}
            <div className="flex items-center gap-1">
              <Filter className="w-3.5 h-3.5 text-slate-400" />
              <select
                value={sentimentFilter}
                onChange={(e) => setSentimentFilter(e.target.value)}
                className="px-2 py-2 border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 text-xs rounded-xl text-slate-600 dark:text-slate-350 focus:outline-none"
              >
                <option value="ALL">All Sentiments</option>
                <option value="POSITIVE">Positive</option>
                <option value="NEGATIVE">Negative</option>
                <option value="NEUTRAL">Neutral</option>
                <option value="MIXED">Mixed</option>
                <option value="SPAM">Flagged Spam</option>
              </select>
            </div>
          </div>
        </div>

        {/* Scrollable list/table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse min-w-[700px]">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                <th className="pb-3 w-32">Reviewer</th>
                <th className="pb-3 w-16 text-center">Rating</th>
                <th className="pb-3 w-28">Sentiment</th>
                <th className="pb-3 w-24">Quality</th>
                <th className="pb-3 w-24">Spam Flag</th>
                <th className="pb-3">Review Comment</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800 text-sm">
              {filteredReviews.map((rev) => (
                <tr key={rev.id} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/20 transition-all">
                  {/* Reviewer / Author */}
                  <td className="py-4 pr-3">
                    <span className="font-semibold text-slate-700 dark:text-slate-300 block truncate max-w-[120px]">
                      {rev.author || 'Amazon Customer'}
                    </span>
                    <span className="text-[10px] text-slate-400 block mt-0.5">{rev.date || 'N/A'}</span>
                  </td>

                  {/* Rating */}
                  <td className="py-4 text-center text-amber-500 font-extrabold pr-3">
                    {rev.rating ? `${rev.rating} ★` : 'N/A'}
                  </td>

                  {/* Sentiment Badge */}
                  <td className="py-4 pr-3">
                    <span 
                      className="px-2 py-0.5 rounded-full text-xs font-bold"
                      style={{ 
                        color: SENTIMENT_COLORS[rev.sentiment], 
                        backgroundColor: `${SENTIMENT_COLORS[rev.sentiment]}15` 
                      }}
                    >
                      {rev.sentiment}
                    </span>
                    <span className="text-[9px] text-slate-400 block mt-1">
                      Conf: {Math.round(rev.sentiment_confidence * 100)}%
                    </span>
                  </td>

                  {/* Quality Score Badge */}
                  <td className="py-4 pr-3">
                    <div className="flex items-center gap-1.5">
                      <span className={`px-2 py-0.5 rounded text-xs font-bold ${getQualityColor(rev.quality_score)}`}>
                        {rev.quality_score}%
                      </span>
                    </div>
                  </td>

                  {/* Spam detection */}
                  <td className="py-4 pr-3">
                    {rev.is_fake ? (
                      <div 
                        className="text-rose-500 bg-rose-50 dark:bg-rose-950/20 border border-rose-200/50 dark:border-rose-900/50 px-2 py-0.5 rounded text-xs font-bold inline-flex items-center gap-1 cursor-help group relative"
                        title={rev.fake_reasons.join(', ')}
                      >
                        <AlertTriangle className="w-3.5 h-3.5" />
                        Flagged
                        
                        {/* Hover Reasons Tooltip */}
                        <div className="absolute bottom-full mb-2 hidden group-hover:block bg-slate-900 text-white text-[10px] p-2.5 rounded-lg w-40 z-20 leading-relaxed shadow-lg left-1/2 -translate-x-1/2">
                          <p className="font-bold border-b border-slate-700 pb-1 mb-1 text-rose-400">Flag Reasons:</p>
                          {rev.fake_reasons.map((reason, i) => (
                            <p key={i}>• {reason}</p>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <span className="text-slate-400 text-xs">-</span>
                    )}
                  </td>

                  {/* Review Text Body */}
                  <td className="py-4 text-xs text-slate-600 dark:text-slate-400 leading-relaxed break-words max-w-sm">
                    {rev.text}
                    {rev.keywords && rev.keywords.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {rev.keywords.map((k, idx) => (
                          <span key={idx} className="bg-slate-100 dark:bg-slate-850 px-1.5 py-0.5 rounded text-[9px] text-slate-400 inline-flex items-center gap-0.5">
                            <Tag className="w-2 h-2" />
                            {k}
                          </span>
                        ))}
                      </div>
                    )}
                  </td>
                </tr>
              ))}

              {filteredReviews.length === 0 && (
                <tr>
                  <td colSpan="6" className="text-center py-8 text-slate-400 dark:text-slate-650">
                    No reviews match the selected filter query.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
