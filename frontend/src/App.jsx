import React, { useState, useEffect } from 'react';
import { Sun, Moon, ArrowLeftRight, BarChart3, Database, Trash2, BrainCircuit } from 'lucide-react';

import InputForm from './components/InputForm';
import Dashboard from './components/Dashboard';
import Compare from './components/Compare';

import { getApiBase } from './utils/apiBase';

// In dev: '' → requests go to /api/... and Vite proxies to uvicorn on :8000
const API_HOST = getApiBase();

function App() {
  const [activeView, setActiveView] = useState('input'); // input, dashboard, compare
  const [analysisData, setAnalysisData] = useState(null);
  const [productsList, setProductsList] = useState([]);
  const [darkMode, setDarkMode] = useState(true);

  // Initialize Dark Mode by default
  useEffect(() => {
    const root = window.document.documentElement;
    if (darkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [darkMode]);

  // Fetch previous sessions list from backend SQLite database
  const fetchProducts = async () => {
    try {
      const response = await fetch(`${API_HOST}/api/products`);
      if (response.ok) {
        const data = await response.json();
        setProductsList(data);
      }
    } catch (err) {
      console.error('Failed to retrieve history sessions:', err);
    }
  };

  useEffect(() => {
    fetchProducts();
  }, [activeView]);

  const handleAnalysisComplete = (data) => {
    setAnalysisData(data);
    setActiveView('dashboard');
  };

  const handleLoadSession = async (productId) => {
    try {
      const response = await fetch(`${API_HOST}/api/analysis/${productId}`);
      if (response.ok) {
        const data = await response.json();
        setAnalysisData(data);
        setActiveView('dashboard');
      }
    } catch (err) {
      console.error('Failed to load session:', err);
    }
  };

  const handleDeleteSession = async (productId, e) => {
    e.stopPropagation();
    if (!confirm('Are you sure you want to delete this analysis session?')) return;
    try {
      const response = await fetch(`${API_HOST}/api/analysis/${productId}`, {
        method: 'DELETE'
      });
      if (response.ok) {
        fetchProducts();
      }
    } catch (err) {
      console.error('Failed to delete session:', err);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100 transition-colors duration-300">
      
      {/* Glow Effects (Premium styling elements) */}
      <div className="fixed top-[-10%] left-[-10%] w-[50%] h-[50%] bg-brand-500/5 rounded-full blur-[120px] pointer-events-none" />
      <div className="fixed bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-violet-500/5 rounded-full blur-[120px] pointer-events-none" />

      {/* Main Navbar */}
      <header className="border-b border-slate-200/60 dark:border-slate-900 bg-white/70 dark:bg-slate-950/70 backdrop-blur-md sticky top-0 z-40 transition-colors">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5 cursor-pointer" onClick={() => setActiveView('input')}>
            <div className="p-2 bg-brand-600 rounded-xl text-white shadow-md shadow-brand-500/10">
              <BrainCircuit className="w-5 h-5" />
            </div>
            <div>
              <h1 className="font-black text-base tracking-tight text-slate-800 dark:text-white leading-none">
                ReviewIntellect
              </h1>
              <span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest mt-1 block">
                Amazon AI Dashboard
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="flex items-center gap-6">
            <button
              onClick={() => setActiveView('input')}
              className={`text-sm font-semibold transition-all ${
                activeView === 'input' || activeView === 'dashboard'
                  ? 'text-brand-600 dark:text-brand-400'
                  : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200'
              }`}
            >
              Analyze
            </button>
            <button
              onClick={() => setActiveView('compare')}
              className={`text-sm font-semibold flex items-center gap-1.5 transition-all ${
                activeView === 'compare'
                  ? 'text-brand-600 dark:text-brand-400'
                  : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200'
              }`}
            >
              <ArrowLeftRight className="w-3.5 h-3.5" />
              Compare Mode
            </button>
            
            <div className="h-4 w-[1px] bg-slate-200 dark:bg-slate-850" />

            {/* Dark Mode Switch */}
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 hover:bg-slate-100 dark:hover:bg-slate-900 rounded-xl text-slate-500 dark:text-slate-400 transition-all cursor-pointer"
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </nav>
        </div>
      </header>

      {/* Main Body Layout */}
      <main className="max-w-7xl mx-auto px-6 py-10 relative z-10">
        {activeView === 'input' && (
          <div className="space-y-12">
            
            {/* Tagline Intro */}
            <div className="text-center max-w-xl mx-auto space-y-3">
              <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight text-slate-850 dark:text-white leading-tight">
                Amazon Review Analysis Powered by Machine Learning
              </h2>
              <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
                Ingest reviews via direct text pasting, CSV file uploads, or URL endpoints. The system extracts aspects dynamically, runs emotion mapping, and filters out spam reviews.
              </p>
            </div>

            {/* Form Component */}
            <InputForm 
              onAnalysisComplete={handleAnalysisComplete}
              apiHost={API_HOST}
            />

            {/* History Sessions List */}
            {productsList.length > 0 && (
              <div className="max-w-4xl mx-auto pt-6">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-450 dark:text-slate-500 mb-4 flex items-center gap-2">
                  <Database className="w-4 h-4" />
                  Previously Analyzed Products ({productsList.length})
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {productsList.map(prod => (
                    <div
                      key={prod.id}
                      onClick={() => handleLoadSession(prod.id)}
                      className="glass-card p-4 border border-slate-200/50 dark:border-slate-850 hover:border-brand-500/40 dark:hover:border-slate-800 transition-all cursor-pointer flex justify-between items-center group relative overflow-hidden"
                    >
                      <div className="pr-4 truncate flex-1">
                        <h4 className="font-extrabold text-sm text-slate-800 dark:text-slate-100 group-hover:text-brand-600 dark:group-hover:text-brand-400 truncate leading-snug">
                          {prod.title}
                        </h4>
                        <span className="text-[10px] text-slate-400 dark:text-slate-500 mt-1 block">
                          ASIN: {prod.id} | {prod.reviews_count} entries | {prod.category || 'General'}
                        </span>
                      </div>
                      
                      {/* Delete icon */}
                      <button
                        onClick={(e) => handleDeleteSession(prod.id, e)}
                        className="p-1.5 hover:bg-rose-50 dark:hover:bg-rose-950/20 text-slate-400 hover:text-rose-500 rounded-lg transition-all cursor-pointer"
                        title="Delete Session"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeView === 'dashboard' && analysisData && (
          <Dashboard 
            data={analysisData}
            onBack={() => setActiveView('input')}
            apiHost={API_HOST}
          />
        )}

        {activeView === 'compare' && (
          <Compare 
            productsList={productsList}
            apiHost={API_HOST}
          />
        )}
      </main>
    </div>
  );
}

export default App;
