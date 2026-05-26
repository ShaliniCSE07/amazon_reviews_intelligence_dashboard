import React, { useState } from 'react';
import { Clipboard, FileSpreadsheet, Globe, UploadCloud, AlertCircle } from 'lucide-react';
import * as XLSX from 'xlsx';

const InputForm = ({ onAnalysisComplete, apiHost }) => {
  const [activeTab, setActiveTab] = useState('url'); // url, paste, csv/excel
  const [urlOrAsin, setUrlOrAsin] = useState('');
  const [maxReviews, setMaxReviews] = useState(30);
  const [pastedText, setPastedText] = useState('');
  const [productName, setProductName] = useState('');
  const [category, setCategory] = useState('Electronics');
  
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [error, setError] = useState(null);

  // CSV / Excel parsing state
  const [uploadedFile, setUploadedFile] = useState(null);
  const [csvHeaders, setCsvHeaders] = useState([]);
  const [csvRows, setCsvRows] = useState([]);
  const [selectedReviewCol, setSelectedReviewCol] = useState('');
  const [selectedRatingCol, setSelectedRatingCol] = useState('');
  const [selectedDateCol, setSelectedDateCol] = useState('');

  const handleUrlSubmit = async (e) => {
    e.preventDefault();
    if (!urlOrAsin.trim()) return;

    setLoading(true);
    setError(null);
    setLoadingStep('Connecting to Amazon reviews gateway...');

    setTimeout(() => setLoadingStep('Downloading review pages...'), 1500);
    setTimeout(() => setLoadingStep('Extracting sentiments and text features...'), 3500);
    setTimeout(() => setLoadingStep('Clustering product aspects...'), 6000);
    setTimeout(() => setLoadingStep('Generating executive summary...'), 8500);

    try {
      const response = await fetch(`${apiHost}/api/analyze/asin`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url_or_asin: urlOrAsin.trim(),
          max_reviews: maxReviews
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to complete review analysis');
      }

      const data = await response.json();
      onAnalysisComplete(data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'An error occurred while fetching reviews. Make sure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handlePasteSubmit = async (e) => {
    e.preventDefault();
    if (!pastedText.trim() || !productName.trim()) return;

    // Split reviews by double newlines or single newlines
    const rawList = pastedText.split(/\n\s*\n/);
    const reviewsList = rawList
      .map(r => r.trim())
      .filter(r => r.length > 5);

    if (reviewsList.length === 0) {
      setError('Please paste at least one review containing text (min 5 characters).');
      return;
    }

    setLoading(true);
    setError(null);
    setLoadingStep('Processing pasted texts...');
    setTimeout(() => setLoadingStep('Extracting aspect-level sentiments...'), 2000);

    try {
      const response = await fetch(`${apiHost}/api/analyze/paste`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          product_name: productName.trim(),
          category: category,
          reviews: reviewsList
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to analyze pasted reviews');
      }

      const data = await response.json();
      onAnalysisComplete(data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'An error occurred while analyzing pasted reviews.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadedFile(file);
    setError(null);
    // Reset previous parsed state
    setCsvHeaders([]);
    setCsvRows([]);
    setSelectedReviewCol('');
    setSelectedRatingCol('');
    setSelectedDateCol('');

    const ext = file.name.split('.').pop().toLowerCase();

    if (ext === 'xlsx' || ext === 'xls') {
      // --- Excel parsing via SheetJS ---
      const reader = new FileReader();
      reader.onload = (evt) => {
        try {
          const workbook = XLSX.read(evt.target.result, { type: 'array' });
          const sheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[sheetName];
          // sheet_to_json with header:1 returns array of arrays
          const rawRows = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: '' });

          if (rawRows.length < 2) {
            setError('Excel file is empty or has insufficient rows.');
            return;
          }

          const headers = rawRows[0].map(h => String(h).trim());
          const dataRows = rawRows.slice(1).map(row =>
            headers.map((_, i) => String(row[i] ?? '').trim())
          );

          setCsvHeaders(headers);
          setCsvRows(dataRows);

          // Smart column matching
          const reviewIndex = headers.findIndex(h => /review|text|comment|body/i.test(h));
          const ratingIndex = headers.findIndex(h => /rating|star|score/i.test(h));
          const dateIndex = headers.findIndex(h => /date|time/i.test(h));

          if (reviewIndex !== -1) setSelectedReviewCol(headers[reviewIndex]);
          if (ratingIndex !== -1) setSelectedRatingCol(headers[ratingIndex]);
          if (dateIndex !== -1) setSelectedDateCol(headers[dateIndex]);
        } catch (err) {
          setError('Failed to parse Excel file. Please ensure it is a valid .xlsx or .xls file.');
        }
      };
      reader.readAsArrayBuffer(file);
    } else {
      // --- CSV parsing ---
      const reader = new FileReader();
      reader.onload = (evt) => {
        const text = evt.target.result;
        const rows = [];
        const lines = text.split(/\r?\n/);

        lines.forEach((line) => {
          if (!line.trim()) return;
          const result = [];
          let current = '';
          let inQuotes = false;

          for (let i = 0; i < line.length; i++) {
            const char = line[i];
            if (char === '"') {
              inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
              result.push(current.trim());
              current = '';
            } else {
              current += char;
            }
          }
          result.push(current.trim());
          rows.push(result);
        });

        if (rows.length < 2) {
          setError('CSV file is empty or has insufficient rows.');
          return;
        }

        const headers = rows[0].map(h => h.replace(/^["']|["']$/g, ''));
        setCsvHeaders(headers);
        setCsvRows(rows.slice(1));

        // Smart column matching
        const reviewIndex = headers.findIndex(h => /review|text|comment|body/i.test(h));
        const ratingIndex = headers.findIndex(h => /rating|star|score/i.test(h));
        const dateIndex = headers.findIndex(h => /date|time/i.test(h));

        if (reviewIndex !== -1) setSelectedReviewCol(headers[reviewIndex]);
        if (ratingIndex !== -1) setSelectedRatingCol(headers[ratingIndex]);
        if (dateIndex !== -1) setSelectedDateCol(headers[dateIndex]);
      };
      reader.readAsText(file);
    }
  };

  const handleCsvSubmit = async (e) => {
    e.preventDefault();
    if (!uploadedFile || !selectedReviewCol || !productName.trim()) {
      setError('Please provide a product name and map the review text column.');
      return;
    }

    const reviewIdx = csvHeaders.indexOf(selectedReviewCol);
    const ratingIdx = csvHeaders.indexOf(selectedRatingCol);
    const dateIdx = csvHeaders.indexOf(selectedDateCol);

    const reviews = [];
    const ratings = [];
    const dates = [];

    csvRows.forEach(row => {
      const text = row[reviewIdx];
      if (text && text.trim().length > 3) {
        reviews.push(text.replace(/^["']|["']$/g, '').trim());
        
        if (ratingIdx !== -1 && row[ratingIdx]) {
          const r = parseInt(row[ratingIdx].replace(/^["']|["']$/g, ''), 10);
          ratings.push(isNaN(r) ? 5 : r);
        }
        
        if (dateIdx !== -1 && row[dateIdx]) {
          dates.push(row[dateIdx].replace(/^["']|["']$/g, '').trim());
        }
      }
    });

    if (reviews.length === 0) {
      setError('No valid review text extracted from CSV based on column mapping.');
      return;
    }

    setLoading(true);
    setError(null);
    const fileExt = uploadedFile.name.split('.').pop().toUpperCase();
    setLoadingStep(`Uploading and parsing ${fileExt} entries...`);
    setTimeout(() => setLoadingStep('Analyzing sentiment and clusters...'), 2000);

    try {
      const bodyPayload = {
        product_name: productName.trim(),
        category: category,
        reviews: reviews.slice(0, 100) // Limit to top 100 for safety
      };

      if (ratings.length === reviews.length) bodyPayload.ratings = ratings.slice(0, 100);
      if (dates.length === reviews.length) bodyPayload.dates = dates.slice(0, 100);

      const response = await fetch(`${apiHost}/api/analyze/paste`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(bodyPayload)
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to analyze CSV reviews');
      }

      const data = await response.json();
      onAnalysisComplete(data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'An error occurred while uploading reviews.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Tab Navigation */}
      <div className="flex border-b border-slate-200 dark:border-slate-800 mb-8 p-1 bg-slate-100 dark:bg-slate-900 rounded-xl max-w-md mx-auto">
        <button
          onClick={() => { setActiveTab('url'); setError(null); }}
          className={`flex-1 py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 text-sm font-semibold transition-all duration-200 ${
            activeTab === 'url'
              ? 'bg-white dark:bg-slate-800 text-brand-600 dark:text-brand-400 shadow-sm'
              : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
          }`}
        >
          <Globe className="w-4 h-4" />
          Amazon Link/ASIN
        </button>
        <button
          onClick={() => { setActiveTab('paste'); setError(null); }}
          className={`flex-1 py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 text-sm font-semibold transition-all duration-200 ${
            activeTab === 'paste'
              ? 'bg-white dark:bg-slate-800 text-brand-600 dark:text-brand-400 shadow-sm'
              : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
          }`}
        >
          <Clipboard className="w-4 h-4" />
          Paste Text
        </button>
        <button
          onClick={() => { setActiveTab('csv'); setError(null); }}
          className={`flex-1 py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 text-sm font-semibold transition-all duration-200 ${
            activeTab === 'csv'
              ? 'bg-white dark:bg-slate-800 text-brand-600 dark:text-brand-400 shadow-sm'
              : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
          }`}
        >
          <FileSpreadsheet className="w-4 h-4" />
          Upload CSV / Excel
        </button>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="mb-6 p-4 bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900/50 rounded-xl flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-500 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-rose-800 dark:text-rose-400 text-sm">Analysis Failed</h4>
            <p className="text-rose-600 dark:text-rose-500 text-sm mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Forms Card */}
      <div className="glass-card p-8 border border-slate-200/60 dark:border-slate-850 shadow-xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 bg-brand-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-0 w-32 h-32 bg-violet-500/10 rounded-full blur-3xl" />

        {/* Tab 1: URL or ASIN */}
        {activeTab === 'url' && (
          <form onSubmit={handleUrlSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                Amazon URL or 10-Digit ASIN
              </label>
              <input
                type="text"
                value={urlOrAsin}
                onChange={(e) => setUrlOrAsin(e.target.value)}
                placeholder="e.g. B09G96T67S or https://www.amazon.com/dp/..."
                className="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 focus:outline-none focus:ring-2 focus:ring-brand-500 transition-all text-sm"
                required
              />
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-2">
                Enter an Amazon product ASIN. If scraped results are throttled, simulated high-quality reviews corresponding to the category (Electronics, Footwear, or Kitchen) will automatically bootstrap the visualization.
              </p>
            </div>

            <div>
              <div className="flex justify-between mb-2">
                <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  Maximum reviews to process
                </label>
                <span className="text-xs font-semibold text-brand-600 dark:text-brand-400">{maxReviews} reviews</span>
              </div>
              <input
                type="range"
                min="10"
                max="100"
                step="5"
                value={maxReviews}
                onChange={(e) => setMaxReviews(parseInt(e.target.value))}
                className="w-full h-2 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-brand-500"
              />
            </div>

            <button
              type="submit"
              disabled={loading || !urlOrAsin.trim()}
              className="w-full py-3.5 bg-brand-600 hover:bg-brand-500 text-white rounded-xl font-semibold shadow-lg hover:shadow-brand-500/20 transition-all text-sm flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
            >
              Analyze Amazon Product
            </button>
          </form>
        )}

        {/* Tab 2: Paste Reviews */}
        {activeTab === 'paste' && (
          <form onSubmit={handlePasteSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                  Product Name
                </label>
                <input
                  type="text"
                  value={productName}
                  onChange={(e) => setProductName(e.target.value)}
                  placeholder="e.g. UltraBass Bluetooth Headphones"
                  className="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 focus:outline-none focus:ring-2 focus:ring-brand-500 transition-all text-sm"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                  Category
                </label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 focus:outline-none focus:ring-2 focus:ring-brand-500 transition-all text-sm dark:text-slate-300"
                >
                  <option value="Electronics">Electronics</option>
                  <option value="Footwear">Footwear</option>
                  <option value="Kitchen">Kitchen</option>
                  <option value="Home & Furniture">Home & Furniture</option>
                  <option value="General">General</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                Paste Reviews (Separate reviews with double newlines)
              </label>
              <textarea
                value={pastedText}
                onChange={(e) => setPastedText(e.target.value)}
                placeholder="Review 1: This battery lasts forever, screen looks gorgeous.&#10;&#10;Review 2: Extremely disappointing experience, the strap broke on the first day."
                rows="8"
                className="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 focus:outline-none focus:ring-2 focus:ring-brand-500 transition-all text-sm font-sans"
                required
              />
            </div>

            <button
              type="submit"
              disabled={loading || !pastedText.trim() || !productName.trim()}
              className="w-full py-3.5 bg-brand-600 hover:bg-brand-500 text-white rounded-xl font-semibold shadow-lg hover:shadow-brand-500/20 transition-all text-sm flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
            >
              Analyze Pasted Text
            </button>
          </form>
        )}

        {/* Tab 3: Upload CSV */}
        {activeTab === 'csv' && (
          <form onSubmit={handleCsvSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                  Product Name
                </label>
                <input
                  type="text"
                  value={productName}
                  onChange={(e) => setProductName(e.target.value)}
                  placeholder="e.g. Ergonomic Office Desk Chair"
                  className="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 focus:outline-none focus:ring-2 focus:ring-brand-500 transition-all text-sm"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                  Category
                </label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 focus:outline-none focus:ring-2 focus:ring-brand-500 transition-all text-sm dark:text-slate-300"
                >
                  <option value="Electronics">Electronics</option>
                  <option value="Footwear">Footwear</option>
                  <option value="Kitchen">Kitchen</option>
                  <option value="Home & Furniture">Home & Furniture</option>
                  <option value="General">General</option>
                </select>
              </div>
            </div>

            {/* File Drag and Drop */}
            <div className="border-2 border-dashed border-slate-300 dark:border-slate-800 rounded-xl p-8 text-center bg-slate-50/50 dark:bg-slate-900/30 hover:border-brand-500 hover:bg-slate-50/80 dark:hover:bg-slate-900/50 transition-all relative">
              <input
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <UploadCloud className="w-10 h-10 text-slate-400 dark:text-slate-600 mx-auto mb-3" />
              <p className="text-sm font-semibold text-slate-600 dark:text-slate-400">
                {uploadedFile ? uploadedFile.name : 'Drag & drop your CSV or Excel file here, or click to browse'}
              </p>
              <p className="text-xs text-slate-400 mt-1">Accepts CSV (.csv) and Excel (.xlsx, .xls) files</p>
            </div>

            {/* Column Mapping (If CSV loaded) */}
            {csvHeaders.length > 0 && (
              <div className="space-y-4 p-5 bg-slate-50 dark:bg-slate-900/60 rounded-xl border border-slate-200/50 dark:border-slate-850">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                  Map CSV Columns
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                      Review Text (Required)
                    </label>
                    <select
                      value={selectedReviewCol}
                      onChange={(e) => setSelectedReviewCol(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-800 text-xs bg-white dark:bg-slate-950 dark:text-slate-300 focus:outline-none"
                      required
                    >
                      <option value="">-- Select --</option>
                      {csvHeaders.map(h => (
                        <option key={h} value={h}>{h}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                      Rating (Optional)
                    </label>
                    <select
                      value={selectedRatingCol}
                      onChange={(e) => setSelectedRatingCol(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-800 text-xs bg-white dark:bg-slate-950 dark:text-slate-300 focus:outline-none"
                    >
                      <option value="">-- None --</option>
                      {csvHeaders.map(h => (
                        <option key={h} value={h}>{h}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                      Review Date (Optional)
                    </label>
                    <select
                      value={selectedDateCol}
                      onChange={(e) => setSelectedDateCol(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-800 text-xs bg-white dark:bg-slate-950 dark:text-slate-300 focus:outline-none"
                    >
                      <option value="">-- None --</option>
                      {csvHeaders.map(h => (
                        <option key={h} value={h}>{h}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !uploadedFile || !selectedReviewCol || !productName.trim()}
              className="w-full py-3.5 bg-brand-600 hover:bg-brand-500 text-white rounded-xl font-semibold shadow-lg hover:shadow-brand-500/20 transition-all text-sm flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
            >
              Analyze Reviews
            </button>
          </form>
        )}
      </div>

      {/* Loading Glass overlay */}
      {loading && (
        <div className="fixed inset-0 bg-slate-950/40 dark:bg-slate-950/60 backdrop-blur-md z-50 flex items-center justify-center p-4">
          <div className="glass-card max-w-sm w-full p-8 text-center shadow-2xl border border-white/20 dark:border-slate-800/40 relative">
            <div className="w-16 h-16 border-4 border-brand-200 border-t-brand-600 rounded-full animate-spin mx-auto mb-6" />
            <h3 className="font-bold text-lg text-slate-800 dark:text-slate-100 mb-1">
              Analyzing Reviews
            </h3>
            <p className="text-slate-500 dark:text-slate-400 text-sm animate-pulse">
              {loadingStep}
            </p>
            <p className="text-xs text-slate-400 mt-6 leading-relaxed">
              ML classifiers are executing server-side. Larger batches can take a minute to process.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default InputForm;
