import React, { useState, useEffect } from 'react';
import { QueryInput } from './components/QueryInput';
import { ModeSelector } from './components/ModeSelector';
import { SubmodeSelector } from './components/SubmodeSelector';
import { ResponseDisplay } from './components/ResponseDisplay';
import { CitationPanel } from './components/CitationPanel';
import { ConfidenceBar } from './components/ConfidenceBar';
import { LoadingSpinner } from './components/LoadingSpinner';
import { useAppStore } from './store/store';
import ApiService from './services/api';
import { QueryMode, SubmodeStyle, AnswerFormat } from './types';
import './styles/globals.css';

function App() {
  const {
    currentMode,
    currentSubmode,
    currentFormat,
    isLoading,
    currentResponse,
    error,
    systemStatus,
    availableSubmodes,
    availableFormats,
    setLoading,
    setCurrentResponse,
    setError,
    setSystemStatus,
    setAvailableSubmodes,
    setAvailableFormats,
    addToHistory
  } = useAppStore();

  const [query, setQuery] = useState('');

  // Load available options on component mount
  useEffect(() => {
    loadSystemInfo();
  }, []);

  const loadSystemInfo = async () => {
    try {
      setLoading(true);
      
      // Load system status and available options
      const [statusResponse, modesResponse] = await Promise.all([
        ApiService.getSystemStatus(),
        ApiService.getAvailableModes()
      ]);
      
      setSystemStatus(statusResponse);
      setAvailableSubmodes(modesResponse.submodes);
      setAvailableFormats(modesResponse.formats);
      
    } catch (err) {
      console.error('Failed to load system info:', err);
      setError('Failed to load system information');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitQuery = async (queryText: string) => {
    if (!queryText.trim()) return;

    try {
      setLoading(true);
      setError(null);
      setCurrentResponse(null);

      const response = await ApiService.submitQuery({
        query: queryText,
        mode: currentMode,
        submode: currentSubmode,
        format_type: currentFormat,
        user_id: 'web_user'
      });

      setCurrentResponse(response);
      addToHistory(response);
      
    } catch (err: any) {
      console.error('Query failed:', err);
      setError(err.response?.data?.detail || err.message || 'Query failed');
    } finally {
      setLoading(false);
    }
  };

  const getCurrentSubmodeInfo = () => {
    return availableSubmodes[currentSubmode];
  };

  const getCurrentFormatInfo = () => {
    return availableFormats[currentFormat];
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="bg-black/20 backdrop-blur-sm border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">
                🔥 Sathik AI Direction Mode
              </h1>
              <p className="text-purple-200">
                RAG-based query processing with multiple response styles
              </p>
            </div>
            
            {/* System Status */}
            <div className="text-right">
              {systemStatus && (
                <div className="text-sm text-purple-200">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${
                      systemStatus.status === 'operational' ? 'bg-green-400' : 'bg-red-400'
                    }`} />
                    <span>{systemStatus.status}</span>
                  </div>
                  <div>v{systemStatus.version}</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Controls */}
          <div className="lg:col-span-1 space-y-6">
            {/* Mode Selector */}
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <h3 className="text-lg font-semibold text-white mb-4">Processing Mode</h3>
              <ModeSelector />
            </div>

            {/* Sub-mode Selector */}
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <h3 className="text-lg font-semibold text-white mb-4">
                Response Style
                {getCurrentSubmodeInfo() && (
                  <span className="ml-2 text-2xl">{getCurrentSubmodeInfo().emoji}</span>
                )}
              </h3>
              <SubmodeSelector />
            </div>

            {/* Format Selector */}
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <h3 className="text-lg font-semibold text-white mb-4">Answer Format</h3>
              <select
                value={currentFormat}
                onChange={(e) => useAppStore.getState().setCurrentFormat(e.target.value as AnswerFormat)}
                className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                {Object.entries(availableFormats).map(([key, format]) => (
                  <option key={key} value={key} className="bg-gray-800">
                    {format.name} - {format.description}
                  </option>
                ))}
              </select>
            </div>

            {/* System Stats */}
            {systemStatus && (
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
                <h3 className="text-lg font-semibold text-white mb-4">System Stats</h3>
                <div className="space-y-2 text-sm text-purple-200">
                  <div className="flex justify-between">
                    <span>Total Queries:</span>
                    <span>{systemStatus.metrics.total_queries}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Cache Hit Rate:</span>
                    <span>{Math.round(systemStatus.metrics.cache_hit_rate * 100)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Avg Confidence:</span>
                    <span>{Math.round(systemStatus.metrics.average_confidence * 100)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>KB Size:</span>
                    <span>{systemStatus.knowledge_base.database_size_mb.toFixed(1)} MB</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Query and Response */}
          <div className="lg:col-span-2 space-y-6">
            {/* Query Input */}
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
              <QueryInput
                onSubmit={handleSubmitQuery}
                disabled={isLoading}
                placeholder="Ask me anything..."
              />
            </div>

            {/* Loading State */}
            {isLoading && (
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10 text-center">
                <LoadingSpinner size="large" />
                <p className="text-purple-200 mt-4">
                  Processing your query with {currentMode} mode...
                </p>
              </div>
            )}

            {/* Error State */}
            {error && (
              <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-6">
                <h3 className="text-red-400 font-semibold mb-2">Error</h3>
                <p className="text-red-200">{error}</p>
                <button
                  onClick={() => setError(null)}
                  className="mt-4 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-200 rounded-lg transition-colors"
                >
                  Dismiss
                </button>
              </div>
            )}

            {/* Response Display */}
            <ResponseDisplay
              response={currentResponse || undefined}
              loading={isLoading}
              error={error}
            />

            {/* Citations */}
            {currentResponse?.citations && currentResponse.citations.length > 0 && (
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
                <h3 className="text-lg font-semibold text-white mb-4">📚 Citations</h3>
                <CitationPanel citations={currentResponse.citations} />
              </div>
            )}

            {/* Query History */}
            {useAppStore.getState().queryHistory.length > 0 && (
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
                <h3 className="text-lg font-semibold text-white mb-4">Recent Queries</h3>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {useAppStore.getState().queryHistory.slice(0, 5).map((response, index) => (
                    <div key={index} className="text-sm text-purple-200 p-2 bg-white/5 rounded">
                      <div className="font-medium">{response.query}</div>
                      <div className="text-xs opacity-70">
                        {response.mode} mode • {Math.round(response.confidence * 100)}% confidence
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;