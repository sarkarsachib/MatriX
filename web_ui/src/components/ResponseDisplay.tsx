import React from 'react';
import { QueryResponse } from '../types';
import { ConfidenceBar } from './ConfidenceBar';

interface ResponseDisplayProps {
  response?: QueryResponse;
  loading: boolean;
  error?: string;
}

export const ResponseDisplay: React.FC<ResponseDisplayProps> = ({
  response,
  loading,
  error
}) => {
  if (loading) {
    return (
      <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10 text-center">
        <div className="animate-pulse">
          <div className="h-4 bg-white/10 rounded w-3/4 mx-auto mb-4"></div>
          <div className="h-4 bg-white/10 rounded w-1/2 mx-auto"></div>
        </div>
        <p className="text-purple-200 mt-4">Processing your query...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-6">
        <h3 className="text-red-400 font-semibold mb-2">❌ Error</h3>
        <p className="text-red-200">{error}</p>
      </div>
    );
  }

  if (!response) {
    return (
      <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10 text-center">
        <div className="text-6xl mb-4">🤖</div>
        <h3 className="text-xl font-semibold text-white mb-2">Welcome to Sathik AI</h3>
        <p className="text-purple-200">
          Ask me anything and I'll provide an answer using advanced AI processing.
        </p>
        <div className="mt-6 text-sm text-purple-300">
          <div className="flex flex-wrap justify-center gap-4">
            <span>🔍 Direction Mode: Search & Retrieve</span>
            <span>🧠 Trained Mode: Neural Networks</span>
            <span>🍬 Multiple Response Styles</span>
          </div>
        </div>
      </div>
    );
  }

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleTimeString();
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'text-green-400';
      case 'error': return 'text-red-400';
      case 'no_results': return 'text-yellow-400';
      case 'low_confidence': return 'text-orange-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success': return '✅';
      case 'error': return '❌';
      case 'no_results': return '⚠️';
      case 'low_confidence': return '🟡';
      default: return '❓';
    }
  };

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <span className="text-2xl">
            {response.mode === 'direction' ? '🔍' : '🧠'}
          </span>
          <div>
            <h3 className="text-lg font-semibold text-white">
              {response.query}
            </h3>
            <div className="flex items-center gap-4 text-sm text-purple-300">
              <span>{response.mode} mode</span>
              <span>•</span>
              <span>{response.submode} style</span>
              <span>•</span>
              <span>{formatTimestamp(response.timestamp)}</span>
            </div>
          </div>
        </div>
        
        <div className="text-right">
          <div className={`text-sm font-medium ${getStatusColor(response.status)}`}>
            {getStatusIcon(response.status)} {response.status}
          </div>
          {response.cache_hit && (
            <div className="text-xs text-purple-400 mt-1">
              ⚡ Cached response
            </div>
          )}
        </div>
      </div>

      {/* Response Content */}
      <div className="mb-6">
        <div className="prose prose-invert max-w-none">
          <div className="text-white leading-relaxed whitespace-pre-wrap">
            {response.answer}
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white/5 rounded-lg p-3 text-center">
          <div className="text-lg font-semibold text-white">
            {Math.round(response.confidence * 100)}%
          </div>
          <div className="text-xs text-purple-300">Confidence</div>
        </div>
        
        <div className="bg-white/5 rounded-lg p-3 text-center">
          <div className="text-lg font-semibold text-white">
            {response.sources_used}
          </div>
          <div className="text-xs text-purple-300">Sources</div>
        </div>
        
        <div className="bg-white/5 rounded-lg p-3 text-center">
          <div className="text-lg font-semibold text-white">
            {response.facts_analyzed}
          </div>
          <div className="text-xs text-purple-300">Facts</div>
        </div>
        
        <div className="bg-white/5 rounded-lg p-3 text-center">
          <div className="text-lg font-semibold text-white">
            {response.processing_time.toFixed(1)}s
          </div>
          <div className="text-xs text-purple-300">Time</div>
        </div>
      </div>

      {/* Confidence Bar */}
      {response.confidence > 0 && (
        <div className="mb-4">
          <ConfidenceBar 
            confidence={response.confidence} 
            label="Overall Confidence"
          />
        </div>
      )}

      {/* Additional Info for Direction Mode */}
      {response.mode === 'direction' && response.validation_results && (
        <div className="bg-white/5 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-white mb-3">🔬 Analysis Results</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-purple-300">Total Facts:</div>
              <div className="text-white font-medium">{response.validation_results.total_facts}</div>
            </div>
            <div>
              <div className="text-purple-300">Valid Facts:</div>
              <div className="text-white font-medium">{response.validation_results.valid_facts}</div>
            </div>
            <div>
              <div className="text-purple-300">Avg Confidence:</div>
              <div className="text-white font-medium">
                {Math.round(response.validation_results.average_confidence * 100)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Query Analysis */}
      {response.query_analysis && (
        <div className="mt-4 bg-white/5 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-white mb-3">🔍 Query Analysis</h4>
          <div className="text-sm text-purple-300">
            <div>Type: <span className="text-white">{response.query_analysis.type}</span></div>
            <div>Confidence: <span className="text-white">{Math.round(response.query_analysis.confidence * 100)}%</span></div>
            {Object.keys(response.query_analysis.entities).length > 0 && (
              <div className="mt-2">
                <span className="text-purple-300">Entities: </span>
                <span className="text-white">
                  {Object.entries(response.query_analysis.entities)
                    .filter(([_, entities]) => entities.length > 0)
                    .map(([type, entities]) => `${type}: ${entities.join(', ')}`)
                    .join('; ')}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};