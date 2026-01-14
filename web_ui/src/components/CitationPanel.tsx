import React from 'react';
import { CitationInfo } from '../types';

interface CitationPanelProps {
  citations: CitationInfo[];
}

export const CitationPanel: React.FC<CitationPanelProps> = ({ citations }) => {
  const getSourceColor = (source: string) => {
    const colors = {
      'wikipedia': 'text-blue-400',
      'google': 'text-red-400',
      'newsapi': 'text-green-400',
      'arxiv': 'text-purple-400',
      'duckduckgo': 'text-yellow-400',
    };
    return colors[source.toLowerCase()] || 'text-gray-400';
  };

  return (
    <div className="space-y-3">
      {citations.map((citation) => (
        <div
          key={citation.number}
          className="bg-white/5 border border-white/10 rounded-lg p-4 hover:bg-white/10 transition-colors"
        >
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="bg-purple-600 text-white text-xs font-bold px-2 py-1 rounded">
                [{citation.number}]
              </span>
              <span className={`text-sm font-medium ${getSourceColor(citation.source)}`}>
                {citation.source}
              </span>
            </div>
            <div className="text-xs text-purple-300">
              {Math.round(citation.confidence * 100)}%
            </div>
          </div>
          
          <div className="text-sm text-purple-200 mb-3">
            {citation.fact}
          </div>
          
          {citation.url && (
            <div className="flex items-center gap-2">
              <a
                href={citation.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-purple-400 hover:text-purple-300 underline break-all"
              >
                {citation.url}
              </a>
              <span className="text-xs text-purple-500">↗</span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};