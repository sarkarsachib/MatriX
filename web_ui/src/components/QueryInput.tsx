import React, { useState } from 'react';
import { QueryMode } from '../types';

interface QueryInputProps {
  onSubmit: (query: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export const QueryInput: React.FC<QueryInputProps> = ({
  onSubmit,
  disabled = false,
  placeholder = "Ask me anything..."
}) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !disabled) {
      onSubmit(query.trim());
      setQuery('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="relative">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          className="w-full h-32 bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-purple-300 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
          style={{ minHeight: '120px' }}
        />
        <div className="absolute bottom-3 right-3 text-xs text-purple-300">
          Press Enter to submit • Shift+Enter for new line
        </div>
      </div>
      
      <div className="flex justify-between items-center">
        <div className="text-sm text-purple-300">
          Current mode: <span className="text-white font-medium">
            {QueryMode.DIRECTION === 'direction' ? '🔍 Direction' : '🧠 Trained'}
          </span>
        </div>
        
        <button
          type="submit"
          disabled={disabled || !query.trim()}
          className={`px-6 py-3 rounded-xl font-medium transition-all duration-200 ${
            disabled || !query.trim()
              ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
          }`}
        >
          {disabled ? (
            <span className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Processing...
            </span>
          ) : (
            'Ask Question'
          )}
        </button>
      </div>
    </form>
  );
};