import React from 'react';
import { QueryMode } from '../types';
import { useAppStore } from '../store/store';

export const ModeSelector: React.FC = () => {
  const { currentMode, setCurrentMode, availableModes } = useAppStore();

  const modes = [
    {
      value: QueryMode.DIRECTION,
      label: 'Direction Mode',
      description: 'RAG-based search and retrieval',
      emoji: '🔍',
      color: 'from-blue-600 to-purple-600'
    },
    {
      value: QueryMode.TRAINED,
      label: 'Trained Mode',
      description: 'Neural network inference',
      emoji: '🧠',
      color: 'from-green-600 to-teal-600'
    }
  ];

  return (
    <div className="space-y-3">
      {modes.map((mode) => (
        <button
          key={mode.value}
          onClick={() => setCurrentMode(mode.value)}
          disabled={!availableModes.find(m => m.name === mode.value)?.available}
          className={`w-full p-4 rounded-xl border-2 transition-all duration-200 text-left ${
            currentMode === mode.value
              ? `border-purple-400 bg-gradient-to-r ${mode.color} text-white shadow-lg`
              : 'border-white/20 bg-white/5 text-purple-200 hover:border-purple-300 hover:bg-white/10'
          } ${!availableModes.find(m => m.name === mode.value)?.available ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <div className="flex items-center gap-3">
            <span className="text-2xl">{mode.emoji}</span>
            <div>
              <div className="font-semibold">{mode.label}</div>
              <div className="text-sm opacity-80">{mode.description}</div>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
};