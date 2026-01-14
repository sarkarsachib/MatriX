import React from 'react';
import { SubmodeStyle } from '../types';
import { useAppStore } from '../store/store';

export const SubmodeSelector: React.FC = () => {
  const { currentSubmode, setCurrentSubmode, availableSubmodes } = useAppStore();

  const getSubmodeDisplay = (key: string, info: any) => {
    const displays = {
      [SubmodeStyle.NORMAL]: {
        label: 'Normal',
        description: 'Standard response',
        emoji: '💬',
        color: 'text-gray-400'
      },
      [SubmodeStyle.SUGARCOTTED]: {
        label: 'Sugarcotted',
        description: 'Sweet and positive',
        emoji: '🍬',
        color: 'text-pink-400'
      },
      [SubmodeStyle.UNHINGED]: {
        label: 'Unhinged',
        description: 'Raw and honest',
        emoji: '🔥',
        color: 'text-red-400'
      },
      [SubmodeStyle.REAPER]: {
        label: 'Reaper',
        description: 'Dark and existential',
        emoji: '☠️',
        color: 'text-gray-300'
      },
      [SubmodeStyle.HEXAGON]: {
        label: '666',
        description: 'Chaotic and demonic',
        emoji: '👹',
        color: 'text-purple-400'
      }
    };
    return displays[key] || {
      label: info.name,
      description: info.description,
      emoji: info.emoji,
      color: 'text-white'
    };
  };

  return (
    <div className="grid grid-cols-1 gap-3">
      {Object.entries(availableSubmodes).map(([key, info]) => {
        const display = getSubmodeDisplay(key, info);
        const isSelected = currentSubmode === key;
        
        return (
          <button
            key={key}
            onClick={() => setCurrentSubmode(key as SubmodeStyle)}
            className={`p-4 rounded-xl border-2 transition-all duration-200 text-left ${
              isSelected
                ? 'border-purple-400 bg-purple-500/20 shadow-lg'
                : 'border-white/20 bg-white/5 hover:border-purple-300 hover:bg-white/10'
            }`}
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">{display.emoji}</span>
              <div>
                <div className={`font-semibold ${display.color}`}>
                  {display.label}
                </div>
                <div className="text-sm text-purple-200">
                  {display.description}
                </div>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
};