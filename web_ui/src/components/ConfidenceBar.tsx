import React from 'react';

interface ConfidenceBarProps {
  confidence: number;
  label?: string;
}

export const ConfidenceBar: React.FC<ConfidenceBarProps> = ({ confidence, label = "Confidence" }) => {
  const percentage = Math.round(confidence * 100);
  
  const getColorClass = (conf: number) => {
    if (conf >= 0.8) return 'from-green-500 to-emerald-500';
    if (conf >= 0.6) return 'from-yellow-500 to-orange-500';
    if (conf >= 0.4) return 'from-orange-500 to-red-500';
    return 'from-red-500 to-red-600';
  };
  
  const getTextColor = (conf: number) => {
    if (conf >= 0.8) return 'text-green-400';
    if (conf >= 0.6) return 'text-yellow-400';
    if (conf >= 0.4) return 'text-orange-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-sm text-purple-300">{label}</span>
        <span className={`text-sm font-medium ${getTextColor(confidence)}`}>
          {percentage}%
        </span>
      </div>
      
      <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
        <div
          className={`h-full bg-gradient-to-r ${getColorClass(confidence)} transition-all duration-500 ease-out`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      
      <div className="text-xs text-purple-400">
        {confidence >= 0.8 ? 'High confidence' : 
         confidence >= 0.6 ? 'Moderate confidence' : 
         confidence >= 0.4 ? 'Low confidence' : 
         'Very low confidence'}
      </div>
    </div>
  );
};