/**
 * Zustand store for Sathik AI Direction Mode state management
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { QueryMode, SubmodeStyle, AnswerFormat, QueryResponse, DirectionModeStatus, ModeInfo, SubmodeInfo, FormatInfo } from '../types';

interface AppState {
  // Current settings
  currentMode: QueryMode;
  currentSubmode: SubmodeStyle;
  currentFormat: AnswerFormat;
  
  // UI state
  isLoading: boolean;
  currentResponse: QueryResponse | null;
  error: string | null;
  queryHistory: QueryResponse[];
  
  // System info
  systemStatus: DirectionModeStatus | null;
  availableModes: ModeInfo[];
  availableSubmodes: Record<string, SubmodeInfo>;
  availableFormats: Record<string, FormatInfo>;
  
  // Actions
  setCurrentMode: (mode: QueryMode) => void;
  setCurrentSubmode: (submode: SubmodeStyle) => void;
  setCurrentFormat: (format: AnswerFormat) => void;
  setLoading: (loading: boolean) => void;
  setCurrentResponse: (response: QueryResponse | null) => void;
  setError: (error: string | null) => void;
  addToHistory: (response: QueryResponse) => void;
  setSystemStatus: (status: DirectionModeStatus | null) => void;
  setAvailableModes: (modes: ModeInfo[]) => void;
  setAvailableSubmodes: (submodes: Record<string, SubmodeInfo>) => void;
  setAvailableFormats: (formats: Record<string, FormatInfo>) => void;
  clearHistory: () => void;
}

export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        currentMode: QueryMode.DIRECTION,
        currentSubmode: SubmodeStyle.NORMAL,
        currentFormat: AnswerFormat.COMPREHENSIVE,
        isLoading: false,
        currentResponse: null,
        error: null,
        queryHistory: [],
        systemStatus: null,
        availableModes: [],
        availableSubmodes: {},
        availableFormats: {},
        
        // Actions
        setCurrentMode: (mode) => set({ currentMode: mode }),
        setCurrentSubmode: (submode) => set({ currentSubmode: submode }),
        setCurrentFormat: (format) => set({ currentFormat: format }),
        setLoading: (loading) => set({ isLoading: loading }),
        setCurrentResponse: (response) => set({ currentResponse: response }),
        setError: (error) => set({ error }),
        addToHistory: (response) => 
          set((state) => ({
            queryHistory: [response, ...state.queryHistory.slice(0, 49)] // Keep last 50
          })),
        setSystemStatus: (status) => set({ systemStatus: status }),
        setAvailableModes: (modes) => set({ availableModes: modes }),
        setAvailableSubmodes: (submodes) => set({ availableSubmodes: submodes }),
        setAvailableFormats: (formats) => set({ availableFormats: formats }),
        clearHistory: () => set({ queryHistory: [] })
      }),
      {
        name: 'sathik-ai-store',
        partialize: (state) => ({
          currentMode: state.currentMode,
          currentSubmode: state.currentSubmode,
          currentFormat: state.currentFormat,
          queryHistory: state.queryHistory
        })
      }
    )
  )
);