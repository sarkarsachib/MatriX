/**
 * TypeScript type definitions for Sathik AI Direction Mode
 */

export interface QueryRequest {
  query: string;
  user_id?: string;
  mode: QueryMode;
  submode: SubmodeStyle;
  format_type: AnswerFormat;
  output_mode?: string;
}

export interface CitationInfo {
  number: number;
  source: string;
  url: string;
  fact: string;
  confidence: number;
}

export interface SourceInfo {
  url: string;
  title: string;
  snippet: string;
  confidence: number;
}

export interface QueryAnalysis {
  type: string;
  confidence: number;
  entities: Record<string, string[]>;
}

export interface ValidationResults {
  total_facts: number;
  valid_facts: number;
  average_confidence: number;
}

export interface KeyInformation {
  main_facts: Array<{
    fact: string;
    confidence: number;
    source: string;
    url: string;
  }>;
  definitions: Array<{
    term: string;
    definition: string;
    confidence: number;
    source: string;
  }>;
  dates: string[];
  people: string[];
  places: string[];
  organizations: string[];
  quantitative_data: Array<{
    value: string;
    context: string;
    confidence: number;
  }>;
  sources: string[];
}

export interface QueryResponse {
  query: string;
  user_id: string;
  mode: string;
  submode: string;
  answer: string;
  confidence: number;
  sources_used: number;
  facts_analyzed: number;
  format: string;
  citations: CitationInfo[];
  key_information?: KeyInformation;
  query_analysis?: QueryAnalysis;
  validation_results?: ValidationResults;
  processing_time: number;
  cache_hit: boolean;
  status: string;
  timestamp: number;
  error?: string;
}

export enum QueryMode {
  TRAINED = 'trained',
  DIRECTION = 'direction'
}

export enum SubmodeStyle {
  NORMAL = 'normal',
  SUGARCOTTED = 'sugarcotted',
  UNHINGED = 'unhinged',
  REAPER = 'reaper',
  HEXAGON = '666'
}

export enum AnswerFormat {
  COMPREHENSIVE = 'comprehensive',
  SUMMARY = 'summary',
  BULLET_POINTS = 'bullet_points'
}

export interface SubmodeInfo {
  name: string;
  description: string;
  emoji: string;
  color: string;
  characteristics: string[];
}

export interface FormatInfo {
  name: string;
  description: string;
  max_length: number;
  includes_citations: boolean;
  includes_confidence: boolean;
}

export interface KnowledgeBaseStats {
  total_queries: number;
  total_facts: number;
  total_concepts: number;
  recent_queries_24h: number;
  average_confidence: number;
  top_sources: Array<{
    source: string;
    count: number;
  }>;
  popular_concepts: Array<{
    concept: string;
    popularity: number;
    last_accessed: number;
  }>;
  database_size_mb: number;
}

export interface DirectionModeStatus {
  system: string;
  status: string;
  version: string;
  components: Record<string, string>;
  metrics: Record<string, any>;
  knowledge_base: KnowledgeBaseStats;
  available_styles: Record<string, SubmodeInfo>;
  available_formats: Record<string, FormatInfo>;
  timestamp: number;
}

export interface ModeInfo {
  name: string;
  description: string;
  available: boolean;
}

export interface ModesResponse {
  modes: ModeInfo[];
  submodes: Record<string, SubmodeInfo>;
  formats: Record<string, FormatInfo>;
}

export interface HealthResponse {
  status: string;
  timestamp: number;
  version: string;
  components: Record<string, string>;
}

export interface SearchRequest {
  search_term: string;
  limit: number;
}

export interface SearchResult {
  concept: string;
  definition: string;
  popularity: number;
  last_accessed: number;
}

export interface SearchResponse {
  results: SearchResult[];
  total_results: number;
  search_term: string;
}

// Component Props Types
export interface QueryInputProps {
  onSubmit: (query: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export interface ModeSelectorProps {
  currentMode: QueryMode;
  onModeChange: (mode: QueryMode) => void;
  availableModes: ModeInfo[];
}

export interface SubmodeSelectorProps {
  currentSubmode: SubmodeStyle;
  onSubmodeChange: (submode: SubmodeStyle) => void;
  availableSubmodes: Record<string, SubmodeInfo>;
}

export interface ResponseDisplayProps {
  response?: QueryResponse;
  loading: boolean;
  error?: string;
}

export interface CitationPanelProps {
  citations: CitationInfo[];
}

export interface ConfidenceBarProps {
  confidence: number;
  label?: string;
}

export interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  color?: string;
}