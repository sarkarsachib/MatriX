/**
 * API service for Sathik AI Direction Mode
 */

import axios from 'axios';
import { QueryRequest, QueryResponse, DirectionModeStatus, ModesResponse, KnowledgeBaseStats, SearchRequest, SearchResponse } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth headers if needed
apiClient.interceptors.request.use(
  (config) => {
    const apiKey = localStorage.getItem('sathik_api_key');
    if (apiKey) {
      config.headers['X-API-Key'] = apiKey;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export class ApiService {
  /**
   * Submit a query for processing
   */
  static async submitQuery(request: QueryRequest): Promise<QueryResponse> {
    const response = await apiClient.post<QueryResponse>('/query', request);
    return response.data;
  }

  /**
   * Get available modes, sub-modes, and formats
   */
  static async getAvailableModes(): Promise<ModesResponse> {
    const response = await apiClient.get<ModesResponse>('/modes');
    return response.data;
  }

  /**
   * Get system status
   */
  static async getSystemStatus(): Promise<DirectionModeStatus> {
    const response = await apiClient.get<DirectionModeStatus>('/status');
    return response.data;
  }

  /**
   * Get knowledge base statistics
   */
  static async getKnowledgeBaseStats(): Promise<KnowledgeBaseStats> {
    const response = await apiClient.get<KnowledgeBaseStats>('/stats');
    return response.data;
  }

  /**
   * Clear cache
   */
  static async clearCache(olderThanDays: number = 30): Promise<{ removed_entries: number; message: string }> {
    const response = await apiClient.post('/clear-cache', null, {
      params: { older_than_days: olderThanDays }
    });
    return response.data;
  }

  /**
   * Search knowledge base
   */
  static async searchKnowledgeBase(request: SearchRequest): Promise<SearchResponse> {
    const response = await apiClient.post<SearchResponse>('/search-knowledge', request);
    return response.data;
  }

  /**
   * Health check
   */
  static async healthCheck(): Promise<{ status: string; timestamp: number; version: string; components: Record<string, string> }> {
    const response = await apiClient.get('/health');
    return response.data;
  }

  /**
   * Test API connectivity
   */
  static async testConnection(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch (error) {
      return false;
    }
  }
}

export default ApiService;