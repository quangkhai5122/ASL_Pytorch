import axios, { AxiosInstance, AxiosError } from 'axios';

// Types
export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface TokenData {
  valid: boolean;
  username: string;
  permissions: string[];
}

export interface FramePredictionRequest {
  frame: string; // base64 encoded JPEG
}

export interface PredictionResponse {
  sign: string;
  confidence: number;
  top5: Array<{ sign: string; confidence: number }>;
  processing_time_ms: number;
  frame_id?: number;
}

export interface BatchPredictionRequest {
  landmarks: number[][][]; // [B, 66, 3]
  enable_gemini?: boolean;
}

export interface BatchPredictionResponse {
  signs: string[];
  confidences: number[];
  sentence?: string;
  frames_processed: number;
  processing_time_ms: number;
}

export interface VideoPredictionRequest {
  file: File;
}

export interface VideoPredictionResponse {
  signs: string[];
  confidences: number[];
  video_duration_seconds: number;
  fps: number;
  total_frames: number;
  frames_processed: number;
  sentence?: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  version: string;
}

export interface MetricsResponse {
  predictions_count: number;
  avg_latency_ms: number;
  uptime_seconds: number;
}

export interface InfoResponse {
  api_version: string;
  model_version: string;
  environment: string;
}

export interface WebSocketServerStats {
  active_connections: number;
  connections: Record<string, unknown>;
  timestamp: string;
}

export interface WebSocketHealthResponse {
  status: string;
  active_connections: number;
  max_connections: string;
  heartbeat_interval: string;
}

export interface GenerateSentenceResponse {
  sentence: string;
  signs: string[];
  method: 'gemini' | 'fallback';
  detail?: string | null;
}

// API Client Class
class APIClient {
  private client: AxiosInstance;
  private apiUrl: string;
  private apiVersion: string;
  private token: string | null = null;

  constructor() {
    this.apiUrl = this.resolveApiUrl(import.meta.env.VITE_API_URL || 'http://localhost:8000');
    this.apiVersion = import.meta.env.VITE_API_VERSION || '/api/v1';

    this.client = axios.create({
      baseURL: `${this.apiUrl}${this.apiVersion}`,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Load token from localStorage
    this.token = localStorage.getItem('access_token');

    // Request interceptor: add JWT token to headers
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor: handle 401 unauthorized
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Token expired or invalid
          localStorage.removeItem('access_token');
          window.location.href = '/';
        }
        return Promise.reject(error);
      }
    );
  }

  private resolveApiUrl(configuredUrl: string): string {
    if (typeof window === 'undefined') return configuredUrl;

    try {
      const apiUrl = new URL(configuredUrl);
      const frontendHost = window.location.hostname;
      const isLocalApiHost = apiUrl.hostname === 'localhost' || apiUrl.hostname === '127.0.0.1';
      const isLocalFrontendHost = frontendHost === 'localhost' || frontendHost === '127.0.0.1';

      if (isLocalApiHost && !isLocalFrontendHost) {
        apiUrl.hostname = frontendHost;
      }

      return apiUrl.toString().replace(/\/$/, '');
    } catch {
      return configuredUrl;
    }
  }

  // =================== Authentication ===================

  async login(username: string, password: string): Promise<LoginResponse> {
    const response = await this.client.post<LoginResponse>('/auth/login', {
      username,
      password,
    });
    this.token = response.data.access_token;
    localStorage.setItem('access_token', this.token);
    return response.data;
  }

  async verifyToken(): Promise<TokenData> {
    const response = await this.client.post<TokenData>('/auth/verify');
    return response.data;
  }

  async logout(): Promise<void> {
    this.token = null;
    localStorage.removeItem('access_token');
  }

  async getTestCredentials(): Promise<{ credentials: Array<{ username: string; password: string }> }> {
    const response = await this.client.get('/auth/test-credentials');
    return response.data;
  }

  isAuthenticated(): boolean {
    return !!localStorage.getItem('access_token');
  }

  // =================== Health & Info ===================

  async getHealth(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health');
    return response.data;
  }

  async getMetrics(): Promise<MetricsResponse> {
    const response = await this.client.get<MetricsResponse>('/metrics');
    return response.data;
  }

  async getInfo(): Promise<InfoResponse> {
    const response = await this.client.get<InfoResponse>('/info');
    return response.data;
  }

  // =================== Predictions ===================

  async predictFrame(frameBase64: string): Promise<PredictionResponse> {
    const response = await this.client.post<PredictionResponse>('/predict/frame', {
      frame: frameBase64,
    });
    return response.data;
  }

  async predictBatch(
    landmarks: number[][][],
    enableGemini: boolean = false
  ): Promise<BatchPredictionResponse> {
    const response = await this.client.post<BatchPredictionResponse>('/predict/batch', {
      landmarks,
      enable_gemini: enableGemini,
    });
    return response.data;
  }

  async predictVideo(file: File): Promise<VideoPredictionResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post<VideoPredictionResponse>('/predict/video', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async generateSentence(signs: string[]): Promise<GenerateSentenceResponse> {
    const response = await this.client.post<GenerateSentenceResponse>('/predict/generate-sentence', {
      signs,
    });
    return response.data;
  }

  // =================== WebSocket ===================

  getWebSocketUrl(token: string): string {
    const wsProtocol = this.apiUrl.startsWith('https') ? 'wss' : 'ws';
    const baseUrl = this.apiUrl.replace('http://', '').replace('https://', '');
    return `${wsProtocol}://${baseUrl}${this.apiVersion}/ws/stream?token=${token}`;
  }

  async getWebSocketStats(): Promise<WebSocketServerStats> {
    const response = await this.client.get<WebSocketServerStats>('/ws/stats');
    return response.data;
  }

  async getWebSocketHealth(): Promise<WebSocketHealthResponse> {
    const response = await this.client.get<WebSocketHealthResponse>('/ws/health');
    return response.data;
  }

  // =================== Utility Methods ===================

  setBaseURL(url: string): void {
    this.apiUrl = url;
    this.client.defaults.baseURL = `${url}${this.apiVersion}`;
  }

  getBaseURL(): string {
    return this.apiUrl;
  }

  setToken(token: string): void {
    this.token = token;
    localStorage.setItem('access_token', token);
  }

  getToken(): string | null {
    return this.token || localStorage.getItem('access_token');
  }
}

// Export singleton instance
export const apiClient = new APIClient();
