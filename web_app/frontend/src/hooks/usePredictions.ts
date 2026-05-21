import { useState, useCallback } from 'react';
import { apiClient, PredictionResponse, BatchPredictionResponse, VideoPredictionResponse } from '@/services/api';

export interface UsePredictionsReturn {
  predictions: PredictionResponse | null;
  batchPredictions: BatchPredictionResponse | null;
  videoPredictions: VideoPredictionResponse | null;
  isLoading: boolean;
  error: string | null;
  predictFrame: (frameBase64: string) => Promise<void>;
  predictBatch: (landmarks: number[][][], enableGemini?: boolean) => Promise<void>;
  predictVideo: (file: File) => Promise<void>;
  clearError: () => void;
}

/**
 * Hook for making predictions from the backend
 */
export function usePredictions(): UsePredictionsReturn {
  const [predictions, setPredictions] = useState<PredictionResponse | null>(null);
  const [batchPredictions, setBatchPredictions] = useState<BatchPredictionResponse | null>(null);
  const [videoPredictions, setVideoPredictions] = useState<VideoPredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predictFrame = useCallback(async (frameBase64: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await apiClient.predictFrame(frameBase64);
      setPredictions(result);

      if (import.meta.env.VITE_LOG_PREDICTIONS === 'true') {
        console.log('Frame prediction:', result);
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Frame prediction failed';
      setError(errorMsg);
      console.error('Frame prediction error:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const predictBatch = useCallback(async (landmarks: number[][][], enableGemini = false) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await apiClient.predictBatch(landmarks, enableGemini);
      setBatchPredictions(result);

      if (import.meta.env.VITE_LOG_PREDICTIONS === 'true') {
        console.log('Batch prediction:', result);
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Batch prediction failed';
      setError(errorMsg);
      console.error('Batch prediction error:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const predictVideo = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await apiClient.predictVideo(file);
      setVideoPredictions(result);

      if (import.meta.env.VITE_LOG_PREDICTIONS === 'true') {
        console.log('Video prediction:', result);
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Video prediction failed';
      setError(errorMsg);
      console.error('Video prediction error:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    predictions,
    batchPredictions,
    videoPredictions,
    isLoading,
    error,
    predictFrame,
    predictBatch,
    predictVideo,
    clearError,
  };
}
