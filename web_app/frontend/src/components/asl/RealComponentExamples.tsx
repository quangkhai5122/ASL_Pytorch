/**
 * REAL COMPONENT INTEGRATION EXAMPLES
 * 
 * Copy the patterns from these examples into your existing components
 * to wire up real backend data.
 */

import React, { useRef, useEffect, useState } from 'react';
import { useASL } from '@/context/ASLContext';

// =====================================================================
// EXAMPLE 1: Status Panel - Show connection status & model info
// =====================================================================
export function IntegratedStatusPanel() {
  const { 
    isAuthenticated, 
    wsConnected, 
    backendStats, 
    backendLoading,
    predictionError,
    wsError,
    authError 
  } = useASL();

  return (
    <div className="rounded-lg border p-4 bg-card">
      <h2 className="text-lg font-semibold mb-4">System Status</h2>
      
      {/* Connection Status */}
      <div className="space-y-2 mb-4">
        <div className="flex items-center justify-between">
          <span>Login Status</span>
          <span className={isAuthenticated ? 'text-green-600' : 'text-red-600'}>
            {isAuthenticated ? '✅ Connected' : '❌ Not Connected'}
          </span>
        </div>
        
        <div className="flex items-center justify-between">
          <span>WebSocket</span>
          <span className={wsConnected ? 'text-green-600' : 'text-yellow-600'}>
            {wsConnected ? '✅ Streaming' : '⏳ Connecting...'}
          </span>
        </div>
      </div>

      {/* Model Stats */}
      {backendLoading ? (
        <div className="animate-pulse">Loading backend stats...</div>
      ) : backendStats ? (
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span>Model Status</span>
            <span className={backendStats.model_loaded ? 'text-green-600' : 'text-yellow-600'}>
              {backendStats.model_loaded ? '✅ Ready' : '⏳ Loading'}
            </span>
          </div>
          
          <div className="flex justify-between">
            <span>Device</span>
            <span className="font-mono text-sm">{backendStats.device_info}</span>
          </div>
          
          <div className="flex justify-between">
            <span>Total Predictions</span>
            <span className="font-mono">{backendStats.predictions_count}</span>
          </div>
          
          <div className="flex justify-between">
            <span>Avg Latency</span>
            <span className="font-mono">{backendStats.avg_latency_ms.toFixed(0)}ms</span>
          </div>
        </div>
      ) : null}

      {/* Errors */}
      {(authError || predictionError || wsError) && (
        <div className="mt-4 p-2 bg-red-100 text-red-800 rounded text-sm">
          {authError && <div>Auth Error: {authError}</div>}
          {predictionError && <div>Prediction Error: {predictionError}</div>}
          {wsError && <div>WebSocket Error: {wsError}</div>}
        </div>
      )}
    </div>
  );
}

// =====================================================================
// EXAMPLE 2: Camera Card with Real WebSocket streaming
// =====================================================================
export function IntegratedCameraCard() {
  const { wsConnected, sendFrameViaWebSocket, wsLastPrediction, predictionLoading } = useASL();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [streaming, setStreaming] = useState(false);
  const frameIntervalRef = useRef<NodeJS.Timeout>();

  // Start camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setStreaming(true);
      }
    } catch (err) {
      console.error('Camera error:', err);
      alert('Could not access camera. Check permissions.');
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      setStreaming(false);
    }
  };

  // Send frames via WebSocket
  useEffect(() => {
    if (!wsConnected || !streaming) {
      frameIntervalRef.current && clearInterval(frameIntervalRef.current);
      return;
    }

    frameIntervalRef.current = setInterval(async () => {
      if (videoRef.current && canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) {
          // Draw video frame to canvas
          ctx.drawImage(videoRef.current, 0, 0, 640, 480);
          
          // Convert canvas to blob and send
          canvasRef.current.toBlob(
            (blob) => {
              if (blob) {
                sendFrameViaWebSocket(blob);
              }
            },
            'image/jpeg',
            0.75 // 75% quality for faster transmission
          );
        }
      }
    }, 200); // Send frame every 200ms (5 FPS)

    return () => {
      frameIntervalRef.current && clearInterval(frameIntervalRef.current);
    };
  }, [wsConnected, streaming, sendFrameViaWebSocket]);

  return (
    <div className="rounded-lg border p-4 bg-card space-y-4">
      <h2 className="text-lg font-semibold">Live Camera</h2>
      
      {/* Connection Status */}
      <div className="text-sm">
        {wsConnected ? (
          <div className="text-green-600">✅ Connected to WebSocket</div>
        ) : (
          <div className="text-yellow-600">⏳ Connecting to WebSocket...</div>
        )}
      </div>

      {/* Camera Preview */}
      <div className="bg-black rounded aspect-video flex items-center justify-center overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="w-full h-full object-cover"
        />
      </div>

      {/* Hidden Canvas for frame capture */}
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        style={{ display: 'none' }}
      />

      {/* Controls */}
      <div className="flex gap-2">
        <button
          onClick={startCamera}
          disabled={streaming}
          className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-gray-400"
        >
          Start Camera
        </button>
        <button
          onClick={stopCamera}
          disabled={!streaming}
          className="px-4 py-2 bg-gray-600 text-white rounded disabled:bg-gray-400"
        >
          Stop Camera
        </button>
      </div>

      {/* Last Prediction */}
      {predictionLoading && <div>Analyzing frame...</div>}
      {wsLastPrediction && (
        <div className="p-3 bg-blue-100 rounded">
          <div className="text-sm text-gray-600">Current Prediction</div>
          <div className="text-xl font-bold text-blue-900">
            {wsLastPrediction.sign}
          </div>
          <div className="text-sm">
            Confidence: {(wsLastPrediction.confidence * 100).toFixed(1)}%
          </div>
        </div>
      )}
    </div>
  );
}

// =====================================================================
// EXAMPLE 3: Predictions List with Real Data
// =====================================================================
export function IntegratedPredictionsList() {
  const { wsLastPrediction, wsStats } = useASL();
  const [history, setHistory] = useState<Array<{
    sign: string;
    confidence: number;
    timestamp: Date;
  }>>([]);

  // Add to history when new prediction comes in
  useEffect(() => {
    if (wsLastPrediction) {
      setHistory(prev => [
        {
          sign: wsLastPrediction.sign,
          confidence: wsLastPrediction.confidence,
          timestamp: new Date()
        },
        ...prev.slice(0, 9) // Keep last 10
      ]);
    }
  }, [wsLastPrediction?.sign]); // Only when sign changes

  return (
    <div className="rounded-lg border p-4 bg-card space-y-4">
      <h2 className="text-lg font-semibold">Predictions</h2>

      {/* Current Prediction */}
      {wsLastPrediction ? (
        <div className="p-4 bg-gradient-to-r from-blue-100 to-purple-100 rounded-lg">
          <div className="text-sm text-gray-600">Current Sign</div>
          <div className="text-3xl font-bold text-blue-900 my-2">
            {wsLastPrediction.sign}
          </div>
          <div className="flex items-center">
            <div className="flex-1 bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full"
                style={{ width: `${wsLastPrediction.confidence * 100}%` }}
              />
            </div>
            <span className="ml-2 text-sm font-semibold">
              {(wsLastPrediction.confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      ) : (
        <div className="p-4 text-center text-gray-500">
          Waiting for prediction... Start camera to begin
        </div>
      )}

      {/* Stats */}
      {wsStats && (
        <div className="grid grid-cols-3 gap-2 text-sm">
          <div className="p-2 bg-gray-100 rounded">
            <div className="text-gray-600">Frames</div>
            <div className="text-lg font-bold">{wsStats.frames_sent}</div>
          </div>
          <div className="p-2 bg-gray-100 rounded">
            <div className="text-gray-600">Predictions</div>
            <div className="text-lg font-bold">{wsStats.predictions_received}</div>
          </div>
          <div className="p-2 bg-gray-100 rounded">
            <div className="text-gray-600">Latency</div>
            <div className="text-lg font-bold">{wsStats.avg_latency_ms.toFixed(0)}ms</div>
          </div>
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-600">Recognition History</h3>
          <div className="space-y-1">
            {history.map((pred, idx) => (
              <div key={idx} className="flex justify-between text-sm p-2 bg-gray-50 rounded">
                <span className="font-medium">{pred.sign}</span>
                <span className="text-gray-600">
                  {(pred.confidence * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// =====================================================================
// EXAMPLE 4: Login Form (if needed)
// =====================================================================
export function IntegratedLoginForm() {
  const { login, authError, isAuthenticated } = useASL();
  const [username, setUsername] = useState('testuser');
  const [password, setPassword] = useState('testpass123');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    setLoading(true);
    try {
      await login(username, password);
    } finally {
      setLoading(false);
    }
  };

  if (isAuthenticated) {
    return <div className="text-green-600">✅ Logged in successfully</div>;
  }

  return (
    <div className="rounded-lg border p-4 bg-card space-y-4 max-w-md">
      <h2 className="text-lg font-semibold">Login</h2>

      <div className="space-y-2">
        <label className="text-sm font-medium">Username</label>
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="w-full px-3 py-2 border rounded"
          placeholder="testuser"
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Password</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full px-3 py-2 border rounded"
          placeholder="••••••••"
        />
      </div>

      {authError && (
        <div className="p-2 bg-red-100 text-red-800 rounded text-sm">
          {authError}
        </div>
      )}

      <button
        onClick={handleLogin}
        disabled={loading}
        className="w-full px-4 py-2 bg-blue-600 text-white rounded disabled:bg-gray-400"
      >
        {loading ? 'Logging in...' : 'Login'}
      </button>

      <div className="text-sm text-gray-600">
        <p>Test credentials:</p>
        <p>testuser / testpass123</p>
        <p>demo / demo123</p>
      </div>
    </div>
  );
}

// =====================================================================
// HOW TO USE THESE IN YOUR COMPONENTS
// =====================================================================
/*
 * 1. In your existing component files (e.g., CameraCard.tsx):
 *    
 *    import { IntegratedCameraCard } from '@/components/asl/RealComponentExamples';
 *    
 *    // Replace the mock component with:
 *    export function CameraCard() {
 *      return <IntegratedCameraCard />;
 *    }
 *
 * 2. Or copy the pattern into your existing component:
 *    
 *    import { useASL } from '@/context/ASLContext';
 *    
 *    export function CameraCard() {
 *      const { wsConnected, sendFrameViaWebSocket } = useASL();
 *      // ... rest of your component
 *    }
 *
 * 3. Key hooks available from useASL():
 *    - isAuthenticated, login, logout, authError, authLoading
 *    - wsConnected, sendFrameViaWebSocket, wsLastPrediction, wsStats, wsError
 *    - lastPrediction, predictFrame, predictionError, predictionLoading
 *    - backendStats, backendLoading
 */
