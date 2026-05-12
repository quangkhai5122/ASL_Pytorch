import { useASL } from "@/context/ASLContext";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useState } from "react";
import React, { useRef, useEffect } from "react";
/**
 * Example: Login Component
 * Shows how to use authentication with the backend
 */
export function LoginPanel() {
  const { isAuthenticated, authLoading, authError, username, login, logout } = useASL();
  const [formData, setFormData] = useState({ username: "", password: "" });

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await login(formData.username, formData.password);
      setFormData({ username: "", password: "" });
    } catch (err) {
      console.error("Login failed:", err);
    }
  };

  if (isAuthenticated) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Logged In</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p>Welcome, <strong>{username}</strong>!</p>
          <Button onClick={logout} variant="outline" className="w-full">
            Logout
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Login to Backend</CardTitle>
        <CardDescription>
          Enter test credentials: testuser / testpass123
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleLogin} className="space-y-4">
          <input
            type="text"
            placeholder="Username"
            value={formData.username}
            onChange={(e) => setFormData({ ...formData, username: e.target.value })}
            className="w-full px-3 py-2 border rounded"
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={formData.password}
            onChange={(e) => setFormData({ ...formData, password: e.target.value })}
            className="w-full px-3 py-2 border rounded"
            required
          />
          <Button type="submit" className="w-full" disabled={authLoading}>
            {authLoading ? "Logging in..." : "Login"}
          </Button>
          {authError && <p className="text-red-500 text-sm">{authError}</p>}
        </form>
      </CardContent>
    </Card>
  );
}

/**
 * Example: Backend Status Component
 * Shows model status and metrics
 */
export function BackendStatusPanel() {
  const { backendStats, backendLoading, isAuthenticated } = useASL();

  if (!isAuthenticated) {
    return <p className="text-muted-foreground">Please login first</p>;
  }

  if (backendLoading) {
    return <p>Loading backend stats...</p>;
  }

  if (!backendStats) {
    return <p className="text-red-500">Failed to load backend stats</p>;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Backend Status</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex justify-between">
          <span>Model Status:</span>
          <span className={backendStats.modelLoaded ? "text-green-600" : "text-red-600"}>
            {backendStats.modelLoaded ? "✓ Loaded" : "✗ Not Loaded"}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Device:</span>
          <span className="font-mono">{backendStats.device.toUpperCase()}</span>
        </div>
        <div className="flex justify-between">
          <span>Predictions:</span>
          <span>{backendStats.predictions}</span>
        </div>
        <div className="flex justify-between">
          <span>Avg Latency:</span>
          <span>{backendStats.latency.toFixed(1)}ms</span>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Example: Frame Prediction Component
 * Shows how to capture and send frames for prediction
 */
export function FramePredictionPanel() {
  const { lastPrediction, predictionLoading, predictFrame, isAuthenticated } = useASL();
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [cameraActive, setCameraActive] = React.useState(false);

  React.useEffect(() => {
    if (!cameraActive || !videoRef.current) return;

    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 } })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => {
        console.error("Camera access denied:", err);
        setCameraActive(false);
      });

    return () => {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((track) => track.stop());
      }
    };
  }, [cameraActive]);

  const handleCapture = async () => {
    if (!canvasRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext("2d");
    ctx?.drawImage(videoRef.current, 0, 0, 640, 480);

    const frameBase64 = canvasRef.current.toDataURL("image/jpeg").split(",")[1];
    await predictFrame(frameBase64);
  };

  if (!isAuthenticated) {
    return <p className="text-muted-foreground">Please login first</p>;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Frame Prediction</CardTitle>
        <CardDescription>Capture and predict from a single frame</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ display: cameraActive ? "block" : "none", width: "100%" }}
          className="rounded"
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          style={{ display: "none" }}
        />

        <div className="flex gap-2">
          <Button
            onClick={() => setCameraActive(!cameraActive)}
            variant={cameraActive ? "default" : "outline"}
            className="flex-1"
          >
            {cameraActive ? "Camera On" : "Camera Off"}
          </Button>
          <Button
            onClick={handleCapture}
            disabled={!cameraActive || predictionLoading}
            className="flex-1"
          >
            {predictionLoading ? "Predicting..." : "Capture & Predict"}
          </Button>
        </div>

        {lastPrediction && (
          <div className="bg-accent p-4 rounded space-y-2">
            <h3 className="font-bold text-lg">{lastPrediction.sign}</h3>
            <p>Confidence: {(lastPrediction.confidence * 100).toFixed(1)}%</p>
            <p className="text-sm text-muted-foreground">
              Latency: {lastPrediction.processing_time_ms}ms
            </p>
            {lastPrediction.top5 && (
              <div className="text-sm space-y-1">
                <p className="font-semibold">Top 5:</p>
                {lastPrediction.top5.slice(0, 5).map((pred: any, idx: number) => (
                  <p key={idx}>
                    {idx + 1}. {pred.sign} ({(pred.confidence * 100).toFixed(1)}%)
                  </p>
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Example: WebSocket Streaming Component
 * Shows real-time predictions from continuous stream
 */
export function WebSocketStreamPanel() {
  const { wsConnected, wsLastPrediction, sendFrameViaWebSocket, wsError, isAuthenticated } = useASL();
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const intervalRef = React.useRef<NodeJS.Timeout | null>(null);

  React.useEffect(() => {
    if (!isAuthenticated) return;

    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 } })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => console.error("Camera access denied:", err));

    return () => {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((track) => track.stop());
      }
    };
  }, [isAuthenticated]);

  React.useEffect(() => {
    if (!wsConnected || !canvasRef.current || !videoRef.current) return;

    intervalRef.current = setInterval(() => {
      const ctx = canvasRef.current!.getContext("2d");
      ctx?.drawImage(videoRef.current!, 0, 0, 640, 480);
      sendFrameViaWebSocket(canvasRef.current!);
    }, 100);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [wsConnected, sendFrameViaWebSocket]);

  if (!isAuthenticated) {
    return <p className="text-muted-foreground">Please login first</p>;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>WebSocket Streaming</CardTitle>
        <CardDescription>Real-time predictions via WebSocket</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2">
          <div
            className={`w-3 h-3 rounded-full ${wsConnected ? "bg-green-500" : "bg-red-500"}`}
          />
          <span>{wsConnected ? "Connected" : "Disconnected"}</span>
        </div>

        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="w-full rounded"
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          style={{ display: "none" }}
        />

        {wsError && (
          <p className="text-red-500 text-sm">{wsError}</p>
        )}

        {wsLastPrediction?.type === 'prediction' && (
          <div className="bg-accent p-4 rounded space-y-2">
            <h3 className="font-bold text-lg">{wsLastPrediction.sign}</h3>
            <p>Confidence: {(wsLastPrediction.confidence * 100).toFixed(1)}%</p>
            <p className="text-sm text-muted-foreground">
              Latency: {wsLastPrediction.processing_time_ms}ms
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
