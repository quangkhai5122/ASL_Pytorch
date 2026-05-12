import { AlertCircle, Camera, CameraOff } from "lucide-react";
import { useASL } from "@/context/ASLContext";
import { Button } from "@/components/ui/button";
import { FPSBadge } from "./FPSBadge";
import { MotionMeter } from "./MotionMeter";
import { useRef, useEffect, useCallback, useState } from "react";

export function CameraCard() {
  const {
    cameraActive, setCameraActive,
    wsConnected, sendFrameViaWebSocket,
  } = useASL();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [videoReady, setVideoReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Start camera: get stream, then set cameraActive
  const startCamera = useCallback(async () => {
    try {
      setCameraError(null);
      console.log("[Camera] Requesting webcam...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          aspectRatio: { ideal: 16 / 9 },
          facingMode: "user",
        },
      });
      console.log("[Camera] Got stream, tracks:", stream.getVideoTracks().length);
      streamRef.current = stream;
      setVideoReady(false);
      setCameraActive(true);
    } catch (err) {
      console.error("[Camera] Error:", err);
      setCameraError(
        err instanceof Error
          ? err.message
          : "Unable to access the camera. Check browser permissions."
      );
      setCameraActive(false);
    }
  }, [setCameraActive]);

  // When cameraActive becomes true AND the video element exists, attach the stream
  useEffect(() => {
    if (cameraActive && videoRef.current && streamRef.current) {
      console.log("[Camera] Attaching stream to video element");
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.onloadedmetadata = () => {
        console.log("[Camera] Video metadata loaded, playing...");
        videoRef.current?.play().then(() => {
          setVideoReady(true);
          console.log("[Camera] Video playing");
        }).catch((e) => console.error("[Camera] Play error:", e));
      };
    }
  }, [cameraActive]);

  const stopCamera = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setVideoReady(false);
    setCameraError(null);
    setCameraActive(false);
  }, [setCameraActive]);

  // Frame streaming loop — captures frames and sends via WebSocket
  useEffect(() => {
    if (cameraActive && wsConnected && videoReady) {
      console.log("[Camera] Starting frame streaming at ~10 FPS");
      intervalRef.current = setInterval(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || video.readyState < 2) return;

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.drawImage(video, 0, 0);
        sendFrameViaWebSocket(canvas);
      }, 100);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [cameraActive, wsConnected, videoReady, sendFrameViaWebSocket]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    };
  }, []);

  const toggleCamera = () => {
    if (cameraActive) {
      stopCamera();
    } else {
      startCamera();
    }
  };

  return (
    <div className="asl-panel flex flex-col h-full">
      <div className="asl-panel-header">
        <h2 className="text-sm font-semibold">Camera Feed</h2>
        <div className="flex items-center gap-2">
          <FPSBadge />
          <MotionMeter />
        </div>
      </div>
      <div className="relative flex-1 min-h-[480px] bg-slate-950 overflow-hidden flex items-center justify-center">
        {cameraActive ? (
          <>
            {/* Real webcam video */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover"
              style={{ transform: "scaleX(-1)" }}
            />
            {/* Hidden canvas for frame capture */}
            <canvas ref={canvasRef} className="hidden" />
            {/* WebSocket status badge */}
            <div className="absolute top-2 right-2 z-10">
              <span
                className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium ${
                  wsConnected
                    ? "bg-green-500/20 text-green-400"
                    : "bg-yellow-500/20 text-yellow-400"
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    wsConnected ? "bg-green-400" : "bg-yellow-400 animate-pulse"
                  }`}
                />
                {wsConnected ? "Streaming" : "Connecting..."}
              </span>
            </div>
          </>
        ) : (
          <div className="text-center space-y-3 p-8">
            <CameraOff className="w-12 h-12 mx-auto text-muted-foreground/50" aria-hidden="true" />
            <p className="text-sm text-muted-foreground">Camera is off</p>
            <p className="text-xs text-muted-foreground/70">
              Tip: place your hands in the center of the frame and ensure even lighting.
            </p>
            {cameraError && (
              <div className="mt-3 inline-flex max-w-sm items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-left text-xs text-destructive">
                <AlertCircle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" aria-hidden="true" />
                <span>{cameraError}</span>
              </div>
            )}
          </div>
        )}
      </div>
      <div className="p-3 flex items-center gap-2 border-t border-border">
        <Button
          onClick={toggleCamera}
          variant={cameraActive ? "destructive" : "default"}
          className="touch-target flex-1"
          aria-label={cameraActive ? "Stop camera" : "Start camera"}>
          {cameraActive ? (
            <><CameraOff className="w-4 h-4 mr-2" aria-hidden="true" />Stop Camera</>
          ) : (
            <><Camera className="w-4 h-4 mr-2" aria-hidden="true" />Start Camera</>
          )}
        </Button>
      </div>
    </div>
  );
}
