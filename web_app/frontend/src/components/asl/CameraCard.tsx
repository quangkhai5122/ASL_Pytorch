import { AlertCircle, Camera, CameraOff } from "lucide-react";
import { useASL } from "@/context/ASLContext";
import { Button } from "@/components/ui/button";
import { FPSBadge } from "./FPSBadge";
import { MotionMeter } from "./MotionMeter";
import { useRef, useEffect, useCallback, useState } from "react";

// ─── MediaPipe Holistic connection definitions (GISLR order) ───────────────
// Pose connections (33 points, indices 489..521 in GISLR → mapped 0..32 in pose subset)
const POSE_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
  [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32],
];

// Hand connections (21 points each)
const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // index
  [0, 9], [9, 10], [10, 11], [11, 12],  // middle
  [0, 13], [13, 14], [14, 15], [15, 16],// ring
  [0, 17], [17, 18], [18, 19], [19, 20],// pinky
  [5, 9], [9, 13], [13, 17],            // palm
];

// Colors matching MediaPipe style
const POSE_COLOR = "rgba(255, 255, 255, 0.7)";
const POSE_POINT_COLOR = "rgba(0, 200, 255, 0.9)";
const LEFT_HAND_COLOR = "rgba(255, 138, 76, 0.9)";
const LEFT_HAND_POINT_COLOR = "rgba(255, 100, 30, 1)";
const RIGHT_HAND_COLOR = "rgba(76, 175, 255, 0.9)";
const RIGHT_HAND_POINT_COLOR = "rgba(30, 130, 255, 1)";

// GISLR index offsets
const FACE_START = 0;
const LEFT_HAND_START = 468;
const POSE_START = 489;
const RIGHT_HAND_START = 522;

/**
 * Draw landmarks and connections on a canvas overlay.
 * landmarks: flat [543 * 3] array or [543][3] from backend (x, y, z normalized 0..1)
 */
function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: number[][],
  width: number,
  height: number,
) {
  ctx.clearRect(0, 0, width, height);

  // Helper: check if a landmark is valid (not NaN/null)
  const isValid = (lm: number[] | undefined): lm is number[] =>
    !!lm && lm.length >= 2 && isFinite(lm[0]) && isFinite(lm[1]);

  // Helper: draw connections between landmarks
  const drawConnections = (
    indices: number[],
    connections: [number, number][],
    color: string,
    lineWidth: number,
  ) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    for (const [i, j] of connections) {
      const a = landmarks[indices[i]];
      const b = landmarks[indices[j]];
      if (isValid(a) && isValid(b)) {
        ctx.beginPath();
        // Mirror X since video is mirrored
        ctx.moveTo((1 - a[0]) * width, a[1] * height);
        ctx.lineTo((1 - b[0]) * width, b[1] * height);
        ctx.stroke();
      }
    }
  };

  // Helper: draw points
  const drawPoints = (
    indices: number[],
    color: string,
    radius: number,
  ) => {
    ctx.fillStyle = color;
    for (const idx of indices) {
      const lm = landmarks[idx];
      if (isValid(lm)) {
        ctx.beginPath();
        ctx.arc((1 - lm[0]) * width, lm[1] * height, radius, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  };

  // Build index arrays
  const poseIndices = Array.from({ length: 33 }, (_, i) => POSE_START + i);
  const leftHandIndices = Array.from({ length: 21 }, (_, i) => LEFT_HAND_START + i);
  const rightHandIndices = Array.from({ length: 21 }, (_, i) => RIGHT_HAND_START + i);

  // Draw pose skeleton
  drawConnections(poseIndices, POSE_CONNECTIONS, POSE_COLOR, 2);
  drawPoints(poseIndices, POSE_POINT_COLOR, 4);

  // Draw left hand
  drawConnections(leftHandIndices, HAND_CONNECTIONS, LEFT_HAND_COLOR, 2);
  drawPoints(leftHandIndices, LEFT_HAND_POINT_COLOR, 3);

  // Draw right hand
  drawConnections(rightHandIndices, HAND_CONNECTIONS, RIGHT_HAND_COLOR, 2);
  drawPoints(rightHandIndices, RIGHT_HAND_POINT_COLOR, 3);
}

export function CameraCard() {
  const {
    cameraActive, setCameraActive,
    wsConnected, sendFrameViaWebSocket,
    wsLastPrediction,
  } = useASL();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [videoReady, setVideoReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const landmarksRef = useRef<number[][] | null>(null);
  const animFrameRef = useRef<number | null>(null);

  // Update landmarks from WebSocket prediction messages
  useEffect(() => {
    if (wsLastPrediction?.landmarks) {
      // Convert null entries to [NaN, NaN, NaN] for the drawing function
      landmarksRef.current = wsLastPrediction.landmarks.map(
        (lm) => lm ?? [NaN, NaN, NaN]
      );
    }
  }, [wsLastPrediction]);

  // Skeleton drawing loop using requestAnimationFrame
  useEffect(() => {
    if (!cameraActive || !videoReady) {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
      return;
    }

    const drawLoop = () => {
      const overlay = overlayCanvasRef.current;
      const video = videoRef.current;
      if (overlay && video && video.readyState >= 2) {
        // Match overlay canvas size to video display size
        const rect = video.getBoundingClientRect();
        if (overlay.width !== rect.width || overlay.height !== rect.height) {
          overlay.width = rect.width;
          overlay.height = rect.height;
        }

        const ctx = overlay.getContext("2d");
        if (ctx && landmarksRef.current) {
          drawSkeleton(ctx, landmarksRef.current, overlay.width, overlay.height);
        }
      }
      animFrameRef.current = requestAnimationFrame(drawLoop);
    };

    animFrameRef.current = requestAnimationFrame(drawLoop);

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
    };
  }, [cameraActive, videoReady]);

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
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    landmarksRef.current = null;
    // Clear skeleton overlay
    const overlay = overlayCanvasRef.current;
    if (overlay) {
      const ctx = overlay.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, overlay.width, overlay.height);
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
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
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
            {/* Skeleton overlay canvas — sits on top of video */}
            <canvas
              ref={overlayCanvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
              style={{ zIndex: 5 }}
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
