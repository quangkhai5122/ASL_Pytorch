import { useState, useEffect, useRef } from "react";
import { Activity } from "lucide-react";
import { useASL } from "@/context/ASLContext";

export function FPSBadge() {
  const { cameraActive, wsStats } = useASL();
  const [fps, setFps] = useState(0);
  const lastFramesRef = useRef(0);
  const latestFramesRef = useRef(0);

  useEffect(() => {
    latestFramesRef.current = wsStats?.frames || 0;
  }, [wsStats?.frames]);

  useEffect(() => {
    const interval = setInterval(() => {
      const currentFrames = latestFramesRef.current;
      setFps(Math.max(0, currentFrames - lastFramesRef.current));
      lastFramesRef.current = currentFrames;
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!cameraActive) setFps(0);
  }, [cameraActive]);

  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-foreground/10 text-xs font-mono"
      aria-label={`${fps} streamed frames per second`}
    >
      <Activity className="w-3 h-3" aria-hidden="true" />
      {fps} FPS
    </span>
  );
}
