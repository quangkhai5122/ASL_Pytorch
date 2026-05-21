import { useMemo } from "react";
import { useASL } from "@/context/ASLContext";

export function MotionMeter() {
  const { cameraActive, wsLastPrediction } = useASL();
  const rawMotion = typeof wsLastPrediction?.motion === "number" ? wsLastPrediction.motion : 0;
  const motion = useMemo(() => Math.max(0, Math.min(1, rawMotion / 0.02)), [rawMotion]);

  const level = !cameraActive ? "Off" : motion > 0.6 ? "High" : motion > 0.3 ? "Med" : "Low";

  return (
    <div className="flex items-center gap-1.5" aria-label={`Motion level: ${level}`}>
      <span className="text-xs text-muted-foreground">Motion</span>
      <div className="w-12 h-2 rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full bg-primary transition-all duration-300"
          style={{ width: `${motion * 100}%` }}
        />
      </div>
      <span className="text-xs font-mono text-muted-foreground">{level}</span>
    </div>
  );
}
