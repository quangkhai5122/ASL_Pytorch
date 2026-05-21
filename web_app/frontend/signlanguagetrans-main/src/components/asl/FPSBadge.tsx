import { useState, useEffect } from "react";
import { Activity } from "lucide-react";

export function FPSBadge() {
  const [fps, setFps] = useState(30);

  useEffect(() => {
    const interval = setInterval(() => {
      setFps(Math.floor(28 + Math.random() * 5));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-foreground/10 text-xs font-mono"
      aria-label={`${fps} frames per second`}
    >
      <Activity className="w-3 h-3" aria-hidden="true" />
      {fps} FPS
    </span>
  );
}
