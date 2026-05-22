import { useState, useEffect } from "react";

export function MotionMeter() {
  const [motion, setMotion] = useState(0.4);

  useEffect(() => {
    const interval = setInterval(() => {
      setMotion(Math.random() * 0.8 + 0.1);
    }, 800);
    return () => clearInterval(interval);
  }, []);

  const level = motion > 0.6 ? "High" : motion > 0.3 ? "Med" : "Low";

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
