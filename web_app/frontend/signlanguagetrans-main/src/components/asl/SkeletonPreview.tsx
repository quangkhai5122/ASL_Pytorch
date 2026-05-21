import { useEffect, useState, useRef } from "react";
import { SKELETON_JOINTS, SKELETON_BONES } from "@/lib/mockData";

type Props = {
  animated?: boolean;
};

export function SkeletonPreview({ animated = false }: Props) {
  const [frame, setFrame] = useState(0);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!animated) return;
    const interval = setInterval(() => {
      setFrame(f => f + 1);
    }, 60);
    return () => clearInterval(interval);
  }, [animated]);

  const getJointPos = (joint: typeof SKELETON_JOINTS[0]) => {
    if (!animated) return { x: joint.x, y: joint.y };
    const wobble = Math.sin(frame * 0.05 + joint.x * 10) * 0.02;
    const wobbleY = Math.cos(frame * 0.04 + joint.y * 8) * 0.015;
    return { x: joint.x + wobble, y: joint.y + wobbleY };
  };

  const jointMap = Object.fromEntries(
    SKELETON_JOINTS.map(j => [j.id, getJointPos(j)])
  );

  return (
    <svg
      ref={svgRef}
      viewBox="0 0 1 1"
      className="w-full h-full"
      role="img"
      aria-label="Skeleton animation preview showing body pose"
    >
      {SKELETON_BONES.map(([from, to], i) => {
        const a = jointMap[from];
        const b = jointMap[to];
        return (
          <line
            key={i}
            x1={a.x} y1={a.y}
            x2={b.x} y2={b.y}
            stroke="hsl(var(--primary))"
            strokeWidth="0.008"
            strokeLinecap="round"
          />
        );
      })}
      {SKELETON_JOINTS.map(joint => {
        const pos = jointMap[joint.id];
        return (
          <circle
            key={joint.id}
            cx={pos.x} cy={pos.y}
            r={joint.id === "head" ? 0.025 : 0.012}
            fill="hsl(var(--primary))"
          />
        );
      })}
    </svg>
  );
}
