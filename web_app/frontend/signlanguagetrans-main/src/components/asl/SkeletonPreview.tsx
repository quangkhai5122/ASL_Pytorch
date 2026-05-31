import { useEffect, useState, useRef, useCallback } from "react";
import { SKELETON_BONES, SKELETON_JOINTS as DEFAULT_JOINTS } from "@/lib/mockData";

type Joint = { id: string; x: number; y: number };
export type FrameData = { frame: number; joints: Joint[] };

type Props = {
  animated?: boolean;
  framesData?: FrameData[];
  // External playback control (overrides internal animation when provided)
  isPlaying?: boolean;
  playbackSpeed?: number;
  currentFrame?: number;
  onFrameUpdate?: (frame: number) => void;
};

export function SkeletonPreview({
  animated = false,
  framesData,
  isPlaying,
  playbackSpeed = 1,
  currentFrame,
  onFrameUpdate,
}: Props) {
  const [internalFrameIdx, setInternalFrameIdx] = useState(0);
  const svgRef = useRef<SVGSVGElement>(null);
  const lastTimeRef = useRef<number>(0);
  const animFrameRef = useRef<number>(0);

  const totalFrames = framesData?.length || 0;

  // Determine if we're externally controlled
  const isExternallyControlled = isPlaying !== undefined;
  const playing = isExternallyControlled ? isPlaying : animated;
  const frameIdx = isExternallyControlled && currentFrame !== undefined ? currentFrame : internalFrameIdx;

  // Animation loop using requestAnimationFrame for smooth speed control
  const animate = useCallback(
    (timestamp: number) => {
      if (!playing || totalFrames === 0) return;

      if (lastTimeRef.current === 0) {
        lastTimeRef.current = timestamp;
      }

      const elapsed = timestamp - lastTimeRef.current;
      // Base interval: ~60ms (~16fps), adjusted by playbackSpeed
      const interval = 60 / playbackSpeed;

      if (elapsed >= interval) {
        lastTimeRef.current = timestamp;
        if (isExternallyControlled) {
          onFrameUpdate?.((frameIdx + 1) % totalFrames);
        } else {
          setInternalFrameIdx((f) => (f + 1) % totalFrames);
        }
      }

      animFrameRef.current = requestAnimationFrame(animate);
    },
    [playing, totalFrames, playbackSpeed, isExternallyControlled, onFrameUpdate, frameIdx]
  );

  useEffect(() => {
    if (playing && totalFrames > 0) {
      lastTimeRef.current = 0;
      animFrameRef.current = requestAnimationFrame(animate);
    }
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [playing, animate, totalFrames]);

  const safeFrameIdx = totalFrames > 0 ? frameIdx % totalFrames : 0;

  const currentJoints =
    framesData && framesData.length > 0
      ? framesData[safeFrameIdx].joints
      : DEFAULT_JOINTS;

  const getJointPos = (joint: Joint) => {
    if (framesData && framesData.length > 0) {
      return { x: joint.x, y: joint.y };
    }
    if (!animated) return { x: joint.x, y: joint.y };
    const wobble = Math.sin(safeFrameIdx * 0.05 + joint.x * 10) * 0.02;
    const wobbleY = Math.cos(safeFrameIdx * 0.04 + joint.y * 8) * 0.015;
    return { x: joint.x + wobble, y: joint.y + wobbleY };
  };

  const jointMap = Object.fromEntries(
    currentJoints.map((j) => [j.id, getJointPos(j)])
  );

  const availableBones = SKELETON_BONES.filter(
    ([from, to]) => jointMap[from as string] && jointMap[to as string]
  );

  return (
    <svg
      ref={svgRef}
      viewBox="0 0 1 1"
      className="w-full h-full"
      role="img"
      aria-label="Skeleton animation preview showing body pose"
    >
      {availableBones.map(([from, to], i) => {
        const a = jointMap[from as string];
        const b = jointMap[to as string];
        return (
          <line
            key={i}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            stroke="hsl(var(--primary))"
            strokeWidth="0.008"
            strokeLinecap="round"
          />
        );
      })}
      {currentJoints.map((joint) => {
        const pos = jointMap[joint.id];
        if (!pos) return null;
        return (
          <circle
            key={joint.id}
            cx={pos.x}
            cy={pos.y}
            r={joint.id === "head" ? 0.025 : 0.012}
            fill="hsl(var(--primary))"
          />
        );
      })}
    </svg>
  );
}
