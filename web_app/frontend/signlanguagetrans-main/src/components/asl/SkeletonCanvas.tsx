import { useEffect, useRef, useCallback } from "react";

// =============================================================================
// Rich frame data from the backend (153 landmarks, segmented)
// =============================================================================
export type RichFrameData = {
  frame: number;
  lips: number[][];       // 40 × [x, y]
  oval: number[][];       // 36 × [x, y]
  eyebrows: number[][];   // 10 × [x, y]
  eyes: number[][];       // 16 × [x, y]
  left_hand: number[][];  // 21 × [x, y]
  pose: number[][];       // 9 × [x, y]
  right_hand: number[][]; // 21 × [x, y]
  hand_valid: { left: boolean; right: boolean };
};

type Props = {
  framesData: RichFrameData[];
  fps?: number;
  isPlaying?: boolean;
  playbackSpeed?: number;
  currentFrame?: number;
  onFrameUpdate?: (frame: number) => void;
};

// =============================================================================
// Constants — mirrors scripts/slp_config.py
// =============================================================================

// Pose order: [Nose(0), LShoulder(1), RShoulder(2), LElbow(3), RElbow(4),
//              LWrist(5), RWrist(6), LHip(7), RHip(8)]
const POSE_BONES: [number, number][] = [
  [1, 2],  // shoulders
  [1, 3],  // L shoulder → L elbow
  [3, 5],  // L elbow → L wrist
  [2, 4],  // R shoulder → R elbow
  [4, 6],  // R elbow → R wrist
  [1, 7],  // L shoulder → L hip
  [2, 8],  // R shoulder → R hip
  [7, 8],  // hips
];

// Hand finger chains (local indices 0-20 within each hand)
const FINGER_CHAINS = [
  [0, 1, 2, 3, 4],       // Thumb
  [0, 5, 6, 7, 8],       // Index
  [0, 9, 10, 11, 12],    // Middle
  [0, 13, 14, 15, 16],   // Ring
  [0, 17, 18, 19, 20],   // Pinky
];

const FINGER_COLORS = ["#DDDD00", "#00DD00", "#00DDDD", "#4466FF", "#DD00DD"];

// Face sub-split: eyebrows 5+5, eyes 8+8
const LEFT_EYEBROW_COUNT = 5;
const LEFT_EYE_COUNT = 8;

// =============================================================================
// Drawing helpers
// =============================================================================

function drawClosedLoop(
  ctx: CanvasRenderingContext2D,
  pts: number[][],
  color: string,
  lineWidth: number,
  w: number,
  h: number
) {
  if (pts.length < 2) return;
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.moveTo(pts[0][0] * w, pts[0][1] * h);
  for (let i = 1; i < pts.length; i++) {
    ctx.lineTo(pts[i][0] * w, pts[i][1] * h);
  }
  ctx.closePath();
  ctx.stroke();
}

function drawOpenPath(
  ctx: CanvasRenderingContext2D,
  pts: number[][],
  color: string,
  lineWidth: number,
  w: number,
  h: number
) {
  if (pts.length < 2) return;
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.moveTo(pts[0][0] * w, pts[0][1] * h);
  for (let i = 1; i < pts.length; i++) {
    ctx.lineTo(pts[i][0] * w, pts[i][1] * h);
  }
  ctx.stroke();
}

function drawLine(
  ctx: CanvasRenderingContext2D,
  p1: number[],
  p2: number[],
  color: string,
  lineWidth: number,
  w: number,
  h: number
) {
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.moveTo(p1[0] * w, p1[1] * h);
  ctx.lineTo(p2[0] * w, p2[1] * h);
  ctx.stroke();
}

function drawGlowDot(
  ctx: CanvasRenderingContext2D,
  pt: number[],
  radius: number,
  color: string,
  w: number,
  h: number
) {
  const x = pt[0] * w;
  const y = pt[1] * h;

  // Glow
  const grad = ctx.createRadialGradient(x, y, 0, x, y, radius * 2.5);
  grad.addColorStop(0, color);
  grad.addColorStop(1, "transparent");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(x, y, radius * 2.5, 0, Math.PI * 2);
  ctx.fill();

  // Solid center
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}

// =============================================================================
// Component
// =============================================================================

export function SkeletonCanvas({
  framesData,
  fps = 20,
  isPlaying = true,
  playbackSpeed = 1,
  currentFrame = 0,
  onFrameUpdate,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);

  const totalFrames = framesData.length;

  // ── Render a single frame ──
  const renderFrame = useCallback(
    (frameIdx: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;
      const frame = framesData[frameIdx % totalFrames];

      // Clear + dark background
      ctx.fillStyle = "#0a0a12";
      ctx.fillRect(0, 0, w, h);

      // Subtle grid lines
      ctx.strokeStyle = "rgba(255,255,255,0.03)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 10; i++) {
        const x = (i / 10) * w;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        const y = (i / 10) * h;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }

      // ── 1. Face ──
      const faceColor = "rgba(180, 140, 140, 0.5)";
      const lipsColor = "rgba(220, 160, 160, 0.7)";
      const faceLineWidth = 1.5;

      // Face oval (closed loop)
      drawClosedLoop(ctx, frame.oval, faceColor, faceLineWidth, w, h);

      // Lips (closed loop)
      drawClosedLoop(ctx, frame.lips, lipsColor, faceLineWidth + 0.5, w, h);

      // Eyebrows (2 separate open arcs)
      const leftBrow = frame.eyebrows.slice(0, LEFT_EYEBROW_COUNT);
      const rightBrow = frame.eyebrows.slice(LEFT_EYEBROW_COUNT);
      drawOpenPath(ctx, leftBrow, faceColor, faceLineWidth, w, h);
      drawOpenPath(ctx, rightBrow, faceColor, faceLineWidth, w, h);

      // Eyes (2 closed loops)
      const leftEye = frame.eyes.slice(0, LEFT_EYE_COUNT);
      const rightEye = frame.eyes.slice(LEFT_EYE_COUNT);
      drawClosedLoop(ctx, leftEye, faceColor, faceLineWidth, w, h);
      drawClosedLoop(ctx, rightEye, faceColor, faceLineWidth, w, h);

      // ── 2. Pose skeleton ──
      const poseColor = "#38bdf8"; // sky blue
      const poseLW = 3;

      for (const [a, b] of POSE_BONES) {
        const p1 = frame.pose[a];
        const p2 = frame.pose[b];
        if (p1 && p2) {
          drawLine(ctx, p1, p2, poseColor, poseLW, w, h);
        }
      }

      // Pose-to-hand connections (wrist → hand wrist)
      if (frame.hand_valid.left) {
        const poseWrist = frame.pose[5]; // L_Wrist
        const handWrist = frame.left_hand[0];
        if (poseWrist && handWrist) {
          drawLine(ctx, poseWrist, handWrist, poseColor, poseLW, w, h);
        }
      }
      if (frame.hand_valid.right) {
        const poseWrist = frame.pose[6]; // R_Wrist
        const handWrist = frame.right_hand[0];
        if (poseWrist && handWrist) {
          drawLine(ctx, poseWrist, handWrist, poseColor, poseLW, w, h);
        }
      }

      // Nose → neck midpoint
      const nose = frame.pose[0];
      const lShoulder = frame.pose[1];
      const rShoulder = frame.pose[2];
      if (nose && lShoulder && rShoulder) {
        const neck = [(lShoulder[0] + rShoulder[0]) / 2, (lShoulder[1] + rShoulder[1]) / 2];
        drawLine(ctx, nose, neck, poseColor, poseLW, w, h);
      }

      // Pose joint dots with glow
      for (let i = 0; i < frame.pose.length; i++) {
        const pt = frame.pose[i];
        if (pt) {
          const r = i === 0 ? 6 : 4; // Nose bigger
          drawGlowDot(ctx, pt, r, poseColor, w, h);
        }
      }

      // ── 3. Hands ──
      const drawHand = (handPts: number[][], valid: boolean) => {
        if (!valid) return;
        for (let fi = 0; fi < FINGER_CHAINS.length; fi++) {
          const chain = FINGER_CHAINS[fi];
          const color = FINGER_COLORS[fi];
          const chainPts = chain.map((idx) => handPts[idx]).filter(Boolean);
          if (chainPts.length >= 2) {
            drawOpenPath(ctx, chainPts, color, 2.5, w, h);
          }
        }
        // Fingertip dots
        for (const pt of handPts) {
          if (pt) {
            drawGlowDot(ctx, pt, 2.5, "rgba(255,255,255,0.7)", w, h);
          }
        }
      };

      drawHand(frame.left_hand, frame.hand_valid.left);
      drawHand(frame.right_hand, frame.hand_valid.right);
    },
    [framesData, totalFrames]
  );

  // ── Animation loop ──
  const animate = useCallback(
    (timestamp: number) => {
      if (!isPlaying || totalFrames === 0) return;

      if (lastTimeRef.current === 0) {
        lastTimeRef.current = timestamp;
      }

      const elapsed = timestamp - lastTimeRef.current;
      const interval = (1000 / fps) / playbackSpeed;

      if (elapsed >= interval) {
        lastTimeRef.current = timestamp;
        const nextFrame = (currentFrame + 1) % totalFrames;
        onFrameUpdate?.(nextFrame);
      }

      renderFrame(currentFrame);
      animRef.current = requestAnimationFrame(animate);
    },
    [isPlaying, totalFrames, fps, playbackSpeed, currentFrame, onFrameUpdate, renderFrame]
  );

  // Start/stop animation
  useEffect(() => {
    if (isPlaying && totalFrames > 0) {
      lastTimeRef.current = 0;
      animRef.current = requestAnimationFrame(animate);
    } else {
      // Render current frame when paused
      renderFrame(currentFrame);
    }
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isPlaying, animate, totalFrames, currentFrame, renderFrame]);

  // ── Resize canvas to match container ──
  useEffect(() => {
    const resize = () => {
      const container = containerRef.current;
      const canvas = canvasRef.current;
      if (!container || !canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = container.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.scale(dpr, dpr);
      // Re-render after resize (use logical size)
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      renderFrame(currentFrame);
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [currentFrame, renderFrame]);

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full rounded-xl" />
    </div>
  );
}
