/**
 * SkeletonCanvas.tsx
 *
 * Render skeleton animation (T × 153 × 3) lên HTML5 Canvas.
 * Giống SignVideoPlayerPIL (PIL version) — vẽ bằng Canvas 2D API.
 *
 * Expose qua forwardRef:
 *   - seekToFrame(idx)   — seek tới frame cụ thể
 *   - captureFramePNG(idx) — render 1 frame thành PNG blob (cho Frames export)
 *
 * Landmark layout 153 points (từ slp_config.py):
 *   [0:40]    Lips (40)
 *   [40:76]   Face oval (36)
 *   [76:86]   Eyebrows (10)
 *   [86:102]  Eyes (16)
 *   [102:123] Left hand (21)
 *   [123:132] Pose: Nose, L/R Shoulder, L/R Elbow, L/R Wrist, L/R Hip (9)
 *   [132:153] Right hand (21)
 */

import {
  useEffect, useRef, useCallback, useState,
  forwardRef, useImperativeHandle,
} from "react";

// ---------------------------------------------------------------------------
// Index constants  (mirror slp_config.py)
// ---------------------------------------------------------------------------

function range(start: number, end: number): number[] {
  return Array.from({ length: end - start }, (_, i) => i + start);
}

const IDX_LIPS       = range(0, 40);
const IDX_FACE_OVAL  = range(40, 76);
const IDX_EYEBROWS   = range(76, 86);
const IDX_EYES       = range(86, 102);
const IDX_LEFT_HAND  = range(102, 123);
const IDX_POSE       = range(123, 132);
const IDX_RIGHT_HAND = range(132, 153);

// Local eye/eyebrow splits
const IDX_LEFT_EYE_LOCAL        = range(0, 8);
const IDX_RIGHT_EYE_LOCAL       = range(8, 16);
const IDX_LEFT_EYEBROW_LOCAL    = range(0, 5);
const IDX_RIGHT_EYEBROW_LOCAL   = range(5, 10);

// Pose connections (absolute global indices)
const POSE_CONNECTIONS: [number, number][] = [
  [IDX_POSE[1], IDX_POSE[2]],  // L_Shoulder — R_Shoulder
  [IDX_POSE[1], IDX_POSE[3]],  // L_Shoulder — L_Elbow
  [IDX_POSE[3], IDX_POSE[5]],  // L_Elbow    — L_Wrist
  [IDX_POSE[2], IDX_POSE[4]],  // R_Shoulder — R_Elbow
  [IDX_POSE[4], IDX_POSE[6]],  // R_Elbow    — R_Wrist
  [IDX_POSE[1], IDX_POSE[7]],  // L_Shoulder — L_Hip
  [IDX_POSE[2], IDX_POSE[8]],  // R_Shoulder — R_Hip
  [IDX_POSE[7], IDX_POSE[8]],  // L_Hip      — R_Hip
];

const POSE_HAND_CONNECTIONS: [number, number][] = [
  [IDX_POSE[5], IDX_LEFT_HAND[0]],   // L_Wrist → Left  hand wrist
  [IDX_POSE[6], IDX_RIGHT_HAND[0]],  // R_Wrist → Right hand wrist
];

// Finger chains (local indices 0–20, same HAND_FINGER_CHAINS in slp_config)
const HAND_FINGER_CHAINS: number[][] = [
  [0, 1, 2, 3, 4],     // Thumb
  [0, 5, 6, 7, 8],     // Index
  [0, 9, 10, 11, 12],  // Middle
  [0, 13, 14, 15, 16], // Ring
  [0, 17, 18, 19, 20], // Pinky
];

// Same as FINGER_COLORS_RGB in sign_video_player_pil.py
const FINGER_COLORS: string[] = [
  "#ff0000",  // Thumb  — red
  "#ffa500",  // Index  — orange
  "#00cc00",  // Middle — green
  "#0000ff",  // Ring   — blue
  "#800080",  // Pinky  — purple
];

// ---------------------------------------------------------------------------
// RichFrameData — format returned by /dictionary/{word}/skeleton
// ---------------------------------------------------------------------------

export interface RichFrameData {
  frame:      number;
  lips:       number[][];   // (40, 2)
  oval:       number[][];   // (36, 2)
  eyebrows:   number[][];   // (10, 2)
  eyes:       number[][];   // (16, 2)
  left_hand:  number[][];   // (21, 2)
  pose:       number[][];   // ( 9, 2)
  right_hand: number[][];   // (21, 2)
  hand_valid?: { left: boolean; right: boolean };
}

/**
 * Convert a RichFrameData → 153-point Frame (each point [x, y, 0]).
 * Layout (mirrors slp_config.py):
 *   [0:40]    lips
 *   [40:76]   oval
 *   [76:86]   eyebrows
 *   [86:102]  eyes
 *   [102:123] left_hand
 *   [123:132] pose
 *   [132:153] right_hand
 */
function richToFrame(r: RichFrameData): Frame {
  const pad = (arr: number[][], n: number): Frame =>
    Array.from({ length: n }, (_, i) => {
      const p = arr?.[i];
      return p ? [p[0], p[1], 0] : [0, 0, 0];
    });
  return [
    ...pad(r.lips, 40),
    ...pad(r.oval, 36),
    ...pad(r.eyebrows, 10),
    ...pad(r.eyes, 16),
    ...pad(r.left_hand, 21),
    ...pad(r.pose, 9),
    ...pad(r.right_hand, 21),
  ];
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------

type Pt = number[];  // [x, y, z]
type Frame = Pt[];   // 153 points

function isInvalidPt(pt: Pt): boolean {
  return !pt || isNaN(pt[0]) || isNaN(pt[1]) ||
    (Math.abs(pt[0]) < 1e-9 && Math.abs(pt[1]) < 1e-9);
}

function isInvalid(pts: Pt[]): boolean {
  if (!pts || pts.length === 0) return true;
  return pts.every(p => isNaN(p[0]) || isNaN(p[1])) ||
    pts.every(p => Math.abs(p[0]) < 1e-9 && Math.abs(p[1]) < 1e-9);
}

function isHandValid(frame: Frame, indices: number[]): boolean {
  const pts = indices.map(i => frame[i]);
  if (!pts[0] || isInvalidPt(pts[0])) return false;
  if (isInvalid(pts)) return false;
  const [fx, fy] = pts[0];
  return pts.some(p => Math.abs(p[0] - fx) > 1e-6 || Math.abs(p[1] - fy) > 1e-6);
}

/** Normalized 0–1 → pixel, with 10% margin like _to_pixel() in PIL player */
function toPixel(x: number, y: number, W: number, H: number): [number, number] {
  const m = 0.1;
  return [
    Math.round((x + m) * W / (1 + 2 * m)),
    Math.round((y + m) * H / (1 + 2 * m)),
  ];
}

function drawLine(
  ctx: CanvasRenderingContext2D, p1: Pt, p2: Pt,
  W: number, H: number, color: string, lw: number
) {
  if (isInvalidPt(p1) || isInvalidPt(p2)) return;
  const [x1, y1] = toPixel(p1[0], p1[1], W, H);
  const [x2, y2] = toPixel(p2[0], p2[1], W, H);
  ctx.strokeStyle = color; ctx.lineWidth = lw;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
}

function drawPolyline(
  ctx: CanvasRenderingContext2D, pts: Pt[],
  W: number, H: number, color: string, lw: number, closed = false
) {
  const valid = pts.filter(p => !isInvalidPt(p));
  if (valid.length < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = lw;
  ctx.beginPath();
  const [sx, sy] = toPixel(valid[0][0], valid[0][1], W, H);
  ctx.moveTo(sx, sy);
  for (let i = 1; i < valid.length; i++) {
    const [px, py] = toPixel(valid[i][0], valid[i][1], W, H);
    ctx.lineTo(px, py);
  }
  if (closed) ctx.closePath();
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Core draw function — mirrors _draw_frame in SignVideoPlayerPIL
// ---------------------------------------------------------------------------

function drawFrame(ctx: CanvasRenderingContext2D, frame: Frame, W: number, H: number) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, W, H);

  const fc = "#8b0000"; // darkred — face

  // ── Face ─────────────────────────────────────────────
  // Face Oval (closed polygon)
  drawPolyline(ctx, IDX_FACE_OVAL.map(i => frame[i]), W, H, fc, 2, true);
  // Lips (closed)
  drawPolyline(ctx, IDX_LIPS.map(i => frame[i]),      W, H, fc, 2, true);
  // Eyebrows
  const eb = IDX_EYEBROWS.map(i => frame[i]);
  if (!isInvalid(eb)) {
    drawPolyline(ctx, IDX_LEFT_EYEBROW_LOCAL.map(i => eb[i]),  W, H, fc, 2);
    drawPolyline(ctx, IDX_RIGHT_EYEBROW_LOCAL.map(i => eb[i]), W, H, fc, 2);
  }
  // Eyes (closed)
  const ey = IDX_EYES.map(i => frame[i]);
  if (!isInvalid(ey)) {
    drawPolyline(ctx, IDX_LEFT_EYE_LOCAL.map(i => ey[i]),  W, H, fc, 2, true);
    drawPolyline(ctx, IDX_RIGHT_EYE_LOCAL.map(i => ey[i]), W, H, fc, 2, true);
  }

  // ── Pose skeleton ─────────────────────────────────────
  const pc = "#ff0000"; // red
  const lhv = isHandValid(frame, IDX_LEFT_HAND);
  const rhv = isHandValid(frame, IDX_RIGHT_HAND);

  for (const [i1, i2] of POSE_CONNECTIONS) {
    drawLine(ctx, frame[i1], frame[i2], W, H, pc, 3);
  }
  for (const [i1, i2] of POSE_HAND_CONNECTIONS) {
    if (i2 === IDX_LEFT_HAND[0]  && !lhv) continue;
    if (i2 === IDX_RIGHT_HAND[0] && !rhv) continue;
    drawLine(ctx, frame[i1], frame[i2], W, H, pc, 3);
  }

  // ── Hands ────────────────────────────────────────────
  const drawHand = (indices: number[]) => {
    if (!isHandValid(frame, indices)) return;
    const pts = indices.map(i => frame[i]);
    HAND_FINGER_CHAINS.forEach((chain, fi) =>
      drawPolyline(ctx, chain.map(j => pts[j]), W, H, FINGER_COLORS[fi], 2)
    );
  };
  drawHand(IDX_LEFT_HAND);
  drawHand(IDX_RIGHT_HAND);
}

// ---------------------------------------------------------------------------
// Public handle
// ---------------------------------------------------------------------------

export interface SkeletonCanvasHandle {
  seekToFrame: (idx: number) => void;
  captureFramePNG: (idx: number) => Promise<Blob | null>;
  captureAllFramesPNG: (onProgress?: (i: number, total: number) => void) => Promise<Blob[]>;
  getCurrentFrame: () => number;
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface SkeletonCanvasProps {
  // ── Mode A: Text2Sign — raw (T, 153, 3) ──────────────────────
  frames?: number[][][];

  // ── Mode B: Dictionary — segmented RichFrameData[] ──────────
  framesData?: RichFrameData[];

  fps?: number;
  playing?: boolean;
  isPlaying?: boolean;       // alias for Dictionary mode
  speed?: number;
  playbackSpeed?: number;    // alias for Dictionary mode
  loop?: boolean;
  onFrameChange?: (frame: number, total: number) => void;
  onFrameUpdate?: (frame: number) => void;  // alias for Dictionary mode
  onPlayEnd?: () => void;

  // Dictionary mode passes currentFrame externally (controlled)
  currentFrame?: number;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const SkeletonCanvas = forwardRef<SkeletonCanvasHandle, SkeletonCanvasProps>(
  function SkeletonCanvas(
    {
      frames: rawFrames,
      framesData,
      fps = 25,
      playing: playingProp,
      isPlaying,
      speed: speedProp,
      playbackSpeed,
      loop = true,
      onFrameChange,
      onFrameUpdate,
      onPlayEnd,
      currentFrame: controlledFrame,
    },
    ref
  ) {
    // Normalise dual-mode props
    const playing = playingProp ?? isPlaying ?? false;
    const speed   = speedProp   ?? playbackSpeed ?? 1;

    // Build unified frames array from whichever source was provided
    const frames: Frame[] = framesData
      ? framesData.map(richToFrame)
      : (rawFrames ?? []) as Frame[];
    const canvasRef    = useRef<HTMLCanvasElement>(null);
    const frameIdxRef  = useRef(0);
    const rafRef       = useRef<number | null>(null);
    const lastTimeRef  = useRef<number>(0);
    const [, forceRender] = useState(0); // triggers re-render for frame counter

    const totalFrames  = frames.length;
    const CANVAS_W     = 480;
    const CANVAS_H     = 480;

    // ── Draw a single frame ──────────────────────────────
    const renderFrame = useCallback((idx: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      if (frames[idx]) {
        drawFrame(ctx, frames[idx] as Frame, CANVAS_W, CANVAS_H);
      }
    }, [frames]);

    // ── Placeholder when no frames ───────────────────────
    const renderPlaceholder = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
      ctx.fillStyle = "#f8f8f8";
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      ctx.fillStyle = "#aaa";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No animation loaded", CANVAS_W / 2, CANVAS_H / 2);
    }, []);

    // ── Reset when frames change ─────────────────────────
    useEffect(() => {
      frameIdxRef.current = 0;
      forceRender(n => n + 1);
      if (frames.length > 0) renderFrame(0);
      else renderPlaceholder();
    }, [frames, renderFrame, renderPlaceholder]);

    // ── Animation loop ───────────────────────────────────
    useEffect(() => {
      if (!playing || totalFrames === 0) {
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
        return;
      }

      const msPerFrame = (1000 / fps) / speed;

      const tick = (ts: number) => {
        if (ts - lastTimeRef.current >= msPerFrame) {
          lastTimeRef.current = ts;
          const idx = frameIdxRef.current;
          renderFrame(idx);
          onFrameChange?.(idx, totalFrames);
          onFrameUpdate?.(idx);  // Dictionary alias
          forceRender(n => n + 1);

          let next = idx + 1;
          if (next >= totalFrames) {
            if (loop) next = 0;
            else { onPlayEnd?.(); return; }
          }
          frameIdxRef.current = next;
        }
        rafRef.current = requestAnimationFrame(tick);
      };

      lastTimeRef.current = 0;
      rafRef.current = requestAnimationFrame(tick);
      return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
    }, [playing, fps, speed, totalFrames, loop, renderFrame, onFrameChange, onPlayEnd]);

    // ── Controlled frame (Dictionary mode) ──────────────
    useEffect(() => {
      if (controlledFrame !== undefined && !playing) {
        const clamped = Math.max(0, Math.min(controlledFrame, frames.length - 1));
        frameIdxRef.current = clamped;
        renderFrame(clamped);
        forceRender(n => n + 1);
      }
    }, [controlledFrame, playing, frames.length, renderFrame]);

    // ── Render when paused & seeked ──────────────────────
    useEffect(() => {
      if (!playing && frames.length > 0) renderFrame(frameIdxRef.current);
    }, [playing, frames, renderFrame]);

    // ── Imperative handle ────────────────────────────────
    useImperativeHandle(ref, () => ({
      seekToFrame(idx: number) {
        if (!frames.length) return;
        const clamped = Math.max(0, Math.min(idx, frames.length - 1));
        frameIdxRef.current = clamped;
        renderFrame(clamped);
        forceRender(n => n + 1);
      },

      async captureFramePNG(idx: number): Promise<Blob | null> {
        if (!frames[idx]) return null;
        const offscreen = document.createElement("canvas");
        offscreen.width  = CANVAS_W;
        offscreen.height = CANVAS_H;
        const ctx = offscreen.getContext("2d");
        if (!ctx) return null;
        drawFrame(ctx, frames[idx] as Frame, CANVAS_W, CANVAS_H);
        return new Promise(resolve => offscreen.toBlob(resolve, "image/png"));
      },

      async captureAllFramesPNG(
        onProgress?: (i: number, total: number) => void
      ): Promise<Blob[]> {
        const blobs: Blob[] = [];
        const offscreen = document.createElement("canvas");
        offscreen.width  = CANVAS_W;
        offscreen.height = CANVAS_H;
        const ctx = offscreen.getContext("2d")!;
        for (let i = 0; i < frames.length; i++) {
          drawFrame(ctx, frames[i] as Frame, CANVAS_W, CANVAS_H);
          const blob: Blob | null = await new Promise(r => offscreen.toBlob(r, "image/png"));
          if (blob) blobs.push(blob);
          onProgress?.(i + 1, frames.length);
          // yield to keep UI responsive
          if (i % 10 === 0) await new Promise(r => setTimeout(r, 0));
        }
        return blobs;
      },

      getCurrentFrame() {
        return frameIdxRef.current;
      },
    }), [frames, renderFrame]);

    // ── Render ───────────────────────────────────────────
    return (
      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={CANVAS_H}
        className="w-full h-full object-contain"
        style={{ background: "#ffffff", display: "block" }}
        aria-label={`ASL skeleton animation — frame ${frameIdxRef.current + 1} of ${totalFrames}`}
      />
    );
  }
);
