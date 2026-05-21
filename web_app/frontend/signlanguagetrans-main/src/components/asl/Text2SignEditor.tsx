/**
 * Text2SignEditor.tsx
 *
 * Pipeline giống app_optimized.py → text_to_sign_tab.py + sign_video_player_pil.py:
 *   1. /translate/gloss          → Gemini / fallback extract ASL glosses
 *   2. /translate/synthesize     → MotionSynthesizer → Hermite transition → frames (T,153,3)
 *   3. SkeletonCanvas             → HTML5 Canvas, mirrors SignVideoPlayerPIL
 *
 * Buttons hoàn chỉnh:
 *   ⏮ Seek start | ⏸/▶ Play/Pause | ⏭ Seek end
 *   Speed: 0.5x | 1x | 1.5x | 2x
 *   Loop toggle
 *   Progress bar (seek by drag)
 *   ⬇ GIF  → POST /export-gif → download animated GIF (PIL/imageio server-side)
 *   ⬇ Frames → client-side render all frames → ZIP of PNGs (jszip)
 */

import { useState, useRef, useCallback } from "react";
import JSZip from "jszip";
import { MOCK_TEXT2SIGN_INPUT } from "@/lib/mockData";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  ArrowRight, Play, Pause, SkipBack, SkipForward,
  Download, X, Loader2, Sparkles, AlertCircle, Zap,
  Film, ImageDown,
} from "lucide-react";
import { SkeletonCanvas, SkeletonCanvasHandle } from "./SkeletonCanvas";

const BACKEND_URL = "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface GlossResult {
  glosses: string[];
  missing_glosses: string[];
  method: "gemini" | "fallback";
}

interface SynthesizeResult {
  success: boolean;
  frames: number[][][];
  n_frames: number;
  fps: number;
  glosses_used: string[];
  missing_glosses: string[];
  error?: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function Text2SignEditor() {
  const [input, setInput]                     = useState(MOCK_TEXT2SIGN_INPUT);

  // ── Gloss state
  const [glossTokens, setGlossTokens]         = useState<string[]>([]);
  const [missingGlosses, setMissingGlosses]   = useState<string[]>([]);
  const [method, setMethod]                   = useState<"gemini" | "fallback" | null>(null);
  const [glossLoading, setGlossLoading]       = useState(false);
  const [glossError, setGlossError]           = useState<string | null>(null);

  // ── Animation state
  const [frames, setFrames]                   = useState<number[][][]>([]);
  const [synthFps, setSynthFps]               = useState(25);
  const [synthLoading, setSynthLoading]       = useState(false);
  const [synthError, setSynthError]           = useState<string | null>(null);
  const [glossesUsed, setGlossesUsed]         = useState<string[]>([]);

  // ── Playback state
  const [playing, setPlaying]                 = useState(false);
  const [speed, setSpeed]                     = useState<number>(1);
  const [loop, setLoop]                       = useState(true);
  const [currentFrame, setCurrentFrame]       = useState(0);

  // ── Export state
  const [gifLoading, setGifLoading]           = useState(false);
  const [framesLoading, setFramesLoading]     = useState(false);
  const [framesProgress, setFramesProgress]   = useState(0);

  // ── Canvas ref (imperative)
  const canvasRef = useRef<SkeletonCanvasHandle>(null);

  const hasGlosses = glossTokens.length > 0;
  const hasFrames  = frames.length > 0;

  // ---------------------------------------------------------------------------
  // Step 1: Translate text → glosses
  // ---------------------------------------------------------------------------

  const handleTranslate = async () => {
    if (!input.trim()) return;
    setGlossLoading(true);
    setGlossError(null);
    setGlossTokens([]);
    setMissingGlosses([]);
    setMethod(null);
    setFrames([]);
    setSynthError(null);
    setPlaying(false);
    setCurrentFrame(0);

    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/translate/gloss`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input, use_gemini: true }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || `Error ${res.status}`);
      const data: GlossResult = await res.json();
      setGlossTokens(data.glosses ?? []);
      setMissingGlosses(data.missing_glosses ?? []);
      setMethod(data.method ?? null);

      if (data.glosses?.length > 0) {
        await handleSynthesize(data.glosses);
      }
    } catch (err: unknown) {
      setGlossError(err instanceof Error ? err.message : "Translation failed");
    } finally {
      setGlossLoading(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Step 2: Synthesize glosses → skeleton frames
  // ---------------------------------------------------------------------------

  const handleSynthesize = async (glosses: string[]) => {
    if (!glosses.length) return;
    setSynthLoading(true);
    setSynthError(null);
    setFrames([]);
    setPlaying(false);
    setCurrentFrame(0);

    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/translate/synthesize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ glosses, fps: 25 }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || `Error ${res.status}`);
      const data: SynthesizeResult = await res.json();
      if (!data.success || !data.frames?.length) throw new Error(data.error || "No frames");

      setFrames(data.frames);
      setSynthFps(data.fps ?? 25);
      setGlossesUsed(data.glosses_used ?? []);
      setPlaying(true);
    } catch (err: unknown) {
      setSynthError(err instanceof Error ? err.message : "Synthesis failed");
    } finally {
      setSynthLoading(false);
    }
  };

  const handleResynthesizeTokens = () => handleSynthesize(glossTokens);
  const removeToken = (index: number) => {
    setGlossTokens(prev => prev.filter((_, i) => i !== index));
    setFrames([]); setPlaying(false);
  };

  // ---------------------------------------------------------------------------
  // Playback controls (mirror SignVideoPlayerPIL)
  // ---------------------------------------------------------------------------

  const handleSeekStart = () => {
    setPlaying(false);
    setCurrentFrame(0);
    canvasRef.current?.seekToFrame(0);
  };

  const handleSeekEnd = () => {
    if (!hasFrames) return;
    setPlaying(false);
    const last = frames.length - 1;
    setCurrentFrame(last);
    canvasRef.current?.seekToFrame(last);
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = Number(e.target.value);
    setCurrentFrame(f);
    setPlaying(false);
    canvasRef.current?.seekToFrame(f);
  };

  const handleFrameChange = useCallback((f: number) => {
    setCurrentFrame(f);
  }, []);

  const handlePlayEnd = useCallback(() => {
    if (!loop) setPlaying(false);
  }, [loop]);

  // ---------------------------------------------------------------------------
  // Export: GIF (server-side, SignVisualizer/PIL)
  // ---------------------------------------------------------------------------

  const handleDownloadGif = async () => {
    if (!hasFrames || !glossesUsed.length) return;
    setGifLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/translate/export-gif`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ glosses: glossesUsed, fps: 15 }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "GIF export failed");
      }
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href     = url;
      a.download = glossesUsed.join("_").slice(0, 40) + ".gif";
      a.click();
      URL.revokeObjectURL(url);
    } catch (err: unknown) {
      setSynthError(err instanceof Error ? err.message : "GIF export failed");
    } finally {
      setGifLoading(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Export: Frames ZIP (client-side, jszip + SkeletonCanvas.captureAllFramesPNG)
  // ---------------------------------------------------------------------------

  const handleDownloadFrames = async () => {
    if (!hasFrames) return;
    setFramesLoading(true);
    setFramesProgress(0);

    try {
      const zip = new JSZip();
      const folder = zip.folder("frames")!;
      const total  = frames.length;

      // Render all frames to PNG via SkeletonCanvas imperative handle
      const blobs = await canvasRef.current?.captureAllFramesPNG((i, t) => {
        setFramesProgress(Math.round((i / t) * 100));
      }) ?? [];

      blobs.forEach((blob, i) => {
        const name = `frame_${String(i).padStart(4, "0")}.png`;
        folder.file(name, blob);
      });

      // Add a manifest.json
      folder.file("manifest.json", JSON.stringify({
        total_frames: total,
        fps: synthFps,
        glosses: glossesUsed,
        width: 480,
        height: 480,
      }, null, 2));

      const zipBlob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(zipBlob);
      const a   = document.createElement("a");
      a.href    = url;
      a.download = `asl_${glossesUsed.join("_").slice(0, 30)}_frames.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err: unknown) {
      setSynthError(err instanceof Error ? err.message : "Frames export failed");
    } finally {
      setFramesLoading(false);
      setFramesProgress(0);
    }
  };

  const isLoading = glossLoading || synthLoading;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4" style={{ height: "calc(100vh - 11.5rem)" }}>

      {/* ── LEFT: Input + Gloss ── */}
      <div className="flex flex-col min-h-0">

        {/* Text Input */}
        <div className="asl-panel">
          <div className="asl-panel-header">
            <h2 className="text-sm font-semibold">English Input</h2>
          </div>
          <div className="asl-panel-body space-y-3">
            <Textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type an English sentence to convert to ASL gloss"
              className="min-h-[80px] max-h-[80px] text-base resize-none"
              aria-label="English text input for ASL translation"
            />
            <Button
              className="w-full touch-target"
              onClick={handleTranslate}
              disabled={isLoading}
              aria-label="Translate text to ASL gloss"
            >
              {glossLoading
                ? <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                : <ArrowRight className="w-4 h-4 mr-2" aria-hidden />}
              {glossLoading ? "Translating…" : "Translate to ASL Gloss"}
            </Button>
            {glossError && (
              <div className="flex items-center gap-2 text-destructive text-xs mt-1" role="alert">
                <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                <span>{glossError}</span>
              </div>
            )}
          </div>
        </div>

        {/* Gloss Output */}
        <div className="asl-panel flex-1 flex flex-col min-h-0 mt-4">
          <div className="asl-panel-header">
            <h2 className="text-sm font-semibold">Extracted Gloss</h2>
            <div className="flex items-center gap-2 flex-wrap">
              {hasGlosses && <span className="text-xs text-muted-foreground">{glossTokens.length} tokens</span>}
              {method === "gemini" && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-600 dark:text-purple-400 text-xs font-medium">
                  <Sparkles className="w-3 h-3" />Gemini
                </span>
              )}
              {method === "fallback" && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 text-xs font-medium">
                  Fallback
                </span>
              )}
              {hasGlosses && !synthLoading && (
                <button
                  onClick={handleResynthesizeTokens}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-green-500/10 text-green-600 dark:text-green-400 text-xs font-medium hover:bg-green-500/20 transition"
                  title="Re-generate animation from current tokens"
                >
                  <Zap className="w-3 h-3" />Animate
                </button>
              )}
            </div>
          </div>

          <div className="asl-panel-body overflow-y-auto flex-1">
            {synthLoading && (
              <div className="flex items-center gap-2 text-muted-foreground text-xs mb-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Synthesizing skeleton animation…</span>
              </div>
            )}
            {synthError && (
              <div className="flex items-center gap-2 text-yellow-600 text-xs mb-2" role="alert">
                <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                <span>{synthError}</span>
              </div>
            )}

            {hasGlosses ? (
              <div className="flex flex-wrap gap-2" role="list" aria-label="ASL gloss tokens">
                {glossTokens.map((token, i) => (
                  <span
                    key={i}
                    role="listitem"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-primary/10 text-primary font-mono text-sm font-semibold"
                  >
                    {token}
                    <button onClick={() => removeToken(i)} className="hover:text-destructive" aria-label={`Remove ${token}`}>
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                {glossLoading ? "Translating with AI…" : 'Press "Translate to ASL Gloss" to begin.'}
              </p>
            )}

            {missingGlosses.length > 0 && (
              <div className="mt-3 pt-3 border-t border-border">
                <p className="text-xs text-muted-foreground mb-1.5">
                  ⚠️ Not in dictionary ({missingGlosses.length} words skipped):
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {missingGlosses.map((g, i) => (
                    <span key={i} className="px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-mono text-xs line-through">{g}</span>
                  ))}
                </div>
              </div>
            )}

            {hasFrames && glossesUsed.length > 0 && (
              <div className="mt-3 pt-3 border-t border-border text-xs text-muted-foreground">
                <span className="font-medium text-green-600">✓ Animated:</span>{" "}
                {glossesUsed.join(" → ")}
                <span className="ml-2">({frames.length} frames @ {synthFps}fps)</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── RIGHT: Animation Preview ── */}
      <div className="flex flex-col min-h-0">
        <div className="asl-panel flex-1 flex flex-col min-h-0">
          <div className="asl-panel-header">
            <h2 className="text-sm font-semibold">Sign Animation Preview</h2>
            {hasFrames && (
              <span className="text-xs text-muted-foreground">
                {currentFrame + 1} / {frames.length}
              </span>
            )}
          </div>

          {/* Canvas */}
          <div className="flex-1 bg-white rounded-lg overflow-hidden min-h-0 flex items-center justify-center">
            {synthLoading ? (
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                <Loader2 className="w-8 h-8 animate-spin" />
                <p className="text-sm">Synthesizing animation…</p>
                <p className="text-xs opacity-70">Hermite splines → IK constraints → Smoothing</p>
              </div>
            ) : (
              <SkeletonCanvas
                ref={canvasRef}
                frames={frames}
                fps={synthFps}
                playing={playing}
                speed={speed}
                loop={loop}
                onFrameChange={handleFrameChange}
                onPlayEnd={handlePlayEnd}
              />
            )}
          </div>

          {/* Progress bar */}
          {hasFrames && (
            <div className="px-3 pt-2">
              <input
                type="range"
                min={0}
                max={frames.length - 1}
                value={currentFrame}
                onChange={handleSliderChange}
                className="w-full h-1.5 accent-primary cursor-pointer"
                aria-label="Animation progress"
                id="anim-progress"
              />
            </div>
          )}

          {/* Controls row 1: Seek + Play + Speed */}
          <div className="px-3 py-2 flex items-center gap-2 flex-wrap">
            {/* Seek start ⏮ */}
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              disabled={!hasFrames}
              onClick={handleSeekStart}
              aria-label="Seek to start"
              title="Seek to start"
            >
              <SkipBack className="w-3.5 h-3.5" />
            </Button>

            {/* Play / Pause ▶ ⏸ */}
            <Button
              variant={playing ? "default" : "outline"}
              className="touch-target px-4"
              onClick={() => setPlaying(p => !p)}
              disabled={!hasFrames}
              aria-label={playing ? "Pause animation" : "Play animation"}
              id="btn-play-pause"
            >
              {playing
                ? <Pause className="w-4 h-4 mr-1.5" />
                : <Play  className="w-4 h-4 mr-1.5" />}
              {playing ? "Pause" : "Play"}
            </Button>

            {/* Seek end ⏭ */}
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              disabled={!hasFrames}
              onClick={handleSeekEnd}
              aria-label="Seek to end"
              title="Seek to end"
            >
              <SkipForward className="w-3.5 h-3.5" />
            </Button>

            {/* Speed buttons — matches SPEED_OPTIONS from sign_video_player_pil.py */}
            <div className="flex items-center gap-1" role="group" aria-label="Playback speed">
              {([0.5, 1, 1.5, 2] as const).map(s => (
                <Button
                  key={s}
                  variant={speed === s ? "default" : "outline"}
                  size="sm"
                  className="touch-target text-xs h-8 px-2.5"
                  onClick={() => setSpeed(s)}
                  aria-label={`Speed ${s}x`}
                  aria-pressed={speed === s}
                  id={`btn-speed-${s}`}
                >
                  {s}x
                </Button>
              ))}
            </div>

            {/* Loop toggle */}
            <Button
              variant={loop ? "default" : "outline"}
              size="sm"
              className="touch-target text-xs h-8 px-3"
              onClick={() => setLoop(l => !l)}
              aria-pressed={loop}
              aria-label={loop ? "Disable loop" : "Enable loop"}
              id="btn-loop"
            >
              {loop ? "Loop ✓" : "Loop"}
            </Button>
          </div>

          {/* Controls row 2: Export */}
          <div className="px-3 pb-3 flex items-center gap-2 border-t border-border pt-2">
            <span className="text-xs text-muted-foreground mr-1">Export:</span>

            {/* Download GIF — server-side PIL render */}
            <Button
              variant="outline"
              size="sm"
              className="touch-target gap-1.5"
              disabled={!hasFrames || gifLoading || framesLoading}
              onClick={handleDownloadGif}
              aria-label="Download animated GIF"
              id="btn-download-gif"
              title="Download animated GIF (server renders with PIL)"
            >
              {gifLoading
                ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                : <Film className="w-3.5 h-3.5" />}
              {gifLoading ? "Generating…" : "GIF"}
            </Button>

            {/* Download Frames ZIP — client-side JSZip */}
            <Button
              variant="outline"
              size="sm"
              className="touch-target gap-1.5"
              disabled={!hasFrames || gifLoading || framesLoading}
              onClick={handleDownloadFrames}
              aria-label="Download all frames as PNG ZIP"
              id="btn-download-frames"
              title="Download all frames as PNG images in a ZIP file"
            >
              {framesLoading
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" />{framesProgress}%</>
                : <><ImageDown className="w-3.5 h-3.5" />Frames</>}
            </Button>

            {/* Frame info */}
            {hasFrames && (
              <span className="ml-auto text-xs text-muted-foreground">
                {frames.length} frames · {synthFps}fps · {(frames.length / synthFps).toFixed(1)}s
              </span>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}
