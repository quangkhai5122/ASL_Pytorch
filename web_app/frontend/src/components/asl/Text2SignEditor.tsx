/**
 * Text2SignEditor.tsx — full API version
 * Pipeline: text → /translate/gloss → /translate/synthesize → SkeletonCanvas
 * Controls: Play/Pause, 0.5x/1x/1.5x/2x speed, seek, loop, GIF export, Frames ZIP
 */

import { useState, useRef, useCallback } from "react";
import JSZip from "jszip";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  ArrowRight, Play, Pause, SkipBack, SkipForward,
  Download, X, Loader2, Sparkles, AlertCircle, Zap,
  Film, ImageDown,
} from "lucide-react";
import { SkeletonCanvas, SkeletonCanvasHandle } from "./SkeletonCanvas";

const BACKEND_URL = "http://localhost:8000";

// ── Types ──────────────────────────────────────────────────────────────────

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

// ── Component ──────────────────────────────────────────────────────────────

export function Text2SignEditor() {
  const [input, setInput] = useState("I study");

  // Gloss state
  const [glossTokens, setGlossTokens]       = useState<string[]>([]);
  const [missingGlosses, setMissingGlosses] = useState<string[]>([]);
  const [method, setMethod]                 = useState<"gemini" | "fallback" | null>(null);
  const [glossLoading, setGlossLoading]     = useState(false);
  const [glossError, setGlossError]         = useState<string | null>(null);

  // Animation state
  const [frames, setFrames]           = useState<number[][][]>([]);
  const [synthFps, setSynthFps]       = useState(25);
  const [synthLoading, setSynthLoading] = useState(false);
  const [synthError, setSynthError]   = useState<string | null>(null);
  const [glossesUsed, setGlossesUsed] = useState<string[]>([]);

  // Playback state
  const [playing, setPlaying]         = useState(false);
  const [speed, setSpeed]             = useState<number>(1);
  const [loop, setLoop]               = useState(true);
  const [currentFrame, setCurrentFrame] = useState(0);

  // Export state
  const [gifLoading, setGifLoading]         = useState(false);
  const [framesLoading, setFramesLoading]   = useState(false);
  const [framesProgress, setFramesProgress] = useState(0);

  const canvasRef = useRef<SkeletonCanvasHandle>(null);

  const hasFrames = frames.length > 0;

  // ── Step 1: text → glosses ─────────────────────────────────────────────

  const handleTranslate = async () => {
    if (!input.trim()) return;
    setGlossLoading(true);
    setGlossError(null);
    setGlossTokens([]);
    setFrames([]);
    setPlaying(false);
    setCurrentFrame(0);

    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/translate/gloss`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input, use_gemini: true }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
      const data: GlossResult = await res.json();
      setGlossTokens(data.glosses ?? []);
      setMissingGlosses(data.missing_glosses ?? []);
      setMethod(data.method ?? null);
      if (data.glosses?.length > 0) await handleSynthesize(data.glosses);
    } catch (err: unknown) {
      setGlossError(err instanceof Error ? err.message : "Translation failed");
    } finally {
      setGlossLoading(false);
    }
  };

  // ── Step 2: glosses → skeleton frames ─────────────────────────────────

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
      if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
      const data: SynthesizeResult = await res.json();
      if (!data.success || !data.frames?.length) throw new Error(data.error || "No frames returned");

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

  const removeToken = (i: number) => {
    setGlossTokens(prev => prev.filter((_, idx) => idx !== i));
    setFrames([]); setPlaying(false);
  };

  // ── Playback controls ──────────────────────────────────────────────────

  const handleSeekStart = () => {
    setPlaying(false); setCurrentFrame(0); canvasRef.current?.seekToFrame(0);
  };
  const handleSeekEnd = () => {
    if (!hasFrames) return;
    const last = frames.length - 1;
    setPlaying(false); setCurrentFrame(last); canvasRef.current?.seekToFrame(last);
  };
  const handleSlider = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = Number(e.target.value);
    setCurrentFrame(f); setPlaying(false); canvasRef.current?.seekToFrame(f);
  };
  const handleFrameChange = useCallback((f: number) => setCurrentFrame(f), []);
  const handlePlayEnd = useCallback(() => { if (!loop) setPlaying(false); }, [loop]);

  // ── Export: GIF (server-side PIL) ─────────────────────────────────────

  const handleDownloadGif = async () => {
    if (!hasFrames || !glossesUsed.length) return;
    setGifLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/translate/export-gif`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ glosses: glossesUsed, fps: 15 }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "GIF export failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = glossesUsed.join("_").slice(0, 40) + ".gif"; a.click();
      URL.revokeObjectURL(url);
    } catch (err: unknown) {
      setSynthError(err instanceof Error ? err.message : "GIF export failed");
    } finally {
      setGifLoading(false);
    }
  };

  // ── Export: Frames ZIP (client-side JSZip) ─────────────────────────────

  const handleDownloadFrames = async () => {
    if (!hasFrames) return;
    setFramesLoading(true); setFramesProgress(0);
    try {
      const zip = new JSZip();
      const folder = zip.folder("frames")!;
      const blobs = await canvasRef.current?.captureAllFramesPNG((i, t) =>
        setFramesProgress(Math.round((i / t) * 100))
      ) ?? [];
      blobs.forEach((blob, i) => folder.file(`frame_${String(i).padStart(4, "0")}.png`, blob));
      folder.file("manifest.json", JSON.stringify({ total_frames: frames.length, fps: synthFps, glosses: glossesUsed }, null, 2));
      const zipBlob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(zipBlob);
      const a = document.createElement("a");
      a.href = url; a.download = `asl_${glossesUsed.join("_").slice(0, 30)}_frames.zip`; a.click();
      URL.revokeObjectURL(url);
    } catch (err: unknown) {
      setSynthError(err instanceof Error ? err.message : "Frames export failed");
    } finally {
      setFramesLoading(false); setFramesProgress(0);
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4" style={{ height: "calc(100vh - 11.5rem)" }}>

      {/* LEFT: Input + Gloss */}
      <div className="flex flex-col min-h-0 gap-4">

        {/* Text Input */}
        <div className="asl-panel">
          <div className="asl-panel-header">
            <h2 className="text-sm font-semibold">English Input</h2>
          </div>
          <div className="asl-panel-body space-y-3">
            <Textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type an English sentence…"
              className="min-h-[80px] max-h-[80px] resize-none text-base"
            />
            <Button className="w-full" onClick={handleTranslate} disabled={glossLoading || synthLoading}>
              {glossLoading
                ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Translating…</>
                : <><ArrowRight className="w-4 h-4 mr-2" />Translate to ASL Gloss</>}
            </Button>
            {glossError && (
              <p className="flex items-center gap-1.5 text-xs text-destructive">
                <AlertCircle className="w-3.5 h-3.5" />{glossError}
              </p>
            )}
          </div>
        </div>

        {/* Gloss tokens */}
        <div className="asl-panel flex-1 flex flex-col min-h-0">
          <div className="asl-panel-header">
            <h2 className="text-sm font-semibold">Extracted Gloss</h2>
            <div className="flex items-center gap-2">
              {method === "gemini" && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-500 text-xs font-medium">
                  <Sparkles className="w-3 h-3" />Gemini
                </span>
              )}
              {method === "fallback" && (
                <span className="px-2 py-0.5 rounded-full bg-yellow-500/10 text-yellow-600 text-xs font-medium">Fallback</span>
              )}
              {glossTokens.length > 0 && !synthLoading && (
                <button
                  onClick={() => handleSynthesize(glossTokens)}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-green-500/10 text-green-600 text-xs font-medium hover:bg-green-500/20 transition"
                >
                  <Zap className="w-3 h-3" />Animate
                </button>
              )}
            </div>
          </div>
          <div className="asl-panel-body overflow-y-auto flex-1">
            {synthLoading && (
              <p className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />Synthesizing skeleton…
              </p>
            )}
            {synthError && (
              <p className="flex items-center gap-1.5 text-xs text-yellow-600 mb-2">
                <AlertCircle className="w-3.5 h-3.5" />{synthError}
              </p>
            )}
            {glossTokens.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {glossTokens.map((tok, i) => (
                  <span key={i} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-primary/10 text-primary font-mono text-sm font-semibold">
                    {tok}
                    <button onClick={() => removeToken(i)} className="hover:text-destructive">
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                {glossLoading ? "Translating…" : 'Press "Translate to ASL Gloss" to begin.'}
              </p>
            )}
            {missingGlosses.length > 0 && (
              <div className="mt-3 pt-3 border-t border-border">
                <p className="text-xs text-muted-foreground mb-1.5">⚠️ Not in dictionary ({missingGlosses.length} skipped):</p>
                <div className="flex flex-wrap gap-1.5">
                  {missingGlosses.map((g, i) => (
                    <span key={i} className="px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-mono text-xs line-through">{g}</span>
                  ))}
                </div>
              </div>
            )}
            {hasFrames && (
              <div className="mt-3 pt-3 border-t border-border text-xs text-muted-foreground">
                <span className="text-green-600 font-medium">✓ Animated:</span>{" "}
                {glossesUsed.join(" → ")}
                <span className="ml-2">({frames.length} frames @ {synthFps}fps)</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* RIGHT: Animation Preview */}
      <div className="flex flex-col min-h-0">
        <div className="asl-panel flex-1 flex flex-col min-h-0">
          <div className="asl-panel-header">
            <h2 className="text-sm font-semibold">Sign Animation Preview</h2>
            {hasFrames && (
              <span className="text-xs text-muted-foreground">{currentFrame + 1} / {frames.length}</span>
            )}
          </div>

          {/* Canvas */}
          <div className="flex-1 bg-white rounded-lg overflow-hidden min-h-0 flex items-center justify-center">
            {synthLoading ? (
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                <Loader2 className="w-8 h-8 animate-spin" />
                <p className="text-sm">Synthesizing animation…</p>
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
                type="range" min={0} max={frames.length - 1} value={currentFrame}
                onChange={handleSlider}
                className="w-full h-1.5 accent-primary cursor-pointer"
              />
            </div>
          )}

          {/* Controls row 1 */}
          <div className="px-3 py-2 flex items-center gap-2 flex-wrap">
            {/* Seek start */}
            <Button variant="outline" size="icon" className="h-8 w-8" disabled={!hasFrames} onClick={handleSeekStart} title="Seek to start">
              <SkipBack className="w-3.5 h-3.5" />
            </Button>

            {/* Play / Pause */}
            <Button
              variant={playing ? "default" : "outline"}
              className="px-4 h-8"
              onClick={() => setPlaying(p => !p)}
              disabled={!hasFrames}
              id="btn-play-pause"
            >
              {playing ? <><Pause className="w-4 h-4 mr-1.5" />Pause</> : <><Play className="w-4 h-4 mr-1.5" />Play</>}
            </Button>

            {/* Seek end */}
            <Button variant="outline" size="icon" className="h-8 w-8" disabled={!hasFrames} onClick={handleSeekEnd} title="Seek to end">
              <SkipForward className="w-3.5 h-3.5" />
            </Button>

            {/* Speed */}
            <div className="flex items-center gap-1">
              {([0.5, 1, 1.5, 2] as const).map(s => (
                <Button
                  key={s}
                  variant={speed === s ? "default" : "outline"}
                  size="sm"
                  className="text-xs h-8 px-2.5"
                  onClick={() => setSpeed(s)}
                  id={`btn-speed-${s}`}
                >
                  {s}x
                </Button>
              ))}
            </div>

            {/* Loop */}
            <Button
              variant={loop ? "default" : "outline"}
              size="sm"
              className="text-xs h-8 px-3"
              onClick={() => setLoop(l => !l)}
              id="btn-loop"
            >
              {loop ? "Loop ✓" : "Loop"}
            </Button>
          </div>

          {/* Controls row 2: Export */}
          <div className="px-3 pb-3 flex items-center gap-2 border-t border-border pt-2">
            <span className="text-xs text-muted-foreground mr-1">Export:</span>

            <Button
              variant="outline" size="sm" className="gap-1.5 h-8"
              disabled={!hasFrames || gifLoading || framesLoading}
              onClick={handleDownloadGif}
              id="btn-download-gif"
              title="Download animated GIF (server-side)"
            >
              {gifLoading
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" />Generating…</>
                : <><Film className="w-3.5 h-3.5" />GIF</>}
            </Button>

            <Button
              variant="outline" size="sm" className="gap-1.5 h-8"
              disabled={!hasFrames || gifLoading || framesLoading}
              onClick={handleDownloadFrames}
              id="btn-download-frames"
              title="Download all frames as PNG ZIP"
            >
              {framesLoading
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" />{framesProgress}%</>
                : <><ImageDown className="w-3.5 h-3.5" />Frames</>}
            </Button>

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
