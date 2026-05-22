import { useState } from "react";
import { MOCK_TEXT2SIGN_INPUT, MOCK_TEXT2SIGN_GLOSS } from "@/lib/mockData";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ArrowRight, Play, Pause, Download, X } from "lucide-react";
import { SkeletonPreview } from "./SkeletonPreview";

export function Text2SignEditor() {
  const [input, setInput] = useState(MOCK_TEXT2SIGN_INPUT);
  const [glossTokens, setGlossTokens] = useState<string[]>([]);
  const [translated, setTranslated] = useState(false);
  const hasTokens = glossTokens.length > 0;
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(1);

  const handleTranslate = () => {
    setGlossTokens(MOCK_TEXT2SIGN_GLOSS);
    setTranslated(true);
  };

  const removeToken = (index: number) => {
    setGlossTokens(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4" style={{ height: 'calc(100vh - 11.5rem)' }}>
      {/* Left column: Input + Gloss */}
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
            <Button className="w-full touch-target" onClick={handleTranslate} aria-label="Translate text to ASL gloss">
              <ArrowRight className="w-4 h-4 mr-2" aria-hidden="true" />
              Translate to ASL Gloss
            </Button>
          </div>
        </div>

        {/* Gloss Output - always visible */}
        <div className="asl-panel flex-1 flex flex-col min-h-0 mt-4">
          <div className="asl-panel-header">
            <h2 className="text-sm font-semibold">Extracted Gloss</h2>
            {hasTokens && <span className="text-xs text-muted-foreground">{glossTokens.length} tokens</span>}
          </div>
          <div className="asl-panel-body overflow-y-auto flex-1">
            {hasTokens ? (
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
              <p className="text-sm text-muted-foreground">Press "Translate to ASL Gloss" to extract tokens.</p>
            )}
          </div>
        </div>
      </div>

      {/* Right column: Animation Preview */}
      <div className="flex flex-col min-h-0">
        <div className="asl-panel flex-1 flex flex-col min-h-0">
          <div className="asl-panel-header">
            <h2 className="text-sm font-semibold">Sign Animation Preview</h2>
          </div>
          <div className="flex-1 bg-foreground/5 rounded-lg overflow-hidden min-h-0">
            <SkeletonPreview animated={playing} />
          </div>
          <div className="p-3 flex items-center gap-2 flex-wrap">
            <Button
              variant="outline"
              className="touch-target"
              onClick={() => setPlaying(!playing)}
              aria-label={playing ? "Pause animation" : "Play animation"}
            >
              {playing ? <Pause className="w-4 h-4 mr-1.5" /> : <Play className="w-4 h-4 mr-1.5" />}
              {playing ? "Pause" : "Play"}
            </Button>
            <div className="flex items-center gap-1" role="group" aria-label="Playback speed">
              {[0.5, 1, 1.5].map(s => (
                <Button
                  key={s}
                  variant={speed === s ? "default" : "outline"}
                  size="sm"
                  className="touch-target text-xs"
                  onClick={() => setSpeed(s)}
                  aria-label={`Speed ${s}x`}
                >
                  {s}x
                </Button>
              ))}
            </div>
            <div className="ml-auto flex gap-1">
              <Button variant="outline" size="sm" className="touch-target" aria-label="Download GIF">
                <Download className="w-3.5 h-3.5 mr-1" />GIF
              </Button>
              <Button variant="outline" size="sm" className="touch-target" aria-label="Export frames">
                <Download className="w-3.5 h-3.5 mr-1" />Frames
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
