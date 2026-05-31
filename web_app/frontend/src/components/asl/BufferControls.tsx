import { useState } from "react";
import { useASL } from "@/context/ASLContext";
import { Button } from "@/components/ui/button";
import { apiClient } from "@/services/api";
import { Sparkles, Trash2, Loader2 } from "lucide-react";

export function BufferControls() {
  const { buffer, clearBuffer, setGeneratedSentence } = useASL();
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (buffer.length === 0) return;
    setIsGenerating(true);
    setError(null);
    try {
      const response = await apiClient.generateSentence(buffer.map((token) => token.gloss));
      setGeneratedSentence(response.sentence);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sentence generation failed");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="asl-panel">
      <div className="asl-panel-body space-y-2">
        <div className="grid grid-cols-2 gap-2">
          <Button
            className="touch-target"
            onClick={handleGenerate}
            disabled={buffer.length === 0 || isGenerating}
            aria-label="Generate sentence from buffer"
          >
            {isGenerating ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" aria-hidden="true" />
            ) : (
              <Sparkles className="w-4 h-4 mr-2" aria-hidden="true" />
            )}
            {isGenerating ? "Generating..." : "Generate Sentence"}
          </Button>
          <Button
            variant="outline"
            className="touch-target"
            onClick={clearBuffer}
            disabled={buffer.length === 0}
            aria-label="Clear session"
          >
            <Trash2 className="w-4 h-4 mr-2" aria-hidden="true" />
            Clear
          </Button>
        </div>
        {error && (
          <p className="rounded-md bg-destructive/10 px-2 py-1.5 text-xs text-destructive">
            {error}
          </p>
        )}
      </div>
    </div>
  );
}
