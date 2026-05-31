import { useState, useRef } from "react";
import { useASL } from "@/context/ASLContext";
import { Button } from "@/components/ui/button";
import { apiClient } from "@/services/api";
import { Sparkles, Trash2, Loader2, Plus } from "lucide-react";

export function BufferControls() {
  const { buffer, clearBuffer, setGeneratedSentence, addToBuffer } = useASL();
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [manualWord, setManualWord] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Gọi API generate sentence từ Word Buffer
  const handleGenerate = async () => {
    if (buffer.length === 0) return;
    setIsGenerating(true);
    setError(null);
    try {
      const signs = buffer.map((token) => token.gloss);
      const response = await apiClient.generateSentence(signs);
      setGeneratedSentence(response.sentence);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sentence generation failed");
    } finally {
      setIsGenerating(false);
    }
  };

  // Thêm từ thủ công vào buffer
  const handleAddWord = () => {
    const word = manualWord.trim().toUpperCase();
    if (!word) return;
    addToBuffer({
      id: `manual-${Date.now()}`,
      gloss: word,
      confidence: 1.0,
      timestamp: new Date().toISOString(),
    });
    setManualWord("");
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleAddWord();
  };

  return (
    <div className="asl-panel">
      <div className="asl-panel-body space-y-2">
        {/* Manual word input */}
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={manualWord}
            onChange={(e) => setManualWord(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Thêm từ vào buffer..."
            className="flex-1 rounded-md border border-input bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            aria-label="Add word to buffer manually"
          />
          <Button
            variant="outline"
            size="sm"
            onClick={handleAddWord}
            disabled={!manualWord.trim()}
            aria-label="Add word"
          >
            <Plus className="w-4 h-4" />
          </Button>
        </div>

        {/* Generate + Clear buttons */}
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
