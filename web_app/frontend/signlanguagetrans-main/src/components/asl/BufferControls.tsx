import { useASL } from "@/context/ASLContext";
import { Button } from "@/components/ui/button";
import { Sparkles, Trash2, Undo2, Download } from "lucide-react";
import { MOCK_GENERATED_SENTENCE } from "@/lib/mockData";

export function BufferControls() {
  const { buffer, clearBuffer, undoBuffer, setGeneratedSentence } = useASL();

  const handleGenerate = () => {
    if (buffer.length === 0) return;
    setGeneratedSentence(MOCK_GENERATED_SENTENCE);
  };

  return (
    <div className="asl-panel">
      <div className="asl-panel-body space-y-2">
        <Button
          className="w-full touch-target"
          onClick={handleGenerate}
          disabled={buffer.length === 0}
          aria-label="Generate sentence from buffer"
        >
          <Sparkles className="w-4 h-4 mr-2" aria-hidden="true" />
          Generate Sentence
          <span className="kbd ml-2">⌘G</span>
        </Button>
        <div className="grid grid-cols-3 gap-2">
          <Button variant="outline" className="touch-target" onClick={clearBuffer} aria-label="Clear session">
            <Trash2 className="w-4 h-4 mr-1" aria-hidden="true" />
            <span className="hidden sm:inline">Clear</span>
          </Button>
          <Button variant="outline" className="touch-target" onClick={undoBuffer} aria-label="Undo last action">
            <Undo2 className="w-4 h-4 mr-1" aria-hidden="true" />
            <span className="hidden sm:inline">Undo</span>
          </Button>
          <Button variant="outline" className="touch-target" aria-label="Save transcript">
            <Download className="w-4 h-4 mr-1" aria-hidden="true" />
            <span className="hidden sm:inline">Save</span>
          </Button>
        </div>
      </div>
    </div>
  );
}
