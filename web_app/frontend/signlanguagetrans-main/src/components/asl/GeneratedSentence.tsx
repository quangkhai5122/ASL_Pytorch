import { useASL } from "@/context/ASLContext";
import { Button } from "@/components/ui/button";
import { Volume2 } from "lucide-react";

export function GeneratedSentence() {
  const { generatedSentence } = useASL();

  return (
    <div className="asl-panel mt-0 h-full flex flex-col">
      <div className="asl-panel-header">
        <h2 className="text-sm font-semibold">Generated Sentence</h2>
      </div>
      <div className="asl-panel-body flex-1 flex flex-col">
        <div className="flex-1">
          {generatedSentence ? (
            <p className="text-lg font-medium leading-relaxed">{generatedSentence}</p>
          ) : (
            <p className="text-sm text-muted-foreground italic">Press "Generate Sentence" to create a sentence from the buffer.</p>
          )}
        </div>
        <div className="flex items-center gap-2 pt-3 mt-auto">
          <Button variant="outline" size="sm" className="touch-target" disabled={!generatedSentence} aria-label="Play audio of generated sentence">
            <Volume2 className="w-4 h-4 mr-1.5" aria-hidden="true" />
            Play Audio
          </Button>
          <span className="text-xs text-muted-foreground">(Mock TTS)</span>
        </div>
      </div>
    </div>);

}