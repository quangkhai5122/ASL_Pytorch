import { useState, useCallback } from "react";
import { useASL } from "@/context/ASLContext";
import { Button } from "@/components/ui/button";
import { Volume2, Loader2, VolumeX } from "lucide-react";

export function GeneratedSentence() {
  const { generatedSentence } = useASL();
  const [isSpeaking, setIsSpeaking] = useState(false);

  const handlePlayAudio = useCallback(() => {
    if (!generatedSentence) return;

    // If already speaking, stop it
    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    // Check browser support
    if (!("speechSynthesis" in window)) {
      alert("Your browser does not support Text-to-Speech.");
      return;
    }

    // Cancel any ongoing speech first
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(generatedSentence);
    utterance.lang = "en-US";
    utterance.rate = 0.9;
    utterance.pitch = 1;

    // Try to pick a good English voice
    const voices = window.speechSynthesis.getVoices();
    const englishVoice = voices.find(
      (v) => v.lang.startsWith("en") && v.name.includes("Google")
    ) || voices.find((v) => v.lang.startsWith("en-US")) || voices.find((v) => v.lang.startsWith("en"));
    if (englishVoice) {
      utterance.voice = englishVoice;
    }

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  }, [generatedSentence, isSpeaking]);

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
          <Button
            variant="outline"
            size="sm"
            className="touch-target"
            disabled={!generatedSentence}
            onClick={handlePlayAudio}
            aria-label={isSpeaking ? "Stop audio" : "Play audio of generated sentence"}
          >
            {isSpeaking ? (
              <>
                <VolumeX className="w-4 h-4 mr-1.5" aria-hidden="true" />
                Stop
              </>
            ) : (
              <>
                <Volume2 className="w-4 h-4 mr-1.5" aria-hidden="true" />
                Play Audio
              </>
            )}
          </Button>
          {isSpeaking && (
            <span className="text-xs text-primary animate-pulse">Speaking...</span>
          )}
        </div>
      </div>
    </div>);

}