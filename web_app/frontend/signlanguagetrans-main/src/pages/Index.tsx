import { useASL } from "@/context/ASLContext";
import { CameraCard } from "@/components/asl/CameraCard";
import { ModeSelector } from "@/components/asl/ModeSelector";
import { WordBuffer } from "@/components/asl/WordBuffer";
import { BufferControls } from "@/components/asl/BufferControls";
import { PredictionsList } from "@/components/asl/PredictionsList";
import { StatusPanel } from "@/components/asl/StatusPanel";
import { GeneratedSentence } from "@/components/asl/GeneratedSentence";
import { Text2SignEditor } from "@/components/asl/Text2SignEditor";
import { DictionarySearch } from "@/components/asl/DictionarySearch";
import { SettingsModal } from "@/components/asl/SettingsModal";
import { OnboardingModal } from "@/components/asl/OnboardingModal";
import { ConfirmingBar } from "@/components/asl/ConfirmingBar";
import { Button } from "@/components/ui/button";
import { Settings, HelpCircle } from "lucide-react";

export default function Index() {
  const { mode, setSettingsOpen, setOnboardingOpen } = useASL();

  const showCameraColumn = mode === "automatic" || mode === "manual";
  const showMiddleColumn = mode === "automatic" || mode === "manual";

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card sticky top-0 z-40">
        <div className="max-w-[1600px] mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center" aria-hidden="true">
              <span className="text-primary-foreground font-bold text-sm">A</span>
            </div>
            <div>
              <h1 className="text-base font-bold leading-tight">ASL-Bridge</h1>
              <p className="text-[10px] text-muted-foreground leading-tight">Two-way ASL Translator</p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="touch-target"
              onClick={() => setOnboardingOpen(true)}
              aria-label="Show setup guide">
              
              <HelpCircle className="w-5 h-5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="touch-target"
              onClick={() => setSettingsOpen(true)}
              aria-label="Open settings">
              
              <Settings className="w-5 h-5" />
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-[1600px] mx-auto p-4">
        {/* Mode selector - always visible */}
        <div className="mb-4">
          <ModeSelector />
        </div>

        {/* Mode-specific layouts */}
        {showCameraColumn && showMiddleColumn ? (
        /* Automatic & Manual: 2-area layout - Camera left, controls right */
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Left: Camera */}
            <div>
              <CameraCard />
            </div>

            {/* Right: Everything else, with GeneratedSentence pushed to bottom */}
            <div className="flex flex-col gap-4 h-full">
              <div className="grid grid-cols-2 gap-4">
                {/* Left: Status & Controls */}
                <div className="space-y-4">
                  <StatusPanel />
                  <BufferControls />
                </div>

                {/* Right: Predictions (manual) or Buffer (automatic) */}
                <div className="space-y-4">
                  {mode === "manual" ?
                <PredictionsList /> :

                <>
                      <ConfirmingBar />
                      <WordBuffer />
                    </>
                }
                </div>
              </div>

              {/* Generated Sentence spanning full width, stretching to match camera bottom */}
              <div className="flex-1">
                <GeneratedSentence />
              </div>
            </div>
          </div>) :
        mode === "text2sign" ? (
        /* Text2Sign: 2-column */
        <div className="mb-0">
            <Text2SignEditor />
          </div>) :
        mode === "dictionary" ? (
        /* Dictionary: centered */
        <div className="max-w-2xl mx-auto">
            <DictionarySearch />
          </div>) :
        null}
      </main>

      {/* Modals */}
      <SettingsModal />
      <OnboardingModal />
    </div>);

}