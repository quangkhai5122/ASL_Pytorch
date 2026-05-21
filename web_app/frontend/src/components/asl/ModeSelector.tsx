import { useASL } from "@/context/ASLContext";
import type { ASLMode } from "@/lib/mockData";
import { Scan, Hand, Type, BookOpen } from "lucide-react";

const modes: { key: ASLMode; label: string; icon: React.ReactNode; shortLabel: string }[] = [
  { key: "automatic", label: "Automatic", shortLabel: "Auto", icon: <Scan className="w-4 h-4" aria-hidden="true" /> },
  { key: "manual", label: "Manual", shortLabel: "Manual", icon: <Hand className="w-4 h-4" aria-hidden="true" /> },
  { key: "text2sign", label: "Text → Sign", shortLabel: "T2S", icon: <Type className="w-4 h-4" aria-hidden="true" /> },
  { key: "dictionary", label: "Dictionary", shortLabel: "Dict", icon: <BookOpen className="w-4 h-4" aria-hidden="true" /> },
];

export function ModeSelector() {
  const { mode, setMode } = useASL();

  return (
    <div className="asl-panel">
      <div className="asl-panel-header">
        <h2 className="text-sm font-semibold">Mode</h2>
      </div>
      <div className="p-2">
        <div
          className="grid grid-cols-4 gap-1 bg-muted rounded-lg p-1"
          role="tablist"
          aria-label="Translation mode selector"
        >
          {modes.map(m => (
            <button
              key={m.key}
              role="tab"
              aria-selected={mode === m.key}
              onClick={() => setMode(m.key)}
              className={`touch-target flex flex-col items-center justify-center gap-1 px-2 py-2.5 rounded-md text-xs font-medium transition-all ${
                mode === m.key
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:bg-background hover:text-foreground"
              }`}
            >
              {m.icon}
              <span className="hidden sm:inline">{m.label}</span>
              <span className="sm:hidden">{m.shortLabel}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
