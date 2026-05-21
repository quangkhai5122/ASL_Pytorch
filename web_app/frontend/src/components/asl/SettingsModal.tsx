import { useASL } from "@/context/ASLContext";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Settings, Monitor, Type, Keyboard, Shield, Download } from "lucide-react";
import type { FontSize, Theme } from "@/lib/mockData";

export function SettingsModal() {
  const { settingsOpen, setSettingsOpen, fontSize, setFontSize, theme, setTheme } = useASL();

  const fontSizes: { key: FontSize; label: string }[] = [
    { key: "a", label: "A (Default)" },
    { key: "a-plus", label: "A+" },
    { key: "a-plus-plus", label: "A++" },
  ];

  const themes: { key: Theme; label: string }[] = [
    { key: "light", label: "Default (Light)" },
    { key: "high-contrast", label: "High Contrast" },
  ];

  return (
    <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
      <DialogContent className="sm:max-w-lg" aria-describedby="settings-description">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" aria-hidden="true" />
            Settings
          </DialogTitle>
        </DialogHeader>
        <p id="settings-description" className="sr-only">Application settings for accessibility, theme, and preferences</p>
        <div className="space-y-6 py-2">
          {/* Theme */}
          <div>
            <h3 className="text-sm font-semibold flex items-center gap-2 mb-3">
              <Monitor className="w-4 h-4" aria-hidden="true" />Theme
            </h3>
            <div className="flex gap-2">
              {themes.map(t => (
                <Button
                  key={t.key}
                  variant={theme === t.key ? "default" : "outline"}
                  className="touch-target flex-1"
                  onClick={() => setTheme(t.key)}
                >
                  {t.label}
                </Button>
              ))}
            </div>
          </div>

          {/* Font Size */}
          <div>
            <h3 className="text-sm font-semibold flex items-center gap-2 mb-3">
              <Type className="w-4 h-4" aria-hidden="true" />Font Size
            </h3>
            <div className="flex gap-2">
              {fontSizes.map(f => (
                <Button
                  key={f.key}
                  variant={fontSize === f.key ? "default" : "outline"}
                  className="touch-target flex-1"
                  onClick={() => setFontSize(f.key)}
                >
                  {f.label}
                </Button>
              ))}
            </div>
          </div>

          {/* Keyboard shortcuts */}
          <div>
            <h3 className="text-sm font-semibold flex items-center gap-2 mb-3">
              <Keyboard className="w-4 h-4" aria-hidden="true" />Keyboard Shortcuts
            </h3>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="flex items-center justify-between p-2 rounded bg-muted/50">
                <span>Generate</span><span className="kbd">⌘G</span>
              </div>
              <div className="flex items-center justify-between p-2 rounded bg-muted/50">
                <span>Undo</span><span className="kbd">⌘Z</span>
              </div>
              <div className="flex items-center justify-between p-2 rounded bg-muted/50">
                <span>Clear</span><span className="kbd">⌘⇧C</span>
              </div>
              <div className="flex items-center justify-between p-2 rounded bg-muted/50">
                <span>Add 1–5</span><span className="kbd">1-5</span>
              </div>
            </div>
          </div>

          {/* Privacy */}
          <div>
            <h3 className="text-sm font-semibold flex items-center gap-2 mb-3">
              <Shield className="w-4 h-4" aria-hidden="true" />Privacy
            </h3>
            <label className="flex items-center gap-3 text-sm cursor-pointer">
              <input type="checkbox" defaultChecked className="rounded" />
              Do not record or store camera video
            </label>
            <p className="text-xs text-muted-foreground mt-1.5">Camera data is processed locally and never sent to a server in this prototype.</p>
          </div>

          {/* Export */}
          <div>
            <h3 className="text-sm font-semibold flex items-center gap-2 mb-3">
              <Download className="w-4 h-4" aria-hidden="true" />Export
            </h3>
            <Button variant="outline" className="touch-target" aria-label="Export session transcript">
              Export Transcript
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
