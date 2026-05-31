import { useState } from "react";
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
import { Button } from "@/components/ui/button";
import { Settings, HelpCircle, LogIn, Loader2 } from "lucide-react";

function LoginGate() {
  const { login, authLoading, authError, clearAuthError } = useASL();
  const [username, setUsername] = useState("testuser");
  const [password, setPassword] = useState("testpass123");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearAuthError();
    await login(username, password);
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-sm">
        <div className="asl-panel">
          <div className="p-6 space-y-6">
            {/* Logo */}
            <div className="text-center space-y-2">
              <div className="w-12 h-12 rounded-xl bg-primary mx-auto flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-lg">A</span>
              </div>
              <h1 className="text-xl font-bold">ASL-Bridge</h1>
              <p className="text-sm text-muted-foreground">Two-way ASL Translator</p>
            </div>

            {/* Login form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <label htmlFor="username" className="text-sm font-medium">Username</label>
                <input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm"
                  placeholder="Enter username"
                  required
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="password" className="text-sm font-medium">Password</label>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm"
                  placeholder="Enter password"
                  required
                />
              </div>

              {authError && (
                <p className="text-sm text-destructive bg-destructive/10 rounded-lg px-3 py-2">
                  {authError}
                </p>
              )}

              <Button type="submit" className="w-full touch-target" disabled={authLoading}>
                {authLoading ? (
                  <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Logging in...</>
                ) : (
                  <><LogIn className="w-4 h-4 mr-2" />Login</>
                )}
              </Button>
            </form>

            <p className="text-xs text-muted-foreground text-center">
              Default: testuser / testpass123
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Index() {
  const { mode, setSettingsOpen, setOnboardingOpen, isAuthenticated, authLoading, username, logout } = useASL();

  // Show loading spinner while checking auth
  if (authLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  // Show login gate if not authenticated
  if (!isAuthenticated) {
    return <LoginGate />;
  }

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
            {username && (
              <span className="text-xs text-muted-foreground mr-2">
                {username}
              </span>
            )}
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
            <Button
              variant="ghost"
              size="sm"
              className="text-xs"
              onClick={() => logout()}>
              Logout
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
        /* Automatic & Manual: Camera-dominant layout */
        <div className="grid grid-cols-1 lg:grid-cols-[60%_1fr] gap-3">
            {/* Left: Camera — stretches to fill full height */}
            <div className="flex flex-col">
              <CameraCard />
            </div>

            {/* Right: Compact control panels */}
            <div className="flex flex-col gap-3">
              {/* Status + Word Buffer side by side */}
              <div className="grid h-[250px] grid-cols-1 gap-3 md:grid-cols-[1fr_2fr]">
                <StatusPanel />
                <div className="min-h-0">
                  {mode === "manual" ?
                <PredictionsList /> :
                <>
                      <WordBuffer />
                    </>
                }
                </div>
              </div>

              {/* Generate Sentence controls */}
              <BufferControls />

              {/* Generated Sentence — fills remaining height */}
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
        /* Dictionary: full width */
        <div className="w-full">
            <DictionarySearch />
          </div>) :
        null}
      </main>

      {/* Modals */}
      <SettingsModal />
      <OnboardingModal />
    </div>);
}
