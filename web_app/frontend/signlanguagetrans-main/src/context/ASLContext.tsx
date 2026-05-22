import React, { createContext, useContext, useState, useCallback } from "react";
import type { ASLMode, FontSize, Theme, BufferToken, SignPrediction } from "@/lib/mockData";
import { MOCK_BUFFER } from "@/lib/mockData";

type ASLContextType = {
  mode: ASLMode;
  setMode: (m: ASLMode) => void;
  fontSize: FontSize;
  setFontSize: (f: FontSize) => void;
  theme: Theme;
  setTheme: (t: Theme) => void;
  buffer: BufferToken[];
  setBuffer: React.Dispatch<React.SetStateAction<BufferToken[]>>;
  addToBuffer: (token: BufferToken) => void;
  removeFromBuffer: (id: string) => void;
  clearBuffer: () => void;
  undoBuffer: () => void;
  cameraActive: boolean;
  setCameraActive: (v: boolean) => void;
  showSkeleton: boolean;
  setShowSkeleton: (v: boolean) => void;
  generatedSentence: string;
  setGeneratedSentence: (s: string) => void;
  status: string;
  setStatus: (s: string) => void;
  settingsOpen: boolean;
  setSettingsOpen: (v: boolean) => void;
  onboardingOpen: boolean;
  setOnboardingOpen: (v: boolean) => void;
};

const ASLContext = createContext<ASLContextType | null>(null);

export function ASLProvider({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<ASLMode>("automatic");
  const [fontSize, setFontSizeState] = useState<FontSize>("a");
  const [theme, setThemeState] = useState<Theme>("light");
  const [buffer, setBuffer] = useState<BufferToken[]>(MOCK_BUFFER);
  const [bufferHistory, setBufferHistory] = useState<BufferToken[][]>([]);
  const [cameraActive, setCameraActive] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [generatedSentence, setGeneratedSentence] = useState("");
  const [status, setStatus] = useState("No hands detected");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [onboardingOpen, setOnboardingOpen] = useState(true);

  const setFontSize = useCallback((f: FontSize) => {
    setFontSizeState(f);
    document.documentElement.className = document.documentElement.className
      .replace(/font-a(-plus(-plus)?)?/g, "")
      .trim();
    document.documentElement.classList.add(`font-${f}`);
  }, []);

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t);
    document.documentElement.classList.remove("high-contrast");
    if (t === "high-contrast") {
      document.documentElement.classList.add("high-contrast");
    }
  }, []);

  const addToBuffer = useCallback((token: BufferToken) => {
    setBuffer(prev => {
      setBufferHistory(h => [...h, prev]);
      return [...prev, token];
    });
  }, []);

  const removeFromBuffer = useCallback((id: string) => {
    setBuffer(prev => {
      setBufferHistory(h => [...h, prev]);
      return prev.filter(t => t.id !== id);
    });
  }, []);

  const clearBuffer = useCallback(() => {
    setBuffer(prev => {
      setBufferHistory(h => [...h, prev]);
      return [];
    });
  }, []);

  const undoBuffer = useCallback(() => {
    setBufferHistory(h => {
      if (h.length === 0) return h;
      const prev = h[h.length - 1];
      setBuffer(prev);
      return h.slice(0, -1);
    });
  }, []);

  return (
    <ASLContext.Provider value={{
      mode, setMode, fontSize, setFontSize, theme, setTheme,
      buffer, setBuffer, addToBuffer, removeFromBuffer, clearBuffer, undoBuffer,
      cameraActive, setCameraActive, showSkeleton, setShowSkeleton,
      generatedSentence, setGeneratedSentence, status, setStatus,
      settingsOpen, setSettingsOpen, onboardingOpen, setOnboardingOpen,
    }}>
      {children}
    </ASLContext.Provider>
  );
}

export function useASL() {
  const ctx = useContext(ASLContext);
  if (!ctx) throw new Error("useASL must be used within ASLProvider");
  return ctx;
}
