import { useASL } from "@/context/ASLContext";

export function ConfirmingBar() {
  const { cameraActive, wsLastPrediction } = useASL();
  
  const isConfirming =
    cameraActive &&
    wsLastPrediction?.type === "prediction" &&
    !wsLastPrediction.commit &&
    (wsLastPrediction.confidence ?? 0) > 0;

  if (!isConfirming) return null;

  return (
    <div className="asl-panel" role="status" aria-label="Confirming sign detection">
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-muted-foreground">Confirming sign...</span>
          <span className="text-xs font-mono text-muted-foreground animate-pulse-soft">Stabilizing</span>
        </div>
        <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
          <div className="h-full rounded-full bg-primary animate-confirming" />
        </div>
      </div>
    </div>
  );
}
