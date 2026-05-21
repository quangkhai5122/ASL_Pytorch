import { useASL } from "@/context/ASLContext";
import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";

export function StatusPanel() {
  const { status, cameraActive } = useASL();

  const getStatusIcon = () => {
    if (!cameraActive) return <AlertCircle className="w-6 h-6 text-muted-foreground" aria-hidden="true" />;
    if (status === "Confirming…") return <Loader2 className="w-6 h-6 text-warning animate-spin" aria-hidden="true" />;
    if (status.includes("detected")) return <AlertCircle className="w-6 h-6 text-muted-foreground" aria-hidden="true" />;
    if (status.includes("Low")) return <AlertCircle className="w-6 h-6 text-warning" aria-hidden="true" />;
    return <CheckCircle2 className="w-6 h-6 text-success" aria-hidden="true" />;
  };

  return (
    <div className="asl-panel">
      <div className="asl-panel-header">
        <h2 className="text-sm font-semibold">Current Status</h2>
      </div>
      <div className="asl-panel-body flex flex-col items-center text-center gap-3 py-6">
        {getStatusIcon()}
        <p className="text-lg font-semibold" role="status" aria-live="polite">
          {cameraActive ? status : "Camera off"}
        </p>
        {status.includes("Low") && (
          <p className="text-xs text-muted-foreground">Switch to Manual mode for better control</p>
        )}
      </div>
    </div>
  );
}
