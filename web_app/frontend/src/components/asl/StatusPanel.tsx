import { useASL } from "@/context/ASLContext";
import { AlertCircle, CheckCircle2, WifiOff } from "lucide-react";
import { useEffect, useState } from "react";

const MIN_DISPLAY_CONFIDENCE = 0.25;
const PREDICTION_HOLD_MS = 2500;

export function StatusPanel() {
  const { cameraActive, wsConnected, isAuthenticated, wsLastPrediction, wsError } = useASL();
  const [heldPrediction, setHeldPrediction] = useState<{
    sign: string;
    confidence: number;
    expiresAt: number;
  } | null>(null);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const confidence = wsLastPrediction?.confidence ?? 0;
    if (wsLastPrediction?.sign && confidence >= MIN_DISPLAY_CONFIDENCE) {
      setHeldPrediction({
        sign: wsLastPrediction.sign.toUpperCase(),
        confidence,
        expiresAt: Date.now() + PREDICTION_HOLD_MS,
      });
    }
  }, [wsLastPrediction]);

  useEffect(() => {
    if (!cameraActive || !wsConnected) {
      setHeldPrediction(null);
    }
  }, [cameraActive, wsConnected]);

  useEffect(() => {
    const interval = setInterval(() => {
      const timestamp = Date.now();
      setNow(timestamp);
      setHeldPrediction((current) =>
        current && current.expiresAt <= timestamp ? null : current
      );
    }, 250);

    return () => clearInterval(interval);
  }, []);

  const getStatusInfo = () => {
    if (!isAuthenticated) {
      return {
        icon: <AlertCircle className="w-8 h-8 text-muted-foreground" aria-hidden="true" />,
        text: "Login required",
        detail: "Sign in to begin",
      };
    }
    if (!cameraActive) {
      return {
        icon: <AlertCircle className="w-8 h-8 text-muted-foreground" aria-hidden="true" />,
        text: "Camera off",
        detail: "Start camera when ready",
      };
    }
    if (!wsConnected) {
      return {
        icon: <WifiOff className="w-8 h-8 text-warning" aria-hidden="true" />,
        text: "Connecting",
        detail: wsError || "Waiting for inference server",
      };
    }

    if (wsLastPrediction?.type === "error") {
      return {
        icon: <AlertCircle className="w-8 h-8 text-destructive" aria-hidden="true" />,
        text: "Stream error",
        detail: wsLastPrediction.error || "Check backend logs",
      };
    }

    if (heldPrediction && heldPrediction.expiresAt > now) {
      return {
        icon: <CheckCircle2 className="w-8 h-8 text-success" aria-hidden="true" />,
        text: heldPrediction.sign,
        detail: `${(heldPrediction.confidence * 100).toFixed(0)}% confidence`,
      };
    }

    return {
      icon: <AlertCircle className="w-8 h-8 text-muted-foreground" aria-hidden="true" />,
      text: "No Hands Detected",
      detail: "Place your hands in the camera frame",
    };
  };

  const { icon, text, detail } = getStatusInfo();

  return (
    <div className="asl-panel h-full">
      <div className="asl-panel-header">
        <h2 className="text-sm font-semibold">Status</h2>
      </div>
      <div className="asl-panel-body flex h-[calc(100%-49px)] min-h-0 flex-col items-center justify-center text-center gap-3 px-4">
        {icon}
        <p className="text-base font-semibold leading-tight break-words w-full" role="status" aria-live="polite">
          {text}
        </p>
        <p className="text-xs leading-snug text-muted-foreground">{detail}</p>
      </div>
    </div>
  );
}
