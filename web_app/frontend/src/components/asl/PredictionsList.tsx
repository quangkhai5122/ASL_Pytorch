import { useEffect, useRef, useState } from "react";
import { useASL } from "@/context/ASLContext";
import { getConfidenceLevel } from "@/lib/mockData";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

type PredictionItem = { sign: string; confidence: number };

const TOP5_UPDATE_INTERVAL_MS = 700;

export function PredictionsList() {
  const { addToBuffer, setMode, wsLastPrediction } = useASL();
  const [stablePredictions, setStablePredictions] = useState<PredictionItem[]>([]);
  const lastUpdateRef = useRef(0);

  useEffect(() => {
    if (wsLastPrediction?.type !== "prediction" || !wsLastPrediction.top5?.length) {
      return;
    }

    const now = Date.now();
    if (now - lastUpdateRef.current < TOP5_UPDATE_INTERVAL_MS) {
      return;
    }

    lastUpdateRef.current = now;
    setStablePredictions(wsLastPrediction.top5.slice(0, 5));
  }, [wsLastPrediction]);

  const handleAdd = (pred: PredictionItem, index: number) => {
    addToBuffer({
      id: `pred-${Date.now()}-${index}`,
      gloss: pred.sign.toUpperCase(),
      confidence: pred.confidence,
      timestamp: Date.now(),
    });
    setMode("automatic");
  };

  return (
    <div className="asl-panel h-full">
      <div className="asl-panel-header">
        <h2 className="text-sm font-semibold">Top 5 Predictions</h2>
        <span className="text-xs text-muted-foreground">Manual Mode</span>
      </div>
      <div
        className="asl-panel-body h-[calc(100%-49px)] min-h-0 space-y-2 overflow-y-auto"
        role="list"
        aria-label="Top 5 sign predictions"
      >
        {stablePredictions.length === 0 ? (
          <div className="flex h-full min-h-[130px] items-center justify-center">
            <p className="text-xs text-muted-foreground">Waiting for predictions...</p>
          </div>
        ) : (
          stablePredictions.map((pred, i) => {
            const level = getConfidenceLevel(pred.confidence);
            return (
              <div
                key={`${pred.sign}-${i}`}
                role="listitem"
                className="flex items-center gap-3 rounded-lg p-2 transition-colors hover:bg-muted/50 group"
              >
                <span className="kbd">{i + 1}</span>
                <div className="flex-1 min-w-0">
                  <p className="font-mono text-sm font-semibold">{pred.sign.toUpperCase()}</p>
                  <div className="flex items-center gap-2 mt-0.5">
                    <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${
                          level === "high" ? "bg-success" : level === "medium" ? "bg-warning" : "bg-destructive"
                        }`}
                        style={{ width: `${pred.confidence * 100}%` }}
                      />
                    </div>
                    <span className={`confidence-chip confidence-${level}`}>
                      {(pred.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                <Button
                  size="icon"
                  variant="outline"
                  className="h-8 w-8 touch-target opacity-60 transition-opacity group-hover:opacity-100"
                  onClick={() => handleAdd(pred, i)}
                  aria-label={`Add ${pred.sign} to buffer`}
                >
                  <Plus className="w-4 h-4" aria-hidden="true" />
                </Button>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
