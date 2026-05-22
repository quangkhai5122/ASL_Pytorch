import { useASL } from "@/context/ASLContext";
import { MOCK_PREDICTIONS, getConfidenceLevel } from "@/lib/mockData";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

export function PredictionsList() {
  const { addToBuffer, setMode } = useASL();

  const handleAdd = (pred: typeof MOCK_PREDICTIONS[0], index: number) => {
    addToBuffer({
      id: `pred-${Date.now()}-${index}`,
      gloss: pred.gloss,
      confidence: pred.confidence,
      timestamp: Date.now()
    });
    setMode("automatic");
  };

  return (
    <div className="asl-panel">
      <div className="asl-panel-header">
        <h2 className="text-sm font-semibold">Top 5 Predictions</h2>
        <span className="text-xs text-muted-foreground">Manual Mode</span>
      </div>
      <div className="asl-panel-body space-y-2 max-h-64 overflow-y-auto" role="list" aria-label="Top 5 sign predictions">
        {MOCK_PREDICTIONS.map((pred, i) => {
          const level = getConfidenceLevel(pred.confidence);
          return (
            <div
              key={i}
              role="listitem"
              className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50 transition-colors group">
              
              <span className="kbd">{i + 1}</span>
              

              
              <div className="flex-1 min-w-0">
                <p className="font-mono text-sm font-semibold">{pred.gloss}</p>
                <div className="flex items-center gap-2 mt-0.5">
                  <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                      level === "high" ? "bg-success" : level === "medium" ? "bg-warning" : "bg-destructive"}`
                      }
                      style={{ width: `${pred.confidence * 100}%` }} />
                    
                  </div>
                  <span className={`confidence-chip confidence-${level}`}>
                    {(pred.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <Button
                size="icon"
                variant="outline"
                className="h-8 w-8 touch-target opacity-60 group-hover:opacity-100 transition-opacity"
                onClick={() => handleAdd(pred, i)}
                aria-label={`Add ${pred.gloss} to buffer`}>
                
                <Plus className="w-4 h-4" aria-hidden="true" />
              </Button>
            </div>);

        })}
      </div>
      

      
    </div>);

}