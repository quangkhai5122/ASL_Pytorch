import { useASL } from "@/context/ASLContext";
import { Button } from "@/components/ui/button";
import { X, GripVertical, Pencil, Check } from "lucide-react";
import { getConfidenceLevel, formatTimestamp } from "@/lib/mockData";
import { useState } from "react";

export function WordBuffer() {
  const { buffer, removeFromBuffer, setBuffer } = useASL();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");

  const startEdit = (id: string, gloss: string) => {
    setEditingId(id);
    setEditValue(gloss);
  };

  const confirmEdit = (id: string) => {
    setBuffer((prev) => prev.map((t) => t.id === id ? { ...t, gloss: editValue.toUpperCase() } : t));
    setEditingId(null);
  };

  return (
    <div className="asl-panel">
      <div className="asl-panel-header">
        <h2 className="text-sm font-semibold">Word Buffer</h2>
        <span className="text-xs text-muted-foreground">{buffer.length} tokens</span>
      </div>
      <div className="asl-panel-body space-y-1.5 max-h-64 overflow-y-auto" role="list" aria-label="Word buffer tokens">
        {buffer.length === 0 &&
        <p className="text-xs text-muted-foreground text-center py-4">No tokens in buffer</p>
        }
        {buffer.map((token) => {
          const level = getConfidenceLevel(token.confidence);
          return (
            <div
              key={token.id}
              role="listitem"
              className="flex items-center gap-2 p-2 rounded-lg bg-muted/50 hover:bg-muted transition-colors group animate-fade-in">
              
              <GripVertical className="w-4 h-4 text-muted-foreground/40 cursor-grab flex-shrink-0" aria-hidden="true" />
              {editingId === token.id ?
              <input
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && confirmEdit(token.id)}
                className="flex-1 bg-background border border-input rounded px-2 py-1 text-sm font-mono"
                autoFocus
                aria-label="Edit token" /> :


              <span className="flex-1 font-mono text-sm font-semibold">{token.gloss}</span>
              }
              <span className={`confidence-chip confidence-${level}`}>
                {(token.confidence * 100).toFixed(0)}%
              </span>
              

              
              <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                {editingId === token.id ?
                <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => confirmEdit(token.id)} aria-label="Confirm edit">
                    <Check className="w-3.5 h-3.5" />
                  </Button> :

                <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => startEdit(token.id, token.gloss)} aria-label="Edit token">
                    <Pencil className="w-3.5 h-3.5" />
                  </Button>
                }
                <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => removeFromBuffer(token.id)} aria-label="Remove token">
                  <X className="w-3.5 h-3.5" />
                </Button>
              </div>
            </div>);

        })}
      </div>
    </div>);

}