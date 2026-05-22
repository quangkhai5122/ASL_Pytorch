import { Camera, CameraOff, Eye, EyeOff } from "lucide-react";
import { useASL } from "@/context/ASLContext";
import { Button } from "@/components/ui/button";
import { FPSBadge } from "./FPSBadge";
import { MotionMeter } from "./MotionMeter";
import { SkeletonPreview } from "./SkeletonPreview";

export function CameraCard() {
  const { cameraActive, setCameraActive, showSkeleton, setShowSkeleton } = useASL();

  return (
    <div className="asl-panel flex flex-col">
      <div className="asl-panel-header">
        <h2 className="text-sm font-semibold">Camera Feed</h2>
        <div className="flex items-center gap-2">
          <FPSBadge />
          <MotionMeter />
        </div>
      </div>
      <div className="relative aspect-video bg-foreground/5 rounded-b-xl overflow-hidden flex items-center justify-center">
        {cameraActive ?
        <>
            {/* Mock camera feed */}
            <div className="absolute inset-0 bg-gradient-to-br from-foreground/10 to-foreground/5 flex items-center justify-center">
              <div className="text-center space-y-2">
                <div className="w-24 h-24 mx-auto rounded-full border-2 border-dashed border-primary/40 flex items-center justify-center">
                  <Camera className="w-8 h-8 text-primary/60" aria-hidden="true" />
                </div>
                <p className="text-xs text-muted-foreground">Camera feed active (mock)</p>
              </div>
            </div>
            {showSkeleton &&
          <div className="absolute inset-0 pointer-events-none">
                <SkeletonPreview animated />
              </div>
          }
          </> :

        <div className="text-center space-y-3 p-8">
            <CameraOff className="w-12 h-12 mx-auto text-muted-foreground/50" aria-hidden="true" />
            <p className="text-sm text-muted-foreground">Camera is off</p>
            <p className="text-xs text-muted-foreground/70">
              Tip: place your hands in the center of the frame and ensure even lighting.
            </p>
          </div>
        }
      </div>
      <div className="p-3 flex items-center gap-2 border-t border-border">
        <Button
          onClick={() => setCameraActive(!cameraActive)}
          variant={cameraActive ? "destructive" : "default"}
          className="touch-target flex-1"
          aria-label={cameraActive ? "Stop camera" : "Start camera"}>
          
          {cameraActive ?
          <><CameraOff className="w-4 h-4 mr-2" aria-hidden="true" />Stop Camera</> :

          <><Camera className="w-4 h-4 mr-2" aria-hidden="true" />Start Camera</>
          }
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="touch-target"
          onClick={() => setShowSkeleton(!showSkeleton)}
          aria-label={showSkeleton ? "Hide skeleton overlay" : "Show skeleton overlay"}>
          
          {showSkeleton ? <Eye className="w-4 h-4" aria-hidden="true" /> : <EyeOff className="w-4 h-4" aria-hidden="true" />}
        </Button>
      </div>
    </div>);

}