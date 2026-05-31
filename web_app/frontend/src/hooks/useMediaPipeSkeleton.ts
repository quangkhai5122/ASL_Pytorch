/**
 * useMediaPipeSkeleton.ts
 *
 * Hook chạy MediaPipe Holistic trên video stream và trả về landmarks.
 * Trả về ref để tránh re-render mỗi frame.
 *
 * Return shape: { gislr: (number[] | null)[] }  — 543 points (GISLR order):
 *   [0..467]   Face (468 pts)
 *   [468..488] Left hand (21 pts)
 *   [489..521] Pose (33 pts)
 *   [522..542] Right hand (21 pts)
 *
 * TODO: tranhuy — fill in MediaPipe Holistic initialization & result processing.
 */

import { useRef, useEffect, RefObject } from "react";

interface SkeletonResult {
  gislr: (number[] | null)[];
}

export function useMediaPipeSkeleton(
  videoRef: RefObject<HTMLVideoElement>,
  enabled: boolean,
): React.RefObject<SkeletonResult | null> {
  const resultRef = useRef<SkeletonResult | null>(null);

  useEffect(() => {
    if (!enabled) {
      resultRef.current = null;
      return;
    }

    // TODO: Initialize MediaPipe Holistic and process videoRef frames.
    // Example skeleton:
    //
    // import { Holistic } from "@mediapipe/holistic";
    // const holistic = new Holistic({ locateFile: ... });
    // holistic.onResults((results) => {
    //   const gislr: (number[] | null)[] = new Array(543).fill(null);
    //   results.faceLandmarks?.forEach((lm, i) => { gislr[i] = [lm.x, lm.y, lm.z ?? 0]; });
    //   results.leftHandLandmarks?.forEach((lm, i) => { gislr[468 + i] = [lm.x, lm.y, lm.z ?? 0]; });
    //   results.poseLandmarks?.forEach((lm, i) => { gislr[489 + i] = [lm.x, lm.y, lm.z ?? 0]; });
    //   results.rightHandLandmarks?.forEach((lm, i) => { gislr[522 + i] = [lm.x, lm.y, lm.z ?? 0]; });
    //   resultRef.current = { gislr };
    // });

    // For now, keep resultRef null → no skeleton overlay displayed
    resultRef.current = null;

    return () => {
      resultRef.current = null;
      // TODO: holistic.close();
    };
  }, [enabled, videoRef]);

  return resultRef;
}
