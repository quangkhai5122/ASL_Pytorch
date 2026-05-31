/**
 * useMediaPipeSkeleton — runs MediaPipe Holistic landmark detection
 * directly in the browser so the skeleton overlay is perfectly in sync
 * with the camera feed (zero network latency).
 *
 * Returns a ref that always holds the latest landmarks for drawing.
 */

import { useRef, useEffect, useCallback } from "react";

// MediaPipe tasks-vision types
interface HolisticLandmarkerType {
  detectForVideo(
    video: HTMLVideoElement,
    timestamp: number,
  ): HolisticResult;
  close(): void;
}

interface NormalizedLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

interface HolisticResult {
  faceLandmarks: NormalizedLandmark[][];
  poseLandmarks: NormalizedLandmark[][];
  leftHandLandmarks: NormalizedLandmark[][];
  rightHandLandmarks: NormalizedLandmark[][];
}

export interface SkeletonLandmarks {
  /** GISLR-ordered flat array [543] of [x,y,z] or null */
  gislr: (number[] | null)[];
}

/**
 * Hook that initialises MediaPipe HolisticLandmarker in the browser
 * and continuously extracts landmarks from a <video> element.
 *
 * @param videoRef  ref to the webcam <video> element
 * @param enabled   whether detection should run (camera active)
 */
export function useMediaPipeSkeleton(
  videoRef: React.RefObject<HTMLVideoElement>,
  enabled: boolean,
) {
  const landmarksRef = useRef<SkeletonLandmarks | null>(null);
  const holisticRef = useRef<HolisticLandmarkerType | null>(null);
  const rafRef = useRef<number | null>(null);
  const initedRef = useRef(false);

  // Initialise the HolisticLandmarker (once)
  const init = useCallback(async () => {
    if (initedRef.current) return;
    initedRef.current = true;

    try {
      const vision = await import("@mediapipe/tasks-vision");
      const { FilesetResolver, HolisticLandmarker } = vision;

      const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
      );

      const holistic = await HolisticLandmarker.createFromOptions(
        filesetResolver,
        {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minHandLandmarksConfidence: 0.5,
        },
      );

      holisticRef.current = holistic as unknown as HolisticLandmarkerType;
      console.log("[MediaPipe] HolisticLandmarker initialised (browser-side)");
    } catch (err) {
      console.error("[MediaPipe] Failed to initialise:", err);
      initedRef.current = false;
    }
  }, []);

  // Detection loop using requestAnimationFrame
  useEffect(() => {
    if (!enabled) {
      // Cleanup when disabled
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      landmarksRef.current = null;
      return;
    }

    // Start init
    init();

    let lastTimestamp = -1;

    const detect = () => {
      const video = videoRef.current;
      const holistic = holisticRef.current;

      if (video && holistic && video.readyState >= 2) {
        const now = performance.now();
        // Only detect if video timestamp changed (new frame)
        if (video.currentTime !== lastTimestamp) {
          lastTimestamp = video.currentTime;

          try {
            const result = holistic.detectForVideo(video, now);
            landmarksRef.current = convertToGISLR(result);
          } catch {
            // Silently skip detection errors (e.g. video not ready)
          }
        }
      }

      rafRef.current = requestAnimationFrame(detect);
    };

    rafRef.current = requestAnimationFrame(detect);

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [enabled, init, videoRef]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (holisticRef.current) {
        try {
          holisticRef.current.close();
        } catch { /* ignore */ }
        holisticRef.current = null;
        initedRef.current = false;
      }
    };
  }, []);

  return landmarksRef;
}

/**
 * Convert HolisticLandmarker result into the GISLR-ordered [543] array.
 *
 * GISLR ordering:
 *   Face:       0..467   (468 pts)
 *   Left hand:  468..488 (21 pts)
 *   Pose:       489..521 (33 pts)
 *   Right hand: 522..542 (21 pts)
 */
function convertToGISLR(result: HolisticResult): SkeletonLandmarks {
  const gislr: (number[] | null)[] = new Array(543).fill(null);

  // Face landmarks (468 points)
  if (result.faceLandmarks?.[0]) {
    const face = result.faceLandmarks[0];
    for (let i = 0; i < Math.min(face.length, 468); i++) {
      gislr[i] = [face[i].x, face[i].y, face[i].z];
    }
  }

  // Left hand (21 points) — offset 468
  if (result.leftHandLandmarks?.[0]) {
    const lh = result.leftHandLandmarks[0];
    for (let i = 0; i < Math.min(lh.length, 21); i++) {
      gislr[468 + i] = [lh[i].x, lh[i].y, lh[i].z];
    }
  }

  // Pose (33 points) — offset 489
  if (result.poseLandmarks?.[0]) {
    const pose = result.poseLandmarks[0];
    for (let i = 0; i < Math.min(pose.length, 33); i++) {
      gislr[489 + i] = [pose[i].x, pose[i].y, pose[i].z];
    }
  }

  // Right hand (21 points) — offset 522
  if (result.rightHandLandmarks?.[0]) {
    const rh = result.rightHandLandmarks[0];
    for (let i = 0; i < Math.min(rh.length, 21); i++) {
      gislr[522 + i] = [rh[i].x, rh[i].y, rh[i].z];
    }
  }

  return { gislr };
}
