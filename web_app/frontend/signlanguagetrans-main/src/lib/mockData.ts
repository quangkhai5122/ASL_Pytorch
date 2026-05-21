// Mock data for ASL-Bridge prototype

export type SignPrediction = {
  gloss: string;
  confidence: number;
  timestamp: number;
};

export type BufferToken = {
  id: string;
  gloss: string;
  confidence: number;
  timestamp: number;
};

export type DictionaryEntry = {
  id: string;
  gloss: string;
  tags: string[];
  handedness: "one-handed" | "two-handed";
  synonyms: string[];
  frequency: "common" | "uncommon" | "rare";
  examples: string[];
  notes: string;
  hasSkeleton: boolean;
  videoQuality: "high" | "medium" | "low";
};

export type ASLMode = "automatic" | "manual" | "text2sign" | "dictionary";

export type FontSize = "a" | "a-plus" | "a-plus-plus";

export type Theme = "light" | "high-contrast";

export const MOCK_PREDICTIONS: SignPrediction[] = [
  { gloss: "NOSE", confidence: 0.21, timestamp: Date.now() - 2000 },
  { gloss: "CRY", confidence: 0.04, timestamp: Date.now() - 1500 },
  { gloss: "RED", confidence: 0.02, timestamp: Date.now() - 1000 },
  { gloss: "MOUTH", confidence: 0.01, timestamp: Date.now() - 500 },
  { gloss: "BROTHER", confidence: 0.01, timestamp: Date.now() },
];

export const MOCK_BUFFER: BufferToken[] = [
  { id: "1", gloss: "GRANDMA", confidence: 0.92, timestamp: Date.now() - 10000 },
  { id: "2", gloss: "READ", confidence: 0.87, timestamp: Date.now() - 8000 },
  { id: "3", gloss: "STORY", confidence: 0.78, timestamp: Date.now() - 6000 },
  { id: "4", gloss: "BEFORE", confidence: 0.85, timestamp: Date.now() - 4000 },
  { id: "5", gloss: "BED", confidence: 0.91, timestamp: Date.now() - 2000 },
];

export const MOCK_DICTIONARY: DictionaryEntry[] = [
  { id: "d1", gloss: "HELLO", tags: ["greeting"], handedness: "one-handed", synonyms: ["HI", "HEY"], frequency: "common", examples: ["HELLO MY NAME H-E-L-E-N"], notes: "Wave hand near face", hasSkeleton: true, videoQuality: "high" },
  { id: "d2", gloss: "THANK-YOU", tags: ["courtesy"], handedness: "one-handed", synonyms: ["THANKS"], frequency: "common", examples: ["THANK-YOU FOR HELP"], notes: "Flat hand moves away from chin", hasSkeleton: true, videoQuality: "high" },
  { id: "d3", gloss: "PLEASE", tags: ["courtesy"], handedness: "one-handed", synonyms: [], frequency: "common", examples: ["PLEASE HELP ME"], notes: "Circular motion on chest", hasSkeleton: true, videoQuality: "medium" },
  { id: "d4", gloss: "SORRY", tags: ["emotion"], handedness: "one-handed", synonyms: ["APOLOGIZE"], frequency: "common", examples: ["SORRY I LATE"], notes: "Fist circles on chest", hasSkeleton: false, videoQuality: "medium" },
  { id: "d5", gloss: "LOVE", tags: ["emotion"], handedness: "two-handed", synonyms: ["ADORE"], frequency: "common", examples: ["I LOVE YOU"], notes: "Cross arms over chest", hasSkeleton: true, videoQuality: "high" },
  { id: "d6", gloss: "UNIVERSITY", tags: ["education"], handedness: "two-handed", synonyms: ["COLLEGE"], frequency: "uncommon", examples: ["I GO UNIVERSITY"], notes: "U handshape circles upward", hasSkeleton: true, videoQuality: "medium" },
  { id: "d7", gloss: "SCIENCE", tags: ["education"], handedness: "two-handed", synonyms: [], frequency: "uncommon", examples: ["SCIENCE CLASS FUN"], notes: "Alternating pouring motion", hasSkeleton: true, videoQuality: "high" },
  { id: "d8", gloss: "TECHNOLOGY", tags: ["education", "modern"], handedness: "one-handed", synonyms: ["TECH"], frequency: "uncommon", examples: ["TECHNOLOGY CHANGE FAST"], notes: "Modified T handshape at temple", hasSkeleton: false, videoQuality: "low" },
  { id: "d9", gloss: "FAMILY", tags: ["people"], handedness: "two-handed", synonyms: [], frequency: "common", examples: ["MY FAMILY BIG"], notes: "F handshapes circle outward", hasSkeleton: true, videoQuality: "high" },
  { id: "d10", gloss: "FRIEND", tags: ["people"], handedness: "two-handed", synonyms: ["PAL", "BUDDY"], frequency: "common", examples: ["MY FRIEND NAME J-O-H-N"], notes: "Linked index fingers", hasSkeleton: true, videoQuality: "medium" },
];

export const MOCK_TEXT2SIGN_INPUT = "I study at science and technology university";
export const MOCK_TEXT2SIGN_GLOSS = ["I", "STUDY", "SCIENCE", "TECHNOLOGY", "UNIVERSITY"];

export const MOCK_GENERATED_SENTENCE = "No, I want to move Generate Sentence";

export const SKELETON_JOINTS = [
  // Simple stick figure joint positions (normalized 0-1)
  { id: "head", x: 0.5, y: 0.12 },
  { id: "neck", x: 0.5, y: 0.2 },
  { id: "l_shoulder", x: 0.38, y: 0.25 },
  { id: "r_shoulder", x: 0.62, y: 0.25 },
  { id: "l_elbow", x: 0.3, y: 0.38 },
  { id: "r_elbow", x: 0.7, y: 0.38 },
  { id: "l_wrist", x: 0.25, y: 0.5 },
  { id: "r_wrist", x: 0.75, y: 0.5 },
  { id: "l_hip", x: 0.42, y: 0.52 },
  { id: "r_hip", x: 0.58, y: 0.52 },
];

export const SKELETON_BONES: [string, string][] = [
  ["head", "neck"],
  ["neck", "l_shoulder"],
  ["neck", "r_shoulder"],
  ["l_shoulder", "l_elbow"],
  ["r_shoulder", "r_elbow"],
  ["l_elbow", "l_wrist"],
  ["r_elbow", "r_wrist"],
  ["neck", "l_hip"],
  ["neck", "r_hip"],
  ["l_hip", "r_hip"],
];

export function getConfidenceLevel(confidence: number): "high" | "medium" | "low" {
  if (confidence >= 0.7) return "high";
  if (confidence >= 0.3) return "medium";
  return "low";
}

export function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
