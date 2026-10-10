export interface MatchResult {
  track_id: string;
  title: string | null;
  artist: string | null;
  vibe_tags: string[];
  ai_generated_score: number | null;
  /** Mean best-window similarity over all query windows (0-1). */
  confidence: number;
  best_window_score: number;
  matched_windows: number;
  query_windows: number;
  /** Where the query clip starts inside the track, in seconds. */
  offset_seconds: number;
}

export interface RecognizeResponse {
  matches: MatchResult[];
  query_windows: number;
  query_duration_seconds: number;
}

export interface IngestResponse {
  track_id: string;
  title: string;
  artist: string;
  windows_indexed: number;
  duration_seconds: number;
}

export interface RecognizeOptions {
  limit: number;
  scoreThreshold: number;
  maxAiScore?: number;
  vibeTags?: string;
}

export interface IngestMetadata {
  trackId: string;
  title: string;
  artist: string;
  vibeTags?: string;
  aiGeneratedScore?: number;
}