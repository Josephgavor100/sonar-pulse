import type {
  IngestMetadata,
  IngestResponse,
  RecognizeOptions,
  RecognizeResponse,
} from "@/types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

async function parse<T>(res: Response): Promise<T> {
  if (res.ok) return res.json() as Promise<T>;
  let detail = `Request failed (${res.status})`;
  try {
    const body = await res.json();
    if (typeof body.detail === "string") detail = body.detail;
    else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    /* non-JSON error body */
  }
  throw new ApiError(detail, res.status);
}

async function post<T>(path: string, form: FormData, signal?: AbortSignal): Promise<T> {
  try {
    return await parse<T>(await fetch(`${API_URL}${path}`, { method: "POST", body: form, signal }));
  } catch (err) {
    if (err instanceof ApiError || (err instanceof DOMException && err.name === "AbortError")) throw err;
    throw new ApiError(`Can't reach the Sonar Pulse API at ${API_URL}. Is the backend running?`, 0);
  }
}

export function recognize(clip: Blob, o: RecognizeOptions, signal?: AbortSignal) {
  const form = new FormData();
  form.append("file", clip, clip instanceof File ? clip.name : "clip.wav");
  form.append("limit", String(o.limit));
  form.append("score_threshold", String(o.scoreThreshold));
  if (o.maxAiScore !== undefined) form.append("max_ai_generated_score", String(o.maxAiScore));
  if (o.vibeTags?.trim()) form.append("vibe_tags", o.vibeTags);
  return post<RecognizeResponse>("/api/v1/recognize", form, signal);
}

export function ingest(file: File, m: IngestMetadata, signal?: AbortSignal) {
  const form = new FormData();
  form.append("file", file, file.name);
  form.append("track_id", m.trackId);
  form.append("title", m.title);
  form.append("artist", m.artist);
  if (m.vibeTags?.trim()) form.append("vibe_tags", m.vibeTags);
  if (m.aiGeneratedScore !== undefined) form.append("ai_generated_score", String(m.aiGeneratedScore));
  return post<IngestResponse>("/api/v1/ingest", form, signal);
}

/** True when the API answers /health. */
export async function getHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}