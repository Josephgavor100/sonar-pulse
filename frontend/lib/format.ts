/** Seconds to m:ss.s (83 -> "1:23.0"). */
export function formatOffset(seconds: number): string {
  const s = Math.max(seconds, 0);
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

export const percent = (v: number) => `${(v * 100).toFixed(1)}%`;

export const slugify = (t: string) => t.toLowerCase().replace(/[\W_]+/g, "-").replace(/^-+|-+$/g, "");

/** "01 - Artist - Title" -> { artist, title }; mirrors the CLI's filename rules. */
export function parseFilename(name: string): { title: string; artist: string } {
  const stem = name.replace(/\.[^.]+$/, "").replace(/^\s*\d{1,3}\s*[-._)]\s*/, "").replace(/_/g, " ").trim();
  const i = stem.indexOf(" - ");
  if (i > 0 && i + 3 < stem.length) return { artist: stem.slice(0, i).trim(), title: stem.slice(i + 3).trim() };
  return { title: stem || name, artist: "Unknown Artist" };
}

export function trackIdFor(artist: string, title: string): string {
  const base = artist === "Unknown Artist" ? title : `${artist} ${title}`;
  return slugify(base).slice(0, 100) || "track";
}