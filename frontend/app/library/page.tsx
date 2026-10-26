"use client";

import { useEffect, useMemo, useState } from "react";
import { Search, Save, Trash2, Music2 } from "lucide-react";
import { toast } from "sonner";
import { API_URL } from "@/lib/api";

type Track = { track_id: string; title: string; artist: string; vibe_tags?: string[]; ai_generated_score?: number | null; preview_url: string };

export default function LibraryPage() {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  async function refresh() {
    try {
      const response = await fetch(`${API_URL}/api/v1/library`, { cache: "no-store" });
      if (!response.ok) throw new Error("Could not load library");
      setTracks(await response.json());
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not load library"); }
    finally { setLoading(false); }
  }
  useEffect(() => { void refresh(); }, []);
  const filtered = useMemo(() => tracks.filter((track) => `${track.title} ${track.artist} ${track.track_id} ${(track.vibe_tags ?? []).join(" ")}`.toLowerCase().includes(query.toLowerCase())), [tracks, query]);
  async function save(track: Track) {
    try {
      const response = await fetch(`${API_URL}/api/v1/library/${encodeURIComponent(track.track_id)}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: track.title, artist: track.artist, vibe_tags: track.vibe_tags ?? [], ai_generated_score: track.ai_generated_score ?? null }) });
      if (!response.ok) throw new Error("Could not save track");
      toast.success("Track updated");
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not save track"); }
  }
  async function remove(track: Track) {
    if (!window.confirm(`Delete ${track.title} by ${track.artist} from your library?`)) return;
    try {
      const response = await fetch(`${API_URL}/api/v1/library/${encodeURIComponent(track.track_id)}`, { method: "DELETE" });
      if (!response.ok) throw new Error("Could not delete track");
      setTracks((current) => current.filter((item) => item.track_id !== track.track_id));
      toast.success("Track deleted");
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not delete track"); }
  }
  function change(trackId: string, field: "title" | "artist", value: string) {
    setTracks((current) => current.map((track) => track.track_id === trackId ? { ...track, [field]: value } : track));
  }
  return <section className="py-10">
    <div className="mb-8"><p className="text-sm uppercase tracking-[0.25em] text-cyan-300">Your collection</p><h1 className="mt-2 font-display text-4xl font-bold">Music library</h1><p className="mt-2 text-slate-400">Search, preview, and manage your indexed tracks.</p></div>
    <label className="glass mb-6 flex items-center gap-3 rounded-2xl px-4 py-3"><Search className="h-5 w-5 text-slate-400"/><input className="w-full bg-transparent outline-none" placeholder="Search title, artist, tags, or ID" value={query} onChange={(event) => setQuery(event.target.value)} /></label>
    {loading ? <p className="text-slate-400">Loading library…</p> : filtered.length === 0 ? <div className="glass rounded-2xl p-10 text-center"><Music2 className="mx-auto mb-3 h-8 w-8 text-cyan-300"/><p className="text-slate-300">{tracks.length ? "No tracks match your search." : "Your library is empty. Index tracks to see them here."}</p></div> : <div className="space-y-4">{filtered.map((track) => <article key={track.track_id} className="glass grid gap-4 rounded-2xl p-5 md:grid-cols-[1fr_auto] md:items-center"><div className="grid gap-3 sm:grid-cols-2"><label className="text-xs text-slate-400">Title<input className="input mt-1" value={track.title} onChange={(event) => change(track.track_id, "title", event.target.value)} /></label><label className="text-xs text-slate-400">Artist<input className="input mt-1" value={track.artist} onChange={(event) => change(track.track_id, "artist", event.target.value)} /></label><p className="text-xs text-slate-500 sm:col-span-2">ID: {track.track_id}{track.vibe_tags?.length ? ` · ${track.vibe_tags.join(", ")}` : ""}</p><audio controls preload="none" src={`${API_URL}${track.preview_url}`} className="h-10 w-full sm:col-span-2" /></div><div className="flex gap-2"><button className="btn-primary !px-4" onClick={() => void save(track)} aria-label={`Save ${track.title}`}><Save className="h-4 w-4"/>Save</button><button className="btn-ghost !px-4 text-rose-300" onClick={() => void remove(track)} aria-label={`Delete ${track.title}`}><Trash2 className="h-4 w-4"/>Delete</button></div></article>)}</div>}
  </section>;
}
