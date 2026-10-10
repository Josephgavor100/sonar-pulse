"use client";

import Link from "next/link";
import { useState } from "react";
import { Library } from "lucide-react";
import { toast } from "sonner";
import { AUDIO_EXT, Dropzone } from "./Controls";
import { ingest } from "@/lib/api";
import { parseFilename, trackIdFor } from "@/lib/format";

/** Compact drop zone: index a few songs without leaving the home page. */
export default function QuickIndex() {
  const [progress, setProgress] = useState<string | null>(null);

  async function addFiles(files: File[]) {
    const audio = files.filter((f) => AUDIO_EXT.test(f.name));
    if (!audio.length) return toast.error("Choose .wav, .mp3, .flac or .m4a files.");

    let ok = 0;
    for (const [i, file] of audio.entries()) {
      const { title, artist } = parseFilename(file.name);
      setProgress(`Indexing ${i + 1}/${audio.length}: ${artist} - ${title}`);
      try {
        await ingest(file, { trackId: trackIdFor(artist, title), title, artist });
        ok++;
      } catch (e) {
        toast.error(`${file.name}: ${e instanceof Error ? e.message : "upload failed"}`);
      }
    }
    setProgress(null);
    if (ok) toast.success(`Added ${ok} track${ok > 1 ? "s" : ""} to your library`);
  }

  return (
    <section className="glass mt-12 rounded-2xl p-6">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="font-display text-xl font-semibold">Teach it new songs</h2>
          <p className="text-sm text-slate-400">Drop a few tracks to index them. Titles come from filenames like &quot;Artist - Title&quot;.</p>
        </div>
        <Link href="/ingest" className="btn-ghost shrink-0 !px-4 !py-2 text-sm">
          <Library className="h-4 w-4" /> Bulk import
        </Link>
      </div>
      <Dropzone multiple onFiles={addFiles}>
        <p className="text-sm text-slate-300">{progress ?? "Drop audio files here, or click to browse"}</p>
      </Dropzone>
    </section>
  );
}