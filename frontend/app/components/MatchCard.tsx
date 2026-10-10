"use client";

import { motion } from "framer-motion";
import type { MatchResult } from "@/types";
import { formatOffset, percent } from "@/lib/format";

const hue = (id: string) => [...id].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 7);

export const listVariants = { show: { transition: { staggerChildren: 0.09 } } };
const itemVariants = { hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0 } };

export default function MatchCard({ match, top }: { match: MatchResult; top: boolean }) {
  const h = hue(match.track_id);
  return (
    <motion.li variants={itemVariants} whileHover={{ y: -3 }}
      className={`rounded-2xl p-px ${top ? "bg-gradient-to-br from-pulse-cyan via-pulse-blue to-violet-500 shadow-[0_0_40px_rgba(0,242,254,0.2)]" : "bg-white/10"}`}
    >
      <div className="glass flex items-center gap-4 rounded-[15px] p-4">
        <div
          className="grid h-16 w-16 shrink-0 place-items-center rounded-xl font-display text-2xl font-semibold text-white/90"
          style={{ background: `linear-gradient(135deg, hsl(${h} 80% 55%), hsl(${(h + 60) % 360} 80% 30%))` }}
          aria-hidden
        >
          {(match.title ?? "?").charAt(0).toUpperCase()}
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate font-display text-lg font-semibold">{match.title ?? "Untitled"}</p>
          <p className="truncate text-slate-300">{match.artist ?? "Unknown artist"}</p>
          <p className="truncate text-xs text-slate-500">{match.track_id}</p>
          {match.vibe_tags.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {match.vibe_tags.map((t) => (
                <span key={t} className="rounded-full bg-violet-500/20 px-2 py-0.5 text-xs text-violet-200">{t}</span>
              ))}
            </div>
          )}
        </div>
        <div className="shrink-0 text-right">
          <span className={`inline-block rounded-full px-3 py-1 text-sm font-semibold tabular-nums ${top ? "bg-pulse-cyan text-slate-950" : "bg-white/10"}`}>
            {percent(match.confidence)}
          </span>
          <p className="mt-2 text-xs text-slate-400">{match.matched_windows}/{match.query_windows} windows</p>
          <p className="text-xs text-slate-400">starts at {formatOffset(match.offset_seconds)}</p>
          {match.ai_generated_score !== null && (
            <p className="text-xs text-slate-500">AI score {match.ai_generated_score.toFixed(2)}</p>
          )}
        </div>
      </div>
    </motion.li>
  );
}

/** Playback of the clip the user recorded or uploaded, shown above the results. */
export function ClipPlayer({ url, label }: { url: string; label: string }) {
  return (
    <div className="glass rounded-2xl p-3">
      <p className="mb-2 truncate text-xs text-slate-400">Your clip: {label}</p>
      <audio controls src={url} className="w-full" />
    </div>
  );
}