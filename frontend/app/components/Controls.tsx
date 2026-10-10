"use client";

import { useRef, useState, type ReactNode } from "react";

export function Slider({
  label, value, onChange, min = 0, max = 1, step = 0.01, format,
}: {
  label: string; value: number; onChange: (v: number) => void;
  min?: number; max?: number; step?: number; format?: (v: number) => string;
}) {
  return (
    <label className="block">
      <span className="flex justify-between text-sm text-slate-300">
        {label}
        <span className="tabular-nums text-pulse-cyan">{format ? format(value) : value.toFixed(2)}</span>
      </span>
      <input
        type="range" className="mt-2 w-full" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export const AUDIO_EXT = /\.(wav|mp3|flac|m4a)$/i;

export function Dropzone({
  onFiles, multiple = false, children,
}: { onFiles: (files: File[]) => void; multiple?: boolean; children: ReactNode }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
    <div
      role="button" tabIndex={0}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); onFiles([...e.dataTransfer.files]); }}
      className={`cursor-pointer rounded-2xl border border-dashed p-6 text-center transition ${
        over ? "border-pulse-cyan bg-pulse-cyan/10" : "border-white/20 hover:border-white/40"
      }`}
    >
      <input
        ref={input} type="file" hidden multiple={multiple} accept=".wav,.mp3,.flac,.m4a,audio/*"
        onChange={(e) => { onFiles([...(e.target.files ?? [])]); e.target.value = ""; }}
      />
      {children}
    </div>
  );
}

export interface Filters {
  threshold: number;
  /** 1 means "no limit". */
  maxAi: number;
  limit: number;
  tags: string;
}

export const DEFAULT_FILTERS: Filters = { threshold: 0.5, maxAi: 1, limit: 5, tags: "" };

const LIMITS = [1, 3, 5, 10];

export function FilterPanel({ value, onChange }: { value: Filters; onChange: (v: Filters) => void }) {
  const set = (p: Partial<Filters>) => onChange({ ...value, ...p });
  return (
    <section className="glass mt-10 grid gap-5 rounded-2xl p-5 sm:grid-cols-2">
      <Slider label="Similarity threshold" value={value.threshold} onChange={(v) => set({ threshold: v })} />
      <Slider
        label="Max AI-generated score" value={value.maxAi} onChange={(v) => set({ maxAi: v })}
        format={(v) => (v >= 1 ? "Any" : v.toFixed(2))}
      />
      <label className="block text-sm text-slate-300">
        Vibe tags
        <input className="input mt-2" placeholder="chill, lofi" value={value.tags} onChange={(e) => set({ tags: e.target.value })} />
      </label>
      <div className="text-sm text-slate-300">
        Matches to show
        <div className="mt-2 flex gap-2">
          {LIMITS.map((n) => (
            <button
              key={n} onClick={() => set({ limit: n })} aria-pressed={value.limit === n}
              className={`flex-1 rounded-xl border px-3 py-2 transition active:scale-95 ${
                value.limit === n ? "border-pulse-cyan bg-pulse-cyan/15 text-pulse-cyan" : "border-white/10 hover:bg-white/5"
              }`}
            >
              {n}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}