"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Mic, RotateCw, SearchX, Square, Upload } from "lucide-react";
import { toast } from "sonner";
import PulseOrb, { type OrbState } from "./PulseOrb";
import MatchCard, { ClipPlayer, listVariants } from "./MatchCard";
import { AUDIO_EXT, DEFAULT_FILTERS, Dropzone, FilterPanel, type Filters } from "./Controls";
import { recognize } from "@/lib/api";
import { toWav16k } from "@/lib/wav";
import type { RecognizeResponse } from "@/types";

type Phase = "idle" | "recording" | "matching";
const MAX_RECORD_MS = 15000;

export default function Recognizer() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [clip, setClip] = useState<{ blob: Blob; url: string; label: string } | null>(null);
  const [result, setResult] = useState<RecognizeResponse | null>(null);
  const [f, setF] = useState<Filters>(DEFAULT_FILTERS);
  const levelRef = useRef(0);
  const stopRec = useRef<(() => void) | null>(null);

  useEffect(() => () => { stopRec.current?.(); }, []);

  const identify = useCallback(
    async (blob: Blob) => {
      setPhase("matching");
      setResult(null);
      try {
        const res = await recognize(blob, {
          limit: f.limit, scoreThreshold: f.threshold,
          maxAiScore: f.maxAi < 1 ? f.maxAi : undefined, vibeTags: f.tags,
        });
        setResult(res);
        res.matches.length
          ? toast.success(`Found ${res.matches.length} match${res.matches.length > 1 ? "es" : ""}`)
          : toast.message("No matches above your threshold");
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Something went wrong. Try again.");
      } finally {
        setPhase("idle");
      }
    },
    [f],
  );

  const submit = (blob: Blob, label: string) => {
    setClip((prev) => {
      if (prev) URL.revokeObjectURL(prev.url);
      return { blob, label, url: URL.createObjectURL(blob) };
    });
    identify(blob);
  };

  const onFiles = (files: File[]) => {
    const file = files.find((x) => AUDIO_EXT.test(x.name));
    file ? submit(file, file.name) : toast.error("Choose a .wav, .mp3, .flac or .m4a file.");
  };

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } });
      const ac = new AudioContext();
      const analyser = ac.createAnalyser();
      analyser.fftSize = 256;
      ac.createMediaStreamSource(stream).connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);
      let raf = 0;
      const tick = () => {
        analyser.getByteTimeDomainData(data);
        levelRef.current = Math.min(1, Math.max(...data.map((v) => Math.abs(v - 128))) / 90);
        raf = requestAnimationFrame(tick);
      };
      tick();

      const mr = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      mr.ondataavailable = (e) => chunks.push(e.data);
      mr.onstop = async () => {
        cancelAnimationFrame(raf);
        levelRef.current = 0;
        stream.getTracks().forEach((t) => t.stop());
        ac.close();
        try {
          submit(await toWav16k(new Blob(chunks, { type: mr.mimeType })), "Microphone clip");
        } catch {
          setPhase("idle");
          toast.error("Couldn't read the recording. Try again, or upload a file.");
        }
      };
      const stop = () => mr.state !== "inactive" && mr.stop();
      stopRec.current = stop;
      mr.start();
      setPhase("recording");
      setTimeout(stop, MAX_RECORD_MS);
    } catch {
      toast.error("Microphone access is blocked. Allow it in your browser, or upload a file.");
    }
  }

  const orb: OrbState = phase === "recording" ? "listening" : phase === "matching" ? "matching" : "idle";
  const busy = phase === "matching";

  return (
    <div>
      <section className="flex flex-col items-center pt-10 text-center">
        <PulseOrb state={orb} levelRef={levelRef} />
        <h1 className="mt-2 font-display text-4xl font-semibold sm:text-5xl">
          {phase === "recording" ? "Listening" : busy ? "Matching your clip" : "What's playing?"}
        </h1>
        <p className="mt-3 max-w-md text-slate-400">
          Record a few seconds or drop an audio file. Sonar Pulse finds the track and where in it your clip starts.
        </p>
        <div className="mt-6 flex gap-3">
          {phase === "recording" ? (
            <button className="btn-primary" onClick={() => stopRec.current?.()}><Square className="h-4 w-4" /> Stop and identify</button>
          ) : (
            <button className="btn-primary" disabled={busy} onClick={startRecording}><Mic className="h-4 w-4" /> Listen</button>
          )}
          {clip && !busy && phase === "idle" && (
            <button className="btn-ghost" onClick={() => identify(clip.blob)}><RotateCw className="h-4 w-4" /> Search again</button>
          )}
        </div>
        <div className="mt-6 w-full max-w-xl">
          <Dropzone onFiles={onFiles}>
            <Upload className="mx-auto h-6 w-6 text-pulse-cyan" />
            <p className="mt-2 text-sm text-slate-300">Drop an audio file here, or click to browse</p>
            <p className="text-xs text-slate-500">WAV, MP3, FLAC or M4A</p>
          </Dropzone>
        </div>
        {clip && (
          <div className="mt-4 w-full max-w-xl text-left">
            <ClipPlayer url={clip.url} label={clip.label} />
          </div>
        )}
      </section>

      <FilterPanel value={f} onChange={setF} />

      <section className="mt-8" aria-live="polite">
        {busy && (
          <ul className="space-y-3">
            {[0, 1, 2].map((i) => <li key={i} className="glass h-24 animate-pulse rounded-2xl" />)}
          </ul>
        )}
        {result && result.matches.length > 0 && (
          <>
            <p className="mb-3 text-sm text-slate-400">
              {result.query_duration_seconds.toFixed(1)}s clip, {result.query_windows} window{result.query_windows > 1 ? "s" : ""} checked
            </p>
            <motion.ul variants={listVariants} initial="hidden" animate="show" className="space-y-3">
              {result.matches.map((m, i) => <MatchCard key={m.track_id} match={m} top={i === 0} />)}
            </motion.ul>
          </>
        )}
        {result && result.matches.length === 0 && (
          <div className="glass rounded-2xl p-8 text-center">
            <SearchX className="mx-auto h-8 w-8 text-slate-500" />
            <p className="mt-3 font-medium">No track matched this clip</p>
            <p className="mt-1 text-sm text-slate-400">Lower the similarity threshold, loosen the filters, or add the song on the Add songs page.</p>
          </div>
        )}
      </section>
    </div>
  );
}