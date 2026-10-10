"use client";

import { motion } from "framer-motion";
import { useEffect, useRef, type MutableRefObject } from "react";

export type OrbState = "idle" | "listening" | "matching";

/** Glowing orb: breathes when idle, follows mic level while listening, pulses while matching. */
export default function PulseOrb({
  state,
  levelRef,
  size = 280,
}: {
  state: OrbState;
  levelRef: MutableRefObject<number>;
  size?: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext("2d");
    if (!el || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = el.height = size * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let raf = 0, t = 0, level = 0;
    const c = size / 2;

    const draw = () => {
      t += calm ? 0 : 0.016;
      const target =
        state === "listening" ? levelRef.current : state === "matching" ? 0.5 + 0.3 * Math.sin(t * 5) : 0.1 + 0.05 * Math.sin(t * 1.6);
      level += (target - level) * 0.15;
      const r = size * 0.2 * (1 + level * 0.9);

      ctx.clearRect(0, 0, size, size);
      for (let i = 0; i < 3; i++) {
        const p = (t * (state === "matching" ? 0.9 : 0.3) + i / 3) % 1;
        ctx.beginPath();
        ctx.arc(c, c, r + p * size * 0.26, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(0,242,254,${0.35 * (1 - p)})`;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      ctx.beginPath();
      for (let k = 0; k <= 72; k++) {
        const a = (k / 72) * Math.PI * 2;
        const rr = r * (1 + 0.07 * (0.4 + level) * Math.sin(a * 5 + t * 3) + 0.04 * Math.sin(a * 3 - t * 2));
        const x = c + Math.cos(a) * rr, y = c + Math.sin(a) * rr;
        k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      const g = ctx.createRadialGradient(c - r * 0.3, c - r * 0.3, r * 0.1, c, c, r * 1.1);
      g.addColorStop(0, "#9ffcff"); g.addColorStop(0.45, "#00f2fe"); g.addColorStop(1, "#4facfe");
      ctx.shadowColor = "rgba(0,242,254,0.8)";
      ctx.shadowBlur = 30 + level * 50;
      ctx.fillStyle = g;
      ctx.fill();
      ctx.shadowBlur = 0;
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [state, size, levelRef]);

  const label = { idle: "Ready to listen", listening: "Listening", matching: "Matching your clip" }[state];
  return (
    <motion.div
      role="img"
      aria-label={label}
      animate={{ scale: state === "idle" ? 1 : 1.04, opacity: state === "matching" ? [1, 0.82, 1] : 1 }}
      transition={{ duration: state === "matching" ? 1.2 : 0.4, repeat: state === "matching" ? Infinity : 0 }}
    >
      <canvas ref={canvas} style={{ width: size, height: size }} aria-hidden />
    </motion.div>
  );
}