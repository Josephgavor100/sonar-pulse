type AudioCtor = typeof AudioContext;
type OfflineCtor = typeof OfflineAudioContext;

const TARGET_RATE = 16000;

/**
 * Decode any audio the browser can play (a MediaRecorder blob or an uploaded
 * file) and re-encode it as 16 kHz, mono, 16-bit PCM WAV. This is the format
 * the API wants, so no FFmpeg is needed on the server for browser recordings.
 *
 * @throws if the browser cannot decode the input.
 */
export async function toWav16k(blob: Blob): Promise<Blob> {
  const w = window as unknown as { AudioContext?: AudioCtor; webkitAudioContext?: AudioCtor; webkitOfflineAudioContext?: OfflineCtor };
  const Ctx = w.AudioContext ?? w.webkitAudioContext;
  const Offline = window.OfflineAudioContext ?? w.webkitOfflineAudioContext;
  if (!Ctx || !Offline) throw new Error("This browser does not support audio processing.");

  const ctx = new Ctx();
  let decoded: AudioBuffer;
  try {
    decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
  } finally {
    ctx.close();
  }

  // Rendering through a 1-channel, 16 kHz offline context downmixes and resamples in one step.
  const offline = new Offline(1, Math.max(1, Math.ceil(decoded.duration * TARGET_RATE)), TARGET_RATE);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start();
  const pcm = (await offline.startRendering()).getChannelData(0);

  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const str = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVEfmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, TARGET_RATE, true); v.setUint32(28, TARGET_RATE * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, pcm.length * 2, true);
  pcm.forEach((s, i) => v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
  return new Blob([buf], { type: "audio/wav" });
}