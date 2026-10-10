export default function Navbar() {
  return (
    <header className="w-full border-b border-white/10 bg-slate-950/60 backdrop-blur-md py-4 px-6 flex items-center justify-between sticky top-0 z-50">
      <div className="flex items-center space-x-3">
        <div className="w-3 h-3 rounded-full bg-cyan-400 animate-pulse" />
        <span className="font-bold tracking-wider text-white text-lg">SONAR PULSE</span>
      </div>
      <div className="text-sm text-slate-400">AI Sound Recognizer & Indexer</div>
    </header>
  );
}