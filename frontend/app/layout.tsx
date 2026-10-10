import type { Metadata } from "next";
import { Manrope, Space_Grotesk } from "next/font/google";
import { Toaster } from "sonner";
import Navbar from "./components/Navbar";
import "./globals.css";

const display = Space_Grotesk({ subsets: ["latin"], variable: "--font-display" });
const body = Manrope({ subsets: ["latin"], variable: "--font-body" });

export const metadata: Metadata = {
  title: "Sonar Pulse",
  description: "Identify any song from a few seconds of audio.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`}>
      <body className="flex min-h-screen flex-col font-sans text-white antialiased">
        <Navbar />
        <main className="mx-auto w-full max-w-5xl flex-1 px-4 pb-16">{children}</main>
        <footer className="border-t border-white/10 py-6 text-center text-sm text-slate-400">
          © 2026 Sonar Pulse. Built by Joseph Gavor.
        </footer>
        <Toaster theme="dark" position="bottom-right" richColors />
      </body>
    </html>
  );
}