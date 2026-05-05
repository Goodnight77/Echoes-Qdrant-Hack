import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmtAgo(ts: number | null): string {
  if (!ts) return '';
  const ms = Date.now() - ts * 1000;
  const d = Math.floor(ms / 86400000);
  if (d === 0) return 'today';
  if (d === 1) return 'yesterday';
  if (d < 30) return `${d} days ago`;
  return `${Math.floor(d / 30)} months ago`;
}

export function badge(type: string): string {
  const map: Record<string, string> = {
    photo: '📷 Photo',
    video: '🎥 Video',
    voice_memo: '🎙️ Voice',
    screenshot: '🖼️ Screenshot',
  };
  return map[type] || type;
}

export function badgeShort(type: string): string {
  const map: Record<string, string> = {
    photo: '📷',
    video: '🎥',
    voice_memo: '🎙️',
    screenshot: '🖼️',
  };
  return map[type] || type;
}

export function highlight(text: string | undefined, query: string): string {
  if (!text || !query) return text || '';
  const safe = query.split(/\s+/).filter(w => w.length > 2);
  let out = text;
  for (const w of safe) {
    const re = new RegExp(`(${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'ig');
    out = out.replace(re, '<mark>$1</mark>');
  }
  return out;
}

export function fmtTs(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}
