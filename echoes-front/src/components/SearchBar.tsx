import { useState, useEffect, type ChangeEvent } from 'react';
import { Search } from 'lucide-react';

interface Props {
  value: string;
  onChange: (v: string) => void;
}

const PLACEHOLDERS = [
  "dog at the beach",
  "what Sarah said about the venue",
  "that error from Tuesday",
  "sunset photos",
  "race condition bug",
  "ceramic mug gift",
];

export default function SearchBar({ value, onChange }: Props) {
  const [phIdx, setPhIdx] = useState(0);
  const [ph, setPh] = useState(PLACEHOLDERS[0]);

  useEffect(() => {
    const iv = setInterval(() => {
      setPhIdx(i => {
        const next = (i + 1) % PLACEHOLDERS.length;
        setPh(PLACEHOLDERS[next]);
        return next;
      });
    }, 3000);
    return () => clearInterval(iv);
  }, []);

  return (
    <div className="relative mb-6">
      <Search className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-text-muted" />
      <input
        type="text"
        autoComplete="off"
        autoFocus
        value={value}
        onChange={(e: ChangeEvent<HTMLInputElement>) => onChange(e.target.value)}
        placeholder={value ? '' : ph}
        className="w-full bg-[#111118]/80 border border-[#1e1e28] rounded-2xl py-5 pl-12 pr-6 text-lg text-text placeholder:text-text-muted focus-ring transition-colors"
      />
    </div>
  );
}
