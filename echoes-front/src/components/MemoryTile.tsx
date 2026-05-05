import { useState, useEffect } from 'react';
import type { MemoryItem } from '@/lib/api';
import { api } from '@/lib/api';
import { badgeShort } from '@/lib/utils';

interface Props {
  item: MemoryItem;
  onClick: () => void;
  onDelete: () => void;
}

export default function MemoryTile({ item, onClick, onDelete }: Props) {
  const [thumb, setThumb] = useState<string | null>(null);

  useEffect(() => {
    if (item.type === 'voice_memo') return;
    let cancelled = false;
    api.thumbnail(item.id).then(t => {
      if (!cancelled) setThumb(t.data_url);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [item.id, item.type]);

  return (
    <div
      onClick={onClick}
      className="group relative aspect-square rounded-lg bg-[#1a1a22] cursor-pointer overflow-hidden
        transition-all duration-160 hover:scale-[1.02] hover:shadow-[0_12px_36px_rgba(167,139,250,0.18)]"
    >
      {item.type === 'voice_memo' ? (
        <div className="w-full h-full flex items-center justify-center text-3xl text-text-muted">🎙️</div>
      ) : thumb ? (
        <img src={thumb} alt="" className="w-full h-full object-cover" />
      ) : (
        <div className="w-full h-full shimmer" />
      )}

      {item.type === 'video' && (
        <div className="absolute inset-0 flex items-center justify-center text-3xl text-white/90 pointer-events-none">▶</div>
      )}

      <div className="absolute bottom-1 left-1 text-[10px] px-1.5 py-0.5 rounded-full bg-black/55 backdrop-blur text-text pointer-events-none">
        {badgeShort(item.type)}
      </div>

      <button
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        className="absolute top-1 right-1 w-5 h-5 rounded-full bg-black/55 text-text text-xs flex items-center justify-center
          opacity-0 group-hover:opacity-100 transition-opacity hover:bg-danger"
        title="forget this memory"
      >
        ×
      </button>
    </div>
  );
}
