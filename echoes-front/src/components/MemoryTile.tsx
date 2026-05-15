import { useState, useEffect } from 'react';
import type { MemoryItem } from '@/lib/api';
import { api } from '@/lib/api';
import { badgeShort, fmtAgo } from '@/lib/utils';
import { Mic, Video } from 'lucide-react';

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
      className="group relative aspect-square rounded-xl bg-[#111118] border border-[#1e1e28] cursor-pointer overflow-hidden
        transition-all duration-200 hover:scale-[1.03] hover:border-accent/50 hover:shadow-[0_8px_32px_rgba(167,139,250,0.15)]"
    >
      {/* Thumbnail */}
      {item.type === 'voice_memo' ? (
        <div className="w-full h-full flex items-center justify-center bg-[#1a1a22]">
          <Mic className="w-10 h-10 text-accent/60" />
        </div>
      ) : thumb ? (
        <img src={thumb} alt="" className="w-full h-full object-cover" />
      ) : (
        <div className="w-full h-full shimmer" />
      )}

      {/* Video play overlay */}
      {item.type === 'video' && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/20">
          <div className="w-12 h-12 rounded-full bg-black/60 backdrop-blur flex items-center justify-center">
            <Video className="w-5 h-5 text-white fill-white" />
          </div>
        </div>
      )}

      {/* Delete button */}
      <button
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        className="absolute top-2 right-2 w-6 h-6 rounded-full bg-black/60 backdrop-blur text-text text-xs flex items-center justify-center
          opacity-0 group-hover:opacity-100 transition-opacity hover:bg-danger z-10"
        title="forget this memory"
      >
        ×
      </button>

      {/* Metadata overlay at bottom */}
      <div className="absolute inset-x-0 bottom-0 p-2 bg-gradient-to-t from-black/80 via-black/40 to-transparent">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-accent/20 text-accent font-medium">
            {badgeShort(item.type)}
          </span>
          <span className="text-[10px] text-text-dim">{fmtAgo(item.timestamp)}</span>
        </div>
        {(item.transcript || item.ocr_text) && (
          <p className="text-[11px] text-text-dim mt-1 line-clamp-2 leading-snug">
            {(item.transcript || item.ocr_text || '').slice(0, 60)}
          </p>
        )}
      </div>
    </div>
  );
}
