import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import type { MemoryItem } from '@/lib/api';
import { api } from '@/lib/api';
import { badge, fmtAgo, highlight, fmtTs } from '@/lib/utils';

interface Props {
  item: MemoryItem;
  query: string;
  onClose: () => void;
}

export default function ImageViewer({ item, query, onClose }: Props) {
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const tail = item.matched_via?.length ? ` · matched via ${item.matched_via.join(' + ')}` : '';

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={(e) => { if (e.target === overlayRef.current) onClose(); }}
    >
      <div className="max-w-3xl w-full bg-[#0b0b10] border border-[#1e1e28] rounded-2xl p-6 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="text-sm text-text-dim">
            {badge(item.type)} · {fmtAgo(item.timestamp)}{tail}
          </div>
          <button onClick={onClose} className="text-text-muted hover:text-text transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content by type */}
        {item.type === 'photo' || item.type === 'screenshot' ? (
          <>
            <img src={api.mediaUrl(item.id)} alt="" className="w-full rounded-lg" />
            {item.ocr_text && (
              <div
                className="mt-4 text-sm text-text-dim bg-[#111118] rounded-lg p-3 whitespace-pre-wrap"
                dangerouslySetInnerHTML={{ __html: highlight(item.ocr_text, query) }}
              />
            )}
          </>
        ) : item.type === 'video' ? (
          <video
            src={api.mediaUrl(item.id)}
            controls
            className="w-full rounded-lg"
            autoPlay={false}
            onLoadedMetadata={(e) => {
              if (item.best_moment_seconds !== undefined) {
                (e.target as HTMLVideoElement).currentTime = item.best_moment_seconds;
              }
            }}
          />
        ) : item.type === 'voice_memo' ? (
          <>
            <audio src={api.mediaUrl(item.id)} controls className="w-full mt-2" />
            {item.transcript && (
              <div
                className="mt-4 text-sm text-text leading-relaxed"
                dangerouslySetInnerHTML={{ __html: highlight(item.transcript, query) }}
              />
            )}
          </>
        ) : null}
      </div>
    </div>
  );
}
