import { useState, useEffect, useRef } from 'react';
import type { FaceCluster, MemoryItem } from '@/lib/api';
import { api } from '@/lib/api';
import MemoryTile from './MemoryTile';
import { X } from 'lucide-react';

interface Props {
  cluster: FaceCluster | null;
  onClose: () => void;
  onSaved: () => void;
}

export default function PersonPanel({ cluster, onClose, onSaved }: Props) {
  const [label, setLabel] = useState('');
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [imgError, setImgError] = useState(false);
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!cluster) return;
    setLabel(cluster.label || '');
    setImgError(false);
    setLoading(true);
    api.faceClusterMemories(cluster.cluster_id)
      .then(data => setMemories(data.items || []))
      .catch(() => setMemories([]))
      .finally(() => setLoading(false));
  }, [cluster]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const handleSave = async () => {
    if (!cluster || !label.trim()) return;
    setSaving(true);
    try {
      await api.faceLabel(cluster.cluster_id, label.trim());
      onSaved();
    } catch (e: any) {
      alert(`failed: ${e.message}`);
    } finally {
      setSaving(false);
    }
  };

  if (!cluster) return null;

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={(e) => { if (e.target === overlayRef.current) onClose(); }}
    >
      <div className="max-w-3xl w-full bg-[#0b0b10] border border-[#1e1e28] rounded-2xl p-6 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center gap-4 mb-5">
          <div className="w-20 h-20 rounded-full overflow-hidden bg-[#111118] border border-[#1e1e28] flex-shrink-0">
            {imgError ? (
              <div className="w-full h-full flex items-center justify-center text-2xl text-text-muted">👤</div>
            ) : (
              <img
                src={api.faceAvatarUrl(cluster.sample_face_id)}
                alt=""
                className="w-full h-full object-cover"
                onError={() => setImgError(true)}
              />
            )}
          </div>
          <div className="flex-1">
            <input
              type="text"
              placeholder="name this person…"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleSave(); }}
              className="w-full bg-[#111118] border border-[#1e1e28] rounded-lg px-3 py-2 text-sm focus-ring placeholder:text-text-muted text-text"
              autoFocus
            />
            <div className="mt-1 text-[11px] text-text-muted">
              {cluster.count} face{cluster.count === 1 ? '' : 's'} across {cluster.memory_count} memor{cluster.memory_count === 1 ? 'y' : 'ies'}
            </div>
          </div>
          <button
            onClick={handleSave}
            disabled={saving}
            className="text-xs px-4 py-2 rounded-full border border-accent text-accent hover:bg-accent hover:text-bg transition-colors disabled:opacity-50"
          >
            {saving ? 'saving…' : 'save'}
          </button>
          <button onClick={onClose} className="text-text-muted hover:text-text transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        {loading ? (
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-1.5">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="aspect-square rounded-lg shimmer" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-1.5">
            {memories.map(it => (
              <MemoryTile
                key={it.id}
                item={it}
                onClick={() => {}}
                onDelete={() => {}}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
