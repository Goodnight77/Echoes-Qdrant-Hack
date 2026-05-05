import { useState } from 'react';
import type { FaceCluster } from '@/lib/api';
import { api } from '@/lib/api';

interface Props {
  clusters: FaceCluster[];
  loading: boolean;
  onSelect: (cluster: FaceCluster) => void;
  onDelete?: (cluster: FaceCluster) => void;
}

export default function PeopleGrid({ clusters, loading, onSelect, onDelete }: Props) {
  const [imgErrors, setImgErrors] = useState<Set<string>>(new Set());

  if (loading) {
    return (
      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="flex flex-col items-center gap-2">
            <div className="w-24 h-24 rounded-full shimmer" />
            <div className="w-16 h-3 rounded shimmer" />
          </div>
        ))}
      </div>
    );
  }

  if (!clusters.length) {
    return (
      <div className="py-16 text-center text-text-muted text-sm">
        no faces found yet. drop some photos with people, then click "rescan faces" if needed.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-4">
      {clusters.map(cl => (
        <div
          key={cl.cluster_id}
          onClick={() => onSelect(cl)}
          className="flex flex-col items-center cursor-pointer group relative"
        >
          {onDelete && (
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(cl); }}
              className="absolute top-0 right-0 w-5 h-5 rounded-full bg-[#1e1e28] hover:bg-danger text-text-muted hover:text-white text-[10px] flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity z-10"
              title="delete this person"
            >
              ×
            </button>
          )}
          <div className="w-24 h-24 rounded-full overflow-hidden border-2 border-[#1e1e28] group-hover:border-accent transition-colors bg-[#111118]">
            {imgErrors.has(cl.cluster_id) ? (
              <div className="w-full h-full flex items-center justify-center text-2xl text-text-muted">👤</div>
            ) : (
              <img
                src={api.faceAvatarUrl(cl.sample_face_id)}
                alt=""
                className="w-full h-full object-cover"
                onError={() => setImgErrors(prev => new Set(prev).add(cl.cluster_id))}
              />
            )}
          </div>
          <div className={`mt-2 text-sm font-medium text-center ${cl.label ? 'text-text' : 'text-text-muted italic'}`}>
            {cl.label || 'unnamed'}
          </div>
          <div className="text-[11px] text-text-muted">
            {cl.memory_count} memor{cl.memory_count === 1 ? 'y' : 'ies'}
          </div>
        </div>
      ))}
    </div>
  );
}
