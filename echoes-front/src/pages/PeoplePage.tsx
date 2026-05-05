import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, type FaceCluster } from '@/lib/api';
import PeopleGrid from '@/components/PeopleGrid';
import PersonPanel from '@/components/PersonPanel';
import { Loader2 } from 'lucide-react';

export default function PeoplePage() {
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<FaceCluster | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ['faceClusters'],
    queryFn: () => api.faceClusters(),
  });

  const deleteMut = useMutation({
    mutationFn: (clusterId: string) => api.faceDeleteCluster(clusterId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['faceClusters'] }),
    onError: () => alert('delete failed'),
  });

  const scanMut = useMutation({
    mutationFn: () => api.faceScan(),
    onSuccess: (data) => {
      alert(`done: ${data.faces_found} faces in ${data.photos_processed} photos`);
      queryClient.invalidateQueries({ queryKey: ['faceClusters'] });
    },
    onError: () => alert('scan failed'),
  });

  const consolidateMut = useMutation({
    mutationFn: () => api.faceConsolidate(),
    onSuccess: (data) => {
      alert(`consolidated · ${data.total_faces_merged} faces merged across ${Object.keys(data.per_label || {}).length} duplicate labels`);
      queryClient.invalidateQueries({ queryKey: ['faceClusters'] });
    },
    onError: () => alert('consolidate failed'),
  });

  const clusters = data?.clusters || [];

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div className="text-xs text-text-dim">
          {isLoading ? 'loading clusters…' :
           error ? 'failed to load clusters' :
           `${clusters.length} cluster${clusters.length === 1 ? '' : 's'} · ${clusters.reduce((a, c) => a + c.count, 0)} faces detected`}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => consolidateMut.mutate()}
            disabled={consolidateMut.isPending}
            className="text-xs text-accent px-3 py-1.5 rounded-full bg-[#111118]/60 border border-[#1e1e28] hover:border-accent disabled:opacity-50 transition-colors"
          >
            {consolidateMut.isPending ? (
              <span className="flex items-center gap-1"><Loader2 className="w-3 h-3 animate-spin" /> merging…</span>
            ) : 'consolidate labels'}
          </button>
          <button
            onClick={() => {
              if (!confirm('rescan faces across all photos? this wipes existing clusters and rebuilds. labels you set will be lost.')) return;
              scanMut.mutate();
            }}
            disabled={scanMut.isPending}
            className="text-xs text-accent px-3 py-1.5 rounded-full bg-[#111118]/60 border border-[#1e1e28] hover:border-accent disabled:opacity-50 transition-colors"
          >
            {scanMut.isPending ? (
              <span className="flex items-center gap-1"><Loader2 className="w-3 h-3 animate-spin" /> scanning…</span>
            ) : 'rescan faces'}
          </button>
        </div>
      </div>

      <PeopleGrid
        clusters={clusters}
        loading={isLoading}
        onSelect={setSelected}
        onDelete={(cl) => {
          const name = cl.label || 'unnamed person';
          if (confirm(`delete ${name}?\n\nremoves ${cl.count} face(s) from the index. this cannot be undone.`)) {
            deleteMut.mutate(cl.cluster_id);
          }
        }}
      />

      {selected && (
        <PersonPanel
          cluster={selected}
          onClose={() => setSelected(null)}
          onSaved={() => {
            queryClient.invalidateQueries({ queryKey: ['faceClusters'] });
            setSelected(null);
          }}
        />
      )}
    </div>
  );
}
