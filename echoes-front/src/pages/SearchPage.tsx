import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, type MemoryItem, type FaceCluster } from '@/lib/api';
import SearchBar from '@/components/SearchBar';
import FilterChips from '@/components/FilterChips';
import MemoryTile from '@/components/MemoryTile';
import ImageViewer from '@/components/ImageViewer';
import UploadBar from '@/components/UploadBar';
import PeopleGrid from '@/components/PeopleGrid';
import PersonPanel from '@/components/PersonPanel';
import { Loader2, Users } from 'lucide-react';

type FilterType = 'all' | 'photo' | 'video' | 'voice_memo' | 'screenshot';
type Mode = 'search' | 'library' | 'people';

const SUGGESTIONS = [
  { q: 'sunset', icon: '🌅' },
  { q: 'screenshot', icon: '🖼️' },
  { q: 'birthday', icon: '🎂' },
  { q: 'error', icon: '🐛' },
  { q: 'voice memo', icon: '🎙️' },
  { q: 'family', icon: '👨‍👩‍👧' },
];

export default function SearchPage() {
  const queryClient = useQueryClient();
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [filter, setFilter] = useState<FilterType>('all');
  const [mode, setMode] = useState<Mode>('search');
  const [viewerItem, setViewerItem] = useState<MemoryItem | null>(null);
  const [libraryBefore, setLibraryBefore] = useState<number | null>(null);
  const [allLibraryItems, setAllLibraryItems] = useState<MemoryItem[]>([]);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query), 200);
    return () => clearTimeout(t);
  }, [query]);

  useEffect(() => {
    if (!debouncedQuery.trim()) return;
    setMode('search');
  }, [debouncedQuery]);

  const { data: searchData, isFetching: searchLoading } = useQuery({
    queryKey: ['search', debouncedQuery],
    queryFn: () => api.search(debouncedQuery, 12),
    enabled: debouncedQuery.trim().length > 0,
  });

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: () => api.stats(),
    staleTime: 15_000,
  });

  const { data: libraryPage, isFetching: libraryLoading } = useQuery({
    queryKey: ['library', libraryBefore],
    queryFn: () => api.library(60, libraryBefore),
    enabled: mode === 'library',
  });

  useEffect(() => {
    if (libraryPage?.items) {
      setAllLibraryItems(prev => {
        const existing = new Set(prev.map(i => i.id));
        const fresh = libraryPage.items.filter(i => !existing.has(i.id));
        return libraryBefore ? [...prev, ...fresh] : libraryPage.items;
      });
    }
  }, [libraryPage, libraryBefore]);

  const enterLibrary = useCallback(() => {
    setMode('library'); setAllLibraryItems([]); setLibraryBefore(null); setQuery('');
  }, []);
  const exitLibrary = useCallback(() => { setMode('search'); setAllLibraryItems([]); }, []);

  const results = searchData?.results || [];
  const filtered = filter === 'all' ? results : results.filter(r => r.type === filter);
  const matchedLabels = searchData?.matched_labels || [];
  const showHero = !debouncedQuery.trim() && mode === 'search';
  const isEmpty = debouncedQuery.trim() && !searchLoading && filtered.length === 0;

  const forgetMut = useMutation({
    mutationFn: (id: string) => api.forget(id, true),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['search'] });
      queryClient.invalidateQueries({ queryKey: ['library'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
    },
  });

  const handleForget = async (id: string, type: string) => {
    const label = ({ photo: 'photo', video: 'video', voice_memo: 'voice memo', screenshot: 'screenshot' } as any)[type] || 'memory';
    if (!confirm(`forget this ${label}?\n\nremoves from index and deletes the file from disk. cannot be undone.`)) return;
    forgetMut.mutate(id);
    setAllLibraryItems(prev => prev.filter(x => x.id !== id));
  };

  return (
    <div>
      <UploadBar onComplete={() => {
        if (debouncedQuery.trim()) queryClient.invalidateQueries({ queryKey: ['search', debouncedQuery] });
        if (mode === 'library') { setAllLibraryItems([]); setLibraryBefore(null); }
      }} />

      <SearchBar value={query} onChange={(v) => { setQuery(v); if (mode === 'library') exitLibrary(); }} />

      <FilterChips
        active={filter} onChange={setFilter} mode={mode}
        onPeopleToggle={() => { if (mode === 'people') { setMode('search'); return; } setMode('people'); setQuery(''); }}
        onLibraryToggle={() => { if (mode === 'library') { exitLibrary(); return; } enterLibrary(); }}
      />

      {/* Stats bar — compact inline */}
      {showHero && stats && (
        <div className="flex items-center justify-center gap-4 mb-4 text-xs text-text-muted">
          <span className="text-text">{stats.total} <span className="text-text-muted">memories</span></span>
          <span className="text-text-dim">·</span>
          <span>{stats.by_type?.photo ?? 0} photos</span>
          <span className="text-text-dim">·</span>
          <span>{stats.by_type?.video ?? 0} videos</span>
          <span className="text-text-dim">·</span>
          <span>{stats.by_type?.voice_memo ?? 0} voice</span>
          <span className="text-text-dim">·</span>
          <span>{stats.by_type?.screenshot ?? 0} screenshots</span>
        </div>
      )}

      {/* Suggested search chips */}
      {showHero && (
        <div className="flex flex-wrap justify-center gap-2 mb-6">
          {SUGGESTIONS.map(s => (
            <button
              key={s.q}
              onClick={() => setQuery(s.q)}
              className="glass px-4 py-2 rounded-full text-xs text-text-dim hover:text-text hover:border-accent transition-all cursor-pointer"
            >
              {s.icon} {s.q}
            </button>
          ))}
        </div>
      )}

      {/* Face filter indicator */}
      {matchedLabels.length > 0 && (
        <div className="glass mb-4 px-4 py-2 rounded-full text-xs text-accent flex items-center gap-2">
          <Users className="w-3 h-3" />
          <span>filtered by {matchedLabels.join(' + ')} · {results.length} hits</span>
        </div>
      )}

      {/* Hero — simple gradient */}
      {showHero && (
        <div className="relative mb-6 rounded-2xl overflow-hidden border border-[#1e1e28] bg-gradient-to-b from-[#0d0d18] to-[#06060a] h-72 flex items-center justify-center">
          <div className="text-center">
            <div className="text-4xl font-semibold text-text mb-2">Échos</div>
            <p className="text-text-muted text-sm px-4">
              search every memory on your phone, in one place
            </p>
            <p className="text-[10px] uppercase tracking-[0.18em] text-text-muted mt-4">
              nothing leaves your device
            </p>
          </div>
        </div>
      )}

      {/* Search results */}
      {mode === 'search' && debouncedQuery.trim() && (
        <>
          {searchLoading ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="shimmer rounded-xl h-56" />
              ))}
            </div>
          ) : isEmpty ? (
            <div className="text-center py-24 text-text-muted">
              <div className="text-3xl mb-3">·</div>
              <div className="text-sm">no memories matched. try a different word, or a name, or a feeling.</div>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5">
              {filtered.map((r, i) => (
                <div key={r.id} className="animate-tile-in" style={{ animationDelay: `${i * 60}ms` }}>
                  <MemoryTile
                    item={r}
                    onClick={() => setViewerItem(r)}
                    onDelete={() => handleForget(r.id, r.type)}
                  />
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Library */}
      {mode === 'library' && (
        <div>
          {libraryLoading && allLibraryItems.length === 0 ? (
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-1.5">
              {Array.from({ length: 12 }).map((_, i) => <div key={i} className="aspect-square rounded-lg shimmer" />)}
            </div>
          ) : allLibraryItems.length === 0 ? (
            <div className="py-24 text-center text-text-muted text-sm">no memories yet. drop files via "+ add memory".</div>
          ) : (
            <>
              <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-1.5">
                {allLibraryItems.map((it, i) => (
                  <div key={it.id} className="animate-tile-in" style={{ animationDelay: `${i * 40}ms` }}>
                    <MemoryTile item={it} onClick={() => setViewerItem(it)} onDelete={() => handleForget(it.id, it.type)} />
                  </div>
                ))}
              </div>
              {libraryPage?.next_before && (
                <div className="text-center mt-6">
                  <button onClick={() => libraryPage.next_before && setLibraryBefore(libraryPage.next_before)} disabled={libraryLoading}
                    className="glass px-4 py-2 rounded-full text-xs hover:border-accent transition-colors disabled:opacity-50">
                    {libraryLoading ? 'loading…' : 'load older'}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {mode === 'people' && <InlinePeople />}

      {viewerItem && <ImageViewer item={viewerItem} query={debouncedQuery} onClose={() => setViewerItem(null)} />}
    </div>
  );
}

// Inline people view — same as PeoplePage but embedded in search page
function InlinePeople() {
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
    onSuccess: (d) => {
      alert(`done: ${d.faces_found} faces in ${d.photos_processed} photos`);
      queryClient.invalidateQueries({ queryKey: ['faceClusters'] });
    },
    onError: () => alert('scan failed'),
  });

  const consolidateMut = useMutation({
    mutationFn: () => api.faceConsolidate(),
    onSuccess: (d) => {
      alert(`consolidated · ${d.total_faces_merged} faces merged across ${Object.keys(d.per_label || {}).length} duplicate labels`);
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
              if (!confirm('rescan faces across all photos? this wipes existing clusters and rebuilds.')) return;
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
