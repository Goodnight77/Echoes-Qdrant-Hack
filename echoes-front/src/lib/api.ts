// API client for Échos backend. Mirrors every endpoint currently in main.py.
// All fetch calls are thin wrappers — no state management, no caching.

const API_BASE = '';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers ?? {});
  if (!(init?.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as any).detail ?? res.statusText);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ── types ──────────────────────────────────────────────────────────────

export interface MemoryItem {
  id: string;
  type: 'photo' | 'video' | 'voice_memo' | 'screenshot';
  timestamp: number | null;
  transcript?: string;
  ocr_text?: string;
  best_moment_seconds?: number;
  matched_via: string[];
}

export interface SearchResult {
  query: string;
  results: MemoryItem[];
  matched_labels: string[];
}

export interface Stats {
  total: number;
  by_type: Record<string, number>;
  last_indexed: number;
}

export interface LibraryPage {
  items: MemoryItem[];
  next_before: number | null;
  total: number | null;
}

export interface FaceCluster {
  cluster_id: string;
  label: string | null;
  sample_face_id: string;
  count: number;
  memory_count: number;
}

export interface FaceClustersResponse {
  clusters: FaceCluster[];
}

export interface ClusterMemoriesResponse {
  cluster_id: string;
  items: MemoryItem[];
}

export interface LabelResponse {
  ok: boolean;
  cluster_id: string;
  canonical_cluster_id: string;
  label: string;
  faces_updated: number;
  merged_faces: number;
}

export interface FaceScanResponse {
  faces_found: number;
  photos_processed: number;
}

export interface ConsolidateResponse {
  total_faces_merged: number;
  per_label: Record<string, number>;
}

export interface UploadResult {
  uploaded: Array<{
    name: string;
    saved_as?: string;
    point_id?: string;
    skipped?: string;
    error?: string;
  }>;
}

export interface VizProjection {
  items: VizItem[];
}

export interface VizItem {
  id: string;
  x: number;
  y: number;
  z: number;
  type: string;
  transcript?: string;
  ocr_text?: string;
  timestamp?: number;
  neighbor_ids: string[];
}

export interface ThreadResult {
  query: string;
  thread: MemoryItem[];
  matched_labels: string[];
}

export interface ResurfaceResult {
  seed: { id: string; payload: Record<string, any> } | null;
  results: any[];
}

export interface ForgetBody {
  point_id: string;
  delete_file: boolean;
}

// ── endpoints ──────────────────────────────────────────────────────────

export const api = {
  // Search
  search(query: string, topK = 12): Promise<SearchResult> {
    return request('/search', {
      method: 'POST',
      body: JSON.stringify({ query, top_k: topK }),
    });
  },

  // Thread (chronological search)
  thread(query: string): Promise<ThreadResult> {
    return request('/thread', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  },

  // Stats
  stats(): Promise<Stats> {
    return request('/stats');
  },

  // Library (newest-first paginated)
  library(limit = 60, before?: number | null): Promise<LibraryPage> {
    const url = before ? `/library?limit=${limit}&before=${before}` : `/library?limit=${limit}`;
    return request(url);
  },

  // Thumbnail
  thumbnail(pointId: string): Promise<{ point_id: string; data_url: string }> {
    return request(`/thumbnail/${pointId}`);
  },

  // Media URL (direct, no JSON)
  mediaUrl(pointId: string): string {
    return `/media/${pointId}`;
  },

  // Upload
  async upload(
    files: File[],
    onProgress?: (done: number, total: number, label: string) => void,
    isCancelled?: () => boolean,
  ): Promise<UploadResult> {
    const CONCURRENCY = 3;
    const total = files.length;
    let done = 0;
    const queue = [...files];
    const results: UploadResult = { uploaded: [] };

    const worker = async () => {
      while (queue.length) {
        if (isCancelled?.()) break;
        const f = queue.shift();
        if (!f) break;
        const fd = new FormData();
        fd.append('files', f, f.name);
        try {
          const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: fd });
          const data = await res.json() as UploadResult;
          results.uploaded.push(...data.uploaded);
        } catch {
          results.uploaded.push({ name: f.name, error: 'network error' });
        }
        done++;
        onProgress?.(done, total, f.name);
      }
    };

    await Promise.all(Array.from({ length: Math.min(CONCURRENCY, total) }, worker));
    return results;
  },

  // Forget
  forget(pointId: string, deleteFile = true): Promise<{ ok: boolean }> {
    return request('/forget', {
      method: 'POST',
      body: JSON.stringify({ point_id: pointId, delete_file: deleteFile }),
    });
  },

  // Face clusters
  faceClusters(): Promise<FaceClustersResponse> {
    return request('/faces/clusters');
  },

  // Delete a face cluster
  faceDeleteCluster(clusterId: string): Promise<{ ok: boolean; cluster_id: string; faces_removed: number }> {
    return request(`/faces/cluster/${encodeURIComponent(clusterId)}`, { method: 'DELETE' });
  },

  // Memories for a cluster
  faceClusterMemories(clusterId: string): Promise<ClusterMemoriesResponse> {
    return request(`/faces/by-cluster/${encodeURIComponent(clusterId)}`);
  },

  // Memories for a label
  faceLabelMemories(label: string): Promise<{ label: string; items: MemoryItem[] }> {
    return request(`/faces/by-label/${encodeURIComponent(label)}`);
  },

  // Label a cluster
  faceLabel(clusterId: string, label: string): Promise<any> {
    return request('/faces/label', {
      method: 'POST',
      body: JSON.stringify({ cluster_id: clusterId, label }),
    });
  },

  // Consolidate labels
  faceConsolidate(): Promise<ConsolidateResponse> {
    return request('/faces/consolidate', { method: 'POST' });
  },

  // Rescan all faces
  faceScan(): Promise<FaceScanResponse> {
    return request('/faces/scan', { method: 'POST' });
  },

  // Face avatar URL
  faceAvatarUrl(faceId: string): string {
    return `/faces/avatar/${faceId}`;
  },

  // Viz projection (3D galaxy)
  vizProjection(space = 'visual', neighbors = 3): Promise<VizProjection> {
    return request(`/viz/projection?space=${space}&neighbors=${neighbors}`);
  },

  // Resurface
  resurface(): Promise<ResurfaceResult> {
    return request('/resurface', { method: 'POST' });
  },

  // Index cancel
  indexCancel(): Promise<{ ok: boolean }> {
    return request('/index-cancel', { method: 'POST' });
  },

  // Reset
  reset(): Promise<{ ok: boolean }> {
    return request('/reset', { method: 'DELETE' });
  },

  // Thumbnail cache clear
  clearThumbnailCache(): Promise<{ ok: boolean }> {
    return request('/thumbnail-cache', { method: 'DELETE' });
  },
};
