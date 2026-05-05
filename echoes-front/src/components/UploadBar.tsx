import { useRef, useState, useCallback, type ChangeEvent } from 'react';
import { Upload, X } from 'lucide-react';
import { api } from '@/lib/api';
import { useQueryClient } from '@tanstack/react-query';

interface Props {
  onComplete?: () => void;
}

export default function UploadBar({ onComplete }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();
  const [visible, setVisible] = useState(false);
  const [done, setDone] = useState(0);
  const [total, setTotal] = useState(0);
  const [label, setLabel] = useState('');
  const [status, setStatus] = useState<'uploading' | 'cancelled' | 'done'>('uploading');
  const [counts, setCounts] = useState({ ok: 0, skipped: 0, failed: 0 });
  const [secs, setSecs] = useState('');
  const cancelRef = useRef(false);

  const handleCancel = useCallback(async () => {
    cancelRef.current = true;
    setStatus('cancelled');
    setLabel('cancelling…');
    try { await api.indexCancel(); } catch {}
    // Wait a beat so in-flight requests settle, then show done-style summary
    setTimeout(() => {
      setCounts({ ok: done, skipped: 0, failed: 0 });
      setSecs('');
      setStatus('done');
      queryClient.invalidateQueries({ queryKey: ['stats'] });
      queryClient.invalidateQueries({ queryKey: ['library'] });
      if (onComplete) onComplete();
      setTimeout(() => {
        setVisible(false);
        if (inputRef.current) inputRef.current.value = '';
      }, 6000);
    }, 300);
  }, [done, queryClient, onComplete]);

  const handleFiles = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    const t0 = performance.now();
    cancelRef.current = false;
    setTotal(files.length);
    setDone(0);
    setVisible(true);
    setStatus('uploading');
    setLabel(`indexing · CLIP + Whisper + OCR`);

    let ok = 0, skipped = 0, failed = 0;
    await api.upload(files, (d, t, name) => {
      if (cancelRef.current) return; // stop updating UI
      setDone(d);
      setTotal(t);
      setLabel(name.length > 28 ? name.slice(0, 25) + '…' : name);
    }, () => cancelRef.current);

    if (!cancelRef.current) {
      setCounts({ ok: files.length, skipped: 0, failed: 0 });
      setSecs(((performance.now() - t0) / 1000).toFixed(1));
      setStatus('done');
      queryClient.invalidateQueries({ queryKey: ['stats'] });
      queryClient.invalidateQueries({ queryKey: ['library'] });
      if (onComplete) onComplete();
      setTimeout(() => {
        setVisible(false);
        if (inputRef.current) inputRef.current.value = '';
      }, 6000);
    }
  };

  if (!visible) {
    return (
      <>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="image/*,video/*,audio/*"
          className="hidden"
          onChange={handleFiles}
        />
        <button
          onClick={() => inputRef.current?.click()}
          className="text-xs text-accent px-3 py-2 rounded-full bg-[#111118]/60 border border-[#1e1e28] hover:border-accent transition-colors"
        >
          + add memory
        </button>
      </>
    );
  }

  return (
    <div className="mb-4 px-4 py-3 rounded-lg bg-[#111118]/70 border border-[#1e1e28]">
      <div className="flex items-center justify-between text-xs text-text-dim mb-2">
        <span>
          {status === 'done'
            ? (secs ? `done in ${secs}s · ${counts.ok} indexed` : `cancelled · ${counts.ok} indexed`)
            : status === 'cancelled'
            ? 'cancelling…'
            : label}
        </span>
        <span className="text-text-muted">{done} / {total}</span>
        <button
          onClick={() => {
            if (status === 'uploading') handleCancel();
            else setVisible(false);
          }}
          className="text-text-muted hover:text-danger transition-colors ml-2"
          title={status === 'uploading' ? 'cancel indexing' : 'dismiss'}
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="h-1.5 rounded-full bg-[#1e1e28] overflow-hidden">
        <div
          className={`h-full transition-all duration-200 ${status === 'cancelled' ? 'bg-warning' : 'bg-accent'}`}
          style={{ width: `${total ? Math.round((done / total) * 100) : 0}%` }}
        />
      </div>
    </div>
  );
}
