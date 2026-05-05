import { cn } from '@/lib/utils';
import { Users, Grid3X3 } from 'lucide-react';

type FilterType = 'all' | 'photo' | 'video' | 'voice_memo' | 'screenshot';

interface Props {
  active: FilterType;
  onChange: (f: FilterType) => void;
  mode: 'search' | 'library' | 'people';
  onPeopleToggle: () => void;
  onLibraryToggle: () => void;
}

const FILTERS: { value: FilterType; label: string }[] = [
  { value: 'all', label: 'all' },
  { value: 'photo', label: 'photos' },
  { value: 'video', label: 'videos' },
  { value: 'voice_memo', label: 'voice' },
  { value: 'screenshot', label: 'screenshots' },
];

export default function FilterChips({ active, onChange, mode, onPeopleToggle, onLibraryToggle }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2 mb-8 text-xs">
      {FILTERS.map(f => (
        <button
          key={f.value}
          onClick={() => onChange(f.value)}
          className={cn(
            'px-3 py-1.5 rounded-full border border-[#1e1e28] transition-colors',
            active === f.value
              ? 'bg-accent text-bg border-accent'
              : 'text-text-dim hover:border-accent'
          )}
        >
          {f.label}
        </button>
      ))}
      <span className="flex-1" />
      <button
        onClick={onPeopleToggle}
        className={cn(
          'px-3 py-1.5 rounded-full border border-[#1e1e28] hover:border-accent flex items-center gap-1.5 transition-colors',
          mode === 'people' && 'bg-accent text-bg border-accent'
        )}
      >
        <Users className="w-3.5 h-3.5" />
        <span>{mode === 'people' ? 'close' : 'people'}</span>
      </button>
      <button
        onClick={onLibraryToggle}
        className={cn(
          'px-3 py-1.5 rounded-full border border-[#1e1e28] hover:border-accent flex items-center gap-1.5 transition-colors',
          mode === 'library' && 'bg-accent text-bg border-accent'
        )}
      >
        <Grid3X3 className="w-3.5 h-3.5" />
        <span>{mode === 'library' ? 'close' : 'browse all'}</span>
      </button>
    </div>
  );
}
