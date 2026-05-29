import { type ReactNode } from 'react';
import { Link, useLocation } from 'wouter';
import { cn } from '@/lib/utils';
import { api } from '@/lib/api';
import { useQuery } from '@tanstack/react-query';

export default function Layout({ children }: { children: ReactNode }) {
  const [loc] = useLocation();
  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: () => api.stats(),
  });

  return (
    <div className="min-h-screen">
      <header className="flex items-center justify-between px-6 py-4 border-b border-[#1e1e28]">
        <div>
          <Link href="/search" className="flex items-center gap-3">
            <img src="/favicon.ico" alt="Échos" className="w-8 h-8 rounded" />
            <h1 className="text-xl font-semibold tracking-tight cursor-pointer hover:text-accent transition-colors">
              Échos
            </h1>
          </Link>
          <p className="text-xs text-text-muted">search every memory on your phone, in one place</p>
        </div>
        <div className="flex items-center gap-2">
          <Link href="/museum">
            <span className="text-xs text-accent px-3 py-2 rounded-full bg-surface border border-border hover:border-accent cursor-pointer transition-colors">
              museum
            </span>
          </Link>
          <div className="text-xs text-text-dim px-3 py-2 rounded-full bg-surface border border-border">
            {stats ? `${stats.total} memories` : 'loading…'}
          </div>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-6 py-6">
        {children}
      </main>
    </div>
  );
}
