import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': '/src' },
  },
  server: {
    proxy: {
      '/search': 'http://localhost:8000',
      '/stats': 'http://localhost:8000',
      '/library': 'http://localhost:8000',
      '/upload': 'http://localhost:8000',
      '/forget': 'http://localhost:8000',
      '/thumbnail': 'http://localhost:8000',
      '/media': 'http://localhost:8000',
      '/faces': 'http://localhost:8000',
      '/viz': 'http://localhost:8000',
      '/museum': 'http://localhost:8000',
      '/index-cancel': 'http://localhost:8000',
      '/reset': 'http://localhost:8000',
      '/thumbnail-cache': 'http://localhost:8000',
      '/debug': 'http://localhost:8000',
      '/thread': 'http://localhost:8000',
      '/resurface': 'http://localhost:8000',
    },
  },
})
