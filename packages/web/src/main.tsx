import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App.js';
import './theme.css';

if (import.meta.env.DEV) {
  // Dev only: lets browser checks read the store (window.__tinker.store) next to liveArrays.
  void import('./state/store.js').then((m) => {
    const w = window as unknown as { __tinker?: Record<string, unknown> };
    w.__tinker = { ...w.__tinker, store: m.useStore };
  });
}

const root = document.getElementById('root');
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
