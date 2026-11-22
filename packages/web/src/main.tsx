import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App.js';
import { loadBundled } from './topbar/PresetMenu.js';
import { loadParsed } from './topbar/FileMenu.js';
import { useStore } from './state/store.js';
import { decodeShare, shareFromHash } from './share/shareLink.js';
import './theme.css';
import 'katex/dist/katex.min.css';
import './tex/tex.css';

if (import.meta.env.DEV) {
  // Dev only: lets browser checks read the store (window.__tinker.store) next to liveArrays.
  void import('./state/store.js').then((m) => {
    const w = window as unknown as { __tinker?: Record<string, unknown> };
    w.__tinker = { ...w.__tinker, store: m.useStore };
  });
}

// `?preset=<name>` loads a bundled preset on start (an unknown name is ignored).
const presetName = new URLSearchParams(window.location.search).get('preset');
if (presetName) loadBundled(presetName);

// `#x=1.<payload>` loads a shared exercise (the file's object, compressed; see CONTRACTS section 12).
const shared = shareFromHash(window.location.hash);
if (shared) {
  void decodeShare(shared).then((r) => {
    const fail = (message: string) => useStore.setState({ error: message });
    if (!r.ok) return fail(r.error);
    const loaded = loadParsed(r.file);
    if (!loaded.ok && loaded.message) fail(loaded.message);
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
