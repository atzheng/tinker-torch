import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App.js';
import { loadBundled } from './topbar/PresetMenu.js';
import { loadParsed } from './topbar/FileMenu.js';
import { hashIsStage, startRouting } from './routes/useRoute.js';
import { useStore } from './state/store.js';
import { startFromUrl } from './share/shareLink.js';
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

// `#x=1.<payload>` loads a shared exercise (CONTRACTS section 12) and wins over `?preset=<name>`,
// which loads a bundled preset (an unknown name is ignored).
// A stage route wins over `?preset=`, so the preset is skipped then (one load, one undo step).
void startFromUrl(
  hashIsStage(window.location.hash) ? '' : window.location.search,
  window.location.hash,
  loadParsed,
  loadBundled,
  (message) => useStore.setState({ error: message }),
);

// `#/<lecture>/stage<k>` opens a lecture stage and wins over `?preset=`; `#/lectures` is the landing page.
startRouting();

const root = document.getElementById('root');
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
