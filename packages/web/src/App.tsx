import type { CSSProperties } from 'react';
import styles from './App.module.css';
import { Canvas } from './canvas/Canvas.js';
import { Palette } from './palette/Palette.js';
import { PromptLine } from './presets/PromptLine.js';
import { ResultsPanels } from './results/ResultsPanels.js';
import { useRegionShown } from './state/disclosure.js';
import { useStore } from './state/store.js';
import { TopBar } from './topbar/TopBar.js';
import { TrainingBar } from './trainbar/TrainingBar.js';

export const NOTICE = 'Tinker needs a wider screen. Rotate your tablet or use a laptop.';

/**
 * The grid for the regions a preset leaves visible (CONTRACTS.md section 12): the rest grow into the
 * room of the hidden ones. With everything shown this is the stylesheet's own grid, so no inline
 * style is returned.
 */
export function gridFor(on: {
  top: boolean;
  palette: boolean;
  results: boolean;
  train: boolean;
}): CSSProperties | undefined {
  if (on.top && on.palette && on.results && on.train) return undefined;
  // The canvas keeps the 55% share only when both side regions are there; otherwise it takes the rest.
  const cols = [
    on.palette ? 'minmax(140px, 15%)' : null,
    on.palette && on.results ? 'minmax(0, 55%)' : 'minmax(0, 1fr)',
    on.results ? 'minmax(0, 30%)' : null,
  ].filter((c) => c !== null);
  const mid = [on.palette ? 'palette' : null, 'canvas', on.results ? 'results' : null].filter(
    (a) => a !== null,
  );
  const rows: string[] = [];
  const areas: string[] = [];
  if (on.top) {
    rows.push('48px');
    areas.push(mid.map(() => 'top').join(' '));
  }
  rows.push('minmax(0, 1fr)');
  areas.push(mid.join(' '));
  if (on.train) {
    rows.push('56px');
    areas.push(mid.map(() => 'train').join(' '));
  }
  return {
    gridTemplateColumns: cols.join(' '),
    gridTemplateRows: rows.join(' '),
    gridTemplateAreas: areas.map((a) => `'${a}'`).join(' '),
  };
}

export function App() {
  const top = useRegionShown('top-bar');
  const results = useRegionShown('results');
  const train = useRegionShown('training-bar');
  const palette = useStore((s) => s.show?.palette?.length !== 0);
  return (
    <>
      <div className={styles.app} style={gridFor({ top, palette, results, train })}>
        {top && (
          <header className={`${styles.region} ${styles.top}`} data-region="top-bar">
            <TopBar />
          </header>
        )}
        {palette && (
          <nav className={`${styles.region} ${styles.palette}`} data-region="palette">
            <Palette />
          </nav>
        )}
        <main className={`${styles.region} ${styles.canvas}`} data-region="canvas">
          <PromptLine />
          <div className={styles.canvasBody}>
            <Canvas />
          </div>
        </main>
        {results && (
          <section className={`${styles.region} ${styles.results}`} data-region="results">
            <ResultsPanels />
          </section>
        )}
        {train && (
          <footer className={`${styles.region} ${styles.train}`} data-region="training-bar">
            <TrainingBar />
          </footer>
        )}
      </div>
      <div className={styles.notice} role="status" data-testid="narrow-notice">
        {NOTICE}
      </div>
    </>
  );
}
