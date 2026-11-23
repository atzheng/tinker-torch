import { useRef, type CSSProperties } from 'react';
import styles from './App.module.css';
import { Canvas } from './canvas/Canvas.js';
import { Palette } from './palette/Palette.js';
import { PromptLine } from './presets/PromptLine.js';
import { SuccessTick } from './presets/SuccessTick.js';
import { InstructionsPanel } from './presets/InstructionsPanel.js';
import { ResultsPanels } from './results/ResultsPanels.js';
import { useRegionShown } from './state/disclosure.js';
import { useLayoutMode } from './layout/useLayoutMode.js';
import { useStore } from './state/store.js';
import { TabBar } from './tabs/TabBar.js';
import { TopBar } from './topbar/TopBar.js';
import type { LayoutMode } from './layout/layoutMode.js';
import { TrainingBar } from './trainbar/TrainingBar.js';
import { useEngine } from './engine.js';
import { ResizeHandle } from './layout/ResizeHandle.js';
import {
  REGION_LIMITS,
  clampSize,
  dragSize,
  pctCss,
  percent,
  stepSize,
  type RegionSizes,
} from './layout/regionSizes.js';

export const NOTICE = 'Tinker needs a larger screen. Use a tablet or a laptop.';

/**
 * The grid for the regions a preset leaves visible (CONTRACTS.md section 12): the rest grow into the
 * room of the hidden ones. With everything shown this is the stylesheet's own grid, so no inline
 * style is returned.
 */
export function gridFor(
  on: {
    top: boolean;
    palette: boolean;
    results: boolean;
    train: boolean;
  },
  mode: LayoutMode = 'landscape',
  sizes: Pick<RegionSizes, 'resultsWidth' | 'resultsBand'> = {
    resultsWidth: null,
    resultsBand: null,
  },
): CSSProperties | undefined {
  const custom = mode === 'portrait' ? sizes.resultsBand !== null : sizes.resultsWidth !== null;
  if (on.top && on.palette && on.results && on.train && !custom) return undefined;
  if (mode === 'portrait') {
    // One column, top to bottom: top bar, palette strip, canvas, results band, training bar.
    const rows: string[] = [];
    const areas: string[] = [];
    const add = (area: string, size: string) => {
      rows.push(size);
      areas.push(`'${area}'`);
    };
    if (on.top) add('top', 'minmax(48px, auto)');
    if (on.palette) add('palette', 'auto');
    add('canvas', 'minmax(0, 1fr)');
    if (on.results) {
      add(
        'results',
        sizes.resultsBand === null
          ? PORTRAIT_RESULTS
          : `clamp(120px, ${pctCss(sizes.resultsBand)}, 50%)`,
      );
    }
    if (on.train) add('train', 'minmax(56px, auto)');
    return {
      gridTemplateColumns: 'minmax(0, 1fr)',
      gridTemplateRows: rows.join(' '),
      gridTemplateAreas: areas.join(' '),
    };
  }
  // The canvas keeps the 55% share only when both side regions are there; otherwise it takes the rest.
  const cols = [
    on.palette ? 'minmax(140px, 15%)' : null,
    on.palette && on.results && sizes.resultsWidth === null ? 'minmax(0, 55%)' : 'minmax(0, 1fr)',
    on.results
      ? `minmax(0, ${sizes.resultsWidth === null ? '30%' : pctCss(sizes.resultsWidth)})`
      : null,
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

/** The portrait results band: about a third of the window, never under 150 px. */
export const PORTRAIT_RESULTS = 'clamp(150px, 32%, 380px)';

export function App() {
  useEngine(); // the engine loads even when a preset hides the training bar (its other caller)
  const mode = useLayoutMode();
  const top = useRegionShown('top-bar');
  const results = useRegionShown('results');
  const train = useRegionShown('training-bar');
  const palette = useStore((s) => s.show?.palette?.length !== 0);
  const sizes = useStore((s) => s.regionSizes);
  const setSize = useStore((s) => s.setRegionSize);
  const appRef = useRef<HTMLDivElement>(null);
  // The results region's size is a drag handle's value: the width of the column (landscape) or the
  // height of the band (portrait), as a fraction of the app grid.
  const portrait = mode === 'portrait';
  const key = portrait ? 'resultsBand' : 'resultsWidth';
  const extent = (): number => {
    const r = appRef.current?.getBoundingClientRect();
    const e = portrait ? r?.height : r?.width;
    return e && e > 0 ? e : portrait ? window.innerHeight : window.innerWidth;
  };
  const current = (): number => {
    const set = sizes[key];
    if (set !== null) return set;
    const r = appRef.current?.querySelector('[data-region="results"]')?.getBoundingClientRect();
    const px = portrait ? r?.height : r?.width;
    return px && px > 0 ? clampSize(key, px / extent()) : REGION_LIMITS[key].def;
  };
  return (
    <>
      <div
        className={styles.app}
        data-layout={mode}
        ref={appRef}
        style={gridFor({ top, palette, results, train }, mode, sizes)}
      >
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
          <TabBar />
          <PromptLine />
          <InstructionsPanel />
          <SuccessTick />
          <div className={styles.canvasBody} id="tab-panel" role="tabpanel">
            <Canvas />
          </div>
        </main>
        {results && (
          <section className={`${styles.region} ${styles.results}`} data-region="results">
            <ResultsPanels />
          </section>
        )}
        {results && (
          <ResizeHandle
            axis={portrait ? 'y' : 'x'}
            strip="centered"
            className={portrait ? styles.resizeBand : styles.resizeWidth}
            testId="resize-results"
            label={portrait ? 'Resize the results band' : 'Resize the results column'}
            now={percent(sizes[key] ?? REGION_LIMITS[key].def)}
            min={percent(REGION_LIMITS[key].min)}
            max={percent(REGION_LIMITS[key].max)}
            begin={() => {
              const start = current();
              const ext = extent();
              return (d) => setSize(key, dragSize(key, start, d, ext, -1));
            }}
            onStep={(dir, big) => setSize(key, stepSize(key, current(), dir, big, -1))}
          />
        )}
        {train && (
          <footer className={`${styles.region} ${styles.train}`} data-region="training-bar">
            <TrainingBar />
          </footer>
        )}
      </div>
      <div
        className={styles.notice}
        data-active={mode === 'notice'}
        role="status"
        data-testid="narrow-notice"
      >
        {NOTICE}
      </div>
    </>
  );
}
