import styles from './App.module.css';
import { Canvas } from './canvas/Canvas.js';
import { Palette } from './palette/Palette.js';
import { PromptLine } from './presets/PromptLine.js';
import { ResultsPanels } from './results/ResultsPanels.js';
import { TopBar } from './topbar/TopBar.js';
import { TrainingBar } from './trainbar/TrainingBar.js';

export const NOTICE = 'Tinker needs a wider screen. Rotate your tablet or use a laptop.';

export function App() {
  return (
    <>
      <div className={styles.app}>
        <header className={`${styles.region} ${styles.top}`} data-region="top-bar">
          <TopBar />
        </header>
        <nav className={`${styles.region} ${styles.palette}`} data-region="palette">
          <Palette />
        </nav>
        <main className={`${styles.region} ${styles.canvas}`} data-region="canvas">
          <PromptLine />
          <div className={styles.canvasBody}>
            <Canvas />
          </div>
        </main>
        <section className={`${styles.region} ${styles.results}`} data-region="results">
          <ResultsPanels />
        </section>
        <footer className={`${styles.region} ${styles.train}`} data-region="training-bar">
          <TrainingBar />
        </footer>
      </div>
      <div className={styles.notice} role="status" data-testid="narrow-notice">
        {NOTICE}
      </div>
    </>
  );
}
