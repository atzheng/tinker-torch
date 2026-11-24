import { LECTURES, stagePresetName } from '../presets/lectures.js';
import { presetTitle } from '../presets/index.js';
import { formatRoute } from './lectureRoute.js';
import styles from './Landing.module.css';

/** The lecture list at `#/lectures`: labels and links only. Each stage link is a plain anchor. */
export function Landing() {
  return (
    <main className={styles.page} data-testid="landing">
      <div className={styles.inner}>
        <header className={styles.head}>
          <h1 className={styles.h1}>Lectures</h1>
          <a className={styles.free} href="#/" data-testid="landing-free">
            Free canvas
          </a>
        </header>
        {LECTURES.map((l) => (
          <section key={l.slug} className={styles.lecture} aria-labelledby={`lec-${l.slug}`}>
            <h2 id={`lec-${l.slug}`} className={styles.h2}>
              {l.title}
            </h2>
            <ol className={styles.stages}>
              {l.stages.map((k) => (
                <li key={k}>
                  <a
                    className={styles.stage}
                    href={formatRoute({ kind: 'stage', lecture: l.slug, stage: k })}
                    data-testid={`landing-${l.slug}-stage${k}`}
                  >
                    <span className={styles.num}>Stage {k}</span>
                    <span className={styles.sub}>
                      {presetTitle(stagePresetName(l.slug, k)).replace(/^[^:]*:\s*/, '')}
                    </span>
                  </a>
                </li>
              ))}
            </ol>
          </section>
        ))}
      </div>
    </main>
  );
}
