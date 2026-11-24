import { useStore } from '../state/store.js';
import { presetTitle } from '../presets/index.js';
import { LECTURES } from '../presets/lectures.js';
import { formatRoute, routePreset, stageNeighbours } from '../routes/lectureRoute.js';
import { useRoute } from '../routes/useRoute.js';
import styles from './LectureNav.module.css';

/**
 * Breadcrumb and previous/next stage links in the top bar, only while a lecture stage is open:
 * the URL is a stage route and the loaded preset is that stage (so it disappears once another file
 * or preset replaces it). Plain anchors: keyboard accessible, and back/forward work.
 */
export function LectureNav() {
  const route = useRoute();
  const title = useStore((s) => s.preset?.title ?? null);
  const name = routePreset(route);
  if (route.kind !== 'stage' || !name || title !== presetTitle(name)) return null;
  const { prev, next } = stageNeighbours(route);
  const lecture = LECTURES.find((l) => l.slug === route.lecture);
  return (
    <nav className={styles.nav} aria-label="Lecture" data-testid="lecture-nav">
      <span className={styles.here} aria-current="page">
        {lecture?.title}, stage {route.stage}
      </span>
      {prev ? (
        <a className={styles.link} href={formatRoute(prev)} data-testid="lecture-nav-prev">
          Previous
        </a>
      ) : (
        <span className={styles.off}>Previous</span>
      )}
      {next ? (
        <a className={styles.link} href={formatRoute(next)} data-testid="lecture-nav-next">
          Next
        </a>
      ) : (
        <span className={styles.off}>Next</span>
      )}
    </nav>
  );
}
