import { BUNDLED, REAL_META } from '@tinker/core';
import { Modal } from './Modal.js';

/** Where each bundled dataset comes from and under which licence (kept in step with docs/datasets.md). */
export function SourcesDialog({ onClose }: { onClose: () => void }) {
  return (
    <Modal title="Dataset sources" testId="sources-dialog" onClose={onClose}>
      <p className="tk-muted">
        These files are bundled with the app and cleaned as described; nothing is downloaded when
        you use them.
      </p>
      {BUNDLED.map((name) => {
        const m = REAL_META[name];
        return (
          <div key={name} data-testid={`source-${name}`}>
            <strong>{m.title}</strong> ({m.rows} rows, {m.featureNames.length} features)
            <div className="tk-muted">
              {m.source}. Licence: {m.licence}.{' '}
              <a href={m.url} target="_blank" rel="noreferrer noopener">
                Source
              </a>
            </div>
            <div className="tk-muted">{m.note}</div>
          </div>
        );
      })}
    </Modal>
  );
}
