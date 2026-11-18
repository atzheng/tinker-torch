import { DataControls } from './DataControls.js';

/** Top bar: the dataset controls; the file and preset menus join them in later beads. */
export function TopBar() {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        height: '100%',
        padding: '0 8px',
        overflow: 'hidden',
      }}
    >
      <DataControls />
    </div>
  );
}
