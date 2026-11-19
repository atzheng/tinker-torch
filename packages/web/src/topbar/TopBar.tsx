import { DataControls } from './DataControls.js';
import { FileMenu } from './FileMenu.js';
import { PresetMenu } from './PresetMenu.js';
import { UndoRedo } from './UndoRedo.js';

/** Top bar: the dataset controls; the file and preset menus join them in later beads. */
export function TopBar() {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        height: '100%',
        padding: '0 8px',
        minWidth: 0,
      }}
    >
      <DataControls />
      <FileMenu />
      <PresetMenu />
      <UndoRedo />
    </div>
  );
}
