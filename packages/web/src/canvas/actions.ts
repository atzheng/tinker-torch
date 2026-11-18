import { useEffect } from 'react';
import { deleteEdge, deleteNode } from '@tinker/core';
import { lockedIds } from '../state/locks.js';
import { useStore } from '../state/store.js';
import { edgeIndexOf } from './pure.js';

/** Delete the selected piece or wire through the undo history; a refused edit shows its toast. */
export function deleteSelection(): void {
  const { graph, selection, apply, select } = useStore.getState();
  if (!selection) return;
  if (selection.kind === 'node') {
    if (apply(deleteNode(graph, selection.id, { locked: lockedIds() })).ok) select(null);
    return;
  }
  const i = edgeIndexOf(graph, selection.from, selection.to);
  if (i < 0) return;
  if (apply(deleteEdge(graph, i, { locked: lockedIds() })).ok) select(null);
}

/** Whether keyboard shortcuts should leave this event target to the browser (typing in a field). */
function isEditable(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  return t.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName);
}

/** Delete or Backspace delete the selection; Ctrl/Cmd+Z undoes, with Shift (or Ctrl+Y) redoes. */
export function useEditShortcuts(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isEditable(e.target)) return;
      const s = useStore.getState();
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) s.redo();
        else s.undo();
      } else if (mod && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        s.redo();
      } else if (!mod && (e.key === 'Delete' || e.key === 'Backspace')) {
        if (!s.selection) return;
        e.preventDefault();
        deleteSelection();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}
