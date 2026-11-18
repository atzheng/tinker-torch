import { useEffect } from 'react';
import { useStore } from '../state/store.js';

const TOAST_MS = 5000;

/** The one-sentence message of the last refused edit (`graphSlice.error`); fades after a few seconds. */
export function Toast() {
  const error = useStore((s) => s.error);
  const clearError = useStore((s) => s.clearError);
  useEffect(() => {
    if (!error) return;
    const id = setTimeout(clearError, TOAST_MS);
    return () => clearTimeout(id);
  }, [error, clearError]);
  if (!error) return null;
  return (
    <div className="tk-toast" role="alert" data-testid="toast" onClick={clearError}>
      {error}
    </div>
  );
}
