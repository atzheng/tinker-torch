import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import { torchScript, type NodeId, type TorchCode } from '@tinker/core';
import { useStore } from '../state/store.js';
import { useAnalysis, useSelection } from '../state/derived.js';
import { useUiShown } from '../state/disclosure.js';
import type { Selection } from '../state/viewSlice.js';
import { buildRows, firstSpan, lineOf, lineStarts, nodeAt, nodeOnLine } from './codeModel.js';
import { useCodeParams } from './useCodeParams.js';
import styles from './CodeView.module.css';

/** The piece a selection is about, for highlighting its spans (a wire is about none). */
function selectedNode(sel: Selection | null): NodeId | null {
  if (!sel || sel.kind === 'wire') return null;
  return sel.kind === 'node' ? sel.id : sel.node;
}

/** `Linear regression` to `linear-regression`: the download's file name. */
export function scriptFileName(title: string | undefined): string {
  const slug = (title ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return `${slug || 'tinker'}.py`;
}

function downloadText(text: string, name: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/x-python' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // no clipboard permission (or an insecure origin): the old route through a hidden textarea
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try {
      return document.execCommand('copy');
    } catch {
      return false;
    } finally {
      ta.remove();
    }
  }
}

/**
 * The Code tab (tinker-img.2, CONTRACTS.md section 14): the whole generated PyTorch script, read
 * only. Selection is shared with the graph tabs: a selected piece lights its spans, a tap on a line
 * selects the piece whose span is the innermost one under the tap. An invalid graph shows its issues.
 */
export function CodeView() {
  const graph = useStore((s) => s.graph);
  const dataset = useStore((s) => s.dataset);
  const title = useStore((s) => s.preset?.title);
  const analysis = useAnalysis();
  const sel = useSelection(analysis);
  const selected = selectedNode(sel);
  const showCopy = useUiShown('code-copy');
  const showDownload = useUiShown('code-download');
  const params = useCodeParams(analysis);
  const code: TorchCode | null = useMemo(
    () => torchScript(graph, analysis, dataset, params ? { params } : undefined),
    [graph, analysis, dataset, params],
  );
  const [open, setOpen] = useState<ReadonlySet<number>>(() => new Set());
  const [copied, setCopied] = useState<'yes' | 'no' | null>(null);
  const rows = useMemo(() => (code ? buildRows(code, open, selected) : []), [code, open, selected]);
  const scroller = useRef<HTMLDivElement>(null);
  // A tap on a line selects; the view must not then jump to that piece's first span.
  const fromTap = useRef(false);
  const press = useRef<{ id: number; x: number; y: number } | null>(null);

  // A piece selected elsewhere: bring its first span into view (inside the code panel only).
  useEffect(() => {
    if (fromTap.current) {
      fromTap.current = false;
      return;
    }
    const box = scroller.current;
    if (!box || !code || selected === null) return;
    const span = firstSpan(code.spans, selected);
    if (!span) return;
    const line = lineOf(lineStarts(code.text), span.from) + 1;
    const row =
      box.querySelector<HTMLElement>(`[data-line="${line}"]`) ??
      [...box.querySelectorAll<HTMLElement>('[data-line]')]
        .filter((r) => Number(r.dataset['start']) <= span.from)
        .pop();
    if (!row) return;
    const top = row.offsetTop;
    const bottom = top + row.offsetHeight;
    if (top < box.scrollTop || bottom > box.scrollTop + box.clientHeight) {
      box.scrollTop = Math.max(0, top - box.clientHeight / 3);
    }
  }, [selected, code]);

  useEffect(() => {
    if (copied === null) return;
    const t = setTimeout(() => setCopied(null), 1800);
    return () => clearTimeout(t);
  }, [copied]);

  if (!code) {
    const lines = [...new Set(analysis.issues.map((i) => i.message))];
    return (
      <div className={styles.view} data-testid="code-view" data-state="invalid">
        <div className={styles.issues} data-testid="issue-global" role="status">
          {lines.length === 0 ? (
            <p>There is no code yet: the graph needs a prediction and a loss.</p>
          ) : (
            lines.map((m) => <p key={m}>{m}</p>)
          )}
        </div>
      </div>
    );
  }

  const toggle = (i: number) =>
    setOpen((o) => {
      const n = new Set(o);
      if (n.has(i)) n.delete(i);
      else n.add(i);
      return n;
    });

  const tap = (target: Element) => {
    if (target.closest('button')) return; // the fold buttons handle their own click
    const seg = target.closest<HTMLElement>('[data-from]');
    const row = target.closest<HTMLElement>('[data-line]');
    if (!row) return;
    const spans = code.spans;
    let node = seg ? nodeAt(spans, Number(seg.dataset['from'])) : null;
    node ??= nodeOnLine(spans, Number(row.dataset['start']), Number(row.dataset['end']));
    fromTap.current = true;
    if (node === null) {
      fromTap.current = false;
      useStore.getState().select(null);
    } else {
      useStore.getState().selectNode(node);
    }
  };

  const onDown = (e: PointerEvent) => {
    press.current = { id: e.pointerId, x: e.clientX, y: e.clientY };
  };
  const onUp = (e: PointerEvent) => {
    const p = press.current;
    press.current = null;
    if (!p || p.id !== e.pointerId) return;
    if (Math.hypot(e.clientX - p.x, e.clientY - p.y) > 8) return; // a scroll or a drag, not a tap
    if (window.getSelection()?.toString()) return; // the person is selecting text to copy
    tap(e.target as Element);
  };

  return (
    <div className={styles.view} data-testid="code-view" data-state="ready">
      <div className={styles.bar}>
        {showCopy && (
          <button
            type="button"
            className={styles.btn}
            data-testid="code-copy"
            onClick={() => void copyText(code.text).then((ok) => setCopied(ok ? 'yes' : 'no'))}
          >
            Copy
          </button>
        )}
        {showDownload && (
          <button
            type="button"
            className={styles.btn}
            data-testid="code-download"
            onClick={() => downloadText(code.text, scriptFileName(title))}
          >
            Download .py
          </button>
        )}
        <span className={styles.note} role="status" data-testid="code-copy-status">
          {copied === 'yes' ? 'Copied' : copied === 'no' ? 'Could not copy' : ''}
        </span>
      </div>
      <div
        className={styles.scroller}
        ref={scroller}
        tabIndex={0}
        role="region"
        aria-label="Generated PyTorch script"
        onPointerDown={onDown}
        onPointerUp={onUp}
        onPointerCancel={() => (press.current = null)}
      >
        <div className={styles.lines}>
          {rows.map((r) => (
            <div
              key={r.n}
              className={`${styles.row} ${
                r.openFold !== undefined || r.segs.some((x) => x.kind === 'fold')
                  ? styles.foldRow
                  : ''
              }`}
              data-line={r.n}
              data-start={r.start}
              data-end={r.end}
              data-hl={r.hl}
            >
              <span className={styles.gutter} aria-hidden={r.openFold === undefined}>
                {r.openFold !== undefined && (
                  <button
                    type="button"
                    className={styles.caret}
                    aria-label="Collapse the data"
                    aria-expanded={true}
                    data-fold={r.openFold}
                    onClick={() => toggle(r.openFold ?? 0)}
                  >
                    ▾
                  </button>
                )}
                <span className={styles.num}>{r.n}</span>
              </span>
              <span className={styles.text}>
                {r.segs.map((s, i) =>
                  s.kind === 'fold' ? (
                    <button
                      key={i}
                      type="button"
                      className={styles.fold}
                      aria-label="Expand the data"
                      aria-expanded={false}
                      data-fold={s.fold}
                      onClick={() => toggle(s.fold ?? 0)}
                    >
                      {s.text}
                    </button>
                  ) : (
                    <span
                      key={i}
                      className={styles[s.kind]}
                      data-hl={s.hl}
                      {...(s.from === null ? {} : { 'data-from': s.from })}
                    >
                      {s.text}
                    </span>
                  ),
                )}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
