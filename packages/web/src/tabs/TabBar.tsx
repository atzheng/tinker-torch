import { useEffect, useRef, type KeyboardEvent } from 'react';
import { TAB_UI_IDS, effectiveTab, type ViewTab } from '@tinker/core';
import { useShownTabs, useUiLabel, useViewTab } from '../state/disclosure.js';
import { useStore } from '../state/store.js';
import styles from './TabBar.module.css';

export const TAB_LABELS: Record<ViewTab, string> = {
  model: 'Model',
  loss: 'Loss',
  full: 'Full graph',
};

function Tab({
  tab,
  active,
  onKey,
}: {
  tab: ViewTab;
  active: boolean;
  onKey: (e: KeyboardEvent) => void;
}) {
  const setTab = useStore((s) => s.setViewTab);
  const label = useUiLabel(TAB_UI_IDS[tab], TAB_LABELS[tab]);
  return (
    <button
      type="button"
      role="tab"
      id={`tab-${tab}`}
      className={styles.tab}
      data-testid={TAB_UI_IDS[tab]}
      aria-selected={active}
      aria-controls="tab-panel"
      tabIndex={active ? 0 : -1}
      onClick={() => setTab(tab)}
      onKeyDown={onKey}
    >
      {label}
    </button>
  );
}

/**
 * The model and loss tabs (CONTRACTS.md section 9, "Tabs"): a proper tablist, so the arrow keys
 * (and Home and End) move between the tabs. Which tabs exist, and which one opens first, comes from
 * the preset (`show.ui`, `show.tab`); with one tab left the bar is not drawn. A load (a new
 * `loadEpoch`) opens the preset's tab.
 */
export function TabBar() {
  const shown = useShownTabs();
  const active = useViewTab();
  const setTab = useStore((s) => s.setViewTab);
  const epoch = useStore((s) => s.loadEpoch);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const { show } = useStore.getState();
    useStore.getState().setViewTab(effectiveTab(show?.tab ?? 'model', show?.ui));
  }, [epoch]);
  if (shown.length < 2) return null;
  const onKey = (e: KeyboardEvent) => {
    const at = shown.indexOf(active);
    let to: number;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') to = (at + 1) % shown.length;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp')
      to = (at - 1 + shown.length) % shown.length;
    else if (e.key === 'Home') to = 0;
    else if (e.key === 'End') to = shown.length - 1;
    else return;
    e.preventDefault();
    const next = shown[to];
    if (next === undefined) return;
    setTab(next);
    ref.current?.querySelector<HTMLElement>(`#tab-${next}`)?.focus();
  };
  return (
    <div
      ref={ref}
      className={styles.bar}
      role="tablist"
      aria-label="Model, loss or full graph"
      data-testid="tab-bar"
    >
      {shown.map((t) => (
        <Tab key={t} tab={t} active={t === active} onKey={onKey} />
      ))}
    </div>
  );
}
