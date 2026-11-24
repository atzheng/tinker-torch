/** Theme preference (CONTRACTS section 10, Themes): System, Light or Dark. A UI preference kept in
 * localStorage only; never in graph files, presets or share links. The resolved theme goes on
 * <html data-theme="light|dark">, which theme.css reads. */
export type ThemePref = 'system' | 'light' | 'dark';
export type ResolvedTheme = 'light' | 'dark';

export const THEME_KEY = 'tinker.theme';
/** Fired on window after the resolved theme changes, so canvas views re-read their CSS variables. */
export const THEME_EVENT = 'tinker-theme';

const ORDER: readonly ThemePref[] = ['system', 'light', 'dark'];

/** A stored value to a preference; anything unknown (or null) is System. */
export function parseThemePref(raw: unknown): ThemePref {
  return raw === 'light' || raw === 'dark' ? raw : 'system';
}

/** The state after one press of the cycling button: System, Light, Dark, System. */
export function nextThemePref(p: ThemePref): ThemePref {
  return ORDER[(ORDER.indexOf(p) + 1) % ORDER.length] ?? 'system';
}

export function resolveTheme(p: ThemePref, osDark: boolean): ResolvedTheme {
  return p === 'system' ? (osDark ? 'dark' : 'light') : p;
}

export function readThemePref(): ThemePref {
  try {
    return parseThemePref(window.localStorage.getItem(THEME_KEY));
  } catch {
    return 'system';
  }
}

export function writeThemePref(p: ThemePref): void {
  try {
    if (p === 'system') window.localStorage.removeItem(THEME_KEY);
    else window.localStorage.setItem(THEME_KEY, p);
  } catch {
    /* storage may be blocked; the choice then lasts for the session only */
  }
}

export function osPrefersDark(): boolean {
  return (
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-color-scheme: dark)').matches
  );
}

/** Put the resolved theme on <html> and tell canvas views when it changed. */
export function applyThemePref(p: ThemePref): ResolvedTheme {
  const t = resolveTheme(p, osPrefersDark());
  const root = document.documentElement;
  if (root.getAttribute('data-theme') !== t) {
    root.setAttribute('data-theme', t);
    window.dispatchEvent(new Event(THEME_EVENT));
  }
  return t;
}
