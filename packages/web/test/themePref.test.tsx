// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import {
  THEME_EVENT,
  THEME_KEY,
  applyThemePref,
  nextThemePref,
  parseThemePref,
  resolveTheme,
} from '../src/themePref.js';
import { ThemeToggle } from '../src/topbar/ThemeToggle.js';

function stubMedia(dark: boolean) {
  window.matchMedia = ((q: string) => ({
    matches: dark && q.includes('dark'),
    addEventListener() {},
    removeEventListener() {},
  })) as unknown as typeof window.matchMedia;
}

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
function mount() {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const r = createRoot(host);
  root = r;
  act(() => r.render(<ThemeToggle />));
  return host.querySelector('[data-testid=theme-toggle]') as HTMLButtonElement;
}
function click(b: HTMLElement) {
  act(() => void b.click());
}

afterEach(() => {
  act(() => root?.unmount());
  root = null;
  document.body.innerHTML = '';
  window.localStorage.clear();
  document.documentElement.removeAttribute('data-theme');
});

describe('theme preference logic', () => {
  it('parses stored values, unknown is system', () => {
    expect(parseThemePref('light')).toBe('light');
    expect(parseThemePref('dark')).toBe('dark');
    expect(parseThemePref('system')).toBe('system');
    expect(parseThemePref(null)).toBe('system');
    expect(parseThemePref('blue')).toBe('system');
  });
  it('cycles system, light, dark', () => {
    expect(nextThemePref('system')).toBe('light');
    expect(nextThemePref('light')).toBe('dark');
    expect(nextThemePref('dark')).toBe('system');
  });
  it('resolves system from the OS and explicit choices over it', () => {
    expect(resolveTheme('system', true)).toBe('dark');
    expect(resolveTheme('system', false)).toBe('light');
    expect(resolveTheme('light', true)).toBe('light');
    expect(resolveTheme('dark', false)).toBe('dark');
  });
});

describe('ThemeToggle', () => {
  it('applies the attribute, persists, announces and cycles', () => {
    stubMedia(true);
    const onTheme = vi.fn();
    window.addEventListener(THEME_EVENT, onTheme);
    const btn = mount();
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    expect(btn.getAttribute('aria-label')).toBe('Theme: System');
    click(btn);
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    expect(window.localStorage.getItem(THEME_KEY)).toBe('light');
    expect(btn.getAttribute('aria-label')).toBe('Theme: Light');
    click(btn);
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    click(btn);
    expect(window.localStorage.getItem(THEME_KEY)).toBeNull();
    expect(onTheme).toHaveBeenCalled();
    window.removeEventListener(THEME_EVENT, onTheme);
  });
  it('starts from the stored choice', () => {
    stubMedia(false);
    window.localStorage.setItem(THEME_KEY, 'dark');
    mount();
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    expect(applyThemePref('light')).toBe('light');
  });
});
