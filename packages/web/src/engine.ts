/**
 * The single Engine instance (CONTRACTS.md section 7) and its React hook.
 *
 * The engine is loaded lazily with a dynamic import so jax-js stays out of the first paint and out
 * of the jsdom tests. No jax array is ever stored in React state or in the zustand store: the
 * train slice keeps its Model and TrainState handles in a module-level runtime (trainSlice.ts) and
 * publishes plain numbers and HostParams only.
 */
import { useEffect, useState } from 'react';
import type { Engine } from '@tinker/engine';

let engine: Engine | null = null;
let loading: Promise<Engine> | null = null;
const listeners = new Set<(e: Engine) => void>();
const errorListeners = new Set<(message: string) => void>();

/** The engine once loaded, otherwise null. */
export function getEngine(): Engine | null {
  return engine;
}

/** Called once when the engine becomes ready (the train slice uses it to compile). */
export function onEngineReady(fn: (e: Engine) => void): () => void {
  listeners.add(fn);
  if (engine) fn(engine);
  return () => listeners.delete(fn);
}

/** Called when loading fails (for example no WebAssembly). */
export function onEngineError(fn: (message: string) => void): () => void {
  errorListeners.add(fn);
  return () => errorListeners.delete(fn);
}

/** Loads the engine once; every later call returns the same promise (StrictMode safe). */
export function loadEngine(): Promise<Engine> {
  loading ??= (async () => {
    if (import.meta.env.DEV) {
      // Dev only: the live-buffer probe must be installed before jax-js is loaded.
      const probe = await import('@tinker/engine/probe');
      probe.installProbe();
    }
    const mod = await import('@tinker/engine');
    const e = await mod.createEngine();
    engine = e;
    if (import.meta.env.DEV) {
      const w = window as unknown as { __tinker?: Record<string, unknown> };
      w.__tinker = { ...w.__tinker, liveArrays: () => e.liveArrays() };
    }
    for (const fn of [...listeners]) fn(e);
    return e;
  })();
  loading.catch(() => {
    for (const fn of [...errorListeners]) fn('The training engine could not be loaded.');
  });
  return loading;
}

/** Starts loading on mount and returns the engine when ready (null while loading or if it failed). */
export function useEngine(): Engine | null {
  const [e, setE] = useState<Engine | null>(engine);
  useEffect(() => {
    let alive = true;
    loadEngine().then(
      (loaded) => {
        if (alive) setE(loaded);
      },
      () => {
        /* the failure is reported through the store (engineError) by the train slice */
      },
    );
    return () => {
      alive = false;
    };
  }, []);
  return e;
}
