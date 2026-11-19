/**
 * Module worker for the resampling overlay. It owns its own engine instance (jax-js is loaded
 * here, not shared with the main thread), so the 20 x steps training never blocks the page.
 * Messages: ResampleRequest in, ResampleResponse (plus the dev-only live-buffer count) out.
 */
import { adaptEngine } from '../legacyEngine.js';
import type { Engine } from '../legacyEngine.js';
import { runResample, type ResampleRequest, type ResampleResponse } from './resampleRun.js';

type Out = ResampleResponse & { live?: number };

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<ResampleRequest>) => void) | null;
  postMessage(message: Out, transfer?: Transferable[]): void;
};

let engine: Promise<Engine> | null = null;

function loadEngine(): Promise<Engine> {
  engine ??= (async () => {
    if (import.meta.env.DEV) {
      // Dev only: the live-buffer probe must be installed before jax-js is loaded.
      (await import('@tinker/engine/probe')).installProbe();
    }
    return adaptEngine(await (await import('@tinker/engine')).createEngine()); // temporary adapter (tinker-8xw.4)
  })();
  return engine;
}

scope.onmessage = (e) => {
  void (async () => {
    let out: Out;
    try {
      const eng = await loadEngine();
      out = runResample(eng, e.data);
      if (import.meta.env.DEV) out.live = eng.liveArrays();
    } catch {
      out = { ok: false, message: 'The training engine could not be loaded.' };
    }
    scope.postMessage(out, out.ok ? out.curves.map((c) => c.buffer) : []);
  })();
};
