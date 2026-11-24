import { useEffect, useState } from 'react';
import type { Analysis, HostParams } from '@tinker/core';
import { useStore } from '../state/store.js';
import { useInitialWeights } from '../state/derived.js';

/** At most this often while a run is going (CONTRACTS.md section 10: no per-frame renders of the Code tab). */
export const CODE_PARAMS_MS = 500;

/**
 * The parameters the Code tab writes into the script: the train slice's `host` (trained or hand-set),
 * else the step-0 weights. Unlike `useWeights` it does not follow `host` frame by frame: while running
 * it takes the latest `host` at most every `CODE_PARAMS_MS`; on pause, step, reset, hand edit or graph
 * edit (anything that is not a running frame) it takes it at once. `torchScript` ignores values whose
 * sizes do not fit the graph, so a stale snapshot never gives a wrong script.
 */
export function useCodeParams(analysis: Analysis): HostParams | undefined {
  const initial = useInitialWeights(analysis);
  const [host, setHost] = useState<HostParams | null>(() => useStore.getState().host);
  useEffect(() => {
    let last = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const take = () => {
      if (timer !== undefined) clearTimeout(timer);
      timer = undefined;
      last = Date.now();
      setHost(useStore.getState().host);
    };
    take();
    const unsub = useStore.subscribe((s, prev) => {
      if (s.host === prev.host && s.graph === prev.graph && s.running === prev.running) return;
      const wait = CODE_PARAMS_MS - (Date.now() - last);
      const frame =
        s.running && prev.running && s.graph === prev.graph && s.host !== prev.host && wait > 0;
      if (!frame) take();
      else if (timer === undefined) timer = setTimeout(take, wait);
    });
    return () => {
      unsub();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, []);
  return host ?? initial;
}
