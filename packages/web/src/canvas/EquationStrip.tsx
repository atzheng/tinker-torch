import { useEffect, useMemo, useState } from 'react';
import {
  initParams,
  modelEquations,
  numericExtras,
  wantsNumeric,
  type Analysis,
  type EquationLine,
  type Graph,
  type HostParams,
} from '@tinker/core';

import { useStore } from '../state/store.js';
import { Tex } from '../tex/Tex.js';

/**
 * The trained host, following the store at most once per animation frame. Only this component
 * re-renders on a training frame; the canvas does not subscribe to `host` (CONTRACTS.md section 10).
 */
function useFrameHost(): HostParams | undefined {
  const [host, setHost] = useState<HostParams | undefined>(
    () => useStore.getState().host ?? undefined,
  );
  useEffect(() => {
    let raf: number | null = null;
    const sync = () => {
      raf = null;
      setHost(useStore.getState().host ?? undefined);
    };
    sync();
    const off = useStore.subscribe((s, prev) => {
      if (s.host === prev.host || raf !== null) return;
      raf = requestAnimationFrame(sync);
    });
    return () => {
      off();
      if (raf !== null) cancelAnimationFrame(raf);
    };
  }, []);
  return host;
}

const Line = ({ l, numeric }: { l: EquationLine; numeric?: boolean }) => (
  <div
    className="tk-eqline"
    data-testid="equation-line"
    data-kind={l.kind}
    data-numeric={numeric === true}
  >
    <Tex src={`${l.lhs} = ${l.rhs}`} />
  </div>
);

/**
 * The model and the objective as equations (tinker-96i), along the bottom of the canvas in both
 * grain views. Numeric lines (live values) appear for scalar-sized models. Collapses to a toggle.
 */
export function EquationStrip(p: { graph: Graph; analysis: Analysis }) {
  const [open, setOpen] = useState(true);
  const info = useStore((s) => s.dataInfo);
  const trained = useFrameHost();
  // Values are only needed for numeric lines: skip them when closed or when the model is too big.
  const numeric = open && wantsNumeric(p.graph, p.analysis);
  const seed = p.graph.train.seed;
  const initial = useMemo(() => {
    if (!numeric) return undefined;
    try {
      return initParams(p.analysis, seed);
    } catch {
      return undefined;
    }
  }, [numeric, p.analysis, seed]);
  const host = numeric ? (trained ?? initial) : undefined;
  const eq = useMemo(
    () => modelEquations(p.graph, p.analysis, info, host),
    [p.graph, p.analysis, info, host],
  );
  if (eq.lines.length === 0) return null;
  return (
    <div className="tk-eqdock" data-open={open}>
      <button
        type="button"
        className="tk-eqtoggle"
        data-testid="equation-toggle"
        aria-expanded={open}
        title={open ? 'Hide the equations' : 'Show the equations'}
        onClick={() => setOpen(!open)}
      >
        <Tex src="f(x)" />
      </button>
      {open && (
        <div className="tk-eqstrip" data-testid="model-equation">
          <div className="tk-eqlines">
            {eq.lines.map((l, i) => (
              <Line key={`${l.kind}${i}`} l={l} />
            ))}
            {numericExtras(eq).map((l, i) => (
              <Line key={`n${l.kind}${i}`} l={l} numeric />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
