import type { Analysis, Graph } from '@tinker/core';
import { findLinearGroups } from '@tinker/core';
import { useMemo } from 'react';
import {
  expandSelected,
  expandTarget,
  groupSelected,
  hasScalarForm,
} from '../state/scalarActions.js';
import type { Selection } from '../state/viewSlice.js';
import { Tex } from '../tex/Tex.js';

/** The equivalence the two buttons convert between: a label, not lesson text. */
export const EQUIVALENCE = String.raw`\hat{y} = \sum_i w_i x_i + b \;\Leftrightarrow\; \hat{y} = W x + b`;

/**
 * Group into Linear / Expand to scalars (tinker-9io.7), in the canvas corner beside the view buttons.
 * Shown when a group pattern or a d to 1 Linear exists. `onPreview(true)` while Group is hovered or
 * focused: the canvas then outlines the pieces it would replace.
 */
export function ScalarBar(p: {
  graph: Graph;
  analysis: Analysis;
  selection: Selection | null;
  onPreview: (on: boolean) => void;
}) {
  const { graph, analysis, selection } = p;
  const groups = useMemo(() => findLinearGroups(graph, analysis), [graph, analysis]);
  const canExpand = expandTarget(graph, selection) !== null;
  if (!hasScalarForm(graph, analysis)) return null;
  return (
    <div className="tk-scalarbar" data-testid="scalar-bar">
      <div className="tk-scalarbtns">
        <button
          type="button"
          className="tk-scalarbtn"
          data-testid="group-linear"
          disabled={groups.length === 0}
          title="Replace these Columns, Weights and Sum by one Linear with the same values"
          onClick={() => {
            groupSelected();
            p.onPreview(false);
          }}
          onPointerEnter={() => p.onPreview(true)}
          onPointerLeave={() => p.onPreview(false)}
          onFocus={() => p.onPreview(true)}
          onBlur={() => p.onPreview(false)}
        >
          Group into Linear
        </button>
        <button
          type="button"
          className="tk-scalarbtn"
          data-testid="expand-scalars"
          disabled={!canExpand}
          title="Replace the selected Linear (one output) by its Columns, Weights and Sum"
          onClick={() => expandSelected()}
        >
          Expand to scalars
        </button>
      </div>
      <div className="tk-equiv" data-testid="equivalence">
        <Tex src={EQUIVALENCE} />
      </div>
    </div>
  );
}
