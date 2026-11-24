import type { TreeCriterion } from '@tinker/core';
import { useMemo } from 'react';
import { useStore } from '../../state/store.js';
import { useUiShown } from '../../state/disclosure.js';
import { treeFitsOf } from '../../state/treeFits.js';
import { NodeShell } from './NodeShell.js';
import { NumberField, SelectField, editSettingOpen } from './fields.js';
import type { NodeOf, WebPiece } from './types.js';

const REGRESSION: readonly (readonly [TreeCriterion, string])[] = [
  ['squared_error', 'Squared error'],
  ['absolute_error', 'Absolute error'],
];
const CLASSIFICATION: readonly (readonly [TreeCriterion, string])[] = [
  ['gini', 'Gini'],
  ['entropy', 'Entropy'],
];

/** Leaves and splits of the fitted tree of `id`, or undefined while it has none (not wired, wrong criterion). */
export function useTreeSummary(id: string): { leaves: number; splits: number } | undefined {
  const graph = useStore((s) => s.graph);
  const dataset = useStore((s) => s.dataset);
  return useMemo(() => {
    const fit = treeFitsOf(graph, dataset)[id];
    if (!fit) return undefined;
    return { leaves: fit.leaves, splits: fit.nodes.filter((n) => n.kind === 'split').length };
  }, [graph, dataset, id]);
}

function TreeFields({ node }: { node: NodeOf<'tree'> }) {
  const kind = useStore((s) => s.dataInfo.kind);
  const summary = useTreeSummary(node.id);
  const showCriterion = useUiShown('tree-criterion');
  const showDepth = useUiShown('tree-depth');
  const showMinSplit = useUiShown('tree-min-split');
  const showMinLeaf = useUiShown('tree-min-leaf');
  const showMinDecrease = useUiShown('tree-min-decrease');
  const base = kind === 'regression' ? REGRESSION : CLASSIFICATION;
  // keep the current value selectable when it does not fit the data: analysis says so in a sentence
  const options = base.some(([v]) => v === node.criterion)
    ? base
    : [...base, [node.criterion, node.criterion.replace('_', ' ')] as const];
  const set = (key: string, uiId: string) => (v: number | string) =>
    editSettingOpen(node.id, key, v, uiId);
  return (
    <>
      {summary && (
        <p className="tk-diag" data-testid="tree-summary">
          {summary.leaves} {summary.leaves === 1 ? 'leaf' : 'leaves'}, {summary.splits}{' '}
          {summary.splits === 1 ? 'split' : 'splits'}
        </p>
      )}
      <p className="tk-diag">Fit on y (training rows)</p>
      {showCriterion && (
        <SelectField
          label="Criterion"
          testid="tree-criterion"
          value={node.criterion}
          options={options}
          onChange={set('criterion', 'tree-criterion')}
        />
      )}
      {showDepth && (
        <NumberField
          label="Depth"
          testid="tree-depth"
          value={node.depth}
          min={1}
          max={8}
          step={1}
          onCommit={set('depth', 'tree-depth')}
        />
      )}
      {showMinSplit && (
        <NumberField
          label="Min samples to split"
          testid="tree-min-split"
          value={node.minSplit}
          min={2}
          step={1}
          onCommit={set('minSplit', 'tree-min-split')}
        />
      )}
      {showMinLeaf && (
        <NumberField
          label="Min samples per leaf"
          testid="tree-min-leaf"
          value={node.minLeaf}
          min={1}
          step={1}
          onCommit={set('minLeaf', 'tree-min-leaf')}
        />
      )}
      {showMinDecrease && (
        <NumberField
          label="Min decrease"
          testid="tree-min-decrease"
          value={node.minDecrease}
          min={0}
          step={0.001}
          onCommit={set('minDecrease', 'tree-min-decrease')}
        />
      )}
    </>
  );
}

/** The Tree piece (CONTRACTS.md section 3): a box `Tree`, `depth d`; the leaf count is its output wire's width. */
export const treePiece: WebPiece<'tree'> = {
  Layer: function TreeLayer({ node, w, h, scale }) {
    return (
      <NodeShell
        piece="tree"
        w={w}
        h={h}
        scale={scale}
        name="Tree"
        glyph="leaf"
        secondary={[`depth ${node.depth}`]}
        inPorts={1}
        outPort
      />
    );
  },
  Neuron: () => <g data-placeholder="tree-neuron" />,
  Fields: TreeFields,
};
