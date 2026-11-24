import { expect, test } from 'vitest';
import { analyse, layoutLayers } from '@tinker/core';
import { dropsOnStub, resolveDrop, type DragPayload } from '../src/canvas/dnd.js';
import { mlpGraph } from './mlpGraph.js';

const info = () => ({ columns: ['x1'], numClasses: 1, n: 10 }) as never;

test('a wire end dropped on a stub port is rejected, like a pin; the same drop on a real piece is not', () => {
  const g = mlpGraph();
  const a = analyse(g, info());
  const layout = layoutLayers(g, a);
  const must = <T>(v: T | undefined): T => {
    if (v === undefined) throw new Error('missing');
    return v;
  };
  const wire: DragPayload = { kind: 'wire', from: must(g.nodes[0]).id };
  const at = (id: string) => {
    const b = must(layout.nodes[id]);
    return { x: b.x + b.w / 2, y: b.y + b.h / 2 };
  };
  const target = g.nodes.slice(1).find((n) => {
    const t = resolveDrop(wire, g, layout, a.edgeWidth, 0.5, at(n.id));
    return t?.kind === 'port' && t.node === n.id;
  });
  const tid = must(target).id;
  const t = resolveDrop(wire, g, layout, a.edgeWidth, 0.5, at(tid));
  expect(dropsOnStub(t, {})).toBe(false);
  expect(dropsOnStub(t, { [tid]: 'model' })).toBe(true);
  expect(dropsOnStub({ kind: 'node', id: tid }, { [tid]: 'model' })).toBe(true);
  expect(dropsOnStub({ kind: 'port', node: 'a', port: 'a.pred' }, { a: 'loss' })).toBe(true);
  expect(dropsOnStub({ kind: 'wire', edge: 0 }, { a: 'loss' })).toBe(false);
  expect(dropsOnStub(null, { a: 'loss' })).toBe(false);
});
