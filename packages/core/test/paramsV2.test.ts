import { describe, expect, test } from 'vitest';
import fixture from '../../engine/test/fixtures/mlp_tanh_sgd_5steps.json';
import tapOnly from '../../engine/test/fixtures/prim_weights_tap_only.json';
import {
  analyse as analyseV1,
  initParams as initV1,
  parseGraph as parseV1,
  v2,
} from '../src/index.js';

const info = { columns: ['x1', 'x2'], kind: 'regression' as const, numClasses: 2 };

describe('v2.initParams', () => {
  test('a migrated version 1 graph starts exactly as in version 1', () => {
    const v1 = parseV1(fixture.graph);
    const g2 = v2.parseGraph(fixture.graph);
    if (!v1.ok || !g2.ok) throw new Error('parse');
    const want = initV1(v1.graph, analyseV1(v1.graph, info), 7);
    const got = v2.initParams(v2.analyse(g2.graph, info), 7);
    expect(Object.keys(got).sort()).toEqual(Object.keys(want).sort());
    for (const id of Object.keys(want)) {
      expect(Array.from(got[id]?.W ?? [])).toEqual(Array.from(want[id]?.W ?? []));
      expect(Array.from(got[id]?.b ?? [])).toEqual(Array.from(want[id]?.b ?? []));
    }
  });

  test('a Linear read only through a Weights tap is initialised', () => {
    const g = v2.parseGraph(tapOnly.graph);
    if (!g.ok) throw new Error('parse');
    const a = v2.analyse(g.graph, info);
    expect(a.order).not.toContain('tap');
    const host = v2.initParams(a, 3);
    expect(Object.keys(host).sort()).toEqual(['lin1', 'lin2', 'tap']);
    expect(host['tap']?.W.length).toBe(2 * 2);
    expect(host['tap']?.b?.length).toBe(2);
  });
});
