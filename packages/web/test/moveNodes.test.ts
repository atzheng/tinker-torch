import { beforeEach, describe, expect, test } from 'vitest';
import { analyse, layoutLayers, moveNodes, parseFileText, serializeFile } from '@tinker/core';
import { useStore } from '../src/state/store.js';
import { PIN, pinCentre, pinHitRect, placePins } from '../src/canvas/pure.js';

beforeEach(() => {
  useStore.getState().loadGraph(useStore.getState().graph);
});

describe('moving a piece through the store', () => {
  test('is one undo step, survives save and load, and undo restores the layout', () => {
    const s = useStore.getState();
    const before = s.graph;
    expect(s.apply(moveNodes(s.graph, { lin1: [200, 150] })).ok).toBe(true);
    const moved = useStore.getState().graph;
    expect(moved.positions).toEqual({ lin1: [200, 150] });
    expect(useStore.getState().past.length).toBe(1);

    const back = parseFileText(serializeFile(moved));
    expect(back.ok && back.file.graph.positions).toEqual({ lin1: [200, 150] });

    useStore.getState().undo();
    expect(useStore.getState().graph).toBe(before);
    useStore.getState().redo();
    expect(useStore.getState().graph.positions).toEqual({ lin1: [200, 150] });
  });
});

describe('pin placement with pieces moved by hand', () => {
  const g = useStore.getState().graph;
  const layout = (positions?: Record<string, [number, number]>) => {
    const gg = positions ? { ...g, positions } : g;
    return layoutLayers(gg, analyse(gg, useStore.getState().dataInfo));
  };

  test('without avoidance the spot is the fixed offset from the port', () => {
    const l = layout();
    const box = l.nodes['lin2'];
    if (!box) throw new Error('no box');
    const m = placePins(l.nodes, [{ role: 'pred', node: 'lin2' }], false);
    expect(m.get('pred:lin2')).toEqual(pinCentre(box, 'pred'));
  });

  test('a piece where the pin would go pushes the pin to a free spot', () => {
    const base = layout();
    const box = base.nodes['lin2'];
    if (!box) throw new Error('no box');
    const [cx, cy] = pinCentre(box, 'pred');
    // put act1 right on the default ŷ spot
    const l = layout({ act1: [cx - 20, cy - 20] });
    const spot = placePins(l.nodes, [{ role: 'pred', node: 'lin2' }], true).get('pred:lin2');
    const act = l.nodes['act1'];
    if (!spot || !act) throw new Error('missing');
    expect(spot).not.toEqual([cx, cy]);
    const hits =
      spot[0] + PIN.r > act.x &&
      spot[0] - PIN.r < act.x + act.w &&
      spot[1] + PIN.r > act.y &&
      spot[1] - PIN.r < act.y + act.h;
    expect(hits).toBe(false);
    // the press rectangle follows the pin and never reaches into the piece
    const hit = pinHitRect(l.nodes['lin2'] ?? box, 'pred', 1, spot);
    expect(hit.x).toBeGreaterThan((l.nodes['lin2'] ?? box).x + (l.nodes['lin2'] ?? box).w);
  });

  test('two pins never share a spot', () => {
    const l = layout();
    const m = placePins(
      l.nodes,
      [
        { role: 'pred', node: 'lin2' },
        { role: 'loss', node: 'lin2' },
      ],
      true,
    );
    const a = m.get('pred:lin2');
    const b = m.get('loss:lin2');
    expect(a && b && Math.hypot(a[0] - b[0], a[1] - b[1])).toBeGreaterThan(2 * PIN.r);
  });
});
