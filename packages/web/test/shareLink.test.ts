import { expect, test } from 'vitest';
import { parseFile, serializeFile, type Graph } from '@tinker/core';
import { mlpGraph } from './mlpGraph.js';
import {
  decodeShare,
  encodeShare,
  SHARE_BAD_LINK,
  SHARE_FRAGMENT_MAX,
  SHARE_INFLATED_MAX,
  startFromUrl,
  SHARE_NEWER,
  SHARE_PREFIX,
  shareFromHash,
} from '../src/share/shareLink.js';

function presetText(g: Graph, extra: Record<string, unknown>): string {
  const params = { lin1: { W: Array(8).fill(0.123456789012345), b: [0.1] } };
  const doc = JSON.parse(serializeFile(g, params)) as Record<string, unknown>;
  return JSON.stringify({ ...doc, ...extra });
}

const failure = (r: Awaited<ReturnType<typeof decodeShare>>) => (r.ok ? 'ok' : r.error);

test('round trip keeps locked, disclosure, prompts and labels', async () => {
  const text = presetText(mlpGraph(), {
    title: 'T',
    prompt: 'Find w1.',
    instructions: ['one', 'two'],
    locked: ['lin1'],
    show: { ui: ['file-menu', 'btn-undo'] },
    labels: { 'data-set': 'Dataset' },
  });
  const value = await encodeShare(text);
  expect(value?.startsWith(SHARE_PREFIX)).toBe(true);
  const r = await decodeShare(value ?? '');
  expect(r).toEqual(parseFile(JSON.parse(text)));
  expect(r.ok).toBe(true);
  if (r.ok) {
    const f = r.file as Record<string, unknown>;
    expect(f['locked']).toEqual(['lin1']);
    expect(f['prompt']).toBe('Find w1.');
    expect(f['instructions']).toEqual(['one', 'two']);
    expect((f['show'] as { ui: string[] }).ui).toContain('btn-undo');
    expect(f['labels']).toEqual({ 'data-set': 'Dataset' });
  }
});

test('an embedded csv survives', async () => {
  const g = mlpGraph();
  const gg = { ...g, data: { ...g.data, set: 'csv' } } as Graph;
  const csv = {
    name: 'mine',
    kind: 'regression',
    features: ['a'],
    target: 'y',
    x: Array.from({ length: 12 }, (_, i) => [i]),
    y: Array.from({ length: 12 }, (_, i) => i / 3),
  };
  const doc = { ...(JSON.parse(serializeFile(gg)) as object), csv };
  const parsed = parseFile(doc);
  expect(parsed.ok).toBe(true);
  const r = await decodeShare((await encodeShare(JSON.stringify(doc))) ?? '');
  expect(r).toEqual(parsed);
  if (r.ok) expect(r.file.csv?.y[1]).toBe(1 / 3);
});

test('values stay at full precision', async () => {
  const r = await decodeShare((await encodeShare(presetText(mlpGraph(), {}))) ?? '');
  expect(r.ok && JSON.stringify(r.file.params)).toContain('0.123456789012345');
});

test('too large gives null', async () => {
  const text = presetText(mlpGraph(), {});
  expect(await encodeShare(text, 0, 50)).toBeNull();
  expect(await encodeShare(text, 0)).not.toBeNull();
});

test('failures give one sentence', async () => {
  expect(failure(await decodeShare('x=1.@@@'))).toBe(SHARE_BAD_LINK);
  expect(failure(await decodeShare('x=1.AAAA'))).toBe(SHARE_BAD_LINK);
  expect(failure(await decodeShare('hello'))).toBe(SHARE_BAD_LINK);
  expect(failure(await decodeShare('x=2.abc'))).toBe(SHARE_NEWER);
  const notAFile = await encodeShare('{"kind":"nope"}');
  const msg = failure(await decodeShare(notAFile ?? ''));
  expect(msg).not.toBe('ok');
  expect(msg).toMatch(/\.$/);
});

test('shareFromHash', () => {
  expect(shareFromHash('#x=1.abc')).toBe('x=1.abc');
  expect(shareFromHash('#a=1&x=1.abc')).toBe('x=1.abc');
  expect(shareFromHash('')).toBeNull();
});

test('a high-ratio payload is refused', async () => {
  const stream = new CompressionStream('deflate-raw');
  const w = stream.writable.getWriter();
  void w.write(new Uint8Array(SHARE_INFLATED_MAX + 1000));
  void w.close();
  const bytes = new Uint8Array(await new Response(stream.readable).arrayBuffer());
  expect(bytes.length).toBeLessThan(SHARE_FRAGMENT_MAX);
  const b64 = btoa(String.fromCharCode(...bytes));
  expect(failure(await decodeShare('x=1.' + b64))).toBe(SHARE_BAD_LINK);
  expect(failure(await decodeShare('x=1.' + 'A'.repeat(SHARE_FRAGMENT_MAX)))).toBe(SHARE_BAD_LINK);
});

test('padding, %3D and standard base64 still load', async () => {
  const value = (await encodeShare(presetText(mlpGraph(), {}))) ?? '';
  const std = value.replace(/-/g, '+').replace(/_/g, '/');
  for (const v of [value + '=', value + '%3D', std, std + '==']) {
    expect((await decodeShare(v)).ok).toBe(true);
  }
});

test('start-up: the link wins over ?preset=; failures call fail', async () => {
  const value = (await encodeShare(presetText(mlpGraph(), { title: 'Shared' }))) ?? '';
  const loaded: string[] = [];
  const presets: string[] = [];
  const fails: string[] = [];
  const load = (f: { graph: Graph }) => {
    loaded.push(f.graph.data.set);
    return { ok: true, message: null };
  };
  const lp = (n: string) => presets.push(n) > 0;
  await startFromUrl('?preset=xor', '#' + value, load, lp, (m) => fails.push(m));
  expect(loaded).toHaveLength(1);
  expect(presets).toEqual([]);
  await startFromUrl('?preset=xor', '', load, lp, (m) => fails.push(m));
  expect(presets).toEqual(['xor']);
  await startFromUrl('', '#x=1.AAAA', load, lp, (m) => fails.push(m));
  expect(fails).toEqual([SHARE_BAD_LINK]);
});
