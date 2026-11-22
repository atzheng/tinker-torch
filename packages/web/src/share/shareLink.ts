/**
 * Shareable exercise links (tinker-6t7.14). A link holds the same object a saved file holds,
 * as compact JSON, deflate-raw compressed (the browser's CompressionStream), base64url, in the
 * URL fragment: `#x=1.<payload>`. The fragment is never sent to a server and has no length limit
 * of its own. `1.` is the format version.
 */
import { parseFileText, type ParseFileResult, type Preset, type SaveFile } from '@tinker/core';

export const SHARE_PREFIX = 'x=1.';
/** A link longer than this many characters is refused (browsers and chat apps truncate long URLs). */
export const SHARE_LINK_MAX = 32000;

/** A fragment longer than this is not decoded. */
export const SHARE_FRAGMENT_MAX = 200000;
/** Decompression stops past this many bytes (a 5000 x 50 CSV is well under it); guards against a tiny fragment inflating hugely. */
export const SHARE_INFLATED_MAX = 16 * 1024 * 1024;

export const SHARE_TOO_LARGE =
  'This exercise is too large for a link (the data would not fit); save a file instead.';
export const SHARE_BAD_LINK = 'This link could not be opened because it is damaged or incomplete.';
export const SHARE_NEWER = 'This link was made by a newer version of the app and cannot be opened.';

function toBase64Url(bytes: Uint8Array): string {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(text: string): Uint8Array {
  // Chat apps and URL decoding may add '=' padding ('%3D') or turn base64url into standard base64.
  const clean = text.replace(/(%3D|=)+$/i, '');
  if (!/^[A-Za-z0-9_+/-]*$/.test(clean)) throw new Error('base64');
  const b = atob(clean.replace(/-/g, '+').replace(/_/g, '/'));
  const out = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) out[i] = b.charCodeAt(i);
  return out;
}

async function pipe(
  bytes: Uint8Array,
  stream: CompressionStream | DecompressionStream,
  maxOut = Infinity,
) {
  const writer = stream.writable.getWriter();
  // Errors surface on the read side; swallow the write-side rejection.
  writer.write(bytes as BufferSource).catch(() => undefined);
  writer.close().catch(() => undefined);
  const reader = stream.readable.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.length;
    if (total > maxOut) {
      await reader.cancel().catch(() => undefined);
      throw new Error('too large');
    }
    chunks.push(value);
  }
  const out = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

/** The fragment value (`x=1.<payload>`) for the text of a saved file; null when it exceeds `max` characters. */
export async function encodeShare(
  fileText: string,
  baseLength = 0,
  max = SHARE_LINK_MAX,
): Promise<string | null> {
  const compact = JSON.stringify(JSON.parse(fileText));
  const packed = await pipe(
    new TextEncoder().encode(compact),
    new CompressionStream('deflate-raw'),
  );
  const value = SHARE_PREFIX + toBase64Url(packed);
  return baseLength + 1 + value.length > max ? null : value;
}

/** The fragment value for the share payload in a URL hash (`#x=...`), or null when there is none. */
export function shareFromHash(hash: string): string | null {
  const h = hash.startsWith('#') ? hash.slice(1) : hash;
  for (const part of h.split('&')) if (part.startsWith('x=')) return part;
  return null;
}

/** Decodes a fragment value into a validated file; one plain sentence on any failure. */
export async function decodeShare(value: string): Promise<ParseFileResult> {
  if (value.length > SHARE_FRAGMENT_MAX) return { ok: false, error: SHARE_BAD_LINK };
  const m = /^x=(\d+)\.(.*)$/.exec(value);
  if (!m) return { ok: false, error: SHARE_BAD_LINK };
  if (m[1] !== '1') return { ok: false, error: SHARE_NEWER };
  let text: string;
  try {
    const bytes = await pipe(
      fromBase64Url(m[2] ?? ''),
      new DecompressionStream('deflate-raw'),
      SHARE_INFLATED_MAX,
    );
    text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch {
    return { ok: false, error: SHARE_BAD_LINK };
  }
  const parsed = parseFileText(text);
  return parsed.ok ? parsed : { ok: false, error: parsed.error };
}

/**
 * Start-up: a `#x=` fragment loads the exercise (it wins over `?preset=`); otherwise `?preset=<name>`
 * loads a bundled preset. `load` opens a parsed file (the same path as Load), `fail` shows one sentence.
 */
export async function startFromUrl(
  search: string,
  hash: string,
  load: (file: SaveFile | Preset) => { ok: boolean; message: string | null },
  loadPreset: (name: string) => boolean,
  fail: (message: string) => void,
): Promise<void> {
  const shared = shareFromHash(hash);
  if (shared) {
    const r = await decodeShare(shared);
    if (!r.ok) return fail(r.error);
    const loaded = load(r.file);
    if (!loaded.ok && loaded.message) fail(loaded.message);
    return;
  }
  const name = new URLSearchParams(search).get('preset');
  if (name) loadPreset(name);
}
