/**
 * Shareable exercise links (tinker-6t7.14). A link holds the same object a saved file holds,
 * as compact JSON, deflate-raw compressed (the browser's CompressionStream), base64url, in the
 * URL fragment: `#x=1.<payload>`. The fragment is never sent to a server and has no length limit
 * of its own. `1.` is the format version.
 */
import { parseFileText, type ParseFileResult } from '@tinker/core';

export const SHARE_PREFIX = 'x=1.';
/** A link longer than this many characters is refused (browsers and chat apps truncate long URLs). */
export const SHARE_LINK_MAX = 32000;

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
  if (!/^[A-Za-z0-9_-]*$/.test(text)) throw new Error('base64');
  const b = atob(text.replace(/-/g, '+').replace(/_/g, '/'));
  const out = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) out[i] = b.charCodeAt(i);
  return out;
}

async function pipe(bytes: Uint8Array, stream: CompressionStream | DecompressionStream) {
  const writer = stream.writable.getWriter();
  // Errors surface on the read side; swallow the write-side rejection.
  writer.write(bytes as BufferSource).catch(() => undefined);
  writer.close().catch(() => undefined);
  return new Uint8Array(await new Response(stream.readable).arrayBuffer());
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
  const m = /^x=(\d+)\.(.*)$/.exec(value);
  if (!m) return { ok: false, error: SHARE_BAD_LINK };
  if (m[1] !== '1') return { ok: false, error: SHARE_NEWER };
  let text: string;
  try {
    const bytes = await pipe(fromBase64Url(m[2] ?? ''), new DecompressionStream('deflate-raw'));
    text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch {
    return { ok: false, error: SHARE_BAD_LINK };
  }
  const parsed = parseFileText(text);
  return parsed.ok ? parsed : { ok: false, error: parsed.error };
}
