import { CSV_LIMITS } from '@tinker/core';

/** Largest file read, in bytes: enough for CSV_LIMITS.rows x CSV_LIMITS.columns of short numbers. */
export const MAX_BYTES = 2_000_000;

export type Loaded = { ok: true; text: string; name: string } | { ok: false; error: string };

/**
 * Fetches CSV text from a URL the user typed, only when called (the dialog calls it on a button
 * press). Failures become one plain sentence. `fetchImpl` is injectable so tests never use the network.
 */
export async function fetchCsvText(
  raw: string,
  fetchImpl: typeof fetch = (...a) => fetch(...a),
): Promise<Loaded> {
  const url = raw.trim();
  let parsed: URL;
  try {
    parsed = new URL(url, window.location.href);
  } catch {
    return { ok: false, error: 'That is not a web address.' };
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    return { ok: false, error: 'The address must start with http:// or https://.' };
  }
  let res: Response;
  try {
    res = await fetchImpl(parsed.href);
  } catch {
    return {
      ok: false,
      error:
        'The file could not be fetched: the server is unreachable or does not allow other sites to read it. Download it and open the file instead.',
    };
  }
  if (!res.ok) return { ok: false, error: `The server answered ${res.status} for that address.` };
  const tooBig: Loaded = {
    ok: false,
    error: `The file is larger than ${MAX_BYTES / 1_000_000} MB; at most ${CSV_LIMITS.rows} rows and ${CSV_LIMITS.columns} columns are supported.`,
  };
  const declared = Number(res.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > MAX_BYTES) {
    void res.body?.cancel();
    return tooBig;
  }
  let text: string;
  try {
    text = await readCapped(res);
  } catch (e) {
    return e === OVER ? tooBig : { ok: false, error: 'The file could not be read.' };
  }
  if (/^\s*<(!doctype|html)/i.test(text)) {
    return { ok: false, error: 'That address returned a web page, not a CSV file.' };
  }
  const last = parsed.pathname.split('/').filter(Boolean).pop();
  return { ok: true, text, name: last ?? parsed.host };
}

const OVER = Symbol('over the size cap');

/** The body as text, read in chunks and abandoned (the request is cancelled) once it passes MAX_BYTES. */
async function readCapped(res: Response): Promise<string> {
  if (!res.body) {
    const t = await res.text();
    if (new TextEncoder().encode(t).length > MAX_BYTES) throw OVER;
    return t;
  }
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.length;
    if (total > MAX_BYTES) {
      void reader.cancel();
      throw OVER;
    }
    chunks.push(value);
  }
  const dec = new TextDecoder();
  return chunks.map((c, i) => dec.decode(c, { stream: i < chunks.length - 1 })).join('');
}

/** Reads a local file chosen by the user. */
export async function readCsvFile(file: File): Promise<Loaded> {
  if (file.size > MAX_BYTES) {
    return {
      ok: false,
      error: `The file is larger than ${MAX_BYTES / 1_000_000} MB; at most ${CSV_LIMITS.rows} rows and ${CSV_LIMITS.columns} columns are supported.`,
    };
  }
  try {
    return { ok: true, text: await file.text(), name: file.name };
  } catch {
    return { ok: false, error: 'The file could not be read.' };
  }
}
