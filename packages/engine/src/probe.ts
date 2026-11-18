// Test-only live-buffer probe (docs/spike.md section 4, CONTRACTS.md section 7).
//
// jax-js 0.1.27 exposes no public count of live buffers. The Wasm backend keeps its buffers in a
// private Map<slot, {ptr, size, ref}>. Importing this module wraps Map.prototype.set until the first
// buffer is allocated, recognises that map by its values, keeps a reference to it and restores
// Map.prototype.set. `installProbe()` must run BEFORE jax-js is loaded: the leak test calls it and
// only then does a dynamic `import('../src/index.js')`. The app never calls it, so `liveArrays()`
// is 0 there and Map.prototype.set is never touched.
// Blind spot: jax-js keeps arrays with fewer than 128 elements whose values are all equal (scalars,
// zeros/ones, 1-element arrays) as lazy symbolic constants with no backend buffer, so they are not
// counted. They use no memory, so leaking them is harmless; real buffers they wrap are counted.
// Tied to the pinned jax-js version: if the structure changes the count stays 0 and the leak test
// fails loudly.
let slotMap: Map<unknown, unknown> | null = null;

let installed = false;

/** Test-only. Idempotent. */
export function installProbe(): void {
  if (installed) return;
  installed = true;
  const origSet = Map.prototype.set;
  Map.prototype.set = function (this: Map<unknown, unknown>, k: unknown, v: unknown) {
    if (
      !slotMap &&
      typeof k === 'number' &&
      typeof v === 'object' &&
      v !== null &&
      'ptr' in v &&
      'size' in v &&
      'ref' in v
    ) {
      // eslint-disable-next-line @typescript-eslint/no-this-alias -- the point: capture the private Map
      slotMap = this;
      Map.prototype.set = origSet;
    }
    return origSet.call(this, k, v);
  };
}

/** Number of live backend buffers (0 until the first buffer exists, or when the probe is not installed). */
export function probeLiveArrays(): number {
  return slotMap ? slotMap.size : 0;
}
