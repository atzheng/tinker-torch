// Live-buffer probe. This module must be imported BEFORE '@jax-js/jax': the Wasm backend creates
// its memory when jax-js loads, and the probe has to be in place first.
// jax-js 0.1.27 exposes no public count of live buffers. The Wasm backend keeps its buffers in a
// private Map<slot, {ptr, size, ref}>. We find that Map by wrapping Map.prototype.set until the
// first buffer is allocated, then keep a reference to it. We also capture the WebAssembly.Memory
// the backend creates, so that memory growth can be reported as a second, coarser signal.
let slotMap: Map<unknown, unknown> | null = null;
let wasmMemory: WebAssembly.Memory | null = null;

function installProbe(): void {
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
      slotMap = this;
      Map.prototype.set = origSet;
    }
    return origSet.call(this, k, v);
  };
  const OrigMemory = WebAssembly.Memory;
  // Replace the constructor so that the backend's memory can be observed (see above).
  const Wrapped = function (this: unknown, desc: WebAssembly.MemoryDescriptor) {
    const m = new OrigMemory(desc);
    wasmMemory ??= m;
    return m;
  } as unknown as typeof WebAssembly.Memory;
  Wrapped.prototype = OrigMemory.prototype;
  WebAssembly.Memory = Wrapped;
}

/** Number of live backend buffers (0 until the first buffer exists). */
export function liveArrays(): number {
  return slotMap ? slotMap.size : 0;
}
/** Bytes of Wasm linear memory (never shrinks; a plateau means no leak). */
export function wasmBytes(): number {
  return wasmMemory ? wasmMemory.buffer.byteLength : 0;
}
/** Whether the Wasm memory is shared (SharedArrayBuffer), i.e. the page can run multi-threaded. */
export function wasmShared(): boolean {
  return (
    wasmMemory !== null &&
    typeof SharedArrayBuffer !== 'undefined' &&
    wasmMemory.buffer instanceof SharedArrayBuffer
  );
}

installProbe();
