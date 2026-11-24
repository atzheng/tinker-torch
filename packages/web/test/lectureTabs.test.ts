import { expect, test } from 'vitest';
import { LECTURES, stagePresetName } from '../src/presets/lectures.js';
import { PRESET_SOURCES } from '../src/presets/index.js';

test('every lesson stage lists both the model tab and the loss tab', () => {
  for (const l of LECTURES) {
    for (const k of l.stages) {
      const name = stagePresetName(l.slug, k);
      const src = PRESET_SOURCES[name] as { show?: { ui?: string[] } } | undefined;
      expect(src, name).toBeDefined();
      const ui = src?.show?.ui ?? [];
      expect(ui, `${name} tab-model`).toContain('tab-model');
      expect(ui, `${name} tab-loss`).toContain('tab-loss');
    }
  }
});
