import { parseFile, type Preset, type SaveFile } from '@tinker/core';
import linear from './linear.json';
import logistic from './logistic.json';
import sine from './sine.json';
import xor from './xor.json';
import skip from './skip.json';
import scalar from './scalar.json';
import mlp from './mlp.json';
import softmax from './softmax.json';
import iris from './iris.json';
import interceptProb from './intercept-prob.json';
import interceptMean from './intercept-mean.json';
import lecture1Stage0 from './lecture1-stage0.json';
import lecture1Stage1 from './lecture1-stage1.json';
import lecture1Stage2 from './lecture1-stage2.json';
import lecture1Stage3 from './lecture1-stage3.json';
import lecture1Stage4 from './lecture1-stage4.json';
import lecture1Stage5 from './lecture1-stage5.json';

/** Bundled presets by the name used in `?preset=<name>`, in menu order. Imported, never fetched. */
export const PRESET_SOURCES: Record<string, unknown> = {
  linear,
  logistic,
  sine,
  xor,
  skip,
  softmax,
  scalar,
  mlp,
  iris,
  'intercept-prob': interceptProb,
  'intercept-mean': interceptMean,
  'lecture1-stage0': lecture1Stage0,
  'lecture1-stage1': lecture1Stage1,
  'lecture1-stage2': lecture1Stage2,
  'lecture1-stage3': lecture1Stage3,
  'lecture1-stage4': lecture1Stage4,
  'lecture1-stage5': lecture1Stage5,
};

export const PRESET_NAMES: string[] = Object.keys(PRESET_SOURCES);

/** A bundled preset, parsed; null for an unknown name. */
export function bundledPreset(name: string): SaveFile | Preset | null {
  if (!Object.prototype.hasOwnProperty.call(PRESET_SOURCES, name)) return null;
  const r = parseFile(PRESET_SOURCES[name]);
  return r.ok ? r.file : null;
}

/** Title shown in the menu for a bundled preset. */
export function presetTitle(name: string): string {
  const f = bundledPreset(name);
  return f && 'title' in f ? f.title : name;
}
