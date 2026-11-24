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
import { lecturePresetNames } from './lectures.js';
import lecture1Stage0 from './lecture1-stage0.json';
import lecture1Stage1 from './lecture1-stage1.json';
import lecture1Stage2 from './lecture1-stage2.json';
import lecture1Stage3 from './lecture1-stage3.json';
import lecture1Stage4 from './lecture1-stage4.json';
import lecture1Stage5 from './lecture1-stage5.json';
import biasVarianceStage1 from './bias-variance-stage1.json';
import biasVarianceStage2 from './bias-variance-stage2.json';
import biasVarianceStage3 from './bias-variance-stage3.json';
import treeClassStage1 from './tree-class-stage1.json';
import treeClassStage2 from './tree-class-stage2.json';
import treeClassStage3 from './tree-class-stage3.json';

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
  'bias-variance-stage1': biasVarianceStage1,
  'bias-variance-stage2': biasVarianceStage2,
  'bias-variance-stage3': biasVarianceStage3,
  'tree-class-stage1': treeClassStage1,
  'tree-class-stage2': treeClassStage2,
  'tree-class-stage3': treeClassStage3,
};

export const PRESET_NAMES: string[] = Object.keys(PRESET_SOURCES);

/** Names in the top bar Presets menu: every bundled preset except lecture stages (those open from `#/lectures`). */
export const MENU_PRESET_NAMES: string[] = PRESET_NAMES.filter((n) => !lecturePresetNames().has(n));

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
