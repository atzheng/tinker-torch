/**
 * Lecture grouping for the landing page (CONTRACTS.md section 12). Metadata only: the preset names
 * are keys of `PRESET_SOURCES`. Lecture and stage slugs form the URL routes and must stay stable.
 */
export type LectureInfo = {
  /** URL slug: `#/<slug>/stage<k>`. */
  slug: string;
  /** Label on the landing page. */
  title: string;
  /** Stage numbers in order, each loading the preset `<slug>-stage<k>`. */
  stages: number[];
};

export const LECTURES: LectureInfo[] = [
  { slug: 'lecture1', title: 'Lecture 1', stages: [0, 1, 2, 3, 4, 5] },
  { slug: 'bias-variance', title: 'Bias and variance', stages: [1, 2, 3] },
];

/** The bundled preset name of a lecture stage. */
export function stagePresetName(slug: string, stage: number): string {
  return `${slug}-stage${stage}`;
}

/** Every preset name that belongs to a lecture (these are not in the presets menu). */
export function lecturePresetNames(): Set<string> {
  return new Set(LECTURES.flatMap((l) => l.stages.map((k) => stagePresetName(l.slug, k))));
}
