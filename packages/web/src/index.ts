import { CORE_NAME } from '@tinker/core';
import { ENGINE_NAME } from '@tinker/engine';

export const REGIONS = ['top bar', 'palette', 'canvas', 'results', 'training bar'] as const;
export const WEB_DEPS = [CORE_NAME, ENGINE_NAME] as const;
