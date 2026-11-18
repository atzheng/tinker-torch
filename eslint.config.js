import js from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: [
      '**/dist/**',
      '**/coverage/**',
      '**/.wrangler/**',
      '.beads/**',
      '.claude/**',
      '**/node_modules/**',
    ],
  },
  js.configs.recommended,
  tseslint.configs.strict,
  {
    // Throwaway spike code (tinker-e1h.1): indexing jax-js arrays and tuples is clearer with `!`.
    files: ['packages/engine/spike/**'],
    rules: {
      '@typescript-eslint/no-non-null-assertion': 'off',
      '@typescript-eslint/no-this-alias': 'off',
    },
  },
  {
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
    },
  },
);
