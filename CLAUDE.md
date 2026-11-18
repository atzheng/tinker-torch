# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

## Git remote and PRs

The remote is `origin` on GitHub (created by the owner, bead `tinker-7x7.1`). Bead work goes through a pull request on a branch named after the bead (`tinker-<id>`, e.g. `tinker-a3f.2`). Do not push bead work straight to `main`. Bead-state-only commits (`.beads/`) may go straight to `main`.

## Multi-agent workflow

One coordinator session runs several implementer and reviewer subagents. The coordinator holds the overall picture and is the only one that merges and changes bead state.

**Roles**

- **Coordinator** (main session):
  - picks beads with `bd ready`, claims them, starts implementers;
  - starts a reviewer per PR;
  - decides on findings, then merges and closes.
  - Do not read whole diffs into the coordinator context: use `gh pr diff --stat` and the reviewer's findings. Read the full diff only for high-risk beads (see below) or when a change touches `CONTRACTS.md`.
- **Implementer** (Sonnet subagent, own git worktree, one bead):
  - builds the bead per "Working a bead" below and opens the PR;
  - never runs `bd`: notes, follow-ups and contract changes go in the PR body.
- **Reviewer** (Sonnet subagent, one PR):
  - reviews independently and does not see the implementer's report;
  - never merges or edits the branch.

**Implementer, before opening the PR**

1. Merge `origin/main` into the branch.
2. Regenerate the lockfile and resolve conflicts.
3. Run `pnpm test && pnpm typecheck && pnpm lint` (and `pnpm build` for web beads).
4. Push the branch and open the PR with `gh pr create`.
5. Include in the PR body:
   - each acceptance point and how it was verified;
   - contract changes or deviations;
   - anything noticed but not done.
6. UI beads must check the result in a browser at the viewports in `CONTRACTS.md` section 10 (laptop and touch tablet), light and dark, and attach at most four screenshots. Touch gestures are checked with `hasTouch: true` and `page.touchscreen` / dispatched pointer events. If no browser is available, say so in the PR. Do not claim the check was done.
   - When no browser extension is connected, use headless Chrome. Install `playwright-core` into a temp dir outside the repo and launch with `executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'`, a `viewport` and `colorScheme`. Check `document.documentElement.scrollWidth <= clientWidth`, then view the screenshots.

**Reviewer brief.** The coordinator gives the reviewer:
- the PR number and bead id;
- the relevant `CONTRACTS.md` sections;
- known risks for the bead;
- what the beads it blocks expect from it (from `bd show` on those beads).

That last item is how problems that span beads get caught.

**Reviewer steps**

1. Check out the branch and run the tests.
2. Try to break the change. Back a suspected bug with a failing test where practical.
3. Post the findings as one PR review comment.
4. Return the same findings to the coordinator. Each finding gives:
   - severity;
   - `file:line`;
   - the failure scenario;
   - whether it is confirmed by a test or only likely.

Keep the reviewer brief self-contained so it does not need to read all of `SPEC.md`.

**Review depth by risk**

- Low (placeholders, styling, config): the coordinator reads the `--stat` and the PR body. No reviewer is needed.
- Normal: one reviewer.
- High (anything in `packages/engine`, `core/analyse.ts`, `core/edit.ts`, the torch fixtures): one reviewer, and the coordinator also reads the diff. Each bead's design notes state its risk.

**After review** the coordinator chooses one of four actions for each finding:
- send it to the *same* implementer via SendMessage, so it keeps its context;
- fix a trivial issue directly;
- file a follow-up bead;
- dismiss it.

Then the coordinator merges with `gh pr merge --merge`, closes the bead with a reason, and starts agents for any newly ready beads.

**Scheduling.** Start a bead only after its dependencies are merged; do not build stand-ins for unmerged work. Start new beads as earlier ones merge rather than in fixed waves.

**Worktrees.**
- Keep an implementer's and a reviewer's worktree until their PR merges. A subagent whose worktree was removed cannot be resumed with SendMessage.
- After a merge, remove the worktrees and delete the local branches.

**Bead state.** Do not silence `bd` output: a failed `bd close` otherwise goes unnoticed.

**Keeping the coordinator lean.**
- Fix environment problems once, before starting agents, and pass known pitfalls in each brief.
- Trim command output (`tail`, `grep`).
- Start a fresh coordinator session every few rounds; beads holds the state.

## Working a bead

Implementer subagents skip the `bd` steps (2, 7 and the note on blocked beads): the coordinator claims and closes, and blockers or follow-ups go in the PR body or the final report.

1. `bd show <id>` and read all of it: description, design, acceptance. Then read the sections of `SPEC.md` and `CONTRACTS.md` it names.
2. `bd update <id> --claim`.
3. Build only what the bead describes. Its design notes list the files it owns; avoid editing files owned by other beads. If you must, keep the change minimal and mention it when closing.
4. `CONTRACTS.md` fixes names, shapes and file paths shared between beads. Follow it exactly. If it is wrong or incomplete, fix it in the same commit and say so when closing — do not silently diverge.
5. Before closing: `pnpm test && pnpm typecheck && pnpm lint` must pass, and every acceptance point must be checked. For UI beads, run `pnpm dev` and look at the result in a browser at the viewports in `CONTRACTS.md` section 10.
6. Commit with the bead id in the message, e.g. `tinker-a3f.2: graph schema and core registry`. Commit only your own files.
7. `bd close <id> --reason "<what was done, anything the next bead should know>"`. File a new bead for anything you found but did not do.

If a bead cannot be finished (missing key, unclear contract, blocked by something not listed as a dependency), do not guess: leave a note with `bd update <id> --append-notes "..."` and stop.


## Build & Test

```bash
pnpm install
pnpm test          # all packages
pnpm typecheck
pnpm lint
pnpm build
pnpm dev           # Vite on :5173
pnpm torch-ref     # opt-in: regenerates engine fixtures from PyTorch (needs uv)
```

These exist once the scaffold bead (`tinker-7x7.2`) is closed.

## Architecture Overview

Read `SPEC.md` (what the tool does) and `CONTRACTS.md` (exact interfaces). In short: one JSON graph is the source of truth. `packages/core` parses and analyses it (widths, issues, parameter counts), samples the synthetic datasets, makes the pure edits behind every canvas gesture and computes the layout. `packages/engine` compiles a graph into jax-js functions and trains it. `packages/web` draws the graph as SVG, the results with canvas 2D, and drives the run loop on the main thread.

- `packages/core` — pure TypeScript; no DOM, no jax-js. Runs in Node and the browser.
- `packages/engine` — the only package that imports `@jax-js/*`. Tests run in Node on the Wasm backend.
- `packages/web` — React app.

## Conventions & Patterns

- TypeScript strict. No `any` without a comment saying why.
- Pure logic (analysis, edits, layout, data, statistics) lives in `packages/core` with unit tests, separate from React components.
- jax-js ownership rules: every engine function documents which arguments it consumes. Use `.ref` for each reuse. No jax array is stored outside `packages/engine`. Engine beads run the leak test.
- Every engine piece and loss is checked against the PyTorch fixtures (CONTRACTS.md section 11) before its bead closes. Never edit a fixture by hand; change `tools/torch-ref/gen.py` and regenerate.
- Randomness only through `core/rng.ts` (and jax random for dropout masks). No `Math.random`.
- The page never crashes on divergence or an invalid graph: invalid graphs show issues and disable training; non-finite losses stop the run and show "diverged".
- Every UI feature works on laptops and touch tablets in landscape, light and dark (CONTRACTS.md section 10). Pointer Events only.
- The tool contains no lesson text. Strings are labels, settings and the plain one-sentence validation messages.
- Tests must not touch the network. The torch script is opt-in and separate.
