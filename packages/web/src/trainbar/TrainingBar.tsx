import { useState } from 'react';
import type { Graph } from '@tinker/core';
import { getEngine, useEngine } from '../engine.js';
import { SPEEDS } from '../state/trainSlice.js';
import { useStore } from '../state/store.js';
import styles from './TrainingBar.module.css';

const LR_MIN = 1e-4;
const LR_MAX = 1;
const toPos = (lr: number) => Math.log10(Math.min(LR_MAX, Math.max(LR_MIN, lr)));
const fromPos = (p: number) => Number((10 ** p).toPrecision(3));
const fmtLr = (lr: number) => (lr >= 0.1 ? lr.toFixed(2) : lr.toPrecision(2));

const BATCH_SIZES = [1, 4, 16, 32, 64];
/** Mini-batch sizes below the training-set size, plus the current one when it is not in the list (loaded graph). */
function batchOptions(nTrain: number, current: 'full' | number): number[] {
  const sizes = BATCH_SIZES.filter((n) => n < nTrain);
  if (current !== 'full' && !sizes.includes(current)) sizes.push(current);
  return sizes.sort((a, b) => a - b);
}

/** Learning rate log slider. The value is committed on release or blur, never per movement: lr is baked in at compile. */
function LearningRate() {
  const lr = useStore((s) => s.graph.train.lr);
  const setLearningRate = useStore((s) => s.setLearningRate);
  const [draft, setDraft] = useState<number | null>(null);
  const shown = draft ?? lr;
  const commit = () => {
    if (draft !== null) setLearningRate(draft);
    setDraft(null);
  };
  return (
    <label className={styles.field}>
      <span>lr</span>
      <input
        type="range"
        min={-4}
        max={0}
        step={0.01}
        value={toPos(shown)}
        onChange={(e) => setDraft(fromPos(Number(e.target.value)))}
        onPointerUp={commit}
        onKeyUp={commit}
        onBlur={commit}
        aria-label="Learning rate"
        data-testid="lr-slider"
      />
      <output className={styles.value} data-testid="lr-value">
        {fmtLr(shown)}
      </output>
    </label>
  );
}

export function TrainingBar() {
  useEngine(); // starts loading the engine; the train slice compiles when it is ready
  const s = {
    stepOnce: useStore((x) => x.stepOnce),
    run: useStore((x) => x.run),
    pause: useStore((x) => x.pause),
    reset: useStore((x) => x.reset),
    restoreBest: useStore((x) => x.restoreBest),
    setSpeed: useStore((x) => x.setSpeed),
    engineStatus: useStore((x) => x.engineStatus),
    trainIssue: useStore((x) => x.trainIssue),
    canTrain: useStore((x) => x.canTrain),
    running: useStore((x) => x.running),
    diverged: useStore((x) => x.diverged),
    step: useStore((x) => x.step),
    speed: useStore((x) => x.speed),
    best: useStore((x) => x.best),
    analysis: useStore((x) => x.analysis),
    optimizer: useStore((x) => x.graph.train.optimizer),
    batch: useStore((x) => x.graph.train.batch),
    nTrain: useStore((x) => x.dataset.nTrain),
    setOptimizer: useStore((x) => x.setOptimizer),
    setBatch: useStore((x) => x.setBatch),
  };
  // The component re-renders on every step change, so the dev live-array readout below is fresh.

  const sinceReset = s.step;
  const message = s.engineStatus === 'loading' ? 'Loading the training engine' : s.trainIssue;
  const runnable = s.canTrain && !s.diverged;
  return (
    <div className={styles.bar} data-testid="training-bar">
      <div className={styles.group}>
        <button
          type="button"
          onClick={s.stepOnce}
          disabled={!runnable || s.running}
          data-testid="btn-step"
        >
          Step
        </button>
        {s.running ? (
          <button type="button" className={styles.primary} onClick={s.pause} data-testid="btn-run">
            Pause
          </button>
        ) : (
          <button
            type="button"
            className={styles.primary}
            onClick={s.run}
            disabled={!runnable}
            data-testid="btn-run"
          >
            Run
          </button>
        )}
        <button
          type="button"
          onClick={s.reset}
          disabled={!s.analysis.trainable}
          data-testid="btn-reset"
        >
          Reset
        </button>
        <button
          type="button"
          onClick={s.restoreBest}
          disabled={!s.best || !s.canTrain}
          title={s.best ? `Step ${s.best.step}` : undefined}
          data-testid="btn-restore-best"
        >
          Restore best
        </button>
      </div>
      <div className={styles.group}>
        <LearningRate />
        <label className={styles.field}>
          <select
            value={s.speed}
            onChange={(e) => s.setSpeed(Number(e.target.value))}
            aria-label="Steps per frame"
            data-testid="speed"
          >
            {SPEEDS.map((n) => (
              <option key={n} value={n}>
                {n}/frame
              </option>
            ))}
          </select>
        </label>
        <label className={styles.field}>
          <select
            value={s.optimizer}
            onChange={(e) => s.setOptimizer(e.target.value as Graph['train']['optimizer'])}
            aria-label="Optimizer"
            data-testid="optimizer"
          >
            <option value="sgd">SGD</option>
            <option value="momentum">SGD with momentum</option>
            <option value="adam">Adam</option>
          </select>
        </label>
        <label className={styles.field}>
          <select
            value={String(s.batch)}
            onChange={(e) =>
              s.setBatch(e.target.value === 'full' ? 'full' : Number(e.target.value))
            }
            aria-label="Batch size"
            data-testid="batch"
          >
            <option value="full">Full batch</option>
            {batchOptions(s.nTrain, s.batch).map((n) => (
              <option key={n} value={n}>
                Batch {n}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className={styles.group}>
        <span className={styles.stat}>
          Step <b data-testid="step-count">{sinceReset}</b>
        </span>
        <span className={styles.stat}>
          Parameters <b data-testid="param-count">{s.analysis.totalParams}</b>
        </span>
        {import.meta.env.DEV && (
          <span className={styles.stat} data-testid="live-arrays">
            live {getEngine()?.liveArrays() ?? 0}
          </span>
        )}
      </div>
      {s.diverged && (
        <span
          className={styles.diverged}
          role="status"
          title="The loss is no longer a number. Lower the learning rate or press Reset."
          data-testid="status-diverged"
        >
          Diverged
        </span>
      )}
      {message && !s.diverged && (
        <span className={styles.issue} title={message} data-testid="train-issue">
          {message}
        </span>
      )}
    </div>
  );
}
