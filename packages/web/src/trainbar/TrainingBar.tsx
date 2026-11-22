import { useState } from 'react';
import type { Graph } from '@tinker/core';
import { getEngine, useEngine } from '../engine.js';
import { SPEEDS, isMiniBatch } from '../state/trainSlice.js';
import { useUiLabel, useUiShown } from '../state/disclosure.js';
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
  const label = useUiLabel('lr-slider', 'lr');
  const shown = draft ?? lr;
  const commit = () => {
    if (draft !== null) setLearningRate(draft);
    setDraft(null);
  };
  return (
    <label className={styles.field}>
      <span className={styles.lbl} title={label}>
        {label}
      </span>
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
        aria-label={label === 'lr' ? 'Learning rate' : label}
        title={label === 'lr' ? undefined : label}
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
    randomStart: useStore((x) => x.randomStart),
    restoreBest: useStore((x) => x.restoreBest),
    setSpeed: useStore((x) => x.setSpeed),
    engineStatus: useStore((x) => x.engineStatus),
    trainIssue: useStore((x) => x.trainIssue),
    canTrain: useStore((x) => x.canTrain),
    running: useStore((x) => x.running),
    diverged: useStore((x) => x.diverged),
    probIssue: useStore((x) => x.probIssue),
    step: useStore((x) => x.step),
    speed: useStore((x) => x.speed),
    best: useStore((x) => x.best),
    analysis: useStore((x) => x.analysis),
    optimizer: useStore((x) => x.graph.train.optimizer),
    // A stored size at or above nTrain (after a data change) behaves as, and shows as, full batch.
    batch: useStore((x) =>
      isMiniBatch(x.graph.train.batch, x.dataset.nTrain) ? x.graph.train.batch : 'full',
    ),
    nTrain: useStore((x) => x.dataset.nTrain),
    setOptimizer: useStore((x) => x.setOptimizer),
    setBatch: useStore((x) => x.setBatch),
  };
  const sh = {
    step: useUiShown('btn-step'),
    run: useUiShown('btn-run'),
    reset: useUiShown('btn-reset'),
    randomStart: useUiShown('btn-random-start'),
    best: useUiShown('btn-restore-best'),
    lr: useUiShown('lr-slider'),
    speed: useUiShown('speed'),
    optimizer: useUiShown('optimizer'),
    batch: useUiShown('batch'),
    count: useUiShown('step-count'),
    params: useUiShown('param-count'),
  };
  const lbl = {
    step: useUiLabel('btn-step', 'Step'),
    run: useUiLabel('btn-run', 'Run'),
    reset: useUiLabel('btn-reset', 'Reset'),
    randomStart: useUiLabel('btn-random-start', 'New random start'),
    best: useUiLabel('btn-restore-best', 'Restore best'),
    speed: useUiLabel('speed', 'Steps per frame'),
    optimizer: useUiLabel('optimizer', 'Optimizer'),
    batch: useUiLabel('batch', 'Batch size'),
    count: useUiLabel('step-count', 'Step'),
    params: useUiLabel('param-count', 'Parameters'),
  };
  // A group whose controls are all hidden is not drawn.
  const g1 = sh.step || sh.run || sh.reset || sh.randomStart || sh.best;
  const g2 = sh.lr || sh.speed || sh.optimizer || sh.batch;
  const g3 = sh.count || sh.params;
  // The component re-renders on every step change, so the dev live-array readout below is fresh.

  const sinceReset = s.step;
  const message = s.engineStatus === 'loading' ? 'Loading the training engine' : s.trainIssue;
  const runnable = s.canTrain && !s.diverged;
  return (
    <div className={styles.bar} data-testid="training-bar">
      {g1 && (
        <div className={styles.group}>
          {sh.step && (
            <button
              type="button"
              onClick={s.stepOnce}
              disabled={!runnable || s.running}
              aria-label={lbl.step}
              data-testid="btn-step"
            >
              {lbl.step}
            </button>
          )}
          {sh.run &&
            (s.running ? (
              <button
                type="button"
                className={styles.primary}
                onClick={s.pause}
                data-testid="btn-run"
              >
                Pause
              </button>
            ) : (
              <button
                type="button"
                className={styles.primary}
                onClick={s.run}
                disabled={!runnable}
                aria-label={lbl.run}
                data-testid="btn-run"
              >
                {lbl.run}
              </button>
            ))}
          {sh.reset && (
            <button
              type="button"
              onClick={s.reset}
              disabled={!s.analysis.trainable}
              aria-label={lbl.reset}
              data-testid="btn-reset"
            >
              {lbl.reset}
            </button>
          )}
          {sh.randomStart && (
            <button
              type="button"
              onClick={s.randomStart}
              disabled={!s.analysis.trainable}
              aria-label={lbl.randomStart}
              data-testid="btn-random-start"
            >
              {lbl.randomStart}
            </button>
          )}
          {sh.best && (
            <button
              type="button"
              onClick={s.restoreBest}
              disabled={!s.best || !s.canTrain}
              title={s.best ? `Step ${s.best.step}` : undefined}
              aria-label={lbl.best}
              data-testid="btn-restore-best"
            >
              {lbl.best}
            </button>
          )}
        </div>
      )}
      {g2 && (
        <div className={styles.group}>
          {sh.lr && <LearningRate />}
          {sh.speed && (
            <label className={styles.field}>
              <select
                value={s.speed}
                onChange={(e) => s.setSpeed(Number(e.target.value))}
                aria-label={lbl.speed}
                title={lbl.speed}
                data-testid="speed"
              >
                {SPEEDS.map((n) => (
                  <option key={n} value={n}>
                    {n}/frame
                  </option>
                ))}
              </select>
            </label>
          )}
          {sh.optimizer && (
            <label className={styles.field}>
              <select
                value={s.optimizer}
                onChange={(e) => s.setOptimizer(e.target.value as Graph['train']['optimizer'])}
                aria-label={lbl.optimizer}
                title={lbl.optimizer}
                data-testid="optimizer"
              >
                <option value="sgd">SGD</option>
                <option value="momentum">Momentum</option>
                <option value="adam">Adam</option>
              </select>
            </label>
          )}
          {sh.batch && (
            <label className={styles.field}>
              <select
                value={String(s.batch)}
                onChange={(e) =>
                  s.setBatch(e.target.value === 'full' ? 'full' : Number(e.target.value))
                }
                aria-label={lbl.batch}
                title={lbl.batch}
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
          )}
        </div>
      )}
      {g3 && (
        <div className={styles.group}>
          {sh.count && (
            <span className={`${styles.stat} tk-diag`}>
              <span className={styles.lbl}>{lbl.count}</span>{' '}
              <b data-testid="step-count">{sinceReset}</b>
            </span>
          )}
          {sh.params && (
            <span className={`${styles.stat} tk-diag`}>
              <span className={styles.lbl}>{lbl.params}</span>{' '}
              <b data-testid="param-count">{s.analysis.totalParams}</b>
            </span>
          )}
          {import.meta.env.DEV && (
            <span className={`${styles.stat} tk-diag`} data-testid="live-arrays">
              live {getEngine()?.liveArrays() ?? 0}
            </span>
          )}
        </div>
      )}
      {s.diverged && (
        <span
          className={`${styles.diverged} tk-issue`}
          role="status"
          title="The loss is no longer a number. Lower the learning rate or press Reset."
          data-testid="status-diverged"
        >
          Diverged
        </span>
      )}
      {s.probIssue && !s.diverged && (
        <span
          className={`${styles.issue} tk-issue`}
          role="status"
          title={s.probIssue}
          data-testid="status-prob"
        >
          {s.probIssue}
        </span>
      )}
      {message && !s.diverged && !s.probIssue && (
        <span className={`${styles.issue} tk-issue`} title={message} data-testid="train-issue">
          {message}
        </span>
      )}
    </div>
  );
}
