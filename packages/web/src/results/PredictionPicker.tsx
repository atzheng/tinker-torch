import { useStore } from '../state/store.js';
import { plotIndex, predictionNodes } from '../state/plotPrediction.js';
import { clearOverlay } from './resampleStore.js';
import styles from './Results.module.css';

/**
 * The fit plot header's choice among the ŷ pins. Shown only when there are several; the plot draws
 * the picked pin's node (the first by default). The choice is by node (`viewSlice.plotPrediction`).
 */
export function PredictionPicker() {
  const graph = useStore((s) => s.graph);
  const pick = useStore((s) => s.plotPrediction);
  const set = useStore((s) => s.setPlotPrediction);
  const nodes = predictionNodes(graph);
  if (nodes.length < 2) return null;
  const current = nodes[plotIndex(graph, pick)] ?? '';
  return (
    <label className={styles.toggle}>
      <span aria-hidden>{'ŷ'}</span>
      <select
        className={styles.picker}
        data-testid="prediction-picker"
        aria-label="Prediction shown in the plot"
        value={current}
        onChange={(e) => {
          set(e.target.value);
          clearOverlay(); // the copies were of the previous prediction
        }}
      >
        {nodes.map((id) => (
          <option key={id} value={id}>
            {id}
          </option>
        ))}
      </select>
    </label>
  );
}
