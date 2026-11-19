import { NodeShell } from './NodeShell.js';
import type { WebPiece } from './types.js';

const NoSettings = () => <p className="tk-muted">No settings.</p>;
const noNeuron = (name: string) => () => <g data-placeholder={`${name}-neuron`} />;

// Minimal drawings; the weight-wire drawing comes in 9io.6.
export const columnPiece: WebPiece<'column'> = {
  Layer: ({ node, w, h, scale }) => (
    <NodeShell
      piece="column"
      role="input"
      w={w}
      h={h}
      scale={scale}
      name="Column"
      glyph={`x_{${node.name.replace(/^x/, '')}}`}
      inPorts={0}
      outPort
    />
  ),
  Neuron: noNeuron('column'),
  Fields: () => <NoSettings />,
};
export const scalewPiece: WebPiece<'scalew'> = {
  Layer: ({ w, h, scale }) => (
    <NodeShell
      piece="scalew"
      w={w}
      h={h}
      scale={scale}
      name="Weight"
      glyph="w"
      inPorts={1}
      outPort
    />
  ),
  Neuron: noNeuron('scalew'),
  Fields: () => <NoSettings />,
};
export const biasPiece: WebPiece<'bias'> = {
  Layer: ({ w, h, scale }) => (
    <NodeShell piece="bias" w={w} h={h} scale={scale} name="Bias" glyph="+b" inPorts={1} outPort />
  ),
  Neuron: noNeuron('bias'),
  Fields: () => <NoSettings />,
};
export const sumPiece: WebPiece<'sum'> = {
  Layer: ({ w, h, scale }) => (
    <NodeShell
      piece="sum"
      w={w}
      h={h}
      scale={scale}
      name="Sum"
      glyph="\\Sigma"
      inPorts={1}
      outPort
    />
  ),
  Neuron: noNeuron('sum'),
  Fields: () => <NoSettings />,
};
