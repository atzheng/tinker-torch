/**
 * The pieces of graph v2 that only compute: links, the pieces of a loss, and the weight taps and
 * norms of a penalty. Each is the shared NodeShell with a name and a TeX glyph; the roles (ŷ, L) are
 * pins on a node and are drawn by the canvas, never by a piece.
 */
import type { FC } from 'react';
import { LINK_FNS, NORM_KINDS, type LinkFn, type NormKind, type PieceType } from '@tinker/core';
import { NodeShell } from './NodeShell.js';
import { NumberField, SelectField, editSetting } from './fields.js';
import type { LayerProps, WebPiece } from './types.js';

const NoSettings = () => <p className="tk-muted">No settings.</p>;
const noNeuron = (name: string) => () => <g data-placeholder={`${name}-neuron`} />;

type Simple = {
  piece: string;
  name: string;
  glyph: string;
  inPorts: number;
  secondary?: string[];
};

/** A piece whose drawing is fixed: the shell with a name, a glyph and the number of input ports. */
function simple<T extends PieceType>(s: Simple): FC<LayerProps<T>> {
  return function Layer({ w, h, scale }: LayerProps<T>) {
    return (
      <NodeShell
        piece={s.piece}
        w={w}
        h={h}
        scale={scale}
        name={s.name}
        glyph={s.glyph}
        secondary={s.secondary}
        inPorts={s.inPorts}
        outPort
      />
    );
  };
}

export const LINK_LABEL: Record<LinkFn, string> = { sigmoid: 'Sigmoid', softmax: 'Softmax' };
const LINK_GLYPH: Record<LinkFn, string> = {
  sigmoid: '\\sigma(z)',
  softmax: '\\mathrm{softmax}',
};

export const linkPiece: WebPiece<'link'> = {
  Layer: ({ node, w, h, scale }) => (
    <NodeShell
      piece="link"
      w={w}
      h={h}
      scale={scale}
      name={LINK_LABEL[node.fn]}
      glyph={LINK_GLYPH[node.fn]}
      inPorts={1}
      outPort
    />
  ),
  Neuron: noNeuron('link'),
  Fields: ({ node }) => (
    <SelectField
      label="Link"
      testid="field-link"
      value={node.fn}
      options={LINK_FNS.map((f) => [f, LINK_LABEL[f]] as const)}
      onChange={(v) => editSetting(node.id, 'fn', v)}
    />
  ),
};

export const subtractPiece: WebPiece<'subtract'> = {
  Layer: simple({ piece: 'subtract', name: 'Subtract', glyph: 'a-b', inPorts: 2 }),
  Neuron: noNeuron('subtract'),
  Fields: NoSettings,
};

export const squarePiece: WebPiece<'square'> = {
  Layer: simple({ piece: 'square', name: 'Square', glyph: 'x^2', inPorts: 1 }),
  Neuron: noNeuron('square'),
  Fields: NoSettings,
};

export const absPiece: WebPiece<'abs'> = {
  Layer: simple({ piece: 'abs', name: 'Absolute', glyph: '|x|', inPorts: 1 }),
  Neuron: noNeuron('abs'),
  Fields: NoSettings,
};

export const huberPiece: WebPiece<'huber'> = {
  Layer: ({ node, w, h, scale }) => (
    <NodeShell
      piece="huber"
      w={w}
      h={h}
      scale={scale}
      name="Huber"
      glyph="\mathrm{H}_\delta"
      secondary={[`$\\delta=${node.delta}$`]}
      inPorts={1}
      outPort
    />
  ),
  Neuron: noNeuron('huber'),
  Fields: ({ node }) => (
    <NumberField
      label="Threshold $\delta$"
      testid="field-delta"
      value={node.delta}
      min={0}
      step={0.1}
      onCommit={(v) => editSetting(node.id, 'delta', v)}
    />
  ),
};

export const logLossPiece: WebPiece<'log_loss'> = {
  Layer: simple({
    piece: 'log_loss',
    name: 'Log loss',
    glyph: '-\\log p',
    inPorts: 2,
  }),
  Neuron: noNeuron('log_loss'),
  Fields: NoSettings,
};

export const crossEntropyPiece: WebPiece<'cross_entropy'> = {
  Layer: simple({
    piece: 'cross_entropy',
    name: 'Cross-entropy',
    glyph: '-\\log p_y',
    inPorts: 2,
  }),
  Neuron: noNeuron('cross_entropy'),
  Fields: NoSettings,
};

export const meanPiece: WebPiece<'mean'> = {
  Layer: simple({
    piece: 'mean',
    name: 'Mean',
    glyph: '\\bar{x}',
    inPorts: 1,
    secondary: ['one number'],
  }),
  Neuron: noNeuron('mean'),
  Fields: NoSettings,
};

export const scalePiece: WebPiece<'scale'> = {
  Layer: ({ node, w, h, scale }) => (
    <NodeShell
      piece="scale"
      w={w}
      h={h}
      scale={scale}
      name="Scale"
      glyph="c\,x"
      secondary={[`$c=${node.c}$`]}
      inPorts={1}
      outPort
    />
  ),
  Neuron: noNeuron('scale'),
  Fields: ({ node }) => (
    <NumberField
      label="Factor $c$"
      testid="field-c"
      value={node.c}
      step={0.01}
      onCommit={(v) => editSetting(node.id, 'c', v)}
    />
  ),
};

export const weightsPiece: WebPiece<'weights'> = {
  Layer: ({ node, w, h, scale }) => (
    <NodeShell
      piece="weights"
      w={w}
      h={h}
      scale={scale}
      name="Weights"
      glyph="W"
      secondary={[`of ${node.of}`]}
      inPorts={0}
      outPort
    />
  ),
  Neuron: noNeuron('weights'),
  Fields: ({ node }) => (
    <p className="tk-muted" data-testid="weights-of">
      Reads the weights of <code>{node.of}</code>.
    </p>
  ),
};

export const NORM_LABEL: Record<NormKind, string> = { l2: 'L2 norm', l1: 'L1 norm' };
const NORM_GLYPH: Record<NormKind, string> = {
  l2: '\\sum w^2',
  l1: '\\sum|w|',
};

export const normPiece: WebPiece<'norm'> = {
  Layer: ({ node, w, h, scale }) => (
    <NodeShell
      piece="norm"
      w={w}
      h={h}
      scale={scale}
      name={NORM_LABEL[node.kind]}
      glyph={NORM_GLYPH[node.kind]}
      inPorts={1}
      outPort
    />
  ),
  Neuron: noNeuron('norm'),
  Fields: ({ node }) => (
    <SelectField
      label="Norm"
      testid="field-norm"
      value={node.kind}
      options={NORM_KINDS.map((k) => [k, NORM_LABEL[k]] as const)}
      onChange={(v) => editSetting(node.id, 'kind', v)}
    />
  ),
};
