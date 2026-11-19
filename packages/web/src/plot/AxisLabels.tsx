import { memo } from 'react';
import { MARGINS } from './draw.js';
import { TexText } from '../tex/Tex.js';
import './axisLabels.css';

/**
 * The two axis titles of a canvas plot, as an HTML overlay in the plot's `position: relative` box
 * (the canvas cannot typeset maths). `x` sits at the right end of the x axis, `y` runs up the left
 * side; both take `$...$` text. Memoised on the strings, so a redraw per frame never touches them.
 */
export const AxisLabels = memo(function AxisLabels(p: { x: string; y: string }) {
  return (
    <>
      <div className="tk-axis-x" style={{ right: MARGINS.r }} data-testid="axis-x">
        <TexText>{p.x}</TexText>
      </div>
      <div className="tk-axis-y" style={{ top: MARGINS.t + 2 }} data-testid="axis-y">
        <TexText>{p.y}</TexText>
      </div>
    </>
  );
});
