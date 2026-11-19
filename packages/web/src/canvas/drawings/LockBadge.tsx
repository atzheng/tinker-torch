import './LockBadge.css';

/** Small padlock for a locked piece, centred on (x, y) in the piece's own units. */
export function LockBadge({ x, y }: { x: number; y: number }) {
  return (
    <g data-testid="lock-badge" transform={`translate(${x} ${y})`} pointerEvents="none">
      <circle r={10} className="tk-lock-bg" />
      <rect x={-5} y={-1} width={10} height={7} rx={1.5} className="tk-lock-body" />
      <path d="M-3 -1V-3.5a3 3 0 0 1 6 0V-1" className="tk-lock-shackle" />
    </g>
  );
}
