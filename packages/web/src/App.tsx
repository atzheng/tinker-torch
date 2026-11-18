import { REGIONS, WEB_DEPS } from './index.js';

export function App() {
  return (
    <div>
      <p>{WEB_DEPS.join(' + ')}</p>
      {REGIONS.map((name) => (
        <section key={name}>{name}</section>
      ))}
    </div>
  );
}
