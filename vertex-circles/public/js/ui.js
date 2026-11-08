// Small DOM helpers used by the section modules.

export const byId = (id) => document.getElementById(id);

export const prefersReducedMotion =
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

/**
 * Row of toggle buttons.
 * options: { key: { label, color? } }
 * Returns a refresh() function that syncs aria-pressed with isActive(key).
 */
export function segmentedControl(container, options, isActive, onSelect) {
  container.innerHTML = '';
  container.classList.add('segmented');

  for (const [key, opt] of Object.entries(options)) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.dataset.key = key;
    btn.innerHTML = (opt.color ? `<span class="dot" style="background:${opt.color}"></span>` : '') + opt.label;
    btn.addEventListener('click', () => onSelect(key));
    container.appendChild(btn);
  }

  return function refresh() {
    container.querySelectorAll('button').forEach((btn) => {
      btn.setAttribute('aria-pressed', String(isActive(btn.dataset.key)));
    });
  };
}

/** Wires a Play/Pause button to state.playing. Returns a function that refreshes its label. */
export function bindPlayButton(button, state) {
  const refresh = () => {
    button.textContent = state.playing ? 'Pause' : 'Play';
  };
  button.addEventListener('click', () => {
    state.playing = !state.playing;
    refresh();
  });
  refresh();
  return refresh;
}

/** Renders the little grey stat tiles. items: [[label, value], ...] */
export function renderMetrics(el, items) {
  el.innerHTML = items
    .map(([label, value]) => `<div><span>${label}</span><b>${value}</b></div>`)
    .join('');
}

/** Shared arrowhead marker for SVG lines. */
export function arrowMarker(id, color, orient = 'auto') {
  return `<marker id="${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="${orient}">` +
    `<path d="M0,0L10,5L0,10z" fill="${color}"/></marker>`;
}

export const formatRadius = (k, threshold) =>
  Math.abs(k) < threshold ? '∞' : (1 / Math.abs(k)).toFixed(2);
