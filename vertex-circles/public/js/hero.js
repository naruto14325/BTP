// Hero figure: three points and the circle through them. The middle point is draggable.

import { DEG, vertexCurvature, circumcenter, clamp } from './geometry.js';
import { byId, renderMetrics, arrowMarker, formatRadius } from './ui.js';

const SCALE = 38;
const ORIGIN_X = 20;
const ORIGIN_Y = 280;
const GRID_W = 10;
const GRID_H = 7;
const KEY_STEP = 0.2;

const toX = (x) => ORIGIN_X + x * SCALE;
const toY = (y) => ORIGIN_Y - y * SCALE;

export function initHero() {
  const svg = byId('h-svg');
  const metricsEl = byId('h-met');
  const points = [[1.2, 1.6], [5, 4.4], [9.2, 2.2]];
  let dragging = false;

  function moveMiddle(x, y) {
    points[1] = [clamp(x, 0, GRID_W), clamp(y, 0, GRID_H)];
    draw();
  }

  function draw() {
    const [a, b, c] = points;
    const { phi, k } = vertexCurvature(a, b, c);
    const centre = circumcenter(a, b, c);

    let s = `<defs>
      <clipPath id="hcl"><rect width="420" height="300"/></clipPath>
      ${arrowMarker('hm', 'var(--ink-3)')}
    </defs>`;

    for (let i = 0; i <= GRID_W; i++) {
      s += `<line x1="${toX(i)}" y1="${toY(0)}" x2="${toX(i)}" y2="${toY(GRID_H)}" stroke="var(--rule)" stroke-width="0.6"/>`;
    }
    for (let j = 0; j <= GRID_H; j++) {
      s += `<line x1="${toX(0)}" y1="${toY(j)}" x2="${toX(GRID_W)}" y2="${toY(j)}" stroke="var(--rule)" stroke-width="0.6"/>`;
    }

    s += '<g clip-path="url(#hcl)">';
    if (centre && Math.abs(k) > 1e-3) {
      const r = SCALE / Math.abs(k);
      s += `<circle cx="${toX(centre[0])}" cy="${toY(centre[1])}" r="${r}" fill="none" stroke="var(--accent)" stroke-width="2"/>`;
      s += `<line x1="${toX(b[0])}" y1="${toY(b[1])}" x2="${toX(centre[0])}" y2="${toY(centre[1])}" stroke="var(--accent)" stroke-dasharray="4 3"/>`;
      s += `<circle cx="${toX(centre[0])}" cy="${toY(centre[1])}" r="4" fill="var(--accent)"/>`;
    }

    // Dashed continuation of the first edge, so the turning angle is visible.
    const ext = [b[0] + (b[0] - a[0]) * 0.6, b[1] + (b[1] - a[1]) * 0.6];
    s += `<line x1="${toX(b[0])}" y1="${toY(b[1])}" x2="${toX(ext[0])}" y2="${toY(ext[1])}" stroke="var(--ink-3)" stroke-dasharray="3 3" marker-end="url(#hm)"/>`;
    s += '</g>';

    s += `<polyline points="${points.map((p) => `${toX(p[0])},${toY(p[1])}`).join(' ')}" fill="none" stroke="var(--ink)" stroke-width="2.5"/>`;

    points.forEach((p, i) => {
      const isHandle = i === 1;
      const handleAttrs = isHandle
        ? 'style="cursor:grab" tabindex="0" role="slider" aria-label="Middle point; use arrow keys to move"'
        : '';
      s += `<circle cx="${toX(p[0])}" cy="${toY(p[1])}" r="${isHandle ? 9 : 6}" fill="${isHandle ? 'var(--accent)' : 'var(--panel)'}" stroke="var(--ink)" stroke-width="2" data-i="${i}" ${handleAttrs}/>`;
      s += `<text x="${toX(p[0])}" y="${toY(p[1]) - 14}" text-anchor="middle" style="font-size:13px;fill:var(--ink-2)">γ${i}</text>`;
    });

    s += `<text x="${toX(ext[0]) + 6}" y="${toY(ext[1])}" style="font-size:13px;fill:var(--ink-3)">φ = ${(phi / DEG).toFixed(1)}°</text>`;

    const hadFocus = document.activeElement?.dataset?.i === '1';
    svg.innerHTML = s;
    if (hadFocus) svg.querySelector('[data-i="1"]').focus();

    renderMetrics(metricsEl, [
      ['γ₁ (x, y)', `(${b[0].toFixed(2)}, ${b[1].toFixed(2)})`],
      ['Turning angle φ', `${(phi / DEG).toFixed(1)}°`],
      ['Curvature κ', k.toFixed(3)],
      ['Radius R', formatRadius(k, 1e-3)],
    ]);
  }

  function pointerToGrid(e) {
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const q = pt.matrixTransform(svg.getScreenCTM().inverse());
    return [(q.x - ORIGIN_X) / SCALE, (ORIGIN_Y - q.y) / SCALE];
  }

  svg.style.touchAction = 'none';

  svg.addEventListener('pointerdown', (e) => {
    if (e.target.dataset?.i !== '1') return;
    dragging = true;
    svg.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  svg.addEventListener('pointermove', (e) => {
    if (dragging) moveMiddle(...pointerToGrid(e));
  });
  svg.addEventListener('pointerup', () => { dragging = false; });

  // Keyboard support for the handle (delegated, since draw() replaces the SVG contents).
  const keyMoves = {
    ArrowLeft: [-KEY_STEP, 0],
    ArrowRight: [KEY_STEP, 0],
    ArrowUp: [0, KEY_STEP],
    ArrowDown: [0, -KEY_STEP],
  };
  svg.addEventListener('keydown', (e) => {
    const move = keyMoves[e.key];
    if (!move || e.target.dataset?.i !== '1') return;
    e.preventDefault();
    moveMiddle(points[1][0] + move[0], points[1][1] + move[1]);
  });

  draw();
}
