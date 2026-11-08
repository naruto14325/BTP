// Eyebrow section: a 7-point curve easing between three facial poses.

import { DEG, vertexCurvature, circumcenter } from './geometry.js';
import { byId, segmentedControl, bindPlayButton, renderMetrics, arrowMarker, formatRadius, prefersReducedMotion } from './ui.js';

const CYCLE = 4;           // four one-second transitions
const PLAY_RATE = 0.6;
const VELOCITY_SCALE = 0.35;
const DT = 0.002;          // step for central differences

const toX = (x) => 70 + x * 90;
const toY = (y) => 210 - y * 60;
const easeInOut = (u) => (1 - Math.cos(Math.PI * u)) / 2;

export function initEyebrow({ eyebrowPoses }) {
  const { neutral, surprised, angry } = eyebrowPoses;
  const phases = [
    [neutral, surprised, 'neutral to surprised'],
    [surprised, neutral, 'surprised to neutral'],
    [neutral, angry, 'neutral to angry'],
    [angry, neutral, 'angry to neutral'],
  ];
  const lastIndex = neutral.length - 1;

  const state = { playing: !prefersReducedMotion, visible: true, el: byId('eyebrow') };
  let t = 0;
  let selected = 3;

  function phaseIndex(time) {
    return Math.min(phases.length - 1, Math.floor(time));
  }

  function poseAt(time) {
    time = ((time % CYCLE) + CYCLE) % CYCLE;
    const i = phaseIndex(time);
    const e = easeInOut(time - i);
    const [from, to] = phases[i];
    return from.map((p, k) => [
      p[0] + (to[k][0] - p[0]) * e,
      p[1] + (to[k][1] - p[1]) * e,
    ]);
  }

  // Static background: grid, axis labels and a faint eye outline.
  let background = `<defs>
    ${arrowMarker('ear', 'var(--vel)', 'auto-start-reverse')}
    <clipPath id="ecl"><rect width="680" height="340"/></clipPath>
  </defs>`;
  for (let x = 0; x <= 6; x++) {
    background += `<line x1="${toX(x)}" y1="40" x2="${toX(x)}" y2="250" stroke="var(--rule)" stroke-width="0.6"/>`;
    background += `<text x="${toX(x)}" y="264" text-anchor="middle" style="font-size:11px;fill:var(--ink-3)">${x}</text>`;
  }
  for (let y = -1; y <= 2; y++) {
    background += `<line x1="50" y1="${toY(y)}" x2="630" y2="${toY(y)}" stroke="var(--rule)" stroke-width="0.6"/>`;
    background += `<text x="44" y="${toY(y) + 4}" text-anchor="end" style="font-size:11px;fill:var(--ink-3)">${y}</text>`;
  }
  background += `<ellipse cx="340" cy="290" rx="150" ry="32" fill="none" stroke="var(--rule-2)" stroke-width="1.5"/>`;
  background += `<circle cx="340" cy="290" r="20" fill="none" stroke="var(--rule-2)" stroke-width="1.5"/>`;
  background += `<text x="14" y="22" style="font-size:12px;fill:var(--ink-3)">units: cm, origin at the inner end of the neutral brow</text>`;

  const vertexOptions = {};
  for (let k = 1; k < lastIndex; k++) vertexOptions[k] = { label: `γ${k}` };
  const refreshVertexButtons = segmentedControl(
    byId('e-vb'), vertexOptions,
    (k) => Number(k) === selected,
    (k) => { selected = Number(k); draw(); },
  );

  function draw() {
    const pts = poseAt(t);
    const before = poseAt(t - DT);
    const after = poseAt(t + DT);
    const velocity = pts.map((_, k) => [
      (after[k][0] - before[k][0]) / (2 * DT),
      (after[k][1] - before[k][1]) / (2 * DT),
    ]);

    const [a, b, c] = [pts[selected - 1], pts[selected], pts[selected + 1]];
    let s = background + '<g clip-path="url(#ecl)">';

    const centre = circumcenter(a, b, c);
    if (centre) {
      const r = Math.hypot(centre[0] - b[0], centre[1] - b[1]);
      s += `<circle cx="${toX(centre[0])}" cy="${toY(centre[1])}" r="${r * 75}" fill="none" stroke="var(--accent)" stroke-width="1.6" stroke-dasharray="5 4"/>`;
      s += `<line x1="${toX(b[0])}" y1="${toY(b[1])}" x2="${toX(centre[0])}" y2="${toY(centre[1])}" stroke="var(--accent)"/>`;
      s += `<circle cx="${toX(centre[0])}" cy="${toY(centre[1])}" r="4" fill="var(--accent)"/>`;
    }
    s += '</g>';

    s += `<polyline points="${pts.map((p) => `${toX(p[0])},${toY(p[1])}`).join(' ')}" fill="none" stroke="var(--ink)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>`;

    pts.forEach((p, k) => {
      const [vx, vy] = velocity[k];
      if (Math.hypot(vx, vy) <= 0.05) return;
      s += `<line x1="${toX(p[0])}" y1="${toY(p[1])}" x2="${toX(p[0] + vx * VELOCITY_SCALE)}" y2="${toY(p[1] + vy * VELOCITY_SCALE)}" stroke="var(--vel)" stroke-width="2" marker-end="url(#ear)"/>`;
    });

    pts.forEach((p, k) => {
      const isSel = k === selected;
      s += `<circle cx="${toX(p[0])}" cy="${toY(p[1])}" r="${isSel ? 6 : 4}" fill="${isSel ? 'var(--accent)' : 'var(--panel)'}" stroke="var(--ink)" stroke-width="1.5"/>`;
      s += `<text x="${toX(p[0])}" y="${toY(p[1]) - 12}" text-anchor="middle" style="font-size:11px;fill:var(--ink-2)">γ${k}</text>`;
    });

    byId('e-svg').innerHTML = s;

    // dκ/dt by central difference on the selected vertex
    const { k } = vertexCurvature(a, b, c);
    const kBefore = vertexCurvature(before[selected - 1], before[selected], before[selected + 1]).k;
    const kAfter = vertexCurvature(after[selected - 1], after[selected], after[selected + 1]).k;

    renderMetrics(byId('e-met'), [
      ['Selected vertex', `γ${selected} (${b[0].toFixed(2)}, ${b[1].toFixed(2)})`],
      ['Curvature κ (1/cm)', k.toFixed(3)],
      ['Radius R (cm)', formatRadius(k, 1e-4)],
      ['dκ/dt (1/cm·s)', ((kAfter - kBefore) / (2 * DT)).toFixed(3)],
    ]);

    byId('e-tb').innerHTML = pts.map((p, i) => {
      let angle = '–';
      let curvature = '–';
      if (i > 0 && i < lastIndex) {
        const r = vertexCurvature(pts[i - 1], p, pts[i + 1]);
        angle = `${(r.phi / DEG).toFixed(1)}°`;
        curvature = r.k.toFixed(3);
      }
      const [vx, vy] = velocity[i];
      return `<tr class="${i === selected ? 'sel' : ''}">
        <td>${i}</td><td>${p[0].toFixed(2)}</td><td>${p[1].toFixed(2)}</td>
        <td>${vx.toFixed(2)}</td><td>${vy.toFixed(2)}</td><td>${Math.hypot(vx, vy).toFixed(2)}</td>
        <td>${angle}</td><td>${curvature}</td></tr>`;
    }).join('');

    byId('e-t').value = t.toFixed(2);
    byId('e-tl').textContent = `t = ${t.toFixed(2)} s, ${phases[phaseIndex(t)][2]}`;
    refreshVertexButtons();
  }

  const refreshPlay = bindPlayButton(byId('e-play'), state);
  byId('e-t').addEventListener('input', (e) => {
    t = Number(e.target.value);
    state.playing = false;
    refreshPlay();
    draw();
  });

  state.tick = (dt) => {
    if (!state.playing) return;
    t = (t + dt * PLAY_RATE) % CYCLE;
    draw();
  };

  draw();
  return state;
}
