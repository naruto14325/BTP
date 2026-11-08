// Hip path by gait: side-view hip trajectory sampled 10 times per step.

import { DEG, vertexCurvature, circumcenter } from './geometry.js';
import { byId, segmentedControl, bindPlayButton, renderMetrics } from './ui.js';

const G = 9.81;
const SAMPLES_PER_STEP = 10;
const VERTEX_COUNT = 2 * SAMPLES_PER_STEP + 1;   // two steps, 21 points
const LAST_INNER = VERTEX_COUNT - 2;              // last vertex that has two neighbours
const STEP_INTERVAL = 0.45;                       // seconds between vertices when playing

const curvatureColor = (k) =>
  Math.abs(k) > 4 ? 'var(--warn)' : k < 0 ? 'var(--neg)' : 'var(--pos)';

export function initHip({ gaits, hipModel }) {
  const state = { playing: false, visible: true, el: byId('hip') };
  let gaitKey = 'walk';
  let selected = 5;
  let showAll = false;
  let sinceStep = 0;

  const params = () => ({ ...gaits[gaitKey], ...hipModel[gaitKey] });

  /**
   * Hip position at time t.
   * Walking: inverted pendulum over a straight leg.
   * Jogging/running: sine dip during stance, parabola during flight.
   */
  function hipAt(t) {
    const p = params();
    const stepTime = p.stepLength / p.speed;
    const x = p.speed * t;

    if (gaitKey === 'walk') {
      const u = (x % p.stepLength) - p.stepLength / 2;
      return { pt: [x, Math.sqrt(p.legLength ** 2 - u * u)], foot: x - u, stance: true };
    }

    const s = (((t % stepTime) + stepTime) % stepTime) / stepTime;
    const stanceTime = p.stanceFraction * stepTime;
    const flightTime = (1 - p.stanceFraction) * stepTime;
    // dip chosen so vertical momentum balances the flight phase
    const dip = G * flightTime * stanceTime / (2 * Math.PI);

    if (s < p.stanceFraction) {
      return {
        pt: [x, p.standingHeight - dip * Math.sin(Math.PI * s / p.stanceFraction)],
        foot: x - (s - p.stanceFraction / 2) * stepTime * p.speed,
        stance: true,
      };
    }
    const tau = (s - p.stanceFraction) * stepTime;
    return { pt: [x, p.standingHeight + G / 2 * tau * (flightTime - tau)], stance: false };
  }

  const refreshGait = segmentedControl(byId('p-gb'), gaits, (k) => k === gaitKey, (k) => { gaitKey = k; draw(); });

  function describe(k) {
    if (gaitKey === 'walk') {
      return Math.abs(k) > 4
        ? 'Step-to-step transition: the path has a corner, so the circle is tiny. This is where walking loses energy as the new leg strikes the ground.'
        : 'Single stance: the hip vaults over a straight leg like an inverted pendulum. The circle’s centre sits on the foot, and R = 0.95 m is the leg length.';
    }
    if (Math.abs(k) < 0.2) {
      return 'Touchdown or take-off: the path is nearly straight here, so the circle is very large.';
    }
    if (k > 0) {
      return 'Stance: the leg compresses like a spring, so the hip dips and the circle’s centre is above the path.';
    }
    const kFlight = G / params().speed ** 2;
    return `Flight: the body is in free fall, so the path is a parabola with κ ≈ g ⁄ v² = ${kFlight.toFixed(2)} 1/m (R ≈ ${(1 / kFlight).toFixed(2)} m).`;
  }

  function draw() {
    const p = params();
    const stepTime = p.stepLength / p.speed;
    const samples = Array.from({ length: VERTEX_COUNT }, (_, i) => hipAt(stepTime * i / SAMPLES_PER_STEP));
    const pts = samples.map((h) => h.pt);
    const last = VERTEX_COUNT - 1;

    const width = 2 * p.stepLength;
    const scale = Math.min(620 / width, 320 / 1.15);   // true scale: same in x and y
    const X = (x) => 30 + x * scale;
    const Y = (y) => 330 - y * scale;

    let s = `<defs><clipPath id="pcl"><rect width="680" height="350"/></clipPath></defs>`;
    s += `<line x1="20" y1="${Y(0)}" x2="660" y2="${Y(0)}" stroke="var(--rule-2)" stroke-width="1.5"/>`;
    for (let m = 0; m <= width + 1e-6; m += 0.25) {
      s += `<line x1="${X(m)}" y1="${Y(0)}" x2="${X(m)}" y2="${Y(0) + 5}" stroke="var(--ink-3)"/>`;
      const onHalfMetre = Math.abs(m * 2 - Math.round(m * 2)) < 1e-6;
      if (onHalfMetre) {
        s += `<text x="${X(m)}" y="${Y(0) + 17}" text-anchor="middle" style="font-size:11px;fill:var(--ink-3)">${m.toFixed(1)} m</text>`;
      }
    }

    const curv = [];
    for (let i = 1; i < last; i++) curv[i] = vertexCurvature(pts[i - 1], pts[i], pts[i + 1]);

    const circleAt = (i, highlight) => {
      const c = circumcenter(pts[i - 1], pts[i], pts[i + 1]);
      if (!c) return '';
      const r = Math.hypot(c[0] - pts[i][0], c[1] - pts[i][1]);
      const color = curvatureColor(curv[i].k);
      let out = `<circle cx="${X(c[0])}" cy="${Y(c[1])}" r="${r * scale}" fill="none" stroke="${color}" stroke-width="${highlight ? 2 : 0.8}" opacity="${highlight ? 1 : 0.35}" ${highlight ? 'stroke-dasharray="6 4"' : ''}/>`;
      if (highlight) {
        out += `<line x1="${X(pts[i][0])}" y1="${Y(pts[i][1])}" x2="${X(c[0])}" y2="${Y(c[1])}" stroke="${color}"/>`;
        out += `<circle cx="${X(c[0])}" cy="${Y(c[1])}" r="4" fill="${color}"/>`;
      }
      return out;
    };

    s += '<g clip-path="url(#pcl)">';
    if (showAll) {
      for (let i = 1; i < last; i++) if (i !== selected) s += circleAt(i, false);
    }
    s += circleAt(selected, true) + '</g>';

    // Simple figure standing at the selected sample
    const cur = samples[selected];
    const hx = X(cur.pt[0]);
    const hy = Y(cur.pt[1]);
    const limb = 'fill="none" stroke="var(--ink-2)" stroke-width="3" stroke-linecap="round"';
    if (cur.stance) {
      s += `<line x1="${hx}" y1="${hy}" x2="${X(cur.foot)}" y2="${Y(0)}" ${limb}/>`;
    } else {
      s += `<polyline points="${hx},${hy} ${hx + 0.25 * scale},${hy + 0.3 * scale} ${hx + 0.05 * scale},${hy + 0.55 * scale}" ${limb}/>`;
      s += `<polyline points="${hx},${hy} ${hx - 0.2 * scale},${hy + 0.35 * scale} ${hx - 0.4 * scale},${hy + 0.5 * scale}" ${limb}/>`;
    }
    s += `<line x1="${hx}" y1="${hy}" x2="${hx}" y2="${hy - 0.5 * scale}" ${limb}/>`;
    s += `<circle cx="${hx}" cy="${hy - 0.62 * scale}" r="${0.1 * scale}" fill="var(--panel)" stroke="var(--ink-2)" stroke-width="2.5"/>`;

    // Smooth path in the background, sampled polygon on top
    let smooth = '';
    for (let i = 0; i <= 200; i++) {
      const q = hipAt(stepTime * 2 * i / 200).pt;
      smooth += `${X(q[0])},${Y(q[1])} `;
    }
    s += `<polyline points="${smooth}" fill="none" stroke="var(--rule-2)"/>`;
    s += `<polyline points="${pts.map((q) => `${X(q[0])},${Y(q[1])}`).join(' ')}" fill="none" stroke="var(--ink)" stroke-width="1.5"/>`;

    pts.forEach((q, i) => {
      const color = i > 0 && i < last ? curvatureColor(curv[i].k) : 'var(--ink-3)';
      const isSel = i === selected;
      s += `<circle cx="${X(q[0])}" cy="${Y(q[1])}" r="${isSel ? 5.5 : 3}" fill="${isSel ? color : 'var(--panel)'}" stroke="${color}" stroke-width="1.5"/>`;
    });
    [selected - 1, selected, selected + 1].forEach((i, j) => {
      s += `<text x="${X(pts[i][0])}" y="${Y(pts[i][1]) - 12 - (j === 1 ? 6 : 0)}" text-anchor="middle" style="font-size:11px;fill:var(--ink-2)">γ${i}</text>`;
    });

    byId('p-svg').innerHTML = s;

    const { phi, k } = curv[selected];
    const centre = circumcenter(pts[selected - 1], pts[selected], pts[selected + 1]);
    renderMetrics(byId('p-met'), [
      ['Vertex γₖ (x, y) m', `(${cur.pt[0].toFixed(3)}, ${cur.pt[1].toFixed(3)})`],
      ['Turning angle φ', `${(phi / DEG).toFixed(2)}°`],
      ['Curvature κ (1/m)', k.toFixed(3)],
      ['Radius R (m)', Math.abs(k) < 0.02 ? '≈ ∞' : (1 / Math.abs(k)).toFixed(2)],
      ['Centre (m)', centre ? `(${centre[0].toFixed(2)}, ${centre[1].toFixed(2)})` : 'none'],
    ]);
    byId('p-note').textContent = describe(k);

    drawCurvatureBars(curv, last);

    byId('p-v').value = selected;
    byId('p-vl').textContent = `γ${selected}`;
    byId('p-all').setAttribute('aria-pressed', String(showAll));
    refreshGait();
  }

  function drawCurvatureBars(curv, last) {
    const barWidth = 600 / LAST_INNER;
    const kMax = gaitKey === 'walk' ? 9 : 2.2;
    const zero = 85;
    const pxPerUnit = 70 / kMax;

    let s = `<line x1="40" y1="${zero}" x2="660" y2="${zero}" stroke="var(--rule-2)"/>`;
    s += `<text x="40" y="12" style="font-size:12px;fill:var(--ink-2)">Discrete curvature κₖ at each vertex (1/m); tap a bar to select it</text>`;
    for (const v of [kMax, kMax / 2, -kMax / 2, -kMax]) {
      s += `<text x="34" y="${zero - v * pxPerUnit + 4}" text-anchor="end" style="font-size:11px;fill:var(--ink-3)">${v.toFixed(1)}</text>`;
      s += `<line x1="40" y1="${zero - v * pxPerUnit}" x2="660" y2="${zero - v * pxPerUnit}" stroke="var(--rule)" stroke-width="0.6"/>`;
    }
    for (let i = 1; i < last; i++) {
      const k = curv[i].k;
      const v = Math.max(-kMax, Math.min(kMax, k));
      const x = 40 + (i - 0.5) * barWidth + 4;
      const top = v > 0 ? zero - v * pxPerUnit : zero;
      const isSel = i === selected;
      s += `<rect x="${x}" y="${top}" width="${barWidth - 8}" height="${Math.max(1, Math.abs(v) * pxPerUnit)}" fill="${curvatureColor(k)}" opacity="${isSel ? 1 : 0.5}" rx="2" style="cursor:pointer" data-i="${i}"/>`;
      s += `<text x="${x + (barWidth - 8) / 2}" y="160" text-anchor="middle" style="font-size:11px;fill:${isSel ? 'var(--ink)' : 'var(--ink-3)'}">${i}</text>`;
    }
    byId('p-kb').innerHTML = s;
  }

  byId('p-kb').addEventListener('click', (e) => {
    const i = e.target.dataset?.i;
    if (i) { selected = Number(i); draw(); }
  });

  bindPlayButton(byId('p-play'), state);
  byId('p-v').addEventListener('input', (e) => { selected = Number(e.target.value); draw(); });
  byId('p-all').addEventListener('click', () => { showAll = !showAll; draw(); });

  state.tick = (dt) => {
    if (!state.playing) return;
    sinceStep += dt;
    if (sinceStep > STEP_INTERVAL) {
      sinceStep = 0;
      selected = selected >= LAST_INNER ? 1 : selected + 1;
      draw();
    }
  };

  draw();
  return state;
}
