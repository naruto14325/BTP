// Speed and prediction: x(t) = v0 t + ½ a t², with ghost figures 1–3 s ahead.

import { byId, segmentedControl, bindPlayButton, renderMetrics, arrowMarker, prefersReducedMotion } from './ui.js';

const DURATION = 10;
const STANDING_HIP = 0.95;
const GROUND_Y = 150;

export function initSpeed({ gaits, paces }) {
  const state = { playing: !prefersReducedMotion, visible: true, el: byId('speed') };
  let gaitKey = 'jog';
  let paceKey = 'constant';
  let t = 0;

  const gait = () => gaits[gaitKey];
  const accel = () => paces[paceKey].accel;

  // Exact motion, including a full stop when decelerating.
  function stateAt(time) {
    const v0 = gait().speed;
    const a = accel();
    if (a < 0 && time >= v0 / -a) {
      return { x: v0 * v0 / (-2 * a), v: 0, a: 0 };
    }
    return { x: v0 * time + 0.5 * a * time * time, v: v0 + a * time, a };
  }

  // Constant-acceleration prediction from the current state.
  function predict(s, dt) {
    if (s.a < 0 && s.v / -s.a < dt) {
      return { x: s.x + s.v * s.v / (-2 * s.a), v: 0 };
    }
    return { x: s.x + s.v * dt + 0.5 * s.a * dt * dt, v: s.v + s.a * dt };
  }

  // Leg-swing phase: integrate step frequency over time (midpoint rule).
  function stridePhase(time) {
    const n = 150;
    let phase = 0;
    for (let i = 0; i < n; i++) {
      const v = stateAt(time * (i + 0.5) / n).v;
      phase += Math.PI * v / gait().stepLength * time / n;
    }
    return phase;
  }

  const refreshGait = segmentedControl(byId('r-gb'), gaits, (k) => k === gaitKey, (k) => { gaitKey = k; t = 0; draw(); });
  const refreshPace = segmentedControl(byId('r-pb'), paces, (k) => k === paceKey, (k) => { paceKey = k; t = 0; draw(); });

  function stickFigure(x0, phase, bob, opacity, color) {
    const hip = [x0, GROUND_Y - (STANDING_HIP + bob) * 70];
    const shoulder = [x0, hip[1] - 40];
    const head = [x0, shoulder[1] - 14];
    const swing = 0.5 * Math.sin(phase);

    const leg = (angle) => {
      const knee = [hip[0] + Math.sin(angle) * 28, hip[1] + Math.cos(angle) * 28];
      const shin = angle - 0.4 * Math.abs(angle);
      const foot = [knee[0] + Math.sin(shin) * 28, knee[1] + Math.cos(shin) * 28];
      return `<polyline points="${hip} ${knee} ${foot}" fill="none"/>`;
    };
    const arm = (angle) =>
      `<line x1="${shoulder[0]}" y1="${shoulder[1] + 4}" x2="${shoulder[0] + Math.sin(angle) * 26}" y2="${shoulder[1] + 4 + Math.cos(angle) * 26}"/>`;

    return `<g stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" opacity="${opacity}">
      ${leg(swing)}${leg(-swing)}${arm(-swing)}${arm(swing)}
      <line x1="${hip[0]}" y1="${hip[1]}" x2="${shoulder[0]}" y2="${shoulder[1]}"/>
      <circle cx="${head[0]}" cy="${head[1]}" r="9" fill="var(--panel)"/>
    </g>`;
  }

  // Small time-series chart: history in blue, prediction dashed.
  function drawPlot(svgId, valueAt, yMax, title, now, isPosition) {
    const W = 320, H = 170, left = 40, right = 10, top = 16, bottom = 30, tMax = 13;
    const X = (time) => left + (W - left - right) * time / tMax;
    const Y = (y) => H - bottom - (H - top - bottom) * y / yMax;

    let s = `<text x="${left}" y="11" style="font-size:12px;fill:var(--ink-2)">${title}</text>`;
    for (let i = 0; i <= tMax; i += 2) {
      s += `<line x1="${X(i)}" y1="${top}" x2="${X(i)}" y2="${H - bottom}" stroke="var(--rule)" stroke-width="0.6"/>`;
      s += `<text x="${X(i)}" y="${H - bottom + 14}" text-anchor="middle" style="font-size:11px;fill:var(--ink-3)">${i}</text>`;
    }
    for (let i = 0; i <= 4; i++) {
      const y = yMax * i / 4;
      s += `<line x1="${left}" y1="${Y(y)}" x2="${W - right}" y2="${Y(y)}" stroke="var(--rule)" stroke-width="0.6"/>`;
      s += `<text x="${left - 5}" y="${Y(y) + 4}" text-anchor="end" style="font-size:11px;fill:var(--ink-3)">${y.toFixed(yMax < 10 ? 1 : 0)}</text>`;
    }
    s += `<text x="${(W + left) / 2}" y="${H - 4}" text-anchor="middle" style="font-size:11px;fill:var(--ink-3)">time (s)</text>`;

    let history = '';
    for (let i = 0; i <= 60; i++) {
      const time = now * i / 60;
      history += `${X(time)},${Y(valueAt(time))} `;
    }
    const current = stateAt(now);
    let future = '';
    for (let i = 0; i <= 30; i++) {
      const dt = 3 * i / 30;
      const q = predict(current, dt);
      future += `${X(now + dt)},${Y(isPosition ? q.x : q.v)} `;
    }
    s += `<polyline points="${history}" fill="none" stroke="var(--blue)" stroke-width="2"/>`;
    s += `<polyline points="${future}" fill="none" stroke="var(--vel)" stroke-width="2" stroke-dasharray="5 4"/>`;

    for (let i = 0; i <= Math.floor(now); i++) {
      s += `<circle cx="${X(i)}" cy="${Y(valueAt(i))}" r="2.5" fill="var(--blue)"/>`;
    }
    s += `<circle cx="${X(now)}" cy="${Y(valueAt(now))}" r="4" fill="var(--blue)"/>`;

    byId(svgId).innerHTML = s;
  }

  function draw() {
    const now = stateAt(t);
    const farthest = predict(stateAt(DURATION), 3);
    const xMax = Math.max(8, farthest.x);
    const toX = (x) => 40 + 600 * x / xMax;
    const phase = stridePhase(t);
    const bob = gait().bob * (1 - Math.cos(2 * phase)) / 2;

    let s = `<defs>${arrowMarker('rar', 'var(--blue)')}</defs>`;
    s += `<line x1="30" y1="${GROUND_Y}" x2="650" y2="${GROUND_Y}" stroke="var(--rule-2)" stroke-width="1.5"/>`;

    const tick = xMax > 40 ? 10 : xMax > 15 ? 5 : 2;
    for (let m = 0; m <= xMax + 0.01; m += tick) {
      s += `<line x1="${toX(m)}" y1="${GROUND_Y}" x2="${toX(m)}" y2="${GROUND_Y + 8}" stroke="var(--ink-3)"/>`;
      s += `<text x="${toX(m)}" y="172" text-anchor="middle" style="font-size:11px;fill:var(--ink-3)">${m} m</text>`;
    }

    for (const ahead of [1, 2, 3]) {
      const q = predict(now, ahead);
      const ghostPhase = phase + Math.PI * q.v / gait().stepLength * ahead;
      s += stickFigure(toX(q.x), ghostPhase, 0, 0.28, 'var(--vel)');
      s += `<text x="${toX(q.x)}" y="30" text-anchor="middle" style="font-size:11px;fill:var(--ink-2)">t+${ahead}s</text>`;
    }

    // one dot per elapsed second
    for (let i = 0; i <= Math.floor(t); i++) {
      s += `<circle cx="${toX(stateAt(i).x)}" cy="${GROUND_Y}" r="3" fill="var(--blue)"/>`;
    }

    s += stickFigure(toX(now.x), phase, bob, 1, 'var(--ink)');
    if (now.v > 0.01) {
      s += `<line x1="${toX(now.x)}" y1="42" x2="${toX(now.x) + now.v * 300 / xMax}" y2="42" stroke="var(--blue)" stroke-width="2" marker-end="url(#rar)"/>`;
    }
    byId('r-svg').innerHTML = s;

    renderMetrics(byId('r-met'), [
      ['Position x (m)', now.x.toFixed(2)],
      ['Hip height y (m)', (STANDING_HIP + bob).toFixed(3)],
      ['Speed v (m/s)', now.v.toFixed(2)],
      ['Accel. a (m/s²)', now.a.toFixed(2)],
      ['Cadence (steps/min)', Math.round(60 * now.v / gait().stepLength)],
    ]);

    byId('r-tb').innerHTML = [0.5, 1, 2, 3].map((ahead) => {
      const q = predict(now, ahead);
      return `<tr><td>t + ${ahead} s</td><td>${(t + ahead).toFixed(2)}</td><td>${q.x.toFixed(2)}</td><td>${q.v.toFixed(2)}</td><td>${(q.x - now.x).toFixed(2)}</td></tr>`;
    }).join('');

    const vMax = Math.max(stateAt(0).v, stateAt(DURATION).v, farthest.v) * 1.15;
    drawPlot('r-px', (time) => stateAt(time).x, Math.ceil(farthest.x / 5) * 5 || 5, 'Position x(t), m', t, true);
    drawPlot('r-pv', (time) => stateAt(time).v, Math.max(1, Math.ceil(vMax)), 'Speed v(t), m/s', t, false);

    byId('r-t').value = t;
    byId('r-tl').textContent = `t = ${t.toFixed(1)} s`;
    refreshGait();
    refreshPace();
  }

  const refreshPlay = bindPlayButton(byId('r-play'), state);
  byId('r-rst').addEventListener('click', () => { t = 0; draw(); });
  byId('r-t').addEventListener('input', (e) => {
    t = Number(e.target.value);
    state.playing = false;
    refreshPlay();
    draw();
  });

  state.tick = (dt) => {
    if (!state.playing) return;
    t += dt;
    if (t > DURATION) t = 0;
    draw();
  };

  draw();
  return state;
}
