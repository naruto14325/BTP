// Leg, hand and body: joint paths of a stick figure, 16 samples per stride.

import { DEG, vertexCurvature, circumcenter } from './geometry.js';
import { byId, segmentedControl, bindPlayButton, prefersReducedMotion } from './ui.js';

// Segment lengths in metres
const THIGH = 0.45;
const SHIN = 0.45;
const UPPER_ARM = 0.30;
const FOREARM = 0.27;
const TRUNK = 0.50;
const FOOT = 0.16;

const SAMPLES = 16;
const SCALE = 190;

const JOINTS = {
  foot: { label: 'Foot (ankle)', color: 'var(--c-foot)' },
  knee: { label: 'Knee', color: 'var(--c-knee)' },
  hand: { label: 'Hand (wrist)', color: 'var(--c-hand)' },
  hip: { label: 'Hip (body)', color: 'var(--c-hip)' },
};

const VIEWS = {
  tread: { label: 'Treadmill (relative to body)' },
  ground: { label: 'Ground (moving forward)' },
};

export function initLimbs({ gaits, limbModel }) {
  const state = { playing: !prefersReducedMotion, visible: true, el: byId('limbs') };
  let gaitKey = 'walk';
  let view = 'tread';
  const tracked = { foot: true, knee: true, hand: true, hip: true };
  let t = 0;
  let rate = 0.2;

  // Hip height and stride time follow from the leg geometry at mid-stance.
  function strideInfo() {
    const p = { ...gaits[gaitKey], ...limbModel[gaitKey] };
    const kneeAtStance = (p.kneeBase + p.kneeStance) * DEG;
    const legLength = Math.sqrt(THIGH ** 2 + SHIN ** 2 + 2 * THIGH * SHIN * Math.cos(kneeAtStance));
    return {
      p,
      hipHeight: p.flight ? legLength + p.bounce / 2 : legLength - p.bounce / 2,
      strideTime: 4 * legLength * Math.sin(p.thighSwing * DEG) / p.speed,
    };
  }

  function legAt(theta, hip, p) {
    const thigh = p.thighSwing * DEG * Math.sin(theta);
    const kneeBend = (
      p.kneeBase +
      p.kneeSwing * ((1 + Math.cos(theta)) / 2) ** 2 +
      p.kneeStance * ((1 - Math.cos(theta)) / 2) ** 4
    ) * DEG;
    const knee = [hip[0] + THIGH * Math.sin(thigh), hip[1] - THIGH * Math.cos(thigh)];
    const shin = thigh - kneeBend;
    const ankle = [knee[0] + SHIN * Math.sin(shin), knee[1] - SHIN * Math.cos(shin)];
    const toe = [ankle[0] + FOOT * Math.cos(shin), ankle[1] + FOOT * Math.sin(shin)];
    return { knee, ankle, toe };
  }

  function armAt(theta, shoulder, p) {
    const swing = -p.armSwing * DEG * Math.sin(theta);
    const bend = (p.elbowBend + 8 * Math.sin(theta)) * DEG;
    const elbow = [shoulder[0] + UPPER_ARM * Math.sin(swing), shoulder[1] - UPPER_ARM * Math.cos(swing)];
    const wrist = [elbow[0] + FOREARM * Math.sin(swing + bend), elbow[1] - FOREARM * Math.cos(swing + bend)];
    return { elbow, wrist };
  }

  function poseAt(time) {
    const { p, hipHeight, strideTime } = strideInfo();
    const theta = 2 * Math.PI * time / strideTime;
    const bob = (p.flight ? -1 : 1) * p.bounce / 2 * Math.cos(2 * theta);
    const forward = view === 'ground' ? p.speed * time : 0;

    const hip = [forward + 0.02 * Math.sin(2 * theta), hipHeight + bob];
    const lean = p.lean * DEG;
    const shoulder = [hip[0] + TRUNK * Math.sin(lean), hip[1] + TRUNK * Math.cos(lean)];
    const head = [shoulder[0] + 0.2 * Math.sin(lean), shoulder[1] + 0.2 * Math.cos(lean)];

    return {
      hip, shoulder, head,
      rightLeg: legAt(theta, hip, p),
      leftLeg: legAt(theta + Math.PI, hip, p),
      rightArm: armAt(theta, shoulder, p),
      leftArm: armAt(theta + Math.PI, shoulder, p),
    };
  }

  function jointPosition(pose, joint) {
    switch (joint) {
      case 'foot': return pose.rightLeg.ankle;
      case 'knee': return pose.rightLeg.knee;
      case 'hand': return pose.rightArm.wrist;
      default: return pose.hip;
    }
  }

  const refreshGait = segmentedControl(byId('l-gb'), gaits, (k) => k === gaitKey, (k) => { gaitKey = k; t = 0; draw(); });
  const refreshView = segmentedControl(byId('l-vb'), VIEWS, (k) => k === view, (k) => { view = k; t = 0; draw(); });
  const refreshJoints = segmentedControl(byId('l-jb'), JOINTS, (k) => tracked[k], (k) => { tracked[k] = !tracked[k]; draw(); });

  function draw() {
    const { strideTime } = strideInfo();
    const pose = poseAt(t);
    const camera = view === 'ground' ? pose.hip[0] : 0;
    const X = (x) => 340 + (x - camera) * SCALE;
    const Y = (y) => 335 - y * SCALE;
    const sampleTime = (i) => strideTime * i / SAMPLES;

    let s = `<defs><clipPath id="lcl"><rect width="680" height="360"/></clipPath></defs>`;
    s += `<line x1="0" y1="${Y(0)}" x2="680" y2="${Y(0)}" stroke="var(--rule-2)" stroke-width="1.5"/>`;

    if (view === 'ground') {
      const start = Math.floor((camera - 2) / 0.5) * 0.5;
      for (let m = start; m < camera + 2; m += 0.5) {
        s += `<line x1="${X(m)}" y1="${Y(0)}" x2="${X(m)}" y2="${Y(0) + 6}" stroke="var(--ink-3)"/>`;
        s += `<text x="${X(m)}" y="${Y(0) + 18}" text-anchor="middle" style="font-size:11px;fill:var(--ink-3)">${m.toFixed(1)}</text>`;
      }
    }

    const k = Math.round(t / strideTime * SAMPLES);
    const rows = [];
    s += '<g clip-path="url(#lcl)">';

    for (const [joint, { label, color }] of Object.entries(JOINTS)) {
      if (!tracked[joint]) continue;

      // Treadmill: one closed stride. Ground: the last stride up to the current sample.
      const indices = view === 'tread'
        ? Array.from({ length: SAMPLES + 1 }, (_, i) => i)
        : Array.from({ length: SAMPLES + 2 }, (_, i) => k - SAMPLES + i);
      const path = indices.map((i) => jointPosition(poseAt(sampleTime(i)), joint));

      s += `<polyline points="${path.map((q) => `${X(q[0])},${Y(q[1])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="1.2" opacity="0.7"/>`;
      path.forEach((q) => { s += `<circle cx="${X(q[0])}" cy="${Y(q[1])}" r="2.2" fill="${color}" opacity="0.7"/>`; });

      const a = jointPosition(poseAt(sampleTime(k - 1)), joint);
      const b = jointPosition(poseAt(sampleTime(k)), joint);
      const c = jointPosition(poseAt(sampleTime(k + 1)), joint);
      const cv = vertexCurvature(a, b, c);
      const speed = Math.hypot(c[0] - a[0], c[1] - a[1]) / (2 * strideTime / SAMPLES);

      let radius = '–';
      let centre = null;
      if (!cv.degenerate && Math.abs(cv.k) > 0.02) {
        centre = circumcenter(a, b, c);
        radius = Math.abs(1 / cv.k).toFixed(3);
      }

      if (centre) {
        s += `<circle cx="${X(centre[0])}" cy="${Y(centre[1])}" r="${Math.abs(1 / cv.k) * SCALE}" fill="none" stroke="${color}" stroke-width="1.8" stroke-dasharray="5 4"/>`;
        s += `<line x1="${X(b[0])}" y1="${Y(b[1])}" x2="${X(centre[0])}" y2="${Y(centre[1])}" stroke="${color}" stroke-width="0.8"/>`;
        s += `<circle cx="${X(centre[0])}" cy="${Y(centre[1])}" r="3" fill="${color}"/>`;
      }
      for (const q of [a, c]) {
        s += `<circle cx="${X(q[0])}" cy="${Y(q[1])}" r="3.5" fill="var(--panel)" stroke="${color}" stroke-width="1.5"/>`;
      }
      s += `<circle cx="${X(b[0])}" cy="${Y(b[1])}" r="5.5" fill="${color}" stroke="var(--panel)" stroke-width="1.5"/>`;

      rows.push(`<tr>
        <td><span class="dot" style="background:${color}"></span>${label}</td>
        <td>${b[0].toFixed(3)}</td><td>${b[1].toFixed(3)}</td><td>${speed.toFixed(2)}</td>
        <td>${cv.degenerate ? '–' : `${(cv.phi / DEG).toFixed(1)}°`}</td>
        <td>${cv.degenerate ? 'planted' : cv.k.toFixed(2)}</td>
        <td>${cv.degenerate ? '–' : radius}</td></tr>`);
    }
    s += '</g>';

    // Stick figure: far side in grey, near side in ink.
    const bone = (p, q, width, color) =>
      `<line x1="${X(p[0])}" y1="${Y(p[1])}" x2="${X(q[0])}" y2="${Y(q[1])}" stroke="${color}" stroke-width="${width}" stroke-linecap="round"/>`;
    const side = (leg, arm, color) =>
      bone(pose.hip, leg.knee, 5, color) + bone(leg.knee, leg.ankle, 5, color) + bone(leg.ankle, leg.toe, 4, color) +
      bone(pose.shoulder, arm.elbow, 4, color) + bone(arm.elbow, arm.wrist, 4, color);

    s += side(pose.leftLeg, pose.leftArm, 'var(--rule-2)');
    s += bone(pose.hip, pose.shoulder, 6, 'var(--ink)');
    s += side(pose.rightLeg, pose.rightArm, 'var(--ink)');
    s += `<circle cx="${X(pose.head[0])}" cy="${Y(pose.head[1])}" r="${0.11 * SCALE}" fill="var(--panel)" stroke="var(--ink)" stroke-width="3"/>`;

    const vertexInStride = ((k % SAMPLES) + SAMPLES) % SAMPLES;
    s += `<text x="12" y="20" style="font-size:12px;fill:var(--ink-2)">vertex γ${vertexInStride} of ${SAMPLES} per stride, t = ${t.toFixed(2)} s, stride time ${strideTime.toFixed(2)} s</text>`;

    byId('l-svg').innerHTML = s;
    byId('l-tb').innerHTML = rows.join('') || '<tr><td colspan="7">Choose at least one joint to track.</td></tr>';
    byId('l-info').textContent = view === 'tread'
      ? 'Treadmill view: each joint is measured relative to the moving body, so its path is a closed loop repeated every stride.'
      : 'Ground view: real paths over the ground. A planted foot stays still, so neighbouring vertices coincide and no circle exists there.';
    byId('l-spl').textContent = `${rate.toFixed(2)}×`;

    refreshGait();
    refreshView();
    refreshJoints();
  }

  const refreshPlay = bindPlayButton(byId('l-play'), state);

  function stepVertex(delta) {
    state.playing = false;
    refreshPlay();
    const { strideTime } = strideInfo();
    const next = Math.round(t / strideTime * SAMPLES) + delta;
    t = Math.max(0, next * strideTime / SAMPLES);
    draw();
  }

  byId('l-prev').addEventListener('click', () => stepVertex(-1));
  byId('l-next').addEventListener('click', () => stepVertex(1));
  byId('l-sp').addEventListener('input', (e) => { rate = Number(e.target.value); draw(); });

  state.tick = (dt) => {
    if (!state.playing) return;
    t += dt * rate;
    if (t > 60) t = 0;
    draw();
  };

  draw();
  return state;
}
