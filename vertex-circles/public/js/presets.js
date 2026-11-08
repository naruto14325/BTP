// Model parameters. Keep in sync with data/presets.json (used by the server API).

export default {
  "eyebrowPoses": {
    "neutral":   [[0, 0],     [1, 0.35],   [2, 0.6],   [3, 0.75], [4, 0.8],  [5, 0.6],  [6, 0.2]],
    "surprised": [[0, 0.5],   [1, 1.15],   [2, 1.6],   [3, 1.85], [4, 1.8],  [5, 1.4],  [6, 0.7]],
    "angry":     [[0.3, -0.4],[1.2, 0],    [2.1, 0.4], [3, 0.7],  [4, 0.85], [5, 0.65], [6, 0.25]]
  },

  "gaits": {
    "walk": { "label": "Walking", "speed": 1.4, "stepLength": 0.75, "bob": 0.04 },
    "jog":  { "label": "Jogging", "speed": 2.5, "stepLength": 1.0,  "bob": 0.07 },
    "run":  { "label": "Running", "speed": 4.5, "stepLength": 1.4,  "bob": 0.09 }
  },

  "paces": {
    "constant": { "label": "Constant",     "accel": 0 },
    "faster":   { "label": "Speeding up",  "accel": 0.3 },
    "slower":   { "label": "Slowing down", "accel": -0.3 }
  },

  "hipModel": {
    "walk": { "legLength": 0.95 },
    "jog":  { "stanceFraction": 0.55, "standingHeight": 0.92 },
    "run":  { "stanceFraction": 0.4,  "standingHeight": 0.92 }
  },

  "limbModel": {
    "walk": { "thighSwing": 25, "kneeBase": 5,  "kneeSwing": 60,  "kneeStance": 8,  "armSwing": 20, "elbowBend": 15, "lean": 3,  "bounce": 0.05, "flight": false },
    "jog":  { "thighSwing": 32, "kneeBase": 15, "kneeSwing": 85,  "kneeStance": 20, "armSwing": 30, "elbowBend": 80, "lean": 6,  "bounce": 0.07, "flight": true },
    "run":  { "thighSwing": 40, "kneeBase": 20, "kneeSwing": 110, "kneeStance": 22, "armSwing": 45, "elbowBend": 90, "lean": 10, "bounce": 0.09, "flight": true }
  }
};
