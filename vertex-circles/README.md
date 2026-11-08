# Vertex circles in motion

Interactive notes on discrete curvature (vertex osculating circles) applied to
an eyebrow, a runner's hip and swinging limbs.

## Running

Needs only Python 3, no packages to install.

```
python3 server.py
```

Then open http://localhost:8000.

## Layout

```
server.py            backend: serves public/ and the /api/presets endpoint
data/presets.json    model parameters (eyebrow poses, gait settings)
public/
  index.html
  css/styles.css
  js/
    main.js          loads presets, starts sections, animation loop
    geometry.js      vertex curvature + circumcentre
    ui.js            buttons, metric tiles, small DOM helpers
    hero.js          draggable three-point demo
    eyebrow.js       eyebrow section
    speed.js         speed and prediction section
    hip.js           hip path section
    limbs.js         leg, hand and body section
```

Edit `data/presets.json` and refresh the page to try different poses or gaits.

## API

- `GET /api/presets` – all parameters
- `GET /api/presets/<name>` – one group, e.g. `/api/presets/gaits`
