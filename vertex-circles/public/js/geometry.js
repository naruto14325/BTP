// Discrete curve geometry shared by every section.

export const DEG = Math.PI / 180;

/**
 * Vertex curvature at b for the discrete curve a -> b -> c.
 *   phi : signed turning angle between the two edges
 *   k   : 2 sin(phi) / |c - a|     (Bobenko, DDG §2.3.1)
 *   degenerate : true when one edge is (almost) zero length,
 *                e.g. a planted foot that hasn't moved
 */
export function vertexCurvature(a, b, c) {
  const e1 = [b[0] - a[0], b[1] - a[1]];
  const e2 = [c[0] - b[0], c[1] - b[1]];

  const cross = e1[0] * e2[1] - e1[1] * e2[0];
  const dot = e1[0] * e2[0] + e1[1] * e2[1];
  const phi = Math.atan2(cross, dot);

  const chord = Math.hypot(c[0] - a[0], c[1] - a[1]);
  const shortestEdge = Math.min(Math.hypot(...e1), Math.hypot(...e2));

  return {
    phi,
    k: 2 * Math.sin(phi) / chord,
    degenerate: shortestEdge < 0.008,
  };
}

/** Centre of the circle through three points, or null if they are collinear. */
export function circumcenter(a, b, c) {
  const d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
  if (Math.abs(d) < 1e-12) return null;

  const A = a[0] ** 2 + a[1] ** 2;
  const B = b[0] ** 2 + b[1] ** 2;
  const C = c[0] ** 2 + c[1] ** 2;

  return [
    (A * (b[1] - c[1]) + B * (c[1] - a[1]) + C * (a[1] - b[1])) / d,
    (A * (c[0] - b[0]) + B * (a[0] - c[0]) + C * (b[0] - a[0])) / d,
  ];
}

export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
