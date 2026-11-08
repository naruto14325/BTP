"""Numba kernels: RNG, BVH queries, Walk on Spheres / Walk on Stars.

Everything here works on plain arrays so it can be JIT-compiled. The public
API lives in `wos.solver`.
"""

import math

import numpy as np
from numba import njit, prange

INF = 1e300
STACK = 64
BATCH = 32
jit = njit(cache=True, error_model="numpy")
jit_inline = njit(cache=True, error_model="numpy", inline="always")


# --------------------------------------------------------------------------
# Counter-based RNG (splitmix64). Each query point gets its own stream, so
# results are reproducible regardless of how threads are scheduled.
# --------------------------------------------------------------------------

@jit_inline
def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@jit_inline
def rand(s):
    s = s + np.uint64(0x9E3779B97F4A7C15)
    return s, (_mix(s) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@jit_inline
def seed_for(seed, i):
    return _mix(np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(i) + np.uint64(1))


# --------------------------------------------------------------------------
# Geometric primitives
# --------------------------------------------------------------------------

@jit_inline
def seg_closest(px, py, ax, ay, bx, by):
    """Squared distance from p to segment ab, and the parameter t of the closest point."""
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + aby * aby
    t = 0.0
    if L2 > 0.0:
        t = ((px - ax) * abx + (py - ay) * aby) / L2
        t = min(1.0, max(0.0, t))
    dx, dy = px - (ax + t * abx), py - (ay + t * aby)
    return dx * dx + dy * dy, t


@jit_inline
def box_d2(px, py, lo, hi, node):
    dx = max(lo[node, 0] - px, 0.0, px - hi[node, 0])
    dy = max(lo[node, 1] - py, 0.0, py - hi[node, 1])
    return dx * dx + dy * dy


@jit_inline
def _push_children(px, py, lo, hi, left, right, node, best, stack, sp):
    l, r = left[node], right[node]
    dl, dr = box_d2(px, py, lo, hi, l), box_d2(px, py, lo, hi, r)
    if dl < dr:
        if dr < best:
            stack[sp] = r; sp += 1
        if dl < best:
            stack[sp] = l; sp += 1
    else:
        if dl < best:
            stack[sp] = l; sp += 1
        if dr < best:
            stack[sp] = r; sp += 1
    return sp


@jit
def closest(px, py, a, b, lo, hi, left, right, start, count, stack):
    """Closest segment to p. Returns (squared distance, segment index, t)."""
    best, bk, bt = INF, -1, 0.0
    if lo.shape[0] == 0:
        return best, bk, bt
    stack[0] = 0
    sp = 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if box_d2(px, py, lo, hi, node) >= best:
            continue
        if left[node] < 0:
            for k in range(start[node], start[node] + count[node]):
                d2, t = seg_closest(px, py, a[k, 0], a[k, 1], b[k, 0], b[k, 1])
                if d2 < best:
                    best, bk, bt = d2, k, t
        else:
            sp = _push_children(px, py, lo, hi, left, right, node, best, stack, sp)
    return best, bk, bt


@jit_inline
def _is_silhouette(px, py, ux, uy, vx, vy, wx, wy):
    """Is vertex v (between segments u->v and v->w) a silhouette as seen from p?

    c1, c2 equal -(n1 . d) and -(n2 . d) for the (unnormalized) left normals
    of the two edges and view direction d = v - p, so v is a silhouette
    exactly when the edges face opposite ways as seen from p.
    """
    c1 = (ux - px) * (vy - py) - (uy - py) * (vx - px)
    c2 = (vx - px) * (wy - py) - (vy - py) * (wx - px)
    return c1 * c2 < 0.0


@jit_inline
def _no_silhouette_possible(px, py, lo, hi, cone_axis, cone_angle, node):
    """Normal-cone test: True if no vertex in this node can be a silhouette.

    View directions from p into the node's box lie in a cone (axis dv,
    half-angle tv); edge normals lie in the node's normal cone (axis na,
    half-angle tn). If no view direction can be perpendicular to a normal,
    every n . d has the same sign and no vertex in the node is a silhouette.
    """
    if cone_angle[node] == 0.0:
        return True
    if cone_angle[node] >= 0.5 * math.pi:
        return False
    cx, cy = 0.5 * (lo[node, 0] + hi[node, 0]), 0.5 * (lo[node, 1] + hi[node, 1])
    rb2 = 0.25 * ((hi[node, 0] - lo[node, 0]) ** 2 + (hi[node, 1] - lo[node, 1]) ** 2)
    dx, dy = cx - px, cy - py
    dist2 = dx * dx + dy * dy
    if dist2 <= rb2:
        return False
    # Total spread s = tn + tv must be < pi/2; then we need |phi - pi/2| > s,
    # i.e. |cos(phi)| > sin(s), with sin(s) from the angle-sum formula.
    sin_tv = math.sqrt(rb2 / dist2)
    cos_tv = math.sqrt(1.0 - rb2 / dist2)
    sin_tn, cos_tn = math.sin(cone_angle[node]), math.cos(cone_angle[node])
    cos_s = cos_tn * cos_tv - sin_tn * sin_tv
    if cos_s <= 0.0:
        return False
    sin_s = sin_tn * cos_tv + cos_tn * sin_tv
    cos_phi = (cone_axis[node, 0] * dx + cone_axis[node, 1] * dy) / math.sqrt(dist2)
    return abs(cos_phi) > sin_s


@jit
def closest_silhouette(px, py, a, b, prev, nxt, cone_axis, cone_angle,
                       lo, hi, left, right, start, count, stack, best):
    """Squared distance from p to the nearest silhouette vertex of the Neumann boundary.

    Only silhouettes closer than sqrt(best) matter (the star radius is capped
    by the Dirichlet distance anyway), so the search starts from that bound
    and returns it if nothing closer exists. Endpoints of open Neumann chains
    (where they meet a Dirichlet segment) are always treated as silhouettes.
    """
    if lo.shape[0] == 0:
        return best
    stack[0] = 0
    sp = 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if box_d2(px, py, lo, hi, node) >= best:
            continue
        if _no_silhouette_possible(px, py, lo, hi, cone_axis, cone_angle, node):
            continue
        if left[node] < 0:
            for k in range(start[node], start[node] + count[node]):
                vx, vy = a[k, 0], a[k, 1]
                p = prev[k]
                sil = p < 0
                if not sil:
                    sil = _is_silhouette(px, py, a[p, 0], a[p, 1], vx, vy, b[k, 0], b[k, 1])
                if sil:
                    d2 = (vx - px) ** 2 + (vy - py) ** 2
                    if d2 < best:
                        best = d2
                if nxt[k] < 0:
                    d2 = (b[k, 0] - px) ** 2 + (b[k, 1] - py) ** 2
                    if d2 < best:
                        best = d2
        else:
            sp = _push_children(px, py, lo, hi, left, right, node, best, stack, sp)
    return best


@jit_inline
def _ray_box(ox, oy, dx, dy, lo, hi, node, tmax):
    t0, t1 = 0.0, tmax
    for ax in range(2):
        o = ox if ax == 0 else oy
        d = dx if ax == 0 else dy
        l, h = lo[node, ax], hi[node, ax]
        if abs(d) < 1e-300:
            if o < l or o > h:
                return False
        else:
            ta, tb = (l - o) / d, (h - o) / d
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return False
    return True


@jit
def ray_hit(ox, oy, dx, dy, tmax, a, b, lo, hi, left, right, start, count, stack):
    """First intersection of ray o + t d (0 < t < tmax) with the segments.

    Returns (t, segment index), index -1 if nothing is hit.
    """
    best, bk = tmax, -1
    if lo.shape[0] == 0:
        return best, bk
    stack[0] = 0
    sp = 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if not _ray_box(ox, oy, dx, dy, lo, hi, node, best):
            continue
        if left[node] < 0:
            for k in range(start[node], start[node] + count[node]):
                ex, ey = b[k, 0] - a[k, 0], b[k, 1] - a[k, 1]
                den = dx * ey - dy * ex
                if abs(den) < 1e-300:
                    continue
                wx, wy = a[k, 0] - ox, a[k, 1] - oy
                t = (wx * ey - wy * ex) / den
                s = (wx * dy - wy * dx) / den
                if 0.0 < t < best and 0.0 <= s <= 1.0:
                    best, bk = t, k
        else:
            stack[sp] = left[node]; sp += 1
            stack[sp] = right[node]; sp += 1
    return best, bk


@njit(cache=True, parallel=True)
def inside_kernel(P, a, b):
    out = np.zeros(P.shape[0], np.bool_)
    for i in prange(P.shape[0]):
        px, py = P[i, 0], P[i, 1]
        c = False
        for k in range(a.shape[0]):
            ay, by = a[k, 1], b[k, 1]
            if (ay > py) != (by > py):
                xi = a[k, 0] + (py - ay) * (b[k, 0] - a[k, 0]) / (by - ay)
                if px < xi:
                    c = not c
        out[i] = c
    return out


@njit(cache=True, parallel=True)
def closest_dist_kernel(P, a, b, lo, hi, left, right, start, count):
    out = np.empty(P.shape[0])
    for i in prange(P.shape[0]):
        stack = np.empty(STACK, np.int64)
        d2, _, _ = closest(P[i, 0], P[i, 1], a, b, lo, hi, left, right, start, count, stack)
        out[i] = math.sqrt(d2)
    return out


# --------------------------------------------------------------------------
# One random walk: Walk on Spheres, generalized to Walk on Stars when the
# scene has Neumann segments (Sawhney, Miller, Gkioulekas & Crane 2023).
# --------------------------------------------------------------------------

@njit(error_model="numpy")
def walk(x, y, on_bnd, nx, ny, s,
         da, db, dga, dgb, dlo, dhi, dl, dr, ds, dc,
         na, nb, nprev, nnext, nnorm, ncax, ncang, nlo, nhi, nl, nr, ns, nc,
         source, has_source, source_samples, eps, rmin, offset, max_steps, stack):
    """Estimate u at (x, y). Returns (estimate, rng state, number of steps).

    (nx, ny) is the inward normal when the walk starts on a Neumann boundary.
    """
    has_neumann = na.shape[0] > 0
    acc = 0.0
    for step in range(max_steps):
        d2, k, t = closest(x, y, da, db, dlo, dhi, dl, dr, ds, dc, stack)
        dD = math.sqrt(d2)
        if dD < eps:
            return acc + dga[k] + t * (dgb[k] - dga[k]), s, step

        # Star radius: the ball may cross Neumann segments, but must not
        # contain a silhouette point, so it stays star-shaped w.r.t. x.
        R = dD
        if has_neumann:
            R = min(R, math.sqrt(closest_silhouette(
                x, y, na, nb, nprev, nnext, ncax, ncang, nlo, nhi, nl, nr, ns, nc, stack, d2)))
            R = max(R, rmin)

        base = math.atan2(ny, nx) if on_bnd else 0.0
        span = math.pi if on_bnd else 2.0 * math.pi

        if has_source:
            # Importance-sample y ~ G(x, y): r = R sqrt(u1 u2) has density
            # proportional to r log(R / r), so the weight is just |B|-integral
            # of G, which is R^2 / 4 in 2D. Several samples per step share
            # the cost of the walk, which matters for localized sources.
            fsum = 0.0
            for _ in range(source_samples):
                s, u1 = rand(s); s, u2 = rand(s); s, u3 = rand(s)
                r = R * math.sqrt(u1 * u2)
                th = base + span * (u3 - 0.5) if on_bnd else span * u3
                cx, cy = math.cos(th), math.sin(th)
                visible = True
                if has_neumann:
                    _, hk = ray_hit(x, y, cx, cy, r, na, nb, nlo, nhi, nl, nr, ns, nc, stack)
                    visible = hk < 0
                if visible:
                    fsum += source(x + r * cx, y + r * cy)
            acc += 0.25 * R * R * fsum / source_samples

        # Next point: uniform direction (hemisphere on a Neumann boundary).
        s, u = rand(s)
        th = base + span * (u - 0.5) if on_bnd else span * u
        cx, cy = math.cos(th), math.sin(th)
        if has_neumann:
            th_hit, hk = ray_hit(x, y, cx, cy, R, na, nb, nlo, nhi, nl, nr, ns, nc, stack)
            if hk >= 0:
                nx, ny = nnorm[hk, 0], nnorm[hk, 1]
                x = x + th_hit * cx + offset * nx
                y = y + th_hit * cy + offset * ny
                on_bnd = True
                continue
        x += R * cx
        y += R * cy
        on_bnd = False

    # Walk did not terminate: fall back to the nearest boundary value (biased).
    d2, k, t = closest(x, y, da, db, dlo, dhi, dl, dr, ds, dc, stack)
    return acc + dga[k] + t * (dgb[k] - dga[k]), s, max_steps


@njit(parallel=True, error_model="numpy")
def solve_kernel(P, n_min, n_max, tol, seed,
                 da, db, dga, dgb, dlo, dhi, dl, dr, ds, dc,
                 na, nb, nprev, nnext, nnorm, ncax, ncang, nlo, nhi, nl, nr, ns, nc,
                 source, has_source, source_samples, eps, rmin, offset, max_steps):
    m = P.shape[0]
    mean = np.empty(m)
    err = np.empty(m)
    used = np.empty(m, np.int64)
    steps = np.empty(m)
    for i in prange(m):
        stack = np.empty(STACK, np.int64)
        s = seed_for(seed, i)
        sm, sq, n, st = 0.0, 0.0, 0, 0
        while n < n_max:
            for _ in range(min(BATCH, n_max - n)):
                v, s, k = walk(P[i, 0], P[i, 1], False, 0.0, 0.0, s,
                               da, db, dga, dgb, dlo, dhi, dl, dr, ds, dc,
                               na, nb, nprev, nnext, nnorm, ncax, ncang, nlo, nhi, nl, nr, ns, nc,
                               source, has_source, source_samples, eps, rmin, offset, max_steps, stack)
                sm += v
                sq += v * v
                st += k
                n += 1
            # Adaptive sampling: stop once the standard error is below tol.
            if tol > 0.0 and n >= n_min:
                var = max(sq - sm * sm / n, 0.0) / (n - 1)
                if math.sqrt(var / n) < tol:
                    break
        mean[i] = sm / n
        err[i] = math.sqrt(max(sq - sm * sm / n, 0.0) / (n - 1) / n) if n > 1 else INF
        used[i] = n
        steps[i] = st / n
    return mean, err, used, steps


@njit(parallel=True, error_model="numpy")
def gradient_kernel(P, n, seed, use_cv,
                    da, db, dga, dgb, dlo, dhi, dl, dr, ds, dc,
                    na, nb, nprev, nnext, nnorm, ncax, ncang, nlo, nhi, nl, nr, ns, nc,
                    source, has_source, source_samples, eps, rmin, offset, max_steps):
    """Estimate u and grad u (Sawhney & Crane 2020, Sec. 3.2).

    The first step uses a ball that touches no boundary at all, then
    continues with the regular walk. grad u = (2/R) E[(u(z) - c) (z - x)/R]
    plus the source contribution; c = mean of u(z) is a control variate
    that removes most of the variance (E[(z - x)] = 0, so it adds no bias).
    """
    m = P.shape[0]
    u_out = np.empty(m)
    g_out = np.empty((m, 2))
    u_err = np.empty(m)
    g_err = np.empty((m, 2))
    for i in prange(m):
        stack = np.empty(STACK, np.int64)
        s = seed_for(seed, i)
        x, y = P[i, 0], P[i, 1]
        R2, _, _ = closest(x, y, da, db, dlo, dhi, dl, dr, ds, dc, stack)
        N2, _, _ = closest(x, y, na, nb, nlo, nhi, nl, nr, ns, nc, stack)
        R = math.sqrt(min(R2, N2))
        if R < eps:
            # On (or within eps of) the boundary the ball estimator is undefined.
            u_out[i] = u_err[i] = np.nan
            g_out[i, 0] = g_out[i, 1] = g_err[i, 0] = g_err[i, 1] = np.nan
            continue
        # Plain sums, plus the second moments needed for standard errors.
        su, su2, sux, suy, svx, svy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        su2x, su2y, sux2, suy2, sx2, sy2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        fu, fu2, fgx, fgy, fgx2, fgy2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for _ in range(n):
            s, w = rand(s)
            th = 2.0 * math.pi * w
            vx, vy = math.cos(th), math.sin(th)
            u, s, _ = walk(x + R * vx, y + R * vy, False, 0.0, 0.0, s,
                           da, db, dga, dgb, dlo, dhi, dl, dr, ds, dc,
                           na, nb, nprev, nnext, nnorm, ncax, ncang, nlo, nhi, nl, nr, ns, nc,
                           source, has_source, source_samples, eps, rmin, offset, max_steps, stack)
            su += u; su2 += u * u
            sux += u * vx; suy += u * vy; svx += vx; svy += vy
            su2x += u * u * vx * vx; su2y += u * u * vy * vy
            sux2 += u * vx * vx; suy2 += u * vy * vy
            sx2 += vx * vx; sy2 += vy * vy
            if has_source:
                # Value: y ~ G (as in walk). Gradient: r ~ U[0, R] cancels the
                # 1/r singularity of grad G, giving weight f R (1 - r^2/R^2).
                s, u1 = rand(s); s, u2 = rand(s); s, u3 = rand(s)
                r = R * math.sqrt(u1 * u2)
                th = 2.0 * math.pi * u3
                fv = 0.25 * R * R * source(x + r * math.cos(th), y + r * math.sin(th))
                fu += fv; fu2 += fv * fv
                s, u1 = rand(s); s, u3 = rand(s)
                r = R * u1
                th = 2.0 * math.pi * u3
                cx, cy = math.cos(th), math.sin(th)
                w = source(x + r * cx, y + r * cy) * R * (1.0 - (r / R) ** 2)
                fgx += w * cx; fgy += w * cy
                fgx2 += (w * cx) ** 2; fgy2 += (w * cy) ** 2
        ubar = su / n
        c = ubar if use_cv else 0.0
        k = 2.0 / R
        u_out[i] = ubar + fu / n
        g_out[i, 0] = k * (sux - c * svx) / n + fgx / n
        g_out[i, 1] = k * (suy - c * svy) / n + fgy / n
        # Variances of the per-walk samples (boundary and source parts are
        # independent, so their variances add).
        var_u = max(su2 / n - ubar ** 2, 0.0) + max(fu2 / n - (fu / n) ** 2, 0.0)
        mx, my = (sux - c * svx) / n, (suy - c * svy) / n
        ex2 = (su2x - 2.0 * c * sux2 + c * c * sx2) / n
        ey2 = (su2y - 2.0 * c * suy2 + c * c * sy2) / n
        var_x = k * k * max(ex2 - mx * mx, 0.0) + max(fgx2 / n - (fgx / n) ** 2, 0.0)
        var_y = k * k * max(ey2 - my * my, 0.0) + max(fgy2 / n - (fgy / n) ** 2, 0.0)
        u_err[i] = math.sqrt(var_u / n)
        g_err[i, 0] = math.sqrt(var_x / n)
        g_err[i, 1] = math.sqrt(var_y / n)
    return u_out, g_out, u_err, g_err


# --------------------------------------------------------------------------
# 3D Walk on Spheres on a signed distance function (sdf < 0 inside).
# --------------------------------------------------------------------------

@njit(parallel=True, error_model="numpy")
def solve_sdf3_kernel(P, n, seed, sdf, g, eps, max_steps):
    m = P.shape[0]
    mean = np.empty(m)
    err = np.empty(m)
    for i in prange(m):
        s = seed_for(seed, i)
        sm, sq = 0.0, 0.0
        for _ in range(n):
            x, y, z = P[i, 0], P[i, 1], P[i, 2]
            for _step in range(max_steps):
                R = abs(sdf(x, y, z))
                if R < eps:
                    break
                s, u1 = rand(s); s, u2 = rand(s)
                cz = 1.0 - 2.0 * u1
                sr = math.sqrt(max(0.0, 1.0 - cz * cz))
                ph = 2.0 * math.pi * u2
                x += R * sr * math.cos(ph)
                y += R * sr * math.sin(ph)
                z += R * cz
            v = g(x, y, z)
            sm += v
            sq += v * v
        mean[i] = sm / n
        err[i] = math.sqrt(max(sq - sm * sm / n, 0.0) / (n - 1) / n) if n > 1 else INF
    return mean, err
