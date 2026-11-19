"""Python mirror of the browser engine (app.js), used for backtesting."""
import numpy as np
from scipy.stats import norm
from model import LEVELS, template_at

EXT = np.concatenate([[0.0], LEVELS, [1.0]])


def quantiles(mu, vol, tpl):
    base = template_at(*tpl, mu)
    q = mu + vol * base
    q = np.maximum(q, mu + base[0])                     # keep the position's floor
    lo = q[0] - (q[1] - q[0]); hi = q[-1] + 2 * (q[-1] - q[-2])
    return np.concatenate([[lo], q, [hi]])


def pair_key(a, b):
    if a['pos'] == 'DST' or b['pos'] == 'DST':
        if a['pos'] == 'DST' and b['pos'] == 'DST':
            return 'DST-DST opp'
        d, o = (a, b) if a['pos'] == 'DST' else (b, a)
        return f"DST-{o['pos']} {'own' if d['team'] == o['team'] else 'vs'}"
    x, y = sorted([a['pos'], b['pos']])
    return f"{x}-{y} {'same' if a['team'] == b['team'] else 'opp'}"


def correlated_normals(players, rho, n, rng):
    """players: list of dicts with pos, team, gid. Returns array (len(players), n)."""
    z = rng.standard_normal((len(players), n))
    games = {}
    for i, p in enumerate(players):
        games.setdefault(p.get('gid'), []).append(i)
    for g, idx in games.items():
        if g is None or len(idx) < 2:
            continue
        C = np.eye(len(idx))
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                C[a, b] = C[b, a] = rho.get(pair_key(players[idx[a]], players[idx[b]]), 0.0)
        s = 1.0
        while True:
            try:
                L = np.linalg.cholesky(np.where(np.eye(len(idx)) == 1, 1.0, C * s)); break
            except np.linalg.LinAlgError:
                s *= 0.9
        z[idx] = L @ z[idx]
    return z


def simulate(players, tpl, rho, n, rng):
    """players: dicts with pos, team, gid, mu, vol, pp (play prob), fixed (optional final points)."""
    z = correlated_normals(players, rho, n, rng)
    u = norm.cdf(z)
    pts = np.empty_like(u)
    for i, p in enumerate(players):
        if p.get('fixed') is not None:
            pts[i] = p['fixed']; continue
        q = quantiles(p['mu'], p.get('vol', 1.0), tpl[p['pos']])
        x = np.interp(u[i], EXT, q)
        if p['pos'] in ('K', 'DST'):
            x = np.round(x)
        if p.get('pp', 1.0) < 1.0:
            x = np.where(rng.random(n) < p['pp'], x, 0.0)
        pts[i] = x
    return pts.sum(0)
