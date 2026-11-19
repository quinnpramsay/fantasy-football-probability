"""Out-of-sample backtest of the win probability.
Distribution settings (templates, volatility, correlations) are fit on 2022-2024 walk-forward predictions only;
matchups are built from 2025-2026 player-weeks whose projections came from models trained on earlier seasons."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import model as M, sim
from fetch import CACHE

SLOTS = ['QB', 'RB', 'RB', 'WR', 'WR', 'TE', 'FLEX', 'K', 'DST']
POOL = {'QB': 16, 'RB': 40, 'WR': 48, 'TE': 16, 'K': 16, 'DST': 16}


def main(n_per_week=150, n_sims=4000, seed=1):
    oos = pd.read_pickle(f'{CACHE}/oos.pkl')
    fit = oos[oos.season.between(2022, 2024)]
    cal = M.fit_calibrators(fit)
    oos = M.calibrate_frame(oos, cal); fit = oos[oos.season.between(2022, 2024)]
    tpl = M.make_templates(fit)['ppr']
    rho, _ = M.pair_correlations(fit, tpl)
    vol, _ = M.volatility(fit, tpl)
    test = oos[oos.season >= 2025].copy()
    rng = np.random.default_rng(seed)
    rows = []
    for (s, w), g in test.groupby(['season', 'week']):
        pools = {p: g[g.pos == p].nlargest(n, 'pred_ppr') for p, n in POOL.items()}
        for _ in range(n_per_week):
            used, teams = set(), []
            for t in range(2):
                lineup = []
                for sl in SLOTS:
                    while True:
                        p = rng.choice(['RB', 'WR', 'TE']) if sl == 'FLEX' else sl
                        r = pools[p].iloc[rng.integers(len(pools[p]))]
                        if r.player_id not in used:
                            used.add(r.player_id); break
                    lineup.append(dict(pos=p, team=r.team, gid=r.gid, mu=float(r.pred_ppr),
                                       vol=float(vol.get(r.player_id, 1.0)), act=float(r.act_ppr)))
                teams.append(lineup)
            A, B = teams
            joint = joint_sim(A + B, tpl, rho, n_sims, rng)           # both teams together: they can share games
            ta, tb = joint[:len(A)].sum(0), joint[len(A):].sum(0)
            p_win = float(np.mean(ta > tb) + 0.5 * np.mean(ta == tb))
            mA, mB = sum(x['mu'] for x in A), sum(x['mu'] for x in B)
            rows.append(dict(season=s, week=w, edge=mA - mB, p=p_win, sd_a=float(ta.std()),
                             act_a=sum(x['act'] for x in A), act_b=sum(x['act'] for x in B),
                             won=float(sum(x['act'] for x in A) > sum(x['act'] for x in B))))
    r = pd.DataFrame(rows)
    r.to_pickle(f'{CACHE}/backtest.pkl')
    return summarize(r)


def joint_sim(players, tpl, rho, n, rng):
    z = sim.correlated_normals(players, rho, n, rng)
    from scipy.stats import norm
    u = norm.cdf(z); out = np.empty_like(u)
    for i, p in enumerate(players):
        x = np.interp(u[i], sim.EXT, sim.quantiles(p['mu'], p['vol'], tpl[p['pos']]))
        out[i] = np.round(x) if p['pos'] in ('K', 'DST') else x
    return out


def summarize(r):
    y = r.won.values
    res = {'matchups': int(len(r)), 'weeks': int(r.groupby(['season', 'week']).ngroups)}
    for c in ['p']:
        p = r[c].clip(1e-4, 1 - 1e-4)
        res[c] = dict(brier=round(float(np.mean((p - y) ** 2)), 4),
                      logloss=round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4))
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    cal = r.groupby(pd.cut(r.p, bins), observed=True).agg(n=('p', 'size'), predicted=('p', 'mean'), actual=('won', 'mean'))
    res['calibration'] = [dict(bin=str(k), n=int(v.n), predicted=round(float(v.predicted), 3), actual=round(float(v.actual), 3))
                          for k, v in cal.iterrows()]
    res['team_sd_model_mean'] = round(float(r.sd_a.mean()), 2)
    res['team_sd_actual'] = round(float(((r.act_a - (r.edge * 0 + r.act_a.mean())).std())), 2)
    return res


if __name__ == '__main__':
    out = main()
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), 'backtest.json'), 'w'), indent=1)
