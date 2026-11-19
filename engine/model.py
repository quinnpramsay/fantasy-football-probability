"""Projection engine: per-stat gradient-boosted models -> expected fantasy points, then an empirical
outcome distribution (skewed, floored, position/projection specific), a per-player volatility factor,
a play-probability from the injury report, and a Gaussian-copula correlation structure."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from data import OFF_W, REC_W
from features import STATS, HL

SKILL = ['QB', 'RB', 'WR', 'TE']
COUNT = {'pass_td', 'int', 'rush_td', 'rec', 'rec_td', 'fum_lost'}
TARGETS = ['pass_yds', 'pass_td', 'int', 'rush_yds', 'rush_td', 'rec', 'rec_yds', 'rec_td', 'fum_lost']
SHARES = ['targets_sh', 'carries_sh', 'air_yards_sh', 'snap_pct']
SKILL_X = [f'{c}_e{h}' for c in STATS + SHARES for h in HL] + [
    'n_prior', 'n_season', 'gap', 'pts_std_season', 'pts_last_season', 'home', 'implied', 'opp_implied',
    'total_line', 'spread', 'dome', 'def_rel', 'vac_tgt', 'vac_car', 'vac_air', 'vac_qb',
    'act_targets_sh', 'act_carries_sh', 'act_air_yards_sh', 'act_pass_att_sh', 'own_q', 'own_prac']
# K and D/ST: ridge regression on few features beat gradient boosting out of sample (GBM overfit: slope 0.5)
K_X = ['implied', 'total_line', 'dome', 'k_pts_e8', 'fg_att_e8', 'wind', 'home']
DST_X = ['opp_implied', 'implied', 'spread', 'home', 'pts_e8', 'sacks_e8', 'takeaways_e8', 'opp_sacked_e', 'opp_give_e']
# probability grid for stored outcome quantiles (denser in the tails)
LEVELS = np.array([0.005, 0.0125] + list(np.round(np.arange(0.025, 0.976, 0.025), 4)) + [0.9875, 0.995])
FMTS = ['ppr', 'half', 'std']
NBINS = 8


def gbm(loss):
    return HistGradientBoostingRegressor(loss=loss, max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def fit_predict(train, test, X, y, loss):
    tr = train[train[y].notna()]
    if tr[y].abs().sum() == 0:
        return np.zeros(len(test))
    m = gbm(loss)
    yy = tr[y].clip(lower=0) if loss == 'poisson' else tr[y]
    m.fit(tr[X], yy)
    return m.predict(test[X])


def skill_points(df_pred, fmt):
    pts = sum(df_pred['x_' + c] * OFF_W[c] for c in ['pass_yds', 'pass_td', 'int', 'rush_yds', 'rush_td',
                                                       'rec_yds', 'rec_td', 'fum_lost'])
    pts = pts + df_pred['x_rec'] * REC_W[fmt]
    pts = pts + df_pred['two_pt_e8'].fillna(0) * OFF_W['two_pt'] + df_pred['st_td_e8'].fillna(0) * OFF_W['st_td']
    return pts


def predict_skill(train, test):
    out = test.copy()
    for pos in SKILL:
        tr, te = train[train.pos == pos], out.pos == pos
        if not te.any():
            continue
        for t in TARGETS:
            out.loc[te, 'x_' + t] = fit_predict(tr, out[te], SKILL_X, t, 'poisson' if t in COUNT else 'squared_error')
    for f in FMTS:
        out['pred_' + f] = skill_points(out, f)
    return out


def predict_simple(train, test, X, y):
    out = test.copy()
    m = make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20)))
    tr = train.dropna(subset=[y])
    m.fit(tr[X], tr[y])
    out['pred'] = m.predict(out[X])
    for f in FMTS:
        out['pred_' + f] = out['pred']
    return out


def forward_oos(feat, predict, first_test=2022):
    """Walk-forward: every season S is predicted by a model trained only on seasons before S."""
    played = feat[feat.played == 1]
    parts = []
    for s in sorted(played.season.unique()):
        if s < first_test:
            continue
        parts.append(predict(played[played.season < s], played[played.season == s]))
    return pd.concat(parts, ignore_index=True)


# ---------- outcome distributions ----------

def build_templates(oos, pos, fmt, actual_col):
    d = oos[(oos.pos == pos)].dropna(subset=['pred_' + fmt, actual_col])
    pred, act = d['pred_' + fmt].values, d[actual_col].values
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, NBINS + 1)))
    centers, qs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred >= lo) & (pred <= hi)
        if m.sum() < 60:
            continue
        r = act[m] - pred[m]
        centers.append(float(np.median(pred[m])))
        qs.append(np.quantile(r, LEVELS))
    return np.array(centers), np.array(qs)


def template_at(centers, qs, mu):
    """Residual quantiles for projection mu (linear interpolation between bin centers)."""
    if mu <= centers[0]:
        return qs[0]
    if mu >= centers[-1]:
        return qs[-1]
    j = np.searchsorted(centers, mu) - 1
    w = (mu - centers[j]) / (centers[j + 1] - centers[j])
    return qs[j] * (1 - w) + qs[j + 1] * w


def tmpl_sd(centers, qs, mu):
    q = template_at(centers, qs, mu)
    # sd of a distribution given by its quantiles on LEVELS (trapezoid in probability)
    w = np.gradient(LEVELS); w = w / w.sum()
    m = (q * w).sum()
    return np.sqrt(((q - m) ** 2 * w).sum())


def residual_u(oos, tpl, pos, fmt, actual_col):
    """Probability-integral transform of each realized outcome within its own template."""
    c, q = tpl
    d = oos[oos.pos == pos].dropna(subset=['pred_' + fmt, actual_col])
    u = np.empty(len(d))
    for i, (mu, a) in enumerate(zip(d['pred_' + fmt].values, d[actual_col].values)):
        qq = template_at(c, q, mu)
        u[i] = np.interp(a - mu, qq + np.arange(len(qq)) * 1e-9, LEVELS, left=LEVELS[0] / 2, right=1 - (1 - LEVELS[-1]) / 2)
    return d.index, u


def volatility(oos, tpls, actual='act_ppr', fmt='ppr', key='player_id'):
    """Per-player spread multiplier: his typical |miss| relative to similar players (same position and projection),
    shrunk toward 1 with k pseudo-games. k is chosen on the last two seasons given, using only earlier games."""
    d = oos.dropna(subset=['pred_' + fmt, actual]).copy()
    d['tsd'] = [tmpl_sd(*tpls[p], mu) for p, mu in zip(d.pos, d['pred_' + fmt])]
    d['az'] = (d[actual] - d['pred_' + fmt]).abs() / d.tsd
    base = float(d.az.mean())
    d = d.sort_values([key, 'season', 'week'])
    g = d.groupby(key)
    d['n_prev'] = g.cumcount()
    d['s_prev'] = g['az'].transform(lambda x: x.shift(1).cumsum())
    ev = d[(d.season >= d.season.max() - 1) & (d.n_prev >= 1)]      # check on the last two seasons given
    best = None
    for k in [5, 10, 20, 30, 40, 60, 80, 120]:
        v = (ev.s_prev + k * base) / (ev.n_prev + k)
        mse = ((ev.az - v) ** 2).mean()
        if best is None or mse < best[1]:
            best = (k, mse)
    k = best[0]
    v = (ev.s_prev + k * base) / (ev.n_prev + k)
    slope = float(np.polyfit(v, ev.az, 1)[0]); corr = float(np.corrcoef(v, ev.az)[0, 1])
    tail = d.groupby(key).tail(40)
    agg = tail.groupby(key).agg(s=('az', 'sum'), n=('az', 'size'))
    vol = (((agg.s + k * base) / (agg.n + k)) / base).clip(0.6, 1.6)
    return vol, dict(k=float(k), calibration_slope=round(slope, 3), corr=round(corr, 3), n_eval=int(len(ev)))


# ---------- play probability ----------

PRAC = {'Did Not Participate In Practice': 'DNP', 'Limited Participation in Practice': 'LP',
        'Full Participation in Practice': 'FP'}


def play_table(pw, games, injuries, max_season):
    """Empirical P(player suits up) for contributors, by injury designation, practice status and whether
    he missed his team's previous game. Returns lookup dict and the table for the report."""
    tg = games[games.game_type == 'REG'][['season', 'week', 'team']].sort_values(['team', 'season', 'week'])
    tg['w1'] = tg.groupby(['team', 'season']).week.shift(1)
    tg['w2'] = tg.groupby(['team', 'season']).week.shift(2)
    tg['w3'] = tg.groupby(['team', 'season']).week.shift(3)
    played = set(zip(pw.player_id, pw.season, pw.week))
    pt = pw[['player_id', 'season', 'team']].drop_duplicates().merge(tg, on=['season', 'team'])
    pt = pt[pt.season <= max_season]
    pl_ = lambda P, S, W: [(p, s, int(w)) in played if w == w else False for p, s, w in zip(P, S, W)]
    pt['m1'] = pl_(pt.player_id, pt.season, pt.w1)
    pt['m2'] = pl_(pt.player_id, pt.season, pt.w2)
    pt['m3'] = pl_(pt.player_id, pt.season, pt.w3)
    # contributor: played 2 of the team's last 3 games and averaged >= 7 PPR over his previous 4 games
    pw2 = pw.sort_values(['player_id', 'season', 'week'])
    pw2 = pw2.assign(r4=pw2.groupby('player_id').pts_ppr.transform(lambda x: x.rolling(4, min_periods=2).mean()))
    pt = pt.merge(pw2[['player_id', 'season', 'week', 'r4']], how='left',
                  left_on=['player_id', 'season', 'w1'], right_on=['player_id', 'season', 'week'], suffixes=('', '_x'))
    pt = pt.merge(pw2[['player_id', 'season', 'week', 'r4']].rename(columns={'r4': 'r4b'}), how='left',
                  left_on=['player_id', 'season', 'w2'], right_on=['player_id', 'season', 'week'], suffixes=('', '_y'))
    pt['form'] = pt.r4.fillna(pt.r4b)
    pt = pt[(pt.m1.astype(int) + pt.m2 + pt.m3 >= 2) & (pt.form >= 7)]
    pt['missed_prev'] = ~pt.m1
    pt['played'] = pl_(pt.player_id, pt.season, pt.week)
    inj = injuries[injuries.game_type == 'REG'].drop_duplicates(['season', 'week', 'gsis_id'], keep='last')
    inj = inj[['season', 'week', 'gsis_id', 'report_status', 'practice_status']].rename(columns={'gsis_id': 'player_id'})
    pt = pt.merge(inj, on=['player_id', 'season', 'week'], how='left')
    pt['rs'] = pt.report_status.fillna('')
    pt['pr'] = pt.practice_status.map(PRAC).fillna('')
    pt['on'] = pt.report_status.notna() | pt.practice_status.notna()
    look, rows = {}, []

    def add(key, d, label):
        if len(d) >= 30:
            look[key] = round(float(d.played.mean()), 3)
            rows.append(dict(case=label, n=int(len(d)), p_play=look[key]))

    for st in ['Out', 'Doubtful', 'Questionable']:
        add(('final', st), pt[pt.rs == st], f'designation {st}')
    for pr in ['DNP', 'LP', 'FP']:
        for mp in (False, True):
            add(('prac', pr, mp), pt[(pt.pr == pr) & (pt.missed_prev == mp)],
                f'practice {pr}, {"missed" if mp else "played"} last game (any designation)')
    for mp in (False, True):
        add(('none', mp), pt[~pt.on & (pt.missed_prev == mp)], f'not on report, {"missed" if mp else "played"} last game')
    return look, rows


def play_prob(look, roster_status, report_status, practice, missed_prev, midweek=False):
    """midweek: the game is 2+ days away and no game designation exists yet. Historical practice rates describe the
    final (Friday) report, so mid-week we assume one level of improvement (DNP -> halfway to LP, LP -> FP, FP -> none).
    This shift is a heuristic, not fitted."""
    if roster_status == 'RES':
        return 0.0, 'IR'
    if isinstance(report_status, str) and report_status in ('Out', 'Doubtful', 'Questionable'):
        return look.get(('final', report_status), 1.0), {'Out': 'O', 'Doubtful': 'D', 'Questionable': 'Q'}[report_status]
    pr = PRAC.get(practice, '') if isinstance(practice, str) else ''
    if pr:
        mp = bool(missed_prev)
        g = lambda k: look.get(('prac', k, mp), 1.0)
        if midweek:
            p = {'DNP': (g('DNP') + g('LP')) / 2, 'LP': g('FP'), 'FP': look.get(('none', mp), 1.0)}[pr]
        else:
            p = g(pr)
        return p, pr if pr != 'FP' or p < 0.9 else ''
    p = look.get(('none', bool(missed_prev)), 1.0)
    return p, ('?' if p < 0.8 else '')


# ---------- correlation (Gaussian copula with game / team factors) ----------

def fit_copula(zt):
    """zt: rows (gid, team, opp, pos, z). Model:
        offense player of team A:  z = a_p*T_A + b_p*G + p_p*P_A + e     (P = team passing-game factor)
        D/ST of team A:            z = c1*T_B + c2*G + c3*T_A + e      (c1, c2 expected negative)
    Fit loadings to empirical pairwise correlations."""
    pos_list = ['QB', 'RB', 'WR', 'TE', 'K']
    pairs = {}
    by_game = {g: d for g, d in zt.groupby('gid')}
    for g, d in by_game.items():
        rows = d.to_dict('records')
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                same = a['team'] == b['team']
                pa, pb = a['pos'], b['pos']
                if pa == 'DST' or pb == 'DST':
                    if pa == 'DST' and pb == 'DST':
                        key = ('DST', 'DST', 'opp')
                    else:
                        dd, oo = (a, b) if pa == 'DST' else (b, a)
                        key = ('DST', oo['pos'], 'own' if dd['team'] == oo['team'] else 'vs')
                else:
                    x, y = sorted([pa, pb])
                    key = (x, y, 'same' if same else 'opp')
                pairs.setdefault(key, ([], []))
                pairs[key][0].append(a['z']); pairs[key][1].append(b['z'])
    emp = {k: (float(np.corrcoef(v[0], v[1])[0, 1]), len(v[0])) for k, v in pairs.items() if len(v[0]) > 200}
    pair_rho = {f'{k[0]}-{k[1]} {k[2]}': (round(v[0], 3) if v[1] >= 400 else 0.0) for k, v in emp.items()}

    PASS = ['QB', 'RB', 'WR', 'TE']

    def unpack(th):
        a = dict(zip(pos_list, th[:5])); b = dict(zip(pos_list, th[5:10])); pp = dict(zip(PASS, th[13:17])); pp['K'] = 0.0
        return a, b, th[10], th[11], th[12], pp

    def model_corr(th, key):
        a, b, c1, c2, c3, pp = unpack(th)
        x, y, kind = key
        if x == 'DST' and y == 'DST':
            return 2 * c1 * c3 + c2 * c2
        if x == 'DST':
            return (c3 * a[y] + c2 * b[y]) if kind == 'own' else (c1 * a[y] + c2 * b[y])
        if kind == 'same':
            return a[x] * a[y] + b[x] * b[y] + pp[x] * pp[y]
        return b[x] * b[y]

    keys = list(emp)
    wts = np.sqrt([emp[k][1] for k in keys])

    def resid(th):
        a, b, c1, c2, c3, pp = unpack(th)
        pen = [max(0, a[p] ** 2 + b[p] ** 2 + pp[p] ** 2 - 0.9) * 10 for p in pos_list] + [max(0, c1 ** 2 + c2 ** 2 + c3 ** 2 - 0.9) * 10]
        return np.concatenate([(np.array([model_corr(th, k) for k in keys]) - np.array([emp[k][0] for k in keys])) * wts / wts.mean(), pen])

    th0 = np.array([0.3] * 5 + [0.1] * 5 + [-0.2, -0.1, 0.0] + [0.2] * 4)
    fit = least_squares(resid, th0, bounds=([-0.95] * 17, [0.95] * 17))
    a, b, c1, c2, c3, pp = unpack(fit.x)
    table = [dict(pair=f'{k[0]}-{k[1]} {k[2]}', n=emp[k][1], empirical=round(emp[k][0], 3),
                  model=round(model_corr(fit.x, k), 3)) for k in keys]
    return pair_rho, table


def median_to_mean_maps(oos):
    """Sportsbook lines are ~medians. Map a median of each stat to its mean (from realized outcomes binned by model)."""
    maps = {}
    for s in ['pass_yds', 'rush_yds', 'rec_yds']:
        d = oos.dropna(subset=['x_' + s, s])
        d = d[d['x_' + s] > 5]
        bins = pd.qcut(d['x_' + s], 20, duplicates='drop')
        g = d.groupby(bins, observed=True)[s].agg(['median', 'mean'])
        g = g.sort_values('median').drop_duplicates('median')
        maps[s] = (g['median'].values.round(2).tolist(), g['mean'].values.round(2).tolist())
    return maps


def make_templates(oos_sub):
    tpl = {}
    for f in FMTS:
        tpl[f] = {pos: build_templates(oos_sub, pos, f, 'act_' + f) for pos in SKILL + ['K', 'DST']}
    return tpl


def pair_correlations(oos_sub, tpl_ppr):
    sub = oos_sub[(oos_sub.pos.isin(['K', 'DST'])) | (oos_sub.pred_ppr >= 5)]
    zt = []
    for pos in SKILL + ['K', 'DST']:
        idx, u = residual_u(sub, tpl_ppr[pos], pos, 'ppr', 'act_ppr')
        part = sub.loc[idx, ['gid', 'team', 'opp', 'pos']].copy()
        part['z'] = norm.ppf(u)
        zt.append(part)
    return fit_copula(pd.concat(zt).dropna(subset=['gid']))


# ---------- projection calibration ----------

def fit_calibrators(oos_sub):
    """Per position/format monotone map raw projection -> expected actual (20 quantile bins, isotonic on bin means).
    Fixes the remaining concavity: very high projections overshoot, mid projections undershoot."""
    from sklearn.isotonic import IsotonicRegression
    cal = {}
    for f in FMTS:
        cal[f] = {}
        for pos in SKILL + ['K', 'DST']:
            d = oos_sub[oos_sub.pos == pos].dropna(subset=['pred_' + f, 'act_' + f])
            b = pd.qcut(d['pred_' + f], 20, duplicates='drop')
            g = d.groupby(b, observed=True).agg(x=('pred_' + f, 'mean'), y=('act_' + f, 'mean'), n=('pred_' + f, 'size'))
            iso = IsotonicRegression(out_of_bounds='clip').fit(g.x, g.y, sample_weight=g.n)
            cal[f][pos] = (g.x.values.round(3).tolist(), iso.predict(g.x).round(3).tolist())
    return cal


def apply_cal(cal, pos, f, x):
    xs, ys = cal[f][pos]
    x = np.asarray(x, float)
    y = np.interp(x, xs, ys)
    lo, hi = x < xs[0], x > xs[-1]                       # outside the fitted range: shift, don't flatten
    y = np.where(lo, x + (ys[0] - xs[0]), y)
    y = np.where(hi, x + (ys[-1] - xs[-1]), y)
    return y


def calibrate_frame(df, cal):
    out = df.copy()
    for f in FMTS:
        col = 'pred_' + f
        out['raw_' + f] = out[col]
        for pos in out.pos.dropna().unique():
            m = out.pos == pos
            out.loc[m, col] = apply_cal(cal, pos, f, out.loc[m, col])
    return out
