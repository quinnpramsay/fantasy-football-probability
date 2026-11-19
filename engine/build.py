"""Build data.js for the site: projections + outcome distributions for every relevant player this week.
usage: python engine/build.py [--season 2026] [--week N] [--fast]   (--fast reuses cached walk-forward results)"""
import argparse, json, math, os, re, sys, time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
import data as D, features as F, model as M, sources as S
from fetch import fetch_all, CACHE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROP_W = 0.6          # weight on sportsbook-implied stat means vs model (prior, not fitted: no free historical props)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def target_week(games, season):
    g = games[(games.season == season) & (games.game_type == 'REG')]
    open_ = g[g.score.isna()]
    return int(open_.week.min()) if len(open_) else int(g.week.max())


_UNSAFE = re.compile(r"[^\w .'\u2019\-:\u00b7/()+,%]")


def clean(o):
    """Everything that reaches the page passes through here: text from outside sources (player names, teams, props,
    expert files) keeps only letters, digits and plain punctuation, so it can never carry HTML or script."""
    if isinstance(o, dict):
        return {clean(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, str):
        return _UNSAFE.sub('', o)[:80]
    if isinstance(o, (bool, int, float)) or o is None:
        return o
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    raise TypeError(f'unexpected type in payload: {type(o)}')


def check_payload(p):
    pl = p['players']
    pos = {}
    for r in pl:
        pos[r['p']] = pos.get(r['p'], 0) + 1
        if not (isinstance(r.get('m'), list) and len(r['m']) == 3 and all(isinstance(v, (int, float)) and -10 <= v < 80 for v in r['m'])):
            raise SystemExit(f"refusing to write data.js: bad projection for {r.get('n')}: {r.get('m')}")
    need = dict(QB=25, RB=50, WR=70, TE=30, K=20, DST=20)
    short = {k: (pos.get(k, 0), v) for k, v in need.items() if pos.get(k, 0) < v}
    if short:
        raise SystemExit(f'refusing to write data.js: too few players {short}')
    if not p['tpl'] or not p['games']:
        raise SystemExit('refusing to write data.js: missing distributions or games')


def walk_forward(fs, fk, fd, fast, tag):
    """Walk-forward predictions are refit once per NFL week (cached by season/week); --fast reuses any cache."""
    path = f'{CACHE}/oos_{tag}.pkl'
    if os.path.exists(path):
        return pd.read_pickle(path)
    if fast and os.path.exists(f'{CACHE}/oos.pkl'):
        return pd.read_pickle(f'{CACHE}/oos.pkl')
    log('walk-forward skill models (slow)')
    o1 = M.forward_oos(fs, M.predict_skill)
    o2 = M.forward_oos(fk, lambda a, b: M.predict_simple(a, b, M.K_X, 'k_pts'))
    o3 = M.forward_oos(fd, lambda a, b: M.predict_simple(a, b, M.DST_X, 'pts'))
    for f in M.FMTS:
        o1['act_' + f] = o1['pts_' + f]; o2['act_' + f] = o2['k_pts']; o3['act_' + f] = o3['pts']
    o3 = o3.assign(pos='DST', player_id='DST_' + o3.team, name=o3.team + ' D/ST')
    oos = pd.concat([o1, o2, o3], ignore_index=True, sort=False)
    for f in os.listdir(CACHE):
        if f.startswith('oos_') and f.endswith('.pkl'):
            os.remove(os.path.join(CACHE, f))
    oos.to_pickle(path); oos.to_pickle(f'{CACHE}/oos.pkl')
    return oos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, default=2026)
    ap.add_argument('--week', type=int)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--no-fetch', action='store_true')
    a = ap.parse_args()
    season = a.season
    if not a.no_fetch:
        log('fetching'); fetch_all(season)
    seasons = range(2021, season + 1)
    games = D.load_games()
    week = a.week or target_week(games, season)
    log(f'season {season} week {week}')
    pl = D.load_players(seasons); dst = D.load_dst(seasons, games)
    # finished games this week: final points (only for games whose box scores are already published)
    cw = pl[(pl.season == season) & (pl.week == week)]
    cwd = dst[(dst.season == season) & (dst.week == week)]
    done_games = set(games[(games.season == season) & (games.week == week) & games.score.notna()].game_id)
    published = done_games & set(cw.game_id)
    finals = {r.player_id: [round(float(r.pts_ppr), 2), round(float(r.pts_half), 2), round(float(r.pts_std), 2)]
              for r in cw.itertuples() if r.game_id in published}
    for r in cwd.itertuples():
        if r.game_id in published:
            finals['DST_' + r.team] = [float(r.pts)] * 3
    pl = pl[(pl.season < season) | (pl.week < week)]
    dst = dst[(dst.season < season) | (dst.week < week)]
    injuries = pd.concat([pd.read_csv(f'{CACHE}/injuries_{y}.csv', low_memory=False) for y in seasons], ignore_index=True)
    roster = pd.read_csv(f'{CACHE}/roster_{season}.csv', low_memory=False)
    rw = roster[roster.week == (week if (roster.week == week).any() else roster.week.max())]
    players_meta = pd.read_csv(f'{CACHE}/players.csv', low_memory=False)

    # ---- who to project ----
    recent = pl[pl.season >= season - 1]
    hist_ids = set(recent.player_id)
    cand = rw[rw.position.isin(M.SKILL + ['K']) & rw.status.isin(['ACT', 'RES']) & rw.gsis_id.isin(hist_ids)]
    cand = cand.drop_duplicates('gsis_id')
    last_name = pl.sort_values(['season', 'week']).groupby('player_id').name.last()
    fut = pd.DataFrame(dict(player_id=cand.gsis_id.values, name=cand.gsis_id.map(last_name).fillna(cand.full_name).values,
                            pos=cand.position.values, team=cand.team.values, season=season, week=week,
                            roster_status=cand.status.values, espn_id=cand.espn_id.values))
    fut_dst = pd.DataFrame(dict(team=sorted(games[games.season == season].team.unique()), season=season, week=week))

    # ---- play probability ----
    look, play_rows = M.play_table(pl, games, injuries, season - 1)
    log('play table', play_rows)
    inj_w = injuries[(injuries.season == season) & (injuries.week == week)].drop_duplicates('gsis_id', keep='last')
    inj_rs = dict(zip(inj_w.gsis_id, inj_w.report_status)); inj_pr = dict(zip(inj_w.gsis_id, inj_w.practice_status))
    tg = games[(games.season == season) & (games.week < week) & (games.game_type == 'REG')]
    team_last = tg.groupby('team').week.max().to_dict()
    played_set = set(zip(pl.player_id, pl.season, pl.week))

    now_et = datetime.now(timezone.utc) - timedelta(hours=5)
    gday = games[(games.season == season) & (games.week == week)].set_index('team').gameday.to_dict()
    rstat = dict(zip(rw.gsis_id, rw.status))

    def p_play_now(pid, team):
        lw = team_last.get(team)
        missed = lw is not None and (pid, season, lw) not in played_set
        midweek = team in gday and (pd.Timestamp(gday[team]) - pd.Timestamp(now_et.date())).days >= 2
        return M.play_prob(look, rstat.get(pid, 'ACT'), inj_rs.get(pid), inj_pr.get(pid), missed, midweek)

    # expected absences -> vacated-opportunity features (history: final designations; this week: current estimate)
    ih = injuries[(injuries.game_type == 'REG') & injuries.report_status.isin(['Out', 'Doubtful', 'Questionable'])]
    ih = ih[(ih.season < season) | (ih.week < week)].drop_duplicates(['season', 'week', 'gsis_id'], keep='last')
    ih = ih.assign(p_out=1 - ih.report_status.map(lambda r: look.get(('final', r), 1.0)))
    cur = inj_w[['season', 'week', 'team', 'gsis_id']].copy()
    cur['p_out'] = [1 - p_play_now(g, t)[0] for g, t in zip(cur.gsis_id, cur.team)]
    inj_feat = pd.concat([ih[['season', 'week', 'team', 'gsis_id', 'p_out']], cur], ignore_index=True)
    inj_feat = inj_feat.rename(columns={'gsis_id': 'player_id'})
    inj_feat = inj_feat[inj_feat.p_out > 0.01]

    # ---- features ----
    log('features')
    sk, kk = pl[pl.pos.isin(M.SKILL)], pl[pl.pos == 'K']
    fs = F.skill_features(sk, games, fut[fut.pos.isin(M.SKILL)][['player_id', 'name', 'pos', 'team', 'season', 'week']], inj_feat, injuries)
    fk = F.kicker_features(kk, games, fut[fut.pos == 'K'][['player_id', 'name', 'pos', 'team', 'season', 'week']])
    fd = F.dst_features(dst, games, fut_dst)

    oos = walk_forward(fs, fk, fd, a.fast, f'{season}w{week}')
    log('final models')
    played_s, played_k, played_d = fs[fs.played == 1], fk[fk.played == 1], fd[fd.played == 1]
    ps = M.predict_skill(played_s, fs[fs.played == 0])
    pk = M.predict_simple(played_k, fk[fk.played == 0], M.K_X, 'k_pts')
    pd_ = M.predict_simple(played_d, fd[fd.played == 0], M.DST_X, 'pts')
    pd_ = pd_.assign(pos='DST', player_id='DST_' + pd_.team, name=pd_.team + ' D/ST')

    # ---- calibration + distributions ----
    log('calibration / templates / volatility / correlations')
    cal = M.fit_calibrators(oos[oos.season >= 2023])
    oos = M.calibrate_frame(oos, cal)
    tpl = M.make_templates(oos[oos.season >= 2023])
    tpl_out = {f: {pos: dict(c=np.round(c, 2).tolist(), q=np.round(q, 2).tolist()) for pos, (c, q) in t.items()}
               for f, t in tpl.items()}
    vol, vol_fit = M.volatility(oos, tpl['ppr'])
    cop, cop_table = M.pair_correlations(oos[oos.season >= 2023], tpl['ppr'])
    log('pair correlations', len(cop))

    # coverage check of the distributions (share of outcomes inside the model's 10-90% band)
    cover = {}
    for pos in M.SKILL + ['K', 'DST']:
        d = oos[(oos.season >= 2023) & (oos.pos == pos)]
        lo = [mu + M.template_at(*tpl['ppr'][pos], mu)[np.searchsorted(M.LEVELS, 0.1)] for mu in d.pred_ppr]
        hi = [mu + M.template_at(*tpl['ppr'][pos], mu)[np.searchsorted(M.LEVELS, 0.9)] for mu in d.pred_ppr]
        cover[pos] = round(float(((d.act_ppr >= lo) & (d.act_ppr <= hi)).mean()), 3)

    # ---- external: props + ECR ----
    wk_games = games[(games.season == season) & (games.week == week)]
    gm_raw = pd.read_csv(f'{CACHE}/games.csv', low_memory=False)
    ev = gm_raw[(gm_raw.season == season) & (gm_raw.week == week) & gm_raw.result.isna()].espn.dropna().astype(int).tolist()
    log(f'props for {len(ev)} unplayed games')
    props = S.fetch_props_espn(ev)
    espn_to_gsis = {}
    for src in (players_meta[['espn_id', 'gsis_id']], roster[['espn_id', 'gsis_id']]):
        for e, gsis in src.dropna().itertuples(index=False):
            try:
                espn_to_gsis[int(float(e))] = gsis
            except ValueError:
                pass
    props_g = {espn_to_gsis[k]: v for k, v in props.items() if k in espn_to_gsis}
    odds_key = os.environ.get('ODDS_API_KEY')
    odds = {}
    if odds_key and len(wk_games):
        days = pd.to_datetime(wk_games.gameday)
        odds = S.fetch_props_oddsapi(odds_key, (days.min() - timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z'),
                                     (days.max() + timedelta(days=1)).strftime('%Y-%m-%dT23:59:59Z'))
    log(f'props: {len(props_g)} players (DraftKings lines){", odds-api " + str(len(odds)) if odds else ""}')
    ecr = S.fetch_ecr()
    log('ecr files', {f'{k[0]}-{k[1]}': len(v) for k, v in ecr.items()})

    # ---- calibrate model points, then blend props into the stat means ----
    ps, pk, pd_ = M.calibrate_frame(ps, cal), M.calibrate_frame(pk, cal), M.calibrate_frame(pd_, cal)
    mm = M.median_to_mean_maps(oos[oos.pos.isin(M.SKILL)])
    ps = ps.copy()
    for f in M.FMTS:
        ps['model_' + f] = ps['pred_' + f]
    nblend = 0
    for i, r in ps.iterrows():
        pr = props_g.get(r.player_id, {})
        od = odds.get(S.norm_name(r['name']), {})
        changed = False
        for st in ('pass_yds', 'rush_yds', 'rec_yds'):
            if st in pr:
                ps.at[i, 'x_' + st] = (1 - PROP_W) * r['x_' + st] + PROP_W * S.line_to_mean(pr[st], mm[st]); changed = True
        if 'rec' in pr:
            ps.at[i, 'x_rec'] = (1 - PROP_W) * r.x_rec + PROP_W * S.poisson_mean_from_line(pr['rec']); changed = True
        if 'td_p' in od and 0 < od['td_p'] < 0.99:
            lam_m = -math.log(1 - od['td_p']); lam = r.x_rush_td + r.x_rec_td
            if lam > 0:
                k = ((1 - PROP_W) * lam + PROP_W * lam_m) / lam
                ps.at[i, 'x_rush_td'] = r.x_rush_td * k; ps.at[i, 'x_rec_td'] = r.x_rec_td * k; changed = True
        nblend += changed
    for f in M.FMTS:   # calibrated model points + (market-blended stat line points - model stat line points)
        ps['pred_' + f] = ps['model_' + f] + M.skill_points(ps, f) - ps['raw_' + f]
    ratio = (ps['pred_ppr'] / M.skill_points(ps, 'ppr').replace(0, np.nan)).fillna(1).clip(0.5, 1.5)
    for c in M.TARGETS:          # keep the displayed stat line consistent with the calibrated points
        ps['x_' + c] = ps['x_' + c] * ratio
    log(f'blended props into {nblend} players')

    # ---- assemble players ----
    def ecr_for(pos, name, team):
        res = {}
        for f in (M.FMTS if pos in ('RB', 'WR', 'TE') else [None]):
            for e in ecr.get((pos, f), []):
                ok = (S.TEAM_NAMES.get(e['name']) == team) if pos == 'DST' else (S.norm_name(e['name']) == S.norm_name(name))
                if ok:
                    res[f or 'all'] = [e['rank'], e['avg'], e['sd'], e['best'], e['worst'], e['tier']]
                    break
        return res or None

    gtab = wk_games.set_index('team')
    last = pd.concat([pl.assign(pts=pl.pts_ppr), dst.assign(player_id='DST_' + dst.team, pts=dst.pts, rec=0)],
                     ignore_index=True, sort=False).sort_values(['season', 'week'])
    last = last[last.season == season].groupby('player_id').tail(8)
    last_map = {pid: [[int(r.week), r.opp, round(float(r.pts), 1), round(float(r.rec), 1)] for r in d.itertuples()]
                for pid, d in last.groupby('player_id')}

    def kickoff(gr):
        try:
            dt = datetime.strptime(f'{gr.gameday} {gr.gametime}', '%Y-%m-%d %H:%M')
            return dt.strftime('%a %-I:%M') + ' ET'
        except Exception:
            return ''

    allp = pd.concat([ps, pk, pd_], ignore_index=True, sort=False)
    meta_f = fut.set_index('player_id')
    out = []
    for r in allp.itertuples():
        pid, pos, team = r.player_id, r.pos, r.team
        g = gtab.loc[team] if team in gtab.index else None
        st, pplay = '', 1.0
        if pos != 'DST':
            pplay, st = p_play_now(pid, team)
        if g is None:
            st, pplay = 'BYE', 0.0
        means = [max(0.0, float(getattr(r, 'pred_' + f))) if pos not in ('DST',) else float(getattr(r, 'pred_' + f)) for f in M.FMTS]
        rec = dict(i=pid, n=r.name, p=pos, t=team, s=st, pp=round(pplay, 3), v=round(float(vol.get(pid, 1.0)), 3),
                   m=[round(x, 2) for x in means])
        if g is not None and g.game_id in published:
            rec['fp'] = finals.get(pid, [0.0, 0.0, 0.0])          # game over: final points (0 if no stats = did not play)
        if g is not None:
            rec.update(o=('' if g.home else '@') + g.opp, g=g.game_id, k=kickoff(g),
                       it=None if pd.isna(g.implied) else round(float(g.implied), 1))
        if pos in M.SKILL:
            rec['x'] = [round(float(getattr(r, 'x_' + c)), 2) for c in M.TARGETS]
            if pid in props_g:
                rec['pr'] = props_g[pid]
            nm = S.norm_name(r.name)
            if nm in odds and 'td_price' in odds[nm]:
                rec.setdefault('pr', {})['td_odds'] = odds[nm]['td_price']
            rec['mm'] = [round(float(getattr(r, 'model_' + f)), 2) for f in M.FMTS]
        e = ecr_for(pos, r.name, team)
        if e:
            rec['e'] = e
        if pid in last_map:
            rec['l'] = last_map[pid]
        out.append(rec)
    out.sort(key=lambda x: -x['m'][0])

    built = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    gm = {r.game_id: dict(h=r.team if r.home else r.opp, a=r.opp if r.home else r.team, k=kickoff(r),
                          tot=None if pd.isna(r.total_line) else float(r.total_line),
                          sp=None if pd.isna(r.spread) else float(r.spread if r.home else -r.spread),
                          done=bool(pd.notna(r.score)))
          for r in wk_games.itertuples() if r.home == 1}
    payload = dict(meta=dict(season=season, week=week, built=built, levels=M.LEVELS.round(4).tolist(),
                             prop_weight=PROP_W, n_players=len(out)),
                   tpl=tpl_out, cop=cop, games=gm, players=out)
    payload = clean(payload)
    check_payload(payload)                     # raises (and the site keeps the last good data.js) if the build looks broken
    js = 'window.FF=' + json.dumps(payload, separators=(',', ':'), allow_nan=False) + ';\n'
    with open(os.path.join(ROOT, 'data.js'), 'w') as fh:
        fh.write(js)
    hist_dir = os.path.join(ROOT, 'engine', 'history'); os.makedirs(hist_dir, exist_ok=True)
    pd.DataFrame([dict(player_id=r['i'], pos=r['p'], team=r['t'], pp=r['pp'], ppr=r['m'][0], half=r['m'][1], std=r['m'][2],
                       model_ppr=(r.get('mm') or r['m'])[0]) for r in out]).to_csv(
        os.path.join(hist_dir, f'{season}_wk{week:02d}.csv'), index=False)
    oos_mae = (oos[oos.season >= 2023].assign(e=lambda d: (d.act_ppr - d.pred_ppr).abs()).groupby('pos').e.mean().round(3).to_dict())
    report = dict(built=built, season=season, week=week, oos_mae_ppr=oos_mae, band_10_90_coverage=cover,
                  volatility_fit=vol_fit, copula=cop, copula_fit=cop_table, play_probability=play_rows, props_players=len(props_g), props_blended=nblend)
    with open(os.path.join(ROOT, 'engine', 'report.json'), 'w') as fh:
        json.dump(report, fh, indent=1)
    log(f'wrote data.js ({len(js)/1024:.0f} KB, {len(out)} players)')


if __name__ == '__main__':
    main()
