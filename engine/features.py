"""Features built only from information available before kickoff (every rolling stat is shifted one game)."""
import numpy as np
import pandas as pd

STATS = ['pass_att', 'pass_yds', 'pass_td', 'int', 'carries', 'rush_yds', 'rush_td',
         'targets', 'rec', 'rec_yds', 'rec_td', 'fum_lost', 'air_yards', 'pts_ppr', 'two_pt', 'st_td']
HL = (2.5, 8)          # EWM half-lives in games


def _ewm_shift(df, key, cols, hl):
    g = df.groupby(key, sort=False)
    out = {}
    for c in cols:
        out[f'{c}_e{hl}'] = g[c].transform(lambda x: x.shift(1).ewm(halflife=hl, ignore_na=True, min_periods=1).mean())
    return pd.DataFrame(out, index=df.index)


def _order(df):
    df = df.sort_values(['season', 'week']).reset_index(drop=True)
    df['t'] = df.season * 100 + df.week
    return df


def vacated(pw, inj):
    """Opportunity left by teammates expected to miss the game.
    inj: rows (season, week, team, player_id, p_out). Uses each absent player's usage shares as of his last game."""
    d = pw.sort_values(['player_id', 'season', 'week']).copy()
    tm = d.groupby(['season', 'week', 'team'])[['targets', 'carries', 'air_yards', 'pass_att']].transform('sum')
    for c in ['targets', 'carries', 'air_yards']:
        d[c + '_sh'] = d[c] / tm[c].replace(0, np.nan)
    d['qb_start'] = ((d.pos == 'QB') & (d.pass_att >= 15)).astype(float)
    g = d.groupby('player_id')
    st = pd.DataFrame({'player_id': d.player_id, 't': d.season * 100 + d.week})
    for c in ['targets_sh', 'carries_sh', 'air_yards_sh', 'qb_start']:
        st[c] = g[c].transform(lambda x: x.ewm(halflife=4, ignore_na=True).mean()).values
    st = st.sort_values('t')
    q = inj.copy(); q['t'] = q.season * 100 + q.week
    q = q.sort_values('t')
    q = pd.merge_asof(q, st, on='t', by='player_id', allow_exact_matches=False, tolerance=100, direction='backward')
    q = q.dropna(subset=['targets_sh'])
    for c in ['targets_sh', 'carries_sh', 'air_yards_sh', 'qb_start']:
        q[c] = q[c] * q.p_out
    agg = q.groupby(['season', 'week', 'team'])[['targets_sh', 'carries_sh', 'air_yards_sh', 'qb_start']].sum()
    agg.columns = ['vac_tgt', 'vac_car', 'vac_air', 'vac_qb']
    own = q.set_index(['season', 'week', 'player_id'])[['targets_sh', 'carries_sh', 'air_yards_sh', 'qb_start']]
    own.columns = ['own_tgt', 'own_car', 'own_air', 'own_qb']
    return agg.reset_index(), own.reset_index()


def active_shares(pw, games, inj, rows):
    """Usage share among teammates expected to play: each player's recent volume divided by the recent volume of
    everyone who played the team's previous game (weighted by their chance of playing this week).
    Handles injured-reserve moves and returns that the injury report alone misses. rows: frame with
    player_id, season, week, team and the shifted EWM volume columns."""
    vol = ['targets', 'carries', 'air_yards', 'pass_att']
    d = pw.sort_values(['player_id', 'season', 'week'])
    g = d.groupby('player_id')
    st = d[['player_id', 'season', 'week', 'team']].copy()
    for c in vol:
        st[c] = g[c].transform(lambda x: x.ewm(halflife=4, ignore_na=True).mean()).values
    tg = games[games.game_type == 'REG'][['season', 'week', 'team']].sort_values(['team', 'season', 'week'])
    tg['pw'] = tg.groupby(['team', 'season']).week.shift(1)
    tg = tg.dropna(subset=['pw']).astype({'pw': int})
    a = tg.merge(st.rename(columns={'week': 'pw'}), on=['season', 'team', 'pw'])     # who played the previous game
    if inj is not None:
        a = a.merge(inj[['season', 'week', 'player_id', 'p_out']].drop_duplicates(['season', 'week', 'player_id']),
                    on=['season', 'week', 'player_id'], how='left')
        a['w'] = 1 - a.p_out.fillna(0)
    else:
        a['w'] = 1.0
    for c in vol:
        a[c] = a[c] * a.w
    den = a.groupby(['season', 'week', 'team'])[vol].sum().add_prefix('den_').reset_index()
    ina = a[['season', 'week', 'player_id']].drop_duplicates().assign(in_active=1)
    r = rows[['player_id', 'season', 'week', 'team'] + [f'{c}_e2.5' for c in vol]].merge(den, on=['season', 'week', 'team'], how='left')
    r = r.merge(ina, on=['season', 'week', 'player_id'], how='left')
    out = {}
    for c in vol:
        own = r[f'{c}_e2.5'].fillna(0)
        out[f'act_{c}_sh'] = (own / (r['den_' + c] + np.where(r.in_active == 1, 0, own)).replace(0, np.nan)).values
    return pd.DataFrame(out, index=rows.index)


def own_status(rows, injuries):
    """The player's own injury designation and practice level this week (known before kickoff)."""
    if injuries is None:
        return pd.DataFrame(index=rows.index)
    i = injuries[injuries.game_type == 'REG'].drop_duplicates(['season', 'week', 'gsis_id'], keep='last')
    i = i[['season', 'week', 'gsis_id', 'report_status', 'practice_status']].rename(columns={'gsis_id': 'player_id'})
    r = rows[['player_id', 'season', 'week']].merge(i, on=['player_id', 'season', 'week'], how='left')
    prac = r.practice_status.map({'Full Participation in Practice': 1, 'Limited Participation in Practice': 2,
                                  'Did Not Participate In Practice': 3}).fillna(0)
    return pd.DataFrame({'own_q': (r.report_status == 'Questionable').astype(float).values,
                         'own_prac': prac.values}, index=rows.index)


def skill_features(pw, games, future=None, inj=None, injuries=None):
    """pw: played player-weeks (QB/RB/WR/TE). future: rows (player_id,name,pos,team,season,week) to project."""
    df = pw.copy()
    df['played'] = 1
    if future is not None and len(future):
        f = future.copy(); f['played'] = 0
        df = pd.concat([df, f], ignore_index=True, sort=False)
    df = _order(df).sort_values(['player_id', 't']).reset_index(drop=True)
    # team shares (computed on played rows, merged back)
    tm = pw.groupby(['season', 'week', 'team'])[['targets', 'carries', 'pass_att', 'air_yards']].sum()
    tm.columns = ['tm_' + c for c in tm.columns]
    df = df.merge(tm.reset_index(), on=['season', 'week', 'team'], how='left')
    for c in ['targets', 'carries', 'air_yards']:
        df[c + '_sh'] = np.where(df.played == 1, df[c] / df['tm_' + c].replace(0, np.nan), np.nan)
    df = df.sort_values(['player_id', 't']).reset_index(drop=True)
    feats = [df]
    for hl in HL:
        feats.append(_ewm_shift(df, 'player_id', STATS + ['targets_sh', 'carries_sh', 'air_yards_sh', 'snap_pct'], hl))
    df = pd.concat(feats, axis=1)
    g = df.groupby('player_id', sort=False)
    df['n_prior'] = g.cumcount()
    df['n_season'] = df.groupby(['player_id', 'season']).cumcount()
    df['prev_t'] = g['t'].shift(1)
    df['gap'] = np.where(df.season * 100 + df.week - df.prev_t < 50,
                         (df.season * 100 + df.week) - df.prev_t, 18)  # weeks since last game (approx.)
    # season-to-date mean of ppr points
    df['pts_std_season'] = df.groupby(['player_id', 'season'])['pts_ppr'].transform(lambda x: x.shift(1).expanding().mean())
    df['pts_last_season'] = _last_season_mean(df)
    df = df.merge(games[['season', 'week', 'team', 'opp', 'home', 'implied', 'opp_implied', 'total_line',
                         'spread', 'roof', 'game_id', 'gameday', 'gametime']].rename(columns={'game_id': 'gid'}),
                  on=['season', 'week', 'team'], how='left', suffixes=('', '_g'))
    df['opp'] = df['opp_g'].fillna(df.get('opp')) if 'opp_g' in df else df['opp']
    df['dome'] = df.roof.isin(['dome', 'closed']).astype(float)
    df = df.merge(defense_allowed(pw), on=['season', 'week', 'opp', 'pos'], how='left')
    df = pd.concat([df, active_shares(pw, games, inj, df), own_status(df, injuries)], axis=1)
    if inj is not None:
        agg, own = vacated(pw, inj)
        df = df.merge(agg, on=['season', 'week', 'team'], how='left').merge(own, on=['season', 'week', 'player_id'], how='left')
        for a, b in [('vac_tgt', 'own_tgt'), ('vac_car', 'own_car'), ('vac_air', 'own_air'), ('vac_qb', 'own_qb')]:
            df[a] = (df[a].fillna(0) - df[b].fillna(0)).clip(lower=0)
        df = df.drop(columns=['own_tgt', 'own_car', 'own_air', 'own_qb'])
    return df


def _last_season_mean(df):
    m = df[df.played == 1].groupby(['player_id', 'season'])['pts_ppr'].mean().reset_index()
    m['season'] += 1
    m = m.rename(columns={'pts_ppr': 'pts_last_season'})
    return df[['player_id', 'season']].merge(m, on=['player_id', 'season'], how='left')['pts_last_season'].values


def defense_allowed(pw):
    """How many PPR points each defense allows to each position, relative to league average, EWM over prior games."""
    a = pw.groupby(['season', 'week', 'opp', 'pos'])['pts_ppr'].sum().reset_index()
    lg = a.groupby(['season', 'week', 'pos'])['pts_ppr'].transform('mean')
    a['rel'] = a.pts_ppr - lg
    # add a placeholder row for the next week of every defense so the shifted EWM exists for future games
    a = _order(a).sort_values(['opp', 'pos', 't'])
    last = a.groupby(['opp', 'pos']).tail(1).copy()
    last['week'] += 1; last['rel'] = np.nan
    a = pd.concat([a, last]).sort_values(['opp', 'pos', 'season', 'week'])
    a['def_rel'] = a.groupby(['opp', 'pos'])['rel'].transform(lambda x: x.shift(1).ewm(halflife=6, ignore_na=True).mean())
    # the placeholder week may collide with a real later week (bye); keep the real one
    a = a.drop_duplicates(['season', 'week', 'opp', 'pos'], keep='first')
    return a[['season', 'week', 'opp', 'pos', 'def_rel']]


def kicker_features(pk, games, future=None):
    df = pk.copy(); df['played'] = 1
    if future is not None and len(future):
        f = future.copy(); f['played'] = 0
        df = pd.concat([df, f], ignore_index=True, sort=False)
    df = _order(df).sort_values(['player_id', 't']).reset_index(drop=True)
    df['k_pts'] = df['pts_ppr']
    df = pd.concat([df] + [_ewm_shift(df, 'player_id', ['k_pts', 'fg_att'], hl) for hl in HL], axis=1)
    df['n_prior'] = df.groupby('player_id').cumcount()
    df = df.merge(games[['season', 'week', 'team', 'opp', 'home', 'implied', 'opp_implied', 'total_line',
                         'spread', 'roof', 'wind', 'game_id', 'gameday', 'gametime']].rename(columns={'game_id': 'gid'}),
                  on=['season', 'week', 'team'], how='left', suffixes=('', '_g'))
    if 'opp_g' in df:
        df['opp'] = df['opp_g'].fillna(df.get('opp'))
    df['dome'] = df.roof.isin(['dome', 'closed']).astype(float)
    df['wind'] = pd.to_numeric(df.wind, errors='coerce')
    return df


def dst_features(dst, games, future=None):
    df = dst.copy(); df['played'] = 1
    df['takeaways'] = df.ints + df.fum_rec
    if future is not None and len(future):
        f = future.copy(); f['played'] = 0
        df = pd.concat([df, f], ignore_index=True, sort=False)
    df = _order(df).sort_values(['team', 't']).reset_index(drop=True)
    cols = ['pts', 'sacks', 'takeaways', 'pts_allowed', 'yds_allowed', 'dst_td']
    df = pd.concat([df] + [_ewm_shift(df, 'team', cols, hl) for hl in HL], axis=1)
    # opponent offense tendencies: sacks taken / giveaways (from the other side of the same games)
    o = dst[['season', 'week', 'team', 'opp', 'sacks', 'ints', 'fum_rec']].copy()
    o['give'] = o.ints + o.fum_rec
    o = o.rename(columns={'opp': 'off', 'sacks': 'sacked'})[['season', 'week', 'off', 'sacked', 'give']]
    o = _order(o).sort_values(['off', 't'])
    last = o.groupby('off').tail(1).copy(); last['week'] += 1; last[['sacked', 'give']] = np.nan
    o = pd.concat([o, last]).sort_values(['off', 'season', 'week']).drop_duplicates(['season', 'week', 'off'])
    for c in ['sacked', 'give']:
        o[f'opp_{c}_e'] = o.groupby('off')[c].transform(lambda x: x.shift(1).ewm(halflife=6, ignore_na=True).mean())
    df = df.drop(columns=[c for c in ['opp'] if c in df and df[c].isna().all()], errors='ignore')
    df = df.merge(games[['season', 'week', 'team', 'opp', 'home', 'implied', 'opp_implied', 'total_line',
                         'spread', 'game_id', 'gameday', 'gametime']].rename(columns={'game_id': 'gid'}),
                  on=['season', 'week', 'team'], how='left', suffixes=('', '_g'))
    if 'opp_g' in df:
        df['opp'] = df['opp_g'].fillna(df.get('opp'))
    df = df.merge(o[['season', 'week', 'off', 'opp_sacked_e', 'opp_give_e']].rename(columns={'off': 'opp'}),
                  on=['season', 'week', 'opp'], how='left')
    df['n_prior'] = df.groupby('team').cumcount()
    return df
