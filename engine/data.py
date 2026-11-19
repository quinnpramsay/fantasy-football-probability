"""Load nflverse data into one player-week table (offense + K) and one team-week table (D/ST),
with every stat ESPN's default scoring uses."""
import os
import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), 'cache')
SKILL = ['QB', 'RB', 'WR', 'TE']

# ESPN default scoring (verified from ESPN's own leaguedefaults scoringItems, 2026)
OFF_W = dict(pass_yds=0.04, pass_td=4, int=-2, rush_yds=0.1, rush_td=6, rec_yds=0.1, rec_td=6,
             fum_lost=-2, two_pt=2, st_td=6)          # + rec * (1 | 0.5 | 0)
REC_W = {'ppr': 1.0, 'half': 0.5, 'std': 0.0}
K_W = dict(fg_0_39=3, fg_40_49=4, fg_50_59=5, fg_60=6, fg_miss=-1, pat=1)
DST_PA = [(0, 0, 5), (1, 6, 4), (7, 13, 3), (14, 17, 1), (18, 27, 0), (28, 34, -1), (35, 45, -3), (46, 999, -5)]
DST_YA = [(-999, 99, 5), (100, 199, 3), (200, 299, 2), (300, 349, 0), (350, 399, -1), (400, 449, -3),
          (450, 499, -5), (500, 549, -6), (550, 9999, -7)]
DST_W = dict(sacks=1, ints=2, fum_rec=2, safeties=2, blocks=2, dst_td=6)


def bracket(v, table):
    for lo, hi, pts in table:
        if lo <= v <= hi:
            return pts
    return 0


def off_points(df, fmt='ppr'):
    pts = sum(df[c] * w for c, w in OFF_W.items()) + df['rec'] * REC_W[fmt]
    return pts


def k_points(df):
    return sum(df[c] * w for c, w in K_W.items())


def dst_points(df):
    pa = df['pts_allowed'].apply(lambda v: bracket(v, DST_PA))
    ya = df['yds_allowed'].apply(lambda v: bracket(round(v), DST_YA))
    return pa + ya + sum(df[c] * w for c, w in DST_W.items())


def load_games():
    g = pd.read_csv(f'{CACHE}/games.csv', low_memory=False)
    g = g[g.season >= 2021]
    rows = []
    for r in g.itertuples():
        tot, spr = r.total_line, r.spread_line          # spread_line > 0 => home favored
        for team, opp, home, ts, os_ in ((r.home_team, r.away_team, 1, r.home_score, r.away_score),
                                         (r.away_team, r.home_team, 0, r.away_score, r.home_score)):
            imp = (tot + (spr if home else -spr)) / 2 if pd.notna(tot) and pd.notna(spr) else np.nan
            rows.append(dict(game_id=r.game_id, season=r.season, week=r.week, game_type=r.game_type,
                             gameday=r.gameday, gametime=r.gametime, team=team, opp=opp, home=home,
                             implied=imp, opp_implied=(tot - imp) if pd.notna(imp) else np.nan,
                             total_line=tot, spread=(spr if home else -spr) if pd.notna(spr) else np.nan,
                             score=ts, opp_score=os_, roof=r.roof, wind=r.wind))
    gt = pd.DataFrame(rows)
    gt['team'] = gt.team.replace({'LA': 'LA', 'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'})
    gt['opp'] = gt.opp.replace({'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'})
    return gt


def load_players(seasons):
    fr = []
    for y in seasons:
        p = os.path.join(CACHE, f'player_{y}.csv')
        if os.path.exists(p):
            fr.append(pd.read_csv(p, low_memory=False))
    s = pd.concat(fr, ignore_index=True)
    s = s[s.season_type == 'REG']
    s = s[s.position.isin(SKILL + ['K'])].copy()
    z = lambda c: s[c].fillna(0) if c in s else 0
    s['pass_yds'] = z('passing_yards'); s['pass_td'] = z('passing_tds'); s['int'] = z('passing_interceptions')
    s['rush_yds'] = z('rushing_yards'); s['rush_td'] = z('rushing_tds')
    s['rec'] = z('receptions'); s['rec_yds'] = z('receiving_yards'); s['rec_td'] = z('receiving_tds')
    s['fum_lost'] = z('sack_fumbles_lost') + z('rushing_fumbles_lost') + z('receiving_fumbles_lost')
    s['two_pt'] = z('passing_2pt_conversions') + z('rushing_2pt_conversions') + z('receiving_2pt_conversions')
    s['st_td'] = z('special_teams_tds')
    s['targets'] = z('targets'); s['carries'] = z('carries'); s['pass_att'] = z('attempts')
    s['air_yards'] = z('receiving_air_yards')
    s['fg_0_39'] = z('fg_made_0_19') + z('fg_made_20_29') + z('fg_made_30_39')
    s['fg_40_49'] = z('fg_made_40_49'); s['fg_50_59'] = z('fg_made_50_59'); s['fg_60'] = z('fg_made_60_')
    s['fg_miss'] = (z('fg_att') - z('fg_made')).clip(lower=0)
    s['pat'] = z('pat_made'); s['fg_att'] = z('fg_att')
    for f in REC_W:
        s['pts_' + f] = np.where(s.position == 'K', k_points(s), off_points(s, f))
    keep = ['player_id', 'player_display_name', 'position', 'season', 'week', 'game_id', 'team', 'opponent_team',
            'headshot_url', 'pass_yds', 'pass_td', 'int', 'rush_yds', 'rush_td', 'rec', 'rec_yds', 'rec_td',
            'fum_lost', 'two_pt', 'st_td', 'targets', 'carries', 'pass_att', 'air_yards', 'fg_att',
            'pts_ppr', 'pts_half', 'pts_std']
    s = s[keep].rename(columns={'player_display_name': 'name', 'position': 'pos', 'opponent_team': 'opp'})
    return s.merge(load_snaps(seasons), on=['player_id', 'season', 'week'], how='left')


def load_snaps(seasons):
    ids = pd.read_csv(f'{CACHE}/players.csv', low_memory=False)[['gsis_id', 'pfr_id']].dropna()
    pf = dict(zip(ids.pfr_id, ids.gsis_id))
    fr = [pd.read_csv(os.path.join(CACHE, f'snaps_{y}.csv')) for y in seasons if os.path.exists(os.path.join(CACHE, f'snaps_{y}.csv'))]
    sn = pd.concat(fr, ignore_index=True)
    sn = sn[sn.game_type == 'REG']
    sn['player_id'] = sn.pfr_player_id.map(pf)
    sn = sn.dropna(subset=['player_id']).rename(columns={'offense_pct': 'snap_pct'})
    return sn[['player_id', 'season', 'week', 'snap_pct']].drop_duplicates(['player_id', 'season', 'week'])


def load_dst(seasons, games):
    fr = [pd.read_csv(os.path.join(CACHE, f'team_{y}.csv'), low_memory=False)
          for y in seasons if os.path.exists(os.path.join(CACHE, f'team_{y}.csv'))]
    t = pd.concat(fr, ignore_index=True)
    t = t[t.season_type == 'REG'].copy()
    z = lambda c: t[c].fillna(0) if c in t else 0
    t['off_yds'] = z('passing_yards') - z('sack_yards_lost').abs() + z('rushing_yards')   # nflverse stores sack yards as negative
    off = t[['season', 'week', 'team', 'off_yds']].rename(columns={'team': 'opp', 'off_yds': 'yds_allowed'})
    t['sacks'] = z('def_sacks'); t['ints'] = z('def_interceptions'); t['fum_rec'] = z('fumble_recovery_opp')
    t['safeties'] = z('def_safeties'); t['blocks'] = z('def_fg_blocks') + z('def_punt_blocks')
    t['dst_td'] = z('def_tds') + z('special_teams_tds')
    d = t[['season', 'week', 'team', 'opponent_team', 'sacks', 'ints', 'fum_rec', 'safeties', 'blocks', 'dst_td']]
    d = d.rename(columns={'opponent_team': 'opp'}).merge(off, on=['season', 'week', 'opp'], how='left')
    gs = games[['season', 'week', 'team', 'opp_score', 'game_id']]
    d = d.merge(gs, on=['season', 'week', 'team'], how='left')
    # points allowed excludes points the opponent's defense/ST scored (ESPN counts points allowed by the D/ST unit)
    opp_def = t[['season', 'week', 'team', 'def_tds', 'special_teams_tds']].fillna(0)
    opp_def = opp_def.assign(opp_nonoff_pts=6 * (opp_def.def_tds + opp_def.special_teams_tds))
    d = d.merge(opp_def[['season', 'week', 'team', 'opp_nonoff_pts']].rename(columns={'team': 'opp'}),
                on=['season', 'week', 'opp'], how='left')
    d['pts_allowed_all'] = d.opp_score
    d['pts_allowed'] = (d.opp_score - d.opp_nonoff_pts.fillna(0)).clip(lower=0)
    d['pts'] = dst_points(d)
    return d
