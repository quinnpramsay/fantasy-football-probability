"""Expert consensus ranks (Boris Chen's weekly files, built from FantasyPros expert rankings) and
sportsbook player-prop lines (DraftKings lines via ESPN's public odds feed; optional The Odds API for prices)."""
import csv, io, json, math, os, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {'User-Agent': 'ffprob/1.0'}
TEAM_NAMES = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL', 'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI', 'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL', 'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX', 'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC', 'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN', 'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT', 'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB', 'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'}


def http(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read().decode('utf-8')


def norm_name(s):
    s = (s or '').lower().replace('.', '').replace("'", '')
    s = re.sub(r'[^a-z ]', ' ', s.replace('-', ' '))
    return ' '.join(w for w in s.split() if w not in ('jr', 'sr', 'ii', 'iii', 'iv', 'v'))


# ---------- expert consensus (Boris Chen weekly files) ----------

ECR_BASE = 'https://s3-us-west-1.amazonaws.com/fftiers/out/weekly-{}.csv'
ECR_FILES = {('QB', None): 'QB', ('K', None): 'K', ('DST', None): 'DST'}
for p in ('RB', 'WR', 'TE'):
    ECR_FILES[(p, 'ppr')] = f'{p}-PPR'; ECR_FILES[(p, 'half')] = f'{p}-HALF'; ECR_FILES[(p, 'std')] = p


def fetch_ecr():
    """-> {(pos, fmt|None): [ {name, rank, avg, sd, best, worst, tier, matchup} ]}"""
    out = {}

    def one(item):
        key, f = item
        try:
            rows = list(csv.DictReader(io.StringIO(http(ECR_BASE.format(f)))))
        except Exception as e:
            print(f'  ecr {f}: {e}', file=sys.stderr); return key, []
        recs = []
        for r in rows:
            try:
                recs.append(dict(name=r['Player.Name'], rank=int(r['Rank']), avg=float(r['Avg.Rank']),
                                 sd=float(r['Std.Dev']), best=int(float(r['Best.Rank'])),
                                 worst=int(float(r['Worst.Rank'])), tier=r.get('Tier', ''), matchup=r.get('Matchup', '')))
            except (KeyError, ValueError):
                pass
        return key, recs

    with ThreadPoolExecutor(8) as ex:
        for k, v in ex.map(one, ECR_FILES.items()):
            out[k] = v
    return out


# ---------- props: DraftKings lines from ESPN's public odds feed ----------

PROP_TYPES = {
    'Total Passing Yards (incl. overtime)': 'pass_yds', 'Total Passing Touchdowns (incl. overtime)': 'pass_td',
    'Total Passing Interceptions (incl. overtime)': 'int', 'Total Passing Attempts (incl. overtime)': 'pass_att',
    'Total Pass Completions (incl. overtime)': 'pass_cmp', 'Total Rushing Yards (incl. overtime)': 'rush_yds',
    'Total Carries (incl. overtime)': 'carries', 'Total Receiving Yards (incl. overtime)': 'rec_yds',
    'Total Receptions (incl. overtime)': 'rec'}
CORE = 'https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{e}/competitions/{e}/odds'


def fetch_props_espn(event_ids):
    """-> {espn_athlete_id: {stat: line}} (DraftKings, provider 100)."""
    out = {}

    def one(e):
        res = {}
        try:
            odds = json.loads(http(CORE.format(e=e)))
            refs = [it['propBets']['$ref'] for it in odds.get('items', []) if it.get('propBets')]
            if not refs:
                return res
            base = refs[0].replace('http://', 'https://')
            page = 1
            while True:
                d = json.loads(http(base + f'&limit=1000&page={page}'))
                for it in d.get('items', []):
                    stat = PROP_TYPES.get(it.get('type', {}).get('name'))
                    tgt = (it.get('current') or {}).get('target', {}).get('value')
                    if not stat or tgt is None or 'athlete' not in it:
                        continue
                    aid = re.search(r'athletes/(\d+)', it['athlete']['$ref'])
                    if aid:
                        res.setdefault(int(aid.group(1)), {}).setdefault(stat, float(tgt))
                if page >= d.get('pageCount', 1):
                    break
                page += 1
        except Exception as ex:
            print(f'  props {e}: {ex}', file=sys.stderr)
        return res

    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(one, event_ids):
            for k, v in r.items():
                out.setdefault(k, {}).update(v)
    return out


# ---------- optional: The Odds API (prices incl. anytime TD) ----------

ODDS = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl'
ODDS_MARKETS = 'player_anytime_td,player_pass_yds,player_rush_yds,player_reception_yds,player_receptions'


def american_to_p(a):
    a = float(a)
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def fetch_props_oddsapi(key, start_iso, end_iso):
    """-> {normalized player name: {'td_p': p_anytime_td (vig-removed), 'odds': {market: [point, over, under]}}}
    Costs ~5 credits per game. Untested without a key; failures are logged and ignored."""
    out = {}
    try:
        evs = json.loads(http(f'{ODDS}/events?apiKey={key}&commenceTimeFrom={start_iso}&commenceTimeTo={end_iso}'))
    except Exception as e:
        print(f'  odds-api events: {e}', file=sys.stderr); return out
    for ev in evs:
        try:
            d = json.loads(http(f"{ODDS}/events/{ev['id']}/odds?apiKey={key}&regions=us&markets={ODDS_MARKETS}"
                                f"&oddsFormat=american&bookmakers=draftkings,fanduel"))
        except Exception as e:
            print(f'  odds-api {ev.get("id")}: {e}', file=sys.stderr); continue
        books = d.get('bookmakers', [])
        if not books:
            continue
        for mk in books[0].get('markets', []):
            for oc in mk.get('outcomes', []):
                nm = norm_name(oc.get('description', ''))
                rec = out.setdefault(nm, {'odds': {}})
                if mk['key'] == 'player_anytime_td' and oc.get('name') == 'Yes':
                    rec['td_price'] = oc['price']
                    rec['td_p'] = american_to_p(oc['price']) / 1.06     # typical hold on one-sided TD markets
                elif oc.get('name') in ('Over', 'Under'):
                    e = rec['odds'].setdefault(mk['key'], [oc.get('point'), None, None])
                    e[1 if oc['name'] == 'Over' else 2] = oc['price']
    return out


def line_to_mean(line, mp):
    """Map a sportsbook median line to a mean using the empirical median->mean table."""
    med, mean = mp
    if not med:
        return line
    if line <= med[0]:
        return line * mean[0] / med[0] if med[0] else line
    if line >= med[-1]:
        return line * mean[-1] / med[-1]
    for i in range(len(med) - 1):
        if med[i] <= line <= med[i + 1]:
            w = (line - med[i]) / (med[i + 1] - med[i])
            return mean[i] + w * (mean[i + 1] - mean[i])
    return line


def poisson_mean_from_line(line):
    """Receptions: find Poisson mean with P(X > line) = 0.5."""
    k = math.floor(line)
    lo, hi = 0.01, 30.0
    for _ in range(60):
        lam = (lo + hi) / 2
        cdf = sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k + 1))
        if 1 - cdf > 0.5:
            hi = lam
        else:
            lo = lam
    return (lo + hi) / 2
