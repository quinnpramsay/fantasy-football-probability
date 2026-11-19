"""Download raw inputs into engine/cache/. Re-downloads current-season files every run."""
import os, sys, time, urllib.request

CACHE = os.path.join(os.path.dirname(__file__), 'cache')
NV = 'https://github.com/nflverse/nflverse-data/releases/download'
FIRST_SEASON = 2021


def get(url, path, force=False):
    if os.path.exists(path) and not force:
        return path
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ffprob/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            tmp = path + '.tmp'
            with open(tmp, 'wb') as f:
                f.write(data)
            os.replace(tmp, path)
            return path
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f'download failed {url}: {e}')
            time.sleep(2 * (attempt + 1))


def fetch_all(season):
    os.makedirs(CACHE, exist_ok=True)
    for y in range(FIRST_SEASON, season + 1):
        cur = y == season
        get(f'{NV}/stats_player/stats_player_week_{y}.csv', f'{CACHE}/player_{y}.csv', cur)
        get(f'{NV}/stats_team/stats_team_week_{y}.csv', f'{CACHE}/team_{y}.csv', cur)
        get(f'{NV}/injuries/injuries_{y}.csv', f'{CACHE}/injuries_{y}.csv', cur)
        get(f'{NV}/snap_counts/snap_counts_{y}.csv', f'{CACHE}/snaps_{y}.csv', cur)
    get(f'{NV}/weekly_rosters/roster_weekly_{season}.csv', f'{CACHE}/roster_{season}.csv', True)
    get('https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv', f'{CACHE}/games.csv', True)
    get(f'{NV}/players/players.csv', f'{CACHE}/players.csv', True)


if __name__ == '__main__':
    fetch_all(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
    print('ok')
