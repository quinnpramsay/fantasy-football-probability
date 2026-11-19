'use strict';
/* fantasy football probability — browser engine + UI. No dependencies. Data: window.FF (data.js). */
(function () {
const FF = window.FF, LV = FF.meta.levels, FMT = ['ppr', 'half', 'std'], N = 20000, LUTN = 256;
const SLOTS = ['QB', 'RB', 'RB', 'WR', 'WR', 'TE', 'FLEX', 'DST', 'K', 'BE', 'BE', 'BE', 'BE', 'BE', 'BE', 'BE', 'IR'];
const NSTART = 9, ELIG = { QB: ['QB'], RB: ['RB'], WR: ['WR'], TE: ['TE'], FLEX: ['RB', 'WR', 'TE'], DST: ['DST'], K: ['K'] };
const SLABEL = { DST: 'D/ST', BE: 'Bench', IR: 'IR' };
const STAT_NAMES = ['pass yds', 'pass TD', 'INT', 'rush yds', 'rush TD', 'receptions', 'rec yds', 'rec TD', 'fumbles lost'];
const PROP_NAMES = { pass_yds: 'pass yds', pass_td: 'pass TD', int: 'INT', pass_att: 'pass att', pass_cmp: 'completions',
  rush_yds: 'rush yds', carries: 'carries', rec_yds: 'rec yds', rec: 'receptions', td_odds: 'anytime TD (odds)' };
const STAT_KEY = { pass_yds: 0, pass_td: 1, int: 2, rush_yds: 3, rec: 5, rec_yds: 6 };
const $ = (s, e = document) => e.querySelector(s);
const byId = new Map(FF.players.map(p => [p.i, p]));
let fmt = 0, view = null;
const state = { A: Array(SLOTS.length).fill(null), B: Array(SLOTS.length).fill(null) };   // {id, fin}

/* ---------- numerics ---------- */
function Phi(z) { const t = 1 / (1 + 0.2316419 * Math.abs(z)); const d = 0.3989422804014327 * Math.exp(-z * z / 2);
  const p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))); return z > 0 ? 1 - p : p; }
function hash(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const noiseCache = new Map(), unifCache = new Map();
function noise(id) {           // standard normals, fixed per player -> common random numbers across lineups
  let z = noiseCache.get(id); if (z) return z;
  z = new Float64Array(N); const r = rng(hash(id + FF.meta.built));
  for (let i = 0; i < N; i += 2) { const u = r() || 1e-12, v = r(); const m = Math.sqrt(-2 * Math.log(u));
    z[i] = m * Math.cos(6.283185307179586 * v); if (i + 1 < N) z[i + 1] = m * Math.sin(6.283185307179586 * v); }
  noiseCache.set(id, z); return z;
}
function unif(id) { let u = unifCache.get(id); if (u) return u; u = new Float64Array(N); const r = rng(hash('play' + id)); for (let i = 0; i < N; i++) u[i] = r(); unifCache.set(id, u); return u; }

/* ---------- player distributions ---------- */
function tmplAt(pos, mu) {
  const t = FF.tpl[FMT[fmt]][pos], c = t.c, q = t.q;
  if (mu <= c[0]) return q[0]; if (mu >= c[c.length - 1]) return q[q.length - 1];
  let j = 0; while (c[j + 1] < mu) j++;
  const w = (mu - c[j]) / (c[j + 1] - c[j]); return q[j].map((v, k) => v * (1 - w) + q[j + 1][k] * w);
}
const lutCache = new Map();
function lut(p) {               // quantile function sampled at u = k/LUTN (conditional on playing)
  const key = p.i + '|' + fmt; let L = lutCache.get(key); if (L) return L;
  const mu = p.m[fmt], base = tmplAt(p.p, mu), v = p.v;
  const q = base.map(b => Math.max(mu + v * b, mu + base[0]));
  const ext = [q[0] - (q[1] - q[0]), ...q, q[q.length - 1] + 2 * (q[q.length - 1] - q[q.length - 2])], lv = [0, ...LV, 1];
  L = new Float64Array(LUTN + 1); let j = 0;
  for (let k = 0; k <= LUTN; k++) { const u = k / LUTN; while (j < lv.length - 2 && lv[j + 1] < u) j++;
    const w = (u - lv[j]) / (lv[j + 1] - lv[j]); L[k] = ext[j] + w * (ext[j + 1] - ext[j]); }
  lutCache.set(key, L); return L;
}
function sample(L, u) { const x = u * LUTN, k = Math.min(LUTN - 1, x | 0), f = x - k; return L[k] + f * (L[k + 1] - L[k]); }
function condMoments(p) { const L = lut(p); let s = 0, s2 = 0; for (let k = 0; k < LUTN; k++) { const x = (L[k] + L[k + 1]) / 2; s += x; s2 += x * x; }
  const m = s / LUTN; return [m, Math.sqrt(Math.max(0, s2 / LUTN - m * m))]; }
function pStats(p, fin) {        // mean, sd, p10, p90 including the chance he does not play
  if (fin != null) return { mean: fin, sd: 0, p10: fin, p90: fin, pp: 1 };
  const [m, s] = condMoments(p), pp = p.pp, mean = pp * m, sd = Math.sqrt(Math.max(0, pp * (s * s + m * m) - mean * mean));
  return { mean, sd, p10: mixQ(p, 0.1), p90: mixQ(p, 0.9), pp };
}
function mixQ(p, t) {           // quantile of (1-pp)*delta(0) + pp*F
  const L = lut(p), pp = p.pp; if (pp >= 0.999) return sample(L, t);
  if (pp <= 0.001) return 0;
  const cdf = x => (x >= 0 ? 1 - pp : 0) + pp * Fx(L, x);
  let lo = Math.min(0, L[0]) - 1, hi = Math.max(0, L[LUTN]) + 1;
  for (let i = 0; i < 50; i++) { const md = (lo + hi) / 2; if (cdf(md) < t) lo = md; else hi = md; } return (lo + hi) / 2;
}
function Fx(L, x) { if (x <= L[0]) return 0; if (x >= L[LUTN]) return 1; let lo = 0, hi = LUTN;
  while (hi - lo > 1) { const md = (lo + hi) >> 1; if (L[md] <= x) lo = md; else hi = md; }
  const d = L[hi] - L[lo]; return (lo + (d > 0 ? (x - L[lo]) / d : 0)) / LUTN; }

/* ---------- simulation ---------- */
function pairKey(a, b) {
  if (a.p === 'DST' || b.p === 'DST') { if (a.p === 'DST' && b.p === 'DST') return 'DST-DST opp';
    const d = a.p === 'DST' ? a : b, o = a.p === 'DST' ? b : a; return 'DST-' + o.p + ' ' + (d.t === o.t ? 'own' : 'vs'); }
  const [x, y] = [a.p, b.p].sort(); return x + '-' + y + ' ' + (a.t === b.t ? 'same' : 'opp');
}
function chol(C) { const n = C.length; for (let s = 1; s > 0.05; s *= 0.9) { const L = Array.from({ length: n }, () => new Float64Array(n)); let ok = true;
  for (let i = 0; i < n && ok; i++) for (let j = 0; j <= i; j++) { let v = i === j ? 1 : C[i][j] * s; for (let k = 0; k < j; k++) v -= L[i][k] * L[j][k];
    if (i === j) { if (v <= 1e-9) { ok = false; break; } L[i][i] = Math.sqrt(v); } else L[i][j] = v / L[j][j]; }
  if (ok) return L; } return null; }
/* entries: [{p, fin, side}] -> {A: Float64Array, B: Float64Array, per: [Float64Array]} */
function simulate(entries) {
  const per = entries.map(() => new Float64Array(N)), Z = entries.map(e => e.fin != null ? null : noise(e.p.i));
  const groups = new Map();
  entries.forEach((e, i) => { if (e.fin == null && e.p.g && e.p.pp > 0) { const g = groups.get(e.p.g) || []; g.push(i); groups.set(e.p.g, g); } });
  const corrZ = entries.map((e, i) => Z[i]);
  for (const idx of groups.values()) {
    if (idx.length < 2) continue; idx.sort((a, b) => entries[a].p.i < entries[b].p.i ? -1 : 1);
    const C = idx.map(a => idx.map(b => a === b ? 1 : (FF.cop[pairKey(entries[a].p, entries[b].p)] || 0)));
    const L = chol(C); if (!L) continue;
    const out = idx.map(() => new Float64Array(N));
    for (let r = 0; r < idx.length; r++) { const o = out[r];
      for (let k = 0; k <= r; k++) { const l = L[r][k]; if (!l) continue; const z = Z[idx[k]]; for (let s = 0; s < N; s++) o[s] += l * z[s]; } }
    idx.forEach((a, r) => corrZ[a] = out[r]);
  }
  const A = new Float64Array(N), B = new Float64Array(N);
  entries.forEach((e, i) => {
    const x = per[i], T = e.side === 'A' ? A : B;
    if (e.fin != null) { x.fill(e.fin); } else if (e.p.pp <= 0) { /* zeros */ } else {
      const L = lut(e.p), z = corrZ[i], pp = e.p.pp, U = pp < 1 ? unif(e.p.i) : null, rnd = e.p.p === 'K' || e.p.p === 'DST';
      for (let s = 0; s < N; s++) { if (U && U[s] >= pp) { x[s] = 0; continue; } let v = sample(L, Phi(z[s])); x[s] = rnd ? Math.round(v) : v; }
    }
    for (let s = 0; s < N; s++) T[s] += x[s];
  });
  return { A, B, per };
}
const finOf = (s, p) => s.fin != null ? s.fin : (p.fp ? p.fp[fmt] : null);
function starters(side, arr) { const out = []; for (let k = 0; k < NSTART; k++) { const s = arr[k]; if (s && byId.has(s.id)) { const p = byId.get(s.id); out.push({ p, fin: finOf(s, p), side, slot: k }); } } return out; }
function evaluate(a = state.A, b = state.B) {
  const ea = starters('A', a), eb = starters('B', b); if (!ea.length || !eb.length) return null;
  const all = ea.concat(eb), r = simulate(all); let w = 0, t = 0;
  for (let s = 0; s < N; s++) { const d = r.A[s] - r.B[s]; if (Math.abs(d) < 0.005) t++; else if (d > 0) w++; }
  return { pw: w / N, pt: t / N, A: r.A, B: r.B, per: r.per, entries: all };
}

/* ---------- statistics ---------- */
function describe(x) {
  const n = x.length, s = Float64Array.from(x).sort(); let m = 0; for (const v of x) m += v; m /= n;
  let v2 = 0, v3 = 0; for (const v of x) { const d = v - m; v2 += d * d; v3 += d * d * d; } v2 /= n; v3 /= n;
  const q = p => s[Math.min(n - 1, Math.max(0, Math.round(p * (n - 1))))], sd = Math.sqrt(v2), k = kde(x, s[0], s[n - 1], sd);
  let mi = 0; for (let i = 1; i < k.y.length; i++) if (k.y[i] > k.y[mi]) mi = i;
  return { mean: m, median: q(0.5), mode: k.x[mi], sd, var: v2, skew: sd > 0 ? v3 / (sd * sd * sd) : 0,
    p1: q(0.01), p5: q(0.05), p10: q(0.1), p25: q(0.25), p75: q(0.75), p90: q(0.9), p95: q(0.95), p99: q(0.99), kde: k };
}
function kde(x, lo, hi, sd) {
  const bw = Math.max(0.6, 1.06 * sd * Math.pow(x.length, -0.2)), a = lo - 3 * bw, b = hi + 3 * bw, M = 240, step = (b - a) / M;
  const h = new Float64Array(M); for (const v of x) { const k = Math.floor((v - a) / step); if (k >= 0 && k < M) h[k]++; }
  const ker = [], R = Math.ceil(3 * bw / step); for (let k = -R; k <= R; k++) ker.push(Math.exp(-0.5 * (k * step / bw) ** 2));
  const ks = ker.reduce((s, v) => s + v, 0), y = new Float64Array(M), xs = new Float64Array(M);
  for (let i = 0; i < M; i++) { let s = 0; for (let k = -R; k <= R; k++) { const j = i + k; if (j >= 0 && j < M) s += h[j] * ker[k + R]; }
    y[i] = s / ks / x.length / step; xs[i] = a + (i + 0.5) * step; }
  return { x: xs, y };
}
const f1 = v => (v == null || isNaN(v)) ? '' : (Math.round(v * 10) / 10).toFixed(1);
const f2 = v => (Math.round(v * 100) / 100).toFixed(2);
const pct = v => (v * 100).toFixed(1) + '%';
const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/* ---------- charts ---------- */
function curves(series, title, wide) {    // series: [{k:kde, dash, label}]
  const W = 720, H = 230, pl = 34, pr = 10, pt = 16, pb = 26;
  let x0 = Infinity, x1 = -Infinity, y1 = 0;
  for (const s of series) { const { x, y } = s.k; for (let i = 0; i < x.length; i++) if (y[i] > y1) y1 = y[i];
    const th = Math.max(...y) * 0.004; for (let i = 0; i < x.length; i++) if (y[i] > th) { x0 = Math.min(x0, x[i]); x1 = Math.max(x1, x[i]); } }
  const X = v => pl + (v - x0) / (x1 - x0) * (W - pl - pr), Y = v => H - pb - v / y1 * (H - pt - pb);
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}" style="max-width:${wide ? 900 : 720}px">`;
  const span = x1 - x0, st = span > 120 ? 20 : span > 50 ? 10 : span > 20 ? 5 : 2;
  for (let v = Math.ceil(x0 / st) * st; v <= x1; v += st) g += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - pb}" y2="${H - pb + 4}" stroke="#000"/><text x="${X(v)}" y="${H - 8}" text-anchor="middle">${v}</text>`;
  g += `<line x1="${pl}" x2="${W - pr}" y1="${H - pb}" y2="${H - pb}" stroke="#000"/>`;
  series.forEach((s, j) => { const { x, y } = s.k; let d = '';
    for (let i = 0; i < x.length; i++) if (x[i] >= x0 && x[i] <= x1) d += (d ? 'L' : 'M') + X(x[i]).toFixed(1) + ' ' + Y(y[i]).toFixed(1);
    g += `<path d="${d}" fill="none" stroke="#000" stroke-width="${s.dash ? 1.5 : 2}" ${s.dash ? 'stroke-dasharray="6 4"' : ''}/>`;
    if (s.mean != null) g += `<line x1="${X(s.mean)}" x2="${X(s.mean)}" y1="${pt}" y2="${H - pb}" stroke="#000" stroke-width="0.7" ${s.dash ? 'stroke-dasharray="2 3"' : ''}/>`;
    g += `<line x1="${W - pr - 150}" x2="${W - pr - 120}" y1="${pt + j * 15}" y2="${pt + j * 15}" stroke="#000" stroke-width="2" ${s.dash ? 'stroke-dasharray="6 4"' : ''}/><text x="${W - pr - 114}" y="${pt + 4 + j * 15}">${esc(s.label)}</text>`; });
  return g + `<text x="${pl}" y="11">${esc(title)}</text></svg>`;
}

/* ---------- lineup tables ---------- */
const label = p => p.n + ' · ' + p.t + ' ' + (p.p === 'DST' ? 'D/ST' : p.p);
const byLabel = new Map(FF.players.map(p => [label(p).toLowerCase(), p]));
const byName = new Map(); FF.players.forEach(p => { const k = p.n.toLowerCase(); byName.set(k, byName.has(k) ? null : p); });
function buildLists() {
  const groups = { QB: [], RB: [], WR: [], TE: [], K: [], DST: [] };
  FF.players.forEach(p => groups[p.p] && groups[p.p].push(p));
  const opt = arr => arr.map(p => `<option value="${esc(label(p))}">`).join('');
  let h = ''; for (const k in groups) h += `<datalist id="dl-${k}">${opt(groups[k])}</datalist>`;
  h += `<datalist id="dl-FLEX">${opt(FF.players.filter(p => ELIG.FLEX.includes(p.p)))}</datalist><datalist id="dl-ALL">${opt(FF.players)}</datalist>`;
  $('#dls').innerHTML = h;
}
function tag(p) { if (!p.s) return ''; const t = p.s === 'BYE' ? 'BYE' : p.s + (p.pp < 1 && p.pp > 0 ? ' ' + Math.round(p.pp * 100) + '%' : '');
  return `<span class="tag" title="chance he plays: ${Math.round(p.pp * 100)}%">${esc(t)}</span>`; }
function renderTable(side) {
  const arr = state[side];
  let h = '<thead><tr><th>slot</th><th>player</th><th>opp</th><th class="n">proj</th><th class="n">sd</th><th class="n">10–90%</th><th class="n">final</th><th></th></tr></thead><tbody>';
  SLOTS.forEach((sl, k) => {
    const s = arr[k], p = s && byId.get(s.id), st = p ? pStats(p, finOf(s, p)) : null;
    if (k === NSTART) h += '<tr class="sep"><td colspan="8"></td></tr>';
    const dl = ELIG[sl] ? (sl === 'FLEX' ? 'FLEX' : sl) : 'ALL';
    h += `<tr class="${k >= NSTART ? 'be' : ''}"><td>${SLABEL[sl] || sl}</td>
      <td class="pc"><div><input list="dl-${dl}" data-side="${side}" data-k="${k}" value="${p ? esc(label(p)) : ''}" placeholder="${sl === 'BE' || sl === 'IR' ? 'any player' : (SLABEL[sl] || sl)}" autocomplete="off" spellcheck="false">${p ? tag(p) : ''}</div></td>
      <td>${p ? esc(p.o || '—') : ''}</td><td class="n">${st ? f1(st.mean) : ''}</td><td class="n">${st ? f1(st.sd) : ''}</td>
      <td class="n">${st ? f1(st.p10) + '–' + f1(st.p90) : ''}</td>
      <td class="n">${p ? `<input class="f" type="number" step="0.01" data-fin="${side}" data-k="${k}" value="${s.fin != null ? s.fin : ''}" placeholder="${p.fp ? f1(p.fp[fmt]) : ''}" title="${p.fp ? 'game over: final points filled in automatically (type to override)' : 'enter final points if his game is over'}">` : ''}</td>
      <td>${p ? `<button type="button" data-info="${esc(p.i)}" title="player details">i</button>` : ''}${p && side === 'A' && k < NSTART ? ` <button type="button" data-swap="${k}" title="compare replacements">⇄</button>` : ''}</td></tr>`;
  });
  $('#t' + side).innerHTML = h + '</tbody>';
}
function slotOk(sl, p) { return !ELIG[sl] || ELIG[sl].includes(p.p); }
function resolve(text, sl) {
  const t = text.trim().toLowerCase(); if (!t) return null;
  let p = byLabel.get(t) || byName.get(t);
  if (!p) { const c = FF.players.filter(q => slotOk(sl, q) && q.n.toLowerCase().includes(t)); if (c.length === 1) p = c[0]; }
  return p && slotOk(sl, p) ? p : undefined;
}

/* ---------- top line + panels ---------- */
let last = null;
function refresh() {
  renderTable('A'); renderTable('B');
  last = evaluate();
  const wp = $('#wp');
  if (!last) { wp.innerHTML = 'enter both lineups<small>&nbsp;</small>'; $('#sumA').textContent = $('#sumB').textContent = ''; }
  else {
    const a = mean(last.A), b = mean(last.B), sa = sd(last.A), sb = sd(last.B);
    wp.innerHTML = `YOU ${pct(last.pw + last.pt / 2)} &nbsp;·&nbsp; ${pct(1 - last.pw - last.pt / 2)} OPP<small>projected ${f1(a)} ± ${f1(sa)} &nbsp;vs&nbsp; ${f1(b)} ± ${f1(sb)}${last.pt > 0.0005 ? ' · tie ' + pct(last.pt) : ''} · click for full statistics</small>`;
    $('#sumA').textContent = 'proj ' + f1(a); $('#sumB').textContent = 'proj ' + f1(b);
  }
  if (view) showView(view);
  save();
}
const mean = x => { let s = 0; for (const v of x) s += v; return s / x.length; };
const sd = x => { const m = mean(x); let s = 0; for (const v of x) s += (v - m) ** 2; return Math.sqrt(s / x.length); };
function closeBtn() { return '<button type="button" class="x" data-close>close</button>'; }
function showView(v) {
  view = v; const P = $('#panel');
  if (!v) { P.innerHTML = ''; return; }
  if (v.type === 'analysis') P.innerHTML = last ? analysisHTML(last, 'Matchup analysis') : '';
  else if (v.type === 'player') P.innerHTML = playerHTML(byId.get(v.id));
  else if (v.type === 'swap') P.innerHTML = swapHTML(v);
}
function analysisHTML(r, title, extra = '') {
  const a = describe(r.A), b = describe(r.B), diff = r.A.map((v, i) => v - r.B[i]), dm = describe(diff);
  let w10 = 0, l10 = 0; for (const d of diff) { if (d >= 10) w10++; if (d <= -10) l10++; }
  const rows = [['mean (projection)', 'mean'], ['median', 'median'], ['mode (most likely)', 'mode'], ['standard deviation', 'sd'], ['variance', 'var'],
    ['skewness', 'skew'], ['1st percentile', 'p1'], ['5th percentile', 'p5'], ['10th percentile (floor)', 'p10'], ['25th percentile', 'p25'],
    ['75th percentile', 'p75'], ['90th percentile (ceiling)', 'p90'], ['95th percentile', 'p95'], ['99th percentile', 'p99']];
  let h = `<div class="h2"><span>${esc(title)}</span>${closeBtn()}</div>${extra}`;
  h += curves([{ k: a.kde, label: 'you', mean: a.mean }, { k: b.kde, dash: true, label: 'opponent', mean: b.mean }], 'team score distributions (' + N.toLocaleString() + ' simulated weeks)', true);
  h += '<div class="grid2"><div class="tw"><table><thead><tr><th>team score</th><th class="n">you</th><th class="n">opp</th></tr></thead><tbody>';
  for (const [n, k] of rows) h += `<tr><td>${n}</td><td class="n">${k === 'var' ? f1(a[k]) : k === 'skew' ? f2(a[k]) : f1(a[k])}</td><td class="n">${k === 'var' ? f1(b[k]) : k === 'skew' ? f2(b[k]) : f1(b[k])}</td></tr>`;
  h += '</tbody></table></div><div class="tw"><table><thead><tr><th>margin (you − opp)</th><th class="n"></th></tr></thead><tbody>';
  h += [['P(win)', pct(r.pw + r.pt / 2)], ['P(lose)', pct(1 - r.pw - r.pt)], ['P(exact tie)', pct(r.pt)], ['expected margin', f1(dm.mean)],
    ['median margin', f1(dm.median)], ['margin sd', f1(dm.sd)], ['P(win by 10+)', pct(w10 / N)], ['P(lose by 10+)', pct(l10 / N)],
    ['margin 10th–90th pct', f1(dm.p10) + ' to ' + f1(dm.p90)]].map(([k, v]) => `<tr><td>${k}</td><td class="n">${v}</td></tr>`).join('');
  h += '</tbody></table></div></div>';
  for (const side of ['A', 'B']) {
    const T = side === 'A' ? r.A : r.B, tv = describe(T).var;
    h += `<div class="h2"><span>${side === 'A' ? 'your' : 'opponent'} starters</span></div><div class="tw"><table><thead><tr><th>player</th><th>opp</th><th class="n">mean</th><th class="n">median</th><th class="n">sd</th><th class="n">10th</th><th class="n">90th</th><th class="n">plays</th><th class="n">share of team variance</th></tr></thead><tbody>`;
    r.entries.forEach((e, i) => { if (e.side !== side) return; const x = r.per[i], d = describe(x);
      let c = 0; const mT = mean(T); for (let s = 0; s < N; s++) c += (x[s] - d.mean) * (T[s] - mT); c /= N;
      h += `<tr><td>${esc(e.p.n)}${e.fin != null ? ' <span class="tag">final</span>' : ''}</td><td>${esc(e.p.o || '')}</td><td class="n">${f1(d.mean)}</td><td class="n">${f1(d.median)}</td><td class="n">${f1(d.sd)}</td><td class="n">${f1(d.p10)}</td><td class="n">${f1(d.p90)}</td><td class="n">${e.fin != null ? '—' : Math.round(e.p.pp * 100) + '%'}</td><td class="n">${tv > 0 ? Math.round(c / tv * 100) + '%' : '—'}</td></tr>`; });
    h += '</tbody></table></div>';
  }
  return h + `<p class="mute">Simulated with each player's own outcome distribution (skewed, floored, from 2023–26 results), his play probability, and measured same-game correlations (QB with his receivers, a D/ST against the offense it faces). Share of variance = covariance with the team total ÷ team variance.</p>`;
}
function swapHTML(v) {
  const k = v.k, cur = state.A[k], cp = cur && byId.get(cur.id); if (!cp) return '';
  const sl = SLOTS[k], cands = [{ label: 'current', id: cur.id, j: null }];
  state.A.forEach((s, j) => { if (j >= NSTART && s && byId.has(s.id) && slotOk(sl, byId.get(s.id))) cands.push({ id: s.id, j }); });
  if (v.extra && byId.has(v.extra) && !cands.some(c => c.id === v.extra)) cands.push({ id: v.extra, j: 'fa' });
  const res = cands.map(c => { const a = lineupWith(k, c); const r = evaluate(a); const p = byId.get(c.id), st = pStats(p, c.j === null ? finOf(cur, p) : (p.fp ? p.fp[fmt] : null));
    return { c, p, st, r, mA: mean(r.A), sA: sd(r.A), win: r.pw + r.pt / 2 }; });
  const base = res[0].win, best = res.reduce((m, x) => x.win > m.win ? x : m, res[0]);
  let h = `<div class="h2"><span>Change ${esc(SLABEL[sl] || sl)}: ${esc(cp.n)}</span>${closeBtn()}</div>`;
  h += `<p>Compare any player: <input list="dl-${ELIG[sl] ? (sl === 'FLEX' ? 'FLEX' : sl) : 'ALL'}" id="faIn" style="width:260px" placeholder="type a name (free agent, trade target)"> <span class="mute">bench players eligible for this slot are listed automatically. Click a row for the full analysis.</span></p>`;
  h += '<div class="tw"><table><thead><tr><th>player</th><th>opp</th><th class="n">proj</th><th class="n">sd</th><th class="n">10–90%</th><th class="n">team mean</th><th class="n">team sd</th><th class="n">P(win)</th><th class="n">Δ</th></tr></thead><tbody>';
  res.forEach((x, i) => { h += `<tr class="pick${x === best ? ' best' : ''}" data-pick="${i}"><td>${esc(x.p.n)}${i === 0 ? ' <span class="mute">(current)</span>' : x.c.j === 'fa' ? ' <span class="mute">(not on roster)</span>' : ''}${tag(x.p)}</td><td>${esc(x.p.o || '—')}</td><td class="n">${f1(x.st.mean)}</td><td class="n">${f1(x.st.sd)}</td><td class="n">${f1(x.st.p10)}–${f1(x.st.p90)}</td><td class="n">${f1(x.mA)}</td><td class="n">${f1(x.sA)}</td><td class="n">${pct(x.win)}</td><td class="n">${i === 0 ? '' : (x.win >= base ? '+' : '') + ((x.win - base) * 100).toFixed(1)}</td></tr>`; });
  h += '</tbody></table></div><p class="mute">Bold = highest win probability. When you are favored a steadier player can beat a higher projection; when you are the underdog the higher-variance option usually helps.</p>';
  const pick = v.pick != null ? res[v.pick] : null;
  if (pick && pick.c.j !== null) h += analysisHTML(pick.r, 'Analysis with ' + pick.p.n + ' at ' + (SLABEL[sl] || sl), `<p><button type="button" data-apply="${v.pick}">apply this change</button></p>`);
  else if (pick) h += analysisHTML(pick.r, 'Analysis with current lineup');
  v._res = res; return h;
}
function lineupWith(k, c) {
  const a = state.A.slice(); if (c.j === null) return a;
  if (c.j === 'fa') { a[k] = { id: c.id, fin: null }; return a; }
  const t = a[k]; a[k] = a[c.j]; a[c.j] = t; return a;
}
function playerHTML(p) {
  if (!p) return '';
  const [m, s] = condMoments(p), st = pStats(p, null), L = lut(p), g = p.g && FF.games[p.g];
  const qx = t => mixQ(p, t);
  let h = `<div class="h2"><span>${esc(p.n)} — ${esc(p.t)} ${p.p === 'DST' ? 'D/ST' : p.p}</span>${closeBtn()}</div>`;
  h += `<p>${p.o ? 'opponent ' + esc(p.o) + (p.k ? ' · ' + esc(p.k) : '') : 'no game this week'}${p.it != null ? ' · Vegas team total ' + f1(p.it) : ''}${g && g.tot != null ? ' · game total ' + g.tot + ' · ' + (g.sp > 0 ? g.h + ' −' + g.sp : g.sp < 0 ? g.a + ' −' + (-g.sp) : 'pick') : ''}
    ${p.p === 'DST' ? '' : '· status ' + (p.s ? esc(p.s) : 'no injury designation') + ' · chance he plays ' + Math.round(p.pp * 100) + '%'}</p>`;
  const xs = []; for (let i = 0; i < 4000; i++) { const u = (i + 0.5) / 4000; xs.push(sample(L, u)); }
  const kd = kde(Float64Array.from(xs), xs[0], xs[xs.length - 1], s);
  h += '<div class="grid2"><div>' + curves([{ k: kd, label: 'if he plays', mean: m }], 'fantasy points distribution (' + FMT[fmt] + ')') + '</div><div class="tw"><table><tbody>';
  const P = t => sample(L, t), above = x => 1 - Fx(L, x);
  h += [['projection (incl. chance he sits)', f1(st.mean)], ['mean if he plays', f1(m)], ['median if he plays', f1(P(0.5))], ['standard deviation', f1(s)],
    ['variance', f1(s * s)], ['5th / 95th pct', f1(P(0.05)) + ' / ' + f1(P(0.95))], ['10th / 90th pct', f1(P(0.1)) + ' / ' + f1(P(0.9))],
    ['25th / 75th pct', f1(P(0.25)) + ' / ' + f1(P(0.75))], ['P(10+ pts)', pct(p.pp * above(10))], ['P(20+ pts)', pct(p.pp * above(20))],
    ['volatility vs similar players', f2(p.v) + '×'], ['model-only projection', p.mm ? f1(p.mm[fmt]) : f1(p.m[fmt])]]
    .map(([a, b]) => `<tr><td>${a}</td><td class="n">${b}</td></tr>`).join('') + '</tbody></table></div></div>';
  // expert consensus
  const e = p.e && (p.e[FMT[fmt]] || p.e.all);
  h += '<div class="grid2"><div><div class="h2"><span>expert consensus rank (FantasyPros experts, via Boris Chen)</span></div>';
  h += e ? `<table><tbody><tr><td>position rank</td><td class="n">${p.p}${e[0]}</td></tr><tr><td>average rank</td><td class="n">${e[1]}</td></tr><tr><td>rank standard deviation</td><td class="n">${e[2]}</td></tr><tr><td>best / worst rank</td><td class="n">${e[3]} / ${e[4]}</td></tr><tr><td>tier</td><td class="n">${esc(e[5])}</td></tr></tbody></table>`
    : '<p class="mute">not ranked this week (outside the experts\' top group, injured, or on bye)</p>';
  h += '</div><div><div class="h2"><span>sportsbook player props (DraftKings lines)</span></div>';
  if (p.pr) { h += '<table><thead><tr><th>market</th><th class="n">line</th><th class="n">our mean</th></tr></thead><tbody>';
    for (const k in p.pr) h += `<tr><td>${PROP_NAMES[k] || k}</td><td class="n">${k === 'td_odds' ? (p.pr[k] > 0 ? '+' : '') + p.pr[k] : p.pr[k]}</td><td class="n">${STAT_KEY[k] != null && p.x ? f1(p.x[STAT_KEY[k]]) : ''}</td></tr>`;
    h += `</tbody></table><p class="mute">Lines are medians. Yards and receptions are blended into the projection at ${Math.round(FF.meta.prop_weight * 100)}% market weight.</p>`;
  } else h += '<p class="mute">no lines posted</p>';
  h += '</div></div>';
  if (p.x) { h += '<div class="h2"><span>expected stat line (model + market)</span></div><div class="tw"><table><tbody><tr>' + STAT_NAMES.map(n => `<th class="n">${n}</th>`).join('') + '</tr><tr>' + p.x.map(v => `<td class="n">${f1(v)}</td>`).join('') + '</tr></tbody></table></div>'; }
  if (p.l) { h += '<div class="h2"><span>recent games (' + FF.meta.season + ')</span></div><table><thead><tr><th>week</th><th>opp</th><th class="n">pts</th></tr></thead><tbody>';
    for (const [w, o, ppr, rec] of p.l.slice().reverse()) h += `<tr><td>${w}</td><td>${esc(o)}</td><td class="n">${f1(ppr - rec * [0, 0.5, 1][fmt])}</td></tr>`;
    h += '</tbody></table>'; }
  return h;
}

/* ---------- events / storage ---------- */
document.addEventListener('change', ev => {
  const t = ev.target;
  if (t.name === 'fmt') { fmt = +t.value; refresh(); return; }
  if (t.id === 'faIn' && view && view.type === 'swap') { const p = resolve(t.value, SLOTS[view.k]); if (p) { view.extra = p.i; view.pick = null; showView(view); } return; }
  if (t.dataset.fin) { const s = state[t.dataset.fin][+t.dataset.k]; if (s) { s.fin = t.value === '' ? null : +t.value; refresh(); } return; }
  if (t.list && t.dataset.side) {
    const side = t.dataset.side, k = +t.dataset.k, p = resolve(t.value, SLOTS[k]);
    if (p === null) state[side][k] = null; else if (p) state[side][k] = { id: p.i, fin: null };
    else { t.setCustomValidity('not eligible for this slot'); t.reportValidity(); return; }
    refresh();
  }
});
document.addEventListener('input', ev => {
  const t = ev.target; if (!(t.list && t.dataset.side)) return;
  const p = byLabel.get(t.value.trim().toLowerCase());
  if (p && slotOk(SLOTS[+t.dataset.k], p)) { state[t.dataset.side][+t.dataset.k] = { id: p.i, fin: null }; refresh(); }
});
document.addEventListener('click', ev => {
  const t = ev.target.closest('button,tr.pick'); if (!t) return;
  if (t.id === 'wp') { showView(view && view.type === 'analysis' ? null : { type: 'analysis' }); }
  else if (t.dataset.close != null) showView(null);
  else if (t.dataset.info) showView({ type: 'player', id: t.dataset.info });
  else if (t.dataset.swap) showView({ type: 'swap', k: +t.dataset.swap });
  else if (t.dataset.pick != null && view && view.type === 'swap') { view.pick = +t.dataset.pick; showView(view); }
  else if (t.dataset.apply != null && view && view.type === 'swap') { const c = view._res[+t.dataset.apply].c, a = lineupWith(view.k, c);
    if (c.j === 'fa') { const old = state.A[view.k]; const e = a.findIndex((s, j) => j >= NSTART && j < SLOTS.length - 1 && !s); if (e >= 0) a[e] = old; }
    state.A = a; view = null; refresh(); }
  else if (t.dataset.clear) { state[t.dataset.clear] = Array(SLOTS.length).fill(null); refresh(); }
});
function save() { try { localStorage.setItem('ffp', JSON.stringify({ fmt, A: state.A, B: state.B, wk: FF.meta.season + '-' + FF.meta.week })); } catch (e) { } }
function load() { try { const s = JSON.parse(localStorage.getItem('ffp') || 'null'); if (!s) return;
  fmt = [0, 1, 2].includes(s.fmt) ? s.fmt : 0; document.querySelector(`input[name=fmt][value="${fmt}"]`).checked = true;
  const fresh = s.wk === FF.meta.season + '-' + FF.meta.week;
  for (const k of ['A', 'B']) if (Array.isArray(s[k])) state[k] = s[k].map(x => x && byId.has(x.id) ? { id: x.id, fin: fresh && x.fin != null && Number.isFinite(+x.fin) ? +x.fin : null } : null).concat(Array(SLOTS.length).fill(null)).slice(0, SLOTS.length);
} catch (e) { } }

$('#wk').textContent = FF.meta.season + ' · week ' + FF.meta.week;
$('#built').textContent = 'data ' + FF.meta.built;
buildLists(); load(); refresh();
})();
