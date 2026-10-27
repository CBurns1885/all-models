"""picks_page.py — Generate a market-grouped, league-sub-grouped HTML picks dashboard."""

import json
from pathlib import Path
from datetime import datetime

FALLBACK_1X2_H = 0.9047619047619047  # 19/21 — home-prior fallback sentinel

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Match Picks — __TITLE_DATE__</title>
<style>
:root{--bg:#0F1620;--surface:#182030;--surface2:#1E2A3D;--border:#253347;--text:#E2EBF5;--muted:#7A8FA6;--accent:#F0A21A;--high:#34C06E;--mid:#F0A21A;--low:#7A8FA6;--warn:#E05A3A;--warn-bg:rgba(224,90,58,.12)}
@media(prefers-color-scheme:light){:root{--bg:#F2F5FA;--surface:#FFF;--surface2:#EFF3F9;--border:#D4DCE8;--text:#1A2535;--muted:#6B7A90;--accent:#C8850E;--high:#1E9A50;--mid:#C8850E;--low:#6B7A90;--warn:#C0401A;--warn-bg:rgba(192,64,26,.08)}}
:root[data-theme=dark]{--bg:#0F1620;--surface:#182030;--surface2:#1E2A3D;--border:#253347;--text:#E2EBF5;--muted:#7A8FA6;--accent:#F0A21A;--high:#34C06E;--mid:#F0A21A;--low:#7A8FA6;--warn:#E05A3A;--warn-bg:rgba(224,90,58,.12)}
:root[data-theme=light]{--bg:#F2F5FA;--surface:#FFF;--surface2:#EFF3F9;--border:#D4DCE8;--text:#1A2535;--muted:#6B7A90;--accent:#C8850E;--high:#1E9A50;--mid:#C8850E;--low:#6B7A90;--warn:#C0401A;--warn-bg:rgba(192,64,26,.08)}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,'Segoe UI',sans-serif;font-size:14px;line-height:1.5;min-height:100vh}
/* header */
.page-header{background:var(--surface);border-bottom:1px solid var(--border);padding:14px 22px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;gap:12px}
.header-left{display:flex;align-items:baseline;gap:12px}
.page-title{font-size:14px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.page-meta{font-size:12px;color:var(--muted)}
.theme-btn{background:var(--surface2);border:1px solid var(--border);color:var(--muted);border-radius:5px;padding:4px 10px;font-size:11px;cursor:pointer;font-family:inherit;flex-shrink:0}
.theme-btn:hover{color:var(--text);border-color:var(--accent)}
/* market nav */
.market-nav{background:var(--surface);border-bottom:1px solid var(--border);padding:0 22px;display:flex;overflow-x:auto;scrollbar-width:none}
.market-nav::-webkit-scrollbar{display:none}
.nav-btn{flex-shrink:0;background:none;border:none;border-bottom:2px solid transparent;color:var(--muted);font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;padding:9px 13px;cursor:pointer;font-family:inherit;white-space:nowrap;transition:color .12s,border-color .12s}
.nav-btn:hover{color:var(--text)}
.nav-btn.active{color:var(--accent);border-bottom-color:var(--accent)}
.nav-count{background:var(--surface2);color:var(--muted);font-size:10px;border-radius:10px;padding:1px 5px;margin-left:3px;font-weight:700}
/* legend */
.legend{display:flex;align-items:center;gap:18px;padding:10px 22px;border-bottom:1px solid var(--border);font-size:11px;color:var(--muted);background:var(--surface);flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:5px}
.ldot{width:7px;height:7px;border-radius:50%}
.fb-badge{font-size:10px;font-weight:700;color:var(--warn);background:var(--warn-bg);border-radius:3px;padding:1px 4px;margin-left:5px;letter-spacing:.04em;vertical-align:middle;display:inline-block}
/* layout */
.content{max-width:1120px;margin:0 auto;padding:22px}
.mkt-section{margin-bottom:32px;scroll-margin-top:86px}
.sec-hdr{display:flex;align-items:baseline;gap:10px;margin-bottom:10px}
.sec-title{font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
.sec-desc{font-size:12px;color:var(--muted)}
.sec-cnt{margin-left:auto;font-size:11px;color:var(--muted)}
/* league filter pills */
.lg-filter{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px}
.lg-pill{background:var(--surface2);border:1px solid var(--border);color:var(--muted);border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600;cursor:pointer;transition:all .12s;white-space:nowrap}
.lg-pill:hover{color:var(--text);border-color:var(--accent)}
.lg-pill.active{background:rgba(240,162,26,.15);color:var(--accent);border-color:var(--accent)}
.lg-pill-count{font-size:10px;opacity:.7;margin-left:2px}
/* league groups */
.lg-group{margin-bottom:14px}
.lg-group-hdr{display:flex;align-items:center;gap:8px;padding:5px 10px;background:var(--surface2);border:1px solid var(--border);border-radius:6px 6px 0 0;border-bottom:none}
.lg-group-name{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--text)}
.lg-group-cnt{font-size:10px;color:var(--muted);margin-left:auto}
/* table */
.tbl-wrap{background:var(--surface);border:1px solid var(--border);border-radius:0 0 6px 6px;overflow:hidden;overflow-x:auto}
.lg-group-hdr+.tbl-wrap{border-top:none}
.ptbl{width:100%;border-collapse:collapse}
.ptbl th{text-align:left;font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);padding:6px 10px;border-bottom:1px solid var(--border)}
.ptbl th.r{text-align:right}
.ptbl td{padding:7px 10px;border-bottom:1px solid var(--border);vertical-align:middle}
.ptbl tr:last-child td{border-bottom:none}
.ptbl tr:hover td{background:var(--surface2)}
.ptbl tr.fb td{background:var(--warn-bg)}
.ptbl tr.fb:hover td{background:var(--warn-bg);filter:brightness(1.08)}
.cell-dt{color:var(--muted);font-size:12px;white-space:nowrap}
.cell-m{font-weight:500}
.cell-m .aw{color:var(--muted)}
.pl{display:inline-block;padding:2px 7px;border-radius:4px;font-size:12px;font-weight:700}
.ph{background:rgba(52,192,110,.15);color:var(--high)}
.pd{background:rgba(240,162,26,.15);color:var(--mid)}
.pa{background:rgba(122,143,166,.15);color:var(--low)}
.py,.po{background:rgba(52,192,110,.15);color:var(--high)}
.pn,.pu{background:rgba(122,143,166,.15);color:var(--low)}
.conf-cell{text-align:right;white-space:nowrap}
.cw{display:inline-flex;align-items:center;gap:8px}
.cn{font-family:'Consolas','Monaco','Lucida Console',monospace;font-size:13px;font-variant-numeric:tabular-nums;min-width:42px;text-align:right}
.ch{color:var(--high)}.cm{color:var(--mid)}.cl{color:var(--low)}
.cbw{width:52px;height:4px;background:var(--surface2);border-radius:2px;overflow:hidden}
.cb{height:100%;border-radius:2px}
.cb.h{background:var(--high)}.cb.m{background:var(--mid)}.cb.l{background:var(--low)}
.empty{padding:20px;text-align:center;color:var(--muted);font-size:13px}
.hidden{display:none!important}
</style>
</head>
<body>
<div class="page-header">
  <div class="header-left">
    <span class="page-title">Match Picks</span>
    <span class="page-meta">__TITLE_DATE__ &middot; __N_FIXTURES__ fixtures</span>
  </div>
  <button class="theme-btn" onclick="var r=document.documentElement;r.setAttribute('data-theme',r.getAttribute('data-theme')==='light'?'dark':'light')">Toggle theme</button>
</div>
<div class="legend">
  <div class="legend-item"><div class="ldot" style="background:var(--high)"></div>&ge;85% confidence</div>
  <div class="legend-item"><div class="ldot" style="background:var(--mid)"></div>70&ndash;84%</div>
  <div class="legend-item"><div class="ldot" style="background:var(--low)"></div>&lt;70%</div>
  <div class="legend-item"><span class="fb-badge">!DATA</span> Home-prior fallback &mdash; no ML training data for this club</div>
</div>
<nav class="market-nav" id="mktNav"></nav>
<div class="content" id="mainContent"></div>
<script>
const RAW=__DATA_JSON__;
const MKTS=[
 {id:'result',label:'Match Result',desc:'Win/Draw/Win ≥70%, data-verified',filter:r=>r.c1>=70&&!r.fb,pick:r=>{const[p,c]=r.p1==='Home'?[r.ht+' Win','ph']:r.p1==='Away'?[r.at+' Win','pa']:['Draw','pd'];return{pred:p,cls:c,conf:r.c1}}},
 {id:'result-fb',label:'Result (!DATA)',desc:'Home-prior fallback — treat as unreliable',filter:r=>r.c1>=70&&r.fb,pick:r=>{const[p,c]=r.p1==='Home'?[r.ht+' Win','ph']:r.p1==='Away'?[r.at+' Win','pa']:['Draw','pd'];return{pred:p,cls:c,conf:r.c1,fb:1}}},
 {id:'btts',label:'BTTS',desc:'Both teams to score ≥70%',filter:r=>!r.fb&&r.cB>=70,pick:r=>({pred:'BTTS '+r.pB,cls:r.pB==='Yes'?'py':'pn',conf:r.cB})},
 {id:'ou05',label:'O/U 0.5',desc:'At least one goal ≥95%',filter:r=>!r.fb&&r.o05O>=95,pick:r=>({pred:'Over 0.5',cls:'po',conf:r.o05O})},
 {id:'ou15',label:'O/U 1.5',desc:'Over or Under 1.5 goals ≥85%',filter:r=>!r.fb&&Math.max(r.o15O,100-r.o15O)>=85,pick:r=>{const ov=r.o15O>=50;return{pred:ov?'Over 1.5':'Under 1.5',cls:ov?'po':'pu',conf:ov?r.o15O:100-r.o15O}}},
 {id:'ou25',label:'O/U 2.5',desc:'Over or Under 2.5 goals ≥70%',filter:r=>!r.fb&&Math.max(r.o25O,r.o25U)>=70,pick:r=>{const ov=r.o25O>=r.o25U;return{pred:ov?'Over 2.5':'Under 2.5',cls:ov?'po':'pu',conf:ov?r.o25O:r.o25U}}},
 {id:'ou35u',label:'O/U 3.5',desc:'Under 3.5 goals ≥88%',filter:r=>!r.fb&&r.o35U>=88,pick:r=>({pred:'Under 3.5',cls:'pu',conf:r.o35U})},
 {id:'ou45u',label:'O/U 4.5',desc:'Under 4.5 goals ≥92%',filter:r=>!r.fb&&r.o45U>=92,pick:r=>({pred:'Under 4.5',cls:'pu',conf:r.o45U})},
 {id:'hometg',label:'Home Goals',desc:'Home team to score ≥90%',filter:r=>!r.fb&&r.hg05>=90,pick:r=>({pred:'Home Scores',cls:'po',conf:r.hg05})},
 {id:'awaytg',label:'Away Goals',desc:'Away team to score ≥90%',filter:r=>!r.fb&&r.ag05>=90,pick:r=>({pred:'Away Scores',cls:'po',conf:r.ag05})},
 {id:'hcard',label:'Home Card',desc:'Home team to receive a card ≥88%',filter:r=>!r.fb&&r.hcard>=88,pick:r=>({pred:'Home Card',cls:'po',conf:r.hcard})},
 {id:'acard',label:'Away Card',desc:'Away team to receive a card ≥88%',filter:r=>!r.fb&&r.acard>=88,pick:r=>({pred:'Away Card',cls:'po',conf:r.acard})},
 {id:'yc15',label:'YC O1.5',desc:'Over 1.5 yellow cards ≥90%',filter:r=>!r.fb&&r.yc1>=90,pick:r=>({pred:'Over 1.5 YC',cls:'po',conf:r.yc1})},
 {id:'yc25',label:'YC O2.5',desc:'Over 2.5 yellow cards ≥88%',filter:r=>!r.fb&&r.yc2>=88,pick:r=>({pred:'Over 2.5 YC',cls:'po',conf:r.yc2})},
 {id:'bp20',label:'Booking Pts',desc:'Over 20.5 booking points ≥88%',filter:r=>!r.fb&&r.bp20>=88,pick:r=>({pred:'O20.5 BP',cls:'po',conf:r.bp20})},
 {id:'tc9',label:'Corners O9.5',desc:'Over 9.5 corners ≥80%',filter:r=>!r.fb&&r.tc9>=80,pick:r=>({pred:'Over 9.5 Corn',cls:'po',conf:r.tc9})},
 {id:'tc10',label:'Corners O10.5',desc:'Over 10.5 corners ≥78%',filter:r=>!r.fb&&r.tc10>=78,pick:r=>({pred:'Over 10.5 Corn',cls:'po',conf:r.tc10})},
];

function cc(c){return c>=85?'h':c>=70?'m':'l'}
function cn(c){return c>=85?'ch':c>=70?'cm':'cl'}
function fd(d){const[,m,dy]=d.split('-');return parseInt(dy)+' '+['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][parseInt(m)-1]}

function buildRow(r, p, isFB) {
  const tr = document.createElement('tr');
  if (p.fb || isFB) tr.classList.add('fb');
  const c = cc(p.conf);
  tr.innerHTML = '<td class="cell-dt">'+fd(r.dt)+'</td>'
    +'<td class="cell-m">'+r.ht+' <span class="aw">vs '+r.at+'</span>'+(p.fb||isFB?'<span class="fb-badge">!DATA</span>':'')+'</td>'
    +'<td><span class="pl '+p.cls+'">'+p.pred+'</span></td>'
    +'<td class="conf-cell"><div class="cw"><div class="cbw"><div class="cb '+c+'" style="width:'+Math.min(p.conf,100)+'%"></div></div>'
    +'<span class="cn '+cn(p.conf)+'">'+p.conf.toFixed(1)+'%</span></div></td>';
  return tr;
}

function buildLeagueGroup(lg, picks, isFB) {
  const div = document.createElement('div');
  div.className = 'lg-group';
  div.dataset.league = lg;

  const hdr = document.createElement('div');
  hdr.className = 'lg-group-hdr';
  hdr.innerHTML = '<span class="lg-group-name">'+lg+'</span>'
    +'<span class="lg-group-cnt">'+picks.length+' pick'+(picks.length!==1?'s':'')+'</span>';
  div.appendChild(hdr);

  const wrap = document.createElement('div');
  wrap.className = 'tbl-wrap';
  const t = document.createElement('table');
  t.className = 'ptbl';
  t.innerHTML = '<thead><tr><th>Date</th><th>Match</th><th>Pick</th><th class="r">Confidence</th></tr></thead>';
  const tb = document.createElement('tbody');
  picks.forEach(({r, p}) => tb.appendChild(buildRow(r, p, isFB)));
  t.appendChild(tb);
  wrap.appendChild(t);
  div.appendChild(wrap);
  return div;
}

function buildSec(mk) {
  const allPicks = RAW.filter(mk.filter).map(r => ({r, p: mk.pick(r)}));
  const isFB = mk.id === 'result-fb';

  // Group by league, sort each group by conf desc
  const byLeague = {};
  allPicks.forEach(item => {
    const lg = item.r.lg || '?';
    if (!byLeague[lg]) byLeague[lg] = [];
    byLeague[lg].push(item);
  });
  Object.values(byLeague).forEach(arr => arr.sort((a,b) => b.p.conf - a.p.conf));

  // Sort leagues: most picks first, then alpha
  const leagues = Object.keys(byLeague).sort((a,b) => byLeague[b].length - byLeague[a].length || a.localeCompare(b));

  const sec = document.createElement('div');
  sec.className = 'mkt-section';
  sec.id = mk.id;

  const hdr = document.createElement('div');
  hdr.className = 'sec-hdr';
  hdr.innerHTML = '<span class="sec-title">'+mk.label+'</span>'
    +'<span class="sec-desc">'+mk.desc+'</span>'
    +'<span class="sec-cnt">'+allPicks.length+' picks across '+leagues.length+' league'+(leagues.length!==1?'s':'')+'</span>';
  sec.appendChild(hdr);

  if (!allPicks.length) {
    const e = document.createElement('div');
    e.className = 'empty';
    e.textContent = 'No picks above threshold';
    sec.appendChild(e);
    return sec;
  }

  // League filter pills (only if >1 league)
  if (leagues.length > 1) {
    const filterDiv = document.createElement('div');
    filterDiv.className = 'lg-filter';

    const allPill = document.createElement('button');
    allPill.className = 'lg-pill active';
    allPill.textContent = 'All';
    allPill.dataset.lg = '';
    filterDiv.appendChild(allPill);

    leagues.forEach(lg => {
      const pill = document.createElement('button');
      pill.className = 'lg-pill';
      pill.innerHTML = lg + '<span class="lg-pill-count">'+byLeague[lg].length+'</span>';
      pill.dataset.lg = lg;
      filterDiv.appendChild(pill);
    });

    filterDiv.addEventListener('click', e => {
      const pill = e.target.closest('.lg-pill');
      if (!pill) return;
      filterDiv.querySelectorAll('.lg-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      const target = pill.dataset.lg;
      sec.querySelectorAll('.lg-group').forEach(g => {
        g.classList.toggle('hidden', target !== '' && g.dataset.league !== target);
      });
    });

    sec.appendChild(filterDiv);
  }

  // League groups
  leagues.forEach(lg => {
    sec.appendChild(buildLeagueGroup(lg, byLeague[lg], isFB));
  });

  return sec;
}

function buildNav() {
  const nav = document.getElementById('mktNav');
  MKTS.forEach(mk => {
    const cnt = RAW.filter(mk.filter).length;
    const btn = document.createElement('button');
    btn.className = 'nav-btn';
    btn.innerHTML = mk.label + ' <span class="nav-count">'+cnt+'</span>';
    btn.onclick = () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(mk.id).scrollIntoView({behavior:'smooth'});
    };
    nav.appendChild(btn);
  });
}

const obs = new IntersectionObserver(es => es.forEach(e => {
  if (e.isIntersecting) {
    const id = e.target.id;
    document.querySelectorAll('.nav-btn').forEach((b,i) => b.classList.toggle('active', MKTS[i].id===id));
  }
}), {rootMargin:'-25% 0px -65% 0px'});

buildNav();
const mc = document.getElementById('mainContent');
MKTS.forEach(mk => { const s = buildSec(mk); mc.appendChild(s); obs.observe(s); });
</script>
</body>
</html>
"""


def _pct(row, col):
    try:
        return round(float(row.get(col) or 0) * 100, 1)
    except (TypeError, ValueError):
        return 0.0


def _build_record(row):
    h = _pct(row, 'P_1X2_H')
    d = _pct(row, 'P_1X2_D')
    a = _pct(row, 'P_1X2_A')
    pred_1x2 = 'Home' if h >= d and h >= a else 'Draw' if d >= a else 'Away'
    conf_1x2 = max(h, d, a)

    by = _pct(row, 'P_BTTS_Y')
    bn = _pct(row, 'P_BTTS_N')
    pred_btts = 'Yes' if by >= bn else 'No'
    conf_btts = max(by, bn)

    raw_h = float(row.get('P_1X2_H') or 0)
    is_fb = abs(raw_h - FALLBACK_1X2_H) < 0.0001

    raw_date = str(row.get('Date', '')).strip()
    try:
        import pandas as pd
        dt = pd.to_datetime(raw_date).strftime('%Y-%m-%d')
    except Exception:
        dt = raw_date

    o25O = _pct(row, 'P_OU_2_5_O')
    o25U = _pct(row, 'P_OU_2_5_U')
    o35O = _pct(row, 'P_OU_3_5_O')
    o35U = _pct(row, 'P_OU_3_5_U')
    o45O = _pct(row, 'P_OU_4_5_O')
    o45U = _pct(row, 'P_OU_4_5_U')

    return {
        'ht':   str(row.get('HomeTeam', '') or '').strip(),
        'at':   str(row.get('AwayTeam', '') or '').strip(),
        'lg':   str(row.get('League', '') or '').strip(),
        'dt':   dt,
        'fb':   is_fb,
        'p1':   pred_1x2,
        'c1':   conf_1x2,
        'pB':   pred_btts,
        'cB':   conf_btts,
        'o05O': _pct(row, 'P_OU_0_5_O'),
        'o15O': _pct(row, 'P_OU_1_5_O'),
        'o25O': o25O,
        'o25U': o25U,
        'o35U': o35U,
        'o45U': o45U,
        'hg05': _pct(row, 'P_HomeTG_0_5_O'),
        'hg15': _pct(row, 'P_HomeTG_1_5_O'),
        'ag05': _pct(row, 'P_AwayTG_0_5_O'),
        'hcard': _pct(row, 'P_HomeTeam_Card_Y'),
        'acard': _pct(row, 'P_AwayTeam_Card_Y'),
        'yc1':  _pct(row, 'P_TotalYC_O1_5_Y'),
        'yc2':  _pct(row, 'P_TotalYC_O2_5_Y'),
        'bp20': _pct(row, 'P_BookingPts_O20_5_Y'),
        'tc9':  _pct(row, 'P_TotalCorners_O9_5_Y'),
        'tc10': _pct(row, 'P_TotalCorners_O10_5_Y'),
    }


def generate_picks_pdf(records, output_dir, title_date):
    """Write picks_page.pdf to output_dir from the same records list used by the HTML page."""
    try:
        from fpdf import FPDF
    except ImportError:
        print("  [WARN] fpdf2 not installed — skipping PDF (run: py -m pip install fpdf2)")
        return None

    # Market definitions mirroring MKTS in the HTML
    _MKTS = [
        ('Match Result',  lambda r: r['c1'] >= 70 and not r['fb'],
         lambda r: (f"{r['ht']} Win" if r['p1'] == 'Home' else f"{r['at']} Win" if r['p1'] == 'Away' else 'Draw', r['c1'])),
        ('BTTS',          lambda r: not r['fb'] and r['cB'] >= 70,
         lambda r: (f"BTTS {r['pB']}", r['cB'])),
        ('O/U 0.5',       lambda r: not r['fb'] and r['o05O'] >= 95,
         lambda r: ('Over 0.5', r['o05O'])),
        ('O/U 1.5',       lambda r: not r['fb'] and max(r['o15O'], 100 - r['o15O']) >= 85,
         lambda r: ('Over 1.5' if r['o15O'] >= 50 else 'Under 1.5', max(r['o15O'], 100 - r['o15O']))),
        ('O/U 2.5',       lambda r: not r['fb'] and max(r['o25O'], r['o25U']) >= 70,
         lambda r: ('Over 2.5' if r['o25O'] >= r['o25U'] else 'Under 2.5', max(r['o25O'], r['o25U']))),
        ('O/U 3.5 Under', lambda r: not r['fb'] and r['o35U'] >= 88,
         lambda r: ('Under 3.5', r['o35U'])),
        ('O/U 4.5 Under', lambda r: not r['fb'] and r['o45U'] >= 92,
         lambda r: ('Under 4.5', r['o45U'])),
        ('Home Goals',    lambda r: not r['fb'] and r['hg05'] >= 90,
         lambda r: ('Home Scores', r['hg05'])),
        ('Away Goals',    lambda r: not r['fb'] and r['ag05'] >= 90,
         lambda r: ('Away Scores', r['ag05'])),
        ('Home Card',     lambda r: not r['fb'] and r['hcard'] >= 88,
         lambda r: ('Home Card', r['hcard'])),
        ('Away Card',     lambda r: not r['fb'] and r['acard'] >= 88,
         lambda r: ('Away Card', r['acard'])),
        ('YC O1.5',       lambda r: not r['fb'] and r['yc1'] >= 90,
         lambda r: ('Over 1.5 YC', r['yc1'])),
        ('YC O2.5',       lambda r: not r['fb'] and r['yc2'] >= 88,
         lambda r: ('Over 2.5 YC', r['yc2'])),
        ('Booking Pts',   lambda r: not r['fb'] and r['bp20'] >= 88,
         lambda r: ('O20.5 BP', r['bp20'])),
        ('Corners O9.5',  lambda r: not r['fb'] and r['tc9'] >= 80,
         lambda r: ('Over 9.5 Corn', r['tc9'])),
        ('Corners O10.5', lambda r: not r['fb'] and r['tc10'] >= 78,
         lambda r: ('Over 10.5 Corn', r['tc10'])),
    ]

    ACCENT  = (240, 162, 26)   # amber
    HIGH    = (52,  192, 110)  # green
    MID_C   = (240, 162, 26)   # amber
    LOW_C   = (122, 143, 166)  # grey
    BG_ROW  = (242, 245, 250)
    BG_HDR  = (24,  32,  48)
    TEXT_HDR= (226, 235, 245)
    TEXT    = (26,  37,  53)
    MUTED   = (107, 122, 144)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_margins(12, 12, 12)
    pdf.set_font('Helvetica', '', 10)

    def _ascii(text):
        import unicodedata
        return unicodedata.normalize('NFKD', str(text)).encode('latin-1', 'ignore').decode('latin-1')

    def _conf_color(c):
        return HIGH if c >= 85 else MID_C if c >= 70 else LOW_C

    def _short_date(d):
        try:
            from datetime import datetime as _dt
            return _dt.strptime(d, '%Y-%m-%d').strftime('%d %b')
        except Exception:
            return d

    for mkt_label, mkt_filter, mkt_pick in _MKTS:
        picks = [(r, mkt_pick(r)) for r in records if mkt_filter(r)]
        if not picks:
            continue

        pdf.add_page()

        # Market header bar
        pdf.set_fill_color(*BG_HDR)
        pdf.set_text_color(*TEXT_HDR)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 9, _ascii(mkt_label.upper()) + f'  ({len(picks)} picks)', fill=True, new_x='LMARGIN', new_y='NEXT')
        pdf.ln(1)

        # Table header
        pdf.set_fill_color(*ACCENT)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 8)
        col_w = [20, 62, 58, 26, 22]  # Date, Match, Pick, Conf%, bar
        hdrs  = ['Date', 'Match', 'Pick', 'Conf', '']
        for w, h in zip(col_w, hdrs):
            pdf.cell(w, 7, h, fill=True)
        pdf.ln()

        # Rows
        pdf.set_font('Helvetica', '', 8)
        for i, (r, (pred, conf)) in enumerate(sorted(picks, key=lambda x: (-x[1][1], x[0]['dt']))):
            bg = BG_ROW if i % 2 == 0 else (255, 255, 255)
            pdf.set_fill_color(*bg)
            pdf.set_text_color(*TEXT)

            match_str = _ascii(f"{r['ht']} vs {r['at']}")
            if len(match_str) > 36:
                match_str = match_str[:35] + '.'

            cc = _conf_color(conf)

            pdf.cell(col_w[0], 6, _short_date(r['dt']), fill=True)
            pdf.cell(col_w[1], 6, match_str, fill=True)
            pdf.set_text_color(*cc)
            pdf.set_font('Helvetica', 'B', 8)
            pdf.cell(col_w[2], 6, _ascii(pred), fill=True)
            pdf.set_text_color(*TEXT)
            pdf.set_font('Helvetica', '', 8)
            pdf.set_text_color(*cc)
            pdf.cell(col_w[3], 6, f'{conf:.1f}%', fill=True, align='R')
            pdf.set_text_color(*TEXT)

            # Mini confidence bar
            bar_x = pdf.get_x()
            bar_y = pdf.get_y()
            pdf.set_fill_color(*bg)
            pdf.cell(col_w[4], 6, '', fill=True)
            bar_w = col_w[4] - 2
            bar_h = 2.5
            pdf.set_fill_color(220, 225, 232)
            pdf.rect(bar_x + 1, bar_y + 1.8, bar_w, bar_h, 'F')
            pdf.set_fill_color(*cc)
            pdf.rect(bar_x + 1, bar_y + 1.8, bar_w * min(conf, 100) / 100, bar_h, 'F')

            pdf.ln()
        pdf.ln(2)

    out_path = Path(output_dir) / 'picks_page.pdf'
    pdf.output(str(out_path))
    total_picks = sum(1 for _, mkt_filter, _ in _MKTS for r in records if mkt_filter(r))
    active_mkts = sum(1 for _, mkt_filter, _ in _MKTS if any(mkt_filter(r) for r in records))
    print(f"  [OK] picks_page.pdf written ({total_picks} picks across {active_mkts} markets)")
    return out_path


def generate_picks_page(csv_path, output_dir):
    """Read weekly_bets_full.csv and write picks_page.html to output_dir."""
    import pandas as pd

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    df = pd.read_csv(csv_path)
    print(f"  Building picks page from {len(df)} rows ({csv_path.name})...")

    records = []
    for _, row in df.iterrows():
        try:
            records.append(_build_record(row))
        except Exception as e:
            ht = str(row.get('HomeTeam', '?'))
            at = str(row.get('AwayTeam', '?'))
            print(f"  [WARN] Skipping {ht} vs {at}: {e}")

    if not records:
        print("  [WARN] No records built — skipping picks page")
        return None

    dates = sorted(r['dt'] for r in records if r.get('dt'))
    first_date = dates[0] if dates else datetime.now().strftime('%Y-%m-%d')
    try:
        title_date = datetime.strptime(first_date, '%Y-%m-%d').strftime('%d %b %Y').lstrip('0')
    except Exception:
        title_date = first_date

    n_fixtures = len(records)
    data_json = json.dumps(records, separators=(',', ':'))

    html = (
        _HTML_TEMPLATE
        .replace('__TITLE_DATE__', title_date)
        .replace('__N_FIXTURES__', str(n_fixtures))
        .replace('__DATA_JSON__', data_json)
    )

    out_path = output_dir / 'picks_page.html'
    out_path.write_text(html, encoding='utf-8')
    print(f"  [OK] picks_page.html written ({n_fixtures} fixtures, {len(html)//1024}KB)")

    generate_picks_pdf(records, output_dir, title_date)

    return out_path
