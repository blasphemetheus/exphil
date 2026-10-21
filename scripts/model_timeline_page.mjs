// Generates the scrollable model timeline page (artifact https://claude.ai/artifact/MaWQVVRLzCz8p1fvcB6jZY) from docs/planning/model_lineage_2026-10-21.json. Run: node scripts/model_timeline_page.mjs
import fs from "fs";
const j = JSON.parse(fs.readFileSync("docs/planning/model_lineage_2026-10-21.json","utf8"));
const esc = s => String(s??"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
// line bucket from the family text
function line(f){ f=f.toLowerCase();
  if (/benchmark|early imitation|candidate sweep|clean_200/.test(f)) return "early";
  if (/mewtwo/.test(f)) return "mewtwo";
  if (/multishine|drill/.test(f)) return "multishine";
  if (/generalist|full-game il/.test(f) && !/sim|search/.test(f)) return "generalist";
  if (/rl on|search-as-teacher|sim/.test(f)) return "sim";
  if (/mamba|coherence/.test(f)) return "mamba";
  return "other"; }
const LINES = {early:"Early + benchmarks", mewtwo:"Mewtwo", multishine:"Fox multishine (drills)", generalist:"Generalist Fox (replay BC)", sim:"Sim / RL / DAgger", mamba:"Fox Mamba + coherence", other:"Other"};
function outcome(v){ v=v.toLowerCase();
  if (/crown/.test(v) && !/revert/.test(v)) return "crowned";
  if (/revert|fail|invalid|not promoted|none promoted/.test(v)) return "failed";
  if (/candidate|met|passed|partial|adopted|kept|= standard|became/.test(v)) return "candidate";
  if (/park/.test(v)) return "parked";
  if (/diagnostic|testbed|artifact/.test(v)) return "diagnostic";
  return "superseded"; }
const OUT = {crowned:"crowned", candidate:"candidate / kept", failed:"failed / reverted", superseded:"superseded", parked:"parked", diagnostic:"diagnostic"};
const eras = j.eras.map(e=>{ const [a,b]=e.range.split(" .. "); return {...e,a,b}; });
const models = j.models.map(m=>({...m, line: line(m.family), out: outcome(m.verdict), d0: m.date.split(" ..")[0]}));
for (const m of models){ m.era = eras.find(e=> m.d0>=e.a && m.d0<=e.b) || eras.reduce((p,e)=> m.d0>=e.a?e:p, eras[0]); }
const byEra = eras.map(e=>({e, ms: models.filter(m=>m.era===e)}));
const counts = {}; for (const m of models) counts[m.out]=(counts[m.out]||0)+1;

const cards = byEra.map(({e,ms})=>`
<section class="era" id="${e.id}">
  <header class="erahead">
    <div class="eratag">${esc(e.id)} · ${esc(e.range)}</div>
    <h2>${esc(e.name)}</h2>
    <p class="focus">${esc(e.focus)}</p>
    <p class="src">${esc(e.sources)}</p>
  </header>
  <div class="cards">${ms.map(m=>`
  <article class="card" data-line="${m.line}" data-out="${m.out}">
    <div class="cardtop"><span class="date">${esc(m.date)}</span><span class="pill pill-${m.out}">${OUT[m.out]}</span><span class="pill line">${LINES[m.line]}</span></div>
    <h3>${esc(m.name)}</h3>
    <div class="meta"><span>${esc(m.family)}</span><span>${esc(m.backbone)}</span></div>
    <dl>
      <dt>Trying to</dt><dd>${esc(m.intent)}</dd>
      <dt>Changed</dt><dd>${esc(m.distinguishes)}</dd>
      <dt>Data</dt><dd>${esc(m.data)}</dd>
      <dt>Result</dt><dd>${esc(m.result)}</dd>
      <dt>Verdict</dt><dd class="verdict">${esc(m.verdict)}</dd>
    </dl>
    <p class="src">${esc(m.sources)}</p>
  </article>`).join("")}${ms.length?"":'<p class="muted">no checkpoints in this era</p>'}</div>
</section>`).join("");

const html = `<title>ExPhil Model Timeline</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: left rail of eras (desktop), one scrolling column of era bands with model cards */
:root{--bg:#f7f6f2;--fg:#1d1c1a;--muted:#6b675e;--line:#d9d5cb;--card:#ffffff;--accent:#8a4b1f;--rail:#efece4;
 --ok:#1f6f43;--okbg:#dff2e6;--bad:#a52c22;--badbg:#f9e0dc;--cand:#1d5c8a;--candbg:#dcebf7;--sup:#6b675e;--supbg:#ebe8e1;--park:#7a5c0f;--parkbg:#f6ecc9;--diag:#5b3f8a;--diagbg:#eae2f6;
 --display:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17161a;--fg:#ebe8e2;--muted:#a39e93;--line:#3a383f;--card:#211f25;--accent:#e0955b;--rail:#1d1c21;
 --ok:#7fd3a0;--okbg:#1b3a2a;--bad:#f0938a;--badbg:#44211d;--cand:#8fc3ec;--candbg:#1c3449;--sup:#a39e93;--supbg:#2b2930;--park:#e5c25a;--parkbg:#3d321a;--diag:#c4aaf0;--diagbg:#302544;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#17161a;--fg:#ebe8e2;--muted:#a39e93;--line:#3a383f;--card:#211f25;--accent:#e0955b;--rail:#1d1c21;
 --ok:#7fd3a0;--okbg:#1b3a2a;--bad:#f0938a;--badbg:#44211d;--cand:#8fc3ec;--candbg:#1c3449;--sup:#a39e93;--supbg:#2b2930;--park:#e5c25a;--parkbg:#3d321a;--diag:#c4aaf0;--diagbg:#302544;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--display);font-size:14px;line-height:1.45;margin:0}
.wrap{display:grid;grid-template-columns:220px minmax(0,1fr);gap:0;min-height:100%}
nav{background:var(--rail);border-right:1px solid var(--line);padding-block:18px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:12px;position:sticky;top:0;height:100vh;overflow:auto;box-sizing:border-box}
nav h1{font-size:15px;margin:0 0 4px;letter-spacing:.01em}
nav .sub{font-size:11px;color:var(--muted);margin:0 0 14px}
nav a{display:block;color:var(--fg);text-decoration:none;font-size:12px;padding:5px 6px;border-radius:4px;border-left:3px solid transparent}
nav a:hover,nav a:focus-visible{background:var(--card);outline:none}
nav a b{display:block;font-family:var(--mono);font-size:10px;color:var(--accent);font-weight:500}
main{padding-block:18px 60px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));min-width:0}
.intro{max-width:72ch}
.intro h1{font-size:24px;margin:0 0 6px;text-wrap:balance}
.intro p{margin:0 0 8px;color:var(--muted)}
.filters{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0 10px;align-items:center}
.filters .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-right:4px}
.chip{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:999px;padding:3px 10px;font-size:12px;cursor:pointer;font-family:inherit}
.chip[aria-pressed="true"]{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}
.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.count{font-family:var(--mono);font-size:11px;color:var(--muted);margin-left:auto}
.era{margin-top:26px;border-top:2px solid var(--accent);padding-top:10px}
.eratag{font-family:var(--mono);font-size:11px;color:var(--accent)}
.era h2{font-size:18px;margin:2px 0 6px;text-wrap:balance}
.focus{max-width:80ch;margin:0 0 4px}
.src{font-family:var(--mono);font-size:10.5px;color:var(--muted);margin:4px 0 0;word-break:break-word}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px;margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:12px 14px;min-width:0}
.card[hidden]{display:none}
.cardtop{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px}
.date{font-family:var(--mono);font-size:11px;color:var(--muted)}
.pill{font-size:10.5px;padding:1px 8px;border-radius:999px;letter-spacing:.02em;font-weight:500}
.pill.line{background:var(--supbg);color:var(--fg)}
.pill-crowned{background:var(--okbg);color:var(--ok)}.pill-failed{background:var(--badbg);color:var(--bad)}.pill-candidate{background:var(--candbg);color:var(--cand)}
.pill-superseded{background:var(--supbg);color:var(--sup)}.pill-parked{background:var(--parkbg);color:var(--park)}.pill-diagnostic{background:var(--diagbg);color:var(--diag)}
.card h3{font-family:var(--mono);font-weight:500;font-size:13px;margin:0 0 4px;word-break:break-word}
.meta{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:11.5px;color:var(--muted);margin-bottom:8px}
dl{display:grid;grid-template-columns:70px minmax(0,1fr);gap:4px 10px;margin:0;font-size:12.5px}
dt{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em;padding-top:2px}
dd{margin:0;min-width:0}
dd.verdict{font-weight:500}
.card[data-out="crowned"]{border-color:var(--ok)}.card[data-out="failed"]{border-left:4px solid var(--bad)}
.tail{margin-top:34px;border-top:2px solid var(--line);padding-top:10px}
.tail h2{font-size:18px;margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;vertical-align:top;padding:6px 8px;border-bottom:1px solid var(--line)}
th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.tablewrap{overflow-x:auto}
ol.lessons{padding-left:22px;max-width:80ch}ol.lessons li{margin-bottom:6px}
.muted{color:var(--muted)}
@media (max-width:860px){.wrap{grid-template-columns:1fr}nav{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line);padding-right:max(16px,env(safe-area-inset-right,0px))}nav a{display:inline-block;margin:2px 2px 2px 0}nav a b{display:inline;margin-right:4px}.cards{grid-template-columns:1fr}dl{grid-template-columns:1fr}}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
</style>
<div class="wrap">
<nav>
  <h1>ExPhil models</h1>
  <p class="sub">${models.length} checkpoint entries · ${eras.length} eras · Mar–Oct 2026</p>
  ${eras.map(e=>`<a href="#${e.id}"><b>${esc(e.range)}</b>${esc(e.name)}</a>`).join("")}
  <a href="#instruments"><b>evals</b>How we measured</a>
  <a href="#lessons"><b>lessons</b>What we learned</a>
</nav>
<main>
  <div class="intro">
    <h1>Every model we trained, and what it was for</h1>
    <p>One card per checkpoint family, in date order, grouped by what the program was focused on at the time. Each card says what we were trying to do, what changed from the one before, what the numbers said, and what happened to it. Numbers are copied from the handoff docs and the registry; the source line on each card says where.</p>
    <p>Only two generalist models were ever crowned for play, and one multishine model is still production. Most entries are superseded by the next step of their own line, which is the normal shape of this work.</p>
  </div>
  <div class="filters" role="group" aria-label="filter by line">
    <span class="lbl">Line</span>
    ${Object.entries(LINES).filter(([k])=>models.some(m=>m.line===k)).map(([k,v])=>`<button class="chip" data-f="line" data-v="${k}" aria-pressed="true" id="f-line-${k}">${v}</button>`).join("")}
  </div>
  <div class="filters" role="group" aria-label="filter by outcome">
    <span class="lbl">Outcome</span>
    ${Object.entries(OUT).map(([k,v])=>`<button class="chip" data-f="out" data-v="${k}" aria-pressed="true" id="f-out-${k}">${v} <span class="muted">${counts[k]||0}</span></button>`).join("")}
    <span class="count" id="shown"></span>
  </div>
  ${cards}
  <section class="tail" id="instruments">
    <h2>How we measured (the eval instruments)</h2>
    <div class="tablewrap"><table><tr><th>Instrument</th><th>What it measures</th><th>Era</th></tr>
    ${j.instruments.map(i=>`<tr><td>${esc(i.name)}</td><td>${esc(i.measures)}</td><td class="muted">${esc(i.era)}</td></tr>`).join("")}</table></div>
  </section>
  <section class="tail" id="lessons">
    <h2>Lessons recorded along the way</h2>
    <ol class="lessons">${j.lessons.map(l=>`<li>${esc(l)}</li>`).join("")}</ol>
  </section>
</main>
</div>
<script>
(function(){
  const on={line:new Set(),out:new Set()};
  document.querySelectorAll(".chip").forEach(c=>{on[c.dataset.f].add(c.dataset.v); c.addEventListener("click",()=>{const p=c.getAttribute("aria-pressed")!=="true"; c.setAttribute("aria-pressed",String(p)); p?on[c.dataset.f].add(c.dataset.v):on[c.dataset.f].delete(c.dataset.v); apply();});});
  function apply(){let n=0; document.querySelectorAll(".card").forEach(k=>{const show=on.line.has(k.dataset.line)&&on.out.has(k.dataset.out); k.hidden=!show; if(show)n++;});
    document.getElementById("shown").textContent=n+" of ${models.length} shown";}
  apply();
})();
</script>`;
fs.writeFileSync("priv/viewer/model_timeline.html", html);
console.log("bytes", html.length, "outcomes", JSON.stringify(counts), "lines", JSON.stringify(models.reduce((a,m)=>(a[m.line]=(a[m.line]||0)+1,a),{})));
