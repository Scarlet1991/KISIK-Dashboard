# -*- coding: utf-8 -*-
"""Baut aus dashboard_station_data.json ein interaktives Stations-Dashboard:
   - dashboard_station.html  (eigenständig, im Browser öffnen)
   - _dashboard_fragment.html (Fragment für Inline-Vorschau)"""
import json
from pathlib import Path
AN = Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung")
data = json.loads((AN/"dashboard_station_data.json").read_text(encoding="utf-8"))
DATA = json.dumps(data, ensure_ascii=False)

STYLE = """
<style>
h2.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}
.dash{font-family:var(--font-sans);color:var(--color-text-primary);padding:1rem 0}
.hd{display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:8px}
.hd h3{font-size:18px;font-weight:500;margin:0}
.hd .sub{font-size:12px;color:var(--color-text-secondary)}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:14px 0}
.mc{background:var(--color-background-secondary);border-radius:8px;padding:.7rem .8rem}
.mc .t{font-size:11px;color:var(--color-text-tertiary);margin:0 0 4px}
.mc .v{font-size:20px;font-weight:500;margin:0}
.lbl{font-size:11px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;color:var(--color-text-tertiary);margin:1.1rem 0 .5rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(118px,1fr));gap:8px}
.bed{background:var(--color-background-primary);border:.5px solid var(--color-border-tertiary);border-radius:8px;padding:.55rem .6rem;cursor:pointer}
.bed:hover{border-color:var(--color-border-primary)}
.bed.sel{outline:2px solid var(--color-border-info);outline-offset:1px}
.bed .bn{font-size:10.5px;color:var(--color-text-tertiary);display:flex;justify-content:space-between;align-items:center;gap:4px}
.bed .los{font-size:21px;font-weight:500;margin:2px 0 0;line-height:1}
.bed .u{font-size:10px;color:var(--color-text-tertiary);font-weight:400}
.bed .meta{font-size:10.5px;color:var(--color-text-secondary);margin-top:3px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex-shrink:0}
.detail{margin-top:6px;background:var(--color-background-secondary);border-radius:12px;padding:1rem 1.1rem}
.detail h4{margin:0 0 2px;font-size:15px;font-weight:500}
.detail .dsub{font-size:12px;color:var(--color-text-secondary);margin:0 0 10px}
.leg{display:flex;gap:14px;font-size:11px;color:var(--color-text-secondary);margin-bottom:8px}
.leg span{display:flex;align-items:center;gap:5px}
.sq{width:11px;height:11px;border-radius:2px}
.frow{display:grid;grid-template-columns:43% 57%;align-items:center;gap:8px;margin:4px 0;font-size:12px}
.fname{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fname .val{color:var(--color-text-tertiary);font-size:10.5px}
.bara{position:relative;height:17px;background:var(--color-background-primary);border-radius:3px}
.bar{position:absolute;top:0;height:17px;border-radius:3px}
.axis{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--color-border-secondary)}
.cval{position:absolute;top:1px;font-size:9.5px;color:var(--color-text-secondary)}
.gi{display:flex;flex-direction:column;gap:4px}
.girow{display:grid;grid-template-columns:55% 45%;align-items:center;gap:8px;font-size:11.5px}
.gibar{height:13px;background:#2166ac;border-radius:3px}
.hint{font-size:11px;color:var(--color-text-tertiary);font-style:italic;margin-top:8px}
</style>
"""

BODY = """
<div class="dash">
<h2 class="sr-only">Interaktives Stations-Dashboard: ICU-Belegung eines Tages mit vorhergesagter Verweildauer und per-Patient-Feature-Importance.</h2>
<div class="hd">
  <div><h3 id="dTitle"></h3><div class="sub" id="dSub"></div></div>
  <div class="sub" id="dModel"></div>
</div>
<div class="cards" id="cards"></div>
<div class="lbl">Stationsbelegung — Klick auf ein Bett zeigt die Begründung der Vorhersage</div>
<div class="grid" id="grid"></div>
<div class="lbl">Vorhersage-Begründung (Patient)</div>
<div class="detail" id="detail"></div>
<div class="lbl">Globale Feature-Wichtigkeit (Modell, Gain)</div>
<div class="gi" id="global"></div>
<div class="hint">Demo-Daten: anonymisierte Betten, Modell XGBoost auf 24h-Features. Beiträge sind SHAP-Werte auf der log-Skala des Modells (rot = erhöht die erwartete Liegedauer, blau = senkt sie). Wert 0 = im 24h-Fenster nicht erfasst.</div>
</div>
"""

SCRIPT = """
<script>
const D = __DATA__;
const RED="#d6604d", BLUE="#2166ac", AMBER="#ef9f27", TEAL="#1d9e75";
function cat(los){ if(los>7) return {c:RED,t:"Langlieger"}; if(los>=3) return {c:AMBER,t:"mittel"}; return {c:TEAL,t:"kurz"}; }
document.getElementById('dTitle').textContent = "Station "+D.ward+" — "+D.date;
document.getElementById('dSub').textContent = D.n_patients+" Patienten belegt";
document.getElementById('dModel').textContent = "Modell: "+D.model;

const nLong = D.patients.filter(p=>p.pred_los>7).length;
const meanP = (D.patients.reduce((s,p)=>s+p.pred_los,0)/D.patients.length).toFixed(1);
const cards=[["Belegung",D.n_patients],["Vorhergesagte Langlieger (>7 d)",nLong],
             ["Ø vorhergesagte LoS",meanP+" d"],["Datenfenster","erste 24 h"]];
document.getElementById('cards').innerHTML = cards.map(c=>
  `<div class="mc"><p class="t">${c[0]}</p><p class="v">${c[1]}</p></div>`).join("");

function bedCard(p,i){
  const k=cat(p.pred_los);
  return `<div class="bed" data-i="${i}" onclick="sel(${i})">
    <div class="bn"><span>${p.bed}</span><span class="dot" style="background:${k.c}"></span></div>
    <div class="los" style="color:${k.c}">${p.pred_los.toFixed(1)}<span class="u"> d</span></div>
    <div class="meta">${p.age?p.age+" J · ":""}Tag ${p.day_of_stay}</div>
  </div>`;
}
document.getElementById('grid').innerHTML = D.patients.map(bedCard).join("");

function sel(i){
  document.querySelectorAll('.bed').forEach(b=>b.classList.remove('sel'));
  document.querySelector(`.bed[data-i="${i}"]`).classList.add('sel');
  const p=D.patients[i], k=cat(p.pred_los);
  const maxabs=Math.max(...p.top_features.map(f=>Math.abs(f.contrib)))||1;
  const rows=p.top_features.map(f=>{
    const w=Math.abs(f.contrib)/maxabs*50;
    const pos=f.contrib>=0;
    const bar=`<div class="bara"><div class="axis"></div>
      <div class="bar" style="${pos?`left:50%`:`left:${50-w}%`};width:${w}%;background:${pos?RED:BLUE}"></div>
      <span class="cval" style="${pos?`left:${50+w}%;margin-left:4px`:`right:${50+w}%;margin-right:4px`}">${f.contrib>0?'+':''}${f.contrib.toFixed(2)}</span></div>`;
    const valtxt = (f.value===null)?"":` <span class="val">(${f.value})</span>`;
    return `<div class="frow"><div class="fname" title="${f.feature}">${f.feature}${valtxt}</div>${bar}</div>`;
  }).join("");
  document.getElementById('detail').innerHTML =
    `<h4>${p.bed} <span style="color:${k.c}">· ${p.pred_los.toFixed(1)} d vorhergesagt</span></h4>
     <div class="dsub">${p.age?p.age+" Jahre · ":""}Aufenthaltstag ${p.day_of_stay} · tatsächliche LoS: ${p.obs_los.toFixed(1)} d</div>
     <div class="leg"><span><span class="sq" style="background:${RED}"></span>erhöht Vorhersage</span>
       <span><span class="sq" style="background:${BLUE}"></span>senkt Vorhersage</span></div>
     ${rows}`;
}
const gmax=Math.max(...D.global_importance.map(g=>g.gain));
document.getElementById('global').innerHTML = D.global_importance.map(g=>
  `<div class="girow"><div class="fname" title="${g.feature}">${g.feature}</div>
   <div class="bara" style="background:transparent"><div class="gibar" style="width:${g.gain/gmax*100}%"></div></div></div>`).join("");
sel(0);
</script>
"""

fragment = STYLE + BODY + SCRIPT.replace("__DATA__", DATA)
(AN/"_dashboard_fragment.html").write_text(fragment, encoding="utf-8")

standalone = ("<!DOCTYPE html><html lang='de'><head><meta charset='utf-8'>"
  "<meta name='viewport' content='width=device-width,initial-scale=1'>"
  "<title>ICU Stations-Dashboard</title>"
  "<style>:root{--font-sans:'Segoe UI',system-ui,sans-serif;"
  "--color-text-primary:#1a1a1a;--color-text-secondary:#555;--color-text-tertiary:#888;"
  "--color-background-primary:#fff;--color-background-secondary:#f4f6f9;"
  "--color-border-tertiary:rgba(0,0,0,.15);--color-border-secondary:rgba(0,0,0,.3);"
  "--color-border-primary:rgba(0,0,0,.4);--color-border-info:#2166ac}"
  "body{background:#fafbfc;margin:0;padding:18px 22px;max-width:1100px;margin:0 auto}</style>"
  + STYLE + "</head><body>" + BODY + SCRIPT.replace("__DATA__", DATA) + "</body></html>")
(AN/"dashboard_station.html").write_text(standalone, encoding="utf-8")

print("Gespeichert:")
print("  ", AN/"dashboard_station.html", "(im Browser öffnen)")
print("  ", AN/"_dashboard_fragment.html")
print("Fragment-Größe:", len(fragment), "Zeichen")
