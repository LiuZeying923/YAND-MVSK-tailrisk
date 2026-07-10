/* YAND-MVSK Tail-Risk Studio — vanilla SPA, zero build, hand-rolled SVG charts. */
"use strict";

const API = "";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const fmtPct = (x, d = 1) => (x >= 0 ? "+" : "") + (x * 100).toFixed(d) + "%";
const fmtPctAbs = (x, d = 1) => (x * 100).toFixed(d) + "%";
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// Categorical palette for weights (colorblind-aware, one hue family per slot).
const PALETTE = ["#5b54e6", "#0e9f6e", "#e08a1e", "#d64d6a", "#2b9bd6",
                 "#8b5cf6", "#4bb377", "#c2485f", "#e0a93b", "#3f6bd6",
                 "#9d5cc0", "#5aa9a0", "#c76b3f", "#6d78e0", "#3aa0b8"];

const state = {
  questions: [], profiles: [], answers: {},
  gamma: 6.0, gammaBreakdown: null,
  catalog: [], filtered: [], market: "all",
  selected: [],   // list of {symbol, name, market, ...}
  maxWeight: 35, horizon: 21, offline: false,
  result: null, busy: false,
};

/* ------------------------------------------------------------------ API */
async function api(path, opts) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = j.detail || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
}
const postJSON = (path, body) =>
  api(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

/* ------------------------------------------------------------------ nav */
function showStep(n) {
  $("#hero").classList.toggle("hidden", n !== 0);
  $("#stepper").classList.toggle("hidden", n === 0);
  [1, 2, 3].forEach((i) => $("#step-" + i).classList.toggle("hidden", i !== n));
  $$(".step").forEach((el) => {
    const s = +el.dataset.step;
    el.classList.toggle("active", s === n);
    el.classList.toggle("done", s < n);
    el.disabled = s > maxReachable();
  });
  if (n === 2 && !state.catalog.length) loadCatalog();
  window.scrollTo({ top: 0, behavior: "smooth" });
  state.step = n;
}
function maxReachable() {
  if (state.result) return 3;
  if (state.selected.length >= 2) return 2;
  return 2; // step 2 always reachable once past hero
}

/* ------------------------------------------------------------------ step 1: questionnaire */
async function loadQuestionnaire() {
  const data = await api("/api/questionnaire");
  state.questions = data.questions;
  state.profiles = data.profiles;
  const box = $("#questions");
  box.innerHTML = "";
  data.questions.forEach((q, i) => {
    const card = document.createElement("div");
    card.className = "panel q-card";
    card.innerHTML = `
      <div class="q-head">
        <span class="q-index">${q.id}</span>
        <span class="q-bias">${q.bias}</span>
      </div>
      <div class="q-prompt">${q.prompt}</div>
      <div class="q-options" data-q="${q.id}">
        ${q.choices.map((c) => `
          <button class="opt" data-key="${c.key}">
            <span class="key">${c.key}</span><span>${c.label}</span>
          </button>`).join("")}
      </div>`;
    box.appendChild(card);
  });
  box.addEventListener("click", (e) => {
    const btn = e.target.closest(".opt");
    if (!btn) return;
    const group = btn.closest(".q-options");
    const qid = group.dataset.q;
    $$(".opt", group).forEach((o) => o.classList.remove("selected"));
    btn.classList.add("selected");
    state.answers[qid] = btn.dataset.key;
    refreshGamma();
  });
}

let gammaTimer = null;
function refreshGamma() {
  const answered = Object.keys(state.answers).length;
  $("#to-step-2").disabled = false; // can proceed anytime; unanswered => base gamma
  clearTimeout(gammaTimer);
  gammaTimer = setTimeout(async () => {
    if (answered === 0) return;
    try {
      const bd = await postJSON("/api/gamma", { answers: state.answers, base_gamma: 6 });
      state.gamma = bd.gamma; state.gammaBreakdown = bd;
      $("#gp-val").textContent = bd.gamma.toFixed(1);
      $("#gp-profile").textContent = bd.profile;
      $("#gp-blurb").textContent = bd.profile_blurb;
      const pos = clamp((bd.gamma - 1.5) / (20 - 1.5), 0, 1) * 100;
      $("#gp-knob").style.left = pos + "%";
    } catch (e) { /* keep last */ }
  }, 120);
}

/* ------------------------------------------------------------------ step 2: picker */
async function loadCatalog() {
  try {
    const data = await api("/api/catalog");
    state.catalog = data.assets;
    renderAssets();
  } catch (e) {
    $("#asset-list").innerHTML = `<div class="banner err">Couldn't load catalog: ${e.message}</div>`;
  }
}

function renderAssets() {
  const q = $("#search").value.trim().toLowerCase();
  let list = state.catalog;
  if (state.market !== "all") list = list.filter((a) => a.market === state.market);
  if (q) list = list.filter((a) =>
    a.symbol.toLowerCase().includes(q) || a.name.toLowerCase().includes(q) || (a.sector || "").toLowerCase().includes(q));
  const picked = new Set(state.selected.map((s) => s.symbol));
  const el = $("#asset-list");
  if (!list.length) {
    el.innerHTML = `<div class="empty-hint" style="padding:20px;text-align:center">No match. You can still type any valid ticker and press Enter.</div>`;
    return;
  }
  el.innerHTML = list.map((a) => `
    <button class="asset-row ${picked.has(a.symbol) ? "picked" : ""}" data-sym="${a.symbol}">
      <span class="mkt-badge mkt-${a.market}">${a.market}</span>
      <span class="asset-main">
        <span class="sym">${a.symbol}</span>
        <span class="nm">${a.name}${a.sector ? " · " + a.sector : ""}</span>
      </span>
      <span class="add">${picked.has(a.symbol) ? "✓" : "+"}</span>
    </button>`).join("");
}

function toggleSelect(sym, meta) {
  const i = state.selected.findIndex((s) => s.symbol === sym);
  if (i >= 0) state.selected.splice(i, 1);
  else {
    if (state.selected.length >= 15) { flashBanner("Max 15 stocks — remove one first.", "warn"); return; }
    const rec = meta || state.catalog.find((a) => a.symbol === sym) ||
      { symbol: sym, name: sym, market: guessMarket(sym) };
    state.selected.push(rec);
  }
  renderChips(); renderAssets(); updateOptimizeBtn();
}
function guessMarket(sym) {
  const s = sym.toUpperCase();
  if (s.endsWith(".HK")) return "HK";
  if (s.endsWith(".SS") || s.endsWith(".SZ")) return "A";
  return "US";
}
function renderChips() {
  $("#sel-count").textContent = state.selected.length;
  const box = $("#chips");
  if (!state.selected.length) { box.innerHTML = `<span class="empty-hint">Add 2–15 stocks to optimize.</span>`; return; }
  box.innerHTML = state.selected.map((s) =>
    `<span class="chip"><span class="mkt-badge mkt-${s.market}">${s.market}</span>${s.symbol}<button data-rm="${s.symbol}" aria-label="remove">×</button></span>`).join("");
}
function updateOptimizeBtn() {
  $("#optimize-btn").disabled = state.selected.length < 2 || state.busy;
}
function flashBanner(msg, kind = "info", target = "#opt-banner") {
  const el = $(target);
  el.innerHTML = `<div class="banner ${kind}" style="margin-top:12px">${msg}</div>`;
  if (kind !== "err") setTimeout(() => { if (el.firstChild) el.innerHTML = ""; }, 4000);
}

/* ------------------------------------------------------------------ optimize */
async function runOptimize() {
  if (state.selected.length < 2) return;
  state.busy = true;
  const btn = $("#optimize-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spin"></span> Optimizing…`;
  $("#opt-banner").innerHTML = "";
  try {
    const body = {
      tickers: state.selected.map((s) => s.symbol),
      max_weight: state.maxWeight / 100,
      horizon: state.horizon,
      offline: state.offline,
      apply_guard: true,
      base_gamma: 6,
    };
    if (state.gammaBreakdown) body.answers = state.answers;
    else body.gamma = state.gamma;
    const res = await postJSON("/api/optimize", body);
    state.result = res;
    renderResults(res);
    showStep(3);
  } catch (e) {
    flashBanner("Optimization failed: " + e.message, "err");
  } finally {
    state.busy = false;
    btn.disabled = false;
    btn.innerHTML = "Optimize portfolio";
  }
}

/* ------------------------------------------------------------------ SVG charts */
function donut(weights, size = 190) {
  const r = size / 2, inner = r * 0.62, cx = r, cy = r;
  let a0 = -Math.PI / 2;
  const arcs = weights.map((w, i) => {
    const frac = w.weight;
    const a1 = a0 + frac * 2 * Math.PI;
    const large = a1 - a0 > Math.PI ? 1 : 0;
    const p = (ang, rad) => [cx + rad * Math.cos(ang), cy + rad * Math.sin(ang)];
    const [x0, y0] = p(a0, r), [x1, y1] = p(a1, r);
    const [xi1, yi1] = p(a1, inner), [xi0, yi0] = p(a0, inner);
    a0 = a1;
    const color = PALETTE[i % PALETTE.length];
    return `<path d="M${x0},${y0} A${r},${r} 0 ${large} 1 ${x1},${y1} L${xi1},${yi1} A${inner},${inner} 0 ${large} 0 ${xi0},${yi0} Z" fill="${color}"><title>${w.symbol} ${fmtPctAbs(frac)}</title></path>`;
  }).join("");
  const top = weights[0];
  return `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" role="img" aria-label="Portfolio weights donut">
    ${arcs}
    <g class="donut-center" text-anchor="middle">
      <text x="${cx}" y="${cy - 4}" font-family="var(--mono)" font-size="26" font-weight="680">${weights.length}</text>
      <text x="${cx}" y="${cy + 15}" font-size="11" fill="var(--muted)">holdings</text>
    </g>
  </svg>`;
}

function lineChart(port, bench, dates, w = 560, h = 220) {
  const pad = { l: 44, r: 14, t: 12, b: 26 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const all = port.concat(bench);
  const lo = Math.min(...all), hi = Math.max(...all);
  const yr = hi - lo || 1;
  const X = (i, n) => pad.l + (i / (n - 1)) * iw;
  const Y = (v) => pad.t + (1 - (v - lo) / yr) * ih;
  const path = (arr) => arr.map((v, i) => (i ? "L" : "M") + X(i, arr.length).toFixed(1) + "," + Y(v).toFixed(1)).join(" ");
  const area = `${path(port)} L${X(port.length - 1, port.length)},${pad.t + ih} L${pad.l},${pad.t + ih} Z`;
  // y gridlines
  const ticks = 4, grid = [];
  for (let k = 0; k <= ticks; k++) {
    const v = lo + (yr * k) / ticks, y = Y(v);
    grid.push(`<line x1="${pad.l}" y1="${y}" x2="${w - pad.r}" y2="${y}" stroke="var(--line)" stroke-width="1"/>
      <text x="${pad.l - 8}" y="${y + 3}" text-anchor="end" class="axis-lab">${v.toFixed(0)}</text>`);
  }
  const nd = dates.length;
  const xlabs = [0, Math.floor(nd / 2), nd - 1].map((i) =>
    `<text x="${X(i, nd)}" y="${h - 8}" text-anchor="middle" class="axis-lab">${(dates[i] || "").slice(0, 7)}</text>`).join("");
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" preserveAspectRatio="none" role="img" aria-label="Equity curve, optimized vs equal weight">
    ${grid.join("")}
    <path d="${area}" fill="var(--accent)" opacity="0.08"/>
    <path d="${path(bench)}" fill="none" stroke="var(--muted)" stroke-width="1.6" stroke-dasharray="4 4" opacity="0.8"/>
    <path d="${path(port)}" fill="none" stroke="var(--accent)" stroke-width="2.4"/>
    ${xlabs}
  </svg>`;
}

function gauge(score, w = 210, h = 128) {
  const cx = w / 2, cy = h - 14, r = 88;
  const a = Math.PI * (1 - clamp(score, 0, 100) / 100);
  const p = (ang, rad = r) => [cx + rad * Math.cos(ang), cy - rad * Math.sin(ang)];
  const [sx, sy] = p(Math.PI), [ex, ey] = p(0), [px, py] = p(a);
  const col = score >= 75 ? "var(--up)" : score >= 55 ? "var(--warn)" : "var(--down)";
  const large = Math.PI - a > Math.PI ? 1 : 0;
  // Filled progress arc + endpoint dot + centered number (no needle — cleaner).
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="Tail-risk resilience score ${Math.round(score)} of 100">
    <path d="M${sx},${sy} A${r},${r} 0 0 1 ${ex},${ey}" fill="none" stroke="var(--line)" stroke-width="13" stroke-linecap="round"/>
    <path d="M${sx},${sy} A${r},${r} 0 ${large} 1 ${px},${py}" fill="none" stroke="${col}" stroke-width="13" stroke-linecap="round"/>
    <circle cx="${px}" cy="${py}" r="7" fill="var(--surface)" stroke="${col}" stroke-width="3"/>
    <text x="${cx}" y="${cy - 12}" text-anchor="middle" font-family="var(--mono)" font-size="34" font-weight="680" fill="${col}">${Math.round(score)}</text>
    <text x="${cx}" y="${cy + 6}" text-anchor="middle" font-size="11" fill="var(--muted)">/ 100 resilience</text>
  </svg>`;
}

/* ------------------------------------------------------------------ results render */
function stat(k, v, cls = "", sub = "") {
  return `<div class="stat"><div class="k">${k}</div><div class="v ${cls}">${v}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}

function renderResults(res) {
  const p = res.performance, risk = res.risk, w = res.weights;
  const legend = w.map((row, i) => `
    <div class="legend-row">
      <span class="sw" style="background:${PALETTE[i % PALETTE.length]}"></span>
      <span class="nm"><span class="sym">${row.symbol}</span> · ${row.name}</span>
      <span class="wt">${fmtPctAbs(row.weight)}</span>
    </div>`).join("");

  const findings = risk.findings.map((f) => {
    const ok = /no significant/i.test(f);
    return `<li class="${ok ? "ok" : ""}"><span class="b">${ok ? "✓" : "!"}</span><span>${f}</span></li>`;
  }).join("");

  const stress = risk.stress.map((s) => {
    const neg = s.portfolio_return < 0;
    return `<div class="stress-row">
      <div><div class="nm">${s.name}</div><div class="ds">${s.description}</div></div>
      <div class="ret" style="color:${neg ? "var(--down)" : "var(--up)"}">${fmtPct(s.portfolio_return, 1)}</div>
    </div>`;
  }).join("");

  const srcNote = res.data_source === "synthetic"
    ? `<div class="banner info" style="margin-bottom:16px">Using <b>offline synthetic</b> demo data — deterministic, no market prices fetched.</div>`
    : (res.synthetic_symbols.length
        ? `<div class="banner warn" style="margin-bottom:16px">Live data used, but ${res.synthetic_symbols.join(", ")} fell back to synthetic (no history returned).</div>` : "");

  let reoptHTML = "";
  const rec = res.recommendation;
  if (rec && rec.action === "reconfigure" && res.reoptimized) {
    const ro = res.reoptimized;
    const applyLabel = ro.kind === "cap" ? `Apply cap ${Math.round(ro.max_weight * 100)}%` : `Apply γ = ${ro.gamma}`;
    reoptHTML = `
      <div class="reopt">
        <div class="ico">↑γ</div>
        <div class="txt">
          <h4>Tail-Risk Guard: a safer configuration is available</h4>
          <p>${cap1(rec.how)} would lift the resilience score
          <b style="color:var(--up)">+${rec.score_delta}</b> → <b>${rec.new_score}</b>.</p>
        </div>
        <button class="btn btn-primary" id="apply-reopt">${applyLabel}</button>
      </div>`;
  } else if (rec && rec.action === "structural") {
    reoptHTML = `
      <div class="reopt" style="border-style:solid;border-color:var(--warn);background:var(--warn-soft)">
        <div class="ico" style="background:var(--warn)">!</div>
        <div class="txt"><h4 style="color:var(--warn)">Residual tail risk is structural</h4>
        <p>${rec.how}</p></div>
      </div>`;
  }

  const bench = res.benchmark;
  $("#results").innerHTML = `
    ${srcNote}
    <div class="panel result-head">
      <div>
        <h2>Your tail-risk-aware portfolio</h2>
        <div class="subtle">${res.request.tickers.length} assets · ${res.n_observations} trading days
          (${res.date_range[0]} → ${res.date_range[1]}) · horizon ${res.request.horizon}d · source: ${res.data_source}</div>
      </div>
      <div class="gamma-badge">
        <span class="g">${res.gamma}</span>
        <span class="t">γ risk<br/>aversion${res.gamma_breakdown ? "<br/>· " + res.gamma_breakdown.profile : ""}</span>
      </div>
    </div>

    ${reoptHTML}

    <div class="result-grid">
      <div class="panel card-pad">
        <div class="card-title">Optimal weights</div>
        <div class="card-note">YAND-MVSK allocation across all four moments, capped at ${fmtPctAbs(res.request.max_weight)} per name.</div>
        <div class="donut-wrap">
          ${donut(w)}
          <div class="legend">${legend}</div>
        </div>
      </div>

      <div class="panel card-pad">
        <div class="card-title">Tail-Risk Guard</div>
        <div class="card-note">${risk.headline}</div>
        <div class="guard-top">
          ${gauge(risk.score)}
          <div>
            <span class="guard-level lvl-${risk.level}">${risk.level}</span>
            <ul class="findings">${findings}</ul>
          </div>
        </div>
      </div>
    </div>

    <div class="panel card-pad">
      <div class="card-title">Growth of 100 · in-sample</div>
      <div class="card-note">Optimized portfolio versus an equal-weight basket of the same stocks.</div>
      <div class="chart-legend">
        <span><i style="background:var(--accent)"></i> Optimized · Sharpe ${p.sharpe.toFixed(2)}</span>
        <span><i style="background:var(--muted)"></i> Equal weight · Sharpe ${bench.sharpe.toFixed(2)}</span>
      </div>
      ${lineChart(res.equity_curve, bench.equity_curve, res.equity_dates)}
    </div>

    <div class="result-grid">
      <div class="panel card-pad">
        <div class="card-title">Performance</div>
        <div class="card-note">Annualized on daily returns.</div>
        <div class="stat-grid">
          ${stat("Ann. return", fmtPct(p.ann_return, 1), p.ann_return >= 0 ? "up" : "down")}
          ${stat("Ann. volatility", fmtPctAbs(p.ann_volatility, 1))}
          ${stat("Sharpe", p.sharpe.toFixed(2), p.sharpe >= 1 ? "up" : "")}
          ${stat("Sortino", p.sortino.toFixed(2))}
          ${stat("Max drawdown", fmtPct(p.max_drawdown, 1), "down")}
          ${stat("95% CVaR", fmtPctAbs(p.cvar_95, 2), "down", "expected shortfall")}
        </div>
      </div>
      <div class="panel card-pad">
        <div class="card-title">Higher moments</div>
        <div class="card-note">What YAND optimizes beyond mean &amp; variance.</div>
        <div class="stat-grid">
          ${stat("Skewness", (p.skewness >= 0 ? "+" : "") + p.skewness.toFixed(2), p.skewness >= 0 ? "up" : "down", p.skewness >= 0 ? "upside tilt" : "downside tilt")}
          ${stat("Excess kurtosis", (p.excess_kurtosis >= 0 ? "+" : "") + p.excess_kurtosis.toFixed(2), p.excess_kurtosis > 1 ? "down" : "", "tail fatness")}
          ${stat("Assets held", p.n_assets_held + "/" + res.request.tickers.length)}
          ${stat("95% VaR", fmtPctAbs(p.var_95, 2), "down")}
          ${stat("Solve time", (p.solver.seconds * 1000).toFixed(0) + " ms", "", p.solver.iterations + " iters")}
          ${stat("Converged", p.solver.converged ? "yes" : "partial", p.solver.converged ? "up" : "")}
        </div>
      </div>
    </div>

    <div class="panel card-pad">
      <div class="card-title">Stress tests</div>
      <div class="card-note">How this exact basket would have fared under tail scenarios.</div>
      <div class="stress-grid">${stress}</div>
    </div>`;

  const applyBtn = $("#apply-reopt");
  if (applyBtn) applyBtn.addEventListener("click", () => applyReopt(res.reoptimized));
}

function applyReopt(ro) {
  // Swap the displayed result to the guard-recommended reconfiguration.
  const res = state.result;
  const merged = Object.assign({}, res, {
    gamma: ro.gamma, weights: ro.weights, performance: ro.performance,
    risk: ro.risk, equity_curve: ro.equity_curve,
    reoptimized: null, recommendation: null, gamma_breakdown: res.gamma_breakdown,
  });
  merged.request = Object.assign({}, res.request, { gamma: ro.gamma, max_weight: ro.max_weight });
  state.result = merged;
  renderResults(merged);
  window.scrollTo({ top: 0, behavior: "smooth" });
}
const cap1 = (s) => s ? s.charAt(0).toUpperCase() + s.slice(1) : s;

/* ------------------------------------------------------------------ wiring */
function init() {
  loadQuestionnaire();

  $("#start-btn").addEventListener("click", () => showStep(1));
  $$(".step").forEach((el) => el.addEventListener("click", () => {
    const s = +el.dataset.step;
    if (s <= maxReachable()) showStep(s);
  }));

  $("#to-step-2").addEventListener("click", () => showStep(2));
  $("#skip-q").addEventListener("click", () => { state.gamma = 6; state.gammaBreakdown = null; showStep(2); });
  $("#back-1").addEventListener("click", () => showStep(1));
  $("#back-2").addEventListener("click", () => showStep(2));
  $("#restart").addEventListener("click", () => { state.result = null; showStep(0); });

  // picker
  $("#search").addEventListener("input", renderAssets);
  $("#search").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const v = e.target.value.trim().toUpperCase();
      if (v && !state.selected.find((s) => s.symbol === v)) { toggleSelect(v); e.target.value = ""; renderAssets(); }
    }
  });
  $("#asset-list").addEventListener("click", (e) => {
    const row = e.target.closest(".asset-row");
    if (row) toggleSelect(row.dataset.sym);
  });
  $("#chips").addEventListener("click", (e) => {
    const b = e.target.closest("[data-rm]");
    if (b) toggleSelect(b.dataset.rm);
  });
  $("#mkt-tabs").addEventListener("click", (e) => {
    const t = e.target.closest(".mkt-tab");
    if (!t) return;
    $$(".mkt-tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active"); state.market = t.dataset.mkt; renderAssets();
  });

  // settings
  const mw = $("#max-weight");
  mw.addEventListener("input", () => { state.maxWeight = +mw.value; $("#mw-val").textContent = mw.value + "%"; });
  $("#horizon-seg").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $$("#horizon-seg button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active"); state.horizon = +b.dataset.h;
  });
  $("#offline-toggle").addEventListener("change", (e) => { state.offline = e.target.checked; });

  $("#optimize-btn").addEventListener("click", runOptimize);

  showStep(0);
}

document.addEventListener("DOMContentLoaded", init);
