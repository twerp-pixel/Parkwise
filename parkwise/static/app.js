import { h, lineChart, legend, heatmap, hBars, dotPlot, stackedColumns, confusion, fmt, hideTip } from "/static/charts.js";

const state = { meta: null, report: null, date: null, zone: 1, studio: null };
const view = document.getElementById("view");
const V = (name) => `var(--${name})`;
// Native replaceChildren() renders null as the text "null"; this drops empty children first.
const fill = (node, ...kids) => node.replaceChildren(...kids.flat().filter(k => k !== null && k !== undefined && k !== false));
const SHORT = ["Indiranagar", "Koramangala", "MG Road", "Whitefield", "Manyata", "Majestic", "Rajajinagar", "Jayanagar", "Stadium", "Hebbal", "Airport", "Sarjapur"];

// ---------------------------------------------------------------------------------------------
async function api(path, opts) {
  let res;
  try { res = await fetch(path, opts); } catch { throw toast("Can't reach the server. Check that it is running and reload."); }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw toast(body.error?.message || `Request failed (${res.status})`);
  return body;
}
function toast(msg) { const t = h("div", { class: "toast", role: "alert" }, msg); document.body.append(t); setTimeout(() => t.remove(), 6000); return new Error(msg); }
const dlabel = (d) => new Date(d + "T00:00:00").toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
const zname = (id) => state.meta.zones[id].name;

function dateSelect(onchange) {
  const s = h("select", { "aria-label": "Date", onchange: e => { state.date = e.target.value; onchange(); } },
    state.meta.dates.slice().reverse().map(d => h("option", { value: d.date, selected: d.date === state.date },
      `${dlabel(d.date)} · ${d.kind === "live" ? "live forecast" : "backtest"}${d.is_offday ? " · off-day" : ""}`)));
  return s;
}
function zoneSelect(onchange) {
  return h("select", { "aria-label": "Zone", onchange: e => { state.zone = +e.target.value; onchange(); } },
    state.meta.zones.map(z => h("option", { value: z.id, selected: z.id === state.zone }, z.name)));
}
function seg(options, value, onpick, label) {
  return h("div", { class: "seg", role: "group", "aria-label": label },
    options.map(([v, text]) => h("button", { type: "button", "aria-pressed": String(v === value), onclick: () => onpick(v) }, text)));
}
function pageHead(title, sub, ...controls) {
  return h("div", { class: "page-head" }, h("div", {}, h("h1", {}, title), sub ? h("p", {}, sub) : null), h("div", { class: "controls" }, ...controls));
}
function panel(title, sub, ...body) {
  return h("section", { class: "panel" }, title ? h("header", {}, h("div", {}, h("h2", {}, title), sub ? h("p", {}, sub) : null)) : null, ...body);
}
function dayBanner(date) {
  const live = date === state.meta.live_date;
  return h("div", { class: "callout info" }, live
    ? `Live forecast for ${dlabel(date)}. No outcomes exist yet, so everything shown is an estimate.`
    : `Out-of-sample evaluation · ${dlabel(date)}. Forecasts are compared with observed demand.`);
}

function kpi(label, value, sub, goodWhen) {
  return h("div", { class: "kpi" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value), h("div", { class: "sub" }, sub));
}
function delta(v, goodUp, text) {
  const cls = Math.abs(v) < 0.0005 ? "neutral" : ((v > 0) === goodUp ? "up" : "down");
  return h("span", { class: `delta ${cls}` }, text ?? fmt.spct(v));
}
function kpiRow(t) {
  const d = t.deltas, p = t.plan, s = t.status_quo;
  return h("div", { class: "kpis" },
    kpi("Revenue per day", fmt.inr(p.revenue), h("span", {}, delta(d.revenue_pct, true), ` vs current schedule (${fmt.inr(s.revenue)})`)),
    kpi("Bay utilisation", fmt.pct(p.utilization, 1), h("span", {}, delta(d.utilization_pts, true, (d.utilization_pts >= 0 ? "+" : "−") + Math.abs(d.utilization_pts * 100).toFixed(1) + " pts"), " vs current")),
    kpi("Drivers turned away at full lots", fmt.int(p.turned_away), h("span", {}, delta(d.turned_away_pct, false), ` vs ${fmt.int(s.turned_away)} now`)),
    kpi("Drivers priced out", fmt.int(p.lost_demand), h("span", {}, delta(d.lost_demand_pct, false), ` vs ${fmt.int(s.lost_demand)} now`)),
    kpi("Overcharge complaints per 10k parked hours", p.complaints_per_10k.toFixed(1), h("span", {}, delta(p.complaints_per_10k / s.complaints_per_10k - 1, false), ` vs ${s.complaints_per_10k.toFixed(1)} now`)));
}
function zoneTable(zones, compact = false) {
  const head = compact ? ["Zone", "Peak occupancy now", "Average price", "Revenue", "Recommended change"]
    : ["Zone", "Busiest hour now", "Peak occupancy now", "Average price", "Revenue", "Recommended change"];
  return h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, head.map((t, i) => h("th", { class: i > 0 && i < head.length - 1 ? "num" : "" }, t)))),
    h("tbody", {}, zones.map(z => h("tr", {},
      h("td", {}, z.name, z.locked ? h("span", { class: "chip amber", style: "margin-left:8px" }, "locked") : null),
      compact ? null : h("td", { class: "num" }, fmt.hour(z.peak_hour)),
      h("td", { class: "num" }, h("span", { class: z.peak_occupancy_status_quo >= .95 ? "chip amber" : "" }, fmt.pct(z.peak_occupancy_status_quo))),
      h("td", { class: "num" }, fmt.mult(z.avg_multiplier)),
      h("td", { class: "num" }, delta(z.revenue_delta_pct, true)),
      h("td", {}, z.action))))));
}

// ---------------------------------------------------------------------------------------------
async function overviewView() {
  const root = h("div", {}, h("div", { class: "skeleton" }, "Loading…"));
  fill(view, pageHead("Overview", "Where lots will fill up and what a guardrailed price plan does about it.", dateSelect(overviewView)), root);
  const o = await api(`/api/overview?date=${state.date}`);
  let mode = "status_quo";
  const heatBox = h("div", {});
  const drawHeat = () => {
    const vals = mode === "status_quo" ? o.heatmap_status_quo : o.heatmap_plan;
    fill(heatBox, heatmap({ rows: o.zones.map(z => z.name), values: vals,
      onTip: (r, c, v) => `<b>${o.zones[r].name}</b> at ${fmt.hour(c)}<br>${fmt.pct(v)} full${v >= .95 ? "<br>Lot is full" : ""}<br>Price ${fmt.mult(o.zones[r][mode === "status_quo" ? "status_quo_multipliers" : "multipliers"][c])}` }),
      h("div", { class: "legend" }, h("span", {}, "Fuller lots are darker. An amber outline means 95% or more of bays are taken.")));
  };
  const segBox = h("div", {});
  const drawSeg = () => fill(segBox, seg([["status_quo", "Current schedule"], ["plan", "Recommended plan"]], mode, v => { mode = v; drawSeg(); drawHeat(); }, "Heatmap scenario"));
  drawSeg(); drawHeat();
  const rc = o.reality_check;
  root.replaceWith(h("div", {},
    dayBanner(state.date), kpiRow(o.totals),
    rc ? h("div", { class: "callout" }, `Simulation validation · Forecast uplift: ${fmt.spct(rc.expected_revenue_pct)} · Simulated uplift: ${fmt.spct(rc.revenue_pct)} · Net of goodwill cost: ${fmt.spct(rc.net_value_pct)}.`) : null,
    o.at_risk.length ? h("div", { class: "callout" }, h("b", {}, "Lots expected to fill: "), o.at_risk.map(a => `${a.zone} (${fmt.hour(a.hour)})`).join(", "), ".") : null,
    panel("Expected occupancy by zone and hour", "Median forecast, using each zone's price schedule for the selected plan.", segBox, heatBox),
    panel("Recommendations by zone", "Compared with today's static schedule. Plans respect the default guardrails (see Pricing studio).", zoneTable(o.zones))));
}

// ---------------------------------------------------------------------------------------------
async function forecastView() {
  const body = h("div", {}, h("div", { class: "skeleton" }, "Loading…"));
  fill(view, pageHead("Forecast", "Day-ahead occupancy forecast with an 80% prediction interval.", zoneSelect(forecastView), dateSelect(forecastView)), body);
  const [f, hist] = await Promise.all([api(`/api/forecast?zone_id=${state.zone}&date=${state.date}`), api(`/api/history?zone_id=${state.zone}`)]);
  const series = [{ name: "Forecast (median)", color: V("s1"), values: f.p50 }, { name: "Same hour last week", color: V("s7"), values: f.naive, dash: "5 4", width: 1.8 }];
  if (f.actual) series.push({ name: "Actual", color: V("ink"), values: f.actual, dots: true, width: 1.4 });
  const chart = lineChart({ series, band: { lo: f.p10, hi: f.p90, color: V("s1") }, refLines: [{ y: 1, label: "Full", color: V("bad") }], yMax: 1.05, ariaLabel: "Forecast occupancy by hour",
    tipRows: i => [`Forecast <b>${fmt.pct(f.p50[i])}</b> (${fmt.pct(f.p10[i])}–${fmt.pct(f.p90[i])})`, `Last week's pattern ${fmt.pct(f.naive[i])}`, f.actual ? `Actual <b>${fmt.pct(f.actual[i])}</b>` : "", `Price ${fmt.mult(f.price_multiplier[i])}`].filter(Boolean) });
  const dm = f.day_metrics;
  const hv = hist.utilization, hd = hist.dates;
  const hchart = lineChart({ series: [{ name: "Utilisation", color: V("s3"), values: hv }], yMax: 0.8, yTicks: [0, .2, .4, .6, .8], height: 170,
    xLabel: i => (i % 20 === 0 ? hd[i].slice(5) : ""), tipRows: i => [`${hd[i]}: <b>${fmt.pct(hv[i], 1)}</b> of bays used`], ariaLabel: "Daily utilisation, last 120 days" });
  body.replaceWith(h("div", {}, dayBanner(state.date),
    panel(`${f.zone}: ${dlabel(f.date)}`, f.is_live ? "Price shown is the current schedule." : "Price shown is what was actually charged that day.",
      chart, legend([{ name: "Forecast (median)", color: V("s1"), line: true }, { name: "80% range", color: "color-mix(in srgb, var(--s1) 22%, transparent)" }, { name: "Same hour last week", color: V("s7"), line: true }, ...(f.actual ? [{ name: "Actual", color: V("ink"), line: true }] : [])]),
      dm ? h("p", { class: "bars-note" }, `Average error today: ${fmt.pct(dm.mae, 1)} of bays (last-week pattern: ${fmt.pct(dm.naive_mae, 1)}). ${fmt.pct(dm.coverage)} of hours landed inside the 80% range.`) : null),
    panel("Utilisation, last 120 days", "Share of bays in use per day. Level shifts like a new competitor or office tower show up here first.", hchart)));
}

// ---------------------------------------------------------------------------------------------
function studioDefaults() {
  const g = state.meta.default_guardrails;
  return { strategy: "optimized", scenario: "p50", spill: 0.25, locked: new Set(), adj: {}, g: { ...g } };
}
async function pricingView() {
  state.studio ||= studioDefaults();
  const S = state.studio, out = h("div", {}), ctl = h("div", {});
  fill(view, pageHead("Pricing studio", "Try a plan, change the guardrails, and see revenue, volume and complaints move together.", dateSelect(pricingView)),
    h("div", { class: "studio" }, h("div", { class: "panel controls-panel" }, ctl), out));
  let timer, seq = 0;
  const run = () => { clearTimeout(timer); timer = setTimeout(simulate, 180); };
  async function simulate() {
    const my = ++seq;
    const g = { ...S.g, avg_cap: Math.min(S.g.avg_cap, S.g.max_mult) };
    const body = { date: state.date, strategy: S.strategy, scenario: S.scenario, guardrails: g, spill_share: S.spill, locked_zones: [...S.locked],
      adjustments: S.strategy === "custom" ? S.adj : {} };
    let r;
    try { r = await api("/api/simulate", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }); } catch { return; }
    if (my !== seq) return;
    const rc = r.reality_check;
    const grid = h("div", { class: "mini-grid" }, r.zones.map(z => {
      const c = lineChart({ series: [{ name: "Current schedule", color: V("s7"), values: z.status_quo_multipliers, dash: "4 3", width: 1.6 }, { name: "Plan", color: V("s1"), values: z.multipliers }],
        yMin: 0.6, yMax: Math.max(1.5, ...z.multipliers) + .05, yTicks: [0.8, 1, 1.2, 1.4], yFormat: v => v.toFixed(1), height: 120, width: 380, xLabel: i => (i % 6 === 0 ? String(i).padStart(2, "0") : ""),
        tipRows: i => [`Plan <b>${fmt.mult(z.multipliers[i])}</b>`, `Current ${fmt.mult(z.status_quo_multipliers[i])}`, `Occupancy ${fmt.pct(z.occupancy[i])}`], ariaLabel: `${z.name} price multipliers` });
      return h("div", { class: "mini" }, h("h3", {}, z.name), c);
    }));
    fill(out, dayBanner(r.date), kpiRow(r.totals),
      rc ? h("div", { class: "callout" }, `Simulation validation · Forecast uplift: ${fmt.spct(rc.expected_revenue_pct)} · Simulated uplift: ${fmt.spct(rc.revenue_pct)} · Net of goodwill cost: ${fmt.spct(rc.net_value_pct)}.`) : null,
      panel("Hourly price multipliers", "Solid blue is the plan, dashed grey is today's schedule.", grid),
      panel("By zone", null, zoneTable(r.zones, true)));
  }
  function slider(label, key, min, max, step, fmtv, hint, target = S.g) {
    const val = h("b", {}, fmtv(target[key]));
    const inp = h("input", { type: "range", min, max, step, value: target[key], "aria-label": label, oninput: e => { target[key] = +e.target.value; val.textContent = fmtv(target[key]); run(); } });
    return h("div", { class: "field" }, h("label", {}, label, val), inp, hint ? h("div", { class: "hint" }, hint) : null);
  }
  function drawControls() {
    const optimized = S.strategy === "optimized";
    fill(ctl, 
      h("div", { class: "field" }, h("label", {}, "Plan"), seg([["optimized", "Recommended"], ["status_quo", "Current"], ["flat", "Flat"], ["custom", "Manual"]], S.strategy, v => { S.strategy = v; drawControls(); run(); }, "Plan")),
      h("div", { class: "field" }, h("label", {}, "Demand assumption"), seg([["p10", "Low"], ["p50", "Expected"], ["p90", "High"]], S.scenario, v => { S.scenario = v; drawControls(); run(); }, "Demand")),
      optimized ? h("div", {},
        slider("Highest price", "max_mult", 1.0, 2.0, .05, fmt.mult, "No hour can exceed this multiple of the base price."),
        slider("Average price cap", "avg_cap", 1.0, 1.5, .01, fmt.mult, "Cap on each zone's daily average multiplier."),
        slider("Hourly change limit", "max_step", .05, .5, .05, v => "±" + v.toFixed(2), "Keeps prices from jumping between hours."),
        slider("Cost of pricing out a driver", "deter_cost", 0, 2, .1, v => v.toFixed(1) + " h", "Higher favours volume over yield, in hours of base price."),
        slider("Cost of turning a driver away", "turn_cost", 0, 2, .1, v => v.toFixed(1) + " h", "Penalty when a lot is full."),
        h("div", { class: "field" }, h("label", {}, "Zones kept on today's schedule"),
          h("div", { class: "locks" }, state.meta.zones.map(z => h("label", {}, h("input", { type: "checkbox", checked: S.locked.has(z.id), onchange: e => { e.target.checked ? S.locked.add(z.id) : S.locked.delete(z.id); run(); } }), SHORT[z.id]))))) : null,
      S.strategy === "custom" ? h("div", {}, h("p", { class: "small muted" }, "Shift each zone's current schedule up or down."),
        state.meta.zones.map(z => { S.adj[z.id] ??= 0; return slider(z.name, z.id, -.3, .5, .05, v => fmt.spct(v, 0), null, S.adj); })) : null,
      slider("Drivers who try a nearby lot", "spill", 0, .8, .05, v => fmt.pct(v), "An assumption, not learned from data. Results barely move across 0–50%.", S),
      h("button", { type: "button", class: "seg-reset", onclick: () => { state.studio = studioDefaults(); pricingView(); } }, "Reset to defaults"));
  }
  drawControls();
  fill(out, h("div", { class: "skeleton" }, "Loading…"));
  simulate();
}

// ---------------------------------------------------------------------------------------------
const table = (head, rows, numFrom = 1) => h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, head.map((t, i) => h("th", { class: i >= numFrom ? "num" : "" }, t)))),
  h("tbody", {}, rows.map(r => h("tr", {}, r.map((c, i) => h("td", { class: i >= numFrom ? "num" : "" }, c)))))));

function healthView() {
  const r = state.report, f = r.forecast, p = r.policy;
  const C = { gbm: V("s1"), naive: V("s7"), hist: V("s3") };
  const pt = f.point;
  const accuracy = hBars({ ariaLabel: "Forecast error by model", format: v => fmt.pct(v, 1), items: [
    { label: "Gradient boosting", values: [{ v: pt.gbm.wmape, color: C.gbm, name: "Error (WMAPE)" }] },
    { label: "Last 4 weeks average", values: [{ v: pt.hist_avg_4w.wmape, color: C.hist, name: "Error (WMAPE)" }] },
    { label: "Same hour last week", values: [{ v: pt.seasonal_naive.wmape, color: C.naive, name: "Error (WMAPE)" }] }] });
  const zoneBars = hBars({ ariaLabel: "Error by zone", format: v => fmt.pct(v, 1), items: f.by_zone.map(z => ({ label: z.zone, values: [{ v: z.gbm_wmape, color: C.gbm, name: "Model" }, { v: z.naive_wmape, color: C.naive, name: "Last week" }] })) });
  const abl = Object.entries(f.ablation_wmape);
  const ablation = hBars({ ariaLabel: "Error when feature groups are removed", format: v => fmt.pct(v, 1), items: abl.map(([k, v]) => ({ label: k, values: [{ v, color: k === "Full model" ? C.gbm : V("s2"), name: "Error" }] })) });
  const imp = Object.entries(f.permutation_importance.mae_increase_when_shuffled).sort((a, b) => b[1] - a[1]);
  const importance = hBars({ ariaLabel: "Permutation importance", format: v => "+" + v.toFixed(3), labelW: 230, items: imp.map(([k, v]) => ({ label: k, values: [{ v: Math.max(v, 0), color: V("s4"), name: "Error increase when shuffled" }] })) });
  const el = r.elasticity;
  const dots = dotPlot({ xMin: -1.5, xMax: 0.1, items: el.map(e => ({ label: e.zone, est: e.beta, lo: e.ci_lo, hi: e.ci_hi, truth: e.true_beta, naive: e.beta_naive_censored })) });
  const covered = el.filter(e => e.true_beta >= e.ci_lo && e.true_beta <= e.ci_hi).length;
  const P = p.policies, order = [["status_quo", "Current schedule"], ["flat", "Flat price"], ["dynamic_naive_forecast", "Optimised on last-week forecast"], ["dynamic_model", "Optimised on model forecast"], ["oracle", "Oracle (knows true demand)"]];
  const policyBars = hBars({ ariaLabel: "Uplift by policy", format: v => fmt.spct(v), labelW: 240, items: order.map(([k, l]) => ({ label: l, values: [{ v: Math.max(P[k].revenue_uplift_vs_status_quo, 0), color: V("s1"), name: "Revenue uplift" }, { v: Math.max(P[k].net_value_uplift_vs_status_quo, 0), color: V("s3"), name: "Net of goodwill cost" }] })) });
  const cs = f.cold_start, med = a => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
  const seen = cs.filter(c => c.archetype_seen_elsewhere);
  const sp = p.sensitivity_true_spill_share, sb = p.sensitivity_elasticity_misspecification, ev = p.expected_vs_realized;
  fill(view, pageHead("Model health", "Everything here is measured on days the models never saw. Because the data is simulated, results can also be checked against the true answer."),
    h("div", { class: "kpis" },
      kpi("Forecast error (WMAPE)", fmt.pct(pt.gbm.wmape, 1), `vs ${fmt.pct(pt.seasonal_naive.wmape, 1)} for last week's pattern`),
      kpi("80% range coverage", fmt.pct(f.interval.coverage), "Target is 80%. Slightly under."),
      kpi("Full-lot warnings", `${fmt.pct(f.full_lot_detection.gbm.precision)} / ${fmt.pct(f.full_lot_detection.gbm.recall)}`, "precision / recall of 'lot will be ≥95% full'"),
      kpi("Pricing replay (net of goodwill)", fmt.spct(P.dynamic_model.net_value_uplift_vs_status_quo), `${fmt.pct(p.share_of_oracle_net_gain)} of the oracle's gain`)),
    panel("Forecast accuracy", `Hold-out window ${dlabel(new Date(Date.parse(r.data.val_end) + 864e5).toISOString().slice(0, 10))} to ${dlabel(r.data.end)}. WMAPE is total absolute error divided by total demand. Even a perfect model would score about ${fmt.pct(f.poisson_noise_floor_mae / (pt.gbm.mae / pt.gbm.wmape), 0)} because arrivals are random.`, accuracy),
    panel("Error by zone", "Model (blue) against the same hour last week (grey).", zoneBars, legend([{ name: "Model", color: C.gbm }, { name: "Same hour last week", color: C.naive }])),
    panel("What the model relies on", "Refit test: error after removing a feature group and retraining. History matters most; events help where they apply.", ablation),
    panel("Reliance in the finished model", "Error added by shuffling one feature group at prediction time. Shuffling punishes any reliance, while the refit test above credits only information found nowhere else, so the two rankings differ on purpose.", importance),
    panel("Price sensitivity, estimated from randomised experiments", `Blue dot and bar: estimate with 95% interval. Black tick: true value. Orange diamond: naive estimate from occupancy, which is capped at capacity. ${covered} of ${el.length} intervals contain the truth, and the naive estimate understates sensitivity in lots that fill up.`, dots),
    panel("Pricing replay on held-out days", `${p.n_days} days, scored against the simulator's true demand. Net of goodwill = revenue minus the cost of drivers turned away or priced out. The model's own expectation was ${fmt.spct(ev.expected_revenue_uplift)} revenue; the replay realised ${fmt.spct(ev.realized_revenue_uplift)}.`, policyBars,
      table(["Policy", "Revenue", "Revenue vs now", "Net vs now", "Utilisation", "Turned away", "Priced out", "Avg price"], order.map(([k, l]) => { const x = P[k]; return [l, fmt.inr(x.revenue), fmt.spct(x.revenue_uplift_vs_status_quo), fmt.spct(x.net_value_uplift_vs_status_quo), fmt.pct(x.utilization, 1), fmt.int(x.turned_away), fmt.int(x.lost_demand), fmt.mult(x.avg_multiplier)]; }))),
    h("div", { class: "grid2" },
      panel("If drivers behave differently", "Revenue uplift when the true share of drivers who try a nearby lot is changed.", table(["Nearby-lot share", "Revenue", "Net"], Object.entries(sp).map(([k, v]) => [fmt.pct(+k), fmt.spct(v.revenue), fmt.spct(v.net_value)]))),
      panel("If elasticity is mis-estimated", "Optimiser fed with sensitivities scaled by a common factor.", table(["Scale", "Revenue", "Net"], Object.entries(sb).map(([k, v]) => [`×${k}`, fmt.spct(v.revenue), fmt.spct(v.net_value)])))),
    panel("Launching a new lot", `With no history, forecasts come from the area's mix of offices, shops and homes. Where that mix exists elsewhere in the network the median error is ${fmt.pct(med(seen.map(c => c.cold_start_wmape)))}, against ${fmt.pct(med(seen.map(c => c.network_average_wmape)))} for a blind network average. Lots with an archetype seen nowhere else are not helped, so they need fast adaptation instead. "Recalibrated" rescales using the first week of real data.`,
      table(["Zone", "Dominant type", "Seen elsewhere", "No history", "After 1 week", "Network average", "Full model"], cs.map(c => [c.zone, c.dominant_archetype, c.archetype_seen_elsewhere ? "yes" : "no", fmt.pct(c.cold_start_wmape), fmt.pct(c.cold_start_recalibrated_wmape), fmt.pct(c.network_average_wmape), fmt.pct(c.full_model_wmape)]), 1)),
    panel("Assumptions and limits", null, h("ul", { class: "plain" }, r.assumptions.map(a => h("li", {}, a)))));
}

function voiceView() {
  const t = state.report.tickets, voc = t.voc, L = voc.labels, cats = voc.weekly.categories;
  const colors = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"].map(V);
  const cols = stackedColumns({ labels: voc.weekly.weeks, series: cats.map((c, i) => ({ name: L[c], values: voc.weekly.counts.map(w => w[i]) })), colors });
  const curve = lineChart({ series: [{ name: "Overcharge complaints", color: V("s2"), values: voc.overcharge_vs_price.map(p => p.per_10k_parked_hours), dots: true }],
    yMax: Math.max(...voc.overcharge_vs_price.map(p => p.per_10k_parked_hours)) * 1.15, yTicks: [0, 1, 2, 3, 4, 5], yFormat: v => v.toFixed(0), height: 250, width: 460,
    xLabel: i => ["×0.8", "×0.9", "×1.0", "×1.1", "×1.25"][i] || "", tipRows: i => [`Price ${["×0.8", "×0.9", "×1.0", "×1.1", "×1.25"][i]}: <b>${voc.overcharge_vs_price[i].per_10k_parked_hours.toFixed(1)}</b> per 10k parked hours`], ariaLabel: "Overcharge complaints by experiment price" });
  const m = t.model, b = t.keyword_baseline;
  const rows = cats.map(c => [L[c], fmt.pct(m.per_class[c].precision), fmt.pct(m.per_class[c].recall), fmt.pct(m.per_class[c].f1), fmt.pct(b.per_class[c].f1)]);
  const zones = voc.by_zone.map(z => { const top = cats.filter(c => c !== "other").sort((a, c2) => (z.current[c2] - z.previous[c2]) - (z.current[a] - z.previous[a]))[0]; return { z, top }; });
  fill(view, pageHead("Customer voice", "Support tickets sorted into causes, so price changes are judged on complaints as well as revenue."),
    h("div", { class: "kpis" },
      kpi("Ticket classifier macro-F1", m.macro_f1.toFixed(2), `vs ${b.macro_f1.toFixed(2)} for keyword rules`),
      kpi("Accuracy on the ticket stream", fmt.pct(t.stream_accuracy_vs_true_category), "against the cause that generated each ticket"),
      kpi("Overcharge complaints at ×1.25", (voc.overcharge_vs_price.at(-1).per_10k_parked_hours / voc.overcharge_vs_price[2].per_10k_parked_hours).toFixed(1) + "×", "compared with an unchanged price")),
    panel("Tickets per week by cause", "Predicted category, whole network.", cols, legend(cats.map((c, i) => ({ name: L[c], color: colors[i] })))),
    h("div", { class: "grid2" },
      panel("Overcharge complaints rise with price", "Per 10,000 parked hours on randomised price-test days. This curve feeds the complaint projection in Pricing studio.", curve),
      panel("Classifier quality", `${t.n_test} held-out tickets. ${t.label_noise_note} The ticket text comes from a small set of templates, so real tickets would score lower.`, table(["Cause", "Precision", "Recall", "F1", "Keyword F1"], rows))),
    panel("Where complaints are moving", "Change over the last 28 days against the 28 before, per 10,000 parked hours.", table(["Zone", "Fastest-growing cause", "Now", "Before"], zones.map(({ z, top }) => [z.zone, L[top], z.current[top].toFixed(1), z.previous[top].toFixed(1)]), 2)),
    h("div", { class: "grid2" },
      panel("Confusion matrix", "Rows are the true label, columns the prediction.", confusion({ labels: t.confusion.labels, matrix: t.confusion.matrix })),
      panel("Latest tickets", null, h("ul", { class: "tickets" }, voc.recent.slice(0, 14).map(k => h("li", {}, h("div", {}, k.text), h("div", { class: "small muted" }, `${zname(k.zone_id)} · `, h("span", { class: "chip" }, L[k.pred]), ` ${fmt.pct(k.confidence)} sure`)))))));
}

function aboutView() {
  const d = state.report.data;
  fill(view, pageHead("How it works", "One pipeline from raw parking logs to a price you can defend."),
    panel("The pipeline", null, h("ul", { class: "plain" },
      h("li", {}, h("b", {}, "Simulate. "), `A ${d.zones}-zone city over ${dlabel(d.start)} to ${dlabel(d.end)} with weather, events, holidays, sensor outages and capacity-capped occupancy. Known ground truth lets every stage be graded.`),
      h("li", {}, h("b", {}, "Estimate price sensitivity. "), `Randomised price tests on ${fmt.pct(d.experiment_share)} of zone-days identify how demand responds to price, using drivers turned away to see demand a full lot hides.`),
      h("li", {}, h("b", {}, "Forecast. "), "Gradient boosting predicts demand at the base price one day ahead, with calibrated 80% ranges. Features use only information available the evening before."),
      h("li", {}, h("b", {}, "Price. "), "Dynamic programming picks hourly multipliers per zone under guardrails on price level, price changes and daily average, trading revenue against turned-away and priced-out drivers."),
      h("li", {}, h("b", {}, "Listen. "), "A text classifier turns tickets into causes, and a fitted complaint curve projects the customer cost of each plan."))),
    panel("What to be careful about", null, h("ul", { class: "plain" },
      h("li", {}, "Results come from a simulator. Real data would add noise, reporting gaps and behaviours not modelled here."),
      h("li", {}, "The pricing replay is an expected-value simulation, not an A/B test. A real launch would start with a few zones and a holdout."),
      h("li", {}, "Spillover to nearby lots is assumed, not learned. The dashboard shows how little it changes the result."),
      h("li", {}, "Cold-start forecasts only help when the new lot resembles existing ones."))));
}

// ---------------------------------------------------------------------------------------------
const routes = { overview: overviewView, forecast: forecastView, pricing: pricingView, health: healthView, voice: voiceView, about: aboutView };
async function route() {
  hideTip();
  const name = (location.hash.replace(/^#\//, "") || "overview");
  const fn = routes[name] || overviewView;
  document.querySelectorAll(".nav a").forEach(a => a.getAttribute("href") === `#/${name}` ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
  try { await fn(); } catch (e) { if (!view.querySelector(".callout")) fill(view, h("div", { class: "callout" }, "This page didn't load. Check the server log, then reload.")); console.error(e); }
  view.focus({ preventScroll: true });
}
document.getElementById("theme").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = next;
  try { localStorage.setItem("pp-theme", next); } catch {}
});
try { const t = localStorage.getItem("pp-theme"); if (t) document.documentElement.dataset.theme = t; } catch {}

(async () => {
  try {
    [state.meta, state.report] = await Promise.all([api("/api/meta"), api("/api/report")]);
  } catch { fill(view, h("div", { class: "callout" }, "The API isn't ready. Run `python -m parkwise build`, then restart the server.")); return; }
  state.date = state.meta.live_date;
  document.getElementById("rail-data").textContent = "Synthetic data · " + state.meta.data.zones + " zones";
  addEventListener("hashchange", route);
  route();
})();
