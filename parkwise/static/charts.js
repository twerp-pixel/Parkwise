// Tiny dependency-free SVG chart kit. All colours come from CSS variables so light/dark just works.
const NS = "http://www.w3.org/2000/svg";

export function h(tag, attrs = {}, ...kids) {
  const svg = ["svg", "g", "path", "line", "rect", "circle", "text", "polyline", "title"].includes(tag) && attrs.__svg !== false;
  const n = svg ? document.createElementNS(NS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "__svg" || v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "class") n.setAttribute("class", v);
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false) n.append(kid.nodeType ? kid : document.createTextNode(kid));
  return n;
}

// ---- tooltip --------------------------------------------------------------------------------
let tipEl;
function tip() { return tipEl || (tipEl = document.body.appendChild(h("div", { class: "tip", role: "status" }))); }
export function showTip(html, ev) {
  const t = tip(); t.innerHTML = html; t.style.opacity = 1;
  const pad = 14, w = t.offsetWidth, hh = t.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + hh > innerHeight - 8) y = ev.clientY - hh - pad;
  t.style.left = x + "px"; t.style.top = y + "px";
}
export function hideTip() { if (tipEl) tipEl.style.opacity = 0; }

// ---- formatting -----------------------------------------------------------------------------
export const fmt = {
  pct: (v, d = 0) => (v === null || v === undefined || isNaN(v)) ? "–" : (v * 100).toFixed(d) + "%",
  spct: (v, d = 1) => (v === null || v === undefined || isNaN(v)) ? "–" : (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d) + "%",
  inr: (v) => "₹" + (Math.abs(v) >= 1e7 ? (v / 1e7).toFixed(2) + " Cr" : Math.abs(v) >= 1e5 ? (v / 1e5).toFixed(2) + " L" : Math.round(v).toLocaleString("en-IN")),
  int: (v) => Math.round(v).toLocaleString("en-IN"),
  hour: (hh) => String(hh).padStart(2, "0") + ":00",
  mult: (v) => "×" + v.toFixed(2),
};

const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / ((d1 - d0) || 1)) * (r1 - r0);

function axisY(g, y, W, left, right, ticks, format) {
  for (const t of ticks) {
    g.append(h("line", { x1: left, x2: W - right, y1: y(t), y2: y(t), stroke: "var(--line)", "stroke-width": 1 }));
    g.append(h("text", { x: left - 8, y: y(t) + 4, "text-anchor": "end" }, format(t)));
  }
}

// ---- line chart over hours with optional band ----------------------------------------------
export function lineChart({ series, band, yMin = 0, yMax = 1, yTicks = [0, .25, .5, .75, 1], yFormat = v => fmt.pct(v), refLines = [],
  height = 250, width = 760, xLabel = h => (h % 3 === 0 ? fmt.hour(h) : ""), tipRows, ariaLabel = "Line chart" }) {
  const W = width, H = height, L = 46, R = 12, T = 12, B = 26, n = series[0].values.length;
  const x = scale(0, n - 1, L, W - R), y = scale(yMin, yMax, H - B, T);
  const svg = h("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": ariaLabel });
  const g = h("g"); svg.append(g);
  axisY(g, y, W, L, R, yTicks, yFormat);
  for (let i = 0; i < n; i++) { const l = xLabel(i); if (l) g.append(h("text", { x: x(i), y: H - 6, "text-anchor": "middle" }, l)); }
  for (const r of refLines) {
    g.append(h("line", { x1: L, x2: W - R, y1: y(r.y), y2: y(r.y), stroke: r.color || "var(--bad)", "stroke-dasharray": "5 4", "stroke-width": 1.2 }));
    if (r.label) g.append(h("text", { x: W - R, y: y(r.y) - 5, "text-anchor": "end", fill: r.color || "var(--bad)" }, r.label));
  }
  if (band) {
    const up = band.hi.map((v, i) => `${x(i)},${y(v)}`), dn = band.lo.map((v, i) => `${x(i)},${y(v)}`).reverse();
    g.append(h("path", { d: "M" + up.concat(dn).join("L") + "Z", fill: band.color || "var(--s1)", opacity: .16 }));
  }
  for (const s of series) {
    const pts = s.values.map((v, i) => (v === null || v === undefined) ? null : [x(i), y(v)]);
    let d = "", pen = false;
    for (const p of pts) { if (!p) { pen = false; continue; } d += (pen ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1); pen = true; }
    g.append(h("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2.2, "stroke-dasharray": s.dash || null, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    if (s.dots) pts.forEach(p => p && g.append(h("circle", { cx: p[0], cy: p[1], r: 2.6, fill: s.color })));
  }
  const cursor = h("line", { y1: T, y2: H - B, stroke: "var(--muted)", "stroke-width": 1, opacity: 0 });
  g.append(cursor);
  const hit = h("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" });
  const move = (ev) => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
    const i = Math.max(0, Math.min(n - 1, Math.round((px - L) / ((W - L - R) / (n - 1)))));
    cursor.setAttribute("x1", x(i)); cursor.setAttribute("x2", x(i)); cursor.setAttribute("opacity", .6);
    const rows = tipRows ? tipRows(i) : series.map(s => `<span style="color:${s.color}">●</span> ${s.name}: <b>${yFormat(s.values[i])}</b>`);
    showTip(`<b>${fmt.hour(i)}</b><br>${rows.join("<br>")}`, ev);
  };
  hit.addEventListener("mousemove", move);
  hit.addEventListener("mouseleave", () => { hideTip(); cursor.setAttribute("opacity", 0); });
  g.append(hit);
  return svg;
}

export function legend(items) {
  return h("div", { class: "legend" }, items.map(i => h("span", {}, h("i", { class: i.line ? "line" : "", style: `background:${i.color}` }), i.name)));
}

// ---- heatmap (zones x hours) -----------------------------------------------------------------
export function heatmap({ rows, values, cell = 28, labelW = 176, onTip }) {
  const n = values[0].length, T = 22, rowH = 26, W = labelW + n * cell + 6, H = T + rows.length * rowH + 4;
  const svg = h("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Expected occupancy by zone and hour" });
  for (let c = 0; c < n; c += 3) svg.append(h("text", { x: labelW + c * cell + cell / 2, y: 13, "text-anchor": "middle" }, String(c).padStart(2, "0")));
  rows.forEach((name, r) => {
    svg.append(h("text", { x: labelW - 10, y: T + r * rowH + rowH / 2 + 4, "text-anchor": "end" }, name));
    values[r].forEach((v, c) => {
      const full = v >= 0.95;
      const rect = h("rect", { x: labelW + c * cell + 1, y: T + r * rowH + 1, width: cell - 2, height: rowH - 2, rx: 3,
        style: `fill: color-mix(in srgb, var(--bay) ${Math.round(8 + Math.min(1, v) * 92)}%, var(--surface))`,
        stroke: full ? "var(--amber)" : "none", "stroke-width": full ? 2 : 0 });
      rect.addEventListener("mousemove", ev => showTip(onTip ? onTip(r, c, v) : `<b>${name}</b> ${fmt.hour(c)}<br>${fmt.pct(v)} full`, ev));
      rect.addEventListener("mouseleave", hideTip);
      svg.append(rect);
    });
  });
  return svg;
}

// ---- horizontal grouped bars -----------------------------------------------------------------
export function hBars({ items, format = v => v.toFixed(2), max, labelW = 190, rowH = 30, ariaLabel = "Bar chart" }) {
  const W = 760, R = 70, groups = Math.max(...items.map(i => i.values.length));
  const m = max || Math.max(...items.flatMap(i => i.values.map(v => Math.abs(v.v)))) * 1.05 || 1;
  const bh = Math.min(14, (rowH - 8) / groups), H = items.length * rowH + 6;
  const x = scale(0, m, labelW, W - R);
  const svg = h("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": ariaLabel });
  items.forEach((it, r) => {
    const y0 = 3 + r * rowH;
    svg.append(h("text", { x: labelW - 10, y: y0 + rowH / 2 + 3, "text-anchor": "end" }, it.label));
    it.values.forEach((v, k) => {
      const top = y0 + (rowH - groups * bh) / 2 + k * bh;
      const rect = h("rect", { x: labelW, y: top, width: Math.max(1, x(Math.abs(v.v)) - labelW), height: bh - 2, rx: 2, fill: v.color });
      rect.addEventListener("mousemove", ev => showTip(`<b>${it.label}</b><br>${v.name}: ${format(v.v)}`, ev));
      rect.addEventListener("mouseleave", hideTip);
      svg.append(rect, h("text", { x: x(Math.abs(v.v)) + 5, y: top + bh - 3 }, format(v.v)));
    });
  });
  return svg;
}

// ---- estimate +/- CI against a reference (elasticity) ----------------------------------------
export function dotPlot({ items, xMin, xMax, labelW = 190, rowH = 30 }) {
  const W = 760, R = 20, T = 6, B = 24, H = T + items.length * rowH + B;
  const x = scale(xMin, xMax, labelW, W - R);
  const svg = h("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Estimated versus true price sensitivity" });
  for (let t = Math.ceil(xMin * 4) / 4; t <= xMax + 1e-9; t += .25) {
    svg.append(h("line", { x1: x(t), x2: x(t), y1: T, y2: H - B, stroke: "var(--line)" }), h("text", { x: x(t), y: H - 6, "text-anchor": "middle" }, t.toFixed(2)));
  }
  items.forEach((it, r) => {
    const cy = T + r * rowH + rowH / 2;
    svg.append(h("text", { x: labelW - 10, y: cy + 4, "text-anchor": "end" }, it.label));
    svg.append(h("line", { x1: x(it.lo), x2: x(it.hi), y1: cy, y2: cy, stroke: "var(--s1)", "stroke-width": 3, opacity: .45, "stroke-linecap": "round" }));
    svg.append(h("circle", { cx: x(it.est), cy, r: 5, fill: "var(--s1)" }));
    if (it.naive !== undefined) svg.append(h("path", { d: `M${x(it.naive)},${cy - 5}l5,5l-5,5l-5,-5z`, fill: "var(--s2)" }));
    if (it.truth !== undefined) svg.append(h("line", { x1: x(it.truth), x2: x(it.truth), y1: cy - 9, y2: cy + 9, stroke: "var(--ink)", "stroke-width": 2 }));
    const hit = h("rect", { x: labelW, y: cy - rowH / 2, width: W - labelW - R, height: rowH, fill: "transparent" });
    hit.addEventListener("mousemove", ev => showTip(`<b>${it.label}</b><br>Estimate ${it.est.toFixed(2)} (${it.lo.toFixed(2)} to ${it.hi.toFixed(2)})` +
      (it.truth !== undefined ? `<br>True value ${it.truth.toFixed(2)}` : "") + (it.naive !== undefined ? `<br>Naive (censored) ${it.naive.toFixed(2)}` : ""), ev));
    hit.addEventListener("mouseleave", hideTip);
    svg.append(hit);
  });
  return svg;
}

// ---- stacked columns (weekly tickets) --------------------------------------------------------
export function stackedColumns({ labels, series, height = 230, colors }) {
  const W = 760, L = 44, R = 8, T = 10, B = 24, H = height, n = labels.length;
  const tot = labels.map((_, i) => series.reduce((s, k) => s + k.values[i], 0));
  const max = Math.max(...tot) * 1.08, y = scale(0, max, H - B, T), bw = (W - L - R) / n;
  const svg = h("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Weekly tickets by category" });
  axisY(svg, y, W, L, R, [0, .5, 1].map(f => Math.round(max * f)), v => String(v));
  labels.forEach((lb, i) => {
    let acc = 0;
    series.forEach((s, k) => {
      const v = s.values[i], top = y(acc + v), r = h("rect", { x: L + i * bw + 1, y: top, width: Math.max(1, bw - 2), height: y(acc) - top, fill: colors[k] });
      r.addEventListener("mousemove", ev => showTip(`<b>Week of ${lb}</b><br>${s.name}: ${v}`, ev));
      r.addEventListener("mouseleave", hideTip);
      svg.append(r); acc += v;
    });
    if (i % Math.ceil(n / 8) === 0) svg.append(h("text", { x: L + i * bw + bw / 2, y: H - 6, "text-anchor": "middle" }, lb.slice(5)));
  });
  return svg;
}

// ---- confusion matrix ------------------------------------------------------------------------
const SHORT_CAT = { no_space: "No space", entry_delay: "Entry/exit", overcharge: "Overcharge", payment_failed: "Payment", safety: "Safety", signage: "Signage", other: "Other" };
export function confusion({ labels, matrix }) {
  const c = 52, L = 98, T = 28, rh = 32, W = L + labels.length * c + 6, H = T + labels.length * rh + 6;
  const svg = h("svg", { class: "chart cm", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Confusion matrix, rows are true labels" });
  matrix.forEach((row, r) => {
    const s = row.reduce((a, b) => a + b, 0) || 1;
    svg.append(h("text", { x: L - 8, y: T + r * rh + 21, "text-anchor": "end" }, SHORT_CAT[labels[r]]));
    row.forEach((v, k) => {
      const f = v / s;
      svg.append(h("rect", { x: L + k * c + 1, y: T + r * rh + 1, width: c - 2, height: rh - 2, rx: 3,
        style: `fill: color-mix(in srgb, var(--bay) ${Math.round(6 + f * 94)}%, var(--surface))` }));
      svg.append(h("text", { x: L + k * c + c / 2, y: T + r * rh + 21, "text-anchor": "middle", style: f > .55 ? "fill:#fff" : "" }, v));
    });
  });
  labels.forEach((l, k) => svg.append(h("text", { x: L + k * c + c / 2, y: 17, "text-anchor": "middle" }, SHORT_CAT[l])));
  return svg;
}
