"""Build the lego-plot playground HTML.

Embeds the cross-locus aggregate CSVs as inline JSON and writes a
self-contained dark-themed HTML file with controls, SVG preview, and a
copy-able prompt.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "supplement" / "lego_playground.html"


# The playground only consumes per-strand frac_* columns
# (rectangle = *_fwd, circle = *_rc).  Combined-strand frac_P/T/M/D are
# unused; dropping them roughly halves the embedded JSON payload.
_FRAC_COLS = [f"frac_{s}_{strand}" for strand in ("fwd", "rc")
              for s in ("P", "T", "M", "D", "score_eq_truth")]

# Compact column-store: emit { cols: [...], rows: [[...], ...] } so we
# don't repeat column names per row.  Drops file size by ~5x for the
# wide frac tables.
def _pack(df: pd.DataFrame, key_cols: list[str]) -> dict:
    cols = key_cols + [c for c in _FRAC_COLS if c in df.columns]
    df = df[cols].copy()
    for c in _FRAC_COLS:
        if c in df.columns:
            df[c] = df[c].round(3)
    return {
        "cols": cols,
        "rows": df.values.tolist(),
    }


def _load_single() -> dict:
    df = pd.read_csv(REPO_ROOT / "supplement/data/single_repeat_cross_locus_aggregate.csv")
    return _pack(df, ["N", "snv_offset", "motif_len", "arm",
                      "delta", "lflank", "n_loci"])


def _load_compound() -> dict:
    df = pd.read_csv(REPO_ROOT / "supplement/data/compound_cross_locus_aggregate.csv")
    return _pack(df, ["N1", "N2", "bridge_len",
                      "motif1_len", "motif2_len", "arm",
                      "delta1", "delta2", "n_loci"])


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NW-flex vs BWA-MEM — lego plot explorer</title>
<style>
  :root {
    --bg: #1a1d23;
    --panel: #23272e;
    --panel-2: #2a2f37;
    --text: #d8dde5;
    --text-dim: #8a92a0;
    --accent: #6baed6;
    --border: #3a4048;
    --good: #08519c;
    --mid: #ffffbf;
    --bad: #c0392b;
  }
  html, body {
    margin: 0; padding: 0; height: 100%;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: var(--text); background: var(--bg);
    font-size: 14px;
  }
  .layout {
    display: grid;
    grid-template-columns: 360px 1fr;
    grid-template-rows: 1fr auto;
    grid-template-areas: "controls preview" "controls prompt";
    height: 100vh; gap: 0;
  }
  .controls { grid-area: controls; padding: 16px 18px; background: var(--panel);
              border-right: 1px solid var(--border); overflow-y: auto; }
  .preview  { grid-area: preview; padding: 16px; display: flex; flex-direction: column;
              align-items: center; justify-content: flex-start; overflow: auto; }
  .prompt   { grid-area: prompt; padding: 12px 16px; background: var(--panel-2);
              border-top: 1px solid var(--border); display: flex; gap: 12px;
              align-items: flex-start; }
  h1 { font-size: 16px; margin: 0 0 6px 0; color: var(--accent); font-weight: 600; }
  h2 { font-size: 12px; margin: 18px 0 8px 0; color: var(--text-dim);
       text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
  h2:first-of-type { margin-top: 8px; }
  label { display: flex; align-items: center; gap: 8px; margin: 6px 0;
          font-size: 13px; cursor: pointer; }
  label.col { flex-direction: column; align-items: stretch; gap: 4px; }
  select, button, input[type="number"] {
    background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 5px 8px; font-family: inherit; font-size: 13px;
  }
  select { flex: 1; }
  button { cursor: pointer; }
  button:hover { background: var(--border); }
  button.preset { background: var(--panel-2); margin: 3px 4px 3px 0;
                  font-size: 12px; padding: 4px 10px; }
  button.preset:hover { background: var(--accent); color: var(--bg); }
  .radio-group { display: flex; gap: 14px; }
  .radio-group label { margin: 0; }
  .filter-row { display: grid; grid-template-columns: 90px 1fr;
                gap: 8px; align-items: center; margin: 5px 0; }
  .filter-row > span { font-size: 12px; color: var(--text-dim); }
  .filter-row > .multiselect {
    display: flex; flex-wrap: wrap; gap: 3px; max-width: 100%;
  }
  .chip {
    background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 12px; padding: 2px 9px; font-size: 11px;
    color: var(--text-dim); cursor: pointer; user-select: none;
    font-family: ui-monospace, monospace;
  }
  .chip.on { background: var(--accent); color: var(--bg); border-color: var(--accent); }
  .chip:hover:not(.on) { color: var(--text); border-color: var(--text-dim); }
  .legend { display: flex; align-items: center; gap: 10px;
            margin: 12px 0 4px 0; font-size: 12px; color: var(--text-dim); }
  .glyph { width: 28px; height: 28px; position: relative;
           border: 1px solid var(--text-dim); }
  .glyph-circle { position: absolute; left: 50%; top: 50%;
                  width: 14px; height: 14px; border-radius: 50%;
                  transform: translate(-50%, -50%);
                  border: 1px solid #222; background: var(--good); }
  .colorbar { display: flex; align-items: center; gap: 8px;
              font-size: 11px; color: var(--text-dim);
              font-family: ui-monospace, monospace; }
  .colorbar .bar { width: 240px; height: 12px;
                   background: linear-gradient(to right,
                     #c0392b 0%, #ffffbf 50%, #08519c 100%);
                   border: 1px solid var(--border); }
  #plot-host { width: 100%; max-width: 900px; }
  #plot { width: 100%; display: block; }
  .axis-label { fill: var(--text); font-family: ui-monospace, monospace;
                font-size: 12px; }
  .axis-tick { fill: var(--text-dim); font-family: ui-monospace, monospace;
               font-size: 11px; }
  .grid-line { stroke: var(--border); stroke-width: 0.5; }
  .zero-frame { stroke: var(--text); stroke-width: 1.6; fill: none; }
  .arm-label { fill: var(--text); font-size: 13px; font-weight: 600;
               font-family: -apple-system, sans-serif; }
  .meta { font-size: 11px; color: var(--text-dim);
          font-family: ui-monospace, monospace; margin-top: 8px; }
  pre#prompt-text {
    flex: 1; margin: 0; white-space: pre-wrap;
    font-family: ui-monospace, "Menlo", "Consolas", monospace;
    font-size: 12px; color: var(--text); line-height: 1.5;
    max-height: 140px; overflow-y: auto;
  }
  #copy-btn {
    flex-shrink: 0; padding: 6px 14px;
    background: var(--accent); color: var(--bg); border: none;
    font-weight: 600;
  }
  #copy-btn:hover { opacity: 0.9; }
  #copy-btn.copied { background: var(--good); color: var(--text); }
  .empty { text-align: center; color: var(--text-dim);
           padding: 80px 20px; font-style: italic; }
</style>
</head>
<body>
<div class="layout">

  <aside class="controls">
    <h1>Lego plot explorer</h1>

    <h2>Dataset</h2>
    <div class="radio-group">
      <label><input type="radio" name="dataset" value="single" checked> Single-repeat</label>
      <label><input type="radio" name="dataset" value="compound"> Compound</label>
    </div>

    <h2>Axes</h2>
    <label class="col">X axis<select id="x-axis"></select></label>
    <label class="col">Y axis<select id="y-axis"></select></label>

    <h2>Glyph metric (rectangle = fwd, circle = rc)</h2>
    <label class="col">Metric<select id="metric">
      <option value="frac_score_eq_truth" selected>frac score = truth (P+T)</option>
      <option value="frac_P">frac P (length+score correct)</option>
      <option value="frac_T">frac T (score = truth, length wrong)</option>
      <option value="frac_M">frac M (score &lt; truth, heuristic miss)</option>
      <option value="frac_D">frac D (score &gt; truth, dominated)</option>
    </select></label>

    <h2>Filters</h2>
    <div id="filters"></div>

    <h2>Faceting</h2>
    <label class="col">Facet columns (one panel per value, left → right)
      <select id="facet-col"></select>
    </label>
    <label class="col">Facet rows (one panel per value, top → bottom)
      <select id="facet-row"></select>
    </label>

    <h2>View</h2>
    <label class="col">Cell size <span id="cell-size-val" style="float:right; color:var(--text-dim); font-family: ui-monospace, monospace;">22 px</span>
      <input type="range" id="cell-size" min="10" max="60" step="2" value="22" style="width: 100%;">
    </label>
    <div style="display:flex; gap:6px; margin-top:8px;">
      <button id="save-svg" style="flex:1;">Save SVG</button>
      <button id="save-png" style="flex:1;">Save PNG</button>
    </div>

    <h2>Presets</h2>
    <div id="presets"></div>

    <h2>Legend</h2>
    <div class="legend">
      <div class="glyph"><div class="glyph-circle"></div></div>
      <div>rectangle = first metric · circle = second metric</div>
    </div>
    <div class="colorbar">
      <span>0.0</span><div class="bar"></div><span>1.0</span>
    </div>
    <div class="meta" id="meta"></div>
  </aside>

  <main class="preview" id="plot-host">
    <svg id="plot" viewBox="0 0 1000 400" preserveAspectRatio="xMidYMid meet"></svg>
  </main>

  <section class="prompt">
    <pre id="prompt-text"></pre>
    <button id="copy-btn">Copy Prompt</button>
  </section>

</div>

<script>
const PACKED = {
  single: __SINGLE_DATA__,
  compound: __COMPOUND_DATA__,
};

// Unpack column-store to a list of row dicts, lazily once per dataset.
const DATA = {};
function ensureUnpacked(kind) {
  if (DATA[kind]) return;
  const p = PACKED[kind];
  const cols = p.cols;
  DATA[kind] = p.rows.map(r => {
    const o = {};
    for (let i = 0; i < cols.length; i++) o[cols[i]] = r[i];
    return o;
  });
}
ensureUnpacked("single");

const AXES = {
  single: ["arm", "delta", "lflank", "N", "snv_offset", "motif_len"],
  compound: ["arm", "delta1", "delta2", "bridge_len", "N1", "N2",
             "motif1_len", "motif2_len"],
};

const AXIS_LABELS = {
  arm: "aligner arm",
  delta: "Δ (hap N − ref N)",
  lflank: "lflank extent",
  N: "ref repeat count N",
  snv_offset: "SNV offset (−1 = no SNV)",
  motif_len: "motif length (bp)",
  delta1: "Δ₁ (block 1)",
  delta2: "Δ₂ (block 2)",
  bridge_len: "bridge length |M| (bp)",
  N1: "block-1 ref count N₁",
  N2: "block-2 ref count N₂",
  motif1_len: "block-1 motif length (bp)",
  motif2_len: "block-2 motif length (bp)",
};

const ARM_ORDER = ["BWA-std", "BWA-no-clip", "NW-flex"];
function sortValues(col, vals) {
  if (col === "arm") {
    return vals.slice().sort((a, b) => ARM_ORDER.indexOf(a) - ARM_ORDER.indexOf(b));
  }
  return vals.slice().sort((a, b) => (a > b) - (a < b));
}

const PRESETS = {
  single: [
    { name: "NB7 (arms)",     x: "delta", y: "lflank", facetCol: "arm", facetRow: "",
      filters: { N: [10], snv_offset: [-1], motif_len: [1, 2, 3] } },
    { name: "Boundary SNV",   x: "delta", y: "lflank", facetCol: "arm", facetRow: "",
      filters: { N: [10], snv_offset: [0], motif_len: [1, 2, 3] } },
    { name: "arm × SNV",      x: "delta", y: "lflank", facetCol: "arm", facetRow: "snv_offset",
      filters: { N: [10], motif_len: [1, 2, 3] } },
    { name: "arm × motif_len",x: "delta", y: "lflank", facetCol: "arm", facetRow: "motif_len",
      filters: { N: [10], snv_offset: [-1] } },
    { name: "arm × N",        x: "delta", y: "lflank", facetCol: "arm", facetRow: "N",
      filters: { snv_offset: [-1], motif_len: [1, 2, 3] } },
  ],
  compound: [
    { name: "NB9 main",       x: "delta1", y: "delta2", facetCol: "arm", facetRow: "",
      filters: { N1: [10], N2: [10], bridge_len: [2],
                 motif1_len: [1, 2, 3], motif2_len: [1, 2, 3] } },
    { name: "arm × bridge",   x: "delta1", y: "delta2", facetCol: "arm", facetRow: "bridge_len",
      filters: { N1: [10], N2: [10],
                 motif1_len: [1, 2, 3], motif2_len: [1, 2, 3] } },
    { name: "motif1 × motif2",x: "delta1", y: "delta2", facetCol: "motif1_len", facetRow: "motif2_len",
      filters: { N1: [10], N2: [10], bridge_len: [2], arm: ["BWA-no-clip"] } },
    { name: "bridge × N1",    x: "delta1", y: "delta2", facetCol: "bridge_len", facetRow: "N1",
      filters: { N2: [10],
                 motif1_len: [1, 2, 3], motif2_len: [1, 2, 3] } },
  ],
};

const state = {
  dataset: "single",
  x: "delta",
  y: "lflank",
  facetCol: "arm",
  facetRow: "",
  metric: "frac_score_eq_truth",
  cellSize: 22,
  filters: {},
};

function reservedCols() {
  return [state.x, state.y, state.facetCol, state.facetRow]
    .filter(c => c !== "" && c != null);
}

// ---------- color ----------
const NAN_COLOR = "#777";
const STOPS = [
  [0.0, [192,  57,  43]],
  [0.5, [255, 255, 191]],
  [1.0, [  8,  81, 156]],
];
function colorAt(v) {
  if (v == null || isNaN(v)) return NAN_COLOR;
  v = Math.max(0, Math.min(1, v));
  let i = 0;
  while (i < STOPS.length - 1 && v > STOPS[i+1][0]) i++;
  const [s, e] = [STOPS[i], STOPS[i+1]];
  const t = (v - s[0]) / (e[0] - s[0]);
  const c = [0,1,2].map(j => Math.round(s[1][j] + t * (e[1][j] - s[1][j])));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ---------- data helpers ----------
function currentData() { ensureUnpacked(state.dataset); return DATA[state.dataset]; }
function currentAxes() { return AXES[state.dataset]; }
function distinctValues(col) {
  const s = new Set(currentData().map(r => r[col]));
  return sortValues(col, [...s]);
}
function filterCols() {
  const used = new Set(reservedCols());
  return currentAxes().filter(c => !used.has(c));
}
function filterData(rows, exclude = []) {
  return rows.filter(r => {
    for (const col of filterCols()) {
      if (exclude.includes(col)) continue;
      const allowed = state.filters[col];
      if (!allowed || allowed.length === 0) return false;
      if (!allowed.includes(r[col])) return false;
    }
    return true;
  });
}

// ---------- controls ----------
function rebuildAxisSelects() {
  const axes = currentAxes();
  const pickers = [
    { id: "x-axis",     key: "x",        nullOpt: false },
    { id: "y-axis",     key: "y",        nullOpt: false },
    { id: "facet-col",  key: "facetCol", nullOpt: true },
    { id: "facet-row",  key: "facetRow", nullOpt: true },
  ];
  pickers.forEach(p => {
    const sel = document.getElementById(p.id);
    const cur = state[p.key];
    sel.innerHTML = "";
    if (p.nullOpt) {
      const o = document.createElement("option");
      o.value = ""; o.textContent = "(none)";
      if (cur === "" || cur == null) o.selected = true;
      sel.appendChild(o);
    }
    axes.forEach(a => {
      const o = document.createElement("option");
      o.value = a; o.textContent = a;
      if (a === cur) o.selected = true;
      sel.appendChild(o);
    });
  });
}
function rebuildFilters() {
  const host = document.getElementById("filters");
  host.innerHTML = "";
  filterCols().forEach(col => {
    const vals = distinctValues(col);
    if (!(col in state.filters)) {
      // default: pick a sensible single value (median of nums, first otherwise)
      const numeric = vals.every(v => typeof v === "number");
      const def = numeric ? vals[Math.floor(vals.length / 2)] : vals[0];
      state.filters[col] = [def];
    }
    const row = document.createElement("div");
    row.className = "filter-row";
    const label = document.createElement("span"); label.textContent = col;
    const multi = document.createElement("div"); multi.className = "multiselect";
    vals.forEach(v => {
      const chip = document.createElement("span");
      chip.className = "chip" + (state.filters[col].includes(v) ? " on" : "");
      chip.textContent = String(v);
      chip.onclick = () => {
        const cur = new Set(state.filters[col]);
        if (cur.has(v)) cur.delete(v); else cur.add(v);
        if (cur.size === 0) cur.add(v);  // keep at least one
        state.filters[col] = [...cur].sort((a, b) => (a > b) - (a < b));
        chip.classList.toggle("on");
        // re-toggle others without rebuilding (faster)
        renderAll();
      };
      multi.appendChild(chip);
    });
    row.appendChild(label); row.appendChild(multi);
    host.appendChild(row);
  });
}
function rebuildPresets() {
  const host = document.getElementById("presets");
  host.innerHTML = "";
  PRESETS[state.dataset].forEach(p => {
    const b = document.createElement("button");
    b.className = "preset"; b.textContent = p.name;
    b.onclick = () => applyPreset(p);
    host.appendChild(b);
  });
}
function applyPreset(p) {
  state.x = p.x; state.y = p.y;
  state.facetCol = (p.facetCol == null) ? "" : p.facetCol;
  state.facetRow = (p.facetRow == null) ? "" : p.facetRow;
  state.filters = {};
  Object.entries(p.filters || {}).forEach(([k, v]) => { state.filters[k] = [...v]; });
  rebuildAxisSelects();
  rebuildFilters();
  renderAll();
}

// ---------- render ----------
function renderAll() {
  renderPreview();
  updatePrompt();
}

function renderPreview() {
  const svg = document.getElementById("plot");
  svg.innerHTML = "";
  const ns = "http://www.w3.org/2000/svg";
  const xs = distinctValues(state.x);
  const ys = distinctValues(state.y);
  const filtered = filterData(currentData());
  if (filtered.length === 0) {
    svg.setAttribute("viewBox", "0 0 600 200");
    const t = document.createElementNS(ns, "text");
    t.setAttribute("x", 300); t.setAttribute("y", 100);
    t.setAttribute("text-anchor", "middle"); t.setAttribute("class", "axis-label");
    t.textContent = "no rows match these filters";
    svg.appendChild(t);
    return;
  }

  // Facet value lists; [null] sentinel means "no faceting on this axis"
  const colVals = state.facetCol
    ? sortValues(state.facetCol, [...new Set(filtered.map(r => r[state.facetCol]))])
    : [null];
  const rowVals = state.facetRow
    ? sortValues(state.facetRow, [...new Set(filtered.map(r => r[state.facetRow]))])
    : [null];

  // Sizing
  const cell = state.cellSize;
  const padL = 48, padR = 8, padT = 4, padB = 4;
  const panelW = padL + xs.length * cell + padR;
  const panelH = padT + ys.length * cell + padB;
  const colGap = 14, rowGap = 14;
  const colHeaderH = state.facetCol ? 22 : 6;
  const rowLabelW = state.facetRow ? 90 : 0;
  const xLabelBlock = 38;  // bottom strip for x-ticks + x-axis label
  const yLabelBlock = 18;  // left strip on first column for y-axis label

  const gridW = colVals.length * panelW + (colVals.length - 1) * colGap;
  const gridH = rowVals.length * panelH + (rowVals.length - 1) * rowGap;
  const W = rowLabelW + yLabelBlock + gridW + 8;
  const H = colHeaderH + gridH + xLabelBlock;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  // Column headers (facetCol labels)
  if (state.facetCol) {
    colVals.forEach((cv, ci) => {
      const x = rowLabelW + yLabelBlock + ci * (panelW + colGap) + panelW / 2;
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", x); t.setAttribute("y", 16);
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("class", "arm-label");
      t.textContent = `${state.facetCol} = ${cv}`;
      svg.appendChild(t);
    });
  }

  // Row headers (facetRow labels) + cells
  rowVals.forEach((rv, ri) => {
    const yBase = colHeaderH + ri * (panelH + rowGap);
    if (state.facetRow) {
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", 4);
      t.setAttribute("y", yBase + panelH / 2 + 4);
      t.setAttribute("text-anchor", "start");
      t.setAttribute("class", "arm-label");
      t.textContent = `${state.facetRow} = ${rv}`;
      svg.appendChild(t);
    }
    colVals.forEach((cv, ci) => {
      const xBase = rowLabelW + yLabelBlock + ci * (panelW + colGap);
      let sub = filtered;
      if (state.facetCol) sub = sub.filter(r => r[state.facetCol] === cv);
      if (state.facetRow) sub = sub.filter(r => r[state.facetRow] === rv);
      const isBottom = ri === rowVals.length - 1;
      const isLeft   = ci === 0;
      drawPanel(svg, xBase, yBase, padL, padT, cell, xs, ys, sub,
                { showXTicks: isBottom, showYTicks: isLeft });
    });
  });

  // Shared X-axis label (under bottom row)
  if (xs.length) {
    const xl = document.createElementNS(ns, "text");
    xl.setAttribute("x", rowLabelW + yLabelBlock + gridW / 2);
    xl.setAttribute("y", colHeaderH + gridH + 32);
    xl.setAttribute("text-anchor", "middle");
    xl.setAttribute("class", "axis-label");
    xl.textContent = AXIS_LABELS[state.x] || state.x;
    svg.appendChild(xl);
  }
  // Shared Y-axis label (left of first column)
  if (ys.length) {
    const yl = document.createElementNS(ns, "text");
    yl.setAttribute("class", "axis-label");
    yl.setAttribute("text-anchor", "middle");
    yl.setAttribute(
      "transform",
      `translate(${rowLabelW + 12}, ${colHeaderH + gridH / 2}) rotate(-90)`,
    );
    yl.textContent = AXIS_LABELS[state.y] || state.y;
    svg.appendChild(yl);
  }

  // Meta line
  const n_loci_vals = filtered.map(r => r.n_loci || 0);
  const n_min = Math.min(...n_loci_vals);
  const n_max = Math.max(...n_loci_vals);
  const colN = colVals.length, rowN = rowVals.length;
  document.getElementById("meta").textContent =
    `${rowN} × ${colN} panel(s) · ${filtered.length} cells · ` +
    `n_loci per cell: ${n_min}–${n_max}`;
}

function drawPanel(svg, x0, y0, padL, padT, cell, xs, ys, rows, opts) {
  const ns = "http://www.w3.org/2000/svg";
  const g = document.createElementNS(ns, "g");
  g.setAttribute("transform", `translate(${x0}, ${y0})`);
  svg.appendChild(g);

  // Aggregate rows by (x,y) — average if multiple match (e.g., when an axis was filtered to multiple values).
  const rectKey = state.metric + "_fwd";
  const circKey = state.metric + "_rc";
  const lookup = new Map();
  rows.forEach(r => {
    const key = `${r[state.x]}|${r[state.y]}`;
    if (!lookup.has(key)) lookup.set(key, { sumR: 0, sumC: 0, n: 0 });
    const e = lookup.get(key);
    const rv = r[rectKey];
    const cv = r[circKey];
    e.sumR += (rv == null ? NaN : rv);
    e.sumC += (cv == null ? NaN : cv);
    e.n += 1;
  });

  const xIdx = new Map(xs.map((v, i) => [v, i]));
  const yIdx = new Map(ys.map((v, i) => [v, i]));

  ys.forEach((yv, yi) => {
    xs.forEach((xv, xi) => {
      const key = `${xv}|${yv}`;
      const cellX = padL + xi * cell;
      const cellY = padT + (ys.length - 1 - yi) * cell;
      if (lookup.has(key)) {
        const e = lookup.get(key);
        const rectV = e.sumR / e.n;
        const circV = e.sumC / e.n;
        const rect = document.createElementNS(ns, "rect");
        rect.setAttribute("x", cellX); rect.setAttribute("y", cellY);
        rect.setAttribute("width", cell); rect.setAttribute("height", cell);
        rect.setAttribute("fill", colorAt(rectV));
        g.appendChild(rect);
        const circ = document.createElementNS(ns, "circle");
        circ.setAttribute("cx", cellX + cell/2);
        circ.setAttribute("cy", cellY + cell/2);
        circ.setAttribute("r", cell * 0.26);
        circ.setAttribute("fill", colorAt(circV));
        circ.setAttribute("stroke", "#222");
        circ.setAttribute("stroke-width", 0.6);
        const tip = document.createElementNS(ns, "title");
        tip.textContent =
          `${state.x}=${xv}, ${state.y}=${yv}` +
          `\\nfwd (${state.metric}_fwd): ${rectV.toFixed(3)}` +
          `\\nrc  (${state.metric}_rc):  ${circV.toFixed(3)}`;
        circ.appendChild(tip);
        g.appendChild(circ);
      } else {
        const rect = document.createElementNS(ns, "rect");
        rect.setAttribute("x", cellX); rect.setAttribute("y", cellY);
        rect.setAttribute("width", cell); rect.setAttribute("height", cell);
        rect.setAttribute("fill", NAN_COLOR);
        rect.setAttribute("opacity", 0.4);
        g.appendChild(rect);
      }
    });
  });

  if (opts.showYTicks) {
    ys.forEach((yv, yi) => {
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", padL - 4);
      t.setAttribute("y", padT + (ys.length - 1 - yi) * cell + cell/2 + 4);
      t.setAttribute("text-anchor", "end");
      t.setAttribute("class", "axis-tick");
      t.textContent = String(yv);
      g.appendChild(t);
    });
  }
  if (opts.showXTicks) {
    xs.forEach((xv, xi) => {
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", padL + xi * cell + cell/2);
      t.setAttribute("y", padT + ys.length * cell + 14);
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("class", "axis-tick");
      t.textContent = String(xv);
      g.appendChild(t);
    });
  }

  function isDeltaAxis(c) {
    return c === "delta" || c === "delta1" || c === "delta2";
  }
  if (isDeltaAxis(state.x) && xIdx.has(0)) {
    const xi = xIdx.get(0);
    const r = document.createElementNS(ns, "rect");
    r.setAttribute("x", padL + xi * cell);
    r.setAttribute("y", padT);
    r.setAttribute("width", cell);
    r.setAttribute("height", ys.length * cell);
    r.setAttribute("class", "zero-frame");
    g.appendChild(r);
  }
  if (isDeltaAxis(state.y) && yIdx.has(0)) {
    const yi = yIdx.get(0);
    const r = document.createElementNS(ns, "rect");
    r.setAttribute("x", padL);
    r.setAttribute("y", padT + (ys.length - 1 - yi) * cell);
    r.setAttribute("width", xs.length * cell);
    r.setAttribute("height", cell);
    r.setAttribute("class", "zero-frame");
    g.appendChild(r);
  }
}

// ---------- prompt ----------
function fmtList(vals) {
  if (vals.length === 1) return String(vals[0]);
  if (vals.length <= 4) return "{" + vals.join(", ") + "}";
  return `${vals[0]}..${vals[vals.length-1]} (${vals.length} values)`;
}
function updatePrompt() {
  const ds = state.dataset === "single" ? "single-repeat" : "compound";
  const parts = [];
  parts.push(
    `Show the cross-locus lego heatmap for the ${ds} sweep with ` +
    `x=${state.x} (${AXIS_LABELS[state.x]}), ` +
    `y=${state.y} (${AXIS_LABELS[state.y]}).`,
  );
  const filterDesc = filterCols().map(col =>
    `${col}=${fmtList(state.filters[col] || [])}`,
  ).join("; ");
  if (filterDesc) parts.push(`Filter to ${filterDesc}.`);
  if (state.facetCol && state.facetRow) {
    parts.push(
      `Lay out a grid of panels with columns over ${state.facetCol} ` +
      `and rows over ${state.facetRow}.`,
    );
  } else if (state.facetCol) {
    parts.push(`Lay out one panel per value of ${state.facetCol} (columns).`);
  } else if (state.facetRow) {
    parts.push(`Lay out one panel per value of ${state.facetRow} (rows).`);
  } else {
    parts.push("Render a single panel (average across remaining axes).");
  }
  parts.push(
    `Color each cell by ${state.metric} per strand: ` +
    `rectangle = fwd strand, circle = rc strand. ` +
    `Use the continuous palette anchored at the discrete heatmap colors ` +
    `(red #c0392b at 0 → yellow #ffffbf at 0.5 → blue #08519c at 1).`,
  );
  if (state.x.startsWith("delta") || state.y.startsWith("delta")) {
    parts.push("Outline the Δ=0 row/column in black.");
  }
  document.getElementById("prompt-text").textContent = parts.join(" ");
}

// ---------- wiring ----------
document.querySelectorAll('input[name="dataset"]').forEach(r => {
  r.onchange = () => {
    state.dataset = r.value;
    // reset axes to sensible defaults
    state.x = (state.dataset === "single") ? "delta" : "delta1";
    state.y = (state.dataset === "single") ? "lflank" : "delta2";
    state.filters = {};
    rebuildAxisSelects();
    rebuildFilters();
    rebuildPresets();
    renderAll();
  };
});
document.getElementById("x-axis").onchange = (e) => {
  state.x = e.target.value;
  if (state.y === state.x) {
    state.y = currentAxes().find(a => a !== state.x);
  }
  rebuildAxisSelects(); rebuildFilters(); renderAll();
};
document.getElementById("y-axis").onchange = (e) => {
  state.y = e.target.value;
  if (state.x === state.y) {
    state.x = currentAxes().find(a => a !== state.y);
  }
  rebuildAxisSelects(); rebuildFilters(); renderAll();
};
document.getElementById("metric").onchange = (e) => {
  state.metric = e.target.value; renderAll();
};
document.getElementById("facet-col").onchange = (e) => {
  state.facetCol = e.target.value;
  if (state.facetCol === state.x || state.facetCol === state.y || state.facetCol === state.facetRow) {
    // pick a different non-conflicting axis
    state.facetCol = currentAxes().find(a => a !== state.x && a !== state.y && a !== state.facetRow) || "";
  }
  rebuildAxisSelects(); rebuildFilters(); renderAll();
};
document.getElementById("facet-row").onchange = (e) => {
  state.facetRow = e.target.value;
  if (state.facetRow === state.x || state.facetRow === state.y || state.facetRow === state.facetCol) {
    state.facetRow = currentAxes().find(a => a !== state.x && a !== state.y && a !== state.facetCol) || "";
  }
  rebuildAxisSelects(); rebuildFilters(); renderAll();
};
document.getElementById("cell-size").oninput = (e) => {
  state.cellSize = parseInt(e.target.value, 10);
  document.getElementById("cell-size-val").textContent = state.cellSize + " px";
  renderPreview();
};

function _viewSummary() {
  const ds = state.dataset === "single" ? "single" : "compound";
  const tag = (s) => s.replace(/[^a-z0-9_+\\-]/gi, "");
  const parts = [ds, `${state.x}-vs-${state.y}`];
  if (state.facetCol) parts.push(`col_${state.facetCol}`);
  if (state.facetRow) parts.push(`row_${state.facetRow}`);
  parts.push(state.metric);
  return parts.map(tag).join("__");
}

function _serializeSvg() {
  const svg = document.getElementById("plot").cloneNode(true);
  // Inline a white background so saved files aren't transparent on dark themes.
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  const vb = svg.getAttribute("viewBox").split(" ").map(Number);
  bg.setAttribute("x", vb[0]); bg.setAttribute("y", vb[1]);
  bg.setAttribute("width", vb[2]); bg.setAttribute("height", vb[3]);
  bg.setAttribute("fill", "#1a1d23");
  svg.insertBefore(bg, svg.firstChild);
  svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  // Bake the CSS styling that the live SVG inherits from the page so
  // standalone files match the live preview.
  const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
  style.textContent = `
    .axis-label { fill: #d8dde5; font-family: ui-monospace, monospace; font-size: 12px; }
    .axis-tick { fill: #8a92a0; font-family: ui-monospace, monospace; font-size: 11px; }
    .arm-label { fill: #d8dde5; font-size: 13px; font-weight: 600;
                 font-family: -apple-system, sans-serif; }
    .zero-frame { stroke: #d8dde5; stroke-width: 1.6; fill: none; }
  `;
  svg.insertBefore(style, svg.firstChild);
  return new XMLSerializer().serializeToString(svg);
}

function _download(filename, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

document.getElementById("save-svg").onclick = () => {
  const data = _serializeSvg();
  _download(`${_viewSummary()}.svg`,
            new Blob([data], { type: "image/svg+xml" }));
};
document.getElementById("save-png").onclick = () => {
  const data = _serializeSvg();
  const svgEl = document.getElementById("plot");
  const vb = svgEl.getAttribute("viewBox").split(" ").map(Number);
  const scale = 2;  // 2x for crisp PNG
  const W = Math.round(vb[2] * scale);
  const H = Math.round(vb[3] * scale);
  const img = new Image();
  const url = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(data);
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, W, H);
    canvas.toBlob(b => _download(`${_viewSummary()}.png`, b), "image/png");
  };
  img.src = url;
};

document.getElementById("copy-btn").onclick = async () => {
  const text = document.getElementById("prompt-text").textContent;
  try { await navigator.clipboard.writeText(text); }
  catch (e) {
    // fallback
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta);
    ta.select(); document.execCommand("copy"); document.body.removeChild(ta);
  }
  const btn = document.getElementById("copy-btn");
  btn.textContent = "Copied!"; btn.classList.add("copied");
  setTimeout(() => {
    btn.textContent = "Copy Prompt";
    btn.classList.remove("copied");
  }, 1200);
};

// initial render
rebuildAxisSelects();
rebuildFilters();
rebuildPresets();
renderAll();
</script>
</body>
</html>
"""


def main() -> None:
    single = _load_single()
    compound = _load_compound()
    print(f"single-repeat rows: {len(single['rows'])}  (cols: {len(single['cols'])})")
    print(f"compound rows:      {len(compound['rows'])}  (cols: {len(compound['cols'])})")
    html = (HTML
            .replace("__SINGLE_DATA__", json.dumps(single, separators=(",", ":")))
            .replace("__COMPOUND_DATA__", json.dumps(compound, separators=(",", ":"))))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
