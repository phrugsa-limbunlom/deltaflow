/*
 * DiT interactive walkthrough for the DeltaFlow docs.
 *
 * Renders the tensors exported by examples/90-showcase/10-dit-tensor-viz/main.py
 * (dit_trace.json) as a guided, step-by-step tour of one DiT forward pass.
 * No dependencies. Boots on any page containing <div id="dit-viewer">,
 * and re-boots after Material's instant navigation swaps the page.
 */
(function () {
  "use strict";

  // ------------------------------------------------------------------ colours
  // Never the docs primary blue. Sequential = viridis, diverging = purple/green.
  const VIRIDIS = ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c",
                   "#28ae80", "#5ec962", "#addc30", "#fde725"].map(hex);
  const DIVERGE = ["#7a5cc0", "#f4f5f8", "#3f9e73"].map(hex);
  const GREYS = ["#161b22", "#fafafa"].map(hex);
  const ACCENT = "#e0b13c";

  function hex(h) {
    return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  }
  function ramp(stops, u) {
    u = Math.min(1, Math.max(0, isFinite(u) ? u : 0.5));
    const x = u * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const f = x - i;
    const a = stops[i], b = stops[i + 1];
    return `rgb(${a.map((v, k) => Math.round(v + f * (b[k] - v))).join(",")})`;
  }
  const colSeq = (v, lo, hi) => ramp(VIRIDIS, (v - lo) / (hi - lo || 1));
  const colDiv = (v, lim) => ramp(DIVERGE, 0.5 + 0.5 * v / (lim || 1));

  // ------------------------------------------------------------------ helpers
  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  }
  const flat = (m) => (Array.isArray(m[0]) ? m.flat(Infinity) : m);
  const absMax = (...ms) => Math.max(1e-9, ...ms.map((m) => Math.max(...flat(m).map(Math.abs))));
  const minOf = (...ms) => Math.min(...ms.map((m) => Math.min(...flat(m))));
  const maxOf = (...ms) => Math.max(...ms.map((m) => Math.max(...flat(m))));
  const map2 = (a, b, f) => a.map((v, i) => Array.isArray(v) ? map2(v, b[i], f) : f(v, b[i]));
  const fmt = (v) => Array.isArray(v)
    ? v.map((value, i) => `${["R", "G", "B"][i]} ${fmt(value)}`).join(", ")
    : (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));
  const shape = (s) => `<code class="dw-shape">${s}</code>`;

  let tip = null;
  function showTip(evt, html) {
    if (!tip) {
      tip = el("div", "dw-tip");
      document.body.appendChild(tip);
    }
    tip.innerHTML = html;
    tip.style.display = "block";
    tip.style.left = evt.clientX + 14 + "px";
    tip.style.top = evt.clientY + 14 + "px";
  }
  function hideTip() {
    if (tip) tip.style.display = "none";
  }

  // ------------------------------------------------------------------ heatmap
  /**
   * Canvas heatmap of a 2D array.
   * opts: cell | cellW,cellH, cmap ("seq" | "div"), lo, hi, lim, title,
   *       rowLabels, grid (draw lines every k cells), onHover(r, c), tipFn(r, c, v)
   * Highlights: hm.rows = [..], hm.rect = {r0, c0, r1, c1}
   */
  class Heatmap {
    constructor(parent, mat, o = {}) {
      this.mat = mat;
      this.o = o;
      this.R = mat.length;
      this.C = mat[0].length;
      this.cw = o.cellW || o.cell || 12;
      this.ch = o.cellH || o.cell || 12;
      if (o.cmap === "seq" || o.cmap === "gray" || o.cmap === "rgb") {
        this.lo = o.lo !== undefined ? o.lo : minOf(mat);
        this.hi = o.hi !== undefined ? o.hi : maxOf(mat);
      } else {
        this.lim = o.lim || absMax(mat);
      }
      this.rows = [];
      this.rect = null;

      const fig = el("figure", "dw-hm");
      if (o.title) fig.appendChild(el("figcaption", "", o.title));
      const body = el("div", "dw-hm-body");
      if (o.rowLabels) {
        const lab = el("div", "dw-rowlabels");
        o.rowLabels.forEach((t) => {
          const d = el("div", "", t);
          d.style.height = this.ch + "px";
          d.style.lineHeight = this.ch + "px";
          lab.appendChild(d);
        });
        body.appendChild(lab);
      }
      const cv = el("canvas");
      const dpr = window.devicePixelRatio || 1;
      cv.width = this.C * this.cw * dpr;
      cv.height = this.R * this.ch * dpr;
      cv.style.width = this.C * this.cw + "px";
      cv.style.height = this.R * this.ch + "px";
      cv.setAttribute("role", "img");
      cv.setAttribute("aria-label", `${fig.textContent}. ${this.R} rows, ${this.C} columns.`);
      this.ctx = cv.getContext("2d");
      this.ctx.scale(dpr, dpr);
      body.appendChild(cv);
      fig.appendChild(body);
      parent.appendChild(fig);
      this.fig = fig;
      const scale = el("div", "dw-scale");
      const lo = this.lim === undefined ? this.lo : -this.lim;
      const hi = this.lim === undefined ? this.hi : this.lim;
      const bar = el("span", "dw-scale-bar");
      bar.style.background = `linear-gradient(to right, ${Array.from({ length: 9 }, (_, i) => this.color(lo + (hi - lo) * i / 8)).join(",")})`;
      scale.append(el("span", "", fmt(lo)), bar, el("span", "", fmt(hi)));
      fig.appendChild(scale);
      if (o.cmap === "rgb" && !o.compact) fig.appendChild(el("div", "dw-image-note",
        `${o.channel === "rgb" ? "RGB" : ["R", "G", "B"][o.channel]} display${o.signed ? ", zero at midpoint" : ", clipped to [0, 1]"}. Hover for raw values.`));

      cv.addEventListener("mousemove", (e) => {
        const b = cv.getBoundingClientRect();
        const c = Math.floor((e.clientX - b.left) / b.width * this.C);
        const r = Math.floor((e.clientY - b.top) / b.height * this.R);
        if (r < 0 || c < 0 || r >= this.R || c >= this.C) return;
        const v = this.mat[r][c];
        showTip(e, o.tipFn ? o.tipFn(r, c, v) : `[${r}, ${c}] = ${fmt(v)}`);
        if (o.onHover) o.onHover(r, c);
      });
      cv.addEventListener("mouseleave", () => {
        hideTip();
        if (o.onHover) o.onHover(null, null);
      });
      this.draw();
    }
    color(v) {
      if (this.o.cmap === "rgb") {
        if (Array.isArray(v) && this.o.channel === "rgb") {
          return `rgb(${v.map((value) => Math.round(255 * Math.max(0, Math.min(1, (value - this.lo) / (this.hi - this.lo || 1))))).join(",")})`;
        }
        const value = Array.isArray(v) ? v[this.o.channel] : v;
        return this.o.signed && this.o.channel !== "rgb"
          ? colDiv(value, Math.max(Math.abs(this.lo), Math.abs(this.hi)))
          : ramp(GREYS, (value - this.lo) / (this.hi - this.lo || 1));
      }
      if (this.o.cmap === "gray") return ramp(GREYS, (v - this.lo) / (this.hi - this.lo || 1));
      return this.o.cmap === "seq" ? colSeq(v, this.lo, this.hi) : colDiv(v, this.lim);
    }
    draw() {
      const { ctx, cw, ch } = this;
      for (let r = 0; r < this.R; r++)
        for (let c = 0; c < this.C; c++) {
          ctx.fillStyle = this.color(this.mat[r][c]);
          ctx.fillRect(c * cw, r * ch, cw, ch);
        }
      if (this.o.grid) {
        const k = this.o.grid;
        ctx.strokeStyle = "rgba(255,255,255,0.85)";
        ctx.lineWidth = 1;
        for (let i = k; i < this.C; i += k) line(ctx, i * cw, 0, i * cw, this.R * ch);
        for (let i = k; i < this.R; i += k) line(ctx, 0, i * ch, this.C * cw, i * ch);
      }
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2.5;
      this.rows.forEach((r) => ctx.strokeRect(1, r * ch + 1, this.C * cw - 2, ch - 2));
      if (this.rect) {
        const q = this.rect;
        ctx.strokeRect(q.c0 * cw + 1, q.r0 * ch + 1, (q.c1 - q.c0) * cw - 2, (q.r1 - q.r0) * ch - 2);
      }
    }
    set(rows, rect) {
      this.rows = rows || [];
      this.rect = rect || null;
      this.draw();
    }
  }
  function line(ctx, x0, y0, x1, y1) {
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }

  function lineChart(parent, series, o = {}) {
    const W = o.width || 360, H = o.height || 150, pad = 6;
    const all = series.flatMap((s) => s.data);
    const lo = Math.min(...all), hi = Math.max(...all);
    const n = series[0].data.length;
    const sx = (i) => pad + (i / (n - 1)) * (W - 2 * pad);
    const sy = (v) => H - pad - ((v - lo) / (hi - lo || 1)) * (H - 2 * pad);
    const paths = series.map((s) => {
      const d = s.data.map((v, i) => `${i ? "L" : "M"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join("");
      return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="1.4"/>`;
    });
    const legend = series.map((s) => `<span style="color:${s.color}">&#9632; ${s.name}</span>`).join(" ");
    const fig = el("figure", "dw-hm");
    if (o.title) fig.appendChild(el("figcaption", "", o.title));
    fig.appendChild(el("div", "", `<svg width="${W}" height="${H}" class="dw-svg">${paths.join("")}</svg>`));
    fig.appendChild(el("div", "dw-legend", legend));
    parent.appendChild(fig);
  }

  function buttons(parent, label, options, current, onPick) {
    const g = el("div", "dw-group");
    if (label) g.appendChild(el("span", "dw-glabel", label));
    options.forEach(([value, text]) => {
      const b = el("button", "dw-btn" + (value === current ? " is-active" : ""), text);
      b.type = "button";
      b.dataset.dwFocus = `${label}-${value}`;
      b.setAttribute("aria-pressed", String(value === current));
      b.addEventListener("click", () => onPick(value));
      g.appendChild(b);
    });
    parent.appendChild(g);
    return g;
  }

  // ------------------------------------------------------------------ the tour
  class Walkthrough {
    constructor(root, data) {
      this.root = root;
      this.d = data;
      this.cfg = data.config;
      this.s = { step: 0, img: 0, ti: 1, li: 0, block: 0, head: "mean", query: 5, cfgS: 2,
                 sampleCfg: 2, frame: 0, modRow: 1, operation: 0, inspectionOpen: false, channel: "rgb" };
      this.timer = null;
      root.innerHTML = "";
      root.tabIndex = 0;
      root.setAttribute("role", "region");
      root.setAttribute("aria-label", "DiT image to velocity walkthrough");
      const intro = el("div", "dw-intro",
        '<span class="dw-kicker">INSIDE A DIFFUSION TRANSFORMER</span>' +
        '<h2>One image. Every transformation.</h2>' +
        `<p>Follow real activations through ${data.config.depth} transformer blocks, from pixels to patch tokens and back to a velocity image.</p>`);
      this.ctrl = el("div", "dw-controls");
      this.nav = el("div", "dw-nav");
      this.head = el("div", "dw-head");
      this.stage = el("div", "dw-stage");
      this.text = el("div", "dw-text");
      this.rail = el("div", "dw-dots");
      this.rail.setAttribute("aria-label", "Layers in forward order");
      this.status = el("div", "dw-status");
      this.status.setAttribute("aria-live", "polite");
      [intro, this.ctrl, this.rail, this.nav, this.head, this.stage, this.text, this.status].forEach((e) => root.appendChild(e));
      root.addEventListener("keydown", (e) => {
        if (e.target !== root) return;
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          e.preventDefault();
          this.advance(e.key === "ArrowRight" ? 1 : -1);
        }
      });
      this.steps = this.defineSteps();
      this.render();
    }

    // ---- data accessors
    tr(li = this.s.li, img = this.s.img, ti = this.s.ti) {
      return this.d.traces[`${img}|${ti}|${li}`];
    }
    get t() {
      return this.d.t_values[this.s.ti];
    }
    labelName(li = this.s.li) {
      return ["y = 0", "y = 1", "null"][li];
    }

    set(patch) {
      const focusKey = document.activeElement.dataset.dwFocus;
      Object.assign(this.s, patch);
      this.render();
      if (focusKey) {
        const target = [...this.root.querySelectorAll("[data-dw-focus]")].find((e) => e.dataset.dwFocus === focusKey);
        (target && !target.disabled ? target : this.root).focus({ preventScroll: true });
      }
    }
    go(i) {
      i = Math.max(0, Math.min(this.steps.length - 1, i));
      if (i !== this.s.step) this.set({ step: i, operation: 0 });
    }
    advance(direction) {
      if (this.steps[this.s.step].block !== undefined) {
        const operation = this.s.operation + direction;
        if (operation >= 0 && operation < 5) {
          this.set({ operation });
          return;
        }
      }
      this.go(this.s.step + direction);
      if (direction < 0 && this.steps[this.s.step].block !== undefined) this.set({ operation: 4 });
    }

    render() {
      if (this.timer) {
        clearInterval(this.timer);
        this.timer = null;
      }
      hideTip();
      const st = this.steps[this.s.step];
      if (st.block !== undefined) this.s.block = st.block;

      this.ctrl.innerHTML = "";
      buttons(this.ctrl, "image x\u2081", [[0, "green top"], [1, "green bottom"]], this.s.img,
              (v) => this.set({ img: v }));
      buttons(this.ctrl, "time t", this.d.t_values.map((t, i) => [i, String(t)]), this.s.ti,
              (v) => this.set({ ti: v }));
      buttons(this.ctrl, "label y", [[0, "0"], [1, "1"], [2, "null"]], this.s.li,
              (v) => this.set({ li: v }));
      buttons(this.ctrl, "image display", [["rgb", "RGB"], [0, "R"], [1, "G"], [2, "B"]], this.s.channel,
              (channel) => this.set({ channel }));

      this.nav.innerHTML = "";
      const prev = el("button", "dw-btn", "&larr; Prev");
      prev.dataset.dwFocus = "previous";
      prev.disabled = this.s.step === 0;
      prev.addEventListener("click", () => this.advance(-1));
      const next = el("button", "dw-btn dw-next",
        `${st.block !== undefined && this.s.operation < 4 ? "Next operation" : "Next"} &rarr;`);
      next.dataset.dwFocus = "next";
      next.disabled = this.s.step === this.steps.length - 1;
      next.addEventListener("click", () => this.advance(1));
      this.rail.innerHTML = "";
      this.steps.forEach((x, i) => {
        const b = el("button", "dw-dot" + (i === this.s.step ? " is-active" : "") + (i < this.s.step ? " is-done" : ""),
          `<span class="dw-node-no">${String(i + 1).padStart(2, "0")}</span><span>${x.short}</span>`);
        b.title = x.title;
        b.type = "button";
        b.dataset.dwFocus = `layer-${i}`;
        b.setAttribute("aria-label", `Step ${i + 1}, ${x.title}`);
        if (i === this.s.step) b.setAttribute("aria-current", "step");
        b.addEventListener("click", () => this.go(i));
        this.rail.appendChild(b);
      });
      this.nav.append(prev, el("span", "dw-progress", `${this.s.step + 1} / ${this.steps.length}`), next);

      this.head.innerHTML = `<span class="dw-stepno">${st.block !== undefined ? "TRANSFORMER STACK" : "TENSOR TRANSFORMATION"}</span>` +
        `<h3>${st.title}</h3>` + (st.shape ? `<div class="dw-shapes">${st.shape()}</div>` : "");
      this.text.innerHTML = st.text();
      this.stage.innerHTML = "";
      st.render(this.stage);
      this.status.textContent = `Step ${this.s.step + 1} of ${this.steps.length}. ${st.title}` +
        (st.block !== undefined ? `. Operation ${this.s.operation + 1} of 5` : "");
    }

    row(parent) {
      const r = el("div", "dw-row");
      parent.appendChild(r);
      return r;
    }

    // ---- reusable views
    imageView(parent, mat, title, o = {}) {
      const lim = o.lim || absMax(mat);
      return new Heatmap(parent, mat, Object.assign({ cell: 24, cmap: "rgb", title,
        compact: (o.cell || 24) < 16,
        channel: this.s.channel, lo: o.signed ? -lim : 0, hi: o.signed ? lim : 1,
        tipFn: (r, c, v) => `pixel (${r}, ${c}) = ${fmt(v)}` }, o));
    }

    patchRect(k) {
      const p = this.cfg.patch, g = this.cfg.grid;
      const r = Math.floor(k / g), c = k % g;
      return { r0: r * p, c0: c * p, r1: (r + 1) * p, c1: (c + 1) * p };
    }
    tokenAt(r, c) {
      const p = this.cfg.patch;
      return Math.floor(r / p) * this.cfg.grid + Math.floor(c / p);
    }

    tensorSequence(parent, tensors) {
      const C = this.cfg;
      const ctl = this.row(parent);
      const label = el("label", "dw-token-picker", "Follow a patch through the tensors ");
      const select = el("select");
      select.setAttribute("aria-label", "Patch token to highlight");
      for (let k = 0; k < C.N; k++) {
        const option = el("option", "", `Token ${k} (row ${Math.floor(k / C.grid)}, column ${k % C.grid})`);
        option.value = k;
        select.appendChild(option);
      }
      select.value = this.s.query;
      label.appendChild(select);
      ctl.appendChild(label);
      const preview = this.imageView(ctl, this.tr().x_t, "patch location", { cell: 8, grid: C.patch });
      const r = this.row(parent);
      r.classList.add("dw-tensors");
      const maps = [];
      const link = (k) => {
        if (k === null) return;
        this.s.query = k;
        select.value = k;
        preview.set([], this.patchRect(k));
        maps.forEach((m) => m.set([k]));
      };
      const lim = absMax(...tensors.map(([, m]) => m));
      tensors.forEach(([title, mat], i) => {
        if (i) r.appendChild(el("div", "dw-op", "&rarr;"));
        maps.push(new Heatmap(r, mat, {
          cellW: Math.min(22, 256 / mat[0].length), cellH: 12, lim,
          title: `${title}<br>${shape(`(1, ${mat.length}, ${mat[0].length})`)}`,
          onHover: link,
          tipFn: (rr, cc, v) => `token ${rr}, channel ${cc}<br>${fmt(v)}`,
        }));
      });
      select.addEventListener("change", () => link(Number(select.value)));
      link(this.s.query);
      parent.appendChild(el("p", "dw-caption",
        "Rows are patch tokens. Columns are feature channels. The yellow row follows the same patch. These heatmaps share one colour scale."));
    }

    blockView(stage, block, details) {
      const tr = this.tr(), state = tr.block_states[block];
      const input = block === 0 ? map2(tr.tokens, this.d.pos, (a, b) => a + b) : tr.block_states[block - 1].output;
      const attentionUpdate = map2(state.after_attention, input, (a, b) => a - b);
      const mlpUpdate = map2(state.output, state.after_attention, (a, b) => a - b);
      const operations = [
        { name: "1  Attention input", tensors: [["Block input h", input], ["LayerNorm + shift and scale", state.attention_input]],
          text: "Normalise each token, then apply the attention shift and scale from c. The token count and channel width stay fixed." },
        { name: "2  Attention + residual", tensors: [["Residual h", input], ["Gated attention update", attentionUpdate], ["h + attention update", state.after_attention]],
          text: "Attention mixes information across all patch tokens. Its output projection returns to D channels. The gate scales that update before the residual addition." },
        { name: "3  MLP input", tensors: [["After attention", state.after_attention], ["LayerNorm + shift and scale", state.mlp_input]],
          text: "A second LayerNorm and a separate shift and scale prepare each token for the feed-forward network." },
        { name: "4  MLP expansion", tensors: [["MLP input", state.mlp_input], ["Linear + GELU", state.mlp_hidden]],
          text: `Each token expands from ${this.cfg.D} to ${state.mlp_hidden[0].length} features. GELU acts on these features independently. No patch tokens are mixed here.` },
        { name: "5  MLP + residual", tensors: [["Residual after attention", state.after_attention], ["Projected, gated MLP update", mlpUpdate], ["Block output", state.output]],
          text: `The second Linear projects back to ${this.cfg.D} channels. Multiply by the MLP gate and add to the residual. This output is the next layer's input.` },
      ];
      buttons(stage, "Inside this block", operations.map((op, i) => [i, op.name]), this.s.operation,
        (operation) => this.set({ operation }));
      const op = operations[this.s.operation];
      stage.appendChild(el("p", "dw-operation-note", op.text));
      this.tensorSequence(stage, op.tensors);
      const more = el("details", "dw-detail");
      more.open = this.s.inspectionOpen;
      more.addEventListener("toggle", () => {
        if (more.isConnected) this.s.inspectionOpen = more.open;
      });
      more.appendChild(el("summary", "", "Inspect attention heads and conditioning gates"));
      const attention = el("div");
      const gates = el("div");
      more.append(el("h4", "", "Attention weights"), attention, el("h4", "", "Conditioning gates"), gates);
      // Reuse the detailed views without their block selectors. The layer rail selects the block.
      details[6].render(attention);
      attention.querySelector(".dw-group").remove();
      details[5].render(gates);
      gates.querySelector(".dw-group").remove();
      stage.appendChild(more);
    }

    // ---- steps
    defineSteps() {
      const C = this.cfg, D = C.D, N = C.N, p = C.patch, g = C.grid, channels = C.channels;
      const patchValues = p * p * channels;
      const self = this;
      const details = [
        {
          title: "Start with the input image",
          shape: () => `x_t ${shape(`(B, ${channels}, ${C.img}, ${C.img})`)} &nbsp; t ${shape("(B,)")} &nbsp; y ${shape("(B,)")}`,
          text: () => `<p>Flow matching moves samples along a straight path from noise
            <b>x\u2080</b> (t = 0) to data <b>x\u2081</b> (t = 1). The network sees the point
            <b>x_t = (1 &minus; t) x\u2080 + t x\u2081</b>, the time <b>t</b>, and a class label
            <b>y</b>, and must predict the velocity that carries x_t towards the data.</p>
            <p>Use the controls above to pick the data image, the time, and the label the model is told.
            The same noise draw x\u2080 is used everywhere, so each control changes exactly one thing.
            This model has ${C.params.toLocaleString()} parameters, ${C.depth} blocks, width D = ${D}.</p>`,
          render(stage) {
            const x0 = self.d.x0, x1 = self.d.x1[self.s.img], xt = self.tr().x_t;
            const r = self.row(stage);
            r.classList.add("dw-input-row");
            self.imageView(r, xt, `MODEL INPUT x_t ${shape(`(1, ${channels}, ${C.img}, ${C.img})`)}`, { cell: 32, grid: p });
            const info = el("div", "dw-input-info",
              `<span class="dw-kicker">WHAT THE MODEL RECEIVES</span><h4>Pixels, not tokens. Yet.</h4>
              <p>One RGB image at t = <b>${self.t}</b>. Each outlined ${p}&times;${p} patch contains ${patchValues} values across R, G, and B. The first layer mixes all three channels into one token.</p>
              <div class="dw-facts"><span><b>1</b> image</span><span><b>${channels}</b> RGB channels</span><span><b>${C.img}&times;${C.img}</b> pixels</span><span><b>${N}</b> patches</span></div>
              <p>The output will have this same shape, but its values describe <b>velocity</b>, not brightness.</p>`);
            r.appendChild(info);
            const source = el("details", "dw-detail");
            source.appendChild(el("summary", "", "Where does the input image come from?"));
            const mix = self.row(source);
            self.imageView(mix, x0, "noise x\u2080", { cell: 14 });
            mix.appendChild(el("div", "dw-op", "&times;(1&minus;t) +"));
            self.imageView(mix, x1, "data x\u2081", { cell: 14 });
            mix.appendChild(el("div", "dw-chip", `x_t = (1 &minus; t) x\u2080 + t x\u2081<br>t = ${self.t}, ${self.labelName()}`));
            stage.appendChild(source);
          },
        },
        {
          title: "Patchify, the image becomes a sequence of tokens",
          shape: () => `${shape(`(B, ${channels}, ${C.img}, ${C.img})`)} &rarr; Conv2d(k = s = ${p}) &rarr; ${shape(`(B, ${D}, ${g}, ${g})`)}
            &rarr; flatten(2) &rarr; ${shape(`(B, ${D}, ${N})`)} &rarr; transpose(1, 2) &rarr; ${shape(`(B, ${N}, ${D})`)}`,
          text: () => `<p>A convolution whose kernel equals its stride sees every ${p}&times;${p} patch exactly once,
            so it is one shared linear map from ${p}&times;${p}&times;${channels} = ${patchValues} channel values to a ${D}-dim vector. The ${g}&times;${g} grid is then
            flattened row by row into <b>N = ${N} tokens</b>, and the transpose puts tokens first.</p>
            <p><b>Hover</b> a patch on the image or a row of the matrix to see which token is which.</p>`,
          render(stage) {
            const r = self.row(stage);
            const tr = self.tr();
            let img, mat, pixels;
            const link = (k) => {
              img.set([], k === null ? null : self.patchRect(k));
              mat.set(k === null ? [] : [k]);
              pixels.set(k === null ? [] : [k]);
            };
            img = self.imageView(r, tr.x_t, `x_t, ${g}&times;${g} grid of ${p}&times;${p} patches`, {
              grid: p, onHover: (rr, cc) => link(rr === null ? null : self.tokenAt(rr, cc)),
              tipFn: (rr, cc, v) => `pixel (${rr}, ${cc}) = ${fmt(v)}<br>belongs to token ${self.tokenAt(rr, cc)}`,
            });
            r.appendChild(el("div", "dw-op", "&rarr;"));
            const patches = Array.from({ length: N }, (_, k) => {
              const q = self.patchRect(k);
              return Array.from({ length: channels }, (_, ch) =>
                tr.x_t.slice(q.r0, q.r1).flatMap((row) => row.slice(q.c0, q.c1).map((pixel) => pixel[ch]))).flat();
            });
            pixels = new Heatmap(r, patches, { cellW: 10, cellH: 12,
              title: `patch values ${shape(`(${N}, ${patchValues})`)}<br>R pixels, then G, then B`,
              tipFn: (rr, cc, v) => `token ${rr}, ${["R", "G", "B"][Math.floor(cc / (p * p))]}<br>patch pixel (${Math.floor((cc % (p * p)) / p)}, ${cc % p}) = ${fmt(v)}`,
              onHover: (rr) => link(rr) });
            r.appendChild(el("div", "dw-op", "&rarr;"));
            mat = new Heatmap(r, tr.tokens, {
              cellW: 4, cellH: 12, title: `tokens h ${shape(`(${N}, ${D})`)}, row = token, column = channel`,
              rowLabels: [...Array(N).keys()].map(String),
              onHover: (rr) => link(rr), tipFn: (rr, cc, v) => `token ${rr}, channel ${cc}<br>${fmt(v)}`,
            });
            stage.appendChild(el("p", "dw-caption",
              `The middle panel is a pixel rearrangement, not an extra model layer. Conv2d learns ${D} features per patch. Flatten and transpose only change the layout.`));
            const conv = el("details", "dw-detail");
            conv.appendChild(el("summary", "", "Inspect the Conv2d output before flatten and transpose"));
            const channelLabel = el("label", "dw-slider", "Feature channel ");
            const channel = el("input");
            Object.assign(channel, { type: "range", min: 0, max: D - 1, step: 1, value: 0 });
            channel.setAttribute("aria-label", "Convolution feature channel");
            const value = el("span", "", "0");
            channelLabel.append(channel, value);
            conv.appendChild(channelLabel);
            const convGrid = (c) => Array.from({ length: g }, (_, rr) => tr.tokens.slice(rr * g, (rr + 1) * g).map((token) => token[c]));
            const hm = new Heatmap(self.row(conv), convGrid(0), { cell: 32, lim: absMax(tr.tokens),
              title: `one of ${D} channels in ${shape(`(1, ${D}, ${g}, ${g})`)}` });
            channel.addEventListener("input", () => {
              value.textContent = channel.value;
              hm.mat = convGrid(Number(channel.value));
              hm.draw();
            });
            stage.appendChild(conv);
          },
        },
        {
          title: "Position embedding, telling each token where it sits",
          shape: () => `h ${shape(`(B, ${N}, ${D})`)} + pos_embed ${shape(`(1, ${N}, ${D})`)} = ${shape(`(B, ${N}, ${D})`)}`,
          text: () => `<p>Self-attention treats its input as an unordered set. Without extra information, token 0
            (top-left) and token ${N - 1} (bottom-right) would be indistinguishable apart from their pixels.
            A learned <b>pos_embed</b>, one row per position, is added so position becomes part of each token.
            It is shared by every image (the leading 1 broadcasts over the batch).</p>
            <p>Hover any row to follow one token through the sum.</p>`,
          render(stage) {
            const r = self.row(stage);
            const tok = self.tr().tokens, pos = self.d.pos;
            const sum = map2(tok, pos, (a, b) => a + b);
            const lim = absMax(tok, pos, sum);
            const hms = [];
            const link = (k) => hms.forEach((h) => h.set(k === null ? [] : [k]));
            const mk = (m, t) => hms.push(new Heatmap(r, m, { cellW: 5, cellH: 12, lim, title: t,
              onHover: link, tipFn: (rr, cc, v) => `token ${rr}, channel ${cc}<br>${fmt(v)}` }));
            mk(tok, "patch tokens");
            r.appendChild(el("div", "dw-op", "+"));
            mk(pos, "pos_embed (learned)");
            r.appendChild(el("div", "dw-op", "="));
            mk(sum, "tokens entering block 0");
          },
        },
        {
          title: "Time embedding, a scalar becomes a vector",
          shape: () => `t ${shape("(B,)")} &times; ${C.time_scale} &rarr; sinusoids ${shape(`(B, ${C.freq_dim})`)} &rarr; MLP &rarr; t_emb ${shape(`(B, ${D})`)}`,
          text: () => `<p>A single number is a poor input for a network, so <b>t</b> is scaled by ${C.time_scale}
            and expanded into ${C.freq_dim / 2} cosines and ${C.freq_dim / 2} sines of geometrically spaced
            frequencies (the transformer positional-encoding trick). Fast frequencies resolve tiny changes in t,
            slow ones encode its coarse value. A two-layer MLP then maps them to <b>t_emb</b>.</p>
            <p>Switch <b>time t</b> above and watch both the sinusoids and t_emb change. The bottom matrix shows
            t_emb at every exported time, with the current one outlined.</p>`,
          render(stage) {
            const r = self.row(stage);
            const s = self.d.sinus[String(self.s.ti)], h = s.length / 2;
            lineChart(r, [{ name: "cos", color: "#3f9e73", data: s.slice(0, h) },
                          { name: "sin", color: "#7a5cc0", data: s.slice(h) }],
                      { title: `sinusoids of t &times; ${C.time_scale}, t = ${self.t} (high to low frequency)` });
            const rows = self.d.t_values.map((_, ti) => self.tr(self.s.li, self.s.img, ti).t_emb);
            const hm = new Heatmap(self.row(stage), rows, { cellW: 7, cellH: 22, title: `t_emb ${shape(`(${D},)`)}`,
              rowLabels: self.d.t_values.map((t) => `t = ${t}`),
              tipFn: (rr, cc, v) => `t = ${self.d.t_values[rr]}, channel ${cc}<br>${fmt(v)}` });
            hm.set([self.s.ti]);
          },
        },
        {
          title: "Label embedding, and the single conditioning vector c",
          shape: () => `y &rarr; embedding_table[y] ${shape(`(B, ${D})`)} &nbsp; c = t_emb + y_emb ${shape(`(B, ${D})`)}`,
          text: () => `<p>The label is a plain row lookup in a table with <b>num_classes + 1</b> rows. The extra
            row is the <b>null</b> token used for classifier-free guidance. During training 10% of labels are
            swapped for it, so one network learns both the conditional and the unconditional velocity.</p>
            <p>Time and label are simply <b>added</b>. The result <b>c</b> is one vector per image. It is never
            turned into tokens or concatenated with the image. Switch <b>label y</b> to see c move.</p>`,
          render(stage) {
            const tr = self.tr();
            const r = self.row(stage);
            const tab = new Heatmap(r, self.d.emb_table, { cellW: 6, cellH: 22,
              title: `embedding_table ${shape(`(3, ${D})`)}`, rowLabels: ["class 0", "class 1", "null"] });
            tab.set([self.s.li]);
            const lim = absMax(tr.t_emb, tr.y_emb, tr.c);
            const r2 = self.row(stage);
            new Heatmap(r2, [tr.t_emb, tr.y_emb, tr.c], { cellW: 6, cellH: 22, lim,
              title: `c = t_emb + y_emb (${self.labelName()})`, rowLabels: ["t_emb", "y_emb", "c"] });
            const cs = [0, 1, 2].map((li) => self.tr(li).c);
            const all = new Heatmap(r2, cs, { cellW: 6, cellH: 22, lim,
              title: "c for each label (same x_t, same t)", rowLabels: ["y = 0", "y = 1", "null"] });
            all.set([self.s.li]);
          },
        },
        {
          title: "adaLN-Zero, c becomes per-channel shift, scale, and gate",
          shape: () => `c ${shape(`(B, ${D})`)} &rarr; SiLU, Linear(${D}, ${6 * D}) &rarr; ${shape(`(B, ${6 * D})`)}
            &rarr; chunk(6) &rarr; 6 &times; ${shape(`(B, ${D})`)} &rarr; unsqueeze(1) &rarr; ${shape(`(B, 1, ${D})`)}`,
          text: () => `<p>Each block turns c into six vectors, a <b>shift</b>, <b>scale</b>, and <b>gate</b> for the
            attention sub-layer and again for the MLP sub-layer, used as</p>
            <p class="dw-eq">h &larr; h + gate &odot; sublayer( LayerNorm(h) &odot; (1 + scale) + shift )</p>
            <p>The unsqueeze to <b>(B, 1, D)</b> means the <b>same</b> values are broadcast to all ${N} tokens.
            Conditioning changes how each channel is normalised, including the attention queries and keys,
            and how much each branch writes. The Linear is zero-initialised, so at the start of training every gate is 0,
            every block is the identity, and v = 0.</p>
            <p><b>Hover</b> a row of the left matrix to see it broadcast over the tokens on the right.</p>`,
          render(stage) {
            buttons(stage, "block", [...Array(C.depth).keys()].map((b) => [b, String(b)]), self.s.block,
                    (v) => self.set({ block: v }));
            const mods = self.tr().mods[self.s.block];
            const names = ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"];
            const r = self.row(stage);
            const lim = absMax(...self.tr().mods);
            let bc;
            const left = new Heatmap(r, mods, { cellW: 6, cellH: 20, lim, rowLabels: names,
              title: `block ${self.s.block} modulation ${shape(`(6, ${D})`)}`,
              tipFn: (rr, cc, v) => `${names[rr]}, channel ${cc}<br>${fmt(v)}`,
              onHover: (rr) => {
                if (rr === null || rr === self.s.modRow) return;
                self.s.modRow = rr;
                left.set([rr]);
                bc.mat = Array(N).fill(mods[rr]);
                bc.draw();
                bc.fig.querySelector("figcaption").innerHTML = `${names[rr]} broadcast to every token ${shape(`(${N}, ${D})`)}`;
              } });
            left.set([self.s.modRow]);
            r.appendChild(el("div", "dw-op", "&rarr;"));
            bc = new Heatmap(r, Array(N).fill(mods[self.s.modRow]), { cellW: 6, cellH: 9, lim,
              title: `${names[self.s.modRow]} broadcast to every token ${shape(`(${N}, ${D})`)}` });
          },
        },
        {
          title: "Self-attention, the only step that mixes tokens",
          shape: () => `qkv ${shape(`(B, ${N}, ${3 * D})`)} &rarr; ${C.heads} heads &rarr; softmax(q k&#7488; / &radic;${D / C.heads}) ${shape(`(B, ${C.heads}, ${N}, ${N})`)}`,
          text: () => `<p>Every token emits a query, a key, and a value. Row <i>i</i> of the attention matrix is a
            softmax over all ${N} keys, telling token <i>i</i> how much of each other token's value to take.
            Everything else in the block acts on each token separately.</p>
            <p><b>Hover</b> a patch of the image or a row of the matrix to choose the query token. The right panel
            paints its attention back onto the ${g}&times;${g} patch grid. Compare blocks and heads to see
            how the weights change. These weights describe token mixing, not a causal explanation of
            the predicted colours.</p>`,
          render(stage) {
            const ctl = self.row(stage);
            buttons(ctl, "block", [...Array(C.depth).keys()].map((b) => [b, String(b)]), self.s.block,
                    (v) => self.set({ block: v }));
            buttons(ctl, "head", [["mean", "mean"], ...[...Array(C.heads).keys()].map((h) => [h, String(h)])],
                    self.s.head, (v) => self.set({ head: v }));
            const A = self.tr().attn[self.s.block];
            const M = self.s.head === "mean"
              ? A[0].map((row, i) => row.map((_, j) => A.reduce((s, h) => s + h[i][j], 0) / A.length))
              : A[self.s.head];
            const r = self.row(stage);
            const grid = (q) => [...Array(g).keys()].map((i) => M[q].slice(i * g, (i + 1) * g));
            let img, mat, over;
            const setQ = (q) => {
              if (q === null) return;
              self.s.query = q;
              img.set([], self.patchRect(q));
              mat.set([q]);
              over.mat = grid(q);
              over.set([], { r0: Math.floor(q / g), c0: q % g, r1: Math.floor(q / g) + 1, c1: q % g + 1 });
            };
            img = self.imageView(r, self.tr().x_t, "x_t (pick a query patch)", { grid: p,
              onHover: (rr, cc) => rr !== null && setQ(self.tokenAt(rr, cc)) });
            mat = new Heatmap(r, M, { cell: 13, cmap: "seq", lo: 0, hi: maxOf(M),
              title: `attention ${shape(`(${N}, ${N})`)}, rows = queries`,
              tipFn: (rr, cc, v) => `query ${rr} &rarr; key ${cc}<br>weight ${fmt(v)}`,
              onHover: (rr) => setQ(rr) });
            over = new Heatmap(r, grid(self.s.query), { cell: 48, cmap: "seq", lo: 0, hi: maxOf(M),
              title: "where the query looks", tipFn: (rr, cc, v) => `key token ${rr * g + cc}<br>weight ${fmt(v)}` });
            setQ(self.s.query);
          },
        },
        {
          title: "The residual stream, tokens accumulate updates block by block",
          shape: () => `h ${shape(`(B, ${N}, ${D})`)} stays the same shape through all ${C.depth} blocks`,
          text: () => `<p>Each block only <b>adds</b> to the tokens (gated by the adaLN gates), so the shape
            never changes. Below, each small grid shows the norm &Vert;h&Vert; of every token, laid out on the
            patch grid, after the position embedding and after each block. A large norm measures activation
            magnitude, not the size of the residual update. Compare the two images, or t = 0 against t = 0.9.</p>`,
          render(stage) {
            const norms = self.tr().tok_norm;
            const lo = minOf(norms), hi = maxOf(norms);
            const r = self.row(stage);
            norms.forEach((nv, i) => {
              const m = [...Array(g).keys()].map((k) => nv.slice(k * g, (k + 1) * g));
              new Heatmap(r, m, { cell: 34, cmap: "seq", lo, hi, title: i === 0 ? "after pos_embed" : `after block ${i - 1}`,
                tipFn: (rr, cc, v) => `token ${rr * g + cc}<br>&Vert;h&Vert; = ${fmt(v)}` });
              if (i < norms.length - 1) r.appendChild(el("div", "dw-op", "&rarr;"));
            });
          },
        },
        {
          title: "Final layer and unpatchify, tokens back to pixels",
          shape: () => `${shape(`(B, ${N}, ${D})`)} &rarr; adaLN, Linear(${D}, ${patchValues}) &rarr; ${shape(`(B, ${N}, ${patchValues})`)}
            &rarr; reshape ${shape(`(B, ${g}, ${g}, ${p}, ${p}, ${channels})`)} &rarr; einsum bhwpqc&rarr;bchpwq &rarr; ${shape(`(B, ${channels}, ${C.img}, ${C.img})`)}`,
          text: () => `<p>A last adaLN modulation, then a Linear layer maps every token to the ${patchValues} channel values
            of <b>its own patch</b>. The einsum interleaves the grid indices (h, w) with the in-patch indices
            (p, q), which exactly undoes the patchify. The output has the same shape as x, as every DeltaFlow
            velocity field must. Output columns group R, G, B for each pixel, unlike the channel-first
            patch layout shown at the input.</p>
            <p><b>Hover</b> a token row or a patch of v to see where each token's output lands.</p>`,
          render(stage) {
            const tr = self.tr();
            const r = self.row(stage);
            let fin, vimg;
            const link = (k) => {
              fin.set(k === null ? [] : [k]);
              vimg.set([], k === null ? null : self.patchRect(k));
            };
            fin = new Heatmap(r, tr.final, { cellW: 16, cellH: 12, rowLabels: [...Array(N).keys()].map(String),
              title: `final_layer output ${shape(`(${N}, ${patchValues})`)}<br>R, G, B per pixel`,
              tipFn: (rr, cc, v) => `token ${rr}, ${["R", "G", "B"][cc % channels]} &rarr; pixel (${Math.floor(rr / g) * p + Math.floor(cc / (p * channels))}, ${(rr % g) * p + (Math.floor(cc / channels) % p)})<br>${fmt(v)}`,
              onHover: (rr) => link(rr) });
            r.appendChild(el("div", "dw-op", "&rarr;"));
            vimg = self.imageView(r, tr.v, `velocity v ${shape(`(${channels}, ${C.img}, ${C.img})`)}`, { cell: 24, grid: p, signed: true,
              tipFn: (rr, cc, v) => `pixel (${rr}, ${cc}) from token ${self.tokenAt(rr, cc)}<br>v = ${fmt(v)}`,
              onHover: (rr, cc) => link(rr === null ? null : self.tokenAt(rr, cc)) });
          },
        },
        {
          title: "Read the output as a velocity, not a clean image",
          shape: () => `v ${shape(`(B, ${channels}, ${C.img}, ${C.img})`)} &asymp; x\u2081 &minus; x\u2080 &nbsp; and &nbsp; x\u0302\u2081 = x_t + (1 &minus; t) v`,
          text: () => {
            const mis = self.s.li !== 2 && self.s.li !== self.s.img;
            return `<p>Flow matching trains v to regress the displacement <b>&Delta; = x\u2081 &minus; x\u2080</b> between
              the noise and data distributions. Following the predicted velocity for the remaining time
              (1 &minus; t) gives a one-step estimate of the clean image x\u0302\u2081.</p>
              <p>At t = 0 the input holds no information about x\u2081, so only the label can set the direction.
              By t = 0.9 the image itself gives the answer away.` +
              (mis ? ` <b>The label currently contradicts the image.</b> Try each t and watch which one wins.` : "") + `</p>`;
          },
          render(stage) {
            const tr = self.tr();
            const x0 = self.d.x0, x1 = self.d.x1[self.s.img];
            const delta = map2(x1, x0, (a, b) => a - b);
            const lim = absMax(tr.v, delta);
            const xhat = map2(tr.x_t, tr.v, (a, b) => a + (1 - self.t) * b);
            const err = flat(map2(tr.v, delta, (a, b) => (a - b) ** 2)).reduce((s, v) => s + v, 0) / (channels * C.img * C.img);
            const r = self.row(stage);
            self.imageView(r, tr.x_t, `x_t (t = ${self.t})`);
            self.imageView(r, tr.v, `predicted v (${self.labelName()})`, { cell: 24, lim, signed: true });
            self.imageView(r, delta, "target \u0394 = x\u2081 &minus; x\u2080", { cell: 24, lim, signed: true });
            self.imageView(r, xhat, "x\u0302\u2081 = x_t + (1 &minus; t) v");
            r.appendChild(el("div", "dw-chip", `MSE(v, &Delta;)<br><b>${err.toFixed(3)}</b>`));
          },
        },
        {
          title: "Classifier-free guidance, extrapolating away from the null token",
          shape: () => `v_cfg = v_null + s (v_cond &minus; v_null)`,
          text: () => `<p>Because the same network knows both the conditional velocity (real label) and the
            unconditional one (null token), guidance can <b>exaggerate</b> the label's effect.
            s = 0 is unconditional, s = 1 is the plain conditional model, s &gt; 1 pushes further.
            <code>DiT.forward_with_cfg</code> computes both terms in one batched pass. The arithmetic here runs
            live in your browser on the exported velocities.</p>
            <p>Best viewed at <b>t = 0</b>, where the label is the only source of class information.</p>`,
          render(stage) {
            const cond = self.s.li === 2 ? 0 : self.s.li;
            const vc = self.tr(cond).v, vn = self.tr(2).v;
            const ctl = self.row(stage);
            const lab = el("label", "dw-slider", `guidance scale s = <b>${self.s.cfgS.toFixed(1)}</b> (guiding towards y = ${cond}) `);
            const inp = el("input");
            Object.assign(inp, { type: "range", min: 0, max: 5, step: 0.1, value: self.s.cfgS });
            lab.appendChild(inp);
            ctl.appendChild(lab);
            const r = self.row(stage);
            const lim = absMax(vc, vn) * 2.5;
            const mk = () => {
              const s = self.s.cfgS;
              const v = map2(vn, vc, (a, b) => a + s * (b - a));
              return [v, map2(self.tr(cond).x_t, v, (a, b) => a + (1 - self.t) * b)];
            };
            self.imageView(r, vn, "v_null", { cell: 22, lim, signed: true });
            self.imageView(r, vc, `v_cond (y = ${cond})`, { cell: 22, lim, signed: true });
            r.appendChild(el("div", "dw-op", "&rarr;"));
            let [v, xh] = mk();
            const hv = self.imageView(r, v, "v_cfg", { cell: 22, lim, signed: true });
            const hx = self.imageView(r, xh, "x\u0302\u2081 under v_cfg", { cell: 22 });
            inp.addEventListener("input", () => {
              self.s.cfgS = parseFloat(inp.value);
              lab.querySelector("b").textContent = self.s.cfgS.toFixed(1);
              [v, xh] = mk();
              hv.mat = v; hv.draw();
              hx.mat = xh; hx.draw();
            });
          },
        },
        {
          title: "Sampling, integrating the velocity from noise to data",
          shape: () => `x \u2190 x + (1/${self.d.sample_steps}) v(x, t), &nbsp; t = 0, 1/${self.d.sample_steps}, &hellip;, 1`,
          text: () => `<p>Sampling repeats the whole forward pass at every step of an Euler solver, starting from the
            same noise x\u2080. Each frame below is the state after one more step. Pick a guidance scale and press
            play. With s = 0 (null token only) the model has no reason to prefer either class. As s grows the
            class becomes cleaner and more saturated.</p>
            <p>The label used is the one selected above (null is treated as y = 0, guided with s = 0 it is
            the same as the null token).</p>`,
          render(stage) {
            const lab = self.s.li === 2 ? 0 : self.s.li;
            const ctl = self.row(stage);
            buttons(ctl, "guidance s", self.d.cfg_values.map((s, i) => [i, String(s)]), self.s.sampleCfg,
                    (v) => self.set({ sampleCfg: v, frame: self.d.sample_steps }));
            const frames = self.d.sampling[`${lab}|${self.s.sampleCfg}`];
            const play = el("button", "dw-btn", "&#9654; play");
            const slider = el("input");
            Object.assign(slider, { type: "range", min: 0, max: frames.length - 1, step: 1, value: self.s.frame });
            const info = el("span", "dw-glabel");
            ctl.append(play, slider, info);
            const r = self.row(stage);
            const big = self.imageView(r, frames[self.s.frame], "current state", { cell: 30 });
            const strip = el("div", "dw-strip");
            r.appendChild(strip);
            const thumbs = [];
            for (let i = 0; i < frames.length; i += 4)
              thumbs.push([i, self.imageView(strip, frames[i], `t = ${(i / self.d.sample_steps).toFixed(1)}`, { cell: 9 })]);
            const show = (i) => {
              self.s.frame = i;
              slider.value = i;
              info.innerHTML = `t = ${(i / self.d.sample_steps).toFixed(2)}`;
              big.mat = frames[i];
              big.draw();
              thumbs.forEach(([k, h]) => h.fig.classList.toggle("is-active", k <= i && i < k + 4));
            };
            slider.addEventListener("input", () => show(parseInt(slider.value, 10)));
            play.addEventListener("click", () => {
              if (self.timer) clearInterval(self.timer);
              let i = 0;
              show(0);
              self.timer = setInterval(() => {
                i += 1;
                if (i >= frames.length) {
                  clearInterval(self.timer);
                  self.timer = null;
                  return;
                }
                show(i);
              }, 140);
            });
            show(self.s.frame);
          },
        },
      ];
      const blocks = Array.from({ length: C.depth }, (_, b) => ({
        short: `Block ${b}`,
        block: b,
        title: `Transformer block ${b}, ${b === 0 ? "the first token updates" : b === C.depth - 1 ? "the last token updates" : "refining the token features"}`,
        shape: () => `${shape(`(1, ${N}, ${D})`)} &rarr; attention + residual &rarr; MLP + residual &rarr; ${shape(`(1, ${N}, ${D})`)}`,
        text: () => `<p>This is <code>blocks[${b}]</code> in the real model. Follow the five operations above in order.
          The shape stays fixed, but the feature values change. The same conditioning vector <b>c</b>
          drives every block, through separately learned modulation weights.</p>`,
        render: (stage) => self.blockView(stage, b, details),
      }));
      const final = {
        short: "Output head",
        title: "The final layer, feature channels become patch pixels",
        shape: () => `${shape(`(1, ${N}, ${D})`)} &rarr; final adaLN &rarr; Linear(${D}, ${patchValues}) &rarr; ${shape(`(1, ${N}, ${patchValues})`)}`,
        text: () => `<p>After the last transformer block, the final LayerNorm is modulated by a new shift and scale from <b>c</b>.
          A learned Linear projects each ${D}-channel token into ${patchValues} velocity values, one for each RGB component of each pixel of its patch.
          This is a learned projection, not a reshape. The following unpatchify operation only rearranges these values.</p>`,
        render(stage) {
          const tr = self.tr();
          self.tensorSequence(stage, [
            ["Last block output", tr.block_states[C.depth - 1].output],
            ["Final LayerNorm + modulation", tr.final_input],
            ["Linear to patch velocities", tr.final],
          ]);
        },
      };
      const main = [details[0], details[1], details[2], details[3], details[4],
        ...blocks, final, { ...details[8], title: "Unpatchify, assemble the output velocity image" },
        details[9], details[10], details[11]];
      const labels = ["Input image", "Patch embed", "Position", "Time", "Label + time",
        ...blocks.map((b) => b.short), "Output head", "Unpatchify", "Read velocity", "Guidance", "Sampling"];
      return main.map((step, i) => ({ ...step, short: labels[i] }));
    }
  }

  // ------------------------------------------------------------------ boot
  let active = null;
  function boot() {
    const root = document.getElementById("dit-viewer");
    if (active && active.root !== root) {
      clearInterval(active.timer);
      active = null;
      hideTip();
    }
    if (!root || root.dataset.ready) return;
    root.dataset.ready = "1";
    root.innerHTML = '<p class="dw-loading">Loading the DiT trace&hellip;</p>';
    fetch(root.dataset.src)
      .then((r) => {
        if (!r.ok) throw new Error(r.status);
        return r.json();
      })
      .then((d) => {
        if (!root.isConnected) return;
        if (d.schema_version !== 3 || d.config.channels !== 3) throw new Error("The RGB trace needs regenerating for this viewer");
        active = new Walkthrough(root, d);
      })
      .catch((e) => {
        root.innerHTML = "";
        const message = el("p", "dw-loading");
        message.setAttribute("role", "alert");
        message.textContent = `Could not load the trace (${e.message}). Regenerate dit_trace.json with the showcase script and copy it to the docs assets.`;
        root.appendChild(message);
      });
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(boot);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
