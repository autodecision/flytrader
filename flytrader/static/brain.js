/* The renderer only displays measured reservoir frames. No generated activity. */
const FLY_DEFAULT_YAW = Math.PI / 2;
const FLY_DEFAULT_PITCH = -.16;

function activityScale(values, setting) {
  if (setting !== 'auto') return Number(setting);
  const magnitudes = values.map(Math.abs).sort((a, b) => a - b);
  // A few unusually strong cells should not hide the rest. Keep quantization noise faint.
  return Math.max(.001, magnitudes[Math.floor((magnitudes.length - 1) * .99)] || 0);
}

/* Color helpers for 3D shading. */
function lighten(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.min(255, (n >> 16) + amt), g = Math.min(255, ((n >> 8) & 255) + amt), b = Math.min(255, (n & 255) + amt);
  return `rgb(${r},${g},${b})`;
}
function darken(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.max(0, (n >> 16) - amt), g = Math.max(0, ((n >> 8) & 255) - amt), b = Math.max(0, (n & 255) - amt);
  return `rgb(${r},${g},${b})`;
}

function activityChanges(previous, frame) {
  if (!previous || previous.mint !== frame.mint || previous.step >= frame.step || previous.values.length !== frame.values.length) {
    return {count: null, comparedStep: null, changes: []};
  }
  const changes = frame.values.map((value, index) => ({index, delta: value - previous.values[index]}))
    .filter(change => change.delta !== 0).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  return {count: changes.length, comparedStep: previous.step, changes};
}

function activityPulse(ageMs, enabled) {
  if (!enabled || ageMs < 0 || ageMs >= 30000) return 0;
  return .5 + .5 * Math.cos(ageMs * Math.PI * 2 / 2400);
}

class BrainViewer {
  constructor(onProgress, onFrame) {
    this.canvas = document.querySelector('#brain-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.priceCanvas = document.querySelector('#price-canvas');
    this.priceCtx = this.priceCanvas.getContext('2d');
    this.priceSamples = [];
    this.priceUnit = 'recording units';
    this.total = 0;
    this.onProgress = onProgress;
    this.onFrame = onFrame;
    this.graph = null;
    this.frames = [];
    this.frame = null;
    this.selected = -1;
    this.playing = false;
    this.live = false;
    this.report = null;
    this.focus = 'all';
    this.yaw = FLY_DEFAULT_YAW;
    this.pitch = FLY_DEFAULT_PITCH;
    this.zoom = 1;
    this.dirty = true;
    this.projected = [];
    this.lastTick = 0;
    this.lastPoll = 0;
    this.loading = false;
    this.recordingReady = false;
    this.generation = 0;
    this.displayRange = .001;
    this.changedCells = new Set();
    this.activity = {count: null, comparedStep: null, changes: []};
    this.frameReceivedAt = 0; this.lastActivityClock = 0;
    this.mapVisible = false;
    this.flyMode = false;
    this.fly = new DeskFly(() => this.dirty = true);
    this.motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.bind();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvas);
    this.resizeObserver.observe(this.priceCanvas);
    this.visibilityObserver = new IntersectionObserver(entries => { this.mapVisible = entries[0].isIntersecting; });
    this.visibilityObserver.observe(this.canvas);
    requestAnimationFrame(t => this.tick(t));
  }

  q(selector) { return document.querySelector(selector); }
  setText(selector, text) {
    const node = this.q(selector);
    if (node && node.textContent !== String(text)) node.textContent = text;
  }

  bind() {
    let drag = null;
    this.canvas.addEventListener('pointerdown', e => {
      drag = {x: e.clientX, y: e.clientY, moved: 0};
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener('pointermove', e => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.moved += Math.abs(dx) + Math.abs(dy);
      this.yaw += dx * .007;
      this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch + dy * .007));
      drag.x = e.clientX; drag.y = e.clientY; this.dirty = true;
    });
    this.canvas.addEventListener('pointerup', e => {
      if (drag && drag.moved < 5) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left, y = e.clientY - rect.top;
        let nearest = null, distance = 14;
        this.projected.forEach((p, i) => {
          if (!p) return;
          const d = Math.hypot(p[0] - x, p[1] - y);
          if (d < distance) { distance = d; nearest = i; }
        });
        if (nearest !== null) this.select(nearest);
      }
      drag = null;
    });
    this.canvas.addEventListener('pointercancel', () => { drag = null; });
    this.canvas.addEventListener('wheel', e => {
      e.preventDefault(); this.zoom = Math.max(.25, Math.min(8, this.zoom * Math.exp(-e.deltaY * .001))); this.dirty = true;
    }, {passive: false});
    this.canvas.addEventListener('keydown', e => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-'].includes(e.key)) return;
      e.preventDefault();
      if (e.key === 'ArrowLeft') this.yaw -= .1;
      if (e.key === 'ArrowRight') this.yaw += .1;
      if (e.key === 'ArrowUp') this.pitch -= .1;
      if (e.key === 'ArrowDown') this.pitch += .1;
      if (['+', '='].includes(e.key)) this.zoom = Math.min(8, this.zoom * 1.1);
      if (e.key === '-') this.zoom = Math.max(.25, this.zoom / 1.1);
      this.dirty = true;
    });
    this.q('#focus-subset').onclick = () => this.setFocus('subset');
    this.q('#focus-all').onclick = () => this.setFocus('all');
    this.q('#reset-camera').onclick = () => {
      this.yaw = this.flyMode ? FLY_DEFAULT_YAW : 0;
      this.pitch = this.flyMode ? FLY_DEFAULT_PITCH : 0;
      this.zoom = 1; this.dirty = true;
    };
    for (const id of ['#show-context', '#show-links']) this.q(id).onchange = () => { this.dirty = true; };
    this.q('#color-range').onchange = () => { this.updateColorScale(); this.dirty = true; };
    this.q('#animate-activity').onchange = () => { this.updateAnimationStatus(); this.dirty = true; };
    this.q('#neuron-search').oninput = () => this.fillNeurons();
    this.q('#neuron-select').onchange = e => this.select(e.target.value === '' ? -1 : Number(e.target.value));
    this.q('#frame-slider').oninput = e => {
      this.playing = false;
      this.setFrame(this.frames[Number(e.target.value)]);
      this.updatePlayback();
    };
    this.q('#play-activity').onclick = () => {
      if (!this.frames.length || this.live) return;
      if (Number(this.q('#frame-slider').value) >= this.frames.length - 1) {
        this.q('#frame-slider').value = 0;
        this.setFrame(this.frames[0]);
      }
      this.playing = !this.playing;
      this.updatePlayback();
    };
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    this.width = rect.width; this.height = rect.height;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const priceRect = this.priceCanvas.getBoundingClientRect();
    this.priceCanvas.width = Math.round(priceRect.width * dpr);
    this.priceCanvas.height = Math.round(priceRect.height * dpr);
    this.priceCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.dirty = true;
  }

  async showGraph(path) {
    const generation = ++this.generation;
    this.live = false; this.report = null; this.playing = false;
    this.frames = []; this.frame = null;
    this.resetActivity();
    this.priceSamples = []; this.total = 0;
    this.graph = null; this.dirty = true;
    this.recordingReady = false;
    this.updatePlayback();
    this.setText('#map-node-count', '—');
    this.setText('#active-count', '—');
    this.setText('#brain-status', 'Loading cell positions…');
    try {
      const response = await fetch('/api/brain?edges=' + encodeURIComponent(path));
      const graph = await response.json();
      if (!response.ok) throw new Error(graph.error);
      if (generation !== this.generation) return;
      this.setGraph(graph);
      this.setText('#brain-status', 'Ready — run a paper test to see activity');
      this.setText('#brain-step', 'No recorded activity');
      this.setText('#brain-clock', 'Each frame corresponds to one processed price observation.');
      this.setText('#brain-price', '');
      this.updatePlayback();
    } catch (error) {
      if (generation !== this.generation) return;
      this.graph = null; this.dirty = true;
      this.setText('#brain-status', error.message);
    }
  }

  setGraph(graph) {
    this.graph = graph;
    this.selected = -1;
    this.frame = null;
    this.resetActivity();
    this.setText('#top-neurons', 'Run a test to see activity.');
    const totalNodes = graph.model_neurons || graph.nodes.length;
    this.setText('#map-node-count', totalNodes.toLocaleString());
    this.setText('#active-count', '—');
    this.setText('#brain-description', graph.sampled ? `All ${totalNodes.toLocaleString()} neurons are computed. Colors show a fixed sample of ${graph.nodes.length.toLocaleString()} cells.` : graph.layout === 'anatomical' ? 'Cell-body positions from MaleCNS. Colors come from the running model.' : 'Schematic layout. Anatomical positions are unavailable for this graph.');
    if (graph.sample_strategy?.startsWith('Dopamine')) this.setText('#brain-description', `All ${totalNodes.toLocaleString()} neurons compute. The ${graph.nodes.length.toLocaleString()} displayed cells include the PAM, PPL1 and MBON populations. Colors show actual model states.`);
    this.setText('#brain-layout-label', graph.layout === 'anatomical' ? `${graph.located.toLocaleString()} anatomical positions · ${graph.nodes.length.toLocaleString()} cells displayed` : 'Schematic · not anatomical');
    this.setText('#unlocated-note', graph.unlocated ? `${graph.unlocated} displayed cell${graph.unlocated === 1 ? '' : 's'} without an anatomical position. Their recorded values are available in this list.` : 'Every displayed cell has an anatomical position.');
    this.setText('#brain-context-note', graph.context.length ? `${graph.context.length.toLocaleString()} background cell bodies shown.` : '');
    this.q('#focus-all').disabled = !graph.context.length;
    // Fly Mode needs real anatomy to wrap the body around; schematic layouts keep the plain view.
    this.flyMode = this.focus === 'all' && graph.layout === 'anatomical';
    this.fly.fit = null;   // re-measure for the new graph
    this.q('#fly-mode-note').hidden = !this.flyMode;
    this.q('#neuron-search').value = '';
    this.fillNeurons();
    this.setFocus(graph.context.length && graph.layout === 'anatomical' ? 'all' : 'subset');
    this.inspect();
  }

  setFocus(mode) {
    this.focus = mode; this.zoom = 1;
    this.q('#focus-subset').classList.toggle('selected', mode === 'subset');
    this.q('#focus-all').classList.toggle('selected', mode === 'all');
    this.flyMode = mode === 'all' && this.graph?.layout === 'anatomical';
    this.fly?.setEnabled?.(this.flyMode);
    this.q('#fly-mode-note').hidden = !this.flyMode;
    // Each mode stages its own default camera so orbiting one view never
    // skews the other. Fly View starts head-on in the office; Neural Map keeps
    // the dorsal inspection camera.
    if (this.flyMode) { this.yaw = FLY_DEFAULT_YAW; this.pitch = FLY_DEFAULT_PITCH; }
    else { this.yaw = 0; this.pitch = 0; }
    this.dirty = true;
  }

  fillNeurons() {
    const select = this.q('#neuron-select'); select.replaceChildren();
    const none = document.createElement('option');
    none.value = ''; none.textContent = 'None'; select.append(none);
    if (!this.graph) return;
    const query = this.q('#neuron-search').value.toLowerCase();
    this.graph.nodes.forEach((n, i) => {
      if (!`${n.id} ${n.type}`.toLowerCase().includes(query)) return;
      const option = document.createElement('option');
      option.value = i; option.textContent = `${n.type} · ${n.id}${n.position ? '' : ' · unplaced'}`;
      select.append(option);
    });
    if (this.selected >= 0 && [...select.options].some(o => o.value !== '' && Number(o.value) === this.selected)) select.value = String(this.selected);
    else { this.selected = -1; select.value = ''; }
    this.inspect(); this.dirty = true;
  }

  select(index) {
    if (index === '' || index == null || Number(index) < 0) {
      this.selected = -1;
      this.q('#neuron-select').value = '';
      this.inspect(); this.dirty = true;
      return;
    }
    index = Number(index);
    this.selected = index;
    if (![...this.q('#neuron-select').options].some(o => o.value !== '' && Number(o.value) === index)) {
      this.q('#neuron-search').value = ''; this.fillNeurons();
    }
    this.q('#neuron-select').value = index;
    this.inspect(); this.dirty = true;
  }

  inspect() {
    const node = this.graph?.nodes[this.selected];
    if (!node) {
      this.setText('#neuron-type', 'No neuron selected');
      for (const id of ['#neuron-id', '#neuron-side', '#neuron-position', '#neuron-state']) this.setText(id, '—');
      this.q('#neuron-state').dataset.sign = 0;
      return;
    }
    this.setText('#neuron-type', node.type);
    this.setText('#neuron-id', node.id);
    this.setText('#neuron-side', node.side);
    this.setText('#neuron-position', node.position ? node.position.map(v => v.toFixed(1)).join(', ') + ' µm' : 'Not available');
    this.setText('#neuron-state', this.frame ? this.frame.values[this.selected].toFixed(6) : '—');
    this.q('#neuron-state').dataset.sign = this.frame ? Math.sign(this.frame.values[this.selected]) : 0;
  }

  async attach(reportPath, live = false) {
    ++this.generation;
    this.report = reportPath; this.live = live; this.playing = false; this.frames = []; this.frame = null;
    this.resetActivity();
    this.priceSamples = []; this.total = 0; this.priceUnit = 'recording units';
    this.trialSource = null;
    this.recordingReady = false;
    this.dirty = true;
    this.inspect();
    this.setText('#active-count', '—');
    this.setText('#top-neurons', 'Waiting for recorded activity…');
    this.setText('#brain-step', 'No observations processed yet');
    this.setText('#brain-price', '');
    this.setText('#brain-status', live ? 'Starting model…' : 'Loading recorded activity…');
    this.updatePlayback();
    await this.loadRecording();
  }

  async loadRecording() {
    const report = this.report;
    const generation = this.generation;
    try {
      const response = await fetch('/api/activity?path=' + encodeURIComponent(report));
      const data = await response.json();
      if (generation !== this.generation) return;
      if (!response.ok) throw new Error(data.error);
      if (data.available === false) {
        if (!this.live) this.setText('#brain-status', 'This older report has no activity recording. Run a new paper test to record it.');
        return;
      }
      this.setGraph(data.graph);
      this.recordingReady = true;
      this.frames = data.frames;
      this.total = data.total;
      this.priceUnit = data.recording?.price_unit || 'recording units';
      this.trialSource = data.trial_source;
      this.recordedStatus = data.status;
      if (!this.live && data.status === 'running') {
        const latest = await fetch('/api/activity?path=' + encodeURIComponent(report) + '&live=1').then(r => r.json());
        if (generation !== this.generation) return;
        if (latest.frame && this.frames.at(-1)?.step !== latest.frame.step) this.frames.push(latest.frame);
      }
      if (data.status !== 'running') this.live = false;
      if (this.frames.length) {
        this.q('#frame-slider').max = this.frames.length - 1;
        this.q('#frame-slider').value = this.frames.length - 1;
        this.setFrame(this.frames.at(-1));
      }
      if (!this.live) this.setText('#brain-status', data.status === 'completed' ? 'Recorded activity — play or drag the timeline' : 'Partial activity — the run was interrupted');
      this.updatePlayback();
    } catch (error) {
      if (generation === this.generation) this.setText('#brain-status', error.message);
    }
  }

  async poll() {
    if (!this.live || this.loading || !this.report) return;
    this.loading = true;
    const generation = this.generation;
    try {
      if (!this.recordingReady) await this.loadRecording();
      if (generation !== this.generation || !this.recordingReady || !this.live) return;
      const response = await fetch('/api/activity?path=' + encodeURIComponent(this.report) + '&live=1');
      const data = await response.json();
      if (generation !== this.generation) return;
      if (response.ok && data.frame && this.frame?.step !== data.frame.step) {
        this.setFrame(data.frame);
        this.onProgress?.(data.frame.step / data.total * 100, data.frame.phase);
      }
      if (data.status && data.status !== 'running') { this.live = false; await this.loadRecording(); }
    } catch (error) {
      if (generation === this.generation) this.setText('#brain-status', 'Waiting for activity data…');
    }
    finally { this.loading = false; }
  }

  async finish() {
    this.live = false;
    await this.loadRecording();
    this.updatePlayback();
  }

  resetActivity() {
    this.activity = {count: null, comparedStep: null, changes: []};
    this.changedCells.clear(); this.frameReceivedAt = 0;
    this.setText('#activity-observation', 'Waiting for a model update');
    this.setText('#activity-freshness', 'The map advances when a new price is processed.');
    this.setText('#activity-changes', 'No comparison yet');
    this.setText('#activity-comparison', 'Changes appear after two observations.');
    this.updateColorScale();
    this.updateAnimationStatus();
  }

  updateColorScale() {
    const setting = this.q('#color-range').value;
    this.displayRange = activityScale(this.frame?.values || [], setting);
    this.setText('#activity-scale', `${setting === 'auto' ? 'Auto contrast' : 'Fixed contrast'} · full color at ±${this.displayRange.toPrecision(2)}. Values in the inspector are unchanged.`);
  }

  updateActivityClock(now) {
    if (!this.frame) return;
    const age = Math.max(0, Math.floor((now - this.frameReceivedAt) / 1000));
    this.setText('#activity-freshness', this.live
      ? `${age > 30 ? 'Waiting for fresh data · last update' : 'Latest update'} received ${age}s ago. One update per new price.`
      : `Recorded at ${new Date(this.frame.timestamp * 1000).toLocaleTimeString()}.`);
    this.updateAnimationStatus(now);
  }

  animationActive(now = performance.now()) {
    return Boolean(this.frame && (this.live || this.playing) && this.changedCells.size &&
      this.q('#animate-activity').checked && !this.motionPreference.matches && now - this.frameReceivedAt < 30000);
  }

  updateAnimationStatus(now = performance.now()) {
    this.setText('#activity-animation-status', !this.frame ? 'Waiting for measured changes.'
      : this.motionPreference.matches ? 'Reduced motion is on; change highlights stay still.'
      : !this.q('#animate-activity').checked ? 'Animation paused; new values still update.'
      : this.activity.count === 0 ? 'No measured changes: the map stays still.'
      : this.animationActive(now) ? 'Rings repeat the latest measured change between price updates.'
      : this.live ? 'Waiting for another model update.' : 'Play the recorded activity to animate its changes.');
  }

  setFrame(frame) {
    if (!frame || !this.graph || frame.values.length !== this.graph.nodes.length) return;
    const previous = this.live ? this.frame : this.frames.findLast(candidate => candidate.step < frame.step);
    this.activity = activityChanges(previous, frame);
    const positions = this.positions();
    this.changedCells = new Set(this.activity.changes.filter(change => positions[change.index]).slice(0, 40).map(change => change.index));
    this.frameReceivedAt = performance.now();
    this.frame = frame;
    this.fly?.onObservation?.();
    this.updateColorScale();
    this.setText('#activity-observation', `Observation ${frame.step} processed`);
    this.setText('#activity-changes', this.activity.count === null ? 'First displayed update'
      : `${this.activity.count.toLocaleString()} / ${frame.values.length.toLocaleString()} displayed values changed`);
    this.setText('#activity-comparison', this.activity.count === null ? 'Waiting for another observation to compare.'
      : `Compared with observation ${this.activity.comparedStep}.${this.activity.count === 0 ? ' The saved states are unchanged.' : ' Rings mark the largest changes with known positions.'}`);
    this.updateActivityClock(this.frameReceivedAt);
    this.onFrame?.(frame);
    this.setText('#active-count', (frame.active_count ?? frame.values.filter(v => Math.abs(v) > .05).length).toLocaleString());
    this.setText('#brain-step', `Observation ${frame.step} · ${frame.phase}`);
    this.setText('#brain-clock', new Date(frame.timestamp * 1000).toLocaleString());
    this.setText('#brain-price', `Input price: ${Number(frame.price).toPrecision(6)} ${this.priceUnit}`);
    if (this.live && this.priceSamples.at(-1)?.step !== frame.step) this.priceSamples.push({step: frame.step, price: frame.price});
    if (this.live) this.setText('#brain-status', 'Model running — showing the latest computed state');
    this.inspect(); this.dirty = true;
    const top = frame.values.map((v, i) => [i, v]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 4);
    const list = this.q('#top-neurons'); list.replaceChildren();
    for (const [i, v] of top) {
      const button = document.createElement('button');
      button.textContent = `${this.graph.nodes[i].type} · ${this.graph.nodes[i].id}     ${v.toFixed(6)}`;
      button.onclick = () => this.select(i); list.append(button);
    }
  }

  updatePlayback() {
    this.q('#play-activity').disabled = !this.frames.length || this.live;
    this.q('#frame-slider').disabled = !this.frames.length || this.live;
    this.q('#play-activity').textContent = this.playing ? 'Pause' : 'Play activity';
    this.updateAnimationStatus();
  }

  positions() {
    if (this.graph.layout === 'anatomical') return this.graph.nodes.map(n => n.position);
    return this.graph.nodes.map((_, i) => {
      const a = i * 2.399963; const radius = Math.sqrt((i + 1) / this.graph.nodes.length) * 200;
      return [Math.cos(a) * radius, Math.sin(a) * radius * .25, Math.sin(a) * radius];
    });
  }

  drawPrices() {
    const ctx = this.priceCtx;
    const rect = this.priceCanvas.getBoundingClientRect();
    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);
    const points = this.frames.length ? this.frames : this.priceSamples;
    if (!points.length) {
      ctx.fillStyle = '#8c9e85'; ctx.font = '10px system-ui';
      ctx.fillText('Input prices appear here as the model processes them.', 0, h / 2);
      return;
    }
    const prices = points.map(p => p.price);
    const low = Math.min(...prices), high = Math.max(...prices);
    const x = step => 4 + (w - 8) * (step - 1) / Math.max(1, this.total - 1);
    const y = price => high === low ? h / 2 : 6 + (h - 12) * (1 - (price - low) / (high - low));
    ctx.strokeStyle = '#829778'; ctx.lineWidth = 1.3; ctx.beginPath();
    points.forEach((point, i) => { if (i) ctx.lineTo(x(point.step), y(point.price)); else ctx.moveTo(x(point.step), y(point.price)); });
    ctx.stroke();
    if (this.frame) {
      const xx = x(this.frame.step), yy = y(this.frame.price);
      ctx.strokeStyle = '#d8af7f'; ctx.lineWidth = .7; ctx.beginPath(); ctx.moveTo(xx, 0); ctx.lineTo(xx, h); ctx.stroke();
      ctx.fillStyle = '#efb07d'; ctx.beginPath(); ctx.arc(xx, yy, 3, 0, Math.PI * 2); ctx.fill();
    }
  }

  draw() {
    const {ctx, width: w, height: h} = this;
    if (!w || !h) return;
    ctx.clearRect(0, 0, w, h);
    this.drawPrices();
    if (!this.graph) { ctx.fillStyle = '#899794'; ctx.font = '13px system-ui'; ctx.fillText('Loading neuron map…', 25, h / 2); return; }
    const positions = this.positions();
    const boundsPoints = (this.focus === 'all' ? [...this.graph.context, ...positions] : positions).filter(Boolean);
    if (!boundsPoints.length) return;
    const limits = [0, 1, 2].map(k => [Math.min(...boundsPoints.map(p => p[k])), Math.max(...boundsPoints.map(p => p[k]))]);
    const center = limits.map(([a, b]) => (a + b) / 2);
    const span = Math.max(...limits.map(([a, b]) => b - a), 1);
    const scale = Math.min(w * .85, h * .84) / span * this.zoom;
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const project = p => {
      const x = p[0] - center[0], y = p[2] - center[2], z = p[1] - center[1];
      const xx = x * cy + z * sy, zz = -x * sy + z * cy;
      return [w / 2 + xx * scale, h / 2 + (y * cp - zz * sp) * scale, y * sp + zz * cp];
    };
    ctx.fillStyle = '#667873';
    // Background context and links use the shared map projection, which does
    // not match the Fly Mode side-view camera — drawing them there leaves a
    // cloud hanging off the body. Fly Mode shows the anatomy through the
    // cutaway windows instead; the toggles still apply to Neural Map.
    if (!this.flyMode && this.q('#show-context').checked) {
      ctx.globalAlpha = .20;
      for (const point of this.graph.context) { const p = project(point); ctx.fillRect(p[0], p[1], 1.3, 1.3); }
      ctx.globalAlpha = 1;
    }
    this.projected = positions.map(p => p ? project(p) : null);
    if (!this.flyMode && this.q('#show-links').checked) {
      ctx.strokeStyle = '#d2a17b'; ctx.lineWidth = .5; ctx.globalAlpha = .075;
      ctx.beginPath();
      for (const [a, b] of this.graph.edges) {
        const p = this.projected[a], q = this.projected[b];
        if (p && q) { ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); }
      }
      ctx.stroke(); ctx.globalAlpha = 1;
    }
    const range = this.displayRange;
    const pulse = activityPulse(performance.now() - this.frameReceivedAt, this.animationActive());
    if (this.flyMode) return this.fly.drawScene(this, w, h, range, pulse);
    const ordered = this.projected.map((p, i) => [p, i]).filter(([p]) => p).sort((a, b) => a[0][2] - b[0][2]);
    for (const [p, i] of ordered) {
      const v = this.frame?.values[i] || 0;
      const strength = Math.min(1, Math.abs(v) / range);
      const radius = 1.9 + strength * 1.8 + (this.changedCells.has(i) ? pulse * 1.8 : 0);
      const neutral = [158, 170, 160], target = v >= 0 ? [239, 165, 100] : [107, 186, 217];
      const color = neutral.map((n, k) => Math.round(n + (target[k] - n) * strength));
      ctx.fillStyle = `rgba(${color.join(',')},${.45 + strength * .5})`;
      ctx.beginPath(); ctx.arc(p[0], p[1], radius, 0, Math.PI * 2); ctx.fill();
      if (strength > .35) {
        ctx.globalAlpha = .12 * strength;
        ctx.beginPath(); ctx.arc(p[0], p[1], radius * 2.4, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = 1;
      }
      if (this.changedCells.has(i)) {
        ctx.strokeStyle = `rgba(244,235,201,${.3 + pulse * .65})`; ctx.lineWidth = 1 + pulse;
        ctx.beginPath(); ctx.arc(p[0], p[1], radius + 2.5 + pulse * 7, 0, Math.PI * 2); ctx.stroke();
      }
      if (i === this.selected) {
        ctx.strokeStyle = '#f2e9d7'; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(p[0], p[1], radius + 5, 0, Math.PI * 2); ctx.stroke();
      }
    }
    const unplaced = positions.filter(p => !p).length;
    if (unplaced) {
      ctx.fillStyle = '#96a29c'; ctx.font = '11px system-ui'; ctx.fillText(`${unplaced} unplaced cell${unplaced === 1 ? '' : 's'} · inspect from the list`, 18, h - 18);
    }
  }

  tick(t) {
    if (this.live && t - this.lastPoll > 250) { this.lastPoll = t; this.poll(); }
    if (this.playing && t - this.lastTick > Number(this.q('#play-speed').value)) {
      this.lastTick = t;
      let index = Number(this.q('#frame-slider').value) + 1;
      if (index >= this.frames.length) { this.playing = false; this.updatePlayback(); }
      else { this.q('#frame-slider').value = index; this.setFrame(this.frames[index]); }
    }
    if (this.live && t - this.lastActivityClock > 1000) { this.lastActivityClock = t; this.updateActivityClock(t); }
    const animating = this.animationActive(t);
    if (animating !== this.wasAnimating) { this.wasAnimating = animating; this.dirty = true; this.updateAnimationStatus(t); }
    if (this.mapVisible && animating && t - (this.lastPulseDraw || 0) > 40) {
      this.lastPulseDraw = t; this.dirty = true;
    }
    if (this.flyMode && this.mapVisible && this.fly.animating() && t - (this.lastFlyDraw || 0) > 40) {
      this.lastFlyDraw = t; this.dirty = true;
    }
    if (this.dirty) { this.draw(); this.dirty = false; }
    requestAnimationFrame(time => this.tick(time));
  }
}


/* Fly Mode: an opaque stylised fly sitting on a desk, keyboard in front.
   The body is fitted around the *measured* MaleCNS anatomy: the head sphere
   encloses the brain cloud, the thorax/abdomen capsules follow the ventral
   nerve cord, ventral = +y so the fly stands on the desk plane. The scene
   uses its own side-view camera (head left, desk flat, keyboard readable);
   yaw/pitch orbit and tilt it. Cutaway windows show each part's own neurons
   (brain cloud in the head, cord cells in thorax/abdomen); the front legs
   type on the keys intermittently (burst on each processed observation,
   ambient otherwise). Typing is decorative — it does not drive the model. */
class DeskFly {
  constructor(markDirty) {
    this.markDirty = markDirty;
    this.enabled = false;
    this.reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.burstUntil = 0;
    this.burstSeed = Math.random() * 997;
    this.fit = null;          // measured anatomy bounds, computed on first draw
  }

  setEnabled(on) {
    this.enabled = on;
    if (on) this.markDirty();
  }

  onObservation() {
    if (!this.enabled) return;
    // A processed observation triggers an active typing burst.
    this.burstUntil = performance.now() + 2400 + Math.random() * 2000;
    this.markDirty();
  }

  ambientActive(now) {
    // Idle rhythm: ~2.2s of typing inside each ~9s window when no data arrives.
    return (now % 9000) < 2200;
  }

  typing(now) {
    return this.enabled && !this.reduced.matches &&
      (now < this.burstUntil || this.ambientActive(now));
  }

  animating(now = performance.now()) { return this.typing(now); }

  /* Measure the anatomy from the actual displayed/context points so the body
     wraps the real data (works for any graph, not just the full model). */
  measure(viewer) {
    if (this.fit) return this.fit;
    const pts = [...viewer.graph.context, ...viewer.positions().filter(Boolean)];
    if (!pts.length) return null;
    const val = (arr, q) => { const s = [...arr].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))]; };
    const cx = (val(pts.map(p => p[0]), .5) + val(pts.map(p => p[0]), .5)) / 2 || val(pts.map(p => p[0]), .5);
    // Brain cloud = anterior mass (low z); cord = posterior rope (high z).
    const brain = pts.filter(p => p[2] < 420);
    const cord = pts.filter(p => p[2] >= 420);
    const bc = [val(brain.map(p => p[0]), .5), val(brain.map(p => p[1]), .5), val(brain.map(p => p[2]), .5)];
    const dists = brain.map(p => Math.hypot(p[0] - bc[0], (p[1] - bc[1]) * .8, (p[2] - bc[2]) * .8));
    const headR = Math.max(60, val(dists, .96) * 1.12);
    const cordY = cord.length ? val(cord.map(p => p[1]), .5) : bc[1] + headR * .8;
    const zc1 = cord.length ? val(cord.map(p => p[2]), .98) : 500;
    const deskY = val(pts.map(p => p[1]), .995) + 45;
    this.fit = {cx, bc, headR, cordY, zc1, deskY,
                z0: val(pts.map(p => p[2]), 0), z1: val(pts.map(p => p[2]), 1)};
    return this.fit;
  }

  /* Body parts in anatomy space (scaled µm), derived from the measured fit.
     Ventral = +y: the head sits dorsal-anterior, the thorax/abdomen descend
     along the ventral nerve cord, legs reach down to the desk. */
  buildBody(f) {
    const cx = f.cx, h = f.headR;
    const thoraxZ0 = Math.max(360, f.bc[2] + h * .55);
    const thoraxZ1 = Math.max(thoraxZ0 + 140, Math.min(f.zc1, thoraxZ0 + 260));
    const thoraxY = (f.bc[1] + f.cordY) / 2;
    const abdomenZ1 = f.zc1 + h * .3;
    return {
      head: {c: [cx, f.bc[1], f.bc[2]], r: h},
      eyeL: {c: [cx - h * .72, f.bc[1] + h * .25, f.bc[2] - h * .5], r: h * .36, side: -1},
      eyeR: {c: [cx + h * .72, f.bc[1] + h * .25, f.bc[2] - h * .5], r: h * .36, side: 1},
      antennae: [-1, 1].map(s => ({
        a: [cx + s * h * .16, f.bc[1] + h * .38, f.bc[2] - h * .82],
        b: [cx + s * h * .3, f.bc[1] + h * .66, f.bc[2] - h * 1.18], side: s})),
      thorax: {a: [cx, thoraxY - h * .2, thoraxZ0], b: [cx, thoraxY + h * .02, thoraxZ1], r: h * .66},
      abdomen: {a: [cx, f.cordY - h * .08, thoraxZ1 - 30], b: [cx, f.cordY + h * .04, abdomenZ1], r: h * .52},
      wings: [-1, 1].map(s => ({
        base: [cx + s * h * .28, thoraxY - h * .52, thoraxZ0 + 50],
        tip: [cx + s * h * .55, thoraxY - h * 1.1, thoraxZ0 + h * 1.1], side: s})),
      halteres: [-1, 1].map(s => ({
        a: [cx + s * h * .4, thoraxY, thoraxZ1 - 30],
        b: [cx + s * h * .78, thoraxY - h * .08, thoraxZ1 + 20], side: s})),
    };
  }

  keyboardExtent(f) {
    // Board in front of the head (-z), resting on the desk, clear of the head.
    return {halfX: f.headR * 1.55, z0: f.bc[2] - f.headR * 2.2, z1: f.bc[2] - f.headR * .8,
            rows: 4, cols: 12, keyRise: 7};
  }

  /* Which key is "pressed" at time t: deterministic chatter across the board. */
  activeKeys(t) {
    const kb = this.kbLayout;
    if (!kb) return [];
    const n = Math.floor(t * 7 + this.burstSeed);
    const r = Math.abs(Math.sin(n * 12.9898) * 43758.5453) % kb.rows | 0;
    const c = Math.abs(Math.sin(n * 78.233) * 12543.197) % kb.cols | 0;
    return [{row: r, col: c, press: .5 + .5 * Math.sin(t * 21)}];
  }

  keyCenter(kb, fit, row, col) {
    const kbE = this.keyboardExtent(fit);
    const x = fit.cx - kbE.halfX + kbE.halfX * .18 + col * (kbE.halfX * 1.64) / (kbE.cols - 1) + (row % 2) * 9;
    const z = kbE.z0 + kbE.halfX * .12 + row * (kbE.z1 - kbE.z0 - kbE.halfX * .24) / (kbE.rows - 1);
    return [x, kbE.keyRise + fit.deskY, z];
  }

  drawScene(viewer, w, h, range, pulse) {
    const ctx = viewer.ctx;
    const fit = this.measure(viewer);
    if (!fit) return;
    const body = this.buildBody(fit);
    const kbE = this.keyboardExtent(fit);
    this.kbLayout = {rows: kbE.rows, cols: kbE.cols};
    const now = performance.now();
    const typing = this.typing(now);
    const keys = typing ? this.activeKeys(now / 1000) : [];
    // Fly Mode side-view camera. The shared map projection puts the
    // dorsal-ventral axis into depth, so every fly angle stages as a top-down
    // blob. Here anterior-posterior runs along the screen (head left),
    // dorsal-ventral runs vertically (desk flat at the bottom), and lateral
    // runs into depth for sorting. viewer.yaw/pitch act as orbit/tilt angles;
    // zoom still applies. Dragging orbits the fly; reset restores the side view.
    const theta = viewer.yaw, phi = viewer.pitch;
    const nodes = viewer.positions();
    const cloud = [...viewer.graph.context, ...nodes.filter(Boolean)];
    const lo = [0, 1, 2].map(k => Math.min(...cloud.map(p => p[k])));
    const hi = [0, 1, 2].map(k => Math.max(...cloud.map(p => p[k])));
    const cc = lo.map((a, k) => (a + hi[k]) / 2);
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1);
    const scale = Math.min(w * .8, h * .78) / (span * 1.3) * viewer.zoom;
    const ct = Math.cos(theta), st = Math.sin(theta);
    const cp = Math.cos(phi), sp = Math.sin(phi);
    const P = (pt) => {
      if (!pt) return null;
      const dx = pt[0] - cc[0], dy = pt[1] - cc[1], dz = pt[2] - cc[2];
      const sx = dz * ct - dx * st;
      const dep = -(dz * st + dx * ct);
      return [w / 2 + sx * scale, h / 2 + (dy * cp - dep * sp) * scale, dy * sp + dep * cp];
    };
    // Keep click-to-inspect and the selection ring consistent with the staged scene.
    viewer.projected = nodes.map(p => P(p));
    const deskY = fit.deskY;
    const officeFloorY = deskY + fit.headR * 1.65;
    // Warm fruit-fly palette: tan body, red eyes, pale wings. Dark slate made
    // the fly unreadable against the page background.
    const headChitin = '#8a6d4f', headEdge = '#b89a74';
    const thoraxChitin = '#7a6247', thoraxEdge = '#9a8262';
    const abdomenChitin = '#6b5439', limbChitin = '#5a4a36', bristleChitin = '#362c22';
    const eyeFill = '#c4443a', eyeEdge = '#e87868';
    const windowRim = 'rgba(200,170,130,.9)';

    /* --- 1. Office + desk: projected together in the fly's 3D world --- */
    // The room uses the same projection as the specimen. Orbiting therefore
    // moves the window, shelves, walls, floor, and desk as one office scene.
    {
      ctx.fillStyle = '#07100d';
      ctx.fillRect(0, 0, w, h);
      const hr = fit.headR;
      const room = {
        x0: fit.cx - hr * 5.4, x1: fit.cx + hr * 5.4,
        y0: deskY - hr * 5.1, y1: officeFloorY,
        z0: kbE.z0 - hr * 2.4, z1: body.abdomen.b[2] + hr * 3.25,
      };
      const officeParts = [];
      const officeLayerCutoff = 1000;
      const addOfficePoly = (points, fill, stroke = null, width = 1, bias = 0) => {
        const q = points.map(P);
        if (!q.every(Boolean)) return;
        const layer = bias >= officeLayerCutoff ? bias : 0;
        officeParts.push({
          layer,
          depth: q.reduce((sum, p) => sum + p[2], 0) / q.length + (layer ? 0 : bias),
          draw: () => {
            ctx.fillStyle = fill;
            ctx.beginPath();
            q.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
            ctx.closePath(); ctx.fill();
            if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.stroke(); }
          },
        });
      };
      const addOfficeLine = (points, stroke, width = 1, bias = 0) => {
        const q = points.map(P);
        if (!q.every(Boolean)) return;
        const layer = bias >= officeLayerCutoff ? bias : 0;
        officeParts.push({
          layer,
          depth: q.reduce((sum, p) => sum + p[2], 0) / q.length + (layer ? 0 : bias),
          draw: () => {
            ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.lineCap = 'round';
            ctx.beginPath(); q.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
            ctx.stroke();
          },
        });
      };
      const quadXY = (x0, y0, x1, y1, z) => [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]];
      const quadZY = (z0, y0, z1, y1, x) => [[x, y0, z0], [x, y0, z1], [x, y1, z1], [x, y1, z0]];
      // Room planes can differ by hundreds of projected depth units. Office
      // details use a separate high layer so no wall repaints over its decor.
      const decorLayer = 10000;
      const wallFacesCamera = (center, inward) => {
        const c = P(center);
        const towardRoom = P(center.map((value, i) => value + inward[i] * hr));
        return c && towardRoom && (towardRoom[2] - c[2]) / (hr * scale) > .035;
      };

      // Enclosing room planes. Sorting lets the nearest side wall frame the
      // far wall correctly as the camera moves around the fly.
      addOfficePoly(quadXY(room.x0, room.y0, room.x1, room.y1, room.z1), '#13251d', '#294638');
      addOfficePoly(quadZY(room.z0, room.y0, room.z1, room.y1, room.x0), '#0e1d17', '#223b30');
      addOfficePoly(quadZY(room.z0, room.y0, room.z1, room.y1, room.x1), '#102019', '#274235');
      addOfficePoly([[room.x0, room.y1, room.z0], [room.x1, room.y1, room.z0],
                     [room.x1, room.y1, room.z1], [room.x0, room.y1, room.z1]], '#0a1511', '#294638');
      addOfficePoly([[room.x0, room.y0, room.z0], [room.x0, room.y0, room.z1],
                     [room.x1, room.y0, room.z1], [room.x1, room.y0, room.z0]], '#0b1712', '#20372c');

      // Window on the back wall with a shallow skyline beyond it.
      const wz = room.z1 - 5;
      // Offset toward +x so the window stays in view at the default three-
      // quarter angle instead of being pushed beyond the far room corner.
      const wx0 = fit.cx + hr * .6, wx1 = fit.cx + hr * 3.45;
      const wy0 = deskY - hr * 2.9, wy1 = deskY - hr * .78;
      const backWallVisible = wallFacesCamera(
        [fit.cx, (room.y0 + room.y1) / 2, room.z1], [0, 0, -1]
      );
      if (backWallVisible) {
        addOfficePoly(quadXY(wx0, wy0, wx1, wy1, wz), '#09191a', '#416957', 2, decorLayer + 2);
        const city = [
          [0, .7, .14, .3], [.12, .55, .15, .45], [.26, .64, .18, .36],
          [.43, .46, .13, .54], [.55, .61, .18, .39], [.72, .52, .14, .48], [.85, .67, .16, .33],
        ];
        for (let i = 0; i < city.length; i++) {
          const [bx, by, bw, bh] = city[i];
          const x0 = wx0 + (wx1 - wx0) * bx, x1 = x0 + (wx1 - wx0) * bw;
          const y0 = wy0 + (wy1 - wy0) * by;
          addOfficePoly(quadXY(x0, y0, x1, wy1, wz - 3), i % 2 ? '#0b1714' : '#0d1d18', null, 1, decorLayer + 3);
          for (let row = 0; row < 3; row++) for (let col = 0; col < 2; col++) {
            if ((row + col + i) % 3 === 0) continue;
            const px0 = x0 + (x1 - x0) * (.2 + col * .42);
            const py0 = y0 + (wy1 - y0) * (.18 + row * .22);
            addOfficePoly(quadXY(px0, py0, px0 + hr * .035, py0 + hr * .045, wz - 5),
                          'rgba(114,211,145,.38)', null, 1, decorLayer + 4);
          }
        }
        // Mullions are a later layer than every skyline element.
        addOfficeLine([[wx0, wy0, wz - 7], [wx1, wy0, wz - 7], [wx1, wy1, wz - 7],
                       [wx0, wy1, wz - 7], [wx0, wy0, wz - 7]],
                      'rgba(93,151,111,.8)', 2.4, decorLayer + 5);
        addOfficeLine([[wx0 + (wx1 - wx0) / 2, wy0, wz - 7], [wx0 + (wx1 - wx0) / 2, wy1, wz - 7]],
                      'rgba(93,151,111,.65)', 2, decorLayer + 5);
        addOfficeLine([[wx0, wy0 + (wy1 - wy0) * .53, wz - 7], [wx1, wy0 + (wy1 - wy0) * .53, wz - 7]],
                      'rgba(93,151,111,.65)', 2, decorLayer + 5);
      }

      // Shelves live on the right wall so they remain readable when the back
      // window becomes edge-on. Books are shallow wall-mounted boxes.
      const shelfX = room.x1 - 6;
      const shelfZ0 = room.z0 + hr * 1.05, shelfZ1 = room.z0 + hr * 4.25;
      const rightWallVisible = wallFacesCamera(
        [room.x1, (room.y0 + room.y1) / 2, (room.z0 + room.z1) / 2], [-1, 0, 0]
      );
      if (rightWallVisible) {
        for (const sy of [deskY - hr * 3.15, deskY - hr * 1.95]) {
          addOfficeLine([[shelfX, sy, shelfZ0], [shelfX, sy, shelfZ1]], 'rgba(91,139,106,.72)', 3, decorLayer + 6);
        }
        const books = [
          [.08, .62, .22, '#365243', 0], [.33, .82, .18, '#756346', 0], [.55, .5, .27, '#294b3c', 0],
          [.12, .72, .2, '#4a624d', 1], [.38, .55, .25, '#725d43', 1], [.7, .86, .16, '#304c3d', 1],
        ];
        for (const [offset, height, breadth, color, row] of books) {
          const sy = row ? deskY - hr * 1.95 : deskY - hr * 3.15;
          const z0 = shelfZ0 + (shelfZ1 - shelfZ0) * offset;
          const z1 = z0 + (shelfZ1 - shelfZ0) * breadth;
          addOfficePoly(quadZY(z0, sy - hr * height, z1, sy, shelfX - 4), color, null, 1, decorLayer + 7);
        }
      }

      // The clock occupies clear wall space opposite the window and shares
      // the back wall's visibility, so it cannot appear through that wall.
      if (backWallVisible) {
        const clockC = [fit.cx - hr * 2.05, deskY - hr * 2.25, wz - 8];
        const clockR = hr * .38;
        const clockPts = [];
        for (let i = 0; i < 24; i++) {
          const a = i / 24 * Math.PI * 2;
          clockPts.push([clockC[0] + Math.cos(a) * clockR, clockC[1] + Math.sin(a) * clockR, clockC[2]]);
        }
        addOfficePoly(clockPts, '#0c1a15', 'rgba(128,190,146,.68)', 2, decorLayer + 8);
        addOfficeLine([clockC, [clockC[0], clockC[1] - clockR * .55, clockC[2] - 1]], 'rgba(181,220,190,.72)', 1.5, decorLayer + 9);
        addOfficeLine([clockC, [clockC[0] + clockR * .47, clockC[1] + clockR * .2, clockC[2] - 1]], 'rgba(181,220,190,.72)', 1.5, decorLayer + 9);
      }

      // Floor grid and ceiling light reinforce the room's changing perspective.
      for (let i = 1; i < 6; i++) {
        const t = i / 6;
        const x = room.x0 + (room.x1 - room.x0) * t;
        const z = room.z0 + (room.z1 - room.z0) * t;
        addOfficeLine([[x, room.y1 - 2, room.z0], [x, room.y1 - 2, room.z1]], 'rgba(48,80,63,.32)', 1, decorLayer + 1);
        addOfficeLine([[room.x0, room.y1 - 2, z], [room.x1, room.y1 - 2, z]], 'rgba(48,80,63,.32)', 1, decorLayer + 1);
      }
      addOfficePoly([[fit.cx - hr * 1.2, room.y0 + 3, body.head.c[2] - hr * .7],
                     [fit.cx + hr * 1.2, room.y0 + 3, body.head.c[2] - hr * .7],
                     [fit.cx + hr * 1.2, room.y0 + 3, body.head.c[2] + hr * .9],
                     [fit.cx - hr * 1.2, room.y0 + 3, body.head.c[2] + hr * .9]],
                    'rgba(130,207,151,.22)', 'rgba(151,224,170,.32)', 1, decorLayer + 3);

      officeParts.sort((a, b) => a.layer - b.layer || a.depth - b.depth);
      for (const part of officeParts) part.draw();
    }
    const deskZ0 = kbE.z0 - fit.headR * 1.2, deskZ1 = fit.z1 + fit.headR * .5;
    const deskHalfX = fit.headR * 4.15;
    {
      // Four splayed desk legs reach the office floor. Draw them before the
      // desktop so the slab naturally covers each upper attachment point.
      const legSpreadX = deskHalfX * .79;
      const legInsetZ = fit.headR * .48;
      const deskLegs = [];
      for (const side of [-1, 1]) for (const end of [-1, 1]) {
        const z = end < 0 ? deskZ0 + legInsetZ : deskZ1 - legInsetZ;
        const top = P([fit.cx + side * legSpreadX, deskY + fit.headR * .08, z]);
        const bottom = P([fit.cx + side * legSpreadX * 1.035, officeFloorY - fit.headR * .06, z]);
        if (!top || !bottom) continue;
        deskLegs.push({top, bottom, depth: (top[2] + bottom[2]) / 2});
      }
      deskLegs.sort((a, b) => a.depth - b.depth);
      const legWidth = Math.max(5, fit.headR * scale * .17);
      for (const leg of deskLegs) {
        ctx.lineCap = 'butt';
        ctx.strokeStyle = '#09110e'; ctx.lineWidth = legWidth + 4;
        ctx.beginPath(); ctx.moveTo(leg.top[0], leg.top[1]); ctx.lineTo(leg.bottom[0], leg.bottom[1]); ctx.stroke();
        const legGrad = ctx.createLinearGradient(leg.top[0], leg.top[1], leg.bottom[0], leg.bottom[1]);
        legGrad.addColorStop(0, '#355345');
        legGrad.addColorStop(1, '#1a2b23');
        ctx.strokeStyle = legGrad; ctx.lineWidth = legWidth;
        ctx.beginPath(); ctx.moveTo(leg.top[0], leg.top[1]); ctx.lineTo(leg.bottom[0], leg.bottom[1]); ctx.stroke();
        ctx.lineCap = 'round';
        ctx.strokeStyle = '#1a2b23'; ctx.lineWidth = Math.max(3, legWidth * .48);
        ctx.beginPath();
        ctx.moveTo(leg.bottom[0] - legWidth * .48, leg.bottom[1]);
        ctx.lineTo(leg.bottom[0] + legWidth * .48, leg.bottom[1]);
        ctx.stroke();
      }
      // A closed apron joins all four upper attachments beneath the desktop,
      // making the support structure continuous at every camera angle.
      const apron = [
        [fit.cx - legSpreadX, deskY + fit.headR * .11, deskZ0 + legInsetZ],
        [fit.cx + legSpreadX, deskY + fit.headR * .11, deskZ0 + legInsetZ],
        [fit.cx + legSpreadX, deskY + fit.headR * .11, deskZ1 - legInsetZ],
        [fit.cx - legSpreadX, deskY + fit.headR * .11, deskZ1 - legInsetZ],
      ].map(P);
      if (apron.every(Boolean)) {
        ctx.strokeStyle = '#263f34'; ctx.lineWidth = Math.max(4, legWidth * .62);
        ctx.lineJoin = 'round';
        ctx.beginPath();
        apron.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
        ctx.closePath(); ctx.stroke();
      }
      const quad = [[fit.cx - deskHalfX, deskY, deskZ0], [fit.cx + deskHalfX, deskY, deskZ0],
                    [fit.cx + deskHalfX, deskY, deskZ1], [fit.cx - deskHalfX, deskY, deskZ1]];
      const q = quad.map(P);
      if (q.every(Boolean)) {
        ctx.fillStyle = '#1c2823'; ctx.strokeStyle = '#33503f'; ctx.lineWidth = 1;
        ctx.beginPath(); q.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
        ctx.closePath(); ctx.fill(); ctx.stroke();
        // Short front face along the head-end edge so the top reads as a slab.
        const f0 = P([fit.cx - deskHalfX, deskY, deskZ0]), f1 = P([fit.cx + deskHalfX, deskY, deskZ0]);
        const f2 = P([fit.cx + deskHalfX, deskY + fit.headR * .22, deskZ0]);
        const f3 = P([fit.cx - deskHalfX, deskY + fit.headR * .22, deskZ0]);
        if (f0 && f1 && f2 && f3) {
          ctx.fillStyle = '#131d18';
          ctx.beginPath(); ctx.moveTo(f0[0], f0[1]); ctx.lineTo(f1[0], f1[1]);
          ctx.lineTo(f2[0], f2[1]); ctx.lineTo(f3[0], f3[1]); ctx.closePath(); ctx.fill();
        }
      }
      // Soft contact shadow grounding the body + keyboard.
      const sc = P([fit.cx, deskY + 4, (body.abdomen.b[2] + kbE.z0) / 2]);
      const shadowRx = Math.max(10, (body.abdomen.b[2] - kbE.z0) / 2 * scale);
      if (sc) {
        ctx.fillStyle = 'rgba(0,0,0,.35)';
        ctx.beginPath(); ctx.ellipse(sc[0], sc[1], shadowRx, Math.max(4, shadowRx * .2), 0, 0, Math.PI * 2); ctx.fill();
      }
    }
    // Pressed-key spots are collected during the keyboard pass and highlighted
    // on top of the body so the tap stays visible even under a foot or the head.
    const pressedSpots = [];
    {
      const c = [[fit.cx - kbE.halfX, deskY, kbE.z0], [fit.cx + kbE.halfX, deskY, kbE.z0],
                 [fit.cx + kbE.halfX, deskY, kbE.z1], [fit.cx - kbE.halfX, deskY, kbE.z1]];
      const q = c.map(P);
      if (q.every(Boolean)) {
        ctx.fillStyle = '#22282b'; ctx.strokeStyle = '#3b444a'; ctx.lineWidth = 1;
        ctx.beginPath(); q.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
        ctx.closePath(); ctx.fill(); ctx.stroke();
        for (let r = 0; r < kbE.rows; r++) for (let col = 0; col < kbE.cols; col++) {
          const pk = P(this.keyCenter(kbE, fit, r, col));
          if (!pk) continue;
          const pressed = keys.some(k => k.row === r && k.col === col);
          const s = Math.max(6, 15 * scale);
          ctx.fillStyle = pressed ? '#efb07d' : '#3a434a';
          ctx.fillRect(pk[0] - s / 2, pk[1] - s / 2, s, s);
          if (pressed) pressedSpots.push([pk[0], pk[1], s]);
        }
      }
    }

    /* --- 1b. Dual side monitors: a workstation for the fly. One on each
       lateral side (±x), standing on the desk slab roughly level with the
       keyboard row, screens toed in to face the head. Single-sided faces:
       the camera side shows the glowing screen, the far side a solid casing
       back (vents, medallion, stand mount). True 3D quads projected
       with P so they orbit correctly; drawn here in the decor pass (before
       the neural map and body) so they can never cover the glass. */
    {
      const H = fit.headR;
      const zmon = (kbE.z0 + kbE.z1) / 2 + H * .35;   // parallel, a bit back
      const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
      const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
      const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
      const at = (c, u, a, v, b) => [c[0] + u[0] * a + v[0] * b, c[1] + u[1] * a + v[1] * b, c[2] + u[2] * a + v[2] * b];
      for (const s of [-1, 1]) {
        const C = [fit.cx + s * H * 2.9, deskY - H * 1.15, zmon];
        const d = norm(sub(body.head.c, C));          // screen normal → head
        const u = norm(cross([0, -1, 0], d));         // screen-right
        const v = norm(cross(d, u));                  // screen-up
        const W = H * 1.7, ScH = H * 1.3;
        const TL = at(C, u, -W / 2, v, ScH / 2), TR = at(C, u, W / 2, v, ScH / 2);
        const BR = at(C, u, W / 2, v, -ScH / 2), BL = at(C, u, -W / 2, v, -ScH / 2);
        const expand = (pt, keep) => { // scale offset from C by keep
          const o = sub(pt, C); return [C[0] + o[0] * keep, C[1] + o[1] * keep, C[2] + o[2] * keep]; };
        const chin = (pt) => [pt[0] - v[0] * H * .14, pt[1] - v[1] * H * .14, pt[2] - v[2] * H * .14];
        const bezel = [expand(TL, 1.09), expand(TR, 1.09), chin(expand(BR, 1.09)), chin(expand(BL, 1.09))].map(P);
        const screen = [expand(TL, .88), expand(TR, .88), expand(BR, .88), expand(BL, .88)].map(P);
        // Stand neck + foot on the desktop.
        const foot = P([C[0], deskY, zmon]);
        const neckB = P(at(C, u, 0, v, -ScH / 2));
        if (foot && neckB) {
          ctx.strokeStyle = '#0d1216'; ctx.lineWidth = Math.max(5, H * scale * .09); ctx.lineCap = 'round';
          ctx.beginPath(); ctx.moveTo(foot[0], foot[1]); ctx.lineTo(neckB[0], neckB[1]); ctx.stroke();
          ctx.strokeStyle = '#2b343c'; ctx.lineWidth = Math.max(3, H * scale * .05);
          ctx.beginPath(); ctx.moveTo(foot[0], foot[1]); ctx.lineTo(neckB[0], neckB[1]); ctx.stroke();
          ctx.fillStyle = '#161d22'; ctx.strokeStyle = '#333e46'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.ellipse(foot[0], foot[1], Math.max(6, H * scale * .3), Math.max(3, H * scale * .1), 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        }
        if (!bezel.every(Boolean) || !screen.every(Boolean)) continue;
        const poly = (pts, fill, stroke) => {
          ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
          ctx.closePath(); if (fill) { ctx.fillStyle = fill; ctx.fill(); }
          if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
        };
        poly(bezel, '#10151a', '#39434c');
        const lerp2 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        const row = (t, x0, x1) => [lerp2(lerp2(screen[0], screen[3], t), lerp2(screen[1], screen[2], t), x0),
                                    lerp2(lerp2(screen[0], screen[3], t), lerp2(screen[1], screen[2], t), x1)];
        // Which face points at the camera? Painter's depth grows nearer, so
        // the screen front faces us when stepping along the normal gets nearer.
        // Note the default camera stands across the desk from the fly, so it
        // sees the casing backs; the screens face the fly and appear on orbit.
        const pC = P(C), pN = P([C[0] + d[0] * H, C[1] + d[1] * H, C[2] + d[2] * H]);
        const facingFront = pC && pN ? pN[2] >= pC[2] : true;
        if (facingFront) {
        // Amber-phosphor screen with a soft glow.
        const sg = ctx.createLinearGradient(screen[0][0], screen[0][1], screen[2][0], screen[2][1]);
        sg.addColorStop(0, '#2a2118'); sg.addColorStop(.5, '#3d2f1c'); sg.addColorStop(1, '#201913');
        ctx.save();
        ctx.shadowColor = 'rgba(232,168,84,.5)'; ctx.shadowBlur = 16;
        poly(screen, sg, '#7a5c33');
        ctx.restore();
        // Screen content: terminal-ish bars + a trace, lerped in screen space.
        ctx.strokeStyle = 'rgba(232,168,84,.85)'; ctx.lineCap = 'round';
        ctx.lineWidth = Math.max(1.5, H * scale * .022);
        [[.3, .1, .62], [.45, .1, .45], [.6, .1, .7], [.75, .1, .38]].forEach(([t, a, b]) => {
          const [p0, p1] = row(t, a, b);
          ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
        });
        ctx.strokeStyle = 'rgba(245,207,142,.9)';
        ctx.beginPath();
        for (let k = 0; k <= 10; k++) {
          const t = k / 10, zz = (k % 3 === 1) ? -.03 : (k % 3 === 2 ? .03 : 0);
          const [p0, p1] = row(.88 + zz, .08 + t * .8, .08 + t * .8);
          const p = lerp2(p0, p1, 0);
          k ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
        }
        ctx.stroke();
        // Power LED on the chin (front only).
        const led = lerp2(lerp2(bezel[3], bezel[2], .5), lerp2(screen[3], screen[2], .5), .5);
        ctx.fillStyle = '#7dd87d';
        ctx.beginPath(); ctx.arc(led[0], led[1], Math.max(1.5, H * scale * .02), 0, Math.PI * 2); ctx.fill();
        } else {
        // Solid back: matte casing with vent slits, maker's medallion, and a
        // stand-mount cover where the neck joins.
        const cg = ctx.createLinearGradient(screen[0][0], screen[0][1], screen[2][0], screen[2][1]);
        cg.addColorStop(0, '#272f36'); cg.addColorStop(.55, '#1c2228'); cg.addColorStop(1, '#141920');
        poly(screen, cg, '#3d4750');
        ctx.strokeStyle = '#0b0e11'; ctx.lineCap = 'round';
        ctx.lineWidth = Math.max(2, H * scale * .028);
        [.16, .22, .28, .34].forEach((t) => {
          const [p0, p1] = row(t, .32, .68);
          ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
        });
        const med = lerp2(lerp2(screen[0], screen[3], .52), lerp2(screen[1], screen[2], .52), .5);
        ctx.fillStyle = '#1b2127'; ctx.strokeStyle = '#4a545e'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(med[0], med[1], Math.max(2.5, H * scale * .035), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        const [m0, m1] = row(.99, .4, .6), [m2, m3] = row(.78, .34, .66);
        poly([m2, m3, m1, m0], '#111518', '#2c343b');
        }
      }
    }

    /* --- 2. Neural map: full dot field drawn underneath the body --- */
    const drawDot = (i) => {
      const p = viewer.projected[i];
      const v = viewer.frame?.values[i] || 0;
      const strength = Math.min(1, Math.abs(v) / range);
      const radius = 1.9 + strength * 1.8 + (viewer.changedCells.has(i) ? pulse * 1.8 : 0);
      const neutral = [158, 170, 160], target = v >= 0 ? [239, 165, 100] : [107, 186, 217];
      const color = neutral.map((n, k) => Math.round(n + (target[k] - n) * strength));
      ctx.fillStyle = `rgba(${color.join(',')},${.45 + strength * .5})`;
      ctx.beginPath(); ctx.arc(p[0], p[1], radius, 0, Math.PI * 2); ctx.fill();
      if (viewer.changedCells.has(i)) {
        ctx.strokeStyle = `rgba(244,235,201,${.3 + pulse * .65})`; ctx.lineWidth = 1 + pulse;
        ctx.beginPath(); ctx.arc(p[0], p[1], radius + 2.5 + pulse * 7, 0, Math.PI * 2); ctx.stroke();
      }
    };
    // Draw the full neural map first so the body occludes it, not the other way around.
    const BRAIN_Z = 420;
    const anatomy = viewer.positions();
    for (let i = 0; i < viewer.projected.length; i++) {
      if (viewer.projected[i]) drawDot(i);
    }
    // Cutaway windows show the neurons that belong to that body part: the head
    // window draws the brain cloud (low z), thorax/abdomen windows draw the
    // ventral-cord population (high z). Same 420 split as measure().
    // Center each cutaway window on the median of the displayed cells it can
    // show (stable across frames — positions, not values), and size it to
    // their projected spread so the glass hugs the dots instead of leaving
    // half the window empty. Centering on the body-part geometry instead
    // leaves the dots piled in part of the glass.
    const median3 = (pts) => {
      if (!pts.length) return null;
      const m = (arr) => { const s = [...arr].sort((a, b) => a - b); return s[Math.floor((s.length - 1) / 2)]; };
      return [0, 1, 2].map(k => m(pts.map(p => p[k])));
    };
    // Keep a fitted center from sliding off the body: at most 0.3 radii out.
    const clampWin = (c, part, r) => {
      if (!c) return part;
      const d = Math.hypot(c[0] - part[0], c[1] - part[1], c[2] - part[2]);
      if (!(d > r * .3)) return c;
      return [0, 1, 2].map(k => part[k] + (c[k] - part[k]) * r * .3 / d);
    };
    const cordSplitZ = (body.thorax.b[2] + body.abdomen.b[2]) / 2;
    const brainIdx = [], cordForeIdx = [], cordAftIdx = [];
    for (let i = 0; i < anatomy.length; i++) {
      const a = anatomy[i];
      if (!a) continue;
      if (a[2] < BRAIN_Z) brainIdx.push(i);
      else (a[2] < cordSplitZ ? cordForeIdx : cordAftIdx).push(i);
    }
    const medA = (idx) => median3(idx.map(i => anatomy[i]));
    const thoraxMid = body.thorax.a.map((v, k) => (v + body.thorax.b[k]) / 2);
    const abdomenMid = body.abdomen.a.map((v, k) => (v + body.abdomen.b[k]) / 2);
    const headWinC = clampWin(medA(brainIdx), body.head.c, body.head.r);
    const thoraxWinC = clampWin(medA(cordForeIdx), thoraxMid, body.thorax.r);
    const abdomenWinC = clampWin(medA(cordAftIdx), abdomenMid, body.abdomen.r);
    // Size each window to the 95th-percentile projected spread of its cells
    // plus padding, clamped inside the body part.
    const q95 = (a) => {
      if (!a.length) return 0;
      const s = [...a].sort((x, y) => x - y);
      return s[Math.min(s.length - 1, Math.floor(.95 * (s.length - 1)))];
    };
    const fitWin = (idx, c, r, only, fx0, fy0) => {
      const pc = P(c);
      if (!pc || !idx.length) return {c, fx: fx0, fy: fy0, only};
      const dx = [], dy = [];
      for (const i of idx) {
        const p = viewer.projected[i];
        if (!p) continue;
        dx.push(Math.abs(p[0] - pc[0])); dy.push(Math.abs(p[1] - pc[1]));
      }
      const pad = 12;
      return {c, fx: Math.min(1, (q95(dx) + pad) / (r * scale)),
              fy: Math.min(1, (q95(dy) + pad) / (r * scale)), only};
    };
    const headWin = fitWin(brainIdx, headWinC, body.head.r, 'brain', .85, .75);
    const thoraxWin = fitWin(cordForeIdx, thoraxWinC, body.thorax.r, 'cord', .68, .6);
    const abdomenWin = fitWin(cordAftIdx, abdomenWinC, body.abdomen.r, 'cord', .64, .58);
    // Appendages are deliberately excluded from every visible neural window.
    // This is a screen-space mask because both the cutaways and limbs have
    // already been projected; it stays exact as the camera orbits the fly.
    const cutoutMask = (window, r) => {
      const c = P(window.c);
      return c ? {x: c[0], y: c[1], rx: r * scale * window.fx, ry: r * scale * window.fy} : null;
    };
    const neuralCutouts = [
      cutoutMask(headWin, body.head.r),
      cutoutMask(thoraxWin, body.thorax.r),
      cutoutMask(abdomenWin, body.abdomen.r),
    ].filter(Boolean);
    const clipOutsideMasks = (masks, padding = 0) => {
      for (const mask of masks || []) {
        const rx = mask.rx ?? mask.r;
        const ry = mask.ry ?? mask.r;
        if (!(rx > 0) || !(ry > 0)) continue;
        // Apply one inverse clip at a time. Intersecting the results keeps
        // overlapping cutaways excluded too.
        ctx.beginPath();
        ctx.rect(0, 0, w, h);
        ctx.ellipse(mask.x, mask.y, rx + padding, ry + padding, 0, 0, Math.PI * 2);
        ctx.clip('evenodd');
      }
    };
    const windowDots = (only) => {
      for (let i = 0; i < viewer.projected.length; i++) {
        if (!viewer.projected[i]) continue;
        if (only) {
          const a = anatomy[i];
          if (!a) continue;
          if (only === 'brain' ? a[2] >= BRAIN_Z : a[2] < BRAIN_Z) continue;
        }
        drawDot(i);
      }
    };

    /* --- 3. Opaque fly body, back-to-front; occludes neurons outside windows --- */
    const parts = [];
    const addPart = (part) => parts.push(part);

    // Each opaque part: c = centre, r = radius, draw paints the disc.
    const spherePart = (c, r, fill, edge, window) => {
      const pc = P(c);
      if (!pc) return;
      // Front-surface depth: orthonormal projection means depth of c + r along view axis.
      addPart({z: pc[2] + r * scale, draw: () => {
        // Radial gradient for 3D spherical shading — light from upper-left.
        const grad = ctx.createRadialGradient(
          pc[0] - r * scale * .3, pc[1] - r * scale * .3, r * scale * .1,
          pc[0], pc[1], r * scale
        );
        grad.addColorStop(0, lighten(fill, 40));
        grad.addColorStop(.5, fill);
        grad.addColorStop(1, darken(fill, 30));
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(pc[0], pc[1], r * scale, 0, Math.PI * 2); ctx.fill();
        if (edge) { ctx.strokeStyle = edge; ctx.lineWidth = 1.2; ctx.stroke(); }
      }});
      if (window) {
        const win = {c: window.c || c, rx: r * scale * (window.fx || .8), ry: r * scale * (window.fy || .66),
                     z: pc[2] + r * scale + .1};
        addPart({z: win.z, draw: () => {
          ctx.save();
          ctx.beginPath(); ctx.ellipse(P(win.c)[0], P(win.c)[1], win.rx, win.ry, 0, 0, Math.PI * 2); ctx.clip();
          // Redraw the window's own neurons clipped inside — measured states.
          windowDots(window.only);
          ctx.restore();
          ctx.strokeStyle = windowRim; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.ellipse(P(win.c)[0], P(win.c)[1], win.rx, win.ry, 0, 0, Math.PI * 2); ctx.stroke();
        }});
      }
    };
    const capsulePart = (a, b, r, fill, edge, window, alpha = 1, zBias = 0, exclusions = []) => {
      const pa = P(a), pb = P(b);
      if (!pa || !pb) return;
      // Approximate capsule as a chain of discs; the head disc carries depth.
      const steps = Math.max(2, Math.ceil(Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) / (r * scale * .6)));
      const discs = [];
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        discs.push([pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t, pa[2] + (pb[2] - pa[2]) * t]);
      }
      addPart({z: Math.max(...discs.map(d => d[2])) + r * scale + zBias, draw: () => {
        ctx.save(); ctx.globalAlpha = alpha;
        clipOutsideMasks(exclusions, r * scale * .55);
        // Draw one continuous round stroke. Filling the old chain of overlapping
        // discs compounded alpha until a nominally translucent shell became
        // effectively opaque and hid the neural cord.
        const grad = ctx.createLinearGradient(pa[0], pa[1], pb[0], pb[1]);
        grad.addColorStop(0, lighten(fill, 25));
        grad.addColorStop(.5, fill);
        grad.addColorStop(1, darken(fill, 25));
        ctx.lineCap = 'round';
        if (edge) {
          ctx.strokeStyle = edge; ctx.lineWidth = r * scale * 2 + 2;
          ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
        }
        ctx.strokeStyle = grad; ctx.lineWidth = r * scale * 2;
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
        ctx.restore();
      }});
      if (window) {
        const wc = P(window.c || [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]);
        if (wc) addPart({z: wc[2] + r * scale + .1, draw: () => {
          ctx.save();
          ctx.beginPath(); ctx.ellipse(wc[0], wc[1], r * scale * (window.fx || .8), r * scale * (window.fy || .8), 0, 0, Math.PI * 2); ctx.clip();
          windowDots(window.only);
          ctx.restore();
          ctx.strokeStyle = windowRim; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.ellipse(wc[0], wc[1], r * scale * (window.fx || .8), r * scale * (window.fy || .8), 0, 0, Math.PI * 2); ctx.stroke();
        }});
      }
    };

    // Wings: translucent membranes sweeping up and back off the thorax, each
    // with a centre vein and iridescent sheen. Halteres stay small and pale behind them.
    for (const wing of body.wings) {
      const pb = P(wing.base), pt = P(wing.tip);
      if (!pb || !pt) continue;
      const mx = (pb[0] + pt[0]) / 2, my = (pb[1] + pt[1]) / 2;
      const ux = pt[0] - pb[0], uy = pt[1] - pb[1];
      const len = Math.hypot(ux, uy) || 1;
      const spread = Math.min(60 * scale, len * .3);
      const nx = -uy / len * spread, ny = ux / len * spread;
      addPart({z: Math.max(pb[2], pt[2]) + 2, draw: () => {
        // Iridescent wing membrane: gradient from base to tip.
        const grad = ctx.createLinearGradient(pb[0], pb[1], pt[0], pt[1]);
        grad.addColorStop(0, 'rgba(215,228,220,.38)');
        grad.addColorStop(.5, 'rgba(195,215,205,.28)');
        grad.addColorStop(1, 'rgba(175,200,190,.18)');
        ctx.fillStyle = grad;
        ctx.strokeStyle = 'rgba(230,240,235,.6)'; ctx.lineWidth = .9;
        ctx.beginPath();
        ctx.moveTo(pb[0], pb[1]); ctx.lineTo(mx + nx, my + ny);
        ctx.lineTo(pt[0], pt[1]); ctx.lineTo(mx - nx, my - ny);
        ctx.closePath(); ctx.fill(); ctx.stroke();
        // Centre vein.
        ctx.strokeStyle = 'rgba(160,175,165,.6)'; ctx.lineWidth = 1.1;
        ctx.beginPath(); ctx.moveTo(pb[0], pb[1]); ctx.lineTo(pt[0], pt[1]); ctx.stroke();
        // Cross veins for detail.
        ctx.strokeStyle = 'rgba(170,185,175,.35)'; ctx.lineWidth = .6;
        for (const t of [.25, .5, .75]) {
          const vx = pb[0] + ux * t, vy = pb[1] + uy * t;
          ctx.beginPath(); ctx.moveTo(vx - nx * .3, vy - ny * .3); ctx.lineTo(vx + nx * .3, vy + ny * .3); ctx.stroke();
        }
      }});
    }
    for (const h of body.halteres) capsulePart(h.a, h.b, fit.headR * .07, '#c9bfae', null);

    // Body core: head (brain-neuron window), thorax + abdomen (cord windows).
    spherePart(body.head.c, body.head.r, headChitin, headEdge, headWin);
    // Keep the head solid, but let the long neural cord remain legible through
    // the thorax and abdomen instead of being reduced to the small windows.
    capsulePart(body.thorax.a, body.thorax.b, body.thorax.r, thoraxChitin, thoraxEdge, thoraxWin, .34);
    capsulePart(body.abdomen.a, body.abdomen.b, body.abdomen.r, abdomenChitin, null, abdomenWin, .28);
    // Antennae remain visible on the chitin without crossing a neural window.
    for (const ant of body.antennae) {
      capsulePart(ant.a, ant.b, fit.headR * .035, bristleChitin, null,
                  null, 1, 0, neuralCutouts);
    }

    // Painter's algorithm: farthest (smallest view depth) first.
    parts.sort((a, b) => a.z - b.z);
    for (const part of parts) part.draw();

    // Six readable foreground limbs. The two front legs route around the head
    // to the keyboard; four walking legs start just outside the lower body
    // outline and bend outward to the desk. Because every segment begins
    // outside the visible silhouette and continues away from it, none passes
    // through the body or its neural cutaway windows (canvas clipping in
    // drawLimb remains as a safety net for extreme camera angles).
    const unit = (x, y, fallback = [1, 0]) => {
      const length = Math.hypot(x, y);
      return length > 1e-6 ? [x / length, y / length] : fallback;
    };
    const drawLimb = (points, width, alpha = 1, occluders = neuralCutouts) => {
      if (points.some(p => !p)) return;
      ctx.save();
      const masks = Array.isArray(occluders) ? occluders : (occluders ? [occluders] : []);
      clipOutsideMasks(masks, width * .55);
      ctx.globalAlpha = alpha;
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.strokeStyle = '#33291f'; ctx.lineWidth = width + 3;
      ctx.beginPath(); ctx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
      ctx.stroke();
      ctx.strokeStyle = limbChitin; ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
      ctx.stroke();
      ctx.restore();
    };
    const headPoint = P(body.head.c);
    if (headPoint) {
      const active = typing && keys[0] ? keys[0] : null;
      const facePoint = P([body.head.c[0], body.head.c[1], body.head.c[2] - body.head.r]);
      const faceFacing = facePoint
        ? (facePoint[2] - headPoint[2]) / (body.head.r * scale)
        : -1;
      const activeSide = active && active.col < kbE.cols / 2 ? -1 : 1;
      const armSpecs = [
        {side: -1, home: {row: 1, col: 3}},
        {side: 1, home: {row: 2, col: 8}},
      ];
      for (const arm of armSpecs) {
        const key = active && arm.side === activeSide ? active : arm.home;
        const target3 = this.keyCenter(kbE, fit, key.row, key.col);
        // These joints live in the fly's coordinate system. Projecting a
        // stable shoulder and elbow prevents the old screen-space tangent
        // from flipping the arm inside out while the camera orbits.
        const shoulder3 = [body.head.c[0] + arm.side * body.head.r * .76,
                           body.head.c[1] + body.head.r * .3,
                           body.head.c[2] - body.head.r * .34];
        const elbow3 = [body.head.c[0] + arm.side * body.head.r * 1.08,
                        deskY - body.head.r * .28,
                        body.head.c[2] - body.head.r * 1.02];
        const shoulder = P(shoulder3), elbow = P(elbow3), target = P(target3);
        if (!shoulder || !elbow || !target) continue;
        // A far-side shoulder belongs behind the opaque head, but the elbow
        // and forearm can still be visible beyond its outline. Clip that arm
        // around the head instead of dropping the entire limb.
        const facing = (shoulder[2] - headPoint[2]) / (body.head.r * scale);
        if (faceFacing <= -.18) continue;
        const headOccluder = facing <= .025
          ? {x: headPoint[0], y: headPoint[1], r: body.head.r * scale}
          : null;
        const armOccluders = headOccluder
          ? [...neuralCutouts, headOccluder]
          : neuralCutouts;
        drawLimb([shoulder, elbow, target], Math.max(4, body.head.r * scale * .065),
                 arm.side < 0 ? .9 : 1, armOccluders);
      }
    }
    const lerp3 = (a, b, t) => a.map((value, i) => value + (b[i] - value) * t);
    const rearLegCenter = lerp3(body.abdomen.a, body.abdomen.b, .64);
    rearLegCenter[1] -= body.abdomen.r * .28;
    const walkingPairs = [
      {center: lerp3(body.thorax.a, body.thorax.b, .62), radius: body.thorax.r,
       spread: 1.02, attachDown: .62, kneeBend: .42, kneeLift: .08},
      // Rear legs attach higher on the abdomen and lift at the knee before
      // dropping to the desk. The extra bend keeps them from reading as rods,
      // and the wider spread keeps them clear of the abdomen neural window.
      {center: rearLegCenter, radius: body.abdomen.r,
       spread: 1.34, attachDown: .24, kneeBend: .7, kneeLift: .42},
    ];
    for (const pair of walkingPairs) {
      const base = P(pair.center);
      const desk = P([pair.center[0], deskY, pair.center[2]]);
      // x is the fly's anatomical left/right axis. Project that axis rather
      // than the head-to-tail z axis so the legs remain attached to the sides
      // of the thorax and abdomen as the camera rotates.
      const lateralPoint = P([pair.center[0] + fit.headR, pair.center[1], pair.center[2]]);
      if (!base || !desk || !lateralPoint) continue;
      const down = unit(desk[0] - base[0], desk[1] - base[1], [0, 1]);
      const lateral = unit(lateralPoint[0] - base[0], lateralPoint[1] - base[1], [1, 0]);
      const radius = pair.radius * scale;
      for (const side of [-1, 1]) {
        const attach = unit(
          down[0] * pair.attachDown + lateral[0] * side * .78,
          down[1] * pair.attachDown + lateral[1] * side * .78,
          [side, .65]
        );
        const start = [base[0] + attach[0] * radius * 1.04,
                       base[1] + attach[1] * radius * 1.04];
        const foot = [desk[0] + lateral[0] * side * radius * pair.spread,
                      desk[1] + lateral[1] * side * radius * pair.spread];
        const knee = [(start[0] + foot[0]) / 2 + lateral[0] * side * radius * pair.kneeBend - down[0] * radius * pair.kneeLift,
                      (start[1] + foot[1]) / 2 + lateral[1] * side * radius * pair.kneeBend - down[1] * radius * pair.kneeLift];
        drawLimb([start, knee, foot], Math.max(3.5, radius * .085),
                 side < 0 ? .62 : .9, neuralCutouts);
      }
    }

    // Pressed-key taps glow on top of everything so typing reads clearly.
    for (const [x, y, s] of pressedSpots) {
      ctx.fillStyle = '#efb07d';
      ctx.beginPath(); ctx.arc(x, y, s * .8, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = 'rgba(239,176,125,.75)'; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.arc(x, y, s * .8 + 3.5, 0, Math.PI * 2); ctx.stroke();
    }

    // Compound eyes sit above the face only while their part of the face is
    // turned toward the camera. This keeps the visible eyes crisp over the
    // head and limbs without making them shine through the back of the head.
    // P()'s depth basis is unit length, so the signed centre-to-eye depth is a
    // stable facing test at every zoom level.
    const headDepth = headPoint?.[2] ?? P(body.head.c)?.[2];
    const eyes = [body.eyeL, body.eyeR]
      .map(e => {
        const p = P(e.c);
        const facing = p && Number.isFinite(headDepth)
          ? (p[2] - headDepth) / (body.head.r * scale)
          : -1;
        return {e, p, facing};
      })
      .filter(item => item.p && item.facing > .025)
      .sort((a, b) => a.p[2] - b.p[2]);
    for (const {e, p} of eyes) {
      const pr = e.r * scale;
      const grad = ctx.createRadialGradient(p[0] - pr * .25, p[1] - pr * .25, pr * .08, p[0], p[1], pr);
      grad.addColorStop(0, lighten(eyeFill, 70));
      grad.addColorStop(.5, eyeFill);
      grad.addColorStop(1, darken(eyeFill, 50));
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.arc(p[0], p[1], pr, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,.22)'; ctx.lineWidth = .9;
      for (let ring = 1; ring <= 3; ring++) {
        const rr = pr * ring / 3.5;
        for (let i = 0; i < 6; i++) {
          const a = i * Math.PI / 3 + ring * .3;
          ctx.beginPath();
          ctx.arc(p[0] + Math.cos(a) * rr, p[1] + Math.sin(a) * rr, pr * .12, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
      ctx.strokeStyle = eyeEdge; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(p[0], p[1], pr, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = 'rgba(255,240,225,.95)';
      ctx.beginPath(); ctx.arc(p[0] - pr * .28, p[1] - pr * .28, pr * .18, 0, Math.PI * 2); ctx.fill();
    }

    // Selection ring on top of everything, at the selected neuron.
    if (viewer.selected >= 0 && viewer.projected[viewer.selected]) {
      const p = viewer.projected[viewer.selected];
      const v = viewer.frame?.values[viewer.selected] || 0;
      const strength = Math.min(1, Math.abs(v) / range);
      ctx.strokeStyle = '#f2e9d7'; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(p[0], p[1], 3.9 + strength * 1.8 + 5, 0, Math.PI * 2); ctx.stroke();
    }

    this.lastScale = scale;
  }
}
