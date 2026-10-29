/* Recorded reserve prices, with actual paper fills and the selected neuron frame. */
class CoinChart {
  constructor(onSelectFrame) {
    this.canvas = document.querySelector('#coin-chart-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.onSelectFrame = onSelectFrame;
    this.coin = null; this.session = null; this.points = []; this.visible = [];
    this.count = 0; this.historyCount = 0; this.historyPoints = []; this.historyLoaded = false;
    this.frame = null; this.hover = null; this.loading = false;
    this.generation = 0; this.lastAttempt = 0;
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(this.canvas);
    document.querySelector('#coin-chart-range').onchange = () => { this.hover = null; this.draw(); };
    this.canvas.onpointermove = event => {
      const box = this.canvas.getBoundingClientRect();
      const x = event.clientX - box.left;
      if (!this.visible.length || !this.x) return;
      this.hover = this.visible.reduce((a, b) => Math.abs(this.x(a.time) - x) <= Math.abs(this.x(b.time) - x) ? a : b);
      this.draw();
    };
    this.canvas.onpointerleave = () => { this.hover = null; this.draw(); };
    this.canvas.onclick = () => { if (this.hover?.step && this.session?.status !== 'running') this.onSelectFrame(this.hover.step); };
    this.canvas.onkeydown = event => {
      if (!this.visible.length || !['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Enter'].includes(event.key)) return;
      event.preventDefault();
      let index = this.hover ? this.visible.findIndex(point => point.time === this.hover.time) : this.visible.length - 1;
      if (index < 0) index = this.visible.length - 1;
      if (event.key === 'Home') index = 0;
      if (event.key === 'End') index = this.visible.length - 1;
      if (event.key === 'ArrowLeft') index = Math.max(0, index - 1);
      if (event.key === 'ArrowRight') index = Math.min(this.visible.length - 1, index + 1);
      this.hover = this.visible[index];
      if (event.key === 'Enter' && this.hover.step && this.session?.status !== 'running') this.onSelectFrame(this.hover.step);
      this.draw();
    };
  }

  async update(coin, session) {
    const panel = document.querySelector('#focused-coin-chart');
    panel.hidden = !coin?.bars;
    if (!coin?.bars) return;
    this.session = session;
    if (coin.activity_path !== this.coin?.activity_path) {
      ++this.generation; this.loading = false; this.lastAttempt = 0;
      this.points = []; this.count = 0; this.frame = null; this.hover = null;
      this.historyPoints = []; this.historyCount = 0; this.historyLoaded = false;
    }
    this.coin = coin;
    const coinLink = document.querySelector('#coin-chart-title');
    coinLink.textContent = `${session.coin_policy === 'pin' ? 'Pinned coin · segment ' : 'Coin '}${coin.index + 1} · ${coin.mint.slice(0, 6)}…${coin.mint.slice(-4)} ↗`;
    coinLink.href = `https://pump.fun/coin/${encodeURIComponent(coin.mint)}`;
    const liveBadge = document.querySelector('#coin-chart-live');
    if (liveBadge) liveBadge.textContent = session.status === 'running' ? 'LIVE PRICES' : 'RECORDED PRICES';
    this.draw();
    if (this.loading || (this.count >= coin.observations && this.points.length) || Date.now() - this.lastAttempt < 2500) return;
    const generation = this.generation;
    this.loading = true; this.lastAttempt = Date.now();
    try {
      const response = await fetch('/api/bars?path=' + encodeURIComponent(coin.bars) + '&full=1');
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Prices unavailable');
      if (generation !== this.generation) return;
      if (coin.history_path && !this.historyLoaded) {
        const historyResponse = await fetch('/api/bars?path=' + encodeURIComponent(coin.history_path) + '&full=1');
        const history = await historyResponse.json();
        if (!historyResponse.ok) throw new Error(history.error || 'Prior history unavailable');
        if (generation !== this.generation) return;
        this.historyPoints = history.points.map(point => ({time: point[0], price: point[1], step: null}));
        this.historyCount = history.count; this.historyLoaded = true;
      }
      this.points = [...this.historyPoints, ...data.points.map((point, i) => ({time: point[0], price: point[1], step: data.steps?.[i] ?? i + 1}))];
      this.count = data.count;
      this.draw();
    } catch (error) {
      if (generation === this.generation) document.querySelector('#coin-chart-note').textContent = `Waiting for recorded prices: ${error.message}`;
    } finally { if (generation === this.generation) this.loading = false; }
  }

  setFrame(frame) {
    if (frame?.mint !== this.coin?.mint) return;
    this.frame = frame; this.draw();
  }

  draw() {
    const box = this.canvas.getBoundingClientRect();
    if (!box.width || !box.height) return;
    const width = box.width, height = box.height, dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (this.canvas.width !== Math.round(width * dpr) || this.canvas.height !== Math.round(height * dpr)) {
      this.canvas.width = Math.round(width * dpr); this.canvas.height = Math.round(height * dpr);
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    if (!this.points.length) {
      ctx.fillStyle = '#a0afa2'; ctx.font = '12px system-ui';
      ctx.fillText('Waiting for the first recorded price…', 18, height / 2);
      document.querySelector('#coin-chart-price').textContent = '—';
      document.querySelector('#coin-chart-change').textContent = '';
      document.querySelector('#coin-chart-note').textContent = 'Prices load from the recording, including observations made before you opened this page.';
      document.querySelector('#coin-chart-hover').textContent = '';
      return;
    }
    const first = this.points[0], last = this.points.at(-1);
    const minutes = Number(document.querySelector('#coin-chart-range').value);
    this.visible = minutes ? this.points.filter(point => point.time >= last.time - minutes * 60) : this.points;
    const points = this.visible, base = first.price;
    const percent = price => (price / base - 1) * 100;
    const changes = points.map(point => percent(point.price));
    let low = Math.min(...changes), high = Math.max(...changes);
    const spread = Math.max(high - low, .1);
    low -= spread * .15; high += spread * .15;
    const left = 67, right = width - 20, top = 22, bottom = height - 38;
    const start = points[0].time, end = Math.max(start + 1, last.time);
    this.x = time => left + (time - start) / (end - start) * (right - left);
    const y = price => bottom - (percent(price) - low) / (high - low) * (bottom - top);
    ctx.font = '10px monospace';
    for (let i = 0; i <= 4; i++) {
      const row = top + (bottom - top) * i / 4;
      const value = high - (high - low) * i / 4;
      ctx.strokeStyle = '#303c35'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(left, row); ctx.lineTo(right, row); ctx.stroke();
      ctx.fillStyle = '#a3b29e'; ctx.textAlign = 'right';
      ctx.fillText(`${value.toFixed(Math.abs(high - low) < 1 ? 3 : 1)}%`, left - 8, row + 3);
    }
    ctx.beginPath();
    points.forEach((point, i) => i ? ctx.lineTo(this.x(point.time), y(point.price)) : ctx.moveTo(this.x(point.time), y(point.price)));
    ctx.strokeStyle = '#b4d2a7'; ctx.lineWidth = 2; ctx.stroke();
    if (points.length === 1) { ctx.beginPath(); ctx.arc(this.x(points[0].time), y(points[0].price), 3, 0, Math.PI * 2); ctx.fillStyle = '#b4d2a7'; ctx.fill(); }
    const orders = (this.session?.events || []).filter(event => event.mint === this.coin.mint && (event.coin_index === undefined || event.coin_index === this.coin.index));
    for (const event of orders) {
      if (event.timestamp < start || event.timestamp > end) continue;
      const px = this.x(event.timestamp), py = y(event.quote_price);
      ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle = event.side === 'BUY' ? '#8fcaaf' : '#efb07d'; ctx.fill();
      ctx.textAlign = 'center'; ctx.fillText(event.side === 'BUY' ? 'B' : 'S', px, Math.max(12, py - 10));
    }
    if (this.frame && this.frame.timestamp >= start && this.frame.timestamp <= end) {
      const px = this.x(this.frame.timestamp);
      ctx.strokeStyle = '#efb07d'; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke(); ctx.setLineDash([]);
    }
    const selected = this.hover || (this.frame ? {time: this.frame.timestamp, price: this.frame.price, step: this.frame.step} : last);
    if (this.hover && this.hover.time >= start) {
      ctx.fillStyle = '#f0dfbb'; ctx.beginPath(); ctx.arc(this.x(selected.time), y(selected.price), 4, 0, Math.PI * 2); ctx.fill();
    }
    for (let i = 0; i <= 2; i++) {
      const time = start + (end - start) * i / 2;
      ctx.textAlign = i === 0 ? 'left' : i === 2 ? 'right' : 'center'; ctx.fillStyle = '#a3b29e';
      ctx.fillText(new Date(time * 1000).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}), left + (right - left) * i / 2, height - 12);
    }
    const change = percent(last.price);
    document.querySelector('#coin-chart-price').textContent = `${last.price.toPrecision(6)} SOL / token`;
    document.querySelector('#coin-chart-change').textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}% since first recorded price`;
    document.querySelector('#coin-chart-change').className = change < 0 ? 'negative' : change > 0 ? 'positive' : '';
    const limitedLedger = this.session?.paper?.orders > (this.session?.events || []).length;
    const observedMinutes = Math.round((last.time - first.time) / 60);
    document.querySelector('#coin-chart-note').textContent = `${this.count.toLocaleString()} segment prices + ${this.historyCount.toLocaleString()} prior context prices · ${observedMinutes} minute${observedMinutes === 1 ? '' : 's'} observed · ${orders.length} ${limitedLedger ? 'recent ' : ''}paper orders in this segment${this.points.length < this.count + this.historyCount ? ' · chart samples the available time span' : ''}. This is collected history, not a full lifetime chart.${limitedLedger ? ' Older orders may be absent from the chart; the complete ledger is saved locally.' : ''}`;
    document.querySelector('#coin-chart-hover').textContent = `${selected.step ? 'Observation ' + selected.step : 'Prior context · no neuron frame'} · ${new Date(selected.time * 1000).toLocaleString()} · ${selected.price.toPrecision(6)} SOL/token${this.hover ? '' : this.frame ? ' · selected neuron frame' : ' · latest recorded price'}`;
    this.canvas.setAttribute('aria-label', `${this.coin.mint} price chart: ${this.count} observations, ${change.toFixed(2)} percent change, ${orders.length} paper orders. Arrow keys inspect prices; Enter selects a saved neuron frame.`);
  }
}
