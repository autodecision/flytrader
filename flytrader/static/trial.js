/* Trial telemetry and account values come from the paper engine and saved frames. */
class TrialDashboard {
  constructor() {
    this.path = localStorage.getItem('flytrader.trial') || '';
    this.session = null;
    this.loading = false;
    this.showAll = false;
    this.lastHistory = '';
    const savedFields = ['coin_policy', 'mint', 'min_cap_sol', 'min_reserves_sol', 'interval', 'profile', 'reward_gain', 'exploration'];
    try {
      const settings = JSON.parse(localStorage.getItem('flytrader.coin-settings') || '{}');
      for (const key of savedFields) if (settings[key] !== undefined) $(`#trial-form [name="${key}"]`).value = settings[key];
    } catch (_) { /* Older preferences do not block the controls. */ }
    $('#trial-form').addEventListener('input', () => {
      localStorage.setItem('flytrader.coin-settings', JSON.stringify(Object.fromEntries(savedFields.map(key => [key, $(`#trial-form [name="${key}"]`).value]))));
    });
    this.chart = new CoinChart(step => {
      if (brain.live || !brain.frames.length) return;
      const frame = brain.frames.reduce((a, b) => Math.abs(a.step - step) <= Math.abs(b.step - step) ? a : b);
      brain.playing = false;
      $('#frame-slider').value = brain.frames.indexOf(frame);
      brain.setFrame(frame); brain.updatePlayback();
    });
    $('#coin-chart-select').onchange = () => {
      const coin = this.session?.coins.find(coin => coin.index === Number($('#coin-chart-select').value));
      if (coin && this.session.status !== 'running') this.openActivity(coin.activity_path);
    };
    $('#trial-form').onsubmit = event => {
      event.preventDefault();
      const settings = Object.fromEntries(new FormData(event.target));
      const continuous = settings.source === 'continuous';
      if (!continuous) settings.profile = 'chart-context-v1';
      start({action: 'trial', edges: activeGraph, ...settings, source: continuous ? 'live' : settings.source, continuous});
    };
    $('#trial-source').onchange = () => this.readiness();
    $('#training-profile').onchange = () => this.readiness();
    for (const button of document.querySelectorAll('[data-circuit-filter]')) button.onclick = () => {
      $('#neuron-search').value = button.dataset.circuitFilter;
      $('#neuron-search').dispatchEvent(new Event('input'));
      $('.brain-panel').scrollIntoView({behavior: 'smooth', block: 'start'});
    };
    $('#coin-policy').onchange = () => this.readiness();
    $('#trial-coins').onchange = () => this.readiness();
    $('#trial-form [name="interval"]').oninput = () => this.readiness();
    $('#trial-mint').oninput = event => { if (event.target.value.trim()) $('#trial-coins').value = '1'; this.readiness(); };
    $('#view-trial').onclick = () => {
      navigate('trial'); this.poll(); $(this.session?.summary ? '#trial-summary' : '#trial-dashboard').scrollIntoView({behavior: 'smooth', block: 'start'});
    };
    $('#summary-inspect').onclick = () => $('#trial-dashboard').scrollIntoView({behavior: 'smooth', block: 'start'});
    $('#stop-trial').onclick = () => $('#stop-button').click();
    $('#trial-all-trades').onclick = () => { this.showAll = !this.showAll; this.renderLedger(); };
    if (state) this.refresh(state);
  }

  short(mint) { return mint ? `${mint.slice(0, 6)}…${mint.slice(-4)}` : 'No coin received'; }
  signed(value, digits = 4) { return `${value > 0 ? '+' : ''}${Number(value).toFixed(digits)}`; }
  running() { return state?.jobs.find(job => job.action === 'trial' && ['running', 'stopping'].includes(job.status)); }

  readiness() {
    if (!state) return;
    const busy = state.jobs.some(job => ['running', 'stopping'].includes(job.status)) || submitting;
    const continuous = $('#trial-source').value === 'continuous';
    const live = $('#trial-source').value !== 'recordings';
    const available = new Set(state.files.filter(file => file.pump_ready).map(file => file.metadata.mint)).size;
    const count = Math.min(Number($('#trial-coins').value), available);
    const interval = Number($('#trial-form [name="interval"]').value) || 10;
    const reward = continuous && $('#training-profile').value === 'mushroom-body-reward-v1';
    const compatible = state.capabilities?.includes(reward ? 'mushroom-body-reward-v1' : 'chart-context-v1');
    $('#server-update-note').hidden = Boolean(compatible);
    const pinOption = $('#coin-policy option[value="pin"]');
    pinOption.disabled = !continuous;
    if (!continuous && $('#coin-policy').value === 'pin') $('#coin-policy').value = 'first';
    const pinned = continuous && $('#coin-policy').value === 'pin';
    for (const option of $('#trial-form [name="count"]').options) {
      const minutes = Math.round(Number(option.value) * interval / 60);
      const duration = minutes < 60 ? `${minutes} minutes` : `${Math.floor(minutes / 60)}h${minutes % 60 ? ' ' + minutes % 60 + 'm' : ''}`;
      option.textContent = `${Number(option.value).toLocaleString()} · about ${duration}`;
    }
    $('#trial-source-hint').textContent = continuous ? `Selects active coins from Pump.fun Explore until you stop. Fresh prices every ${interval}s; the model learns from completed outcomes and keeps that learning across coins and sessions. First use needs about ${Math.round(interval)} minutes of prices; later coins need about ${Math.round(interval / 3)} minutes to warm up. Orders follow the model’s signals, so it may hold cash.` : live ? `Limited trial: each coin trains on its first 60 observations (about ${Math.round(interval)} minutes). The output then stays fixed and the trial ends after the selected number of coins.` : available ? `${count} available coin${count === 1 ? '' : 's'} will run automatically on saved real Pump.fun prices. Historical trials do not alter your live learning memory.` : 'No ready Pump.fun recording yet. Choose continuous mode to discover and learn from new coins.';
    if (pinned) $('#trial-source-hint').textContent = `Stays on your coin until Stop. The brain processes a confirmed price about every ${interval}s, learns from completed outcomes, and keeps its neuron state across recording segments. Quiet or risky conditions pause buys, not observation.`;
    if (reward) $('#trial-source-hint').textContent = `${pinned ? 'Stays on your coin' : 'Follows active Pump.fun coins'} until Stop. After six fresh quotes, tries a small paper position when checks allow. Learns from completed trades after fees, using 30 / 60 / 180-second holds. Pumps reduce position size instead of automatically blocking every buy. Teaching gain strengthens positive dopamine updates; it does not create a reward for buying.`;
    $('#trial-coins-label').hidden = continuous;
    $('#trial-continuous-note').hidden = !continuous;
    const graph = state.files.find(file => file.path === activeGraph);
    const trialModel = $('#trial-model');
    if (trialModel) trialModel.textContent = graph ? `${graph.label} · all model neurons compute` : 'Choose a brain in Brain settings';
    $('#start-trial').disabled = busy || !compatible || !activeGraph || (!live && !available);
    $('#start-trial').textContent = this.running() ? 'Session is running…' : continuous ? 'Start trading & learning ▶' : 'Start paper trial ▶';
    $('#view-trial').disabled = !this.path;
    $('#view-trial').textContent = this.session?.summary ? 'View session results ↗' : 'View trial progress ↗';
    for (const input of $('#trial-form').querySelectorAll('input,select')) input.disabled = busy;
    $('#reward-settings').hidden = !continuous;
    for (const input of $('#reward-settings').querySelectorAll('select')) input.disabled = busy || !continuous;
    for (const knob of document.querySelectorAll('.reward-knob')) { knob.hidden = !reward; knob.querySelector('select').disabled = busy || !reward; }
    for (const label of document.querySelectorAll('.trial-live-setting')) {
      label.hidden = !live;
      label.querySelector('input,select').disabled = busy || !live;
    }
    const needsCoin = live && $('#coin-policy').value !== 'auto';
    $('#coin-input-label').hidden = !needsCoin;
    $('#trial-mint').disabled = busy || !needsCoin;
    $('#trial-mint').required = needsCoin;
    if (needsCoin && !continuous) $('#trial-coins').value = '1';
    $('#observation-count-label').textContent = pinned ? 'Observations per saved segment · same coin continues' : 'Live observations before switching coins';
    const running = this.running();
    $('#stop-trial').hidden = !running;
    $('#stop-trial').disabled = running?.status === 'stopping';
  }

  begin(job) {
    this.path = job.output; this.session = null; this.showAll = false;
    $('#trial-summary').hidden = true;
    localStorage.setItem('flytrader.trial', this.path);
    navigate('trial');
    $('#results').hidden = true;
    $('#trial-phase').textContent = 'Starting the paper trial…';
    if ($('#trial-status-badge')) $('#trial-status-badge').textContent = 'STARTING';
    $('#trial-coin-detail').textContent = 'Loading the selected brain and preparing prices.';
    const capital = Number($('#trial-form [name="capital"]').value);
    const live = $('#trial-source').value !== 'recordings';
    $('#trial-learning').hidden = true;
    $('#reward-circuit').hidden = true;
    $('#trial-source-label').textContent = live ? 'FRESH PUMP.FUN PRICES · PAPER ORDERS' : 'SAVED PUMP.FUN PRICES · HISTORICAL TRIAL';
    $('#trial-map-phase').textContent = 'STARTING PAPER TRIAL';
    $('#trial-freshness').textContent = live ? 'Waiting for discovery and a fresh confirmed price…' : 'Opening the saved Pump.fun prices…';
    $('#trial-frame-note').textContent = 'Waiting for the first computed neuron frame.';
    this.renderPaper({initial_sol: capital, cash_sol: capital, equity_sol: capital, pnl_sol: 0, return_pct: 0,
      realized_sol: 0, unrealized_sol: 0, closed_trades: 0, wins: 0, drawdown_pct: 0,
      position: null, pending: null, prediction: null, decision: 'Waiting for prices', valuation_stale: false});
    $('#trial-progress').value = 0;
    $('#trial-coin-list').replaceChildren();
    this.renderLedger();
    brain.showGraph(activeGraph);
    this.scrollToActivity = true;
    this.poll();
  }

  ended(job) {
    if (job.output === this.path) { this.scrollToSummary = true; this.poll(); }
  }

  refresh(nextState) {
    const running = this.running();
    if (running && this.path !== running.output) {
      this.path = running.output; this.session = null;
      localStorage.setItem('flytrader.trial', this.path);
    }
    if (!running && !nextState.trials?.some(item => item.path === this.path)) this.path = nextState.trials?.[0]?.path || '';
    this.readiness();
    this.renderHistory(nextState.trials || []);
    if (this.path) this.poll();
  }

  async poll() {
    if (!this.path || this.loading) return;
    this.loading = true;
    const path = this.path;
    try {
      const session = await request('/api/trial?path=' + encodeURIComponent(path));
      if (path !== this.path) return;
      this.session = session;
      this.render();
      this.readiness();
      const activity = session.activity_path;
      if (currentView === 'trial' && activity && (brain.report !== activity) && (session.status === 'running' || !brain.report || !session.coins.some(coin => coin.activity_path === brain.report))) {
        await this.openActivity(activity, session.status === 'running');
        if (this.scrollToActivity) {
          $('#trial-dashboard').scrollIntoView({behavior: 'smooth', block: 'start'});
          this.scrollToActivity = false;
        }
      } else if (currentView === 'trial' && activity === brain.report && brain.live && session.status !== 'running') {
        await brain.finish();
      }
      if (currentView === 'trial' && session.summary && this.scrollToSummary) {
        this.scrollToSummary = false; this.scrollToActivity = false;
        $('#trial-summary').scrollIntoView({behavior: 'smooth', block: 'start'});
        $('#trial-summary').focus({preventScroll: true});
      }
    } catch (error) {
      if (path === this.path && state?.trials?.some(item => item.path === path)) {
        $('#trial-freshness').textContent = `Cannot update trial: ${error.message}`;
      }
    } finally { this.loading = false; }
  }

  async openActivity(path, live = false) {
    const graph = state.files.find(file => this.session.edges === file.path || this.session.edges.endsWith('/' + file.path));
    if (graph && activeGraph !== graph.path) setGraph(graph.path);
    const coin = this.session.coins.find(coin => coin.activity_path === path);
    this.chart.update(coin, this.session);
    if (coin) $('#coin-chart-select').value = coin.index;
    await brain.attach(path, live);
  }

  render() {
    const trial = this.session;
    if (!trial) return;
    this.renderSummary(trial.summary);
    const coin = trial.coins[trial.active_coin];
    $('#trial-source-label').textContent = trial.source === 'live' ? 'FRESH PUMP.FUN PRICES · PAPER ORDERS' : 'SAVED PUMP.FUN PRICES · HISTORICAL TRIAL';
    $('#trial-phase').textContent = trial.summary ? 'Recorded account & neuron details' : trial.phase;
    $('#trial-map-phase').textContent = `${trial.source === 'live' ? 'LIVE PRICES' : 'SAVED PRICES'} · ${trial.phase}`;
    if ($('#trial-status-badge')) $('#trial-status-badge').textContent = trial.status === 'running' && trial.continuous ? 'RUNS UNTIL STOP' : trial.status.replaceAll('_', ' ').toUpperCase();
    $('#trial-coin-detail').textContent = coin ? `Coin ${coin.index + 1}${trial.continuous ? ' · continuous session' : ' of ' + trial.planned_coins} · ${this.short(coin.mint)} · ${coin.observations} / ${coin.total || '—'} observations${trial.continuous ? ' before next coin' : ''}` : `${trial.continuous ? 'Finding the next coin · runs until Stop' : 'Preparing up to ' + trial.planned_coins + ' coins'} · ${fmt(trial.neurons || 0)} model neurons`;
    if (trial.coin_policy === 'pin') $('#trial-coin-detail').textContent = coin
      ? `Pinned to ${this.short(coin.mint)} · recording segment ${coin.index + 1} · ${coin.observations} / ${coin.total || '—'} prices. The next segment continues the same coin.`
      : 'Preparing your pinned coin · stays selected until Stop';
    $('.trial-coins-details > summary').textContent = trial.coin_policy === 'pin' ? 'Saved segments of this coin' : 'Coins in this session';
    $('#trial-coin-detail').title = coin?.mint || '';
    if (coin?.total) $('#trial-progress').value = 100 * coin.observations / coin.total;
    else if (trial.status === 'running') $('#trial-progress').removeAttribute('value');
    else $('#trial-progress').value = 0;
    const observed = trial.last_observation_at ? new Date(trial.last_observation_at).getTime() : null;
    const age = observed ? Math.max(0, Math.round((Date.now() - observed) / 1000)) : null;
    const marketTime = coin?.last_timestamp ? new Date(coin.last_timestamp * 1000).toLocaleString() : null;
    $('#trial-freshness').textContent = trial.feed_status === 'retrying' ? `${trial.status === 'running' ? 'FEED PAUSED' : 'FEED UNAVAILABLE'} · ${trial.feed_message}. ${trial.status === 'running' ? 'No new fills until a fresh quote arrives.' : 'The trial has ended; saved values are retained.'}` : trial.source === 'live' ? observed ? `Last fresh observation ${age}s ago · ${marketTime}${trial.status === 'running' ? ' · keep the local server running' : ' · trial has ended'}` : trial.status === 'running' ? 'Waiting for token discovery and a confirmed price. No prices or trades are simulated while waiting.' : 'Trial ended before a supported price was received. See the coin details.' : marketTime ? `Historical price from ${marketTime} · accelerated replay, not the current market` : 'Loading a saved Pump.fun recording.';
    if (trial.status === 'running' && this.running()) {
      $('#operation-detail').textContent = trial.phase;
      if (coin?.total) $('#operation-progress').value = coin.observations / coin.total * 100;
      else $('#operation-progress').removeAttribute('value');
    }
    const frame = this.currentFrame();
    this.renderChart();
    this.renderCoinHealth();
    this.renderMarketAnalysis();
    this.renderPaper(frame ? this.paperForFrame(frame) : trial.paper);
    this.renderLearning(frame?.learning || trial.learning);
    this.renderReward(frame ? frame.reward : trial.reward, frame?.learning || trial.learning);
    this.renderCoins();
    this.renderLedger();
    if (frame) this.frameNote(frame);
    else $('#trial-frame-note').textContent = 'Waiting for the first computed neuron frame. Account figures above are the saved session state.';
  }

  currentFrame() {
    return this.session?.coins.some(coin => coin.activity_path === brain.report) && brain.frame?.paper ? brain.frame : null;
  }

  renderCoinHealth() {
    const trial = this.session;
    const policy = trial.coin_selection;
    $('#trial-coin-health').hidden = trial.source !== 'live' || !policy?.activity_screening;
    if (trial.source !== 'live' || !policy?.activity_screening) return;
    const coin = trial.coins[trial.active_coin];
    const health = coin?.activity;
    const searching = trial.status === 'running' && !trial.activity_path;
    $('#coin-health-title').textContent = searching ? 'Finding an active coin' : coin?.status === 'quiet' ? 'Moved on from a quiet coin' : 'Market activity check';
    $('#coin-health-detail').textContent = searching ? trial.phase : health
      ? `${health.changes_in_window} price changes in ${Math.round(health.window_seconds)} seconds · ${Math.round(health.seconds_since_change)}s since the last movement.${coin.rotation_reason ? ' ' + coin.rotation_reason + '.' : ''}`
      : 'Candidates must pass three fresh price checks before training starts.';
    $('#coin-health-rule').textContent = trial.coin_policy === 'pin'
      ? `Pinned to your coin. Quiet conditions pause new buys while observation continues; it will not switch to another coin.`
      : `Selects recently traded curves from Pump.fun Explore. Moves on after ${policy.quiet_seconds}s without meaningful price movement, or fewer than ${policy.minimum_changes} changes in ${policy.window_seconds}s. Activity is based on observed quotes.`;
  }

  renderMarketAnalysis() {
    const trial = this.session;
    if (!trial) return;
    const frame = this.currentFrame();
    const coin = trial.coins.find(c => c.activity_path === brain.report) || trial.coins[trial.active_coin];
    const context = frame ? frame.chart_context : coin?.chart_context;
    const risk = frame ? frame.risk : coin?.risk;
    const market = (frame ? frame.market : coin?.market) || {};
    const enhanced = ['chart-context-v1', 'mushroom-body-reward-v1'].includes(trial.model_profile);
    $('#market-trend').textContent = risk?.trend || (enhanced ? 'Collecting chart context' : 'Original short-window model');
    const riskLabel = $('#market-risk-label');
    if (riskLabel) {
      riskLabel.textContent = risk?.label || (enhanced ? 'WAITING FOR DATA' : 'OLDER SESSION');
      riskLabel.classList.toggle('negative', Boolean(risk && !risk.entry_allowed));
    }
    const number = value => Number.isFinite(value) ? Number(value).toLocaleString(undefined, {maximumFractionDigits: 2}) : 'Unknown';
    $('#market-cap').textContent = number(market.market_cap_sol);
    $('#market-reserves').textContent = number(market.real_sol);
    $('#market-holder').textContent = Number.isFinite(market.largest_account_pct) ? `${number(market.largest_account_pct)}% of supply` : 'Unknown';
    $('#market-returns').textContent = ['1m', '5m', '15m'].map(k => Number.isFinite(context?.returns_pct?.[k]) ? this.signed(context.returns_pct[k], 1) + '%' : 'Not covered').join(' / ');
    $('#market-history').textContent = context ? `${fmt(context.observations)} context prices since ${new Date(context.first_timestamp * 1000).toLocaleString()}. ${this.signed(context.return_since_start_pct, 2)}% from first price; ${this.signed(context.drawdown_pct, 2)}% from the observed high. Uses 1/5/15-minute windows and longer collected history. ${coin?.history?.points || 0} prior prices loaded. Full lifetime chart is not available.`
      : enhanced ? 'Will use previously captured prices for this coin, then all new observations. History coverage is shown here; unavailable history is not invented.' : 'This session started with the original short-window inputs. New runs use longer chart context and risk checks; old results are retained.';
    const age = trial.last_observation_at ? Math.max(0, Math.round((Date.now() - new Date(trial.last_observation_at)) / 1000)) : null;
    $('#market-timing').textContent = `Brain + risk evaluation: once per confirmed price, requested every ${trial.interval_seconds}s.${context ? ` Last price interval: ${Math.round(context.gap_seconds)}s.` : ''}${age !== null ? ` Latest processing: ${age}s ago.` : ''} The moving map repeats the latest state change; it does not trade between quotes.`;
    $('#market-risk-reasons').replaceChildren(...[...(risk?.reasons || []), ...(risk?.warnings || [])].map(reason => element('li', reason)));
    $('#market-risk-unknown').textContent = risk ? [...risk.unknowns, market.holder_check || ''].filter(Boolean).join(' · ') : '';
  }

  renderChart() {
    const trial = this.session;
    const select = $('#coin-chart-select');
    const coins = trial.coins.filter(coin => coin.bars && coin.observations);
    const signature = JSON.stringify(coins.map(coin => [coin.index, coin.mint]));
    if (select.dataset.signature !== signature) {
      select.replaceChildren();
      for (const coin of coins) {
        const option = element('option', `${trial.coin_policy === 'pin' ? 'Segment' : 'Coin'} ${coin.index + 1} · ${this.short(coin.mint)}`);
        option.value = coin.index; select.append(option);
      }
      select.dataset.signature = signature;
    }
    const current = trial.status === 'running' ? trial.coins[trial.active_coin] : trial.coins.find(coin => coin.activity_path === brain.report) || trial.coins[trial.active_coin];
    if (current) select.value = current.index;
    select.disabled = trial.status === 'running';
    if (currentView === 'trial') this.chart.update(current, trial);
  }

  renderSummary(summary) {
    $('#trial-summary').hidden = !summary;
    if (!summary) return;
    const money = summary.money, learning = summary.learning, activity = summary.activity;
    const pct = value => value === null ? 'not recorded' : `${this.signed(Math.expm1(value) * 100, 3)}%`;
    const noOrders = money.orders === 0;
    $('#summary-headline').textContent = noOrders ? 'No trades were placed.' : money.open_positions ? money.valuation_stale ? 'Session ended with unresolved positions.' : 'Session ended with an open paper position.' : money.pnl_sol > 0 ? 'Session ended with a paper profit.' : money.pnl_sol < 0 ? 'Session ended with a paper loss.' : 'Paper balance unchanged.';
    $('#summary-status').textContent = summary.end_status.replaceAll('_', ' ').toUpperCase();
    const minutes = Math.round((summary.duration_seconds || 0) / 60);
    const duration = summary.duration_seconds === null ? 'Duration unavailable' : minutes >= 60 ? `${Math.floor(minutes / 60)}h ${minutes % 60}m` : minutes ? `${minutes}m` : `${Math.round(summary.duration_seconds)}s`;
    $('#summary-duration').textContent = `${duration} · ${fmt(summary.coins)} coin${summary.coins === 1 ? '' : 's'} · ${fmt(summary.observations)} prices processed · ${new Date(this.session.started).toLocaleString()}`;
    $('#summary-profit').textContent = `${this.signed(money.pnl_sol)} SOL`;
    $('#summary-profit').className = money.valuation_stale ? '' : money.pnl_sol > 0 ? 'positive' : money.pnl_sol < 0 ? 'negative' : '';
    $('#summary-balance').textContent = `${money.pnl_sol === 0 ? 'No profit or loss. ' : ''}${Number(money.initial_sol).toFixed(4)} → ${Number(money.equity_sol).toFixed(4)} SOL${money.valuation_stale ? ' · includes stale, unresolved values' : ' · ' + this.signed(money.return_pct, 2) + '%'}${money.open_positions ? '. ' + money.open_positions + ' open/unresolved position(s); only ' + this.signed(money.realized_sol) + ' SOL realized.' : ''}`;
    $('#summary-orders').textContent = `${fmt(money.orders)} order${money.orders === 1 ? '' : 's'}`;
    $('#summary-fills').textContent = `${fmt(money.buys)} buy${money.buys === 1 ? '' : 's'} · ${fmt(money.sells)} sell${money.sells === 1 ? '' : 's'} · ${fmt(money.closed_trades)} completed trade${money.closed_trades === 1 ? '' : 's'}`;
    $('#summary-learning').textContent = learning.enabled ? `${fmt(learning.new_updates)} new updates` : 'Comparison run';
    const saved = {verified: 'Checkpoint verified on disk', missing: 'Checkpoint file is missing', unverified: 'Checkpoint could not be verified', no_updates: 'No completed learning updates yet', not_used: 'Live learning memory unchanged'}[learning.checkpoint_status];
    $('#summary-memory').textContent = learning.enabled ? `${fmt(learning.restored_updates)} restored + ${fmt(learning.new_updates)} new = ${fmt(learning.total_updates)} total. ${saved}.` : 'This finite trial fitted its own model. Live learning memory was unchanged.';
    $('#summary-reason-title').textContent = noOrders ? 'Why no trades?' : 'How the session traded';
    const reasons = {
      no_data: 'The session ended before it received a usable price. No orders or new price-based learning were possible.',
      unknown: 'The saved decision history is incomplete, so the exact reason cannot be reconstructed.',
      training: `All ${fmt(activity.training)} recorded observations were used for training or warmup. The model had not started issuing trade forecasts.`,
      below_threshold: `None of ${fmt(activity.forecasts)} forecasts reached the buy threshold. The strongest forecast was ${pct(activity.max_forecast)}; the rule requires more than ${pct(activity.entry_threshold)} over the modeled holding period.`,
      sampled_below_threshold: `The available activity samples show forecasts below the buy threshold (${pct(activity.entry_threshold)}). Some observations were not saved, so they cannot establish what happened at every step.`,
      risk_halt: 'The risk limit paused new entries. Price observation and learning could continue while orders were paused.',
      risk_checks: 'Some forecasts qualified, but no buys filled while market-risk or activity checks paused entries. The decision breakdown records the reasons; observation and learning could continue.',
      reward_waiting: 'The reward policy did not fill a paper buy. Check the decision counts for market checks, warmup, cancelled entries, or waiting for exploration. No completed trade means no reward-learning update.',
      no_fills: 'Some forecasts reached the threshold, but no buys filled. Timing, cancelled queued orders, or risk rules can prevent a fill; see the decision breakdown.',
      traded: `${fmt(money.closed_trades)} paper position${money.closed_trades === 1 ? ' was' : 's were'} completed${money.open_positions ? '; ' + money.open_positions + ' remain open or unresolved' : ''}. Costs are included in the displayed paper result.`,
    };
    $('#summary-reason').textContent = reasons[summary.no_trade_reason];
    const quiet = activity.unchanged_price_pct;
    $('#summary-market').textContent = `${quiet === null ? '' : `${quiet.toFixed(1)}% of ${fmt(activity.price_intervals)} checked price intervals were unchanged. `}${summary.holding_periods_seconds ? `The policy chose ${summary.holding_periods_seconds.join(' / ')}-second holds, with earlier risk or segment exits. ` : summary.holding_period_seconds ? `Each paper trade is designed to hold for one price interval (about ${summary.holding_period_seconds}s). ` : ''}${summary.cost_bps_per_side === null || summary.cost_bps_per_side === undefined ? '' : `Assumed trading cost: ${(summary.cost_bps_per_side / 100).toFixed(2)}% per side.`}`;
    const comparison = learning.comparison;
    const quality = comparison.assessment;
    $('#summary-quality-title').textContent = !learning.enabled ? 'Persistent learning was not part of this trial.' : quality === 'worse' ? 'It updated, but predictions were worse than no change.' : quality === 'better' ? 'Predictions beat the no-change comparison.' : quality === 'same' ? 'Predictions matched the no-change comparison.' : 'Not enough checked predictions to judge learning.';
    $('#summary-quality').classList.toggle('negative', quality === 'worse' && learning.enabled);
    $('#summary-quality-detail').textContent = !learning.enabled ? 'Use continuous mode to retain learning across sessions.' : comparison.count < 20 ? `${fmt(comparison.count)} forecasts checked (${comparison.scope}). Learning updates mean parameters were trained; they do not by themselves show better predictions or profit.` : `${comparison.error_ratio === null ? 'The no-change comparison had zero error.' : 'Model error was ' + comparison.error_ratio.toFixed(2) + '× the error from predicting no price change.'} Based on ${fmt(comparison.count)} forecasts (${comparison.scope}), scored before their outcomes were used for training. This comparison measures prediction quality, not trading profitability.`;
    if (learning.profile === 'mushroom-body-reward-v1') {
      $('#summary-market').textContent += ` Reserve impact and an assumed 0.000005 SOL per order are also included. Costs paid: ${Number(money.costs_paid_sol || 0).toFixed(6)} SOL.`;
      $('#summary-quality-title').textContent = `${fmt(learning.new_updates)} closed trades taught the reward circuit this session.`;
      $('#summary-quality-detail').textContent = `${fmt(comparison.count)} net-return predictions checked before learning from their outcomes. ${comparison.error_ratio == null ? 'More completed trades are needed to compare errors.' : `Prediction error was ${comparison.error_ratio.toFixed(2)}× the error of predicting zero net return.`} This measures prediction accuracy; the paper P/L above measures trading results.`;
    }
    const decisions = $('#summary-decisions'); decisions.replaceChildren();
    for (const [reason, count] of Object.entries(activity.decisions).sort((a, b) => b[1] - a[1])) {
      const row = element('div', undefined, 'summary-decision-row');
      row.append(element('span', reason), element('strong', fmt(count)));
      decisions.append(row);
    }
    $('#summary-coverage').textContent = `${fmt(activity.observations)} / ${fmt(summary.observations)} observations have decision details · ${activity.coverage === 'complete' ? 'complete coverage' : activity.coverage === 'sampled' ? 'sampled activity; counts cover saved samples only' : 'decision history unavailable'}.`;
  }

  renderPaper(paper) {
    if (!paper) return;
    $('#trial-equity').textContent = `${Number(paper.equity_sol).toFixed(4)} SOL`;
    $('#trial-map-equity').textContent = `${Number(paper.equity_sol).toFixed(4)} SOL${paper.valuation_stale ? ' · stale value' : ''}`;
    $('#trial-cash').textContent = `${Number(paper.cash_sol).toFixed(4)} SOL available cash`;
    $('#trial-pnl').textContent = `${this.signed(paper.pnl_sol)} SOL`;
    $('#trial-pnl').className = paper.pnl_sol < 0 ? 'negative' : 'positive';
    $('#trial-map-pnl').textContent = `${this.signed(paper.pnl_sol)} SOL`;
    $('#trial-map-pnl').className = paper.pnl_sol < 0 ? 'negative' : 'positive';
    $('#trial-realized').textContent = `${this.signed(paper.return_pct, 2)}% · realized ${this.signed(paper.realized_sol)} SOL`;
    $('#trial-trades').textContent = fmt(paper.closed_trades);
    $('#trial-map-trades').textContent = fmt(paper.closed_trades);
    $('#trial-wins').textContent = `${paper.wins} profitable · ${Number(paper.drawdown_pct).toFixed(2)}% closed-trade drawdown`;
    const stalePosition = paper.position_stale ?? paper.valuation_stale;
    $('#trial-position').textContent = paper.position ? stalePosition ? 'Unresolved' : 'Open' : paper.pending ? 'Buy queued' : paper.unresolved?.length ? 'Unresolved tokens' : 'In cash';
    $('#trial-exposure').textContent = paper.position ? `${Number(paper.position.units).toPrecision(5)} tokens · ${stalePosition ? 'last known value; no exit quote' : paper.position.exit_timestamp ? `${paper.position.entry_kind} · planned exit ${new Date(paper.position.exit_timestamp * 1000).toLocaleTimeString()}` : 'exit on the next fresh observation'}` : paper.pending ? 'Fills on the next fresh observation' : paper.unresolved?.length ? 'Some tokens await an exit quote; see details below' : 'No open paper position';
    $('#trial-decision').textContent = paper.decision;
    $('#trial-map-decision').textContent = paper.decision;
    $('#trial-signal').textContent = paper.prediction === null ? 'Learning from earlier prices; orders begin after training.' : `Predicted log return ${Number(paper.prediction).toFixed(4)} · entry threshold > 0.0400`;
    if (this.session?.model_profile === 'mushroom-body-reward-v1') $('#trial-signal').textContent = paper.prediction == null ? 'Warming the circuit with six fresh quotes.' : `Best MBON estimate: ${this.signed(Math.expm1(paper.prediction) * 100, 2)}% net return. Exploration can buy while this estimate is weak; learned entries require positive estimates and completed examples.`;
    if (paper.execution_model === 'constant-product shadow reserves') $('#trial-cash').textContent += ` · costs paid ${Number(paper.costs_paid_sol || 0).toFixed(6)} SOL`;
    $('#trial-unresolved').hidden = !paper.unresolved?.length;
    $('#trial-unresolved').textContent = `${paper.unresolved?.length || 0} position(s) have no supported exit quote. Their last values are stale, their allocated cash stays locked, and they are not recorded as sales. New entries pause when locked cost reaches 10% of starting cash; learning continues.`;
  }

  renderLearning(learning) {
    $('#trial-learning').hidden = !learning?.enabled;
    $('#trial-map-learning').hidden = !learning?.enabled;
    if (!learning?.enabled) return;
    const reward = learning.profile === 'mushroom-body-reward-v1';
    $('#learning-comparison-label').textContent = reward ? 'Compared with predicting zero net return' : 'Compared with predicting no price change';
    $('#learning-scope').textContent = `${learning.scope}. Error is measured before each outcome is used for learning. Lower prediction error does not establish profitable trading.`;
    $('#trial-map-learning').textContent = `${fmt(learning.examples)} outcomes learned · ${fmt(learning.resumed_examples)} restored`;
    $('#learning-status').textContent = this.session?.status !== 'running' ? 'Learning paused · memory retained' : learning.examples >= 38 ? 'Learning from new outcomes' : 'Building initial experience';
    if ($('#learning-badge')) $('#learning-badge').textContent = this.session?.status === 'running' ? 'AUTO-SAVING' : 'MEMORY SAVED';
    $('#learning-examples').textContent = fmt(learning.examples);
    $('#learning-session').textContent = `${fmt(learning.session_updates)} this session · ${fmt(learning.resumed_examples)} restored`;
    $('#learning-error').textContent = learning.evaluated < 20 ? `${learning.evaluated} / 20 checked` : Math.abs(learning.prediction_mae - learning.baseline_mae) < 1e-8 ? 'Same error' : learning.prediction_mae < learning.baseline_mae ? 'Lower error' : 'Higher error';
    $('#learning-baseline').textContent = learning.baseline_mae === null ? 'Waiting for predictions to mature' : `Model error ${Number(learning.prediction_mae).toFixed(4)} · no-change error ${Number(learning.baseline_mae).toFixed(4)} · ${fmt(learning.evaluated)} forecasts checked before learning from them. Mean absolute error in log-return units.`;
    const outcome = learning.last_outcome;
    $('#learning-latest').textContent = outcome ? `Latest learned outcome: ${this.short(outcome.mint)} · signal #${outcome.signal_step} · price change ${this.signed(Math.expm1(outcome.observed) * 100, 2)}%${outcome.predicted === null ? ' · initial training' : ' · forecast ' + this.signed(Math.expm1(outcome.predicted) * 100, 2) + '%'}. Saved ${new Date(learning.last_update).toLocaleTimeString()}.` : 'Waiting for two later observations to make the first eligible outcome known.';
    if (reward) {
      $('#learning-status').textContent = this.session?.status === 'running' ? 'Learning after each completed paper trade' : 'Reward learning paused · memory retained';
      $('#learning-baseline').textContent = learning.prediction_mae == null ? 'The first completed paper sale supplies the first reward.' : `Net-return prediction error ${Number(learning.prediction_mae).toFixed(4)} · zero-return error ${Number(learning.baseline_mae).toFixed(4)} · ${fmt(learning.evaluated)} closed trades. Mean absolute error in log-return units.`;
      $('#learning-latest').textContent = outcome ? `Latest: ${outcome.entry_kind} · ${outcome.hold_seconds}s planned hold · ${this.signed(outcome.pnl_sol, 5)} SOL after fees. Saved ${new Date(learning.last_update).toLocaleTimeString()}.` : 'Open positions and cancelled orders do not count as reward-learning updates.';
    }
  }

  renderReward(reward, learning) {
    $('#reward-circuit').hidden = !reward;
    if (!reward) return;
    const outcome = learning?.last_outcome;
    $('#reward-decision').textContent = this.session?.status === 'running' ? reward.decision : 'Learning paused · saved circuit state';
    $('#reward-gain').textContent = `POSITIVE TEACHING ${learning?.reward_gain ?? 1}×`;
    $('#reward-outcome').textContent = outcome ? `${this.signed(outcome.pnl_sol, 5)} SOL` : 'No closed trade yet';
    $('#reward-outcome').className = outcome?.pnl_sol < 0 ? 'negative' : 'positive';
    $('#reward-error').textContent = outcome ? `Net return ${this.signed(Math.expm1(outcome.observed) * 100, 2)}% · prediction error ${this.signed(outcome.reward_error, 4)}` : 'A buy alone earns no teaching reward.';
    $('#reward-changed').textContent = fmt(reward.changed_connections);
    $('#reward-weights').textContent = `of ${fmt(reward.plastic_connections)} KC → MBON connections · mean absolute change ${Number(reward.mean_weight_change_pct).toFixed(5)}%`;
    for (const [name, key] of [['pam', 'PAM'], ['ppl', 'PPL1']]) {
      const value = reward.dopamine_activity?.[key] || 0;
      $(`#${name}-activity`).value = value;
      $(`#${name}-value`).textContent = Number(value).toFixed(3);
    }
    $('#reward-cells').textContent = Object.entries(reward.neurons || {}).map(([name, count]) => `${fmt(count)} ${name} cells`).join(' · ') + ` · ${reward.approach_mbons} approach / ${reward.avoid_mbons} avoidance outputs`;
    $('#reward-opponents').textContent = `${Number(reward.approach || 0).toFixed(3)} / ${Number(reward.avoid || 0).toFixed(3)}`;
    $('#reward-bias').textContent = `Bias ${this.signed(reward.bias || 0, 4)} · normalized against unchanged connections`;
    $('#reward-sparsity').textContent = `${((reward.kc_active_fraction || 0) * 100).toFixed(1)}% of KCs active`;
    $('#reward-regime').textContent = `${reward.regime === 'pumpswap' ? 'PumpSwap' : reward.regime === 'bonding_curve' ? 'Bonding curve' : 'Regime unavailable'} · novelty ${((reward.novelty || 0) * 100).toFixed(0)}% · ${((reward.kc_used_fraction || 0) * 100).toFixed(1)}% of KCs used so far`;
    $('#reward-actions').textContent = 'MBON net-return estimates: ' + (reward.action_values || []).map(action => `${action.hold_seconds}s: ${this.signed(Math.expm1(action.expected_net_log_return) * 100, 2)}% (${action.completed_examples} closed)`).join(' · ');
  }

  onFrame(frame) {
    if (!frame.paper || !this.session?.coins.some(coin => coin.activity_path === brain.report)) return;
    this.renderPaper(this.paperForFrame(frame));
    this.renderLearning(frame.learning || this.session.learning);
    this.renderReward(frame.reward, frame.learning || this.session.learning);
    this.chart.setFrame(frame);
    this.frameNote(frame);
    this.renderMarketAnalysis();
    this.renderLedger();
  }

  paperForFrame(frame) {
    const coin = this.session.coins[this.session.active_coin];
    // A terminal or paused session can cancel a queued order without receiving another price.
    if (frame.mint === coin?.mint && frame.step === coin.observations &&
        (this.session.status !== 'running' || this.session.feed_status === 'retrying')) return this.session.paper;
    return frame.paper;
  }

  frameNote(frame) {
    $('#trial-frame-note').textContent = `Account + neurons: ${this.short(frame.mint)} · observation ${frame.step} · ${frame.phase} · ${new Date(frame.timestamp * 1000).toLocaleString()}${brain.live ? '' : ' · saved playback. Whole-session results above stay fixed.'}`;
  }

  renderCoins() {
    const list = $('#trial-coin-list'); list.replaceChildren();
    const trial = this.session;
    for (const candidate of trial.skipped_candidates || []) {
      const note = element('div', undefined, 'trial-coin');
      note.append(element('strong', `Skipped ${this.short(candidate.mint)}`), element('small', candidate.error, 'muted'));
      list.append(note);
    }
    trial.coins.forEach(coin => {
      const card = element('div', undefined, 'trial-coin');
      card.append(element('strong', `${coin.index + 1}. ${this.short(coin.mint)}`), element('span', `${coin.status} · ${coin.observations} prices · ${coin.closed_trades} trade${coin.closed_trades === 1 ? '' : 's'}`));
      if (coin.error) card.append(element('small', coin.error, 'negative'));
      if (coin.rotation_reason) card.append(element('small', coin.rotation_reason, 'muted'));
      if (coin.observations) {
        const button = element('button', 'Inspect neurons ↗', 'text-button');
        button.disabled = state.jobs.some(job => ['running', 'stopping'].includes(job.status));
        button.onclick = async () => { await this.openActivity(coin.activity_path); $('.brain-panel').scrollIntoView({behavior: 'smooth'}); };
        card.append(button);
      }
      list.append(card);
    });
  }

  renderLedger() {
    const body = $('#trial-ledger-body'); body.replaceChildren();
    let events = this.session?.events || [];
    const frame = this.currentFrame();
    if (frame) {
      const coinIndex = this.session.coins.findIndex(coin => coin.activity_path === brain.report);
      events = events.filter(event => {
        const index = event.coin_index ?? this.session.coins.find(coin => coin.mint === event.mint)?.index;
        const selectedIndex = this.session.coins[coinIndex].index;
        return index < selectedIndex || (index === selectedIndex && event.step <= frame.step);
      });
    }
    const sessionOrders = this.session?.summary?.money.orders ?? this.session?.paper.orders;
    const emptyReason = sessionOrders === 0 && this.session?.status !== 'running' ? 'No orders were placed during this session. The session results above explain why.' : `No paper orders through this observation.${sessionOrders ? ' The whole session placed ' + fmt(sessionOrders) + ' orders.' : ' The model may be training or holding cash.'}`;
    $('#trial-ledger-note').textContent = events.length ? `${events.length} recent paper orders through the displayed observation · fills include assumed costs${this.session?.paper.orders > 500 ? ' · full ledger saved locally' : ''}` : emptyReason;
    $('#trial-all-trades').hidden = events.length <= 12;
    $('#trial-all-trades').textContent = this.showAll ? 'Show latest 12' : 'Show all trades';
    if (!events.length) {
      const row = element('tr'), cell = element('td', emptyReason, 'empty');
      cell.colSpan = 6; row.append(cell); body.append(row); return;
    }
    for (const event of (this.showAll ? events : events.slice(-12)).toReversed()) {
      const row = element('tr');
      row.append(element('td', `#${event.step} · ${new Date(event.timestamp * 1000).toLocaleTimeString()}`),
        element('td', this.short(event.mint)), element('td', event.side + (event.entry_kind ? ` · ${event.entry_kind}` : ''), event.side === 'BUY' ? 'positive' : ''),
        element('td', Number(event.fill_price).toPrecision(6)), element('td', Number(event.amount_sol).toFixed(5)),
        element('td', event.pnl_sol === null ? '—' : this.signed(event.pnl_sol, 5), event.pnl_sol < 0 ? 'negative' : 'positive'));
      body.append(row);
    }
  }

  renderHistory(items) {
    const signature = JSON.stringify(items.map(item => [item.path, item.status, item.updated]));
    if (signature === this.lastHistory) return;
    this.lastHistory = signature;
    const list = $('#trial-history'); list.replaceChildren();
    if (!items.length) list.append(element('p', 'Your trials will stay here after you close the page or restart the server.'));
    for (const item of items) {
      const row = element('div', undefined, 'trial-history-row');
      const label = element('div');
      const trades = item.paper?.closed_trades || 0;
      label.append(element('strong', `${item.source === 'live' ? 'Live prices' : 'Saved prices'} · ${new Date(item.started).toLocaleString()}`), element('small', `${item.status} · ${item.coin_count} coin${item.coin_count === 1 ? '' : 's'} · ${trades} completed paper trade${trades === 1 ? '' : 's'} · ${this.signed(item.paper?.pnl_sol || 0)} SOL${item.learning?.enabled ? ' · ' + fmt(item.learning.session_updates) + ' learning updates' : ''}`));
      const button = element('button', 'Open trial ↗', 'text-button');
      button.onclick = () => {
        if (state.jobs.some(job => ['running', 'stopping'].includes(job.status))) {
          message('#error', 'Finish or stop the current operation before opening another trial.'); return;
        }
        this.path = item.path; this.session = null; this.showAll = false;
        this.scrollToSummary = true;
        localStorage.setItem('flytrader.trial', this.path);
        navigate('trial'); this.poll();
      };
      row.append(label, button); list.append(row);
    }
  }

  disconnected() {
    $('#start-trial').disabled = true;
    $('#trial-freshness').textContent = 'Dashboard disconnected. Last displayed values are unchanged; reconnect to check the trial process.';
  }
}

window.trialDashboard = new TrialDashboard();
