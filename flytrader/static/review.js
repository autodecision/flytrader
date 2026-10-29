/* Learning checks use isolated replays; this view never alters the live circuit. */
class LearningReview {
  constructor() {
    this.selected = '';
    this.signature = '';
    $('#run-learning-check').onclick = () => {
      const form = $('#trial-form');
      const body = {action: 'learning-check', edges: activeGraph};
      this.selected = '';
      for (const name of ['cost', 'interval', 'reward_gain', 'exploration', 'min_cap_sol', 'min_reserves_sol']) body[name] = form.elements[name].value;
      start(body);
    };
    $('#learning-check-history').onchange = event => {
      this.selected = event.target.value; this.signature = ''; this.refresh(state);
    };
    if (state) this.refresh(state);
  }

  refresh(next) {
    if (!next) return;
    const supported = next.capabilities?.includes('learning-check-v1');
    const plan = next.evaluation_plan;
    const running = next.jobs.find(job => ['running', 'stopping'].includes(job.status));
    const full = next.files.find(file => file.path === activeGraph)?.metadata.scope === 'full-retained';
    const button = $('#run-learning-check');
    button.disabled = !supported || !plan?.ready || !full || Boolean(running) || submitting;
    button.textContent = running?.action === 'learning-check' ? 'Checking learning…' : 'Check learning ↗';
    $('#learning-check-ready').textContent = !supported ? 'Restart the dashboard server to enable learning checks.' :
      running?.action === 'learning-check' ? running.log.at(-1) || 'Preparing isolated replay copies…' :
      running ? 'Available after the current operation stops.' : !full ? 'Choose the full retained model in Brain settings.' :
      plan?.ready ? `Ready: ${plan.train_coins} training coins · ${plan.test_coins} later unseen coins · ${plan.test_observations} test prices.` : plan?.reason || 'No completed reserve recordings yet.';
    const reports = next.evaluations || [];
    const signature = JSON.stringify([reports, this.selected]);
    if (signature === this.signature) return;
    this.signature = signature;
    $('#learning-check-result').hidden = !reports.length;
    if (!reports.length) return;
    const current = reports.find(report => report.path === this.selected) || reports[0];
    const select = $('#learning-check-history'); select.replaceChildren();
    for (const report of reports) {
      const option = element('option', new Date(report.created).toLocaleString());
      option.value = report.path; select.append(option);
    }
    select.value = current.path;
    $('#learning-check-verdict').textContent = current.summary.verdict;
    $('#learning-check-sample').textContent = current.summary.note;
    const names = {opponent: 'Trained fly + synaptic learning', readout_only: 'Output learning only',
      no_learning: 'Untrained fly', random: 'Random entries', momentum: 'Simple momentum', cash: 'Hold cash'};
    const tbody = $('#learning-check-scores'); tbody.replaceChildren();
    const signed = value => `${value > 0 ? '+' : ''}${Number(value).toFixed(4)}`;
    for (const [policy, name] of Object.entries(names)) {
      const score = current.aggregate[policy];
      if (!score) continue;
      const row = element('tr');
      const label = element('td'); label.append(element('strong', name));
      if (['opponent', 'readout_only', 'no_learning'].includes(policy)) label.append(element('small', `${score.learned_entries.toFixed(1)} learned / ${score.exploration_entries.toFixed(1)} exploratory entries`));
      if (score.unresolved) label.append(element('small', `${score.unresolved} unresolved exit(s) · valuation incomplete`));
      const pnl = element('td', signed(score.net_pnl_sol), score.net_pnl_sol < 0 ? 'negative' : score.net_pnl_sol > 0 ? 'positive' : '');
      pnl.append(element('small', `${signed(score.min_pnl_sol)} to ${signed(score.max_pnl_sol)}`));
      row.append(label, pnl, element('td', score.costs_sol.toFixed(4)), element('td', score.closed_trades.toFixed(1)));
      tbody.append(row);
    }
    $('#learning-check-circuit').textContent = `Effect of synaptic learning in this check: ${signed(current.summary.advantage_vs_readout_only_sol)} SOL versus output learning alone.`;
    $('#learning-check-limits').textContent = 'These are tests of the training method, not a score for your saved live brain. A small historical comparison does not establish a profitable strategy.';
    const settings = current.settings;
    $('#learning-check-settings').textContent = `10 SOL starting paper balance per policy · ${settings.cost_bps} bps assumed cost per side plus fixed transaction costs and reserve impact · ${settings.interval_seconds}s reference interval (recorded timestamps preserved) · minimum cap ${settings.min_cap_sol} SOL / reserves ${settings.min_reserves_sol} SOL · ${Math.round(settings.exploration * 100)}% exploration · ${settings.reward_gain}× teaching gain · seeds ${current.seeds.join(', ')}.`;
    $('#learning-check-files').textContent = `${current.data.train_observations} training / ${current.data.test_observations} test observations. ${current.data.excluded.length} recording(s) excluded for split or data requirements. Saved report: ${current.path}`;
  }
}
window.learningReview = new LearningReview();
