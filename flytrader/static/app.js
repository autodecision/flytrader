const $ = selector => document.querySelector(selector);
const fmt = value => Number(value).toLocaleString();
let state = null;
let activeGraph = localStorage.getItem('flytrader.graph.v2') || '';
let activeBars = localStorage.getItem('flytrader.bars') || '';
let latestJob = null;
let selectedJob = null;
let submitting = false;
let refreshing = false;
let loadedGraph = null;
let viewedReport = null;
let currentView = 'trial';
const titles = {
  trial: ['Flytrader Lab', 'Train on active Pump.fun coins and watch the model work.', 'Pump.fun trials'],
  experiment: ['Compare models on a recording.', 'Choose a recording, run a paper test, and watch real model activity.', 'Replay lab'],
  capture: ['Record the market as it happens.', 'Save Pump.fun price observations for a paper test.', 'Pump.fun'],
  connectome: ['Choose the scale of your model.', 'Use the full retained connectome or a smaller circuit for a quick comparison.', 'Brain settings'],
  history: ['Return to an experiment.', 'Compare saved results and revisit recorded neuron activity.', 'Past results'],
  about: ['About Flytrader', '', 'About Flytrader'],
};

function element(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

function navigate(view) {
  const computing = state?.jobs.find(job => ['running', 'stopping'].includes(job.status) && ['replay', 'trial'].includes(job.action));
  if ((view === 'trial' && computing?.action === 'replay') || (view === 'experiment' && computing?.action === 'trial')) {
    message('#error', 'Finish or stop the current run before switching between trials and the replay lab.');
    return;
  }
  currentView = view;
  if (view === 'trial') $('#results').hidden = true;
  document.querySelectorAll('.view').forEach(node => node.hidden = node.id !== view);
  document.querySelectorAll('.nav').forEach(node => node.classList.toggle('active', node.dataset.view === view));
  $('#page-title').textContent = titles[view][0];
  $('#page-subtitle').textContent = titles[view][1];
  $('#page-subtitle').hidden = !titles[view][1];
  $('#breadcrumb').textContent = titles[view][2];
  $('#help-toggle').hidden = view === 'about';
  if (view === 'about') {
    $('#help-box').hidden = true;
    $('#help-toggle').setAttribute('aria-expanded', 'false');
  }
  if (['trial', 'experiment'].includes(view)) {
    $('#trial-form').hidden = view !== 'trial';
    $('#trial-map-summary').hidden = view !== 'trial';
    $(view === 'trial' ? '#trial-brain-slot' : '#replay-brain-slot').append($('.brain-panel'));
    requestAnimationFrame(() => brain.resize());
  }
}

document.querySelectorAll('[data-view]').forEach(node => node.onclick = event => {
  if (node.tagName === 'A') event.preventDefault();
  navigate(node.dataset.view);
});

function getSoundTheme() {
  try {
    const raw = localStorage.getItem('backgroundShaderSettings');
    if (raw) {
      const parsed = JSON.parse(raw);
      const theme = parsed?.panelTheme || parsed?.shaderMode;
      if (theme) {
        const lower = theme.toLowerCase();
        if (lower === 'xp' || lower.startsWith('xp-') || lower.startsWith('xp/')) return 'xp';
        if (lower === 'cicada' || lower.startsWith('cicada-') || lower.startsWith('cicada/')) return 'cicada';
        return 'original';
      }
    }
    const siteTheme = localStorage.getItem('siteTheme');
    if (siteTheme) {
      const lower = siteTheme.toLowerCase();
      if (lower === 'xp' || lower.startsWith('xp-') || lower.startsWith('xp/')) return 'xp';
      if (lower === 'cicada' || lower.startsWith('cicada-') || lower.startsWith('cicada/')) return 'cicada';
      return 'original';
    }
  } catch {}
  return 'cicada';
}

function playNavSoundAndGo(url) {
  const theme = getSoundTheme();
  let isMuted = false;
  try {
    isMuted = localStorage.getItem('funthing.sound.muted') === 'true';
  } catch {}

  if (!isMuted) {
    const soundUrl = theme === 'cicada'
      ? '/media/audio/ocean/message-attention.oga'
      : (theme === 'xp' ? '/media/audio/xp/Windows%20Navigation%20Start.wav' : '/media/audio/floraphonic-cute-level-up-3-189853.mp3');
    try {
      const audio = new Audio(soundUrl);
      let volume = 1;
      const storedVol = localStorage.getItem('funthing.sound.volume');
      if (storedVol !== null) {
        const parsed = parseFloat(storedVol);
        if (!isNaN(parsed)) volume = Math.max(0, Math.min(1, parsed));
      }
      audio.volume = volume;
      audio.play().catch(() => {});
    } catch {}
    setTimeout(() => {
      window.location.assign(url);
    }, 350);
  } else {
    window.location.assign(url);
  }
}

document.querySelectorAll('a[href="/"]').forEach(link => {
  link.onclick = event => {
    event.preventDefault();
    playNavSoundAndGo('/');
  };
});

try {
  if (localStorage.getItem('funthing.sound.muted') !== 'true') {
    const theme = getSoundTheme();
    const preloadUrl = theme === 'cicada'
      ? '/media/audio/ocean/message-attention.oga'
      : (theme === 'xp' ? '/media/audio/xp/Windows%20Navigation%20Start.wav' : '/media/audio/floraphonic-cute-level-up-3-189853.mp3');
    const preloadAudio = new Audio(preloadUrl);
    preloadAudio.preload = 'auto';
  }
} catch {}

$('#help-toggle').onclick = () => {
  $('#help-box').hidden = !$('#help-box').hidden;
  $('#help-toggle').setAttribute('aria-expanded', String(!$('#help-box').hidden));
};

function message(selector, text) {
  $(selector).textContent = text;
  $(selector).hidden = !text;
}

async function request(url, body) {
  if (body && state?.public_mode) {
    const allowed = url.endsWith('/api/stop') ? state.can_stop : state.can_run;
    if (!allowed) {
      if (url.endsWith('/api/stop')) {
        throw new Error(state.override_cooldown_seconds
          ? `You recently stopped another user's paper test. Please wait ${state.override_cooldown_seconds}s before stopping another test.`
          : 'No paper test is currently running to stop.');
      } else {
        throw new Error(state.override_cooldown_seconds
          ? `You recently stopped another user's paper test. Please wait ${state.override_cooldown_seconds}s before starting another test.`
          : 'The shared Flytrader worker is busy. Try again when the current paper test finishes.');
      }
    }
  }
  const response = await fetch(url, body ? {
    method: 'POST', headers: {'Content-Type': 'application/json', 'X-Flytrader-Token': state?.token || ''}, body: JSON.stringify(body),
  } : {});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'The operation could not be completed.');
  return data;
}

function applyPublicMode() {
  const publicMode = Boolean(state?.public_mode);
  $('#public-mode').hidden = !publicMode;
  if (!publicMode) return;

  const running = state.jobs?.find(job => ['running', 'stopping'].includes(job.status));
  const cooldown = state.override_cooldown_seconds || 0;
  const statusEl = $('#public-mode-status');

  if (running) {
    if (running.status === 'stopping') {
      if (statusEl) statusEl.textContent = 'Stopping simulation…';
      $('#public-mode-copy').textContent = 'The current run is stopping safely. You can start a new run as soon as it finishes.';
    } else if (cooldown > 0) {
      if (statusEl) statusEl.textContent = '● A run is currently active';
      $('#public-mode-copy').textContent = `You recently stopped another user's test, with ${cooldown}s remaining on your cooldown before you can stop or start another test.`;
    } else if (state.is_owner) {
      if (statusEl) statusEl.textContent = '● A run is currently active';
      $('#public-mode-copy').textContent = 'You can stop the run at any time and begin your own, or you can view the run in progress.';
    } else {
      if (statusEl) statusEl.textContent = '● A run is currently active';
      $('#public-mode-copy').textContent = 'You can stop the current run to start your own, or observe the one in progress.';
    }
  } else {
    if (cooldown > 0) {
      if (statusEl) statusEl.textContent = '○ Cooldown active';
      $('#public-mode-copy').textContent = `No simulation is currently running. Please wait ${cooldown}s for your override cooldown to expire before starting a new run.`;
    } else {
      if (statusEl) statusEl.textContent = '○ Server idle';
      $('#public-mode-copy').textContent = 'No simulation is currently running. You can set your parameters and start a new run now.';
    }
  }

  for (const selector of [
    '#run-learning-check', '#demo-button', '#build-full-button',
    '#capture-form button[type="submit"]', '#prepare-form button[type="submit"]',
  ]) {
    const control = $(selector);
    if (control) {
      control.disabled = true;
      control.title = 'This operation is disabled on the public server';
    }
  }
  $('#run-button').disabled = !state.can_run || submitting;
  $('#start-trial').disabled = !state.can_run || submitting;
  $('#stop-button').disabled = !state.can_stop;
  $('#stop-trial').disabled = !state.can_stop;
  if (cooldown > 0) {
    const cdNotice = `Cooldown: wait ${cooldown}s before starting or stopping another test`;
    $('#stop-button').title = cdNotice;
    $('#stop-trial').title = cdNotice;
    $('#run-button').title = cdNotice;
    $('#start-trial').title = cdNotice;
  } else {
    $('#stop-button').removeAttribute('title');
    $('#stop-trial').removeAttribute('title');
    $('#run-button').removeAttribute('title');
    $('#start-trial').removeAttribute('title');
  }
}

const brain = new BrainViewer((percent, phase) => {
  $('#operation-progress').value = percent;
  $('#operation-detail').textContent = `${Math.round(percent)}% of observations processed · ${phase}`;
}, frame => window.trialDashboard?.onFrame(frame));

function fillOptions(select, files, value) {
  const signature = JSON.stringify(files.map(file => [file.path, file.label, file.ready]));
  if (select.dataset.signature !== signature) {
    select.replaceChildren();
    for (const file of files) {
      const date = new Date(file.modified * 1000).toLocaleString([], {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'});
      const showDate = file.kind === 'bars' || file.label.includes('saved subset');
      const option = element('option', file.label + (showDate ? ` · ${date}` : '') + (!file.ready ? ' · not enough data yet' : ''));
      option.value = file.path;
      select.append(option);
    }
    select.dataset.signature = signature;
  }
  if (files.some(file => file.path === value)) select.value = value;
  return select.value;
}

function graphDetails() {
  const graph = state.files.find(file => file.path === activeGraph);
  const metadata = graph?.metadata || {};
  const full = metadata.scope === 'full-retained';
  $('#graph-summary').textContent = graph?.synthetic ? 'Synthetic graph · for testing the tool' : `${fmt(metadata.matched_neurons || 0)} neurons · ${full ? 'full retained brain + nerve cord' : 'small selected circuit'}`;
  $('#full-model-note').textContent = state.files.some(file => file.metadata.scope === 'full-retained') ? 'The full retained model is ready. Every retained neuron and connection is used during a test; only the activity display is sampled.' : 'Build the full model from the downloaded source to use all annotated neuronal superclasses.';
  const details = $('#graph-details');
  details.replaceChildren();
  for (const [name, value] of [
    ['Dataset', graph?.synthetic ? 'Synthetic' : metadata.dataset || 'Imported graph'],
    ['Model neurons', metadata.matched_neurons ? fmt(metadata.matched_neurons) : 'Not recorded'],
    ['Connections', metadata.edges ? fmt(metadata.edges) : 'Not recorded'],
    ['Model scope', full ? 'Full retained neuronal graph' : 'Selected circuit'],
  ]) {
    const row = element('div', undefined, 'fact');
    row.append(element('span', name), element('strong', value));
    details.append(row);
  }
  if (activeGraph && loadedGraph !== activeGraph) {
    loadedGraph = activeGraph;
    brain.showGraph(activeGraph);
  }
}

function setGraph(path) {
  activeGraph = path;
  localStorage.setItem('flytrader.graph.v2', path);
  $('#replay-edges').value = path;
  $('#active-graph').value = path;
  graphDetails();
}
$('#replay-edges').onchange = event => setGraph(event.target.value);
$('#active-graph').onchange = event => setGraph(event.target.value);
$('#replay-bars').onchange = event => {
  activeBars = event.target.value;
  localStorage.setItem('flytrader.bars', activeBars);
  updateReadiness();
};

function updateReadiness() {
  const recording = state.files.find(file => file.path === activeBars);
  const running = state.jobs.some(job => ['running', 'stopping'].includes(job.status));
  const ready = !!recording?.ready && !!activeGraph;
  $('#data-summary').textContent = recording ? recording.ready ? (recording.synthetic ? 'Synthetic prices · tool test only' : 'Ready for a paper test') : `Needs ${Math.max(0, 100 - recording.count)} more prices before a test` : 'Record Pump.fun prices to begin';
  $('#run-button').disabled = !ready || running || submitting;
  $('#empty-start').hidden = state.files.some(file => file.kind === 'bars' && file.ready);
  const full = state.files.find(file => file.path === activeGraph)?.metadata.scope === 'full-retained';
  $('#run-hint').textContent = !ready ? 'Choose a recording with at least 100 prices' : full ? 'All retained neurons run · duration depends on recording length' : 'Simulated trades · typically 5–20 seconds';
  for (const selector of ['#replay-edges', '#active-graph', '#replay-bars']) $(selector).disabled = running;
}

function recordingName(report) {
  const path = report.input_files?.bars || '';
  return state.files.find(file => path.endsWith('/' + file.path) || path === file.path)?.label || path.split('/').at(-1) || 'Unknown recording';
}

function showReport(item, scroll = false) {
  viewedReport = item;
  const report = item.report;
  $('#results').hidden = false;
  $('#result-name').textContent = recordingName(report);
  const box = $('#result-content'); box.replaceChildren();
  const grid = element('div', undefined, 'result-grid');
  for (const [name, score] of [['Your brain model', report.reservoir], ['Simple linear model', report.linear_baseline], ['Keep cash', report.cash_baseline]]) {
    const card = element('div', undefined, 'result-stat');
    card.append(element('small', name), element('strong', `${Number(score.return_pct).toFixed(2)}%`, score.return_pct < 0 ? 'negative' : 'positive'));
    const trades = score.trades || 0;
    card.append(element('p', `${trades} simulated trade${trades === 1 ? '' : 's'}`));
    card.append(element('p', `${Number(score.closed_trade_drawdown_pct || 0).toFixed(2)}% closed-trade drawdown`));
    grid.append(card);
  }
  box.append(grid);
  box.append(element('p', `${fmt(report.neurons)} neurons · ${report.train_bars} training / ${report.test_bars} test prices · ${report.cost_bps_per_side} bps assumed cost per side`, 'result-note'));
  if (report.activity_file) {
    const button = element('button', 'Play this test’s neuron activity ↗', 'secondary');
    button.onclick = () => {
      if (state.jobs.some(job => ['running', 'stopping'].includes(job.status))) {
        message('#error', 'Finish or stop the current operation before opening another recording.');
        return;
      }
      const graph = state.files.find(file => file.kind === 'edges' && (report.input_files?.edges === file.path || report.input_files?.edges?.endsWith('/' + file.path)));
      if (graph) setGraph(graph.path);
      const bars = state.files.find(file => file.kind === 'bars' && (report.input_files?.bars === file.path || report.input_files?.bars?.endsWith('/' + file.path)));
      if (bars) {
        activeBars = bars.path;
        $('#replay-bars').value = activeBars;
        localStorage.setItem('flytrader.bars', activeBars);
        updateReadiness();
      }
      navigate('experiment'); brain.attach(item.path); $('.brain-panel').scrollIntoView({behavior: 'smooth'});
    };
    box.append(button);
  } else {
    box.append(element('p', 'This older result has no neuron recording. Run a new test to capture activity.', 'result-note'));
  }
  const details = element('details', undefined, 'report-details');
  details.append(element('summary', 'Test assumptions and reproducibility'));
  details.append(element('p', report.limitations.join(' · ')));
  if (report.provenance) details.append(element('p', `Input SHA-256: ${report.provenance.inputs.bars.sha256}. Random seed: ${report.provenance.seed}.`));
  box.append(details);
  box.append(element('p', 'A single paper test does not demonstrate a profitable strategy.', 'result-note'));
  if (scroll) $('#results').scrollIntoView({behavior: 'smooth', block: 'center'});
}
$('#close-results').onclick = () => $('#results').hidden = true;

function renderReports() {
  const body = $('#recent-reports'); body.replaceChildren();
  if (!state.reports.length) {
    const row = element('tr'), cell = element('td', 'Your completed paper tests will appear here.', 'empty'); cell.colSpan = 5; row.append(cell); body.append(row);
  }
  for (const item of state.reports) {
    const report = item.report, row = element('tr');
    const name = element('td'); name.append(element('strong', recordingName(report)), element('small', item.path.split('/').at(-1)));
    row.append(name, element('td', fmt(report.neurons)), element('td', `${Number(report.reservoir.return_pct).toFixed(2)}%`, report.reservoir.return_pct < 0 ? 'negative' : 'positive'), element('td', fmt(report.reservoir.trades)));
    const cell = element('td'), button = element('button', 'Open result ↗', 'text-button');
    button.onclick = () => showReport(item, true); cell.append(button); row.append(cell); body.append(row);
  }
}

const actionNames = {trial: 'Pump.fun paper trial', prepare: 'Small circuit preparation', 'prepare-full': 'Full neuronal graph build', capture: 'Pump.fun price recording', replay: 'Brain model replay', 'learning-check': 'Learning check', demo: 'Demo data creation'};
const runningActionNames = {trial: 'Currently running a Pump.fun paper trial', prepare: 'Preparing small circuit', 'prepare-full': 'Building the full neuronal graph', capture: 'Recording Pump.fun prices', replay: 'Running brain model', 'learning-check': 'Checking learning on unseen coins', demo: 'Creating demo data'};
function renderJobs() {
  const running = state.jobs.find(job => ['running', 'stopping'].includes(job.status));
  document.querySelectorAll('button[type=submit],#demo-button,#build-full-button').forEach(button => button.disabled = !!running || submitting);
  $('#build-full-button').disabled = !!running || submitting || !state.bulk_ready;
  $('#operation-strip').hidden = !running;
  if (running) {
    $('#operation-title').textContent = running.status === 'stopping' ? 'Stopping safely…' : runningActionNames[running.action];
    if (!['replay', 'trial'].includes(running.action)) {
      $('#operation-progress').removeAttribute('value');
      $('#operation-detail').textContent = running.action === 'learning-check' ? running.log.at(-1) || 'Preparing isolated training copies…' : running.action === 'capture' ? 'The brain is idle while prices are being recorded.' : running.action.startsWith('prepare') ? running.log.at(-1) || 'Reading the downloaded connectome…' : 'Your new recording will be selected when it is ready.';
    }
    $('#stop-button').disabled = running.status === 'stopping';
  }
  updateReadiness();
  window.trialDashboard?.readiness();
  window.learningReview?.refresh(state);
  if (latestJob) {
    const job = state.jobs.find(job => job.id === latestJob);
    if (job && !['running', 'stopping'].includes(job.status)) {
      latestJob = null;
      if (job.action === 'replay') brain.finish();
      if (job.action === 'trial') window.trialDashboard?.ended(job);
      if (job.status === 'failed') message('#error', job.log.findLast(line => line.includes('error:'))?.split('error:').at(-1).trim() || 'This operation could not finish.');
      if (job.status === 'cancelled') message('#success', job.action === 'learning-check' ? 'Learning check stopped. Saved live learning is unchanged; previous completed checks remain available.' : ['replay', 'trial'].includes(job.action) ? 'Stopped. Saved prices, trades, and neuron activity remain available.' : 'Stopped. Any completed prices remain in your recordings.');
      if (job.status === 'completed') {
        if (job.action === 'replay') {
          const report = state.reports.find(item => item.path === job.output);
          if (report) showReport(report);
          message('#success', 'Paper test complete. Scrub the neuron timeline or review the results below.');
        } else if (job.action === 'learning-check') {
          message('#success', 'Learning check complete. Review the comparison under “Is the training helping?”');
        } else if (job.action === 'trial') {
          message('#success', 'Paper trial ended. Review its status, trade history, and recorded neuron activity.');
        } else if (job.action.startsWith('prepare')) {
          setGraph(job.output); navigate('experiment'); message('#success', 'Your new brain model is selected. Choose a recording and run a test.');
        } else if (job.action === 'capture') {
          activeBars = job.output; $('#replay-bars').value = activeBars;
          localStorage.setItem('flytrader.bars', activeBars); navigate('experiment'); updateReadiness();
          message('#success', 'Recording saved and selected. Press Run paper test when you’re ready.');
        } else message('#success', 'Synthetic demo data is ready in your recording and brain lists.');
      }
    }
  }
}

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    state = await request('/api/state');
    if ($('#connection')) $('#connection').textContent = state.public_mode ? '● Public lab' : '● Connected locally';
    if ($('#error').textContent.startsWith('The local server is unavailable.')) message('#error', '');
    const graphs = state.files.filter(file => file.kind === 'edges');
    const bars = state.files.filter(file => file.kind === 'bars');
    if (!graphs.some(file => file.path === activeGraph)) activeGraph = state.default_graph;
    activeGraph = fillOptions($('#replay-edges'), graphs, activeGraph);
    fillOptions($('#active-graph'), graphs, activeGraph);
    if (!bars.some(file => file.path === activeBars)) activeBars = bars.find(file => file.ready && !file.synthetic)?.path || bars[0]?.path || '';
    activeBars = fillOptions($('#replay-bars'), bars, activeBars);
    graphDetails(); renderReports(); renderJobs();
    window.trialDashboard?.refresh(state);
    if (!latestJob) {
      const running = state.jobs.find(job => ['replay', 'trial'].includes(job.action) && job.status === 'running');
      if (running) {
        latestJob = running.id; selectedJob = running.id;
        if (running.action === 'replay') brain.attach(running.output, true);
      }
    }
    applyPublicMode();
  } catch (error) {
    if ($('#connection')) $('#connection').textContent = 'Disconnected';
    message('#error', 'The local server is unavailable. ' + error.message);
    $('#run-button').disabled = true;
    window.trialDashboard?.disconnected();
  } finally { refreshing = false; }
}

async function start(body) {
  if (submitting) return;
  message('#error', ''); message('#success', '');
  submitting = true; renderJobs();
  try {
    const job = await request('/api/jobs', body);
    latestJob = job.id; selectedJob = job.id;
    if (body.action === 'replay') {
      $('#results').hidden = true; $('#operation-progress').value = 0;
      await brain.attach(job.output, true);
    }
    if (body.action === 'trial') window.trialDashboard?.begin(job);
    await refresh();
  } catch (error) { message('#error', error.message); }
  finally { submitting = false; if (state) renderJobs(); }
}

for (const action of ['prepare', 'capture', 'replay']) {
  $('#' + action + '-form').onsubmit = event => {
    event.preventDefault();
    start({action, ...Object.fromEntries(new FormData(event.target)), watch: $('#watch-run').checked});
  };
}
$('#demo-button').onclick = () => start({action: 'demo'});
$('#build-full-button').onclick = () => start({action: 'prepare-full'});
$('#stop-button').onclick = async () => {
  try {
    const res = await request('/api/stop', {});
    await refresh();
    if (res?.override_cooldown_seconds) {
      message('#success', "Stopped test. As you stopped another user's test, you have a 5-minute cooldown before you can start or stop another test.");
    }
  }
  catch (error) { message('#error', error.message); }
};
const initialView = location.hash.replace(/^#/, '');
navigate(titles[initialView] ? initialView : 'trial');
refresh();
setInterval(refresh, 1500);
