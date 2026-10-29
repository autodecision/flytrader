# Flytrader

An experimental **paper-trading research prototype** using fruit-fly connectivity in a recurrent model. Active training now includes sparse Kenyon-cell inputs and dopamine-inspired plasticity on identified KC→MBON connections. No wallet, signing, or order submission is implemented.

Google's September 2026 release is a structural connectome, not executable brain software or a pretrained trading model. This project's neural dynamics, market input mapping, and trained output are engineering assumptions. Its current unsigned, normalized graph does not reproduce biological neural activity. No trading advantage has been established.

The dashboard defaults to **Active exploration** for continuous paper training: six
fresh quotes of warmup, small exploratory buys, 30/60/180-second planned holds,
and learning from completed trades after costs and reserve-based price impact.
The **Conservative** option preserves the original forecast-based method.
See [reward training](docs/reward-training.md) for circuit roles, execution assumptions,
saved telemetry, and frozen evaluation on later, unseen coins.

After stopping a session, use **Check learning** under **Is the training helping?**
to compare fresh training copies against fixed-synapse, untrained, random,
momentum and cash baselines. It reports net results after assumed costs across
three seeds, while preserving the saved live brain and all recordings.

## Run the working offline demo

```bash
cd /home/allister/Tools/flytrader
python3 -m venv .venv
.venv/bin/pip install -e '.[connectome,market]'
.venv/bin/python -m flytrader.cli demo
.venv/bin/python -m flytrader.cli replay \
  --edges data/demo/synthetic-edges.csv \
  --bars data/demo/synthetic-bars.csv
.venv/bin/python -m unittest discover -s tests -v
```

The demo graph and prices are **synthetic**, and the output is only a pipeline smoke test. Results appear in `reports/replay.json`.

## Import actual male-fly connectivity

Create a neuPrint account and obtain an API token using the [official instructions](https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html). Set it in your local shell, without putting it in chat or committing it:

```bash
read -rs -p 'neuPrint token: ' NEUPRINT_TOKEN
export NEUPRINT_TOKEN
.venv/bin/python -m flytrader.cli fetch-connectome \
  --type-regex 'YOUR_SELECTED_CELL_TYPE_REGEX' \
  --out data/malecns-edges.csv
unset NEUPRINT_TOKEN
```

Choose cell types using the [MaleCNS explorer](https://male-cns.janelia.org/explore/). The importer retrieves the induced graph between matching neurons from `male-cns:v1.0`, aggregates primary-ROI weights, and writes attribution/selection metadata alongside the CSV. It refuses an empty graph or a selection larger than 2,000 neurons by default. An arbitrary small selection is not a validated trading circuit. This authenticated import has not yet been exercised against the service.

Alternatively, Janelia provides the complete connectivity table as a roughly 1.1 GB Feather download on its [download page](https://male-cns.janelia.org/download/). The public connectivity table and neuron annotations are installed in `data/`. Run `.venv/bin/python -m flytrader.cli prepare-connectome` to extract the default induced subgraph: the 439 annotated cells with type prefixes MBON, APL, PAM, or PPL. Only internal edges are retained; connected node counts may be lower. This is an experimental selection, not a complete biological learning circuit. Morphology and microscope volumes are unnecessary for this graph experiment. MaleCNS data is CC-BY; preserve the dataset attribution when sharing derived data.

## Full retained MaleCNS model

The GUI now defaults to `data/malecns-full.npz`: **166,700 annotated neurons,
25,582,938 directed connections, and 124,177,617 synaptic contacts**. The 439-cell
circuit remains available for comparisons. More cells make broader topology
experiments possible; they do not establish better trading performance.

The full model includes every assigned neuronal superclass, including uncertain
classes, and excludes explicit glia and unresolved segmentation objects. It
retains every released internal edge, including single-contact edges and self
connections. The 217 retained cells without internal edges remain explicit model
nodes. This is the full graph under that inclusion policy, not every biological
cell or synapse and not a calibrated living brain. It includes the brain and
ventral nerve cord. This policy follows the useful distinction in
[Stonkfly's model documentation](https://github.com/nftechie/stonkfly/blob/main/docs/model.md).

To rebuild or run it from the CLI:

```bash
.venv/bin/python -m flytrader.cli prepare-full
.venv/bin/python -m flytrader.cli replay \
  --edges data/malecns-full.npz --bars data/observations.csv \
  --out reports/full-test.json --activity-out reports/full-test.activity.json
```

The compiler scans Arrow batches, maps neuron identities to sparse matrix
indices, and writes an atomic NPZ file with raw contact counts. The matrix uses
rows for postsynaptic and columns for presynaptic cells. Source and compiled-file
SHA-256 fingerprints, inclusion policy, counts and storage size are saved in the
metadata. The installed graph is about 202 MiB on disk.

**All retained neurons update at every observation.** The unsigned, normalized
tanh dynamics and artificial market inputs remain the same model family. For
graphs larger than 2,000 cells, the output combines 128 individual neuron states
with 64 deterministic pooled means covering every model neuron. Pool membership
is the stable neuron index modulo 64; these are numerical summaries, not anatomical
regions. This keeps regression small while including the activity of every cell.
The 439-cell comparison retains its original readout. Reports record both sizes.

Local validation of the full graph on the existing 120-price Pump.fun recording
took 7.39 seconds with activity recording and no visual pacing, peaking at about
724 MiB of process memory. This is a runtime benchmark, not a trading result.
The GUI uses one BLAS thread per job to keep the desktop responsive.

## Collect free price observations

```bash
.venv/bin/python -m flytrader.cli capture --out data/observations.csv
```

This discovers the first new token using PumpPortal's free creation stream, then records 120 reserve-price snapshots about 10 seconds apart from public Solana RPC (about 20 minutes). Use `--mint TOKEN_MINT` to select a token. Selection is for collection, not an investment recommendation. No API key or wallet is needed. The output file must not already exist.

The collector checks the derived account owner, discriminator, native SOL quote and advancing RPC slot. It stops on migration or errors. Newly created accounts may be temporarily unavailable, and public RPC may throttle requests; this prototype has no automatic reconnect. Use `--rpc` to configure a provider later.

Prices are virtual-reserve ratios, not executable quotes. Volume is unavailable and set to zero to disable that input; metadata records this limitation. After capture, use `data/observations.csv` as `--bars` for replay with `data/malecns-edges.csv`. Three real observations were successfully collected in `data/live-smoke.csv`; this is a connectivity check, not a strategy evaluation. The GUI can also run continuous collection and paper learning until you stop it.

## Replay market observations

Supply a CSV with one token's chronological bars:

```csv
timestamp,mint,close,volume_sol
1700000000,TOKEN_MINT,0.00000010,2.4
1700000060,TOKEN_MINT,0.00000011,3.1
```

Timestamps must be strictly increasing Unix seconds; close prices are SOL per token; volume is SOL per bar. Use a consistent bar interval and at least 100 rows. Keep tokens in separate files. The included collector can also supply reserve-price snapshots. It does not normalize trade events.

```bash
.venv/bin/python -m flytrader.cli replay \
  --edges data/malecns-edges.csv --bars data/token-bars.csv \
  --cost-bps 150 --out reports/token.json
```

The model uses causal returns, momentum, and relative volume. A ridge-regression output is fitted on the first 70% of bars, with labels crossing the split excluded. The remaining bars are held out. A signal at close t can enter only at close t+1 and exits at close t+2. Simulated positions allocate 5% of remaining capital, do not overlap, and stop opening after a 10% closed-trade drawdown. These are experiment defaults, not recommended trading parameters. Losses can exceed the stop threshold on the final trade.

Reports compare the graph model with a linear model on identical inputs and with cash. The default 150 basis points per side is an assumed combined fee/slippage stress value, **not a claim about current Pump.fun fees**. This coarse fill model omits bonding-curve impact, liquidity constraints, priority fees, failed transactions, MEV, and intratrade drawdown. Single-token/single-split results cannot establish generalization or profitability.

## Remaining strategy and execution research

1. Compare the imported real graph with degree-preserving shuffled graphs to test whether fly topology helps.
2. Collect a universe of tokens including failures and migrations, then build timestamped bars. Use a selected Solana RPC/indexer or a documented data provider.
3. Implement reserve-based paper fills, current protocol fees, latency, failed exits, and stale-data handling.
4. Evaluate across held-out dates and tokens; continuous online learning and finite trials do not establish generalization or profitability.

[Pump.fun's official program documentation](https://github.com/pump-fun/pump-public-docs) is the protocol reference. [PumpPortal](https://pumpportal.fun/data-api/real-time/) is a separate third-party service: its documented token trade subscriptions currently require a funded API key and charge 0.01 SOL per 10,000 events. The collector uses only the free creation stream and public RPC reads, with no paid trade subscriptions.

Sources checked September 10, 2026:

- [Google Research announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
- [MaleCNS data, licensing and access](https://male-cns.janelia.org/download/)
- [neuPrint query documentation](https://connectome-neuprint.github.io/neuprint-python/docs/queries.html)

## Local GUI

Run `./start-gui.sh` from this folder, then open [the local workspace](http://127.0.0.1:8765).
The opening page is **Pump.fun trials**. The default **Continuous · live prices + learning**
mode has one **Start trading & learning** button. It runs until **Stop trial** is pressed,
following activity-screened coins and showing the paper account, decisions, and actual neuron states.
**View trial progress** returns to the coin chart and account. After Stop, it becomes
**View session results**. All orders remain simulated.

The **Coin in focus** chart follows the current coin as prices arrive, including its
full observed history and paper buy/sell markers. Use its range selector to zoom into
recent minutes. After stopping, choose a recorded coin and click a price to inspect
the nearest saved neuron frame; the dashed cursor follows neuron playback. Arrow
keys inspect chart prices and Enter selects a saved frame. **Trial settings → Live
observations per coin** offers longer tracking: 20 minutes, 50 minutes, 2 hours, or
about 2 hours 47 minutes at the default 10-second interval. The session continues
across coins until stopped; this setting changes when it rotates to another coin.

The coin heading opens its Pump.fun page in another tab. The neuron map defaults
to **Auto contrast** so small computed states remain visible; its current color
range is labeled and the inspector always shows the original values. The latest
observation, update age, and count of changed displayed values appear above the
map. Rings highlight up to 40 of the largest changes with known positions and
repeat a visible pulse between live updates. **Animate changes** pauses that motion;
it represents the latest measured change, not additional observations. Identical
states produce no change rings, and motion stops if no fresh frame arrives for 30 seconds.
These comparisons use recorded model values, not biological spikes. Auto contrast
uses the displayed states' 99th-percentile magnitude with a 0.001 floor; fixed
color ranges remain available. Reduced-motion settings disable the animation.

### Active coin selection

Live trials shortlist recently traded coins from [Pump.fun Explore](https://pump.fun/explore)
using its public frontend coin feed. Candidates must be at least two minutes old,
have a listed trade in the last 90 seconds, and pass configurable market-size checks.
The chart-context profile defaults to a minimum 100 SOL price-times-supply value and
10 SOL of real quote reserves, and ranks eligible candidates by market cap. The runner
supports native-SOL Pump bonding curves and canonical SOL PumpSwap pools. Pool ownership,
canonical mint/creator/quote fields, vault authorities and advancing confirmed slots are
verified; PumpSwap prices include the pool's signed virtual quote reserves while the
liquidity limit uses only actual quote-vault balances. Other exchanges and quote assets
remain unsupported. These are research heuristics, not investment recommendations.
The runner independently verifies the market on Solana, then requires
two price movements across three fresh quotes before starting the model. The three
real screening quotes are retained as the beginning of the recording. An explicitly
pinned coin can be quiet: the runner observes it while pausing paper entries.
No funded API key, wallet, or paid subscription is used.

At the default 10-second interval, live trials move on after 90 seconds without a
meaningful price change, or fewer than three changes in a rolling two-minute window.
Slower sampling extends those windows to at least three and four price intervals,
respectively, so an active coin is not rejected simply because fewer quotes were requested.
Changes smaller than one part per million are ignored; accumulated movement still counts.
The dashboard shows observed activity and the reason for rotation. The ongoing
session's paper cash, risk limits, and learned readout carry forward to the next coin.
Pending buys are cancelled before rotation; existing positions finish on fresh quotes
under the normal exit rule. A missing exit quote remains unresolved, never a fake sale.
Saved comparison trials retain their original inputs without activity filtering.

Public frontend feeds can change or fail. If no candidate passes, the session displays
its search/retry state and waits; it does not silently select an unchecked new coin.
Quiet quotes do not prove that every on-chain trade has stopped, and a qualifying coin
can become inactive later. The activity checks therefore remain enabled throughout training.

### Chart context, risk checks, and selecting a coin

**Coin selection** accepts automatic selection, a chosen first coin, or **Stay on my
coin until Stop**. Paste a `https://pump.fun/coin/<mint>` link or the mint address.
Pinned mode retains the same model state and learning across saved recording segments,
keeps watching quiet periods, and retries the same coin during an outage. It never
silently changes to another coin. Unsupported quotes pause the pinned session's feed.

Conservative trials use the separately versioned `chart-context-v1` profile. The actual
reservoir receives short returns, combined time-based momentum, and longer trend/drawdown
context. Its readout also receives 15 context features including 1/5/15-minute returns,
observed full-history return and high-water drawdown, volatility, market size, reserves,
reserve loss, concentration when available, and missing-data indicators. It learns
output weights from subsequently observed outcomes; rules are not presented as a learned
probability of a rug pull. The original Replay lab and `trial --profile legacy` remain
available with their original checkpoints and input mapping.

Before a new coin, the context loader reads previously captured, verified local prices
for that mint strictly before the first live observation. It searches the 500 most recent
recording metadata files and retains up to 20,000 context points with endpoints and
extrema preserved. Historical inputs prime context and the last 21 reservoir updates;
they do not place orders or train live memory again. The chart includes this retained
history. Coverage and missing windows are explicit: this is **collected price history,
not the coin's full lifetime chart**. No working public lifetime-candle feed is configured.

Every new quote evaluates observable risk rules. An 8% single-quote drop or 12% one-minute
drop flags a dump; 15% one-minute or 35% five-minute rises pause entries during a rapid
pump. A 25% run-up followed by an 8% retreat flags a reversal. A 35% drawdown from the
observed high or 25% reserve loss also pauses entries. Dumps/reversals add a two-minute
cooldown (at least three configured intervals). These thresholds are transparent
heuristics, not validated trading parameters or predictions of an imminent dump.

Live entries also require the configured cap/reserve minimums and verified mint/freeze
controls. Active authorities and unreviewed token extensions pause entries. The largest
20 token accounts are queried at startup and at most once per five minutes, excluding
the curve and canonical pool vaults. An observed non-vault account holding 20% of minted
supply blocks entry. This is account concentration, not a complete owner/bundle analysis.
RPC refusal is shown as **unknown**, and incomplete checks reduce allocation to 1% of cash.
Otherwise allocation is at most 5%; all live orders are capped at 1% of real SOL reserves.
New warnings cancel queued buys before filling. Existing exits use the actual fresh price,
including a dump price. Market cap cannot override other warnings or guarantee safety.

The brain and risk guard run once per fresh confirmed quote: 10 seconds by default,
configurable from 10 to 3,600 seconds. The panel shows the actual last interval and update
age. Display pulses are not extra observations or decisions. Long gaps do not train a
multi-minute return as a normal one-interval label. Quote sampling can miss fast moves;
fees, slippage, failed execution, and sellability remain paper-fill limitations.

### Conservative continuous paper trading and learning

This section describes the retained **Conservative** option. The new default
**Active exploration** has the different warmup, opponent synapse learning and
reserve-based fills documented in [reward training](docs/reward-training.md).

- Fresh confirmed native-SOL reserve prices arrive about every 10 seconds. The first
  session needs 60 observations (about 10 minutes) to build its initial fit. Later coins
  warm up their neuron states for 21 observations (about 3.5 minutes). The trading
  readout retains its learned parameters across coins and sessions.
- An active coin runs for up to 120 observations by default, then the runner selects
  another active Pump.fun coin. Quiet coins end sooner. This is a rotation interval,
  **not a session time limit**. You can
  change the interval and observations per coin in Trial settings. An optional mint
  chooses the first coin; pinned mode instead continues on that coin across segments.
- Learning uses a regularized, exponentially weighted least-squares readout with a
  forgetting factor of 0.995. All 166,700 retained neuron states still compute each
  observation. The connectivity, reservoir dynamics, feature mapping, and risk rules
  remain fixed; the model learns the output weights that produce trading signals.
- A signal at observation t forecasts the log return from t+1 to t+2. Its label becomes
  available only at t+2. That outcome can then update the model before predicting for
  the current observation. The last two unobserved labels are discarded at a coin
  boundary or Stop. There are no cross-coin labels and no future prices in a fit.
- The dashboard shows total learned outcomes, updates this session, restored experience,
  the last learned outcome, and mean absolute prediction error against a zero-return
  baseline. Errors use the original forecast made before its outcome arrived. Initial
  fitting examples are not counted as evaluated forecasts. Learning includes holds as
  well as paper-trade outcomes. Lower error alone is not evidence of profitable trades.
- Checkpoints under `learning/pump-context-<model-id>.npz` (new profile) or
  `learning/pump-<model-id>.npz` (original profile) are atomically saved after every update
  and on Stop. The identity includes the graph, model/learner code, sampling interval,
  and seed. Compatible sessions restore automatically; a different model or interval
  has separate memory. A file lock prevents simultaneous writers. A timestamp high-water
  mark rejects duplicate or older outcomes. Learning statistics remain bounded in memory.

The dashboard shows equity and cash in SOL, net and realized profit/loss, open and queued
positions, closed trades, and a buy/sell ledger. One session shares its starting paper
capital across all coins (10 SOL by default). A new session starts a new paper account
while retaining compatible learning memory. New model inputs have separate memory;
old checkpoints are preserved. Paper buys allocate at most 5% of available cash,
enter on the next observed price, and exit on the following observation. Costs default
to 150 bps per side. Signals near a coin rotation cannot open a position without enough
observations to close it. At 10% closed-trade drawdown, entries pause but observations
and learning continue. These fill assumptions do not model executable liquidity.

The live trial's shortlist and three-quote checks are followed by continuous activity
monitoring. Transient reads retry with bounded backoff and cancel queued buys.
No price, neuron observation, learning example, or fill is invented during an outage.
Supported canonical PumpSwap and verified SOL-paired Raydium CLMM liquidity continue on
the same mint. Token-2022 transfer fees are included in paper fills. If a coin becomes
temporarily unavailable, automatic mode moves on after a short delay and pinned mode
retries it; a permanently unsupported venue ends a pinned trial with an explicit reason.
An open position without an exit quote is retained as **unresolved**, with its cost
locked out of available cash and its last valuation explicitly marked stale. It is never
counted as a sale or realized profit. At 10% of initial cash locked in unresolved tokens,
new entries pause while learning continues. Discovery outages keep retrying until Stop.

**Stop trial** preserves learning, prices, trades, and partial activity. Browser refresh
or closure does not stop the worker; reopening reconnects to the active session. The
computer and local server must remain running. Stopping the server ends its worker;
starting another continuous session restores compatible learning memory. Fatal local
errors (for example, inability to save a checkpoint) end the session with an error.

### Finite comparison modes

**Quick trial** chooses the newest ready recording per distinct Pump.fun coin, up to
its selected count. This is accelerated historical playback with original dates. It
fits only the first 70% and keeps that output fixed on the remaining observations.
**Limited trial · live prices** collects fresh coins, trains on the first 60 observations,
then keeps the output fixed and ends after the selected count of coins. Neither finite
mode reads or updates the continuous learner's memory. Only verified Pump.fun recordings
are used in automatic trials; the Replay lab also supports synthetic and imported recordings.

### Saved sessions

After **Stop trial**, the dashboard opens **Whole session results**. This shows the
entire session's paper profit/loss, starting and final balance, buy/sell order counts,
new learning updates, checkpoint verification, and prediction error compared with
predicting no price change. These totals remain fixed while you inspect older neuron
frames. Updating parameters is reported separately from improving prediction quality.

If there were no trades, results show the strongest forecast, actual entry threshold,
training/hold counts, and how often consecutive observed prices were unchanged. The
current rule forecasts one price interval and holds from the next observation to the
following one (about 10 seconds at default live settings). Tracking a coin for longer
does not by itself lengthen that holding period. The model can learn price outcomes
without buying; extra simulated orders alone provide no extra market information.

New sessions persist exact decision counters even when neuron recordings are sampled.
Older results are reconstructed read-only from their activity files; incomplete or
sampled history is labeled. Resumed learners are evaluated on this session's prediction
errors when starting counters are available, otherwise the comparison explicitly uses
all saved learning. Opening results never restarts a stopped session.

Each session is `reports/trial-<id>.json`. Per-coin activity files contain matching
paper and learning snapshots; saved playback keeps these aligned with neuron colors.
Live prices are in `data/trial-<id>-coin-<n>.csv`. The dashboard keeps the most recent
50 coin summaries and 500 orders so an ongoing session stays responsive. The complete
append-only order ledger is `reports/trial-<id>.orders.jsonl`; individual coin summaries
and activity remain on disk, with at most 401 sampled frames per coin. Graph, code, and
price-file fingerprints are recorded. Saved data grows with session duration.

CLI equivalents (use a new output name):

```bash
.venv/bin/python -m flytrader.cli trial --source live --continuous \
  --count 120 --interval 10 --out reports/my-learning-session.json
.venv/bin/python -m flytrader.cli trial --source recordings \
  --bars data/observations.csv --coins 1 --out reports/my-trial.json
.venv/bin/python -m flytrader.cli trial --source live \
  --coins 3 --count 120 --interval 10 --out reports/my-limited-trial.json
```

The default graph is `data/malecns-full.npz`. Online prediction and delayed evaluation
follow the streaming formulation discussed in [MIT's system identification notes](https://underactuated.mit.edu/sysid.html).
Live data references checked September 17, 2026: [PumpPortal's public streams](https://pumpportal.fun/data-api/real-time/),
[its shared Bonk creation stream](https://pumpportal.fun/data-api/bonk-fun-data-api/),
and [Pump's protocol documentation](https://github.com/pump-fun/pump-public-docs).

In the separate **Replay lab**, choose a brain model and a price recording, then press
**Run paper test**. Recordings with fewer than 100 prices show how much data
is missing and cannot start a test. Use the **Pump.fun** data screen to collect prices;
a completed recording is selected automatically. Test settings and
operation logs are tucked into expandable sections.
Each operation creates a new output file. Existing reports remain available after
restarting the GUI; operation logs last for the current server session.

### Watch and inspect neuron activity

The rotatable map uses actual MaleCNS cell-body coordinates. The full model has
139,662 neurons with positions. To keep the browser and recordings bounded, it
colors a fixed, uniformly sampled set of 2,000 model neurons; the current sample
contains 1,702 positioned cells and 298 cells available through the inspector
without invented coordinates. The heading shows the full model count separately
from the displayed count. Searching and individual values apply to this displayed
sample. The 439-cell circuit still displays every cell, with 438 real positions.

The gray background is sampled anatomical context, not an individual activity
reading; its cells may belong to the running model. **Neural Map** shows the
displayed cells; **Fly Mode** places a stylised opaque fly in side view —
head enclosing the brain cloud, thorax and striped abdomen following the
ventral nerve cord, swept wings, red eyes — sitting on a desk with a
keyboard in front of it and a glowing monitor on each side, screens toed in
to face the fly. The front legs type intermittently when the
model processes observations; the typing is decorative and does not drive the
model. Each cutaway window shows that part's own neurons: brain-cloud cells
in the head, cord cells in the thorax and abdomen.
Synthetic graphs use a labeled schematic layout.

During a paper test, colors and the inspector display the model's actual signed
reservoir states (blue negative, orange positive, gray near zero), on a −1 to +1
scale. The color-range control changes saturation only. All model neurons are
computed at every observation; the “above |0.05|” count covers the **entire model**,
not just the displayed sample, and is not a count of biological spikes. Connections
optionally show up to 1,200 of the strongest edges between displayed cells as
straight lines, not reconstructed neural branches. Display sampling never prunes
the computational graph.
The positions use the dataset's 8 nm coordinate space, converted to micrometres;
see [MaleCNS coordinate and data documentation](https://male-cns.janelia.org/download/).

After the run, use **Play activity**, the speed menu, or the timeline to inspect
saved states. Search by cell type or ID, click a positioned cell, or select one
of the largest displayed activations. Each displayed frame includes its observation number,
timestamp, input price and units, and whether the input belongs to training or the holdout.
A price chart follows the selected observation, so changes in the recording can
be compared with the displayed neuron states.
**Past results** can reopen the activity attached to a new report. Older reports
retain their results but need a new run to record neuron activity.

GUI replays save `reports/<name>.activity.json` with the full graph fingerprint,
displayed cell identities and their model indices, and up to 401 sampled frames
(values rounded to six decimals), plus a small live sidecar. Each frame includes
an activation count and maximum absolute activation measured across the full model.
Runs of 400 observations or fewer retain every frame. Stopping preserves the last
computed frame in a partial recording. The optional slow-run setting adds at most
20 seconds of pacing; turning it off retains the same model and saved activity.
CLI users can add `--activity-out reports/test.activity.json --watch-delay 0.06`
to `replay` to produce the same recording.

The GUI binds only to localhost. The selected computational graph and the
displayed activity sample are explicitly distinct.
Stop the GUI with Ctrl+C; this also interrupts its active operation.

Replay reports include input/source fingerprints and environment details.
See `docs/stonkfly-review.md` for the reviewed upstream architecture and what was
actually incorporated.

The current desktop session runs the GUI as a transient user service named
`flytrader-gui`. Stop it with `systemctl --user stop flytrader-gui`. It is not
configured to start at login. After stopping it, `./start-gui.sh` starts a normal
foreground instance. Keep only one instance on port 8765.
