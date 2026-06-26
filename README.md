# LLM Framework for Predicting Significant Equity Index Fluctuations

Reference implementation for the paper *"A Large Language Model Framework for
Predicting Significant Equity Index Fluctuations."* The system reformulates
next-day equity-index movement prediction as a threshold-based **rare-event
classification** problem and solves it with a **confidence-weighted ensemble of
six commercial large language models**, benchmarked against LSTM and GRU
baselines under both fixed train/test splits and rolling-window validation. It
targets three Indian indices — **NIFTY50**, **BANKNIFTY** (labelled `NIFTYBANK`
in the data files), and **NIFTYIT** — over the period **17 September 2007 to
21 July 2025**.

The central idea is to replace the frequency-driven classifier at the heart of a
conventional forecasting pipeline with **context-aware semantic inference**: each
day's market state is rendered as a structured natural-language prompt, several
LLMs reason over it independently, and their votes are combined by a
confidence-weighted aggregation rule. Because large single-session moves are rare
and heavily outnumbered by ordinary trading days, the evaluation deliberately
pairs minority-sensitive classification metrics with economic backtests, so that
a model cannot look good merely by predicting "no significant move" every day.

---

## Table of contents

- [Conceptual overview](#conceptual-overview)
- [Repository layout](#repository-layout)
- [The prediction tasks](#the-prediction-tasks)
- [Pipeline stages](#pipeline-stages)
- [The LLM ensemble and the aggregation rule](#the-llm-ensemble-and-the-aggregation-rule)
- [Causality and leakage control](#causality-and-leakage-control)
- [The backtesting engine](#the-backtesting-engine)
- [Recurrent baselines](#recurrent-baselines)
- [Metrics](#metrics)
- [Installation](#installation)
- [Configuration and API keys](#configuration-and-api-keys)
- [Usage](#usage)
- [Output files](#output-files)
- [Caching and reproducibility](#caching-and-reproducibility)
- [Cost and throughput](#cost-and-throughput)
- [Data format](#data-format)
- [Extending the framework](#extending-the-framework)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

---

## Conceptual overview

A standard supervised classifier trained on financial returns is dominated by the
majority class: most days are unremarkable, so cross-entropy and accuracy reward a
model that simply learns to say "nothing happens." This is fatal for tail-event
detection, where the rare positive class — a large single-day move — is exactly
what matters economically.

This framework approaches the problem differently. Rather than fitting parameters
to the empirical class frequencies, it presents each day's market configuration
to pre-trained LLMs as a semantic description and asks them to reason about
whether the configuration looks like one that precedes a significant move. The
models are not retrained or fine-tuned; they act as context-aware classifiers
over structured market summaries. The contribution is therefore a **framework and
evaluation protocol**, not a new learning algorithm: the empirical payoff is a
clear, reproducible characterisation of where this framing helps (directional
forecasting, low thresholds) and where it does not (the rarest 3% tail events).

## Repository layout

```
llm-stock-analyzer/
├── llm_stock_analyzer.py          # Multi-provider LLM stack
├── ensemble_predictor.py          # Confidence-weighted ensemble (Eq. 5) + cache
├── advanced_backtest_analyzer.py  # Backtesting engine (fixed + rolling splits)
├── verify_dataset_stats.py        # API-free Table 1 integrity check
├── baselines/
│   └── recurrent_baselines.py     # LSTM / GRU baselines
├── backtest-comparison-dashboard.html  # Static results dashboard
├── data/                          # Daily OHLCV + movement labels (21 CSVs)
├── llm_cache/                     # Per-day cached ensemble responses (generated)
├── results/                       # Per-symbol LLM prediction dumps (generated)
├── advanced_backtest_results/     # Backtest workbooks + text reports (generated)
├── requirements.txt               # Core dependencies (LLM pipeline)
├── requirements-baselines.txt     # TensorFlow, for the recurrent baselines only
└── .env.example                   # Template for provider API keys
```

| File | Responsibility |
|------|----------------|
| `llm_stock_analyzer.py` | The LLM stack. Computes ~50 technical indicators (`TechnicalAnalyzer`), derives probabilistic movement statistics (`MovementAnalyzer`), constructs the templated prompt `g(X_t, θ, Task)` (`create_movement_prediction_prompt`), holds the six-model registry with per-model provider/temperature/token settings, dispatches queries asynchronously through a `UnifiedLLMInterface`, and parses each model's JSON reply with a neutral fallback on malformed output. |
| `ensemble_predictor.py` | The aggregation layer. Wraps the LLM stack in a synchronous, causal, cached `predict_day(...)` call that the backtest consumes one day at a time, and combines the member votes with the confidence-weighted rule of **Eq. (5)**. |
| `advanced_backtest_analyzer.py` | The evaluation engine. Implements fixed-ratio and rolling-window backtests, computes the full classification and trading metric suite, and writes a multi-sheet Excel workbook plus a text summary. The next-day prediction at every test step comes from the LLM ensemble. |
| `baselines/recurrent_baselines.py` | LSTM and GRU baselines, trained and evaluated under identical labels and the same fixed split, producing per-index/per-task Accuracy and F1 for the comparison table. |
| `verify_dataset_stats.py` | Recomputes the paper's Table 1 jump-day band counts directly from the CSVs; requires no API access. |

## The prediction tasks

Let `P_t` be the closing price on day `t` and `R_t = ln(P_t / P_{t-1})` the
log-return. Each task defines a binary target `Y_{t+1}`:

| Task | Positive class (`Y_{t+1} = 1`) | Label files |
|------|--------------------------------|-------------|
| `direction` | `R_{t+1} > 0` (up day) | `*_with_Direction.csv` |
| `sig_move_1pct` | `|R_{t+1}| > 1%` | `*_SigMove_1pct.csv` |
| `sig_move_2pct` | `|R_{t+1}| > 2%` | `*_SigMove_2pct.csv` |
| `sig_move_3pct` | `|R_{t+1}| > 3%` | `*_SigMove_3pct.csv` |

Thresholds are fixed in **absolute** return terms rather than scaled per index.
Holding the threshold fixed while the indices differ in volatility is what lets
the study observe how event rarity — and therefore imbalance severity — varies
across assets: the same 3% threshold is rarer for NIFTY50 than for the more
volatile BANKNIFTY, inducing a controlled gradient of imbalance. The data also
ships negative-threshold label files (`*_SigMove_neg{1,2,3}pct.csv`) for the
symmetric bearish-tail task, which the framework supports without modification.

## Pipeline stages

The system is organised as a six-stage modular pipeline:

1. **Data ingestion** — load daily OHLCV for each index, coerce numeric columns
   (volume may contain thousands separators), sort chronologically, compute
   log-returns, and attach the binary target column for the requested task.
2. **Feature engineering** — `TechnicalAnalyzer` produces trend, momentum,
   volatility, volume, and market-structure indicators (simple and exponential
   moving averages, RSI, rate-of-change, MACD variants, historical volatility,
   ATR, Bollinger-band positioning, volume anomalies). `MovementAnalyzer` layers
   on probabilistic descriptors: unconditional and trailing movement rates over
   10- and 30-day windows, and consecutive positive/negative streak lengths.
3. **Prompt construction** — `g(X_t, θ, Task)` renders the feature vector into a
   fixed, fully templated prompt. Only the numeric field values and the task
   description change between days; the structure is constant. The template emits
   labelled sections in this order: `CURRENT TECHNICAL DATA`, `MOMENTUM
   INDICATORS`, `MOVING AVERAGES`, `SUPPORT/RESISTANCE`, `BOLLINGER BANDS`,
   `VOLUME ANALYSIS`, `HISTORICAL MOVEMENT STATISTICS`, then the prediction task
   and a strict-JSON output instruction.
4. **Ensemble inference** — the prompt is sent to all configured models in
   parallel; each returns a binary prediction, a confidence in `[0, 1]`, and a
   short reasoning string.
5. **Aggregation** — member votes are combined by the confidence-weighted rule
   (Eq. 5) into a single ensemble decision.
6. **Backtesting and evaluation** — the per-day ensemble decisions drive a
   trading simulation and a classification scorecard, under both fixed and
   rolling-window protocols.

## The LLM ensemble and the aggregation rule

Six models drawn from three providers, paired by cost/capability tier:

| Provider | Cost-efficient tier | Higher-capability tier |
|----------|---------------------|------------------------|
| OpenAI | `gpt-4o-mini` | `gpt-4o` |
| Anthropic | `claude-3-haiku-20240307` | `claude-3-5-sonnet-20241022` |
| Google | `gemini-1.5-flash` | `gemini-1.5-pro` |

All members are queried at **temperature 0.3** (to reduce output variance) with a
**500-token** cap (sufficient for the structured JSON reply), and are dispatched
asynchronously via `asyncio.gather`, so per-day latency is bounded by the slowest
model rather than the sum.

Each model `i` returns a binary vote `ŷ_i ∈ {0, 1}` and a self-reported
confidence `c_i ∈ [0, 1]`. The ensemble decision is

```
                ⎛  Σ_i  w_i · c_i · ŷ_i        ⎞
  Ŷ_{t+1} = 1 ⎜ ─────────────────────────  ≥  τ ⎟
                ⎝     Σ_i  w_i · c_i            ⎠
```

where:

- **`w_i` — static reliability weights.** Fixed *before* evaluation and never
  estimated from or tuned on the held-out test data; the same weights apply to
  every index, task, and split. The default is **equal weights**, so the
  aggregate reduces to confidence-weighted voting. Passing `--tiered-weights`
  switches to a heterogeneous prior in which the higher-capability member of each
  provider pair carries a modestly larger weight (1.25 vs 1.0). Neither setting
  consults any test-period outcome.
- **`c_i` — per-query confidence.** A within-prompt, per-day quantity reported by
  the model itself; it carries no test-set information. Confidences are clamped to
  `[0, 1]` on ingestion.
- **`τ` — decision threshold.** Defaults to `0.5`; override with `--tau`.

If a model call fails or returns unparseable output, it contributes a **neutral
fallback** (prediction `0`, confidence `0.5`) rather than aborting the day, and
the aggregation simply proceeds over the remaining members. Aggregating across six
independent models from three providers dilutes any single model's idiosyncratic
bias or hallucinated response and keeps behaviour more stable across regimes.

The ensemble's own output confidence is reported as the agreement margin on the
winning class (the normalised weighted score for a positive call, or its
complement for a negative call), giving downstream ROC-AUC a usable
probability-like score.

**A note on the self-reported confidences.** They are best read as heuristic
ranking signals, not calibrated probabilities — the repository does not ship a
reliability-diagram or expected-calibration-error analysis establishing that a
stated 0.8 corresponds to an 80% empirical hit rate. Treat them accordingly.

**Model deprecation.** The specific 2024 model snapshots above are deprecated or
superseded by their providers. The framework is model-agnostic: any
instruction-following model that consumes a structured prompt and returns the
expected JSON can serve as a drop-in replacement. To migrate, edit the
`model_configs` registry in `llm_stock_analyzer.py` to reference successor models
(for example GPT-4.1, Claude 4, Gemini 2.5); no other code changes are required.

## Causality and leakage control

Because the system is evaluated as a forecaster, every feature presented for day
`t` is constructed from information available no later than the close of day `t`:

- **Technical indicators are trailing by construction** — each is a function of a
  look-back window ending at day `t` and references no future observation.
- **Probabilistic descriptors are computed from the causal history ending at day
  `t`** — the trailing movement-rate features use only the most recent 10- and
  30-day windows, and streak counts are read backward from day `t`. In the
  backtest loop the movement statistics are estimated from the concatenation of
  the training block and the test rows already seen up to `t`, never from the
  prediction target at `t+1` or beyond.
- **Rolling windows forecast strictly forward** — each window estimates its
  statistics from a bounded block of recent history and predicts only the
  subsequent test segment, so conditional and streak-based probabilities remain
  causal with respect to the prediction date.

No feature uses full-sample estimates, and no label from day `t+1` or later enters
the day-`t` feature vector. This rules out the look-ahead bias that would
otherwise inflate rare-event performance.

A residual, structural caveat applies to any study built on closed black-box
LLMs: the commercial models may have encountered descriptions of major historical
market episodes during pre-training. The prompt withholds calendar dates and
absolute index levels and presents only scale-normalised indicator values (a
market *state* rather than an identifiable point in history), and the pattern of
results runs opposite to what memorisation-driven leakage would produce (the
extreme-threshold tasks are the *worst*, not the best, performers). The effect
cannot be fully excluded with models of unknown training cutoff; a
leakage-controlled replication on open-weight models with a verified cutoff is the
clean way to settle it.

## The backtesting engine

`AdvancedBacktestEngine` evaluates predictive performance by simulating trading.
On each test day the ensemble decision becomes a position for the next day; the
realised next-day return is booked long for a positive prediction and short for a
negative one. Two validation protocols are supported:

**Fixed train/test splits.** The series is partitioned at a chosen ratio
(`0.70, 0.75, 0.80, 0.85, 0.90, 0.95`). A burn-in `lookback_window` of up to 30
days at the start of the test segment seeds the trailing features before the first
prediction. A minimum test size of 50 observations is enforced.

**Rolling windows.** A fixed-length train/test block slides forward through time.
Three horizon/step configurations are evaluated — **12-month window / 3-month
step**, **18-month / 6-month**, and **24-month / 12-month** — with months
approximated as 21 trading days and an 80/20 train/test split inside each window.
Up to 20 windows are evaluated per configuration. Note that "rolling window" here
denotes a *validation protocol* (repeatedly sliding a fixed-length block forward
and re-evaluating), not a "rolling return"; the financial quantities within each
window are computed exactly as in the fixed-split case.

The Sharpe Ratio is annualised (`mean / std × √252`); maximum drawdown is computed
on the compounded equity curve. Datasets with fewer than 200 rows are skipped.

A `--mode rule` switch replaces the ensemble with a deterministic, hand-coded
technical-rule predictor (`technical_rule_prediction`). This path makes no API
calls and is provided only as a fast smoke test and ablation reference; it is not
the LLM framework and does not reproduce the LLM-ensemble results.

## Recurrent baselines

`baselines/recurrent_baselines.py` provides the LSTM and GRU baselines used for
the head-to-head comparison. They are deliberately standard so the comparison is
fair: a single recurrent layer (32 units) followed by dropout and a dense sigmoid
head, trained on a look-back window of trailing daily features (log-return and its
magnitude, rolling means/standard deviations, multi-horizon momentum, RSI(14),
and a volume ratio). Features are standardised using statistics fit on the
training portion only — no leakage. Class imbalance is left intact (no resampling,
consistent with the framework's stance); only an imbalance-aware class weight is
passed to the binary cross-entropy loss so the rare class is not entirely ignored.
Each run reports test-set Accuracy, Precision, Recall, and F1 per index and task.

These baselines depend on TensorFlow, kept in a separate requirements file so the
core LLM pipeline carries no deep-learning dependency.

## Metrics

**Classification:** Accuracy, Precision, Recall, F1-Score, ROC-AUC.
**Trading:** Total Return, Sharpe Ratio, Win Rate, Maximum Drawdown, Profit
Factor, average win/loss, volatility.

Reporting both families is essential under heavy imbalance: a `sig_move_3pct`
model can post very high Accuracy while its F1, Precision, and risk-adjusted
return collapse, because it is mostly predicting the dominant "no significant
move" class. The minority-sensitive metrics (Precision, Recall, F1) and the
economic backtest are what reveal genuine tail-event competence. Under severe
imbalance the precision–recall view is more informative than ROC-AUC, since
ROC-AUC can stay optimistic when the negative class dominates.

## Installation

Requires Python 3.8+.

```bash
pip install -r requirements.txt
```

For the recurrent baselines only (pulls in TensorFlow):

```bash
pip install -r requirements-baselines.txt
```

## Configuration and API keys

Copy the template and fill in at least one provider key:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

You need at least one key. The ensemble silently drops any model whose provider
key is absent, so a single-provider run (for example Gemini-only) works for
validating the pipeline before committing to the full six-model ensemble. The
model registry, including provider mapping and per-model temperature and token
settings, lives in `model_configs` inside `llm_stock_analyzer.py`.

## Usage

**Verify the data matches the paper (no API key needed):**

```bash
python verify_dataset_stats.py
```

Recomputes the jump-day band counts (1–2%, 2–3%, ≥3%) for each index directly
from the CSVs and checks them against the published Table 1 values.

**Run the LLM-ensemble backtest:**

```bash
# Quick run: NIFTY50, two split ratios — good for validating keys
python advanced_backtest_analyzer.py --scope quick --mode ensemble

# Full grid: 3 indices × 4 tasks × 6 ratios × {fixed, rolling 12/18/24m}
python advanced_backtest_analyzer.py --scope full --mode ensemble
```

Command-line options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--scope` | `quick` | `quick` = NIFTY50 + 2 ratios; `full` = all indices, all ratios, all rolling windows |
| `--mode` | `ensemble` | `ensemble` = LLM ensemble (Eq. 5); `rule` = deterministic technical-rule ablation (no API calls) |
| `--tau` | `0.5` | ensemble decision threshold τ |
| `--tiered-weights` | off | use heterogeneous reliability weights instead of equal weights |

**Run the recurrent baselines:**

```bash
python baselines/recurrent_baselines.py \
    --symbols NIFTY50 NIFTYBANK NIFTYIT \
    --movements direction sig_move_1pct sig_move_2pct sig_move_3pct \
    --models lstm gru
```

Writes `baselines/baseline_results.json` with Accuracy/Precision/Recall/F1 per
index, task, and model. Additional flags: `--train-ratio`, `--lookback`,
`--epochs`, `--data-dir`, `--out`.

**Generate single-day LLM predictions (illustrative):**

```python
import asyncio
from llm_stock_analyzer import LLMStockPredictor

async def run():
    predictor = LLMStockPredictor("data")
    results = await predictor.predict_all_movements(['gemini-flash'], 'NIFTY50')
    predictor.display_movement_results('NIFTY50', results)
    predictor.save_movement_results('NIFTY50', results)

asyncio.run(run())
```

## Output files

**LLM prediction dumps** — `results/{SYMBOL}_all_movements_{timestamp}.json`.
Per-symbol JSON with each model's prediction, confidence, reasoning string, and
the technical snapshot, grouped by movement type.

**Backtest workbook** — `advanced_backtest_results/comprehensive_backtest_results_{timestamp}.xlsx`,
with sheets:

| Sheet | Contents |
|-------|----------|
| `All_Results` | Every backtest run, full metric set |
| `Train_Ratio_Summary` | Performance aggregated by train/test ratio |
| `Stock_Movement_Summary` | Performance by index and movement type |
| `Split_Type_Comparison` | Fixed vs rolling-window comparison |
| `Best_Performers` | Top configuration per metric |
| `Ratio_XX_Detail` | Detailed runs for each fixed ratio |

A companion `comprehensive_backtest_report_{timestamp}.txt` gives an overall and
per-segment summary.

## Caching and reproducibility

Every distinct day's ensemble query is cached on disk under `llm_cache/`, keyed by
a SHA-256 hash of the exact prompt text and the active model set. Because the same
market state recurs across overlapping splits and rolling windows, the cache makes
the first pass the only expensive one and makes repeated runs deterministic —
re-running an identical configuration reuses cached responses and reproduces the
same numbers. To force fresh API calls, delete `llm_cache/` (or the relevant
entries).

The generated directories (`llm_cache/`, `results/`, `advanced_backtest_results/`)
ship empty; running the commands above populates them. A clean
end-to-end reproduction is therefore:

1. `python verify_dataset_stats.py` — confirm the data is intact (Table 1 match).
2. `python advanced_backtest_analyzer.py --scope full --mode ensemble` —
   regenerate the LLM-ensemble result workbook backing the headline tables.
3. `python baselines/recurrent_baselines.py` — regenerate the LSTM/GRU comparison.

## Cost and throughput

A full-grid ensemble run issues one six-model query for every test day across
every index, task, split ratio, and rolling window — on the order of thousands of
LLM calls, each billed by the respective provider. The on-disk cache amortises
this across re-runs, but the first full pass is genuinely expensive in both time
and API spend. Recommended practice: start with `--scope quick`, or restrict the
registry to a single cheap model, to confirm your keys and environment before
launching the full grid. Per-query latency for the lightweight tier is typically a
few seconds; the parallel dispatch keeps per-day latency near that of the slowest
single model.

## Data format

Files follow the naming convention `{SYMBOL}_*_Direction.csv` and
`{SYMBOL}_*_SigMove_{threshold}pct.csv` (with `neg` variants for the bearish
tasks). Note that the NIFTY50 direction file is `NIFTY50_2007_2025_with_Direction.csv`
(without `_Daily`), while the other index files include `_Daily`; the loader keys
on the `Direction` / `SigMove` substrings, so this is handled transparently.

Required columns:

```csv
Date,Open,High,Low,Close,Adj_Close,Volume,Direction
2024-01-01,19000.50,19150.25,18950.75,19100.00,19100.00,250000000,1
```

- Price columns are numeric; `Adj_Close` is used for return computation.
- `Volume` may be a string with thousands separators (handled on load).
- The movement column is `Direction` (1 = up, 0 = down) or `Significant_Move`
  (1 = significant, 0 = normal). The loader auto-detects which is present.

The loader coerces dates, strips thousands separators, drops invalid rows, and
sorts chronologically.

## Extending the framework

- **Successor models** — edit `model_configs` in `llm_stock_analyzer.py`.
- **Negative-tail prediction** — point a run at the `*_SigMove_neg{1,2,3}pct.csv`
  label files; no code change is needed.
- **Richer features** — `g(·)` is the single integration point for new inputs
  (for example microstructure or liquidity descriptors); extend the prompt
  template and the feature builders in `llm_stock_analyzer.py`.
- **Calibration analysis** — the cached per-model responses retain confidences,
  which is the natural raw material for a reliability-diagram or
  expected-calibration-error study.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `API key not found for {provider}` | The provider's key is missing from `.env`; that model is skipped. Supply the key or remove the model from the registry. |
| `No datasets found for {SYMBOL}` | Data directory or file-naming mismatch; check the `data/` filenames against the convention above. |
| `Insufficient data` warning | The dataset has fewer than 200 rows after cleaning. |
| Failed to parse JSON response | A model returned malformed output; it contributes the neutral fallback for that day. Persistent failures usually indicate a rate limit. |
| Full run is slow / costly | Expected — use `--scope quick` or a single provider first; rely on `llm_cache/` for re-runs. |

## Disclaimer

For educational and research purposes only. Market predictions are inherently
uncertain and past performance does not guarantee future results. Ensure
compliance with each API provider's terms of service, the data source's
licensing, and the financial regulations in your jurisdiction.
