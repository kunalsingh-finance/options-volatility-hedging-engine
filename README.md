# Options Volatility & Dynamic Hedging Engine

Calibrate a volatility surface and stochastic-volatility model to observed BTC/ETH option quotes, then compare delta hedges through a fully accounted simulation.

**[Open the interactive research report](https://kunalsingh-finance.github.io/options-volatility-hedging-engine/)**. For offline use, download the repository and double-click **Open Research Report.cmd** on Windows, or open [output/report.html](output/report.html) locally. The report includes market and expiry filters, a volatility smile and surface, precomputed risk shocks, hedge distributions, paired confidence intervals, and complete sample cash ledgers.

Market quotes are observed. Model prices and hedge outcomes are hypothetical. The project does not place trades or present a historical hedging backtest.

## Research question

How much does a richer volatility model improve option-price fit, and does its conventional price delta improve hedging outcomes after transaction costs?

The two questions have separate evidence. Cross-sectional quote holdouts assess interpolation. Common simulated paths assess hedge mechanics under specified processes and costs; they do not establish future trading performance.

## The saved experiment

- **88 observed quotes:** 44 BTC and 44 ETH, eleven selected out-of-the-money options per expiry across four expiries. The sequential capture began October 17, 2026 at 04:16:32 UTC; book timestamps span 47.146 seconds.
- **64 calibration observations and 24 holdouts:** every fourth strike within each expiry is reserved before fitting. No holdout prices enter the model objective.
- **Three pricing comparisons:** a fitted constant Black volatility for each expiry, constrained SSVI across all expiries, and a five-parameter Heston model with three deterministic optimization starts and quote-uncertainty refits.
- **120 precomputed risk scenarios:** 60 per currency, varying the normalized underlying, volatility level, and correlation/skew parameter.
- **24 hedge cases:** two models × three rebalance policies × four scenarios, with 256 common paths and 120 steps over a 30-day hypothetical contract. A separate, larger Monte Carlo calculation checks prices at two coupled timestep resolutions.

## Findings

The saved cross-sectional holdout price errors are:

| Market | Constant-vol Black RMSE | SSVI RMSE | Heston RMSE |
| --- | ---: | ---: | ---: |
| BTC | $191.00 | $44.24 | $64.99 |
| ETH | $5.63 | $1.20 | $2.13 |

Values are USD equivalents per one unit of the underlying, so BTC and ETH dollar errors are not directly comparable. The report also shows spread-scaled errors and the fraction of modeled prices inside the observed bid–ask interval. The objective uses half the USD spread with a 5 bp forward-price floor.

SSVI fits the withheld quotes better than the two alternatives in this snapshot. Fit remains imperfect, especially in the shortest-expiry wings. Heston reaches a parameter bound in each currency: BTC mean reversion and ETH volatility of variance. Both fitted models fail the sufficient Feller condition, which is allowed in the model but makes zero-variance behavior and discretization checks particularly relevant.

The baseline Heston every-step hedge has lower sample P&L dispersion than its Black counterpart, but **all twelve paired mean-P&L confidence intervals include zero** in the saved run. The experiment does not establish a mean-profit advantage. Large Greek-attribution residuals remain visible rather than being reassigned to convenient risk buckets.

## Reproduce offline

Python 3.11 is the tested environment. From this directory:

```powershell
python -m pip install -r requirements-lock.txt
python scripts/run_research.py
python scripts/verify_outputs.py
python -m pytest -q --basetemp=.pytest-local
```

The default run verifies and reuses the recorded raw snapshot. It does not request new data. The report and results are regenerated from the saved inputs. `run_status.json` remains failed or building until a complete validated pack is ready.

Explicitly acquire a new public snapshot:

```powershell
python scripts/run_research.py --refresh
```

A capture failure raises an error; it never substitutes synthetic quotes. Refresh changes the research input and may change findings or fail the minimum-data/numerical checks. Larger simulated experiments can use `--paths 1024 --steps 240 --seed 1729`. Rebalancing every six steps then represents a different calendar interval; compare resolutions accordingly.

## What is implemented

| Layer | Method and controls |
| --- | --- |
| Source capture | Production Deribit instrument metadata, summary selection, timestamped selected order books, raw JSON and SHA-256 manifests |
| Quote normalization | Coin premiums converted using each book's index; expiry-specific reported forwards; zero-discount convention; ACT/365 |
| Quote controls | Missing/crossed sides, excessive spread, stale snapshots, unsupported contracts, bounds, duplicate instruments, and implied-volatility failures |
| SSVI | Monotone total-variance knots, sufficient butterfly/wing constraints, density and calendar grids, constrained interpolation |
| Heston | Characteristic-function pricing, multi-start least squares, parameter bounds, numerical integration refinement, independent probability-integral checks |
| Risk | Forward delta/gamma, instantaneous-volatility vega, precomputed spot/volatility/correlation shocks |
| Hedging | Common paths and premium, current-state signals, fixed-interval and delta-band rules, initial/ongoing/terminal costs, settlement |
| Attribution | Explicit conditional-Black reference, premium gap, delta/gamma/theta/variance terms and an exact remaining residual |
| Audit | Reconstructed saved cash ledgers, distribution/summary consistency, paired confidence intervals, file and code fingerprints |

## Scope and interpretation

The captured contracts are Deribit inverse options. USD equivalents are a normalization of the observed coin quotes; calibration assumes zero discounting and uses the reported expiry forward. It does not infer a USD interest-rate curve or replicate native coin collateral, expiry-futures settlement, or perpetual funding.

The hedge experiment uses a separate short, USD cash-settled call with initial underlying and strike both 100, zero funding, and a common Heston premium. It is parameterized by the BTC fit; changing the report's market selector does not change the hedge experiment. Variance is perfectly observed for the Heston hedge. Its partial-price delta is not the variance-optimal hedge in a market with correlated, unhedgeable variance risk. Testing both models on Heston-generated paths favors that model structurally; the jump and volatility stresses are specified sensitivity cases, not fitted crisis forecasts.

The surface constraints apply to the modeled normalized price family. They do not certify that asynchronous raw quotes are mutually arbitrage-free or executable. Finite-grid diagnostics complement the analytical restrictions; neither establishes trading profitability.

Full-truncation Euler variance proposals, quadrature refinements, Monte Carlo uncertainty, boundary parameters, and model-fit failures remain visible. Two simulation resolutions do not prove asymptotic convergence. This is a research implementation, not a production risk system or validated trading strategy.

## Files and technical references

- [Methodology and equations](docs/METHODOLOGY.md)
- [Sources and provenance](docs/SOURCES.md)
- [Validation record](docs/VALIDATION.md)
- [Machine-readable research pack](output/research.json), [quote/model export](output/quotes.csv), [validation checks](output/validation.json)
- [Technical interfaces](docs/CONTRACT.md)

## Authorship

Independent research project developed with OpenAI Codex assistance for design, implementation, documentation and testing. The project is not affiliated with Deribit or the cited researchers.
