# Technical interfaces

The `options_engine` package supports Python 3.11. Scripts load it from the repository root. The default workflow replays saved data offline; acquisition of a new snapshot is explicit.

## Pricing

`black.price(forward, strike, maturity, volatility, option_type='call', discount=1.0)` broadcasts numerical arrays. `black.greeks(...)` returns forward delta, gamma, vega and calendar theta. Vega is per absolute volatility unit; theta is per year at fixed forward. `black.implied_vol(value, forward, strike, maturity, option_type='call')` inverts one admissible price.

`heston.price(forward, strike, maturity, params, option_type='call', nodes=128, upper=200.0)` broadcasts forward, strike, maturity and initial variance. Parameters are `kappa`, `theta`, `sigma`, `rho` and `v0`. Here theta and v0 are variances; sigma is volatility of variance. Pricing assumes zero discounting under a forward measure. `heston.delta(...)` uses centered price differences, while `heston.delta_adaptive(...)` provides a scalar probability-integral check and its integration-error estimate.

## Market data

`data.capture(root: Path)` saves a bounded production Deribit BTC/ETH snapshot under `data/raw/`, with a manifest. It selects up to four expiry slices and eleven OTM strikes per slice, using instrument metadata and summaries, then retrieves the selected books. Retrieval failures raise errors; synthetic prices are never substituted.

`data.load_quotes(root: Path)` verifies recorded hashes and returns `(quotes, metadata)`. Each accepted row includes currency, instrument, expiry, timestamp, ACT/365 maturity, forward, index, strike, option type, USD-equivalent bid/ask/mid, native coin bid/ask, implied volatility, USD spread, relative spread and source URL. Rejections and reasons are saved separately.

The USD conversion uses the contemporaneous index. The model uses the reported expiry forward. Mark and last-trade prices are not substitutes for missing bid/ask sides. Exact selection and cleaning thresholds are recorded in the source manifest and methodology.

## Calibration and risk

`calibration.fit_currency(quotes)` assigns calibration and cross-sectional holdout rows, fits Black, SSVI and Heston models on calibration rows, and returns quotes with modeled prices, model parameters, diagnostics, error metrics and precomputed risk scenarios.

SSVI total variance uses shared rho and eta parameters and monotone maturity knots. `surface.iv(k, maturity, surface)` interpolates the fitted surface. Heston diagnostics include optimizer outcomes, boundary parameters, Feller-condition status, integration stability and quote-uncertainty refits.

Each risk scenario records `spot_shock`, `vol_shock`, `skew_shock`, effective rho, model prices, forward sensitivities and Heston repricing P&L. Exact shock mappings and Greek units accompany the results.

## Hedge simulation

`hedge.run_experiment(params, atm_vol, seed=1729, paths=256, steps=120)` returns JSON-compatible results for a hypothetical USD call with initial underlying and strike 100, maturity 30/365 and zero funding.

The experiment compares frozen-volatility Black and current-variance Heston deltas across `every_step`, `every_6_steps` and `delta_band` policies. Scenarios are `base`, `high_cost`, `volatility_stress` and `spot_jump`. Models share paths, initial premium and cost conventions within each comparison. Opening and closing costs, cash financing and terminal payoff are included.

Outputs contain assumptions, 24 case summaries, paired model comparisons, complete sample ledgers, terminal P&L samples, path diagnostics and a coupled-resolution price benchmark. Gross P&L equals initial premium plus cumulative hedge trading gains minus payoff; net P&L subtracts all costs. Conditional-Black attribution includes a separate premium gap and exact remaining residual.

## Report and audit

`report.write_report(result, path)` creates a standalone HTML report with embedded data, CSS, JavaScript and SVG. The main result contains `metadata`, per-currency models and quotes, `hedge` results, and `validation` checks. Controls select saved scenarios; the browser does not fetch prices, recalibrate models or run simulations.

`validation.validate(result)` checks numerical bounds, surface restrictions, hedge-case completeness, saved cash ledgers, marked wealth, distributions, paired intervals and integration diagnostics. `scripts/verify_outputs.py` also checks source/code/artifact fingerprints and reconstructs source quotes. A changed model or input requires a fresh build before the output is accepted.
