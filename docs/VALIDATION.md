# Executed validation record

Validated October 17, 2026 against the recorded production Deribit snapshot. These are executed local checks, not a claim of professional model validation or successful GitHub Actions execution.

## Results

- **58 pytest tests passed**, covering quote filtering, source integrity, Black inversion and Greeks, Heston moments and independent integration, constrained SSVI geometry, hedge timing/accounting, common paths, report serialization, and corrupted-output rejection.
- **82 research checks passed** in `output/validation.json`.
- `scripts/verify_outputs.py` passed: 88 observed quotes reconstructed from raw files, 82 saved numerical/accounting checks, and four artifact hashes verified. Source-code fingerprints match the build.
- All 24 hedge-case identities and sample ledgers are present. Cash, holdings, payoff, costs, marked wealth, attribution and terminal P&L reconcile. The maximum reported cash-accounting discrepancy is below 2e-13 USD per normalized option.
- All twelve paired confidence intervals are reconstructed from common-path P&L differences. Regression tests reject altered interval endpoints, missing ledgers and corrupted reference option marks.

## Pricing and calibration checks

Black reference pricing is checked against a known at-the-money value, put–call parity, implied-volatility round trips and numerical derivatives. Heston uses an independent P1/P2 integration check against the damped pricing formula, a deterministic-variance Black limit, characteristic-function moments, vector variance inputs and quadrature refinement. An adaptive oscillatory-integral delta check covers a near-zero-variance, short-dated case.

SSVI sufficient-condition margins are positive, total-variance knots are nondecreasing, and the tested density/calendar grids pass. Independent call-price convexity tests complement the implementation's density formula. Both optimizers terminate successfully, but boundary parameters and Feller-condition violations are retained as substantive limitations. Neither optimization success nor numerical consistency means the model is empirically validated.

## Independent Monte Carlo price benchmark

The saved BTC-normalized call has analytic value **3.96110 USD**. A separate 32,768-path experiment gives:

| Resolution | Simulated price | Monte Carlo SE | Negative latent variance fraction |
| --- | ---: | ---: | ---: |
| 120 steps | 3.95119 | 0.03604 | 4.455% |
| 240 steps | 3.94893 | 0.03601 | 3.171% |

The coupled fine-minus-coarse price difference is −0.00226, with a 95% Monte Carlo interval of approximately [−0.01004, 0.00551]. Both analytic-price comparisons pass the declared four-standard-error diagnostic. This is evidence of numerical consistency at the sampled resolutions, not proof that discretization bias is zero. Negative latent variance is handled by the disclosed full-truncation scheme.

The principal hedge comparison uses 256 paths. All twelve paired mean-P&L intervals include zero, so the saved study does not establish mean-profit superiority. Its larger independent price benchmark does not increase the hedge comparison's effective path count.

## Browser verification

The final report was served on an isolated localhost port and inspected using Playwright in a real browser:

- Market selection changes BTC/ETH quote and calibration views.
- Risk controls select saved spot, volatility and skew shocks; an inspected ETH +20% spot / +10 vol points / −0.15 skew scenario displayed a Heston price of $21.77 and repricing P&L of $16.29.
- Hedge scenario, model and policy selectors update distributions, statistics, paired intervals and the full 121-row example ledger.
- The hedge view explicitly identifies its BTC-calibrated hypothetical USD contract, independent of the other market views.
- The evidence view exposes source records, calibration diagnostics and simulation numerical evidence.
- Browser JSON export equals the saved `research.json` content exactly.
- A 390 × 844 mobile viewport has no page-level horizontal overflow; the tab strip and wide ledgers scroll within their containers. Desktop layout was also inspected visually.
- The report embeds its data, styles, graphics, script and favicon. No external runtime chart/font/library dependency is required.

Browser screenshots are local inspection artifacts in ignored `output/playwright/`. The browser preview is optional: the generated HTML also opens directly from disk.

## Rechecking

```powershell
python -m pytest -q --basetemp=.pytest-local
python scripts/verify_outputs.py
```

After changing model code, tests, inputs or report generation, rebuild with `python scripts/run_research.py` before verification. The verifier refuses stale code/artifact fingerprints. A failed build replaces the report with an explicit failure page and records the error, avoiding an old green report beside a failed current run.
