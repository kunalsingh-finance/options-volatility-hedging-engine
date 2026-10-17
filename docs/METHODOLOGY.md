# Methodology

## Observations and conventions

Select four active inverse-option expiries nearest 14, 35, 75 and 150 days within a 10–190 day window, then at most eleven representative OTM strikes per expiry. Selection uses the contemporaneous summary and instrument list before requesting individual books. Selection exclusions and subsequent cleaning rejections are distinct counts. The snapshot is sequential, not synchronized tick data.

Each book supplies coin bid/ask, index, expiry forward, and exchange timestamp. Multiply coin prices by that book's index to obtain USD equivalents per base unit. Use the expiry forward F, strike K, and ACT/365 maturity T with discount factor one. No rate curve is estimated. A standing order may have been unchanged for a while: the book timestamp demonstrates response freshness, not the last amendment time of each quote.

Reject unsupported metadata, missing/nonpositive sides, crossed prices, excessive relative spread, stale observations, invalid pricing bounds, inversion failures and duplicates. Verify raw SHA-256 digests on replay. After capture, freshness is evaluated relative to the recorded retrieval time so reproducibility does not depend on today's date.

## Black benchmark

For a European call, C = F N(d1) − K N(d2), d1 = log(F/K)/(sigma sqrt(T)) + sigma sqrt(T)/2 and d2 = d1 − sigma sqrt(T). Put prices use put–call parity. The implementation supports positive discount factors but this research fixes the factor at one.

Delta and gamma differentiate the forward, with variance parameters fixed. Vega is per absolute annualized volatility unit, not per one percentage point. Theta is the calendar-time derivative at fixed forward. Prices at expiry and at zero volatility use exact intrinsic values.

Fit one constant Black volatility per expiry, using calibration rows only. This is a benchmark with an expiry term structure but no strike smile.

## SSVI

Let k = log(K/F) and theta(T) be at-the-money total variance. Use

    w(k,theta) = theta/2 * [1 + rho*phi(theta)*k
                       + sqrt((phi(theta)*k + rho)^2 + 1-rho^2)]
    phi(theta) = eta / sqrt(theta*(1+theta))
    implied_vol = sqrt(w/T)

Positive cumulative increments produce nondecreasing theta knots. Linear interpolation includes (0,0); beyond the last expiry theta is held flat. Extrapolation is a declared convention, not a forecast. Gamma in the modified power-law family is fixed at 1/2; rho and eta are shared across expiries.

The parameter transform imposes |rho| < 0.95 and eta²(1+|rho|) < 3.95. It also implies eta(1+|rho|) < 4. For this phi, d(theta*phi)/dtheta divided by phi equals 1/[2(1+theta)], satisfying the calendar-spread sufficient restriction. With monotone theta, the Gatheral–Jacquier sufficient conditions hold under the model's normalized forward-price convention.

The density diagnostic is

    g(k) = (1 − k*w'/(2w))² − (w'²/4)*(1/w + 1/4) + w''/2.

Record its minimum and calendar violations over a broad grid, plus the analytical constraint margins. Separate tests independently check call-price monotonicity and convexity. A positive diagnostic does not mean every observed quote fits inside its spread.

## Heston

Use a zero-drift forward-price process with variance dynamics

    dF/F = sqrt(v) dW_s
    dv = kappa*(theta − v) dt + sigma*sqrt(v) dW_v
    corr(dW_s,dW_v) = rho.

In Heston, theta and v0 denote variance and sigma is volatility of variance. These names are distinct from SSVI theta(T), which denotes total variance. Price European options with the normalized log-return characteristic function and a damped Lewis integral. Gauss–Legendre quadrature defaults to 128 nodes and upper cutoff 200 for calibration; a 256-node/400-cutoff calculation independently checks stability on each saved quote. Exact characteristic-function moments avoid a removable algebraic singularity at u = −i.

Fit five parameters using three deterministic starts, least squares on scaled price residuals, and bounds: kappa [0.15,12], theta/v0 [0.01,2.25], sigma [0.05,3], rho [−0.95,0.5]. The Feller condition 2*kappa*theta >= sigma² is reported, not imposed. Boundary fits are retained as model/identification warnings. Repeat fits after shifting calibration midpoints by ± one-quarter of the full bid–ask width to expose quote uncertainty. These refits are sensitivities, not confidence intervals.

Use residual scale max((ask−bid)/2, 0.0005*F), in USD. Calibrations minimize squared residuals under this scale. The same scale is used to report held-out errors. Every fourth strike per expiry is withheld; this is interpolation across one snapshot, not out-of-time validation.

## Risk scenarios

Reprice a normalized 30-day call at F=K=100. Spot shocks range from −20% to +20%. Volatility shocks add −0.10, 0, 0.10 or 0.20 to both sqrt(v0) and sqrt(theta), floored at 0.02. Skew shocks add −0.15, 0 or 0.15 to rho, clipped to [−0.98,0.98]; effective rho is saved. Prices are precomputed, and the browser selects saved rows.

Heston delta/gamma are forward-price sensitivities. Heston vega bumps sqrt(v0) alone, holding long-run variance fixed. The parallel volatility scenario changes both quantities, so the displayed vega is not its full derivative.

## Hedge mechanics

Sell one hypothetical USD call at a common initial Heston premium. Buy the chosen model delta and finance it through an unrestricted zero-rate cash account. At time i, observe only the current state, determine the hedge, and then expose that holding to the next price increment. Liquidate the underlying and pay the option payoff at expiry.

Cash recurrence is cash_new = cash_old − trade*spot − cost, with the terminal payoff also subtracted. The independent terminal identity is

    net_PnL = initial_premium + sum(held_delta_i*(spot_(i+1)−spot_i))
              − payoff − all_transaction_costs.

Both models use identical paths, premiums, costs and maturity. Black uses a frozen ATM SSVI volatility; Heston uses baseline structural parameters with the current simulated variance observed exactly. Heston's ordinary partial-price delta does not include the correlated-variance term of a minimum-variance hedge.

Policies rebalance each step, every six steps, or when target minus held delta reaches 0.05 in absolute value. Costs are 5 bp of absolute dollar turnover, including opening and closing; the high-cost case uses 20 bp on the identical base paths. Other scenarios multiply initial/long-run variance by 2.25 or apply an unpriced deterministic 12% downward jump at the midpoint. Keep the common baseline premium even in stress cases: those are risk shocks, not repriced fair-premium comparisons.

Simulation uses log-Euler spot and full-truncation latent variance. Negative latent proposals are retained and counted; the positive part enters drift and diffusion. No path deletion or P&L winsorization. Heston deltas must agree within 0.0002 on successive quadrature grids; difficult states receive higher-order integration. A numerical delta bound tolerance of 0.00001 is explicit; small excursions are retained, and unresolved states abort. A separate adaptive probability-integral delta is independently tested as an additional numerical reference.

## Attribution and uncertainty

For both hedge models, use the same conditional-Black reference price C(S,t,sqrt(v)). Compute local delta, gamma, theta and variance terms from the beginning of each interval. An explicit variance floor is used only for singular diagnostic Greeks, never for trading or path simulation. Save its activation count. Initial Heston premium minus initial reference price is a separate premium gap.

The exact reference repricing remainder is retained. Premium gap plus Greek contributions plus that remainder minus transaction costs equals net P&L. A large residual means the Taylor terms explain only part of the movement; it is not an attribution failure to hide.

Sample means use normal-approximation 95% Monte Carlo intervals. Paired model differences use the same path index and their own standard error. Sample quantiles have no asserted confidence bounds. These intervals omit parameter, model and discretization uncertainty.

The default independent price benchmark uses 32,768 paths at 120 and 240 steps, coupled by aggregating the finer Brownian increments. Compare both price levels to the analytic price and retain their paired difference and negative-variance frequencies. Two resolutions are a sensitivity check, not a proof of convergence. See the saved numerical results before interpreting apparent strategy differences.
