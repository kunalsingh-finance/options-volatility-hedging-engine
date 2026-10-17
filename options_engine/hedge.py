"""Auditable hypothetical USD cash-settled option hedge experiments.

Market snapshots do not supply the paths here. Paths are generated under a
zero-drift Heston diffusion, with explicitly specified stress overrides. Every
policy trades after observing the current state and earns the next increment.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import black, heston


POLICIES = ("every_step", "every_6_steps", "delta_band")
MODELS = ("black_scholes", "heston")
F0, STRIKE, MATURITY = 100.0, 100.0, 30.0 / 365.0
ATTRIBUTION_VARIANCE_FLOOR = 1e-8


def _parameters(params: dict) -> dict:
    p = {key: float(params[key]) for key in ("kappa", "theta", "sigma", "rho", "v0")}
    if not all(np.isfinite(x) for x in p.values()):
        raise ValueError("Heston parameters must be finite")
    if p["kappa"] <= 0 or p["theta"] <= 0 or p["sigma"] < 0 or p["v0"] < 0 or abs(p["rho"]) >= 1:
        raise ValueError("Invalid Heston diffusion parameters")
    return p


def simulate_paths(params: dict, *, paths: int = 256, steps: int = 120,
                   seed: int = 1729, initial_spot: float = F0,
                   maturity: float = MATURITY, shocks: np.ndarray | None = None,
                   jump_fraction: float = 0.0) -> dict:
    """Full-truncation Euler variance and log-Euler spot, returning all states.

    The latent variance is allowed to be negative; its positive part enters
    BOTH drift and diffusion. This is the full-truncation scheme, not a silently
    floored Euler state. Effective (nonnegative) variance is used for pricing.
    shocks[..., 0] drives spot; shocks[..., 1] is independent of the first.
    A specified jump multiplies spot by (1+jump_fraction) at the midpoint.
    """
    p = _parameters(params)
    if not isinstance(paths, (int, np.integer)) or not isinstance(steps, (int, np.integer)) or paths < 1 or steps < 1:
        raise ValueError("Positive integer paths and steps required")
    if not (np.isfinite(initial_spot) and initial_spot > 0 and np.isfinite(maturity) and maturity > 0):
        raise ValueError("Positive finite spot and maturity required")
    if not np.isfinite(jump_fraction) or jump_fraction <= -1:
        raise ValueError("Jump must leave a positive spot")
    if shocks is None:
        shocks = np.random.default_rng(seed).standard_normal((paths, steps, 2))
    shocks = np.asarray(shocks, float)
    if shocks.shape != (paths, steps, 2) or not np.all(np.isfinite(shocks)):
        raise ValueError("shocks must be finite with shape (paths, steps, 2)")
    spot = np.empty((paths, steps + 1))
    variance = np.empty_like(spot)
    raw_variance = np.empty_like(spot)
    spot[:, 0], variance[:, 0], raw_variance[:, 0] = initial_spot, p["v0"], p["v0"]
    dt, root_dt = maturity / steps, np.sqrt(maturity / steps)
    jump_step = max(1, steps // 2)
    for t in range(steps):
        v = np.maximum(raw_variance[:, t], 0.0)
        z_spot = shocks[:, t, 0]
        z_variance = p["rho"] * z_spot + np.sqrt(1 - p["rho"] ** 2) * shocks[:, t, 1]
        raw_variance[:, t + 1] = raw_variance[:, t] + p["kappa"] * (p["theta"] - v) * dt + p["sigma"] * np.sqrt(v) * root_dt * z_variance
        variance[:, t + 1] = np.maximum(raw_variance[:, t + 1], 0.0)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            spot[:, t + 1] = spot[:, t] * np.exp(-0.5 * v * dt + np.sqrt(v) * root_dt * z_spot)
        if jump_fraction and t + 1 == jump_step:
            spot[:, t + 1] *= 1 + jump_fraction
        if not (np.all(np.isfinite(spot[:, t + 1])) and np.all(spot[:, t + 1] > 0) and np.all(np.isfinite(raw_variance[:, t + 1]))):
            raise FloatingPointError(f"Nonfinite/zero simulated state at step {t + 1}; no path was discarded")
    return dict(spot=spot, variance=variance, raw_variance=raw_variance,
                dt=float(dt), negative_variance_proposals=int(np.count_nonzero(raw_variance[:, 1:] < 0)),
                negative_variance_fraction=float(np.mean(raw_variance[:, 1:] < 0)),
                jump_step=jump_step if jump_fraction else None)


def account_strategy(spot: np.ndarray, targets: np.ndarray, premium: float,
                     *, strike: float = STRIKE, cost_rate: float = 0.0005,
                     policy: str = "every_step", band: float = 0.05) -> dict:
    """Trade at current spot, then hold through the next interval.

    The initial option premium funds the stock hedge. The final row sells all
    stock, charges liquidation costs and settles the option payoff. Direct cash
    recursion is independently checked against the stochastic-integral identity.
    Arrays have one row per path and one column per state, including expiry.
    """
    s, target = np.asarray(spot, float), np.asarray(targets, float)
    if s.ndim != 2 or s.shape[1] < 2 or target.shape != s.shape:
        raise ValueError("spot and targets must have equal (paths, steps+1) shape")
    if not (np.all(np.isfinite(s)) and np.all(s > 0) and np.all(np.isfinite(target))):
        raise ValueError("Finite targets and positive finite spots required")
    if policy not in POLICIES or cost_rate < 0 or not np.isfinite(cost_rate) or band <= 0 or not np.isfinite(band):
        raise ValueError("Invalid trading policy or transaction costs")
    if not np.isfinite(premium) or not np.isfinite(strike) or strike <= 0:
        raise ValueError("Finite premium and positive strike required")
    held, cash, trades, costs = [np.zeros_like(s) for _ in range(4)]
    held[:, 0] = target[:, 0]
    trades[:, 0] = held[:, 0]
    costs[:, 0] = cost_rate * np.abs(trades[:, 0]) * s[:, 0]
    cash[:, 0] = premium - trades[:, 0] * s[:, 0] - costs[:, 0]
    for t in range(1, s.shape[1] - 1):
        rebalance = (policy == "every_step") or (policy == "every_6_steps" and t % 6 == 0)
        trade_flag = np.abs(target[:, t] - held[:, t - 1]) >= band if policy == "delta_band" else rebalance
        held[:, t] = np.where(trade_flag, target[:, t], held[:, t - 1])
        trades[:, t] = held[:, t] - held[:, t - 1]
        costs[:, t] = cost_rate * np.abs(trades[:, t]) * s[:, t]
        cash[:, t] = cash[:, t - 1] - trades[:, t] * s[:, t] - costs[:, t]
    payoff = np.maximum(s[:, -1] - strike, 0.0)
    trades[:, -1] = -held[:, -2]
    costs[:, -1] = cost_rate * np.abs(trades[:, -1]) * s[:, -1]
    cash[:, -1] = cash[:, -2] - trades[:, -1] * s[:, -1] - costs[:, -1] - payoff
    total_cost = costs.sum(axis=1)
    gross = premium + np.sum(held[:, :-1] * np.diff(s, axis=1), axis=1) - payoff
    net = gross - total_cost
    error = cash[:, -1] - net
    if np.max(np.abs(error)) > 1e-9 * max(1.0, float(np.max(s))):
        raise ArithmeticError("Self-financing cash recursion failed reconciliation")
    return dict(held=held, cash=cash, trades=trades, costs=costs,
                cumulative_cost=np.cumsum(costs, axis=1), payoff=payoff,
                gross_pnl=gross, net_pnl=net, total_cost=total_cost,
                trade_count=np.count_nonzero(np.abs(trades) > 1e-12, axis=1),
                accounting_error=error)


def _validated_heston_delta(spot: np.ndarray, maturity: float, params: dict) -> tuple[np.ndarray, dict]:
    """Refine numerical integration when deltas fail convergence or bounds.

    Nothing is clipped. Small quadrature differences are retained and reported.
    Unresolved deltas fail the experiment rather than manufacture hedge results.
    """
    coarse = np.asarray(heston.delta(spot, STRIKE, maturity, params, nodes=128, upper=200.0), float)
    fine = np.asarray(heston.delta(spot, STRIKE, maturity, params, nodes=256, upper=400.0), float)
    difference = np.abs(fine - coarse)
    bad = (~np.isfinite(fine)) | (fine < -1e-5) | (fine > 1 + 1e-5) | (difference > 2e-4)
    initial_bad = int(np.count_nonzero(bad))
    refined = 0
    for nodes, upper in ((512, 800.0), (1024, 1600.0), (2048, 3200.0), (4096, 6400.0), (8192, 12800.0), (16384, 25600.0)):
        if not np.any(bad):
            break
        idx = np.flatnonzero(bad)
        conditional = dict(params, v0=np.asarray(params["v0"])[idx])
        next_delta = np.asarray(heston.delta(spot[idx], STRIKE, maturity, conditional, nodes=nodes, upper=upper), float)
        difference[idx] = np.abs(next_delta - fine[idx])
        fine[idx] = next_delta
        refined += len(idx)
        bad = (~np.isfinite(fine)) | (fine < -1e-5) | (fine > 1 + 1e-5) | (difference > 2e-4)
    if np.any(bad):
        idx = int(np.flatnonzero(bad)[0])
        raise ArithmeticError(f"Heston delta did not converge: T={maturity:.7g}, S={spot[idx]:.7g}, v={np.asarray(params['v0'])[idx]:.7g}, delta={fine[idx]:.7g}, difference={difference[idx]:.7g}; no clipping applied")
    return fine, dict(initial_refinement_count=initial_bad, extra_evaluations=refined,
                     max_accepted_grid_difference=float(np.max(difference)),
                     min_delta=float(np.min(fine)), max_delta=float(np.max(fine)))


def _reference_attribution(spot: np.ndarray, variance: np.ndarray, dt: float) -> dict:
    remaining = np.maximum(MATURITY - np.arange(spot.shape[1]) * dt, 0.0)
    remaining[-1] = 0.0  # Settlement is exact, including an exactly-at-the-money path.
    diagnostic_variance = np.maximum(variance, ATTRIBUTION_VARIANCE_FLOOR)
    vol = np.sqrt(diagnostic_variance)
    value = np.asarray(black.price(spot, STRIKE, remaining, vol), float)
    greeks = black.greeks(spot[:, :-1], STRIKE, remaining[:-1], vol[:, :-1])
    ds = np.diff(spot, axis=1)
    dv = np.diff(diagnostic_variance, axis=1)
    delta = np.asarray(greeks["delta"]) * ds
    gamma = 0.5 * np.asarray(greeks["gamma"]) * ds ** 2
    theta = np.asarray(greeks["theta"]) * dt
    variance_term = np.asarray(greeks["vega"]) / (2 * vol[:, :-1]) * dv
    residual = np.diff(value, axis=1) - delta - gamma - theta - variance_term
    return dict(option_value=value, delta=delta, gamma=gamma, theta=theta,
                variance=variance_term, residual=residual,
                variance_floor_states=int(np.count_nonzero(variance < ATTRIBUTION_VARIANCE_FLOOR)))


def _stats(values: np.ndarray) -> dict:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    se = std / np.sqrt(len(values))
    return dict(mean=mean, std=std, mc_se=float(se), ci95_low=float(mean - 1.96 * se), ci95_high=float(mean + 1.96 * se))


def _convergence_benchmark(params: dict, premium: float, seed: int, steps: int, paths: int) -> dict:
    n = min(32768, max(1024, paths * 128))
    shocks = np.random.default_rng(seed + 104729).standard_normal((n, 2 * steps, 2))
    coarse_shocks = shocks.reshape(n, steps, 2, 2).sum(axis=2) / np.sqrt(2)
    coarse = simulate_paths(params, paths=n, steps=steps, shocks=coarse_shocks)
    fine = simulate_paths(params, paths=n, steps=2 * steps, shocks=shocks)
    levels = []
    payoffs = []
    for count, simulation in ((steps, coarse), (2 * steps, fine)):
        payoff = np.maximum(simulation["spot"][:, -1] - STRIKE, 0.0)
        payoffs.append(payoff)
        st = _stats(payoff)
        levels.append(dict(steps=count, paths=n, price=st["mean"], mc_se=st["mc_se"],
                           ci95_low=st["ci95_low"], ci95_high=st["ci95_high"],
                           difference_from_analytic=st["mean"] - premium,
                           negative_variance_fraction=simulation["negative_variance_fraction"]))
    return dict(analytic_price=premium, levels=levels,
                paired_fine_minus_coarse=_stats(payoffs[1] - payoffs[0]),
                note="Independent Monte Carlo benchmark; nested Brownian increments. Confidence intervals describe sampling error, not Euler discretization bias. Two levels cannot establish asymptotic convergence.")


def run_experiment(params: dict, atm_vol: float, seed: int = 1729,
                   paths: int = 256, steps: int = 120) -> dict[str, Any]:
    """Compare two model deltas and three policies on common synthetic paths."""
    p = _parameters(params)
    if paths < 2 or steps < 2 or not np.isfinite(atm_vol) or atm_vol <= 0:
        raise ValueError("At least two paths/steps and positive finite ATM volatility required")
    premium = float(heston.price(F0, STRIKE, MATURITY, p, nodes=256, upper=400.0))
    premium_check = float(heston.price(F0, STRIKE, MATURITY, p, nodes=512, upper=800.0))
    if not 0 < premium < F0 or abs(premium_check - premium) > 1e-4:
        raise ArithmeticError("Common Heston premium fails bounds or quadrature convergence")
    shocks = np.random.default_rng(seed).standard_normal((paths, steps, 2))
    scenario_specs = {
        "base": dict(cost_rate=0.0005, process="base"),
        "high_cost": dict(cost_rate=0.0020, process="base"),
        "volatility_stress": dict(cost_rate=0.0005, process="volatility_stress"),
        "spot_jump": dict(cost_rate=0.0005, process="spot_jump"),
    }
    simulations, targets_cache, reference_cache, path_summary = {}, {}, {}, {}
    numerical = dict(initial_refinement_count=0, extra_evaluations=0,
                     max_accepted_grid_difference=0.0, min_delta=1.0, max_delta=0.0,
                     grid_agreement_tolerance=2e-4, delta_bounds_tolerance=1e-5,
                     refinement_grids=[[128, 200], [256, 400], [512, 800], [1024, 1600],
                                       [2048, 3200], [4096, 6400], [8192, 12800], [16384, 25600]])
    for process in ("base", "volatility_stress", "spot_jump"):
        process_params = dict(p)
        if process == "volatility_stress":
            process_params.update(v0=p["v0"] * 2.25, theta=p["theta"] * 2.25)
        sim = simulate_paths(process_params, paths=paths, steps=steps, shocks=shocks,
                             jump_fraction=-0.12 if process == "spot_jump" else 0.0)
        s, v, dt = sim["spot"], sim["variance"], sim["dt"]
        remaining = np.maximum(MATURITY - np.arange(steps + 1) * dt, 0)
        bs_target = np.zeros_like(s)
        bs_target[:, :-1] = black.greeks(s[:, :-1], STRIKE, remaining[:-1], atm_vol)["delta"]
        heston_target = np.zeros_like(s)
        for t in range(steps):
            heston_target[:, t], diagnostic = _validated_heston_delta(s[:, t], float(remaining[t]), dict(p, v0=v[:, t]))
            for key in ("initial_refinement_count", "extra_evaluations"):
                numerical[key] += diagnostic[key]
            numerical["max_accepted_grid_difference"] = max(numerical["max_accepted_grid_difference"], diagnostic["max_accepted_grid_difference"])
            numerical["min_delta"] = min(numerical["min_delta"], diagnostic["min_delta"])
            numerical["max_delta"] = max(numerical["max_delta"], diagnostic["max_delta"])
        # The terminal target is liquidation, independent of in-the-money delta.
        bs_target[:, -1] = heston_target[:, -1] = 0.0
        simulations[process] = sim
        targets_cache[process] = dict(black_scholes=bs_target, heston=heston_target)
        reference_cache[process] = _reference_attribution(s, v, dt)
        terminal_stats = _stats(s[:, -1])
        expected_variance = process_params["theta"] + (process_params["v0"] - process_params["theta"]) * np.exp(-process_params["kappa"] * MATURITY)
        path_summary[process] = dict(terminal_spot=terminal_stats,
                                    theoretical_terminal_spot=F0 * (0.88 if process == "spot_jump" else 1.0),
                                    terminal_variance_mean=float(np.mean(v[:, -1])),
                                    continuous_model_expected_variance=float(expected_variance),
                                    negative_variance_proposals=sim["negative_variance_proposals"],
                                    negative_variance_fraction=sim["negative_variance_fraction"],
                                    attribution_variance_floor_states=reference_cache[process]["variance_floor_states"])
    cases, ledgers, distributions, pairs = [], [], [], []
    pnl_cache = {}
    max_accounting_error, max_attribution_error = 0.0, 0.0
    for scenario, spec in scenario_specs.items():
        sim = simulations[spec["process"]]
        s, v, dt = sim["spot"], sim["variance"], sim["dt"]
        reference = reference_cache[spec["process"]]
        for model in MODELS:
            targets = targets_cache[spec["process"]][model]
            for policy in POLICIES:
                accounting = account_strategy(s, targets, premium, cost_rate=spec["cost_rate"], policy=policy)
                pnl = accounting["net_pnl"]
                pnl_cache[scenario, model, policy] = pnl
                contributions = dict(
                    delta=accounting["held"][:, :-1] * np.diff(s, axis=1) - reference["delta"],
                    gamma=-reference["gamma"], theta=-reference["theta"],
                    variance=-reference["variance"], residual=-reference["residual"])
                premium_gap = premium - reference["option_value"][:, 0]
                reconstructed = premium_gap + sum(term.sum(axis=1) for term in contributions.values()) - accounting["total_cost"]
                attribution_error = float(np.max(np.abs(reconstructed - pnl)))
                accounting_error = float(np.max(np.abs(accounting["accounting_error"])))
                max_accounting_error = max(max_accounting_error, accounting_error)
                max_attribution_error = max(max_attribution_error, attribution_error)
                if attribution_error > 1e-8 * max(1.0, float(np.max(s))):
                    raise ArithmeticError("Greek attribution does not reconcile to net P&L")
                stats = _stats(pnl)
                case = dict(scenario=scenario, model=model, policy=policy,
                            mean_pnl=stats["mean"], pnl_std=stats["std"], mc_se=stats["mc_se"],
                            ci95_low=stats["ci95_low"], ci95_high=stats["ci95_high"],
                            p05=float(np.quantile(pnl, .05)), p95=float(np.quantile(pnl, .95)),
                            mean_cost=float(np.mean(accounting["total_cost"])),
                            mean_trades=float(np.mean(accounting["trade_count"])),
                            accounting_max_error=accounting_error,
                            attribution_max_error=attribution_error,
                            mean_premium_gap=float(np.mean(premium_gap)), paths=paths)
                for key, term in contributions.items():
                    case["mean_attribution_residual" if key == "residual" else f"mean_{key}_attribution"] = float(np.mean(term.sum(axis=1)))
                cases.append(case)
                distributions.append(dict(scenario=scenario, model=model, policy=policy, pnl=pnl.tolist()))
                rows = []
                for t in range(steps + 1):
                    # Terminal cash already includes option settlement; avoid subtracting payoff twice.
                    option_value = float(reference["option_value"][0, t])
                    wealth = float(accounting["cash"][0, t] + accounting["held"][0, t] * s[0, t] - (option_value if t < steps else 0))
                    row = dict(step=t, time=float(t * dt), spot=float(s[0, t]), variance=float(v[0, t]),
                               option_value=option_value, target_delta=float(targets[0, t]),
                               held_delta=float(accounting["held"][0, t]), trade=float(accounting["trades"][0, t]),
                               trade_cost=float(accounting["costs"][0, t]), cumulative_cost=float(accounting["cumulative_cost"][0, t]),
                               cash=float(accounting["cash"][0, t]), wealth=wealth)
                    for key, term in contributions.items():
                        row["attribution_residual" if key == "residual" else f"{key}_attribution"] = float(term[0, t - 1]) if t else 0.0
                    rows.append(row)
                ledgers.append(dict(scenario=scenario, model=model, policy=policy, path_index=0,
                                    rows=rows, premium=premium, premium_gap=float(premium_gap[0]),
                                    payoff=float(accounting["payoff"][0]), gross_pnl=float(accounting["gross_pnl"][0]),
                                    net_pnl=float(pnl[0]), accounting_error=float(accounting["accounting_error"][0])))
        for policy in POLICIES:
            diff = pnl_cache[scenario, "heston", policy] - pnl_cache[scenario, "black_scholes", policy]
            stats = _stats(diff)
            pairs.append(dict(scenario=scenario, policy=policy, model_a="heston", model_b="black_scholes",
                              mean_difference=stats["mean"], mc_se=stats["mc_se"], ci95_low=stats["ci95_low"], ci95_high=stats["ci95_high"],
                              interpretation="Heston net P&L minus Black-Scholes net P&L on identical paths; positive favors Heston."))
    return dict(assumptions=dict(
        status="Hypothetical simulated experiment; not historical or live trading performance.",
        contract="Short one European USD cash-settled call; F0=K=100, 30/365 years; zero drift and zero funding.",
        initial_spot=F0, strike=STRIKE, maturity=MATURITY, premium=premium, atm_vol=float(atm_vol),
        params=p, seed=seed, paths=paths, steps=steps,
        position="Short call, long model delta. Same initial Heston premium for every strategy and stress.",
        costs="Proportional absolute dollar turnover; 5 bp base and stress, 20 bp high-cost; includes initial hedge and terminal liquidation.",
        policies={"every_step": "Rebalance after every simulated state observation.",
                  "every_6_steps": "Rebalance at steps divisible by 6; actual interval changes with step count.",
                  "delta_band": "Rebalance when absolute target-minus-held delta reaches 0.05."},
        scenarios={"base": "Heston baseline diffusion.", "high_cost": "Exactly the base paths and targets, four times transaction-cost rate.",
                   "volatility_stress": "Initial variance and long-run variance multiplied by 2.25; same normal shocks. Heston hedge retains baseline structural parameters and observes current variance.",
                   "spot_jump": "Deterministic unpriced 12% downward spot jump at midpoint; same diffusion shocks and variance as base. Not a fitted jump process."},
        simulation="Log-Euler spot; full-truncation Euler latent variance. Negative variance proposals counted and positive part used in drift/diffusion. No path deletion, delta clipping, or P&L winsorization.",
        numerical_deltas="Heston deltas must agree on successive quadrature grids within 0.0002 and lie within [−0.00001, 1.00001]. Small numerical bound excursions are retained, not clipped. Unresolved states abort the experiment.",
        hedge_information="Targets use only current spot, observed current variance and time; no future path information. Variance is perfectly observed in this hypothetical experiment.",
        attribution="Common conditional-Black reference with volatility sqrt(current variance), not Heston Greeks. Interval-start delta/gamma/theta/variance Taylor terms, exact repricing residual, and initial premium gap reconcile to net P&L after costs.",
        attribution_variance_floor=ATTRIBUTION_VARIANCE_FLOOR,
        attribution_floor_note="Explicit diagnostic-only variance floor avoids the singular zero-variance Greek; does not affect paths, trading deltas, or accounting. Floor activations and the full residual are retained.",
        intervals="95% normal-approximation Monte Carlo intervals estimate mean uncertainty only; quantiles are sample estimates. Pairwise intervals use common-path differences.",
        limitations="No bid/ask dynamics, financing spreads, margin, liquidity, slippage beyond proportional cost, execution delay, stochastic rates, or coin collateral. Path count and Euler resolution limit precision."),
        cases=cases, paired_comparisons=pairs, sample_ledgers=ledgers,
        distribution_samples=distributions, path_summary=path_summary,
        validation=dict(accounting_max_error=max_accounting_error, attribution_max_error=max_attribution_error,
                        common_premium_grid_difference=abs(premium_check - premium), delta_quadrature=numerical,
                        convergence_benchmark=_convergence_benchmark(p, premium, seed, steps, paths)))
