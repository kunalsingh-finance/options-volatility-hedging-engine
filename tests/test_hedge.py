import numpy as np
import pytest

from options_engine import black
from options_engine.hedge import account_strategy, simulate_paths, _reference_attribution, _validated_heston_delta, run_experiment


PARAMS = dict(kappa=2.0, theta=.09, sigma=.25, rho=-.6, v0=.09)


def test_self_financing_cash_identity_and_entry_exit_costs():
    spot = np.array([[100., 110., 90., 120.]])
    targets = np.array([[.5, .7, .2, 99.]])  # Terminal target is ignored: liquidate.
    out = account_strategy(spot, targets, 10., cost_rate=.001)
    expected_gross = 10 + .5 * 10 + .7 * -20 + .2 * 30 - 20
    expected_cost = .001 * (50 + .2 * 110 + .5 * 90 + .2 * 120)
    assert out["gross_pnl"][0] == pytest.approx(expected_gross)
    assert out["total_cost"][0] == pytest.approx(expected_cost)
    assert out["net_pnl"][0] == pytest.approx(expected_gross - expected_cost)
    assert out["cash"][0, -1] == pytest.approx(out["net_pnl"][0])
    assert out["held"][0, -1] == 0
    assert out["trade_count"][0] == 4
    assert np.max(np.abs(out["accounting_error"])) < 1e-12


def test_cost_increase_is_exactly_turnover_penalty():
    spot = np.array([[100., 103., 102., 106.]])
    targets = np.array([[.5, .6, .55, 0.]])
    low = account_strategy(spot, targets, 5., cost_rate=.0005)
    high = account_strategy(spot, targets, 5., cost_rate=.002)
    assert high["gross_pnl"] == pytest.approx(low["gross_pnl"])
    assert high["total_cost"] == pytest.approx(4 * low["total_cost"])
    assert low["net_pnl"] - high["net_pnl"] == pytest.approx(3 * low["total_cost"])


def test_rebalance_observes_current_state_without_earning_previous_move():
    spot = np.array([[100., 120., 125.]])
    before = account_strategy(spot, np.array([[0., 0., 0.]]), 0., cost_rate=0.)
    after = account_strategy(spot, np.array([[0., 1., 0.]]), 0., cost_rate=0.)
    assert after["gross_pnl"] - before["gross_pnl"] == pytest.approx([5.])
    assert after["cash"][0, 0] == before["cash"][0, 0]


def test_future_targets_cannot_change_earlier_holdings_or_cash():
    spot = np.array([[100., 110., 90., 120., 118.]])
    a = np.array([[.5, .6, .55, .7, 0.]])
    b = a.copy()
    b[:, 3:] = [0.2, -10.]
    for policy in ("every_step", "every_6_steps", "delta_band"):
        out_a = account_strategy(spot, a, 10., policy=policy)
        out_b = account_strategy(spot, b, 10., policy=policy)
        np.testing.assert_array_equal(out_a["held"][:, :3], out_b["held"][:, :3])
        np.testing.assert_array_equal(out_a["cash"][:, :3], out_b["cash"][:, :3])


def test_policy_schedule_and_band_have_distinct_holdings():
    spot = np.full((1, 9), 100.)
    targets = np.array([[.5, .52, .53, .57, .58, .59, .61, .62, 0.]])
    periodic = account_strategy(spot, targets, 2., policy="every_6_steps")
    band = account_strategy(spot, targets, 2., policy="delta_band")
    np.testing.assert_allclose(periodic["held"], [[.5, .5, .5, .5, .5, .5, .61, .61, 0.]])
    np.testing.assert_allclose(band["held"], [[.5, .5, .5, .57, .57, .57, .57, .62, 0.]])


def test_full_truncation_retains_negative_latent_variance():
    params = dict(kappa=1., theta=.04, sigma=2., rho=0., v0=.04)
    shocks = np.zeros((1, 3, 2))
    shocks[0, 0, 1] = -10
    sim = simulate_paths(params, paths=1, steps=3, maturity=.3, shocks=shocks)
    assert sim["raw_variance"][0, 1] < 0
    assert sim["raw_variance"][0, 2] < 0  # Not projected to zero after first step.
    assert sim["variance"][0, 1] == 0
    assert sim["negative_variance_proposals"] == 3
    assert sim["spot"][0, 2] == sim["spot"][0, 1]


def test_jump_reuses_diffusion_draws_and_changes_only_spot():
    a = simulate_paths(PARAMS, paths=16, steps=12, seed=4)
    b = simulate_paths(PARAMS, paths=16, steps=12, seed=4, jump_fraction=-.12)
    np.testing.assert_array_equal(a["variance"], b["variance"])
    np.testing.assert_array_equal(a["spot"][:, :6], b["spot"][:, :6])
    np.testing.assert_allclose(b["spot"][:, 6:], .88 * a["spot"][:, 6:], rtol=1e-14)


def test_simulation_reproducible_and_rejects_bad_inputs():
    a = simulate_paths(PARAMS, paths=4, steps=6, seed=8)
    b = simulate_paths(PARAMS, paths=4, steps=6, seed=8)
    np.testing.assert_array_equal(a["spot"], b["spot"])
    with pytest.raises(ValueError):
        simulate_paths(dict(PARAMS, rho=1.0))
    with pytest.raises(ValueError):
        simulate_paths(PARAMS, paths=2, steps=3, shocks=np.zeros((2, 2, 2)))


def test_monte_carlo_spot_martingale_and_variance_first_moment():
    sim = simulate_paths(PARAMS, paths=24000, steps=64, seed=5981, maturity=.25)
    terminal = sim["spot"][:, -1]
    standard_error = terminal.std(ddof=1) / np.sqrt(len(terminal))
    assert abs(terminal.mean() - 100.) < 4 * standard_error
    variance = sim["variance"][:, -1]
    variance_se = variance.std(ddof=1) / np.sqrt(len(variance))
    assert abs(variance.mean() - PARAMS["theta"]) < 4 * variance_se + 1e-4
    assert sim["negative_variance_fraction"] < 1e-4


def test_greek_attribution_has_explicit_exact_residual_and_premium_gap():
    sim = simulate_paths(PARAMS, paths=12, steps=12, seed=18, jump_fraction=-.12)
    s, v = sim["spot"], sim["variance"]
    reference = _reference_attribution(s, v, sim["dt"])
    t = np.maximum(30 / 365 - np.arange(13) * sim["dt"], 0)
    targets = black.greeks(s, 100., t, .3)["delta"]
    out = account_strategy(s, targets, 3.1)
    terms = out["held"][:, :-1] * np.diff(s, axis=1)
    terms -= sum(reference[k] for k in ("delta", "gamma", "theta", "variance", "residual"))
    reconstructed = 3.1 - reference["option_value"][:, 0] + terms.sum(axis=1) - out["total_cost"]
    np.testing.assert_allclose(reconstructed, out["net_pnl"], atol=1e-12)
    assert np.max(np.abs(reference["residual"])) > .01


def test_nonconverged_delta_is_rejected_instead_of_clipped(monkeypatch):
    from options_engine import heston

    monkeypatch.setattr(heston, "delta", lambda spot, *args, **kwargs: np.full_like(spot, 1.2))
    with pytest.raises(ArithmeticError, match="no clipping applied"):
        _validated_heston_delta(np.array([100.]), .01, dict(PARAMS, v0=np.array([.09])))


def test_small_experiment_is_json_safe_and_uses_common_premium_paths_and_costs():
    import json

    result = run_experiment(PARAMS, .3, seed=42, paths=8, steps=12)
    json.dumps(result, allow_nan=False)
    assert len(result["cases"]) == 24
    assert len(result["paired_comparisons"]) == 12
    assert len(result["sample_ledgers"]) == 24
    assert len({entry["premium"] for entry in result["sample_ledgers"]}) == 1
    for model in ("black_scholes", "heston"):
        for policy in ("every_step", "every_6_steps", "delta_band"):
            low, high = [next(case for case in result["cases"] if case["model"] == model and case["policy"] == policy and case["scenario"] == scenario) for scenario in ("base", "high_cost")]
            assert high["mean_cost"] == pytest.approx(4 * low["mean_cost"])
            assert low["mean_pnl"] - high["mean_pnl"] == pytest.approx(3 * low["mean_cost"])
    for ledger in result["sample_ledgers"]:
        assert len(ledger["rows"]) == 13
        assert ledger["rows"][-1]["cash"] == pytest.approx(ledger["net_pnl"])
        assert ledger["rows"][-1]["wealth"] == pytest.approx(ledger["net_pnl"])
    assert result["validation"]["accounting_max_error"] < 1e-10
    assert result["validation"]["attribution_max_error"] < 1e-10
