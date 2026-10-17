"""Discounted Black forward-option prices. Volatility is annualized, time ACT/365."""
import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq


def _inputs(forward, strike, maturity, volatility, option_type):
    if option_type not in ('call', 'put'):
        raise ValueError('option_type must be call or put')
    f, k, t, v = np.broadcast_arrays(*map(lambda x: np.asarray(x, dtype=float),
                                         (forward, strike, maturity, volatility)))
    if not all(np.all(np.isfinite(x)) for x in (f, k, t, v)):
        raise ValueError('nonfinite pricing input')
    if np.any(f <= 0) or np.any(k <= 0) or np.any(t < 0) or np.any(v < 0):
        raise ValueError('F,K positive; T,vol nonnegative required')
    return f, k, t, v


def price(forward, strike, maturity, volatility, option_type='call', discount=1.0):
    f, k, t, v = _inputs(forward, strike, maturity, volatility, option_type)
    if not np.isfinite(discount) or discount <= 0:
        raise ValueError('discount must be positive')
    st = v * np.sqrt(t)
    d1 = np.log(f / k) / np.maximum(st, 1e-300) + 0.5 * st
    d2 = d1 - st
    sign = 1 if option_type == 'call' else -1
    regular = sign * (f * ndtr(sign * d1) - k * ndtr(sign * d2))
    return discount * np.where(st > 1e-14, regular, np.maximum(sign * (f-k), 0))


def greeks(forward, strike, maturity, volatility, option_type='call', discount=1.0):
    f, k, t, v = _inputs(forward, strike, maturity, volatility, option_type)
    if not np.isfinite(discount) or discount<=0:
        raise ValueError('discount must be positive')
    st = v * np.sqrt(t)
    d1 = np.log(f / k) / np.maximum(st, 1e-300) + 0.5 * st
    density = np.exp(-0.5 * np.minimum(np.abs(d1),40.)**2) / np.sqrt(2*np.pi)
    smooth = st > 1e-14
    terminal_delta = np.where(f > k, 1., np.where(f < k, 0., .5))
    delta = np.where(smooth, ndtr(d1), terminal_delta) - (option_type == 'put')
    return dict(delta=discount*delta,
                gamma=discount*np.where(smooth, density/(f*np.maximum(st, 1e-300)), 0.),
                vega=discount*np.where(smooth, f*density*np.sqrt(t), 0.),
                theta=discount*np.where(smooth, -f*density*v/(2*np.sqrt(np.maximum(t,1e-300))), 0.))


def implied_vol(value, forward, strike, maturity, option_type='call'):
    _inputs(forward, strike, maturity, 0., option_type)
    if not np.isfinite(value) or maturity <= 0:
        raise ValueError('finite price and positive maturity required')
    lo = max((forward-strike) * (1 if option_type == 'call' else -1), 0.)
    hi = forward if option_type == 'call' else strike
    tol = 1e-10 * max(forward, strike)
    if value < lo-tol or value >= hi:
        raise ValueError('option price outside intrinsic/upper bound')
    if value <= lo+tol:
        return 0.
    return float(brentq(lambda vol: float(price(forward,strike,maturity,vol,option_type))-value,
                        1e-10, 20., xtol=1e-12))
