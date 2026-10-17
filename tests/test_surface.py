import numpy as np
import pytest
from options_engine import surface,black

def test_ssvi_density_and_calendar_sufficient_conditions():
    for rho in (-.9,0,.9):
        eta=.99*np.sqrt(4/(1+abs(rho)))
        ks=np.linspace(-10,10,2001)
        previous=None
        for theta in np.geomspace(1e-5,10,30):
            w=surface.total_variance(ks,theta,rho,eta)
            assert np.min(surface.density_factor(ks,theta,rho,eta))>=-1e-9
            if previous is not None: assert np.all(w>=previous)
            previous=w

def test_ssvi_call_convexity_independent_price_check():
    k=np.linspace(40,250,2001); vol=np.sqrt(surface.total_variance(np.log(k/100),.1,-.6,.8)/.5)
    prices=black.price(100,k,.5,vol)
    assert np.max(np.diff(prices))<=0
    assert np.min(np.diff(prices,2))>=-1e-9

def test_theta_interpolation_is_monotone():
    s=dict(maturities=[.1,.3,.8],theta=[.02,.05,.1])
    vals=surface.theta_at(np.linspace(0,2,1000),s)
    assert vals[0]==0 and vals[-1]==.1
    assert np.min(np.diff(vals))>=0
