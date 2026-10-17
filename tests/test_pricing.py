import numpy as np
import pytest
from scipy.integrate import quad
from options_engine import black,heston

P=dict(kappa=2.,theta=.09,sigma=.5,rho=-.6,v0=.09)

@pytest.mark.parametrize('kind',['call','put'])
@pytest.mark.parametrize('strike',[70.,100.,140.])
def test_black_inversion_and_greeks(kind,strike):
    v=black.price(100,strike,.7,.4,kind)
    assert black.implied_vol(v,100,strike,.7,kind)==pytest.approx(.4,abs=1e-10)
    g=black.greeks(100,strike,.7,.4,kind); h=.001
    assert g['delta']==pytest.approx((black.price(100+h,strike,.7,.4,kind)-black.price(100-h,strike,.7,.4,kind))/(2*h),abs=1e-7)
    assert g['gamma']==pytest.approx((black.price(100+h,strike,.7,.4,kind)-2*v+black.price(100-h,strike,.7,.4,kind))/h**2,abs=1e-7)

def test_black_known_atm_and_terminal():
    assert black.price(100,100,1,.2)==pytest.approx(7.9655674554,abs=1e-9)
    assert black.price(90,100,0,.2,'put')==10
    with pytest.raises(ValueError): black.implied_vol(105,100,100,1)

def test_heston_black_limit():
    p=dict(P,sigma=0.,v0=.09,theta=.09)
    np.testing.assert_allclose(heston.price(100,np.array([70,100,130]),.8,p),black.price(100,np.array([70,100,130]),.8,.3),atol=1e-10)

def test_heston_characteristic_moments_and_parity():
    assert heston.characteristic(0,.5,P)==pytest.approx(1)
    assert heston.characteristic(-1j,.5,P)==pytest.approx(1)
    c=heston.price(100,110,.5,P); p=heston.price(100,110,.5,P,'put')
    assert c-p==pytest.approx(-10,abs=1e-10)

def test_heston_independent_probability_integral():
    # P1/P2 inversion rather than the implementation's damped Lewis integral.
    f,k,t=100.,110.,.5
    def integrand(u,shift):
        cf=heston.characteristic(u-1j*shift,t,P)
        return float(np.real(np.exp(1j*u*np.log(f/k))*cf/(1j*u)))
    p1=.5+quad(lambda u:integrand(u,1),1e-9,400,epsabs=1e-9)[0]/np.pi
    p2=.5+quad(lambda u:integrand(u,0),1e-9,400,epsabs=1e-9)[0]/np.pi
    assert heston.price(f,k,t,P,nodes=256,upper=400)==pytest.approx(f*p1-k*p2,abs=1e-7)

def test_heston_convergence_and_vector_variance():
    k=np.linspace(75,130,9); p=dict(P,v0=np.linspace(.04,.16,9))
    a=heston.price(100,k,.1,p,nodes=128,upper=200)
    b=heston.price(100,k,.1,p,nodes=256,upper=400)
    np.testing.assert_allclose(a,b,atol=3e-5)
    assert np.all((a>=np.maximum(100-k,0))&(a<100))

def test_heston_moment_singular_algebraic_branch():
    p=dict(kappa=.15,theta=.09,sigma=3.,rho=.5,v0=.09)
    assert heston.characteristic(-1j,.2,p)==1
    assert heston.characteristic(0,.2,p)==1

def test_reject_nonfinite_parameters():
    with pytest.raises(ValueError):heston.price(100,100,1,dict(P,kappa=np.inf))
    with pytest.raises(ValueError):black.greeks(100,100,1,.2,discount=-1)

def test_adaptive_delta_against_price_gradient_and_short_expiry():
    d,error=heston.delta_adaptive(100,105,.2,P)
    assert error<1e-6
    assert d==pytest.approx(float(heston.delta(100,105,.2,P,nodes=256,upper=400)),abs=1e-6)
    p=dict(kappa=12.,theta=.169018769,sigma=2.9561296,rho=-.180576,v0=0.)
    d,error=heston.delta_adaptive(96.3609,100,.0006849315,p)
    assert error<1e-6 and -1e-7<=d<=1+1e-7
