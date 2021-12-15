from numpy import interp, sqrt, pi
from numba import jit
from Model import get_rise_speed, J_lambda_prep, Fr_contribution, Fr2_crit_getter, Re_contribution, We_contribution
# @jit(nopython=True, cache=True)
def Calc_Para_Func(zp,l,lst,ul2_lst,rhoc,sig,kt,et,nu,cL,cEta,g,Refitcoefs,FrXcoefs,Fr2_lst,zcoa_lst,F_tab,F_tab_NP,Fr2_crt_PolyExtra):
	Reg,Bog,Weg,circ_p,n_lam,x,tau_vort = J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
	zcoa=-1*zp/(l/2)
	b=Refitcoefs[1]+Refitcoefs[4]*zcoa
	a=Refitcoefs[3]
	B = Re_contribution(zcoa,Refitcoefs,Reg)
	W = We_contribution(Bog,Weg)
	wz = get_rise_speed(l,2*zp,kt,et,nu,cL,cEta,method=2)
	Fr2=circ_p*wz/(l**2/4*g)
	F = Fr_contribution(zcoa,Fr2,F_tab_NP,zcoa_lst,F_tab,Fr2_lst)
	Fr2_crit = Fr2_crit_getter(l,zp,FrXcoefs,Fr2_crt_PolyExtra)
	return B, Reg, -b/2/a,\
	       W, Weg, \
	       F, Fr2, Fr2_crit, tau_vort