from numpy import interp, sqrt, pi
from numba import jit
from Model import get_rise_speed
# @jit(nopython=True, cache=True)
def Calc_Para_Func(zp,l,lst,ul2_lst,rhoc,sig,kt,et,nu,cL,cEta,g,Refitcoefs,FrXcoefs,Fr2_lst,zoa_lst,F_tab,zl_min,zl_max):
######### Calculating Vortex properties #########
	#---- Eddy velocity ----#
	ulamsq=interp(l,lst,ul2_lst);	ulam=sqrt(ulamsq)
	#---- Circulation (parallel component) ----#
	circ_p=pi*pi/4*l*ulam
	#---- Eddy lifetime
	tau_vort=l**(2.0/3.0)/et**(1.0/3.0)
	#---- MIT model input ----#
	Reg=circ_p/nu;	Weg=circ_p**2*rhoc/(0.5*l*sig);	Bog=g*(l/2)**2/(sig/rhoc)

	zcoa=-1*zp/(l/2)
	if zcoa > -2*zl_min:
		zcoa=-2*zl_min
	elif zcoa<-2*zl_max:
		zcoa=-2*zl_max
	# ===== Reynolds number dependence =====
	b=Refitcoefs[1]+Refitcoefs[4]*zcoa
	a=Refitcoefs[3]
	if Reg < -b/2/a:
		B=Refitcoefs[0] + Refitcoefs[1]*Reg + Refitcoefs[2]*zcoa + \
		Refitcoefs[3]*Reg**2 + Refitcoefs[4]*Reg*zcoa
	else:
		B=Refitcoefs[0] + Refitcoefs[1]*(-b/2/a) + Refitcoefs[2]*zcoa + \
		Refitcoefs[3]*(-b/2/a)**2 + Refitcoefs[4]*(-b/2/a)*zcoa
	B=max(0.0,B)
	# ===== Weber number dependence =====
	if Bog<1:
		W=0
	elif Bog<50:
		W=1.5e-2+8.1e-5*Weg
	else:
		W=1
	# ===== Friude number dependence =====
	# Rise velocity
	wz = get_rise_speed(l,2*zp,kt,et,nu,cL,cEta,method=2)
	Fr2=circ_p*wz/(l**2/4*g)
	# Critical Fr2
	zcoa_scl=(zcoa-FrXcoefs[7])/FrXcoefs[8]
	# Linear extrapolation
	if(zcoa <= -6):
		zcoa_lim=(-6-FrXcoefs[7])/FrXcoefs[8]
	elif(zcoa >= -4):
		zcoa_lim=(-4-FrXcoefs[7])/FrXcoefs[8]
	if(zcoa<=-6 or zcoa>=-4):
		dzcoa = zcoa_scl-zcoa_lim
		Fr2_crt_lim = FrXcoefs[0]*zcoa_lim**6+FrXcoefs[1]*zcoa_lim**5+FrXcoefs[2]*zcoa_lim**4+\
					  FrXcoefs[3]*zcoa_lim**3+FrXcoefs[4]*zcoa_lim**2+FrXcoefs[5]*zcoa_lim + FrXcoefs[6]
		# >>>> Linear extrapolation
		Fr2_crit = Fr2_crt_lim+(FrXcoefs[0]*zcoa_lim**5*6+FrXcoefs[1]*zcoa_lim**4*5+FrXcoefs[2]*zcoa_lim**3*4+\
				   FrXcoefs[3]*zcoa_lim**2*3+FrXcoefs[4]*zcoa_lim*2+FrXcoefs[5])*dzcoa
		# >>>> Polynomial extrapolation
		# Fr2_crit=FrXcoefs[0]*zcoa_scl**6+FrXcoefs[1]*zcoa_scl**5+FrXcoefs[2]*zcoa_scl**4+\
		# FrXcoefs[3]*zcoa_scl**3+FrXcoefs[4]*zcoa_scl**2+FrXcoefs[5]*zcoa_scl + FrXcoefs[6]
	else:
		Fr2_crit=FrXcoefs[0]*zcoa_scl**6+FrXcoefs[1]*zcoa_scl**5+FrXcoefs[2]*zcoa_scl**4+\
		FrXcoefs[3]*zcoa_scl**3+FrXcoefs[4]*zcoa_scl**2+FrXcoefs[5]*zcoa_scl + FrXcoefs[6]
	# if abs(l-100) < 1e-6 :
		# print("zcoa_scl: {:.3e}, Fr2_crit: {:.3e}, zcoa: {:.3e}, zp: {:.3e}".format(zcoa_scl,Fr2_crit,zcoa,zp))
	if Fr2 < Fr2_crit:
		F=0
	else:
		# z_a_data goes from -6 to -4; flxfr_data goes from 0 to 4
		izcoa=-1
		if zcoa >= max(zoa_lst):
			izcoa=len(zoa_lst)-2; zcoa_lw=0; 
		elif zcoa <= min(zoa_lst):
			izcoa=0; 			zcoa_lw=1; 
		if izcoa == -1:
			for i in range(len(zoa_lst)-1):
				if zcoa>=zoa_lst[i] and zcoa<zoa_lst[i+1]:
					izcoa=i; zcoa_lw=(zoa_lst[i+1]-zcoa)/(zoa_lst[i+1]-zoa_lst[i])
		if izcoa == -1 or zcoa_lw>1 or zcoa_lw<0:
			print('Sthg is very wrong')
		F_lst=F_tab[izcoa,:]*zcoa_lw+F_tab[izcoa+1,:]*(1-zcoa_lw)
		F=interp(Fr2,Fr2_lst,F_lst)
		F=max(0.0,F)
	return B, Reg, -b/2/a,\
	       W, Weg, \
	       F, Fr2, Fr2_crit