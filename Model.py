from numpy import sqrt, exp, log, pi, logspace, zeros, log10, mod, floor, interp, linspace, flip
from scipy.special import erfc
from scipy.integrate import quad, quadrature, trapezoid
#from scipy.optimize import fsolve
from scipy.interpolate import interp2d
from Utilities import findcLceta, ulambda_sq
import matplotlib.pyplot as plt
from numba import jit, float64
#import sys
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
#==============================================#
#                Previous Model
#==============================================#
def rms_velocity(k,kt,et,nu):
	cL=6.78
	u0 = (2/3*kt)**0.5
	L=kt**1.5/et; L11=0.43*L
	km=k*L
	fL = km/((cL+km**2)**0.5)
	uelli = sqrt(2)*((2*pi)**(1.0/3.0))*((et/k)**(1.0/3.0))
	p0=1.0
	uell = uelli*(fL**(1.0/3.0+p0))
	ell = 2*pi/k
	p0=2
	nell = 0.8413423/(ell**4)*(fL**(p0-1))
	return uell,uelli,fL,nell
####################################################################
def vortex_kernel(ell,kt,et):
	# PROPERTIES
	rho = 1000; rho_air = 1.24; nu = 1.0e-6; sigma = 0.072; g = 9.81
	k=2*pi/ell  #wave number

	uell,uelli,fL,nell=rms_velocity(k,kt,et,nu)

	# Variables (other forms of ell
	We = rho*ell*(uell**2)/sigma #Weber number
	Fr=uell/sqrt(g*ell)
	Fr23=Fr**(2.0/3.0)

	# Breakage probability
	x = sqrt(2/We)
	PB=erfc(x)+2/sqrt(pi)*x*exp(-(x**2))

	# Integral of the single vortex entrainment source
	#IQ = 1.615*log(2.336*Fr23); #delta=u_\ell^2/g
	IQ = 0.8075*(5.46*(Fr23**2)-1) #delta=u_s^2/g
	IQ = (IQ+abs(IQ))/2

	# Vortex kernel

	ze = 1.168*ell*Fr23

	#factor = ze*(uell.^3)*ell/g; #delta=u_\ell^2/g
	factor = ((uell*ell)**3)/ze/g #delta=u_s^2/g

	Q_integral = factor*IQ

	Jell = nell*PB*Q_integral

	J2 =  nell*factor*IQ

	J3 = nell*PB*factor
	return Jell
####################################################################
def Jent_numerical_prev(kt,et):
	# PROPERTIES
	x1=1.0e-5;	x2=10.0;
	def intgrd(u,kt,et):
		return vortex_kernel(exp(u),kt,et)*exp(u)
	J=quad(intgrd, log(x1), log(x2), args=(kt,et), limit = 100, epsrel=1e-6, epsabs=1.0e-10)[0]
	return J
####################################################################
def Ds_div_ze(z_ze):
	# scr = tan(18*pi/180)
	# # s_tilde=scr*1.168**3/2*z_ze**3
	# s_tilde=scr*0.824*z_ze**3
	# def x_tilde_eqn(x):
	# 	return s_tilde-x/(1+x**2)**3
	# 	# return x**(1/3)-s_tilde**(1/3)*(1+x**2)
	# x_tilde=fsolve(x_tilde_eqn,1e-5)[0]
	# if z_ze > 1:
	# 	out=0
	# elif z_ze >0.688:
	# 	out=2*z_ze*x_tilde*(0.824*scr*z_ze**3)
	# else:
	# 	out=1.615
	# return out,x_tilde
	if z_ze > 1:
		out=0
	else:
		out=1.615
	return out
####################################################################
def Ent_rate_prev(ell,zp,kt,et):
	# PROPERTIES
	rho = 1000; rho_air = 1.24; nu = 1.0e-6; sigma = 0.072; g = 9.81
	Sl0=1.9e-2
	k=2*pi/ell  #wave number
	uell,uelli,fL,nell=rms_velocity(k,kt,et,nu)
	# Variables (other forms of ell
	We = rho*ell*(uell**2)/sigma #Weber number
	Fr=uell/sqrt(g*ell)
	Fr2 =Fr**2
	Fr23=Fr**(2.0/3.0)
	ze = 1.168*ell*Fr23
	Ds=Ds_div_ze(zp/ze)*ze
	Q=Sl0*uell**3*ell**3/g/ze**2*(ze/zp)**3*Ds_div_ze(zp/ze)
	return Q, ze

#==============================================#
#                   New Model
#==============================================#
@jit(nopython=True, cache=True)
def ulam_nlam(logl,kt,et,nu,cL,cEta,lst,ul2_lst,output):
	l=exp(logl)
	ulamsq=interp(l,lst,ul2_lst)
	C=1.5;	p0=2.0;   be=5.2; k=2*pi/l
	L=kt**1.5/et; eta=(nu**3/et)**0.25
	fl  = (k*L/((k*L)**2+cL)**0.5)**(5.0/3.0+p0)
	feta= exp(-be*(((k*eta)**4+cEta**4)**0.25-cEta))
	Ek  = C*et**(2.0/3.0)*k**(-5.0/3.0)*fl*feta
	n_lam=24*Ek/(l**5*ulamsq)
	if output == 1: #numerator
		return sqrt(ulamsq)*n_lam * l
	elif output == 2: #denominator
		return n_lam * l
@jit(nopython=True, cache=True)
def ulam_nlam_nolog(l,kt,et,nu,cL,cEta,lst,ul2_lst,output):
	ulamsq=interp(l,lst,ul2_lst)
	C=1.5;	p0=2.0;   be=5.2; k=2*pi/l
	L=kt**1.5/et; eta=(nu**3/et)**0.25
	fl  = (k*L/((k*L)**2+cL)**0.5)**(5.0/3.0+p0)
	feta= exp(-be*(((k*eta)**4+cEta**4)**0.25-cEta))
	Ek  = C*et**(2.0/3.0)*k**(-5.0/3.0)*fl*feta
	n_lam=24*Ek/(l**5*ulamsq)
	if output == 1: #numerator
		return sqrt(ulamsq)*n_lam
	elif output == 2: #denominator
		return n_lam
####################################################################
@jit(nopython=True, cache=True)
def Ek_logint(logk,c1,c2,C,L,p0,beta,eta,et):
	k=exp(logk)
	return C*et**(2/3)*k**(-5.0/3.0)*(k*L/((k*L)**2+c1)**0.5)**(5.0/3.0+p0)*exp(-beta*(((k*eta)**4+c2**4)**0.25-c2))*k
@jit(nopython=True, cache=True)
def Ek_int(k,c1,c2,C,L,p0,beta,eta,et):
	return C*et**(2/3)*k**(-5.0/3.0)*(k*L/((k*L)**2+c1)**0.5)**(5.0/3.0+p0)*exp(-beta*(((k*eta)**4+c2**4)**0.25-c2))
####################################################################
# def get_rise_speed(l1,l2,kt,et,nu,cL,cEta,lst,ul2_lst,method):
def get_rise_speed(l1,l2,kt,et,nu,cL,cEta,method):
	if l1 > l2:
		print('==WARNING: Length scale l1 should be smaller than l2!!!===')
	if method == 1 : # Use second-order longitudinal structure function
		if l1 == l2:
			return 0
		# numerator   = quad(ulam_nlam, log(l1), log(l2), args=(kt,et,nu,cL,cEta,lst,ul2_lst,1),limit = 100)[0]
		# denominator = quad(ulam_nlam, log(l1), log(l2), args=(kt,et,nu,cL,cEta,lst,ul2_lst,2),limit = 100)[0]
		# numerator   = quadrature(ulam_nlam_nolog, l1, l2, args=(kt,et,nu,cL,cEta,lst,ul2_lst,1),vec_func=False,maxiter=100)[0]
		# denominator = quadrature(ulam_nlam_nolog, l1, l2, args=(kt,et,nu,cL,cEta,lst,ul2_lst,2),vec_func=False,maxiter=100)[0]
		# return numerator/denominator
		return -9999
	elif method == 2: # Use Energy spectrum
		L=kt**1.5/et;  eta=(nu**3/et)**0.25
		C=1.5;	p0=2.0;	beta=5.2

		# p1 = quadrature(Ek, 2*pi/l2, 2*pi/l1, args=(cL,cEta,C,L,p0,beta,eta),vec_func=False,maxiter=100)[0]
		# p1 = quad(Ek, 2*pi/l2, 2*pi/l1, args=(cL,cEta,C,L,p0,beta,eta),limit = 100)[0]
		p1 = quad(Ek_logint, log(2*pi/l2), log(2*pi/l1), args=(cL,cEta,C,L,p0,beta,eta,et),limit = 100)[0]

		p1 = (p1 *2.0/3.0)**(0.5)
		return p1
####################################################################
@jit(nopython=True, cache=True)
def max_entrainement(l,ulamsq,kt,et,cL,cEta,nu,g,rhoc,sig):
	# ulamsq=ulambda_sq(l,kt,et,cL,cEta,nu,pope_spec=1.01)
	# def E_ent_intgrand(d):
	# 	x=sqrt(2*sig/rhoc/d/ulamsq)
	# 	p0=rhoc*ulamsq/sig*x**5*exp(-x**2)/sqrt(pi)
	# 	# return p0*(sig*pi*d**2+(rhoc-rhod)/6*pi*d**3*g)
	# 	return p0*(2*sig/d/4*pi*l**3)
	# def avg_volume(d):
	# 	x=sqrt(2*sig/rhoc/d/ulamsq)
	# 	p0=rhoc*ulamsq/sig*x**5*exp(-x**2)/sqrt(pi)
	# 	# return p0*(sig*pi*d**2+(rhoc-rhod)/6*pi*d**3*g)
	# 	return p0*(1/6*pi*d**3)
	# E_ent_per_bubb=quad(E_ent_intgrand,0,inf)[0]
	# d_lst=logspace(-5,-1,200); E_lst=zeros(200)
	# for i in range(200):
	# 	E_lst[i]=avg_volume(d_lst[i])
	# plt.plot(d_lst,E_lst)
	# # Vmax=0.5*(rhoc/4*pi*l**3)*ulamsq*v0/E_ent_per_bubb
	#=====================================================
	ulam3=ulamsq**(1.5)
	Vmax=pi/6*(rhoc/8/sig*l**3)**(1.5)*ulam3
	return Vmax
####################################################################
#           Single depth version (z_c=3\lambda)
####################################################################
@jit(nopython=True, cache=True)
def Ent_Volume(zp,l,lst,ul2_lst,Reg,Bog,Weg,kt,et,cL,cEta,nu,g,circ_p,Vmax):
	if Reg<70:
		B=0
	elif Reg<2580:
		B=-3.4e-3+5.1e-5*Reg+(-9.8e-9)*Reg**2
	else:
		B=0.062
	if Bog<1:
		W=0
	elif Bog<50:
		W=1.5e-2+8.1e-5*Weg
	else:
		W=1
	# Rise velocity
	# ulam_z=sqrt(ulambda_sq(zp-l/2,kt,et,cL,cEta,nu,pope_spec=1.01)); wz=ulam_z/sqrt(2)
	ulam_z=sqrt(interp(zp-l/2,lst,ul2_lst)); wz=ulam_z/sqrt(2)
	Fr2=circ_p*wz/(l**2/4*g); fz=2.5*l/(zp-0.5*l)
	if Fr2*fz<0.4:
		F=0
	else:
		F=-0.43+1.2*Fr2*fz
	V_Ent=pi*l**3/4.0*F*B*W
	V_Ent=min(Vmax,V_Ent)
	return V_Ent
####################################################################
#           Multiple depth version (z_c=3\lambda)
####################################################################
# def Ent_Volume_intgrand_jit(logzp,l,kt,et,nu,cL,cEta,g,circ_p,Reg,Bog,Weg,Refitcoefs,FrXcoefs,Fr2_lst,zoa_lst,F_tab):
@jit(nopython=True, cache=True)
def Ent_Volume_intgrand_jit(logzp,l,zp_lst,wz_lst,g,circ_p,Reg,Bog,Weg,Refitcoefs,FrXcoefs,Fr2_lst,zoa_lst,F_tab):
	zp=exp(logzp)
	zcoa=-1*zp/(l/2)
	if zcoa > -4:
		zcoa=-4
	elif zcoa<-6:
		zcoa=-6
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
	# Prev Method
	# ulam_z=sqrt(interp(zp-l/2,lst,ul2_lst)); wz=ulam_z/sqrt(2)
	# New Method
	wz=interp(zp,zp_lst,wz_lst)
	# wz = get_rise_speed(l,2*zp,kt,et,nu,cL,cEta,method=2)
	Fr2=circ_p*wz/(l**2/4*g)
	# Critical Fr2
	zcoa_scl=(zcoa-FrXcoefs[7])/FrXcoefs[8]
	Fr2_crit=FrXcoefs[0]*zcoa_scl**6+FrXcoefs[1]*zcoa_scl**5+FrXcoefs[2]*zcoa_scl**4+\
	FrXcoefs[3]*zcoa_scl**3+FrXcoefs[4]*zcoa_scl**2+FrXcoefs[5]*zcoa_scl + FrXcoefs[6]
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
	V_Ent=pi*l**3/6.0*F*B*W
	# print("F,B,W:{:.3e}, {:.3e}, {:.3e}".format(F,B,W))
	if V_Ent < 0:
		print('WTF?????????')
	return V_Ent*zp
####################################################################
def Ent_Volume_intgrand_jit_dbg(logzp,l,zp_lst,wz_lst,g,circ_p,Reg,Bog,Weg,Refitcoefs,FrXcoefs,Fr2_lst,zoa_lst,F_tab):
	zp=exp(logzp)
	zcoa=-1*zp/(l/2)
	if zcoa > -4:
		zcoa=-4
	elif zcoa<-6:
		zcoa=-6
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
	# Prev Method
	# ulam_z=sqrt(interp(zp-l/2,lst,ul2_lst)); wz=ulam_z/sqrt(2)
	# New Method
	wz=interp(zp,zp_lst,wz_lst)
	# wz = get_rise_speed(l,2*zp,kt,et,nu,cL,cEta,method=2)
	Fr2=circ_p*wz/(l**2/4*g)
	# Critical Fr2
	zcoa_scl=(zcoa-FrXcoefs[7])/FrXcoefs[8]
	Fr2_crit=FrXcoefs[0]*zcoa_scl**6+FrXcoefs[1]*zcoa_scl**5+FrXcoefs[2]*zcoa_scl**4+\
	FrXcoefs[3]*zcoa_scl**3+FrXcoefs[4]*zcoa_scl**2+FrXcoefs[5]*zcoa_scl + FrXcoefs[6]
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
	V_Ent=pi*l**3/6.0*F*B*W
	# print("F,B,W:{:.3e}, {:.3e}, {:.3e}".format(F,B,W))
	if V_Ent < 0:
		print('WTF?????????')
	return V_Ent, F, B, W, Fr2, wz
####################################################################
@jit(nopython=True, cache=True)
def J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig):
	#---- Eddy velocity ----#
	ulamsq=interp(l,lst,ul2_lst)
	# ulamsq=ulambda_sq(l,kt,et,cL,cEta,nu,pope_spec=1.01)
	ulam  =sqrt(ulamsq)
	#---- Circulation (parallel component) ----#
	circ_p=pi*pi/4*l*ulam
	#---- Eddy lifetime
	# tau_vort=l**(2.0/3.0)/et**(1.0/3.0)
	tau_vort=ulamsq*0.5/et

	#---- MIT model input ----#
	Reg=circ_p/nu; 				Weg=circ_p**2*rhoc/(0.5*l*sig)
	Bog=g*(l/2)**2/(sig/rhoc)
	#---- Eddy number density ----#
	C=1.5;	p0=2.0;   be=5.2; k=2*pi/l
	L=kt**1.5/et; eta=(nu**3/et)**0.25
	fl  = (k*L/((k*L)**2+cL)**0.5)**(5.0/3.0+p0)
	feta= exp(-be*(((k*eta)**4+cEta**4)**0.25-cEta))
	Ek  = C*et**(2.0/3.0)*k**(-5.0/3.0)*fl*feta
	n_lam=24*Ek/(l**5*ulamsq)
	#---- Breakage probability ----#
	We = rhoc*l*ulamsq/sig #Weber number
	x = sqrt(2/We)
	return ulamsq,Reg,Bog,Weg,circ_p,n_lam,x,tau_vort
####################################################################
# @jit(nopython=True, cache=True)
def int_seg_find(nwz,l,zp_lst,wz_lst,zoa_lst,FrXcoefs,circ_p,g):
	Fr2_lst=circ_p*wz_lst/(l**2/4*g)
	Fr2_crit_lst=zeros(nwz)
	for i in range(nwz):
		zp=zp_lst[i]
		zcoa=-1*zp/(l/2)
		zcoa_scl=(zcoa-FrXcoefs[7])/FrXcoefs[8]
		# Critical Fr2
		Fr2_crit_lst[i]=FrXcoefs[0]*zcoa_scl**6+FrXcoefs[1]*zcoa_scl**5+FrXcoefs[2]*zcoa_scl**4+\
		FrXcoefs[3]*zcoa_scl**3+FrXcoefs[4]*zcoa_scl**2+FrXcoefs[5]*zcoa_scl + FrXcoefs[6]
	diff=Fr2_lst-Fr2_crit_lst
	num_seg=0; zp_seg=[]
	if ~(diff.any()>0): # integrating over all domain, Fr2>Fr2_crit
		num_seg = 1
		zp_seg=[zp_lst.min(), zp_lst.max()]
	else:
		if diff[0] > 0: #Rising edge
			num_seg = 1; zp_R_edge = zp_lst[0]
		i=0
		while i < nwz-1:
			if diff[i] > 0 and diff[i+1] <= 0: #Falling edge
				zp_F_edge=zp_lst[i]-diff[i]/((diff[i]-diff[i+1])/(zp_lst[i]-zp_lst[i+1]))
				zp_seg.append((zp_R_edge,zp_F_edge))
			elif diff[i] < 0 and diff[i+1] >= 0: #Rising edge
				zp_R_edge=zp_lst[i]-diff[i]/((diff[i]-diff[i+1])/(zp_lst[i]-zp_lst[i+1]))
				num_seg=num_seg+1
			i = i + 1
		if diff[nwz-1] > 0: #Rising edge
			zp_seg.append((zp_R_edge,zp_lst[nwz-1]))
	return num_seg, zp_seg
####################################################################
def J_lambda(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange):
	if rrange > 0:
		z_st_crt_max=2.5698*rrange	
		z_st_crt_min=2.5698		
	else:
		z_st_crt_max=3
		z_st_crt_min=2
	zmax=z_st_crt_max*l;	zmin=z_st_crt_min*l
	ulamsq,Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=\
	J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
	# Calculate rising speed table
	nwz=100
	zp_lst=linspace(zmin,zmax,nwz)
	wz_lst=zeros(nwz)
	for iz in range(nwz):
		wz_lst[iz] = get_rise_speed(l,2*zp_lst[iz],kt,et,nu,cL,cEta,method=2)
	# Figure out number of segments in integration
	Refitcoefs=Table['Refitcoefs'][0];	FrXcoefs=Table['FrXcoefs'][0]
	Fr2_lst=Table['flxfr_data'][0,:]; zoa_lst=Table['z_a_data'][:,0]; F_tab=Table['F_lookuptable']
	num_seg, zp_seg = int_seg_find(nwz,l,zp_lst,wz_lst,zoa_lst,FrXcoefs,circ_p,g)
	if num_seg == 0:
		return 0
	# print(num_seg,zp_seg[0])
	#---- Breakage probability ----#
	PB=erfc(x)+2/sqrt(pi)*x*exp(-(x**2))

	V_int = 0
	for iseg in range(num_seg):
		nwz=100
		zp_lst=linspace(zp_seg[iseg][0],zp_seg[iseg][1],nwz)
		wz_lst=zeros(nwz)
		for iz in range(nwz):
			wz_lst[iz] = get_rise_speed(l,2*zp_lst[iz],kt,et,nu,cL,cEta,method=2)
		# print(iseg,zp_seg[iseg][0]/l,zp_seg[iseg][1]/l)
		V_int=V_int+quad(Ent_Volume_intgrand_jit, log(zp_seg[iseg][0]), log(zp_seg[iseg][1]), \
		                 args=(l,zp_lst,wz_lst,g,circ_p,Reg,Bog,Weg,Refitcoefs,FrXcoefs,Fr2_lst,zoa_lst,F_tab), \
		                 limit = 100)[0]
	#---- J_lambda ----#
	# print("cL:{:.6e},cEta:{:.6e}".format(cL,cEta))
	# print("n_lam:{:.3e}".format(n_lam))
	# print("PB:{:.3e}".format(PB))
	# print("V_int:{:.3e}".format(V_int))
	# print("tau_vort:{:.3e}".format(tau_vort))
	# print("WZ test:{:.3e}".format(get_rise_speed(l,2.23*l,kt,et,nu,cL,cEta,method=2)))
	# llst = logspace(log10(l),log10(2.23*l),500)
	# klst = 2*pi/llst
	# Ek_out=zeros(500)
	# eta=(nu**3/et)**0.25; L=kt**1.5/et;
	# C=1.5;	p0=2.0;	beta=5.2
	# for i in range(500):
	# 	Ek_out[i]=Ek_int(klst[i],cL,cEta,C,L,p0,beta,eta,et)
	# Ekint_trapz=trapezoid(flip(Ek_out),flip(klst))
	# print("Ek_int={:.4e}, resulting wz:{:.4e}".format(Ekint_trapz,(Ekint_trapz *2.0/3.0)**(0.5)))
	# # V_debug, F, B, W, Fr2, wz=Ent_Volume_intgrand_jit_dbg(log(l*2.5),l,zp_lst,wz_lst,g,circ_p,Reg,Bog,Weg,Refitcoefs,FrXcoefs,Fr2_lst,zoa_lst,F_tab)
	# # print("V_debug, F, B, W:{:.4e}m^3, {:.4e}, {:.4e}, {:.4e}, {:.4e}, {:.4e}".format(V_debug, F, B, W, Fr2, wz))
	J_lam=n_lam*PB*V_int/tau_vort
	return J_lam
####################################################################
def Jent_numerical_New(kt,et,nu,g,rhoc,sig,Table,rrange,wmeth):
	if (wmeth!=2 and wmeth>0):
		print('Not working in this mode.')
		return -1

	cL,cEta=findcLceta(kt,et,nu,mode=1)
	L=kt**1.5/et
	x1=sqrt(4*sig/rhoc/g); x2=sqrt(200*sig/rhoc/g); x3=L; x4=50*L; # Lambda range
	# For speed get a table of ulambda_square
	nlst=1500
	# lst=logspace(-8,2,nlst);
	lst=logspace(log10(x1),log10(x4),nlst);	ul2_lst=zeros(nlst) #with dimension!
	for i in range(nlst):
		ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
	def intgrd(u,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange):
		return J_lambda(exp(u),lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange)*exp(u)
	J=quadrature(intgrd,  log(x1), log(x2),
	             args=(kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange),
	             vec_func=False,maxiter=51)[0] +\
	quadrature(intgrd,  log(x2), log(x4),
	            args=(kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange),
	            vec_func=False,maxiter=52)[0]
	return J