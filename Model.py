from numpy import sqrt, exp, log, pi, inf, logspace, zeros, log10, mod, floor
from scipy.special import erfc
from scipy.integrate import quad
from Utilities import findcLceta, ulambda_sq
import matplotlib.pyplot as plt
import sys
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
#==============================================#
#                   New Model
#==============================================#
####################################################################
def max_entrainement(l,kt,et,cL,cEta,nu,g,rhoc,sig):
	ulamsq=ulambda_sq(l,kt,et,cL,cEta,nu,pope_spec=1.01)
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
def Ent_Volume(zp,l,Reg,Bog,Weg,kt,et,cL,cEta,nu,g,circ_p,Vmax):
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
	ulam_z=sqrt(ulambda_sq(zp-l/2,kt,et,cL,cEta,nu,pope_spec=1.01)); wz=ulam_z/sqrt(2)
	Fr2=circ_p*wz/(l**2/4*g); fz=2.5*l/(zp-0.5*l)
	if Fr2*fz<0.4:
		F=0
	else:
		F=-0.43+1.2*Fr2*fz
	V_Ent=pi*l**3/4.0*F*B*W
	V_Ent=min(Vmax,V_Ent)
	return V_Ent
####################################################################
def Ent_Volume_debug(zp,l,Reg,Bog,Weg,kt,et,cL,cEta,nu,g,circ_p,Vmax):
	reason0=0
	if Reg<70:
		B=0
		reason0=1
	elif Reg<2580:
		B=-3.4e-3+5.1e-5*Reg+(-9.8e-9)*Reg**2
	else:
		B=0.062
	if Bog<1:
		W=0
		reason0=reason0+10
	elif Bog<50:
		W=1.5e-2+8.1e-5*Weg
	else:
		W=1
	# Rise velocity
	ulam_z=sqrt(ulambda_sq(zp-l/2,kt,et,cL,cEta,nu,pope_spec=1.01)); wz=ulam_z/sqrt(2)
	Fr2=circ_p*wz/(l**2/4*g); fz=2.5*l/(zp-0.5*l)
	if Fr2*fz<0.4:
		F=0
		reason0=reason0+100
	else:
		F=-0.43+1.2*Fr2*fz
	V_Ent=pi*l**3/4.0*F*B*W
	if V_Ent > Vmax:
		V_Ent=Vmax
		reason0=1000
	return V_Ent, reason0
####################################################################
def J_lambda(l,kt,et,cL,cEta,nu,g,rhoc,sig):
	#---- Eddy velocity ----#
	ulamsq=ulambda_sq(l,kt,et,cL,cEta,nu,pope_spec=1.01)
	ulam  =sqrt(ulamsq)
	#---- Circulation (parallel component) ----#
	circ_p=pi*pi/4*l*ulam
	#---- Eddy lifetime
	tau_vort=l**(2.0/3.0)/et**(1.0/3.0)
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
	PB=erfc(x)+2/sqrt(pi)*x*exp(-(x**2))
	#---- Maximum Entrainment Volume ----#
	Vmax=max_entrainement(l,kt,et,cL,cEta,nu,g,rhoc,sig)
	#---- Entrainment Volume ----#
	z_max=l/2+(2*sqrt(2)*g/(25*pi**2*et**(2/3)*l**(1/3)))**(-1.5)
		
	#---- Entrainment volume ----#
	V=quad(Ent_Volume, (l/2+1e-10), z_max, args=(l,Reg,Bog,Weg,kt,et,cL,cEta,nu,g,circ_p,Vmax), \
	       limit = 100, epsrel=1e-6, epsabs=1.0e-10)[0]

	#---- J_lambda ----#
	J_lam=n_lam*PB*V/tau_vort
	return J_lam
####################################################################
def J_lambda_plot(kt,et,cL,cEta,nu,g,rhoc,sig):
	fig=plt.figure(figsize=(8,8),dpi=200)
	plt.subplots_adjust(hspace=0.35,wspace=0.3)
	L = (kt**1.5)/et;
	fig.suptitle(r'$k_t={:.2f}m^2/s^2, \ \varepsilon={:.2f}m^2/s^3, \ L={:.2f}m$'.format(kt,et,L),fontsize=12,y=0.92)
	ifig=-2
	for l in [1e-2, 1e-1, 1]:
		ifig = ifig + 2
		#---- Eddy velocity ----#
		ulamsq=ulambda_sq(l,kt,et,cL,cEta,nu,pope_spec=1.01)
		ulam  =sqrt(ulamsq)
		#---- Circulation (parallel component) ----#
		circ_p=pi*pi/4*l*ulam
		#---- Eddy lifetime
		tau_vort=l**(2.0/3.0)/et**(1.0/3.0)
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
		PB=erfc(x)+2/sqrt(pi)*x*exp(-(x**2))
		#---- Maximum Entrainment Volume ----#
		Vmax=max_entrainement(l,kt,et,cL,cEta,nu,g,rhoc,sig)
		#---- Entrainment Volume ----#
		z_max=l/2+(2*sqrt(2)*g/(25*pi**2*et**(2/3)*l**(1/3)))**(-1.5)
			
		#---- Entrainment volume ----#
		V=quad(Ent_Volume, (l/2+1e-10), z_max, args=(l,Reg,Bog,Weg,kt,et,cL,cEta,nu,g,circ_p,Vmax), \
		       limit = 100, epsrel=1e-6, epsabs=1.0e-10)[0]

		# PLOT GENERATION
		zp_lst=logspace(log10(l/2+1e-15),log10(z_max),200)
		V_lst=zeros(200); Vmax_lst=zeros(200)+Vmax
		Re_cut=zeros(200); Bo_cut=zeros(200)
		Fr2_cut=zeros(200); Vmx_cut=zeros(200); 
		for i in range(200):
			V_lst[i],reason0=Ent_Volume_debug(zp_lst[i],l,Reg,Bog,Weg,kt,et,cL,cEta,nu,g,circ_p,Vmax)
			if mod(reason0,2) == 1:
				Re_cut[i]=1
			if mod(floor(reason0/10),2) == 1:
				Bo_cut[i]=1
			if mod(floor(reason0/100),2) == 1:
				Fr2_cut[i]=1
			if mod(floor(reason0/1000),2) == 1:
				Vmx_cut[i]=1
		ax=fig.add_subplot(321+ifig)
		plt.plot(zp_lst,V_lst,label=r'$\forall, \lambda={:.1f}mm$'.format(l*1000))
		plt.plot(zp_lst,Vmax_lst,label=r'$\forall_{max}$')
		plt.xlabel(r'$z^\prime [m]$',fontsize=12)
		plt.ylabel(r'$\forall [m^3]$',fontsize=12)
		plt.yscale('log'); plt.xscale('log'); plt.legend(fontsize=12); plt.grid()
		ax=fig.add_subplot(322+ifig)
		plt.plot(zp_lst,Re_cut, label=r'$Re \  cut \  off$')
		plt.plot(zp_lst,Bo_cut, label=r'$Bo \  cut \  off$')
		plt.plot(zp_lst,Fr2_cut,label=r'$Fr^{2} \  cut \  off$')
		plt.plot(zp_lst,Vmx_cut,label=r'$\forall_{max} \  cut \ off$')
		plt.xlabel(r'$z^\prime [m]$',fontsize=12); plt.ylabel('Reason of cutoff',fontsize=12)
		plt.xscale('log'); plt.legend(fontsize=12)

	return 0

####################################################################
def Jent_numerical_New(kt,et,nu,g,rhoc,sig):
	cL,cEta=findcLceta(kt,et,nu,mode=1)
	x1=1e-3;	x2=10.0;
	def intgrd(u,kt,et,cL,cEta,nu,g,rhoc,sig):
		return J_lambda(exp(u),kt,et,cL,cEta,nu,g,rhoc,sig)*exp(u)
	J=quad(intgrd,log(x1), log(x2), args=(kt,et,cL,cEta,nu,g,rhoc,sig),\
	      limit = 100, epsrel=1e-6, epsabs=1.0e-10)[0]
	return J
	# J_lambda_plot(kt,et,cL,cEta,nu,g,rhoc,sig)
	# exsit()
	# return 100
####################################################################

