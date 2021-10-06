from numpy import sqrt, logspace, log10, zeros, linspace, pi, interp, exp, log
from Model import J_lambda_prep, Ent_Volume, Ent_Volume_Z, max_entrainement, Ent_rate_prev, get_rise_speed
from Utilities import findcLceta, ulambda_sq
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from scipy.io import loadmat
from matplotlib.ticker import ScalarFormatter
from scipy.integrate import quad
from numba import jit #, float64
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
fs=12

##########################
#   PHYSICAL PARAMETERS  #
##########################
kt=1.23; et=1.83; nu=1e-6; g=9.81; rhoc=1000; sig=0.072
cL,cEta=findcLceta(kt,et,nu,mode=1)
nlst=2000
lst=logspace(-8,3,nlst);	ul2_lst=zeros(nlst) #with dimension!
for i in range(nlst):
	ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
# Load the depth table:
Table=loadmat("UoIEntrainment.mat"); color_lst=['black','red','blue']
############################################################################
#=========================      Rising speed      =========================#
############################################################################
nz=200; nl=3
l_lst=[1e-2,1e-1,1]; wz_lst=zeros((nz,nl,4))
l1=1e-8; l2=logspace(-7,3,100)
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
numerator=zeros(100); denominator=zeros(100)
fig=plt.figure(figsize=(6,3),dpi=300); ax1=fig.add_subplot(121); ax2=fig.add_subplot(122)
for i in range(100):

	numerator[i]   = quad(ulam_nlam, log(l1), log(l2[i]), args=(kt,et,nu,cL,cEta,lst,ul2_lst,1),\
	                   limit = 200, epsrel=1e-6, epsabs=1.0e-10)[0]
	denominator[i] = quad(ulam_nlam, log(l1), log(l2[i]), args=(kt,et,nu,cL,cEta,lst,ul2_lst,2),\
	                   limit = 200, epsrel=1e-6, epsabs=1.0e-10)[0]

ax1.plot(l2,numerator,	label='numerator')
ax2.plot(l2,denominator, label='denominator')
ax1.legend(); ax2.legend()
ax1.set_xscale('log'); ax1.set_yscale('log')
ax2.set_xscale('log'); ax2.set_yscale('log')
# l2=5.1
# ldebug=logspace(log10(l1),log10(l2),200); deno=zeros(200)
# for i in range(200):
# 	l=ldebug[i]
# 	deno[i]=ulam_nlam(l,kt,et,nu,cL,cEta,lst,ul2_lst,output=2)
# 
# ax=fig.add_subplot(111)
# ax.plot(ldebug,deno); ax.set_xscale('log'); ax.set_yscale('log')
# denominator,abserr,infodict = quad(ulam_nlam, l1, l2, full_output=1,\
#                                        args=(kt,et,nu,cL,cEta,lst,ul2_lst,2),\
#                                        limit = 100, epsrel=1e-6, epsabs=1.0e-10)
# # ax.plot(infodict['alist'],infodict['blist'],linestyle='none',marker='0')