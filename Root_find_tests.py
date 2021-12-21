from Model import int_seg_find, J_lambda_prep, get_rise_speed, Fr2_crit_getter, Re_contribution
from numpy import logspace, log10, load, sqrt, zeros
from Utilities import findcLceta, ulambda_sq
import matplotlib.pyplot as plt
kt = 1 #m2/s2
et = 0.000014 #m2/s3
nu = 1.0e-6
sig = 0.072
rhoc=1000
g = 9.81
cL,cEta=findcLceta(kt,et,nu,mode=1)
Table=load("Ent_table_org.npz")
Refitcoefs=Table['Refitcoefs'];	FrXcoefs=Table['FrXcoefs']
Fr2_lst=Table['flxfr_data']; zcoa_lst=Table['z_a_data']; F_tab=Table['F_lookuptable']
nlst = 400
x1=sqrt(4*sig/rhoc/g); x4=2
lst=logspace(log10(x1),log10(x4),nlst);	ul2_lst=zeros(nlst) #with dimension!
for i in range(nlst):
	ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
#! ---------------------------------------------------------!#
#!                  Froude segment find test                !#
#! ---------------------------------------------------------!#
# zlam_min = 2
# zlam_max = 5
# l = 1
# def_size = 500
# Fr2_crt_PolyExtra = False
# zp = logspace(log10(l*zlam_min),log10(l*zlam_max),def_size)
# Fr2 = zeros(def_size)
# Fr2_crt = zeros(def_size)
# Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
# num_seg, zp_seg = int_seg_find(l,zlam_min,zlam_max,kt,et,nu,cL,cEta,FrXcoefs,circ_p,g,Fr2_crt_PolyExtra)

# for i in range(def_size):
# 	wz = get_rise_speed(l,2*zp[i],kt,et,nu,cL,cEta,method=2)
# 	Fr2[i] = circ_p*wz/(l**2/4*g)
# 	Fr2_crt[i] = Fr2_crit_getter(l,zp[i],FrXcoefs,Fr2_crt_PolyExtra)

# fig=plt.figure(figsize=(6,4), dpi=200)
# ax = fig.add_subplot(111)
# ax.plot(zp,Fr2,color = 'red')
# ax.plot(zp,Fr2_crt,color = 'black')
# for i in range(num_seg):
# 	ax.plot([zp_seg[i],zp_seg[i]],[0,1],color = 'blue', linestyle = ':')

#! ---------------------------------------------------------!#
#!                 Reynolds segment find test               !#
#! ---------------------------------------------------------!#
colors=['red','green','blue']
fig=plt.figure(figsize=(7,3), dpi=200)
axRe = fig.add_subplot(121); axB = fig.add_subplot(122)
def_size = 500
zlam_min = 1e-10
zlam_max = 90
# llst = logspace(log10(x1),log10(10),def_size)
llst = [x1,5*x1,25*x1]
Reg = zeros(len(llst))
Reg_crt = zeros((len(llst),def_size))
B=zeros((len(llst),def_size))
# zlamlst = [zlam_min,0.5*(zlam_min+zlam_max),zlam_max]
for i in range(len(llst)):
	l = llst[i]
	zp = logspace(log10(l*zlam_min), log10(l*zlam_max), def_size)
	Reg[i],Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
	for j in range(len(zp)):
		zcoa = -1*zp[j]/(l/2)
		B[i,j] = Re_contribution(zcoa,Refitcoefs,Reg[i])
		zcoa_1=-2*zlam_min
		b=Refitcoefs[1]+Refitcoefs[4]*zcoa
		a=Refitcoefs[3]
		Reg_crt[i,j] = -b/2/a	
	axRe.plot([zp.min(), zp.max()], [Reg[i], Reg[i]], color=colors[i])
	axRe.plot(zp, Reg_crt[i, :], color=colors[i], linestyle=':')
	axRe.set_yscale("log"), axRe.set_xscale("log")
	axB.plot(zp/l,B[i,:],color = colors[i],label = r"$\lambda={:.4f}m$".format(llst[i]))

axB.plot(zp/l, zeros(def_size), color='black')
zp_lam_root1 = -1.411709818619592e+02 * -1 / 2
zp_lam_root2 = -13.502406389481719 * -1 / 2
axB.plot([zp_lam_root1],[0],marker='o',color='purple')
axB.plot([zp_lam_root2],[0],marker='o',color='purple')

# axB.plot([min(zp/l),max(zp/l)],[B[0,0],B[0,def_size-1]])

axB.legend(); axB.set_ylim([-0.06,0.06])  #axB.set_ylim([-0.2,0.1]) 

axRe.set_xlabel(r"$z'$ [m]");	axRe.set_ylabel(r"$\mathrm{Re}_{\Gamma,crt}$")
axB.set_xlabel (r"$z'/\lambda$ [-]");	axB.set_ylabel (r"$\mathrm{B}$")

plt.tight_layout()
plt.ioff()
plt.show(block=True)
