from numpy import sqrt, exp, log, pi, logspace, zeros, log10, mod, floor, interp, linspace#, tan, inf
from scipy.special import erfc
from scipy.integrate import quad, trapezoid, quadrature
#from scipy.optimize import fsolve
from scipy.interpolate import interp2d
from Utilities import findcLceta, ulambda_sq
import matplotlib.pyplot as plt
from numba import jit #, float64
from scipy.io import loadmat
from Model import get_rise_speed, J_lambda, Ent_Volume_intgrand_jit_debug, J_lambda_prep
import time as t
# debug input
# kt/et[1]
kt = 1.238320701922703
et = 1.8408136363813399
nu= 1.0e-6;
g=9.81;
rhoc= 1000;
sig= 0.072;
Table=loadmat("UoIEntrainment.mat");
rrange=-1
##################################################
t1=t.time()
L=kt**1.5/et
cL,cEta=findcLceta(kt,et,nu,mode=1)
# x1=5e-3;	x2=10.0 # Lambda range
x1=sqrt(4*sig/rhoc/g); x2=sqrt(200*sig/rhoc/g); x3=L; x4=10;
# For speed get a table of ulambda_square
nlst=1000
lst=logspace(-4,2,nlst);	ul2_lst=zeros(nlst) #with dimension!
for i in range(nlst):
	ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
# Risisng speed list
rsp_lst=zeros(nlst)
for i in range(nlst):
	rsp_lst[i]=get_rise_speed(lst[i],1000,kt,et,nu,cL,cEta,method=2)
# fig0=plt.figure(figsize=(3,3),dpi=300)
# ax=fig0.add_subplot(111)
# ax.plot(lst,rsp_lst,color='red')
# ax.plot(lst,sqrt(ul2_lst),color='black')
# ax.set_xscale('log')
t2=t.time()
print('============= lam tab:{:.2e}s'.format(t2-t1))
def intgrd(u,lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange):
	return J_lambda(exp(u),lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange)*exp(u)
# J1=quad(intgrd,log(x1), log(x2), args=(lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange),limit = 100)[0]
# J2=quad(intgrd,log(x2), log(x3), args=(lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange),limit = 100)[0]
# J3=quad(intgrd,log(x3), log(x4), args=(lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange),limit = 100)[0]
# J1=quadrature(J_lambda,x1, x2,args=(lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange),vec_func=False)[0]
# J2=quadrature(J_lambda,x2, x3,args=(lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange),vec_func=False)[0]
# J3=quadrature(J_lambda,x3, x4,args=(lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange),vec_func=False)[0]
# J=J1+J2+J3
# J=quadrature(J_lambda,x1, x4,args=(lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange),vec_func=False)[0]
t3=t.time()
# print('============= Quad:{:.2e}s J = {:.4e}'.format(t3-t2,J))
# J_0=quadrature(intgrd,  log(x1), log(x2),
# 	             args=(lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange),
# 	             vec_func=False,maxiter=90)[0]
# J_1=quadrature(intgrd,  log(x2), log(x4),
# 	             args=(lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange),
# 	             vec_func=False,maxiter=91)[0]
# J=J_0+J_1
J=0
t4=t.time()
print('============= Quad (log int):{:.2e}s J = {:.4e}'.format(t4-t3,J))
##############################
# fig1=plt.figure(figsize=(6,6),dpi=300)
# ax=fig1.add_subplot(111)

# nlam=400
# lam=logspace(log10(x1),log10(x2),nlam); J_lst=zeros(nlam)
# for i in range(nlam):
# 	l=lam[i]
# 	J_lst[i]=J_lambda(l,lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange)
# J_trap1=trapezoid(J_lst,lam); #ax.plot(lam,J_lst)
# lam=logspace(log10(x2),log10(x3),nlam); J_lst=zeros(nlam)
# for i in range(nlam):
# 	l=lam[i]
# 	J_lst[i]=J_lambda(l,lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange)
# J_trap2=trapezoid(J_lst,lam); #ax.plot(lam,J_lst)
# lam=logspace(log10(x3),log10(x4),nlam); J_lst=zeros(nlam)
# for i in range(nlam):
# 	l=lam[i]
# 	J_lst[i]=J_lambda(l,lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange)
# J_trap3=trapezoid(J_lst,lam); #ax.plot(lam,J_lst)
# J_trap=J_trap1+J_trap2+J_trap3
# # print('J1={:.4e}, J_trapz1={:.4e}.'.format(J1,J_trap1))
# # print('J2={:.4e}, J_trapz2={:.4e}.'.format(J2,J_trap2))
# # print('J3={:.4e}, J_trapz3={:.4e}.'.format(J3,J_trap3))
# print('J_trapz ={:.4e}.'.format(J_trap ))
# # ax.set_xscale('log')
# # ax.set_yscale('log'); 
##############################
print('========= WARNING: DEBUG MODE!!! =========')
print('>> Plotting the J_lambda integrand')
# lamlist = logspace(log10(x1),log10(x4),500)
# 	# return J_lam, n_lam, PB, V_int, tau_vort
# Jlamint = zeros(lamlist.size);	nlamint = zeros(lamlist.size)
# PBint = zeros(lamlist.size);	Vint = zeros(lamlist.size)
# Tauint = zeros(lamlist.size)
# for i in range(lamlist.size):
# 	l = lamlist[i]
# 	Jlamint[i],nlamint[i],PBint[i],Vint[i],Tauint[i]= J_lambda_tmp(l,lst,ul2_lst,rsp_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange)
# fig = plt.figure(figsize=(5,4),dpi=300)
# ax=fig.add_subplot(111)
# ax.plot(lamlist,Jlamint,	label='Jlambda_int')
# # ax.plot(lamlist,Jlamint/Jlamint.max(),	label='Jlambda_int')
# # ax.plot(lamlist,nlamint/nlamint.max(),	label='n_lambda')
# # ax.plot(lamlist,PBint  /PBint.max(),	label='PB')
# # ax.plot(lamlist,Vint   /Vint.max(),		label='V_entrain')
# # ax.plot(lamlist,Tauint /Tauint.max(),	label='Ent_tau')
# ax.legend()
# ax.set_xlabel(r'$\lambda$');
# # ax.set_ylabel("J lambda integrand")
# ax.set_ylabel("J lambda components")
# ax.set_xscale('log'); ax.set_yscale('log')
# # ax.set_xlim([1e-3,20])
# ax.set_xlim([1e-1,2e-1])
# ax.set_ylim([1e-3,1])
# # ax.plot([x2,x2],[Jlamint.min(),Jlamint.max()])

fig = plt.figure(figsize=(6,5),dpi=300)
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)
ax3=fig.add_subplot(223)
ax4=fig.add_subplot(224)
z_st_crt_max=3
z_st_crt_min=2
Refitcoefs=Table['Refitcoefs'][0];	FrXcoefs=Table['FrXcoefs'][0]; 
Fr2_lst=Table['flxfr_data'][0,:]; zoa_lst=Table['z_a_data'][:,0]
F_tab=Table['F_lookuptable']
l_i=-1
# for l in linspace(9.2e-1,1.6e0,10):
nline=1
for l in linspace(4,4,nline):
	l_i = l_i + 1
	ulamsq,Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=\
	J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
	zmax=z_st_crt_max*l;	zmin=z_st_crt_min*l
	zlst=logspace(log10(zmin),log10(zmax),500)
	ent_v_out=zeros(500); Flst=zeros(500); Fr2lst=zeros(500); reason=zeros(500)
	tmp=J_lambda(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange)
	for i in range(500):
		logzp = log(zlst[i])
		ent_v_out[i],Flst[i],Fr2lst[i],reason[i]=\
		Ent_Volume_intgrand_jit_debug(logzp,l,kt,et,nu,cL,cEta,g,circ_p,
		                              Reg,Bog,Weg,Refitcoefs,FrXcoefs,
		                              Fr2_lst,zoa_lst,F_tab)
	ax1.plot(zlst/l,ent_v_out,	label=r'$\lambda={:.1e}$'.format(l),color=plt.cm.gist_rainbow(l_i/nline))
	ax2.plot(zlst/l,Flst,		label=r'$\lambda={:.1e}$'.format(l),color=plt.cm.gist_rainbow(l_i/nline))
	ax3.plot(zlst/l,Fr2lst,		label=r'$\lambda={:.1e}$'.format(l),color=plt.cm.gist_rainbow(l_i/nline))
	ax4.plot(zlst/l,reason,		label=r'$\lambda={:.1e}$'.format(l),color=plt.cm.gist_rainbow(l_i/nline))


ax1.legend(bbox_to_anchor=(-0.35,0)); ax1.set_yscale('log')
ax1.set_ylabel('Vint'); ax1.set_xlim([2.49,2.64])
ax2.set_ylabel('F'); ax2.set_xlim([2.49,2.64])
ax3.set_ylabel('Fr2'); ax3.set_xlim([2.49,2.64])
ax4.set_ylabel('Reason'); ax4.set_xlim([2.49,2.64])


#ax2.legend(); #ax1.set_yscale('log')
#ax3.legend(); #ax1.set_yscale('log')
#ax4.legend(); #ax1.set_yscale('log')
# NO CONCLUSION WHY THE JLAMBDA IS NOT SMOOTH AT LAMBDA AROUND 1e-1. 