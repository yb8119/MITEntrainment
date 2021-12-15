from numpy import load, sqrt, linspace, zeros, log10, logspace, pi
from Model import int_seg_find, J_lambda_prep, max_lambda
from Utilities import findcLceta, ulambda_sq
from Plt_funcs import Calc_Para_Func
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

zlam_min = 30
zlam_max = 50
kt = 1.0 #m2/s2
et = 4.0 #m2/s3
nu = 1.0e-6
sig = 0.072
rhoc=1000
g = 9.81
Table=load("Ent_table_org.npz")
Refitcoefs=Table['Refitcoefs'];	FrXcoefs=Table['FrXcoefs']
Fr2_lst=Table['flxfr_data']; zcoa_lst=Table['z_a_data']; F_tab=Table['F_lookuptable']
cL,cEta=findcLceta(kt,et,nu,mode=1)
Fr2_crt_PolyExtra = False
x1=sqrt(4*sig/rhoc/g)
L=kt**1.5/et
x4=2
x4_alt = max_lambda(kt,et,nu,g,cL,cEta,zlam_min,zlam_max,FrXcoefs,Fr2_crt_PolyExtra)
nlst = 400
lst=logspace(log10(x1),log10(x4),nlst);	ul2_lst=zeros(nlst) #with dimension!
llst = linspace(x1,x4,nlst)
for i in range(nlst):
	ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
for il in range(len(llst)):
	l = llst[il]
	Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
	num_seg, zp_seg = int_seg_find(l,zlam_min,zlam_max,kt,et,nu,cL,cEta,FrXcoefs,circ_p,g,Fr2_crt_PolyExtra)
	if num_seg != 0:
		lam_act = l

# ********************* Entrainment source for single vortex for fixed lambda
#region
colors=["Black","Red","Green","Blue","Purple"]
def_size = 1000; 
pltlst=[]
fig=plt.figure(figsize=(6,4), dpi=200)
fig2=plt.figure(figsize=(8,3), dpi=200)
ax = fig.add_subplot(111)
ax1_2 = fig2.add_subplot(121)
ax2_2 = fig2.add_subplot(122)
# z_lam_lst=logspace(log10(2),log10(3000),def_size);
z_lam_lst=logspace(log10(zlam_min),log10(zlam_max),def_size);
lam_lst = [x1,lam_act,x4_alt]
Q_lst=zeros((len(lam_lst),def_size));
F_out=zeros((len(lam_lst),def_size));
Fr2_out=zeros((len(lam_lst),def_size));
Fr2c_out=zeros((len(lam_lst),def_size));
for il in range(len(lam_lst)):
	for i in range(def_size):
		(B, tmp, tmp,
		 W, tmp,
		 F, Fr2, Fr2_crit, tau_vort)=\
		Calc_Para_Func(z_lam_lst[i]*lam_lst[il],lam_lst[il],lst,ul2_lst,rhoc,
					   sig,kt,et,nu,cL,cEta,g,
					   Refitcoefs,FrXcoefs,Fr2_lst,zcoa_lst,F_tab,F_tab_NP=True,Fr2_crt_PolyExtra=False)
		Q_lst[il, i] = pi*lam_lst[il]**3/6.0*F*B*W/tau_vort
		F_out[il, i] = F
		Fr2_out[il, i] = Fr2
		Fr2c_out[il, i] = Fr2_crit
for il in range(len(lam_lst)):
	pl,=ax.plot(z_lam_lst,Q_lst[il,:],color=colors[il+1]); pltlst.append(pl)
	ax1_2.plot(z_lam_lst, F_out[il, :], color=colors[il+1]);  ax1_2.set_xscale('log')
	ax2_2.plot(z_lam_lst, Fr2_out[il, :], color=colors[il+1]);  ax2_2.set_xscale('log')
	ax2_2.plot(z_lam_lst, Fr2c_out[il, :], color=colors[il+1], linestyle='--')
ax2_2.set_ylim([0,3])
ax.legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$\lambda={}m, {}m, {}m$".format(lam_lst[0],lam_lst[1],lam_lst[2]))],
				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1),labelspacing=0.2)
ax.set_xscale('log')
ax.set_xlabel(r"$z'/\lambda$ [-]")
ax.set_ylabel(r"$\dot Q\ \mathrm{[m^3/s]}$")
#endregion
