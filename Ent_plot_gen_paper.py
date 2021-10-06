from numpy import logspace, log10, zeros, sqrt, pi
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Utilities import findcLceta, ulambda_sq
from Model import get_rise_speed
from Plt_funcs import Calc_Para_Func
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
fs=12
plt.rcParams.update({'font.size': fs})
def_size=1500
colors=["Black","Red","Green","Blue"]
LineSty=["-",(0,(5,3)),(0,(3,3,1,3))]
############# Physical/Flow field parameters #############
nu =1e-6; g=9.81; sig=0.072;rhoc=1000;nu=1e-6
# kt = 1.0 #m2/s2
# et = 4.0 #m2/s3
kt = 1.75 #m2/s2
et = 3.32 #m2/s3
cL,cEta=findcLceta(kt,et,nu,mode=1); L=kt**1.5/et
######## Get the basics for the MIT model ########
# Load the depth table:
Table=loadmat("UoIEntrainment.mat")
# Figure out number of segments in integration
Refitcoefs=Table['Refitcoefs'][0];	FrXcoefs=Table['FrXcoefs'][0]
Fr2_lst=Table['flxfr_data'][0,:]; zoa_lst=Table['z_a_data'][:,0]; F_tab=Table['F_lookuptable']
nlst=1500; x1=sqrt(4*sig/rhoc/g); x4=50*L
lst=logspace(log10(x1),log10(x4),nlst);	ul2_lst=zeros(nlst) #with dimension!
for i in range(nlst):
	ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
#====================== Figure 5: Vortex rising speeds ======================#
rsp_vs_lam=zeros((def_size,3))
fig=plt.figure(figsize=(6,2.5),dpi=300)
plt.subplots_adjust(hspace=0.2,wspace=0.2)
axlst=[]; pltlst=[]
ax=fig.add_subplot(121); axlst.append(ax); ax=fig.add_subplot(122); axlst.append(ax) 
#const depth
zp_lst = [0.1, 0.5, 2.5]
for idepth in range(3):
	lam_range=logspace(-3,log10(2*zp_lst[idepth]-1e-10),def_size)
	for i in range(def_size):
		rsp_vs_lam[i,idepth] = get_rise_speed(lam_range[i],2.0*zp_lst[idepth],kt,et,nu,cL,cEta,method=2)
	pl,=axlst[0].plot(lam_range,rsp_vs_lam[:,idepth],color=colors[idepth])
	pltlst.append(pl)
axlst[0].set_xlabel(r"$\lambda \ \mathrm{[m]}$"); axlst[0].set_ylabel(r"$w(z',\lambda) \ \mathrm{[m/}s]$")
#const lambda
lam_lst = [0.2, 1, 5]
for ilambda in range(3):
	zp_range=logspace(log10(lam_lst[ilambda]/2+1e-10),1,def_size)
	for i in range(def_size):
		rsp_vs_lam[i,ilambda] = get_rise_speed(lam_lst[ilambda],2.0*zp_range[i],kt,et,nu,cL,cEta,method=2)
	axlst[1].plot(zp_range,rsp_vs_lam[:,ilambda],color=colors[ilambda])
	pltlst.append(pl)
axlst[1].set_xlabel(r"$z' \ \mathrm{[m]}$")
axlst[0].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$z'={}, {}, {}$m".format(zp_lst[0],zp_lst[1],zp_lst[2]))],
                handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1))
axlst[1].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$\lambda={}, {}, {}$m".format(lam_lst[0],lam_lst[1],lam_lst[2]))],
                handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1))
axlst[0].set_xlim([1e-3,10]); axlst[1].set_xlim([0,10]); axlst[0].set_xscale("log")
axlst[0].set_ylim([0,1.1]); axlst[1].set_ylim([0,1.1])
#====================== Figure 6&7: parameter values and their function values ======================#
# zp_lst = [0.2, 1, 5]; lam_lst = [0.2, 1, 5]
# #Re_Gamma
# B_list=zeros((def_size,3)); Recrt_list=zeros((def_size,3)); Re_list=zeros(def_size)
# #We_Gamma
# W_list=zeros(def_size); We_list=zeros(def_size)
# #Fr^2_Xi
# F_list=zeros((def_size,6)); Fr2_list=zeros((def_size,6)); Fr2_crt_list=zeros((def_size,6));

# for idepth in range(3):
# 	lam_range=logspace(-4,log10(2*zp_lst[idepth]),def_size)
# 	for i in range(def_size):
# 		(B_list[i,idepth], Re_list[i], Recrt_list[i,idepth],
# 		W_list[i], We_list[i],
# 		F_list[i,idepth], Fr2_list[i,idepth], Fr2_crt_list[i,idepth])=\
# 		Calc_Para_Func(zp_lst[idepth],lam_range[i],lst,ul2_lst,rhoc,
# 		                 sig,kt,et,nu,cL,cEta,g,
# 		                 Refitcoefs,FrXcoefs,Fr2_lst,zoa_lst,F_tab)
# lam_lst = [0.2, 1, 5]
# for ilam in range(3):
# 	zp_range=logspace(log10(lam_lst[ilam]/2+1e-10),1,def_size)
# 	for i in range(def_size):
# 		(tmp, tmp, tmp,
# 		tmp,tmp,
# 		F_list[i,ilam+3], Fr2_list[i,ilam+3], Fr2_crt_list[i,ilam+3])=\
# 		Calc_Para_Func(zp_range[i],lam_lst[ilam],lst,ul2_lst,rhoc,
# 		                 sig,kt,et,nu,cL,cEta,g,
# 		                 Refitcoefs,FrXcoefs,Fr2_lst,zoa_lst,F_tab)
# #$$$$$$$$$$$$$$$$$$ Figure 6: Parameters $$$$$$$$$$$$$$$$$$
# fig=plt.figure(figsize=(6,6), dpi=200)
# plt.subplots_adjust(wspace=0.4, hspace=0.8)
# axlst=[]; pltlst=[]
# ax=fig.add_subplot(221); axlst.append(ax); ax=fig.add_subplot(222); axlst.append(ax)
# ax=fig.add_subplot(223); axlst.append(ax); ax=fig.add_subplot(224); axlst.append(ax)

# # Re_gamma
# axlst[0].plot(lam_range,Re_list,color="Black")
# for idepth in range(3):
# 	lam_range=logspace(-4,log10(2*zp_lst[idepth]),def_size)
# 	pl,=axlst[0].plot(lam_range,Recrt_list[:,idepth],color=colors[idepth+1],linestyle="--"); pltlst.append(pl)
# axlst[0].set_xscale("log"); axlst[0].set_yscale("log"); axlst[0].set_xlim([1e-4,2*zp_lst[2]])#; axlst[0].set_ylim([10,10000])
# axlst[0].set_xlabel(r"$\lambda$ [m]"); axlst[0].set_ylabel(r"$\mathrm{Re}_\Gamma$ [-]")
# axlst[0].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$\mathrm{Re}_{\Gamma,crt}$"+r"$, z_c={}, {}, {}$m".format(zp_lst[0],zp_lst[1],zp_lst[2]))],
#                 handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1))

# # We_gamma
# axlst[1].plot(lam_range,We_list,color="Black"); pltlst=[]
# pl,=axlst[1].plot([5.42e-3,5.42e-3],[min(We_list),max(We_list)],color="black",linestyle="--");  pltlst.append(pl)
# pl,=axlst[1].plot([38.3e-3,38.3e-3],[min(We_list),max(We_list)],color="black",linestyle=":"); pltlst.append(pl)
# axlst[1].set_xscale("log"); axlst[1].set_yscale("log"); 
# axlst[1].set_xlim([1e-4,2*zp_lst[2]])
# axlst[1].set_xlabel(r"$\lambda$ [m]"); axlst[1].set_ylabel(r"$\mathrm{We}_\Gamma$ [-]")
# axlst[1].legend([(pltlst[0],pltlst[1])],[(r"$\mathrm{Bo}_\Gamma = 1, 50$")],
#                 handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1),labelspacing=0.2)
# # Fr_gamma
# #Const depth
# pltlst=[]
# for idepth in range(3):
# 	lam_range=logspace(-4,log10(2*zp_lst[idepth]),def_size)
# 	# for i in range(def_size-1):
# 	# 	if Fr2_list[i,idepth]>=Fr2_crt_list[i,idepth] and Fr2_list[i+1,idepth]<=Fr2_crt_list[i+1,idepth]:
# 	# 		axlst[2].plot(lam_range[i],Fr2_list[i,idepth],color=colors[idepth+1],linestyle="none",marker="o")
# 	axlst[2].fill_between(lam_range, Fr2_list[:,idepth], 
# 	                      where=((lam_range > zp_lst[idepth]/3)*(lam_range < zp_lst[idepth]/2)), color=colors[idepth+1], alpha=0.3)
# 	pl,=axlst[2].plot(lam_range,Fr2_list[:,idepth],color=colors[idepth+1]); pltlst.append(pl)
# 	pl,=axlst[2].plot(lam_range,Fr2_crt_list[:,idepth],color=colors[idepth+1],linestyle=(0,(1,0.8))); pltlst.append(pl)
# axlst[2].set_xscale("log"); axlst[2].set_yscale("log"); 
# axlst[2].set_xlim([1e-2,2*zp_lst[2]]); axlst[2].set_ylim([1e-4,100])
# axlst[2].set_xlabel(r"$\lambda$ [m]"); axlst[2].set_ylabel(r"$\mathrm{Fr}^2_\Xi$ [-]")
# axlst[2].legend([(pltlst[0],pltlst[2],pltlst[4]),(pltlst[1],pltlst[3],pltlst[5])],\
#                 [(r"$\mathrm{Fr}^2_\Xi,$"+r"$ z_c={}, {}, {}$m".format(zp_lst[0],zp_lst[1],zp_lst[2])),(r"$\mathrm{Fr}^2_{\Xi,crt},$"+r"$ z_c={}, {}, {}$m".format(zp_lst[0],zp_lst[1],zp_lst[2]))],
#                 handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.25),labelspacing=0.2)
# #Const Vortex Size
# pltlst=[]
# for ilam in range(3):
# 	i = ilam+3
# 	zp_range=logspace(log10(lam_lst[ilam]/2+1e-10),1,def_size)
# 	axlst[3].fill_between(zp_range, Fr2_list[:,i], 
# 	                      where=((zp_range > lam_lst[ilam]*2)*(zp_range < lam_lst[ilam]*3)), color=colors[ilam+1], alpha=0.3)
# 	pl,=axlst[3].plot(zp_range,Fr2_list[:,i],color=colors[ilam+1]); pltlst.append(pl)
# 	pl,=axlst[3].plot(zp_range,Fr2_crt_list[:,i],color=colors[ilam+1],linestyle=(0,(1,0.5))); pltlst.append(pl)
# axlst[3].set_xscale("log"); axlst[3].set_yscale("log"); 
# # axlst[3].set_yscale("log"); 
# axlst[3].set_xlim([1e-1,10]); axlst[3].set_ylim([1e-3,10])
# axlst[3].set_xlabel(r"$z_c$ [m]"); axlst[3].set_ylabel(r"$\mathrm{Fr}^2_\Xi$ [-]")
# axlst[3].legend([(pltlst[0],pltlst[2],pltlst[4]),(pltlst[1],pltlst[3],pltlst[5])],\
#                 [(r"$\mathrm{Fr}^2_\Xi,$"+r"$ \lambda={}, {}, {}$m".format(lam_lst[0],lam_lst[1],lam_lst[2])),(r"$\mathrm{Fr}^2_{\Xi,crt},$"+r"$ \lambda={}, {}, {}$m".format(lam_lst[0],lam_lst[1],lam_lst[2]))],
#                 handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.25),labelspacing=0.2)

# #$$$$$$$$$$$$$$$$$$ Figure 7: Function values $$$$$$$$$$$$$$$$$$
# fig=plt.figure(figsize=(6,5), dpi=200)
# plt.subplots_adjust(wspace=0.4, hspace=0.6)
# axlst=[]; pltlst=[]
# ax=fig.add_subplot(221); axlst.append(ax); ax=fig.add_subplot(222); axlst.append(ax)
# ax=fig.add_subplot(223); axlst.append(ax); ax=fig.add_subplot(224); axlst.append(ax)

# # B
# for idepth in range(3):
# 	lam_range=logspace(-4,log10(2*zp_lst[idepth]),def_size)
# 	pl,=axlst[0].plot(lam_range,B_list[:,idepth],color=colors[idepth+1]); pltlst.append(pl)
# 	axlst[0].fill_between(lam_range, B_list[:,idepth], 
# 	                      where=((lam_range > zp_lst[idepth]/3)*(lam_range < zp_lst[idepth]/2)), color=colors[idepth+1], alpha=0.3)
# axlst[0].set_xscale("log"); axlst[0].set_yscale("log"); axlst[0].set_xlim([1e-3,2*zp_lst[2]]); axlst[0].set_ylim([1e-2,1e-1])
# axlst[0].set_xlabel(r"$\lambda$ [m]"); axlst[0].set_ylabel(r"$B\left(\mathrm{Re}_\Gamma,\ z_c^*\right)$ [-]")
# axlst[0].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$z_c={}, {}, {}$m".format(zp_lst[0],zp_lst[1],zp_lst[2]))],
#                 handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1))

# # We_gamma
# pltlst=[]
# axlst[1].plot(lam_range,W_list,color="Black")
# pl,=axlst[1].plot([5.42e-3,5.42e-3],[min(W_list),max(W_list)],color="black",linestyle="--"); pltlst.append(pl)
# pl,=axlst[1].plot([38.3e-3,38.3e-3],[min(W_list),max(W_list)],color="black",linestyle=":");  pltlst.append(pl)
# axlst[1].set_xscale("log")
# axlst[1].set_xlim([1e-4,2*zp_lst[2]])
# axlst[1].set_xlabel(r"$\lambda$ [m]"); axlst[1].set_ylabel(r"$W\left(\mathrm{We}_\Gamma\right)$ [-]")
# axlst[1].legend([(pltlst[0],pltlst[1])],[(r"$\mathrm{Bo}_\Gamma = 1, 50$")],
#                 handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1),labelspacing=0.2)

# #Const depth
# pltlst=[]
# for idepth in range(3):
# 	lam_range=logspace(-4,log10(2*zp_lst[idepth]),def_size)
# 	pl,=axlst[2].plot(lam_range,F_list[:,idepth],color=colors[idepth+1]); pltlst.append(pl)
# 	axlst[2].fill_between(lam_range, 0,100,
# 	                      where=((lam_range > zp_lst[idepth]/3)*(lam_range < zp_lst[idepth]/2)), color=colors[idepth+1], alpha=0.3)
# axlst[2].set_xscale("log"); axlst[2].set_yscale("log"); 
# axlst[2].set_xlim([1e-2,2*zp_lst[2]]); axlst[2].set_ylim([1e-4,10])
# axlst[2].set_xlabel(r"$\lambda$ [m]"); axlst[2].set_ylabel(r"$F\left(\mathrm{Fr}^2_\Xi,z_c^*\right)$ [-]")
# axlst[2].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$z_c={}, {}, {}$m".format(zp_lst[0],zp_lst[1],zp_lst[2]))],
#                 handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1),labelspacing=0.2)

# #Const vortex size
# pltlst=[]
# for ilam in range(3):
# 	i = ilam+3; zp_range=logspace(log10(lam_lst[ilam]/2+1e-10),1,def_size)
# 	pl,=axlst[3].plot(zp_range,F_list[:,i],color=colors[ilam+1]); pltlst.append(pl)
# 	axlst[3].fill_between(zp_range,F_list[:,i],
# 	                      where=((zp_range > lam_lst[ilam]*2)*(zp_range < lam_lst[ilam]*3)), color=colors[ilam+1], alpha=0.3)
# axlst[3].set_xscale("log"); axlst[3].set_yscale("log"); 
# # axlst[3].set_yscale("log"); 
# axlst[3].set_xlim([1e-1,10]); axlst[3].set_ylim([1e-3,10])
# axlst[3].set_xlabel(r"$z_c$ [m]"); axlst[3].set_ylabel(r"$F\left(\mathrm{Fr}^2_\Xi,z_c^*\right)$ [-]")
# axlst[3].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$\lambda={}, {}, {}$m".format(lam_lst[0],lam_lst[1],lam_lst[2]))],
#                 handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1),labelspacing=0.2)

# #$$$$$$$$$$$$$$$$$$ Figure 8: Entrainment source for single vortex $$$$$$$$$$$$$$$$$$
# fig=plt.figure(figsize=(6,4), dpi=200)
# plt.subplots_adjust(wspace=0.4, hspace=0.3)
# zlam_rlst = [2,2.5,3]
# axlst=[]; pltlst=[]
# ax=fig.add_subplot(221); axlst.append(ax); ax=fig.add_subplot(222); axlst.append(ax)
# ax=fig.add_subplot(223); axlst.append(ax); ax=fig.add_subplot(224); axlst.append(ax)

# axlst[0].set_xscale("log"); axlst[0].set_yscale("log"); axlst[0].set_xlim([1e-2,20])
# axlst[1].set_xscale("log"); axlst[1].set_yscale("log"); axlst[1].set_xlim([1e-2,20])
# axlst[2].set_xscale("log"); axlst[2].set_yscale("log"); axlst[2].set_xlim([1e-2,20])
# axlst[3].set_xscale("log"); axlst[3].set_yscale("log"); axlst[3].set_xlim([1e-2,20])

# axlst[0].set_ylabel(r"$\forall\ \mathrm{[m^3]}$")
# axlst[1].set_ylabel(r"$\forall/V_\lambda\ \mathrm{[-]}$")
# axlst[2].set_ylabel(r"$\dot Q\ \mathrm{[m^3/s]}$")
# axlst[3].set_ylabel(r"$\dot Q/V_\lambda\ \mathrm{[1/s]}$")
# axlst[2].set_xlabel(r"$z'$ [m]");	axlst[3].set_xlabel(r"$z'$ [m]")
# depth_lst=logspace(-3,log10(100),def_size);
# A_lst=zeros((3,def_size)); 	Q_lst=zeros((3,def_size));
# Ast_lst=zeros((3,def_size)); 	Qst_lst=zeros((3,def_size));

# for ir in range(3):
# 	lam_lst=depth_lst/zlam_rlst[ir]
# 	for i in range(def_size):
# 		(B, tmp, tmp,
# 		 W, tmp,
# 		 F, tmp, tmp)=\
# 		Calc_Para_Func(depth_lst[i],lam_lst[i],lst,ul2_lst,rhoc,
# 		               sig,kt,et,nu,cL,cEta,g,
# 		               Refitcoefs,FrXcoefs,Fr2_lst,zoa_lst,F_tab)
# 		A_lst[ir,i]   = pi*lam_lst[i]**3/6.0*F*B*W
# 		Ast_lst[ir,i] = F*B*W
# 		tau_vort=lam_lst[i]**(2.0/3.0)/et**(1.0/3.0)
# 		Q_lst[ir,i]   = A_lst[ir,i]/tau_vort
# 		Qst_lst[ir,i] = Ast_lst[ir,i]/tau_vort
# 	pl,=axlst[0].plot(depth_lst,A_lst[ir,:],color=colors[ir+1]); pltlst.append(pl)
# 	axlst[1].plot(depth_lst,Ast_lst[ir,:],color=colors[ir+1])
# 	axlst[2].plot(depth_lst,Q_lst[ir,:],color=colors[ir+1])
# 	axlst[3].plot(depth_lst,Qst_lst[ir,:],color=colors[ir+1])

# axlst[0].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$z'/\lambda={}, {}, {}$".format(zlam_rlst[0],zlam_rlst[1],zlam_rlst[2]))],
#                 handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(1.15,1.2),labelspacing=0.2)