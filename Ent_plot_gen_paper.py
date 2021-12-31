from numpy import array, logspace, log10, zeros, sqrt, pi, linspace, meshgrid, savez, concatenate
import numpy as np
from scipy.io import loadmat
from Utilities import findcLceta, ulambda_sq
from Model import Fr2_crit_getter, F_func_table_ext, J_lambda_prep, get_rise_speed
from Plt_funcs import Calc_Para_Func, myax
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import matplotlib.colors as matcolors
import matplotlib.patches as patches

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
fs=12
plt.rcParams.update({'font.size': fs})
def_size=1000
colors=["Red","Green","Blue","Purple"]
LineSty=["-",(0,(5,3)),(0,(3,3,1,3))]
Re_gamma_crt_str=r'$\mathrm{Re}_{\Gamma,crt}$'
Re_gamma_str=r'$\mathrm{Re}_{\Gamma}$'
Fr2_xi_str=r"$\mathrm{Fr}^2_\Xi$"
Fr2_xi_crt_str=r"Fr$^2_{\Xi,crt}$"
B_str=r"B$\left(\mathrm{Re}_\Gamma,\ z'/\lambda\right)$"
F_str=r"F$\left(\mathrm{Fr}^2_\Xi, \, z_c^*\right)$"
V_str=r"$\forall$"
Q_nlam_str=r"$\mathrm{Q}\cdot n_{\lambda}$"
Q_str=r"Q"
zlam_str=r"$z'/\lambda$ [-]"
# ****************** Physical/Flow field parameters ******************#
#region
nu =1e-6; g=9.81; sig=0.072;rhoc=1000;nu=1e-6
kt = 1.0 #m2/s2 # kt = 1.75 #m2/s2
et = 4.0 #m2/s3 # et = 3.32 #m2/s3
zl_min=1;	zl_max=6.751203194740859
cL,cEta=findcLceta(kt,et,nu,mode=1); L=kt**1.5/et
######## Get the basics for the MIT model ########
# Load the depth table:
Table=loadmat("UoIEntrainment.mat")
Refitcoefs=Table['Refitcoefs'][0];	FrXcoefs=Table['FrXcoefs'][0]
Fr2_lst=Table['flxfr_data'][0,:]; zcoa_lst=Table['z_a_data'][:,0]; F_tab=Table['F_lookuptable']
zp_lst = [0.2, 1, 5]
lam_lst = [0.2, 1, 5]
F_tab_NP = True
Fr2_crt_PolyExtra = False
nlst=1500; #x1=sqrt(4*sig/rhoc/g); x4=10*L
lst=logspace(-5,2,nlst);	ul2_lst=zeros(nlst) #with dimension!'
for i in range(nlst):
	ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
#endregion
# ********************* Froude number function (extended range) 
#region
# # fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(6,6), dpi=200)
# # Make data.
# X = zcoa_lst
# Y = Fr2_lst
# Z = F_tab.transpose()
# # Linear extrapolation
# zoa_ext= concatenate((linspace(-2*zl_max, -2*3-1e-2, 100),linspace(-2*3, -2*zl_min, 100)))
# # zoa_ext	= linspace(-2*zl_max, -2*zl_min, 2000)
# Fr2_ext	= concatenate((linspace(0, 4, 50),linspace(4+1e-2, 8, 100)))
# Z_ext_LN = zeros((Fr2_ext.size,zoa_ext.size))
# # Z_ext_NP = zeros((Fr2_ext.size,zoa_ext.size))
# Z_ext_LN = F_func_table_ext(Fr2_lst,Fr2_ext,zcoa_lst,zoa_ext,F_tab,"Nearest Point",np.array([1,5,3])).transpose()
# # Z_ext_NP = F_func_table_ext(Fr2_lst,Fr2_ext,zcoa_lst,zoa_ext,F_tab,method="Nearest Point").transpose()

# # f = interp2d(X, Y, Z, kind = 'cubic')
# # f = RectBivariateSpline(X, Y, F_tab, bbox=[min(X_ext)])
# # for i in range(X_ext.size):
# # 	for j in range(Y_ext.size):
# # 		Z_ext[j,i] = f(X_ext[i],Y_ext[j])
# X, Y = meshgrid(X, Y)
# X_ext, Y_ext = meshgrid(zoa_ext, Fr2_ext)
# zcoa_frc = linspace(-2*zl_max, -2*zl_min, 250); Frc2=zeros(250)

# for i in range(250):
# 	Frc2[i] = Fr2_crit_getter(1,zcoa_frc[i]*-0.5,FrXcoefs,Fr2_crt_PolyExtra)
# def plot_F(X,Y,Z,zcoa_frc,Frc2):
# 	vmin =max(1e-10,Z.min()); vmax = Z.max()
# 	fig, ax = plt.subplots(figsize=(5,4), dpi=200)
# 	rect = patches.Rectangle((-6,0),2,4, linewidth=1, edgecolor='black', facecolor='none', linestyle=':')
# 	cb_range=linspace(vmin,vmax,101)
# 	# cb_range=logspace(log10(vmin),log10(vmax),101)
# 	cs = ax.contourf(X, Y, Z, 
# 				   cmap=cm.gist_rainbow, antialiased=True, levels=cb_range, 
# 				   #norm=matcolors.LogNorm(vmin=vmin, vmax=vmax),
# 				   vmin=vmin,vmax=vmax, extend="both")
# 	# ax.add_patch(rect)
# 	pl,= ax.plot(zcoa_frc,Frc2,color="Black")
# 	ax.set_ylabel(r"$\mathrm{Fr}^2_\Xi$ [-]")
# 	ax.set_xlabel(r"$z_c^*$ [-]")
# 	ax.set_title(r"$\mathrm{Fr}^2_{\Xi,crt}, \,\, F\left(\mathrm{Fr}^2_\Xi, \, z_c^*\right)$ [-]")
# 	# ax.set_ylim([Y_ext.min(),Y_ext.max()])
# 	# plt.colorbar()
# 	plt.colorbar(cs,ax=ax)
# 	return ax
# ax_LN = plot_F(X_ext,Y_ext,Z_ext_LN,zcoa_frc,Frc2)
# # ax_NP = plot_F(X_ext,Y_ext,Z_ext_NP,z_o_l_frc,Frc2)
# ************************ Plot the surface. ************************
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=True)
# ax1.plot([-6,-6],[0,4],color="black",linewidth=2)
# ax1.legend([pl],[(r"$\mathrm{Fr}^2_{\Xi,crt}$")],
#            handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,
#            handlelength=3,loc="center",bbox_to_anchor=(0.5,1.05),frameon=False,fontsize=14)
# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')
#endregion
# ********************* Validate the linear extrapolation 
#region
# fig3=plt.figure(figsize=(8,4),dpi=300);
# axFr2=fig3.add_subplot(121); axzoa=fig3.add_subplot(122)
# axFr2.set_xlabel(r"$z_c^*$ [-]"); axFr2.set_ylabel(r"$ F\left(\mathrm{Fr}^2_\Xi, \, z_c^*\right)$ [-]")
# axzoa.set_xlabel(r"$\mathrm{Fr}^2_\Xi$ [-]"); axzoa.set_ylabel(r"$ F\left(\mathrm{Fr}^2_\Xi, \, z_c^*\right)$ [-]")
# def add_Fr2_slice(slc,ax,ax_2D,zoa_ext,Z_ext_LN):
# 	ax.plot(zoa_ext,Z_ext_LN[slc,:],marker='o',mfc='none',markevery=21,color="black")
# 	ymin = ax.get_ylim()[0]; ymax = ax.get_ylim()[1]
# 	ax.plot([-4, -4], [ymin, ymax], color = 'grey', linestyle = '--')
# 	ax.plot([-6, -6], [ymin, ymax], color = 'grey', linestyle = '--')
# 	ax_2D.plot(linspace(zoa_ext.min(),zoa_ext.max(),15),linspace(Fr2_ext[slc],Fr2_ext[slc],15),
# 			   marker='o',mfc='none',color="black")
# def add_zoa_slice(slc,ax,ax_2D,Fr2_ext,Z_ext_LN):
# 	ax.plot(Fr2_ext,Z_ext_LN[:,slc],marker='v',mfc='none',markevery=21,color="black")
# 	ymin = ax.get_ylim()[0]; ymax = ax.get_ylim()[1]
# 	ax.plot([0, 0], [ymin, ymax], color = 'grey', linestyle = '--')
# 	ax.plot([4, 4], [ymin, ymax], color = 'grey', linestyle = '--')
# 	ax_2D.plot(linspace(zoa_ext[slc],zoa_ext[slc],15),linspace(Fr2_ext.min(),Fr2_ext.max(),15),
# 			   marker='v',mfc='none',color="black")
# add_Fr2_slice(10,axFr2,ax_LN,zoa_ext,Z_ext_LN)
# add_zoa_slice(33,axzoa,ax_LN,Fr2_ext,Z_ext_LN)
#************************************************************************************
# Add a color bar which maps values to colors.
# fig1.colorbar(cs1,ticks=[Z.min(),0,1,2,3,4,Z.max()])
# fig2.colorbar(cs2,ticks=[-0.8,0,0.8,1.6,2.4,3.2,4.0,4.8,5.6])
# Regenerate the table:
# Original table
# Refitcoefs=Table['Refitcoefs'][0];	FrXcoefs=Table['FrXcoefs'][0]
# Fr2_lst=Table['flxfr_data'][0,:]; zcoa_lst=Table['z_a_data'][:,0]; F_tab=Table['F_lookuptable']
# savez("Ent_table_org.npz",Refitcoefs=Refitcoefs, FrXcoefs=FrXcoefs, flxfr_data=Fr2_lst,\
#       z_a_data=zcoa_lst, F_lookuptable=F_tab)
# # Nearest point table
# Z_ext = F_func_table_ext(Fr2_lst,Fr2_ext,zcoa_lst,zoa_ext,F_tab,method="Nearest Point")
# savez("Ent_table_NP.npz",Refitcoefs=Refitcoefs, FrXcoefs=FrXcoefs, flxfr_data=Fr2_ext,\
#       z_a_data=zoa_ext, F_lookuptable=Z_ext)
# # Linear Exrapolation table
# Z_ext = F_func_table_ext(Fr2_lst,Fr2_ext,zcoa_lst,zoa_ext,F_tab,method="Linear Exrapolation")
# savez("Ent_table_LE.npz",Refitcoefs=Refitcoefs, FrXcoefs=FrXcoefs, flxfr_data=Fr2_ext,\
#       z_a_data=zoa_ext, F_lookuptable=Z_ext)
#endregion
# ********************* Vortex rising speeds 
#region
# rsp_vs_lam=zeros((def_size,3))
# fig=plt.figure(figsize=(6,2.5),dpi=300)
# plt.subplots_adjust(hspace=0.2,wspace=0.2)
# axlst=[]; pltlst=[]
# ax=fig.add_subplot(121); axlst.append(ax); ax=fig.add_subplot(122); axlst.append(ax) 
# #const depth
# for idepth in range(3):
# 	lam_range=logspace(-3,log10(2*zp_lst[idepth]-1e-10),def_size)
# 	for i in range(def_size):
# 		rsp_vs_lam[i,idepth] = get_rise_speed(lam_range[i],2.0*zp_lst[idepth],kt,et,nu,cL,cEta,method=2)
# 	pl,=axlst[0].plot(lam_range,rsp_vs_lam[:,idepth],color=colors[idepth])
# 	pltlst.append(pl)
# axlst[0].set_xlabel(r"$\lambda \ \mathrm{[m]}$"); axlst[0].set_ylabel(r"$w(z',\lambda) \ \mathrm{[m/}s]$")
# #const lambda
# for ilambda in range(3):
# 	zp_range=logspace(log10(lam_lst[ilambda]/2+1e-10),1,def_size)
# 	for i in range(def_size):
# 		rsp_vs_lam[i,ilambda] = get_rise_speed(lam_lst[ilambda],2.0*zp_range[i],kt,et,nu,cL,cEta,method=2)
# 	axlst[1].plot(zp_range,rsp_vs_lam[:,ilambda],color=colors[ilambda])
# 	pltlst.append(pl)
# axlst[1].set_xlabel(r"$z' \ \mathrm{[m]}$")
# axlst[0].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$z'={}, {}, {}$m".format(zp_lst[0],zp_lst[1],zp_lst[2]))],
# 				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1))
# axlst[1].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$\lambda={}, {}, {}$m".format(lam_lst[0],lam_lst[1],lam_lst[2]))],
# 				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1))
# axlst[0].set_xlim([1e-3,10]); axlst[1].set_xlim([0,10]); #axlst[0].set_xscale("log")
# axlst[0].set_ylim([0,1.1]); axlst[1].set_ylim([0,1.1])
#endregion
# ********************* parameter values and their function values (Calculation, not plot)
#region
# #Re_Gamma
# B_list=zeros((def_size,3)); Recrt_list=zeros((def_size,3)); Re_list=zeros(def_size)
# #We_Gamma
# W_list=zeros(def_size); We_list=zeros(def_size)
# #Fr^2_Xi
# F_list=zeros((def_size,6)); Fr2_list=zeros((def_size,6)); Fr2_crt_list=zeros((def_size,6)) # 8: As a func of lambda(3) and depth(3)
# Re_gamma_roots = zeros((3,2))
# # \lambda as the independent variable
# lam_range=logspace(-4,1,def_size)
# for j in range(3):
# 	for i in range(def_size):
# 		(B_list[i,j], Re_list[i], Recrt_list[i,j],
# 		W_list[i], We_list[i],
# 		F_list[i,j], Fr2_list[i,j], Fr2_crt_list[i,j], tau_vort)=\
# 		Calc_Para_Func(zp_lst[j],lam_range[i],lst,ul2_lst,rhoc,
# 					   sig,kt,et,nu,cL,cEta,g,
# 					   Refitcoefs,FrXcoefs,Fr2_lst,zcoa_lst,F_tab,F_tab_NP,Fr2_crt_PolyExtra)
# # zp as the independent variable
# zp_range=logspace(-1,1,def_size)
# for j in range(3):
# 	for i in range(def_size):
# 		(tmp, tmp, tmp,
# 		tmp,tmp,
# 		F_list[i,j+3], Fr2_list[i,j+3], Fr2_crt_list[i,j+3], tau_vort)=\
# 		Calc_Para_Func(zp_range[i],lam_lst[j],lst,ul2_lst,rhoc,
# 					   sig,kt,et,nu,cL,cEta,g,
# 					   Refitcoefs,FrXcoefs,Fr2_lst,zcoa_lst,F_tab,F_tab_NP,Fr2_crt_PolyExtra)
#endregion
# ********************* Parameters 
#region
# fig=plt.figure(figsize=(6,6), dpi=200)
# plt.subplots_adjust(wspace=0.4, hspace=0.8)
# axlst=[]; pltlst=[]
# ax=fig.add_subplot(221); axlst.append(ax); ax=fig.add_subplot(222); axlst.append(ax)
# ax=fig.add_subplot(223); axlst.append(ax); ax=fig.add_subplot(224); axlst.append(ax)

# # Re_gamma
# axlst[0].plot(lam_range,Re_list,color="Black")
# for j in range(3):
# 	pl,=axlst[0].plot(lam_range,Recrt_list[:,j],color=colors[j+1],linestyle="--"); pltlst.append(pl)
# 	axlst[0].fill_between(lam_range, Re_list, 
#                       where=((lam_range > zp_lst[j]/3)*(lam_range < zp_lst[j]/2)), 
#                       color=colors[j+1], alpha=0.3)
# axlst[0].set_xscale("log"); axlst[0].set_yscale("log"); axlst[0].set_xlim([1e-4,10]); axlst[0].set_ylim([10,1e8])
# axlst[0].set_xlabel(r"$\lambda$ [m]"); axlst[0].set_ylabel(r"$\mathrm{Re}_\Gamma$ [-]")
# axlst[0].legend([(pltlst[0],pltlst[1],pltlst[2])],
# 				[(r"$\mathrm{Re}_{\Gamma,crt}$"+r"$, z'={}, {}, {}m$".format(zp_lst[0],zp_lst[1],zp_lst[2]))],
# 				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1))

# # We_gamma
# axlst[1].plot(lam_range,We_list,color="Black"); pltlst=[]
# pl,=axlst[1].plot([5.42e-3,5.42e-3],[min(We_list),max(We_list)],color="black",linestyle="--");  pltlst.append(pl)
# pl,=axlst[1].plot([38.3e-3,38.3e-3],[min(We_list),max(We_list)],color="black",linestyle=":"); pltlst.append(pl)
# axlst[1].set_xscale("log"); axlst[1].set_yscale("log"); 
# axlst[1].set_xlim([1e-4,10])
# axlst[1].set_xlabel(r"$\lambda$ [m]"); axlst[1].set_ylabel(r"$\mathrm{We}_\Gamma$ [-]")
# axlst[1].legend([(pltlst[0],pltlst[1])],[(r"$\mathrm{Bo}_\Gamma = 1, 50$")],
# 				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1),labelspacing=0.2)
# # Fr_gamma
# #Const depth
# pltlst=[]
# for j in range(3):
# 	pl,=axlst[2].plot(lam_range,Fr2_list[:,j],color=colors[j+1]); pltlst.append(pl)
# 	pl,=axlst[2].plot(lam_range,Fr2_crt_list[:,j],color=colors[j+1],linestyle=(0,(1,0.8))); pltlst.append(pl)
# 	axlst[2].fill_between(lam_range, Fr2_list[:,j], 
#                       where=((lam_range > zp_lst[j]/3)*(lam_range < zp_lst[j]/2)), 
#                       color=colors[j+1], alpha=0.3)
# axlst[2].set_xscale("log"); axlst[2].set_yscale("log"); 
# axlst[2].set_xlim([1e-2,10]); axlst[2].set_ylim([1e-4,25])
# axlst[2].set_xlabel(r"$\lambda$ [m]"); axlst[2].set_ylabel(r"$\mathrm{Fr}^2_\Xi$ [-]")
# axlst[2].legend([(pltlst[0],pltlst[2],pltlst[4]),(pltlst[1],pltlst[3],pltlst[5])],\
# 				[(r"$\mathrm{Fr}^2_\Xi,$"+r"$ z'={}, {}, {}m$".format(zp_lst[0],zp_lst[1],zp_lst[2])),
# 				(r"$\mathrm{Fr}^2_{\Xi,crt},$"+r"$ z'={}, {}, {}m$".format(zp_lst[0],zp_lst[1],zp_lst[2]))],
# 				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.25),labelspacing=0.2)


# #Const Vortex Size
# pltlst=[]
# for j in range(3):
# 	i = j+3
# 	pl,=axlst[3].plot(zp_range,Fr2_list[:,i],color=colors[j+1]); pltlst.append(pl)
# 	pl,=axlst[3].plot(zp_range,Fr2_crt_list[:,i],color=colors[j+1],linestyle=(0,(1,0.5))); pltlst.append(pl)
# 	axlst[3].fill_between(zp_range, Fr2_list[:,i], 
#                       where=((lam_lst[j] > zp_range/3)*(lam_lst[j] < zp_range/2)), 
#                       color=colors[j+1], alpha=0.3)
# axlst[3].set_xscale("log"); axlst[3].set_yscale("log"); 
# axlst[3].set_xlim([1e-1,10]); axlst[3].set_ylim([1e-3,5])
# axlst[3].set_xlabel(r"$z'$ [m]"); axlst[3].set_ylabel(r"$\mathrm{Fr}^2_\Xi$ [-]")
# axlst[3].legend([(pltlst[0],pltlst[2],pltlst[4]),(pltlst[1],pltlst[3],pltlst[5])],\
# 				[(r"$\mathrm{Fr}^2_\Xi,$"+r"$ \lambda={}, {}, {}m$".format(lam_lst[0],lam_lst[1],lam_lst[2])),
# 				(r"$\mathrm{Fr}^2_{\Xi,crt},$"+r"$ \lambda={}, {}, {}m$".format(lam_lst[0],lam_lst[1],lam_lst[2]))],
# 				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.25),labelspacing=0.2)
#endregion
# ********************* Function values 
#region
# fig=plt.figure(figsize=(6,5), dpi=200)
# plt.subplots_adjust(wspace=0.4, hspace=0.6)
# axlst=[]; pltlst=[]
# ax=fig.add_subplot(221); axlst.append(ax); ax=fig.add_subplot(222); axlst.append(ax)
# ax=fig.add_subplot(223); axlst.append(ax); ax=fig.add_subplot(224); axlst.append(ax)

# # B
# for idepth in range(3):
# 	# lam_range=logspace(-4,log10(2*zp_lst[idepth]),def_size)
# 	pl,=axlst[0].plot(lam_range,B_list[:,idepth],color=colors[idepth+1]); pltlst.append(pl)
# 	axlst[0].fill_between(lam_range, B_list[:,idepth], 
# 						  where=((lam_range > zp_lst[idepth]/3)*(lam_range < zp_lst[idepth]/2)), color=colors[idepth+1], alpha=0.3)
# 	axlst[0].plot([Re_gamma_roots[idepth,0]],[0],color=colors[idepth+1],marker='x')
# 	axlst[0].plot([Re_gamma_roots[idepth,1]],[0],color=colors[idepth+1],marker='s')
	
# axlst[0].set_xscale("log"); axlst[0].set_yscale("linear"); axlst[0].set_xlim([1e-3,2*zp_lst[2]]); axlst[0].set_ylim([0,1e-1])
# axlst[0].set_xlabel(r"$\lambda$ [m]"); axlst[0].set_ylabel(r"$B\left(\mathrm{Re}_\Gamma,\ z'/\lambda\right)$ [-]")
# axlst[0].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$z'={}, {}, {}$m".format(zp_lst[0],zp_lst[1],zp_lst[2]))],
# 				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1))

# # We_gamma
# pltlst=[]
# axlst[1].plot(lam_range,W_list,color="Black")
# pl,=axlst[1].plot([5.42e-3,5.42e-3],[min(W_list),max(W_list)],color="black",linestyle="--"); pltlst.append(pl)
# pl,=axlst[1].plot([38.3e-3,38.3e-3],[min(W_list),max(W_list)],color="black",linestyle=":");  pltlst.append(pl)
# axlst[1].set_xscale("log")
# axlst[1].set_xlim([1e-4,2*zp_lst[2]])
# axlst[1].set_xlabel(r"$\lambda$ [m]"); axlst[1].set_ylabel(r"$W\left(\mathrm{We}_\Gamma\right)$ [-]")
# axlst[1].legend([(pltlst[0],pltlst[1])],[(r"$\mathrm{Bo}_\Gamma = 1, 50$")],
# 				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1),labelspacing=0.2)

# #Const depth
# pltlst=[]
# for idepth in range(3):
# 	# lam_range=logspace(-4,log10(2*zp_lst[idepth]),def_size)
# 	pl,=axlst[2].plot(lam_range,F_list[:,idepth],color=colors[idepth+1]); pltlst.append(pl)
# 	axlst[2].fill_between(lam_range, 0,100,
# 						  where=((lam_range > zp_lst[idepth]/3)*(lam_range < zp_lst[idepth]/2)), color=colors[idepth+1], alpha=0.3)
# axlst[2].set_xscale("log"); axlst[2].set_yscale("log"); 
# axlst[2].set_xlim([1e-2,2*zp_lst[2]]); axlst[2].set_ylim([1e-4,10])
# axlst[2].set_xlabel(r"$\lambda$ [m]"); axlst[2].set_ylabel(r"$F\left(\mathrm{Fr}^2_\Xi,z'/\lambda\right)$ [-]")
# axlst[2].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$z'={}, {}, {}$m".format(zp_lst[0],zp_lst[1],zp_lst[2]))],
# 				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.11),labelspacing=0.2)

# #Const vortex size
# pltlst=[]
# for ilam in range(3):
# 	i = ilam+3; # zp_range=logspace(log10(lam_lst[ilam]/2+1e-10),1,def_size)
# 	pl,=axlst[3].plot(zp_range,F_list[:,i],color=colors[ilam+1]); pltlst.append(pl)
# 	axlst[3].fill_between(zp_range,F_list[:,i],
# 						  where=((zp_range > lam_lst[ilam]*2)*(zp_range < lam_lst[ilam]*3)), color=colors[ilam+1], alpha=0.3)
# axlst[3].set_xscale("log"); axlst[3].set_yscale("log"); 
# axlst[3].set_xlim([1e-1,10]); axlst[3].set_ylim([1e-3,10])
# axlst[3].set_xlabel(r"$z'$ [m]"); axlst[3].set_ylabel(r"$F\left(\mathrm{Fr}^2_\Xi,z'/\lambda\right)$ [-]")
# axlst[3].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$\lambda={}, {}, {}$m".format(lam_lst[0],lam_lst[1],lam_lst[2]))],
# 				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.11),labelspacing=0.2)
#endregion
#! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#! $$$$$$$$$$$$$$$$$$$$$$$$$ Sector filtering plots $$$$$$$$$$$$$$$$$$$$$$$$$
#! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# ********************* parameter values and their function values (Calculation, not plot)
sector = 0
sz = 200
lam_lst = array([5.42e-3, 1e-1, 5e-1, 1])
def legend_string_get(lam_lst):
	lgd = r"$\lambda$ = "
	for l in lam_lst:
		if l <1e-2:
			lgd += "{:}mm".format(l*1000)
		else:
			lgd += "{:}m".format(l)
		if l != lam_lst[len(lam_lst)-1]:
			lgd += ", "
	return lgd
lgd=legend_string_get(lam_lst)
zl_range = logspace(log10(1),log10(6.751203194740859),sz)
Re_lst = []; Re_crt_lst = []; B_lst = []
Fr2lst = []; Fr2_crt_lst = []; F_lst = [];
V_lst = []; Q_lst=[];
Fr2_crt = zeros(sz)
for l in lam_lst:
	Re	= zeros(sz); Re_crt = zeros(sz); B = zeros(sz)
	Fr2 = zeros(sz); F = zeros(sz)
	V	= zeros(sz); Q = zeros(sz)
	for j in range(sz):
		zp = l*zl_range[j]
		(B[j], Re[j], Re_crt[j], \
		W, tmp, \
		F[j], Fr2[j], Fr2_crt[j], tau_vort, n_lam)=\
		Calc_Para_Func(zp, l, lst, ul2_lst, rhoc,
						sig, kt, et, nu, cL, cEta, g,
						Refitcoefs, FrXcoefs, Fr2_lst, zcoa_lst, F_tab, F_tab_NP, Fr2_crt_PolyExtra)
		V[j] = pi*l**3/6.0*F[j]*B[j]*W
		Q[j] = V[j] / tau_vort * n_lam
	Re_lst.append(Re);	Re_crt_lst=[Re_crt]; B_lst.append(B)
	Fr2lst.append(Fr2); Fr2_crt_lst=[Fr2_crt]; F_lst.append(F)
	V_lst.append(V); Q_lst.append(Q) 
# ********************* Parameters 
fig_para = plt.figure(figsize=(8,4), dpi=200)
ax_Re = fig_para.add_subplot(121); ax_Fr = fig_para.add_subplot(122)
myax(ax_Re, zlam_str, Re_gamma_str+" [-]",
     (zl_range.min(), zl_range.max()), (1e2, 1e7), 'linear', 'log')
myax(ax_Fr, zlam_str, Fr2_xi_str+" [-]",
     (zl_range.min(), zl_range.max()), (1e-2, 1e2), 'linear', 'log')
def plot_all_lines(ax,x,vl_lst,crt_lst,cls,lgd_str,crt_str):
	pl_lst=(); pl_crt_lst=()
	for i in range(len(vl_lst)):
		pl,=ax.plot(x,vl_lst[i],color=cls[i],linewidth = 2)
		pl_lst += (pl,)
	for i in range(len(crt_lst)):
		if len(crt_lst) == 1:
			cl='black'
		else:
			cl = cls[i]
		pl,=ax.plot(x,crt_lst[i],color=cl,linestyle="-.",linewidth = 2)
		pl_crt_lst += (pl,)
	if len(crt_lst) !=0 :
		lst_1 = [pl_lst, pl_crt_lst]
		lst_2 = [lgd_str, crt_str]
	else:
		lst_1 = [pl_lst]
		lst_2 = [lgd_str]
	ax.legend(lst_1,lst_2,
				loc="center",
				bbox_to_anchor=(0.5, 1.1),
				handler_map={tuple: HandlerTuple(ndivide=None)},
				handlelength=4,labelspacing=0.2,frameon=False)
	plt.tight_layout()
#--------------------------------------------------------------------
plot_all_lines(ax_Re,zl_range,Re_lst,Re_crt_lst,colors,lgd,Re_gamma_crt_str)
plot_all_lines(ax_Fr,zl_range,Fr2lst,Fr2_crt_lst,colors,lgd,Fr2_xi_crt_str)
# ********************* Function values for each sector
fig_func = plt.figure(figsize=(8,4), dpi=200)
ax_B = fig_func.add_subplot(121); ax_F = fig_func.add_subplot(122)
myax(ax_B, zlam_str, B_str+" [-]",
     (zl_range.min(), zl_range.max()), (0, 0.1), 'linear', 'linear')
myax(ax_F, zlam_str, F_str+" [-]",
     (zl_range.min(), zl_range.max()), (1e-4, 10), 'linear', 'log')
plot_all_lines(ax_B,zl_range,B_lst,[],colors,lgd,())
plot_all_lines(ax_F,zl_range,F_lst,[],colors,lgd,())
# ********************* Entrainment source for single vortex for fixed lambda
fig_sgsr = plt.figure(figsize=(8,4), dpi=200)
ax_V = fig_sgsr.add_subplot(121); ax_Q = fig_sgsr.add_subplot(122)
myax(ax_V, zlam_str, V_str+r" $\mathrm{[m^3]}$",
     (zl_range.min(), zl_range.max()), (1e-10, 1e-1), 'linear', 'log')
myax(ax_Q, zlam_str, Q_nlam_str+r" $\mathrm{[1/(ms)]}$",
     (zl_range.min(), zl_range.max()), (1e-8, 100), 'linear', 'log')
plot_all_lines(ax_V,zl_range,V_lst,[],colors,lgd,())
plot_all_lines(ax_Q,zl_range,Q_lst,[],colors,lgd,())
# ********************* Entrainment source for single vortex for fixed z/lam
#region
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
# 					   sig,kt,et,nu,cL,cEta,g,
# 					   Refitcoefs,FrXcoefs,Fr2_lst,zcoa_lst,F_tab,zl_min,zl_max)
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
# 				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(1.15,1.2),labelspacing=0.2)
#endregion
# ********************* Specifically asked figures
# 1. circulation/lambda^2 vs lambda
lam_range=logspace(log10(5.76e-3),log10(1),sz)
circ = zeros(sz); nlam = zeros(sz)
for i in range(len(lam_range)):
	Reg,Bog,Weg,circ[i],nlam[i],x,tmp = J_lambda_prep(lam_range[i],lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
fig_1 = plt.figure(figsize=(4,4), dpi=200); ax_1 = fig_1.add_subplot(111);
myax(ax_1, r"$\lambda$ [m]", r"$\Gamma_p/\lambda^2$ [1/s]",
     (lam_range.min(), lam_range.max()), (0, 200), 'log', 'linear')
ax_1.plot(lam_range,circ/lam_range**2,color = 'black')
# 2. vortex number density
fig_2 = plt.figure(figsize=(4,4), dpi=200); ax_2 = fig_2.add_subplot(111);
myax(ax_2, r"$\lambda$ [m]", r"n$_\lambda \mathrm{[1/m^4]}$",
     (lam_range.min(), lam_range.max()), (1e-1,1e9), 'log', 'log')
ax_2.plot(lam_range,nlam,color = 'black')
# 3. circ flux vs depth
sz2 = len(lam_lst)
circ_flux_lst=[]; time_ratio_lst=[]
for i in range(sz2):
	l = lam_lst[i]
	circ_flx = zeros(sz); rise_time = zeros(sz)
	tmp,tmp,tmp,circ,tmp,tmp,vort_lifetime = J_lambda_prep(lam_range[i],lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
	for j in range(sz):
		wz = get_rise_speed(l,2*l*zl_range[j],kt,et,nu,cL,cEta,method=2)
		circ_flx[j] = wz * circ
		rise_time[j] = (l*zl_range[j]) / wz
	circ_flux_lst.append(circ_flx)
	time_ratio_lst.append(rise_time/vort_lifetime)
fig_3 = plt.figure(figsize=(4,4), dpi=200); ax_3 = fig_3.add_subplot(111);
myax(ax_3, zlam_str, r"$w_z \cdot \Gamma_p \mathrm{[m^3/s^2]}$",
     (zl_range.min(), zl_range.max()), (0,4e-3), 'linear', 'linear')
plot_all_lines(ax_3,zl_range,circ_flux_lst,[],colors,lgd,())
# 4. life time vs rise time
fig_4 = plt.figure(figsize=(4,4), dpi=200); ax_4 = fig_4.add_subplot(111);
myax(ax_4, zlam_str, r"Life time/Rise time [-]",
     (zl_range.min(), zl_range.max()), (1e0,1e4), 'linear', 'log')
plot_all_lines(ax_4,zl_range,time_ratio_lst,[],colors,lgd,())