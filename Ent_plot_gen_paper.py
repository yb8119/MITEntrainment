from numpy import logspace, log10, zeros, sqrt, pi, linspace, meshgrid, savez, concatenate
import numpy as np
from scipy.io import loadmat
from Utilities import findcLceta, ulambda_sq, F_func_table_ext
from Model import Fr2_crit_getter
from Plt_funcs import Calc_Para_Func
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
colors=["Black","Red","Green","Blue","Purple"]
LineSty=["-",(0,(5,3)),(0,(3,3,1,3))]
# ****************** Physical/Flow field parameters ******************#
#region
nu =1e-6; g=9.81; sig=0.072;rhoc=1000;nu=1e-6
kt = 1.0 #m2/s2 # kt = 1.75 #m2/s2
et = 4.0 #m2/s3 # et = 3.32 #m2/s3
zl_min=2;	zl_max=3000
cL,cEta=findcLceta(kt,et,nu,mode=1); L=kt**1.5/et
######## Get the basics for the MIT model ########
# Load the depth table:
Table=loadmat("UoIEntrainment.mat")
Refitcoefs=Table['Refitcoefs'][0];	FrXcoefs=Table['FrXcoefs'][0]
Fr2_lst=Table['flxfr_data'][0,:]; zcoa_lst=Table['z_a_data'][:,0]; F_tab=Table['F_lookuptable']
nlst=1500; #x1=sqrt(4*sig/rhoc/g); x4=10*L
lst=logspace(-5,2,nlst);	ul2_lst=zeros(nlst) #with dimension!'
zp_lst = [0.2, 1, 5]
lam_lst = [0.2, 1, 5]
F_tab_NP = True
Fr2_crt_PolyExtra = False
for i in range(nlst):
	ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
#endregion
# ********************* Critical Froude number square (extended range)
#region
#endregion
# ********************* Froude number function (extended range) 
#region
# # fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(6,6), dpi=200)
# # Make data.
# X = zcoa_lst
# Y = Fr2_lst
# Z = F_tab.transpose()
# # Linear extrapolation
# zoa_ext= concatenate((linspace(-2*zl_max, -2*3-1e-2, 100),linspace(-2*3, -2*2, 100)))
# # zoa_ext	= linspace(-2*zl_max, -2*zl_min, 2000)
# Fr2_ext	= concatenate((linspace(0, 4, 50),linspace(4+1e-2, 100, 100)))
# Z_ext_LN = zeros((Fr2_ext.size,zoa_ext.size))
# # Z_ext_NP = zeros((Fr2_ext.size,zoa_ext.size))
# Z_ext_LN = F_func_table_ext(Fr2_lst,Fr2_ext,zcoa_lst,zoa_ext,F_tab,method="Linear Exrapolation").transpose()
# # Z_ext_NP = F_func_table_ext(Fr2_lst,Fr2_ext,zcoa_lst,zoa_ext,F_tab,method="Nearest Point").transpose()

# # f = interp2d(X, Y, Z, kind = 'cubic')
# # f = RectBivariateSpline(X, Y, F_tab, bbox=[min(X_ext)])
# # for i in range(X_ext.size):
# # 	for j in range(Y_ext.size):
# # 		Z_ext[j,i] = f(X_ext[i],Y_ext[j])
# X, Y = meshgrid(X, Y)
# X_ext, Y_ext = meshgrid(zoa_ext, Fr2_ext)
# z_o_l_frc = linspace(-2*zl_max, -2*zl_min, 250); Frc2=zeros(250)
# for i in range(250):
# 	(tmp, tmp, tmp,
# 	 tmp, tmp,
# 	 tmp, tmp, Frc2[i])=\
# 	Calc_Para_Func(z_o_l_frc[i]/-2*100,100,lst,ul2_lst,rhoc,
# 	               sig,kt,et,nu,cL,cEta,g,
# 	               Refitcoefs,FrXcoefs,Fr2_lst,zcoa_lst,F_tab,zl_min,zl_max) 
# def plot_F(X,Y,Z,z_o_l_frc,Frc2):
# 	vmin =max(1e-10,Z.min()); vmax = Z.max()
# 	fig, ax = plt.subplots(figsize=(5,4), dpi=200)
# 	rect = patches.Rectangle((-6,0),2,4, linewidth=1, edgecolor='white', facecolor='none')
# 	cb_range=linspace(vmin,vmax,101)
# 	# cb_range=logspace(log10(vmin),log10(vmax),101)
# 	cs = ax.contourf(X, Y, Z, 
# 				   cmap=cm.gist_rainbow, antialiased=True, levels=cb_range, 
# 				   #norm=matcolors.LogNorm(vmin=vmin, vmax=vmax),
# 				   vmin=vmin,vmax=vmax, extend="both")
# 	ax.add_patch(rect)
# 	pl,= ax.plot(z_o_l_frc,Frc2,color="Black")
# 	ax.set_ylabel(r"$\mathrm{Fr}^2_\Xi$ [-]")
# 	ax.set_xlabel(r"$z_c^*$ [-]")
# 	ax.set_title(r"$\mathrm{Fr}^2_{\Xi,crt}, \,\, F\left(\mathrm{Fr}^2_\Xi, \, z_c^*\right)$ [-]")
# 	# ax.set_ylim([Y_ext.min(),Y_ext.max()])
# 	# plt.colorbar()
# 	plt.colorbar(cs,ax=ax)
# 	return ax
# ax_LN = plot_F(X_ext,Y_ext,Z_ext_LN,z_o_l_frc,Frc2)
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
# zp_lam_lst = [zl_min, 2, 3, zl_max]
#Re_Gamma
B_list=zeros((def_size,3)); Recrt_list=zeros((def_size,3)); Re_list=zeros(def_size)
#We_Gamma
W_list=zeros(def_size); We_list=zeros(def_size)
#Fr^2_Xi
F_list=zeros((def_size,6)); Fr2_list=zeros((def_size,6)); Fr2_crt_list=zeros((def_size,6)) # 8: As a func of lambda(3) and depth(3)
# \lambda as the independent variable
lam_range=logspace(-4,1,def_size)
for j in range(3):
	for i in range(def_size):
		(B_list[i,j], Re_list[i], Recrt_list[i,j],
		W_list[i], We_list[i],
		F_list[i,j], Fr2_list[i,j], Fr2_crt_list[i,j], tau_vort)=\
		Calc_Para_Func(zp_lst[j],lam_range[i],lst,ul2_lst,rhoc,
					   sig,kt,et,nu,cL,cEta,g,
					   Refitcoefs,FrXcoefs,Fr2_lst,zcoa_lst,F_tab,F_tab_NP,Fr2_crt_PolyExtra)
# zp as the independent variable
zp_range=logspace(-1,1,def_size)
for j in range(3):
	for i in range(def_size):
		(tmp, tmp, tmp,
		tmp,tmp,
		F_list[i,j+3], Fr2_list[i,j+3], Fr2_crt_list[i,j+3], tau_vort)=\
		Calc_Para_Func(zp_range[i],lam_lst[j],lst,ul2_lst,rhoc,
					   sig,kt,et,nu,cL,cEta,g,
					   Refitcoefs,FrXcoefs,Fr2_lst,zcoa_lst,F_tab,F_tab_NP,Fr2_crt_PolyExtra)
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
#                       where=((lam_range > zp_lst[j]/zl_max)*(lam_range < zp_lst[j]/zl_min)), 
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
#                       where=((lam_range > zp_lst[j]/zl_max)*(lam_range < zp_lst[j]/zl_min)), 
#                       color=colors[j+1], alpha=0.3)
# axlst[2].set_xscale("log"); axlst[2].set_yscale("log"); 
# axlst[2].set_xlim([1e-2,10]); axlst[2].set_ylim([1e-4,100])
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
#                       where=((lam_lst[j] > zp_range/zl_max)*(lam_lst[j] < zp_range/zl_min)), 
#                       color=colors[j+1], alpha=0.3)
# axlst[3].set_xscale("log"); axlst[3].set_yscale("log"); 
# axlst[3].set_xlim([1e-1,10]); axlst[3].set_ylim([1e-3,10])
# axlst[3].set_xlabel(r"$z'$ [m]"); axlst[3].set_ylabel(r"$\mathrm{Fr}^2_\Xi$ [-]")
# axlst[3].legend([(pltlst[0],pltlst[2],pltlst[4]),(pltlst[1],pltlst[3],pltlst[5])],\
# 				[(r"$\mathrm{Fr}^2_\Xi,$"+r"$ \lambda={}, {}, {}m$".format(lam_lst[0],lam_lst[1],lam_lst[2])),
# 				(r"$\mathrm{Fr}^2_{\Xi,crt},$"+r"$ \lambda={}, {}, {}m$".format(lam_lst[0],lam_lst[1],lam_lst[2]))],
# 				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.25),labelspacing=0.2)
#endregion
# ********************* Function values 
#region
fig=plt.figure(figsize=(6,5), dpi=200)
plt.subplots_adjust(wspace=0.4, hspace=0.6)
axlst=[]; pltlst=[]
ax=fig.add_subplot(221); axlst.append(ax); ax=fig.add_subplot(222); axlst.append(ax)
ax=fig.add_subplot(223); axlst.append(ax); ax=fig.add_subplot(224); axlst.append(ax)

# B
for idepth in range(3):
	# lam_range=logspace(-4,log10(2*zp_lst[idepth]),def_size)
	pl,=axlst[0].plot(lam_range,B_list[:,idepth],color=colors[idepth+1]); pltlst.append(pl)
	axlst[0].fill_between(lam_range, B_list[:,idepth], 
						  where=((lam_range > zp_lst[idepth]/3)*(lam_range < zp_lst[idepth]/2)), color=colors[idepth+1], alpha=0.3)
axlst[0].set_xscale("log"); axlst[0].set_yscale("linear"); axlst[0].set_xlim([1e-3,2*zp_lst[2]]); axlst[0].set_ylim([-0.2,1e-1])
axlst[0].set_xlabel(r"$\lambda$ [m]"); axlst[0].set_ylabel(r"$B\left(\mathrm{Re}_\Gamma,\ z'/\lambda\right)$ [-]")
axlst[0].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$z'={}, {}, {}$m".format(zp_lst[0],zp_lst[1],zp_lst[2]))],
				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1))

# We_gamma
pltlst=[]
axlst[1].plot(lam_range,W_list,color="Black")
pl,=axlst[1].plot([5.42e-3,5.42e-3],[min(W_list),max(W_list)],color="black",linestyle="--"); pltlst.append(pl)
pl,=axlst[1].plot([38.3e-3,38.3e-3],[min(W_list),max(W_list)],color="black",linestyle=":");  pltlst.append(pl)
axlst[1].set_xscale("log")
axlst[1].set_xlim([1e-4,2*zp_lst[2]])
axlst[1].set_xlabel(r"$\lambda$ [m]"); axlst[1].set_ylabel(r"$W\left(\mathrm{We}_\Gamma\right)$ [-]")
axlst[1].legend([(pltlst[0],pltlst[1])],[(r"$\mathrm{Bo}_\Gamma = 1, 50$")],
				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.1),labelspacing=0.2)

#Const depth
pltlst=[]
for idepth in range(3):
	# lam_range=logspace(-4,log10(2*zp_lst[idepth]),def_size)
	pl,=axlst[2].plot(lam_range,F_list[:,idepth],color=colors[idepth+1]); pltlst.append(pl)
	axlst[2].fill_between(lam_range, 0,100,
						  where=((lam_range > zp_lst[idepth]/3)*(lam_range < zp_lst[idepth]/2)), color=colors[idepth+1], alpha=0.3)
axlst[2].set_xscale("log"); axlst[2].set_yscale("log"); 
axlst[2].set_xlim([1e-2,2*zp_lst[2]]); axlst[2].set_ylim([1e-4,10])
axlst[2].set_xlabel(r"$\lambda$ [m]"); axlst[2].set_ylabel(r"$F\left(\mathrm{Fr}^2_\Xi,z'/\lambda\right)$ [-]")
axlst[2].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$z'={}, {}, {}$m".format(zp_lst[0],zp_lst[1],zp_lst[2]))],
				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.11),labelspacing=0.2)

#Const vortex size
pltlst=[]
for ilam in range(3):
	i = ilam+3; # zp_range=logspace(log10(lam_lst[ilam]/2+1e-10),1,def_size)
	pl,=axlst[3].plot(zp_range,F_list[:,i],color=colors[ilam+1]); pltlst.append(pl)
	axlst[3].fill_between(zp_range,F_list[:,i],
						  where=((zp_range > lam_lst[ilam]*2)*(zp_range < lam_lst[ilam]*3)), color=colors[ilam+1], alpha=0.3)
axlst[3].set_xscale("log"); axlst[3].set_yscale("log"); 
axlst[3].set_xlim([1e-1,10]); axlst[3].set_ylim([1e-3,10])
axlst[3].set_xlabel(r"$z'$ [m]"); axlst[3].set_ylabel(r"$F\left(\mathrm{Fr}^2_\Xi,z'/\lambda\right)$ [-]")
axlst[3].legend([(pltlst[0],pltlst[1],pltlst[2])],[(r"$\lambda={}, {}, {}$m".format(lam_lst[0],lam_lst[1],lam_lst[2]))],
				handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.3,handlelength=4,loc="center",bbox_to_anchor=(0.5,1.11),labelspacing=0.2)
#endregion
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
