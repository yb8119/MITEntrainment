from numpy import sqrt, logspace, log10, zeros, linspace, pi
from Model import J_lambda_prep, Ent_Volume, Ent_Volume_Z, max_entrainement, Ent_rate_prev
from Utilities import findcLceta, ulambda_sq
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from scipy.io import loadmat
from matplotlib.ticker import ScalarFormatter

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
fs=12

##########################
#   PHYSICAL PARAMETERS  #
##########################
kt=1.23; et=1.83; nu=1e-6; g=9.81; rhoc=1000; sig=0.072
cL,cEta=findcLceta(kt,et,nu,mode=1)
nlst=1800
lst=logspace(-8,2,nlst);	ul2_lst=zeros(nlst) #with dimension!
for i in range(nlst):
	ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
# Load the depth table:
Table=loadmat("UoIEntrainment.mat");
############################################################################
#================== Critical Froude number ==================#
############################################################################
nz=100; l=1e-3 # l does not matter here
z_lst=linspace(2*l,3*l,nl);
fig1=plt.figure(figsize=(3,2),dpi=300)
color_lst=['black','red','blue']
style_lst=['-','--','dotted']
Re_out=zeros(nl); B_out=zeros((nl,nz)); We_out=zeros(nl); W_out=zeros(nl);
Fr2_out=zeros((nl,nz)); F_out=zeros((nl,nz)); Fr2c_out=zeros(nz); Q_out=zeros((nl,nz));

for il in range(nl):
	l=l_lst[il]
	ulamsq,Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
	Re_out[il] = Reg; We_out[il] = Weg;
	for iz in range(nz):
		V_Ent, Fr2_crit, B, F, W, Fr2= Ent_Volume_Z(zlst_ratio[iz]*l,l,lst,ul2_lst,g,circ_p,Reg,Bog,Weg,Table,mode=1)
		Fr2c_out[iz] = Fr2_crit; B_out[il,iz] =  B; W_out[il] =  W; F_out[il,iz] =  F; Fr2_out[il,iz]=Fr2
		Q_out[il,iz] = V_Ent
#### Parameter figure
ax_1=fig11.add_subplot(131)
ax_1.plot(l_lst,Re_out,color='black');
ax_1.plot(l_lst,70*l_lst/l_lst,color='black',linestyle='--',label=r'$\mathrm{Re}_\Gamma=70$');
ax_1.plot(l_lst,2580*l_lst/l_lst,color='black',linestyle='--',label=r'$\mathrm{Re}_\Gamma=2580$');
ax_1.legend(labelspacing=0.15); ax_1.set_xscale('log'); ax_1.set_yscale('log'); ax_1.set_ylim([1e1,1e7])
ax_1.set_ylabel(r'$\mathrm{Re}_\Gamma$'); ax_1.set_xlim([min(l_lst),max(l_lst)])
ax_1.set_xlabel(r'$\lambda$ [m]')

ax_2=fig11.add_subplot(132)
for iz in range(nz):
	ax_2.plot(l_lst,B_out[:,iz],color=color_lst[iz],label=r"$z'/\lambda={:g}$".format(zlst_ratio[iz]));
ax_2.legend(labelspacing=0.15); ax_2.set_xscale('log'); ax_2.set_yscale('log'); ax_2.set_ylim([1e-3,1e-1])
ax_2.set_ylabel(r"$B\left(\mathrm{Re}_\Gamma,z'/\lambda\right)$"); ax_2.set_xlim([min(l_lst),max(l_lst)])
ax_2.set_xlabel(r'$\lambda$ [m]')

ax_3=fig12.add_subplot(121)
ax_3.plot(l_lst,We_out,color='black');
lam_Bo1=sqrt(4*sig/rhoc/g); lam_Bo50=sqrt(4*sig/rhoc/g*50)
ax_3.plot([lam_Bo1, lam_Bo1 ],[min(We_out),max(We_out)],color='black',linestyle='--',label=r'$\mathrm{Bo}=1$');
ax_3.plot([lam_Bo50,lam_Bo50],[min(We_out),max(We_out)],color='black',linestyle='--',label=r'$\mathrm{Bo}=50$');
ax_3.legend(labelspacing=0.15); ax_3.set_xscale('log'); ax_3.set_yscale('log'); ax_3.set_ylim([1e-2,1e8])
ax_3.set_ylabel(r'$\mathrm{We}_\Gamma$'); ax_3.set_xlim([min(l_lst),max(l_lst)])
ax_3.set_xlabel(r'$\lambda$ [m]')

ax_4=fig12.add_subplot(122)
ax_4.plot(l_lst,W_out,color='black');
ax_4.set_xscale('log'); ax_4.set_yscale('log'); ax_4.set_ylim([1e-2,2])
ax_4.set_ylabel(r"$W\left(\mathrm{We}_\Gamma\right)$"); ax_4.set_xlim([min(l_lst),max(l_lst)])
ax_4.set_xlabel(r'$\lambda$ [m]')


ax_5=fig13.add_subplot(221); pllsst=[]
for iz in range(nz):
	pl,=ax_5.plot(l_lst,Fr2_out[:,iz],color=color_lst[iz]); pllsst.append(pl)
	pl,=ax_5.plot(l_lst,Fr2c_out[iz]*l_lst/l_lst,color=color_lst[iz],linestyle='--'); pllsst.append(pl)
ax_5.set_xscale('log'); ax_5.set_yscale('log')
ax_5.set_ylabel(r'$\mathrm{Fr}^2_\Xi$'); ax_5.set_xlim([min(l_lst),max(l_lst)]); ax_5.set_xlabel(r'$\lambda$ [m]')
ax_5.set_ylim([1e-2,100]);
leg=ax_5.legend([(pllsst[0],pllsst[1]),(pllsst[2],pllsst[3]),(pllsst[4],pllsst[5])],
               [r"$z'/\lambda$={:g},".format(zlst_ratio[0])+ r" $\mathrm{Fr}^{2}_{crt,\Xi}$",
                r"$z'/\lambda$={:g},".format(zlst_ratio[1])+ r" $\mathrm{Fr}^{2}_{crt,\Xi}$",
                r"$z'/\lambda$={:g},".format(zlst_ratio[2])+ r" $\mathrm{Fr}^{2}_{crt,\Xi}$"],
                numpoints=1,handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.2,\
                handlelength=4,fontsize=fs-2.5,frameon=True,labelspacing=0.1,loc='lower left',handletextpad=0.4)
ax_6=fig13.add_subplot(222)
for iz in range(nz):
	ax_6.plot(l_lst,F_out[:,iz],color=color_lst[iz],label=r"$z'/\lambda={:g}$".format(zlst_ratio[iz]));
ax_6.legend(labelspacing=0.12); ax_6.set_xscale('log'); ax_6.set_yscale('log'); ax_6.set_xlabel(r'$\lambda$ [m]')
ax_6.set_ylabel(r"$F\left(\mathrm{Fr}^2_\Xi,z'/\lambda\right)$"); ax_6.set_ylim([1e-3,10]); ax_6.set_xlim([min(l_lst),max(l_lst)])
####################################################################################################
#================== Entrainment volume/rate, parameters verses z'and timescales ===================#
####################################################################################################
fig2=plt.figure(figsize=(7,3),dpi=200)
plt.subplots_adjust(wspace=0.35,hspace=0.25)
ax=fig2.add_subplot(121)
for iz in range(nz):
	ax.plot(l_lst,Q_out[:,iz],color=color_lst[iz],label=r"$z'/\lambda={:g}$".format(zlst_ratio[iz]))
ax.legend(labelspacing=0.15); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$\lambda$ [m]'); ax.set_ylabel(r'$\forall \ \mathrm{[m^3]}$');
ax.set_xlim([min(l_lst),max(l_lst)])
ax=fig2.add_subplot(122)
# Now we use constant depth instead of constant **scaled** depth
nz=3; nl=150; z_lst=logspace(log10(5e-2),log10(5e-2*25),nz); F_out_backup=F_out
Q_out=zeros((nl,nz));B_out=zeros((nl,nz));F_out=zeros((nl,nz)); Fr2_out=zeros((nl,nz)); Fr2c_out=zeros((nl,nz))
Q_dot1=zeros((nl,nz)); Q_dot2=zeros((nl,nz)); tau1=zeros((nl,nz)); tau2=zeros((nl,nz)); Q_dot_prev=zeros((nl,nz))
# constant dimensional depth
for iz in range(nz):
	zp=z_lst[iz]
	l_lst=logspace(log10(zp/2),log10(zp/3),nl)
	for il in range(nl):
		l=l_lst[il]
		ulamsq,Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
		V_Ent, Fr2_crit, B, F, W, Fr2= Ent_Volume_Z(zp,l,lst,ul2_lst,g,circ_p,Reg,Bog,Weg,Table,mode=1)
		tmp1,tmp2=Ent_rate_prev(l,zp,kt,et)
		Q_dot_prev[il,iz]=tmp1
		Q_out[il,iz] = V_Ent; B_out[il,iz] = B; F_out[il,iz]=F; Fr2_out[il,iz]=Fr2; Fr2c_out[il,iz]=Fr2_crit
		tau1[il,iz]   = tau_vort;			tau2[il,iz]   = 0.1*pi*(l/2)**2/circ_p
		Q_dot1[il,iz] = V_Ent/tau1[il,iz];	Q_dot2[il,iz] = V_Ent/tau2[il,iz]
for iz in range(nz):
	l_lst=logspace(log10(z_lst[iz]/2),log10(z_lst[iz]/3),nl)
	ax.plot(l_lst,Q_out[:,iz],color=color_lst[iz],label=r"$z'={:g}m$".format(z_lst[iz]))
	ax.plot([z_lst[iz]/2,z_lst[iz]/2],[1e-8,2e-2],color=color_lst[iz],linestyle=(0, (3,1,1,1)),linewidth=0.7);
	ax.plot([z_lst[iz]/3,z_lst[iz]/3],[1e-8,2e-2],color=color_lst[iz],linestyle=(0, (3,1,1,1)),linewidth=0.7);
ax.legend(loc='upper left', labelspacing=0.15); ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(); 
ax.set_xlabel(r'$\lambda$ [m]'); ax.set_ylabel(r'$\forall \ \mathrm{[m^3]}$'); ax.set_ylim([1e-8,2e-2])

ax_7=fig11.add_subplot(133)
for iz in range(nz):
	l_lst=logspace(log10(z_lst[iz]/2),log10(z_lst[iz]/3),nl)
	ax_7.plot(l_lst,B_out[:,iz],color=color_lst[iz],label=r"$z'={:g}$m".format(z_lst[iz]));
	ax_7.plot([z_lst[iz]/2,z_lst[iz]/2],[0.04,0.09],color=color_lst[iz],linestyle=(0, (3,1,1,1)),linewidth=0.7);
	ax_7.plot([z_lst[iz]/3,z_lst[iz]/3],[0.04,0.09],color=color_lst[iz],linestyle=(0, (3,1,1,1)),linewidth=0.7);
ax_7.legend(labelspacing=0.15); ax_7.set_xscale('log')
ax_7.set_ylabel(r"$B\left(\mathrm{Re}_\Gamma,z',\lambda\right)$"); ax_7.set_xlim([1e-2,1]); ax_7.set_ylim([0.04,0.065])
ax_7.set_xlabel(r'$\lambda$ [m]')

ax_8=fig13.add_subplot(223); plt.subplots_adjust(wspace=0.35,hspace=0.25); pllsst=[]
for iz in range(nz):
	l_lst=logspace(log10(z_lst[iz]/2),log10(z_lst[iz]/3),nl)
	pl,=ax_8.plot(l_lst,Fr2_out[:,iz],color=color_lst[iz]); pllsst.append(pl)
	pl,=ax_8.plot(l_lst,Fr2c_out[:,iz],color=color_lst[iz],linestyle='--'); pllsst.append(pl)
	ax_8.plot([z_lst[iz]/2,z_lst[iz]/2],[1e-2,10000],color=color_lst[iz],linestyle=(0, (3,1,1,1)),linewidth=0.7);
	ax_8.plot([z_lst[iz]/3,z_lst[iz]/3],[1e-2,10000],color=color_lst[iz],linestyle=(0, (3,1,1,1)),linewidth=0.7);
ax_8.set_xscale('log'); ax_8.set_yscale('log')
ax_8.set_ylabel(r'$\mathrm{Fr}^2_\Xi$'); ax_8.set_xlabel(r'$\lambda$ [m]'); ax_8.set_xlim([1e-2,0.8]); ax_8.set_ylim([1e-2,10000]);
leg=ax_8.legend([(pllsst[0],pllsst[1]),(pllsst[2],pllsst[3]),(pllsst[4],pllsst[5])],
               [r"$z'$={:g}m,".format(z_lst[0])+ r" $\mathrm{Fr}^{2}_{crt,\Xi}$",
                r"$z'$={:g}m,".format(z_lst[1])+ r" $\mathrm{Fr}^{2}_{crt,\Xi}$",
                r"$z'$={:g}m,".format(z_lst[2])+ r" $\mathrm{Fr}^{2}_{crt,\Xi}$"],
                numpoints=1,handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.2,\
                handlelength=4,fontsize=fs-2.5,frameon=True,labelspacing=0.1,handletextpad=0.4)
ax_9=fig13.add_subplot(224)
for iz in range(nz):
	l_lst=logspace(log10(z_lst[iz]/2),log10(z_lst[iz]/3),nl)
	ax_9.plot(l_lst,F_out[:,iz],color=color_lst[iz],label=r"$z'$={:g}m,".format(z_lst[iz]));
	ax_9.plot([z_lst[iz]/2,z_lst[iz]/2],[1,5],color=color_lst[iz],linestyle=(0, (3,1,1,1)),linewidth=0.7);
	ax_9.plot([z_lst[iz]/3,z_lst[iz]/3],[1,5],color=color_lst[iz],linestyle=(0, (3,1,1,1)),linewidth=0.7);
ax_9.legend(labelspacing=0.12); ax_9.set_xscale('log'); ax_9.set_xlim([1e-2,0.8]); ax_9.set_ylim([1,5]);
ax_9.set_ylabel(r"$F\left(\mathrm{Fr}^2_\Xi,z'/\lambda\right)$"); ax_9.set_xlabel(r'$\lambda$ [m]')

fig3=plt.figure(figsize=(6,2.5),dpi=200); pllsst=[]; plt.subplots_adjust(wspace=0.35,hspace=0.25);
ax_10=fig3.add_subplot(121)
ax_11=fig3.add_subplot(122)
for iz in range(nz):
	l_lst=logspace(log10(z_lst[iz]/2),log10(z_lst[iz]/3),nl)
	pl,=ax_10.plot(l_lst,tau1[:,iz],  color=color_lst[iz],linestyle='solid'); 	pllsst.append(pl)
	pl,=ax_10.plot(l_lst,tau2[:,iz],  color=color_lst[iz],linestyle='dashed'); 	pllsst.append(pl)
	ax_10.plot([z_lst[iz]/2,z_lst[iz]/2],[1e-4,1],color=color_lst[iz],linestyle=(0, (3,1,1,1)),linewidth=0.7);
	ax_10.plot([z_lst[iz]/3,z_lst[iz]/3],[1e-4,1],color=color_lst[iz],linestyle=(0, (3,1,1,1)),linewidth=0.7);
	pl,=ax_11.plot(l_lst,Q_dot1[:,iz],color=color_lst[iz],linestyle='solid'); 	pllsst.append(pl)
	pl,=ax_11.plot(l_lst,Q_dot2[:,iz],color=color_lst[iz],linestyle='dashed'); 	pllsst.append(pl)
	ax_11.plot([z_lst[iz]/2,z_lst[iz]/2],[1e-7,1],color=color_lst[iz],linestyle=(0, (3,1,1,1)),linewidth=0.7);
	ax_11.plot([z_lst[iz]/3,z_lst[iz]/3],[1e-7,1],color=color_lst[iz],linestyle=(0, (3,1,1,1)),linewidth=0.7);
	pl,=ax_11.plot(l_lst,Q_dot_prev[:,iz],color=color_lst[iz],linestyle='dashdot');	pllsst.append(pl)
ax_10.set_xscale('log'); ax_10.set_yscale('log'); ax_11.set_xscale('log'); ax_11.set_yscale('log')
ax_10.set_ylabel(r'$\tau$ [s]'); ax_11.set_ylabel(r'$\dot Q \ \mathrm{[m^3/s]}$ ')
ax_10.set_xlabel(r'$\lambda$ [m]'); ax_11.set_xlabel(r'$\lambda$ [m]')
ax_10.set_xlim([1e-2,1]); ax_10.set_ylim([1e-4,1]);
ax_11.set_xlim([1e-2,1]); ax_11.set_ylim([1e-7,1]);
# leg=ax_10.legend([(pllsst[0],pllsst[1]),(pllsst[4],pllsst[5]),(pllsst[8],pllsst[9])],
#                [r"$z'$={:g}m".format(z_lst[0])+ '\n'+ r" $\tau_{\mathrm{vortex}},\ \tau_{\mathrm{DNS}}$",
#                 r"$z'$={:g}m".format(z_lst[1])+ '\n'+ r" $\tau_{\mathrm{vortex}},\ \tau_{\mathrm{DNS}}$",
#                 r"$z'$={:g}m".format(z_lst[2])+ '\n'+ r" $\tau_{\mathrm{vortex}},\ \tau_{\mathrm{DNS}}$"],
#                 numpoints=1,handler_map={tuple: HandlerTuple(ndivide=None)},borderpad=0.2,\
#                 handlelength=3.5,fontsize=fs-2.5,frameon=True,labelspacing=0.1)
leg=ax_11.legend([(pllsst[2],pllsst[3],pllsst[4]),(pllsst[7],pllsst[8],pllsst[9]),(pllsst[12],pllsst[13],pllsst[14])],
               [r"$z'$={:g}m".format(z_lst[0])+ r" $\tau_{\mathrm{vortex}},\ \tau_{\mathrm{DNS}}, \ \dot Q_{prev}$",
                r"$z'$={:g}m".format(z_lst[1])+ r" $\tau_{\mathrm{vortex}},\ \tau_{\mathrm{DNS}}, \ \dot Q_{prev}$",
                r"$z'$={:g}m".format(z_lst[2])+ r" $\tau_{\mathrm{vortex}},\ \tau_{\mathrm{DNS}}, \ \dot Q_{prev}$"],
                numpoints=1,handler_map={tuple: HandlerTuple(ndivide=None,pad=0.5)},borderpad=0.2,\
                handlelength=9,fontsize=fs-2.5,frameon=True,labelspacing=0.2,handletextpad=0.4,loc='center',
                bbox_to_anchor=(-0.25, 1.20))
####### Compare with single depth ########
# for il in range(nl):
# 	l=l_lst[il]
# 	ulamsq,Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
# 	Re_out[il] = Reg; We_out[il] = Weg;
# 	for iz in range(nz):
# 		V_Ent, Fr2_crit, B, F, W, Fr2= Ent_Volume_Z(zlst_ratio[iz]*l,l,lst,ul2_lst,g,circ_p,Reg,Bog,Weg,Table,mode=0)
# 		Fr2c_out[iz] = Fr2_crit; B_out[il,iz] =  B; W_out[il] =  W; F_out[il,iz] =  F; Fr2_out[il,iz]=Fr2
# for iz in range(nz):
# 	ax1.plot(l_lst,B_out[:,iz],color=color_lst[iz],linestyle='-.',label=r"$z'/\lambda={:g} prev$".format(zlst_ratio[iz]));
# 	ax3.plot(l_lst,F_out[:,iz],color=color_lst[iz],linestyle='-.',label=r"$z'/\lambda={:g} prev$".format(zlst_ratio[iz]));
# ax1.legend(); ax3.legend();
##########################################

