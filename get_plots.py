from numpy import sqrt, logspace, log10, zeros, linspace, pi, interp
from Model import J_lambda_prep, Ent_Volume, max_entrainement, Ent_rate_prev, get_rise_speed, Ent_Volume_Z
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
kt=1.238320701922703; et=1.8408136363813399; nu=1e-6; g=9.81; rhoc=1000; sig=0.072
cL,cEta=findcLceta(kt,et,nu,mode=1)
nlst=2000
lst=logspace(-8,3,nlst);	ul2_lst=zeros(nlst) #with dimension!
for i in range(nlst):
	ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
# Load the depth table:
Table=loadmat("UoIEntrainment.mat"); color_lst=['black','red','blue','green']
############################################################################
#=========================      Rising speed      =========================#
############################################################################
# nz=200; nl=3
# l_lst=[1e-2,1e-1,1]; wz_lst=zeros((nz,nl,5))
# almost0=1e-10; almostinf=1e3 # 1000m diameter vortex should be large enough...

# z_lst=logspace(log10(l_lst[0]/2),1,nz)
# for iz in range(nz):
# 	zp=z_lst[iz]
# 	wz_lst[iz,0,1]=get_rise_speed(almost0, almostinf,	kt,et,nu,cL,cEta,lst,ul2_lst,1)
# 	wz_lst[iz,0,2]=get_rise_speed(almost0, 2*zp,		kt,et,nu,cL,cEta,lst,ul2_lst,1)

# for il in range(nl):
# 	l=l_lst[il]; z_lst=logspace(log10(l/2)+1e-10,1,nz)
# 	for iz in range(nz):
# 		zp=z_lst[iz]
# 		wz_lst[iz,il,0]=sqrt(interp(zp-l/2,lst,ul2_lst))/sqrt(2)
# 		wz_lst[iz,il,3]=get_rise_speed(l,       2*zp,		kt,et,nu,cL,cEta,lst,ul2_lst,1)
# 		wz_lst[iz,il,4]=get_rise_speed(l,       2*zp,		kt,et,nu,cL,cEta,lst,ul2_lst,2)
# fig4=plt.figure(figsize=(6,6),dpi=300); plt.subplots_adjust(wspace=0.35,hspace=0.45)
# ax1=fig4.add_subplot(221); ax2=fig4.add_subplot(222); ax3=fig4.add_subplot(223); ax4=fig4.add_subplot(224)

# for il in range(nl):
# 	l=l_lst[il]; z_lst=logspace(log10(l/2),1,nz)
# 	ax1.plot(z_lst,wz_lst[:,il,0],color=color_lst[il],label=r"${:g}$m".format(l))
# 	if il == 0:
# 		ax2.plot(z_lst,wz_lst[:,il,1],color='black',label=r"$l_2=\infty$")
# 		ax2.plot(z_lst,wz_lst[:,il,2],color='black',marker='x',label=r"$l_2=2z'$",markevery=20)
		
# 	ax3.plot(z_lst,wz_lst[:,il,3],color=color_lst[il],label=r"${:g}$m".format(l))
# 	ax4.plot(z_lst,wz_lst[:,il,4],color=color_lst[il],label=r"${:g}$m".format(l))
# ax1.set_xscale('log'); ax1.set_ylabel(r'$w_1$ [m/s]',fontsize=fs); ax1.set_xlabel(r"$z'$ [m]",fontsize=fs);
# ax2.set_xscale('log'); ax2.set_ylabel(r'$w_2$ [m/s]',fontsize=fs); ax2.set_xlabel(r"$z'$ [m]",fontsize=fs); ax2.set_title(r"$l_1=0$",fontsize=fs)
# ax3.set_xscale('log'); ax3.set_ylabel(r'$w_2$ [m/s]',fontsize=fs); ax3.set_xlabel(r"$z'$ [m]",fontsize=fs); ax3.set_title(r"$l_1=\lambda,\ l_2=2z'$",fontsize=fs); 
# ax4.set_xscale('log'); ax4.set_ylabel(r'$w_3$ [m/s]',fontsize=fs); ax4.set_xlabel(r"$z'$ [m]",fontsize=fs); ax4.set_title(r"$l_1=\lambda,\ l_2=2z'$",fontsize=fs); 
# ax1.legend(fontsize=fs,labelspacing=0.15,loc='upper left',title=r"$\lambda$"); ax2.legend(fontsize=fs,labelspacing=0.15)
# ax3.legend(fontsize=fs,labelspacing=0.15,title=r"$\lambda$"); ax4.legend(fontsize=fs,labelspacing=0.15,title=r"$\lambda$")
# ax2.set_ylim([0.068,0.075]); ax3.set_ylim([0.3,1.3]); #ax4.set_ylim([0.068,0.075])
# ax1.set_ylim([0,1]); ax4.set_ylim([0,1]);
# ax1.set_xlim([4e-3,10]); ax2.set_xlim([4e-3,10]); ax3.set_xlim([4e-3,10]); ax4.set_xlim([4e-3,10])

# # ==== Find the terminal rising speed
# nl=200
# l_lst=logspace(log10(5e-3),log10(10),nl)
# wz_lst_term=zeros(nl); ulam_lst=zeros(nl);
# for il in range(nl):
# 	l=l_lst[il]; zp=500
# 	wz_lst_term[il]=get_rise_speed(l,       2*zp,		kt,et,nu,cL,cEta,lst,ul2_lst,1)
# 	ulam_lst[il] = sqrt(interp(l,lst,ul2_lst))
# fig5=plt.figure(figsize=(4,3),dpi=300)
# ax5=fig5.add_subplot(111)
# ax5.plot(l_lst,wz_lst_term,color='black',label=r"$w_2\left(z'\rightarrow \infty, \lambda\right)$")
# ax5.plot(l_lst,ulam_lst,color='red',label=r"$\overline{u}_\lambda\left(\lambda\right)$")
# ax5.set_xscale('log'); ax5.set_ylabel(r'Speed [m/s]'); ax5.set_xlabel(r"$\lambda$ [m]"); ax5.legend()

# # For the program review slides (no previous w_1)
# fig4=plt.figure(figsize=(3,8),dpi=300); plt.subplots_adjust(wspace=0.2,hspace=0.18)
# ax2=fig4.add_subplot(311); ax3=fig4.add_subplot(312); ax4=fig4.add_subplot(313)

# for il in range(nl):
# 	l=l_lst[il]; z_lst=logspace(log10(l/2)+1e-10,1,nz)
# 	if il == 0:
# 		ax2.plot(z_lst,wz_lst[:,il,1],color='black')
# 		ax3.plot(z_lst,wz_lst[:,il,2],color='black')
# 	ax4.plot(z_lst,wz_lst[:,il,3],color=color_lst[il],label=r"$\lambda={:g}$m".format(l))

# ax2.set_xscale('log'); ax2.set_ylabel(r'$w$ [m/s]');  ax2.set_title(r"   $l_1=0,\ l_2=\infty$",loc='left',y=0.83) #ax2.set_xlabel(r"$z'$ [m]");
# ax3.set_xscale('log'); ax3.set_ylabel(r'$w$ [m/s]');  ax3.set_title(r"   $l_1=0,\ l_2=2z'$",loc='left',y=0.83) #ax3.set_xlabel(r"$z'$ [m]");
# ax4.set_xscale('log'); ax4.set_ylabel(r'$w$ [m/s]');  ax4.set_title(r"   $l_1=\lambda,\ l_2=2z'$",loc='left',y=0.83)
# ax4.legend(labelspacing=0.15,loc='lower left',bbox_to_anchor=(0.52,0.12),borderpad=0.1); ax4.set_xlabel(r"$z'$ [m]");
# ax3.set_ylim([0.068,0.075]); ax2.set_ylim([0.068,0.075])
# ax2.set_xlim([4e-3,10]); ax3.set_xlim([4e-3,10]); ax4.set_xlim([4e-3,10])
# # Find the terminal rising speed
# nl=200
# l_lst=logspace(log10(5e-3),log10(10),nl)
# wz_lst_term=zeros(nl); ulam_lst=zeros(nl);
# for il in range(nl):
# 	l=l_lst[il]; zp=500
# 	wz_lst_term[il]=get_rise_speed(l,       2*zp,		kt,et,nu,cL,cEta,lst,ul2_lst)
# 	ulam_lst[il] = sqrt(interp(l,lst,ul2_lst))
# fig5=plt.figure(figsize=(3.3,2.5),dpi=300)
# ax5=fig5.add_subplot(111); ax5.set_title(r"   $l_1=\lambda,\ l_2=2z'$",loc='left',y=0.83)
# ax5.plot(l_lst,wz_lst_term,color='black',label=r"$w\left(z'\rightarrow \infty, \lambda\right)$")
# ax5.plot(l_lst,ulam_lst,color='red',label=r"$\overline{u}_\lambda\left(\lambda\right)$")
# ax5.set_xscale('log'); ax5.set_ylabel(r'Speed [m/s]'); ax5.set_xlabel(r"$\lambda$ [m]"); ax5.legend(fontsize=12)

############################################################################
#========================= Critical Froude number =========================#
############################################################################
# nz=10000; l=1e-3 # l does not matter here
# z_lst=linspace(2*l,3*l,nz);
# fig1=plt.figure(figsize=(3,2),dpi=300)
# color_lst=['black','red','blue']
# style_lst=['-','--','dotted']
# Fr2c_out=zeros(nz);

# ulamsq,Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
# for iz in range(nz):
# 	zp=z_lst[iz]
# 	V_Ent, Fr2_crit, B, F, W, Fr2= Ent_Volume_Z(zp,l,lst,ul2_lst,g,circ_p,Reg,Bog,Weg,Table,mode=1)
# 	Fr2c_out[iz] = Fr2_crit

# ax_1=fig1.add_subplot(111)
# ax_1.plot(z_lst/l,Fr2c_out,color='black')
# ax_1.set_ylabel(r"$\mathrm{Fr}^2_{crt,\Xi}\left( z'/\lambda \right)$ [-]"); ax_1.set_ylim([0,0.4])
# ax_1.set_xlabel(r"$z'/\lambda$ [-]"); ax_1.set_xlim([2,3])

##############################################################################
#====================== Parameters and functions (SDSS) ======================#
##############################################################################
# nz=200; z_st_crt=2.5698; z_lst=logspace(-3,log10(10),nz); l_lst=z_lst/z_st_crt
# Re_out=zeros(nz); B_out=zeros(nz); We_out=zeros(nz); W_out=zeros(nz);
# Fr2_out=zeros(nz); F_out=zeros(nz); Fr2c_out=zeros(nz); 
# for iz in range(nz):
# 	l=l_lst[iz]; zp=z_lst[iz]
# 	ulamsq,Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
# 	Re_out[iz] = Reg; We_out[iz] = Weg
# 	V_Ent, Fr2_crit, B, F, W, Fr2= Ent_Volume_Z(zp,l,lst,ul2_lst,kt,et,nu,cL,cEta,g,circ_p,Reg,Bog,Weg,Table,mode=1)
# 	Fr2_out[iz] = Fr2; B_out[iz] = B; W_out[iz] = W; F_out[iz] = F; Fr2c_out[iz] = Fr2_crit
# fig2=plt.figure(figsize=(7,3),dpi=300); plt.subplots_adjust(wspace=0.55,hspace=0.3)
# ax_1=fig2.add_subplot(231)
# # ax_1.plot(z_lst,70*z_lst/z_lst,  color='black',linestyle='--',label=r'$\mathrm{Re}_\Gamma=70$');
# # ax_1.plot(z_lst,2580*z_lst/z_lst,color='black',linestyle='--',label=r'$\mathrm{Re}_\Gamma=2580$');
# ax_1.plot(z_lst,Re_out,color='black');	
# ax_1.set_xscale('log'); ax_1.set_yscale('log'); ax_1.set_ylabel(r'$\mathrm{Re}_\Gamma$ [-]'); #ax_1.legend(labelspacing=0.15,borderpad=0.2,handletextpad=0.4)
# ax_1.set_xlim([1e-3,10])

# ax_2=fig2.add_subplot(232)
# ax_2.plot(z_lst,We_out,color='black');	
# ax_2.set_xscale('log'); ax_2.set_yscale('log'); ax_2.set_ylabel(r'$\mathrm{We}_\Gamma$ [-]');
# ax_2.set_xlim([1e-3,10])

# ax_3=fig2.add_subplot(233)
# ax_3.plot(z_lst,Fr2_out,color='black');	
# ax_3.set_xscale('log'); ax_3.set_yscale('log'); ax_3.set_ylabel(r'$\mathrm{Fr}^2_\Xi$ [-]');
# ax_3.set_xlim([1e-3,10])

# ax_4=fig2.add_subplot(234)
# ax_4.plot(z_lst,B_out,color='black');	
# ax_4.set_xscale('log'); ax_4.set_yscale('log'); ax_4.set_xlabel(r"$z'$ [m]"); ax_4.set_ylabel(r"$B\left(\mathrm{Re}_\Gamma,z'/\lambda\right)$ [-]");
# ax_4.set_xlim([1e-3,10])

# ax_5=fig2.add_subplot(235)
# ax_5.plot(z_lst,W_out,color='black');	
# ax_5.set_xscale('log'); ax_5.set_yscale('log'); ax_5.set_xlabel(r"$z'$ [m]"); ax_5.set_ylabel(r"$W\left(\mathrm{We}_\Gamma\right)$ [-]");
# ax_5.set_xlim([1e-3,10])

# ax_6=fig2.add_subplot(236)
# ax_6.plot(z_lst,F_out,color='black');	
# ax_6.set_xscale('log'); ax_6.set_yscale('log'); ax_6.set_xlabel(r"$z'$ [m]"); ax_6.set_ylabel(r"$F\left(\mathrm{Fr}^2_\Xi,z'/\lambda\right)$ [-]");
# ax_6.set_xlim([1e-3,10])
# For program review slides (added few depth)
nz=200; nzr = 3; zlr_lst=[2,2.5698,3] #z_st_crt=2.5698; 
Re_out=zeros(nz); B_out=zeros((nz,nzr)); We_out=zeros(nz); W_out=zeros(nz)
Fr2_out=zeros((nz,nzr)); F_out=zeros((nz,nzr)); Fr2c_out=zeros((nz,nzr))
z_lst=logspace(-3,1,nz) 
for izlr in range(nzr):
	zlr=zlr_lst[izlr]
	l_lst=z_lst/zlr
	for iz in range(nz):
		l=l_lst[iz]; zp=z_lst[iz]
		ulamsq,Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
		Re_out[iz] = Reg; We_out[iz] = Weg
		# V_Ent, Fr2_crit, B, F, W, Fr2= Ent_Volume_Z(zp,l,lst,ul2_lst,kt,et,nu,cL,cEta,g,circ_p,Reg,Bog,Weg,Table,mode=1)
		(B, tmp, tmp,
		 W, tmp,
		 F, tmp, tmp)=\
		Calc_Para_Func(depth_lst[i],lam_lst[i],lst,ul2_lst,rhoc,
		               sig,kt,et,nu,cL,cEta,g,
		               Refitcoefs,FrXcoefs,Fr2_lst,zoa_lst,F_tab)
		Fr2_out[iz,izlr] = Fr2; B_out[iz,izlr] = B; W_out[iz] = W; F_out[iz,izlr] = F; Fr2c_out[iz,izlr] = Fr2_crit
fig2=plt.figure(figsize=(6,6),dpi=300); plt.subplots_adjust(wspace=0.4,hspace=0.23)
ax_1=fig2.add_subplot(321)
# ax_1.plot(z_lst,70*z_lst/z_lst,  color='black',linestyle='--',label=r'$\mathrm{Re}_\Gamma=70$');
# ax_1.plot(z_lst,2580*z_lst/z_lst,color='black',linestyle='--',label=r'$\mathrm{Re}_\Gamma=2580$');
ax_1.plot(z_lst,Re_out,color='black');	
ax_1.set_xscale('log'); ax_1.set_yscale('log'); ax_1.set_ylabel(r'$\mathrm{Re}_\Gamma$ [-]'); #ax_1.legend(labelspacing=0.15,borderpad=0.2,handletextpad=0.4)
ax_1.set_xlim([1e-3,10])

ax_2=fig2.add_subplot(323)
ax_2.plot(z_lst,We_out,color='black');	
ax_2.set_xscale('log'); ax_2.set_yscale('log'); ax_2.set_ylabel(r'$\mathrm{We}_\Gamma$ [-]');
ax_2.set_xlim([1e-3,10])

ax_3=fig2.add_subplot(325)
for izlr in range(nzr):
	zlr=zlr_lst[izlr]
	ax_3.plot(z_lst,Fr2_out[:,izlr],color=color_lst[izlr],label='{:g}'.format(zlr))
ax_3.set_xscale('log'); ax_3.set_yscale('log'); ax_3.set_ylabel(r'$\mathrm{Fr}^2_\Xi$ [-]'); ax_3.set_xlabel(r"$z'$ [m]"); 
ax_3.set_xlim([1e-3,10]); ax_3.legend(title=r"$z'/\lambda$",labelspacing=0.15,borderpad=0.2,handletextpad=0.2)

ax_4=fig2.add_subplot(322)
for izlr in range(nzr):
	zlr=zlr_lst[izlr]
	ax_4.plot(z_lst,B_out[:,izlr],color=color_lst[izlr],label='{:g}'.format(zlr))
ax_4.set_xscale('log'); ax_4.set_yscale('log'); ax_4.set_ylabel(r"$B\left(\mathrm{Re}_\Gamma,z'/\lambda\right)$ [-]");
ax_4.set_xlim([1e-3,10]); ax_4.legend(title=r"$z'/\lambda$",labelspacing=0.15,borderpad=0.2,handletextpad=0.2)

ax_5=fig2.add_subplot(324)
ax_5.plot(z_lst,W_out,color='black');	
ax_5.set_xscale('log'); ax_5.set_yscale('log'); ax_5.set_ylabel(r"$W\left(\mathrm{We}_\Gamma\right)$ [-]");
ax_5.set_xlim([1e-3,10])

ax_6=fig2.add_subplot(326)
for izlr in range(nzr):
	zlr=zlr_lst[izlr]
	ax_6.plot(z_lst,F_out[:,izlr],color=color_lst[izlr],label='{:g}'.format(zlr))
ax_6.set_xscale('log'); ax_6.set_yscale('log'); ax_6.set_xlabel(r"$z'$ [m]"); ax_6.set_ylabel(r"$F\left(\mathrm{Fr}^2_\Xi,z'/\lambda\right)$ [-]");
ax_6.set_xlim([1e-3,10]); ax_6.legend(title=r"$z'/\lambda$",labelspacing=0.15,borderpad=0.2,handletextpad=0.2)
##############################################################################
#=============== Parameters and functions (constant z/lambda) ===============#
##############################################################################
# fig4=plt.figure(figsize=(7,3),dpi=300); plt.subplots_adjust(wspace=0.35,hspace=0.3)
# ax_1=fig4.add_subplot(121)
# ax_2=fig4.add_subplot(122)
# nz=200; nl=3; l_lst=[1e-2,1e-1,1]
# Fr2_out=zeros((nz,nl)); F_out=zeros((nz,nl)); Fr2c_out=zeros((nz,nl))
# for il in range(nl):
# 	l=l_lst[il]; z_lst=logspace(log10(l/2+1e-10),1,nz)
# 	ulamsq,Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
# 	for iz in range(nz):
# 		zp=z_lst[iz]
# 		V_Ent, Fr2_crit, B, F, W, Fr2= Ent_Volume_Z(zp,l,lst,ul2_lst,kt,et,nu,cL,cEta,g,circ_p,Reg,Bog,Weg,Table,mode=1)
# 		Fr2_out[iz,il] = Fr2; F_out[iz,il] = F; Fr2c_out[iz,il] = Fr2_crit
# 	ax_1.plot(z_lst,Fr2_out[:,il],color=color_lst[il],label=r"$\lambda={:g}$m".format(l))
# 	ax_2.plot(z_lst,F_out[:,il],  color=color_lst[il],label=r"$\lambda={:g}$m".format(l))
# 	ax_1.plot([l/2,l/2],[0,Fr2_out[:,il].max()],	color=color_lst[il],linestyle='--')	
# 	ax_2.plot([l/2,l/2],[0,F_out[:,il].max()],		color=color_lst[il],linestyle='--')

# ax_1.set_xscale('log'); ax_1.set_ylabel(r'$\mathrm{Fr}^2_\Xi$ [-]')
# ax_1.set_xlim([1e-3,10]); ax_1.legend(loc='center',bbox_to_anchor=(0.75,0.65)); ax_1.set_xlabel(r"$z'$ [m]")
# ax_1.set_ylim([0,16])
# ax_2.set_xscale('log'); ax_2.set_ylabel(r"$F\left(\mathrm{Fr}^2_\Xi,z'/\lambda\right)$ [-]")
# ax_2.set_xlim([1e-3,10]); ax_2.legend(loc='center',bbox_to_anchor=(0.75,0.55)); ax_2.set_xlabel(r"$z'$ [m]")
# ax_2.set_ylim([0,6])
##############################################################################
#=========================    Entrainment Volume    =========================#
##############################################################################
# nz=200; z_st_crt=[2.5698,2.5698*1.05,2.5698*1.10]; z_lst=logspace(-3,2,nz);
# Q_out=zeros((nz, 3)); Q_dot_out=zeros((nz, 3));
# for iratio in range(3):
# 	l_lst=z_lst/z_st_crt[iratio]
# 	for iz in range(nz):
# 		zp=z_lst[iz]; l=l_lst[iz]
# 		ulamsq,Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
# 		V_Ent, Fr2_crit, B, F, W, Fr2= Ent_Volume_Z(zp,l,lst,ul2_lst,kt,et,nu,cL,cEta,g,circ_p,Reg,Bog,Weg,Table,mode=1)
# 		# Q_out[iz,iratio] = V_Ent/(pi*l**3/6.0); Q_dot_out[iz,iratio] = V_Ent/tau_vort/(pi*l**3/6.0);
# 		Q_out[iz,iratio] = V_Ent; Q_dot_out[iz,iratio] = V_Ent/tau_vort;
# fig3=plt.figure(figsize=(7,3),dpi=300); plt.subplots_adjust(wspace=0.35,hspace=0.3)
# ax_7=fig3.add_subplot(121)
# for iratio in range(3):
# 	ax_7.plot(z_lst,Q_out[:,iratio],color=color_lst[iratio],label=r"$z'/\lambda={:g}$".format(z_st_crt[iratio]));	
# ax_7.set_xscale('log'); ax_7.set_yscale('log'); ax_7.set_xlabel(r"$z'$ [m]")
# # ax_7.set_ylabel(r'$\forall/V_\lambda \ \mathrm{[-]}$');
# ax_7.set_ylabel(r'$\forall \ \mathrm{[m^3]}$');
# ax_7.set_xlim([1e-3,100]); ax_7.legend(labelspacing=0.15,borderpad=0.2,handletextpad=0.4,loc='lower center',bbox_to_anchor=(0.6,0))
# ax_8=fig3.add_subplot(122)
# for iratio in range(3):
# 	ax_8.plot(z_lst,Q_dot_out[:,iratio],color=color_lst[iratio],label=r"$z'/\lambda={:g}$".format(z_st_crt[iratio]));	
# ax_8.set_xscale('log'); ax_8.set_yscale('log'); ax_8.set_xlabel(r"$z'$ [m]")
# # ax_8.set_ylabel(r'$Q_\lambda/V_\lambda \ \mathrm{[1/s]}$');
# ax_8.set_ylabel(r'$Q_\lambda \ \mathrm{[m^3/s]}$');
# ax_8.set_xlim([1e-3,100]); ax_8.legend(labelspacing=0.15,borderpad=0.2,handletextpad=0.4,loc='lower center',bbox_to_anchor=(0.6,0))