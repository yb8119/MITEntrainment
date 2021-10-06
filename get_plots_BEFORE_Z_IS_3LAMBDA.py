from numpy import logspace, log10, zeros, linspace
from Model import J_lambda_prep, Ent_Volume, max_entrainement, Ent_rate_prev
from Utilities import findcLceta, ulambda_sq
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
fs=12

##########################
#   PHYSICAL PARAMETERS  #
##########################
kt=1.23; et=1.83; nu=1e-6; g=9.81; rhoc=1000; sig=0.072
cL,cEta=findcLceta(kt,et,nu,mode=1)
nlst=1500
lst=logspace(-8,2,nlst);	ul2_lst=zeros(nlst) #with dimension!
#================== Entrainment volume/rate vs lambda ==================
# nl=20
# l_lst=linspace(1e-4,5,nl);
# # l_lst=logspace(-3,log10(3),nl);
# Q_lst=zeros(nl); Q_dot_lst=zeros(nl); Q_dot_lst_old=zeros(nl); zp_lst=zeros(nl); ze_lst=zeros(nl)
# for il in range(nl):
# 	l=l_lst[il]
# 	for i in range(nlst):
# 		ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
# 	ulamsq,Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
# 	zp=3*l
# 	Vmax=max_entrainement(l,ulamsq,kt,et,cL,cEta,nu,g,rhoc,sig)
# 	Q_lst[il]=Ent_Volume(zp,l,lst,ul2_lst,Reg,Bog,Weg,kt,et,cL,cEta,nu,g,circ_p,Vmax)
# 	Q_dot_lst[il]=Q_lst[il]/tau_vort
# 	Q_dot_lst_old[il],zp_lst[il],ze_lst[il]=Ent_rate_prev(l,zp,kt,et)

# fig=plt.figure(figsize=(7,3),dpi=200)
# plt.subplots_adjust(wspace=0.35,hspace=0.25)
# fig.suptitle(r"$k_t={:.2f}m^2/s^2, \ \varepsilon={:.2f}m^2/s^3, \ z'=3\lambda$".format(kt,et),fontsize=fs,y=0.95)
# axa=fig.add_subplot(121)
# axb=fig.add_subplot(122)
# axa.plot(l_lst, Q_lst,color='black')
# axb.plot(l_lst, Q_dot_lst,color='black')
# axb.plot(l_lst, Q_dot_lst_old,color='black', linestyle='--')
# # axb.set_yscale('log')
# axa.set_xlabel(r'$\lambda \ [m]$ ' ); axb.set_xlabel(r'$\lambda \ [m]$ ' )
# axa.set_ylabel(r'$\forall\ [m^3]$' ); axb.set_ylabel(r'$\dot \forall \ [m^3/s]$ ' )


#================== Entrainment volume/rate vs depth ==================
nz=200; nl=3
l_lst=[0.05,0.1,0.5];
Q_lst=zeros((nl,nz)); Q_dot_lst=zeros((nl,nz)); Q_dot_lst_old=zeros((nl,nz)); Q_dot_lst_cmp=zeros((nl,nz));
fig=plt.figure(figsize=(7,3),dpi=200)
color_lst=['black','red','blue']
# plt.subplots_adjust(wspace=0.35,hspace=0.25)
fig.suptitle(r"$k_t={:.2f}m^2/s^2, \ \varepsilon={:.2f}m^2/s^3$".format(kt,et),fontsize=fs,y=0.97)
axa=fig.add_subplot(121)
axb=fig.add_subplot(122)
pllsst=[]
for i in range(nlst):
	ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
for il in range(nl):
	l=l_lst[il]; z_lst=linspace(0.5*l+1e-10,8,nz); z_lst2=linspace(0.5*l+1e-10,0.5,nz);
	# get ze for old model
	tmp,ze=Ent_rate_prev(l,l,kt,et)
	z_lst_old=linspace(0.5*l+1e-10,ze,nz);
	ulamsq,Reg,Bog,Weg,circ_p,n_lam,x,tau_vort=J_lambda_prep(l,lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig)
	for iz in range(nz):
		zp=z_lst[iz]; zp_old=z_lst_old[iz]; zp2=z_lst2[iz]
		Vmax=max_entrainement(l,ulamsq,kt,et,cL,cEta,nu,g,rhoc,sig)
		Q_lst[il,iz]=Ent_Volume(zp,l,lst,ul2_lst,Reg,Bog,Weg,kt,et,cL,cEta,nu,g,circ_p,Vmax)
		Q_dot_lst[il,iz]=Q_lst[il,iz]/tau_vort

		Q_tmp=Ent_Volume(zp2,l,lst,ul2_lst,Reg,Bog,Weg,kt,et,cL,cEta,nu,g,circ_p,Vmax)
		Q_dot_lst_cmp[il,iz]=Q_tmp/tau_vort

		Q_dot_lst_old[il,iz],tmp=Ent_rate_prev(l,zp_old,kt,et)
	pl,=axa.plot(z_lst,     Q_dot_lst[il,:], 	 color=color_lst[il]); pllsst.append(pl)
	axb.plot(z_lst2,        Q_dot_lst_cmp[il,:], color=color_lst[il])
	pl,=axb.plot(z_lst_old, Q_dot_lst_old[il,:], color=color_lst[il], linestyle='-.'); pllsst.append(pl)
	# axa.plot(z_lst, Q_dot_lst[il,:], 	 linestyle = '-',  color=color_lst[il], label=r'This work, $\lambda={:.2g}m$'.format(l))
	# axa.plot(z_lst, Q_dot_lst_old[il,:], linestyle = '--', color=color_lst[il], label=r'Castro et al. (2016), $\lambda={:.2g}m$'.format(l))
# axb.set_yscale('log')
axa.set_xlabel(r'$z \ [m]$ ' );axb.set_xlabel(r'$z \ [m]$ ' );axa.set_ylabel(r'$\dot Q [m^3/s]$' );axa.set_yscale('log'); axb.set_yscale('log')
leg=axa.legend([(pllsst[0],pllsst[2],pllsst[4])],
               ["$\lambda={:.2g}, {:.2g}, {:.2g}m$ \n This work".format(l_lst[0],l_lst[1],l_lst[2])],
                numpoints=1,handler_map={tuple: HandlerTuple(ndivide=None)},\
                handlelength=7,fontsize=fs-2.5,frameon=True,labelspacing=0.2,loc='upper right')

leg=axb.legend([(pllsst[1],pllsst[3],pllsst[5])],
               ["$\lambda={:.2g}, {:.2g}, {:.2g}m$ \n Castro et al. (2016)".format(l_lst[0],l_lst[1],l_lst[2])],
                numpoints=1,handler_map={tuple: HandlerTuple(ndivide=None)},\
                handlelength=7,fontsize=fs-2.5,frameon=True,labelspacing=0.2,loc='upper right')
axb.set_ylim([3e-5,15])
axa.set_ylim([1e-6,10])
axa.set_xlim([0,8])
axb.set_xlim([0,0.5])
# for t in leg.texts:
#     t.set_multialignment('center')
