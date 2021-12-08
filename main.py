from numpy import zeros, linspace, loadtxt, interp, insert, savez, load
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Ent_body import Turb_entrainment
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
fs=12

####   EVERYTHING IS DIMENSIONAL   ####
# Input parameters
model = 2
Sl0 = 1.9e-2
L0  = 47 				#Athena Ship Length
U0  = 10.5 * 0.514444	# 10.5 knot speed
#######################################################
# Phyical properties    #Numerical parameters
rhoc = 1000;            zmax = 1.5
rhod = 1.204;           Tend = 60
nuc = 1.0e-6;           Nx = 45
sigma = 0.072;          Nt = 200
g = 9.81
z = linspace(0,zmax,Nx)   #Domain
dt = Tend/Nt
zlam_min = 2; zlam_max = 100000
############################################################
#Read groups info
ds = loadtxt('groups.dat', skiprows=1)
D =ds[:,0]*L0*2	#Diameters
Dg=ds[:,1]		#Entrainment distribution
G = D.size
turb_data = loadtxt('stern_turbulence.dat',skiprows=18);
kt  = -1*interp(-1*z,-1*insert(turb_data[:,5]*L0,0,10), -1*insert(turb_data[:,6] ,0,turb_data[0,6] ));
et  = -1*interp(-1*z,-1*insert(turb_data[:,5]*L0,0,10), -1*insert(turb_data[:,11],0,turb_data[0,11]));
nut = -1*interp(-1*z,-1*insert(turb_data[:,5]*L0,0,10), -1*insert(turb_data[:,8] ,0,turb_data[0,8] ));
# kt=zeros(Nx); 	kt[:]  = 1.2346934417890287; et=zeros(Nx); 	et[:]  = 1.8320709103157145; nut=zeros(Nx); 	nut[:] = 0.07489538904880445
Fr2_crt_PolyExtra = False
F_tab_NP = False # Using Nearest-Point for F table
wmeth=2			# rising speed model
# Load the depth table:
# Table=loadmat("UoIEntrainment.mat");
if F_tab_NP:
	print("========  Using Original Table (no extrapolation)  ========")
	Table=load("Ent_table_org.npz")
else:
	# print("========  Using Nearest-Point-Extrapolated Table  ========")
	# Table=load("Ent_table_NP.npz");
	print("========  Using Linearly-Extrapolated Table  ========")
	Table=load("Ent_table_LE.npz");
print(" >> z_c/(0.5*lam) dimension: {:}, range: {:}, {:}".format(Table['z_a_data'].size,   Table['z_a_data'].min(),    Table['z_a_data'].max()))
print(" >> Fr2           dimension: {:}, range: {:}, {:}".format(Table['flxfr_data'].size, Table['flxfr_data'].min(),  Table['flxfr_data'].max()))

# # Castro et al. 2016
# time, alpha_out1, alpha01, alpha_mid1, S1, VT = \
# 	Turb_entrainment(Nx,Nt,kt,et,g,rhoc,rhod,sigma,z,nuc,nut,G,D,Dg,dt,Sl0,Table,
# 					 1,zlam_min,zlam_max,wmeth,Fr2_crt_PolyExtra,F_tab_NP) #rising speed method not used
# # New all the DNS data
# time, alpha_out4, alpha04, alpha_mid4, S4, VT = \
# 	Turb_entrainment(Nx,Nt,kt,et,g,rhoc,rhod,sigma,z,nuc,nut,G,D,Dg,dt,Sl0,Table,
# 					 2,zlam_min,zlam_max,wmeth,Fr2_crt_PolyExtra,F_tab_NP)

# ======================================== Converge Validation ========================================
zlam_max_range=[3,30,300,3000]
alpha_d_convg=[]
for irange in range(4):
	time, alpha_out4, alpha04, alpha_mid4, S4, VT = \
	Turb_entrainment(Nx,Nt,kt,et,g,rhoc,rhod,sigma,z,nuc,nut,G,D,Dg,dt,Sl0,Table,
					 2,zlam_min,zlam_max_range[irange],wmeth,Fr2_crt_PolyExtra,F_tab_NP)
	alpha_d_convg.append(alpha_out4)
# === plot part ===
Terrill = [[1.07, 0.96, 0.81, 0.562, 0.435, 0.309, 0.167, 0.053],
           [0.0298, 0.0282, 0.0366, 0.697, 6.32, 11.2, 13, 18.5]]
Johansen = [[0.1016, 0.4064, 0.5588, 0.7112, 0.1016, 0.4064, 0.7112], 
			[11.65, 3.54244, 0.8087, 0.03, 12.475, 4.101, 2.474]]
color_plate = ['red', 'green', 'blue', 'black']
# ; plt.subplots_adjust(wspace=0.3,hspace=0.4)
fig_convg = plt.figure(figsize=(4, 3), dpi=300)
axconvg = fig_convg.add_subplot(111)
axconvg.set_title(r"$\left(z/\lambda\right)_{min} =$"+"{:}".format(zlam_min))
axconvg.plot(Terrill[0], Terrill[1], color='black',
             linestyle='none', marker='o', ms=5, label='Terrill (2005)')
axconvg.plot(Johansen[0], Johansen[1], color='black', linestyle='none',
             marker='o', mfc='none', ms=5, label='Johansen et al. (2010)')
axconvg.set_xlabel('Depth [m]')

axconvg.set_ylabel(r'$\alpha_d$ [%]')
axconvg.set_xlim([min(z), max(z)])
axconvg.set_ylim([-1, 30])
for irange in range(4):
	axconvg.plot(z, alpha_d_convg[irange][:, Nt-1]*100,
              color=color_plate[irange], label=r"$\left(z/\lambda\right)_{max}=$"+"{}".format(zlam_max_range[irange]))
axconvg.legend(labelspacing=0.15, borderpad=0.2, handletextpad=0.4,
               loc='center', bbox_to_anchor=(0.67, 0.72))
#====================================================================================================
for i in range(Nx):
	if z[i] > 0.5 :
		nut[i] = nuc
#================================== Turb Condition (in 1 figure) ==================================
# fig0=plt.figure(figsize=(5,5),dpi=300); plt.subplots_adjust(wspace=0.4,hspace=0.35)
# ax01=fig0.add_subplot(221); ax02=fig0.add_subplot(222)
# ax03=fig0.add_subplot(223); ax04=fig0.add_subplot(224)
# ax01.plot(z,kt,color='black');		ax01.set_xlim([0,1.5]);			ax01.set_ylim([0,2])
# ax02.plot(z,et,color='black');		ax02.set_xlim([0,1.5]);			ax02.set_ylim([0,8])
# ax03.plot(z,nut,color='black');		ax03.set_xlim([0,1.5]);			ax03.set_ylim([0,0.08])
# ax04.plot(1000*D,VT,color='black');	ax04.set_xlim([0,max(1000*D)]);	ax04.set_ylim([0,0.25])
# ax01.set_xlabel('Depth [m]'); ax02.set_xlabel('Depth [m]')
# ax03.set_xlabel('Depth [m]'); ax04.set_xlabel('Diameter [mm]')
# ax01.set_ylabel(r'$\mathrm{TKE} \ \mathrm{[m^2/s^2]}$',fontsize=fs);    ax02.set_ylabel(r'$\epsilon \ \mathrm{[m^2/s^3]}$',fontsize=fs)
# ax03.set_ylabel(r'$\nu_t \ \mathrm{[m^2/s]}$',fontsize=fs);    ax04.set_ylabel('Terminal velocity [m/s]',fontsize=fs)
#================================== Turb Condition (PLOT in Seperate figures for program review) ==================================
# fig01=plt.figure(figsize=(2,2),dpi=300); ax01=fig01.add_subplot(111)
# fig02=plt.figure(figsize=(2,2),dpi=300); ax02=fig02.add_subplot(111)
# fig03=plt.figure(figsize=(2,2),dpi=300); ax03=fig03.add_subplot(111)
# ax01.plot(z,kt,color='black');		ax01.set_xlim([0,1.5]);			ax01.set_ylim([0,2]);	ax01.set_xlabel('Depth [m]');	ax01.set_ylabel(r'$\mathrm{TKE} \ \mathrm{[m^2/s^2]}$',fontsize=fs)
# ax02.plot(z,et,color='black');		ax02.set_xlim([0,1.5]);			ax02.set_ylim([0,8]);	ax02.set_xlabel('Depth [m]');	ax02.set_ylabel(r'$\epsilon \ \mathrm{[m^2/s^3]}$',fontsize=fs)
# ax03.plot(z,nut,color='black');		ax03.set_xlim([0,1.5]);			ax03.set_ylim([0,0.08]);ax03.set_xlabel('Depth [m]');	ax03.set_ylabel(r'$\nu_t \ \mathrm{[m^2/s]}$',fontsize=fs)
#================================== Ad at free surface and mid-depth ==================================
# fig1=plt.figure(figsize=(7,3),dpi=300); plt.subplots_adjust(wspace=0.3,hspace=0.4)
# ax1=fig1.add_subplot(121); ax2=fig1.add_subplot(122)
# ax1.plot(time,alpha01,		color='black',	label='Castro et al. 2016')
# ax2.plot(time,alpha_mid1,	color='black',	label='Castro et al. 2016')
# ax1.plot(time,alpha02,		color='red',	label='C=1.05')
# ax2.plot(time,alpha_mid2,	color='red',	label='C=1.05')
# ax1.plot(time,alpha03,		color='blue',	label='C=1.10')
# ax2.plot(time,alpha_mid3,	color='blue',	label='C=1.10')
# ax1.set_xlabel('Time [s]'); ax2.set_xlabel('Time [s]');
# ax1.legend(labelspacing=0.15,borderpad=0.2,handletextpad=0.4)
# ax2.legend(labelspacing=0.15,borderpad=0.2,handletextpad=0.4)
# ax1.set_ylabel(r'$\alpha_d\left( z=0 \right)$'); ax2.set_ylabel(r'$\alpha_d\left( z={:.1f}m \right)$'.format(z[int(Nx/2)]));
#================================== Ad as a function depth ==================================
# fig2=plt.figure(figsize=(7,3),dpi=300); plt.subplots_adjust(wspace=0.3,hspace=0.4)
# ax3=fig2.add_subplot(121);  ax4=fig2.add_subplot(122);
# ax3.plot(z,alpha_out1[:,int(Nt/2)-1], color='black', linestyle='-',
# 		 label=r'$t={:.1f}s$, Castro et al. 2016'.format(time[int(Nt/2)-1]))
# ax3.plot(z,alpha_out1[:,Nt-1],      color='black', linestyle='--',
# 		 label=r'$t={:.1f}s$, Castro et al. 2016'.format(time[Nt-1]))
# ax4.plot(z,alpha_out2[:,int(Nt/2)-1], color='red', linestyle='-',
# 		 label=r'$t={:.1f}s$, C=1.05'.format(time[int(Nt/2)-1]))
# ax4.plot(z,alpha_out2[:,Nt-1],      color='red', linestyle='--',
# 		 label=r'$t={:.1f}s$, C=1.05'.format(time[Nt-1]))
# ax4.plot(z,alpha_out3[:,int(Nt/2)-1], color='blue', linestyle='-',
# 		 label=r'$t={:.1f}s$, C=1.10'.format(time[int(Nt/2)-1]))
# ax4.plot(z,alpha_out3[:,Nt-1],      color='blue', linestyle='--',
# 		 label=r'$t={:.1f}s$, C=1.10'.format(time[Nt-1]))
# ax4.plot(z,alpha_out4[:,int(Nt/2)-1], color='green', linestyle='-',
# 		 label=r'$t={:.1f}s$, All range'.format(time[int(Nt/2)-1]))
# ax4.plot(z,alpha_out4[:,Nt-1],      color='green', linestyle='--',
# 		 label=r'$t={:.1f}s$, All range'.format(time[Nt-1]))

# ax3.set_xlabel('Depth [m]'); ax4.set_xlabel('Depth [m]');
# ax3.legend(labelspacing=0.15,borderpad=0.2,handletextpad=0.4,loc='center',bbox_to_anchor=(0.5, 1.1))
# ax4.legend(labelspacing=0.15,borderpad=0.2,handletextpad=0.4,loc='center',bbox_to_anchor=(0.5, 1.15),ncol=2)
# ax3.set_ylabel(r'$\alpha_d$'); ax4.set_ylabel(r'$\alpha_d$');
# ax3.set_xlim([min(z),max(z)]); ax4.set_xlim([min(z),max(z)]);
# ax3.set_ylim([0,0.16]); ax4.set_ylim([0,0.16]);
# ================================== For program review ==================================
# Terrill=[[1.07,0.96,0.81,0.562,0.435,0.309,0.167,0.053],[0.0298,0.0282,0.0366,0.697,6.32,11.2,13,18.5]]
# Johansen=[[0.1016,0.4064,0.5588,0.7112,0.1016,0.4064,0.7112]
# ,[11.65,3.54244,0.8087,0.03,12.475,4.101,2.474]]
# fig2=plt.figure(figsize=(3,3),dpi=300)#; plt.subplots_adjust(wspace=0.3,hspace=0.4)
# ax3=fig2.add_subplot(111); ax3.set_title(r"$\left(z/\lambda\right)_{min} =$"+"{:}".format(zlam_min)+ ", "+\
#                                          r"$\left(z/\lambda\right)_{max} =$"+"{:}".format(zlam_max))
# ax3.plot(z,alpha_out1[:,Nt-1]*100,      color='black', linestyle='-',
# 		 # label=r'$t={:.1f}s$, Castro et al. (2016)'.format(time[Nt-1]))
# 		 label=r'Castro et al. (2016)')
# ax3.plot(z,alpha_out4[:,Nt-1]*100,      color='red', linestyle='-',
# 		 # label=r'$t={:.1f}s$, This work'.format(time[Nt-1]))
# 		 label=r'This work')
# ax3.plot(Terrill[0],Terrill[1],		color='black', linestyle='none', marker='o', ms=5,
# 		 label='Terrill (2005)')
# ax3.plot(Johansen[0],Johansen[1],	color='black', linestyle='none', marker='o', mfc='none', ms=5,
# 		 label='Johansen et al. (2010)')
# ax3.set_xlabel('Depth [m]')
# ax3.legend(labelspacing=0.15,borderpad=0.2,handletextpad=0.4,loc='center',bbox_to_anchor=(0.57, 0.82))
# ax3.set_ylabel(r'$\alpha_d$ [%]')
# ax3.set_xlim([min(z),max(z)])
# # ax3.set_ylim([0,20])

# fig3=plt.figure(figsize=(3,3),dpi=300)
# ax4=fig3.add_subplot(111); ax4.set_title(r"$\left(z/\lambda\right)_{min} =$"+"{:}".format(zlam_min)+ ", "+\
#                                          r"$\left(z/\lambda\right)_{max} =$"+"{:}".format(zlam_max))
# ax4.plot(z,S4,label='This work',color='red')
# ax4.plot(z,S1,label='Castro et al. (2016)',color='black')
# ax4.legend(); ax4.set_xlim([0,1.5]); ax4.set_xlabel('Depth [m]'); ax4.set_ylabel(r'$S\ \mathrm{[1/s]}$ ')

# fo=open("0D_out.npz","wb")
# savez(fo,
#       z=z,
#       alpha_out1=alpha_out1[:,Nt-1],
#       alpha_out4=alpha_out4[:,Nt-1],
#       S4=S4,
#       S1=S1,
#       kt=kt,
#       et=et)
# fo.close()



# fig2=plt.figure(figsize=(3.5,3.3),dpi=300)#; plt.subplots_adjust(wspace=0.3,hspace=0.4)
# ax3=fig2.add_subplot(111)
# ax3.plot(z,alpha_out1[:,Nt-1]*100,      color='black', linestyle='-',
# 		 # label=r'$t={:.1f}s$, Castro et al. (2016)'.format(time[Nt-1]))
# 		 label=r'Castro et al. (2016)')
# ax3.plot(z,alpha_out4[:,Nt-1]*100,      color='red', linestyle='-',
# 		 # label=r'$t={:.1f}s$, This work'.format(time[Nt-1]))
# 		 label=r'This work $\lambda_{max}=100L$')
# ax3.plot(z,alpha_out4_10lam[:,Nt-1]*100,      color='red', linestyle='-.',
# 		 # label=r'$t={:.1f}s$, This work'.format(time[Nt-1]))
# 		 label=r'This work $\lambda_{max}=10L$')
# ax3.plot(Terrill[0],Terrill[1],		color='black', linestyle='none', marker='o', ms=5,
# 		 label='Terrill (2005)')
# ax3.plot(Johansen[0],Johansen[1],	color='black', linestyle='none', marker='o', mfc='none', ms=5,
# 		 label='Johansen et al. (2010)')
# ax3.set_xlabel('Depth [m]')
# ax3.legend(labelspacing=0.15,borderpad=0.2,handletextpad=0.4,loc='center',bbox_to_anchor=(0.60, 0.8),frameon=False)
# ax3.set_ylabel(r'$\alpha_d$ [%]')
# ax3.set_xlim([min(z),max(z)])
# ax3.set_ylim([0.0,20])
