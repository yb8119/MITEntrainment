from numpy import logspace, linspace, zeros
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
fs=12
plt.rcParams.update({'font.size': fs})
def_sz=200;
Table=loadmat("UoIEntrainment.mat")
z_lam_ratio_l=linspace(0.5,2,def_sz);
z_lam_ratio_m=linspace(2,3,def_sz);
z_lam_ratio_r=linspace(3,6,def_sz);
Fr2_crt_list_l=zeros((def_sz,2));
Fr2_crt_list_m=zeros(def_sz);
Fr2_crt_list_r=zeros((def_sz,2));
Fr2_crt_extr=zeros(def_sz);
########### Fr2 plot ###########
fig=plt.figure(figsize=(4,3),dpi=300)
ax=fig.add_subplot(111)
FrXcoefs=Table['FrXcoefs'][0]
for i in range(def_sz):
	# Critical Fr2
	zcoa = -1 * z_lam_ratio_m[i]*2
	zcoa_scl=(zcoa-FrXcoefs[7])/FrXcoefs[8]
	Fr2_crt_list_m[i]=FrXcoefs[0]*zcoa_scl**6+FrXcoefs[1]*zcoa_scl**5+FrXcoefs[2]*zcoa_scl**4+\
	FrXcoefs[3]*zcoa_scl**3+FrXcoefs[4]*zcoa_scl**2+FrXcoefs[5]*zcoa_scl + FrXcoefs[6]
for i in range(def_sz):
	zcoa = -1 * z_lam_ratio_l[i]*2
	zcoa_scl=(zcoa-FrXcoefs[7])/FrXcoefs[8]
	zcoa_l=(-4-FrXcoefs[7])/FrXcoefs[8]
	dzcoa = zcoa_scl-zcoa_l
	Fr2_crt_list_l[i,0] = Fr2_crt_list_m[0]+(FrXcoefs[0]*zcoa_l**5*6+FrXcoefs[1]*zcoa_l**4*5+FrXcoefs[2]*zcoa_l**3*4+\
			FrXcoefs[3]*zcoa_l**2*3+FrXcoefs[4]*zcoa_l*2+FrXcoefs[5])*dzcoa
	Fr2_crt_list_l[i,1] = FrXcoefs[0]*zcoa_scl**6+FrXcoefs[1]*zcoa_scl**5+FrXcoefs[2]*zcoa_scl**4+\
	FrXcoefs[3]*zcoa_scl**3+FrXcoefs[4]*zcoa_scl**2+FrXcoefs[5]*zcoa_scl + FrXcoefs[6]

	zcoa = -1 * z_lam_ratio_r[i]*2
	zcoa_scl=(zcoa-FrXcoefs[7])/FrXcoefs[8]
	zcoa_r=(-6-FrXcoefs[7])/FrXcoefs[8]
	dzcoa = zcoa_scl-zcoa_r
	Fr2_crt_list_r[i,0] = Fr2_crt_list_m[def_sz-1]+(FrXcoefs[0]*zcoa_r**5*6+FrXcoefs[1]*zcoa_r**4*5+FrXcoefs[2]*zcoa_r**3*4+\
			FrXcoefs[3]*zcoa_r**2*3+FrXcoefs[4]*zcoa_r*2+FrXcoefs[5])*dzcoa
	Fr2_crt_list_r[i,1] = FrXcoefs[0]*zcoa_scl**6+FrXcoefs[1]*zcoa_scl**5+FrXcoefs[2]*zcoa_scl**4+\
	FrXcoefs[3]*zcoa_scl**3+FrXcoefs[4]*zcoa_scl**2+FrXcoefs[5]*zcoa_scl + FrXcoefs[6]

ax.plot(z_lam_ratio_m,Fr2_crt_list_m,color="black")
ax.plot(z_lam_ratio_l,Fr2_crt_list_l[:,0],color="red",linestyle="-.")
ax.plot(z_lam_ratio_r,Fr2_crt_list_r[:,0],color="red",linestyle="-.",label="Extrapolation")
ax.plot(z_lam_ratio_l,Fr2_crt_list_l[:,1],color="red",linestyle="-")
ax.plot(z_lam_ratio_r,Fr2_crt_list_r[:,1],color="red",linestyle="-",label="Polynomial")
# ax.plot([2,2],[ax.get_ylim()[0],ax.get_ylim()[1]],color="black",linestyle="--")
# ax.plot([3,3],[ax.get_ylim()[0],ax.get_ylim()[1]],color="black",linestyle="--")
ax.set_xlabel(r"$z'/\lambda$ [-]")
ax.legend()
ax.set_xlim([0.5,6])
ax.set_ylabel(r"$\mathrm{Fr}^2_{crt,\Xi}$ [-]")
ax.set_yscale("log")