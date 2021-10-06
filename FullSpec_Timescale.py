from numpy import logspace, log10, zeros, sqrt, pi
import matplotlib.pyplot as plt
from Utilities import findcLceta, ulambda_sq
#############
# nu =1e-6; g=9.81; sig=0.072;rhoc=1000;nu=1e-6
# kt_lst = [0.25,1.0,1.75] #m2/s2
# et = 3.32 #m2/s3
# fig=plt.figure(figsize=(4,3),dpi=300)
# ax=fig.add_subplot(111);
# for ik in range(3):
# 	kt = kt_lst[ik]
# 	cL,cEta=findcLceta(kt,et,nu,mode=1); L=kt**1.5/et
# 	nlst=1500; x1=sqrt(4*sig/rhoc/g); x4=20
# 	lst=logspace(log10(x1),log10(x4),nlst);	ul2_lst=zeros(nlst) #with dimension!
# 	for i in range(nlst):
# 		ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
# 	tao_lst = 0.5*ul2_lst/et
# 	ax.plot(lst,tao_lst, label = r"$k_t = {:}".format(kt)+r" \mathrm{[m^2/s^2]}$")

# ax.plot(lst,lst**(2.0/3.0)/et**(1.0/3.0),label="Kolmogorov")
# ax.legend(); ax.set_xscale("Log"); ax.set_yscale("Log"); ax.set_xlabel(r"$\lambda \mathrm{[m]}$")
# ax.set_ylabel(r"$\tau \, \mathrm{[s]}$")
# ax.set_title(r"$\epsilon={}$".format(et)+r" $\mathrm{[m^2/s^3]}$")
#############
nu =1e-6; g=9.81; sig=0.072;rhoc=1000;nu=1e-6
kt_lst = [0.25,1.0,1.75] #m2/s2
et = 3.32 #m2/s3
fig=plt.figure(figsize=(4,3),dpi=300)
ax=fig.add_subplot(111);
for ik in range(3):
	kt = kt_lst[ik]
	cL,cEta=findcLceta(kt,et,nu,mode=1); L=kt**1.5/et
	nlst=1500; x1=sqrt(4*sig/rhoc/g); x4=20
	lst=logspace(log10(x1),log10(x4),nlst);	ul2_lst=zeros(nlst) #with dimension!
	for i in range(nlst):
		ul2_lst[i]=ulambda_sq(lst[i],kt,et,cL,cEta,nu,pope_spec=1.01)
	tao_lst = 0.5*ul2_lst/et
	ax.plot(lst,tao_lst/(lst**(2.0/3.0)/et**(1.0/3.0)), label = r"$k_t = {:}".format(kt)+r" \mathrm{[m^2/s^2]}$")
ax.legend(); ax.set_xscale("Log"); ax.set_yscale("Log"); ax.set_xlabel(r"$\lambda \mathrm{[m]}$")
ax.set_ylabel(r"$\tau/\tau_{Kolmogorov} \, \mathrm{[-]}$")
ax.set_title(r"$\epsilon={}$".format(et)+r" $\mathrm{[m^2/s^3]}$")