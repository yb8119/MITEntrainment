from numpy import logspace, pi, log10, zeros, trapz
from Model import Ek_int, get_rise_speed
import matplotlib.pyplot as plt
from Utilities import findcLceta

nu =1e-6; g=9.81; sig=0.072;rhoc=1000;nu=1e-6
kt = 1.0 #m2/s2
et = 4.0 #m2/s3
cL,cEta=findcLceta(kt,et,nu,mode=1)
lam = 1

C=1.5;	p0=2.0;	beta=5.2
L=kt**1.5/et;  eta=(nu**3/et)**0.25

zp_lst = [1, 0.512]
rsp = get_rise_speed(lam,2*zp_lst[0],kt,et,nu,cL,cEta,method=2)
print("zp: {:.3f}, lambda: {:.3f}, rising spd: {:.5f}".format(zp_lst[0], lam, rsp))
rsp = get_rise_speed(lam,2*zp_lst[1],kt,et,nu,cL,cEta,method=2)
print("zp: {:.3f}, lambda: {:.3f}, rising spd: {:.5f}".format(zp_lst[1], lam, rsp))

klst_full = logspace(log10(2*pi/2/zp_lst[1]/100),   log10(2*pi/lam*100), 1000)
klst      = logspace(log10(2*pi/2/zp_lst[1]),       log10(2*pi/lam),     1000)
Eklst_full= zeros(1000)
Eklst     = zeros(1000)
fig=plt.figure(figsize=(4,2.5),dpi=300)
ax=fig.add_subplot(111)
for i in range(1000):
	Eklst[i]      = Ek_int(klst[i],     cL,cEta,C,L,p0,beta,eta,et)
	Eklst_full[i] = Ek_int(klst_full[i],cL,cEta,C,L,p0,beta,eta,et)

ax.plot(klst_full,Eklst_full,color='Black')
ax.plot(klst,     Eklst,     color='Red')
ax.plot([2*pi/2/zp_lst[0],2*pi/2/zp_lst[0]],[min(Eklst_full),max(Eklst_full)],color="Black",linestyle="--")
ax.plot([2*pi/2/zp_lst[1],2*pi/2/zp_lst[1]],[min(Eklst_full),max(Eklst_full)],color="Black",linestyle="-.")
ax.plot([2*pi/lam,2*pi/lam],[min(Eklst_full),max(Eklst_full)],color="Black",linestyle="-")

ax.set_xscale("log")
# ax.set_yscale("log"); ax.set_ylim([1e-10,10])
print("trapz for zp = {:.4f} is {:.5f}".format(zp_lst[1], (trapz(Eklst,klst)*2.0/3.0)**(0.5)))