#########################################
################DIMED###################
#########################################
from numpy import logspace, zeros, sqrt, log10
from Model import Jent_numerical_New,J_lambda
from Utilities import u2_intgrand, findcLceta, ulambda_sq
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import time as t
# debug input
# kt/et[1]
kt = 1.77947018E-01
et = 1.78672232E+02
nu= 1.0e-6;
g=9.81;
rhoc= 1000;
sig= 0.072;
Table=loadmat("UoIEntrainment.mat");
rrange=-1
##################################################
L=kt**1.5/et
eta = nu**(0.75)/et**(0.25)/L
Ree = et**(1.0/3.0)*L**(4.0/3.0)/nu
Wee = rhoc*et**(2.0/3.0)*L**(5.0/3.0)/sig
Fr2e= et**(2.0/3.0)*L**(-1.0/3.0)/g

t1=t.time()
# nlam=1200
# x1=sqrt(4*sig/rhoc/g); x4=5*L;
# lam_lst=logspace(log10(x1),log10(x4),nlam); ul2_lst = zeros(nlam)
# cL,cEta=findcLceta(kt,et,nu,mode=1)
# # # print(u2_intgrand(1e-2,1.0,cL,cEta,L,eta,et))
# print(ulambda_sq(0.2,kt,et,cL,cEta,nu,pope_spec=1))
# for il in range(nlam):
# 	ul2_lst[il] = ulambda_sq(lam_lst[il],kt,et,cL,cEta,nu,pope_spec=1)
# print(J_lambda(0.2,lam_lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange))
# print(J_lambda(2*x1,lam_lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange))
# print(J_lambda(2.5*x1,lam_lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange))
J_dim=Jent_numerical_New(kt,et,nu,g,rhoc,sig,Table,rrange,wmeth=2)
# Jl_lst=zeros(nlam)
# for il in range(nlam):
# 	Jl_lst[il]=J_lambda(lam_lst[il],lam_lst,ul2_lst,kt,et,cL,cEta,nu,g,rhoc,sig,Table,rrange)

# plt.plot(lam_lst*L,Jl_lst) ; plt.title("Dimensioned")
# J_trap=trapezoid(Jl_lst,lam_lst)
t2=t.time()
print('============= Dimensioned VERSION J = {:.4e}m/s, time:{:.2f}s.'.format(J_dim,t2-t1))