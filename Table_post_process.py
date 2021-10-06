from numpy import load, log10, array, abs, log
from numpy.random import default_rng
from scipy.interpolate import interpn
from Model import Jent_numerical_New
from scipy.io import loadmat

nuc = 1e-6
g = 9.81; rhoc = 1000; sig = 0.072
# L = 1 #meter
data=load("J_star_table_64.npz")
# n_Ree = data["n_Ree"]; Ree_lst = data["Ree_lst"]; Reemax=max(Ree_lst); Reemin=min(Ree_lst)
n_Wee = data["n_Wee"]; Wee_lst = data["Wee_lst"]; Weemax=max(Wee_lst); Weemin=min(Wee_lst)
n_Fr2 = data["n_Fr2"]; Fr2_lst = data["Fr2_lst"]; Fr2max=max(Fr2_lst); Fr2min=min(Fr2_lst)
n_Eta = data["n_Eta"]; Eta_lst = data["Eta_lst"]; Etamax=max(Eta_lst); Etamin=min(Eta_lst)
J_fin_table = data["J_fin_table"]
#######################################################################
#                     Check if the table is valid                     #
#######################################################################
logWee_lst	=log10(Wee_lst)
logFr2_lst	=log10(Fr2_lst)
logEta_lst	=log10(Eta_lst)
logJ_tab	=log10(J_fin_table)

coords=(logWee_lst,logFr2_lst,logEta_lst)
Table=loadmat("UoIEntrainment.mat");
rng = default_rng()
for i in range(1):
	# vals = abs(rng.random(4))
	# Wee	= vals[1] * (Weemax-Weemin)	+ Weemin
	# Fr2	= vals[2] * (Fr2max-Fr2min) + Fr2min
	# Eta	= vals[3] * (Etamax-Etamin) + Etamin
	# kt =  5.57172823E-01; et = 1.05084015E+00
	kt =  1.769E-06; et = 1.226E-05
	
	L =kt**1.5/et
	Eta = nuc**(0.75)/et**(0.25)/L
	Wee = rhoc*et**(2.0/3.0)*L**(5.0/3.0)/sig
	Fr2 = et**(2.0/3.0)*L**(-1.0/3.0)/g
	
	print('----------------------------------------------------------------------------------')
	print(' >>Inputs  Wee:{:.3e}, Fr2:{:.3e}, Eta*:{:.3e}'.format(Wee,Fr2,Eta))
	logWee  = log10(Wee);
	logFr2  = log10(Fr2);
	logEta  = log10(Eta);

	et = (nuc**(3/4)/(Eta*L))**4
	kt = (L*et)**(1/1.5)
	print(' >>Inputs  kt:{:.3e}m^2/s^2, et:{:.3e}m^2/s^3, L:{:.3e}m, Eta:{:.3e}m'.format(kt,et,L,Eta*L))

	J_dim=Jent_numerical_New(kt,et,nuc,g,rhoc,sig,Table,rrange=-1,wmeth=2)

	point = array([logWee, logFr2, logEta])
	J_scl_int=10**(interpn(coords, logJ_tab, point))[0] * (et*L)**(1/3) 
	print(' >>J: Exact:{:.5e}m/s, Interp:{:.5e}m/s, Rel err:{:6.2f}%'.format(J_dim,J_scl_int,(J_scl_int-J_dim)/J_dim*100))
#######################################################################
#                 Print the table in ASCII for FORTRAN                #
#######################################################################
# print('start writting')
# fid1=open('Tab_J_ascii.dat','w')
# fid3=open('Tab_Jpa_ascii.dat','w')
# #J_fin_table=zeros((n_Wee,n_Fr2,n_Eta))
# for iWee in range(n_Wee):
# 	for iFr2 in range(n_Fr2):
# 		for iEta in range(n_Eta):
# 			if J_fin_table[iWee,iFr2,iEta] <=0:
# 				J_out=-1e50
# 			else:
# 				J_out=log(J_fin_table[iWee,iFr2,iEta])
# 			print('{:.16e} '.format(J_out),		file=fid1,	end='')

# print('{:d} {:d} {:d}'.format(n_Wee, n_Fr2, n_Eta),		file=fid3)
# print('{:.16e} {:.16e} {:.16e}'.format(Weemin, Fr2min, Etamin),	file=fid3)
# print('{:.16e} {:.16e} {:.16e}'.format(Weemax, Fr2max, Etamax),	file=fid3)

# for iWee in range(n_Wee):
# 	print('{:.16e} '.format(Wee_lst[iWee]),	file=fid3, end='')
# print('\n', file=fid3)

# for iFr2 in range(n_Fr2):
# 	print('{:.16e} '.format(Fr2_lst[iFr2]),	file=fid3, end='')
# print('\n', file=fid3)

# for iEta in range(n_Eta):
# 	print('{:.16e} '.format(Eta_lst[iEta]),	file=fid3, end='')
# print('\n', file=fid3)

# fid1.close()
# fid3.close()