from numpy import loadtxt, zeros, pi
from scipy.io import loadmat
from Model import Jent_numerical_New, Jent_numerical_prev
from Utilities import get_ug
L0  = 47 				#Athena Ship Length
U0  = 10.5 * 0.514444	# 10.5 knot speed
z  = 0.00084403593791648746* L0
kt = 0.009532545693218708 * U0 * U0
et = 0.1691356748342514 * U0**3/L0
nut= 4.8688365495763719e-005 * U0*L0
ad = 0.15011142194271088
print(">> Input: kt: {:.3e} m^2/s^2, et: {:.3e}m^2/s^3".format(kt,et))
# Water-air properties
nu= 1.0e-6; g=9.81; rhoc= 1000; sig= 0.072; rhod = 1.204
rrange=-1; Sl0 = 0.026e0
###########################################################
#Groups Info
ds = loadtxt('groups.dat', skiprows=1)
D =ds[:,0]*L0*2	#Diameters
Dg=ds[:,1]		#Entrainment distribution
G = D.size
#
#Compute group sizes
dDg=zeros(G)
dDg[0] = 0.5*(D[0]+D[1])
for ig in range(1,G-1):
	dDg[ig] = 0.5*(D[ig+1]-D[ig-1])
dDg[G-1] = 2*dDg[G-2]-dDg[G-3]
vg = pi/6*(D**3)	#Bubble volume
v0 = sum(Dg*vg)		#Mean entrainned bubble volume
#Terminal velocity
VT=zeros(G)
for ig in range(G):
	VT[ig] = get_ug(D[ig],rhoc,rhod,sig,nu,1.0)

# Calculate pz:
L11 = 0.43*(kt**1.5)/et
LD=zeros(G); Lz=zeros(G); pz=zeros(G); n=zeros(G); pz0=zeros(G);
for ig in range(G):
	LD[ig]=nut/VT[ig]
	Lz[ig]=1/(1/L11+1/LD[ig])
	n[ig] = L11/LD[ig]
	#Variable turbulence
	arg = ((1-z/L11)+abs(1-z/L11))/2
	pz[ig]=(arg**n[ig])/Lz[ig];
	arg = ((1-0/L11)+abs(1-0/L11))/2
	pz0[ig]=(arg**n[ig])/Lz[ig];
#Find ph
phg = Dg/pz0
phg = phg/sum(phg)
###########################################################
# MIT model
Table=loadmat("UoIEntrainment.mat");
J_MIT=Jent_numerical_New(kt,et,nu,g,rhoc,sig,Table,rrange,wmeth=2)
# Castro model
J_Castro=Jent_numerical_prev(kt,et)
print(">>Superficial velocity (J): MIT:{:.5e}m/s, Casto:{:.5e}m/s".format(J_MIT,J_Castro))
S_MIT = 0; S_Castro = 0
for ig in range(G):
	S_MIT    += J_MIT/v0*phg[ig]*pz[ig]*vg[ig]*(1-ad/0.3)
	S_Castro += Sl0*J_Castro/v0*phg[ig]*pz[ig]*vg[ig]*(1-ad/0.3)
print(">>Total source: MIT:{:.5e}/s, Casto:{:.5e}/s".format(S_MIT,S_Castro))
print(">>Total source (REX SCALE): MIT:{:.5e}, Casto:{:.5e}".format(S_MIT/(U0/L0),S_Castro/(U0/L0)))

# Analysis parameters
L=kt**1.5/et
eta = nu**(0.75)/et**(0.25)/L
Ree = et**(1.0/3.0)*L**(4.0/3.0)/nu
Wee = rhoc*et**(2.0/3.0)*L**(5.0/3.0)/sig
Fr2e= et**(2.0/3.0)*L**(-1.0/3.0)/g

#